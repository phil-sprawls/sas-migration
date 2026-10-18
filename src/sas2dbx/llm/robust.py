"""Robustness of converted code beyond the rows it was validated on (differential testing).

For every golden case, synthetic variants of the inputs (``validate.fuzz``) are run
through a trusted reference and through each strategy's conversion; outputs must
agree to 2 decimals.  The reference is the rule-based conversion where the rules
cover the case (validated against SAS semantics by the conformance suite), else the
case's independent double-programmed ``reference_pyspark``.  A disagreement means
at least one side is wrong - each one is evidence to adjudicate, not a verdict.

LLM answers are replayed from the response cache (``ReplayOnly``): no spend.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..validate.compare import CompareOptions, compare
from ..validate.conformance import execute_case_notebook, run_conformance_notebook
from ..validate.fuzz import fuzz_tables
from .assist import fill_unsupported, units
from .eval import PREAMBLE, GoldenCase, Trial, _whole_program, convert_golden
from .provider import Completion, ProviderError


class ReplayOnly:
    """Inner provider for a CachedProvider that must never call a model."""

    name = "replay-only"

    def __init__(self, model: str):
        self.model = model

    def complete(self, system, user, **kw) -> Completion:
        raise ProviderError("answer not in the response cache (replay-only: no spend)")


def reference_notebook(case: GoldenCase) -> tuple[str | None, str]:
    conv = convert_golden(case)
    if not units(conv):
        return conv.notebook, "rules"
    ref = getattr(case.payload, "reference_pyspark", "")
    if ref:
        return PREAMBLE + "\n# COMMAND ----------\n\n" + ref + "\n", "reference"
    return None, ""


def candidate_notebook(strategy: str, case: GoldenCase, provider, effort: str | None) -> str | None:
    """The strategy's first-attempt conversion (replayed); None when it was rejected before execution."""
    if strategy == "rules":
        conv = convert_golden(case)
        return None if units(conv) else conv.notebook
    if strategy == "hybrid":
        conv = convert_golden(case)
        if units(conv):
            conv, recs = fill_unsupported(conv, provider, original=case.sas, effort=effort)
            if not all(r.ok for r in recs):
                return None
        return conv.notebook
    mode = "runtime" if strategy == "llm_runtime" else "plain"
    code, problem = _whole_program(provider, case, mode, effort, Trial(strategy, case.id, case.family, case.frontier, 0))
    return None if code is None or problem else PREAMBLE + "\n# COMMAND ----------\n\n" + code + "\n"


def _run(case: GoldenCase, spark, notebook: str, inputs, timeout: float):
    """Execute with a time limit (synthetic inputs can make SAS logic loop forever, e.g. DO WHILE with rate 0)."""
    timed_out = threading.Event()

    def stop() -> None:
        timed_out.set()
        spark.sparkContext.cancelAllJobs()

    timer = threading.Timer(timeout, stop)
    timer.start()
    try:
        exe, ctx = execute_case_notebook(case.payload, spark, notebook, inputs)
        if exe["ok"]:                        # Spark is lazy: surface run-time errors now, under the time limit
            inputs_ = {k.split(".")[-1].upper() for k in (inputs or {})}
            for ds in ctx.names("WORK"):
                if ds not in inputs_:
                    try:
                        ctx.get(ds).count()
                    except Exception as e:  # noqa: BLE001
                        exe = {"ok": False, "title": f"materializing {ds}", "error": f"{type(e).__name__}: {e}"}
                        break
    finally:
        timer.cancel()
    if timed_out.is_set():
        exe = {"ok": False, "title": "timeout", "error": f"exceeded {timeout:.0f}s"}
    return exe, ctx


def _stem(name: str) -> str:
    """Data-dependent output names (CALL EXECUTE out_<value>, %DO part<i>) are matched by their stem."""
    n = name.upper().rstrip("0123456789")
    return n[: n.rfind("_") + 1] if "_" in n else n


def _outputs(case: GoldenCase, ctx, input_names: set[str]) -> set[str]:
    stems = {_stem(k.split(".")[-1]) for k in case.payload.expected}
    exact = {k.split(".")[-1].upper() for k in case.payload.expected}
    names = set()
    for ds in ctx.names("WORK"):
        if ds in input_names:
            continue
        if ds in exact or any(ds.startswith(s) and s for s in stems if s.endswith("_") or ds[len(s):].isdigit()):
            names.add(ds)
    return names


@dataclass
class Variant:
    strategy: str
    case: str
    family: str
    frontier: bool
    variant: int
    verdict: str = ""        # agree | disagree | candidate_error | skipped
    detail: str = ""
    reference: str = ""


def _compare_ctx(case: GoldenCase, ref_ctx, cand_ctx, input_names: set[str]) -> str:
    expected = {k.split(".")[-1].upper(): t for k, t in case.payload.expected.items()}
    want = _outputs(case, ref_ctx, input_names)
    got = _outputs(case, cand_ctx, input_names)
    problems = []
    if want != got:
        problems.append(f"data sets differ: reference {sorted(want)} vs candidate {sorted(got)}")
    for ds in sorted(want & got):
        t = expected.get(ds) or next(iter(expected.values()))
        r, c = ref_ctx.get(ds), cand_ctx.get(ds)
        if case.payload.compare_columns == "subset":
            keep = {col.name for col in t.columns} | {"_SAS_SEQ"}
            r = r.select(*[x for x in r.columns if x.upper() in keep])
            c = c.select(*[x for x in c.columns if x.upper() in keep])
        try:
            res = compare(r, c, CompareOptions(keys=t.keys or None, decimals=2, mode="keyed" if t.keys else "ordered"), ds)
        except Exception as e:  # noqa: BLE001 - e.g. incomparable types
            problems.append(f"{ds}: comparison failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        if not res.passed:
            problems.append(res.summary() + (f" | sample {res.samples[0]}" if res.samples else ""))
    return "; ".join(problems)


def robustness(cases: list[GoldenCase], strategies: list[str], spark, provider, *, variants: int = 5, seed: int = 0,
               effort: str | None = None, timeout: float = 30.0, log=None) -> dict:
    results: list[Variant] = []
    for case in cases:
        if case.kind != "conformance":
            continue                         # real programs: SAS ground truth on real data (G1/G3) already
        ref_nb, ref_kind = reference_notebook(case)
        if ref_nb is None:
            continue
        cands: dict[str, str | None] = {}
        why: dict[str, str] = {}
        for s in strategies:
            if s == "rules" or (s == "hybrid" and ref_kind == "rules"):
                continue                     # identical to the reference
            try:
                cands[s] = candidate_notebook(s, case, provider, effort)
                why[s] = "" if cands[s] else "first attempt rejected before execution"
            except ProviderError as e:
                cands[s], why[s] = None, str(e)
            if cands[s] is not None:         # robustness is asked only of conversions that passed the parity gate
                gate = run_conformance_notebook(case.payload, spark, cands[s])
                if not gate.passed:
                    cands[s], why[s] = None, "first attempt failed the parity gate"
        input_names = {k.split(".")[-1].upper() for k in case.payload.inputs}
        for v in range(variants):
            inputs = fuzz_tables(case.payload.inputs, case.sas, seed * 1000 + v,
                                 protect={k for t in case.payload.expected.values() for k in t.keys})
            ref_exe, ref_ctx = _run(case, spark, ref_nb, inputs, timeout)
            for s, nb in cands.items():
                row = Variant(s, case.id, case.family, case.frontier, v, reference=ref_kind)
                if nb is None:
                    row.verdict, row.detail = "skipped", why[s]
                elif not ref_exe["ok"]:
                    row.verdict, row.detail = "skipped", f"reference failed: {str(ref_exe.get('error'))[:200]}"
                else:
                    exe, ctx = _run(case, spark, nb, inputs, timeout)
                    if not exe["ok"]:
                        row.verdict, row.detail = "candidate_error", f"{exe.get('title')}: {str(exe.get('error'))[:300]}"
                    else:
                        diff = _compare_ctx(case, ref_ctx, ctx, input_names)
                        row.verdict, row.detail = ("disagree", diff[:1500]) if diff else ("agree", "")
                results.append(row)
                if log and row.verdict not in ("agree",):
                    log(f"{s:<12} {case.id:<26} v{v} {row.verdict}: {row.detail[:160]}")
        if log:
            log(f"{case.id}: done ({ref_kind} reference)")
    return summarize(results)


def summarize(results: list[Variant]) -> dict:
    out: dict = {"strategies": {}, "families": defaultdict(dict), "disagreements": [], "variants": [asdict(r) for r in results]}
    for s in sorted({r.strategy for r in results}):
        rs = [r for r in results if r.strategy == s and r.verdict != "skipped"]
        bad_cases = sorted({r.case for r in rs if r.verdict != "agree"})
        out["strategies"][s] = {
            "judged_variants": len(rs),
            "agree_pct": round(100 * sum(r.verdict == "agree" for r in rs) / len(rs), 1) if rs else None,
            "disagree": sum(r.verdict == "disagree" for r in rs),
            "candidate_error": sum(r.verdict == "candidate_error" for r in rs),
            "skipped": sum(1 for r in results if r.strategy == s and r.verdict == "skipped"),
            "cases_with_any_disagreement": bad_cases,
        }
        for f in sorted({r.family for r in rs}):
            fr = [r for r in rs if r.family == f]
            out["families"][f][s] = round(100 * sum(r.verdict == "agree" for r in fr) / len(fr), 1)
    out["families"] = dict(out["families"])
    out["disagreements"] = [asdict(r) for r in results if r.verdict in ("disagree", "candidate_error")]
    return out
