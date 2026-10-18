"""Golden-set bake-off: how much conversion should be deterministic, how much LLM?

Every strategy is judged by the same deterministic gate - output data parity to
2 decimals against SAS-derived expected outputs.  Primary metric: parity pass
rate (first attempt, and after the bounded repair loop).  Secondary: repair
attempts, cost, latency, run-to-run agreement.

Strategies
  rules        deterministic converter only (unsupported steps fail the case)
  hybrid       rules; the LLM converts only the unsupported units; repair loop
  llm_runtime  the LLM converts the whole program, with the sas2dbx runtime + hazard catalog
  llm_plain    the LLM converts the whole program to plain PySpark (no helper library)

Decision rule per construct family: rules where rules pass; the LLM (behind the
gate) where only the LLM passes; otherwise extend the rules or convert manually.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..runlog import usd
from ..semantic.schema import Catalog, Schema
from . import prompts
from .assist import fill_unsupported, repair_unit, unit_for_problem, units
from .guard import check_code
from .provider import PRICES, Provider, ProviderError

STRATEGIES = ("rules", "hybrid", "llm_runtime", "llm_plain")

PREAMBLE = """# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from sas2dbx.runtime import S, P, SasContext, context_from_widgets

if "ctx" not in globals():
    ctx = context_from_widgets(spark)
"""


@dataclass
class GoldenCase:
    id: str
    kind: str                         # conformance | parity
    sas: str                          # the original program text
    schemas: dict[str, Schema]        # input data sets
    outputs: list[str]                # data sets the gate compares
    family: str
    payload: object = None
    macro_vars: dict[str, str] = field(default_factory=dict)
    frontier: bool = False            # the rules are expected to route it to LLM/manual


def golden_set(conformance_dir: str = "conformance", cases_dir: str = "cases", only: list[str] | None = None,
               families: list[str] | None = None) -> list[GoldenCase]:
    from ..validate.conformance import catalog_for, load_all
    from ..validate.runner import ParityCase, build_catalog

    out: list[GoldenCase] = []
    for c in load_all(conformance_dir):
        if not c.expected:            # nothing to compare: not a parity case
            continue
        if c.param_module:            # tests the parameter-notebook conversion, not SAS semantics an LLM converts
            continue
        fam = c.family or (c.hazards[0] if c.hazards else "general")
        out.append(GoldenCase(c.id, "conformance", c.sas, dict(catalog_for(c).datasets), [k.upper() for k in c.expected],
                              fam, c, dict(c.macro_vars), c.expect_unsupported))
    if cases_dir and os.path.isdir(cases_dir):
        for name in sorted(os.listdir(cases_dir)):
            if not name.endswith(".yaml"):
                continue
            pc = ParityCase.from_yaml(os.path.join(cases_dir, name))
            with open(pc.program, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            cat = build_catalog(pc.libnames)
            referenced = {k: v for k, v in cat.datasets.items()
                          if re.search(r"\b" + re.escape(k.split(".")[1]) + r"\b", text, re.I)}
            out.append(GoldenCase(pc.name, "parity", text, referenced, [o.dataset.upper() for o in pc.outputs],
                                  "real-program", pc, dict(pc.macro_vars)))
    if only:
        out = [g for g in out if g.id in only or g.family in only]
    if families:
        out = [g for g in out if g.family in families]
    return out


# ---------------------------------------------------------------------------
# conversion + the gate
def convert_golden(case: GoldenCase):
    from ..validate import conformance, runner

    if case.kind == "conformance":
        return conformance.convert_case(case.payload)
    return runner.convert_case(case.payload)    # as `validate` does: its parameter module stays a parameter notebook


def gate(case: GoldenCase, notebook: str, spark) -> tuple[bool, str]:
    """Execute the notebook against the case inputs; compare every expected output at 2 decimals."""
    if case.kind == "conformance":
        from ..validate.conformance import run_conformance_notebook

        r = run_conformance_notebook(case.payload, spark, notebook)
        return r.passed, r.detail
    from ..validate.runner import run_case

    r = run_case(case.payload, spark, notebook_override=notebook)
    if not r.execution.get("ok"):
        return False, f"execution failed in {r.execution.get('title')}: {str(r.execution.get('error', ''))[:800]}"
    bad = []
    for c in r.comparisons:
        if not c.passed:
            bad.append(c.summary() + (f" | sample {c.samples[0]}" if c.samples else ""))
    return (not bad and bool(r.comparisons)), "; ".join(bad)[:2000] or "ok"


# ---------------------------------------------------------------------------
@dataclass
class Trial:
    strategy: str
    case: str
    family: str
    frontier: bool
    run: int
    first_pass: bool = False
    final_pass: bool = False
    attempts: int = 0                # answers requested (1 + repairs); rules: 0
    gate_runs: int = 0               # executions of the parity gate
    llm_calls: int = 0
    llm_units: int = 0               # hybrid: units the LLM had to convert
    steps: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_seconds: float = 0.0
    seconds: float = 0.0
    error: str = ""
    code_sha: str = ""               # what finally ran (run-to-run agreement on the code itself)

    def charge(self, comp) -> None:
        self.llm_calls += 1
        self.cost_usd += comp.cost_usd
        self.llm_seconds += comp.seconds
        u = comp.usage
        self.tokens_in += u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        self.tokens_out += u.get("output_tokens", 0)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _whole_program(provider: Provider, case: GoldenCase, mode: str, effort: str | None, t: Trial,
                   previous: str | None = None, problem: str | None = None) -> tuple[str | None, str]:
    user = prompts.program_prompt(case.sas, case.schemas, case.outputs, case.macro_vars)
    if previous is not None:
        user = prompts.repair_prompt(user, previous, problem or "")
    comp = provider.complete(prompts.system_prompt(mode), user, schema=prompts.OUTPUT_SCHEMA, effort=effort)
    t.charge(comp)
    if comp.refused:
        return None, "model declined the request (refusal)"
    try:
        ans = prompts.parse_answer(comp.text)
    except ValueError as e:
        return None, f"unparseable answer: {e}"
    problems = check_code(ans["code"])
    return ans["code"], ("rejected by the static guard: " + "; ".join(problems)) if problems else ""


def run_trial(strategy: str, case: GoldenCase, spark, provider: Provider | None, *, attempts: int = 3,
              effort: str | None = None, run: int = 0, log=None) -> Trial:
    """attempts = maximum gate runs (1 + repairs)."""
    t = Trial(strategy, case.id, case.family, case.frontier, run)
    t0 = time.time()
    try:
        if strategy in ("rules", "hybrid"):
            conv = convert_golden(case)
            t.steps = len([s for s in conv.steps if s.kind != "GLOBAL"])
            pending = units(conv)
            if strategy == "rules" or not pending:
                if pending:
                    reasons = "; ".join(r for u in pending for s in u.steps for r in s.reasons)
                    ok, detail = False, f"not converted by the rules: {reasons}"
                else:
                    t.gate_runs, t.attempts = 1, 1
                    ok, detail = gate(case, conv.notebook, spark)
                t.first_pass = t.final_pass = ok
                t.error = "" if ok else detail
                t.code_sha = _sha(conv.notebook)
            else:
                if provider is None:
                    raise RuntimeError("the hybrid strategy needs an LLM provider")
                t.llm_units = len(pending)
                conv, recs = fill_unsupported(conv, provider, original=case.sas, effort=effort)
                for r in recs:
                    t.llm_calls += r.attempts
                    t.cost_usd += r.cost_usd
                    t.llm_seconds += r.seconds
                    t.tokens_in += r.tokens_in
                    t.tokens_out += r.tokens_out
                by_unit = {u.index: (u, r) for u, r in zip(pending, recs)}

                def rejected() -> list[str]:
                    return [p for _, r in by_unit.values() if not r.ok for p in r.problems]

                t.attempts = 1
                if rejected():
                    ok, detail = False, "rejected by the static guard: " + "; ".join(rejected())
                else:
                    t.gate_runs += 1
                    ok, detail = gate(case, conv.notebook, spark)
                t.first_pass = ok
                while not ok and t.attempts < attempts:
                    t.attempts += 1
                    target = next((u for u, r in by_unit.values() if not r.ok), None) or unit_for_problem(detail, pending)
                    u, rec = by_unit[target.index]
                    before = (rec.attempts, rec.cost_usd, rec.seconds, rec.tokens_in, rec.tokens_out)
                    rec.ok = repair_unit(provider, conv, u, detail, rec, original=case.sas, effort=effort)
                    t.llm_calls += rec.attempts - before[0]
                    t.cost_usd += rec.cost_usd - before[1]
                    t.llm_seconds += rec.seconds - before[2]
                    t.tokens_in += rec.tokens_in - before[3]
                    t.tokens_out += rec.tokens_out - before[4]
                    if rejected():
                        ok, detail = False, "rejected by the static guard: " + "; ".join(rejected())
                        continue
                    t.gate_runs += 1
                    ok, detail = gate(case, conv.notebook, spark)
                t.final_pass = ok
                t.error = "" if ok else detail
                t.code_sha = _sha(conv.notebook)
        elif strategy in ("llm_runtime", "llm_plain"):
            if provider is None:
                raise RuntimeError(f"the {strategy} strategy needs an LLM provider")
            mode = "runtime" if strategy == "llm_runtime" else "plain"
            code, problem = _whole_program(provider, case, mode, effort, t)
            t.attempts = 1
            ok, detail = False, problem
            if code is not None and not problem:
                t.gate_runs += 1
                ok, detail = gate(case, PREAMBLE + "\n# COMMAND ----------\n\n" + code + "\n", spark)
            t.first_pass = ok
            while not ok and t.attempts < attempts:
                t.attempts += 1
                new, problem = _whole_program(provider, case, mode, effort, t, previous=code or "(no usable answer)",
                                              problem=detail)
                code = new if new is not None else code
                if new is None or problem:
                    ok, detail = False, problem
                    continue
                t.gate_runs += 1
                ok, detail = gate(case, PREAMBLE + "\n# COMMAND ----------\n\n" + code + "\n", spark)
            t.final_pass = ok
            t.error = "" if ok else detail
            t.code_sha = _sha(code or "")
        else:
            raise ValueError(f"unknown strategy {strategy}")
    except ProviderError:
        raise                          # infrastructure, not the model: abort rather than score it
    except Exception as e:  # noqa: BLE001 - a crashing trial is a failed trial, never a skipped one
        t.error = f"{type(e).__name__}: {e}"
    t.seconds = round(time.time() - t0, 2)
    t.cost_usd = round(t.cost_usd, 6)
    if log:
        log(f"{strategy:<12} {case.id:<28} run {run}  first={'PASS' if t.first_pass else 'fail'}  "
            f"final={'PASS' if t.final_pass else 'fail'}  attempts={t.attempts} calls={t.llm_calls} "
            f"${t.cost_usd:.3f}  {t.error[:100]}")
    return t


def prefetch(cases: list[GoldenCase], strategies: list[str], provider_for_run, *, runs: int = 1,
             effort: str | None = None, workers: int = 6, log=None) -> int:
    """Request every first-attempt answer concurrently, so the (sequential, Spark-bound) evaluation
    replays them from the response cache.  Repairs depend on gate results and stay sequential."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = []
    for s in strategies:
        if s == "rules":
            continue
        for r in range(runs):
            provider = provider_for_run(r)
            for c in cases:
                jobs.append((s, c, provider, r))

    def one(job) -> int:
        s, c, provider, r = job
        if s == "hybrid":
            conv = convert_golden(c)
            if not units(conv):
                return 0
            _, recs = fill_unsupported(conv, provider, original=c.sas, effort=effort)
            return sum(r_.attempts for r_ in recs)
        _whole_program(provider, c, "runtime" if s == "llm_runtime" else "plain", effort, Trial(s, c.id, c.family, c.frontier, r))
        return 1

    done = calls = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        try:
            for f in as_completed(futures):
                calls += f.result()     # a ProviderError surfaces here and aborts the run
                done += 1
                if log and (done % 10 == 0 or done == len(jobs)):
                    log(f"prefetched {done}/{len(jobs)} first attempts")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return calls


def run_eval(cases: list[GoldenCase], strategies: list[str], spark, provider_for_run, *, runs: int = 1,
             attempts: int = 3, effort: str | None = None, log=None) -> dict:
    """provider_for_run(run) -> Provider | None (a fresh cache namespace per run = independent samples)."""
    trials: list[Trial] = []
    for strategy in strategies:
        n_runs = 1 if strategy == "rules" else runs
        for r in range(n_runs):
            provider = None if strategy == "rules" else provider_for_run(r)
            for case in cases:
                trials.append(run_trial(strategy, case, spark, provider, attempts=attempts, effort=effort, run=r, log=log))
    return summarize(trials, meta={"spark_version": spark.version, "ansi": spark.conf.get("spark.sql.ansi.enabled"),
                                   "runs": runs, "attempts": attempts, "effort": effort,
                                   "cases": len(cases), "strategies": list(strategies)})


# ---------------------------------------------------------------------------
def _pct(xs) -> float | None:
    xs = list(xs)
    return round(100.0 * sum(xs) / len(xs), 1) if xs else None


def summarize(trials: list[Trial], meta: dict | None = None) -> dict:
    strategies: dict[str, dict] = {}
    for s in [x for x in STRATEGIES if any(t.strategy == x for t in trials)]:
        ts = [t for t in trials if t.strategy == s]
        groups = defaultdict(list)
        for t in ts:
            groups[t.case].append(t)
        multi = [v for v in groups.values() if len(v) > 1]
        strategies[s] = {
            "trials": len(ts),
            "first_attempt_pass_pct": _pct(t.first_pass for t in ts),
            "final_pass_pct": _pct(t.final_pass for t in ts),
            "frontier_final_pass_pct": _pct(t.final_pass for t in ts if t.frontier),
            "covered_final_pass_pct": _pct(t.final_pass for t in ts if not t.frontier),
            "mean_attempts": round(sum(t.attempts for t in ts) / len(ts), 2),
            "gate_runs": sum(t.gate_runs for t in ts),
            "llm_calls": sum(t.llm_calls for t in ts),
            "cost_usd": round(sum(t.cost_usd for t in ts), 4),
            "cost_per_pass_usd": round(sum(t.cost_usd for t in ts) / max(1, sum(t.final_pass for t in ts)), 4),
            "tokens_in": sum(t.tokens_in for t in ts),
            "tokens_out": sum(t.tokens_out for t in ts),
            "llm_seconds": round(sum(t.llm_seconds for t in ts), 1),
            "seconds": round(sum(t.seconds for t in ts), 1),
            # same verdict on every run of a case (and, stricter, identical code)
            "verdict_agreement_pct": _pct(len({t.final_pass for t in v}) == 1 for v in multi) if multi else None,
            "code_agreement_pct": _pct(len({t.code_sha for t in v}) == 1 for v in multi) if multi else None,
        }
    fams: dict[str, dict] = {}
    for f in sorted({t.family for t in trials}):
        row: dict = {"cases": len({t.case for t in trials if t.family == f}),
                     "frontier": any(t.frontier for t in trials if t.family == f)}
        for s in strategies:
            ts = [t for t in trials if t.family == f and t.strategy == s]
            row[s] = _pct(t.final_pass for t in ts)
            row[f"{s}_first"] = _pct(t.first_pass for t in ts)
        row["decision"] = decide(row, list(strategies))
        fams[f] = row
    failures = [{"strategy": t.strategy, "case": t.case, "run": t.run, "error": t.error[:400]}
                for t in trials if not t.final_pass]
    return {"meta": meta or {}, "strategies": strategies, "families": fams, "failures": failures,
            "trials": [asdict(t) for t in trials]}


def decide(row: dict, strategies: list[str]) -> str:
    """Rules where they reach parity; else the cheapest LLM strategy that always did; else extend rules / manual."""
    if row.get("rules") == 100.0:
        return "rules"
    for s in ("hybrid", "llm_runtime", "llm_plain"):
        if s in strategies and row.get(s) == 100.0:
            return f"LLM via {s} (behind the parity gate)"
    best = max((row.get(s) or 0.0, s) for s in strategies) if strategies else (0.0, "")
    if best[0] > 0:
        return f"unreliable: best {best[1]} {best[0]}% - extend rules/runtime or convert manually"
    return "extend rules/runtime or convert manually"


def estimate_cost(cases: list[GoldenCase], strategies: list[str], *, runs: int, attempts: int,
                  model: str, out_tokens=(1500, 5000, 12000), repair_rate=(0.15, 0.35, 0.6)) -> dict:
    """Token-based cost envelope before spending anything (chars/3.5 per token; output incl. thinking)."""
    p_in, p_out, p_cache_read, p_cache_write = PRICES.get(model, PRICES["claude-opus-5-5"])
    sys_tok = {m: len(prompts.system_prompt(m)) / 3.5 for m in ("runtime", "plain")}
    calls = 0.0
    est = [0.0, 0.0, 0.0]
    for s in strategies:
        if s == "rules":
            continue
        mode = "plain" if s == "llm_plain" else "runtime"
        for c in cases:
            if s == "hybrid":
                conv = convert_golden(c)
                n_units = len(units(conv))
                if not n_units:
                    continue
                user_tok = len(c.sas) / 3.5 + 1200 * n_units
                first = n_units
            else:
                user_tok = len(c.sas) / 3.5 + 400 + 60 * sum(len(v.vars) for v in c.schemas.values())
                first = 1
            for k in range(3):
                n = runs * first * (1 + repair_rate[k] * (attempts - 1))
                calls += n / 3
                # repairs resend the previous code and the gate's diff
                in_cost = n * (user_tok * (1 + repair_rate[k]) * p_in + sys_tok[mode] * p_cache_read) / 1e6
                est[k] += in_cost + n * out_tokens[k] * p_out / 1e6
    writes = sum(sys_tok.values()) * p_cache_write / 1e6 * 20          # occasional cache re-writes (5-minute TTL)
    return {"model": model, "approx_calls": round(calls), "usd_low": round(est[0] + writes, 2),
            "usd_expected": round(est[1] + writes, 2), "usd_high": round(est[2] + writes, 2),
            "system_prompt_tokens": {k: round(v) for k, v in sys_tok.items()}}


def write_report(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def render_markdown(summary: dict) -> str:
    st = summary["strategies"]
    names = list(st)
    lines = ["| strategy | first-attempt parity | final parity | frontier | covered by rules | mean attempts | LLM calls | cost $ | $ per pass | verdict agreement |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in names:
        r = st[s]
        pct = lambda v: "-" if v is None else f"{v}%"  # noqa: E731
        lines.append(f"| {s} | {pct(r['first_attempt_pass_pct'])} | {pct(r['final_pass_pct'])} | {pct(r['frontier_final_pass_pct'])} | "
                     f"{pct(r['covered_final_pass_pct'])} | {r['mean_attempts']} | {r['llm_calls']} | {usd(r['cost_usd'])[1:]} | "
                     f"{r['cost_per_pass_usd']:.3f} | {r['verdict_agreement_pct'] if r['verdict_agreement_pct'] is not None else '-'} |")
    lines += ["", "| family | cases | " + " | ".join(names) + " | decision |", "|---|---|" + "---|" * len(names) + "---|"]
    for f, row in summary["families"].items():
        cells = " | ".join("-" if row.get(s) is None else f"{row[s]:.0f}%" for s in names)
        lines.append(f"| {f}{' *' if row['frontier'] else ''} | {row['cases']} | {cells} | {row['decision']} |")
    lines.append("\n`*` frontier family: the rules route it to LLM/manual by design.")
    return "\n".join(lines)
