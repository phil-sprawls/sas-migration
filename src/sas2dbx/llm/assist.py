"""Hybrid conversion: rules first; the LLM converts only what the rules marked unsupported.

Contiguous unsupported steps form one *unit* (one prompt, one cell): run-time macro
logic and the step that consumes it must be converted together to be coherent.
Nothing here decides success - the caller runs the parity gate.
"""
from __future__ import annotations

import re
import types
from dataclasses import dataclass, field

from ..codegen.program import ConversionResult, StepReport, render_notebook
from ..lang.lexer import Kind, tokenize
from ..lang.parser import Parser
from ..semantic.schema import Catalog, Schema
from . import prompts
from .guard import check_code
from .provider import Completion, Provider

LLM_MARK = "# Converted by the LLM assist (provenance: llm"


@dataclass
class Unit:
    steps: list[StepReport]
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    needs_source: bool = False        # macro logic: the expanded text alone is not enough

    @property
    def index(self) -> int:
        return self.steps[0].index


@dataclass
class UnitResult:
    unit: int
    ok: bool = False
    attempts: int = 0
    problems: list[str] = field(default_factory=list)
    notes: str = ""
    cost_usd: float = 0.0            # what the answers cost to produce (also when replayed from the cache)
    billed_usd: float = 0.0          # what this run spent (cache replays are free)
    seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    last_code: str = ""              # the latest answer, kept even when the guard rejected it (for repair)

    def add(self, comp: Completion) -> None:
        self.attempts += 1
        self.cost_usd += comp.cost_usd
        self.billed_usd += comp.billed_usd
        self.seconds += comp.seconds
        u = comp.usage
        self.tokens_in += u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        self.tokens_out += u.get("output_tokens", 0)
        self.model = comp.model


# ---------------------------------------------------------------------------
# what a step reads and writes (the rules record this; fall back to a token scan
# when they gave up before recording it)
_READ_LISTS = ("SET", "MERGE", "UPDATE", "MODIFY")
_READ_ONE = {"FROM", "JOIN"}
_OPT_READ = {"DATA"}
_OPT_WRITE = {"OUT", "OUTPUT", "BASE", "OUTFILE"}


def scan_io(sas_text: str) -> tuple[list[str], list[str]]:
    p = Parser(sas_text)
    toks = p.toks
    reads: list[str] = []
    writes: list[str] = []

    def refs_at(j: int, many: bool, into: list[str]) -> None:
        while j < len(toks) and toks[j].kind in (Kind.NAME, Kind.RTMACRO) and not toks[j].is_name("SELECT", "SET", "WHERE"):
            p.i = j
            try:
                ref = p.parse_dataset_ref()
            except Exception:  # noqa: BLE001 - best effort
                return
            if p.i < len(toks) and toks[p.i].is_op("="):
                return                                  # `name = value`: an assignment (SQL SET), not a data set
            if not ref.is_null:
                into.append(ref.key)
            if not many or p.i == j:
                return
            j = p.i

    stmt_start = True
    for i, t in enumerate(toks):
        nxt_eq = i + 1 < len(toks) and toks[i + 1].is_op("=")
        if t.kind is Kind.NAME:
            u = t.upper
            if stmt_start and u == "DATA" and not nxt_eq:
                refs_at(i + 1, True, writes)
            elif (stmt_start or (i > 0 and toks[i - 1].is_name("THEN", "ELSE"))) and u in _READ_LISTS and not nxt_eq:
                refs_at(i + 1, True, reads)
            elif u in _READ_ONE:
                refs_at(i + 1, False, reads)
            elif nxt_eq and u in _OPT_READ:
                refs_at(i + 2, False, reads)
            elif nxt_eq and u in _OPT_WRITE:
                refs_at(i + 2, False, writes)
            elif u == "TABLE" and i > 0 and toks[i - 1].is_name("CREATE"):
                refs_at(i + 1, False, writes)
            elif u == "INTO" and i > 0 and toks[i - 1].is_name("INSERT"):
                refs_at(i + 1, False, writes)
        stmt_start = t.kind is Kind.SEMI
    return list(dict.fromkeys(reads)), list(dict.fromkeys(writes))


def _needs_source(s: StepReport) -> bool:
    return s.kind in ("MACRO", "UNPARSED") or "__sas2dbx_unresolved__" in s.sas_text


def units(conv: ConversionResult) -> list[Unit]:
    out: list[Unit] = []
    cur: Unit | None = None
    for s in conv.steps:
        if s.status != "unsupported" or s.kind == "GLOBAL":
            if s.kind == "GLOBAL" and s.status == "noop" and cur is not None:
                continue                       # comments/options between unsupported steps keep the unit open
            cur = None
            continue
        if cur is None:
            cur = Unit([])
            out.append(cur)
        cur.steps.append(s)
        r, w = (s.reads, s.writes) if (s.reads or s.writes) else scan_io(s.sas_text)
        cur.reads += [k for k in r if k not in cur.writes]
        cur.writes += w
        cur.needs_source |= _needs_source(s)
    for u in out:
        u.reads = list(dict.fromkeys(u.reads))
        u.writes = list(dict.fromkeys(u.writes))
    return out


def runtime_macros_read(notebook: str) -> list[str]:
    return sorted({m.group(2).upper() for m in re.finditer(r"ctx\.symget(_num)?\(['\"](\w+)['\"]\)", notebook)})


def _unit_prompt(u: Unit, conv: ConversionResult, original: str | None) -> str:
    names = u.reads or ([] if not u.needs_source else list(conv.catalog.datasets)[:30])
    schemas = {k: conv.catalog.get(k) or Schema([], known=False) for k in names}
    reasons = "; ".join(r for s in u.steps for r in s.reasons)
    before = [s.sas_text for s in conv.steps if s.index < u.index and s.sas_text.strip()]
    after_reads = sorted({k for s in conv.steps if s.index > u.steps[-1].index for k in s.reads})
    extra = []
    if after_reads:
        extra.append("Later (already converted) steps read these data sets: " + ", ".join(k.lower() for k in after_reads))
    rt = runtime_macros_read(conv.notebook)
    if rt:
        extra.append("Later converted steps read these run-time macro variables with ctx.symget/symget_num "
                     "(set them with ctx.symput if this code defines them): " + ", ".join(rt))
    if u.needs_source and original:
        sas = original
        what = ("The rule-based converter could not expand this macro logic statically because it depends on run-time "
                f"data ({reasons}). Below is the ORIGINAL program; convert only the part that corresponds to these "
                f"unresolved steps:\n" + "\n".join(f"- {s.title} at {s.origin}" for s in u.steps))
        return prompts.step_prompt(sas, schemas, what, u.writes, "\n".join(before), extra="\n".join(extra))
    sas = "\n".join(s.sas_text.strip() for s in u.steps)
    return prompts.step_prompt(sas, schemas, reasons, u.writes, "\n".join(before), extra="\n".join(extra))


def _apply(conv: ConversionResult, u: Unit, code: str, tag: str) -> None:
    first, *rest = u.steps
    first.code = [f"{LLM_MARK}{tag}). Validated only by the parity gate.", *code.splitlines()]
    first.status, first.provenance = "converted", "llm"
    for s in rest:
        s.code = [f"# converted together with step {first.index} (LLM)"]
        s.status, s.provenance = "converted", "llm"
    conv.notebook = render_notebook(conv, types.SimpleNamespace(runtime_macros=set()))


def _ask(provider: Provider, user: str, mode: str, effort: str | None, rec: UnitResult) -> str | None:
    comp = provider.complete(prompts.system_prompt(mode), user, schema=prompts.OUTPUT_SCHEMA, effort=effort)
    rec.add(comp)
    if comp.refused:
        rec.problems = ["model declined the request (refusal)"]
        return None
    try:
        ans = prompts.parse_answer(comp.text)
    except ValueError as e:
        rec.problems = [f"unparseable answer: {e}"]
        return None
    rec.last_code = ans["code"]
    rec.problems = check_code(ans["code"])
    rec.notes = ans.get("notes", "")
    return None if rec.problems else ans["code"]


def fill_unsupported(conv: ConversionResult, provider: Provider, *, original: str | None = None, mode: str = "runtime",
                     effort: str | None = None, log=None) -> tuple[ConversionResult, list[UnitResult]]:
    """Convert every unsupported unit with the LLM (static guard only - execution is the caller's gate)."""
    results: list[UnitResult] = []
    for u in units(conv):
        rec = UnitResult(u.index)
        code = _ask(provider, _unit_prompt(u, conv, original), mode, effort, rec)
        if code is not None:
            _apply(conv, u, code, "")
            rec.ok = True
        results.append(rec)
        if log:
            log(f"unit {u.index}: {'ok' if rec.ok else 'rejected'} {rec.problems[:2]}")
    return conv, results


def repair_unit(provider: Provider, conv: ConversionResult, u: Unit, problem: str, rec: UnitResult, *,
                original: str | None = None, mode: str = "runtime", effort: str | None = None) -> bool:
    if rec.problems:
        problem = "; ".join(rec.problems) + ("\n" + problem if problem else "")
    user = prompts.repair_prompt(_unit_prompt(u, conv, original), rec.last_code or "(no usable answer)", problem)
    code = _ask(provider, user, mode, effort, rec)
    if code is None:
        return False
    _apply(conv, u, code, ", repaired")
    return True


def unit_for_problem(problem: str, llm_units: list[Unit]) -> Unit:
    """The unit to repair: the one whose cell failed, else the one that writes the failing data set."""
    m = re.search(r"\[(\d+)\]", problem)
    if m:
        idx = int(m.group(1))
        for u in llm_units:
            if any(s.index == idx for s in u.steps):
                return u
    low = problem.lower()
    for u in reversed(llm_units):
        if any(re.search(r"\b" + re.escape(w.split(".")[1].lower()) + r"\b", low) for w in u.writes):
            return u
    return llm_units[-1]


# ---------------------------------------------------------------------------
# the loop: convert, check, repair what failed, check again
@dataclass
class RepairOutcome:
    ok: bool = False
    detail: str = ""                  # the last failure ("ok" once it passed)
    answers: int = 0                  # rounds: the first answers, then one per repair
    gate_runs: int = 0
    first_pass: bool = False
    stopped: str = ""                 # why it stopped short of a pass: attempts | budget | rules
    history: list[dict] = field(default_factory=list)
    results: list[UnitResult] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 6)

    @property
    def billed_usd(self) -> float:
        return round(sum(r.billed_usd for r in self.results), 6)


def convert_until_pass(conv: ConversionResult, provider: Provider, gate, *, original: str | None = None,
                       attempts: int = 3, mode: str = "runtime", effort: str | None = None,
                       budget_usd: float | None = None, log=None) -> tuple[ConversionResult, RepairOutcome]:
    """Convert the unsupported units with the LLM and run ``gate(notebook) -> (ok, detail)``.  While it fails, the unit
    to blame gets its own code back with the problem (the static guard's findings, or the gate's error or output
    diff) and answers again, up to ``attempts`` answers in all; repairs stop early once ``budget_usd`` is billed.
    Only LLM-written units are ever changed: a failure with no LLM unit to blame is the rules' to fix."""
    pending = units(conv)
    conv, results = fill_unsupported(conv, provider, original=original, mode=mode, effort=effort, log=log)
    by_unit = {u.index: (u, r) for u, r in zip(pending, results)}
    out = RepairOutcome(results=results)

    def check(repaired: int | None) -> tuple[bool, str]:
        rejected = [p for _, r in by_unit.values() if not r.ok for p in r.problems]
        if rejected:
            ok, detail = False, "rejected by the static guard: " + "; ".join(rejected)
        else:
            out.gate_runs += 1
            ok, detail = gate(conv.notebook)
        out.answers += 1
        out.history.append({"answer": out.answers, "repaired_unit": repaired,
                            "result": "PASS" if ok else "REJECTED" if rejected else "FAIL",
                            "problem": None if ok else detail[:1000]})
        return ok, detail

    ok, detail = check(None)
    out.first_pass = ok
    while not ok:
        if not pending:
            out.stopped = "rules"
            break
        if out.answers >= attempts:
            out.stopped = "attempts"
            break
        if budget_usd is not None and out.billed_usd >= budget_usd:
            out.stopped = "budget"
            break
        target = next((u for u, r in by_unit.values() if not r.ok), None) or unit_for_problem(detail, pending)
        u, rec = by_unit[target.index]
        rec.ok = repair_unit(provider, conv, u, detail, rec, original=original, mode=mode, effort=effort)
        if log:
            log(f"unit {u.index}: repaired ({'accepted' if rec.ok else 'rejected'} by the static guard)")
        ok, detail = check(target.index)
    out.ok, out.detail = ok, "ok" if ok else detail
    return conv, out
