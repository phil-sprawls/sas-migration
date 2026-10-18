"""SAS program -> Databricks notebook (+ conversion report)."""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import threading
import tokenize as pytokenize
from dataclasses import dataclass, field
from datetime import datetime

from .. import __version__
from ..lang import ast as A
from ..lang.lexer import RT_CLOSE, RT_OPEN, Kind
from ..lang.lexer import tokenize as sas_tokenize
from ..lang.parser import parse
from ..macro.processor import RUNTIME_AUTOMATIC, MacroProcessor
from ..semantic.schema import Catalog
from .datastep import DataStepCompiler, StepCode
from .params import default_value, read_param_module, render_param_notebook, resolve_kind
from .procs import ProcCompiler


@dataclass
class StepReport:
    index: int
    kind: str
    title: str
    origin: str
    status: str
    tier: str = ""
    reasons: list[str] = field(default_factory=list)
    hazards: list[tuple[str, str, int]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    sas_text: str = ""
    code: list[str] = field(default_factory=list)
    provenance: str = "rules"            # rules | llm | manual


@dataclass
class ConversionResult:
    file: str
    sha256: str
    notebook: str
    steps: list[StepReport]
    diagnostics: list
    parse_errors: list[str]
    catalog: Catalog
    expanded: str
    params: list[str] = field(default_factory=list)   # run-time parameters the notebook reads from the job
    unresolved: list[str] = field(default_factory=list)       # macro variables used but defined nowhere
    includes: list[str] = field(default_factory=list)         # %INCLUDE targets as written
    runtime_inputs: list[str] = field(default_factory=list)   # values of the run: SYSPARM, SYSDATE9, %SYSGET, ...
    param_notebook: str | None = None    # the parameter notebook end users edit (from a parameter module)
    param_notebook_name: str = ""        # it is %run by name, next to the program's notebook
    param_kinds: dict[str, str] = field(default_factory=dict)   # parameter -> text|number|date|filter|values|columns

    @property
    def stats(self) -> dict:
        data_steps = [s for s in self.steps if s.kind not in ("GLOBAL",)]
        by_status: dict[str, int] = {}
        for s in data_steps:
            by_status[s.status] = by_status.get(s.status, 0) + 1
        conv = by_status.get("converted", 0) + by_status.get("display", 0) + by_status.get("noop", 0)
        return {
            "steps": len(data_steps),
            "by_status": by_status,
            "auto_converted_pct": round(100.0 * conv / len(data_steps), 1) if data_steps else 100.0,
            "hazards": sorted({h[0] for s in self.steps for h in s.hazards}),
        }


class ProgramState:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.user_formats: dict[str, str] = {}
        self.runtime_macros: set[str] = set()
        self.last_dataset: str | None = None
        self.libnames: dict[str, str] = {}
        self.infer_schemas = True
        self.pending_ods: dict = {}
        self.params: list[str] = []          # run-time parameters: read from Databricks job/notebook parameters
        self.param_specs: dict = {}          # parameter module: name -> ParamSpec (today's value)
        self.param_kinds: dict[str, set[str]] = {}
        self.param_columns: dict[str, dict] = {}   # filter parameter -> {COLUMN: (type, format)} it filters
        self._id = 0

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def note_param(self, name: str, kind: str, schema=None) -> bool:
        """Record how a parameter-module variable is used; False when ``name`` is not one."""
        key = name.upper()
        if key not in self.param_specs:
            return False
        self.param_kinds.setdefault(key, set()).add(kind)
        if schema is not None and key not in self.param_columns:
            self.param_columns[key] = {v.name.upper(): (v.type, v.format) for v in schema.vars}
        return True


def convert(text: str, *, file: str = "program.sas", catalog: Catalog | None = None,
            macro_vars: dict[str, str] | None = None, include_paths=(), now: datetime | None = None,
            comment_macros: str = "skip", params=(), param_module: str | None = None,
            param_module_text: str | None = None) -> ConversionResult:
    """Convert one SAS program.

    ``macro_vars`` are fixed at conversion time (environment constants: paths, library names).  ``params`` are
    macro variables the program receives from outside when it runs (prompts, -SYSPARM): they stay run-time values
    read by ``ctx.symget`` from the job's parameters, so one notebook serves every value.  ``param_module`` is the
    %INCLUDE'd file of %LET statements end users edit: it becomes a separate parameter notebook
    (``result.param_notebook``) that the program %runs, with filters in Spark SQL and lists as Python lists.  A
    macro variable that is neither defined by the program nor given here makes the steps that use it unsupported -
    it is never frozen to a guess.
    """
    return _deep(_convert, text, file, catalog, macro_vars, include_paths, now, comment_macros, params,
                 param_module, param_module_text)


_STACK_BYTES = 512 * 1024 * 1024
_RECURSION_LIMIT = 100_000
_STACK_LOCK = threading.Lock()


def _deep(fn, *args):
    """Run the converter with room for very large constructs: parsing and compiling recurse once per level, and
    enterprise code has 1,000+ branch ELSE IF chains and 1,000+ term expressions (often on one very long line)."""
    if sys.getrecursionlimit() < _RECURSION_LIMIT:
        sys.setrecursionlimit(_RECURSION_LIMIT)
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = fn(*args)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e

    with _STACK_LOCK:
        old = threading.stack_size(_STACK_BYTES)
        try:
            worker = threading.Thread(target=run, name="sas2dbx-convert", daemon=True)
            worker.start()
        finally:
            threading.stack_size(old)
    worker.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _convert(text, file, catalog, macro_vars, include_paths, now, comment_macros, params, param_module,
             param_module_text) -> ConversionResult:
    catalog = catalog or Catalog()
    specs, module_notes = {}, []
    if param_module or param_module_text is not None:
        param_module = param_module or "params.sas"
        if param_module_text is None:
            with open(param_module, encoding="utf-8", errors="replace") as fh:
                param_module_text = fh.read()
        specs, module_notes = read_param_module(param_module_text, os.path.basename(param_module))
    mp = MacroProcessor(include_paths=include_paths, symbols=macro_vars, now=now, comment_macros=comment_macros,
                        runtime_vars={p.upper() for p in params} | set(specs),
                        param_modules={os.path.basename(param_module)} if param_module else None)
    exp = mp.expand(text, file)
    prog = parse(exp.text)
    state = ProgramState(catalog)
    state.params = sorted({p.upper() for p in params})
    state.param_specs = specs
    expanded_lines = exp.text.split("\n")

    def source_of(line: int) -> str:
        return expanded_lines[line - 1] if 0 < line <= len(expanded_lines) else ""

    def origin(step: A.Step) -> str:
        a = exp.origin_at(step.start)
        b = exp.origin_at(max(step.end - 1, step.start))
        loc = f"{os.path.basename(a.file)}:{a.line}" + (f"-{b.line}" if b.line != a.line and b.file == a.file else "")
        if a.stack:
            loc += f" via {' > '.join(a.stack)}"
        return loc

    reports: list[StepReport] = []
    for i, step in enumerate(prog.steps, start=1):
        sas_text = exp.text[step.start:step.end] if step.end > step.start else ""
        try:
            if isinstance(step, A.DataStep):
                outs = " ".join(o.display for o in step.outputs)
                code = DataStepCompiler(step, state, source_of).compile()
                rep = _report(i, "DATA", f"DATA {outs}", origin(step), code, sas_text)
                if code.writes:
                    state.last_dataset = code.writes[-1]
            elif isinstance(step, A.ProcStep):
                code = ProcCompiler(step, state).compile()
                title = f"PROC {step.name} {step.options_raw.strip()}"
                rep = _report(i, f"PROC {step.name}", " ".join(title.split())[:120], origin(step), code, sas_text)
                if code.writes:
                    state.last_dataset = code.writes[-1]
            elif isinstance(step, A.GlobalStmt):
                rep = _global(i, step, state, origin(step), sas_text)
            elif isinstance(step, A.Unresolved) and step.message.startswith("parse error"):
                rep = StepReport(i, "UNPARSED", "code the parser could not read", origin(step), "unsupported",
                                 reasons=[step.message], sas_text=sas_text)
            elif isinstance(step, A.Unresolved):
                rep = StepReport(i, "MACRO", "macro logic that depends on run-time data", origin(step), "unsupported",
                                 reasons=step.message.split("\n"), sas_text=sas_text)
            else:  # pragma: no cover
                continue
        except Exception as e:  # noqa: BLE001 - one step must never take the program (or an estate batch) down
            kind = "DATA" if isinstance(step, A.DataStep) else f"PROC {step.name}" if isinstance(step, A.ProcStep) else "GLOBAL"
            if isinstance(e, RecursionError):
                title, why = "construct too large to convert", f"nesting deeper than the converter's limit ({_RECURSION_LIMIT:,} levels)"
            else:
                title, why = "converter error", f"converter error (a sas2dbx bug to report): {type(e).__name__}: {str(e)[:160]}"
            rep = StepReport(i, kind, title, origin(step), "unsupported", reasons=[why], sas_text=sas_text)
        if _MARKER in sas_text and rep.status == "unsupported":
            # run-time macro logic inside the step: name the real cause, not the parse/compile symptom
            rep.reasons = [f"run-time macro logic: {m.replace(chr(39) * 2, chr(39))}" for m in _MARKER_RE.findall(sas_text)] \
                + [r for r in rep.reasons if "__sas2dbx_unresolved__" not in r.lower()]
            if rep.kind == "UNPARSED":
                rep.kind, rep.title = "MACRO", "macro logic that depends on run-time data"
        if rep.status != "unsupported":
            rep.code = _bound_plan_depth(rep.code)
            _check_converted(rep, exp.unresolved)
        reports.append(rep)

    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    used = {"SYSPARM"} & state.runtime_macros       # -SYSPARM is an input of the run like any parameter
    run_values = sorted(n for n in state.runtime_macros if n in RUNTIME_AUTOMATIC or n.startswith(("SYSGET_", "SYSFUNC_")))
    result = ConversionResult(file, sha, "", reports, exp.diagnostics, prog.errors, catalog, exp.text,
                              params=sorted(set(state.params) | used), unresolved=sorted(exp.unresolved),
                              runtime_inputs=run_values, includes=list(exp.includes))
    if specs:
        defaults = {}
        for name, spec in specs.items():
            kind, note = resolve_kind(state.param_kinds.get(name))
            result.param_kinds[name] = kind
            value, why = default_value(spec, kind, state.param_columns.get(name))
            defaults[name] = (value, "; ".join(x for x in (note, why) if x))
        result.param_notebook_name = os.path.splitext(os.path.basename(param_module))[0]
        result.param_notebook = render_param_notebook(param_module, file, specs, result.param_kinds, defaults, module_notes)
    result.notebook = render_notebook(result, state)
    return result


_MARKER = "__sas2dbx_unresolved__"
_MARKER_RE = re.compile(r"__sas2dbx_unresolved__\s*'((?:[^']|'')*)'", re.I)
_AMP_REF_RE = re.compile(r"&([A-Za-z_][A-Za-z0-9_]*)")
_RT_ESCAPES = tuple(f"\\u{ord(c):04x}" for c in (RT_OPEN, RT_CLOSE))


def _check_converted(rep: StepReport, unresolved: set[str]) -> None:
    """Last gate before a step counts as converted: nothing it depends on may be a guess."""
    refs = _unresolved_refs(rep.sas_text, unresolved)
    if refs:
        names = ", ".join("&" + r for r in refs)
        rep.status, rep.tier = "unsupported", ""
        rep.reasons = [f"uses {names}, which nothing defines: a value supplied from outside the program? "
                       "(define it with --define NAME=VALUE or declare a run-time parameter with --param NAME)"]
        return
    if _leaks_placeholder(rep.code):
        rep.status, rep.tier = "unsupported", ""
        rep.reasons = ["a run-time macro value is used in a position the converter does not handle yet"]
        return
    try:
        compile("\n".join(rep.code), f"<step {rep.index}>", "exec")
    except (SyntaxError, RecursionError, MemoryError, ValueError) as e:
        rep.status, rep.tier = "unsupported", ""
        rep.reasons = [f"generated code is not valid Python: {type(e).__name__}: {str(e)[:160]}"]


_PLAN_STEP = "df = df."
_MAX_CHAINED = 300


def _bound_plan_depth(code: list[str]) -> list[str]:
    """Cut the lineage every _MAX_CHAINED chained projections (macro-generated DATA steps run to thousands of
    statements; Spark analyses each as one more nested plan node and overflows the JVM stack near ~3,000)."""
    if sum(ln.startswith(_PLAN_STEP) for ln in code) <= _MAX_CHAINED:
        return code
    out: list[str] = []
    run = 0
    for k, ln in enumerate(code):
        out.append(ln)
        if not ln.startswith(_PLAN_STEP):
            continue
        run += 1
        nxt = code[k + 1] if k + 1 < len(code) else ""
        if run >= _MAX_CHAINED and nxt.startswith(_PLAN_STEP):   # between two complete top-level statements
            out.append("df = ctx.cut(df)  # bound the Spark plan depth")
            run = 0
    return out


def _unresolved_refs(sas_text: str, unresolved: set[str]) -> list[str]:
    """'&name' references left in the step (outside comments and single-quoted strings) to undefined variables."""
    if not unresolved or "&" not in sas_text:
        return []
    try:
        toks = sas_tokenize(sas_text)
    except Exception:  # noqa: BLE001 - unlexable text: fall back to a plain scan without single-quoted strings
        plain = re.sub(r"'(?:[^']|'')*'", "''", sas_text)
        return sorted({m.group(1).upper() for m in _AMP_REF_RE.finditer(plain) if m.group(1).upper() in unresolved})
    found: set[str] = set()
    for k, t in enumerate(toks):
        if t.kind is Kind.STRING and t.quote == '"':
            found |= {m.group(1).upper() for m in _AMP_REF_RE.finditer(t.value) if m.group(1).upper() in unresolved}
        elif t.kind is Kind.OP and t.value == "&" and k + 1 < len(toks):
            nxt = toks[k + 1]
            if nxt.kind is Kind.NAME and not nxt.ws_before and nxt.upper in unresolved:
                found.add(nxt.upper)
    return sorted(found)


def _leaks_placeholder(code: list[str]) -> bool:
    """True when a run-time macro placeholder reached generated code (outside comments) instead of ctx.symget."""
    src = "\n".join(code)
    if RT_OPEN not in src and RT_CLOSE not in src and not any(e in src for e in _RT_ESCAPES):
        return False
    try:
        for tok in pytokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != pytokenize.COMMENT and (RT_OPEN in tok.string or RT_CLOSE in tok.string
                                                    or any(e in tok.string for e in _RT_ESCAPES)):
                return True
    except (pytokenize.TokenError, SyntaxError):
        return True
    return False


def _report(i: int, kind: str, title: str, origin: str, code: StepCode, sas_text: str) -> StepReport:
    return StepReport(i, kind, title, origin, code.status, code.tier if code.status == "converted" else "",
                      code.reasons, code.hazards, code.reads, code.writes, sas_text, code.lines)


def _global(i: int, step: A.GlobalStmt, state: ProgramState, origin: str, sas_text: str) -> StepReport:
    kw = step.keyword
    lines: list[str] = []
    status = "noop"
    reasons: list[str] = []
    if kw == "LIBNAME":
        toks = step.raw.split()
        if toks:
            state.libnames[toks[0].upper()] = step.raw
            lines.append(f"# LIBNAME {' '.join(step.raw.split())[:100]}")
            lines.append(f"#   -> map libref {toks[0].upper()} in the context configuration (e.g. a Unity Catalog schema)")
    elif kw in ("OPTIONS", "OPTION", "TITLE", "FOOTNOTE", "ODS", "GOPTIONS", "DM", "RESETLINE", "PAGE", "SKIP", "LOCK") \
            or kw.startswith(("TITLE", "FOOTNOTE", "AXIS", "SYMBOL", "LEGEND", "PATTERN")):
        lines.append(f"# {kw} {' '.join(step.raw.split())[:100]}  (no effect on data)")
    elif kw == "ENDSAS":
        lines.append("# ENDSAS: the SAS program stops here")
    elif kw == "FILENAME":
        lines.append(f"# FILENAME {' '.join(step.raw.split())[:100]}  -> map to a Unity Catalog volume path")
    elif kw == "X":
        status = "unsupported"
        reasons.append("X command (operating-system command)")
    elif kw == "":
        status = "noop"
    else:
        status = "unsupported"
        reasons.append(f"global statement {kw}")
    return StepReport(i, "GLOBAL", ' '.join(f"{kw} {step.raw}".split())[:80], origin, status, reasons=reasons, sas_text=sas_text, code=lines)


# ---------------------------------------------------------------------------
# notebook rendering
def _md(text: str) -> list[str]:
    return [f"# MAGIC {ln}" if ln else "# MAGIC" for ln in text.split("\n")]


def render_notebook(result: ConversionResult, state: ProgramState) -> str:
    st = result.stats
    out: list[str] = ["# Databricks notebook source"]
    rows = []
    for s in result.steps:
        if s.kind == "GLOBAL" and s.status == "noop":
            continue
        hz = ", ".join(sorted({h[0] for h in s.hazards}))
        rows.append(f"| {s.index} | {s.title.replace('|', '/')} | {s.origin} | {s.status} | {hz} |")
    header = (
        f"# `{os.path.basename(result.file)}` → PySpark\n"
        f"Converted by **sas2dbx {__version__}** · source SHA-256 `{result.sha256[:16]}…`\n\n"
        f"**{st['steps']} steps** — {st['by_status']} — auto-converted {st['auto_converted_pct']}%\n\n"
        "Every cell shows the SAS it came from. Cells marked *unsupported* raise until converted (LLM assist or manual) "
        "and validated for 2-decimal output parity.\n\n"
        "| # | Step | SAS source | Status | Semantic hazards handled |\n|---|---|---|---|---|\n" + "\n".join(rows)
    )
    out += _md("%md\n" + header)
    if result.param_notebook is not None:
        # like %INCLUDE today: end users edit only the parameter notebook, which defines PARAMETERS
        out += ["", "# COMMAND ----------", "", f"# MAGIC %run ./{result.param_notebook_name}"]
    out += ["", "# COMMAND ----------", "",
            "from pyspark.sql import functions as F",
            "from pyspark.sql import Window as W",
            "from sas2dbx.runtime import S, P, SasContext, context_from_widgets",
            "",
            "if \"ctx\" not in globals():"]
    if result.params:
        out.append("    # run-time parameters: Databricks job/notebook parameters of the same name (required)")
        out.append(f"    ctx = context_from_widgets(spark, params={result.params!r})")
    else:
        out.append("    ctx = context_from_widgets(spark)  # librefs -> Unity Catalog, frozen clock for validation runs")
    if result.param_notebook is not None:
        spec = {k: result.param_kinds[k] for k in result.param_kinds}
        out.append(f"PARAM_SPEC = {spec!r}")
        out.append("ctx.load_params(PARAM_SPEC, globals().get(\"PARAMETERS\"))  # checks every value before reading data")
    if state.runtime_macros:
        out.append(f"# run-time macro variables used: {', '.join(sorted(state.runtime_macros))}")
    for s in result.steps:
        if s.kind == "GLOBAL" and s.status == "noop" and not s.code:
            continue
        out += ["", "# COMMAND ----------", ""]
        out.append(f"# DBTITLE 1,[{s.index}] {' '.join(s.title.split())[:90]}  ({s.origin})")
        if s.sas_text.strip():
            out.append("# ---- SAS " + "-" * 60)
            for ln in s.sas_text.strip("\n").split("\n")[:60]:
                out.append(("# " + ln.rstrip()).rstrip())
            if s.sas_text.count("\n") > 60:
                out.append("# ... (truncated)")
            out.append("# " + "-" * 69)
        if s.status == "unsupported":
            reasons = "; ".join(s.reasons) or "not supported"
            out.append(f"raise NotImplementedError({('sas2dbx: step ' + str(s.index) + ' needs conversion: ' + reasons)!r})")
        else:
            out += s.code
    return "\n".join(out) + "\n"
