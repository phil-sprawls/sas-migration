"""Estate analysis: inventory, construct census, hazards, rule coverage, lineage and waves."""
from __future__ import annotations

import glob
import hashlib
import os
import re
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from ..codegen.exprs import NUMERIC_AGG_FUNCS, ExprCompiler, _SIMPLE_NUM1
from ..codegen.program import convert
from ..lang import ast as A
from ..semantic.schema import Catalog

SUPPORTED_PROCS = {"SORT", "SQL", "FREQ", "MEANS", "SUMMARY", "TRANSPOSE", "FORMAT", "PRINT", "CONTENTS", "DATASETS",
                   "DELETE", "APPEND"}


def supported_functions() -> set[str]:
    names = {n[3:].upper() for n in dir(ExprCompiler) if n.startswith("_f_")}
    return names | set(NUMERIC_AGG_FUNCS) | set(_SIMPLE_NUM1) | {"LAG", "DIF"} | {f"LAG{i}" for i in range(1, 100)} | {f"DIF{i}" for i in range(1, 100)}


@dataclass
class ProgramAnalysis:
    path: str
    sha256: str = ""
    lines: int = 0
    code_lines: int = 0
    category: str = "program"                 # program | macro-library | empty
    steps: dict = field(default_factory=dict)
    procs: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    unsupported_functions: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    librefs: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    hazards: dict = field(default_factory=dict)
    conversion: dict = field(default_factory=dict)
    unsupported: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    parse_errors: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)   # values from outside the program: undefined macro vars, run values
    complexity: float = 0.0
    error: str = ""


_MACRO_PATTERNS = {
    "definitions": r"%macro\s+\w+",
    "do_loops": r"%do\b",
    "if_statements": r"%if\b",
    "indirect_refs": r"&&",
    "symput": r"call\s+symputx?\b",
    "into": r"\binto\s*:",
    "includes": r"%inc(lude)?\b",
    "sysfunc": r"%q?sysfunc\b",
    "lets": r"%let\b",
    "execute": r"call\s+execute\b",
}


def analyze_program(path: str, text: str | None = None, *, include_paths=(), catalog: Catalog | None = None) -> ProgramAnalysis:
    pa = ProgramAnalysis(path)
    try:
        if text is None:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        pa.sha256 = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        raw_lines = text.splitlines()
        pa.lines = len(raw_lines)
        no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        pa.code_lines = sum(1 for ln in no_block.splitlines() if ln.strip() and not ln.strip().startswith("*"))
        low = no_block.lower()
        pa.macro = {k: len(re.findall(v, low)) for k, v in _MACRO_PATTERNS.items()}
        conv = convert(text, file=os.path.basename(path), catalog=catalog or Catalog(), include_paths=include_paths)
        pa.diagnostics = dict(Counter(d.code for d in conv.diagnostics))
        pa.parse_errors = list(conv.parse_errors)[:10]
        steps = [s for s in conv.steps if s.kind != "GLOBAL"]
        pa.steps = dict(Counter(s.kind.split()[0] for s in steps))
        pa.procs = dict(Counter(s.kind.split()[1] for s in steps if s.kind.startswith("PROC ")))
        pa.conversion = conv.stats
        pa.hazards = dict(Counter(h[0] for s in conv.steps for h in s.hazards))
        pa.unsupported = [{"step": s.index, "kind": s.kind, "origin": s.origin, "reason": "; ".join(s.reasons)}
                          for s in conv.steps if s.status == "unsupported"]
        pa.reads = sorted({r for s in conv.steps for r in s.reads})
        pa.writes = sorted({w for s in conv.steps for w in s.writes})
        pa.librefs = sorted({k.split(".")[0] for k in pa.reads + pa.writes if not k.startswith("WORK.")})
        pa.inputs = {"undefined_macro_vars": conv.unresolved, "run_time_values": conv.runtime_inputs,
                     "includes": conv.includes}
        funcs = Counter()
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", conv.expanded):
            funcs[m.group(1).upper()] += 1
        sup = supported_functions()
        known_sas = _known_sas_functions()
        pa.functions = {k: v for k, v in funcs.items() if k in known_sas}
        pa.unsupported_functions = {k: v for k, v in pa.functions.items() if k not in sup}
        if not steps:
            pa.category = "macro-library" if pa.macro.get("definitions") else "empty"
        pa.complexity = _complexity(pa)
    except Exception as e:  # noqa: BLE001 - analysis must never stop an estate scan
        pa.error = f"{type(e).__name__}: {e}"
        pa.parse_errors.append(traceback.format_exc(limit=3))
    return pa


def _complexity(pa: ProgramAnalysis) -> float:
    m = pa.macro
    score = pa.code_lines / 50
    score += 2 * m.get("definitions", 0) + 1.5 * m.get("do_loops", 0) + m.get("if_statements", 0)
    score += 3 * m.get("indirect_refs", 0) + 3 * (m.get("symput", 0) + m.get("into", 0)) + 5 * m.get("execute", 0)
    score += sum(pa.hazards.values()) * 0.5 + 4 * len(pa.unsupported)
    return round(score, 1)


def _known_sas_functions() -> set[str]:
    from ..runtime import pyfuncs  # noqa: F401 - imported for completeness of the name list

    base = supported_functions()
    extra = {"RAND", "RANUNI", "RANNOR", "RANBIN", "STREAMINIT", "SYMGET", "RESOLVE", "VVALUE", "VLABEL", "VNAME",
             "VTYPE", "EXIST", "OPEN", "CLOSE", "ATTRN", "VARNUM", "FETCH", "FETCHOBS", "GETVARC", "GETVARN", "DIM",
             "HBOUND", "LBOUND", "NWKDOM", "HOLIDAY", "JULDATE", "DATEJUL", "WEEK", "ZIPSTATE", "PROBNORM", "PROBIT",
             "PROBT", "TINV", "CINV", "FINV", "PDF", "CDF", "QUANTILE", "LOGPDF", "LGAMMA", "GAMMA", "BETA", "FACT",
             "COMB", "PERM", "ARCOS", "ARSIN", "ATAN", "COS", "SIN", "TAN", "SOUNDEX", "SPEDIS", "COMPGED", "COMPLEV",
             "PRXPARSE", "PRXPOSN", "PRXSUBSTR", "PRXNEXT", "CHOOSEC", "CHOOSEN", "WHICHC", "WHICHN", "MD5", "SHA256",
             "UUIDGEN", "PATHNAME", "GETOPTION", "SLEEP", "FILEEXIST", "FEXIST", "FILENAME", "SYSGET", "LARGEST",
             "SMALLEST", "ORDINAL", "IQR", "PCTL", "KURTOSIS", "SKEWNESS", "CSS", "USS", "CV", "STDERR", "SUMABS",
             "SIGN", "TRUNC", "FUZZ", "BAND", "BOR", "BXOR", "BNOT", "COLLATE", "ANYCNTRL", "ANYPRINT", "ANYGRAPH",
             "NOTPRINT", "NOTUPPER", "NOTLOWER", "NOTPUNCT", "KSCAN", "KSUBSTR", "KUPCASE", "KLENGTH", "KINDEX",
             "SUBSTRN", "TRANSTRN", "IFC", "IFN", "INPUTC", "INPUTN", "PUTC", "PUTN", "COUNTW", "CATQ"}
    return base | extra


# ---------------------------------------------------------------------------
@dataclass
class EstateReport:
    programs: list[ProgramAnalysis]
    summary: dict
    lineage: dict
    waves: list[list[str]]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "lineage": self.lineage, "waves": self.waves,
                "programs": [asdict(p) for p in self.programs]}


def analyze_estate(root: str, *, include_paths=None, catalog: Catalog | None = None, pattern: str = "**/*.sas",
                   log=None) -> EstateReport:
    paths = sorted(set(glob.glob(os.path.join(root, pattern), recursive=True)))
    if include_paths is None:
        include_paths = sorted({os.path.dirname(p) for p in paths})
    progs = []
    for i, p in enumerate(paths, start=1):
        progs.append(analyze_program(p, include_paths=include_paths, catalog=catalog))
        if log:
            log(f"[{i}/{len(paths)}] {os.path.relpath(p, root)}: {progs[-1].conversion.get('auto_converted_pct', '-')}%")
    return build_report(progs, root)


def build_report(progs: list[ProgramAnalysis], root: str = "") -> EstateReport:
    runnable = [p for p in progs if p.category == "program" and not p.error]
    total_steps = sum(p.conversion.get("steps", 0) for p in runnable)
    status = Counter()
    for p in runnable:
        for k, v in p.conversion.get("by_status", {}).items():
            status[k] += v
    reasons = Counter()
    for p in runnable:
        for u in p.unsupported:
            reasons[_reason_family(u["reason"])] += 1
    fun_unsup = Counter()
    for p in progs:
        for k, v in p.unsupported_functions.items():
            fun_unsup[k] += v
    procs = Counter()
    for p in progs:
        for k, v in p.procs.items():
            procs[k] += v
    hazards = Counter()
    for p in progs:
        for k, v in p.hazards.items():
            hazards[k] += v
    fully = sum(1 for p in runnable if p.conversion.get("by_status", {}).get("unsupported", 0) == 0)
    undefined = Counter(n for p in runnable for n in p.inputs.get("undefined_macro_vars", []))
    run_values = Counter(n for p in runnable for n in p.inputs.get("run_time_values", []))
    summary = {
        "root": root,
        "files": len(progs),
        "programs": len(runnable),
        "macro_libraries": sum(1 for p in progs if p.category == "macro-library"),
        "errors": [{"path": p.path, "error": p.error} for p in progs if p.error],
        "lines": sum(p.lines for p in progs),
        "code_lines": sum(p.code_lines for p in progs),
        "steps": total_steps,
        "steps_by_status": dict(status),
        "step_auto_conversion_pct": round(100 * (status["converted"] + status["display"] + status.get("noop", 0)) / total_steps, 1) if total_steps else None,
        "programs_fully_converted": fully,
        "programs_fully_converted_pct": round(100 * fully / len(runnable), 1) if runnable else None,
        "unsupported_reasons": reasons.most_common(25),
        "unsupported_functions": fun_unsup.most_common(25),
        "procs": procs.most_common(),
        "unsupported_procs": [(k, v) for k, v in procs.most_common() if k not in SUPPORTED_PROCS],
        "hazards": hazards.most_common(),
        # the parameter catalog to confirm: each name -> number of programs (declare with --param or --define)
        "external_inputs": {
            "programs_using_undefined_macro_vars": sum(1 for p in runnable if p.inputs.get("undefined_macro_vars")),
            "undefined_macro_vars": undefined.most_common(50),
            "run_time_values": run_values.most_common(),
        },
    }
    summary["parameter_modules"] = _parameter_modules(progs)
    lineage, waves = _lineage(runnable)
    return EstateReport(progs, summary, lineage, waves)


def _parameter_modules(progs: list[ProgramAnalysis]) -> list[dict]:
    """%INCLUDE'd files made only of %LET statements: candidates for --param-module (a parameter notebook)."""
    from ..codegen.params import read_param_module

    by_name = {os.path.basename(p.path).lower(): p.path for p in progs}
    users: dict[str, list[str]] = defaultdict(list)
    for p in progs:
        for inc in p.inputs.get("includes", []):
            users[os.path.basename(inc.replace("\\", "/")).lower()].append(p.path)
    out = []
    for name, programs in sorted(users.items()):
        path = by_name.get(name)
        if path is None:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                specs, notes = read_param_module(fh.read(), name)
        except OSError:
            continue
        if specs and not notes:
            out.append({"module": path, "parameters": [s.lower() for s in specs], "included_by": sorted(set(programs))})
    return out


def _reason_family(reason: str) -> str:
    m = re.match(r"run-time macro logic: (RUNTIME_\w+|UNRESOLVED_MACRO_CONTROL|MACRO_EVAL_ERROR)", reason)
    if m:
        what = re.search(r"%sysfunc\((\w+)", reason, re.I)
        return f"run-time macro logic ({m.group(1)}{', %SYSFUNC(' + what.group(1).upper() + ')' if what else ''})"
    if "which nothing defines" in reason:
        return "uses macro variables nothing defines (values supplied from outside?)"
    if reason.startswith(("generated code is not valid Python", "nesting deeper than")):
        return "construct too large (" + reason.split(":")[0] + ")"
    if reason.startswith("converter error"):
        return "converter error (" + reason.split(": ")[1] + ")"
    if reason.startswith("parse error"):
        return "parse error: " + re.sub(r"\(near .*\)$", "", re.sub(r"^parse error: line \d+: ", "", reason)).strip()
    r = re.sub(r"^line \d+: ", "", reason)
    r = re.sub(r"'[^']*'", "'…'", r)
    r = re.sub(r"\b[A-Z_][A-Z0-9_]{2,}\b(?=[^()]*$)", lambda m: m.group(0) if m.group(0) in ("PROC", "DATA", "SET", "MERGE", "SQL") else "X", r)
    return r[:110]


def _lineage(progs: list[ProgramAnalysis]):
    producers = defaultdict(set)
    for p in progs:
        for w in p.writes:
            if not w.startswith("WORK."):
                producers[w].add(p.path)
    deps = {p.path: set() for p in progs}
    for p in progs:
        for r in p.reads:
            for prod in producers.get(r, ()):
                if prod != p.path:
                    deps[p.path].add(prod)
    waves: list[list[str]] = []
    done: set[str] = set()
    remaining = set(deps)
    while remaining:
        ready = sorted(p for p in remaining if deps[p] <= done)
        if not ready:
            waves.append(sorted(remaining))  # cycle: convert together
            break
        waves.append(ready)
        done |= set(ready)
        remaining -= set(ready)
    lineage = {"datasets": {k: sorted(v) for k, v in producers.items()},
               "dependencies": {k: sorted(v) for k, v in deps.items() if v}}
    return lineage, waves
