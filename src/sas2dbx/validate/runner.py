"""Parity cases: convert a SAS program, run the notebook, compare outputs with ground truth."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, field

import yaml

from ..codegen.params import edit_param_notebook
from ..codegen.program import convert
from ..semantic.schema import Catalog
from .compare import CompareOptions, CompareResult, compare


@dataclass
class OutputSpec:
    dataset: str
    expected: str
    keys: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)
    mode: str = "auto"
    column_map: dict[str, str] = field(default_factory=dict)
    deviations: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)       # compare only these columns (both sides)


@dataclass
class ParityCase:
    name: str
    program: str
    libnames: dict[str, str] = field(default_factory=dict)
    outputs: list[OutputSpec] = field(default_factory=list)
    macro_vars: dict[str, str] = field(default_factory=dict)
    as_of: str | None = None
    decimals: int = 2
    include_paths: list[str] = field(default_factory=list)
    ground_truth: str = ""            # G1 | G2 | G3 | G4 (docs/ARCHITECTURE.md §5.8)
    description: str = ""
    allow_row_samples: bool = True
    param_module: str = ""            # the %INCLUDE'd parameter file -> parameter notebook (run with its defaults)
    param_values: dict = field(default_factory=dict)   # what a user typed into the parameter notebook (overrides)
    sas_reader: str = "auto"          # .sas7bdat libraries: auto | spark-sas7bdat | pyreadstat

    @classmethod
    def from_yaml(cls, path: str) -> "ParityCase":
        with open(path) as fh:
            d = yaml.safe_load(fh)
        base = os.path.dirname(os.path.abspath(path))

        def rel(p: str) -> str:
            if not p or os.path.isabs(p) or ":" in p.split("/")[0] and not p.startswith(("sasfiles:", "parquet:", "delta:")):
                return p
            if p.startswith(("sasfiles:", "parquet:", "delta:")):
                kind, _, rest = p.partition(":")
                return f"{kind}:{rel(rest)}"
            return os.path.normpath(os.path.join(base, p))

        outs = [OutputSpec(o["dataset"], rel(o["expected"]), o.get("keys", []), o.get("ignore", []), o.get("mode", "auto"),
                           o.get("column_map", {}) or {}, o.get("deviations", []) or [],
                           [c.upper() for c in o.get("columns", []) or []])
                for o in d.get("outputs", [])]
        return cls(
            name=d.get("name") or os.path.splitext(os.path.basename(path))[0],
            program=rel(d["program"]),
            libnames={k: (rel(v) if isinstance(v, str) else v) for k, v in (d.get("libnames") or {}).items()},
            outputs=outs,
            macro_vars={k: str(v) for k, v in (d.get("macro_vars") or {}).items()},
            as_of=str(d["as_of"]) if d.get("as_of") else None,
            decimals=int(d.get("decimals", 2)),
            include_paths=[rel(p) for p in d.get("include_paths", [])],
            ground_truth=d.get("ground_truth", ""),
            description=d.get("description", ""),
            allow_row_samples=bool(d.get("allow_row_samples", True)),
            param_module=rel(d["param_module"]) if d.get("param_module") else "",
            param_values=dict(d.get("param_values") or {}),
            sas_reader=d.get("sas_reader", "auto"),
        )


@dataclass
class CaseResult:
    case: str
    passed: bool
    conversion: dict
    execution: dict
    comparisons: list[CompareResult] = field(default_factory=list)
    notebook: str = ""
    evidence: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["comparisons"] = [c.to_dict() for c in self.comparisons]
        return d


def split_cells(source: str) -> list[str]:
    return source.split("\n# COMMAND ----------\n")


def execute_notebook(source: str, namespace: dict) -> dict:
    """Execute a Databricks .py notebook locally, cell by cell."""
    t0 = time.time()
    timings = []
    for i, cell in enumerate(split_cells(source)):
        lines = [ln for ln in cell.split("\n") if ln.strip() and not ln.startswith("# MAGIC") and ln.strip() != "# Databricks notebook source"]
        if not lines or all(ln.lstrip().startswith("#") for ln in lines):
            continue
        title = next((ln for ln in cell.split("\n") if ln.startswith("# DBTITLE")), f"cell {i}").replace("# DBTITLE 1,", "")
        tc = time.time()
        try:
            exec(compile(cell, f"<notebook cell {i}>", "exec"), namespace)  # noqa: S102 - running generated code is the point
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "cell": i, "title": title, "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=6), "seconds": round(time.time() - t0, 2), "cells": timings}
        timings.append({"cell": i, "title": title[:80], "seconds": round(time.time() - tc, 2)})
        if os.environ.get("SAS2DBX_TRACE"):
            print(f"  cell {i:>3} {timings[-1]['seconds']:>7.2f}s  {title[:70]}", flush=True)
    return {"ok": True, "seconds": round(time.time() - t0, 2), "cells": timings}


def _read_expected(spark, path: str):
    """Parquet ground truth (e.g. from an independent double-programming oracle)."""
    import pandas as pd

    from pyspark.sql import functions as F

    pdf = pd.read_parquet(path)
    if pdf.shape[1] == 0:
        return spark.range(0).select(F.lit(0).cast("long").alias("_sas_seq"))
    df = spark.createDataFrame(pdf)
    return df.withColumn("_sas_seq", F.monotonically_increasing_id())


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_catalog(libnames: dict[str, str], spark=None) -> Catalog:
    cat = Catalog()
    for lib, spec in libnames.items():
        if isinstance(spec, str) and spec.startswith("sasfiles:"):
            cat.harvest_dir(lib, spec[len("sasfiles:"):])
        elif isinstance(spec, str) and spec.startswith(("parquet:", "delta:")):
            cat.harvest_tables(lib, spec.split(":", 1)[1])
        elif isinstance(spec, str) and spec.startswith("uc:") and spark is not None:
            cat.harvest_uc(lib, spark, spec[len("uc:"):])
        elif isinstance(spec, str) and os.path.isdir(spec):
            cat.harvest_dir(lib, spec)
    return cat


def program_text(case: ParityCase) -> str:
    with open(case.program, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def convert_case(case: ParityCase, spark=None):
    """Convert the case's program against its libraries' schemas (``uc:`` libraries are read through ``spark``)."""
    return convert(program_text(case), file=os.path.basename(case.program), catalog=build_catalog(case.libnames, spark),
                   macro_vars=case.macro_vars, include_paths=case.include_paths, param_module=case.param_module or None)


def case_param_notebook(case: ParityCase, conv) -> str | None:
    """The parameter notebook as the case's user filled it in (its defaults when the case types no values)."""
    if conv.param_notebook is None:
        return None
    return edit_param_notebook(conv.param_notebook, case.param_values) if case.param_values else conv.param_notebook


def notebook_name(case: ParityCase) -> str:
    return re.sub(r"\W+", "_", os.path.splitext(os.path.basename(case.program))[0])


def write_notebooks(case: ParityCase, conv, out_dir: str, notebook: str | None = None) -> str:
    """The case's notebooks in one folder, laid out as deployed: the program %runs ./<parameter notebook>."""
    folder = os.path.join(out_dir, case.name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{notebook_name(case)}.py"), "w") as fh:
        fh.write(notebook or conv.notebook)
    params = case_param_notebook(case, conv)
    if params is not None:
        with open(os.path.join(folder, f"{conv.param_notebook_name}.py"), "w") as fh:
            fh.write(params)
    return folder


def compare_outputs(case: ParityCase, spark, get) -> list[CompareResult]:
    """Compare every output the case checks with its ground truth; ``get(dataset)`` returns it in SAS space."""
    from ..runtime.sasfile import read_sas_file

    comparisons: list[CompareResult] = []
    for o in case.outputs:
        opts = CompareOptions(keys=o.keys or None, decimals=case.decimals, mode=o.mode, ignore_columns=o.ignore,
                              column_map=o.column_map, allow_row_samples=case.allow_row_samples)
        try:
            actual = get(o.dataset)
        except KeyError as e:
            comparisons.append(CompareResult(o.dataset, False, "n/a", [], -1, -1, notes=[f"output not produced: {e}"]))
            continue
        if o.expected.lower().endswith((".xpt", ".sas7bdat")):
            expected, _, _, _ = read_sas_file(spark, o.expected, "pyreadstat")
        else:
            expected = _read_expected(spark, o.expected)
        if o.columns:
            keep = set(o.columns) | {"_SAS_SEQ"}
            actual = actual.select(*[c for c in actual.columns if c.upper() in keep])
            expected = expected.select(*[c for c in expected.columns if c.upper() in keep]) if expected.columns else expected
        res = compare(expected, actual, opts, name=o.dataset)
        res.notes.extend(f"documented deviation: {d}" for d in o.deviations)
        comparisons.append(res)
    return comparisons


def case_evidence(case: ParityCase, conv, notebook: str, spark, **extra) -> dict:
    return {
        "program_sha256": conv.sha256,
        "notebook_sha256": hashlib.sha256(notebook.encode()).hexdigest(),
        "expected_sha256": {o.dataset: _sha(o.expected) for o in case.outputs if os.path.exists(o.expected)},
        "spark_version": spark.version,
        "ansi_mode": spark.conf.get("spark.sql.ansi.enabled"),
        "decimals": case.decimals,
        "tolerance": 0.5 * 10 ** (-case.decimals),
        "ground_truth": case.ground_truth,
        **extra,
    }


def case_steps(conv) -> list[dict]:
    return [{"index": s.index, "kind": s.kind, "title": s.title, "origin": s.origin, "status": s.status,
             "tier": s.tier, "reasons": s.reasons, "hazards": sorted({h[0] for h in s.hazards})} for s in conv.steps]


def gate_detail(result: CaseResult) -> tuple[bool, str]:
    """A run's verdict and, for a failure, what whoever fixes it needs: the failing cell and its error, or each
    output's differences with a sample row (no sample values when the case sets ``allow_row_samples: false``)."""
    exe = result.execution
    if not exe.get("ok"):
        return False, f"execution failed in {exe.get('title')}: {str(exe.get('error', ''))[:800]}"
    bad = [c.summary() + (f" | sample {c.samples[0]}" if c.samples else "") for c in result.comparisons if not c.passed]
    return (not bad and bool(result.comparisons)), "; ".join(bad)[:2000] or "ok"


def execute_case(case: ParityCase, spark, conv, notebook: str) -> CaseResult:
    """Run one notebook for the case and compare its outputs with the ground truth: the parity gate."""
    from ..runtime.context import SasContext

    ctx = SasContext(spark, case.libnames, as_of=case.as_of, write_outputs=False, macro_vars=case.macro_vars,
                     sas_reader=case.sas_reader)
    ns = {"spark": spark, "ctx": ctx}
    parameters = None
    params = case_param_notebook(case, conv)
    if params is not None:      # the program %runs its parameter notebook: run it as the case's user filled it in
        exec(compile(params, f"{conv.param_notebook_name}.py", "exec"), pns := {})  # noqa: S102
        parameters = ns["PARAMETERS"] = pns["PARAMETERS"]
    exe = execute_notebook(notebook, ns)
    comparisons = compare_outputs(case, spark, ctx.get) if exe["ok"] else []
    passed = exe["ok"] and bool(comparisons) and all(c.passed for c in comparisons)
    evidence = case_evidence(case, conv, notebook, spark, parameters=parameters,
                             sas_readers={k: {"file": f, "reader": r} for k, (f, r) in ctx.readers.items()})
    return CaseResult(case.name, passed, conv.stats, exe, comparisons, notebook, evidence, case_steps(conv))


def run_case(case: ParityCase, spark, out_dir: str | None = None, notebook_override: str | None = None, *,
             provider=None, repairs: int = 2, effort: str | None = None, budget_usd: float | None = None,
             log=None) -> CaseResult:
    """Convert the case's program, run it and compare its outputs.  With an LLM ``provider``, the steps the rules
    leave unsupported are converted by the LLM, and a failing run goes back to it with the failure, up to
    ``repairs`` times (``llm.assist.convert_until_pass``).  The notebook written to ``out_dir`` is the last one
    the LLM produced, passing or not, so a person can take over from it."""
    from ..llm.assist import units

    conv = convert_case(case, spark)
    if provider is not None and notebook_override is None and units(conv):
        result = _run_with_llm(case, spark, conv, provider, repairs, effort, budget_usd, log)
        notebook = conv.notebook
    else:
        notebook = notebook_override or conv.notebook
        result = execute_case(case, spark, conv, notebook)
    if out_dir:
        write_notebooks(case, conv, out_dir, notebook)
        result.notebook = ""
    return result


def _run_with_llm(case: ParityCase, spark, conv, provider, repairs: int, effort, budget_usd, log) -> CaseResult:
    from ..llm.assist import convert_until_pass

    runs: list[CaseResult] = []

    def gate(notebook: str) -> tuple[bool, str]:
        runs.append(execute_case(case, spark, conv, notebook))
        return gate_detail(runs[-1])

    conv, out = convert_until_pass(conv, provider, gate, original=program_text(case), attempts=1 + max(0, repairs),
                                   effort=effort, budget_usd=budget_usd, log=log)
    result = runs[-1] if runs else CaseResult(      # every answer was rejected by the static guard: nothing ran
        case.name, False, conv.stats, {"ok": False, "title": "static guard", "error": out.detail}, [], conv.notebook,
        case_evidence(case, conv, conv.notebook, spark), case_steps(conv))
    result.passed = out.ok
    result.evidence["llm_repair"] = {
        "provider": getattr(provider, "name", None), "model": getattr(provider, "model", None),
        "answers": out.answers, "max_answers": 1 + max(0, repairs), "first_pass": out.first_pass,
        "passed_on_answer": out.answers if out.ok else None, "stopped": out.stopped or None,
        "units": [r.unit for r in out.results], "llm_calls": sum(r.attempts for r in out.results),
        "cost_usd": out.cost_usd, "billed_usd": out.billed_usd,
        "tokens_in": sum(r.tokens_in for r in out.results), "tokens_out": sum(r.tokens_out for r in out.results),
        "history": out.history,
    }
    return result


def write_result(result: CaseResult, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
