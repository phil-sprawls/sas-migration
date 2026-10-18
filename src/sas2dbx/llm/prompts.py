"""Prompts for LLM-assisted conversion.  The system prompt is stable (and cached): it is
generated from the runtime library's own signatures and the hazard registry, so it
always matches the code that will execute."""
from __future__ import annotations

import inspect
import json

from ..semantic.hazards import HAZARDS
from ..semantic.schema import Schema

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source for the notebook cell(s)."},
        "outputs": {"type": "array", "items": {"type": "string"}, "description": "Data sets written with ctx.put."},
        "notes": {"type": "string", "description": "Assumptions and SAS behaviours you had to decide."},
    },
    "required": ["code", "outputs", "notes"],
    "additionalProperties": False,
}

_CONTRACT = """\
## Execution contract
Your code runs as a Databricks notebook cell (Python) with these names already defined:
`spark` (SparkSession), `F` (pyspark.sql.functions), `W` (pyspark.sql.Window), `ctx` (data-set context){runtime_names}.

* Read every SAS data set with `ctx.get("lib.name")` (one-level names are WORK: `ctx.get("name")`).
* Write every SAS output data set with `ctx.put("lib.name", df, meta={{"VAR": {{"type": "num"|"char", "length": n, "format": "DATE9.", "label": "..."}}}})`.
  Give `meta` for every variable the step creates or changes; existing columns keep their metadata automatically.
* `ctx.get` returns data in **SAS space**, and outputs must be in SAS space too:
  - numeric variables are DOUBLE; a SAS missing value is NULL;
  - character variables are STRING, right-trimmed, and a blank/missing value is `''` (never NULL);
  - SAS dates are DOUBLE days since 1960-01-01, datetimes DOUBLE seconds since 1960-01-01T00:00:00,
    times seconds since midnight (formats in `meta` say which is which);
  - every DataFrame carries `_sas_seq` (BIGINT), the SAS row order. Keep it on outputs, and when the SAS order changes
    (sorting, BY-group output, MERGE) give the output a new `_sas_seq` that is increasing in the SAS order
    (e.g. `df.orderBy(...).withColumn("_sas_seq", F.monotonically_increasing_id())`).
* Variable names: SAS names as written in the program (Spark resolves them case-insensitively); output columns in SAS PDV order.
* Run-time macro variables: `ctx.symput(name, text)` / `ctx.symget(name)` / `ctx.symget_num(name)`.
* Do not use the RDD API, `spark.conf.set`, `toPandas()`/`collect()` on unbounded data, or anything that writes outside `ctx`.
* The code must behave identically with `spark.sql.ansi.enabled` true or false (use `F.try_divide`, guarded casts).
"""

_GOAL = """\
You convert SAS programs to PySpark for Databricks. The converted code is accepted only if every output data set
matches what SAS produces: same rows, same columns, character values equal, numbers equal to 2 decimal places.
Reproduce SAS semantics exactly - including the behaviours listed under "SAS semantics that differ from Spark" -
even where SAS behaviour looks surprising. Do not "improve" the logic.
"""

_FORMAT = """\
## Answer format
Return JSON with `code` (the cell body), `outputs` (data sets written with ctx.put) and `notes` (assumptions).
"""


def hazard_catalog() -> str:
    lines = ["## SAS semantics that differ from Spark (handle all of these)"]
    for h in HAZARDS.values():
        lines.append(f"- **{h.title}** — SAS: {h.sas} Spark: {h.spark}")
    return "\n".join(lines)


def runtime_reference() -> str:
    """Signatures + summaries of S.* and P.* (generated from the runtime itself)."""
    from ..runtime import functions as S_mod
    from ..runtime import procs as P_mod
    from ..runtime import stats as stats_mod

    out = ["## Runtime library (preferred: these encode SAS semantics and are tested against SAS behaviour)",
           "`S` = SAS functions/operators returning Columns; `P` = SAS step semantics on DataFrames.", "", "### S"]

    def doc(f) -> str:
        d = (inspect.getdoc(f) or "").strip().splitlines()
        return f" — {d[0]}" if d else ""

    for name, f in sorted(vars(S_mod).items()):
        if name.startswith("_") or not inspect.isfunction(f) or f.__module__ != S_mod.__name__:
            continue
        try:
            sig = str(inspect.signature(f)).replace(" -> pyspark.sql.column.Column", "").replace(" -> Column", "")
        except (TypeError, ValueError):
            sig = "(...)"
        out.append(f"- `S.{name}{sig}`{doc(f)}")
    out.append("- `S.MISSING` numeric missing literal Column; `S.d('01JAN2020')` / `S.dt(...)` / `S.t(...)` SAS literals -> float")
    out += ["", "### P"]
    for mod in (P_mod, stats_mod):
        for name, f in sorted(vars(mod).items()):
            if name.startswith("_") or not inspect.isfunction(f) or f.__module__ != mod.__name__:
                continue
            try:
                sig = str(inspect.signature(f)).replace(" -> DataFrame", "").replace("pyspark.sql.dataframe.", "")
            except (TypeError, ValueError):
                sig = "(...)"
            out.append(f"- `P.{name}{sig}`{doc(f)}")
    return "\n".join(out)


_EXAMPLE = '''\
## Example
SAS:
```
data out; set work.vitals; by usubjid;
  retain n; if first.usubjid then n = 0; n + 1;
  if sysbp < 90 then flag = 'LOW'; else flag = 'OK';
  if last.usubjid then output;
run;
```
Cell:
```python
df = ctx.get("vitals")
df = P.first_last(df, by=["USUBJID"])
w = W.partitionBy("USUBJID").orderBy("_sas_seq").rowsBetween(W.unboundedPreceding, W.currentRow)
df = df.withColumn("n", F.lit(0.0) + F.sum(F.lit(1.0)).over(w))
df = df.withColumn("flag", F.when(S.lt(F.col("SYSBP"), 90.0), F.lit("LOW")).otherwise(F.lit("OK")))  # missing < 90 is TRUE in SAS
df = df.where(F.col("_sas_last_USUBJID"))
ctx.put("out", df.drop("_sas_first_USUBJID", "_sas_last_USUBJID"), meta={"n": {"type": "num", "length": 8}, "flag": {"type": "char", "length": 3}})
```
'''


def system_prompt(mode: str = "runtime") -> str:
    """mode: 'runtime' (use the sas2dbx runtime library) | 'plain' (PySpark only)."""
    runtime_names = ", `S` (SAS functions), `P` (SAS step semantics)" if mode == "runtime" else ""
    parts = [_GOAL, _CONTRACT.format(runtime_names=runtime_names), hazard_catalog()]
    if mode == "runtime":
        parts += [runtime_reference(), _EXAMPLE]
    else:
        parts.append("Use only PySpark (`spark`, `F`, `W`) and `ctx`; no helper library is available.")
    parts.append(_FORMAT)
    return "\n\n".join(parts)


def describe_schemas(schemas: dict[str, Schema]) -> str:
    lines = []
    for name, sch in schemas.items():
        cols = ", ".join(f"{v.name} {'$' + str(v.length) if v.type == 'char' else 'num'}"
                         + (f" {v.format}" if v.format else "") for v in sch.vars)
        flag = "" if sch.known else " (partial: inferred from usage)"
        lines.append(f"- `{name.lower()}`{flag}: {cols or '(unknown columns)'}")
    return "\n".join(lines) or "(none)"


def step_prompt(sas: str, schemas: dict[str, Schema], reason: str, outputs: list[str], context: str = "",
                extra: str = "") -> str:
    """One unit of a program that the rule-based converter could not handle (the rest is converted)."""
    return (f"Convert this SAS code. The rule-based converter could not handle it: {reason}\n\n"
            "The other steps of the program are converted separately and share `ctx` with your code: read their "
            "outputs with ctx.get and write this code's outputs with ctx.put.\n\n"
            f"Input data sets (SAS attributes):\n{describe_schemas(schemas)}\n\n"
            f"Output data sets to write: {', '.join(o.lower() for o in outputs) or '(determine from the code)'}\n\n"
            + (f"{extra}\n\n" if extra else "")
            + (f"Earlier steps in the program (already converted; for context only):\n```sas\n{context[-4000:]}\n```\n\n"
               if context else "")
            + f"SAS code to convert:\n```sas\n{sas.strip()}\n```")


def program_prompt(sas: str, schemas: dict[str, Schema], outputs: list[str], macro_vars: dict[str, str] | None = None) -> str:
    mv = "".join(f"%let {k} = {v};\n" for k, v in (macro_vars or {}).items())
    return (f"Convert this whole SAS program into one notebook cell.\n\n"
            f"Input data sets (SAS attributes):\n{describe_schemas(schemas)}\n\n"
            + (f"Macro variables defined before the program runs (also available via ctx.symget):\n```sas\n{mv}```\n\n"
               if mv else "")
            + f"The program must produce these data sets with ctx.put: {', '.join(o.lower() for o in outputs)}\n\n"
            f"SAS program:\n```sas\n{sas.strip()}\n```")


def repair_prompt(original_user: str, code: str, problem: str) -> str:
    return (f"{original_user}\n\nYour previous answer:\n```python\n{code}\n```\n\n"
            f"It failed validation:\n{problem}\n\nReturn a corrected answer (same JSON format).")


def parse_answer(text: str) -> dict:
    """JSON answer (structured output), tolerating a fenced block if the provider ignored the schema."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    try:
        d = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        d = json.loads(t[start:end + 1]) if start >= 0 and end > start else {"code": t, "outputs": [], "notes": ""}
    if not isinstance(d, dict) or "code" not in d:
        raise ValueError("answer has no 'code'")
    d.setdefault("outputs", [])
    d.setdefault("notes", "")
    return d
