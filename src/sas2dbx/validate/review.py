"""Review tables: what a person needs to check a parity result against the data itself, kept inside the platform.

For each compared output, three tables: ``<case>__<dataset>__sas`` (the SAS reference as the gate read it),
``__dbx`` (the converted output) and ``__diff`` (one row per differing cell, from ``compare.diff_rows``).  On
Databricks they are Unity Catalog tables in a review schema, governed by its permissions and pinned by Delta version;
on local Spark they are Parquet folders, readable in SQL as ``parquet.`<path>```.  Row values stay where the data
already lives, which is why this is allowed even when a case sets ``allow_row_samples: false``.
"""
from __future__ import annotations

import os
import re

from .compare import CompareResult, diff_rows


def _ident(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _visible(df):
    """The columns a reviewer sees, with SAS dates and datetimes (numbers with a date format) shown as dates."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    from ..report import _DATE_FORMATS, _DATETIME_FORMATS
    from ..runtime.meta import normalize_format, schema_meta

    meta = schema_meta(df.schema)
    cols = []
    for f in df.schema.fields:
        if f.name.lower().startswith("_sas_"):
            continue
        c = F.col(f"`{f.name}`")
        m = meta.get(f.name.upper())
        fmt = (normalize_format(m.format) or "").upper() if m else ""
        if isinstance(f.dataType, T.NumericType) and fmt.startswith(_DATETIME_FORMATS):
            c = F.timestamp_seconds(c.cast("double") - F.lit(315619200)).alias(f.name)
        elif isinstance(f.dataType, T.NumericType) and fmt.startswith(_DATE_FORMATS):
            c = F.date_add(F.lit("1960-01-01").cast("date"), c.cast("int")).alias(f.name)
        cols.append(c)
    return df.select(*cols)


class ReviewSink:
    """Writes the review tables: to ``catalog.schema`` on Databricks, or to Parquet under ``local_dir``."""

    def __init__(self, spark, target: str, *, local_dir: str | None = None):
        self.spark = spark
        self.target = target
        self.local_dir = local_dir

    @property
    def kind(self) -> str:
        return "parquet" if self.local_dir else "unity-catalog"

    def name(self, case: str, dataset: str, part: str) -> str:
        return f"{_ident(case)}__{_ident(dataset)}__{part}"

    def write(self, df, name: str) -> dict:
        if self.local_dir:
            path = os.path.abspath(os.path.join(self.local_dir, name))
            df.write.mode("overwrite").parquet(path)
            rows = self.spark.read.parquet(path).count()
            return {"name": name, "sql": f"parquet.`{path}`", "path": path, "rows": rows}
        table = f"{self.target}.{name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        version = self.spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").collect()[0]["version"]
        rows = self.spark.table(table).count()
        return {"name": name, "sql": table, "table": table, "version": int(version), "rows": rows}

    def write_output(self, case: str, dataset: str, expected, actual, res: CompareResult) -> dict:
        out = {"kind": self.kind, "target": self.local_dir or self.target, "keys": list(res.keys), "tables": {}}
        out["tables"]["sas"] = self.write(_visible(expected), self.name(case, dataset, "sas"))
        out["tables"]["dbx"] = self.write(_visible(actual), self.name(case, dataset, "dbx"))
        diff = diff_rows(res)
        if diff is not None:
            out["tables"]["diff"] = self.write(diff, self.name(case, dataset, "diff"))
        return out
