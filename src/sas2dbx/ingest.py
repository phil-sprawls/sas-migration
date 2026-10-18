"""SAS data sets -> lakehouse tables (``sas2dbx ingest``).

Reads .sas7bdat files in parallel with spark-sas7bdat (XPORT and the fallback via pyreadstat), checks the read
against pyreadstat (an independent reader), and writes Delta/Parquet tables as Databricks users expect them: DATE /
TIMESTAMP_NTZ columns by SAS format, NULL for blank text and missing numbers, upper-case names, SAS formats, labels and
lengths kept as column metadata, and the file's row order as ``_sas_seq``.  Converted programs read the tables back
into SAS space (``SasContext._normalize_in``), so nothing is lost on the way.

Only plain Spark functions are used, so ingestion also runs on Spark 3.x (Scala 2.12), where spark-sas7bdat is
available; the converted programs need Spark 4 or Databricks Runtime (SQL functions).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

from .runtime.meta import META_KEY, schema_meta
from .runtime.sasfile import crosscheck, read_sas_file


def delta_typed(df):
    """A SAS-space frame as written to Delta (no SQL functions: works on Spark 3.x and 4.x)."""
    from pyspark.sql import functions as F

    meta = schema_meta(df.schema)
    cols = []
    for f in df.schema.fields:
        c = F.col(f"`{f.name}`")
        m = meta.get(f.name.upper())
        if f.name == "_sas_seq" or m is None:
            cols.append(c)
            continue
        if m.type == "char":
            new = F.when(c == "", F.lit(None).cast("string")).otherwise(c)
        elif m.kind == "date":
            new = F.expr(f"date_add(DATE'1960-01-01', CAST(floor(`{f.name}`) AS INT))")
        elif m.kind == "datetime":
            new = F.timestamp_seconds(c - F.lit(315619200)).cast("timestamp_ntz")
        else:
            new = c
        md = {META_KEY: m.to_dict()}
        if m.label:
            md["comment"] = m.label
        cols.append(new.alias(f.name, metadata=md))
    return df.select(*cols)


def ingest(spark, source: str, dest: str, *, reader: str = "auto", fmt: str = "parquet", check_rows: int = 5000,
           log=print) -> dict:
    """Ingest every SAS data set in ``source`` (a directory or one file) into ``dest/<name>``; returns the evidence."""
    paths = [source] if os.path.isfile(source) else sorted(
        glob.glob(os.path.join(source, "*.sas7bdat")) + glob.glob(os.path.join(source, "*.xpt")))
    evidence = {"spark_version": spark.version, "reader_requested": reader, "format": fmt, "tables": {}}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0].lower()
        df, cols, label, used = read_sas_file(spark, path, reader)
        rec = {"file": path, "sha256": _sha(path), "reader": used, "rows": df.count(), "label": label,
               "columns": {k: v.to_dict() for k, v in cols.items()}}
        if used == "spark-sas7bdat":
            check = crosscheck(spark, path, df, rows=check_rows)
            rec["crosscheck"] = {"against": "pyreadstat", "rows": min(check_rows, rec["rows"]), "passed": check.passed,
                                 "summary": check.summary()}
            if not check.passed:
                raise RuntimeError(f"{path}: spark-sas7bdat and pyreadstat disagree, nothing written: {check.summary()}")
        out = os.path.join(dest, name)
        delta_typed(df).write.format(fmt).mode("overwrite").save(out)
        rec["table"] = out
        evidence["tables"][name] = rec
        log(f"{name}: {rec['rows']} rows via {used}" + (f", cross-check {'passed' if rec.get('crosscheck', {}).get('passed') else 'n/a'}"
                                                         if used == "spark-sas7bdat" else "") + f" -> {out}")
    with open(os.path.join(dest, "INGEST.json"), "w") as fh:
        json.dump(evidence, fh, indent=2, default=str)
    return evidence


def publish(spark, source: str, schema: str, *, volume: str = "sas2dbx_ingest", profile: str | None = None,
            log=print) -> dict:
    """Ingested tables (a ``sas2dbx ingest`` directory) -> Unity Catalog Delta tables ``<schema>.<name>``.

    For workspaces where spark-sas7bdat cannot run (serverless compute loads no JARs, e.g. Databricks Free Edition):
    ingest on Spark 3.x elsewhere, then publish.  The Parquet files are uploaded unchanged to a UC volume, in a folder
    named by their content hash, and copied into Delta tables with the ingested column metadata (SAS formats, labels,
    lengths) and ``_sas_seq``.  Each table's row count and metadata are checked after the copy.
    """
    import pyarrow.parquet as pq
    from databricks.sdk import WorkspaceClient
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType

    catalog, _, sch = schema.partition(".")
    if not catalog or not sch or "." in sch:
        raise ValueError(f"publish needs a Unity Catalog schema as catalog.schema, got {schema!r}")
    w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {schema}.{volume}")
    ingested = {}
    if os.path.exists(os.path.join(source, "INGEST.json")):
        with open(os.path.join(source, "INGEST.json")) as fh:
            ingested = json.load(fh).get("tables", {})
    evidence = {"schema": schema, "volume": f"{schema}.{volume}", "tables": {}}
    for table in sorted(glob.glob(os.path.join(source, "*"))):
        files = sorted(glob.glob(os.path.join(table, "*.parquet")))
        if not os.path.isdir(table) or not files:
            continue
        name = os.path.basename(table).lower()
        digest = hashlib.sha256("".join(_sha(p) for p in files).encode()).hexdigest()
        folder = f"/Volumes/{catalog}/{sch}/{volume}/{name}/{digest[:16]}"
        for p in files:
            with open(p, "rb") as fh:
                w.files.upload(f"{folder}/{os.path.basename(p)}", fh, overwrite=True)
        local = StructType.fromJson(json.loads(pq.read_schema(files[0]).metadata[b"org.apache.spark.sql.parquet.row.metadata"]))
        df = spark.read.parquet(folder).select(*[F.col(f"`{f.name}`").alias(f.name, metadata=f.metadata) for f in local.fields])
        fq = f"{schema}.{name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq)
        rec = ingested.get(name) or {}
        if rec.get("label"):
            label = rec["label"].replace("\\", "\\\\").replace("'", "\\'")
            spark.sql(f"COMMENT ON TABLE {fq} IS '{label}'")
        rows = spark.table(fq).count()
        expected = sum(pq.read_metadata(p).num_rows for p in files)
        got = {f.name: f.metadata.get(META_KEY) for f in spark.table(fq).schema.fields}
        lost = [f.name for f in local.fields if f.metadata.get(META_KEY) != got.get(f.name)]
        if rows != expected or lost:
            raise RuntimeError(f"{fq}: {rows} rows (expected {expected}); SAS metadata lost on {lost}")
        evidence["tables"][name] = {"table": fq, "rows": rows, "files": folder, "sha256": digest,
                                    "source": rec.get("file"), "reader": rec.get("reader"), "label": rec.get("label")}
        log(f"{name}: {rows} rows -> {fq} (SAS metadata kept on {sum(bool(f.metadata) for f in local.fields)} columns)")
    with open(os.path.join(source, "PUBLISH.json"), "w") as fh:
        json.dump(evidence, fh, indent=2, default=str)
    return evidence


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
