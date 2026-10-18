"""Read SAS datasets (XPORT .xpt, .sas7bdat) into Spark in "SAS space".

SAS space = what SAS itself holds: numerics as DOUBLE (dates/datetimes as SAS
numbers), character values right-trimmed, plus the SAS metadata (formats,
labels, lengths) and the physical row order as ``_sas_seq``.
"""
from __future__ import annotations

import math
import os

from .meta import META_KEY, ColumnMeta, normalize_format


SPARK_SAS7BDAT = "com.github.saurfang.sas.spark"      # saurfang/spark-sas7bdat (Spark 3.x, Scala 2.12)
READERS = ("auto", "spark-sas7bdat", "pyreadstat")


def read_sas_pandas(path: str, row_limit: int = 0):
    """(pandas DataFrame, {COL: ColumnMeta}, dataset label) via pyreadstat."""
    import pyreadstat

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xpt":
        df, meta = pyreadstat.read_xport(path, disable_datetime_conversion=True, row_limit=row_limit)
    elif ext == ".sas7bdat":
        df, meta = pyreadstat.read_sas7bdat(path, disable_datetime_conversion=True, row_limit=row_limit)
    else:
        raise ValueError(f"not a SAS dataset file: {path}")
    cols, label = _metadata(meta, list(df.columns))
    return df, cols, label


def _metadata(meta, columns: list[str]) -> tuple[dict[str, "ColumnMeta"], str | None]:
    cols: dict[str, ColumnMeta] = {}
    types = getattr(meta, "readstat_variable_types", {}) or {}
    fmts = getattr(meta, "original_variable_types", {}) or {}
    widths = getattr(meta, "variable_storage_width", {}) or {}
    labels = getattr(meta, "column_names_to_labels", {}) or {}
    for c in columns:
        is_char = types.get(c) == "string"
        fmt = normalize_format(fmts.get(c))
        if fmt and is_char and not fmt.startswith("$"):
            fmt = None
        if fmt and not is_char and fmt.startswith("$"):
            fmt = None
        if fmt in ("BEST12.", "BEST.", "F."):
            pass
        cols[c.upper()] = ColumnMeta(
            type="char" if is_char else "num",
            length=int(widths.get(c) or (8 if not is_char else 1)),
            format=fmt if fmt and not (is_char and fmt in (f"${widths.get(c)}.", "$.")) else None,
            label=labels.get(c) or None,
        )
    label = getattr(meta, "file_label", None) or None
    return cols, label


def pandas_to_spark(spark, pdf, cols: dict[str, "ColumnMeta"]):
    """Build a Spark DataFrame with explicit SAS-space types, metadata and _sas_seq."""
    from pyspark.sql import types as T

    fields = []
    rows_cols = []
    for c in pdf.columns:
        m = cols.get(c.upper()) or ColumnMeta()
        dtype = T.StringType() if m.type == "char" else T.DoubleType()
        fields.append(T.StructField(c.upper(), dtype, True, {META_KEY: m.to_dict()}))
        series = pdf[c]
        if m.type == "char":
            rows_cols.append(["" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).rstrip(" ") for v in series])
        else:
            rows_cols.append([None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in series])
    fields.append(T.StructField("_sas_seq", T.LongType(), False))
    n = len(pdf)
    data = [tuple(col[i] for col in rows_cols) + (i,) for i in range(n)]
    return spark.createDataFrame(data, T.StructType(fields))


def read_sas_file(spark, path: str, reader: str = "auto"):
    """(SAS-space DataFrame, {COL: ColumnMeta}, label, reader used).

    ``.sas7bdat`` files are read in parallel by spark-sas7bdat when it is on the cluster ("auto"; "spark-sas7bdat"
    insists on it); XPORT files and the fallback go through pyreadstat on the driver."""
    if reader not in READERS:
        raise ValueError(f"unknown SAS reader {reader!r}: one of {READERS}")
    if path.lower().endswith(".sas7bdat") and reader != "pyreadstat":
        if reader == "spark-sas7bdat" or spark_sas7bdat_available(spark):
            df, cols, label = read_sas7bdat_spark(spark, path)
            return df, cols, label, "spark-sas7bdat"
    pdf, cols, label = read_sas_pandas(path)
    return pandas_to_spark(spark, pdf, cols), cols, label, "pyreadstat"


def spark_sas7bdat_available(spark) -> bool:
    try:
        spark.sparkContext._jvm.java.lang.Class.forName(SPARK_SAS7BDAT + ".DefaultSource")
        return True
    except Exception:  # noqa: BLE001 - class missing, or no JVM gateway (Spark Connect / serverless)
        return False


def read_sas7bdat_spark(spark, path: str):
    """Read a .sas7bdat in parallel with spark-sas7bdat, into SAS space: dates back to SAS numbers, missing as NULL,
    text right-trimmed ('' for blank), names upper-case, SAS formats/labels/lengths from the file header (pyreadstat,
    header only) and the file's row order as ``_sas_seq``."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    _require_utc_jvm(spark)
    raw = spark.read.format(SPARK_SAS7BDAT).load(path)
    cols, label = _header(path, raw.schema)
    sel = []
    for f in raw.schema.fields:
        name = f.name.upper()
        c = F.col(f"`{f.name}`")
        if isinstance(f.dataType, T.DateType):
            v = F.datediff(c, F.lit("1960-01-01").cast("date")).cast("double")
        elif isinstance(f.dataType, T.TimestampType):
            v = (F.unix_timestamp(c) + F.lit(315619200)).cast("double")       # seconds since 1960-01-01
        elif isinstance(f.dataType, T.StringType):
            v = F.coalesce(F.rtrim(c), F.lit(""))
        else:
            d = c.cast("double")
            v = F.when(F.isnan(d), F.lit(None).cast("double")).otherwise(d)
        sel.append(v.alias(name, metadata={META_KEY: cols[name].to_dict()}))
    return raw.select(*sel).withColumn("_sas_seq", F.monotonically_increasing_id()), cols, label


def _require_utc_jvm(spark) -> None:
    """spark-sas7bdat turns SAS dates into java.sql.Date in the JVM's default time zone: west of UTC every date
    lands one day early, silently (verified: America/Kentucky/Louisville reads 1993-01-01 as 1992-12-31)."""
    try:
        tz = spark.sparkContext._jvm.java.util.TimeZone.getDefault()
        tz_id, offset = tz.getID(), tz.getRawOffset()
    except Exception:  # noqa: BLE001 - no JVM gateway: cannot check here (Databricks clusters run in UTC)
        return
    if offset != 0 or tz.useDaylightTime():
        raise RuntimeError(
            f"spark-sas7bdat converts SAS dates in the JVM time zone ({tz_id}), which shifts dates by a day. Start "
            "Spark with -Duser.timezone=UTC in spark.driver.extraJavaOptions and spark.executor.extraJavaOptions "
            "(Databricks clusters already run in UTC), or read with reader='pyreadstat'.")


def _header(path: str, schema) -> tuple[dict[str, "ColumnMeta"], str | None]:
    """SAS formats, labels and lengths from the file header (pyreadstat reads the header only); without pyreadstat
    they are inferred from spark-sas7bdat's types (date/datetime kinds kept, character lengths unknown -> 200)."""
    from pyspark.sql import types as T

    try:
        import pyreadstat

        _, meta = pyreadstat.read_sas7bdat(path, metadataonly=True)
        cols, label = _metadata(meta, list(meta.column_names))
        return {k.upper(): v for k, v in cols.items()}, label
    except ImportError:
        pass
    cols = {}
    for f in schema.fields:
        if isinstance(f.dataType, T.StringType):
            cols[f.name.upper()] = ColumnMeta(type="char", length=200)
        elif isinstance(f.dataType, T.DateType):
            cols[f.name.upper()] = ColumnMeta(type="num", length=8, format="DATE9.")
        elif isinstance(f.dataType, T.TimestampType):
            cols[f.name.upper()] = ColumnMeta(type="num", length=8, format="DATETIME20.")
        else:
            cols[f.name.upper()] = ColumnMeta(type="num", length=8)
    return cols, None


def crosscheck(spark, path: str, df, rows: int = 2000):
    """Differential check of a spark-sas7bdat read: its first ``rows`` rows must equal pyreadstat's (an independent
    reader) at full precision, in the file's order.  Returns a CompareResult."""
    from ..validate.compare import CompareOptions, compare

    pdf, cols, _ = read_sas_pandas(path, row_limit=rows)
    expected = pandas_to_spark(spark, pdf, cols)
    actual = df.orderBy("_sas_seq").limit(rows)
    return compare(expected, actual, CompareOptions(decimals=9, mode="ordered"), name=os.path.basename(path))
