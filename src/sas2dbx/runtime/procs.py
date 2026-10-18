"""``P``: SAS step semantics on Spark DataFrames (sorting, MERGE, SET, BY-group flags ...).

All functions take/return DataFrames in SAS space carrying ``_sas_seq`` (the
logical row order).  Order is re-established with ``orderBy`` followed by
``monotonically_increasing_id()``, which is monotonic in the sorted order and
needs no single-partition renumbering.
"""
from __future__ import annotations

from functools import reduce

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

SEQ = "_sas_seq"


def _key_cols(by) -> list[tuple[str, bool]]:
    out = []
    for b in by:
        if isinstance(b, (tuple, list)):
            out.append((b[0], bool(b[1])))
        else:
            out.append((b, False))
    return out


def _sort_exprs(by) -> list[Column]:
    return [F.col(c).desc_nulls_last() if desc else F.col(c).asc_nulls_first() for c, desc in _key_cols(by)]


def reseq(df: DataFrame, *order: Column) -> DataFrame:
    """Re-establish _sas_seq after an order-changing operation."""
    return df.orderBy(*order).withColumn(SEQ, F.monotonically_increasing_id())


def visible(df: DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_sas_")]


# ---------------------------------------------------------------------------
# PROC SORT
def sort(df: DataFrame, by, *, nodupkey: bool = False, nodup: bool = False, dupout: bool = False):
    """Stable sort (SAS EQUALS).  NODUPKEY keeps the first row of each key in input order;
    NODUP removes a row identical to its predecessor in the sorted order."""
    keys = _key_cols(by)
    names = [c for c, _ in keys]
    dups = None
    if nodupkey:
        w = Window.partitionBy(*names).orderBy(SEQ)
        df = df.withColumn("_sas_rn", F.row_number().over(w))
        if dupout:
            dups = reseq(df.where("_sas_rn > 1").drop("_sas_rn"), *_sort_exprs(keys), F.col(SEQ))
        df = df.where("_sas_rn = 1").drop("_sas_rn")
    out = reseq(df, *_sort_exprs(keys), F.col(SEQ))
    if nodup:
        cols = visible(out)
        w = Window.partitionBy(*names).orderBy(SEQ)
        same = reduce(lambda a, b: a & b, [F.col(c).eqNullSafe(F.lag(c).over(w)) for c in cols])
        flagged = out.withColumn("_sas_dup", F.coalesce(same, F.lit(False)) & F.lag(F.lit(1)).over(w).isNotNull())
        if dupout:
            dups = flagged.where("_sas_dup").drop("_sas_dup")
        out = flagged.where(~F.col("_sas_dup")).drop("_sas_dup")
    return (out, dups) if dupout else out


# ---------------------------------------------------------------------------
# data set options
def keep(df: DataFrame, cols: list[str]) -> DataFrame:
    upper = {c.upper(): c for c in df.columns}
    missing = [c for c in cols if c.upper() not in upper]
    if missing:
        raise KeyError(f"KEEP: variables not found: {missing}")
    sel = [upper[c.upper()] for c in cols]
    extras = [c for c in df.columns if c.startswith("_sas_") and c not in sel]
    return df.select(*sel, *extras)


def drop(df: DataFrame, cols: list[str]) -> DataFrame:
    dropset = {c.upper() for c in cols}
    return df.select(*[c for c in df.columns if c.upper() not in dropset])


def rename(df: DataFrame, pairs: list[tuple[str, str]]) -> DataFrame:
    mapping = {o.upper(): n for o, n in pairs}
    return df.select(*[F.col(f"`{c}`").alias(mapping[c.upper()]) if c.upper() in mapping else F.col(f"`{c}`") for c in df.columns])


def obs(df: DataFrame, firstobs: int | None = None, obs_: int | None = None) -> DataFrame:
    if firstobs is None and obs_ is None:
        return df
    w = Window.orderBy(SEQ)
    df = df.withColumn("_sas_n", F.row_number().over(w))
    if firstobs is not None:
        df = df.where(F.col("_sas_n") >= firstobs)
    if obs_ is not None:
        df = df.where(F.col("_sas_n") <= obs_)
    return df.drop("_sas_n")


# ---------------------------------------------------------------------------
# BY-group processing
def first_last(df: DataFrame, by, *, notsorted: bool = False) -> DataFrame:
    """Add boolean _sas_first_<VAR>/_sas_last_<VAR> for each BY variable (SAS FIRST./LAST.)."""
    keys = _key_cols(by)
    names = [c for c, _ in keys]
    w = Window.orderBy(SEQ) if notsorted else Window.partitionBy(names[0]).orderBy(SEQ)
    prev_exists = F.lag(F.lit(1)).over(w).isNotNull()
    next_exists = F.lead(F.lit(1)).over(w).isNotNull()
    out = df
    changed_prev = F.lit(False)
    changed_next = F.lit(False)
    for c in names:
        changed_prev = changed_prev | ~F.col(c).eqNullSafe(F.lag(c).over(w))
        changed_next = changed_next | ~F.col(c).eqNullSafe(F.lead(c).over(w))
        out = out.withColumn(f"_sas_first_{c.upper()}", ~prev_exists | changed_prev)
        out = out.withColumn(f"_sas_last_{c.upper()}", ~next_exists | changed_next)
    return out


def row_number(df: DataFrame, name: str = "_sas_n_") -> DataFrame:
    """_N_ (iteration number) in logical order."""
    return df.withColumn(name, F.row_number().over(Window.orderBy(SEQ)).cast("double"))


def end_flag(df: DataFrame, name: str) -> DataFrame:
    """END= variable: 1 on the last row read."""
    return df.withColumn(name, F.when(F.lead(F.lit(1)).over(Window.orderBy(SEQ)).isNull(), 1.0).otherwise(0.0))


# ---------------------------------------------------------------------------
# SET (concatenate / interleave)
def set_concat(inputs: list[DataFrame], *, in_vars: list[str | None] | None = None, by=None) -> DataFrame:
    in_vars = in_vars or [None] * len(inputs)
    order: list[str] = []
    for df in inputs:
        for c in visible(df):
            if c.upper() not in {o.upper() for o in order}:
                order.append(c)
    parts = []
    for i, df in enumerate(inputs):
        upper = {c.upper(): c for c in df.columns}
        cols = []
        for c in order:
            if c.upper() in upper:
                cols.append(F.col(f"`{upper[c.upper()]}`").alias(c))
            else:
                dtype = _dtype_of(inputs, c)
                cols.append((F.lit("") if dtype == "string" else F.lit(None)).cast(dtype).alias(c))
        for j, iv in enumerate(in_vars):
            if iv:
                cols.append(F.lit(1.0 if j == i else 0.0).alias(iv))
        cols.append(F.lit(i).alias("_sas_ds"))
        cols.append(F.col(SEQ).alias("_sas_inseq"))
        parts.append(df.select(*cols))
    u = reduce(lambda a, b: a.unionByName(b), parts)
    if by:
        out = reseq(u, *_sort_exprs(by), F.col("_sas_ds"), F.col("_sas_inseq"))
    else:
        out = reseq(u, F.col("_sas_ds"), F.col("_sas_inseq"))
    return out.drop("_sas_ds", "_sas_inseq")


def _dtype_of(inputs: list[DataFrame], col: str) -> str:
    for df in inputs:
        for f in df.schema.fields:
            if f.name.upper() == col.upper():
                return f.dataType.simpleString()
    return "double"


# ---------------------------------------------------------------------------
# MERGE
def merge(inputs: list[DataFrame], by, *, in_vars: list[str | None] | None = None) -> DataFrame:
    """SAS match-merge.

    Within each BY group SAS reads the i-th row of every data set in parallel;
    a data set with fewer rows keeps contributing its last row's values
    (retained), a data set with no rows in the group contributes missing values.
    When several data sets share a variable, the value comes from the data set
    read most recently (later in the MERGE statement wins).  This is *not* a SQL
    join: many-to-many groups pair rows positionally instead of multiplying.
    """
    in_vars = in_vars or [None] * len(inputs)
    if not by:
        return _merge_positional(inputs, in_vars)
    keys = _key_cols(by)
    names = [c for c, _ in keys]
    knames = [f"_sas_k{j}" for j in range(len(names))]
    prepared = []
    col_sources: dict[str, list[int]] = {}
    order: list[str] = []
    for i, df in enumerate(inputs):
        upper = {c.upper(): c for c in df.columns}
        w = Window.partitionBy(*[upper[n.upper()] for n in names]).orderBy(SEQ)
        d = df.withColumn("_sas_r", F.row_number().over(w)).withColumn("_sas_cnt", F.count(F.lit(1)).over(w))
        sel = [F.col(f"`{upper[n.upper()]}`").alias(k) for n, k in zip(names, knames)]
        sel += [F.col("_sas_r").alias(f"_sas_r{i}"), F.col("_sas_cnt").alias(f"_sas_n{i}")]
        for c in visible(df):
            if c.upper() in {n.upper() for n in names}:
                continue
            sel.append(F.col(f"`{c}`").alias(f"_sas_c{i}_{c.upper()}"))
            col_sources.setdefault(c.upper(), []).append(i)
            if c.upper() not in {o.upper() for o in order}:
                order.append(c)
        prepared.append(d.select(*sel))
    # group frame: every key with the max group size across inputs, exploded to 1..max
    sizes = reduce(lambda a, b: a.unionByName(b),
                   [p.select(*knames, F.col(f"_sas_n{i}").alias("_sas_n")) for i, p in enumerate(prepared)])
    frame = sizes.groupBy(*knames).agg(F.max("_sas_n").alias("_sas_m"))
    frame = frame.select(*knames, F.explode(F.sequence(F.lit(1), F.col("_sas_m").cast("int"))).alias("_sas_r"))
    for i, p in enumerate(prepared):
        # unique column names on each side keep Spark's self-join detection out of the way
        cnt = p.groupBy(*knames).agg(F.max(f"_sas_n{i}").alias(f"_sas_g{i}"))
        cnt = cnt.select(*[F.col(k).alias(f"{k}_j") for k in knames], f"_sas_g{i}")
        cond = reduce(lambda a, b: a & b, [F.col(k).eqNullSafe(F.col(f"{k}_j")) for k in knames])
        frame = frame.join(cnt, on=cond, how="left").drop(*[f"{k}_j" for k in knames])
    for i, p in enumerate(prepared):
        data_cols = [c for c in p.columns if c.startswith(f"_sas_c{i}_")]
        rows = p.select(*[F.col(k).alias(f"{k}_r") for k in knames], F.col(f"_sas_r{i}").alias("_sas_rr"), *data_cols)
        cond = reduce(lambda a, b: a & b, [F.col(k).eqNullSafe(F.col(f"{k}_r")) for k in knames])
        cond = cond & (F.col("_sas_rr") == F.least(F.col("_sas_r"), F.col(f"_sas_g{i}")))
        frame = frame.join(rows, on=cond, how="left").drop(*[f"{k}_r" for k in knames], "_sas_rr")
    out_cols = [F.col(k).alias(n) for k, n in zip(knames, _first_names(inputs, names))]
    for c in order:
        srcs = col_sources[c.upper()]
        if len(srcs) == 1:
            out_cols.append(F.col(f"_sas_c{srcs[0]}_{c.upper()}").alias(c))
            continue
        # value from the source with the latest read: priority = min(r, n_i) * K + i
        K = len(inputs) + 1
        structs = [
            F.when(F.col(f"_sas_g{i}").isNotNull(),
                   F.struct((F.least(F.col("_sas_r"), F.col(f"_sas_g{i}")) * K + i).alias("p"),
                            F.col(f"_sas_c{i}_{c.upper()}").alias("v")))
            for i in srcs
        ]
        out_cols.append(F.greatest(*structs).getField("v").alias(c))
    for i, iv in enumerate(in_vars):
        if iv:
            out_cols.append(F.when(F.col(f"_sas_g{i}").isNotNull(), 1.0).otherwise(0.0).alias(iv))
    out = frame.select(*out_cols, *[F.col(k) for k in knames], F.col("_sas_r"))
    out = reseq(out, *[(F.col(k).desc_nulls_last() if desc else F.col(k).asc_nulls_first()) for k, (_, desc) in zip(knames, keys)],
                F.col("_sas_r"))
    return out.drop(*knames, "_sas_r")


def _first_names(inputs: list[DataFrame], names: list[str]) -> list[str]:
    out = []
    for n in names:
        for c in inputs[0].columns:
            if c.upper() == n.upper():
                out.append(c)
                break
        else:
            out.append(n)
    return out


def _merge_positional(inputs: list[DataFrame], in_vars) -> DataFrame:
    """MERGE without BY: one-to-one by position; shorter data sets contribute missing values."""
    numbered = []
    order: list[str] = []
    for i, df in enumerate(inputs):
        d = df.withColumn("_sas_r", F.row_number().over(Window.orderBy(SEQ)))
        cols = [F.col("_sas_r")]
        for c in visible(df):
            cols.append(F.col(f"`{c}`").alias(f"_sas_c{i}_{c.upper()}"))
            if c.upper() not in {o.upper() for o in order}:
                order.append(c)
        cols.append(F.lit(1.0).alias(f"_sas_in{i}"))
        numbered.append(d.select(*cols))
    joined = reduce(lambda a, b: a.join(b, on="_sas_r", how="full_outer"), numbered)
    out_cols = []
    for c in order:
        srcs = [i for i, d in enumerate(numbered) if f"_sas_c{i}_{c.upper()}" in d.columns]
        val = None
        for i in srcs:
            ci = F.col(f"_sas_c{i}_{c.upper()}")
            val = ci if val is None else F.when(F.col(f"_sas_in{i}").isNotNull(), ci).otherwise(val)
        out_cols.append(val.alias(c))
    for i, iv in enumerate(in_vars):
        if iv:
            out_cols.append(F.coalesce(F.col(f"_sas_in{i}"), F.lit(0.0)).alias(iv))
    return reseq(joined.select(*out_cols, "_sas_r"), F.col("_sas_r")).drop("_sas_r")


def align(df: DataFrame, cols: list[str]) -> DataFrame:
    """Select ``cols`` (PROC APPEND BASE= layout); variables absent from df become missing."""
    upper = {c.upper(): c for c in df.columns}
    out = []
    for c in cols:
        if c.upper() in upper:
            out.append(F.col(f"`{upper[c.upper()]}`").alias(c))
        else:
            out.append(F.lit(None).cast("double").alias(c))
    return df.select(*out, SEQ)


def into_text(value, trimmed: bool = False) -> str:
    """Text of a value stored by SELECT ... INTO :macrovar (numbers use BEST8., right aligned)."""
    from . import pyfuncs

    if value is None or isinstance(value, (int, float)):
        txt = pyfuncs.put_best(None if value is None else float(value), 8)
        return txt.strip() if trimmed else txt
    return str(value).strip() if trimmed else str(value)


def display(df: DataFrame, n: int = 50) -> None:
    """Report output of PROC PRINT / SQL SELECT: Databricks display() when available."""
    shown = df.drop(*[c for c in df.columns if c.startswith("_sas_")])
    try:  # pragma: no cover - Databricks notebooks
        import IPython

        disp = IPython.get_ipython().user_ns.get("display")
        if disp is not None:
            disp(shown)
            return
    except Exception:
        pass
    shown.show(n, truncate=False)


# statistical PROCs live in .stats; re-exported so generated code can use P.freq / P.means / P.transpose
from .stats import freq, means, transpose  # noqa: E402,F401


def define_format(spark, fn_name: str, kind: str, ranges: list, other=None) -> None:
    """Register a PROC FORMAT VALUE format as a SQL function ``fn_name(x) -> STRING``.

    ranges: (lo, hi, lo_exclusive, hi_exclusive, label) with lo/hi = None for LOW/HIGH;
    ('=', value, label) for single values; ('miss', label) for '.'/blank.
    """
    from . import sqlfuncs

    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return repr(float(v)) + "D"

    whens = []
    for r in ranges:
        if r[0] == "miss":
            whens.append(f"WHEN x IS NULL OR {'x = ' + chr(39) + chr(39) if kind == 'char' else 'false'} THEN {lit(r[1])}")
        elif r[0] == "=":
            whens.append(f"WHEN x = {lit(r[1])} THEN {lit(r[2])}")
        else:
            lo, hi, lo_ex, hi_ex, label = r
            conds = []
            if lo is not None:
                conds.append(f"x {'>' if lo_ex else '>='} {lit(lo)}")
            elif kind == "num":
                conds.append("x IS NOT NULL")
            if hi is not None:
                conds.append(f"x {'<' if hi_ex else '<='} {lit(hi)}")
            whens.append(f"WHEN {' AND '.join(conds) or 'true'} THEN {lit(label)}")
    if other is not None:
        default = lit(other)
    elif kind == "num":
        default = "trim(sas_put_best(x, 12))"
        sqlfuncs.ensure(spark, ["sas_put_best"])
    else:
        default = "x"
    typ = "STRING" if kind == "char" else "DOUBLE"
    body = "CASE " + " ".join(whens) + f" ELSE {default} END"
    spark.sql(f"CREATE OR REPLACE TEMPORARY FUNCTION {fn_name}(x {typ}) RETURNS STRING RETURN {body}")


def display_schema(df: DataFrame) -> None:
    """PROC CONTENTS: variables with their SAS attributes."""
    from .meta import schema_meta

    meta = schema_meta(df.schema)
    rows = [(i + 1, f.name, (meta.get(f.name.upper()).type if meta.get(f.name.upper()) else ""),
             (meta.get(f.name.upper()).length if meta.get(f.name.upper()) else None),
             (meta.get(f.name.upper()).format if meta.get(f.name.upper()) else None),
             (meta.get(f.name.upper()).label if meta.get(f.name.upper()) else None))
            for i, f in enumerate([f for f in df.schema.fields if not f.name.startswith("_sas_")])]
    display(df.sparkSession.createDataFrame(rows, "num int, variable string, type string, len int, format string, label string"))


def select_pdv(df: DataFrame, known: dict[str, str], drop: list[str] | None = None,
               rename: dict[str, str] | None = None) -> DataFrame:
    """Output columns when the input schema was not known at conversion time.

    ``known`` maps the PDV variables the converter knows (in PDV order) to 'num'/'char';
    they come first (missing ones are created as missing values), followed by every
    other input column the program never mentioned, minus DROP.
    """
    dropset = {d.upper() for d in (drop or [])}
    rename = {k.upper(): v for k, v in (rename or {}).items()}
    present = {c.upper(): c for c in df.columns}
    cols = []
    for name, typ in known.items():
        out_name = rename.get(name.upper(), name)
        if name.upper() in present:
            cols.append(F.col(f"`{present[name.upper()]}`").alias(out_name))
        else:
            cols.append((F.lit("") if typ == "char" else F.lit(None).cast("double")).alias(out_name))
    known_upper = {k.upper() for k in known}
    rest = [F.col(f"`{c}`") for c in df.columns
            if not c.startswith("_sas_") and c.upper() not in dropset and c.upper() not in known_upper]
    return df.select(*cols, *rest, SEQ)


def dedupe_columns(df: DataFrame) -> DataFrame:
    """SELECT * over joins: SAS keeps the first column of a given name; drop later duplicates."""
    seen, keep = set(), []
    for i, c in enumerate(df.columns):
        if c.upper() in seen:
            continue
        seen.add(c.upper())
        keep.append(i)
    if len(keep) == len(df.columns):
        return df
    tmp = df.toDF(*[f"_c{i}" for i in range(len(df.columns))])
    return tmp.select(*[F.col(f"_c{i}").alias(df.columns[i]) for i in keep])
