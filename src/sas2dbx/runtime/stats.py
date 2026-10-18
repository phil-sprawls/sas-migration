"""Statistical / reporting PROCs with SAS semantics: FREQ, MEANS/SUMMARY, TRANSPOSE.

Results are DataFrames in SAS space with ``_sas_seq`` ordered the way SAS
writes the output data sets.
"""
from __future__ import annotations

import itertools
import math
from functools import reduce

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .procs import SEQ, reseq


def _is_char(df: DataFrame, c: str) -> bool:
    return isinstance(df.schema[c].dataType, T.StringType)


def _missing(df: DataFrame, c: str) -> Column:
    return (F.col(c) == "") if _is_char(df, c) else F.col(c).isNull()


def _order(df: DataFrame, cols: list[str], order: str = "internal", count_col: str | None = None) -> list[Column]:
    if order == "freq" and count_col:
        return [F.col(count_col).desc()] + [F.col(c).asc_nulls_first() for c in cols]
    return [F.col(c).asc_nulls_first() for c in cols]


# ---------------------------------------------------------------------------
# PROC FREQ
def freq(df: DataFrame, table: list[str], *, by: list[str] | None = None, weight: str | None = None,
         missing: bool = False, outpct: bool = False, order: str = "internal",
         formats: dict[str, Column] | None = None) -> dict[str, DataFrame]:
    """One PROC FREQ table request (one-way or two-way).

    Returns {'OUT': ..., 'OneWayFreqs' | 'CrossTabFreqs': ..., 'FishersExact': ..., 'ChiSq': ...}.
    Percentages exclude missing levels unless MISSING; OUT= keeps a row for the
    missing level with PERCENT missing.  FORMATs group on the formatted value.
    """
    by = by or []
    formats = formats or {}
    w = F.col(weight).cast("double") if weight else F.lit(1.0)
    data = df
    if weight:
        data = data.where(F.col(weight).isNotNull() & (F.col(weight) > 0))
    group_cols = []
    for c in table:
        if c in formats:
            data = data.withColumn(f"_sas_fmt_{c}", formats[c])
            group_cols.append(f"_sas_fmt_{c}")
        else:
            group_cols.append(c)
    miss_any = reduce(lambda a, b: a | b, [_missing(df, c) for c in table], F.lit(False))
    data = data.withColumn("_sas_miss", miss_any).withColumn("_sas_w", w)
    keys = by + group_cols
    agg = data.groupBy(*keys, "_sas_miss").agg(F.sum("_sas_w").alias("COUNT"),
                                               *[F.min(c).alias(c) for c in table if c in formats])
    wb = Window.partitionBy(*by) if by else Window.partitionBy(F.lit(1))
    denom = F.sum(F.when(F.lit(missing) | ~F.col("_sas_miss"), F.col("COUNT"))).over(wb)
    agg = agg.withColumn("PERCENT", F.when(F.lit(missing) | ~F.col("_sas_miss"), F.col("COUNT") / denom * 100))
    out: dict[str, DataFrame] = {}
    if len(table) == 2 and outpct:
        r, c = group_cols
        wr = Window.partitionBy(*by, r)
        wc = Window.partitionBy(*by, c)
        valid = F.lit(missing) | ~F.col("_sas_miss")
        agg = agg.withColumn("PCT_ROW", F.when(valid, F.col("COUNT") / F.sum(F.when(valid, F.col("COUNT"))).over(wr) * 100))
        agg = agg.withColumn("PCT_COL", F.when(valid, F.col("COUNT") / F.sum(F.when(valid, F.col("COUNT"))).over(wc) * 100))
    sel = [F.col(b) for b in by] + [F.col(c) for c in table] + [F.col("COUNT"), F.col("PERCENT")]
    if len(table) == 2 and outpct:
        sel += [F.col("PCT_ROW"), F.col("PCT_COL")]
    out["OUT"] = reseq(agg.select(*sel, *[F.col(g) for g in group_cols if g not in table]),
                       *[F.col(b).asc_nulls_first() for b in by], *_order(agg, group_cols, order, "COUNT")).select(*sel, SEQ)
    valid_rows = agg.where(F.lit(missing) | ~F.col("_sas_miss"))
    if len(table) == 1:
        v = table[0]
        fv = formats.get(v)
        wcum = (Window.partitionBy(*by) if by else Window.partitionBy(F.lit(1))).orderBy(*_order(valid_rows, group_cols, order, "COUNT")).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ow = valid_rows.select(
            *[F.col(b) for b in by], F.lit(f"Table {v}").alias("Table"), F.col(v),
            (F.col(group_cols[0]) if fv is not None else F.trim(F.col(v).cast("string"))).alias(f"F_{v}"),
            F.col("COUNT").alias("Frequency"), F.col("PERCENT").alias("Percent"),
            F.sum("COUNT").over(wcum).alias("CumFrequency"), F.sum("PERCENT").over(wcum).alias("CumPercent"), *[F.col(g) for g in group_cols if g not in table])
        out["OneWayFreqs"] = reseq(ow, *[F.col(b).asc_nulls_first() for b in by], *_order(ow, group_cols, order, "Frequency")).drop(*[g for g in group_cols if g not in table])
    if len(table) == 2:
        r, c = table
        ct = valid_rows.select(*[F.col(b) for b in by], F.lit(f"Table {r} * {c}").alias("Table"), F.col(r), F.col(c),
                               F.col("COUNT").alias("Frequency"), F.col("PERCENT").alias("Percent"))
        out["CrossTabFreqs"] = reseq(ct, *[F.col(b).asc_nulls_first() for b in by], F.col(r).asc_nulls_first(), F.col(c).asc_nulls_first())
        out["FishersExact"] = _fisher(valid_rows, by, group_cols[0], group_cols[1], f"Table {r} * {c}")
        out["ChiSq"] = _chisq(valid_rows, by, group_cols[0], group_cols[1], f"Table {r} * {c}")
    return out


def _cells(rows: DataFrame, by: list[str], r: str, c: str):
    """Collect the (small) contingency tables to the driver: {by tuple: {(r, c): count}}."""
    tables: dict[tuple, dict] = {}
    for row in rows.select(*by, r, c, "COUNT").collect():
        key = tuple(row[b] for b in by)
        tables.setdefault(key, {})[(row[r], row[c])] = row["COUNT"]
    return tables


def _fisher(rows: DataFrame, by: list[str], r: str, c: str, label: str) -> DataFrame:
    from scipy import stats as st

    spark = rows.sparkSession
    out = []
    for key, cells in sorted(_cells(rows, by, r, c).items(), key=lambda kv: tuple((x is None, x) for x in kv[0])):
        rl = sorted({k[0] for k in cells}, key=lambda x: (x is None, x))
        cl = sorted({k[1] for k in cells}, key=lambda x: (x is None, x))
        if len(rl) != 2 or len(cl) != 2:
            continue  # SAS: "No statistics are computed ... has less than 2 nonmissing levels"
        a = cells.get((rl[0], cl[0]), 0.0)
        b = cells.get((rl[0], cl[1]), 0.0)
        cc = cells.get((rl[1], cl[0]), 0.0)
        d = cells.get((rl[1], cl[1]), 0.0)
        table = [[int(round(a)), int(round(b))], [int(round(cc)), int(round(d))]]
        _, p2 = st.fisher_exact(table, alternative="two-sided")
        _, pl = st.fisher_exact(table, alternative="less")
        _, pr = st.fisher_exact(table, alternative="greater")
        M, n, N = int(a + b + cc + d), int(a + cc), int(a + b)
        ptab = float(st.hypergeom.pmf(int(a), M, n, N))
        for name, lab, val in (("XPL_FISH", "Left-sided Pr <= F", pl), ("XPR_FISH", "Right-sided Pr >= F", pr),
                               ("P_TABLE", "Table Probability (P)", ptab), ("XP2_FISH", "Two-sided Pr <= P", p2)):
            out.append(tuple(key) + (label, name, lab, _pvalue_text(val), float(val)))
    schema = T.StructType([*[rows.schema[b] for b in by],
                           T.StructField("Table", T.StringType()), T.StructField("Name1", T.StringType()),
                           T.StructField("Label1", T.StringType()), T.StructField("cValue1", T.StringType()),
                           T.StructField("nValue1", T.DoubleType())])
    df = spark.createDataFrame(out, schema)
    return df.withColumn(SEQ, F.monotonically_increasing_id())


def _pvalue_text(p: float) -> str:
    return "<.0001" if p < 0.0001 else f"{p:.4f}"


def _chisq(rows: DataFrame, by: list[str], r: str, c: str, label: str) -> DataFrame:
    spark = rows.sparkSession
    out = []
    for key, cells in sorted(_cells(rows, by, r, c).items(), key=lambda kv: tuple((x is None, x) for x in kv[0])):
        rl = sorted({k[0] for k in cells}, key=lambda x: (x is None, x))
        cl = sorted({k[1] for k in cells}, key=lambda x: (x is None, x))
        if len(rl) < 2 or len(cl) < 2:
            continue
        from scipy import stats as st

        mat = [[cells.get((i, j), 0.0) for j in cl] for i in rl]
        chi2, p, dof, _ = st.chi2_contingency(mat, correction=False)
        out.append(tuple(key) + (label, "Chi-Square", float(dof), float(chi2), float(p)))
    schema = T.StructType([*[rows.schema[b] for b in by], T.StructField("Table", T.StringType()),
                           T.StructField("Statistic", T.StringType()), T.StructField("DF", T.DoubleType()),
                           T.StructField("Value", T.DoubleType()), T.StructField("Prob", T.DoubleType())])
    return spark.createDataFrame(out, schema).withColumn(SEQ, F.monotonically_increasing_id())


# ---------------------------------------------------------------------------
# PROC MEANS / SUMMARY
_PCTL = {"P1": 0.01, "P5": 0.05, "P10": 0.10, "Q1": 0.25, "P25": 0.25, "MEDIAN": 0.5, "P50": 0.5,
         "Q3": 0.75, "P75": 0.75, "P90": 0.90, "P95": 0.95, "P99": 0.99}


def _pctl_qntldef5(arr: Column, n: Column, p: float) -> Column:
    """SAS QNTLDEF=5 (default): j = floor(np), g = np - j;  g>0 -> x[j+1], g=0 -> (x[j]+x[j+1])/2."""
    np_ = n * F.lit(p)
    j = F.floor(np_ + F.lit(1e-9)).cast("int")
    g = np_ - j
    xj1 = F.element_at(arr, F.least(j + 1, n.cast("int")))
    xj = F.element_at(arr, F.greatest(j, F.lit(1)))
    return F.when(n == 0, F.lit(None).cast("double")).when(g > F.lit(1e-9), xj1).otherwise((xj + xj1) / 2)


def _stat_exprs(var: str, stats: list[str], vardef: str = "DF", alpha: float = 0.05) -> dict[str, Column]:
    """Aggregate expressions for one analysis variable (to be used in a single groupBy)."""
    x = F.col(var).cast("double")
    n = F.count(x)
    s = F.sum(x)
    mean = F.avg(x)
    css = F.sum(x * x) - F.sum(x) * F.sum(x) / n
    denom = (n - 1) if vardef.upper() == "DF" else n
    var_ = F.when(denom > 0, css / denom)
    std = F.sqrt(F.greatest(var_, F.lit(0.0)))
    arr = F.array_sort(F.collect_list(x))
    out: dict[str, Column] = {}
    for st_ in stats:
        u = st_.upper()
        if u == "N":
            out[u] = n.cast("double")
        elif u == "NMISS":
            out[u] = F.sum(F.when(x.isNull(), 1.0).otherwise(0.0))
        elif u == "MEAN":
            out[u] = mean
        elif u == "SUM":
            out[u] = s
        elif u == "MIN":
            out[u] = F.min(x)
        elif u == "MAX":
            out[u] = F.max(x)
        elif u == "RANGE":
            out[u] = F.max(x) - F.min(x)
        elif u in ("STD", "STDDEV"):
            out[u] = std
        elif u == "VAR":
            out[u] = var_
        elif u == "USS":
            out[u] = F.sum(x * x)
        elif u == "CSS":
            out[u] = css
        elif u == "CV":
            out[u] = std / mean * 100
        elif u == "STDERR":
            out[u] = std / F.sqrt(n)
        elif u in _PCTL:
            out[u] = _pctl_qntldef5(arr, F.size(arr), _PCTL[u])
        elif u == "QRANGE":
            out[u] = _pctl_qntldef5(arr, F.size(arr), 0.75) - _pctl_qntldef5(arr, F.size(arr), 0.25)
        elif u == "SUMWGT":
            out[u] = n.cast("double")
        elif u in ("LCLM", "UCLM"):
            tq = F.call_function("sas_tinv", F.lit(1 - alpha / 2), (n - 1).cast("double"))
            half = tq * std / F.sqrt(n)
            out[u] = mean - half if u == "LCLM" else mean + half
        elif u == "SKEWNESS":
            # Spark gives g1 = m3 / m2^1.5; SAS (VARDEF=DF) reports G1 = g1 * sqrt(n(n-1)) / (n-2)
            out[u] = F.skewness(x) * F.sqrt(n * (n - 1)) / (n - 2)
        elif u == "KURTOSIS":
            g2 = F.kurtosis(x)
            out[u] = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
        else:
            raise ValueError(f"PROC MEANS statistic {st_} not supported")
    return out


def means(df: DataFrame, *, var: list[str], stats: list[str], class_: list[str] | None = None, by: list[str] | None = None,
          nway: bool = False, missing: bool = False, weight: str | None = None, vardef: str = "DF",
          names: dict[tuple[str, str], str] | None = None, types: list[tuple[str, ...]] | None = None,
          formats: dict[str, Column] | None = None) -> DataFrame:
    """PROC MEANS/SUMMARY OUTPUT OUT= data set.

    ``names`` maps (statistic, variable) -> output column.  Rows carry _TYPE_ and
    _FREQ_ like SAS; without NWAY all class combinations (types) are produced,
    ordered by _TYPE_ then class values.
    """
    if weight:
        raise ValueError("WEIGHT in PROC MEANS is not supported yet")
    class_ = class_ or []
    by = by or []
    names = names or {(s.upper(), v): v for s in stats for v in var}
    data = df
    for c in class_:
        if not missing:
            data = data.where(~_missing(df, c))
        if formats and c in formats:
            data = data.withColumn(c, formats[c])
    k = len(class_)
    if types is None:
        combos = [tuple(class_)] if nway else [tuple(c for i, c in enumerate(class_) if (t >> (k - 1 - i)) & 1) for t in range(2 ** k)]
    else:
        combos = types
    parts = []
    for combo in combos:
        t = sum(1 << (k - 1 - class_.index(c)) for c in combo)
        aggs = [F.count(F.lit(1)).cast("double").alias("_FREQ_")]
        for v in var:
            for s_, expr in _stat_exprs(v, stats, vardef).items():
                col_name = names.get((s_, v)) or names.get((s_.upper(), v))
                if col_name is None:
                    continue
                aggs.append(expr.alias(col_name))
        g = data.groupBy(*by, *combo).agg(*aggs)
        sel = [F.col(b) for b in by]
        for c in class_:
            sel.append(F.col(c) if c in combo else (F.lit("") if _is_char(df, c) else F.lit(None).cast("double")).alias(c))
        sel += [F.lit(float(t)).alias("_TYPE_"), F.col("_FREQ_")]
        sel += [F.col(n_) for n_ in dict.fromkeys(names.values()) if n_ in g.columns]
        parts.append(g.select(*sel))
    u = reduce(lambda a, b: a.unionByName(b), parts)
    order = [F.col(b).asc_nulls_first() for b in by] + [F.col("_TYPE_")] + [F.col(c).asc_nulls_first() for c in class_]
    return reseq(u, *order)


# ---------------------------------------------------------------------------
# PROC TRANSPOSE
def transpose(df: DataFrame, *, var: list[str], by: list[str] | None = None, id_: str | None = None, prefix: str = "COL",
              name: str = "_NAME_", label: str | None = "_LABEL_", labels: dict[str, str] | None = None,
              id_values: list[str] | None = None) -> DataFrame:
    """PROC TRANSPOSE.  Without ID the new columns are COL1..COLn by row order within the BY group."""
    by = by or []
    labels = labels or {}
    any_char = any(_is_char(df, v) for v in var)
    def val(v):
        c = F.col(v)
        if any_char and not _is_char(df, v):
            c = F.trim(F.call_function("sas_put_best", c, F.lit(12)))
        return c.cast("string") if any_char else c.cast("double")
    stacked = df.select(*by, SEQ, *( [F.col(id_)] if id_ else []),
                        F.explode(F.array(*[F.struct(F.lit(i).alias("_k"), F.lit(v).alias("_n"), val(v).alias("_v"))
                                             for i, v in enumerate(var)])).alias("_s"))
    stacked = stacked.select(*by, SEQ, *([F.col(id_)] if id_ else []), "_s._k", "_s._n", "_s._v")
    if id_:
        key = F.trim(F.col(id_).cast("string")) if not _is_char(df, id_) else F.col(id_)
        stacked = stacked.withColumn("_pivot", F.concat(F.lit(prefix if prefix != "COL" else ""), key))
    else:
        w = Window.partitionBy(*by, "_k").orderBy(SEQ)
        stacked = stacked.withColumn("_pivot", F.concat(F.lit(prefix), F.row_number().over(w).cast("string")))
    grp = stacked.groupBy(*by, "_k", "_n")
    pivoted = grp.pivot("_pivot", id_values).agg(F.first("_v"))
    cols = [c for c in pivoted.columns if c not in (*by, "_k", "_n")]
    if not id_:
        cols = sorted(cols, key=lambda c: int(c[len(prefix):]) if c[len(prefix):].isdigit() else 10**9)
    out = pivoted.select(*by, F.col("_n").alias(name),
                         *([F.coalesce(F.create_map(*itertools.chain.from_iterable((F.lit(k), F.lit(v)) for k, v in labels.items()))[F.col("_n")], F.lit("")).alias(label)] if (label and labels) else []),
                         *[F.col(f"`{c}`") for c in cols], "_k")
    out = reseq(out, *[F.col(b).asc_nulls_first() for b in by], F.col("_k"))
    return out.drop("_k")
