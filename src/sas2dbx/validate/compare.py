"""Spark-native data comparison with SAS-aware normalisation and decimal-parity tolerance.

Parity rule (default ``decimals=2``): two numbers match when both are missing or
``|a - b| < 0.5 * 10**-decimals`` — the standard definition of agreeing to *d*
decimal places.  Character values match after right-trimming (SAS trailing
blanks are not significant) with NULL treated as blank.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


@dataclass
class CompareOptions:
    keys: list[str] | None = None
    decimals: int = 2
    rel_tol: float = 0.0
    mode: str = "auto"                  # auto | keyed | ordered | unordered
    ignore_columns: list[str] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)   # expected name -> actual name
    max_samples: int = 20
    allow_row_samples: bool = True      # False => only aggregate statistics leave the cluster (PHI/PII)
    float_noise: float = 1e-9

    @property
    def abs_tol(self) -> float:
        return 0.5 * 10 ** (-self.decimals)


@dataclass
class ColumnResult:
    name: str
    kind: str                           # num | char
    status: str                         # ok | mismatch | missing_in_actual | extra_in_actual | type_mismatch
    mismatches: int = 0
    max_abs_diff: float | None = None
    display_mismatches: int = 0         # values within tolerance whose 2-decimal display still differs


@dataclass
class CompareResult:
    dataset: str
    passed: bool
    mode: str
    keys: list[str]
    expected_rows: int
    actual_rows: int
    matched_rows: int = 0
    only_in_expected: int = 0
    only_in_actual: int = 0
    duplicate_keys: int = 0
    columns: list[ColumnResult] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    decimals: int = 2

    @property
    def mismatched_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.status != "ok"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mismatched_columns"] = self.mismatched_columns
        return d

    def summary(self) -> str:
        if self.passed:
            return (f"PASS {self.dataset}: {self.expected_rows} rows, {len(self.columns)} columns match "
                    f"to {self.decimals} decimals ({self.mode})")
        parts = []
        if self.expected_rows != self.actual_rows:
            parts.append(f"rows expected {self.expected_rows} vs actual {self.actual_rows}")
        if self.only_in_expected or self.only_in_actual:
            parts.append(f"{self.only_in_expected} rows only in expected, {self.only_in_actual} only in actual")
        bad = [c for c in self.columns if c.status != "ok"]
        if bad:
            parts.append("columns: " + ", ".join(f"{c.name} ({c.status}{', ' + str(c.mismatches) if c.mismatches else ''})" for c in bad[:12]))
        return f"FAIL {self.dataset}: " + "; ".join(parts)


def _kind(dt: T.DataType) -> str:
    return "char" if isinstance(dt, T.StringType) else "num"


def normalize(df: DataFrame) -> DataFrame:
    """SAS space: upper-case names, trimmed strings (NULL->''), doubles (NaN->NULL)."""
    cols = []
    for f in df.schema.fields:
        if f.name.startswith("_sas_") and f.name != "_sas_seq":
            continue
        c = F.col(f"`{f.name}`")
        if f.name == "_sas_seq":
            cols.append(c.alias("_sas_seq"))
        elif isinstance(f.dataType, T.StringType):
            cols.append(F.coalesce(F.rtrim(c), F.lit("")).alias(f.name.upper()))
        elif isinstance(f.dataType, (T.DateType,)):
            cols.append((F.unix_date(c) + 3653).cast("double").alias(f.name.upper()))
        elif isinstance(f.dataType, (T.TimestampType, T.TimestampNTZType)):
            cols.append((F.unix_seconds(c.cast("timestamp")) + 315619200).cast("double").alias(f.name.upper()))
        elif isinstance(f.dataType, T.NumericType):
            d = c.cast("double")
            cols.append(F.when(F.isnan(d), F.lit(None).cast("double")).otherwise(d).alias(f.name.upper()))
        elif isinstance(f.dataType, T.BooleanType):
            cols.append(F.when(c, 1.0).when(~c, 0.0).alias(f.name.upper()))
        else:
            cols.append(c.cast("string").alias(f.name.upper()))
    return df.select(*cols)


def _num_mismatch(a, b, opts: CompareOptions):
    diff = F.abs(a - b)
    within = diff < F.lit(opts.abs_tol)
    if opts.rel_tol:
        within = within | (diff <= F.lit(opts.rel_tol) * F.greatest(F.abs(a), F.abs(b)))
    return F.when(a.isNull() & b.isNull(), F.lit(False)).when(a.isNull() | b.isNull(), F.lit(True)).otherwise(~within)


def _display_diff(a, b, decimals: int):
    """Both values present, within tolerance, but their rounded display differs (informational)."""
    ra = F.round(a, decimals)
    rb = F.round(b, decimals)
    return a.isNotNull() & b.isNotNull() & (ra != rb)


def compare(expected: DataFrame, actual: DataFrame, opts: CompareOptions | None = None, name: str = "") -> CompareResult:
    opts = opts or CompareOptions()
    exp = normalize(expected)
    act = normalize(actual)
    if opts.column_map:
        back = {a.upper(): e.upper() for e, a in opts.column_map.items()}
        act = act.select(*[F.col(c).alias(back.get(c, c)) for c in act.columns])
    ignore = {c.upper() for c in opts.ignore_columns}
    exp_cols = {f.name: _kind(f.dataType) for f in exp.schema.fields if f.name != "_sas_seq" and f.name not in ignore}
    act_cols = {f.name: _kind(f.dataType) for f in act.schema.fields if f.name != "_sas_seq" and f.name not in ignore}
    n_exp, n_act = exp.count(), act.count()
    result = CompareResult(name, False, "", [], n_exp, n_act, decimals=opts.decimals)
    if n_exp == 0 and not exp_cols:
        # an empty ground truth without a schema (e.g. SAS produced no rows): only the row count is comparable
        result.mode = "row-count"
        result.passed = n_act == 0
        result.notes.append("expected data set is empty (no schema); compared on row count only")
        return result
    common = [c for c in exp_cols if c in act_cols]
    for c in exp_cols:
        if c not in act_cols:
            result.columns.append(ColumnResult(c, exp_cols[c], "missing_in_actual"))
    for c in act_cols:
        if c not in exp_cols:
            result.columns.append(ColumnResult(c, act_cols[c], "extra_in_actual"))
    typed = []
    for c in common:
        if exp_cols[c] != act_cols[c]:
            result.columns.append(ColumnResult(c, exp_cols[c], "type_mismatch"))
        else:
            typed.append(c)

    mode = opts.mode
    keys = [k.upper() for k in (opts.keys or [])]
    if mode == "auto":
        if keys:
            mode = "keyed"
        elif "_sas_seq" in exp.columns and "_sas_seq" in act.columns:
            mode = "ordered"
        else:
            mode = "unordered"
    result.mode = mode
    result.keys = keys
    if mode == "keyed":
        missing = [k for k in keys if k not in exp_cols or k not in act_cols]
        if missing:
            result.notes.append(f"key columns missing: {missing}; falling back to unordered comparison")
            mode = result.mode = "unordered"
    if mode == "keyed":
        dup_e = exp.groupBy(*keys).count().where("count > 1").count()
        dup_a = act.groupBy(*keys).count().where("count > 1").count()
        result.duplicate_keys = dup_e + dup_a
        if result.duplicate_keys:
            result.notes.append(f"keys not unique ({dup_e} duplicate groups in expected, {dup_a} in actual); "
                                "pairing duplicates by row order within key")
            w = Window.partitionBy(*keys).orderBy("_sas_seq")
            exp = exp.withColumn("_k_dup", F.row_number().over(w))
            act = act.withColumn("_k_dup", F.row_number().over(w))
            keys = keys + ["_k_dup"]
        _compare_joined(exp, act, keys, [c for c in typed if c not in keys], exp_cols, opts, result)
    elif mode == "ordered":
        e = exp.withColumn("_k_pos", F.row_number().over(Window.orderBy("_sas_seq")))
        a = act.withColumn("_k_pos", F.row_number().over(Window.orderBy("_sas_seq")))
        _compare_joined(e, a, ["_k_pos"], typed, exp_cols, opts, result)
    else:
        _compare_unordered(exp, act, typed, exp_cols, opts, result)
    for c in typed:
        if not any(r.name == c for r in result.columns):
            result.columns.append(ColumnResult(c, exp_cols[c], "ok"))
    order = list(exp_cols) + [c for c in act_cols if c not in exp_cols]
    result.columns.sort(key=lambda r: order.index(r.name) if r.name in order else 10**6)
    result.passed = (n_exp == n_act and result.only_in_expected == 0 and result.only_in_actual == 0
                     and all(c.status == "ok" for c in result.columns))
    return result


def _compare_joined(exp, act, keys, cols, kinds, opts, result: CompareResult) -> None:
    e = exp.select(*keys, *[F.col(c).alias(f"e__{c}") for c in cols], F.lit(1).alias("_in_e"))
    a = act.select(*keys, *[F.col(c).alias(f"a__{c}") for c in cols], F.lit(1).alias("_in_a"))
    cond = reduce(lambda x, y: x & y, [e[k].eqNullSafe(a[k]) for k in keys])
    j = e.join(a, cond, "full_outer").select(
        *[F.coalesce(e[k], a[k]).alias(k) for k in keys],
        *[F.col(f"e__{c}") for c in cols], *[F.col(f"a__{c}") for c in cols], "_in_e", "_in_a")
    flags = {}
    aggs = [F.sum(F.when(F.col("_in_e").isNull(), 1).otherwise(0)).alias("_only_a"),
            F.sum(F.when(F.col("_in_a").isNull(), 1).otherwise(0)).alias("_only_e"),
            F.sum(F.when(F.col("_in_a").isNotNull() & F.col("_in_e").isNotNull(), 1).otherwise(0)).alias("_both")]
    both = F.col("_in_a").isNotNull() & F.col("_in_e").isNotNull()
    for c in cols:
        ec, ac = F.col(f"e__{c}"), F.col(f"a__{c}")
        if kinds[c] == "num":
            mm = both & _num_mismatch(ec, ac, opts)
            aggs.append(F.max(F.when(both, F.abs(ec - ac))).alias(f"_d__{c}"))
            aggs.append(F.sum(F.when(both & ~_num_mismatch(ec, ac, opts) & _display_diff(ec, ac, opts.decimals), 1).otherwise(0)).alias(f"_disp__{c}"))
        else:
            mm = both & (ec != ac)
        flags[c] = mm
        aggs.append(F.sum(F.when(mm, 1).otherwise(0)).alias(f"_m__{c}"))
    row = j.agg(*aggs).collect()[0]
    result.only_in_actual = int(row["_only_a"] or 0)
    result.only_in_expected = int(row["_only_e"] or 0)
    result.matched_rows = int(row["_both"] or 0)
    bad_cols = []
    for c in cols:
        m = int(row[f"_m__{c}"] or 0)
        cr = ColumnResult(c, kinds[c], "mismatch" if m else "ok", m)
        if kinds[c] == "num":
            d = row[f"_d__{c}"]
            cr.max_abs_diff = float(d) if d is not None else None
            cr.display_mismatches = int(row[f"_disp__{c}"] or 0)
        if m:
            bad_cols.append(c)
        result.columns.append(cr)
    if opts.allow_row_samples and (bad_cols or result.only_in_actual or result.only_in_expected):
        any_bad = reduce(lambda x, y: x | y, [flags[c] for c in bad_cols], F.lit(False))
        any_bad = any_bad | F.col("_in_e").isNull() | F.col("_in_a").isNull()
        show = bad_cols[:8]
        rows = j.where(any_bad).select(*keys, *[F.col(f"e__{c}") for c in show], *[F.col(f"a__{c}") for c in show],
                                       "_in_e", "_in_a").limit(opts.max_samples).collect()
        for r in rows:
            d = {k: r[k] for k in keys if not k.startswith("_k_")}
            if r["_in_e"] is None:
                d["_side"] = "only in actual"
            elif r["_in_a"] is None:
                d["_side"] = "only in expected"
            for c in show:
                ev, av = r[f"e__{c}"], r[f"a__{c}"]
                if r["_in_e"] is not None and r["_in_a"] is not None:
                    same = (ev == av) if kinds[c] == "char" else (
                        (ev is None and av is None) or (ev is not None and av is not None and abs(ev - av) < opts.abs_tol))
                    if same:
                        continue
                d[c] = {"expected": ev, "actual": av}
            result.samples.append(d)


def _canon(c: str, kind: str, decimals: int):
    col = F.col(c)
    if kind == "num":
        return F.when(col.isNull(), F.lit("∅")).otherwise(F.format_string(f"%.{decimals}f", F.round(col, decimals) + F.lit(0.0)))
    return col


def _compare_unordered(exp, act, cols, kinds, opts, result: CompareResult) -> None:
    sig_e = exp.select(F.concat_ws("␟", *[_canon(c, kinds[c], opts.decimals) for c in cols]).alias("_sig"))
    sig_a = act.select(F.concat_ws("␟", *[_canon(c, kinds[c], opts.decimals) for c in cols]).alias("_sig"))
    ce = sig_e.groupBy("_sig").agg(F.count(F.lit(1)).alias("ne"))
    ca = sig_a.groupBy("_sig").agg(F.count(F.lit(1)).alias("na"))
    j = ce.join(ca, "_sig", "full_outer").select("_sig", F.coalesce("ne", F.lit(0)).alias("ne"), F.coalesce("na", F.lit(0)).alias("na"))
    row = j.agg(F.sum(F.greatest(F.col("ne") - F.col("na"), F.lit(0))).alias("oe"),
                F.sum(F.greatest(F.col("na") - F.col("ne"), F.lit(0))).alias("oa"),
                F.sum(F.least("ne", "na")).alias("both")).collect()[0]
    result.only_in_expected = int(row["oe"] or 0)
    result.only_in_actual = int(row["oa"] or 0)
    result.matched_rows = int(row["both"] or 0)
    if result.only_in_expected or result.only_in_actual:
        result.notes.append("unordered comparison: row signatures use values rounded to the parity decimals; "
                            "rows that straddle a rounding boundary are reported as differences — supply keys for a column-level diff")
        if opts.allow_row_samples:
            for r in j.where("ne <> na").limit(opts.max_samples).collect():
                result.samples.append({"_signature": r["_sig"], "expected_count": r["ne"], "actual_count": r["na"]})
    for c in cols:
        result.columns.append(ColumnResult(c, kinds[c], "ok"))
