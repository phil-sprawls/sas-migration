"""Spark-native data comparison with SAS-aware normalisation: output data parity.

Numeric rule (default, ``decimals=None``): two numbers match when both are missing, or when they are the same number
up to floating-point noise: ``|a - b| < 5e-13`` or ``|a - b| <= 1e-9 * max(|a|, |b|)``.  The relative part keeps
small values strict (0.001 and 0.004 differ) and tolerates only the last-digit noise of a different summation order.
A case whose reference values were themselves rounded can relax this to ``decimals=d``: ``|a - b| < 0.5 * 10**-d``.

Character values match after right-trimming (SAS trailing blanks are not significant) with NULL treated as blank.

Metadata is part of the output: column order always, and SAS length, format and label for every column whose
expected side carries SAS metadata.  ``metadata_ignore`` drops a check for a documented reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..runtime.meta import normalize_format, schema_meta

STRICT_ABS = 5e-13
STRICT_REL = 1e-9
METADATA_CHECKS = ("order", "length", "format", "label")


@dataclass
class CompareOptions:
    keys: list[str] | None = None
    decimals: int | None = None         # None: strict (floating-point noise only); d: relaxed to d decimals
    rel_tol: float = 0.0
    mode: str = "auto"                  # auto | keyed | ordered | unordered
    ignore_columns: list[str] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)   # expected name -> actual name
    max_samples: int = 20
    allow_row_samples: bool = True      # False => only aggregate statistics leave the cluster (PHI/PII)
    metadata_ignore: list[str] = field(default_factory=list)   # "check" or "check:COLUMN", each with a documented reason
    control_by: list[str] = field(default_factory=list)        # columns whose row counts per value the report shows

    @property
    def abs_tol(self) -> float:
        return STRICT_ABS if self.decimals is None else 0.5 * 10 ** (-self.decimals)

    @property
    def rel(self) -> float:
        return self.rel_tol or (STRICT_REL if self.decimals is None else 0.0)

    @property
    def display_decimals(self) -> int:
        return 2 if self.decimals is None else self.decimals

    @property
    def tolerance(self) -> str:
        return tolerance_text(self.decimals, self.rel_tol)


def tolerance_text(decimals: int | None, rel_tol: float = 0.0) -> str:
    if decimals is None and not rel_tol:
        return f"strict (|a-b| < {STRICT_ABS:g} or <= {STRICT_REL:g} relative)"
    if decimals is None:
        return f"|a-b| < {STRICT_ABS:g} or <= {rel_tol:g} relative"
    return f"{decimals} decimals" + (f" or <= {rel_tol:g} relative" if rel_tol else "")


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
    decimals: int | None = None
    tolerance: str = ""
    metadata: list[dict] = field(default_factory=list)   # metadata differences: {check, column?, expected, actual}
    controls: dict = field(default_factory=dict)         # control totals per side (aggregates only): see control_totals
    settings: dict = field(default_factory=dict)         # what the comparison left out or renamed, for the report
    review: dict = field(default_factory=dict)           # where the review tables were written (validate.review)

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
                    f"({self.tolerance}, {self.mode})")
        parts = []
        if self.expected_rows != self.actual_rows:
            parts.append(f"rows expected {self.expected_rows} vs actual {self.actual_rows}")
        if self.only_in_expected or self.only_in_actual:
            parts.append(f"{self.only_in_expected} rows only in expected, {self.only_in_actual} only in actual")
        bad = [c for c in self.columns if c.status != "ok"]
        if bad:
            parts.append("columns: " + ", ".join(f"{c.name} ({c.status}{', ' + str(c.mismatches) if c.mismatches else ''})" for c in bad[:12]))
        if self.metadata:
            parts.append("metadata: " + ", ".join(
                f"{m['check']}{' ' + m['column'] if m.get('column') else ''} expected {m['expected']!r} actual {m['actual']!r}"
                for m in self.metadata[:8]))
        if not parts and self.notes:
            parts.append(self.notes[-1])
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
    if opts.rel:
        within = within | (diff <= F.lit(opts.rel) * F.greatest(F.abs(a), F.abs(b)))
    return F.when(a.isNull() & b.isNull(), F.lit(False)).when(a.isNull() | b.isNull(), F.lit(True)).otherwise(~within)


def _within(ev, av, opts: CompareOptions) -> bool:
    """The numeric rule on two Python values (for samples)."""
    if ev is None or av is None:
        return ev is None and av is None
    d = abs(ev - av)
    return d < opts.abs_tol or (bool(opts.rel) and d <= opts.rel * max(abs(ev), abs(av)))


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
    result = CompareResult(name, False, "", [], n_exp, n_act, decimals=opts.decimals, tolerance=opts.tolerance)
    result.settings = {"ignored_columns": sorted(ignore), "column_map": {k.upper(): v.upper() for k, v in opts.column_map.items()},
                       "metadata_ignore": list(opts.metadata_ignore), "control_by": [c.upper() for c in opts.control_by]}
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
            # the keys define which rows must agree; comparing without them would not be the same check
            result.notes.append(f"key columns missing: {missing}; rows cannot be identified, so the outputs do not match")
            return result
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
    result.metadata = _compare_metadata(expected, actual, list(exp_cols), list(act_cols), opts)
    result.controls = control_totals(exp, act, [c for c in typed if exp_cols[c] == "num"],
                                     [c.upper() for c in opts.control_by if c.upper() in typed], n_exp, n_act,
                                     schema_meta(expected.schema))
    result.passed = (n_exp == n_act and result.only_in_expected == 0 and result.only_in_actual == 0
                     and all(c.status == "ok" for c in result.columns) and not result.metadata)
    return result


def control_totals(exp: DataFrame, act: DataFrame, nums: list[str], by: list[str], n_exp: int, n_act: int,
                   meta: dict | None = None, max_groups: int = 40) -> dict:
    """Totals a person can reconcile by hand, side by side: rows, then per numeric column the count of non-missing
    values, sum, minimum and maximum, then row counts per value of each ``by`` column.  Aggregates only, so they
    are safe to show when row samples are not (``allow_row_samples: false``), apart from small group counts."""
    meta = meta or {}

    def stats(df):
        if not nums:
            return {}
        aggs = []
        for c in nums:
            aggs += [F.count(c).alias(f"n__{c}"), F.sum(c).alias(f"s__{c}"), F.min(c).alias(f"lo__{c}"),
                     F.max(c).alias(f"hi__{c}")]
        r = df.agg(*aggs).collect()[0]
        return {c: (r[f"n__{c}"], r[f"s__{c}"], r[f"lo__{c}"], r[f"hi__{c}"]) for c in nums}

    se, sa = stats(exp), stats(act)
    numeric = []
    for c in nums:
        fmt = normalize_format(meta[c].format) if c in meta else None
        numeric.append({"column": c, "format": fmt,
                        "expected": dict(zip(("n", "sum", "min", "max"), se[c])),
                        "actual": dict(zip(("n", "sum", "min", "max"), sa[c]))})
    groups = []
    for c in by:
        ce = {r[c]: r["count"] for r in exp.groupBy(c).count().collect()}
        ca = {r[c]: r["count"] for r in act.groupBy(c).count().collect()}
        values = sorted(set(ce) | set(ca), key=lambda v: (v is None, str(v)))
        groups.append({"by": c, "truncated": len(values) > max_groups,
                       "rows": [{"value": v, "expected": ce.get(v, 0), "actual": ca.get(v, 0)} for v in values[:max_groups]]})
    return {"rows": {"expected": n_exp, "actual": n_act}, "numeric": numeric, "groups": groups}


def _compare_metadata(expected: DataFrame, actual: DataFrame, exp_cols: list[str], act_cols: list[str],
                      opts: CompareOptions) -> list[dict]:
    """Column order of the shared columns, then SAS length, format and label where the expected side has SAS metadata.

    A reference without SAS metadata (e.g. a Parquet oracle) cannot say what the formats, labels and lengths should
    be, so only the order is checked against it."""
    skip, skip_col = set(), set()
    for entry in opts.metadata_ignore:
        check, _, column = str(entry).partition(":")
        check = check.strip().lower()
        if check not in METADATA_CHECKS:
            raise ValueError(f"metadata_ignore: unknown check {entry!r}; use one of {list(METADATA_CHECKS)}, "
                             "optionally as check:COLUMN")
        if column.strip():
            skip_col.add((check, column.strip().upper()))
        else:
            skip.add(check)
    out: list[dict] = []
    common = [c for c in exp_cols if c in act_cols]
    if "order" not in skip:
        act_order = [c for c in act_cols if c in common]
        if act_order != common:
            out.append({"check": "order", "expected": common, "actual": act_order})
    back = {a.upper(): e.upper() for e, a in opts.column_map.items()}
    em = schema_meta(expected.schema)
    am = {back.get(k, k): v for k, v in schema_meta(actual.schema).items()}
    for c in common:
        e = em.get(c)
        if e is None:
            continue
        a = am.get(c)
        if a is None:
            out.append({"check": "metadata", "column": c, "expected": e.to_dict(), "actual": None})
            continue
        pairs = {"length": (e.length, a.length),
                 "format": (normalize_format(e.format), normalize_format(a.format)),
                 "label": ((e.label or "").strip() or None, (a.label or "").strip() or None)}
        for check, (ev, av) in pairs.items():
            if check not in skip and (check, c) not in skip_col and ev != av:
                out.append({"check": check, "column": c, "expected": ev, "actual": av})
    return out


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
            aggs.append(F.sum(F.when(both & ~_num_mismatch(ec, ac, opts) & _display_diff(ec, ac, opts.display_decimals), 1).otherwise(0)).alias(f"_disp__{c}"))
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
    result._diff_source = (j, keys, cols, kinds, flags)     # for diff_rows(); not part of the evidence
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
                    same = (ev == av) if kinds[c] == "char" else _within(ev, av, opts)
                    if same:
                        continue
                d[c] = {"expected": ev, "actual": av}
            result.samples.append(d)


_KEY_NAMES = {"_k_pos": "ROW_NUMBER", "_k_rank": "PAIR_RANK", "_k_dup": "DUP_SEQ"}


def diff_rows(result: CompareResult) -> DataFrame | None:
    """Every difference the comparison found, one row per differing cell, plus one per row present on only one side:
    the row's key, ``status``, ``column_name``, ``sas_value``, ``dbx_value`` (as text) and ``difference`` (Databricks minus
    SAS, numbers only).  The same cells the counts in ``result.columns`` count, so the two always agree.  ``None``
    when the comparison never paired rows (missing keys, an empty reference)."""
    src = getattr(result, "_diff_source", None)
    if src is None:
        return None
    j, keys, cols, kinds, flags = src
    key_cols = [F.col(k).alias(_KEY_NAMES.get(k, k)) for k in keys if k != "_k_nulls"]
    none_s, none_d = F.lit(None).cast("string"), F.lit(None).cast("double")
    parts = []
    for c in cols:
        ec, ac = F.col(f"e__{c}"), F.col(f"a__{c}")
        diff = (ac - ec) if kinds[c] == "num" else none_d
        parts.append(j.where(flags[c]).select(*key_cols, F.lit("value differs").alias("status"), F.lit(c).alias("column_name"),
                                              ec.cast("string").alias("sas_value"), ac.cast("string").alias("dbx_value"),
                                              diff.alias("difference")))
    for side, cond in (("only in SAS", F.col("_in_a").isNull()), ("only in Databricks", F.col("_in_e").isNull())):
        parts.append(j.where(cond).select(*key_cols, F.lit(side).alias("status"), none_s.alias("column_name"),
                                          none_s.alias("sas_value"), none_s.alias("dbx_value"), none_d.alias("difference")))
    return reduce(DataFrame.unionByName, parts)


def _compare_unordered(exp, act, cols, kinds, opts, result: CompareResult) -> None:
    """Rows have no identity, so pair them, then compare the pairs cell by cell with the same rule as keyed mode.

    Rows can only match when their character values are equal and the same numeric columns are missing, so those
    form the pairing group.  Within a group both sides are ranked by their numeric values (rounded to the parity
    decimals, or 9 when strict, first, so near-ties on one column defer to the next, then raw) and paired by rank.  For one numeric
    column this pairing finds a tolerance match whenever one exists.  Every pair is checked against the tolerance,
    so a pairing can produce a false FAIL (rare: several numeric columns tied near a rounding boundary) but never a
    false PASS."""
    chars = [c for c in cols if kinds[c] == "char"]
    nums = [c for c in cols if kinds[c] == "num"]
    nulls = (F.concat(*[F.when(F.col(c).isNull(), F.lit("1")).otherwise(F.lit("0")) for c in nums])
             if nums else F.lit(""))
    order = [F.round(F.col(c), 9 if opts.decimals is None else opts.decimals).asc_nulls_first() for c in nums] + [F.col(c).asc_nulls_first() for c in nums]
    w = Window.partitionBy(*chars, "_k_nulls").orderBy(*(order or [F.lit(0)]))

    def ranked(df):
        return df.withColumn("_k_nulls", nulls).withColumn("_k_rank", F.row_number().over(w))

    _compare_joined(ranked(exp), ranked(act), [*chars, "_k_nulls", "_k_rank"], nums, kinds, opts, result)
    if result.only_in_expected or result.only_in_actual or result.mismatched_columns:
        result.notes.append("unordered comparison: rows are paired within groups of equal character values by their "
                            "sorted numeric values; a row with no partner differs in a character value or in which "
                            "columns are missing. Supply keys for an exact row identity.")
