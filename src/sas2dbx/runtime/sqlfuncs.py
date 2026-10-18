"""SAS semantics as Spark SQL functions (single source of truth for DataFrame and SQL code).

Every function is registered per session as ``CREATE OR REPLACE TEMPORARY FUNCTION``.
Generated PySpark calls them through ``sas2dbx.runtime.functions`` (``S.*``) and
generated PROC SQL text calls them by name, so DATA steps and SQL share one
implementation.  Bodies avoid anything that changes behaviour with
``spark.sql.ansi.enabled`` (guarded casts, ``try_*`` functions, no reliance on
parse failures) and avoid non-foldable arguments where Spark requires literals.

Value model (see docs/ARCHITECTURE.md): numerics are DOUBLE with NULL = missing;
character values are right-trimmed STRING, blank = ''; dates are DOUBLE days
since 1960-01-01, datetimes DOUBLE seconds since 1960-01-01T00:00:00.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

try:  # pandas UDF type hints are resolved against module globals
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is always present on Databricks
    pd = None

SAS_EPOCH_UNIX_DAYS = 3653          # days from 1960-01-01 to 1970-01-01
YEARCUTOFF = 1926


@dataclass(frozen=True)
class SqlFn:
    name: str
    params: str
    returns: str
    body: str
    deps: tuple[str, ...] = ()
    doc: str = ""


@dataclass
class PyFn:
    name: str
    factory: Callable  # (spark) -> registered UDF
    deps: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, SqlFn | PyFn] = {}


def fn(name: str, params: str, returns: str, body: str, deps: tuple[str, ...] = (), doc: str = "") -> None:
    REGISTRY[name] = SqlFn(name, params, returns, body.strip(), deps, doc)


# ---------------------------------------------------------------------------
# dates & times
fn("sas_date", "d DOUBLE", "DATE",
   "CASE WHEN d IS NULL OR d < -715510 OR d > 2936549 THEN NULL "
   "ELSE date_from_unix_date(CAST(floor(d) AS INT) - 3653) END",
   doc="SAS date number -> DATE")
fn("sas_from_date", "dt DATE", "DOUBLE", "CAST(unix_date(dt) + 3653 AS DOUBLE)")
fn("sas_from_ts", "ts TIMESTAMP_NTZ", "DOUBLE",
   "CAST(unix_date(CAST(ts AS DATE)) + 3653 AS DOUBLE) * 86400 + hour(ts) * 3600 + minute(ts) * 60 "
   "+ CAST(extract(SECOND FROM ts) AS DOUBLE)")
fn("sas_expand_year", "y DOUBLE", "DOUBLE",
   f"CASE WHEN y >= 0 AND y < 100 THEN (CASE WHEN 1900 + floor(y) < {YEARCUTOFF} THEN 2000 + floor(y) ELSE 1900 + floor(y) END) "
   "ELSE y END")
fn("sas_mdy", "m DOUBLE, d DOUBLE, y DOUBLE", "DOUBLE",
   "CASE WHEN m IS NULL OR d IS NULL OR y IS NULL THEN NULL "
   "WHEN sas_expand_year(y) < 1 OR sas_expand_year(y) > 9999 OR floor(m) < 1 OR floor(m) > 12 OR floor(d) < 1 THEN NULL "
   "WHEN floor(d) > dayofmonth(last_day(make_date(CAST(sas_expand_year(y) AS INT), CAST(floor(m) AS INT), 1))) THEN NULL "
   "ELSE CAST(unix_date(make_date(CAST(sas_expand_year(y) AS INT), CAST(floor(m) AS INT), CAST(floor(d) AS INT))) + 3653 AS DOUBLE) END",
   deps=("sas_expand_year",))
fn("sas_year", "d DOUBLE", "DOUBLE", "CAST(year(sas_date(d)) AS DOUBLE)", ("sas_date",))
fn("sas_month", "d DOUBLE", "DOUBLE", "CAST(month(sas_date(d)) AS DOUBLE)", ("sas_date",))
fn("sas_day", "d DOUBLE", "DOUBLE", "CAST(dayofmonth(sas_date(d)) AS DOUBLE)", ("sas_date",))
fn("sas_qtr", "d DOUBLE", "DOUBLE", "CAST(quarter(sas_date(d)) AS DOUBLE)", ("sas_date",))
fn("sas_weekday", "d DOUBLE", "DOUBLE", "CAST(dayofweek(sas_date(d)) AS DOUBLE)", ("sas_date",),
   doc="1=Sunday..7=Saturday (same convention as Spark dayofweek)")
fn("sas_datepart", "dt DOUBLE", "DOUBLE", "CAST(floor(dt / 86400) AS DOUBLE)")
fn("sas_timepart", "dt DOUBLE", "DOUBLE", "dt - CAST(floor(dt / 86400) AS DOUBLE) * 86400")
fn("sas_dhms", "d DOUBLE, h DOUBLE, mi DOUBLE, s DOUBLE", "DOUBLE", "d * 86400 + h * 3600 + mi * 60 + s")
fn("sas_hms", "h DOUBLE, mi DOUBLE, s DOUBLE", "DOUBLE", "h * 3600 + mi * 60 + s")
fn("sas_hour", "t DOUBLE", "DOUBLE", "CAST(floor(pmod(t, 86400D) / 3600) AS DOUBLE)")
fn("sas_minute", "t DOUBLE", "DOUBLE", "CAST(floor(pmod(t, 3600D) / 60) AS DOUBLE)")
fn("sas_second", "t DOUBLE", "DOUBLE", "pmod(t, 60D)")

# INTNX / INTCK, month-based intervals (MONTH=1, QTR=3, SEMIYEAR=6, YEAR=12 months * multiplier),
# aligned to 1960-01 + (shift-1) months exactly like the Python reference.
fn("sas_mon_index", "d DOUBLE, span INT, shift INT", "BIGINT",
   "CAST(floor((year(sas_date(d)) * 12 + month(sas_date(d)) - 1 - (1960 * 12 + shift - 1)) / span) AS BIGINT)",
   ("sas_date",))
fn("sas_mon_start", "idx BIGINT, span INT, shift INT", "DATE",
   "make_date(CAST(floor((1960 * 12 + shift - 1 + idx * span) / 12) AS INT), "
   "CAST(pmod(1960 * 12 + shift - 1 + idx * span, 12) + 1 AS INT), 1)")
fn("sas_intnx_mon", "d DOUBLE, n DOUBLE, span INT, shift INT, align STRING", "DOUBLE",
   "CASE WHEN d IS NULL OR n IS NULL OR sas_date(d) IS NULL THEN NULL ELSE CAST(unix_date("
   "CASE upper(substring(coalesce(align, 'B'), 1, 1)) "
   "WHEN 'E' THEN date_add(add_months(sas_mon_start(sas_mon_index(d, span, shift) + CAST(n AS BIGINT), span, shift), span), -1) "
   "WHEN 'M' THEN date_add(sas_mon_start(sas_mon_index(d, span, shift) + CAST(n AS BIGINT), span, shift), "
   "  CAST(floor(datediff(date_add(add_months(sas_mon_start(sas_mon_index(d, span, shift) + CAST(n AS BIGINT), span, shift), span), -1), "
   "  sas_mon_start(sas_mon_index(d, span, shift) + CAST(n AS BIGINT), span, shift)) / 2) AS INT)) "
   "WHEN 'S' THEN add_months(sas_date(d), CAST(CAST(n AS BIGINT) * span AS INT)) "
   "ELSE sas_mon_start(sas_mon_index(d, span, shift) + CAST(n AS BIGINT), span, shift) END) + 3653 AS DOUBLE) END",
   ("sas_date", "sas_mon_index", "sas_mon_start"))
fn("sas_intnx_day", "d DOUBLE, n DOUBLE, mult INT, align STRING", "DOUBLE",
   "CASE WHEN d IS NULL OR n IS NULL THEN NULL ELSE "
   "CASE upper(substring(coalesce(align, 'B'), 1, 1)) "
   "WHEN 'E' THEN (floor(d / mult) + CAST(n AS BIGINT)) * mult + mult - 1 "
   "WHEN 'M' THEN (floor(d / mult) + CAST(n AS BIGINT)) * mult + floor((mult - 1) / 2) "
   "WHEN 'S' THEN d + CAST(n AS BIGINT) * mult "
   "ELSE (floor(d / mult) + CAST(n AS BIGINT)) * mult END END")
fn("sas_intnx_week", "d DOUBLE, n DOUBLE, mult INT, shift INT, align STRING", "DOUBLE",
   "CASE WHEN d IS NULL OR n IS NULL THEN NULL ELSE "
   "CASE upper(substring(coalesce(align, 'B'), 1, 1)) "
   "WHEN 'E' THEN (floor((d + 5 - (shift - 1)) / (7 * mult)) + CAST(n AS BIGINT)) * 7 * mult - 5 + (shift - 1) + 7 * mult - 1 "
   "WHEN 'M' THEN (floor((d + 5 - (shift - 1)) / (7 * mult)) + CAST(n AS BIGINT)) * 7 * mult - 5 + (shift - 1) + floor((7 * mult - 1) / 2) "
   "WHEN 'S' THEN d + 7 * CAST(n AS BIGINT) * mult "
   "ELSE (floor((d + 5 - (shift - 1)) / (7 * mult)) + CAST(n AS BIGINT)) * 7 * mult - 5 + (shift - 1) END END")
fn("sas_intnx_time", "t DOUBLE, n DOUBLE, size DOUBLE, align STRING", "DOUBLE",
   "CASE WHEN t IS NULL OR n IS NULL THEN NULL ELSE "
   "CASE upper(substring(coalesce(align, 'B'), 1, 1)) "
   "WHEN 'E' THEN (floor(t / size) + CAST(n AS BIGINT) + 1) * size - 1 "
   "WHEN 'M' THEN (floor(t / size) + CAST(n AS BIGINT)) * size + size / 2 "
   "ELSE (floor(t / size) + CAST(n AS BIGINT)) * size END END")
fn("sas_intck_mon", "a DOUBLE, b DOUBLE, span INT, shift INT", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL OR sas_date(a) IS NULL OR sas_date(b) IS NULL THEN NULL "
   "ELSE CAST(sas_mon_index(b, span, shift) - sas_mon_index(a, span, shift) AS DOUBLE) END",
   ("sas_date", "sas_mon_index"))
fn("sas_intck_mon_c", "a DOUBLE, b DOUBLE, span INT", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL OR sas_date(a) IS NULL OR sas_date(b) IS NULL THEN NULL "
   "ELSE CAST(((year(sas_date(b)) * 12 + month(sas_date(b))) - (year(sas_date(a)) * 12 + month(sas_date(a))) "
   "- CASE WHEN b >= a AND add_months(sas_date(a), (year(sas_date(b)) * 12 + month(sas_date(b))) - (year(sas_date(a)) * 12 + month(sas_date(a)))) > sas_date(b) THEN 1 "
   "       WHEN b < a AND add_months(sas_date(a), (year(sas_date(b)) * 12 + month(sas_date(b))) - (year(sas_date(a)) * 12 + month(sas_date(a)))) < sas_date(b) THEN -1 "
   "       ELSE 0 END) div span AS DOUBLE) END",
   ("sas_date",))
fn("sas_intck_day", "a DOUBLE, b DOUBLE, mult INT", "DOUBLE",
   "CAST(floor(b / mult) - floor(a / mult) AS DOUBLE)")
fn("sas_intck_week", "a DOUBLE, b DOUBLE, mult INT, shift INT", "DOUBLE",
   "CAST(floor((b + 5 - (shift - 1)) / (7 * mult)) - floor((a + 5 - (shift - 1)) / (7 * mult)) AS DOUBLE)")
fn("sas_intck_cont", "a DOUBLE, b DOUBLE, size DOUBLE", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL THEN NULL ELSE CAST(CAST((b - a) / size AS BIGINT) AS DOUBLE) END")
fn("sas_intck_time", "a DOUBLE, b DOUBLE, size DOUBLE", "DOUBLE",
   "CAST(floor(b / size) - floor(a / size) AS DOUBLE)")
fn("sas_days_in_year", "y INT", "INT",
   "CASE WHEN (y % 4 = 0 AND y % 100 <> 0) OR y % 400 = 0 THEN 366 ELSE 365 END")
fn("sas_yrdif_act", "a DOUBLE, b DOUBLE", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL THEN NULL "
   "WHEN year(sas_date(a)) = year(sas_date(b)) THEN datediff(sas_date(b), sas_date(a)) / sas_days_in_year(year(sas_date(a))) "
   "ELSE datediff(make_date(year(sas_date(a)) + 1, 1, 1), sas_date(a)) / sas_days_in_year(year(sas_date(a))) "
   "+ (year(sas_date(b)) - year(sas_date(a)) - 1) "
   "+ datediff(sas_date(b), make_date(year(sas_date(b)), 1, 1)) / sas_days_in_year(year(sas_date(b))) END",
   ("sas_date", "sas_days_in_year"))
fn("sas_anniv", "a DATE, y INT", "DATE",
   "CASE WHEN month(a) = 2 AND dayofmonth(a) = 29 AND sas_days_in_year(y) = 365 THEN make_date(y, 3, 1) "
   "ELSE make_date(y, month(a), dayofmonth(a)) END", ("sas_days_in_year",))
fn("sas_age_years", "a DATE, b DATE", "INT",
   "year(b) - year(a) - CASE WHEN sas_anniv(a, year(b)) > b THEN 1 ELSE 0 END", ("sas_anniv",))
fn("sas_yrdif_age", "a DOUBLE, b DOUBLE", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL THEN NULL ELSE "
   "sas_age_years(sas_date(a), sas_date(b)) + "
   "datediff(sas_date(b), sas_anniv(sas_date(a), year(sas_date(a)) + sas_age_years(sas_date(a), sas_date(b)))) / "
   "datediff(sas_anniv(sas_date(a), year(sas_date(a)) + sas_age_years(sas_date(a), sas_date(b)) + 1), "
   "         sas_anniv(sas_date(a), year(sas_date(a)) + sas_age_years(sas_date(a), sas_date(b)))) END",
   ("sas_date", "sas_anniv", "sas_age_years"))
fn("sas_yrdif_act360", "a DOUBLE, b DOUBLE", "DOUBLE", "(b - a) / 360")
fn("sas_yrdif_act365", "a DOUBLE, b DOUBLE", "DOUBLE", "(b - a) / 365")

# ---------------------------------------------------------------------------
# comparisons, booleans, arithmetic
fn("sas_lt", "a DOUBLE, b DOUBLE", "BOOLEAN",
   "CASE WHEN a IS NULL THEN b IS NOT NULL WHEN b IS NULL THEN false ELSE a < b END",
   doc="missing sorts below every number")
fn("sas_le", "a DOUBLE, b DOUBLE", "BOOLEAN",
   "CASE WHEN a IS NULL THEN true WHEN b IS NULL THEN false ELSE a <= b END")
fn("sas_gt", "a DOUBLE, b DOUBLE", "BOOLEAN",
   "CASE WHEN a IS NULL THEN false WHEN b IS NULL THEN true ELSE a > b END")
fn("sas_ge", "a DOUBLE, b DOUBLE", "BOOLEAN",
   "CASE WHEN b IS NULL THEN true WHEN a IS NULL THEN false ELSE a >= b END")
fn("sas_truth", "x DOUBLE", "BOOLEAN", "coalesce(x <> 0, false)")
fn("sas_bool", "b BOOLEAN", "DOUBLE", "CASE WHEN b THEN 1.0D ELSE 0.0D END")
fn("sas_minop", "a DOUBLE, b DOUBLE", "DOUBLE", "CASE WHEN a IS NULL OR b IS NULL THEN NULL ELSE least(a, b) END")
fn("sas_maxop", "a DOUBLE, b DOUBLE", "DOUBLE", "greatest(a, b)")
fn("sas_finite", "x DOUBLE", "DOUBLE",
   "CASE WHEN isnan(x) OR x = double('Infinity') OR x = double('-Infinity') THEN NULL ELSE x END")
fn("sas_pow", "a DOUBLE, b DOUBLE", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL THEN NULL ELSE sas_finite(power(a, b)) END", ("sas_finite",))
fn("sas_sqrt", "x DOUBLE", "DOUBLE", "CASE WHEN x >= 0 THEN sqrt(x) END")
fn("sas_log", "x DOUBLE", "DOUBLE", "CASE WHEN x > 0 THEN ln(x) END")
fn("sas_log10", "x DOUBLE", "DOUBLE", "CASE WHEN x > 0 THEN log10(x) END")
fn("sas_log2", "x DOUBLE", "DOUBLE", "CASE WHEN x > 0 THEN log2(x) END")
fn("sas_exp", "x DOUBLE", "DOUBLE", "CASE WHEN x <= 709.78 THEN exp(x) END")
fn("sas_is_pow10", "u DOUBLE", "BOOLEAN", "abs(log10(u) - round(log10(u))) < 1e-9")
fn("sas_round", "x DOUBLE, u DOUBLE", "DOUBLE",
   "CASE WHEN x IS NULL OR u IS NULL OR u <= 0 THEN NULL "
   "WHEN abs(x) >= 1e15 OR u < 1e-7 OR u > 1e6 THEN x "
   "WHEN sas_is_pow10(u) THEN CAST(round(CAST(x AS DECIMAL(28,12)) * CAST(1 / u AS DECIMAL(13,6)), 0) / CAST(1 / u AS DECIMAL(13,6)) AS DOUBLE) "
   "ELSE CAST(round(CAST(x AS DECIMAL(28,12)) / CAST(u AS DECIMAL(28,12)), 0) * CAST(u AS DECIMAL(28,12)) AS DOUBLE) END",
   ("sas_is_pow10",),
   doc="half away from zero on the decimal representation (2.675 -> 2.68)")
fn("sas_round_d", "x DOUBLE, d INT", "DOUBLE",
   "CASE WHEN x IS NULL THEN NULL WHEN abs(x) >= 1e15 THEN x "
   "ELSE CAST(round(CAST(x AS DECIMAL(28,12)) * CAST(power(10, d) AS DECIMAL(16,0)), 0) / CAST(power(10, d) AS DECIMAL(16,0)) AS DOUBLE) + 0.0D END")
fn("sas_fuzz_int", "x DOUBLE", "BOOLEAN", "abs(x - round(x)) <= 1e-12 * greatest(1.0D, abs(x))")
fn("sas_int", "x DOUBLE", "DOUBLE",
   "CASE WHEN x IS NULL THEN NULL WHEN abs(x) >= 9e15 THEN x WHEN sas_fuzz_int(x) THEN CAST(round(x) AS DOUBLE) "
   "WHEN x >= 0 THEN CAST(floor(x) AS DOUBLE) ELSE CAST(ceil(x) AS DOUBLE) END", ("sas_fuzz_int",))
fn("sas_floor", "x DOUBLE", "DOUBLE",
   "CASE WHEN x IS NULL THEN NULL WHEN abs(x) >= 9e15 THEN x WHEN sas_fuzz_int(x) THEN CAST(round(x) AS DOUBLE) "
   "ELSE CAST(floor(x) AS DOUBLE) END", ("sas_fuzz_int",))
fn("sas_ceil", "x DOUBLE", "DOUBLE",
   "CASE WHEN x IS NULL THEN NULL WHEN abs(x) >= 9e15 THEN x WHEN sas_fuzz_int(x) THEN CAST(round(x) AS DOUBLE) "
   "ELSE CAST(ceil(x) AS DOUBLE) END", ("sas_fuzz_int",))
fn("sas_mod", "a DOUBLE, b DOUBLE", "DOUBLE",
   "CASE WHEN a IS NULL OR b IS NULL OR b = 0 THEN NULL "
   "WHEN abs(a % b) <= 1e-12 * greatest(1.0D, abs(b)) OR abs(abs(a % b) - abs(b)) <= 1e-12 * greatest(1.0D, abs(b)) THEN 0.0D "
   "ELSE a % b END")
fn("sas_nn", "a ARRAY<DOUBLE>", "ARRAY<DOUBLE>", "filter(a, x -> x IS NOT NULL)")
fn("sas_sum_arr", "a ARRAY<DOUBLE>", "DOUBLE",
   "CASE WHEN size(sas_nn(a)) = 0 THEN NULL ELSE aggregate(sas_nn(a), 0.0D, (acc, x) -> acc + x) END", ("sas_nn",))
fn("sas_mean_arr", "a ARRAY<DOUBLE>", "DOUBLE",
   "CASE WHEN size(sas_nn(a)) = 0 THEN NULL ELSE aggregate(sas_nn(a), 0.0D, (acc, x) -> acc + x) / size(sas_nn(a)) END", ("sas_nn",))
fn("sas_min_arr", "a ARRAY<DOUBLE>", "DOUBLE", "array_min(sas_nn(a))", ("sas_nn",))
fn("sas_max_arr", "a ARRAY<DOUBLE>", "DOUBLE", "array_max(sas_nn(a))", ("sas_nn",))
fn("sas_n_arr", "a ARRAY<DOUBLE>", "DOUBLE", "CAST(size(sas_nn(a)) AS DOUBLE)", ("sas_nn",))
fn("sas_nmiss_arr", "a ARRAY<DOUBLE>", "DOUBLE", "CAST(size(a) - size(sas_nn(a)) AS DOUBLE)", ("sas_nn",))
fn("sas_range_arr", "a ARRAY<DOUBLE>", "DOUBLE", "array_max(sas_nn(a)) - array_min(sas_nn(a))", ("sas_nn",))
fn("sas_var_arr", "a ARRAY<DOUBLE>", "DOUBLE",
   "CASE WHEN size(sas_nn(a)) < 2 THEN NULL ELSE "
   "aggregate(sas_nn(a), 0.0D, (acc, x) -> acc + power(x - sas_mean_arr(a), 2)) / (size(sas_nn(a)) - 1) END",
   ("sas_nn", "sas_mean_arr"))
fn("sas_std_arr", "a ARRAY<DOUBLE>", "DOUBLE", "sqrt(sas_var_arr(a))", ("sas_var_arr",))
fn("sas_median_arr", "a ARRAY<DOUBLE>", "DOUBLE",
   "CASE WHEN size(sas_nn(a)) = 0 THEN NULL WHEN size(sas_nn(a)) % 2 = 1 "
   "THEN element_at(array_sort(sas_nn(a)), CAST(size(sas_nn(a)) / 2 AS INT) + 1) "
   "ELSE (element_at(array_sort(sas_nn(a)), CAST(size(sas_nn(a)) / 2 AS INT)) + element_at(array_sort(sas_nn(a)), CAST(size(sas_nn(a)) / 2 AS INT) + 1)) / 2 END",
   ("sas_nn",))
fn("sas_coalesce_arr", "a ARRAY<DOUBLE>", "DOUBLE", "try_element_at(sas_nn(a), 1)", ("sas_nn",))
fn("sas_coalescec_arr", "a ARRAY<STRING>", "STRING",
   "coalesce(try_element_at(filter(a, x -> trim(coalesce(x, '')) <> ''), 1), '')")

# ---------------------------------------------------------------------------
# character functions (inputs are right-trimmed, blank = '')
fn("sas_substr2", "s STRING, p DOUBLE", "STRING",
   "CASE WHEN p IS NULL OR p < 1 THEN '' ELSE coalesce(substring(s, CAST(floor(p) AS INT)), '') END")
fn("sas_substr3", "s STRING, p DOUBLE, n DOUBLE", "STRING",
   "CASE WHEN p IS NULL OR n IS NULL OR p < 1 OR n < 1 THEN '' "
   "ELSE coalesce(substring(s, CAST(floor(p) AS INT), CAST(floor(n) AS INT)), '') END")
fn("sas_substrn3", "s STRING, p DOUBLE, n DOUBLE", "STRING",
   "CASE WHEN p IS NULL OR n IS NULL OR floor(p) + floor(n) <= greatest(floor(p), 1) THEN '' "
   "ELSE coalesce(substring(s, CAST(greatest(floor(p), 1) AS INT), CAST(floor(p) + floor(n) - greatest(floor(p), 1) AS INT)), '') END")
fn("sas_index", "s STRING, t STRING", "DOUBLE",
   "CAST(instr(coalesce(s, ''), CASE WHEN coalesce(t, '') = '' THEN ' ' ELSE t END) AS DOUBLE)")
fn("sas_chars", "s STRING", "ARRAY<STRING>",
   "CASE WHEN coalesce(s, '') = '' THEN array() ELSE split(s, '') END")
fn("sas_indexc", "s STRING, chars STRING", "DOUBLE",
   "CAST(array_position(transform(sas_chars(s), c -> instr(chars, c) > 0), true) AS DOUBLE)", ("sas_chars",))
fn("sas_verify", "s STRING, chars STRING", "DOUBLE",
   "CAST(array_position(transform(sas_chars(s), c -> instr(chars, c) = 0), true) AS DOUBLE)", ("sas_chars",))
fn("sas_find", "s STRING, t STRING, mods STRING, start DOUBLE", "DOUBLE",
   "CAST(locate("
   "CASE WHEN instr(lower(coalesce(mods, '')), 'i') > 0 THEN lower(CASE WHEN instr(lower(coalesce(mods, '')), 't') > 0 THEN rtrim(t) ELSE t END) "
   "     ELSE (CASE WHEN instr(lower(coalesce(mods, '')), 't') > 0 THEN rtrim(t) ELSE t END) END, "
   "CASE WHEN instr(lower(coalesce(mods, '')), 'i') > 0 THEN lower(s) ELSE s END, "
   "CAST(greatest(coalesce(start, 1), 1) AS INT)) AS DOUBLE)")
fn("sas_scan", "s STRING, n DOUBLE, rx STRING", "STRING",
   "CASE WHEN n IS NULL OR CAST(n AS INT) = 0 THEN '' "
   "ELSE coalesce(try_element_at(filter(split(coalesce(s, ''), rx), w -> w <> ''), CAST(n AS INT)), '') END",
   doc="rx is a regex character class of delimiters followed by '+'")
fn("sas_scan_m", "s STRING, n DOUBLE, rx STRING", "STRING",
   "CASE WHEN n IS NULL OR CAST(n AS INT) = 0 THEN '' "
   "ELSE coalesce(try_element_at(split(coalesce(s, ''), rx), CAST(n AS INT)), '') END")
fn("sas_countw", "s STRING, rx STRING", "DOUBLE",
   "CAST(size(filter(split(coalesce(s, ''), rx), w -> w <> '')) AS DOUBLE)")
fn("sas_compress_rx", "s STRING, rx STRING", "STRING", "regexp_replace(coalesce(s, ''), rx, '')",
   doc="rx is a character class of characters to remove (or a negated class to keep)")
fn("sas_compbl", "s STRING", "STRING", "regexp_replace(coalesce(s, ''), ' {2,}', ' ')")
fn("sas_tranwrd", "s STRING, f STRING, t STRING", "STRING",
   "replace(coalesce(s, ''), CASE WHEN coalesce(f, '') = '' THEN ' ' ELSE f END, CASE WHEN coalesce(t, '') = '' THEN ' ' ELSE t END)")
fn("sas_transtrn", "s STRING, f STRING, t STRING", "STRING",
   "CASE WHEN coalesce(f, '') = '' THEN coalesce(s, '') ELSE replace(coalesce(s, ''), f, coalesce(t, '')) END")
fn("sas_translate", "s STRING, to_ STRING, from_ STRING", "STRING",
   "translate(coalesce(s, ''), from_, rpad(coalesce(to_, ''), length(from_), ' '))",
   doc="SAS argument order (source, to, from); unmatched 'from' characters become blanks")
fn("sas_right", "s STRING, l INT", "STRING", "lpad(rtrim(coalesce(s, '')), l, ' ')")
fn("sas_trim", "s STRING", "STRING", "CASE WHEN rtrim(coalesce(s, '')) = '' THEN ' ' ELSE rtrim(s) END",
   doc="TRIM of a blank value is one blank")
fn("sas_propcase", "s STRING, delims STRING", "STRING",
   "CASE WHEN coalesce(s, '') = '' THEN '' ELSE array_join(transform(sequence(1, length(s)), i -> "
   "CASE WHEN i = 1 OR instr(delims, substring(s, i - 1, 1)) > 0 THEN upper(substring(s, i, 1)) "
   "ELSE lower(substring(s, i, 1)) END), '') END")
fn("sas_length", "s STRING", "DOUBLE",
   "CASE WHEN rtrim(coalesce(s, '')) = '' THEN 1.0D ELSE CAST(length(rtrim(s)) AS DOUBLE) END")
fn("sas_lengthn", "s STRING", "DOUBLE", "CAST(length(rtrim(coalesce(s, ''))) AS DOUBLE)")
fn("sas_repeat", "s STRING, n DOUBLE", "STRING", "repeat(coalesce(s, ''), CAST(n AS INT) + 1)")
fn("sas_cat_arr", "a ARRAY<STRING>", "STRING", "array_join(transform(a, x -> coalesce(x, '')), '')")
fn("sas_cats_arr", "a ARRAY<STRING>", "STRING", "array_join(transform(a, x -> trim(coalesce(x, ''))), '')")
fn("sas_catt_arr", "a ARRAY<STRING>", "STRING", "array_join(transform(a, x -> rtrim(coalesce(x, ''))), '')")
fn("sas_catx_arr", "sep STRING, a ARRAY<STRING>", "STRING",
   "array_join(filter(transform(a, x -> trim(coalesce(x, ''))), x -> x <> ''), coalesce(sep, ''))")
fn("sas_quote", "s STRING", "STRING", "concat('\"', replace(rtrim(coalesce(s, '')), '\"', '\"\"'), '\"')")
fn("sas_dequote", "s STRING", "STRING",
   "CASE WHEN length(s) >= 2 AND substring(s, 1, 1) IN ('\"', '\\'') AND substring(s, length(s), 1) = substring(s, 1, 1) "
   "THEN replace(substring(s, 2, length(s) - 2), repeat(substring(s, 1, 1), 2), substring(s, 1, 1)) ELSE coalesce(s, '') END")
fn("sas_count", "s STRING, t STRING", "DOUBLE",
   "CASE WHEN coalesce(t, '') = '' THEN 0.0D ELSE CAST((length(coalesce(s, '')) - length(replace(coalesce(s, ''), t, ''))) / length(t) AS DOUBLE) END")
fn("sas_countc", "s STRING, chars STRING", "DOUBLE",
   "CAST(size(filter(sas_chars(s), c -> instr(chars, c) > 0)) AS DOUBLE)", ("sas_chars",))
for _name, _cls in [("anydigit", "[0-9]"), ("anyalpha", "[A-Za-z]"), ("anyalnum", "[A-Za-z0-9]"),
                    ("anyspace", "\\\\s"), ("anyupper", "[A-Z]"), ("anylower", "[a-z]"),
                    ("anypunct", "\\\\p{Punct}")]:
    fn(f"sas_{_name}", "s STRING", "DOUBLE",
       f"CAST(array_position(transform(sas_chars(s), c -> c RLIKE '^{_cls}$'), true) AS DOUBLE)", ("sas_chars",))
for _name, _cls in [("notdigit", "[0-9]"), ("notalpha", "[A-Za-z]"), ("notspace", "\\\\s"), ("notalnum", "[A-Za-z0-9]")]:
    fn(f"sas_{_name}", "s STRING", "DOUBLE",
       f"CAST(array_position(transform(sas_chars(rtrim(s)), c -> NOT (c RLIKE '^{_cls}$')), true) AS DOUBLE)", ("sas_chars",))
fn("sas_prxmatch", "rx STRING, s STRING", "DOUBLE",
   "CASE WHEN coalesce(s, '') RLIKE rx THEN CAST(length(regexp_extract(s, concat('^(.*?)(', rx, ')'), 1)) + 1 AS DOUBLE) ELSE 0.0D END")
fn("sas_prxchange", "rx STRING, s STRING, rep STRING", "STRING", "regexp_replace(coalesce(s, ''), rx, rep)")

# ---------------------------------------------------------------------------
# formats (PUT)
fn("sas_fmt_fixed", "x DOUBLE, d INT", "STRING",
   "format_string(concat('%.', d, 'f'), sas_round_d(x, d))", ("sas_round_d",))
fn("sas_fit", "s STRING, x DOUBLE, w INT", "STRING",
   "CASE WHEN length(s) <= w THEN lpad(s, w, ' ') ELSE sas_put_best(x, w) END", ("sas_put_best",),
   doc="a formatted value wider than the field falls back to BESTw. (as SAS does)")
fn("sas_put_wd", "x DOUBLE, w INT, d INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') "
   "WHEN length(sas_fmt_fixed(x, d)) <= w THEN lpad(sas_fmt_fixed(x, d), w, ' ') "
   "ELSE sas_put_best(x, w) END",
   ("sas_fmt_fixed", "sas_put_best"))
fn("sas_put_z", "x DOUBLE, w INT, d INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') "
   "WHEN length(sas_fmt_fixed(abs(x), d)) + CASE WHEN x < 0 AND sas_round_d(x, d) <> 0 THEN 1 ELSE 0 END > w THEN sas_put_best(x, w) "
   "WHEN x < 0 AND sas_round_d(x, d) <> 0 THEN concat('-', lpad(sas_fmt_fixed(abs(x), d), w - 1, '0')) "
   "ELSE lpad(sas_fmt_fixed(abs(x), d), w, '0') END", ("sas_fmt_fixed", "sas_round_d", "sas_put_best"))
fn("sas_put_comma", "x DOUBLE, w INT, d INT, dollar BOOLEAN", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') ELSE sas_fit(concat("
   "CASE WHEN sas_round_d(x, d) < 0 THEN '-' ELSE '' END, CASE WHEN dollar THEN '$' ELSE '' END, "
   "format_number(abs(sas_round_d(x, d)), d)), x, w) END", ("sas_round_d", "sas_fit"))
fn("sas_put_percent", "x DOUBLE, w INT, d INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') "
   "WHEN sas_round_d(x * 100, d) < 0 THEN sas_fit(concat('(', sas_fmt_fixed(abs(x * 100), d), '%)'), x, w) "
   "ELSE sas_fit(concat(sas_fmt_fixed(x * 100, d), '%'), x, w) END", ("sas_round_d", "sas_fmt_fixed", "sas_fit"))
fn("sas_put_best", "x DOUBLE, w INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') "
   "WHEN x = 0 THEN lpad('0', w, ' ') "
   "WHEN abs(x) < 1e15 AND x = floor(x) AND length(CAST(CAST(x AS BIGINT) AS STRING)) <= w "
   "THEN lpad(CAST(CAST(x AS BIGINT) AS STRING), w, ' ') "
   "ELSE sas_put_best_py(x, w) END", ("sas_put_best_py",),
   doc="integers in SQL; other values through the Python reference (BESTw. selection rules)")
fn("sas_put_date", "x DOUBLE, w INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') WHEN sas_date(x) IS NULL THEN repeat('*', w) "
   "WHEN w >= 11 THEN lpad(upper(date_format(sas_date(x), 'dd-MMM-yyyy')), w, ' ') "
   "WHEN w >= 9 THEN lpad(upper(date_format(sas_date(x), 'ddMMMyyyy')), w, ' ') "
   "WHEN w >= 7 THEN lpad(upper(date_format(sas_date(x), 'ddMMMyy')), w, ' ') "
   "WHEN w >= 5 THEN lpad(upper(date_format(sas_date(x), 'ddMMM')), w, ' ') "
   "ELSE lpad(date_format(sas_date(x), 'dd'), w, ' ') END", ("sas_date",))
fn("sas_ymd_parts", "x DOUBLE, y4 BOOLEAN", "ARRAY<STRING>",
   "array(CASE WHEN y4 THEN lpad(CAST(year(sas_date(x)) AS STRING), 4, '0') ELSE lpad(CAST(year(sas_date(x)) % 100 AS STRING), 2, '0') END, "
   "lpad(CAST(month(sas_date(x)) AS STRING), 2, '0'), lpad(CAST(dayofmonth(sas_date(x)) AS STRING), 2, '0'))", ("sas_date",))
fn("sas_put_ymd", "x DOUBLE, w INT, sep STRING, ord STRING", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') WHEN sas_date(x) IS NULL THEN repeat('*', w) "
   "ELSE lpad(array_join(transform(split(ord, ''), c -> element_at(sas_ymd_parts(x, "
   "CASE WHEN sep = '' THEN w >= 8 ELSE w >= 10 END), CASE c WHEN 'y' THEN 1 WHEN 'm' THEN 2 ELSE 3 END)), sep), w, ' ') END",
   ("sas_date", "sas_ymd_parts"))
fn("sas_put_dfmt", "x DOUBLE, w INT, pattern STRING, up BOOLEAN", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') WHEN sas_date(x) IS NULL THEN repeat('*', w) "
   "ELSE lpad(CASE WHEN up THEN upper(date_format(sas_date(x), pattern)) ELSE date_format(sas_date(x), pattern) END, w, ' ') END",
   ("sas_date",), doc="date formats expressible as a single Java pattern (MONYY, WORDDATE, E8601DA ...)")
fn("sas_hms_str", "secs DOUBLE, pad_hours BOOLEAN", "STRING",
   "concat(CASE WHEN pad_hours THEN lpad(CAST(CAST(floor(secs / 3600) AS BIGINT) AS STRING), 2, '0') "
   "ELSE CAST(CAST(floor(secs / 3600) AS BIGINT) AS STRING) END, ':', "
   "lpad(CAST(CAST(floor(pmod(secs, 3600D) / 60) AS BIGINT) AS STRING), 2, '0'), ':', "
   "lpad(CAST(CAST(floor(pmod(secs, 60D)) AS BIGINT) AS STRING), 2, '0'))")
fn("sas_put_datetime", "x DOUBLE, w INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') WHEN sas_date(floor(x / 86400)) IS NULL THEN repeat('*', w) "
   "ELSE lpad(concat(upper(date_format(sas_date(floor(x / 86400)), CASE WHEN w >= 18 THEN 'ddMMMyyyy' ELSE 'ddMMMyy' END)), ':', "
   "CASE WHEN w >= 16 THEN sas_hms_str(pmod(round(x), 86400D), true) "
   "ELSE substring(sas_hms_str(pmod(round(x), 86400D), true), 1, 5) END), w, ' ') END",
   ("sas_date", "sas_hms_str"))
fn("sas_put_isodt", "x DOUBLE, w INT", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') WHEN sas_date(floor(x / 86400)) IS NULL THEN repeat('*', w) "
   "ELSE lpad(concat(date_format(sas_date(floor(x / 86400)), 'yyyy-MM-dd'), 'T', sas_hms_str(pmod(round(x), 86400D), true)), w, ' ') END",
   ("sas_date", "sas_hms_str"))
fn("sas_put_time", "x DOUBLE, w INT, pad_hours BOOLEAN", "STRING",
   "CASE WHEN x IS NULL THEN lpad('.', w, ' ') "
   "WHEN w < 8 THEN lpad(concat(CASE WHEN x < 0 THEN '-' ELSE '' END, substring(sas_hms_str(round(abs(x)), pad_hours), 1, "
   "  length(sas_hms_str(round(abs(x)), pad_hours)) - 3)), w, ' ') "
   "ELSE lpad(concat(CASE WHEN x < 0 THEN '-' ELSE '' END, sas_hms_str(round(abs(x)), pad_hours)), w, ' ') END",
   ("sas_hms_str",))
fn("sas_put_char", "s STRING, w INT", "STRING", "rpad(substring(coalesce(s, ''), 1, w), w, ' ')")

# ---------------------------------------------------------------------------
# informats (INPUT)
_NUM_RX = "^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?$"
fn("sas_inw", "s STRING, w INT", "STRING", "trim(CASE WHEN w IS NULL THEN coalesce(s, '') ELSE substring(coalesce(s, ''), 1, w) END)")
fn("sas_input_num", "s STRING, w INT, d INT", "DOUBLE",
   f"CASE WHEN sas_inw(s, w) RLIKE '{_NUM_RX}' THEN CAST(sas_inw(s, w) AS DOUBLE) / "
   "(CASE WHEN coalesce(d, 0) > 0 AND instr(sas_inw(s, w), '.') = 0 AND NOT (sas_inw(s, w) RLIKE '[eE]') THEN power(10, d) ELSE 1 END) END",
   ("sas_inw",))
fn("sas_input_comma", "s STRING, w INT, d INT", "DOUBLE",
   "CASE WHEN regexp_replace(sas_inw(s, w), '[,$ ]', '') RLIKE '^[(].*[)]$' "
   "THEN -sas_input_num(substring(regexp_replace(sas_inw(s, w), '[,$ ]', ''), 2, length(regexp_replace(sas_inw(s, w), '[,$ ]', '')) - 2), NULL, d) "
   "ELSE sas_input_num(regexp_replace(sas_inw(s, w), '[,$ ]', ''), NULL, d) END", ("sas_inw", "sas_input_num"))
fn("sas_input_pct", "s STRING, w INT", "DOUBLE",
   "CASE WHEN regexp_replace(sas_inw(s, w), '[%,]', '') RLIKE '^[(].*[)]$' "
   "THEN -sas_input_num(substring(regexp_replace(sas_inw(s, w), '[%,]', ''), 2, length(regexp_replace(sas_inw(s, w), '[%,]', '')) - 2), NULL, 0) / 100 "
   "ELSE sas_input_num(regexp_replace(sas_inw(s, w), '[%,]', ''), NULL, 0) / 100 END", ("sas_inw", "sas_input_num"))
_SEP = "[-/.: ]"
_YMD_RX = f"^([0-9]{{4}}|[0-9]{{2}}){_SEP}([0-9]{{1,2}}){_SEP}([0-9]{{1,2}})$"
fn("sas_input_ymd", "s STRING, w INT", "DOUBLE",
   "CASE WHEN sas_inw(s, w) RLIKE '^[0-9]{8}$' THEN sas_mdy(try_cast(substring(sas_inw(s, w), 5, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 7, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 1, 4) AS DOUBLE)) "
   "WHEN sas_inw(s, w) RLIKE '^[0-9]{6}$' THEN sas_mdy(try_cast(substring(sas_inw(s, w), 3, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 5, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 1, 2) AS DOUBLE)) "
   f"WHEN sas_inw(s, w) RLIKE '{_YMD_RX}' THEN sas_mdy(try_cast(regexp_extract(sas_inw(s, w), '{_YMD_RX}', 2) AS DOUBLE), "
   f"try_cast(regexp_extract(sas_inw(s, w), '{_YMD_RX}', 3) AS DOUBLE), try_cast(regexp_extract(sas_inw(s, w), '{_YMD_RX}', 1) AS DOUBLE)) END",
   ("sas_inw", "sas_mdy"))
_MDY_RX = f"^([0-9]{{1,2}}){_SEP}([0-9]{{1,2}}){_SEP}([0-9]{{4}}|[0-9]{{2}})$"
for _name, _a, _b in (("sas_input_mdy", 1, 2), ("sas_input_dmy", 2, 1)):
    fn(_name, "s STRING, w INT", "DOUBLE",
       f"CASE WHEN sas_inw(s, w) RLIKE '^[0-9]{{8}}$' THEN sas_mdy(try_cast(substring(sas_inw(s, w), {1 + (_a - 1) * 2}, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), {1 + (_b - 1) * 2}, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 5, 4) AS DOUBLE)) "
       f"WHEN sas_inw(s, w) RLIKE '^[0-9]{{6}}$' THEN sas_mdy(try_cast(substring(sas_inw(s, w), {1 + (_a - 1) * 2}, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), {1 + (_b - 1) * 2}, 2) AS DOUBLE), try_cast(substring(sas_inw(s, w), 5, 2) AS DOUBLE)) "
       f"WHEN sas_inw(s, w) RLIKE '{_MDY_RX}' THEN sas_mdy(try_cast(regexp_extract(sas_inw(s, w), '{_MDY_RX}', {_a}) AS DOUBLE), "
       f"try_cast(regexp_extract(sas_inw(s, w), '{_MDY_RX}', {_b}) AS DOUBLE), try_cast(regexp_extract(sas_inw(s, w), '{_MDY_RX}', 3) AS DOUBLE)) END",
       ("sas_inw", "sas_mdy"))
_MON = "(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
_D9_RX = f"^([0-9]{{1,2}})[-/ ]?{_MON}[-/ ]?([0-9]{{4}}|[0-9]{{2}})$"
fn("sas_mon_num", "m STRING", "DOUBLE", "CAST((instr('JANFEBMARAPRMAYJUNJULAUGSEPOCTNOVDEC', m) + 2) / 3 AS DOUBLE)")
fn("sas_input_date", "s STRING, w INT", "DOUBLE",
   f"CASE WHEN upper(sas_inw(s, w)) RLIKE '{_D9_RX}' THEN sas_mdy(sas_mon_num(regexp_extract(upper(sas_inw(s, w)), '{_D9_RX}', 2)), "
   f"try_cast(regexp_extract(upper(sas_inw(s, w)), '{_D9_RX}', 1) AS DOUBLE), try_cast(regexp_extract(upper(sas_inw(s, w)), '{_D9_RX}', 3) AS DOUBLE)) END",
   ("sas_inw", "sas_mdy", "sas_mon_num"))
_DT_RX = f"^([0-9]{{1,2}}){_MON}([0-9]{{4}}|[0-9]{{2}})[: ]([0-9]{{1,2}}):([0-9]{{2}})(:([0-9]{{2}}([.][0-9]*)?))?$"
fn("sas_input_datetime", "s STRING, w INT", "DOUBLE",
   f"CASE WHEN upper(sas_inw(s, w)) RLIKE '{_DT_RX}' AND sas_mdy(sas_mon_num(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 2)), "
   f"try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 1) AS DOUBLE), try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 3) AS DOUBLE)) IS NOT NULL "
   f"THEN sas_mdy(sas_mon_num(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 2)), try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 1) AS DOUBLE), "
   f"try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 3) AS DOUBLE)) * 86400 "
   f"+ try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 4) AS DOUBLE) * 3600 + try_cast(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 5) AS DOUBLE) * 60 "
   f"+ coalesce(try_cast(nullif(regexp_extract(upper(sas_inw(s, w)), '{_DT_RX}', 7), '') AS DOUBLE), 0) END",
   ("sas_inw", "sas_mdy", "sas_mon_num"))
_ISO_RX = "^([0-9]{4})-([0-9]{2})-([0-9]{2})[T ]([0-9]{2}):([0-9]{2})(:([0-9]{2}([.][0-9]*)?))?$"
fn("sas_input_isodt", "s STRING, w INT", "DOUBLE",
   f"CASE WHEN sas_inw(s, w) RLIKE '{_ISO_RX}' AND try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 4) AS INT) <= 23 "
   f"AND try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 5) AS INT) <= 59 "
   f"THEN sas_mdy(try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 2) AS DOUBLE), try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 3) AS DOUBLE), "
   f"try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 1) AS DOUBLE)) * 86400 "
   f"+ try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 4) AS DOUBLE) * 3600 + try_cast(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 5) AS DOUBLE) * 60 "
   f"+ coalesce(try_cast(nullif(regexp_extract(sas_inw(s, w), '{_ISO_RX}', 7), '') AS DOUBLE), 0) END",
   ("sas_inw", "sas_mdy"))
_TIME_RX = "^(-?[0-9]{1,2}):([0-9]{1,2})(:([0-9]{1,2}([.][0-9]*)?))? *(AM|PM)?$"
fn("sas_input_time", "s STRING, w INT", "DOUBLE",
   f"CASE WHEN upper(sas_inw(s, w)) RLIKE '{_TIME_RX}' AND try_cast(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 2) AS INT) <= 59 "
   f"AND coalesce(try_cast(nullif(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 4), '') AS DOUBLE), 0) < 60 THEN "
   f"(try_cast(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 1) AS DOUBLE) "
   f" + CASE WHEN regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 6) = 'PM' AND try_cast(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 1) AS INT) < 12 THEN 12 "
   f"        WHEN regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 6) = 'AM' AND try_cast(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 1) AS INT) = 12 THEN -12 ELSE 0 END) * 3600 "
   f"+ try_cast(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 2) AS DOUBLE) * 60 "
   f"+ coalesce(try_cast(nullif(regexp_extract(upper(sas_inw(s, w)), '{_TIME_RX}', 4), '') AS DOUBLE), 0) END",
   ("sas_inw",))
fn("sas_input_anydtdte", "s STRING, w INT", "DOUBLE",
   "coalesce(sas_input_ymd(s, w), sas_input_date(s, w), sas_input_mdy(s, w), sas_datepart(sas_input_isodt(s, w)))",
   ("sas_input_ymd", "sas_input_date", "sas_input_mdy", "sas_datepart", "sas_input_isodt"))
fn("sas_input_char", "s STRING, w INT", "STRING", "ltrim(CASE WHEN w IS NULL THEN coalesce(s, '') ELSE substring(coalesce(s, ''), 1, w) END)")


# ---------------------------------------------------------------------------
# Python-backed functions (vectorised pandas UDFs over the reference implementation)
def _best_udf(spark):
    from pyspark.sql import functions as F

    from . import pyfuncs

    @F.pandas_udf("string")
    def sas_put_best_py(x: pd.Series, w: pd.Series) -> pd.Series:
        return pd.Series([pyfuncs.put_best(None if pd.isna(a) else float(a), int(b)) for a, b in zip(x, w)])

    return sas_put_best_py


def _put_any_udf(spark):
    from pyspark.sql import functions as F

    from . import pyfuncs

    @F.pandas_udf("string")
    def sas_put_py(x: pd.Series, fmt: pd.Series) -> pd.Series:
        return pd.Series([pyfuncs.put(None if pd.isna(a) else float(a), f) for a, f in zip(x, fmt)])

    return sas_put_py


def _input_any_udf(spark):
    from pyspark.sql import functions as F

    from . import pyfuncs

    @F.pandas_udf("double")
    def sas_input_py(s: pd.Series, fmt: pd.Series) -> pd.Series:
        out = []
        for a, f in zip(s, fmt):
            try:
                v = pyfuncs.input_value("" if a is None else a, f)
            except ValueError:
                v = None
            out.append(v)
        return pd.Series(out, dtype="float64")

    return sas_input_py


def _tinv_udf(spark):
    from pyspark.sql import functions as F

    @F.pandas_udf("double")
    def sas_tinv(p: pd.Series, df: pd.Series) -> pd.Series:
        from scipy import stats as st

        return pd.Series([None if (pd.isna(a) or pd.isna(b) or b <= 0) else float(st.t.ppf(a, b)) for a, b in zip(p, df)],
                         dtype="float64")

    return sas_tinv


REGISTRY["sas_tinv"] = PyFn("sas_tinv", _tinv_udf)
REGISTRY["sas_put_best_py"] = PyFn("sas_put_best_py", _best_udf)
REGISTRY["sas_put_py"] = PyFn("sas_put_py", _put_any_udf)
REGISTRY["sas_input_py"] = PyFn("sas_input_py", _input_any_udf)


# ---------------------------------------------------------------------------
# registration
_REGISTERED: dict[int, set[str]] = {}


def _session_key(spark) -> int:
    return id(spark)


def ensure(spark, names, *, today_sas: float | None = None) -> None:
    """Register the named functions (and their dependencies) in this session once."""
    done = _REGISTERED.setdefault(_session_key(spark), set())
    order: list[str] = []

    def visit(n: str) -> None:
        if n in done or n in order:
            return
        f = REGISTRY.get(n)
        if f is None:
            raise KeyError(f"unknown sas2dbx SQL function {n!r}")
        for dep in f.deps:
            visit(dep)
        order.append(n)

    for n in names:
        visit(n)
    for n in order:
        f = REGISTRY[n]
        if isinstance(f, PyFn):
            spark.udf.register(n, f.factory(spark))
        else:
            spark.sql(f"CREATE OR REPLACE TEMPORARY FUNCTION {f.name}({f.params}) RETURNS {f.returns} RETURN {f.body}")
        done.add(n)


def register_clock(spark, today_sas: float | None) -> None:
    """sas_today()/sas_now(): frozen for reproducible validation runs when today_sas is given."""
    if today_sas is None:
        spark.sql("CREATE OR REPLACE TEMPORARY FUNCTION sas_today() RETURNS DOUBLE RETURN CAST(unix_date(current_date()) + 3653 AS DOUBLE)")
        spark.sql("CREATE OR REPLACE TEMPORARY FUNCTION sas_now() RETURNS DOUBLE RETURN "
                  "CAST(unix_date(current_date()) + 3653 AS DOUBLE) * 86400 + hour(current_timestamp()) * 3600 "
                  "+ minute(current_timestamp()) * 60 + second(current_timestamp())")
    else:
        spark.sql(f"CREATE OR REPLACE TEMPORARY FUNCTION sas_today() RETURNS DOUBLE RETURN {float(today_sas)}D")
        spark.sql(f"CREATE OR REPLACE TEMPORARY FUNCTION sas_now() RETURNS DOUBLE RETURN {float(today_sas) * 86400 + 9 * 3600}D")
    _REGISTERED.setdefault(_session_key(spark), set()).update({"sas_today", "sas_now"})


def all_names() -> list[str]:
    return sorted(REGISTRY)
