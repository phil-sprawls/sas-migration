"""``S``: PySpark Column builders with SAS semantics (thin wrappers over the SQL runtime).

Generated code always passes Columns for data (``F.col("AGE")``) and Python
scalars for literals, so a Python ``str`` argument is a string *literal*.
"""
from __future__ import annotations

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

from . import dispatch as D
from . import pyfuncs, sqlfuncs



def __getattr__(name: str):
    # Columns need an active SparkSession, so the constants are built on first use.
    if name == "MISSING":
        return F.lit(None).cast("double")
    if name == "BLANK":
        return F.lit("")
    raise AttributeError(name)


def _missing() -> Column:
    return F.lit(None).cast("double")


def _blank() -> Column:
    return F.lit("")


def _spark() -> SparkSession:
    s = SparkSession.getActiveSession()
    if s is None:
        s = SparkSession.active()
    return s


def ensure(*names: str) -> None:
    sqlfuncs.ensure(_spark(), names)


def _col(x) -> Column:
    if isinstance(x, Column):
        return x
    if x is None:
        return _missing()
    if isinstance(x, bool):
        return F.lit(x)
    if isinstance(x, (int, float)):
        return F.lit(float(x))
    return F.lit(x)


def _call(name: str, *args) -> Column:
    ensure(name)
    return F.call_function(name, *[a if isinstance(a, Column) else F.lit(a) for a in args])


def call(spec: D.CallSpec, *values) -> Column:
    """Render a dispatch CallSpec with value Columns."""
    ensure(*[n for n in spec.functions() if n in sqlfuncs.REGISTRY])

    def render(node):
        if isinstance(node, D.Arg):
            return _col(values[node.i])
        if isinstance(node, D.Lit):
            v = node.v
            return F.lit(v)
        if isinstance(node, D.CallSpec):
            return F.call_function(node.name, *[render(a) for a in node.args])
        raise TypeError(node)

    return render(spec)


# ---------------------------------------------------------------------------
# literals
def d(text: str) -> float:
    """'01JAN2020'd -> SAS date number."""
    v = pyfuncs.input_value(text, "DATE11.") if "-" in text else pyfuncs.input_value(text, "DATE9.")
    if v is None:
        raise ValueError(f"invalid SAS date literal {text!r}")
    return v


def dt(text: str) -> float:
    v = pyfuncs.input_value(text, "DATETIME40.")
    if v is None:
        raise ValueError(f"invalid SAS datetime literal {text!r}")
    return v


def t(text: str) -> float:
    v = pyfuncs.input_value(text, "TIME20.")
    if v is None:
        raise ValueError(f"invalid SAS time literal {text!r}")
    return v


def sql_esc(value: str) -> str:
    """A run-time value placed inside a Spark SQL '...' literal: data, never SQL code (backslash escapes)."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------------------
# booleans & comparisons
def truth(x) -> Column:
    """SAS truth value of a numeric: not missing and not zero."""
    return _call("sas_truth", _col(x))


def b2n(cond: Column) -> Column:
    """Boolean -> 1/0 (SAS comparison results are numbers)."""
    return F.when(cond, F.lit(1.0)).otherwise(F.lit(0.0))


def lt(a, b) -> Column:
    return _call("sas_lt", _col(a), _col(b))


def le(a, b) -> Column:
    return _call("sas_le", _col(a), _col(b))


def gt(a, b) -> Column:
    return _call("sas_gt", _col(a), _col(b))


def ge(a, b) -> Column:
    return _call("sas_ge", _col(a), _col(b))


def eq(a, b) -> Column:
    """Numeric equality with SAS missing semantics (. = . is true)."""
    return _col(a).eqNullSafe(_col(b))


def ne(a, b) -> Column:
    return ~_col(a).eqNullSafe(_col(b))


def isin(x, *values) -> Column:
    """Numeric IN list; missing literals (None) match missing values."""
    vals = [v for v in values if v is not None]
    cond = _col(x).isin(*vals) if vals else F.lit(False)
    cond = F.coalesce(cond, F.lit(False))
    if any(v is None for v in values):
        cond = cond | _col(x).isNull()
    return cond


def prefix_eq(a, b) -> Column:
    """=: comparison: compare on the length of the shorter operand."""
    a, b = _col(a), _col(b)
    n_ = F.least(F.length(a), F.length(b))
    return a.substr(F.lit(1), n_) == b.substr(F.lit(1), n_)


# ---------------------------------------------------------------------------
# arithmetic
def div(a, b) -> Column:
    """Division; x/0 is missing (never an ANSI error)."""
    return F.try_divide(_col(a), _col(b))


def pow(a, b) -> Column:  # noqa: A001 - SAS name
    return _call("sas_pow", _col(a), _col(b))


def minop(a, b) -> Column:
    """a >< b (MIN operator: missing if either is missing)."""
    return _call("sas_minop", _col(a), _col(b))


def maxop(a, b) -> Column:
    """a <> b (MAX operator)."""
    return _call("sas_maxop", _col(a), _col(b))


def neg(a) -> Column:
    return -_col(a)


def _f1(name):
    def f(x) -> Column:
        return _call(name, _col(x))

    f.__name__ = name.replace("sas_", "")
    return f


sqrt = _f1("sas_sqrt")
log = _f1("sas_log")
log10 = _f1("sas_log10")
log2 = _f1("sas_log2")
exp = _f1("sas_exp")
int_ = _f1("sas_int")
floor = _f1("sas_floor")
ceil = _f1("sas_ceil")


def abs(x) -> Column:  # noqa: A001
    return F.abs(_col(x))


def sign(x) -> Column:
    return F.signum(_col(x))


def round(x, unit=1.0) -> Column:  # noqa: A001
    return _call("sas_round", _col(x), _col(unit))


def mod(a, b) -> Column:
    return _call("sas_mod", _col(a), _col(b))


def _arr(xs) -> Column:
    return F.array(*[_col(x).cast("double") for x in xs])


def sum(*xs) -> Column:  # noqa: A001
    return _call("sas_sum_arr", _arr(xs))


def mean(*xs) -> Column:
    return _call("sas_mean_arr", _arr(xs))


def min(*xs) -> Column:  # noqa: A001
    return _call("sas_min_arr", _arr(xs))


def max(*xs) -> Column:  # noqa: A001
    return _call("sas_max_arr", _arr(xs))


def n(*xs) -> Column:
    return _call("sas_n_arr", _arr(xs))


def nmiss(*xs) -> Column:
    return _call("sas_nmiss_arr", _arr(xs))


def std(*xs) -> Column:
    return _call("sas_std_arr", _arr(xs))


def var(*xs) -> Column:
    return _call("sas_var_arr", _arr(xs))


def range(*xs) -> Column:  # noqa: A001
    return _call("sas_range_arr", _arr(xs))


def median(*xs) -> Column:
    return _call("sas_median_arr", _arr(xs))


def coalesce(*xs) -> Column:
    return F.coalesce(*[_col(x) for x in xs])


def coalescec(*xs) -> Column:
    return _call("sas_coalescec_arr", F.array(*[_col(x) for x in xs]))


def missing_num(x) -> Column:
    return _col(x).isNull()


def missing_char(s) -> Column:
    return F.trim(F.coalesce(_col(s), _blank())) == ""


def cmiss(*items) -> Column:
    """items: (column, kind) pairs."""
    flags = [F.when(missing_char(c) if k == "char" else missing_num(c), 1.0).otherwise(0.0) for c, k in items]
    out = flags[0]
    for f in flags[1:]:
        out = out + f
    return out


# ---------------------------------------------------------------------------
# dates & times
def mdy(m, d_, y) -> Column:
    return _call("sas_mdy", _col(m), _col(d_), _col(y))


year = _f1("sas_year")
month = _f1("sas_month")
day = _f1("sas_day")
qtr = _f1("sas_qtr")
weekday = _f1("sas_weekday")
datepart = _f1("sas_datepart")
timepart = _f1("sas_timepart")
hour = _f1("sas_hour")
minute = _f1("sas_minute")
second = _f1("sas_second")


def today() -> Column:
    return F.call_function("sas_today")


def now() -> Column:
    return F.call_function("sas_now")


def dhms(d_, h, mi, s) -> Column:
    return _call("sas_dhms", _col(d_), _col(h), _col(mi), _col(s))


def hms(h, mi, s) -> Column:
    return _call("sas_hms", _col(h), _col(mi), _col(s))


def intnx(interval: str, start, n_, alignment: str = "B") -> Column:
    return call(D.intnx_call(interval, alignment), start, n_)


def intck(interval: str, a, b, method: str = "DISCRETE") -> Column:
    return call(D.intck_call(interval, method), a, b)


def yrdif(a, b, basis: str = "ACT/ACT") -> Column:
    return call(D.yrdif_call(basis), a, b)


def to_date(x) -> Column:
    """SAS date number -> Spark DATE (for output boundaries)."""
    return _call("sas_date", _col(x))


def from_date(col) -> Column:
    return _call("sas_from_date", _col(col))


# ---------------------------------------------------------------------------
# character functions
def pad(s, length: int) -> Column:
    """The value as SAS holds it: blank-padded to the variable's declared length."""
    return F.rpad(_col(s), int(length), " ")


def store(s, length: int | None = None) -> Column:
    """Value as stored in a character variable: truncated to its length, trailing blanks dropped."""
    c = _col(s)
    if length is not None:
        c = F.substring(c, 1, int(length))
    return F.rtrim(c)


def concat(*parts) -> Column:
    return F.concat(*[F.coalesce(_col(p), _blank()) for p in parts])


def substr(s, p, length=None) -> Column:
    if length is None:
        return _call("sas_substr2", _col(s), _col(p))
    return _call("sas_substr3", _col(s), _col(p), _col(length))


def substrn(s, p, length) -> Column:
    return _call("sas_substrn3", _col(s), _col(p), _col(length))


def index(s, t_) -> Column:
    return _call("sas_index", _col(s), _col(t_))


def indexc(s, *chars) -> Column:
    return _call("sas_indexc", _col(s), F.concat(*[_col(c) for c in chars]))


def verify(s, *chars) -> Column:
    return _call("sas_verify", _col(s), F.concat(*[_col(c) for c in chars]))


def find(s, t_, modifiers: str = "", start=1) -> Column:
    return _call("sas_find", _col(s), _col(t_), F.lit(modifiers or ""), _col(start))


def scan(s, n_, delims: str | None = None, modifiers: str = "") -> Column:
    rx, m = D.scan_regex(delims, modifiers)
    return _call("sas_scan_m" if m else "sas_scan", _col(s), _col(n_), F.lit(rx))


def countw(s, delims: str | None = None, modifiers: str = "") -> Column:
    rx, _ = D.scan_regex(delims, modifiers)
    return _call("sas_countw", _col(s), F.lit(rx))


def compress(s, chars: str | None = None, modifiers: str = "") -> Column:
    return _call("sas_compress_rx", _col(s), F.lit(D.compress_regex(chars, modifiers)))


def compbl(s) -> Column:
    return _call("sas_compbl", _col(s))


def tranwrd(s, target, replacement) -> Column:
    return _call("sas_tranwrd", _col(s), _col(target), _col(replacement))


def transtrn(s, target, replacement) -> Column:
    return _call("sas_transtrn", _col(s), _col(target), _col(replacement))


def translate(s, *to_from) -> Column:
    out = _col(s)
    for k in __import__("builtins").range(0, len(to_from) - 1, 2):
        out = _call("sas_translate", out, _col(to_from[k]), _col(to_from[k + 1]))
    return out


def left(s) -> Column:
    return F.ltrim(_col(s))


def right(s, length: int) -> Column:
    return _call("sas_right", _col(s), F.lit(int(length)))


def trim(s) -> Column:
    return _call("sas_trim", _col(s))


def trimn(s) -> Column:
    return F.rtrim(_col(s))


def strip(s) -> Column:
    return F.trim(_col(s))


def upcase(s) -> Column:
    return F.upper(_col(s))


def lowcase(s) -> Column:
    return F.lower(_col(s))


def propcase(s, delims: str | None = None) -> Column:
    return _call("sas_propcase", _col(s), F.lit(pyfuncs.PROPCASE_DEFAULT_DELIMS if delims is None else delims))


def length(s) -> Column:
    return _call("sas_length", _col(s))


def lengthn(s) -> Column:
    return _call("sas_lengthn", _col(s))


def lengthc(s, declared: int) -> Column:
    return F.lit(float(declared))


def repeat(s, n_) -> Column:
    return _call("sas_repeat", _col(s), _col(n_))


def reverse(s) -> Column:
    return F.reverse(_col(s))


def num_for_cat(x, *, catx: bool = False) -> Column:
    """Numeric argument of CAT*/CATX: BEST12. then stripped ('.' for missing, '' in CATX)."""
    best = F.trim(put_best(x, 12))
    if catx:
        return F.when(_col(x).isNull(), _blank()).otherwise(best)
    return best


def cat(*xs) -> Column:
    return _call("sas_cat_arr", F.array(*[_col(x) for x in xs]))


def cats(*xs) -> Column:
    return _call("sas_cats_arr", F.array(*[_col(x) for x in xs]))


def catt(*xs) -> Column:
    return _call("sas_catt_arr", F.array(*[_col(x) for x in xs]))


def catx(sep, *xs) -> Column:
    return _call("sas_catx_arr", _col(sep), F.array(*[_col(x) for x in xs]))


def quote(s) -> Column:
    return _call("sas_quote", _col(s))


def dequote(s) -> Column:
    return _call("sas_dequote", _col(s))


def count(s, t_) -> Column:
    return _call("sas_count", _col(s), _col(t_))


def countc(s, chars) -> Column:
    return _call("sas_countc", _col(s), _col(chars))


def _any(name):
    def f(s) -> Column:
        return _call(f"sas_{name}", _col(s))

    f.__name__ = name
    return f


anydigit = _any("anydigit")
anyalpha = _any("anyalpha")
anyalnum = _any("anyalnum")
anyspace = _any("anyspace")
anyupper = _any("anyupper")
anylower = _any("anylower")
anypunct = _any("anypunct")
notdigit = _any("notdigit")
notalpha = _any("notalpha")
notspace = _any("notspace")
notalnum = _any("notalnum")


def prxmatch(pattern: str, s) -> Column:
    rx, _ = D.perl_to_java(pattern)
    return _call("sas_prxmatch", F.lit(rx), _col(s))


def prxchange(pattern: str, times, s) -> Column:
    rx, rep = D.perl_to_java(pattern)
    return _call("sas_prxchange", F.lit(rx), _col(s), F.lit(rep or ""))


# ---------------------------------------------------------------------------
# formats / informats / conversions
def put(x, spec: str, kind: str = "num") -> Column:
    return call(D.put_call(spec, kind), x)


def put_best(x, w: int = 12) -> Column:
    return _call("sas_put_best", _col(x), F.lit(int(w)))


def input(s, spec: str) -> Column:  # noqa: A001 - SAS name
    return call(D.input_call(spec), s)


def num_to_char(x) -> Column:
    """Automatic numeric->character conversion (SAS uses BEST12., right-aligned)."""
    return put_best(x, 12)


def char_to_num(s) -> Column:
    """Automatic character->numeric conversion (BEST32. informat; invalid -> missing)."""
    return _call("sas_input_num", _col(s), F.lit(32), F.lit(0))
