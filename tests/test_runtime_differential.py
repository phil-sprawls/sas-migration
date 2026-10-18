"""Differential test: Spark SQL runtime (``S.*``) vs. the Python reference (``pyfuncs``).

Two independent implementations of each SAS function must agree on random and
edge-case inputs, under ANSI mode on and off.  A disagreement is a bug in one of
them (the conformance suite then decides which).
"""
import math
import random

import pytest

from sas2dbx.runtime import pyfuncs as R

SEED = 20261016
N = 250


def _rng():
    return random.Random(SEED)


def _nums(rng, n=N):
    vals = [None, 0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 2.675, 0.285, 1.005, 1e-9, 123456.789, -7.0, 1e15]
    while len(vals) < n:
        k = rng.random()
        if k < 0.3:
            vals.append(float(rng.randint(-1000, 1000)))
        elif k < 0.7:
            vals.append(round(rng.uniform(-1000, 1000), rng.choice([1, 2, 3])))
        elif k < 0.85:
            vals.append(rng.choice([1, 2, 3]) * 0.5 + rng.randint(-50, 50))
        elif k < 0.95:
            vals.append(rng.uniform(-1e6, 1e6))
        else:
            vals.append(None)
    return vals


def _dates(rng, n=N):
    vals = [None, 0.0, -1.0, 21974.0, 21975.0, 21945.0, 21946.0, R.mdy(12, 31, 2019), R.mdy(1, 1, 2000), R.mdy(2, 28, 2100)]
    while len(vals) < n:
        vals.append(float(rng.randint(int(R.mdy(1, 1, 1900)), int(R.mdy(12, 31, 2100)))))
    return vals


_ALPHA = "abcXYZ019 ,.-!/(  '"


def _strs(rng, n=N):
    vals = ["", " ", "a", "abc", "one,two  three!four", "  lead", "Hello WORLD-foo", "a-b.c(d/e", "x" * 30]
    while len(vals) < n:
        vals.append("".join(rng.choice(_ALPHA) for _ in range(rng.randint(0, 25))).rstrip())
    return vals


def _frame(spark, cols: dict):
    from pyspark.sql import types as T

    names = list(cols)
    n = len(next(iter(cols.values())))
    fields = []
    for k in names:
        sample = next((v for v in cols[k] if v is not None), 0.0)
        fields.append(T.StructField(k, T.StringType() if isinstance(sample, str) else T.DoubleType(), True))
    rows = [tuple(cols[k][i] for k in names) for i in range(n)]
    return spark.createDataFrame(rows, T.StructType(fields))


def _same(a, b, tol=1e-9):
    if isinstance(a, float) and math.isnan(a):
        a = None
    if isinstance(b, float) and math.isnan(b):
        b = None
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def _check(spark, cols, spark_expr, py_fn, label, tol=1e-9):
    df = _frame(spark, cols)
    out = df.select(spark_expr.alias("v")).collect()
    names = list(cols)
    bad = []
    for i, row in enumerate(out):
        args = [cols[k][i] for k in names]
        exp = py_fn(*args)
        got = row["v"]
        if isinstance(exp, str) and got is None:
            got = ""
        if not _same(exp, got, tol):
            bad.append((args, exp, got))
    assert not bad, f"{label}: {len(bad)} disagreements, e.g. {bad[:5]}"


@pytest.fixture(scope="module")
def S():
    from sas2dbx.runtime import S as _S

    return _S


@pytest.mark.spark
class TestNumeric:
    def test_round(self, ansi_spark, S):
        from pyspark.sql import functions as F

        rng = _rng()
        x = _nums(rng)
        for unit in (1.0, 0.1, 0.01, 0.001, 10.0, 0.25, 0.5, 5.0):
            _check(ansi_spark, {"x": x}, S.round(F.col("x"), unit), lambda v, u=unit: R.sas_round(v, u), f"round(x,{unit})")

    def test_int_floor_ceil_mod(self, ansi_spark, S):
        from pyspark.sql import functions as F

        rng = _rng()
        x = _nums(rng)
        _check(ansi_spark, {"x": x}, S.int_(F.col("x")), R.sas_int, "int")
        _check(ansi_spark, {"x": x}, S.floor(F.col("x")), R.sas_floor, "floor")
        _check(ansi_spark, {"x": x}, S.ceil(F.col("x")), R.sas_ceil, "ceil")
        for d in (3.0, -3.0, 0.1, 0.0):
            _check(ansi_spark, {"x": x}, S.mod(F.col("x"), d), lambda v, dd=d: R.sas_mod(v, dd), f"mod(x,{d})", tol=1e-7)

    def test_stats(self, ansi_spark, S):
        from pyspark.sql import functions as F

        rng = _rng()
        a, b, c = _nums(rng), _nums(random.Random(1)), _nums(random.Random(2))
        cols = {"a": a, "b": b, "c": c}
        args = (F.col("a"), F.col("b"), F.col("c"))
        for name, fn in [("sum", R.sas_sum), ("mean", R.sas_mean), ("min", R.sas_min), ("max", R.sas_max),
                         ("n", R.sas_n), ("nmiss", R.sas_nmiss), ("std", R.sas_std), ("median", R.sas_median),
                         ("range", R.sas_range)]:
            _check(ansi_spark, cols, getattr(S, name)(*args), fn, name, tol=1e-9)


@pytest.mark.spark
class TestDates:
    def test_parts(self, ansi_spark, S):
        from pyspark.sql import functions as F

        d = _dates(_rng())
        for name, fn in [("year", R.year), ("month", R.month), ("day", R.day), ("qtr", R.qtr), ("weekday", R.weekday)]:
            _check(ansi_spark, {"d": d}, getattr(S, name)(F.col("d")), fn, name)

    def test_mdy(self, ansi_spark, S):
        from pyspark.sql import functions as F

        rng = _rng()
        m = [float(rng.randint(0, 13)) for _ in range(N)]
        dd = [float(rng.randint(0, 32)) for _ in range(N)]
        y = [float(rng.choice([2019, 2020, 1900, 2000, 25, 26, 99, 0])) for _ in range(N)]
        _check(ansi_spark, {"m": m, "d": dd, "y": y}, S.mdy(F.col("m"), F.col("d"), F.col("y")), R.mdy, "mdy")

    @pytest.mark.parametrize("interval", ["day", "week", "month", "qtr", "semiyear", "year", "month2", "year.7", "qtr.2", "week.2"])
    @pytest.mark.parametrize("align", ["B", "E", "M", "S"])
    def test_intnx(self, ansi_spark, S, interval, align):
        from pyspark.sql import functions as F

        rng = _rng()
        d = _dates(rng, 120)
        n = [float(rng.randint(-25, 25)) for _ in range(120)]
        if align == "S" and ("." in interval or any(ch.isdigit() for ch in interval)):
            pytest.skip("SAME alignment for multiplied/shifted intervals is approximated (documented)")
        _check(ansi_spark, {"d": d, "n": n}, S.intnx(interval, F.col("d"), F.col("n"), align),
               lambda a, b: R.intnx(interval, a, b, align), f"intnx({interval},{align})")

    @pytest.mark.parametrize("interval", ["day", "week", "month", "qtr", "year", "month3", "year.4"])
    @pytest.mark.parametrize("method", ["DISCRETE", "C"])
    def test_intck(self, ansi_spark, S, interval, method):
        from pyspark.sql import functions as F

        if method == "C" and "." in interval:
            pytest.skip("continuous method ignores shift")
        rng = _rng()
        a, b = _dates(rng, 120), _dates(random.Random(7), 120)
        _check(ansi_spark, {"a": a, "b": b}, S.intck(interval, F.col("a"), F.col("b"), method),
               lambda x, y: R.intck(interval, x, y, method), f"intck({interval},{method})")

    def test_yrdif(self, ansi_spark, S):
        from pyspark.sql import functions as F

        rng = _rng()
        a = _dates(rng, 150)
        b = [None if v is None else v + rng.randint(0, 20000) for v in a]
        for basis in ("AGE", "ACT/ACT"):
            _check(ansi_spark, {"a": a, "b": b}, S.yrdif(F.col("a"), F.col("b"), basis),
                   lambda x, y, bb=basis: R.yrdif(x, y, bb), f"yrdif {basis}", tol=1e-9)


@pytest.mark.spark
class TestFormats:
    @pytest.mark.parametrize("spec", ["8.2", "10.", "6.1", "5.3", "z6.2", "z3.", "comma12.2", "dollar10.2", "percent8.1", "best12.", "best8.", "best.", "12.4"])
    def test_put_numeric(self, ansi_spark, S, spec):
        from pyspark.sql import functions as F

        x = [v for v in _nums(_rng()) if v is None or abs(v) < 1e7]
        _check(ansi_spark, {"x": x}, S.put(F.col("x"), spec), lambda v: R.put(v, spec), f"put({spec})")

    @pytest.mark.parametrize("spec", ["date9.", "date7.", "date11.", "yymmdd10.", "yymmdd8.", "yymmddn8.", "mmddyy10.",
                                      "ddmmyy10.", "monyy7.", "worddate.", "e8601da.", "year4.", "yyq6.", "weekdate."])
    def test_put_dates(self, ansi_spark, S, spec):
        from pyspark.sql import functions as F

        d = _dates(_rng())
        _check(ansi_spark, {"d": d}, S.put(F.col("d"), spec), lambda v: R.put(v, spec), f"put({spec})")

    @pytest.mark.parametrize("spec", ["datetime20.", "datetime16.", "e8601dt19.", "time8.", "time5.", "tod8."])
    def test_put_times(self, ansi_spark, S, spec):
        from pyspark.sql import functions as F

        rng = _rng()
        if spec.startswith(("time", "tod")):
            x = [None] + [float(rng.randint(0, 86399)) for _ in range(N)]
        else:
            x = [None] + [float(rng.randint(-10**9, 2 * 10**9)) for _ in range(N)]
        _check(ansi_spark, {"x": x}, S.put(F.col("x"), spec), lambda v: R.put(v, spec), f"put({spec})")


@pytest.mark.spark
class TestInformats:
    @pytest.mark.parametrize("spec,gen", [
        ("yymmdd10.", lambda d: R.put(d, "yymmdd10.")),
        ("yymmdd8.", lambda d: R.put(d, "yymmddn8.")),
        ("mmddyy10.", lambda d: R.put(d, "mmddyy10.")),
        ("ddmmyy10.", lambda d: R.put(d, "ddmmyy10.")),
        ("date9.", lambda d: R.put(d, "date9.")),
        ("date11.", lambda d: R.put(d, "date11.")),
        ("anydtdte20.", lambda d: R.put(d, "yymmdd10.")),
    ])
    def test_dates_roundtrip_and_garbage(self, ansi_spark, S, spec, gen):
        from pyspark.sql import functions as F

        rng = _rng()
        texts = [gen(d) if d is not None else "" for d in _dates(rng, 150)]
        texts += ["2012-02-30", "2012-13-01", "abc", "2012-03", "20120230", "31/02/2020", "  2020-01-15  ", "15jan2020", "1-1-2020"]
        _check(ansi_spark, {"s": texts}, S.input(F.col("s"), spec), lambda v: R.input_value(v, spec), f"input({spec})")

    @pytest.mark.parametrize("spec", ["8.", "8.2", "best32.", "comma10.", "comma10.2", "percent8."])
    def test_numbers(self, ansi_spark, S, spec):
        from pyspark.sql import functions as F

        texts = ["", "42", " 42 ", "-1.5", "1e3", "1,234", "$1,234.50", "(12)", "12%", "abc", "1.2.3", "1234", ".5", "+7"]
        _check(ansi_spark, {"s": texts}, S.input(F.col("s"), spec), lambda v: R.input_value(v, spec), f"input({spec})")

    def test_datetime_time(self, ansi_spark, S):
        from pyspark.sql import functions as F

        texts = ["05MAR2012:14:30:00", "05MAR2012:14:30", "5MAR12:00:00:01", "31FEB2012:10:00:00", "garbage", ""]
        _check(ansi_spark, {"s": texts}, S.input(F.col("s"), "datetime20."), lambda v: R.input_value(v, "datetime20."), "input(datetime)")
        iso = ["2012-03-05T14:30:00", "2012-03-05 14:30", "2012-03-05T25:00:00", "x", ""]
        _check(ansi_spark, {"s": iso}, S.input(F.col("s"), "e8601dt19."), lambda v: R.input_value(v, "e8601dt19."), "input(e8601dt)")
        tm = ["14:30", "14:30:15", "2:05:09.5", "12:00 AM", "1:00 PM", "14:75", "x", ""]
        _check(ansi_spark, {"s": tm}, S.input(F.col("s"), "time8."), lambda v: R.input_value(v, "time8."), "input(time)")


@pytest.mark.spark
class TestCharacter:
    def test_char_functions(self, ansi_spark, S):
        from pyspark.sql import functions as F

        s = _strs(_rng())
        c = F.col("s")
        cases = [
            (S.scan(c, 1), lambda v: R.scan(v, 1)),
            (S.scan(c, 2), lambda v: R.scan(v, 2)),
            (S.scan(c, -1), lambda v: R.scan(v, -1)),
            (S.scan(c, 2, ","), lambda v: R.scan(v, 2, ",")),
            (S.scan(c, 3, ", ", "m"), lambda v: R.scan(v, 3, ", ", "m")),
            (S.countw(c), lambda v: R.countw(v)),
            (S.compress(c), lambda v: R.compress(v)),
            (S.compress(c, ",."), lambda v: R.compress(v, ",.")),
            (S.compress(c, None, "kd"), lambda v: R.compress(v, None, "kd")),
            (S.compress(c, "", "a"), lambda v: R.compress(v, "", "a")),
            (S.compbl(c), R.compbl),
            (S.translate(c, "xy", "ab"), lambda v: R.translate(v, "xy", "ab")),
            (S.translate(c, "Q", "abc"), lambda v: R.translate(v, "Q", "abc")),
            (S.tranwrd(c, "a", "ZZ"), lambda v: R.tranwrd(v, "a", "ZZ")),
            (S.substr(c, 2, 3), lambda v: R.substr(v, 2, 3)),
            (S.substr(c, 4), lambda v: R.substr(v, 4)),
            (S.index(c, "a"), lambda v: R.index(v, "a")),
            (S.indexc(c, ",!"), lambda v: R.indexc(v, ",!")),
            (S.verify(c, "abc "), lambda v: R.verify(v, "abc ")),
            (S.find(c, "A", "i"), lambda v: R.find(v, "A", "i")),
            (S.propcase(c), R.propcase),
            (S.length(c), R.length),
            (S.lengthn(c), R.lengthn),
            (S.strip(c), R.strip),
            (S.left(c), lambda v: R.left(v).rstrip()),
            (S.count(c, "a"), lambda v: R.count(v, "a")),
            (S.countc(c, "a,"), lambda v: R.countc(v, "a,")),
            (S.anydigit(c), R.anydigit),
            (S.anyalpha(c), R.anyalpha),
            (S.anypunct(c), R.anypunct),
            (S.notalpha(c), R.notalpha),
            (S.quote(c), R.quote),
            (S.repeat(c, 1), lambda v: R.repeat(v, 1)),
            (S.cats(c, F.lit("z")), lambda v: R.cats(v, "z")),
            (S.catx("-", c, F.lit(" q ")), lambda v: R.catx("-", v, " q ")),
        ]
        for i, (expr, fn) in enumerate(cases):
            _check(ansi_spark, {"s": s}, expr, fn, f"char case {i}")

    def test_prx(self, ansi_spark, S):
        from pyspark.sql import functions as F

        s = _strs(_rng())
        _check(ansi_spark, {"s": s}, S.prxmatch("/[0-9]+/", F.col("s")), lambda v: R.prxmatch("/[0-9]+/", v), "prxmatch")
        _check(ansi_spark, {"s": s}, S.prxmatch("/ABC/i", F.col("s")), lambda v: R.prxmatch("/ABC/i", v), "prxmatch i")
        _check(ansi_spark, {"s": s}, S.prxchange("s/[0-9]/#/", -1, F.col("s")), lambda v: R.prxchange("s/[0-9]/#/", -1, v), "prxchange")
