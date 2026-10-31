"""Parity comparator: every mode must apply the same rule, |a - b| < 0.5 * 10**-decimals per cell."""
import pytest

from sas2dbx.validate.compare import CompareOptions, compare

pytestmark = pytest.mark.spark


def _df(spark, rows, schema="id int, x double"):
    return spark.createDataFrame(rows, schema)


@pytest.mark.parametrize("mode", ["keyed", "unordered"])
@pytest.mark.parametrize("e,a,expected", [
    (0.9951, 1.0049, False),   # diff 0.0098: same 2-dp rounding bucket, but outside tolerance
    (1.0049, 1.0051, True),    # diff 0.0002: straddles a rounding boundary, but inside tolerance
    (1.0, 1.0049, True),
    (1.0, 1.0051, False),
    (None, None, True),        # missing matches missing
    (None, 0.0, False),        # missing never matches a value
])
def test_tolerance_is_the_same_in_every_mode(spark, mode, e, a, expected):
    opts = CompareOptions(keys=["id"] if mode == "keyed" else None, mode=mode)
    assert compare(_df(spark, [(1, e)]), _df(spark, [(1, a)]), opts).passed is expected


def test_unordered_pairs_rows_regardless_of_order(spark):
    exp = _df(spark, [(1, 10.0), (2, 20.0), (3, 30.0)])
    act = _df(spark, [(3, 30.004), (1, 9.996), (2, 20.0)])
    r = compare(exp, act, CompareOptions(mode="unordered"))
    assert r.passed and r.matched_rows == 3


def test_unordered_multi_column_near_ties(spark):
    """Lexicographic order on raw values would pair (1.000, 5) with (1.000, 3); rounded order pairs them right."""
    schema = "x double, y double"
    exp = spark.createDataFrame([(1.000, 5.0), (1.001, 3.0)], schema)
    act = spark.createDataFrame([(1.001, 5.0), (1.000, 3.0)], schema)
    assert compare(exp, act, CompareOptions(mode="unordered")).passed


def test_unordered_duplicates_fail(spark):
    # numeric-only rows form one pairing group, so a different duplicate count surfaces as cell mismatches
    exp = _df(spark, [(1, 1.0), (1, 1.0), (2, 2.0)])
    act = _df(spark, [(1, 1.0), (2, 2.0), (2, 2.0)])
    r = compare(exp, act, CompareOptions(mode="unordered"))
    assert not r.passed and r.mismatched_columns
    # with a character column, the unmatched duplicates are reported as rows without a partner
    schema = "k string, x double"
    exp = spark.createDataFrame([("a", 1.0), ("a", 1.0), ("b", 2.0)], schema)
    act = spark.createDataFrame([("a", 1.0), ("b", 2.0), ("b", 2.0)], schema)
    r = compare(exp, act, CompareOptions(mode="unordered"))
    assert not r.passed and r.only_in_expected == 1 and r.only_in_actual == 1 and r.matched_rows == 2


def test_unordered_character_values_are_exact(spark):
    schema = "name string, x double"
    exp = spark.createDataFrame([("A", 1.0), ("B", 2.0)], schema)
    act = spark.createDataFrame([("A", 1.0), ("b", 2.0)], schema)
    r = compare(exp, act, CompareOptions(mode="unordered"))
    assert not r.passed and r.only_in_expected == 1 and r.only_in_actual == 1
    # trailing blanks are not significant, NULL is blank
    act2 = spark.createDataFrame([("A  ", 1.0), ("B", 2.0)], schema)
    assert compare(exp, act2, CompareOptions(mode="unordered")).passed


def test_unordered_reports_the_mismatched_column(spark):
    schema = "g string, x double, y double"
    exp = spark.createDataFrame([("A", 1.0, 5.0)], schema)
    act = spark.createDataFrame([("A", 1.0, 5.2)], schema)
    r = compare(exp, act, CompareOptions(mode="unordered"))
    assert not r.passed
    assert r.mismatched_columns == ["Y"]
    assert r.columns[[c.name for c in r.columns].index("Y")].max_abs_diff == pytest.approx(0.2)


def test_unordered_character_only(spark):
    schema = "a string, b string"
    exp = spark.createDataFrame([("x", "1"), ("y", "2")], schema)
    assert compare(exp, spark.createDataFrame([("y", "2"), ("x", "1")], schema), CompareOptions(mode="unordered")).passed
    assert not compare(exp, spark.createDataFrame([("y", "2"), ("x", "9")], schema), CompareOptions(mode="unordered")).passed


def test_unordered_respects_decimals(spark):
    exp, act = _df(spark, [(1, 1.0)]), _df(spark, [(1, 1.0004)])
    assert compare(exp, act, CompareOptions(mode="unordered", decimals=3)).passed
    assert not compare(exp, act, CompareOptions(mode="unordered", decimals=4)).passed
