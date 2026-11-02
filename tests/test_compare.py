"""Parity comparator: every mode applies the same numeric rule per cell, and metadata is part of the output."""
import pytest
from pyspark.sql import types as T

from sas2dbx.runtime.meta import META_KEY
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
def test_relaxed_tolerance_is_the_same_in_every_mode(spark, mode, e, a, expected):
    opts = CompareOptions(keys=["id"] if mode == "keyed" else None, mode=mode, decimals=2)
    assert compare(_df(spark, [(1, e)]), _df(spark, [(1, a)]), opts).passed is expected


@pytest.mark.parametrize("mode", ["keyed", "unordered"])
@pytest.mark.parametrize("e,a,expected", [
    (0.001, 0.004, False),                 # small values: 2 decimals would call these equal
    (1e-6, 1.1e-6, False),
    (0.1 + 0.2, 0.3, True),                # floating-point noise
    (1e12 + 0.0001, 1e12, True),           # last-digit noise at a large magnitude (relative 1e-16)
    (1e6, 1e6 + 0.01, False),              # a real difference at a large magnitude (relative 1e-8)
    (0.0, 1e-14, True),
    (0.0, 1e-9, False),
    (None, None, True),
    (None, 0.0, False),
])
def test_strict_tolerance_by_default(spark, mode, e, a, expected):
    opts = CompareOptions(keys=["id"] if mode == "keyed" else None, mode=mode)
    assert compare(_df(spark, [(1, e)]), _df(spark, [(1, a)]), opts).passed is expected


def test_unordered_pairs_rows_regardless_of_order(spark):
    exp = _df(spark, [(1, 10.0), (2, 20.0), (3, 30.0)])
    act = _df(spark, [(3, 30.004), (1, 9.996), (2, 20.0)])
    r = compare(exp, act, CompareOptions(mode="unordered", decimals=2))
    assert r.passed and r.matched_rows == 3


def test_unordered_multi_column_near_ties(spark):
    """Lexicographic order on raw values would pair (1.000, 5) with (1.000, 3); rounded order pairs them right."""
    schema = "x double, y double"
    exp = spark.createDataFrame([(1.000, 5.0), (1.001, 3.0)], schema)
    act = spark.createDataFrame([(1.001, 5.0), (1.000, 3.0)], schema)
    assert compare(exp, act, CompareOptions(mode="unordered", decimals=2)).passed


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


def test_missing_key_columns_fail_instead_of_falling_back(spark):
    exp = _df(spark, [(1, 1.0)])
    act = spark.createDataFrame([(1.0,)], "x double")
    r = compare(exp, act, CompareOptions(keys=["id"]))
    assert not r.passed and any("key columns missing" in n for n in r.notes)


def test_column_order_is_part_of_parity(spark):
    exp = spark.createDataFrame([(1, 2.0)], "id int, x double")
    act = spark.createDataFrame([(2.0, 1)], "x double, id int")
    r = compare(exp, act, CompareOptions(keys=["id"]))
    assert not r.passed and r.metadata == [{"check": "order", "expected": ["ID", "X"], "actual": ["X", "ID"]}]
    assert compare(exp, act, CompareOptions(keys=["id"], metadata_ignore=["order"])).passed


def _with_meta(spark, meta: dict, schema="name string, x double"):
    fields = [T.StructField(f.name, f.dataType, True, {META_KEY: meta[f.name]} if f.name in meta else {})
              for f in spark.createDataFrame([], schema).schema.fields]
    return spark.createDataFrame([("a", 1.0)], T.StructType(fields))


@pytest.mark.parametrize("check,exp_meta,act_meta", [
    ("length", {"type": "char", "length": 10}, {"type": "char", "length": 20}),
    ("format", {"type": "char", "length": 10, "format": "$10."}, {"type": "char", "length": 10, "format": "$20."}),
    ("label", {"type": "char", "length": 10, "label": "Name"}, {"type": "char", "length": 10, "label": "Full name"}),
])
def test_sas_metadata_is_part_of_parity(spark, check, exp_meta, act_meta):
    exp, act = _with_meta(spark, {"name": exp_meta}), _with_meta(spark, {"name": act_meta})
    r = compare(exp, act, CompareOptions(keys=["name"]))
    assert not r.passed and [m["check"] for m in r.metadata] == [check]
    assert compare(exp, act, CompareOptions(keys=["name"], metadata_ignore=[check])).passed


def test_metadata_equal_after_normalisation(spark):
    exp = _with_meta(spark, {"x": {"type": "num", "length": 8, "format": "date9", "label": "Day "}})
    act = _with_meta(spark, {"x": {"type": "num", "length": 8, "format": "DATE9.", "label": "Day"}})
    assert compare(exp, act, CompareOptions(keys=["name"])).passed


def test_metadata_missing_in_actual_fails(spark):
    exp = _with_meta(spark, {"x": {"type": "num", "length": 8, "format": "DATE9."}})
    act = _with_meta(spark, {})
    r = compare(exp, act, CompareOptions(keys=["name"]))
    assert not r.passed and r.metadata[0]["check"] == "metadata"


def test_reference_without_sas_metadata_checks_order_only(spark):
    exp = _with_meta(spark, {})
    act = _with_meta(spark, {"x": {"type": "num", "length": 8, "format": "DATE9.", "label": "Day"}})
    assert compare(exp, act, CompareOptions(keys=["name"])).passed


def test_unknown_metadata_check_is_rejected(spark):
    with pytest.raises(ValueError):
        compare(_df(spark, [(1, 1.0)]), _df(spark, [(1, 1.0)]), CompareOptions(keys=["id"], metadata_ignore=["colour"]))


def test_metadata_ignore_can_name_one_column(spark):
    meta = lambda label: {"name": {"type": "char", "length": 10, "label": label}, "x": {"type": "num", "length": 8, "label": label}}
    exp, act = _with_meta(spark, meta("A")), _with_meta(spark, meta("B"))
    r = compare(exp, act, CompareOptions(keys=["name"], metadata_ignore=["label:NAME"]))
    assert not r.passed and [(m["check"], m["column"]) for m in r.metadata] == [("label", "X")]
