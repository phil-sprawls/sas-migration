"""Review tables: the differences table holds exactly the cells the comparison counted, and the sink writes all three."""
import pytest

from sas2dbx.validate.compare import CompareOptions, compare, diff_rows
from sas2dbx.validate.review import ReviewSink

pytestmark = pytest.mark.spark


def _pair(spark):
    e = spark.createDataFrame([(1, 1.0, "A"), (2, 2.0, "B"), (3, 3.0, "C"), (4, 4.0, "D")],
                              "id int, x double, arm string")
    a = spark.createDataFrame([(1, 1.0, "A"), (2, 2.5, "B"), (3, 3.0, "Z"), (5, 5.0, "E")],
                              "id int, x double, arm string")
    return e, a


def test_diff_rows_match_the_counts(spark):
    e, a = _pair(spark)
    r = compare(e, a, CompareOptions(keys=["id"]))
    rows = {(x["ID"], x["status"], x["column_name"]): x for x in diff_rows(r).collect()}
    assert set(rows) == {(2, "value differs", "X"), (3, "value differs", "ARM"),
                         (4, "only in SAS", None), (5, "only in Databricks", None)}
    assert rows[(2, "value differs", "X")]["difference"] == pytest.approx(0.5)
    assert rows[(3, "value differs", "ARM")]["sas_value"] == "C" and rows[(3, "value differs", "ARM")]["dbx_value"] == "Z"
    counted = sum(c.mismatches for c in r.columns) + r.only_in_expected + r.only_in_actual
    assert counted == len(rows)


def test_diff_rows_empty_on_a_pass_and_absent_without_pairing(spark):
    e, _ = _pair(spark)
    assert diff_rows(compare(e, e, CompareOptions(keys=["id"]))).count() == 0
    assert diff_rows(compare(e, e.drop("id"), CompareOptions(keys=["id"]))) is None   # key missing: nothing paired


def test_unordered_diff_uses_readable_row_ids(spark):
    e, a = _pair(spark)
    cols = diff_rows(compare(e.drop("id"), a.drop("id"), CompareOptions(mode="unordered"))).columns
    assert "PAIR_RANK" in cols and not any(c.startswith("_") for c in cols)


def test_local_sink_writes_three_tables(spark, tmp_path):
    e, a = _pair(spark)
    r = compare(e, a, CompareOptions(keys=["id"]))
    out = ReviewSink(spark, "demo", local_dir=str(tmp_path)).write_output("My Case", "work.out", e, a, r)
    assert out["kind"] == "parquet" and out["keys"] == ["ID"]
    assert {k: v["rows"] for k, v in out["tables"].items()} == {"sas": 4, "dbx": 4, "diff": 4}
    assert out["tables"]["diff"]["name"] == "my_case__work_out__diff"
    assert spark.sql(f"SELECT count(*) AS n FROM {out['tables']['diff']['sql']}").collect()[0]["n"] == 4


def test_sas_dates_become_dates_for_reviewers(spark, tmp_path):
    from pyspark.sql import types as T

    from sas2dbx.runtime.meta import META_KEY

    schema = T.StructType([T.StructField("id", T.IntegerType()),
                           T.StructField("d", T.DoubleType(), metadata={META_KEY: {"type": "num", "format": "DATE9."}})])
    e = spark.createDataFrame([(1, 0.0), (2, 19725.0)], schema)
    r = compare(e, e, CompareOptions(keys=["id"]))
    out = ReviewSink(spark, "demo", local_dir=str(tmp_path)).write_output("c", "work.out", e, e, r)
    got = [str(x["d"]) for x in spark.read.parquet(out["tables"]["sas"]["path"]).orderBy("id").collect()]
    assert got == ["1960-01-01", "2014-01-02"]
