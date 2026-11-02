"""Column order is part of output parity: runtime steps must build the SAS PDV order."""
import pytest

from sas2dbx.runtime import P

pytestmark = pytest.mark.spark


def test_merge_keeps_pdv_order_with_by_variables_in_place(spark):
    summary = spark.createDataFrame([("C", "W", "P", 3.0, 0)], "country string, region string, product string, n double, _sas_seq long")
    targets = spark.createDataFrame([("C", "P", 9.0, 0)], "country string, product string, target double, _sas_seq long")
    out = P.merge([summary, targets], by=["country", "product"], in_vars=["in_s", None])
    assert [c for c in out.columns if not c.startswith("_sas_")] == ["country", "region", "product", "n", "target", "in_s"]


def test_select_pdv_keeps_unseen_input_columns_in_pdv_position(spark):
    """A column the converter never saw (e.g. added by a run-time parameter) enters the PDV at the SET/MERGE,
    in the input's order - not after the variables the converter knew."""
    df = spark.createDataFrame([("C", "W", "P", 3.0, 1)], "country string, region string, product string, n double, _sas_seq long")
    df = df.withColumn("pct", df.n * 2)
    out = P.select_pdv(df, {"country": "char", "product": "char", "n": "num", "pct": "num"})
    assert out.columns == ["country", "region", "product", "n", "pct", "_sas_seq"]
    tail = spark.createDataFrame([("C", "x", 1)], "country string, extra string, _sas_seq long")
    assert P.select_pdv(tail, {"country": "char", "new": "num"}).columns == ["country", "extra", "new", "_sas_seq"]
