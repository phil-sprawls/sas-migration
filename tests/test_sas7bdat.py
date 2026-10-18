"""Reading .sas7bdat files: spark-sas7bdat (parallel, Spark 3.x / Scala 2.12) checked against pyreadstat, ingestion to
lakehouse tables and the round trip back into SAS space.

The real public file is SASHELP.PRDSALE (productsales.sas7bdat, ``sas2dbx corpus fetch --only public-sas7bdat``).
spark-sas7bdat tests run where its jars load (``sas2dbx jars fetch`` on Spark 3.x, e.g. .venv-spark35); the rest
run everywhere."""
import os

import pytest

from sas2dbx.cli import spark_sas7bdat_jars
from sas2dbx.runtime.sasfile import _require_utc_jvm, crosscheck, read_sas_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRDSALE = os.path.join(ROOT, "corpus", "public-sas7bdat", "productsales.sas7bdat")
needs_file = pytest.mark.skipif(not os.path.exists(PRDSALE), reason="sas2dbx corpus fetch --only public-sas7bdat")
needs_jars = pytest.mark.skipif(not spark_sas7bdat_jars(os.path.join(ROOT, ".tools", "jars")),
                                reason="spark-sas7bdat loads on Spark 3.x only (sas2dbx jars fetch)")


@needs_file
@needs_jars
@pytest.mark.spark
def test_spark_sas7bdat_reads_the_real_file_like_pyreadstat(spark):
    df, cols, label, reader = read_sas_file(spark, PRDSALE, "spark-sas7bdat")
    assert reader == "spark-sas7bdat" and label == "Furniture sales data" and df.count() == 1440
    assert cols["MONTH"].format == "MONNAME3." and cols["ACTUAL"].label == "Actual Sales"
    assert df.orderBy("_sas_seq").first()["MONTH"] == 12054.0      # 1993-01-01, not a day early
    res = crosscheck(spark, PRDSALE, df, rows=2000)
    assert res.passed, res.summary()


class _Zone:
    def __init__(self, tz_id: str, offset: int, dst: bool):
        self.tz_id, self.offset, self.dst = tz_id, offset, dst

    def getID(self):
        return self.tz_id

    def getRawOffset(self):
        return self.offset

    def useDaylightTime(self):
        return self.dst


class _FakeSpark:
    def __init__(self, zone: _Zone):
        jvm = type("J", (), {})()
        jvm.java = type("Jj", (), {})()
        jvm.java.util = type("Ju", (), {})()
        jvm.java.util.TimeZone = type("TZ", (), {"getDefault": staticmethod(lambda: zone)})
        self.sparkContext = type("SC", (), {"_jvm": jvm})()


def test_spark_sas7bdat_is_refused_outside_utc():
    """A JVM west of UTC reads every SAS date one day early (America/Kentucky/Louisville: 1993-01-01 -> 1992-12-31)."""
    with pytest.raises(RuntimeError, match="-Duser.timezone=UTC"):
        _require_utc_jvm(_FakeSpark(_Zone("America/Kentucky/Louisville", -5 * 3600 * 1000, True)))
    _require_utc_jvm(_FakeSpark(_Zone("UTC", 0, False)))


@needs_file
@pytest.mark.spark
def test_ingested_table_reads_back_identically(spark, tmp_path):
    """sas7bdat -> Delta-typed Parquet (DATE columns, NULL for blank, SAS metadata) -> back into SAS space."""
    pytest.importorskip("pyspark", minversion="4.0")        # reading back uses SQL functions (Spark 4 / Databricks)
    from sas2dbx.ingest import ingest
    from sas2dbx.runtime.context import SasContext
    from sas2dbx.runtime.meta import schema_meta
    from sas2dbx.validate.compare import CompareOptions, compare

    ev = ingest(spark, PRDSALE, str(tmp_path), reader="auto", log=lambda *_: None)
    raw = spark.read.parquet(str(tmp_path / "productsales"))
    assert dict(raw.dtypes)["MONTH"] == "date" and ev["tables"]["productsales"]["rows"] == 1440
    ctx = SasContext(spark, {"lake": f"parquet:{tmp_path}", "sas": f"sasfiles:{os.path.dirname(PRDSALE)}"},
                     write_outputs=False, sas_reader="pyreadstat")
    lake, direct = ctx.get("lake.productsales"), ctx.get("sas.productsales")
    assert schema_meta(lake.schema)["MONTH"].format == "MONNAME3."
    res = compare(direct, lake, CompareOptions(decimals=9, mode="ordered"), name="productsales")
    assert res.passed, res.summary()


def test_merge_in_flag_is_not_written_when_columns_pass_through():
    """IN= is temporary in SAS: an input whose columns are only known at run time must not leak it."""
    from sas2dbx.codegen.program import convert

    conv = convert("data out; merge a(in=ina) b; by id; if ina; run;", file="t.sas")
    code = "\n".join(conv.steps[0].code)
    assert "P.select_pdv(" in code and "'ina'" in code.split("P.select_pdv(")[1].split("drop=")[1]
