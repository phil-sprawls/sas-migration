"""Validation on Databricks without a workspace: the job harness notebooks and Unity Catalog schemas for the converter.

The harness notebooks only run on the cluster, where a mistake would surface late, so they are compiled here.
`sas2dbx validate|conformance --engine databricks --as-job` exercises them for real (README, "Validating on Databricks")."""
import re

import pytest

from sas2dbx.validate.jobs import CONFORMANCE_HARNESS, PARITY_HARNESS


def _python(notebook: str) -> str:
    return "\n".join(ln for ln in notebook.split("\n") if not ln.startswith("# MAGIC"))


def test_harness_notebooks_are_python_and_run_the_program_unchanged():
    parity = (PARITY_HARNESS.replace("__WHEEL__", "/Volumes/c/s/sas2dbx_jobs/sas2dbx-0.1.0-py3-none-any.whl")
              .replace("__MAIN__", "report").replace("__OUTPUTS__", repr({"detail": "c.s.validation_case_detail"})))
    suite = (CONFORMANCE_HARNESS.replace("__WHEEL__", "/Volumes/c/s/sas2dbx_jobs/w.whl")
             .replace("__REQUIREMENTS__", "'sqlglot>=30,<31'").replace("__DIR__", "/Volumes/c/s/sas2dbx_jobs/suite"))
    for notebook in (parity, suite):
        assert not re.search(r"__[A-Z]+__", notebook)
        compile(_python(notebook), "harness.py", "exec")
    assert "# MAGIC %run ./report" in parity and "%pip install /Volumes/" in parity


@pytest.mark.spark
def test_uc_library_schemas_come_from_the_tables(spark, tmp_path):
    """A `uc:` library: the converter takes SAS attributes from the tables' column metadata, as `publish` keeps it."""
    from sas2dbx.runtime.meta import META_KEY
    from sas2dbx.semantic.schema import Catalog
    from sas2dbx.validate.runner import build_catalog

    spark.sql(f"CREATE DATABASE IF NOT EXISTS sas2dbx_uc_test LOCATION '{tmp_path}'")
    try:
        df = spark.createDataFrame([(1.5, "A", 1)], "amount double, region string, _sas_seq long")
        df.select(df.amount.alias("AMOUNT", metadata={META_KEY: {"type": "num", "length": 8, "format": "DOLLAR12.2",
                                                                 "label": "Actual sales"}}),
                  df.region.alias("REGION", metadata={META_KEY: {"type": "char", "length": 10}}),
                  "_sas_seq").write.mode("overwrite").saveAsTable("sas2dbx_uc_test.sales")
        spark.range(1).createOrReplaceTempView("scratch")         # temporary views are not library members
        cat = build_catalog({"SALES": "uc:sas2dbx_uc_test"}, spark)
        schema = cat.get(Catalog.key("SALES", "sales"))
        assert [(v.name, v.type, v.length, v.format, v.label) for v in schema.vars] == [
            ("AMOUNT", "num", 8, "DOLLAR12.2", "Actual sales"), ("REGION", "char", 10, None, None)]
        assert cat.get(Catalog.key("SALES", "scratch")) is None
        assert build_catalog({"SALES": "uc:sas2dbx_uc_test"}).get(Catalog.key("SALES", "sales")) is None   # no session
    finally:
        spark.sql("DROP DATABASE IF EXISTS sas2dbx_uc_test CASCADE")
