import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
_jdk = os.path.join(ROOT, ".tools", "jdk17")
if os.path.isdir(_jdk):
    os.environ.setdefault("JAVA_HOME", _jdk)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    from sas2dbx.cli import spark_sas7bdat_jars

    b = (SparkSession.builder.master("local[2]")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.driver.memory", "2g")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
         .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC"))
    jars = spark_sas7bdat_jars(os.path.join(ROOT, ".tools", "jars"))
    if jars:                                    # Spark 3.x only (spark-sas7bdat is Scala 2.12)
        b = b.config("spark.jars", ",".join(jars))
    s = b.getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture(params=["true", "false"], ids=["ansi", "legacy"])
def ansi_spark(spark, request):
    """Run a test with ANSI mode on (Spark 4 / DBR 17+ default) and off."""
    old = spark.conf.get("spark.sql.ansi.enabled")
    spark.conf.set("spark.sql.ansi.enabled", request.param)
    yield spark
    spark.conf.set("spark.sql.ansi.enabled", old)
