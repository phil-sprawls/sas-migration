#!/usr/bin/env bash
# End to end: a real public .sas7bdat -> spark-sas7bdat ingestion (Spark 3.5, cross-checked against pyreadstat)
# -> the converted program and its parameter notebook (Spark 4) -> 2-decimal parity against an independent oracle,
# with today's parameter values and with values a user typed into the parameter notebook.
# Needs .venv (Spark 4) and .venv-spark35 (Spark 3.5, Scala 2.12): see examples/sales_review/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
export JAVA_HOME="${JAVA_HOME:-$PWD/.tools/jdk17}"

.venv/bin/sas2dbx corpus fetch --only public-sas7bdat                     # SASHELP.PRDSALE copy, pinned commit
.venv-spark35/bin/sas2dbx jars fetch                                        # spark-sas7bdat + deps, SHA-256 checked
.venv-spark35/bin/sas2dbx ingest corpus/public-sas7bdat out/ingest/sales --reader spark-sas7bdat
.venv/bin/python cases/oracles/sales_review.py corpus/public-sas7bdat/productsales.sas7bdat corpus/oracles/sales_review
.venv/bin/sas2dbx convert examples/sales_review/report.sas --param-module examples/sales_review/params.sas \
    --libname SALES=parquet:out/ingest/sales --out out/sales_review/report.py
.venv/bin/sas2dbx validate cases/sales_review.yaml cases/sales_review_user.yaml cases/sales_review_direct.yaml \
    --out out/sales_review/evidence
