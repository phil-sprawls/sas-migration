# Sales review: a synthetic program over a real public `.sas7bdat`

The program is synthetic, written in the style of enterprise SAS: a parameter file that business users edit, then filters, list parameters, a merge and summaries. The data is a real public SAS data set, read with spark-sas7bdat.

| file | what it is |
|---|---|
| `params.sas` | the parameter file business users edit: text, two dates, a value list, two filters, two column lists |
| `report.sas` | the program: it `%INCLUDE`s `params.sas`, reads `SALES.PRODUCTSALES`, and writes `detail`, `summary`, `review`, `flagged` and `report_title` |
| `../../cases/oracles/sales_review.py` | an independent pandas + pyreadstat implementation of the same specification (double programming) |
| `../../cases/sales_review*.yaml` | three parity cases: today's values; a user's values; a direct `.sas7bdat` read |

**Data:** `productsales.sas7bdat` is a copy of the SAS sample data set `SASHELP.PRDSALE`. It holds 1,440 rows of furniture sales, actual vs predicted, for 1993–94. It has SAS formats (`DOLLAR12.2`, a `MONNAME3.` date), labels and a data set label. It comes from the pandas test suite (BSD-3-Clause), is pinned to a commit in `src/sas2dbx/corpus/manifest.yaml`, and is never vendored.

## Run it

```bash
# once: a Spark 3.5 (Scala 2.12) environment next to the main Spark 4 one, because spark-sas7bdat has no Scala 2.13 build
uv venv .venv-spark35 --python 3.11
uv pip install --python .venv-spark35/bin/python "pyspark==3.5.9" "pandas>=2.2,<2.3" "numpy>=1.26,<2" \
    "pyarrow>=14,<18" pyreadstat pyyaml "sqlglot>=30,<31"
uv pip install --python .venv-spark35/bin/python -e . --no-deps

examples/sales_review/run.sh      # about a minute
```

The script does six things:

1. Fetches the data file from its pinned commit.
2. Fetches spark-sas7bdat 3.0.0 and its dependencies, with SHA-256 checks.
3. Runs **`sas2dbx ingest`** on Spark 3.5, which:
   - reads the file in parallel with spark-sas7bdat;
   - checks every row against pyreadstat, an independent reader, and refuses to write anything if they disagree;
   - writes a Parquet table as Databricks users expect it: `MONTH` as a `DATE`, SAS formats and labels as column metadata, and the file's row order as `_sas_seq`.
4. Writes the oracle's expected outputs.
5. Converts `report.sas` with `--param-module params.sas`. It writes `out/sales_review/report.py` and the parameter notebook `out/sales_review/params.py`.
6. Runs the parity checks on Spark 4, at 2 decimals.

Expected result:

```
PASS  sales_review         (today's values: SAS rules, e.g. a missing percentage is flagged)
PASS  sales_review_user    (a user's Spark SQL filters, other dates, countries, columns and groups)
PASS  sales_review_direct  (the .sas7bdat read directly instead of the ingested table)
```

## On Databricks

Serverless-only workspaces, such as Databricks Free Edition, cannot load spark-sas7bdat. So ingest locally as above, then publish the table to Unity Catalog and run the converted notebooks there, using the Databricks Connect environment from the main README:

```bash
.venv-dbconnect/bin/sas2dbx publish out/ingest/sales workspace.sas2dbx_demo
.venv-dbconnect/bin/sas2dbx validate cases/databricks/sales_review.yaml cases/databricks/sales_review_user.yaml \
    --engine databricks --as-job --schema workspace.sas2dbx_demo
```

- **Publishing:** the ingested Parquet goes unchanged to a volume and into the Delta table `workspace.sas2dbx_demo.productsales`. It keeps its SAS formats and labels, and `PUBLISH.json` records the copy.
- **The cases:** `cases/databricks/` holds the same two cases with `SALES` mapped to `uc:workspace.sas2dbx_demo`.
  - The converter reads the table's schema from Unity Catalog and produces the same notebook, byte for byte.
  - With `--as-job`, the `report` and `params` notebooks run in a serverless job exactly as deployed: `%run ./params`, the user's edits in the parameter notebook, and the sas2dbx wheel installed from a volume.
- **Without `--as-job`:** the same cases run through Databricks Connect.

Result on Free Edition (Runtime 19.9): all 10 outputs match the oracle at 2 decimals, both as a job and through Databricks Connect.

## What it shows

- **Today's filters, translated:** `actual > 100 and prodtype ne 'OFFICE'` becomes `(actual > 100 AND (prodtype <> 'OFFICE' OR prodtype IS NULL))`. The flag filter `pct_of_target < 95` becomes `(pct_of_target < 95 OR pct_of_target IS NULL)`, so groups with no target are flagged, as in SAS.
- **A user's filter keeps Spark SQL rules:** a missing percentage is not flagged by `pct_of_target < 105`.
- **One notebook serves every value:** the user run keeps an extra column (`division`) and groups by an extra column (`region`). The run-time columns pass through the sort and the merge, and `IN=` flags are never written.

## spark-sas7bdat

**Where it runs:**

- spark-sas7bdat 3.0.0 is built for Scala 2.12 only. On Spark 4 and Databricks Runtime 17+ (Scala 2.13), it fails with `NoClassDefFoundError: scala/Serializable`.
- On Databricks classic compute with Runtime ≤ 16.x, install the Maven library `saurfang:spark-sas7bdat:3.0.0-s_2.12` from `https://repos.spark-packages.org/`. A `sasfiles:` library then reads `.sas7bdat` files with it directly.
- Serverless compute cannot load custom JARs. There, ingest on classic compute, or read with pyreadstat (`sas_reader: pyreadstat`) for small files.

**The time-zone bug:** spark-sas7bdat converts SAS dates in the **JVM's time zone**. West of UTC, every date is read **one day early**; on this machine's US time zone, `1993-01-01` came back as `1992-12-31`. sas2dbx refuses to read with spark-sas7bdat unless the JVM runs in UTC. Databricks clusters already do; locally, `-Duser.timezone=UTC` is set. The pyreadstat cross-check is the second line of defence.
