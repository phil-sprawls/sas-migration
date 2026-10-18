# sas2dbx

Convert SAS programs to Databricks (PySpark notebooks) and prove each conversion with **output data parity**: every output data set must match the SAS result, with character values equal and numbers equal to 2 decimal places (|a − b| < 0.005; two missing values count as equal).

- **Rules first.** A deterministic SAS front end covers the macro processor, DATA step, PROC SQL and the common PROCs. It generates readable PySpark on top of a small runtime library (`sas2dbx.runtime`) that encodes SAS semantics: missing values, BY-group processing, MERGE, RETAIN/LAG, formats and informats, and PROC FREQ/MEANS/TRANSPOSE statistics.
- **Never approximate.** Whatever the rules can't convert exactly, they mark *unsupported*. That covers hash objects, CALL EXECUTE, run-time macro logic, `%SYSFUNC(OPEN/ATTRN/VARNUM)`, unsupported PROCs, and code that doesn't parse. An unsupported cell raises until it is converted.
- **The LLM is optional and gated.** Claude (through the Anthropic API or Databricks Model Serving) converts only the unsupported units. Its code goes through a static guard, then the same parity gate, with a bounded repair loop.
- **Measured, not assumed.** `sas2dbx eval` runs a golden-set bake-off that decides, per construct family, what should be deterministic and what should go to the LLM.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the semantic hazard registry.

## Quick start

```bash
uv venv && uv pip install -e '.[convert,validate,stats,dev]'   # local Spark needs Java 17 (.tools/jdk17 is picked up)
sas2dbx corpus fetch                                           # pinned public corpus: CDISC pilot + PhUSE scripts
sas2dbx analyze corpus/phuse-scripts --out reports/estate.json # readiness, unsupported reasons, migration waves
sas2dbx convert corpus/cdisc-pilot/programs/adae.sas --libname SDTMNEW=sasfiles:corpus/cdisc-pilot/sdtm \
    --libname ADAMNEW=sasfiles:corpus/cdisc-pilot/adam --out out/adae.py
sas2dbx validate cases/*.yaml                                  # convert, run, compare at 2 decimals -> out/*.json evidence
sas2dbx conformance                                            # 66 SAS-semantics cases (export with --export-sas to re-baseline on SAS)
sas2dbx runs                                                   # run log: who ran what, scenario, outcome, LLM tokens and cost
```

## Parameters and values from outside the program

Enterprise programs often get macro variables from outside the program: prompts, a parameter module, the autoexec or `-SYSPARM`. The converter never freezes these into the code.

- **Undefined variables:** a step that uses a variable nothing defines stays *unsupported*, and its reason names the variable. So does any `%IF`/`%DO`/`%LET` logic that depends on one. `sas2dbx analyze` lists these names per program and across the estate, as the parameter catalog to confirm.
- **Run values:** `&SYSDATE9`, `&SYSPARM`, `%SYSGET(...)` and `%SYSFUNC(TODAY(), fmt.)` are read when the notebook runs, never taken from the converter's clock or environment.
- **Declaring them:** each name is declared one of two ways:

```bash
sas2dbx convert report.sas --param region,start_dt --define outroot=/Volumes/main/reports
# region, start_dt -> read from the Databricks job/notebook parameters REGION and START_DT (one notebook, every value)
# outroot          -> fixed at conversion time (an environment constant)
```

A parameter that the job doesn't supply fails the run where it is first used. It never runs with a blank value. Conformance cases take `params: [...]` and get those values only at run time, so the parity gate checks the notebook that will actually be deployed. Run such a case with several values, because parity for one value proves nothing about the others.

### Parameter files → parameter notebooks

When end users edit a `%INCLUDE`d file of `%LET` statements, that file becomes a separate **parameter notebook**. Users edit only that notebook. The program's notebook loads it with `%run`, just as SAS includes the file, so users stay soft-gated from the code.

```bash
sas2dbx convert report.sas --param-module params.sas --out out/report.py   # writes out/report.py and out/params.py
sas2dbx analyze <estate>     # summary.parameter_modules: included files made only of %LET statements
```

Each parameter's kind comes from how the program uses it:

| use in SAS | kind | users write in the notebook |
|---|---|---|
| `where &filter;` (DATA step, PROC, single-table SQL) | filter | a **Spark SQL** condition, run on the data as Databricks tables show it (`DATE` columns, `NULL` for blank text) |
| `in (&regions)` | value list | `["EAST", "WEST"]` |
| `keep &vars;`, `keep=`/`drop=`, `by &vars;` (PROC SORT), SQL `select`/`group by`/`order by` | column list | `["id", "region", "amount"]` |
| `"&region"`, `&n`, `"&start"d` | text / number / date | `"EAST"`, `5`, `"2024-01-31"` |

Today's values become the defaults. A SAS filter is translated to Spark SQL that selects the same rows as SAS. SAS treats a missing value as smaller than any number, so `amount < 100` becomes `(amount < 100 OR amount IS NULL)`. The notebook header explains this to users, because filters they write themselves follow Spark's rules: a comparison with NULL is never true.

Every run checks all values before reading any data:
- a filter must be one valid condition over existing columns, and may not call `reflect` or `java_method`;
- lists must contain valid names, and the columns must exist;
- dates must be valid.

All problems are reported together. Without SAS access, the filter translation is tested against a second implementation, the converter's own SAS-semantics compiler. Both must select the same rows on generated data full of missing values, blanks and dates. A deliberately wrong translation fails this check.

A filter parameter combined with `OR`/`NOT`, used in a multi-table SQL query, or written in a PROC statement other than `WHERE`/`SORT BY` is reported as unsupported. It is never guessed. `%run` executes the parameter notebook with the job's identity, the same trust model as `%INCLUDE` today. If users must not be able to run code, keep the values as job parameters (`--param`).

## SAS data files: `.sas7bdat` with spark-sas7bdat

```bash
sas2dbx jars fetch                                         # spark-sas7bdat 3.0.0 + deps for local Spark 3.x (SHA-256 checked)
sas2dbx ingest <dir-of-.sas7bdat> <dest> --reader spark-sas7bdat --format delta
```

- **Ingestion:** `sas2dbx ingest` reads SAS data sets in parallel with [spark-sas7bdat](https://github.com/saurfang/spark-sas7bdat). It checks each read against pyreadstat, row by row, and stops if they disagree. It writes Delta/Parquet tables:
  - dates as `DATE`, and blank or missing values as `NULL`;
  - SAS formats, labels and lengths as column metadata;
  - the file's row order as `_sas_seq`.

  Converted programs read those tables back into SAS space exactly, and the converter takes their schemas from that metadata.
- **Direct reads:** a `sasfiles:` library also reads `.sas7bdat` files with spark-sas7bdat when it is installed (`sas_reader: auto`), and falls back to pyreadstat otherwise. The reader each run used is recorded in the evidence.
- **Version limits:**
  - spark-sas7bdat is Scala 2.12 only: Spark 3.x and Databricks Runtime ≤ 16.x classic compute (Maven `saurfang:spark-sas7bdat:3.0.0-s_2.12`). Spark 4 and Runtime 17+ cannot load it, and serverless cannot load JARs. On serverless-only workspaces, ingest elsewhere and use `sas2dbx publish` (see "Validating on Databricks").
  - Converted programs need Spark 4 or Databricks Runtime, because they use SQL functions.
  - Locally, ingestion runs on Spark 3.5 and the programs run on Spark 4.
- **The time-zone bug:** spark-sas7bdat converts dates in the JVM time zone, so every date is a day early west of UTC. sas2dbx refuses to read with it unless the JVM runs in UTC.

`examples/sales_review/` runs the whole flow end to end: a real public `.sas7bdat` (`SASHELP.PRDSALE`), a synthetic program with a parameter file, and parity against an independent oracle for today's values and a user's values.

## Validating on Databricks

```bash
uv venv .venv-dbconnect --python 3.12      # Databricks Connect replaces pyspark, so it gets its own environment
uv pip install --python .venv-dbconnect/bin/python -e '.[convert,validate,stats]' "databricks-connect~=18.3.0"
sas2dbx conformance --engine databricks                                  # through Databricks Connect
sas2dbx conformance --engine databricks --as-job --schema main.sas2dbx   # inside a notebook job on the cluster
sas2dbx validate cases/databricks/*.yaml --engine databricks --as-job --schema main.sas2dbx
```

- **`--engine databricks`** uses Databricks Connect:
  - the Spark work runs on serverless compute, or on the cluster named in `SAS2DBX_CLUSTER_ID`;
  - the notebook's Python runs on your machine;
  - the Databricks Connect version must suit the compute: 18.x with Python 3.12 for current serverless.
- **`--as-job`** runs exactly what users will run:
  - Each case's converted notebook, its parameter notebook (with the case's values filled in) and a `sas2dbx.yaml` go to `~/sas2dbx-validation` in the workspace, next to a small harness notebook.
  - The harness installs the sas2dbx wheel, `%run`s the program unchanged, and saves the outputs the case checks as tables in `--schema`. Those tables are then compared on your machine.
  - The wheel is built from this source tree and uploaded to the volume `<schema>.sas2dbx_jobs`.
  - Case libraries must be Unity Catalog schemas (`uc:catalog.schema`). The converter reads their schemas from Unity Catalog.
- **Serverless-only workspaces** can't load spark-sas7bdat. Ingest where it runs, then publish:

  ```bash
  sas2dbx publish out/ingest/sales workspace.sas2dbx_demo   # Parquet -> UC volume -> Delta tables, metadata kept and checked
  ```

First run, on a Databricks Free Edition workspace (serverless, Runtime 19.9, Photon, ANSI on):

| run | result |
|---|---|
| conformance suite as a job | 66 of 66 pass |
| conformance suite through Databricks Connect | 62 of 66 pass: Python UDFs could not start (known issue below) |
| sales review, Unity Catalog table: today's values and a user's values, as a job | 2 of 2 cases, 10 of 10 outputs |
| sales review through Databricks Connect: Unity Catalog table, and the `.sas7bdat` read directly | 3 of 3 cases |

**Known issue on that workspace:** Python UDFs from Databricks Connect sessions fail with `SANDBOX_STARTUP … exec format error`, because the UDF sandbox's Python was built for another CPU architecture. This happens with Connect 17.3 and 18.3 alike, while the same UDFs work in notebooks. Four runtime functions use Python UDFs: `BEST.` number-to-text, `PUT`/`INPUT` with formats not compiled to SQL, and `TINV`. Cases that use them need `--as-job` there.

## Run log: who ran what, on which code, the outcome and the token spend

Every command except `runs` appends one record to `logs/sas2dbx-runs.jsonl`:

| field | what it holds |
|---|---|
| `user` | the OS account and host; for runs that use Databricks, also the Databricks user they authenticated as. In pipelines, `on_behalf_of` records who started them: the CI system's actor (GitHub, GitLab, Azure DevOps, Jenkins) or `SAS2DBX_REQUESTED_BY` |
| `command`, `args`, `context` | what ran (`--define` values redacted), the engine, and Databricks job run pages |
| `scenarios` | per program or case: SHA-256, lines, steps by kind, semantic hazards, run-time parameters (names only), how much converted, and the case's description and ground truth |
| `outcome` | per case: PASS, FAIL or ERROR, with each output's rows, its mismatched columns and the evidence file. For the run: status, exit code, cases passed and failed |
| `llm` | calls and tokens (input, output, cache read and write) with their cost at list price. `billed` covers live calls and `replayed` covers answers replayed from the response cache at no charge. Calls with no list price, such as Databricks Model Serving (billed in DBUs), count as `unpriced_calls` |

```bash
sas2dbx runs                        # the last 20 runs as a table, with totals per person
sas2dbx runs --user alice --json    # one person's runs as JSON Lines
```

For a team, set `SAS2DBX_RUN_LOG_VOLUME=/Volumes/<catalog>/<schema>/<volume>`. Each run then also uploads its record there, and Databricks SQL reads them all. Each record is read as a single VARIANT value, so fields that are new or absent from some records never break the query (Databricks Runtime 15.3+ or serverless):

```sql
SELECT rec:started_at::timestamp AS started_at,
       coalesce(rec:user.databricks_user::string, rec:user.os_user::string) AS who,
       rec:command::string AS command, rec:outcome.status::string AS status,
       rec:llm.billed.usd::double AS cost_usd
FROM read_files('/Volumes/<catalog>/<schema>/<volume>', format => 'json', singleVariantColumn => 'rec')
```

`docs/HANDOFF.md` §4.7 has the team dashboard queries.

- **Other settings:** `SAS2DBX_RUN_LOG` sets another local path, and `off` turns the local log off.
- **What it never holds:** keys, tokens, parameter values, or the `--define` values that SAS programs sometimes use for passwords.
- **Failures:** if a write or upload fails, the run still completes.

## How much should be deterministic and how much LLM? `sas2dbx eval`

The golden set has 67 cases:

- 47 conformance cases with expected outputs, which pin down SAS semantics the rules already cover (4 of them for run-time values: macro values in strings, parameters, automatic values).
- 14 *frontier* cases that the rules route to LLM/manual by design: random access, multiple SET statements, hash objects, data-dependent DO loops, SCL metadata functions, PROC UNIVARIATE/TTEST/RANK, SQL DDL, retained recurrences, UPDATE (two cases, one with duplicate master keys), CALL EXECUTE and run-time macro loops.
- 6 real programs: 3 CDISC programs, compared against SAS-produced output (G1) or an independent double-programming oracle (G3), and the sales review over a real `.sas7bdat` with today's values, a user's values and a direct read (G3).

Rules alone pass 53 of 67 on the first attempt (79.1%): every case the rules cover, and none of the 14 frontier cases, by design.

Four strategies are compared:

| strategy | what it does |
|---|---|
| `rules` | the deterministic converter only |
| `hybrid` | rules, plus the LLM for the unsupported units only |
| `llm_runtime` | the LLM converts the whole program using the sas2dbx runtime and the hazard catalog |
| `llm_plain` | the LLM converts the whole program to plain PySpark |

The primary metric is the parity pass rate, both on the first attempt and after the repair loop. Secondary metrics are attempts, cost, latency and run-to-run agreement (`--runs N`). The report ends with a per-family decision: use rules where rules reach parity; use the LLM (behind the gate) where only the LLM does; otherwise extend the runtime or convert manually.

```bash
sas2dbx eval --strategies rules                    # free baseline
sas2dbx eval --estimate --runs 3                   # cost envelope, no spend
sas2dbx eval --runs 3 --yes                        # live run (needs ANTHROPIC_API_KEY in .env)
sas2dbx eval --provider databricks:<endpoint>@sas2dbx --yes   # Claude on Databricks Model Serving
```

Passing the gate on a case's own rows is necessary but not sufficient. A conversion can fit the rows it was checked against and still be wrong on other data. `sas2dbx robust` re-runs every conversion that passed on synthetic edge-case variants of the inputs: missing values, blanks, ties, duplicated rows and boundary numbers, with BY groups kept intact. Each conversion must agree to 2 decimals with a trusted reference. The reference is the rules conversion where the rules cover the case, otherwise the case's independent `reference_pyspark` (double programming). With SAS available, the same variants can be exported and re-baselined on SAS instead. It replays cached answers, so it never spends.

```bash
sas2dbx robust --variants 5                         # differential test on synthetic inputs (no LLM spend)
```

LLM answers are cached by content hash in `.sas2dbx-cache/`, so re-runs replay at no cost. Each `--runs` sample uses its own cache namespace, so the samples are independent. The Anthropic provider streams responses, caches the system prompt, and enables the server-side refusal fallback (`fallbacks: "default"`) by default.

## Credentials

- **Anthropic:** `ANTHROPIC_API_KEY=...` in `.env`. The file is gitignored. Never pass the key on the command line or paste it in chat.
- **Databricks:** run `databricks auth login --host <workspace-url>` to create an OAuth profile in `~/.databrickscfg`. It is named `DEFAULT` unless you pass `--profile <name>`. sas2dbx uses the SDK's default profile unless `--profile` or `SAS2DBX_PROFILE` names another.
