# sas2dbx: enterprise handoff

**From:** the build team (design, build and first validation, 2026-10-16 to 2026-10-18)
**To:** the enterprise migration team (migration engineers, SAS subject-matter experts, data owners, platform team)
**Version:** sas2dbx 0.1.0

This document says what was built and what it has proved, how to test it in your environment, how to make its evaluations useful to a team, and a plan for launch. The design is in [ARCHITECTURE.md](ARCHITECTURE.md), and the commands are in the [README](../README.md).

---

## 1. Start here

**sas2dbx converts SAS programs into Databricks notebooks and proves each conversion by output parity.** Every output data set of the converted notebook must match the SAS result, with character values equal and numbers equal to 2 decimal places (|a − b| < 0.005). That check is the *parity gate*.

- A deterministic converter (the *rules*) handles everything it can convert exactly, and marks everything else *unsupported*. It never approximates.
- Claude (an LLM) converts only the unsupported pieces, and its code must pass the same gate.
- Every run is logged: who ran it, the code scenario, the outcome and the LLM spend.

| Area | State | Evidence |
|---|---|---|
| Converter (rules) | built: macro processor, DATA step, PROC SQL, common PROCs, parameter modules | 66 conformance cases pass locally and on Databricks |
| LLM assist (hybrid) | built: Claude for unsupported steps only, behind the gate | pilot: 100% first-attempt parity on the frontier cases (2026-10-16) |
| Validation | built: parity cases, conformance suite, LLM bake-off (`eval`), synthetic-input differential test (`robust`) | 526 automated tests: 515 pass, 11 skip without optional pieces (spark-sas7bdat jars, Spark 3.5) |
| Databricks | built and tested on **Databricks Free Edition (serverless only)**: Databricks Connect, deployed notebooks as jobs, Unity Catalog | §2.3 |
| SAS data sets | `.sas7bdat` via spark-sas7bdat (cross-checked against pyreadstat), XPORT and fallback via pyreadstat | real public `.sas7bdat` end to end |
| Run log | built: local JSON Lines plus an optional team copy in a Unity Catalog volume | §4.7 |
| Not yet tested | classic compute, enterprise SSO and service principals, production data volumes, Databricks Model Serving, real enterprise SAS code | §3.6 |

**Your first three steps:**
1. Work through the test sequence in §3.3, T1 to T9, in your workspace.
2. Run `sas2dbx analyze` over your SAS estate (§3.4).
3. Start the Phase 1 pilot (§5.1) with SAS-produced baselines (G1).

---

## 2. What we did

### 2.1 Decisions (made with the project sponsor) and why

| Decision | Why |
|---|---|
| Cover all four categories: DATA step ETL, PROC SQL, macro-heavy code, statistical and reporting PROCs | Enterprise estates mix all four. The analyzer reports readiness per program, so waves can be planned around what is hard. |
| **Hybrid conversion:** deterministic rules first, then a pluggable LLM (Anthropic API or Databricks Model Serving) for what the rules mark unsupported | Rules are free, deterministic and reviewable. The LLM covers the long tail, and only behind the same gate. |
| **Acceptance = output parity to 2 decimals** | A deterministic gate. "Code that runs" is not success; matching outputs are. |
| No SAS access for the build team: public corpora plus synthetic data | Ground truth came from SAS-produced public outputs (G1) and independent double programming (G3); see §4.4. **Your team has SAS, so prefer G1 baselines from production runs.** |
| An `%INCLUDE`d `%LET` file of user parameters becomes a separate **parameter notebook** that the program `%run`s | "If it comes from a separate file in SAS, it should come from a separate notebook in Databricks": users stay soft-gated from the code. Filters are written in Spark SQL; value and column lists are Python lists. |
| Macro variables that nothing defines are never frozen at conversion | Values from prompts, schedulers or `-SYSPARM` must come from the run, not from the converter's environment. |

### 2.2 What was built

| Component | Path | Purpose |
|---|---|---|
| Analyzer | `src/sas2dbx/analyze` | Estate inventory and readiness: unsupported reasons, macro variables nothing defines (the parameter catalog), parameter modules |
| Macro processor, parser | `src/sas2dbx/macro`, `src/sas2dbx/lang` | SAS macro language and SAS syntax to an intermediate representation. Robust to very long lines and deeply nested code. |
| Semantic model | `src/sas2dbx/semantic` | Schemas plus the **hazard registry** (`hazards.py`, 37 SAS-vs-Spark behaviour differences) |
| Code generation | `src/sas2dbx/codegen` | Rules: readable PySpark notebooks, one cell per SAS step, with the SAS source shown above each cell |
| Runtime library | `src/sas2dbx/runtime` | The only dependency of generated code. It holds SAS semantics: missing values, BY groups, MERGE, RETAIN/LAG, formats and informats, PROC statistics, parameters. |
| LLM assist | `src/sas2dbx/llm` | Providers (Anthropic, Databricks Model Serving), static guard, repair loop, response cache, `eval`, `robust` |
| Validation | `src/sas2dbx/validate` | Comparator, parity cases, conformance suite, Databricks job mode (`jobs.py`) |
| Data | `src/sas2dbx/ingest.py`, `runtime/sasfile.py` | `.sas7bdat`/XPORT to Delta/Parquet (`ingest`), then to Unity Catalog (`publish`) |
| Run log | `src/sas2dbx/runlog.py` | Who, scenario, outcome and token spend for every run (`sas2dbx runs`) |
| Test assets | `conformance/`, `cases/`, `examples/sales_review/` | 66 conformance cases, parity cases with independent oracles, a worked example |

### 2.3 Results to date

| Date | What | Where | Result |
|---|---|---|---|
| 2026-10-16 | LLM pilot on the 60-case golden set of the day, Claude Opus 5.5 | local Spark 4 | first-attempt parity: hybrid 100%, llm_runtime 100%, llm_plain 96.6% (100% after one repair); **$4.86** billed |
| 2026-10-16 | Synthetic-input differential test (`robust`, no spend) | local | agreement with a trusted reference: llm_plain 100%, llm_runtime 98.9%, hybrid 95.2% (see note 1) |
| 2026-10-17 | Real public `.sas7bdat` (SASHELP.PRDSALE, 1,440 rows) with a synthetic parameterised program | Spark 3.5 ingest, Spark 4 run | spark-sas7bdat read cross-checked against pyreadstat at 9 decimals; parity PASS for today's values, a user's values and a direct read |
| 2026-10-18 | Conformance suite (66 cases) | Databricks Free Edition, serverless Runtime 19.9 (ARM, Photon, ANSI on) | **66/66 as a notebook job**; 62/66 through Databricks Connect (see note 2) |
| 2026-10-18 | Sales review against Unity Catalog | same | 2/2 cases as deployed notebooks in a job (10/10 outputs); 3/3 through Connect |
| 2026-10-18 | Rules-only baseline on today's 67-case golden set | local | **53/67 (79.1%)**: every case the rules cover passes; the 14 frontier cases go to the LLM by design |
| 2026-10-18 | Paid demo: one hybrid conversion (hash-object lookup) with Claude Haiku 5.5 | local | PASS on the first attempt, **$0.0031** for 13,497 tokens, recorded in the run log |
| 2026-10-18 | Readiness of a public, macro-heavy estate (PhUSE scripts) | local | 93 programs, 2,308 steps: **52.1% of steps** and **8.6% of programs** fully converted by the rules (see note 3) |

1. All the `robust` disagreements were the SAS `UPDATE` statement with duplicate master keys. In that case the reference oracle was wrong, not the LLM's code. The oracle was fixed against the SAS documentation, and hazard `H-UPDATE-DUPMASTER` plus case `FRONTIER-UPDATE-02` were added. Adding the hazard changed the LLM system prompt, so the pilot's cached answers no longer replay, and re-running the pilot costs money again (about $5 at Opus pricing).
2. The four Connect failures are a Databricks-side defect on that workspace. Python UDFs started from Databricks Connect failed with `SANDBOX_STARTUP … exec format error`: the sandbox's Python was built for a different CPU architecture. The same UDFs work in notebooks. **Check this in your workspace (T3).**
3. Expect a large LLM or manual share for macro-heavy utility code. The top blockers were run-time macro logic (431 steps), `%SYSFUNC(OPEN())` metadata calls (221), data-dependent DO loops (60), the `X` statement (40) and unsupported PROCs (39). Your estate will differ: measure it with `analyze` (§3.4).

### 2.4 What a PASS proves, and what it does not

- **It proves** the converted notebook reproduces the expected outputs for the tested inputs and parameter values, to 2 decimals, with exact character values. Evidence files record the SHA-256 of the SAS source, the notebook and the expected outputs.
- **It does not prove** correctness on other data. A conversion can fit the tested rows and still be wrong on inputs with more missing values, ties, duplicate keys or new categories. Mitigations:
  - several production cycles (§5.1);
  - several parameter value sets (§4.4);
  - `sas2dbx robust` synthetic variants for LLM-written code.
- **It does not prove** the expected outputs are right. Prefer SAS-produced outputs (G1). The build team found one wrong oracle this way (note 1 above).
- **It does not cover** performance, scheduling, security or downstream consumers.

### 2.5 Defects the gate caught during the build (why the gate matters)

| Defect | Caught by |
|---|---|
| Run-time columns lost after a sort and merge when a user's parameter values added columns | sales review, user values |
| `MERGE`'s temporary `IN=` flags leaked into outputs | sales review |
| spark-sas7bdat read every date one day early in JVMs west of UTC (silently) | cross-check against pyreadstat; now refused unless the JVM runs in UTC |
| Converter would not run on Python 3.10/3.11 (Databricks Runtime 14.3/15.4 LTS) | compile check on both versions |
| Reference oracle wrong for `UPDATE` with duplicate master keys | `robust` disagreements |
| `ctx.cut` reported "PERSIST TABLE is not supported" instead of the real error on serverless | Databricks conformance run |
| `eval` ignored parameter modules for parity cases, so a user's values were never applied | rules baseline re-run for this handoff |

### 2.6 Known gaps (you will meet these)

| Gap | Impact | Workaround today | Roadmap ([ARCHITECTURE.md](ARCHITECTURE.md) §9) |
|---|---|---|---|
| **Steps that need one global row order run in a single partition.** `LAG`/`_N_`/`RETAIN` without BY groups, and `_sas_seq` order, use windows with no partition. | Slow or failing on large tables | Not measured. **Test with production volumes (T9).** | new: partition-friendly ordering |
| No BY-sort check: SAS stops when a `MERGE … BY` input is unsorted, but the converted code carries on | Silent wrong results possible if an input is unsorted | Keep `PROC SORT` before merges (as SAS requires) | 8 |
| spark-sas7bdat is Scala 2.12 only | Works on Spark 3.x / Runtime ≤ 16.x classic compute only, and serverless cannot load JARs | Ingest on Runtime ≤ 16.x or on a workstation, then `publish`; or read with pyreadstat | rebuild for Scala 2.13 |
| Python UDFs (four runtime functions: `BEST.`, `PUT`/`INPUT` with formats not compiled to SQL, `TINV`) | Slower than SQL; failed over Connect on Free Edition | Validate those cases `--as-job` | 9: SQL `BEST.` and formats |
| Filter parameters inside `OR`/joins, column lists in PROC MEANS/FREQ, run-time table names | Marked unsupported, never guessed | LLM or manual | 6 |
| No SAS capture kit, step-by-step fault localisation or HTML review page | Reviewers read JSON evidence | `sas2dbx runs`, evidence JSON, notebook header table | 11 |
| No Asset Bundle deployment template | Manual job setup | Deploy notebooks with the Databricks CLI | 10 |
| The parameter notebook is a soft gate: `%run` executes it as code | A user with edit rights can run code with the job's identity | Use job parameters (`--param`) where users must not run code | 6: hard gate |

---

## 3. Testing it in your environment

### 3.1 Prerequisites

- [ ] A Databricks workspace with Unity Catalog, and a **sandbox** catalog/schema for sas2dbx with `CREATE SCHEMA`, `CREATE VOLUME` and `CREATE TABLE`, e.g. `sas_migration.validation`.
- [ ] Compute: serverless (notebooks and jobs). If you will read `.sas7bdat` files on Databricks, also a **classic, dedicated (single-user) cluster on Runtime 16.4 LTS or earlier** for spark-sas7bdat.
- [ ] An LLM provider approved by security: the Anthropic API (key in `.env`) or **Databricks Model Serving**, which keeps code inside the workspace. Name a budget owner.
- [ ] A workstation or CI runner with Python 3.12, Java 17 (local Spark) and the Databricks CLI. The package supports Python 3.10 and later.
- [ ] Package access: PyPI or an internal mirror for `databricks-connect`, `sqlglot` and the other dependencies. The sas2dbx wheel itself has no dependencies.

### 3.2 Set up

```bash
git clone <your repo> && cd sas-migration
uv venv && uv pip install -e '.[convert,validate,stats,llm,dev]'           # local Spark 4 environment
uv venv .venv-dbconnect --python 3.12                                     # Databricks Connect replaces pyspark
uv pip install --python .venv-dbconnect/bin/python -e '.[convert,validate,stats]' "databricks-connect~=18.3.0"
databricks auth login --host https://<your-workspace>                     # OAuth; DEFAULT profile unless --profile
export SAS2DBX_SCHEMA=sas_migration.validation                            # wheel, configuration and outputs of job runs
export SAS2DBX_RUN_LOG_VOLUME=/Volumes/sas_migration/validation/run_log   # team run log (create the volume first)
sas2dbx corpus fetch                                                      # public test corpora at pinned commits
```

- **Databricks Connect version:** pick the one for your compute. Current serverless uses 18.x on Python 3.12; for a cluster, match its Runtime. Check the [Databricks Connect requirements](https://docs.databricks.com/aws/en/dev-tools/databricks-connect/requirements).
- **CI:** use a service principal (OAuth M2M) and set `SAS2DBX_REQUESTED_BY`. The run log then names the person behind a pipeline run.

### 3.3 Test sequence

Run these in order. Each result lands in the run log (§4.7).

| # | Test | Command | Pass when |
|---|---|---|---|
| T1 | Unit tests | `.venv/bin/python -m pytest` | all pass. Tests needing optional pieces (spark-sas7bdat jars, Spark 3.5) skip |
| T2 | Conformance, local | `sas2dbx conformance` | 66/66 |
| T3 | Conformance through Databricks Connect | `.venv-dbconnect/bin/sas2dbx conformance --engine databricks` | 66/66. If exactly the four Python-UDF cases fail with `SANDBOX_STARTUP … exec format error`, it is the Databricks defect in §2.3: raise it with Databricks support and use T4 |
| T4 | Conformance as a notebook job | `… conformance --engine databricks --as-job` | 66/66; the job run link is printed |
| T5 | Worked example, local | `examples/sales_review/run.sh` (needs `.venv-spark35`, see its README) | 3 × PASS |
| T6 | Worked example on Databricks | `… publish out/ingest/sales $SAS2DBX_SCHEMA`, then `… validate cases/databricks/*.yaml --engine databricks --as-job` (first change `uc:workspace.sas2dbx_demo` in those YAML files to your schema) | 2 × PASS |
| T7 | `.sas7bdat` on classic compute | see below | rows and values match pyreadstat |
| T8 | LLM provider | `sas2dbx eval --strategies hybrid --only FRONTIER-HASH-01 --estimate`, then the same with `--yes` (add `--provider databricks:<endpoint>` for Model Serving) | PASS, with the spend shown in `sas2dbx runs`. About $0.003 on Haiku 5.5, about $0.1–$1 on Opus 5.5 |
| T9 | Scale | a converted program over production-size inputs, run as a job | finishes within your batch window. Record runtime against SAS, and look for "No Partition Defined for Window operation" warnings (§2.6) |

**T7 (not tested by the build team, so expect to adapt):** on a dedicated cluster with Runtime 16.4 LTS or earlier, install the Maven library `saurfang:spark-sas7bdat:3.0.0-s_2.12` from `https://repos.spark-packages.org/`. Clusters run in UTC, which the reader requires. Then, in a notebook:

```python
%pip install /Volumes/<catalog>/<schema>/sas2dbx_jobs/sas2dbx-0.1.0-py3-none-any.whl pyreadstat
from sas2dbx.ingest import ingest
ingest(spark, "/Volumes/<catalog>/<schema>/raw_sas", "/Volumes/<catalog>/<schema>/ingested",
       reader="spark-sas7bdat", fmt="delta")     # cross-checks against pyreadstat; writes nothing on a difference
```

Shared ("standard") access mode does not expose the JVM that the reader's UTC check needs: use dedicated access mode.

### 3.4 Your first real programs

1. **Inventory:** `sas2dbx analyze <estate root> --out reports/estate.json`. This gives:
   - readiness per program;
   - unsupported reasons, ranked;
   - the macro variables nothing defines, which are your parameter catalog: confirm each as `--param` (run-time) or `--define` (fixed);
   - the parameter modules found.
2. **Choose pilot programs** (§5.1) and, for each, **capture G1 baselines from SAS**: the input and output data sets of one to three production runs, as `.sas7bdat` or XPORT.
3. **Load the data:** ingest the inputs and expected outputs (T7, or `ingest` locally then `publish`).
4. **Write a parity case** (template in Appendix C):
   - outputs with keys (or `mode: ordered` when row order matters);
   - the ground-truth tier;
   - for parameter modules, today's values plus at least two real users' value sets.
5. **Run it:** `sas2dbx validate <case.yaml> --engine databricks --as-job`, then triage (§4.6) and get sign-off (§4.3).

### 3.5 Configuration you will touch

See Appendix B for the full list. The essentials:
- `SAS2DBX_PROFILE`: which Databricks profile to use.
- `SAS2DBX_SCHEMA`: the sandbox schema for job runs.
- `SAS2DBX_RUN_LOG_VOLUME`: the team run log.
- `SAS2DBX_CLUSTER_ID`: run on a cluster instead of serverless.
- `ANTHROPIC_API_KEY`: in `.env`, never on the command line.

### 3.6 What the build team could not test

Classic compute; shared access mode; enterprise SSO and service principals; Databricks Model Serving as the LLM provider; production data volumes; real enterprise SAS code; outbound-network restrictions on serverless; Runtime 14.3 and 15.4 LTS, which were checked by compiling on Python 3.10 and 3.11, not by running. Treat T3–T9 as first runs, not regression tests.

---

## 4. Making the evaluations useful to a human team

### 4.1 Principles

1. **The gate decides.** People review what the gate cannot see: whether the expected outputs are right, whether the tested data is representative, and whether accepted deviations are acceptable. Nobody waives a FAIL by opinion.
2. **Evidence is the record.** Every validation writes a JSON evidence file and a run-log record. Sign-offs cite them.
3. **The golden set is a contract.** Expected outputs change only through review (§4.9).
4. **Spend is visible before and after.** `--estimate` before, the run log after.

### 4.2 Roles

| Role | Does |
|---|---|
| Migration engineer | converts, writes cases, runs validations, triages FAILs, reviews LLM-written cells line by line |
| SAS subject-matter expert | owns semantics: reviews hazards and deviations, approves documented deviations |
| Data owner | supplies G1 baselines, signs off outputs for cutover |
| Platform team | workspaces, compute, packages, run-log volume and dashboard |
| Budget owner | approves LLM spend above the agreed per-run limit |

### 4.3 Per-program workflow and statuses

`Inventoried → Converted → Parity PASS → Reviewed → Parallel-run PASS (K cycles) → Signed off → Cut over`

- One pull request per program, holding the notebook, the parameter notebook, the case YAML and a link to the evidence.
- Reviewers: one engineer, plus the SME when the notebook header lists hazards or the case has deviations.
- Cells written by the LLM are marked in the notebook (step provenance) and always get line-by-line review.

### 4.4 Writing cases people can trust

- **One case per program and parameter set**, named `<domain>_<program>[_<scenario>]`. The `description` states the business purpose and what the scenario exercises, and appears in the run log.
- **Ground truth, strongest first:**
  - **G1** (SAS outputs from the same inputs) for every pilot and wave program.
  - **G3** (double programming) only where G1 is impossible. The oracle is written by someone other than the converter's reviewer, without reading the generated code (as in `cases/oracles/`).
- **Parameter modules:** parity for one value set proves nothing about the others. Cover today's values, at least two values that real users have typed, and an edge case (an empty filter, a single value).
- **Keys and modes:**
  - Use `keys:` when outputs are logically sets.
  - Use `mode: ordered` when row order is part of the output, for example after `PROC SORT` or in reports.
  - Use `columns:` to compare only some columns, and only when the description justifies the rest.
- **Accepted differences:** `ignore: [column]` on an output excludes that column from the comparison. Always pair it with a `deviations:` note (what differs, why, the approving SME and the date), which is written into the evidence. Keep them rare.

### 4.5 Reading results

- **Console:**
  - `validate` prints PASS or FAIL per case, then one line per output, e.g. `FAIL review: … columns: PCT (mismatch, 2) | sample {...}`.
  - `sas2dbx runs` prints the run log as a table with totals per person.
- **Evidence JSON** (`out/<case>.json`):
  - `comparisons[]`: rows, keys, mismatched columns, max |diff| and samples;
  - `execution`: which cell failed, and the job run link;
  - `steps[]`: conversion status, hazards, LLM reasons;
  - `evidence`: SHA-256 fingerprints, Spark version, ANSI mode, parameters.
- **Notebook header:** a table of every SAS step, with its status and the semantic hazards handled. It is the quickest review aid for SMEs.

### 4.6 Triage playbook

| Category | Signs | Action |
|---|---|---|
| Converter defect | the same SAS construct fails across programs, and the hazard is named in the notebook header | open an issue and add a **conformance case** that reproduces it (small SAS, inputs, expected output with a SAS documentation reference); fix the rules or runtime |
| Wrong expected output | only one program fails, the logic is plausible, G3 oracle | the SME re-derives the expected output from SAS documentation or a SAS run; fix the oracle (this happened with `UPDATE`) |
| Data or environment difference | dates off by one, blank vs NULL, encodings | check ingestion (UTC JVM, cross-check), and the formats in column metadata |
| Unsupported construct | the step is marked unsupported with a reason | hybrid LLM behind the gate, or manual conversion, then re-validate |
| Accepted deviation | a known, explainable difference (e.g. a SAS display-only rounding artefact) | an SME-approved `ignore:` for that output, with a `deviations:` note giving the reason |

Track counts per category weekly. A rising "converter defect" count is a rules-backlog signal; a rising "wrong expected output" count is a ground-truth quality signal.

### 4.7 Team dashboard (Databricks SQL)

Point `SAS2DBX_RUN_LOG_VOLUME` at one volume for every person and pipeline, then build an AI/BI dashboard on these queries.
- They read each record as a VARIANT, so new or missing fields never break them (Runtime 15.3+ or serverless).
- They were tested against the build team's run-log volume.

```sql
-- the run log as a view (one row per run)
CREATE OR REPLACE VIEW sas_migration.validation.sas2dbx_runs AS
SELECT rec:run_id::string AS run_id, rec:started_at::timestamp AS started_at, rec:command::string AS command,
       coalesce(rec:user.databricks_user::string, rec:user.os_user::string || '@' || rec:user.host::string) AS who,
       rec:user.on_behalf_of AS on_behalf_of, rec:outcome.status::string AS status,
       rec:outcome.cases::int AS cases, rec:outcome.passed::int AS passed, rec:outcome.failed::int AS failed,
       rec:llm.billed.calls::int AS llm_calls, rec:llm.billed.usd::double AS llm_usd, rec
FROM read_files('/Volumes/sas_migration/validation/run_log', format => 'json', singleVariantColumn => 'rec');

-- every output of every parity case: what failed, where, and by whom
SELECT r.started_at, r.who, s.value:name::string AS case_name, s.value:outcome::string AS outcome,
       o.value:dataset::string AS dataset, o.value:passed::boolean AS passed, o.value:rows::int AS rows,
       o.value:mismatched_columns AS mismatched_columns, s.value:evidence::string AS evidence
FROM sas_migration.validation.sas2dbx_runs r, LATERAL variant_explode(r.rec:scenarios) s,
     LATERAL variant_explode_outer(s.value:outputs) o
WHERE s.value:kind::string = 'parity_case';

-- latest status per case (the migration board)
SELECT case_name, max_by(outcome, started_at) AS latest_outcome, max(started_at) AS last_run, count(*) AS runs
FROM (SELECT r.started_at, s.value:name::string AS case_name, s.value:outcome::string AS outcome
      FROM sas_migration.validation.sas2dbx_runs r, LATERAL variant_explode(r.rec:scenarios) s
      WHERE s.value:kind::string = 'parity_case')
GROUP BY case_name ORDER BY last_run DESC;

-- LLM spend per person and week (at list price, the invoice is authoritative)
SELECT date_trunc('WEEK', started_at) AS week, who, sum(llm_calls) AS calls, round(sum(llm_usd), 4) AS usd
FROM sas_migration.validation.sas2dbx_runs GROUP BY ALL ORDER BY week, who;
```

Suggested tiles: cases by latest status; pass rate by week; failing outputs with links to evidence; LLM spend by person, model and week against budget; top unsupported reasons (from `analyze` runs).

### 4.8 LLM evaluations: when and how

- **When to re-run `sas2dbx eval`:** before Phase 2 with your chosen provider and model; after any model, provider or prompt change, or new hazards; and when a construct family moves between rules and LLM.
- **How:**
  - Run `--estimate` first, which is free.
  - Then `--yes`, with `--runs 3` for run-to-run agreement on families you will rely on.
  - Keep the reports under `reports/` and link them in decisions.
- **Reading it:** the per-family decision table says which families to use rules for and which to send to the LLM through hybrid behind the gate. Families flagged "extend rules/runtime or convert manually" need either a rules or runtime extension or manual conversion. **Decide per family, not per run.**
- **Cost control:**
  - Responses are cached by content, so re-runs replay at no charge.
  - The run log separates billed from replayed spend.
  - Smaller models can be cheaper per pass: Haiku passed the hash-object case at $0.003. Adopt one only after `eval` shows it passes the families you need.
- **Security:** with Databricks Model Serving (`--provider databricks:<endpoint>`), SAS code never leaves the workspace. The run log never stores code, keys or parameter values.

### 4.9 Keeping the golden set healthy

- Every converter defect becomes a conformance case before it is fixed. Every migrated program contributes its parity case.
- Changes to expected outputs or `deviations:` require SME approval in review. Never edit expected outputs to make a case pass.
- **Licensing:** do not commit third-party SAS code or data. Public corpora are fetched at pinned commits (`src/sas2dbx/corpus/manifest.yaml`), and derived variants are built from a recipe there. Enterprise SAS code and baselines stay in your governed repository and Unity Catalog.

---

## 5. Launch plan

### 5.1 Phases

**Phase 0: Platform readiness.** Owners: platform team, migration lead.
- **Do:**
  - Create the sandbox schema and the run-log volume.
  - Approve the LLM provider and budget.
  - Set up compute and its policies, including a Runtime 16.4 LTS dedicated cluster policy if `.sas7bdat` is read on Databricks.
  - Build and host the wheel.
  - Set up CI identities.
  - Run T1–T9.
- **Exit:**
  - T1–T6 pass in your workspace.
  - T7 is answered: either "works" or "ingest elsewhere".
  - T8 shows its spend in the run log.
  - T9 has a baseline runtime.
  - The dashboard (§4.7) is live.

**Phase 1: Pilot, 10–20 programs.** Owners: migration lead, SMEs, data owners.
- **Choose the programs from `analyze`:**
  - mostly fully convertible ones;
  - two or three with parameter modules (soft-gated users);
  - two or three that need the LLM;
  - one macro-heavy utility;
  - at least one with a large input.
- **For each:** G1 baselines from one to three production runs, a parity case, validation as a job, triage and sign-off.
- **Measure:**
  - engineer hours per program;
  - first-attempt and final parity;
  - LLM cost per program;
  - triage categories;
  - runtime against SAS.
- **Exit:**
  - At least 90% of the pilot programs PASS against G1 and are signed off.
  - No open high-severity converter defects.
  - This runbook has been updated with what you learned.
  - A go/no-go decision is recorded.

**Phase 2: Waves.** Owners: migration lead, domain owners.
- **Plan waves by readiness band:**
  - (A) fully converted by rules;
  - (B) a few unsupported steps, using the hybrid LLM behind the gate;
  - (C) macro-heavy work or unsupported PROCs: extend the rules or runtime first, or convert manually.
- **Parallel run:** each converted program runs on schedule next to SAS for **K production cycles** (we suggest K = 3, or one full business cycle for month-end jobs), with parity checked every cycle as a job.
- **Cutover:** after K consecutive PASS cycles and the data owner's sign-off.
- **Weekly evaluation review:** the dashboard, triage counts, golden-set growth, spend against budget.
- **Exit per wave:** every program cut over or explicitly deferred with a reason. The SAS jobs are disabled, not deleted, for a retention period.

**Phase 3: Scale and decommission.** Owners: platform team, migration lead.
- **CI:** every change to a SAS source, case or rule runs conversion and validation as a job, and blocks the merge on FAIL.
- **Deployment:** move to Asset Bundles (roadmap 10).
- **Decommission:** retire SAS per domain after its retention period, archiving the SAS code and the last G1 outputs.

### 5.2 Cutover checklist (per program)

- [ ] Parity PASS against G1 on K consecutive production cycles, for every parameter set used in practice.
- [ ] Every `ignore:` and `deviations:` entry approved by an SME, and every LLM-written cell reviewed.
- [ ] Runtime within the batch window on production volumes.
- [ ] The parameter notebook reviewed with its users: filters are Spark SQL, and NULL rules are explained in its header.
- [ ] Run-log records and evidence files linked in the sign-off.
- [ ] Rollback ready: the SAS job can be re-enabled.

### 5.3 RACI

| Activity | Migration engineer | SAS SME | Data owner | Platform | Migration lead | Budget owner |
|---|---|---|---|---|---|---|
| Environments, dashboard | C | – | – | **R/A** | C | – |
| Conversion and validation | **R** | C | – | – | A | – |
| Ground truth (G1/G3) | R | C | **A** | – | I | – |
| Accepted differences (`ignore:`) | R | **A** | C | – | I | – |
| LLM spend above limit | R | – | – | – | C | **A** |
| Cutover sign-off | R | C | **A** | I | R | – |

### 5.4 Risks

| Risk | Mitigation |
|---|---|
| The expected outputs are wrong (oracle error) | G1 baselines from SAS; independent oracles; SME review of disagreements |
| A conversion passes on tested data but fails on other data | several production cycles, parameter value sets, `robust` variants for LLM code |
| Performance at scale (global-order windows, materialising every step) | T9 in Phase 0; a large-input program in the pilot; repartitioning work before Phase 2 |
| Unsorted BY input merges silently | keep SAS's `PROC SORT`; implement roadmap 8 before wave C |
| Sending code to an external LLM | Databricks Model Serving; security review in Phase 0 |
| LLM cost overruns | `--estimate`, cache replays, model choice per family via `eval`, run-log spend tiles |
| `.sas7bdat` reading constraints (Scala 2.12, time zone) | ingest on Runtime ≤ 16.x dedicated or on a workstation; the UTC guard and cross-check are built in |
| Users edit parameter notebooks (`%run` runs code) | job parameters where users must not run code; roadmap 6 hard gate |
| Platform defects (as with Connect UDFs on Free Edition) | validate as jobs (`--as-job`), which is the deployed form anyway |

### 5.5 Metrics

Programs by status (§4.3), first-attempt and final parity rates, mean time from conversion to sign-off, triage categories per week, LLM spend per converted program, runtime against SAS, and defects found in parallel runs (should trend to zero).

### 5.6 Prioritised backlog for the enterprise team

1. Performance at scale: measure T9, then partition-friendly ordering for global-order steps.
2. BY-sort check (roadmap 8).
3. G1 capture kit and step-by-step fault localisation (roadmap 11). You have SAS, so this gives the highest value per effort.
4. In-cluster `.sas7bdat` ingestion path (T7), hardened into a documented job.
5. Re-run `eval` with your provider and model, and record the per-family decisions.
6. SQL `BEST.` and the common formats (roadmap 9).
7. HTML review page per case, generated from the evidence JSON.
8. Asset Bundle template (roadmap 10).
9. Parameter kinds the estate needs (roadmap 6), driven by `analyze`.

---

## Appendix A: Commands

| Command | Purpose |
|---|---|
| `sas2dbx analyze <dir> [--out estate.json]` | estate inventory and readiness |
| `sas2dbx convert <program.sas> [--out nb.py] [--libname LIB=uc:cat.schema] [--param NAME] [--define NAME=VALUE] [--param-module params.sas] [--llm anthropic\|databricks:<endpoint>]` | one program to a notebook (plus its parameter notebook) |
| `sas2dbx validate <case.yaml>… [--engine databricks] [--as-job --schema cat.schema] [--out-dir out]` | parity cases |
| `sas2dbx conformance [--engine databricks] [--as-job --schema cat.schema] [--export-sas DIR]` | SAS-semantics suite (`--export-sas` re-baselines it on SAS) |
| `sas2dbx eval [--strategies rules,hybrid,llm_runtime,llm_plain] [--provider …] [--only …] [--runs N] --estimate \| --yes` | LLM bake-off on the golden set |
| `sas2dbx robust [--variants N]` | synthetic-input differential test (replays cached answers, no spend) |
| `sas2dbx ingest <dir> <dest> [--reader spark-sas7bdat\|pyreadstat] [--format delta]` | SAS data sets to Delta/Parquet, cross-checked |
| `sas2dbx publish <ingest dir> <catalog.schema>` | ingested tables to Unity Catalog |
| `sas2dbx runs [--user U] [--last N] [--json]` | the run log |
| `sas2dbx corpus fetch`, `sas2dbx jars fetch` | pinned public corpora; spark-sas7bdat jars for local Spark 3.x |

## Appendix B: Environment variables

| Variable | Effect |
|---|---|
| `SAS2DBX_PROFILE` | `~/.databrickscfg` profile (default: the SDK's choice, `DEFAULT`) |
| `SAS2DBX_CLUSTER_ID` | Databricks Connect and job runs on this cluster instead of serverless |
| `SAS2DBX_SCHEMA` | default `--schema` for `--as-job` |
| `SAS2DBX_CONFIG` | path to the notebook context configuration (`sas2dbx.yaml`) when no `sas2dbx_config` job parameter is given |
| `SAS2DBX_RUN_LOG` | local run-log path (`logs/sas2dbx-runs.jsonl`), or `off` |
| `SAS2DBX_RUN_LOG_VOLUME` | Unity Catalog volume that receives one run-log file per run |
| `SAS2DBX_REQUESTED_BY` | the person a pipeline runs for (CI actors from GitHub, GitLab, Azure DevOps and Jenkins are picked up automatically) |
| `SAS2DBX_TRACE` | print per-cell timings while notebooks execute |
| `ANTHROPIC_API_KEY` | in `.env` (gitignored); never on the command line |

## Appendix C: Case templates

```yaml
# cases/<domain>/<program>_<scenario>.yaml  (parity case)
name: finance_month_end_defaults
description: >
  Month-end revenue rollup (finance/month_end.sas) with today's parameter values. Business owner: <name>.
ground_truth: G1                       # G1 SAS outputs | G3 double programming
program: ../../sas/finance/month_end.sas
param_module: ../../sas/finance/month_end_params.sas     # optional: becomes the parameter notebook
param_values: {region: [EAST, WEST], start_dt: "2026-09-01"}   # optional: a user's values (names as in the module)
include_paths: [../../sas/finance, ../../sas/macros]
libnames:
  src: uc:sas_migration.finance_ingested     # inputs in Unity Catalog
as_of: "2026-09-30"                    # freezes TODAY()/&SYSDATE9 for reproducible runs
decimals: 2
outputs:
  - {dataset: rollup, expected: ../../baselines/month_end/rollup.sas7bdat, keys: [region, account]}
  - {dataset: report, expected: ../../baselines/month_end/report.sas7bdat, mode: ordered}
  # an accepted difference: exclude the column and say why (the note is written into the evidence)
  # - {dataset: audit, expected: ..., keys: [id], ignore: [RUN_TS], deviations: ["RUN_TS is the run time (approved: <SME>, <date>)"]}
```

Conformance cases (`conformance/*.yaml`) pin down one SAS behaviour each: `id`, `title`, `hazards`, `reference` (the SAS documentation section), `sas`, `inputs` and `expected` tables. See `conformance/merge_one_to_many.yaml`.

## Appendix D: Glossary

| Term | Meaning |
|---|---|
| Parity gate | the 2-decimal output comparison that decides a conversion's success |
| G1 / G2 / G3 / G4 | ground truth: SAS outputs / SAS logs / double programming or documented semantics / synthetic edge-case data |
| Hazard | a documented SAS-vs-Spark difference, e.g. missing values sort low in SAS (`H-MISS-CMP`) |
| Covered / frontier | constructs the rules convert / constructs the rules route to the LLM or manual conversion by design |
| hybrid / llm_runtime / llm_plain | LLM for unsupported steps only / LLM writes the whole program on the sas2dbx runtime / LLM writes plain PySpark |
| Parameter module / notebook | the `%INCLUDE`d `%LET` file users edit / its Databricks counterpart, `%run` by the program |
| SAS space | the runtime's internal value model: dates as SAS numbers, blanks as `''`, row order as `_sas_seq` |
| Run log | one record per command: who, scenario, outcome, LLM spend |
