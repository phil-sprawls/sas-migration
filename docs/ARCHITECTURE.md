# sas2dbx — Architecture

Convert SAS programs to Databricks (PySpark + Spark SQL) **and prove the converted code
produces the same data** — parity to 2 decimal places — before anyone relies on it.

---

## 1. Success criteria

| Criterion | Definition |
|---|---|
| **Output parity** (the acceptance gate) | Every output dataset of a converted program matches the SAS output: same columns, same row multiset, character values equal after SAS normalisation, numeric values equal to 2 decimal places, i.e. `abs(a − b) < 0.005` (the standard "correct to *d* decimal places" definition, \|a−b\| < ½·10^−d). Both-missing counts as equal. |
| **Traceability** | Every generated line maps back to a SAS file and line; every output dataset maps to a parity result; every LLM-produced block records model, prompt hash and iteration. |
| **Fail loud, never silently wrong** | Constructs the rules cannot convert *faithfully* become explicit TODO stubs that raise, or go to the LLM path. In both cases they are marked unverified until they pass parity. Nothing is approximated without a recorded note. |
| **Coverage** | DATA step ETL, PROC SQL, macro-heavy code, and STAT/reporting PROCs. The deterministic rules convert the common core, and the hybrid LLM path handles the long tail. |

## 2. Expertise applied

This design reflects four roles:

1. **Migration architect (SAS ⇄ Databricks).** Owns the method: assess, convert, validate, then cut over. Owns the semantic-hazard catalog.
2. **Compiler engineer.** Builds the SAS macro preprocessor, the parser and IR, the semantic analysis (PDV, types, lengths, ordering) and code generation with source maps.
3. **Reconciliation and test architect.** Owns the ground-truth tiers, the parity definition, the Spark-native comparator, step-level fault localisation, the conformance suite and synthetic data.
4. **Statistical programmer.** Owns PROC statistical semantics: QNTLDEF=5 percentiles, VARDEF, FREQ percent bases, Fisher exact, grouping on formatted values, and rounding.

## 3. Principles

1. **Validation is the product.** Converting 80% of SAS is easy. *Proving* it is right is the hard part, and it is where migrations fail.
2. **Semantics over syntax.** SAS and Spark disagree on missing values, row order, string padding, booleans, numeric precision, MERGE, and ANSI errors. We encode SAS semantics *explicitly*, in a tested runtime library, instead of hoping a syntax mapping is enough.
3. **Deterministic first, LLM for the long tail, validation as arbiter.** Rules are cheap, consistent and auditable. The LLM handles what the rules cannot. Both must pass the same parity gates.
4. **Order is explicit.** SAS datasets are ordered and Spark DataFrames are not. The converter tracks a *logical order* for every dataset and uses it wherever SAS semantics depend on order: BY groups, FIRST./LAST., LAG, NODUPKEY, `_N_`, positional MERGE.
5. **Correct first, idiomatic second.** Generate faithful code. Refactoring toward idiomatic code happens *under the parity harness*, so it cannot silently change results.

## 4. Pipeline

```
 SAS estate (.sas, %include trees, logs, datasets)
   │
   ├─▶ 1. ANALYZE ──────── inventory · construct census · hazards · lineage graph · waves
   │
   ├─▶ 2. MACRO EXPAND ─── %let/%macro/%if/%do/%eval/%sysfunc/&&x&i/%include
   │                       static expansion + runtime placeholders (SYMPUT / INTO:)
   │                       provenance: every output char → file:line (+ macro call stack)
   ├─▶ 3. PARSE ────────── steps: DATA · PROC SQL · PROC <x> · globals  →  IR
   │
   ├─▶ 4. SEMANTICS ────── schema propagation (type, length, format, label, PDV order)
   │                       logical-order tracking · hazard detection · support tiering
   ├─▶ 5. GENERATE ─────── Databricks notebook (.py source) + source map + conversion report
   │        │               rule-based: Tier A vectorised · Tier B window patterns · PROCs → runtime
   │        └─▶ 5b. LLM ASSIST (pluggable) for blocks marked UNSUPPORTED/PARTIAL
   │
   ├─▶ 6. VALIDATE ─────── static (compile/lint) → execute (local Spark | Databricks)
   │                       → parity vs ground truth (2 dp) → step-level localisation
   │                       → bounded repair loop for LLM blocks
   └─▶ 7. REPORT ───────── parity report (HTML+JSON) · evidence manifest (SHA-256s) · estate dashboard
```

## 5. Components

### 5.1 Analyzer (`sas2dbx.analyze`)
- Tokenises every program with the real SAS lexer, not regexes. It handles `* ;` statement comments, `/* */`, `%* ;`, quoted strings with suffixes (`'01JAN2020'd`, `'x y'n`), and DATALINES blocks.
- **Construct census**: steps by type, PROC names, DATA-step statements, functions, formats, macro features (definitions, calls, `%DO` depth, `&&` indirection, `CALL SYMPUT`, `INTO:`), and I/O datasets by libref.
- **Hazards**: every construct is matched against the hazard registry (§6), giving each program a risk profile.
- **Support tiering**: each construct is classified as `RULE` (deterministic, conformance-tested), `RULE+RT` (runtime-backed), `LLM`, or `MANUAL`. This yields an auto-conversion estimate per program.
- **Lineage**: program ⇄ dataset read/write graph across the estate, used for topological *waves*: convert producers before consumers.

### 5.2 Macro processor (`sas2dbx.macro`)
SAS macro is a text preprocessor, interleaved with execution. For static conversion we:
- Implement `%LET`, `%MACRO/%MEND` (positional and keyword parameters), `%IF/%THEN/%ELSE`, `%DO` (iterative, `%WHILE`, `%UNTIL`), `%EVAL`/`%SYSEVALF` (integer and float rules, empty operands), quoting functions (`%STR`, `%NRSTR`, `%BQUOTE`, `%SUPERQ`, `%UNQUOTE`), `%SCAN/%SUBSTR/%UPCASE/%LENGTH/%INDEX`, `%SYSFUNC` (common functions), `&&var&i` multi-pass resolution, `%GLOBAL/%LOCAL` scoping, and `%INCLUDE` via a search path.
- Quoting uses masking sentinels, the way SAS uses internal delta characters, so quoted `;`, `,` and `&` are not re-scanned.
- **Runtime boundary**: macro variables created at run time (`CALL SYMPUT[X]`, `SELECT … INTO :`) cannot be expanded statically. References in *value* positions become placeholders that the code generator turns into Python variables, including inside quoted strings and date literals. References that drive *control flow* (`%IF`, `%DO`) are flagged `RUNTIME_MACRO_CONTROL` and sent to the LLM or a manual path. A macro function applied to a run-time value, or one that cannot be evaluated, gives an opaque run-time result, never a computed guess or a blank.
- **Values from outside the program** (H-MACRO-EXTERNAL): enterprise programs get parameters from prompts, a parameter module, the autoexec or the scheduler. A macro variable that the program and its includes never define is **never frozen**: SAS would keep the text `&name`, so a conversion that copied that text, or picked a `%IF` branch, would be wrong for every real run. Steps that use it, and `%IF`/`%DO`/`%LET` logic that depends on it, stay unsupported with a reason that names it. The analyzer lists these names per program and across the estate, which gives a first draft of the parameter catalog. Each is then declared as:
  - `--param NAME`: a run-time parameter. It becomes a run-time placeholder like a `CALL SYMPUTX` value. The notebook reads it from the Databricks job or notebook parameter of the same name, so one notebook serves every value. Parity must then be shown for several values.
  - `--define NAME=VALUE`: fixed at conversion, for environment constants.
- **Parameter modules** (`--param-module`, `sas2dbx.codegen.params`): the `%INCLUDE`d file of `%LET` statements that end users edit becomes a separate parameter notebook defining `PARAMETERS`.
  - **Loading:** the program's notebook `%run`s it and calls `ctx.load_params(PARAM_SPEC, PARAMETERS)`, which checks every value before reading data. The macro processor skips the include, and each `%LET` name becomes a run-time placeholder.
  - **Kinds:** the code generator records each parameter's kind from where its placeholder lands.
    - A whole `WHERE`/subsetting-`IF`/`WHERE=` conjunct, or a single-table SQL `WHERE` conjunct, is a filter. It is applied by `ctx.param_filter` on the Delta-typed view, then mapped back to SAS-space columns.
    - An `IN` item is a value list (`ctx.param_values`).
    - A `KEEP`/`DROP`/`KEEP=`/`DROP=` item, PROC SORT `BY`, or SQL select/group/order item is a column list. Static analysis uses today's list, and the output schema becomes run-time-dependent (`known=False`).
  - **Defaults:** today's SAS filter is translated to SAS-equivalent Spark SQL. `NOT` is pushed down, because SAS comparisons are two-valued with missing as the lowest value. `x < c` becomes `(x < c OR x IS NULL)`, and `x = y` becomes `x <=> y`. The translation is checked differentially against the rules' own compiler on synthetic data.
  - **Guard:** PROC statements now raise unsupported on any run-time macro value they do not explicitly handle. Before, `class &cv` silently produced an empty CLASS list.
- **Values of the run** (H-RUNTIME-AUTOMATIC): `&SYSDATE9`, `&SYSDATE`, `&SYSDAY`, `&SYSTIME`, `&SYSUSERID`, `&SYSJOBID`, `&SYSHOSTNAME`, `&SYSPARM`, `%SYSGET(...)` and `%SYSFUNC(TODAY()|DATE()|DATETIME()|TIME()[, format])` are run-time placeholders that the context supplies when the notebook runs. They never come from the converter's clock or environment, which could hold secrets. Values derived from them by macro functions (for example `%SYSFUNC(INTNX(MONTH, %SYSFUNC(TODAY()), -1))`) are opaque and go to the LLM or a manual path.
- Output: expanded text, provenance (file, line, macro stack) per character, and diagnostics.

### 5.3 Parser and IR (`sas2dbx.lang`)
- Step segmentation follows SAS rules: `RUN;`, `QUIT;`, the next `DATA`/`PROC`, `ENDSAS;`, and the end of a DATALINES block.
- DATA step: a recursive-descent statement parser (IF/THEN/ELSE, DO forms, SELECT/WHEN, SET/MERGE/UPDATE, BY, RETAIN, ARRAY, OUTPUT, DELETE, KEEP/DROP/RENAME/LENGTH/FORMAT/LABEL/ATTRIB, CALL, sum statement…).
- Expressions: a Pratt parser with SAS precedence groups I–VII, chained comparisons (`a < b < c` ≡ `a<b and b<c`), `IN` lists and ranges, `=:` prefix comparison, `><`/`<>` MIN/MAX (but `<>` is *not-equal* inside WHERE), `OF` variable lists, special missing literals, and date/time literals.
- PROCs parse into a generic statement IR. Each PROC handler interprets its own statements.
- **Very long lines and very large constructs**: the lexer is a linear character scanner, so line length itself costs nothing. A one-line 129k-character `IN` list converts in 0.4 s. Recursion depth tracks construct size (a 1,500-branch `ELSE IF` chain nests 1,500 deep), so conversion runs on a thread with a 512 MB stack and a 100k recursion limit. A construct that still exceeds it becomes one unsupported step, never a failed program.
- PROC SQL: SAS-specific syntax is rewritten at token level (dataset options in FROM, `CALCULATED`, mnemonics `EQ/NE/…`, `'…'d` literals, `.`, `IS MISSING`, `INTO :`, `CONTAINS`, `=*`), then parsed with **sqlglot** into an AST for semantic transforms.

### 5.4 Semantic model (`sas2dbx.semantic`)
- **Schema propagation**: (name, type num/char, length, format, informat, label) flows through every step, seeded from input metadata (XPT/sas7bdat headers, Delta schemas, or a schema catalog).
- **SAS compile-time rules emulated**:
  - PDV variable order, which is how the `RETAIN`-to-reorder idiom works.
  - Character length taken from the first occurrence, including the `x='A'` truncation trap.
  - Function result lengths.
  - Automatic retention of SET/MERGE variables.
  - Num↔char auto-conversion (BEST12. / BEST32.).
- **Logical order**: each dataset carries an order spec, `[keys…, _sas_rowid]`. PROC SORT is stable (EQUALS), so its order is `[by keys, previous order]`. Hidden key copies are materialised before a sort key would be dropped or overwritten. Order-dependent operations use windows over this spec, so the converter never needs a single-partition global renumbering. Inputs carry `_sas_rowid`, the physical row order in the SAS dataset, which the loader preserves.

### 5.5 Code generation (`sas2dbx.codegen`)
Output is a Databricks notebook in `.py` source format. Each SAS step becomes a cell, with the original SAS shown above it and the `file:line` range noted. The output also includes a source-map JSON and a conversion report.

**DATA step tiers**
| Tier | Applies when | Spark strategy |
|---|---|---|
| **A: vectorised** | Row-local logic: assignments, IF/ELSE, SELECT, DELETE, subsetting IF, OUTPUT, constant-bound DO loops, arrays | Predicated if-conversion. Each statement runs under the conjunction of its enclosing conditions and becomes `F.when(pred, new).otherwise(old)`, simplified back into readable `when/otherwise` chains. Multiple OUTPUTs per iteration become `posexplode` of an array of row structs, which preserves order. |
| **B: window patterns** | FIRST./LAST., unconditional and conditional LAG/DIF, `_N_`, `END=`, sum-statement accumulators with FIRST. resets, LOCF retains | Window functions over the logical order, partitioned by BY keys where SAS semantics allow. Conditional LAG uses a window partitioned by its predicate (a faithful model of the LAG queue). |
| **C: PDV emulation** (roadmap) | Data-dependent loops and arbitrary cross-row state | A generated Python row loop over SAS-semantics Python functions, run via `applyInPandas` per BY group when state provably resets, or else a single ordered partition. The same code doubles as a **reference oracle** for synthetic-data differential tests. |
| **LLM** | Anything else | See §5.7. |

**Last checks before a step counts as converted**:
- it uses no macro variable that nothing defines;
- no run-time placeholder reached the generated code outside comments;
- the generated Python compiles. Python allows at most 200 nested parentheses, so a 2,000-term `+` or `OR` chain is flagged until chains are flattened; see the roadmap.

DATA steps with more than 300 chained projections (macro-generated steps run to thousands of statements) get `ctx.cut(df)` every 300. That materialises the frame and drops its lineage. Spark otherwise overflows the JVM stack while analysing the plan; in our test that happened at about 3,000.

**MERGE**: one-to-one and one-to-many BY merges become full outer joins with IN= flags and SAS overwrite precedence for overlapping variables. Many-to-many merges (SAS pairs rows positionally within a BY group; it does not take a Cartesian product) are joined on `(by keys, row-number-within-group)` with SAS's last-row carry-forward. A step that modifies a variable from a dataset that is not re-read gets hazard `H-MERGE-RETAIN`, which escalates it to Tier C or the LLM.

### 5.6 Runtime library (`sas2dbx.runtime`) — the only dependency of generated code
- **SAS functions as Spark SQL functions**: each function is defined once as a temporary SQL function (`sas_scan`, `sas_intnx`, `sas_compress`…). Generated SQL calls them by name, and generated PySpark calls them through thin wrappers (`S.scan(col, 2)` → `F.call_function("sas_scan", …)`). One definition means DATA steps and PROC SQL share one semantics. Formats too complex to express in SQL (BESTw.) are registered vectorised pandas UDFs.
- **Value model**:
  - Numerics are `double`, and missing is `NULL`. NaN is normalised to NULL at read, because Spark orders NaN above all numbers.
  - Character values are right-trimmed, and blank is `''`, never NULL internally. That makes SAS blank equality and joins behave correctly by construction.
  - Dates, datetimes and times stay as SAS numbers internally (days or seconds since 1960-01-01), so SAS arithmetic (`AENDT-ASTDT+1`) is exact. At read and write boundaries they convert to and from `DATE`/`TIMESTAMP` according to the variable's format.
- **SAS comparisons**: missing sorts below every number, so `x < 5` is TRUE for missing `x`, and `.=.` is TRUE. These are emitted only where operands can be NULL. Boolean-as-number (1/0) is applied where a comparison is used as a value.
- **ANSI-safe by construction**: `try_divide`, guarded casts, and validated date assembly (no reliance on parse failures), so the same code gives the same results whether `spark.sql.ansi.enabled` is on (the Spark 4 / DBR 17+ default) or off. Tests run in both modes.
- **PROCs**: SORT (stable, NODUPKEY/NODUP/DUPOUT), MEANS/SUMMARY (exact SAS statistics, `_TYPE_`/`_FREQ_`, QNTLDEF=5), FREQ (one-way and n-way, MISSING, OUT=/OUTPCT, ODS tables incl. Fisher exact), TRANSPOSE, APPEND, DATASETS, FORMAT (value and range formats → lookups, *grouping on formatted values*), CONTENTS, PRINT, IMPORT/EXPORT, RANK. REG/LOGISTIC/GLM go through statsmodels and are flagged for statistical review.
- **Context**: maps librefs to Unity Catalog `catalog.schema` (or paths). WORK is held as in-memory DataFrames, cut from their lineage after each step (`localCheckpoint`, like SAS writing WORK). The context also handles:
  - read/write boundary conversion;
  - the clock: `TODAY()` and `&SYSDATE9` come from when the run started, frozen by `as_of` for parity runs;
  - the macro variable store, including the automatic values of the run and `%SYSGET` from the run's environment;
  - run-time parameters, read from job or notebook parameters by `context_from_widgets(spark, params=[...])`. A parameter that is not supplied stays undefined, so the run fails where it is first used instead of using a blank;
  - optional per-step checkpoints for fault localisation.

### 5.7 LLM assist (`sas2dbx.llm`): hybrid, pluggable, gated
- **Providers**:
  - the Anthropic Claude API (`claude-opus-5-5` by default; streaming, a cached system prompt, explicit effort, server-side refusal fallback enabled),
  - Databricks Model Serving (Claude on Databricks, so code stays inside the workspace perimeter),
  - `none`, which is deterministic only.

  Every provider is wrapped in a content-hash response cache, which makes replays reproducible and free.
- **Units**: the LLM sees only what the rules marked unsupported. Contiguous unsupported steps form one unit, so run-time macro logic and the step that consumes it are converted together. Macro units get the *original* source, because the expanded text has no body to convert.
- **Prompt**: the system prompt is generated from the runtime library's own signatures (`S.*`, `P.*`) and the hazard registry, so it can't drift from the code that executes. The user prompt carries:
  - input schemas,
  - the outputs to write,
  - the run-time macro variables that later steps read,
  - the earlier SAS steps, as context.

  The answer is structured JSON (`code`, `outputs`, `notes`).
- **Gates**: a static guard runs first (syntax; no RDD API, `toPandas`, `spark.conf.set`, `eval`/`exec`/`open`; an import allowlist; must write through `ctx.put`). Then execute, then 2-decimal parity. A failure feeds a bounded repair loop with the gate's diff *summary* and one sample row. Row samples are only included when data governance allows (`allow_row_samples`), which matters for PHI and PII.
  - The loop is `llm.assist.convert_until_pass`, shared by `validate --llm` and `eval`.
  - The unit to blame gets back its own code and the problem: the static guard's findings, the failing cell's error, or the output differences. Rules-written cells are never sent.
  - It stops at a pass, after the allowed answers, or at the spend cap.
  - `validate` saves the last notebook and records every round in the evidence.
- **Provenance**: LLM cells are marked `provenance: llm` in the notebook and in the conversion report.
- **Bake-off** (`sas2dbx eval`, `sas2dbx.llm.eval`): the strategies `rules`, `hybrid`, `llm_runtime` and `llm_plain` run over the golden set. The golden set is conformance cases, 14 frontier cases with hand-derived expectations, and real CDISC programs.
  - The primary metric is parity pass rate, first-attempt and after repair.
  - Secondary metrics are attempts, cost, latency and run-to-run agreement on both verdict and code.
  - A per-family decision table says: rules where rules pass; the LLM (gated) where only it passes; otherwise extend the runtime or convert manually.
  - Mutation tests prove the gate rejects plausible wrong conversions: TIES=LOW instead of MEAN, an inner join where the hash lookup needs a default, missing values overwriting in UPDATE, a lookup value that isn't retained, wrong read order.
  - A first-attempt pass is the unbiased number. After-repair numbers include feedback from the expected outputs, so a held-out input set is the next safeguard.

### 5.8 Validation (`sas2dbx.validate`)

**Ground-truth tiers (strongest first)**
| Tier | Source | What it proves |
|---|---|---|
| G1 | Real SAS outputs from the same inputs (public CDISC pilot pair; data sets exported from production SAS runs; archived outputs; a capture kit is on the roadmap) | End-to-end parity |
| G2 | SAS logs (`NOTE: The data set WORK.X has N observations and M variables`, MERGE-repeats, missing-value and conversion notes) | Per-step row and column counts, plus data-dependent hazards, when only logs survive |
| G3 | Conformance suite: one case per construct and hazard with hand-derived expected output citing SAS documentation; `sas2dbx conformance --export-sas` writes each case as a SAS program for re-baselining on real SAS or Altair SLC | Construct-level semantics |
| G4 | Synthetic edge-case data (boundary values mined from code literals, missing values, duplicate BY keys, many-to-many keys, blanks, leap days, zero divisors), run as smoke plus invariants, and differentially against the Tier-C reference executor (roadmap) | Robustness and execution safety |

**Comparator** (Spark-native, so it scales to production tables):
- Normalises column-name case, trims character values, treats `NULL ≡ ''` for char and `NULL ≡ NaN` for numerics, and maps date/datetime by format.
- Checks schema, then row count, then rows. Row modes:
  - **keyed**: null-safe full outer join on keys.
  - **ordered**: positional, using SAS physical order against the logical order.
  - **unordered**: multiset match on rounded fingerprints, followed by a tolerance-aware pass for rows left unmatched.
- Reports per-column mismatch counts, max |diff| and sample mismatching rows (`allow_row_samples: false` keeps data values out of the evidence).
- Writes one **JSON evidence file** per case: the comparisons, SHA-256 of the SAS source, the generated notebook and the expected outputs, the Spark version, ANSI mode and the parity parameters. HTML review pages are on the roadmap (11).

**Fault localisation** (roadmap 11): the runtime already offers per-step checkpoints (`SasContext(checkpoint_hook=…)`). Comparing them with intermediate WORK data sets exported from SAS, to name the first step that diverges, is not built yet.

**Run log** (`sas2dbx.runlog`, `sas2dbx runs`): every command writes one record when it ends, including when it fails. The record holds:
- **who:** the OS account and host, plus the Databricks user when the run used Databricks;
- **scenario:** per program or case, the SHA-256, steps by kind, hazards, parameter names, how much converted, the description and the ground-truth tier;
- **outcome:** per case and per output;
- **LLM spend:** tokens and cost at list price. Every provider reports each answer to one ledger, and live calls (`billed`) are kept apart from cache replays (`replayed`, no charge).

Records go to local JSON Lines and, optionally, as one file per run to a Unity Catalog volume, which Databricks SQL can query team-wide with `read_files`. No secrets, parameter values or `--define` values are recorded.

## 6. Semantic hazard registry
`sas2dbx/semantic/hazards.py` is the single source of truth. The analyzer uses it to flag programs, the code generator uses it to choose a strategy, the LLM prompt uses it as context, and the conformance tests are keyed by hazard ID. The full catalog (37 hazards) is in that file. Highlights:

| ID | SAS behaviour | Naive Spark result | Handling |
|---|---|---|---|
| H-MISS-CMP | missing < every number; `.=.` true | NULL → row dropped | null-aware comparisons |
| H-SQL-JOIN-MISS | PROC SQL joins match missing keys | no match | `<=>` in ON |
| H-ORDER | datasets are ordered | arbitrary order | logical order + `_sas_rowid` |
| H-NODUPKEY | keeps first row in input order | `dropDuplicates` keeps any | window over order |
| H-UPDATE-DUPMASTER | UPDATE applies transactions only to the first master row of a BY group; duplicates pass through | group collapsed or every duplicate updated | LLM/manual + golden case + synthetic inputs (found by `sas2dbx robust`) |
| H-MACRO-EXTERNAL | prompts, parameter modules and schedulers set macro variables the program never defines | a value frozen at conversion, or a guessed `%IF` branch | never frozen: unsupported until declared `--param` (job parameter; one notebook for every value) or `--define` |
| H-RUNTIME-AUTOMATIC | `&SYSDATE9`, `&SYSPARM`, `%SYSGET`, `%SYSFUNC(TODAY())` are values of the run | the converter's date and environment baked in | read from the context when the notebook runs |
| H-RUNTIME-MACRO | `CALL SYMPUTX`/`INTO :` values substituted into later code, also inside `"…"` and `"…"d` | the placeholder text compared as data | `ctx.symget` at run time; SQL literals escaped |
| H-MERGE-M2M | positional pairing within BY group | Cartesian product | row-number join |
| H-CHAR-LEN | length from first use, truncation | unbounded strings | length tracking + truncation |
| H-CONCAT-PAD | `\|\|` keeps trailing pad | no padding | pad to declared length |
| H-ROUND | half away from zero, fuzzed | banker's or binary artefacts | `sas_round` |
| H-DIV0 | missing + NOTE | NULL (non-ANSI) or **error** (ANSI) | `try_divide` |
| H-LAG-COND | queue updates only when executed | n/a | predicate-partitioned window |
| H-REMERGE | SQL remerges aggregates | analysis error | window aggregates |
| H-PCTL | QNTLDEF=5 | `percentile_approx` approximate | exact SAS definition |
| H-FMT-GROUP | FREQ/MEANS group by *formatted* value | raw value | apply format before grouping |
| H-TRANSLATE-ARGS | `TRANSLATE(s, to, from)` | `translate(s, from, to)` | argument swap + pad |
| H-INTCK | counts boundaries crossed | `months_between` fractional | discrete/continuous implementation |

## 7. Databricks deployment
- `sas2dbx` is built as a wheel with no dependencies beyond the Databricks Runtime; only the converter needs `sqlglot`. Notebooks install it from a Unity Catalog volume (`%pip install /Volumes/…/sas2dbx-….whl`, as the validation harness does), a serverless environment or a cluster library.
- Libraries map to Unity Catalog in a `sas2dbx.yaml` that the `sas2dbx_config` job or notebook parameter names, for example `libnames: {SDTM: "uc:main.cdisc_sdtm"}`. `WORK` stays in memory.
- **Validation on Databricks** has two modes:
  - `--engine databricks` runs parity cases and the conformance suite through **Databricks Connect**. The Spark work runs on serverless compute or a named cluster; the notebook's Python runs on the client. Databricks Connect replaces `pyspark`, so it needs its own environment, with a version that matches the compute (current serverless: Connect 18.x, Python 3.12).
  - `--as-job` runs **deployed notebooks**. Each case gets its converted notebook, its parameter notebook with the case's values typed in, and a `sas2dbx.yaml`. A harness notebook installs the wheel, `%run`s the program unchanged and saves the outputs the case checks to Unity Catalog. One job run holds one task per case, on serverless compute or `--cluster-id`, and the outputs are compared from the client. `conformance --as-job` runs the whole suite in one notebook on the cluster. This mode also covers what Connect does not: `%run`, job parameters, the installed wheel, and Python UDFs on the cluster's Python.
  - **First run** (2026-10-18, Databricks Free Edition, serverless Runtime 19.9, Photon, ANSI on):
    - conformance: 66/66 as a job, 62/66 through Connect;
    - sales review: 3/3 through Connect, 2/2 as a job.

    The four Connect failures are the runtime's Python-backed functions (`BEST.`, `PUT`/`INPUT` with formats not compiled to SQL, `TINV`). The Python UDF sandbox for Connect sessions would not start (`exec format error`: its Python was built for another CPU architecture than the ARM serverless hosts). UDFs in notebooks worked.
  - Serverless compute has no `cache()`/`persist()`. `ctx.cut` uses `localCheckpoint`, and when that fails it reports the checkpoint's own error rather than the `cache()` one.
- Planned (roadmap 10): a Databricks Asset Bundle template that deploys converted notebooks as jobs, with a validation task that writes parity results to a Delta table for the migration dashboard.
- **SAS data sets** (`sas2dbx ingest`, `sas2dbx.runtime.sasfile`):
  - **Reading:** `.sas7bdat` files are read in parallel with spark-sas7bdat (saurfang, 3.0.0). XPORT files and the fallback use pyreadstat. Each spark-sas7bdat read is checked against pyreadstat, the first 5,000 rows at full precision, and ingestion stops on any difference.
  - **Tables written:** Delta-typed, meaning `DATE`/`TIMESTAMP_NTZ` by format and `NULL` for blank or missing, with SAS formats, labels and lengths as column metadata and `_sas_seq` for row order. `Catalog.harvest_tables` reads those schemas back for conversion, and `_normalize_in` restores SAS space at run time.
  - **Constraints:**
    - spark-sas7bdat is Scala 2.12 only, so Spark 3.x and Databricks Runtime ≤ 16.x classic compute (Maven `saurfang:spark-sas7bdat:3.0.0-s_2.12` from repos.spark-packages.org).
    - Runtime 17+ and Spark 4 fail with `NoClassDefFoundError: scala/Serializable`, and serverless cannot load JARs.
    - The runtime's SQL functions need Spark 4 or Databricks Runtime; open-source Spark 3.5 rejects `CREATE FUNCTION … RETURN`. So locally, ingestion runs on Spark 3.5 and programs run on Spark 4.
  - **Time zone:** spark-sas7bdat builds `java.sql.Date` values in the JVM's default time zone, so west of UTC every date is one day early. The reader refuses a non-UTC JVM, and the local Spark sessions start with `-Duser.timezone=UTC`, as Databricks clusters do.
  - **Serverless-only workspaces** (no JARs, e.g. Free Edition): ingest where spark-sas7bdat runs, then run `sas2dbx publish <ingest dir> <catalog.schema>`.
    - The Parquet files go unchanged to a UC volume, in a folder named by their content hash, and from there into Delta tables with the same column metadata.
    - Row counts and metadata are checked after the copy.
    - For `uc:` libraries, `Catalog.harvest_uc` reads those schemas for conversion. The sales-review notebook converted from Unity Catalog is byte-identical to the one converted from the local Parquet.

## 8. Public corpus (no SAS licence required)
Fetched by `sas2dbx corpus fetch`, pinned to commit SHAs, and never vendored. Licences are recorded in the manifest.
- **CDISC SDTM/ADaM pilot** (`cdisc-org/sdtm-adam-pilot-project`): `adae.sas` plus SDTM inputs plus the SAS-produced `adae.xpt`. A real G1 pair covering DATA steps, MERGE, macros, PROC SORT and informats. `at14-5-02.sas` covers PROC SQL and PROC FREQ with exact tests.
- **PhUSE scripts** (`phuse-org/phuse-scripts`, MIT): 164 SAS programs (macro-heavy, reporting, statistics) with SAS `.log`/`.lst` files (G2) and XPT data.
- **Synthetic**: generated per program from schemas and code literals, and exported as XPT, so the same inputs can later be re-baselined on real SAS.

## 9. Roadmap
1. Tier C PDV emulation and the reference oracle; differential fuzzing on synthetic data.
2. Macro lifting: turn parameter-only macros into Python functions instead of expanding them.
3. Idiomatic refactor pass (dates as `DATE`, SQL-first rewrites), allowed only when parity is preserved.
4. Wider PROC coverage (TABULATE, REPORT → AI/BI dashboards; MIXED, PHREG).
5. Altair SLC / SAS engine adapters for automatic G1 baselines.
6. Parameters beyond today's kinds:
   - run-time table names (`set lib.sales_&yyyymm`);
   - `%IF`/`%DO` on parameters as Python control flow;
   - filter parameters inside `OR` and in joins;
   - column lists in PROC MEANS/FREQ statements;
   - parity runs over a matrix of parameter values;
   - a hard gate that reads the parameter notebook's literals without executing it.
7. Flatten very long `+`/`OR`/`AND` chains (balanced for boolean logic; left fold via `aggregate` to keep SAS's left-to-right floating-point order) and merge consecutive `IF … THEN v = c;` statements into one CASE.
8. BY-group order checks. SAS stops a `MERGE`/`SET … BY` whose input is not sorted ("BY variables are not properly sorted"), but the converted code merges anyway. Add a run-time sortedness check, or flag inputs whose order the converter cannot prove. Writing the sales-review oracle exposed this gap.
9. SQL implementations of `BEST.` and the common `PUT`/`INPUT` formats, so ordinary number-to-text conversion needs no Python UDF. Python UDFs are slower everywhere, and they were the one part of the runtime that Databricks Connect could not run on Free Edition (§7).
10. A Databricks Asset Bundle template: converted notebooks deployed as jobs, plus a validation task that writes parity results to a Delta table behind the migration dashboard.
11. G1 capture kit, step-by-step fault localisation and an HTML review page per case.
    - A SAS-side add-on exports each run's inputs, outputs and WORK data sets with their metadata.
    - Case files are generated from those exports: library mapping, plus keys taken from BY and sort order.
    - Intermediate data sets are compared through `checkpoint_hook`, to name the first step that diverges.
12. Batch runner:
    - resumable: it skips unchanged cases by fingerprint;
    - runs cases as parallel Databricks job tasks, with a time limit per case;
    - isolates failures, so one failing case doesn't stop the run;
    - supports `--llm` repairs in job mode.

    CI then re-validates only what changed.
13. SAS format catalogs: read `.sas7bcat` catalogs, or their `CNTLOUT` exports, and honour `OPTIONS FMTSEARCH=`.
14. Runtime at scale:
    - partition-friendly row order for steps that need the whole table's order (BY-group partitions; two-pass running values);
    - adaptive materialisation instead of a `localCheckpoint` after every step: save only deep plans, and write large intermediate tables to Delta;
    - no driver-side collection for per-row `CALL SYMPUT`.
