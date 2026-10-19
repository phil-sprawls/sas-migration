"""sas2dbx command line.

  sas2dbx analyze  <dir>                      estate inventory, readiness, migration waves
  sas2dbx convert  <program.sas>              rules (+ optional LLM for unsupported steps) -> notebook
  sas2dbx validate <case.yaml>...             convert, run, compare outputs at 2 decimals (evidence JSON)
  sas2dbx conformance                         SAS-semantics conformance suite (or --export-sas)
  sas2dbx eval                                golden-set bake-off: rules vs hybrid vs LLM
  sas2dbx ingest | publish                    SAS data sets -> Delta/Parquet -> Unity Catalog
  sas2dbx runs                                the run log: who ran what, scenario, outcome, LLM spend
  sas2dbx corpus fetch                        download the pinned public SAS corpus
  sas2dbx schema harvest <LIB> <dir>          catalog of SAS data set attributes

Every command except `runs` appends a record to the run log (sas2dbx.runlog).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def local_spark(engine: str = "local", profile: str | None = None):
    """A Spark session for running converted code: local (bundled JDK if present) or Databricks Connect."""
    if engine == "databricks":
        try:
            from databricks.connect import DatabricksSession
        except ImportError as e:  # pragma: no cover - optional dependency
            raise SystemExit("--engine databricks needs databricks-connect (pip install 'databricks-connect')") from e
        b = DatabricksSession.builder
        b = b.profile(profile) if profile else b
        return b.serverless(True).getOrCreate() if os.environ.get("SAS2DBX_CLUSTER_ID") is None \
            else b.clusterId(os.environ["SAS2DBX_CLUSTER_ID"]).getOrCreate()
    jdk = os.path.join(os.getcwd(), ".tools", "jdk17")
    if os.path.isdir(jdk):
        os.environ.setdefault("JAVA_HOME", jdk)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.master("local[2]").appName("sas2dbx")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
         .config("spark.driver.memory", "2g").config("spark.sql.session.timeZone", "UTC")
         # a UTC JVM like Databricks clusters: spark-sas7bdat converts SAS dates in the JVM time zone
         .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
         .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC"))
    jars = spark_sas7bdat_jars()
    if jars:
        b = b.config("spark.jars", ",".join(jars))
    s = b.getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    return s


def spark_sas7bdat_jars(directory: str = os.path.join(".tools", "jars")) -> list[str]:
    """The fetched spark-sas7bdat jars (``sas2dbx jars fetch``) when this Spark can load them: the published build
    is Scala 2.12, i.e. Spark 3.x (Databricks Runtime up to 16.x); Spark 4 is Scala 2.13 only."""
    import glob

    import pyspark

    if not pyspark.__version__.startswith("3."):
        return []
    return sorted(glob.glob(os.path.join(directory, "*.jar")))


def _libnames(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        lib, _, spec = it.partition("=")
        out[lib.upper()] = spec
    return out


# ---------------------------------------------------------------------------- commands
def cmd_analyze(a) -> int:
    from . import runlog
    from .analyze.inventory import analyze_estate

    rep = analyze_estate(a.root, log=_log if a.verbose else None)
    s = rep.summary
    runlog.add_scenario({"kind": "estate", "name": a.root, "files": s["files"], "programs": s["programs"],
                         "steps": s["steps"], "auto_converted_pct": s["step_auto_conversion_pct"],
                         "programs_fully_converted_pct": s["programs_fully_converted_pct"]})
    print(f"{s['files']} files, {s['programs']} programs, {s['steps']} steps; "
          f"auto-converted steps {s['step_auto_conversion_pct']}%; programs fully converted {s['programs_fully_converted_pct']}%")
    for reason, n in s["unsupported_reasons"][:10]:
        print(f"  {n:>5}  {reason}")
    ext = s["external_inputs"]
    if ext["undefined_macro_vars"]:
        top = ", ".join(f"&{k} ({v})" for k, v in ext["undefined_macro_vars"][:12])
        print(f"{ext['programs_using_undefined_macro_vars']} programs use macro variables nothing defines "
              f"(parameters from outside? declare with --param/--define): {top}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=2, default=str)
        print(f"report: {a.out}")
    return 0


def cmd_convert(a) -> int:
    from . import runlog
    from .codegen.program import convert
    from .semantic.schema import Catalog
    from .validate.runner import build_catalog

    with open(a.program, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    catalog = Catalog.from_json(open(a.catalog).read()) if a.catalog else build_catalog(_libnames(a.libname))
    defines = dict(d.split("=", 1) for d in a.define or [])
    conv = convert(text, file=os.path.basename(a.program), catalog=catalog, macro_vars=defines,
                   params=[p for ps in a.param or [] for p in ps.split(",") if p], param_module=a.param_module,
                   include_paths=a.include or [os.path.dirname(os.path.abspath(a.program))])
    if conv.unresolved:
        _log(f"macro variables nothing defines: {', '.join('&' + n for n in conv.unresolved)} - the steps using "
             "them stay unsupported; declare each with --param NAME (run-time) or --define NAME=VALUE (fixed)")
    llm_units = None
    if a.llm:
        from .llm.assist import fill_unsupported
        from .llm.provider import make_provider

        conv, recs = fill_unsupported(conv, make_provider(a.llm, effort=a.effort), original=text, effort=a.effort, log=_log)
        cost = sum(r.cost_usd for r in recs)
        llm_units = {"units": len(recs), "accepted_by_guard": sum(r.ok for r in recs), "validated": False}
        _log(f"LLM: {sum(r.ok for r in recs)}/{len(recs)} units accepted by the static guard, ${cost:.3f}. "
             "They are NOT validated until `sas2dbx validate` passes the parity gate.")
    unsupported = conv.stats["by_status"].get("unsupported", 0)
    runlog.add_scenario(runlog.program_scenario(a.program, conv, outcome="PARTIAL" if unsupported else "CONVERTED",
                                                llm_units=llm_units, param_module=a.param_module))
    out = a.out or os.path.splitext(os.path.basename(a.program))[0] + ".py"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(conv.notebook)
    if conv.param_notebook is not None:     # %run by name from the program's notebook: keep them side by side
        pnb = os.path.join(os.path.dirname(out), conv.param_notebook_name + ".py")
        with open(pnb, "w") as fh:
            fh.write(conv.param_notebook)
        print(f"{pnb}: parameter notebook for end users ({', '.join(f'{k.lower()}={v}' for k, v in conv.param_kinds.items())})")
    st = conv.stats
    print(f"{out}: {st['steps']} steps {st['by_status']} auto-converted {st['auto_converted_pct']}%")
    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"stats": st, "steps": [{k: v for k, v in vars(s).items() if k != "code"} for s in conv.steps],
                       "diagnostics": [vars(d) | {"origin": str(d.origin)} for d in conv.diagnostics],
                       "parse_errors": conv.parse_errors}, fh, indent=2, default=str)
    return 0 if st["by_status"].get("unsupported", 0) == 0 else 2


def _llm_provider(spec: str, effort: str):
    """The LLM behind --llm, with its credentials checked before anything is spent."""
    from .llm.provider import load_dotenv, make_provider

    load_dotenv()
    if spec.startswith("anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set: put it in .env (gitignored), never on the command line")
    return make_provider(spec, effort=effort)


def cmd_validate(a) -> int:
    from . import runlog
    from .validate.runner import ParityCase, notebook_name, run_case, write_result

    if a.llm and a.as_job:
        raise SystemExit("--llm repairs run on local Spark or through Databricks Connect, not --as-job: drop --as-job "
                         "(--engine databricks still runs them on Databricks)")
    provider = _llm_provider(a.llm, a.effort) if a.llm else None
    spark = local_spark(a.engine, a.profile)
    os.makedirs(a.out_dir, exist_ok=True)
    notebooks = os.path.join(a.out_dir, "notebooks")
    cases = [ParityCase.from_yaml(path) for path in a.cases]

    def local_results():
        billed = 0.0                           # --max-usd caps the whole run, across cases
        for case in cases:
            budget = None if a.max_usd is None else a.max_usd - billed
            llm = provider if budget is None or budget > 0 else None
            if provider is not None and llm is None:
                _log(f"{case.name}: the LLM budget (--max-usd {a.max_usd}) is spent: rules only")
            res = run_case(case, spark, out_dir=notebooks, provider=llm, repairs=a.repair, effort=a.effort,
                           budget_usd=budget, log=_log)
            billed += (res.evidence.get("llm_repair") or {}).get("billed_usd", 0.0)
            yield res

    if a.as_job:
        from .validate.jobs import run_cases_as_job

        results = run_cases_as_job(cases, spark, _job_workspace(a, spark))
    else:
        results = local_results()
    failed = 0
    for case, res in zip(cases, results):
        evidence = os.path.join(a.out_dir, f"{case.name}.json")
        write_result(res, evidence)
        runlog.add_scenario(runlog.case_scenario(case, res, evidence, engine=a.engine))
        verdict = "PASS" if res.passed else "FAIL"
        print(f"{verdict}  {case.name}  ({case.ground_truth})")
        for c in res.comparisons:
            print(f"   {c.summary()}")
        if not res.execution.get("ok"):
            print(f"   execution failed in {res.execution.get('title')}: {str(res.execution.get('error'))[:400]}")
        rep = res.evidence.get("llm_repair")
        if rep:
            outcome = (f"passed on answer {rep['passed_on_answer']}" if rep["passed_on_answer"]
                       else f"still failing (stopped: {rep['stopped']})")
            print(f"   LLM: {len(rep['units'])} step(s), {rep['answers']} answer(s), {outcome}, "
                  f"{runlog.usd(rep['billed_usd'])} billed")
            if not res.passed:
                print(f"   the last attempt is saved for a person to finish: "
                      f"{os.path.join(notebooks, case.name, notebook_name(case) + '.py')}")
        failed += not res.passed
    return 1 if failed else 0


def _job_workspace(a, spark):
    from .validate.jobs import JobWorkspace

    if a.engine != "databricks":
        raise SystemExit("--as-job runs on Databricks: add --engine databricks")
    if not a.schema:
        raise SystemExit("--as-job needs --schema catalog.schema: the Unity Catalog schema for the wheel and outputs")
    return JobWorkspace(spark, a.schema, profile=a.profile, folder=a.workspace_dir, wheel=a.wheel,
                        cluster_id=a.cluster_id, log=_log)


def cmd_conformance(a) -> int:
    from . import runlog
    from .validate.conformance import export_sas, load_all, run_conformance

    cases = load_all(a.dir)
    if a.export_sas:
        os.makedirs(a.export_sas, exist_ok=True)
        for c in cases:
            with open(os.path.join(a.export_sas, f"{c.id}.sas"), "w") as fh:
                fh.write(export_sas(c))
        print(f"{len(cases)} SAS programs written to {a.export_sas}")
        return 0
    spark = local_spark(a.engine, a.profile)
    if a.as_job:
        from .validate.jobs import run_conformance_as_job

        report = run_conformance_as_job(a.dir, _job_workspace(a, spark))
        results = report["results"]
        runlog.annotate(run_page_url=report["run_page_url"], databricks_runtime=report["dbr"])
        by_id = {c.id: c for c in cases}
        for r in results:
            runlog.add_scenario(runlog.conformance_scenario(by_id[r["id"]], r["passed"], r["status"], r["detail"]))
            if not r["passed"] or a.verbose:
                print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}  {r['detail'][:300]}")
        bad = sum(not r["passed"] for r in results)
        print(f"{len(results) - bad}/{len(results)} conformance cases pass on Databricks Runtime {report['dbr']} "
              f"(Python {report['python']}, ANSI {report['ansi_mode']}): {report['run_page_url']}")
        return 1 if bad else 0
    bad = 0
    for c in cases:
        r = run_conformance(c, spark)
        runlog.add_scenario(runlog.conformance_scenario(c, r.passed, r.status, r.detail))
        bad += not r.passed
        if not r.passed or a.verbose:
            print(f"{'PASS' if r.passed else 'FAIL'}  {c.id}  {r.detail[:300]}")
    print(f"{len(cases) - bad}/{len(cases)} conformance cases pass")
    return 1 if bad else 0


def cmd_eval(a) -> int:
    from . import runlog
    from .llm.eval import STRATEGIES, estimate_cost, golden_set, render_markdown, run_eval, write_report
    from .llm.provider import DEFAULT_MODEL, CachedProvider, load_dotenv, make_provider

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies {sorted(unknown)}; choose from {', '.join(STRATEGIES)}")
    cases = golden_set(a.conformance_dir, a.cases_dir, only=a.only.split(",") if a.only else None)
    scope = f"golden set: {a.only}" if a.only else "golden set"
    needs_llm = any(s != "rules" for s in strategies)
    model = (a.provider.partition(":")[2] or DEFAULT_MODEL) if a.provider.startswith("anthropic") else a.provider
    print(f"golden set: {len(cases)} cases ({sum(c.frontier for c in cases)} frontier); strategies: {', '.join(strategies)}")
    if needs_llm:
        est = estimate_cost(cases, strategies, runs=a.runs, attempts=a.attempts, model=model)
        print(f"estimated LLM spend ({model}, effort={a.effort}, runs={a.runs}, attempts<= {a.attempts}): "
              f"~${est['usd_expected']} (range ${est['usd_low']}-${est['usd_high']}, ~{est['approx_calls']} calls; "
              "cached answers replay at no cost)")
        runlog.annotate(model=model, estimated_usd=est["usd_expected"])     # compare with what the run spends
        if a.estimate:
            runlog.add_scenario({"kind": "estimate", "name": scope, "cases": len(cases),
                                 "estimated_usd": est["usd_expected"], "range_usd": [est["usd_low"], est["usd_high"]]})
            return 0
        if not a.yes:
            print("re-run with --yes to spend it (or --estimate to only estimate)")
            return 3
        load_dotenv()
        if a.provider.startswith("anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set: put it in .env (gitignored), never on the command line")
    base = make_provider(a.provider, effort=a.effort) if needs_llm else None

    def provider_for_run(r: int):
        return base.with_namespace(f"eval-run{r}") if isinstance(base, CachedProvider) else base

    if needs_llm:
        from .llm import prompts
        from .llm.eval import prefetch

        inner = getattr(base, "inner", base)
        if hasattr(inner, "count_tokens"):     # free call: proves the key works and measures the cached prefix
            sizes = {m: inner.count_tokens(prompts.system_prompt(m), "ping") for m in ("runtime", "plain")}
            print(f"credentials OK; system prompt tokens: {sizes}")
        calls = prefetch(cases, strategies, provider_for_run, runs=a.runs, effort=a.effort, workers=a.workers, log=_log)
        print(f"first attempts ready ({calls} answers); evaluating through the parity gate")
    spark = local_spark(a.engine, a.profile)
    summary = run_eval(cases, strategies, spark, provider_for_run, runs=a.runs, attempts=a.attempts, effort=a.effort,
                       log=_log)
    summary["meta"]["provider"] = a.provider if needs_llm else None
    if isinstance(base, CachedProvider):
        summary["meta"]["spend_this_invocation"] = dict(base.spend)
    runlog.add_scenario({
        "kind": "evaluation", "name": scope, "cases": len(cases), "frontier": sum(c.frontier for c in cases),
        "runs": a.runs, "report": a.out,
        "strategies": {s: {**{k: v.get(k) for k in ("trials", "first_attempt_pass_pct", "final_pass_pct", "cost_usd",
                                                     "tokens_in", "tokens_out")},
                           "final_passes": round((v.get("final_pass_pct") or 0) * v["trials"] / 100)}
                       for s, v in summary["strategies"].items()}})
    write_report(summary, a.out)
    md = render_markdown(summary)
    with open(os.path.splitext(a.out)[0] + ".md", "w") as fh:
        fh.write(md + "\n")
    print(md)
    if isinstance(base, CachedProvider):
        sp = base.spend
        print(f"\nbilled this invocation: {runlog.usd(sp['billed_usd'])} over {sp['live_calls']} live calls "
              f"({sp['replays']} answers replayed from the cache); the table's cost column is the cost of the answers")
    print(f"\nreport: {a.out}")
    return 0


def cmd_robust(a) -> int:
    from . import runlog
    from .llm.eval import golden_set
    from .llm.provider import DEFAULT_MODEL, CachedProvider
    from .llm.robust import ReplayOnly, robustness

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    cases = golden_set(a.conformance_dir, cases_dir="", only=a.only.split(",") if a.only else None)
    provider = CachedProvider(ReplayOnly(a.model or DEFAULT_MODEL), namespace=f"eval-run{a.run}")
    spark = local_spark(a.engine, a.profile)
    res = robustness(cases, strategies, spark, provider, variants=a.variants, seed=a.seed, effort=a.effort,
                     timeout=a.timeout, log=_log)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    runlog.add_scenario({
        "kind": "robustness", "name": "synthetic input variants", "cases": len(cases), "variants": a.variants,
        "report": a.out, "strategies": {s: {k: r.get(k) for k in ("judged_variants", "agree_pct", "disagree", "candidate_error")}
                                        for s, r in res["strategies"].items()}})
    print("| strategy | judged variants | agree with reference | disagree | crashed | cases with a disagreement |")
    print("|---|---|---|---|---|---|")
    for s, r in res["strategies"].items():
        print(f"| {s} | {r['judged_variants']} | {r['agree_pct']}% | {r['disagree']} | {r['candidate_error']} | "
              f"{', '.join(r['cases_with_any_disagreement']) or '-'} |")
    print(f"\nreport: {a.out}")
    return 0


def cmd_ingest(a) -> int:
    from . import runlog
    from .ingest import ingest

    ev = ingest(local_spark(a.engine, a.profile), a.source, a.dest, reader=a.reader, fmt=a.format,
                check_rows=a.check_rows, log=_log)
    for name, t in ev["tables"].items():       # a spark-sas7bdat read is validated against pyreadstat
        check = t.get("crosscheck")
        runlog.add_scenario({"kind": "data", "name": name, "file": t["file"], "sha256": t["sha256"], "rows": t["rows"],
                             "reader": t["reader"], "table": t["table"],
                             "outcome": ("PASS" if check["passed"] else "FAIL") if check else None})
    print(f"{len(ev['tables'])} tables -> {a.dest} (evidence: {os.path.join(a.dest, 'INGEST.json')})")
    return 0


def cmd_publish(a) -> int:
    from . import runlog
    from .ingest import publish

    ev = publish(local_spark("databricks", a.profile), a.source, a.schema, volume=a.volume, profile=a.profile, log=_log)
    for name, t in ev["tables"].items():       # rows and SAS metadata are checked after the copy (else it raises)
        runlog.add_scenario({"kind": "data", "name": name, "table": t["table"], "rows": t["rows"], "sha256": t["sha256"],
                             "source": t.get("source"), "outcome": "PASS"})
    print(f"{len(ev['tables'])} tables -> {a.schema} (evidence: {os.path.join(a.source, 'PUBLISH.json')})")
    return 0


def cmd_runs(a) -> int:
    from . import runlog

    path = a.log or os.environ.get("SAS2DBX_RUN_LOG") or runlog.DEFAULT_PATH
    records = runlog.read(path)
    if a.user:
        records = [r for r in records if a.user.lower() in runlog.person(r).lower()]
    if a.last:
        records = records[-a.last:]
    if a.json:
        for r in records:
            print(json.dumps(r, default=str))
    elif records:
        print(runlog.render(records))
    else:
        print(f"no runs logged in {path}")
    return 0


def cmd_jars(a) -> int:
    from .corpus.fetch import fetch_jars

    for path in fetch_jars(a.dest, log=_log):
        print(path)
    return 0


def cmd_corpus(a) -> int:
    from .corpus.fetch import fetch

    fetch(a.dest, only=a.only.split(",") if a.only else None, log=_log)
    return 0


def cmd_schema(a) -> int:
    from .semantic.schema import Catalog

    cat = Catalog.from_json(open(a.out).read()) if os.path.exists(a.out) and not a.replace else Catalog()
    names = cat.harvest_dir(a.lib, a.dir)
    with open(a.out, "w") as fh:
        fh.write(cat.to_json())
    print(f"{len(names)} data sets from {a.dir} as {a.lib.upper()} -> {a.out}")
    return 0


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sas2dbx", description="SAS -> Databricks conversion with output-parity validation")
    sub = p.add_subparsers(dest="cmd", required=True)

    def engine(sp):
        sp.add_argument("--engine", choices=["local", "databricks"], default="local",
                        help="where converted code runs (databricks = Databricks Connect)")
        sp.add_argument("--profile", default=os.environ.get("SAS2DBX_PROFILE"),
                        help="~/.databrickscfg profile (default: the SDK's own choice, e.g. DEFAULT)")

    def as_job(sp):
        sp.add_argument("--as-job", action="store_true",
                        help="run as a Databricks job: deployed notebooks, the runtime wheel and Python on the cluster")
        sp.add_argument("--schema", default=os.environ.get("SAS2DBX_SCHEMA"),
                        help="Unity Catalog schema (catalog.schema) for the job's wheel, configuration and outputs")
        sp.add_argument("--workspace-dir", help="workspace folder for the notebooks (default: ~/sas2dbx-validation)")
        sp.add_argument("--wheel", help="sas2dbx wheel to install on the cluster (default: built from this source tree)")
        sp.add_argument("--cluster-id", default=os.environ.get("SAS2DBX_CLUSTER_ID"),
                        help="run on this cluster instead of serverless compute")

    s = sub.add_parser("analyze", help="estate inventory and readiness")
    s.add_argument("root")
    s.add_argument("--out")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("convert", help="convert one program to a Databricks notebook")
    s.add_argument("program")
    s.add_argument("--out")
    s.add_argument("--report", help="conversion report JSON")
    s.add_argument("--catalog", help="schema catalog JSON (sas2dbx schema harvest)")
    s.add_argument("--libname", action="append", help="LIB=sasfiles:/path (harvest attributes from SAS files)")
    s.add_argument("--include", action="append", help="%%INCLUDE / autocall search path")
    s.add_argument("--param", action="append", metavar="NAME[,NAME]",
                   help="macro variable supplied when the program runs (prompt, parameter module): read from the "
                        "Databricks job/notebook parameter of the same name")
    s.add_argument("--param-module", metavar="FILE",
                   help="the %%INCLUDE'd file of %%LET parameters end users edit: becomes a separate parameter "
                        "notebook (filters in Spark SQL, lists as Python lists) that the program %%runs")
    s.add_argument("--define", action="append", metavar="NAME=VALUE",
                   help="macro variable fixed at conversion time (environment constant: path, library name)")
    s.add_argument("--llm", help="convert unsupported steps with an LLM: anthropic[:model] | databricks:<endpoint>[@profile]")
    s.add_argument("--effort", default="high")
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("validate", help="run parity cases (convert, execute, compare at 2 decimals)")
    s.add_argument("cases", nargs="+")
    s.add_argument("--out-dir", default="out")
    s.add_argument("--llm", help="convert the steps the rules leave unsupported with an LLM, and send failing runs back "
                                 "to it: anthropic[:model] | databricks:<endpoint>[@profile]")
    s.add_argument("--repair", type=int, default=2, help="with --llm: repairs per case after the first answers (default 2)")
    s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--max-usd", type=float, help="with --llm: stop calling the LLM once this run has billed this much")
    engine(s)
    as_job(s)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("conformance", help="SAS-semantics conformance suite")
    s.add_argument("--dir", default="conformance")
    s.add_argument("--export-sas", metavar="DIR", help="write each case as a SAS program (re-baseline on real SAS)")
    s.add_argument("-v", "--verbose", action="store_true")
    engine(s)
    as_job(s)
    s.set_defaults(fn=cmd_conformance)

    s = sub.add_parser("eval", help="golden-set bake-off: how much rules vs LLM")
    s.add_argument("--strategies", default="rules,hybrid,llm_runtime,llm_plain")
    s.add_argument("--provider", default="anthropic", help="anthropic[:model] | databricks:<endpoint>[@profile]")
    s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--runs", type=int, default=1, help="independent samples per case (run-to-run agreement)")
    s.add_argument("--attempts", type=int, default=3, help="answers per case: 1 + repairs driven by the gate's diff")
    s.add_argument("--only", help="comma-separated case ids or families")
    s.add_argument("--conformance-dir", default="conformance")
    s.add_argument("--cases-dir", default="cases")
    s.add_argument("--out", default="reports/eval.json")
    s.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    s.add_argument("--yes", action="store_true", help="confirm the estimated LLM spend")
    s.add_argument("--workers", type=int, default=6, help="concurrent LLM requests for first attempts")
    engine(s)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("robust", help="differential test of conversions on synthetic edge-case inputs (no LLM spend)")
    s.add_argument("--strategies", default="hybrid,llm_runtime,llm_plain")
    s.add_argument("--variants", type=int, default=5, help="synthetic input variants per case")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--run", type=int, default=0, help="which eval run's cached answers to test")
    s.add_argument("--model", help="model whose cached answers to replay (default: the provider default)")
    s.add_argument("--effort", default="high")
    s.add_argument("--timeout", type=float, default=30.0, help="seconds per execution")
    s.add_argument("--only", help="comma-separated case ids or families")
    s.add_argument("--conformance-dir", default="conformance")
    s.add_argument("--out", default="reports/robust.json")
    engine(s)
    s.set_defaults(fn=cmd_robust)

    s = sub.add_parser("ingest", help="SAS data sets -> Delta/Parquet tables (spark-sas7bdat, cross-checked)")
    s.add_argument("source", help="a directory of .sas7bdat/.xpt files, or one file")
    s.add_argument("dest")
    s.add_argument("--reader", default="auto", choices=["auto", "spark-sas7bdat", "pyreadstat"])
    s.add_argument("--format", default="parquet", choices=["parquet", "delta"])
    s.add_argument("--check-rows", type=int, default=5000, help="rows compared with pyreadstat (spark-sas7bdat reads)")
    s.add_argument("--engine", default="local", choices=["local", "databricks"])
    s.add_argument("--profile")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("publish", help="ingested tables -> Unity Catalog Delta tables (when spark-sas7bdat can't run there)")
    s.add_argument("source", help="a directory written by `sas2dbx ingest`")
    s.add_argument("schema", help="Unity Catalog schema as catalog.schema (created if missing)")
    s.add_argument("--volume", default="sas2dbx_ingest", help="UC volume in that schema that keeps the uploaded files")
    s.add_argument("--profile", default=os.environ.get("SAS2DBX_PROFILE"), help="~/.databrickscfg profile")
    s.set_defaults(fn=cmd_publish)

    s = sub.add_parser("runs", help="the run log: who ran what, on which code scenario, the outcome and LLM spend")
    s.add_argument("--log", help="JSON Lines run log (default: SAS2DBX_RUN_LOG or logs/sas2dbx-runs.jsonl)")
    s.add_argument("--last", type=int, default=20, help="the last N runs (0 = all); totals cover the runs shown")
    s.add_argument("--user", help="only runs by this person (part of the Databricks user or OS account)")
    s.add_argument("--json", action="store_true", help="print the records as JSON Lines")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("jars", help="spark-sas7bdat jars for local Spark 3.x (pinned, SHA-256 checked)")
    js = s.add_subparsers(dest="jars_cmd", required=True)
    f = js.add_parser("fetch")
    f.add_argument("--dest", default=os.path.join(".tools", "jars"))
    f.set_defaults(fn=cmd_jars)

    s = sub.add_parser("corpus", help="public SAS corpus")
    cs = s.add_subparsers(dest="corpus_cmd", required=True)
    f = cs.add_parser("fetch")
    f.add_argument("--dest", default="corpus")
    f.add_argument("--only")
    f.set_defaults(fn=cmd_corpus)

    s = sub.add_parser("schema", help="schema catalog")
    ss = s.add_subparsers(dest="schema_cmd", required=True)
    h = ss.add_parser("harvest")
    h.add_argument("lib")
    h.add_argument("dir")
    h.add_argument("--out", default="catalog.json")
    h.add_argument("--replace", action="store_true")
    h.set_defaults(fn=cmd_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "runs":                       # reading the run log is not itself a run
        return a.fn(a)
    from . import runlog

    spec = str(getattr(a, "llm", None) or getattr(a, "provider", None) or "")
    profile = getattr(a, "profile", None) or (spec.rpartition("@")[2] if spec.startswith("databricks:") and "@" in spec
                                              else None)
    databricks = getattr(a, "engine", None) == "databricks" or a.cmd == "publish" or spec.startswith("databricks")
    with runlog.run(a.cmd, sys.argv[1:] if argv is None else list(argv), databricks=databricks, profile=profile) as log:
        for key in ("engine", "as_job", "schema", "llm", "provider", "strategies"):
            if getattr(a, key, None) not in (None, False):
                runlog.annotate(**{key: getattr(a, key)})
        log.exit_code = a.fn(a)
    return log.exit_code


if __name__ == "__main__":
    sys.exit(main())
