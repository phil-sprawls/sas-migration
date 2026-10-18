"""Validation inside Databricks: deployed notebooks and the conformance suite run as job runs (``--as-job``).

``--engine databricks`` runs converted code through Databricks Connect: the Spark work runs on Databricks, but the
notebook's Python runs on this machine.  So ``%run`` of the parameter notebook, job parameters, the runtime package
installed on the cluster and Python UDFs on the cluster's Python are not exercised.  ``--as-job`` runs them there:

* ``validate --as-job``: each case is deployed as users will run it.  The converted notebook and its parameter
  notebook (with the case's values typed in) go to a workspace folder, the context configuration to a Unity Catalog
  volume, and a harness notebook installs the sas2dbx wheel, ``%run``s the program unchanged and saves every output
  the case checks to Unity Catalog.  The outputs are read back and compared here, as in any other run.
* ``conformance --as-job``: the whole suite runs inside one notebook on the cluster.

One job run holds all cases (one task each) and runs on serverless compute unless ``cluster_id`` is given.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import yaml

from .runner import (CaseResult, ParityCase, case_evidence, case_param_notebook, case_steps, compare_outputs,
                     convert_case, notebook_name)

VOLUME = "sas2dbx_jobs"
SQLGLOT = "sqlglot>=30,<31"       # the converter's one dependency beyond the Databricks Runtime (conformance runs)

# The harness only %runs the program: the deployed notebooks are exactly the ones under test.
PARITY_HARNESS = """# Databricks notebook source
# MAGIC %pip install __WHEEL__

# COMMAND ----------

# MAGIC %run ./__MAIN__

# COMMAND ----------

import json
import sys

for name, table in __OUTPUTS__.items():
    ctx.to_output(ctx.get(name)).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
dbutils.notebook.exit(json.dumps({
    "python": sys.version.split()[0],
    "dbr": spark.sql("SELECT current_version().dbr_version").first()[0],
    "ansi_mode": spark.conf.get("spark.sql.ansi.enabled"),
}))
"""

CONFORMANCE_HARNESS = """# Databricks notebook source
# MAGIC %pip install __WHEEL__ __REQUIREMENTS__

# COMMAND ----------

import json
import sys

from sas2dbx.validate.conformance import load_all, run_conformance

results = []
for case in load_all("__DIR__"):
    try:
        r = run_conformance(case, spark)
        results.append({"id": case.id, "passed": r.passed, "status": r.status, "detail": r.detail[:600]})
    except Exception as e:  # noqa: BLE001 - one broken case must not hide the others
        results.append({"id": case.id, "passed": False, "status": "error", "detail": f"{type(e).__name__}: {e}"[:600]})
dbutils.notebook.exit(json.dumps({
    "results": results,
    "python": sys.version.split()[0],
    "dbr": spark.sql("SELECT current_version().dbr_version").first()[0],
    "ansi_mode": spark.conf.get("spark.sql.ansi.enabled"),
}))
"""


def build_wheel(out_dir: str) -> str:
    """Build the sas2dbx wheel from this source tree (uv, else pip)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.exists(os.path.join(root, "pyproject.toml")):
        raise FileNotFoundError("sas2dbx is not installed from a source tree: build its wheel and pass --wheel")
    cmd = (["uv", "build", "--wheel", "--out-dir", out_dir, root] if shutil.which("uv")
           else [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", out_dir, root])
    subprocess.run(cmd, check=True, capture_output=True)
    return max(glob.glob(os.path.join(out_dir, "sas2dbx-*.whl")), key=os.path.getmtime)


class JobWorkspace:
    """Where job runs keep things: notebooks in a workspace folder; the wheel, configurations and inputs in the UC
    volume ``<schema>.sas2dbx_jobs``; outputs as tables in ``<schema>``."""

    def __init__(self, spark, schema: str, *, profile: str | None = None, folder: str | None = None,
                 wheel: str | None = None, cluster_id: str | None = None, log=print) -> None:
        from databricks.sdk import WorkspaceClient

        catalog, _, sch = schema.partition(".")
        if not catalog or not sch or "." in sch:
            raise ValueError(f"--schema must be a Unity Catalog schema as catalog.schema, got {schema!r}")
        self.spark, self.schema, self.cluster_id, self.log = spark, schema, cluster_id, log
        self.w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self.folder = folder or f"/Workspace/Users/{self.w.current_user.me().user_name}/sas2dbx-validation"
        self.volume = f"/Volumes/{catalog}/{sch}/{VOLUME}"
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {schema}.{VOLUME}")
        local = wheel or build_wheel(tempfile.mkdtemp(prefix="sas2dbx-wheel-"))
        self.wheel = self.upload(local, os.path.basename(local))

    def upload(self, local_path: str, name: str) -> str:
        path = f"{self.volume}/{name}"
        with open(local_path, "rb") as fh:
            self.w.files.upload(path, fh, overwrite=True)
        return path

    def upload_text(self, name: str, text: str) -> str:
        import io

        path = f"{self.volume}/{name}"
        self.w.files.upload(path, io.BytesIO(text.encode()), overwrite=True)
        return path

    def notebook(self, path: str, source: str) -> None:
        from databricks.sdk.service.workspace import ImportFormat, Language

        self.w.workspace.mkdirs(os.path.dirname(path))
        self.w.workspace.import_(path, content=base64.b64encode(source.encode()).decode(), format=ImportFormat.SOURCE,
                                 language=Language.PYTHON, overwrite=True)

    def submit(self, name: str, tasks: dict[str, tuple[str, dict]], timeout_minutes: int = 120) -> dict[str, dict]:
        """One job run, a notebook task per entry ``key: (notebook path, parameters)``; waits for it to finish."""
        from databricks.sdk.service import jobs

        submit = [jobs.SubmitTask(task_key=key, existing_cluster_id=self.cluster_id,
                                  notebook_task=jobs.NotebookTask(notebook_path=path, base_parameters=params))
                  for key, (path, params) in tasks.items()]
        t0 = time.time()
        waiter = self.w.jobs.submit(run_name=name, tasks=submit)
        run = self.w.jobs.get_run(waiter.run_id)
        self.log(f"job run {waiter.run_id}: {run.run_page_url}")
        try:
            run = waiter.result(timeout=timedelta(minutes=timeout_minutes))
        except Exception as e:  # noqa: BLE001 - an internal error still leaves task states to report
            self.log(f"job run {waiter.run_id}: {type(e).__name__}: {e}")
            run = self.w.jobs.get_run(waiter.run_id)
        out = {}
        for t in run.tasks or []:
            o = self.w.jobs.get_run_output(t.run_id)
            ok = bool(t.state and t.state.result_state and t.state.result_state.value == "SUCCESS")
            out[t.task_key] = {
                "ok": ok, "result": o.notebook_output.result if o.notebook_output else None,
                "error": o.error or (t.state.state_message if t.state else None), "traceback": o.error_trace,
                "run_id": t.run_id, "run_page_url": run.run_page_url,
                "seconds": round(((t.end_time or 0) - (t.start_time or 0)) / 1000, 1),
            }
        self.log(f"job run {waiter.run_id} finished in {time.time() - t0:.0f}s: "
                 f"{sum(v['ok'] for v in out.values())}/{len(tasks)} tasks succeeded")
        return out


def _tag(text: str) -> str:
    return re.sub(r"\W+", "_", text.lower()).strip("_")


def run_cases_as_job(cases: list[ParityCase], spark, ws: JobWorkspace) -> list[CaseResult]:
    """Deploy each case's notebooks, run them all in one job run, compare the outputs they saved."""
    from ..runtime.context import SasContext

    plans, tasks = {}, {}
    for case in cases:
        local = {k: v for k, v in case.libnames.items() if not str(v).startswith("uc:")}
        if local:
            raise ValueError(f"{case.name}: a job reads Unity Catalog libraries only (uc:catalog.schema), not {local} "
                             "(see `sas2dbx publish`)")
        conv = convert_case(case, spark)
        tag = _tag(case.name)
        folder = f"{ws.folder}/{tag}"
        main = notebook_name(case)
        ws.notebook(f"{folder}/{main}", conv.notebook)
        params = case_param_notebook(case, conv)
        if params is not None:
            ws.notebook(f"{folder}/{conv.param_notebook_name}", params)
        config = {"libnames": case.libnames, "macro_vars": case.macro_vars, "write_outputs": False}
        if case.as_of:
            config["as_of"] = case.as_of
        config_path = ws.upload_text(f"{tag}/sas2dbx.yaml", yaml.safe_dump(config, sort_keys=False))
        outputs = {o.dataset: f"{ws.schema}.validation_{tag}_{_tag(o.dataset)}" for o in case.outputs}
        harness = (PARITY_HARNESS.replace("__WHEEL__", ws.wheel).replace("__MAIN__", main)
                   .replace("__OUTPUTS__", repr(outputs)))
        ws.notebook(f"{folder}/_parity", harness)
        tasks[tag] = (f"{folder}/_parity", {"sas2dbx_config": config_path})
        plans[tag] = (case, conv, outputs, folder)
    runs = ws.submit("sas2dbx validate", tasks)
    reader = SasContext(spark, {"_RESULT": f"uc:{ws.schema}"}, write_outputs=False)
    results = []
    for tag, (case, conv, outputs, folder) in plans.items():
        r = runs.get(tag) or {"ok": False, "error": "task did not run", "result": None}
        env = json.loads(r["result"]) if r.get("ok") and r.get("result") else {}
        execution = {"ok": r["ok"], "engine": "databricks-job", "notebooks": folder, "run_id": r.get("run_id"),
                     "run_page_url": r.get("run_page_url"), "seconds": r.get("seconds")}
        if not r["ok"]:
            execution.update(title="job task", error=r.get("error"), traceback=r.get("traceback"))
        comparisons = (compare_outputs(case, spark, lambda ds: reader.get(f"_RESULT.{outputs[ds].rsplit('.', 1)[1]}"))
                       if r["ok"] else [])
        passed = r["ok"] and bool(comparisons) and all(c.passed for c in comparisons)
        evidence = case_evidence(case, conv, conv.notebook, spark, job_environment=env, output_tables=outputs)
        results.append(CaseResult(case.name, passed, conv.stats, execution, comparisons, "", evidence, case_steps(conv)))
    return results


def run_conformance_as_job(directory: str, ws: JobWorkspace) -> dict:
    """Run the conformance suite inside a notebook on the cluster; returns the notebook's report."""
    root = f"conformance-{int(time.time())}"
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.yaml"), recursive=True)):
        ws.upload(path, f"{root}/{os.path.relpath(path, directory)}")
    harness = (CONFORMANCE_HARNESS.replace("__WHEEL__", ws.wheel).replace("__REQUIREMENTS__", f"'{SQLGLOT}'")
               .replace("__DIR__", f"{ws.volume}/{root}"))
    ws.notebook(f"{ws.folder}/_conformance", harness)
    r = ws.submit("sas2dbx conformance", {"conformance": (f"{ws.folder}/_conformance", {})})["conformance"]
    if not r["ok"]:
        raise RuntimeError(f"the conformance notebook failed: {r['error']}\n{r.get('traceback') or ''}")
    report = json.loads(r["result"])
    report["run_page_url"] = r["run_page_url"]
    return report
