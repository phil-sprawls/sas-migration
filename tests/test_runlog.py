"""Run log: every command records who ran it, the code scenario, the validation outcome and the LLM token spend."""
import json

import pytest

from sas2dbx import runlog
from sas2dbx.cli import main
from sas2dbx.llm.provider import CachedProvider, ScriptedProvider


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setenv("SAS2DBX_RUN_LOG", str(path))
    monkeypatch.delenv("SAS2DBX_RUN_LOG_VOLUME", raising=False)
    return path


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_define_values_never_reach_the_log():
    args = ["convert", "p.sas", "--define", "pwd=hunter2", "--define=user=me", "--def", "k=v", "--dir", "x"]
    assert runlog.redact(args) == ["convert", "p.sas", "--define", "pwd=<redacted>", "--define=user=<redacted>",
                                   "--def", "k=<redacted>", "--dir", "x"]


def test_token_ledger_keeps_billed_replayed_and_unpriced_apart(log_path, tmp_path):
    priced = CachedProvider(ScriptedProvider(["a"], model="claude-opus-5-5"), cache_dir=str(tmp_path / "cache"))
    unpriced = ScriptedProvider(["b"], model="my-serving-endpoint")     # e.g. Databricks Model Serving: DBUs
    with runlog.run("eval", ["eval"]) as log:
        first = priced.complete("system prompt", "convert this")         # live: billed
        again = priced.complete("system prompt", "convert this")         # same prompt: replayed from the cache
        other = unpriced.complete("system prompt", "convert that too")   # live, tokens counted, no list price
    led = log.record["llm"]
    assert not first.cached and again.cached and first.cost_usd > 0 and other.usage["input_tokens"] > 0
    assert led["billed"]["calls"] == 2 and led["replayed"]["calls"] == 1 and led["unpriced_calls"] == 1
    assert led["billed"]["input_tokens"] == first.usage["input_tokens"] + other.usage["input_tokens"]
    assert led["billed"]["usd"] == pytest.approx(first.cost_usd) and led["replayed"]["usd"] == pytest.approx(first.cost_usd)
    assert led["models"] == ["claude-opus-5-5", "my-serving-endpoint"]
    assert _records(log_path)[0]["llm"] == led


def test_a_conversion_is_logged_with_who_and_what_the_program_contains(log_path, tmp_path):
    prog = tmp_path / "report.sas"
    prog.write_text("%let region = EAST;\ndata out;\n  set lib.sales;\n  where region = \"&region\";\n  t = amt * 2;\nrun;\n"
                    "proc sort data=out; by t; run;\n")
    code = main(["convert", str(prog), "--out", str(tmp_path / "report.py"), "--define", "pwd=hunter2"])
    assert code == 0
    r = _records(log_path)[-1]
    assert "hunter2" not in log_path.read_text()
    assert r["command"] == "convert" and r["user"]["os_user"] and r["user"]["host"]
    s = r["scenarios"][0]
    assert s["kind"] == "program" and s["lines"] == 7 and s["steps"] == {"DATA": 1, "PROC SORT": 1}
    assert len(s["program_sha256"]) == 64 and s["outcome"] == "CONVERTED"
    assert r["outcome"] == {"status": "completed", "exit_code": 0, "cases": 0, "passed": 0, "failed": 0}
    assert r["llm"]["billed"]["calls"] == 0


def test_pipeline_runs_name_who_started_them(log_path, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_ACTOR", "alice")
    with runlog.run("convert", ["convert", "x.sas"]) as log:
        pass
    assert log.record["user"]["on_behalf_of"] == {"github_actor": "alice"}
    assert runlog.person(log.record).endswith(" for alice")


def test_a_run_that_crashes_is_logged_as_an_error(log_path):
    with pytest.raises(FileNotFoundError):
        main(["convert", "no/such/program.sas"])
    r = _records(log_path)[-1]
    assert r["outcome"]["status"] == "error" and r["outcome"]["exit_code"] == 1
    assert "FileNotFoundError" in r["outcome"]["error"]


@pytest.mark.spark
def test_a_refused_run_is_logged_with_its_reason(spark, log_path):
    import os

    suite = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conformance", "params")
    with pytest.raises(SystemExit):
        main(["conformance", "--dir", suite, "--as-job"])               # --as-job needs --engine databricks
    r = _records(log_path)[-1]
    assert r["outcome"]["status"] == "error" and r["outcome"]["exit_code"] == 1
    assert "--as-job runs on Databricks" in r["outcome"]["error"] and r["context"]["as_job"] is True


@pytest.mark.spark
def test_validation_outcome_per_case_and_output(spark, log_path, tmp_path, capsys):
    import pandas as pd
    import pyreadstat

    pyreadstat.write_xport(pd.DataFrame({"X": [1.0, 2.5, None]}), str(tmp_path / "inp.xpt"), table_name="INP")
    (tmp_path / "double.sas").write_text("data out; set lib.inp; y = x * 2; run;\n")
    pd.DataFrame({"X": [1.0, 2.5, None], "Y": [2.0, 5.0, None]}).to_parquet(tmp_path / "good.parquet")
    pd.DataFrame({"X": [1.0, 2.5, None], "Y": [2.0, 5.1, None]}).to_parquet(tmp_path / "bad.parquet")
    for name, expected in (("doubles", "good"), ("doubles_wrong_key", "bad")):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: {name}\ndescription: doubling a number, missing stays missing\nground_truth: G3\n"
            f"program: double.sas\nlibnames: {{lib: 'sasfiles:{tmp_path}'}}\n"
            f"outputs:\n  - {{dataset: out, expected: {expected}.parquet, mode: ordered}}\n")
    code = main(["validate", str(tmp_path / "doubles.yaml"), str(tmp_path / "doubles_wrong_key.yaml"),
                 "--out-dir", str(tmp_path / "out")])
    assert code == 1
    r = _records(log_path)[-1]
    assert r["outcome"] == {"status": "failed", "exit_code": 1, "cases": 2, "passed": 1, "failed": 1}
    good, bad = r["scenarios"]
    assert (good["outcome"], bad["outcome"]) == ("PASS", "FAIL")
    assert good["description"] == "doubling a number, missing stays missing" and good["ground_truth"] == "G3"
    assert bad["outputs"] == [{"dataset": "out", "passed": False, "rows": 3, "expected_rows": 3,
                               "mismatched_columns": ["Y"]}]
    assert good["evidence"].endswith("doubles.json") and good["engine"] == "local"
    assert "run log:" in capsys.readouterr().err

    assert main(["runs", "--log", str(log_path)]) == 0
    table = capsys.readouterr().out
    assert "validate" in table and "1/2 FAIL" in table and "$0.00" in table
