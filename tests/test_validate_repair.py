"""`validate --llm`: steps the rules leave unsupported are converted by the LLM, and a failing run goes back to it.

Two kinds of failure are fed back: code that does not run (the failing cell and its error) and code that runs but
produces different outputs (each output's differences with a sample row).  The loop stops at the first pass, after
the allowed repairs, or when the spend cap is reached; only LLM-written steps are ever sent back.  The LLM here is
scripted: these tests spend nothing."""
import json
import os

import pytest

from sas2dbx import runlog
from sas2dbx.cli import main
from sas2dbx.codegen.program import convert
from sas2dbx.llm.assist import LLM_MARK, convert_until_pass
from sas2dbx.llm.provider import ScriptedProvider
from sas2dbx.validate.conformance import convert_case, load_case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The hash-object lookup the rules route to the LLM, reading from a library (a real program would)
PROGRAM = """data out;
  if 0 then set lib.dm(keep=usubjid arm);
  if _n_ = 1 then do;
    declare hash h(dataset: 'lib.dm');
    h.defineKey('usubjid');
    h.defineData('arm');
    h.defineDone();
  end;
  set lib.ae;
  if h.find() ne 0 then arm = 'UNKNOWN';
run;
"""
RIGHT = """first = W.partitionBy("usubjid").orderBy("_sas_seq")
lk = (ctx.get("lib.dm").withColumn("_r", F.row_number().over(first)).where(F.col("_r") == 1)
      .select(F.col("usubjid").alias("_k"), F.col("arm").alias("_arm")))
ae = ctx.get("lib.ae")
df = ae.join(lk, ae["usubjid"] == lk["_k"], "left")
df = df.withColumn("arm", F.when(F.col("_k").isNull(), F.lit("UNKNOWN")).otherwise(F.col("_arm")))
ctx.put("out", df.select("usubjid", "arm", "aeterm", "_sas_seq"), meta={"arm": {"type": "char", "length": 8}})
"""
DOES_NOT_RUN = RIGHT.replace('F.col("_arm")', 'F.col("_arm_typo")')     # an unresolved column: fails when run
WRONG_OUTPUT = RIGHT.replace('"left")', '"inner")')                     # runs, but drops subjects with no match


def answer(code: str) -> str:
    return json.dumps({"code": code, "outputs": [], "notes": ""})


def hash_conv():
    return convert_case(load_case(os.path.join(ROOT, "conformance", "frontier", "hash_lookup.yaml")))


# ---------------------------------------------------------------------------- the loop itself (no Spark)
def test_loop_feeds_each_failure_back_and_stops_at_the_first_pass():
    sp = ScriptedProvider([answer("ctx.put('out', ctx.get('ae'))"), answer("ctx.put('out', ctx.get('dm'))")])
    verdicts = iter([(False, "FAIL out: rows expected 4 vs actual 3; columns: ARM (mismatch, 1)"), (True, "ok")])
    conv, out = convert_until_pass(hash_conv(), sp, lambda notebook: next(verdicts), attempts=3)
    assert out.ok and out.answers == 2 and out.gate_runs == 2 and not out.first_pass and out.stopped == ""
    assert [h["result"] for h in out.history] == ["FAIL", "PASS"] and out.history[1]["repaired_unit"] is not None
    repair_prompt = sp.calls[1][1]
    assert "failed validation" in repair_prompt and "ARM (mismatch, 1)" in repair_prompt
    assert "ctx.get('ae')" in repair_prompt                       # the LLM sees its own previous answer
    assert f"{LLM_MARK}, repaired)" in conv.notebook and "ctx.get('dm')" in conv.notebook


def test_loop_stops_after_the_allowed_answers_or_at_the_spend_cap():
    always_fails = lambda notebook: (False, "execution failed in [1] DATA out: AnalysisException: boom")  # noqa: E731
    _, out = convert_until_pass(hash_conv(), ScriptedProvider([answer("ctx.put('out', x)")] * 5), always_fails,
                                attempts=2)
    assert not out.ok and out.answers == 2 and out.stopped == "attempts"
    priced = ScriptedProvider([answer("ctx.put('out', x)")] * 5, model="claude-haiku-5-5")
    _, out = convert_until_pass(hash_conv(), priced, always_fails, attempts=5, budget_usd=1e-9)
    assert out.answers == 1 and out.stopped == "budget" and out.billed_usd > 0 and len(priced.calls) == 1


def test_a_guard_rejection_is_repaired_before_anything_runs():
    sp = ScriptedProvider([answer("x = df.rdd\nctx.put('out', x)"), answer("ctx.put('out', ctx.get('ae'))")])
    runs = []
    _, out = convert_until_pass(hash_conv(), sp, lambda nb: runs.append(nb) or (True, "ok"), attempts=3)
    assert out.ok and out.gate_runs == 1 and len(runs) == 1 and out.history[0]["result"] == "REJECTED"
    assert "RDD" in sp.calls[1][1]


def test_a_failure_in_rules_code_is_never_sent_to_the_llm():
    sp = ScriptedProvider([])
    _, out = convert_until_pass(convert("data out; x = 1; run;", file="t.sas"), sp, lambda nb: (False, "FAIL out"))
    assert not out.ok and out.stopped == "rules" and sp.calls == []


# ---------------------------------------------------------------------------- `sas2dbx validate --llm` (Spark)
@pytest.fixture
def hash_case(tmp_path, monkeypatch):
    import pandas as pd
    import pyreadstat

    monkeypatch.setenv("SAS2DBX_RUN_LOG", str(tmp_path / "runs.jsonl"))
    monkeypatch.delenv("SAS2DBX_RUN_LOG_VOLUME", raising=False)
    pyreadstat.write_xport(pd.DataFrame({"USUBJID": ["01", "02"], "ARM": ["PLACEBO", "DRUG"]}),
                           str(tmp_path / "dm.xpt"), table_name="DM")
    pyreadstat.write_xport(pd.DataFrame({"USUBJID": ["01", "03", "02", "01"],
                                         "AETERM": ["HEADACHE", "NAUSEA", "RASH", "FEVER"]}),
                           str(tmp_path / "ae.xpt"), table_name="AE")
    pd.DataFrame({"USUBJID": ["01", "03", "02", "01"], "ARM": ["PLACEBO", "UNKNOWN", "DRUG", "PLACEBO"],
                  "AETERM": ["HEADACHE", "NAUSEA", "RASH", "FEVER"]}).to_parquet(tmp_path / "out.parquet")
    (tmp_path / "ae_arm.sas").write_text(PROGRAM)

    def case(name: str) -> str:
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"name: {name}\ndescription: adverse events with the subject's arm (hash lookup)\n"
                        f"ground_truth: G3\nprogram: ae_arm.sas\nlibnames: {{lib: 'sasfiles:{tmp_path}'}}\n"
                        "outputs:\n  - {dataset: out, expected: out.parquet, mode: ordered}\n")
        return str(path)
    return case


def _use_llm(monkeypatch, sp):
    monkeypatch.setattr("sas2dbx.cli._llm_provider", lambda spec, effort: sp)


@pytest.mark.spark
def test_validate_repairs_code_that_does_not_run_then_wrong_output(spark, hash_case, tmp_path, monkeypatch, capsys):
    sp = ScriptedProvider([answer(DOES_NOT_RUN), answer(WRONG_OUTPUT), answer(RIGHT)], model="claude-haiku-5-5")
    _use_llm(monkeypatch, sp)
    out = tmp_path / "out"
    assert main(["validate", hash_case("ae_arm"), "--llm", "anthropic:claude-haiku-5-5", "--repair", "2",
                 "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "PASS  ae_arm" in printed and "LLM: 1 step(s), 3 answer(s), passed on answer 3" in printed

    rep = json.loads((out / "ae_arm.json").read_text())["evidence"]["llm_repair"]
    assert (rep["answers"], rep["passed_on_answer"], rep["first_pass"], rep["llm_calls"]) == (3, 3, False, 3)
    assert [h["result"] for h in rep["history"]] == ["FAIL", "FAIL", "PASS"]
    assert rep["history"][0]["problem"].startswith("execution failed in")          # 1: the code did not run
    assert "rows expected 4 vs actual 3" in rep["history"][1]["problem"]          # 2: it ran, outputs differed
    assert "execution failed in" in sp.calls[1][1] and "_arm_typo" in sp.calls[1][1]
    assert "rows expected 4 vs actual 3" in sp.calls[2][1]

    saved = (out / "notebooks" / "ae_arm" / "ae_arm.py").read_text()                 # the notebook that passed
    assert f"{LLM_MARK}, repaired)" in saved and '"left")' in saved and "_arm_typo" not in saved

    record = runlog.read()[-1]
    assert record["llm"]["billed"]["calls"] == 3 and record["llm"]["billed"]["usd"] > 0
    s = record["scenarios"][0]
    assert s["outcome"] == "PASS" and s["llm_repair"]["answers"] == 3 and s["llm_repair"]["passed_on_answer"] == 3
    assert "history" not in s["llm_repair"]          # failures can quote data: they stay in the evidence file


@pytest.mark.spark
def test_validate_saves_the_last_attempt_when_repairs_run_out(spark, hash_case, tmp_path, monkeypatch, capsys):
    sp = ScriptedProvider([answer(WRONG_OUTPUT)] * 2)
    _use_llm(monkeypatch, sp)
    out = tmp_path / "out"
    assert main(["validate", hash_case("ae_arm"), "--llm", "anthropic", "--repair", "1", "--out-dir", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "still failing (stopped: attempts)" in printed and "saved for a person to finish" in printed
    assert '"inner")' in (out / "notebooks" / "ae_arm" / "ae_arm.py").read_text()


@pytest.mark.spark
def test_spend_cap_covers_the_whole_run(spark, hash_case, tmp_path, monkeypatch, capsys):
    sp = ScriptedProvider([answer(WRONG_OUTPUT)] * 5, model="claude-haiku-5-5")
    _use_llm(monkeypatch, sp)
    code = main(["validate", hash_case("first"), hash_case("second"), "--llm", "anthropic", "--max-usd", "0.000001",
                 "--out-dir", str(tmp_path / "out")])
    assert code == 1 and len(sp.calls) == 1           # one answer for the first case, then the cap stops everything
    first = json.loads((tmp_path / "out" / "first.json").read_text())["evidence"]["llm_repair"]
    second = json.loads((tmp_path / "out" / "second.json").read_text())["evidence"]
    assert first["stopped"] == "budget" and "llm_repair" not in second


@pytest.mark.spark
def test_validate_never_sends_rules_failures_to_the_llm(spark, tmp_path, monkeypatch):
    import pandas as pd
    import pyreadstat

    monkeypatch.setenv("SAS2DBX_RUN_LOG", "off")
    pyreadstat.write_xport(pd.DataFrame({"X": [1.0, 2.0]}), str(tmp_path / "inp.xpt"), table_name="INP")
    (tmp_path / "double.sas").write_text("data out; set lib.inp; y = x * 2; run;\n")
    pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 5.0]}).to_parquet(tmp_path / "bad.parquet")
    (tmp_path / "case.yaml").write_text(f"name: doubles\nprogram: double.sas\nlibnames: {{lib: 'sasfiles:{tmp_path}'}}\n"
                                        "outputs:\n  - {dataset: out, expected: bad.parquet, mode: ordered}\n")
    sp = ScriptedProvider([])
    _use_llm(monkeypatch, sp)
    assert main(["validate", str(tmp_path / "case.yaml"), "--llm", "anthropic", "--out-dir", str(tmp_path / "o")]) == 1
    assert sp.calls == []
    assert "llm_repair" not in json.loads((tmp_path / "o" / "doubles.json").read_text())["evidence"]


def test_llm_repairs_do_not_combine_with_job_mode(monkeypatch):
    monkeypatch.setenv("SAS2DBX_RUN_LOG", "off")
    with pytest.raises(SystemExit, match="not --as-job"):
        main(["validate", "x.yaml", "--llm", "anthropic", "--as-job"])
