"""LLM assist + golden-set bake-off, exercised with scripted answers (no API spend)."""
import json
import os

import pytest

from sas2dbx.llm import prompts
from sas2dbx.llm.assist import scan_io, units
from sas2dbx.llm.eval import (PREAMBLE, Trial, convert_golden, decide, estimate_cost, gate, golden_set, render_markdown,
                              run_trial, summarize)
from sas2dbx.llm.guard import check_code
from sas2dbx.llm.provider import CachedProvider, Completion, ScriptedProvider

from llm_fixtures.frontier_solutions import UNIT, WHOLE

ROOT = os.path.join(os.path.dirname(__file__), "..")
GOLDEN = {c.id: c for c in golden_set(os.path.join(ROOT, "conformance"), cases_dir="")}


def answer(code: str, notes: str = "") -> str:
    return json.dumps({"code": code, "outputs": [], "notes": notes})


# ---------------------------------------------------------------------------- static parts
def test_golden_set_has_frontier_and_covered_cases():
    frontier = [c for c in GOLDEN.values() if c.frontier]
    assert len(frontier) >= 14 and len(GOLDEN) - len(frontier) >= 40
    assert {"hash-object", "call-execute", "scl-metadata", "stat-proc"} <= {c.family for c in frontier}


def test_system_prompts_are_generated_from_the_runtime():
    rt, plain = prompts.system_prompt("runtime"), prompts.system_prompt("plain")
    assert "`S.lt(" in rt and "`P.merge(" in rt and "### S" not in plain
    for text in (rt, plain):
        assert "ctx.put" in text and "Missing values in comparisons" in text or "missing" in text.lower()


def test_guard():
    assert check_code("df = ctx.get('a')\nctx.put('b', df)") == []
    bad = check_code("import os\ndf = ctx.get('a').rdd\nspark.conf.set('x', 1)\nctx.put('b', df)")
    assert any("import of os" in p for p in bad) and any("RDD" in p for p in bad) and any("conf.set" in p for p in bad)
    assert check_code("def f(:\n  pass")[0].startswith("syntax error")
    assert "never writes" in check_code("x = 1")[0]


def test_scan_io():
    r, w = scan_io("data out(keep=a) x.y; set lib.in1 in2(where=(a>1)); if _n_=1 then set stats; run;")
    assert w == ["WORK.OUT", "X.Y"] and r == ["LIB.IN1", "WORK.IN2", "WORK.STATS"]
    r, w = scan_io("proc univariate data=lab noprint; var x; output out=pct mean=m; run;")
    assert r == ["WORK.LAB"] and w == ["WORK.PCT"]
    r, w = scan_io("proc sql; create table t as select * from a join b on a.k=b.k; insert into u values (1); quit;")
    assert set(r) == {"WORK.A", "WORK.B"} and w == ["WORK.T", "WORK.U"]


def test_units_group_runtime_macro_logic_with_its_consumer():
    conv = convert_golden(GOLDEN["FRONTIER-SCL-01"])
    us = units(conv)
    assert len(us) == 1 and [s.kind for s in us[0].steps] == ["MACRO", "DATA"] and us[0].needs_source


def test_parse_errors_are_visible_unsupported_steps():
    conv = convert_golden(GOLDEN["FRONTIER-HASH-01"])
    assert [s.kind for s in conv.steps if s.kind != "GLOBAL"] == ["UNPARSED"]
    assert conv.stats["auto_converted_pct"] == 0.0


def test_runtime_sysfunc_never_picks_a_branch_statically():
    conv = convert_golden(GOLDEN["FRONTIER-SCL-01"])
    assert all(s.status == "unsupported" for s in conv.steps if s.kind != "GLOBAL")
    assert "flag = 'Y'" not in conv.notebook.split("# COMMAND")[-1].replace('"', "'").split("# ----")[-1]


def test_cached_provider_namespaces(tmp_path):
    inner = ScriptedProvider([answer("a"), answer("b"), answer("c")])
    c0 = CachedProvider(inner, str(tmp_path))
    assert c0.complete("s", "u").text == answer("a")
    assert c0.complete("s", "u").cached
    assert c0.with_namespace("run1").complete("s", "u").text == answer("b")   # independent sample


def test_decide_and_summarize():
    assert decide({"rules": 100.0, "hybrid": 100.0}, ["rules", "hybrid"]) == "rules"
    assert decide({"rules": 0.0, "hybrid": 100.0}, ["rules", "hybrid"]).startswith("LLM via hybrid")
    assert decide({"rules": 0.0, "hybrid": 50.0}, ["rules", "hybrid"]).startswith("unreliable")
    trials = [Trial("rules", "a", "f", False, 0, True, True), Trial("hybrid", "b", "g", True, 0, False, True, 2),
              Trial("hybrid", "b", "g", True, 1, True, True, 1)]
    s = summarize(trials)
    assert s["strategies"]["hybrid"]["first_attempt_pass_pct"] == 50.0
    assert s["strategies"]["hybrid"]["final_pass_pct"] == 100.0
    assert s["strategies"]["hybrid"]["verdict_agreement_pct"] == 100.0
    assert s["families"]["g"]["decision"].startswith("LLM via hybrid")
    assert "| hybrid |" in render_markdown(s)


def test_estimate_cost_is_offline():
    cases = [GOLDEN["FRONTIER-RANK-01"], GOLDEN["H-MERGE-OVERLAP-01"]]
    est = estimate_cost(cases, ["rules", "hybrid", "llm_runtime"], runs=2, attempts=3, model="claude-opus-5-5")
    assert 0 < est["usd_low"] <= est["usd_expected"] <= est["usd_high"]
    assert est["system_prompt_tokens"]["runtime"] > est["system_prompt_tokens"]["plain"]


def test_unit_prompt_for_macro_logic_carries_the_original_source():
    sp = ScriptedProvider([answer(UNIT["FRONTIER-SCL-01"])])
    from sas2dbx.llm.assist import fill_unsupported

    case = GOLDEN["FRONTIER-SCL-01"]
    conv, recs = fill_unsupported(convert_golden(case), sp, original=case.sas)
    assert recs[0].ok and "%sysfunc(open(&ds))" in sp.calls[0][1]
    assert "provenance: llm" in conv.notebook and "NotImplementedError" not in conv.notebook


# ---------------------------------------------------------------------------- through the gate (Spark)
@pytest.mark.spark
def test_rules_strategy(spark):
    t = run_trial("rules", GOLDEN["H-MERGE-OVERLAP-01"], spark, None)
    assert t.first_pass and t.final_pass and t.llm_calls == 0
    t = run_trial("rules", GOLDEN["FRONTIER-RANK-01"], spark, None)
    assert not t.final_pass and "not converted by the rules" in t.error and t.gate_runs == 0


@pytest.mark.spark
@pytest.mark.parametrize("case_id", sorted(UNIT))
def test_frontier_reference_solutions_pass_the_gate(spark, case_id):
    case = GOLDEN[case_id]
    t = run_trial("hybrid", case, spark, ScriptedProvider([answer(UNIT[case_id])]))
    assert t.first_pass and t.final_pass and t.llm_calls == t.llm_units, t.error
    ok, detail = gate(case, PREAMBLE + "\n# COMMAND ----------\n\n" + WHOLE[case_id], spark)
    assert ok, detail


WRONG = {   # plausible conversions with a classic semantic slip - the gate must reject every one
    "FRONTIER-RANK-01": UNIT["FRONTIER-RANK-01"].replace(" + (F.count(F.lit(1)).over(ties) - 1) / 2.0", ""),    # TIES=LOW
    "FRONTIER-HASH-01": UNIT["FRONTIER-HASH-01"].replace('"left")', '"inner")'),                              # drops misses
    "FRONTIER-UPDATE-01": UNIT["FRONTIER-UPDATE-01"].replace("F.last(F.col(c), ignorenulls=True)", "F.last(F.col(c))"),
    "FRONTIER-MULTISET-01": UNIT["FRONTIER-MULTISET-01"].replace(
        'df = lab.withColumn("mu", F.lit(mu).cast("double"))',
        'df = lab.withColumn("mu", F.when(F.col("_sas_seq") == lab.agg(F.min("_sas_seq")).first()[0], F.lit(mu)).cast("double"))'),
    "FRONTIER-POINT-01": UNIT["FRONTIER-POINT-01"].replace(".desc()", ".asc()"),
}


@pytest.mark.spark
@pytest.mark.parametrize("case_id", sorted(WRONG))
def test_gate_rejects_plausible_wrong_conversions(spark, case_id):
    assert WRONG[case_id] != UNIT[case_id]
    t = run_trial("hybrid", GOLDEN[case_id], spark, ScriptedProvider([answer(WRONG[case_id])]), attempts=1)
    assert not t.final_pass and t.error


@pytest.mark.spark
def test_repair_loop_feeds_the_gate_diff_back(spark):
    sp = ScriptedProvider([answer(WRONG["FRONTIER-RANK-01"]), answer(UNIT["FRONTIER-RANK-01"])])
    t = run_trial("hybrid", GOLDEN["FRONTIER-RANK-01"], spark, sp, attempts=3)
    assert not t.first_pass and t.final_pass and t.attempts == 2 and t.llm_calls == 2
    repair_user = sp.calls[1][1]
    assert "failed validation" in repair_user and "RNK" in repair_user.upper()


@pytest.mark.spark
def test_guard_rejection_counts_as_a_failed_attempt(spark):
    bad = UNIT["FRONTIER-RANK-01"] + "\nx = df.rdd\n"
    sp = ScriptedProvider([answer(bad), answer(UNIT["FRONTIER-RANK-01"])])
    t = run_trial("hybrid", GOLDEN["FRONTIER-RANK-01"], spark, sp, attempts=3)
    assert not t.first_pass and t.final_pass and t.gate_runs == 1
    assert "RDD" in sp.calls[1][1]


@pytest.mark.spark
def test_whole_program_strategies(spark):
    case = GOLDEN["FRONTIER-EXECUTE-01"]
    t = run_trial("llm_runtime", case, spark, ScriptedProvider([answer(WHOLE[case.id])]))
    assert t.first_pass and t.final_pass
    t = run_trial("llm_plain", case, spark, ScriptedProvider(["not json at all {", answer(WHOLE[case.id])]), attempts=2)
    assert not t.first_pass and t.final_pass and t.attempts == 2


@pytest.mark.spark
def test_refusal_is_a_failed_trial(spark):
    class Refuser(ScriptedProvider):
        def complete(self, system, user, **kw):
            self.calls.append((system, user))
            return Completion("", "m", "refusal", {}, refused=True)

    t = run_trial("llm_runtime", GOLDEN["FRONTIER-RANK-01"], spark, Refuser(), attempts=2)
    assert not t.final_pass and "refusal" in t.error and t.llm_calls == 2


def test_prefetch_warms_the_cache_so_the_evaluation_replays(tmp_path):
    from sas2dbx.llm.eval import prefetch

    case = GOLDEN["FRONTIER-RANK-01"]
    inner = ScriptedProvider([answer(UNIT[case.id]), answer(WHOLE[case.id])])
    cached = CachedProvider(inner, str(tmp_path))
    calls = prefetch([case], ["rules", "hybrid", "llm_runtime"], lambda r: cached, workers=2)
    assert calls == 2 and len(inner.calls) == 2
    from sas2dbx.llm.assist import fill_unsupported

    conv, recs = fill_unsupported(convert_golden(case), cached, original=case.sas)
    assert recs[0].ok and len(inner.calls) == 2          # served from the cache


def test_provider_errors_abort_instead_of_scoring():
    from sas2dbx.llm.provider import ProviderError

    class Down(ScriptedProvider):
        def complete(self, system, user, **kw):
            raise ProviderError("Anthropic API: AuthenticationError")

    with pytest.raises(ProviderError):
        run_trial("llm_runtime", GOLDEN["FRONTIER-RANK-01"], None, Down())


# ---------------------------------------------------------------------------- synthetic inputs + differential testing
def test_fuzz_keeps_by_variables_and_is_reproducible():
    from sas2dbx.validate.fuzz import fuzz_tables, protected_columns

    case = GOLDEN["FRONTIER-RETAIN-01"]
    assert "USUBJID" in protected_columns(case.sas)
    a = fuzz_tables(case.payload.inputs, case.sas, seed=7)
    b = fuzz_tables(case.payload.inputs, case.sas, seed=7)
    assert [t.rows for t in a.values()] == [t.rows for t in b.values()]
    ids = [r[0] for r in a["ex"].rows]
    assert ids == sorted(ids)                     # BY groups stay sorted and intact
    for t in a.values():
        for r in t.rows:
            for v, col in zip(r, t.columns):
                assert v is None or (isinstance(v, str) == (col.type == "char"))


HARDCODED_RANK = '''
df = ctx.get("scores")
df = df.withColumn("rnk", F.when(F.col("score") == 90, 1.5).when(F.col("score") == 80, 3.0)
                   .when(F.col("score") == 70, 1.0).otherwise(F.lit(None)).cast("double"))
ctx.put("ranked", df, meta={"rnk": {"type": "num", "length": 8}})
'''


@pytest.mark.spark
def test_synthetic_inputs_catch_a_conversion_that_only_fits_the_case_rows(spark):
    from sas2dbx.llm.robust import robustness

    case = GOLDEN["FRONTIER-RANK-01"]
    ok, _ = gate(case, PREAMBLE + "\n# COMMAND ----------\n\n" + HARDCODED_RANK, spark)
    assert ok                                          # the parity gate on the case rows is fooled ...
    res = robustness([case], ["llm_runtime"], spark, ScriptedProvider([answer(HARDCODED_RANK)]), variants=4, seed=1)
    r = res["strategies"]["llm_runtime"]
    assert r["disagree"] >= 1 and r["agree_pct"] < 100    # ... synthetic inputs are not


@pytest.mark.spark
def test_robustness_references(spark):
    from sas2dbx.llm.robust import reference_notebook

    assert reference_notebook(GOLDEN["H-MERGE-OVERLAP-01"])[1] == "rules"
    assert reference_notebook(GOLDEN["FRONTIER-RANK-01"])[1] == "reference"
