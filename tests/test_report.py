"""Parity certificate: plain-language verdicts, control totals and the off-by hint, from evidence alone."""
import pytest

from sas2dbx.report import fmt_num, render
from sas2dbx.validate.compare import CompareOptions, compare


def _evidence(passed=True, mismatches=0, expected_sum=10.0, actual_sum=10.0, samples=None, allow=True):
    cols = [{"name": "ID", "kind": "num", "status": "ok", "mismatches": 0, "max_abs_diff": 0.0},
            {"name": "DAY", "kind": "num", "status": "mismatch" if mismatches else "ok", "mismatches": mismatches,
             "max_abs_diff": 1.0 if mismatches else 0.0},
            {"name": "ARM", "kind": "char", "status": "ok", "mismatches": 0, "max_abs_diff": None}]
    stats = lambda s: {"n": 4, "sum": s, "min": 1.0, "max": 4.0}  # noqa: E731
    return {
        "case": "demo", "passed": passed, "conversion": {"steps": 3, "auto_converted_pct": 100.0, "hazards": []},
        "execution": {"ok": True},
        "evidence": {"program": "demo.sas", "ground_truth": "G1", "decimals": None, "tolerance": "strict",
                     "program_sha256": "ab" * 32, "expected_sha256": {"work.out": "cd" * 32}, "spark_version": "4.3.0",
                     "ansi_mode": "true", "allow_row_samples": allow, "as_of": "2024-01-31",
                     "run_at": "2026-11-02T12:00:00+00:00"},
        "comparisons": [{
            "dataset": "work.out", "passed": passed, "mode": "keyed", "keys": ["ID"], "expected_rows": 4,
            "actual_rows": 4, "matched_rows": 4, "only_in_expected": 0, "only_in_actual": 0, "duplicate_keys": 0,
            "columns": cols, "samples": samples or [], "notes": ["documented deviation: ARM labels changed in v2."],
            "metadata": [],
            "settings": {"ignored_columns": [], "column_map": {}, "metadata_ignore": ["length"], "control_by": ["ARM"],
                         "expected_file": "out.sas7bdat", "deviations": ["ARM labels changed in v2."]},
            "controls": {"rows": {"expected": 4, "actual": 4},
                         "numeric": [{"column": "DAY", "format": None, "expected": stats(expected_sum),
                                      "actual": stats(actual_sum)}],
                         "groups": [{"by": "ARM", "truncated": False,
                                     "rows": [{"value": "A", "expected": 2, "actual": 2}, {"value": "", "expected": 2, "actual": 2}]}]},
        }],
    }


def test_pass_says_same_as_sas_in_words():
    page = render(_evidence())
    assert "Same as SAS" in page
    assert "produced the same work.out as SAS: 4 rows, 12 values, every one equal" in page
    assert "one billionth of the value" in page          # the tolerance in words, not a formula
    assert "ARM labels changed in v2." in page and "Accepted by" in page
    assert "What this does not prove" in page and "2024-01-31" in page
    assert "Observation Summary" in page and "Values Comparison Summary" in page   # PROC COMPARE's section names
    assert "Not compared" in page                        # lengths skipped by metadata_ignore are shown as such


def test_fail_names_the_column_and_the_constant_offset():
    page = render(_evidence(passed=False, mismatches=3, actual_sum=7.0,
                            samples=[{"ID": 1.0, "DAY": {"expected": 2.0, "actual": 1.0}}]))
    assert "Not the same as SAS" in page
    assert "3 values differ in 1 column (DAY)" in page
    assert "Each of the 3 differing values is off by exactly 1 (Databricks lower)" in page
    assert "<td class=n>2</td><td class=n>1</td>" in page          # the sample row: SAS value, Databricks value


def test_no_offset_claim_when_totals_do_not_show_one():
    page = render(_evidence(passed=False, mismatches=3, actual_sum=9.0))
    assert "off by exactly" not in page


def test_row_values_withheld_when_samples_not_allowed():
    page = render(_evidence(passed=False, mismatches=3, actual_sum=7.0, allow=False,
                            samples=[{"ID": 1.0, "DAY": {"expected": 2.0, "actual": 1.0}}]))
    assert "Row values are withheld" in page


def test_execution_failure_is_could_not_run():
    d = _evidence(passed=False)
    d["execution"] = {"ok": False, "title": "[3] DATA out", "error": "AnalysisException: boom"}
    d["comparisons"] = []
    page = render(d)
    assert "Could not run" in page and "AnalysisException: boom" in page


def test_bare_page_has_no_document_wrapper():
    page = render(_evidence(), standalone=False)
    assert page.startswith("<title>") and "<html" not in page


def test_sas_number_display():
    assert fmt_num(None) == "."
    assert fmt_num(0.0, "DATE9.") == "1960-01-01"
    assert fmt_num(23066826.0) == "23,066,826"
    assert fmt_num(0.1 + 0.2) == "0.3"


@pytest.mark.spark
def test_control_totals_side_by_side(spark):
    e = spark.createDataFrame([(1, 1.0, "A"), (2, 2.0, "B"), (3, None, "B")], "id int, x double, arm string")
    a = spark.createDataFrame([(1, 1.0, "A"), (2, 3.0, "B"), (3, None, "B")], "id int, x double, arm string")
    r = compare(e, a, CompareOptions(keys=["id"], control_by=["arm"]))
    x = next(n for n in r.controls["numeric"] if n["column"] == "X")
    assert x["expected"] == {"n": 2, "sum": 3.0, "min": 1.0, "max": 2.0}
    assert x["actual"]["sum"] == 4.0
    assert r.controls["groups"] == [{"by": "ARM", "truncated": False, "rows": [
        {"value": "A", "expected": 1, "actual": 1}, {"value": "B", "expected": 2, "actual": 2}]}]
    assert r.controls["rows"] == {"expected": 3, "actual": 3}
    assert r.settings["control_by"] == ["ARM"]


def test_review_section_has_tables_queries_and_spot_check():
    d = _evidence(passed=False, mismatches=3, actual_sum=7.0,
                  samples=[{"ID": 1.0, "DAY": {"expected": 2.0, "actual": 1.0}}])
    d["comparisons"][0]["review"] = {"kind": "unity-catalog", "target": "main.review", "keys": ["ID"], "tables": {
        "sas": {"sql": "main.review.demo__work_out__sas", "table": "main.review.demo__work_out__sas", "rows": 4, "version": 0},
        "dbx": {"sql": "main.review.demo__work_out__dbx", "table": "main.review.demo__work_out__dbx", "rows": 4, "version": 0},
        "diff": {"sql": "main.review.demo__work_out__diff", "table": "main.review.demo__work_out__diff", "rows": 3, "version": 2}}}
    page = render(d)
    assert "Review the data" in page and "Unity Catalog" in page and "version 2" in page
    assert "WHERE column_name = &#x27;DAY&#x27;" in page
    assert "FROM main.review.demo__work_out__dbx WHERE ID = 1;" in page   # the sample row's key, ready to run
    assert "Your own spot check" in page
