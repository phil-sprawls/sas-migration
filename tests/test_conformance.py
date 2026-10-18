"""Every conformance case: convert the SAS, run it on Spark (ANSI on and off), compare at 2 decimals."""
import os

import pytest

from sas2dbx.validate.conformance import export_sas, load_all, run_conformance

CASES = load_all(os.path.join(os.path.dirname(__file__), "..", "conformance"))


@pytest.mark.spark
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_conformance(case, ansi_spark):
    res = run_conformance(case, ansi_spark)
    assert res.passed, res.detail


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_exports_to_sas(case):
    text = export_sas(case)
    assert case.sas.strip().splitlines()[0] in text
