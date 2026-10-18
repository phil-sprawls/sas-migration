"""Values from outside the program (prompts, parameter modules, the run's clock and environment) and very large
constructs: nothing may be frozen into a conversion that claims to be converted, and nothing may crash it."""
import os

import pytest

from sas2dbx.analyze.inventory import analyze_program
from sas2dbx.codegen.program import convert
from sas2dbx.semantic.schema import Catalog, Schema, Var
from sas2dbx.validate.conformance import ColSpec, ConformanceCase, TableSpec, run_conformance_notebook

CAT = Catalog()
CAT.put("WORK.IN", Schema([Var("ID", "num", 8, None), Var("X", "num", 8, None), Var("REGION", "char", 8, None)]))


def _statuses(conv):
    return [s.status for s in conv.steps if s.kind != "GLOBAL"]


def _code(conv):
    return "\n".join(ln for ln in conv.notebook.split("\n") if not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("sas", [
    'data out; set in; where region = "&region"; run;',
    '%let r2 = &region; data out; set in; where region = "&r2"; run;',
    "%macro m; %if &mode = FULL %then %do; data a; set in; run; %end; %else %do; data a; set in(obs=1); run; %end; "
    "%mend; %m",
    "%macro m; %let i = 1; %do %while(&i <= &n); data p&i; set in; run; %let i = %eval(&i + 1); %end; %mend; %m",
    "%macro m; %do i = 1 %to &n; data p&i; set in; run; %end; %mend; %m",
], ids=["value", "let-chain", "if", "do-while", "do-to"])
def test_undefined_macro_variable_is_never_frozen(sas):
    conv = convert(sas, file="t.sas", catalog=CAT)
    assert _statuses(conv) and set(_statuses(conv)) == {"unsupported"}
    assert "nothing defines" in " ".join(r for s in conv.steps for r in s.reasons)


def test_failed_macro_function_is_not_a_blank_value():
    conv = convert("%let p = %sysfunc(pathname(work)); data out_&p; set in; run;", file="t.sas", catalog=CAT)
    assert "unsupported" in _statuses(conv)


def test_run_values_are_read_when_the_notebook_runs(monkeypatch):
    monkeypatch.setenv("SAS2DBX_PROBE", "converter-machine-value")
    sas = ('data out; set in; length a b c d $40; a = "&sysdate9"; b = "%sysget(SAS2DBX_PROBE)"; c = "&sysparm"; '
           'd = "%sysfunc(today(), date9.)"; run;')
    conv = convert(sas, file="t.sas", catalog=CAT)
    code = _code(conv)
    assert _statuses(conv) == ["converted"]
    assert "converter-machine-value" not in conv.notebook and "01JAN2026" not in code
    for name in ("SYSDATE9", "SYSGET_SAS2DBX_PROBE", "SYSPARM", "SYSFUNC_TODAY__DATE9"):
        assert f"ctx.symget('{name}')" in code
    assert conv.params == ["SYSPARM"] and "params=['SYSPARM']" in conv.notebook


def test_declared_parameter_comes_from_the_job_and_define_fixes_a_value():
    sas = 'data out; set in; where region = "&region" and x > &threshold; run;'
    conv = convert(sas, file="t.sas", catalog=CAT, params=["region"], macro_vars={"threshold": "5"})
    assert _statuses(conv) == ["converted"] and conv.unresolved == []
    assert "ctx.symget('REGION')" in _code(conv) and "params=['REGION']" in conv.notebook
    assert "symget('THRESHOLD')" not in _code(conv) and "5.0" in _code(conv)


def test_placeholder_never_reaches_converted_code():
    # CONTAINS builds its LIKE pattern from the literal: a run-time value there is not handled (yet) -> unsupported
    sas = ("data _null_; set in; if id = 1 then call symputx('r', region); run;\n"
           'proc sql; create table out as select * from in where region contains "&r"; quit;')
    conv = convert(sas, file="t.sas", catalog=CAT)
    assert _statuses(conv) == ["converted", "unsupported"]
    assert "\\ue100" not in _code(conv)


def test_very_long_else_if_chain_converts():
    chain = " else ".join(f"if x={i} then y={i};" for i in range(1, 1501))
    conv = convert(f"data out; set in; {chain} run;", file="t.sas", catalog=CAT)
    assert _statuses(conv) == ["converted"]


def test_oversized_expression_is_flagged_not_crashed():
    conv = convert("data out; set in; t = " + " + ".join(["x"] * 2000) + "; run;", file="t.sas", catalog=CAT)
    assert _statuses(conv) == ["unsupported"] and conv.steps[0].reasons


def test_notebook_compiles_with_multi_line_global_statements():
    sas = 'goptions reset=all device=gif\n   gsfname=grafout\n   xpixels=600;\nlibname a\n  "/x";\ndata out; x=1; run;\n'
    conv = convert(sas, file="t.sas", catalog=CAT)
    compile(conv.notebook, "notebook", "exec")


def test_one_bad_step_never_takes_the_program_down():
    sas = "data a; set in; substr(newvar, 1, 1) = 'x'; run; data b; set in; run;"
    conv = convert(sas, file="t.sas", catalog=CAT)
    assert _statuses(conv) == ["unsupported", "converted"]


def test_inventory_lists_values_from_outside():
    pa = analyze_program("t.sas", 'data out; set in; where region = "&region" and id < &sysparm; run;')
    assert (pa.inputs["undefined_macro_vars"], pa.inputs["run_time_values"]) == (["REGION"], ["SYSPARM"])


@pytest.mark.spark
def test_one_notebook_serves_every_parameter_value(spark):
    rows = [[1, 3, "EAST"], [2, 7, "WEST"], [3, 9, "O'NEIL"]]
    cols = [ColSpec("ID", "num", 8), ColSpec("X", "num", 8), ColSpec("REGION", "char", 8)]
    sas = 'proc sql; create table out as select * from in where region = "&region"; quit;'
    conv = convert(sas, file="t.sas", catalog=CAT, params=["region"])
    for region in ("WEST", "O'NEIL"):
        case = ConformanceCase(id="p", title="p", sas=sas, inputs={"in": TableSpec(cols, rows)},
                               macro_vars={"region": region},
                               expected={"out": TableSpec(cols, [r for r in rows if r[2] == region])})
        res = run_conformance_notebook(case, spark, conv.notebook)
        assert res.passed, res.detail


@pytest.mark.spark
def test_thousands_of_statements_in_one_step_run(spark):
    n = 3000
    sas = f"%macro g; %do i=1 %to {n}; if x=&i then y=&i; %end; %mend; data out; set in; y=0; %g run;"
    conv = convert(sas, file="t.sas", catalog=CAT)
    assert "ctx.cut(df)" in conv.notebook
    rows = [[1, 5.0, "A"], [2, 2999.0, "B"], [3, 4000.0, "C"]]
    cols = [ColSpec("ID", "num", 8), ColSpec("X", "num", 8), ColSpec("REGION", "char", 8)]
    case = ConformanceCase(id="g", title="g", sas=sas, inputs={"in": TableSpec(cols, rows)},
                           expected={"out": TableSpec(cols + [ColSpec("Y", "num", 8)],
                                                      [r + [r[1] if r[1] <= n else 0.0] for r in rows])})
    res = run_conformance_notebook(case, spark, conv.notebook)
    assert res.passed, res.detail


def test_cli_param_and_define_flags(tmp_path):
    from sas2dbx.cli import build_parser

    prog = tmp_path / "p.sas"
    prog.write_text('data out; set in; where region = "&region" and x > &threshold; run;')
    a = build_parser().parse_args(["convert", str(prog), "--param", "region", "--define", "threshold=5",
                                   "--out", str(tmp_path / "p.py")])
    a.fn(a)
    nb = (tmp_path / "p.py").read_text()
    assert "params=['REGION']" in nb and "ctx.symget('REGION')" in nb and os.path.exists(tmp_path / "p.py")
