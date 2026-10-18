"""Parameter modules: the %INCLUDE'd file of %LET statements becomes a parameter notebook end users edit.

Without SAS, the translation of today's SAS filters to Spark SQL is checked against a second implementation: the
converter's SAS-semantics expression compiler (itself pinned by the conformance suite).  Both must select exactly
the same rows on data full of missing values, blanks, boundary numbers and dates."""
import itertools
import random

import pytest

from sas2dbx.codegen.params import read_param_module, sas_filter_to_sql
from sas2dbx.codegen.program import convert
from sas2dbx.semantic.schema import Catalog, Schema, Var
from sas2dbx.validate.conformance import ColSpec, ConformanceCase, TableSpec, execute_case_notebook

COLS = [ColSpec("ID", "num", 8), ColSpec("AMOUNT", "num", 8), ColSpec("QTY", "num", 8), ColSpec("STATUS", "char", 1),
        ColSpec("REGION", "char", 8), ColSpec("NAME", "char", 12), ColSpec("ORDER_DATE", "num", 8, "DATE9.")]
TYPES = {c.name: (c.type, c.format) for c in COLS}

FILTERS = [
    "amount > 100 and status ne 'X'", "amount < 100", "not (amount > 50)", "amount = .", "amount ^= .",
    "status = ' '", "status ne 'X'", "region in ('EAST', 'WEST')", "region not in ('EAST')", "qty in (1:3)",
    "qty not in (1, 2)", "order_date >= '01JAN2024'd", "order_date < '15FEB2024'd",
    "order_date between '01JAN2024'd and '31MAR2024'd", "order_date not between '01JAN2024'd and '31MAR2024'd",
    "name like 'A%'", "name contains 'b'", "name =: 'Ab'", "upcase(region) = 'EAST'", "year(order_date) = 2024",
    "qty < amount", "qty = amount", "qty >= amount", "amount / qty > 2", "missing(amount) or amount = 0",
    "not (status = 'X' or amount < 50)", "amount between 50 and 150", "(amount > 100 or qty > 2) and region ne 'WEST'",
    "length(name) > 3", "coalesce(amount, 0) > 10", "order_date > '01JAN2024'd + 30",
]


def _rows(n: int = 400, seed: int = 7) -> list[list]:
    rng = random.Random(seed)
    pools = [[None, -5, 0, 50, 99.99, 100, 100.004, 150, 1e6], [None, 0, 1, 2, 3, 5], [None, "A", "X", "B"],
             ["EAST", "WEST", None, "NORTH", "east"], [None, "Bob", "Alice", "Abner", "x_y", "50%"],
             [None, "31DEC2019", "01JAN2024", "15FEB2024", "31MAR2024", "01APR2024", "02FEB2024"]]
    return [[i + 1] + [rng.choice(p) for p in pools] for i in range(n)]


@pytest.mark.parametrize("sas", FILTERS)
def test_filter_translation_is_spark_sql(sas):
    sql = sas_filter_to_sql(sas, TYPES)
    assert sql and "&" not in sql


@pytest.mark.spark
def test_translated_filters_select_the_same_rows_as_sas(spark):
    """Differential: SAS semantics (DATA step WHERE, converted by the rules) vs the Spark SQL translation applied
    with ctx.param_filter on the data as Databricks tables show it."""
    inputs = {"t": TableSpec(COLS, _rows())}
    mismatches = []
    for sas in FILTERS:
        case = ConformanceCase(id="f", title="f", sas="", inputs=inputs, expected={})
        conv = convert(f"data out; set t; where {sas}; run;", file="f.sas", catalog=_catalog())
        assert [s.status for s in conv.steps if s.kind != "GLOBAL"] == ["converted"], sas
        exe, ctx = execute_case_notebook(case, spark, conv.notebook)
        assert exe["ok"], (sas, exe)
        want = {r[0] for r in ctx.get("out").select("ID").collect()}
        ctx.params["F"] = sas_filter_to_sql(sas, TYPES)
        got = {r[0] for r in ctx.param_filter(ctx.get("t"), "F").select("ID").collect()}
        if want != got:
            mismatches.append((sas, ctx.params["F"], sorted(want ^ got)[:8]))
    assert not mismatches, mismatches


def _catalog() -> Catalog:
    cat = Catalog()
    cat.put("WORK.T", Schema([Var(c.name, c.type, c.length, c.format) for c in COLS]))
    return cat


def test_reads_values_comments_and_other_statements():
    specs, notes = read_param_module("/* region to report */\n%let region = EAST;\n%let f = %str(a > 1);  /* the filter */\n"
                                     "%let g = &region._x;\noptions mprint;\n", "p.sas")
    assert [(s.name, s.default, s.comment) for s in specs.values()] == [
        ("REGION", "EAST", "region to report"), ("F", "a > 1", "the filter"), ("G", "EAST_x", "")]
    assert specs["G"].refs == ["REGION"] and notes and "options mprint" in notes[0]


@pytest.mark.parametrize("sas,why", [
    ("scan(name, 1) = 'a'", "function SCAN"),
    ("name || 'x' = 'ax'", "in a filter"),
])
def test_untranslatable_filters_are_named(sas, why):
    from sas2dbx.codegen.exprs import Unsupported
    with pytest.raises(Unsupported, match=why.split()[0]):
        sas_filter_to_sql(sas, TYPES)


@pytest.mark.spark
def test_parameter_check_reports_every_problem_before_reading_data(spark):
    from sas2dbx.runtime.context import SasContext

    ctx = SasContext(spark, write_outputs=False)
    spec = {"FILTER": "filter", "VARS": "columns", "START": "date", "N": "number", "REGION": "text"}
    with pytest.raises(ValueError) as e:
        ctx.load_params(spec, {"filter": "amount >>> 5", "vars": ["id", "bad name"], "start": "31/01/2024",
                               "n": "five"})
    msg = str(e.value)
    for part in ("filter: not a valid Spark SQL condition", "vars: not column names", "start: expected a date",
                 "n: expected a number", "region: no value"):
        assert part in msg, msg
    ctx.load_params(spec, {"filter": "amount > 5 AND typo_col = 1", "vars": ["id"], "start": "2024-01-31", "n": 5,
                           "region": "EAST"})
    df = spark.createDataFrame([(1.0, 7.0, 0)], "ID double, AMOUNT double, _sas_seq long")
    with pytest.raises(ValueError, match="typo_col"):
        ctx.param_filter(df, "FILTER")
    with pytest.raises(ValueError, match="columns not in the data"):
        ctx.load_params({"VARS": "columns"}, {"vars": ["id", "nope"]})
        ctx.param_keep(df, "VARS")
    assert ctx.symget("START") == "31JAN2024" and ctx.symget("N") == "5"


def test_proc_statements_never_read_a_run_time_value_as_a_name():
    """class &cv used to become an empty CLASS list (silently wrong totals): now it is visible unsupported work."""
    cat = Catalog()
    cat.put("WORK.IN", Schema([Var("ID", "num", 8, None), Var("REGION", "char", 8, None), Var("AMOUNT", "num", 8, None)]))
    sas = ("data _null_; call symputx('cv', 'region'); run;\n"
           "proc means data=in noprint; class &cv; var amount; output out=out sum=total; run;")
    conv = convert(sas, file="t.sas", catalog=cat)
    assert [s.status for s in conv.steps if s.kind != "GLOBAL"] == ["converted", "unsupported"]
    assert "&CV in PROC MEANS CLASS" in conv.steps[-1].reasons[0]


def test_filter_parameter_under_or_is_not_split():
    conv = convert("%include 'p.sas'; data out; set in; where x > 1 or &f; run;", file="t.sas",
                   catalog=_catalog(), param_module_text="%let f = amount > 5;")
    assert [s.status for s in conv.steps if s.kind != "GLOBAL"] == ["unsupported"]
    assert "combined with OR/NOT" in conv.steps[-1].reasons[0]


def test_parameter_notebook_is_python_and_main_notebook_runs_it():
    conv = convert("%include 'params.sas'; data out; set t; where &f; keep &v; run;", file="r.sas", catalog=_catalog(),
                   param_module_text="%let f = amount < 100; /* small orders */\n%let v = id amount;\n")
    ns: dict = {}
    exec(compile(conv.param_notebook, "params.py", "exec"), ns)
    assert ns["PARAMETERS"] == {"f": "(amount < 100 OR amount IS NULL)", "v": ["id", "amount"]}
    assert "# MAGIC %run ./params" in conv.notebook and "ctx.load_params(PARAM_SPEC" in conv.notebook
    assert conv.param_kinds == {"F": "filter", "V": "columns"}
    assert list(itertools.islice((s.status for s in conv.steps if s.kind != "GLOBAL"), 2)) == ["converted"]


def test_a_user_edit_of_the_parameter_notebook_changes_only_the_values():
    """Parity runs with a user's values run the parameter notebook as that user would leave it."""
    from sas2dbx.codegen.params import edit_param_notebook

    conv = convert("%include 'params.sas'; data out; set t; where &f; keep &v; title \"&t\"; run;", file="r.sas",
                   catalog=_catalog(), param_module_text=("%let f = amount < 100; /* small orders */\n%let v = id amount;\n"
                                                          f"%let t = {'Ré' + 'x' * 80};\n"))
    edited = edit_param_notebook(conv.param_notebook, {"F": "region = 'WEST'", "t": "Été"})
    ns: dict = {}
    exec(compile(edited, "params.py", "exec"), ns)
    assert ns["PARAMETERS"] == {"f": "region = 'WEST'", "v": ["id", "amount"], "t": "Été"}
    assert "# f: a Spark SQL condition" in edited and "small orders" in edited      # comments and layout kept
    assert edited.count("\n") < conv.param_notebook.count("\n")                     # the long text was replaced
    with pytest.raises(KeyError, match="no nope"):
        edit_param_notebook(conv.param_notebook, {"nope": 1})
