"""Static checks on the SQL function bodies in ``sas2dbx.runtime.sqlfuncs`` (no Spark needed).

Newer Spark rejects a SQL UDF called inside a higher-order-function lambda
(``UNSUPPORTED_FEATURE.LAMBDA_FUNCTION_WITH_SQL_UDF``; verified on 4.3.0, accepted by
4.0.1), and Databricks Runtimes built on it can be expected to as well.  A runtime test
on older Spark cannot catch the regression: this check does, on any installed version.
"""
import re

from sas2dbx.runtime.sqlfuncs import REGISTRY, SqlFn


def lambda_bodies(sql: str) -> list[str]:
    """The text of every ``(...) -> body`` / ``x -> body`` lambda: from the arrow to the
    comma or closing parenthesis that ends the enclosing argument."""
    out = []
    for m in re.finditer(r"->", sql):
        depth, i = 0, m.end()
        while i < len(sql):
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            i += 1
        out.append(sql[m.end():i])
    return out


def test_lambda_bodies_parser():
    body = "aggregate(sas_nn(a), 0.0D, (acc, x) -> acc + f(x, 1), acc -> acc * 2)"
    assert lambda_bodies(body) == [" acc + f(x, 1)", " acc * 2"]


def test_no_sql_udf_inside_lambda():
    names = [n for n, f in REGISTRY.items() if isinstance(f, SqlFn)]
    call = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\s*\(")
    offenders = {}
    for name, f in REGISTRY.items():
        if not isinstance(f, SqlFn):
            continue
        hits = sorted({m.group(1) for b in lambda_bodies(f.body) for m in call.finditer(b)})
        if hits:
            offenders[name] = hits
    assert not offenders, f"SQL UDFs called inside a lambda (rejected by newer Spark, e.g. 4.3): {offenders}"


# ---------------------------------------------------------------------------
# the two functions that used to call a SQL UDF inside a lambda, run for real
import pytest  # noqa: E402

from sas2dbx.runtime.sqlfuncs import ensure  # noqa: E402

D_2024_03_05 = 23440.0   # SAS date of 2024-03-05 (days since 1960-01-01)


@pytest.mark.spark
def test_var_std_arr_values(spark):
    ensure(spark, ["sas_var_arr", "sas_std_arr"])
    r = spark.sql("SELECT sas_var_arr(array(1D, 2D, NULL, 4D)) v, sas_std_arr(array(2D, 4D, 4D, 4D, 5D, 5D, 7D, 9D)) s, "
                  "sas_var_arr(array(3D, NULL)) one, sas_var_arr(CAST(array() AS ARRAY<DOUBLE>)) none").first()
    assert r.v == pytest.approx(7 / 3)             # SAS VAR(1, 2, ., 4): missing ignored, n-1 divisor
    assert r.s == pytest.approx(2.138089935299395)  # SAS STD of the classic 8-value sample
    assert r.one is None and r.none is None         # fewer than 2 non-missing values -> missing


@pytest.mark.spark
@pytest.mark.parametrize("w,sep,order,expected", [
    (10, "-", "ymd", "2024-03-05"), (8, "", "ymd", "20240305"), (6, "", "ymd", "240305"),
    (10, "/", "mdy", "03/05/2024"), (8, "/", "dmy", "05/03/24"),
])
def test_put_ymd_values(spark, w, sep, order, expected):
    ensure(spark, ["sas_put_ymd"])
    assert spark.sql(f"SELECT sas_put_ymd({D_2024_03_05}D, {w}, '{sep}', '{order}') r").first().r == expected
