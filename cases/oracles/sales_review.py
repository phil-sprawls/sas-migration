"""Independent double-programming oracle for examples/sales_review/report.sas (synthetic program, real data).

Written from the program's specification with pandas, reading the original productsales.sas7bdat with pyreadstat:
it shares no code with sas2dbx or spark-sas7bdat, so agreement is evidence that the converted program, the parameter
notebook and the spark-sas7bdat read are all right (industry-standard "independent double programming" QC).

Each run's filters are written out by hand from their meaning: today's SAS filters follow SAS rules (a missing value
is lower than any number; blank text satisfies "ne"), filters a user writes in the parameter notebook follow Spark
SQL rules (a comparison with NULL is never true).

Usage: python cases/oracles/sales_review.py <productsales.sas7bdat> <out_dir>   (writes <out_dir>/<run>/<table>.parquet)
"""
import math
import os
import sys
from datetime import date

import pandas as pd


def sas_date(d: date) -> float:
    return float((d - date(1960, 1, 1)).days)


def missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def sas_round(x, unit: float):
    """SAS ROUND: to the nearest multiple of ``unit``, halves away from zero."""
    if missing(x):
        return None
    q = abs(x) / unit
    return math.copysign(math.floor(q + 0.5 + 1e-9) * unit, x)


RUNS = {
    # today's values of params.sas: SAS rules
    "defaults": dict(
        start=date(1993, 4, 1), end=date(1994, 3, 31), countries={"CANADA", "GERMANY", "U.S.A."},
        rows=lambda r: (not missing(r.ACTUAL) and r.ACTUAL > 100) and r.PRODTYPE != "OFFICE",
        flag=lambda pct: missing(pct) or pct < 95,          # SAS: a missing percentage is lower than 95
        keep=["COUNTRY", "REGION", "PRODUCT", "MONTH", "ACTUAL", "PREDICT"],
        group=["COUNTRY", "PRODUCT"], title="Furniture sales review"),
    # what a user typed into the parameter notebook (cases/sales_review_user.yaml): Spark SQL rules
    "user": dict(
        start=date(1993, 1, 1), end=date(1993, 12, 31), countries={"GERMANY", "U.S.A."},
        rows=lambda r: (not missing(r.ACTUAL) and r.ACTUAL > 500) and r.REGION == "WEST",
        flag=lambda pct: not missing(pct) and pct < 105,     # Spark SQL: NULL < 105 is not true
        keep=["COUNTRY", "REGION", "DIVISION", "PRODUCT", "MONTH", "ACTUAL", "PREDICT"],
        group=["COUNTRY", "REGION", "PRODUCT"], title="West region review"),
}

TARGETS = pd.DataFrame([("CANADA", "SOFA", 30000.0), ("CANADA", "BED", 26000.0), ("GERMANY", "SOFA", 25000.0),
                        ("GERMANY", "CHAIR", 24000.0), ("U.S.A.", "SOFA", 31000.0), ("U.S.A.", "BED", None)],
                       columns=["COUNTRY", "PRODUCT", "TARGET"])


def run(prdsale: pd.DataFrame, p: dict) -> dict[str, pd.DataFrame]:
    lo, hi = sas_date(p["start"]), sas_date(p["end"])
    keep_row = [p["rows"](r) and lo <= r.MONTH <= hi and r.COUNTRY in p["countries"] for r in prdsale.itertuples()]
    detail = prdsale[keep_row][p["keep"]]
    detail = detail.sort_values(["COUNTRY", "PRODUCT", "MONTH"], kind="mergesort")          # PROC SORT is stable
    summary = (detail.groupby(p["group"], sort=False)
               .agg(N_ROWS=("ACTUAL", "size"), ACTUAL=("ACTUAL", "sum"), PREDICT=("PREDICT", "sum")).reset_index())
    summary = summary.sort_values(["COUNTRY", "PRODUCT"], kind="mergesort")
    targets = TARGETS.sort_values(["COUNTRY", "PRODUCT"], kind="mergesort")
    review = summary.merge(targets, on=["COUNTRY", "PRODUCT"], how="left")                  # MERGE ... ; IF in_sales
    review["PCT_OF_TARGET"] = [None if missing(t) else sas_round(100 * a / t, 0.1) for a, t in zip(review.ACTUAL, review.TARGET)]
    flagged = review[[p["flag"](v) for v in review.PCT_OF_TARGET]]
    title = pd.DataFrame({"TITLE": [p["title"]]})
    return {"detail": detail, "summary": summary, "review": review, "flagged": flagged, "report_title": title}


def main(path: str, out: str) -> None:
    import pyreadstat

    prdsale, _ = pyreadstat.read_sas7bdat(path, disable_datetime_conversion=True)
    for c in prdsale.columns:
        if prdsale[c].dtype == object:
            prdsale[c] = prdsale[c].fillna("").astype(str).str.rstrip()
    for name, p in RUNS.items():
        os.makedirs(os.path.join(out, name), exist_ok=True)
        for table, df in run(prdsale, p).items():
            df.reset_index(drop=True).to_parquet(os.path.join(out, name, f"{table}.parquet"), index=False)
            print(f"{name}/{table}: {len(df)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
