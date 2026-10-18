"""Independent double-programming oracle for CDISC pilot at14-5-02.sas (serious TEAE table).

Written directly from the SAS program's specification with pandas + scipy; it shares
no code with sas2dbx, so agreement between the two is evidence of correctness
(industry-standard "independent double programming" QC).

Usage: python cases/oracles/at14_5_02.py <adam_dir> <out_dir> [high_arm low_arm]
"""
import os
import sys

import pandas as pd
from scipy.stats import fisher_exact


def best(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else repr(float(n))


def pct81(x: float) -> str:
    # SAS w.d rounds half away from zero; format to 1 decimal
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main(adam: str, out: str, high: float = 2, low: float = 1) -> None:
    # pyreadstat, not pandas.read_sas: pandas decodes some XPORT zeros as 5.4e-79 (IBM float quirk)
    import pyreadstat

    adsl, _ = pyreadstat.read_xport(os.path.join(adam, "adsl.xpt"))
    adae, _ = pyreadstat.read_xport(os.path.join(adam, "adae.xpt"))
    for df in (adsl, adae):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].fillna("").astype(str).str.rstrip()
    saf = adsl[adsl.SAFFL == "Y"]
    sae = adae[(adae.TRTEMFL == "Y") & (adae.AESER == "Y")]
    denom = saf.groupby("TRT01AN").USUBJID.nunique().rename("DENOM").reset_index().rename(columns={"TRT01AN": "TRTAN"})

    def result(levels, level_no):
        cnt = sae.groupby(["TRTAN", *levels]).USUBJID.nunique().rename("count").reset_index()
        ev = sae.groupby(["TRTAN", *levels]).USUBJID.count().rename("events").reset_index()
        r = cnt.merge(denom, on="TRTAN", how="left").merge(ev, on=["TRTAN", *levels], how="left")
        r["SUMMARYLEVEL"] = float(level_no)
        r["RESULT"] = [f"{best(c)} ({pct81(c / d * 100)}%) [{best(e)}]" for c, d, e in zip(r["count"], r.DENOM, r.events)]
        return r[["count", *levels, "TRTAN", "SUMMARYLEVEL", "DENOM", "events", "RESULT"]]

    os.makedirs(out, exist_ok=True)
    result(["AEBODSYS", "AEDECOD"], 3).to_parquet(os.path.join(out, "resultpt.parquet"), index=False)
    result(["AEBODSYS"], 2).to_parquet(os.path.join(out, "resultsoc.parquet"), index=False)
    result([], 1).to_parquet(os.path.join(out, "resultsub.parquet"), index=False)

    subjects = saf[["USUBJID", "TRT01AN"]].drop_duplicates().rename(columns={"TRT01AN": "TRTAN"})

    def fisher(level, arms, name):
        rows = []
        if level is None:
            had = set(zip(sae.USUBJID, sae.TRTAN))
            groups = [(None, [(s, t, 1 if (s, t) in had else 0) for s, t in zip(subjects.USUBJID, subjects.TRTAN)])]
        else:
            had = set(zip(sae.USUBJID, sae.TRTAN, sae[level]))
            groups = []
            for lv in sorted(sae[level].unique()):
                groups.append((lv, [(s, t, 1 if (s, t, lv) in had else 0) for s, t in zip(subjects.USUBJID, subjects.TRTAN)]))
        for lv, recs in groups:
            recs = [r for r in recs if r[1] in arms]
            tab = {}
            for _, t, e in recs:
                tab[(t, e)] = tab.get((t, e), 0) + 1
            trt = sorted({t for t, _ in tab})
            evs = sorted({e for _, e in tab})
            if len(trt) != 2 or len(evs) != 2:
                continue  # SAS computes no statistics for tables with a single level
            m = [[tab.get((t, e), 0) for e in evs] for t in trt]
            p = fisher_exact(m, alternative="two-sided")[1]
            row = {"Name1": "XP2_FISH", "nValue1": float(p)}
            if level is not None:
                row = {level: lv, **row}
            rows.append(row)
        pd.DataFrame(rows).to_parquet(os.path.join(out, f"{name}.parquet"), index=False)

    fisher("AEDECOD", (0, high), "f1pt")
    fisher("AEDECOD", (0, low), "f2pt")
    fisher("AEBODSYS", (0, high), "f1soc")
    fisher("AEBODSYS", (0, low), "f2soc")
    fisher(None, (0, high), "f1sub")
    fisher(None, (0, low), "f2sub")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], *(float(a) for a in args[2:4]))
