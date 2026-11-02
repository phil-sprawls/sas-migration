"""Parity certificate: a plain-language HTML page per validated case, built from the evidence JSON `validate` writes.

The audience is the people who sign off a migrated program (business owners, SMEs, auditors), not the migration
team, so it says what was compared and what "the same" means in words, mirrors the section names of SAS PROC COMPARE
(Data Set, Variables, Observation and Values Comparison Summaries) that SAS users already read, puts control totals
beside the cell-by-cell result, and says what a pass does not prove.  Everything shown comes from the evidence file.
"""
from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta

_SAS_EPOCH = date(1960, 1, 1)
_DATE_FORMATS = ("DATE", "YYMMDD", "MMDDYY", "DDMMYY", "E8601DA", "IS8601DA", "B8601DA", "WORDDATE", "WEEKDATE", "MONYY")
_DATETIME_FORMATS = ("DATETIME", "E8601DT", "IS8601DT", "B8601DT")

GROUND_TRUTH = {
    "G1": "output that SAS itself produced from the same inputs",
    "G2": "row and column counts taken from SAS logs",
    "G3": "an independently programmed reference (not SAS output)",
    "G4": "synthetic test data (robustness only, not parity with SAS)",
}
METADATA_WORDS = {"order": "column order", "length": "column lengths", "format": "display formats",
                  "label": "column labels"}


def e(v) -> str:
    return html.escape("" if v is None else str(v))


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def fmt_num(v, fmt: str | None = None) -> str:
    """A number the way a SAS user expects to read it: '.' for missing, dates as dates when the column has a date
    format, integers with thousands separators, otherwise up to 12 significant digits."""
    if v is None:
        return "."
    f = (fmt or "").upper()
    try:
        if f.startswith(_DATETIME_FORMATS):
            return (datetime(1960, 1, 1) + timedelta(seconds=float(v))).strftime("%Y-%m-%d %H:%M:%S")
        if f.startswith(_DATE_FORMATS) and float(v).is_integer():
            return (_SAS_EPOCH + timedelta(days=int(v))).isoformat()
    except (OverflowError, ValueError):
        pass
    if isinstance(v, (int, float)) and float(v).is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    if isinstance(v, (int, float)):
        return format(v, ",.12g")
    return str(v)


def fmt_value(v) -> str:
    if v is None:
        return "(missing)"
    if v == "":
        return "(blank)"
    return fmt_num(v) if isinstance(v, (int, float)) else str(v)


def tolerance_words(decimals) -> str:
    if decimals is None:
        return ("Text values must be identical (trailing blanks are ignored, as in SAS). Numbers must be identical, "
                "except for differences smaller than one billionth of the value: the last-digit noise two computers "
                "produce when they add the same numbers in a different order. Nothing is rounded to make values match.")
    return (f"Text values must be identical (trailing blanks are ignored, as in SAS). Numbers must agree to {decimals} "
            f"decimal places, because the SAS reference values were recorded rounded to {decimals} decimals.")


def _deviations(c: dict) -> list[str]:
    s = c.get("settings") or {}
    if s.get("deviations") is not None:
        return list(s["deviations"])
    return [n.removeprefix("documented deviation: ") for n in c.get("notes", []) if n.startswith("documented deviation: ")]


def _other_notes(c: dict) -> list[str]:
    return [n for n in c.get("notes", []) if not n.startswith("documented deviation: ")]


def _compared_columns(c: dict) -> list[dict]:
    return [col for col in c.get("columns", []) if col["status"] in ("ok", "mismatch")]


def _values_compared(c: dict) -> int:
    return c.get("matched_rows", 0) * len(_compared_columns(c))


def _value_differences(c: dict) -> int:
    return sum(col.get("mismatches", 0) for col in c.get("columns", []))


def _problems(c: dict) -> list[str]:
    """Why one output failed, in plain words, most important first."""
    out = []
    if c.get("expected_rows", 0) < 0:
        return [n for n in c.get("notes", [])] or ["the output was not produced"]
    if c.get("mode") == "keyed" and not c.get("matched_rows") and _other_notes(c):
        out += _other_notes(c)
    if c["expected_rows"] != c["actual_rows"]:
        out.append(f"SAS has {plural(c['expected_rows'], 'row')}, Databricks has {c['actual_rows']:,}")
    if c.get("only_in_expected") or c.get("only_in_actual"):
        out.append(f"{plural(c['only_in_expected'], 'row')} only in SAS, {c['only_in_actual']:,} only in Databricks")
    bad = [col for col in c.get("columns", []) if col["status"] == "mismatch"]
    if bad:
        out.append(f"{plural(_value_differences(c), 'value')} differ in {plural(len(bad), 'column')} "
                   f"({', '.join(col['name'] for col in bad[:6])}{', ...' if len(bad) > 6 else ''})")
    missing = [col["name"] for col in c.get("columns", []) if col["status"] == "missing_in_actual"]
    extra = [col["name"] for col in c.get("columns", []) if col["status"] == "extra_in_actual"]
    typed = [col["name"] for col in c.get("columns", []) if col["status"] == "type_mismatch"]
    if missing:
        out.append(f"missing from Databricks: {', '.join(missing)}")
    if extra:
        out.append(f"not in SAS: {', '.join(extra)}")
    if typed:
        out.append(f"text in one and number in the other: {', '.join(typed)}")
    if c.get("metadata"):
        kinds = sorted({METADATA_WORDS.get(m["check"], m["check"]) for m in c["metadata"]})
        out.append(f"{plural(len(c['metadata']), 'attribute difference')} ({', '.join(kinds)})")
    return out


def _hint(c: dict) -> str | None:
    """One sentence on where to look first."""
    bad = [col for col in c.get("columns", []) if col["status"] == "mismatch"]
    rows_off = c.get("only_in_expected") or c.get("only_in_actual") or c["expected_rows"] != c["actual_rows"]
    if rows_off:
        return ("Rows are missing or extra, so start with the steps that select or combine rows "
                "(WHERE and IF conditions, MERGE and joins, de-duplication).")
    if len(bad) == 1:
        col = bad[0]
        hint = (f"Every difference is in {col['name']}, so the likely cause is the one derivation that computes "
                f"{col['name']} (or a value it is computed from).")
        return hint + (_constant_offset(c, col) or "")
    if bad:
        return "Differences span several columns; compare the earliest derivation they share."
    if c.get("metadata") and not _value_differences(c):
        return "Every value matches; only the column descriptions differ, which affects how the data set is labelled and displayed."
    return None


def _constant_offset(c: dict, col: dict) -> str | None:
    """' Each of the N differing values is off by exactly X' when the control totals show it: the totals differ by
    N times the largest difference, and the number of non-missing values is the same on both sides."""
    if col["kind"] != "num" or not col.get("mismatches") or not col.get("max_abs_diff"):
        return None
    ctl = next((x for x in (c.get("controls") or {}).get("numeric", []) if x["column"] == col["name"]), None)
    if not ctl or ctl["expected"]["n"] != ctl["actual"]["n"] or None in (ctl["expected"]["sum"], ctl["actual"]["sum"]):
        return None
    gap = ctl["expected"]["sum"] - ctl["actual"]["sum"]
    step = col["max_abs_diff"]
    if abs(abs(gap) - col["mismatches"] * step) > 1e-9 * max(1.0, abs(gap)):
        return None
    side = "lower" if gap > 0 else "higher"
    return (f" Each of the {col['mismatches']:,} differing values is off by exactly {fmt_num(step)} (Databricks "
            f"{side}), the pattern of an off-by-{fmt_num(step)} in a formula.")


def _kpis(c: dict) -> str:
    diffs = _value_differences(c) + c.get("only_in_expected", 0) + c.get("only_in_actual", 0)
    items = [
        ("Rows matched", f"{c.get('matched_rows', 0):,}", f"of {c['expected_rows']:,} in SAS"),
        ("Columns compared", f"{len(_compared_columns(c)):,}", f"of {len([x for x in c.get('columns', []) if x['status'] != 'extra_in_actual']):,} in SAS"),
        ("Values compared", f"{_values_compared(c):,}", "cell by cell"),
        ("Differences", f"{diffs:,}", "values or rows" if diffs else "none found"),
    ]
    cells = "".join(f'<div class="kpi{" bad" if lab == "Differences" and diffs else ""}"><span class="kl">{e(lab)}</span>'
                    f'<span class="kval">{e(v)}</span><span class="ks">{e(sub)}</span></div>' for lab, v, sub in items)
    return f'<div class="kpis">{cells}</div>'


def _table(head: list[str], rows: list[list[str]], cls: str = "", num_cols: tuple[int, ...] = ()) -> str:
    th = "".join(f'<th{" class=n" if i in num_cols else ""}>{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td{" class=n" if i in num_cols else ""}>{v}</td>' for i, v in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _status_pill(ok: bool, yes: str = "Match", no: str = "Differs") -> str:
    return f'<span class="pill {"ok" if ok else "no"}">{yes if ok else no}</span>'


def _section(eyebrow: str, title: str, body: str) -> str:
    return f'<section class="blk"><p class="eyebrow">{e(eyebrow)}</p><h3>{e(title)}</h3>{body}</section>'


def _dataset(c: dict, ev: dict, allow_samples: bool) -> str:
    s = c.get("settings") or {}
    name = c["dataset"]
    parts = []
    ok = c["passed"]
    problems = [] if ok else _problems(c)
    head = (f'<header class="dshead"><h2><span class="mono">{e(name)}</span></h2>{_status_pill(ok, "Same as SAS", "Not the same")}</header>')
    if problems:
        hint = _hint(c)
        head += ('<div class="callout no"><strong>What differs</strong><ul>' + "".join(f"<li>{e(p)}</li>" for p in problems)
                 + "</ul>" + (f'<p class="hint">{e(hint)}</p>' if hint else "") + "</div>")
    parts.append(head)
    if c.get("expected_rows", 0) < 0:
        return '<article class="ds">' + "".join(parts) + "</article>"
    parts.append(_kpis(c))

    sha = (ev.get("expected_sha256") or {}).get(name)
    parts.append(_section("PROC COMPARE · Data Set Summary", "What was compared", _table(
        ["", "Source", "Rows", "Columns"],
        [["SAS (reference)", f'<span class="mono">{e(s.get("expected_file") or "reference")}</span>'
          + (f'<br><span class="fine mono">sha256 {e(sha[:16])}…</span>' if sha else ""),
          f"{c['expected_rows']:,}", f"{len([x for x in c['columns'] if x['status'] != 'extra_in_actual']):,}"],
         ["Databricks (converted)", f'notebook from <span class="mono">{e(ev.get("program", "the program"))}</span>',
          f"{c['actual_rows']:,}", f"{len([x for x in c['columns'] if x['status'] != 'missing_in_actual']):,}"]],
        "kv", (2, 3))))

    cols = c.get("columns", [])
    n_char = sum(1 for x in cols if x["status"] in ("ok", "mismatch") and x["kind"] == "char")
    n_num = sum(1 for x in cols if x["status"] in ("ok", "mismatch") and x["kind"] == "num")
    var_rows = [["In both", f"{n_char + n_num:,}", f"{n_char} text, {n_num} numeric"]]
    for status, label in (("missing_in_actual", "Only in SAS"), ("extra_in_actual", "Only in Databricks"),
                          ("type_mismatch", "Text in one, number in the other")):
        names = [x["name"] for x in cols if x["status"] == status]
        if names:
            var_rows.append([f'<span class="bad">{label}</span>', str(len(names)), e(", ".join(names))])
    for old, new in (s.get("column_map") or {}).items():
        var_rows.append(["Renamed", "1", f'SAS <span class="mono">{e(old)}</span> is compared with Databricks '
                         f'<span class="mono">{e(new)}</span> (see deviations)'])
    if s.get("ignored_columns"):
        var_rows.append(["Not compared", str(len(s["ignored_columns"])),
                         e(", ".join(s["ignored_columns"])) + " (see deviations)"])
    parts.append(_section("PROC COMPARE · Variables Summary", "Columns", _table(["", "Count", "Detail"], var_rows, "kv", (1,))))

    mode = c.get("mode")
    if mode == "keyed":
        how = f'Rows are paired by <span class="mono">{e(" + ".join(c.get("keys", [])))}</span>, which identifies each row.'
    elif mode == "ordered":
        how = "Rows are paired by position: the first SAS row with the first Databricks row, and so on."
    elif mode == "unordered":
        how = ("Rows have no identifying key, so rows are paired by their values; any row without an identical "
               "partner counts as a difference.")
    else:
        how = "The reference has no columns, so only the number of rows is compared."
    obs = [["Rows in SAS", f"{c['expected_rows']:,}"], ["Rows in Databricks", f"{c['actual_rows']:,}"],
           ["Rows paired and compared", f"{c.get('matched_rows', 0):,}"],
           ["Only in SAS", f"{c.get('only_in_expected', 0):,}"], ["Only in Databricks", f"{c.get('only_in_actual', 0):,}"]]
    if c.get("duplicate_keys"):
        obs.append(["Key values that repeat", f"{c['duplicate_keys']:,}"])
    notes = "".join(f'<p class="fine">{e(n)}</p>' for n in _other_notes(c))
    parts.append(_section("PROC COMPARE · Observation Summary", "Rows", f"<p>{how}</p>" + _table(["", "Rows"], obs, "kv", (1,)) + notes))

    bad = [x for x in cols if x["status"] == "mismatch"]
    vtxt = (f"<p>All {_values_compared(c):,} values compared are equal ({plural(len(_compared_columns(c)), 'column')} × "
            f"{c.get('matched_rows', 0):,} rows).</p>" if not bad else
            f"<p>{plural(len(bad), 'column')} with differences, {len(_compared_columns(c)) - len(bad)} with every value equal.</p>")
    if bad:
        vtxt += _table(["Column", "Type", "Values that differ", "Largest numeric difference"],
                       [[f'<span class="mono">{e(x["name"])}</span>', "number" if x["kind"] == "num" else "text",
                         f"{x['mismatches']:,}", fmt_num(x.get("max_abs_diff")) if x["kind"] == "num" else "–"] for x in bad],
                       "", (2, 3))
    full = _table(["Column", "Type", "Result", "Values that differ", "Largest numeric difference"],
                  [[f'<span class="mono">{e(x["name"])}</span>', "number" if x["kind"] == "num" else "text",
                    _status_pill(x["status"] == "ok", "Equal", x["status"].replace("_", " ")),
                    f"{x.get('mismatches', 0):,}",
                    (fmt_num(x["max_abs_diff"]) if x.get("max_abs_diff") is not None else "–")] for x in cols],
                  "dense", (3, 4))
    vtxt += f'<details><summary>Every column ({len(cols)})</summary>{full}</details>'
    parts.append(_section("PROC COMPARE · Values Comparison Summary", "Values", vtxt))

    parts.append(_section("Attributes", "Column order, lengths, formats and labels", _attributes(c)))
    if c.get("controls"):
        parts.append(_section("Control totals", "Totals you can reconcile yourself", _controls(c)))
    if not ok and (bad or c.get("only_in_expected") or c.get("only_in_actual")):
        parts.append(_section("Differences", "First differing rows", _samples(c, allow_samples)))
    if c.get("review"):
        parts.append(_section("Review the data", "Check it yourself", _review(c)))
    devs = _deviations(c)
    if devs:
        items = "".join(f'<li><p>{e(d)}</p><p class="accept">Accepted by ____________________ &nbsp; Date __________</p></li>'
                        for d in devs)
        parts.append(_section("Documented deviations", "Where the comparison was adjusted, and why",
                              "<p>Each adjustment needs a reviewer's acceptance. A deviation nobody accepts counts as a failure.</p>"
                              f'<ol class="devs">{items}</ol>'))
    return '<article class="ds">' + "".join(parts) + "</article>"


def _sql_literal(v) -> str:
    if isinstance(v, (int, float)):
        return fmt_num(v).replace(",", "")
    return "'" + str(v).replace("'", "''") + "'"


def _review(c: dict) -> str:
    """Where the reference, the output and the differences can be queried, with queries ready to paste."""
    r = c["review"]
    t = r.get("tables", {})
    where = ("Unity Catalog, under that schema's permissions" if r.get("kind") == "unity-catalog"
             else "Parquet files on the machine that ran the comparison (query them with Spark SQL)")
    rows = []
    for part, label in (("sas", "SAS reference"), ("dbx", "Databricks output"), ("diff", "Differences, one row per cell")):
        x = t.get(part)
        if x:
            ver = f"version {x['version']}" if x.get("version") is not None else "–"
            rows.append([label, f'<span class="mono wrap">{e(x.get("table") or x.get("path"))}</span>', f"{x['rows']:,}", ver])
    out = [f"<p>The full tables behind this certificate are saved in {e(where)}. Row values stay there.</p>",
           _table(["", "Table", "Rows", "Delta version"], rows, "kv", (2,))]
    keys = [k for k in (r.get("keys") or []) if not k.startswith("_")]
    sample = next((smp for smp in c.get("samples") or [] if "_side" not in smp), None)
    if keys:
        cond = " AND ".join(f"{k} = {_sql_literal(sample[k]) if sample and sample.get(k) is not None else '<' + k + '>'}"
                            for k in keys)
    else:
        cond = "<condition>"
    q = []
    if "diff" in t:
        d = t["diff"]["sql"]
        q.append(f"-- 1. Where the differences are ({'none: this table is empty' if not t['diff']['rows'] else t['diff']['rows']} rows)\n"
                 f"SELECT status, column_name, count(*) AS cells\nFROM {d}\nGROUP BY status, column_name\nORDER BY cells DESC;")
        bad = [x["name"] for x in c.get("columns", []) if x["status"] == "mismatch"]
        if bad:
            q.append(f"-- 2. The differing values in {bad[0]}, largest first\nSELECT *\nFROM {d}\n"
                     f"WHERE column_name = '{bad[0]}'\nORDER BY abs(difference) DESC\nLIMIT 100;")
    if "sas" in t and "dbx" in t:
        q.append(f"-- {len(q) + 1}. One row on each side{'' if sample else ' (fill in a key you know)'}\n"
                 f"SELECT * FROM {t['sas']['sql']} WHERE {cond};\nSELECT * FROM {t['dbx']['sql']} WHERE {cond};")
    out.append("<pre class=\"sql\">" + e("\n\n".join(q)) + "</pre>")
    spot = "".join("<tr><td></td><td></td><td></td></tr>" for _ in range(5))
    out.append("<h4>Your own spot check</h4><p>Pick five or more rows you know from your own SAS work, not the ones "
               "shown on this page. Look each one up with the last query and record what you found. Rows you choose "
               "test the conversion independently of the tool that produced this certificate.</p>"
               f'<div class="tw"><table class="spot"><thead><tr><th>Row ({e(" + ".join(keys) or "key")})</th>'
               f"<th>Same in Databricks?</th><th>Initials</th></tr></thead><tbody>{spot}</tbody></table></div>")
    return "".join(out)


def _attributes(c: dict) -> str:
    s = c.get("settings") or {}
    ign = s.get("metadata_ignore") or []
    skipped = {x for x in ign if ":" not in x}
    per_col = [x for x in ign if ":" in x]
    md = c.get("metadata") or []
    rows = []
    for check in ("order", "length", "format", "label"):
        found = [m for m in md if m["check"] == check]
        if check in skipped:
            res = '<span class="pill skip">Not compared</span>'
            det = "See deviations."
        elif found:
            res = _status_pill(False)
            det = "; ".join(f'{e(m.get("column", ""))} SAS {e(m["expected"])} vs Databricks {e(m["actual"])}' if check != "order"
                            else "Databricks puts the columns in a different order" for m in found[:6])
        else:
            res = _status_pill(True)
            exc = [x.split(":", 1)[1] for x in per_col if x.split(":", 1)[0] == check]
            det = (f"Except {', '.join(exc)} (see deviations)." if exc else
                   "Same as SAS." if check == "order" else "Same as SAS wherever the reference records it.")
        rows.append([METADATA_WORDS[check].capitalize(), res, det])
    other = [m for m in md if m["check"] not in METADATA_WORDS]
    for m in other:
        rows.append([e(m["check"]), _status_pill(False), e(m.get("column", ""))])
    return _table(["Attribute", "Result", "Detail"], rows, "kv")


def _controls(c: dict) -> str:
    ctl = c["controls"]
    out = []
    r = ctl.get("rows", {})
    out.append(_table(["", "SAS", "Databricks", ""], [["Row count", f"{r.get('expected', 0):,}", f"{r.get('actual', 0):,}",
                                                       _status_pill(r.get("expected") == r.get("actual"))]], "kv", (1, 2)))
    for g in ctl.get("groups", []):
        rows = [[e(fmt_value(x["value"])), f"{x['expected']:,}", f"{x['actual']:,}", _status_pill(x["expected"] == x["actual"])]
                for x in g["rows"]]
        trunc = '<p class="fine">Showing the first 40 values.</p>' if g.get("truncated") else ""
        out.append(f'<h4>Rows by <span class="mono">{e(g["by"])}</span></h4>'
                   + _table([e(g["by"]), "SAS rows", "Databricks rows", ""], rows, "", (1, 2)) + trunc)
    nums = ctl.get("numeric", [])
    if nums:
        def agree(a, b):
            if a is None or b is None:
                return a is None and b is None
            return abs(a - b) <= 1e-9 * max(abs(a), abs(b)) or abs(a - b) < 5e-13
        rows, n_ok = [], 0
        for x in nums:
            ex, ac = x["expected"], x["actual"]
            same = ex["n"] == ac["n"] and all(agree(ex[k], ac[k]) for k in ("sum", "min", "max"))
            n_ok += same
            rows.append([f'<span class="mono">{e(x["column"])}</span>', f"{ex['n']:,} / {ac['n']:,}",
                         f"{fmt_num(ex['sum'])}<br><span class=fine>{fmt_num(ac['sum'])}</span>",
                         f"{fmt_num(ex['min'], x.get('format'))} – {fmt_num(ex['max'], x.get('format'))}"
                         f"<br><span class=fine>{fmt_num(ac['min'], x.get('format'))} – {fmt_num(ac['max'], x.get('format'))}</span>",
                         _status_pill(same)])
        out.append(f"<h4>Numeric columns</h4><p>Count of non-missing values, total, and range for each of the "
                   f"{len(nums)} numeric columns: {n_ok} of {len(nums)} agree. In each cell the first line is SAS, the "
                   "second Databricks; dates are shown as dates.</p>"
                   f'<details><summary>Totals for all {len(nums)} numeric columns</summary>'
                   + _table(["Column", "Non-missing (SAS / DBX)", "Total", "Range", ""], rows, "dense", (1, 2)) + "</details>")
    out.append('<p class="fine">These are the figures a PROC MEANS or PROC FREQ on the SAS output would give, so they can be checked independently.</p>')
    return "".join(out)


def _samples(c: dict, allow: bool) -> str:
    if not allow:
        return ("<p>Row values are withheld for this program because its data is sensitive. The counts above are "
                "aggregates; the migration team can inspect rows inside the secure environment.</p>")
    samples = c.get("samples") or []
    if not samples:
        return "<p>No sample rows were recorded.</p>"
    keys = [k for k in (c.get("keys") or []) if not k.startswith("_")]
    rows = []
    for smp in samples:
        key = ", ".join(f"{k}={fmt_value(smp.get(k))}" for k in keys) or "(row)"
        if "_side" in smp:
            rows.append([f'<span class="mono">{e(key)}</span>', "–", "row " + e(smp["_side"].replace("expected", "SAS").replace("actual", "Databricks")), ""])
            continue
        for col, v in smp.items():
            if isinstance(v, dict) and "expected" in v:
                rows.append([f'<span class="mono">{e(key)}</span>', f'<span class="mono">{e(col)}</span>',
                             e(fmt_value(v["expected"])), e(fmt_value(v["actual"]))])
    return (f"<p>Up to {len(samples)} rows with a difference, as recorded by the comparison.</p>"
            + _table(["Row", "Column", "SAS value", "Databricks value"], rows, "dense", (2, 3)))


def _limits(ev: dict) -> str:
    gt = ev.get("ground_truth") or ""
    ref = GROUND_TRUTH.get(gt, "the reference named in the case")
    items = [
        f"It compares <strong>one run</strong> on one set of inputs. Other input data, such as next month's extract, "
        "can exercise logic this data never reached.",
        ("The reference is output that SAS itself produced from the same inputs (tier G1), the strongest kind of "
         "reference." if gt == "G1" else
         f"The reference is {e(ref)}{f' (tier {e(gt)})' if gt else ''}, not output from a SAS run, so agreement with "
         "it is only as good as the reference."),
        "It covers the data sets listed above, not reports, emails or files the program may also write.",
        "It does not show the converted job runs on its production schedule; that is shown by a period of parallel runs.",
    ]
    if ev.get("parameters"):
        items.insert(1, "It covers the parameter values listed under Traceability, not every value a user could enter.")
    if ev.get("as_of"):
        items.insert(1, f"Anything in the program that depends on today's date was run as of {e(ev['as_of'])}.")
    return '<ul class="limits">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _trace(d: dict) -> str:
    ev = d.get("evidence") or {}
    conv = d.get("conversion") or {}
    rows = [["Program", f'<span class="mono">{e(ev.get("program", ""))}</span>'],
            ["Program fingerprint (sha256)", f'<span class="mono wrap">{e(ev.get("program_sha256", ""))}</span>'],
            ["Notebook fingerprint (sha256)", f'<span class="mono wrap">{e(ev.get("notebook_sha256", ""))}</span>']]
    for ds, sha in (ev.get("expected_sha256") or {}).items():
        rows.append([f"SAS reference {e(ds)} (sha256)", f'<span class="mono wrap">{e(sha)}</span>'])
    rows.append(["Comparison rule", e(ev.get("tolerance", ""))])
    rows.append(["Runtime", f"Spark {e(ev.get('spark_version', ''))}, ANSI SQL mode {e(ev.get('ansi_mode', ''))}"])
    if conv:
        hz = ", ".join(conv.get("hazards") or []) or "none"
        rows.append(["Conversion", f"{conv.get('steps', 0)} steps, {conv.get('auto_converted_pct', 0):g}% converted by rules; "
                     f"SAS behaviours handled explicitly: {e(hz)}"])
    llm = ev.get("llm_repair")
    if llm:
        outcome = f"passed on answer {llm['passed_on_answer']}" if llm.get("passed_on_answer") else "did not pass"
        rows.append(["AI assistance", f"{len(llm.get('units') or [])} step(s) converted by {e(llm.get('model'))}; "
                     f"{llm.get('answers')} answer(s), {outcome}. The verdict above comes from the comparison, not the AI."])
    if ev.get("parameters"):
        rows.append(["Parameters", f'<span class="mono">{e(json.dumps(ev["parameters"]))}</span>'])
    if ev.get("run_at"):
        rows.append(["Compared at", e(ev["run_at"])])
    return _table(["", ""], rows, "kv trace")


def _signoff() -> str:
    fields = ["Reviewer", "Role", "Risk tier", "Decision &nbsp; ☐ Accept &nbsp; ☐ Reject", "Date", "Signature"]
    return ('<div class="sign">' + "".join(f'<div class="sf"><span>{f}</span><i></i></div>' for f in fields)
            + '</div><p class="fine">Accepting means: the outputs listed match SAS for this run, the deviations are '
            'acceptable for how the data is used, and the limits above are understood.</p>')


def render(d: dict, *, standalone: bool = True) -> str:
    ev = d.get("evidence") or {}
    comps = d.get("comparisons") or []
    exe = d.get("execution") or {}
    passed = bool(d.get("passed"))
    program = ev.get("program") or d.get("case", "")
    datasets = ", ".join(c["dataset"] for c in comps) or "its outputs"
    if not exe.get("ok", True):
        verdict = "Could not run"
        sentence = (f"The converted notebook stopped with an error in {exe.get('title', 'a step')}, so no output could "
                    "be compared with SAS.")
    elif passed:
        verdict = "Same as SAS"
        rows = sum(c.get("matched_rows", 0) for c in comps)
        vals = sum(_values_compared(c) for c in comps)
        sentence = (f"The Databricks version of {program} produced the same {datasets} as SAS: "
                    f"{plural(rows, 'row')}, {vals:,} values, every one equal.")
    else:
        verdict = "Not the same as SAS"
        bad = [c for c in comps if not c["passed"]]
        sentence = (f"The Databricks version of {program} did not reproduce SAS for "
                    f"{', '.join(c['dataset'] for c in bad)}: {'; '.join(_problems(bad[0])[:2])}.")
    meta = []
    if ev.get("ground_truth"):
        meta.append(f"Reference: {e(GROUND_TRUTH.get(ev['ground_truth'], ev['ground_truth']))}")
    if ev.get("run_at"):
        meta.append(f"Compared {e(ev['run_at'][:10])}")
    body = [
        f'<header class="top"><div class="tl"><p class="eyebrow">Output data parity certificate</p>'
        f'<h1><span class="mono">{e(program)}</span></h1>'
        f'<p class="sub">{e(ev.get("description") or "")}</p></div>'
        f'<div class="stamp {"ok" if passed else "no"}"><span>{e(verdict)}</span></div></header>',
        f'<p class="lede">{e(sentence)}</p>',
        f'<p class="meta">{" · ".join(meta)}</p>' if meta else "",
        f'<section class="rule"><h2>What “the same” means here</h2><p>{e(tolerance_words(ev.get("decimals")))}</p>'
        "<p>Column order, lengths, display formats and labels are part of the output and are checked too.</p></section>",
    ]
    if not exe.get("ok", True):
        body.append(f'<div class="callout no"><strong>Error</strong><pre>{e(str(exe.get("error", ""))[:1200])}</pre></div>')
    allow = ev.get("allow_row_samples", True)
    body += [_dataset(c, ev, allow) for c in comps]
    body.append(f'<section class="rule"><h2>What this does not prove</h2>{_limits(ev)}</section>')
    body.append(f'<section class="rule"><h2>Traceability</h2>{_trace(d)}</section>')
    body.append(f'<section class="rule"><h2>Sign-off</h2>{_signoff()}</section>')
    title = f"{program} parity"
    page = (f"<title>{e(title)}</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
            f"{_FONTS}\n<style>{_CSS}</style>\n<main class=\"sheet\">{''.join(body)}</main>\n")
    if not standalone:
        return page
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            f"{page.split('<main', 1)[0]}</head><body><main{page.split('<main', 1)[1]}</body></html>\n")


_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
          '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
          'family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">')

_CSS = """
/* layout: one printable sheet; summary first, PROC COMPARE sections per data set, then limits, trace, sign-off */
:root{--paper:#eef1f5;--sheet:#ffffff;--ink:#16202c;--muted:#566273;--rule:#d6dce5;--soft:#f4f6f9;--accent:#24497a;
--ok:#16703f;--ok-bg:#e3f2e8;--no:#b3261e;--no-bg:#fbe8e6;--skip:#7a5a00;--skip-bg:#fbf1d6;
--sans:"IBM Plex Sans",ui-sans-serif,"Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,"Cascadia Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0d1217;--sheet:#151c25;--ink:#e5eaf0;--muted:#9aa6b4;
--rule:#2a3542;--soft:#1b2430;--accent:#93b4e4;--ok:#62c88f;--ok-bg:#11291c;--no:#ff8f82;--no-bg:#32171a;--skip:#e6b850;--skip-bg:#2c2412;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#0d1217;--sheet:#151c25;--ink:#e5eaf0;--muted:#9aa6b4;--rule:#2a3542;--soft:#1b2430;
--accent:#93b4e4;--ok:#62c88f;--ok-bg:#11291c;--no:#ff8f82;--no-bg:#32171a;--skip:#e6b850;--skip-bg:#2c2412;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--sans);
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px}
.sheet{max-width:940px;margin:0 auto;background:var(--sheet);border:1px solid var(--rule);border-radius:6px;
padding:clamp(18px,4vw,44px);display:flex;flex-direction:column;gap:22px}
h1,h2,h3,h4{margin:0;text-wrap:balance;line-height:1.25}
h1{font-size:clamp(24px,4vw,32px);font-weight:600}
h2{font-size:19px;font-weight:600}
h3{font-size:16px;font-weight:600}
h4{font-size:14px;font-weight:600;margin-top:14px}
p{margin:0;max-width:70ch}
.mono{font-family:var(--mono);font-size:.94em}
.wrap{overflow-wrap:anywhere}
.eyebrow{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:600}
.top{display:flex;gap:20px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;
border-bottom:2px solid var(--ink);padding-bottom:18px}
.tl{display:flex;flex-direction:column;gap:6px;min-width:0;flex:1 1 360px}
.sub{color:var(--muted);font-size:14px}
.stamp{border:2.5px solid currentColor;border-radius:4px;padding:10px 16px;font-weight:700;font-size:15px;
letter-spacing:.06em;text-transform:uppercase;transform:rotate(-2deg);white-space:nowrap}
.stamp.ok{color:var(--ok);background:var(--ok-bg)}.stamp.no{color:var(--no);background:var(--no-bg)}
.lede{font-size:18px;line-height:1.5;max-width:62ch}
.meta{color:var(--muted);font-size:13px}
.rule{border-top:1px solid var(--rule);padding-top:18px;display:flex;flex-direction:column;gap:10px}
.ds{border:1px solid var(--rule);border-radius:6px;padding:clamp(14px,3vw,24px);display:flex;flex-direction:column;gap:18px}
.dshead{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.kpi{background:var(--soft);padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.kl{font-size:12px;color:var(--muted)}.kval{font:600 24px/1.15 var(--mono);font-variant-numeric:tabular-nums}.ks{font-size:12px;color:var(--muted)}
.kpi.bad .kval{color:var(--no)}
@media (max-width:620px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.blk{display:flex;flex-direction:column;gap:8px}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--muted);font-size:12px;border-bottom:1px solid var(--rule);padding:6px 10px 6px 0}
td{border-bottom:1px solid var(--rule);padding:7px 10px 7px 0;vertical-align:top}
th.n,td.n{text-align:right}
table.kv td:first-child{color:var(--muted);width:30%}
table.trace td:first-child{width:34%}
table.dense td{padding-block:4px}
.fine{font-size:12px;color:var(--muted)}
.bad{color:var(--no)}
.pill{display:inline-block;font-size:11.5px;font-weight:600;padding:1px 8px;border-radius:999px;white-space:nowrap}
.pill.ok{color:var(--ok);background:var(--ok-bg)}.pill.no{color:var(--no);background:var(--no-bg)}.pill.skip{color:var(--skip);background:var(--skip-bg)}
.dshead .pill{font-size:13px;padding:3px 12px}
.callout{border-radius:4px;padding:12px 16px;display:flex;flex-direction:column;gap:6px}
.callout.no{background:var(--no-bg);color:var(--ink);border-left:3px solid var(--no)}
.callout ul{margin:0;padding-left:20px}.callout pre{white-space:pre-wrap;font:12px/1.45 var(--mono);margin:0}
.hint{font-weight:500}
details{border:1px solid var(--rule);border-radius:4px;padding:8px 12px}
summary{cursor:pointer;font-weight:500;color:var(--accent)}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details[open] summary{margin-bottom:8px}
.devs{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:12px}
.devs li{display:flex;flex-direction:column;gap:6px}
.accept{font-size:13px;color:var(--muted)}
pre.sql{margin:0;padding:12px 14px;background:var(--soft);border:1px solid var(--rule);border-radius:4px;overflow-x:auto;
font:12.5px/1.5 var(--mono);white-space:pre}
table.spot td{height:34px}
.limits{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:75ch}
.sign{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px 28px;margin-top:4px}
.sf{display:flex;flex-direction:column;gap:22px}.sf span{font-size:12.5px;color:var(--muted)}.sf i{border-bottom:1px solid var(--ink)}
@media print{body{background:#fff;padding:0}.sheet{border:0}details{border:0}details>*:not(summary){display:block}}
"""
