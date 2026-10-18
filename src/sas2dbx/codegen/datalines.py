"""DATALINES / CARDS: parse in-stream data at conversion time into literal rows.

Supports list input (with DLM=/DSD), modified list input (``name :$20.``),
column input (``name $ 1-10``) and formatted input (``@1 x date9.``) with
MISSOVER/TRUNCOVER/FLOWOVER semantics.  Values are parsed with the reference
informats, so the resulting rows are exactly what SAS would read.
"""
from __future__ import annotations

import csv
import io

from ..lang import ast as A
from ..runtime import pyfuncs
from .exprs import Unsupported


def compile_datalines(step: A.DataStep, pdv) -> list[str]:
    inp = next((s for s in step.stmts if isinstance(s, A.InputStmt)), None)
    data = next((s for s in step.stmts if isinstance(s, A.DatalinesStmt)), None)
    infile = next((s for s in step.stmts if isinstance(s, A.InfileStmt)), None)
    if inp is None or data is None:
        raise Unsupported("DATALINES without INPUT", step.start_line)
    opts = {k.upper(): v for k, v in (infile.options.items() if infile else [])}
    if infile is not None and infile.fileref.upper() not in ("DATALINES", "CARDS", "DATALINES4", "CARDS4"):
        raise Unsupported(f"INFILE {infile.fileref}", infile.line)
    dsd = "DSD" in opts
    dlm = opts.get("DLM") or opts.get("DELIMITER")
    if dlm is not None:
        dlm = dlm.strip("'\"")
        if dlm.lower() == "09x":
            dlm = "\t"
    elif dsd:
        dlm = ","
    flowover = not any(k in opts for k in ("MISSOVER", "TRUNCOVER", "STOPOVER"))
    firstobs = int(opts.get("FIRSTOBS", "1"))
    lines = data.text.split("\n")
    if lines and lines[-1].strip() == "":
        lines = lines[:-1]
    lines = lines[firstobs - 1:]
    items = inp.items
    hold_all = any(i.kind == "linehold" and i.pointer == "@@" for i in items)
    vars_ = [i for i in items if i.kind == "var"]
    for it in items:
        if it.kind == "pointer" and not (it.pointer or "").startswith("@"):
            raise Unsupported(f"INPUT line pointer {it.pointer}", inp.line)
    column_mode = any(i.col_start is not None for i in vars_) or any(i.kind == "pointer" for i in items)
    rows: list[list] = []
    if column_mode:
        for ln in lines:
            if not ln.strip():
                continue
            rows.append(_read_columns(ln, items, pdv, inp.line))
    else:
        tokens_per_line = [_split(ln, dlm, dsd) for ln in lines if ln.strip() != "" or dsd]
        if hold_all:
            stream = [t for toks in tokens_per_line for t in toks]
            k = len(vars_)
            for i in range(0, len(stream) - k + 1, k):
                rows.append([_value(v, stream[i + j], pdv) for j, v in enumerate(vars_)])
        else:
            it = iter(tokens_per_line)
            for toks in it:
                toks = list(toks)
                while flowover and len(toks) < len(vars_):
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    toks += list(nxt)
                rows.append([_value(v, toks[j] if j < len(toks) else None, pdv) for j, v in enumerate(vars_)])
    names = [pdv.get(v.var).name for v in vars_]
    schema = ", ".join(f"{n} {'string' if pdv.get(n).type == 'char' else 'double'}" for n in names)
    out = [f"_rows = {rows!r}" if len(repr(rows)) < 4000 else "_rows = ["]
    if len(repr(rows)) >= 4000:
        out = ["_rows = ["] + [f"    {r!r}," for r in rows] + ["]"]
    out.append(f"df = spark.createDataFrame(_rows, {schema!r}).withColumn('_sas_seq', F.monotonically_increasing_id())  # DATALINES ({len(rows)} rows)")
    return out


def _split(line: str, dlm: str | None, dsd: bool) -> list[str]:
    if dsd:
        return next(csv.reader(io.StringIO(line), delimiter=dlm or ","))
    if dlm:
        return [t for t in _split_any(line, dlm) if t != ""]
    return line.split()


def _split_any(line: str, chars: str) -> list[str]:
    out, cur = [], []
    for ch in line:
        if ch in chars:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _value(item: A.InputItem, token: str | None, pdv):
    var = pdv.get(item.var)
    if token is None or token.strip() in ("", "."):
        return "" if var.type == "char" else None
    if var.type == "char":
        if item.informat:
            v = pyfuncs.input_value(token, item.informat)
        else:
            v = token
        return str(v)[:var.length].rstrip()
    if item.informat:
        return pyfuncs.input_value(token, item.informat)
    return pyfuncs.input_value(token, "BEST32.")


def _read_columns(line: str, items: list[A.InputItem], pdv, stmt_line: int) -> list:
    pos = 0
    row = []
    for it in items:
        if it.kind == "pointer":
            p = it.pointer or "@1"
            if p.startswith("@") and p[1:].isdigit():
                pos = int(p[1:]) - 1
            elif p.startswith("+") and p[1:].isdigit():
                pos += int(p[1:])
            else:
                raise Unsupported(f"INPUT pointer {p}", stmt_line)
            continue
        if it.kind != "var":
            continue
        var = pdv.get(it.var)
        if it.col_start is not None:
            end = it.col_end or it.col_start
            raw = line[it.col_start - 1:end]
            pos = end
        elif it.informat:
            _, _, w, _ = pyfuncs.parse_format(it.informat)
            if w is None:
                raise Unsupported(f"informat without width in formatted input: {it.informat}", stmt_line)
            raw = line[pos:pos + w]
            pos += w
        else:
            rest = line[pos:]
            stripped = rest.lstrip()
            pos += len(rest) - len(stripped)
            tok = stripped.split(" ", 1)[0]
            raw = tok
            pos += len(tok)
        if var.type == "char":
            val = pyfuncs.input_value(raw, it.informat) if it.informat else raw.strip()
            row.append(str(val)[:var.length].rstrip())
        else:
            row.append(pyfuncs.input_value(raw, it.informat or "BEST32.") if raw.strip() not in ("", ".") else None)
    return row
