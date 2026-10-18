"""Synthetic edge-case inputs that strengthen the parity gate.

A conversion that matches SAS on the rows at hand can still be wrong on rows
it has not seen.  The fuzzer derives variants of a case's inputs - missing
values, blanks, ties, duplicated rows, boundary numbers - while keeping BY
variables intact so BY-group processing stays valid.  Expected outputs for a
variant come from an oracle: SAS itself (export with ``export_sas``) or,
without SAS, a trusted reference conversion (differential testing).
"""
from __future__ import annotations

import random
import re

from .conformance import ColSpec, TableSpec

_BY_RE = re.compile(r"\b(?:by|class|id)\s+([^;]+);", re.I)
_NOT_NAMES = {"DESCENDING", "NOTSORTED", "GROUPFORMAT", "_ALL_", "_NUMERIC_", "_CHARACTER_"}
_NUM_POOL = [None, None, 0.0, -1.0, 0.5, 1.5, 2.5, -2.5, 10.0, 1234.5678]
_CHAR_POOL = ["", "", "A", "z", "Mixed Case", " lead", "a-b", "x y"]


def protected_columns(sas: str) -> set[str]:
    """BY/CLASS/ID variables: fuzzing them would make the data invalid for the program (unsorted BY groups)."""
    cols: set[str] = set()
    for m in _BY_RE.finditer(sas):
        for w in re.findall(r"[A-Za-z_]\w*", m.group(1)):
            if w.upper() not in _NOT_NAMES:
                cols.add(w.upper())
    return cols


def _kind(col: ColSpec) -> str:
    f = (col.format or "").upper()
    if re.match(r"(DATETIME|E8601DT|IS8601DT)", f):
        return "datetime"
    if re.match(r"(TIME|HHMM|E8601TM)", f):
        return "time"
    if re.match(r"(DATE|YYMMDD|MMDDYY|DDMMYY|E8601DA|IS8601DA|MONYY|WORDDATE|WEEKDATE|YYMON)", f):
        return "date"
    return col.type


def _mutate(v, col: ColSpec, rng: random.Random):
    k = _kind(col)
    if k == "char":
        c = rng.choice(_CHAR_POOL + ["upper", "dup"])
        out = (str(v or "")).upper() if c == "upper" else (str(v or "")) * 2 if c == "dup" else c
        return out[: col.length or 8].rstrip() or None
    base = v if isinstance(v, (int, float)) else None
    if k == "date":
        return rng.choice([None, float((base or 21915) + rng.randint(-400, 400))])
    if k == "datetime":
        return rng.choice([None, float((base or 1.89e9) + rng.randint(-30, 30) * 86400 + rng.randint(0, 86399))])
    if k == "time":
        return rng.choice([None, float(rng.randint(0, 86399))])
    c = rng.choice(_NUM_POOL + ["near", "near"])
    if c == "near":
        return None if base is None else float(base) + rng.choice([0.004, -0.004, 0.005, 1.0, -1.0])
    return c


def fuzz_tables(inputs: dict[str, TableSpec], sas: str, seed: int, *, protect: set[str] | None = None,
                p_cell: float = 0.3) -> dict[str, TableSpec]:
    rng = random.Random(seed)
    keep = protected_columns(sas) | {c.upper() for c in (protect or set())}
    out: dict[str, TableSpec] = {}
    for name, t in inputs.items():
        rows = [list(r) for r in t.rows]
        if rows and rng.random() < 0.5:                       # a duplicate: ties, repeated keys
            i = rng.randrange(len(rows))
            rows.insert(i + 1, list(rows[i]))
        if len(rows) >= 3 and rng.random() < 0.3:
            del rows[rng.randrange(len(rows))]
        if rows and rng.random() < 0.3:                       # one more row in the last BY group
            rows.append(list(rows[-1]))
        for r in rows:
            for j, col in enumerate(t.columns):
                if j < len(r) and col.name.upper() not in keep and rng.random() < p_cell:
                    r[j] = _mutate(r[j], col, rng)
        out[name] = TableSpec(t.columns, rows, t.keys)
    return out
