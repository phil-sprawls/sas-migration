"""Map SAS function/format/informat/interval specs to runtime SQL function calls.

Shared by the PySpark wrappers (``S.*``) and the Spark SQL emitter so DataFrame
code and SQL code always agree.  A ``CallSpec`` is a small call tree whose
leaves are positional value arguments (``Arg``) or Python literals (``Lit``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import pyfuncs


@dataclass(frozen=True)
class Arg:
    i: int


@dataclass(frozen=True)
class Lit:
    v: object


@dataclass(frozen=True)
class CallSpec:
    name: str
    args: tuple
    returns: str = "num"     # 'num' | 'char'

    def functions(self) -> set[str]:
        out = {self.name}
        for a in self.args:
            if isinstance(a, CallSpec):
                out |= a.functions()
        return out


class UnsupportedSpec(ValueError):
    pass


# ---------------------------------------------------------------------------
# formats
_YMD_SEPS = {"": None, "N": "", "D": "-", "S": "/", "P": ".", "B": " ", "C": ":"}
_DFMT_PATTERNS = {
    "MONYY": (lambda w: "MMMyyyy" if w >= 7 else "MMMyy", True, 5),
    "WORDDATE": (lambda w: "MMMM d, yyyy", False, 18),
    "WORDDATX": (lambda w: "d MMMM yyyy", False, 18),
    "WEEKDATE": (lambda w: "EEEE, MMMM d, yyyy", False, 29),
    "DOWNAME": (lambda w: "EEEE", False, 9),
    "MONNAME": (lambda w: "MMMM", False, 9),
    "E8601DA": (lambda w: "yyyy-MM-dd", False, 10),
    "IS8601DA": (lambda w: "yyyy-MM-dd", False, 10),
    "B8601DA": (lambda w: "yyyyMMdd", False, 8),
    "YEAR": (lambda w: "yyyy" if w >= 4 else "yy", False, 4),
    "YYMON": (lambda w: "yyyyMMM", True, 7),
    "YYQ": (lambda w: "yyyy'Q'q" if w >= 6 else "yy'Q'q", False, 6),
    "YYMM": (lambda w: "yyyy'M'MM", False, 7),
    "MMYY": (lambda w: "MM'M'yyyy", False, 7),
    "QTR": (lambda w: "q", False, 1),
    "MONTH": (lambda w: "M", False, 2),
    "DAY": (lambda w: "d", False, 2),
}


def put_call(spec: str, kind: str = "num") -> CallSpec:
    """PUT(x, spec): returns the runtime call for a built-in format."""
    dollar, name, w, d = pyfuncs.parse_format(spec)
    if dollar or kind == "char":
        width = w if w is not None else 200
        if name in ("", "CHAR", "F"):
            return CallSpec("sas_put_char", (Arg(0), Lit(width)), "char")
        if name == "UPCASE":
            return CallSpec("upper", (CallSpec("sas_put_char", (Arg(0), Lit(width)), "char"),), "char")
        if name == "QUOTE":
            return CallSpec("sas_quote", (Arg(0),), "char")
        raise UnsupportedSpec(f"character format {spec!r}")
    if name in pyfuncs.DATE_FORMATS:
        width = w if w is not None else pyfuncs._DATE_FMT_DEFAULT_W[name]
        if name == "DATE":
            return CallSpec("sas_put_date", (Arg(0), Lit(width)), "char")
        for base, order in (("YYMMDD", "ymd"), ("MMDDYY", "mdy"), ("DDMMYY", "dmy")):
            if name.startswith(base) and name[len(base):] in _YMD_SEPS:
                suffix = name[len(base):]
                sep = _YMD_SEPS[suffix]
                if sep is None:
                    sep = "-" if base == "YYMMDD" else "/"
                if base == "YYMMDD" and suffix == "" and width < 6:
                    raise UnsupportedSpec(spec)
                return CallSpec("sas_put_ymd", (Arg(0), Lit(width), Lit(sep), Lit(order)), "char")
        if name in _DFMT_PATTERNS:
            pat, up, _ = _DFMT_PATTERNS[name]
            return CallSpec("sas_put_dfmt", (Arg(0), Lit(width), Lit(pat(width)), Lit(up)), "char")
        if name == "WEEKDAY":
            return CallSpec("sas_put_dfmt", (Arg(0), Lit(width), Lit("e"), Lit(False)), "char")
        return CallSpec("sas_put_py", (Arg(0), Lit(spec)), "char")
    if name in pyfuncs.DATETIME_FORMATS:
        width = w if w is not None else pyfuncs._DATETIME_FMT_DEFAULT_W[name]
        if name == "DATETIME" and not d:
            return CallSpec("sas_put_datetime", (Arg(0), Lit(width)), "char")
        if name in ("E8601DT", "IS8601DT") and not d:
            return CallSpec("sas_put_isodt", (Arg(0), Lit(width)), "char")
        if name in ("DTDATE",):
            return CallSpec("sas_put_date", (CallSpec("sas_datepart", (Arg(0),)), Lit(width)), "char")
        if name in ("DTYYMMDD",):
            return CallSpec("sas_put_ymd", (CallSpec("sas_datepart", (Arg(0),)), Lit(width), Lit("-"), Lit("ymd")), "char")
        if name in ("E8601DN",):
            return CallSpec("sas_put_dfmt", (CallSpec("sas_datepart", (Arg(0),)), Lit(width), Lit("yyyy-MM-dd"), Lit(False)), "char")
        return CallSpec("sas_put_py", (Arg(0), Lit(spec)), "char")
    if name in pyfuncs.TIME_FORMATS:
        width = w if w is not None else pyfuncs._TIME_FMT_DEFAULT_W[name]
        if name in ("TIME", "TOD", "E8601TM", "IS8601TM") and not d:
            return CallSpec("sas_put_time", (Arg(0), Lit(width), Lit(name != "TIME")), "char")
        return CallSpec("sas_put_py", (Arg(0), Lit(spec)), "char")
    if name == "BEST" or (name == "" and w is None):
        return CallSpec("sas_put_best", (Arg(0), Lit(w or 12)), "char")
    if name in ("", "F"):
        return CallSpec("sas_put_wd", (Arg(0), Lit(w), Lit(d or 0)), "char")
    if name == "Z":
        return CallSpec("sas_put_z", (Arg(0), Lit(w or 1), Lit(d or 0)), "char")
    if name in ("COMMA", "DOLLAR"):
        return CallSpec("sas_put_comma", (Arg(0), Lit(w or 6), Lit(d or 0), Lit(name == "DOLLAR")), "char")
    if name == "PERCENT":
        return CallSpec("sas_put_percent", (Arg(0), Lit(w or 6), Lit(d or 0)), "char")
    if pyfuncs.format_kind(spec) == "user":
        raise UnsupportedSpec(f"user-defined format {spec!r} (needs PROC FORMAT definition)")
    return CallSpec("sas_put_py", (Arg(0), Lit(spec)), "char")


def format_width(spec: str, kind: str = "num") -> int:
    """Width of PUT(x, spec) output (the length SAS gives the result)."""
    dollar, name, w, _ = pyfuncs.parse_format(spec)
    if w is not None:
        return w
    if dollar or kind == "char":
        return 200
    for table in (pyfuncs._DATE_FMT_DEFAULT_W, pyfuncs._DATETIME_FMT_DEFAULT_W, pyfuncs._TIME_FMT_DEFAULT_W):
        if name in table:
            return table[name]
    return 12


def input_call(spec: str) -> CallSpec:
    """INPUT(s, spec)."""
    dollar, name, w, d = pyfuncs.parse_format(spec)
    if dollar:
        if name == "CHAR":
            return CallSpec("substring", (Arg(0), Lit(1), Lit(w or 32767)), "char")
        return CallSpec("sas_input_char", (Arg(0), Lit(w)), "char")
    if name in ("", "BEST", "F", "D", "E"):
        return CallSpec("sas_input_num", (Arg(0), Lit(w), Lit(d or 0)))
    if name in ("COMMA", "DOLLAR", "COMMAX", "DOLLARX"):
        return CallSpec("sas_input_comma", (Arg(0), Lit(w), Lit(d or 0)))
    if name == "PERCENT":
        return CallSpec("sas_input_pct", (Arg(0), Lit(w)))
    if name in ("YYMMDD", "E8601DA", "IS8601DA", "B8601DA", "YYMMDDN"):
        return CallSpec("sas_input_ymd", (Arg(0), Lit(w)))
    if name == "MMDDYY":
        return CallSpec("sas_input_mdy", (Arg(0), Lit(w)))
    if name == "DDMMYY":
        return CallSpec("sas_input_dmy", (Arg(0), Lit(w)))
    if name == "DATE":
        return CallSpec("sas_input_date", (Arg(0), Lit(w)))
    if name == "ANYDTDTE":
        return CallSpec("sas_input_anydtdte", (Arg(0), Lit(w)))
    if name == "DATETIME":
        return CallSpec("sas_input_datetime", (Arg(0), Lit(w)))
    if name in ("E8601DT", "IS8601DT"):
        return CallSpec("sas_input_isodt", (Arg(0), Lit(w)))
    if name in ("TIME", "E8601TM", "IS8601TM"):
        return CallSpec("sas_input_time", (Arg(0), Lit(w)))
    return CallSpec("sas_input_py", (Arg(0), Lit(spec)))


def input_kind(spec: str) -> str:
    return "char" if spec.strip().startswith("$") else "num"


# ---------------------------------------------------------------------------
# intervals
_TIME_SIZES = {"SECOND": 1, "MINUTE": 60, "HOUR": 3600}
_MONTH_SPANS = {"MONTH": 1, "QTR": 3, "SEMIYEAR": 6, "YEAR": 12}


def _interval(interval: str):
    try:
        return pyfuncs.parse_interval(interval)
    except ValueError as e:
        raise UnsupportedSpec(str(e)) from e


def intnx_call(interval: str, alignment: str = "B") -> CallSpec:
    is_dt, base, mult, shift = _interval(interval)
    align = (alignment or "B").strip().upper()[:1] or "B"
    if align not in "BEMS":
        raise UnsupportedSpec(f"INTNX alignment {alignment!r}")
    if base in _TIME_SIZES:
        return CallSpec("sas_intnx_time", (Arg(0), Arg(1), Lit(float(_TIME_SIZES[base] * mult)), Lit(align)))
    start = CallSpec("sas_datepart", (Arg(0),)) if is_dt else Arg(0)
    if base in _MONTH_SPANS:
        inner = CallSpec("sas_intnx_mon", (start, Arg(1), Lit(_MONTH_SPANS[base] * mult), Lit(shift), Lit(align)))
    elif base == "DAY":
        inner = CallSpec("sas_intnx_day", (start, Arg(1), Lit(mult), Lit(align)))
    elif base == "WEEK":
        inner = CallSpec("sas_intnx_week", (start, Arg(1), Lit(mult), Lit(shift), Lit(align)))
    else:
        raise UnsupportedSpec(f"INTNX interval {interval!r}")
    if is_dt:
        return CallSpec("sas_dhms", (inner, Lit(0.0), Lit(0.0), Lit(0.0)))
    return inner


def intck_call(interval: str, method: str = "DISCRETE") -> CallSpec:
    is_dt, base, mult, shift = _interval(interval)
    m = (method or "D").strip().upper()[:1] or "D"
    if base in _TIME_SIZES:
        size = Lit(float(_TIME_SIZES[base] * mult))
        return CallSpec("sas_intck_cont" if m == "C" else "sas_intck_time", (Arg(0), Arg(1), size))
    a = CallSpec("sas_datepart", (Arg(0),)) if is_dt else Arg(0)
    b = CallSpec("sas_datepart", (Arg(1),)) if is_dt else Arg(1)
    if base in _MONTH_SPANS:
        if m == "C":
            return CallSpec("sas_intck_mon_c", (a, b, Lit(_MONTH_SPANS[base] * mult)))
        return CallSpec("sas_intck_mon", (a, b, Lit(_MONTH_SPANS[base] * mult), Lit(shift)))
    if base == "DAY":
        if m == "C":
            return CallSpec("sas_intck_cont", (a, b, Lit(float(mult))))
        return CallSpec("sas_intck_day", (a, b, Lit(mult)))
    if base == "WEEK":
        if m == "C":
            return CallSpec("sas_intck_cont", (a, b, Lit(float(7 * mult))))
        return CallSpec("sas_intck_week", (a, b, Lit(mult), Lit(shift)))
    raise UnsupportedSpec(f"INTCK interval {interval!r}")


def yrdif_call(basis: str = "ACT/ACT") -> CallSpec:
    b = (basis or "ACT/ACT").strip().upper()
    name = {"AGE": "sas_yrdif_age", "ACT/ACT": "sas_yrdif_act", "ACTUAL": "sas_yrdif_act",
            "ACT/360": "sas_yrdif_act360", "ACT/365": "sas_yrdif_act365"}.get(b)
    if name is None:
        raise UnsupportedSpec(f"YRDIF basis {basis!r}")
    return CallSpec(name, (Arg(0), Arg(1)))


# ---------------------------------------------------------------------------
# character classes (Java regex)
def java_char_class(chars, negate: bool = False) -> str:
    chars = "".join(sorted(set(chars), key=ord)) if not isinstance(chars, str) else "".join(dict.fromkeys(chars))
    if not chars:
        return "[^\\s\\S]" if not negate else "[\\s\\S]"
    out = []
    for ch in chars:
        if ch in "\\]^-[&":
            out.append("\\" + ch)
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "[" + ("^" if negate else "") + "".join(out) + "]"


def scan_regex(delims: str | None, modifiers: str = "") -> tuple[str, bool]:
    """(regex, m_modifier) for SCAN/COUNTW."""
    mods = (modifiers or "").lower()
    d = pyfuncs.SCAN_DEFAULT_DELIMS if delims is None or delims == "" else delims
    extra = ""
    if "a" in mods:
        extra += "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if "d" in mods:
        extra += "0123456789"
    if "s" in mods:
        extra += " \t\n\r\f\v"
    cls = java_char_class(d + extra)
    if "m" in mods:
        return cls, True
    return cls + "+", False


def compress_regex(chars: str | None, modifiers: str = "") -> str:
    mods = modifiers or ""
    if chars is None and not mods:
        return "[ ]"
    cls = pyfuncs._char_class(chars if chars is not None else "", mods)
    if chars is None and not (set(mods.lower()) - set("kit")):
        cls |= {" "}
    keep = "k" in mods.lower()
    return java_char_class("".join(sorted(cls, key=ord)), negate=keep)


def perl_to_java(pattern: str) -> tuple[str, str | None]:
    """'/re/i' -> ('(?i)re', None);  's/a/b/' -> ('a', 'b')."""
    p = pattern.strip()
    m = re.match(r"^s/(.*?)(?<!\\)/(.*?)(?<!\\)/([a-z]*)$", p, re.S)
    if m:
        pat, rep, opts = m.groups()
        flags = "(?i)" if "i" in opts else ""
        return flags + pat, re.sub(r"\$(\d)", r"$\1", rep)
    m = re.match(r"^m?/(.*)(?<!\\)/([a-z]*)$", p, re.S)
    if m:
        pat, opts = m.groups()
        flags = "(?i)" if "i" in opts else ""
        return flags + pat, None
    return p, None
