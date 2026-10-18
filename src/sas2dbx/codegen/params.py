"""Parameter modules: the %INCLUDE'd SAS file of %LET statements that end users edit becomes a parameter notebook.

The converted program %runs that notebook and checks every value before it reads data (``ctx.load_params``).  End
users edit only the parameter notebook: they stay soft-gated from the program, as with the %INCLUDE file today.

A parameter's kind comes from how the program uses it:

* ``text`` / ``number`` / ``date``: a value ("&region", &threshold, "&start"d);
* ``filter``: a whole condition (``where &filter;``).  Users write it in Spark SQL, evaluated on the data as it
  appears in Databricks tables (DATE columns, NULL for blank text);
* ``values``: a list for IN (``in (&regions)``);
* ``columns``: a list of column names (KEEP/DROP, SQL select / group by / order by, PROC SORT BY).

Today's values become the notebook's defaults.  A SAS filter is translated to Spark SQL that selects the same rows
SAS does.  SAS treats a missing value as a value that sorts below everything, so its comparisons are two-valued
where Spark's are three-valued; ``amount < 100`` becomes ``(amount < 100 OR amount IS NULL)``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..lang import ast as A
from ..lang.parser import ParseError, Parser
from ..macro.quoting import unmask
from ..runtime import pyfuncs
from .exprs import Unsupported

KINDS = ("text", "number", "date", "filter", "values", "columns")
_CODE_KINDS = ("filter", "values", "columns")
_LET_RE = re.compile(r"%let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.I)
_LET_STMT_RE = re.compile(r"%let\s+[A-Za-z_][A-Za-z0-9_]*\s*=(?:[^;'\"]|'[^']*'|\"[^\"]*\")*;", re.I)
_REF_RE = re.compile(r"&([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class ParamSpec:
    name: str                    # upper case
    default: str                 # today's value (macro-resolved text)
    raw: str = ""                # the %LET value as written
    comment: str = ""            # the comment next to the %LET (shown to users)
    line: int = 0
    refs: list[str] = field(default_factory=list)   # other parameters its value was built from


def read_param_module(text: str, file: str = "params.sas") -> tuple[dict[str, ParamSpec], list[str]]:
    """The parameters of a module, in order, with today's values; plus notes on anything that is not a %LET."""
    from ..macro.processor import MacroProcessor

    exp = MacroProcessor().expand(text, file)
    specs: dict[str, ParamSpec] = {}
    for m in _LET_RE.finditer(text):
        name = m.group(1).upper()
        end = _stmt_end(text, m.end())
        raw = text[m.end():end].strip()
        line = text.count("\n", 0, m.start()) + 1
        refs = [r.upper() for r in _REF_RE.findall(raw) if r.upper() in specs]
        specs[name] = ParamSpec(name, unmask(exp.symbols.get(name, raw)), raw, _comment_near(text, m.start(), end), line, refs)
    notes = []
    rest = _strip_comments(_LET_STMT_RE.sub("", text)).strip()
    if rest:
        notes.append(f"{file} has statements other than %LET; they are not part of the parameter notebook: "
                     f"{' '.join(rest.split())[:120]}")
    return specs, notes



def _stmt_end(text: str, i: int) -> int:
    """End of a %LET value: the first ';' outside quotes and %STR(...)-style parentheses."""
    depth, quote = 0, ""
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == ";" and depth == 0:
            return i
        i += 1
    return i


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"(^|;)\s*\*[^;]*;", r"\1", text)


def _comment_near(text: str, start: int, end: int) -> str:
    """A /* */ comment on the %LET's own line, else the comment line(s) just above it."""
    line_end = text.find("\n", end)
    same = re.search(r"/\*(.*?)\*/", text[end:line_end if line_end >= 0 else len(text)], re.S)
    if same:
        return " ".join(same.group(1).split())
    above = text[:start].rstrip().split("\n")
    out = []
    for ln in reversed(above):
        s = ln.strip()
        m = re.fullmatch(r"/\*(.*)\*/", s) or re.fullmatch(r"\*(.*);", s)
        if not m:
            break
        out.append(" ".join(m.group(1).split()))
    return " ".join(reversed(out))


def resolve_kind(kinds: set[str] | None) -> tuple[str, str]:
    """One kind per parameter (and a note when its uses disagree)."""
    kinds = set(kinds or ())
    code = [k for k in _CODE_KINDS if k in kinds]
    if len(code) > 1 or (code and kinds - set(code) - {"text"}):
        return code[0], f"used as {' and '.join(sorted(kinds))}: check the converted uses"
    if code:
        return code[0], ""
    for k in ("date", "number"):
        if kinds == {k}:
            return k, ""
    return "text", ""


# ---------------------------------------------------------------------------------------------- SAS filter -> Spark SQL
def sas_filter_to_sql(text: str, columns: dict[str, tuple[str, str | None]] | None = None) -> str:
    """Translate a SAS WHERE condition to a Spark SQL condition that selects the same rows on the data as written to
    Databricks (DATE/TIMESTAMP by format, NULL for blank and missing).  ``columns`` maps upper-case names to
    (type, format).  Raises Unsupported for anything it cannot translate exactly."""
    try:
        p = Parser(text)
        e = p.parse_expr(where=True)
        if not p.at_eof():
            raise Unsupported(f"unexpected text after the condition: {p.peek().value!r}")
    except ParseError as ex:
        raise Unsupported(f"not a SAS condition: {ex}") from ex
    return _FilterSql(columns or {}).cond(e)


_FLIP = {"EQ": "EQ", "NE": "NE", "LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE"}
_NEGATE = {"EQ": "NE", "NE": "EQ", "LT": "GE", "LE": "GT", "GT": "LE", "GE": "LT"}
_SQL_OP = {"EQ": "=", "NE": "<>", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}
_FUNCS1 = {"UPCASE": "upper", "LOWCASE": "lower", "STRIP": "trim", "TRIM": "rtrim", "TRIMN": "rtrim", "LEFT": "ltrim",
           "ABS": "abs"}
_DATE_FUNCS = {"YEAR": "year", "MONTH": "month", "DAY": "day", "QTR": "quarter", "WEEKDAY": "dayofweek"}


class _FilterSql:
    def __init__(self, columns: dict[str, tuple[str, str | None]]):
        self.columns = columns

    # ---- kinds
    def kind(self, e) -> str | None:
        """'num' | 'char' | 'date' | 'datetime' | 'time' | None (unknown column)."""
        if isinstance(e, A.Var):
            c = self.columns.get(e.name.upper())
            if c is None:
                return None
            typ, fmt = c
            if typ == "char":
                return "char"
            k = pyfuncs.format_kind(fmt) if fmt else "num"
            return k if k in ("date", "datetime", "time") else "num"
        if isinstance(e, A.Str):
            return {"d": "date", "dt": "datetime", "t": "time"}.get(e.suffix, "char")
        if isinstance(e, A.Num):
            return "num"
        if isinstance(e, A.Call):
            if e.name in ("TODAY", "DATE"):
                return "date"
            return "char" if e.name in ("UPCASE", "LOWCASE", "STRIP", "TRIM", "TRIMN", "LEFT", "SUBSTR", "COALESCEC") else "num"
        if isinstance(e, A.Binary) and e.op in ("+", "-"):
            kl, kr = self.kind(e.left), self.kind(e.right)
            if "date" in (kl, kr):
                return "num" if kl == kr == "date" else "date"
        return "num" if isinstance(e, (A.Binary, A.Unary, A.Call)) else None

    @staticmethod
    def _missing_literal(e) -> bool:
        return isinstance(e, A.Missing) or (isinstance(e, A.Str) and e.suffix == "" and e.value.strip() == "")

    @staticmethod
    def _literal(e) -> bool:
        return isinstance(e, (A.Num, A.Str)) and not _FilterSql._missing_literal(e)

    @staticmethod
    def _nullable(e) -> bool:
        """Can be missing: refers to a column (a literal or today() - 30 cannot)."""
        if isinstance(e, (A.Var, A.Missing, A.RtMacro)):
            return True
        if isinstance(e, A.Binary):
            return _FilterSql._nullable(e.left) or _FilterSql._nullable(e.right)
        if isinstance(e, A.Unary):
            return _FilterSql._nullable(e.operand)
        if isinstance(e, A.Call):
            return any(_FilterSql._nullable(a) for a in e.args)
        return False

    # ---- conditions (NOT is pushed down: SAS comparisons are two-valued, so NOT (x > 5) is exactly x <= 5)
    def cond(self, e, neg: bool = False) -> str:
        if isinstance(e, A.Unary) and e.op == "NOT":
            return self.cond(e.operand, not neg)
        if isinstance(e, A.Binary) and e.op in ("AND", "OR"):
            op = e.op if not neg else ("OR" if e.op == "AND" else "AND")
            return f"({self.cond(e.left, neg)} {op} {self.cond(e.right, neg)})"
        if isinstance(e, A.Compare):
            return self.compare(e, neg)
        if isinstance(e, A.InList):
            return self.in_list(e, neg != e.negated)
        if isinstance(e, A.Between):
            return self.between(e, neg != e.negated)
        if isinstance(e, A.IsMissing) or (isinstance(e, A.Call) and e.name == "MISSING" and len(e.args) == 1):
            operand = e.operand if isinstance(e, A.IsMissing) else e.args[0]
            negated = (e.negated if isinstance(e, A.IsMissing) else False) != neg
            return f"{self.value(operand)} IS {'NOT ' if negated else ''}NULL"
        if isinstance(e, A.Like):
            return self.like(e.operand, self._pattern(e.pattern, like=True), neg != e.negated)
        if isinstance(e, A.Contains):
            return self.like(e.operand, "%" + self._pattern(e.sub, like=False) + "%", neg != e.negated)
        if isinstance(e, A.Var) and self.kind(e) == "num":        # a numeric value as a truth value
            v = self.value(e)
            return f"({v} IS NULL OR {v} = 0)" if neg else f"({v} IS NOT NULL AND {v} <> 0)"
        raise Unsupported(f"condition {type(e).__name__} in a filter")

    def compare(self, e: A.Compare, neg: bool) -> str:
        op = _NEGATE[e.op] if neg else e.op
        left, right = e.left, e.right
        if e.colon:
            if op not in ("EQ", "NE") or not isinstance(right, A.Str) or right.suffix:
                raise Unsupported("=: prefix comparison other than with a character literal")
            return self.like(left, _like_escape(right.value.rstrip()) + "%", op == "NE")
        if self._missing_literal(left) and not self._missing_literal(right):
            left, right, op = right, left, _FLIP[op]
        if self._missing_literal(right):
            if self._missing_literal(left):
                return "TRUE" if op in ("EQ", "LE", "GE") else "FALSE"
            x = self.value(left)
            return {"EQ": f"{x} IS NULL", "NE": f"{x} IS NOT NULL", "LT": "FALSE", "LE": f"{x} IS NULL",
                    "GT": f"{x} IS NOT NULL", "GE": "TRUE"}[op]
        if not self._nullable(left) and self._nullable(right):
            left, right, op = right, left, _FLIP[op]
        hint = self.kind(left) or self.kind(right)
        x, y = self.value(left, hint=self.kind(right)), self.value(right, hint=hint)
        if not self._nullable(right):       # missing sorts below every value in SAS
            return {"EQ": f"{x} = {y}", "NE": f"({x} <> {y} OR {x} IS NULL)", "LT": f"({x} < {y} OR {x} IS NULL)",
                    "LE": f"({x} <= {y} OR {x} IS NULL)", "GT": f"{x} > {y}", "GE": f"{x} >= {y}"}[op]
        return {"EQ": f"{x} <=> {y}", "NE": f"NOT ({x} <=> {y})",
                "LT": f"({x} < {y} OR ({x} IS NULL AND {y} IS NOT NULL))", "LE": f"({x} <= {y} OR {x} IS NULL)",
                "GT": f"({x} > {y} OR ({x} IS NOT NULL AND {y} IS NULL))", "GE": f"({x} >= {y} OR {y} IS NULL)"}[op]

    def in_list(self, e: A.InList, negated: bool) -> str:
        hint = self.kind(e.operand)
        vals, has_missing = [], False
        for it in e.items:
            if isinstance(it, A.InRange):
                lo, hi = it.lo, it.hi
                if not (isinstance(lo, A.Num) and isinstance(hi, A.Num)) or hi.value - lo.value > 10_000:
                    raise Unsupported("IN range other than a small integer range")
                vals += [str(k) for k in range(int(lo.value), int(hi.value) + 1)]
            elif self._missing_literal(it):
                has_missing = True
            elif self._literal(it):
                vals.append(self.value(it, hint=hint))
            else:
                raise Unsupported("non-constant IN list item in a filter")
        x = self.value(e.operand)
        if not vals:
            return f"{x} IS {'NOT ' if negated else ''}NULL"
        lst = ", ".join(vals)
        if not negated:
            return f"({x} IN ({lst}) OR {x} IS NULL)" if has_missing else f"{x} IN ({lst})"
        return f"{x} NOT IN ({lst})" if has_missing else f"({x} NOT IN ({lst}) OR {x} IS NULL)"

    def between(self, e: A.Between, negated: bool) -> str:
        if self._literal(e.lo) and self._literal(e.hi) and not self._literal(e.operand):
            hint = self.kind(e.operand)
            x, lo, hi = self.value(e.operand), self.value(e.lo, hint=hint), self.value(e.hi, hint=hint)
            return f"({x} NOT BETWEEN {lo} AND {hi} OR {x} IS NULL)" if negated else f"{x} BETWEEN {lo} AND {hi}"
        both = A.Binary("AND", A.Compare("GE", e.operand, e.lo), A.Compare("LE", e.operand, e.hi))
        return self.cond(both, negated)

    def like(self, operand, pattern: str, negated: bool) -> str:
        x = self.value(operand)
        lit = _sql_str(pattern)
        return f"({x} NOT LIKE {lit} OR {x} IS NULL)" if negated else f"{x} LIKE {lit}"

    def _pattern(self, e, like: bool) -> str:
        if not isinstance(e, A.Str) or e.suffix:
            raise Unsupported("LIKE/CONTAINS with a non-literal pattern")
        v = e.value.rstrip() if like else e.value
        return v.replace("\\", "\\\\") if like else _like_escape(v)

    # ---- values
    def value(self, e, hint: str | None = None) -> str:
        if isinstance(e, A.Num):
            return _num(e.value)
        if isinstance(e, A.Str):
            if e.suffix == "":
                return _sql_str(e.value.rstrip())
            if e.suffix == "d":
                v = pyfuncs.input_value(e.value, "DATE11." if "-" in e.value else "DATE9.")
                if v is None:
                    raise Unsupported(f"invalid date literal '{e.value}'d")
                return f"DATE'{pyfuncs.sas_to_date(v).isoformat()}'" if hint in ("date", None) else _num(v)
            if e.suffix == "dt":
                v = pyfuncs.input_value(e.value, "DATETIME40.")
                if v is None:
                    raise Unsupported(f"invalid datetime literal '{e.value}'dt")
                ts = datetime(1960, 1, 1) + timedelta(seconds=v)
                return f"TIMESTAMP'{ts.isoformat(sep=' ')}'" if hint in ("datetime", None) else _num(v)
            if e.suffix == "t":
                v = pyfuncs.input_value(e.value, "TIME20.")
                if v is None:
                    raise Unsupported(f"invalid time literal '{e.value}'t")
                return _num(v)
            raise Unsupported(f"'...'{e.suffix} literal in a filter")
        if isinstance(e, A.Missing):
            return "NULL"
        if isinstance(e, A.Var):
            return e.name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", e.name) else f"`{e.name}`"
        if isinstance(e, A.Unary) and e.op in ("-", "+"):
            v = self.value(e.operand)
            return f"-{v}" if e.op == "-" else v
        if isinstance(e, A.Binary) and e.op in ("+", "-", "*", "/", "**"):
            kl, kr = self.kind(e.left), self.kind(e.right)
            a, b = self.value(e.left), self.value(e.right)
            if e.op in ("+", "-") and "date" in (kl, kr):
                if kl == kr == "date" and e.op == "-":
                    return f"datediff({a}, {b})"
                if kl == "date" and kr == "num":
                    return f"date_add({a}, {b if e.op == '+' else '-' + _paren(b)})"
                if kr == "date" and kl == "num" and e.op == "+":
                    return f"date_add({b}, {a})"
                raise Unsupported("date arithmetic other than date +/- days and date - date")
            if "datetime" in (kl, kr) or "time" in (kl, kr):
                raise Unsupported("datetime arithmetic in a filter")
            if e.op == "/":
                return f"try_divide({a}, {b})"
            if e.op == "**":
                return f"power({a}, {b})"
            return f"({a} {e.op} {b})"
        if isinstance(e, A.Call):
            return self.call(e)
        raise Unsupported(f"{type(e).__name__} in a filter")

    def call(self, e: A.Call) -> str:
        n, args = e.name, e.args
        if n in _FUNCS1 and len(args) == 1:
            return f"{_FUNCS1[n]}({self.value(args[0])})"
        if n in _DATE_FUNCS and len(args) == 1 and self.kind(args[0]) == "date":
            return f"{_DATE_FUNCS[n]}({self.value(args[0])})"
        if n in ("TODAY", "DATE") and not args:
            return "current_date()"
        if n == "SUBSTR" and len(args) in (2, 3):
            return f"substr({', '.join(self.value(a) for a in args)})"
        if n in ("INDEX", "FIND") and len(args) == 2:
            return f"instr({self.value(args[0])}, {self.value(args[1])})"
        if n in ("LENGTH", "LENGTHN") and len(args) == 1:      # SAS LENGTH of a blank value is 1
            return f"coalesce(length({self.value(args[0])}), {1 if n == 'LENGTH' else 0})"
        if n in ("COALESCE", "COALESCEC") and args:
            return f"coalesce({', '.join(self.value(a) for a in args)})"
        raise Unsupported(f"function {n} in a filter")


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))


def _paren(s: str) -> str:
    return s if re.fullmatch(r"[\w.`]+", s) else f"({s})"


def _sql_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------------------------------------- notebook defaults
def default_value(spec: ParamSpec, kind: str, columns: dict[str, tuple[str, str | None]] | None):
    """Today's value as the parameter notebook's default; returns (python value, note)."""
    text = spec.default.strip()
    if kind == "filter":
        try:
            return sas_filter_to_sql(text, columns), ""
        except Unsupported as e:
            return "", f"today's SAS filter could not be translated ({e}); write it in Spark SQL: {text}"
    if kind == "values":
        return [_number(x) if _is_num(x) and not _quoted(x, text) else x for x in _list_items(text)], ""
    if kind == "columns":
        return [c for c in re.split(r"[\s,]+", text) if c], ""
    if kind == "date":
        if re.fullmatch(r"\d{4}-\d\d-\d\d", text):
            return text, ""
        v = pyfuncs.input_value(text, "DATE11." if "-" in text else "DATE9.")
        if v is None:
            return text, "not a date today: check the value"
        return pyfuncs.sas_to_date(v).isoformat(), ""
    if kind == "number":
        return (_number(text), "") if _is_num(text) else (text, "not a number today: check the value")
    return text, ""


_ITEM_RE = re.compile(r"'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\"|([^\s,]+)")


def _list_items(text: str) -> list[str]:
    out = []
    for m in _ITEM_RE.finditer(text):
        if m.group(1) is not None:
            out.append(m.group(1).replace("''", "'"))
        elif m.group(2) is not None:
            out.append(m.group(2).replace('""', '"'))
        else:
            out.append(m.group(3))
    return out


def _quoted(item: str, text: str) -> bool:
    return f"'{item}'" in text or f'"{item}"' in text


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _number(s: str):
    v = float(s)
    return int(v) if v.is_integer() and abs(v) < 1e15 else v


_HELP = {
    "text": "text",
    "number": "a number",
    "date": "a date, YYYY-MM-DD",
    "filter": "a Spark SQL condition on the input's columns (\"\" = no filter)",
    "values": "a list of values",
    "columns": "a list of column names",
}


def render_param_notebook(module_file: str, program_file: str, specs: dict[str, ParamSpec], kinds: dict[str, str],
                          defaults: dict[str, tuple], notes: list[str]) -> str:
    """The notebook end users edit: one PARAMETERS dict, today's values as defaults, one comment per parameter."""
    out = ["# Databricks notebook source", "# MAGIC %md",
           f"# MAGIC # Parameters: `{os.path.basename(program_file)}`",
           f"# MAGIC Converted from `{os.path.basename(module_file)}` by sas2dbx. Edit the values below, then run the job.",
           "# MAGIC The program checks every value before it reads any data.",
           "# MAGIC",
           "# MAGIC **Filters are Spark SQL conditions**, evaluated on the data as it appears in Databricks tables:",
           "# MAGIC dates are `DATE` values (`order_date >= DATE'2024-01-01'`) and blank text is `NULL`.",
           "# MAGIC Unlike SAS, a comparison with a missing value is never true: write `OR amount IS NULL` to keep",
           "# MAGIC missing values (today's SAS filters were translated that way).",
           "", "# COMMAND ----------", "", "PARAMETERS = {"]
    for name, spec in specs.items():
        kind = kinds.get(name, "text")
        value, note = defaults[name]
        what = _HELP[kind] + (f" - {spec.comment}" if spec.comment else "")
        out.append(f"    # {name.lower()}: {what}")
        if spec.refs:
            out.append(f"    #   in SAS this was built from {', '.join('&' + r.lower() for r in spec.refs)}: "
                       "change it together with them")
        if note:
            out.append(f"    #   NOTE: {note}")
        out.append(f"    {name.lower()!r}: {_py(value)},")
    out.append("}")
    if notes:
        out += ["", "# COMMAND ----------", ""] + [f"# NOTE: {n}" for n in notes]
    return "\n".join(out) + "\n"


def edit_param_notebook(text: str, values: dict) -> str:
    """The parameter notebook with some values changed the way a user edits it: only the values are replaced, so the
    comments and layout stay.  Raises KeyError for a name the notebook does not define."""
    import ast

    data = text.encode()                  # ast offsets are UTF-8 byte offsets
    starts = [0]
    for line in data.split(b"\n"):
        starts.append(starts[-1] + len(line) + 1)
    wanted = {k.lower(): v for k, v in values.items()}
    found = {}
    for node in ast.parse(text).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "PARAMETERS" for t in node.targets)):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and str(k.value).lower() in wanted:
                    found[str(k.value).lower()] = v
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise KeyError(f"the parameter notebook has no {', '.join(missing)}")
    for name, v in sorted(found.items(), key=lambda kv: (kv[1].lineno, kv[1].col_offset), reverse=True):
        start = starts[v.lineno - 1] + v.col_offset
        end = starts[v.end_lineno - 1] + v.end_col_offset
        before, after = data[:start].rstrip(), data[end:].lstrip()
        if before.endswith(b"(") and after.startswith(b")"):      # long text: _py's parentheses are not in the node
            start, end = len(before) - 1, len(data) - len(after) + 1
        data = data[:start] + _py(wanted[name]).encode() + data[end:]
    return data.decode()


def _py(v) -> str:
    if isinstance(v, str) and len(v) > 70:
        return "(\n        " + "\n        ".join(repr(chunk) for chunk in _chunks(v, 70)) + "\n    )"
    return repr(v)


def _chunks(s: str, n: int) -> list[str]:
    """Split long text on spaces into adjacent string literals (Python joins them)."""
    words, out, cur = s.split(" "), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > n:
            out.append(cur + " ")
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return out + [cur]
