"""Infer variable types from usage when a data set's schema is not in the catalog.

Used for estate analysis (programs without data) and as a fallback during
conversion.  Inferred schemas are flagged ``known=False`` so generated code
passes through columns it cannot see instead of dropping them, and the report
marks the step's types as inferred.
"""
from __future__ import annotations

from ..lang import ast as A
from .schema import Schema, Var

_CHAR_ARG1 = {"SUBSTR", "SUBSTRN", "UPCASE", "LOWCASE", "PROPCASE", "SCAN", "INDEX", "INDEXC", "INDEXW", "FIND",
              "COMPRESS", "COMPBL", "STRIP", "TRIM", "TRIMN", "LEFT", "RIGHT", "LENGTH", "LENGTHN", "LENGTHC",
              "TRANWRD", "TRANSTRN", "TRANSLATE", "REVERSE", "COUNTW", "COUNT", "COUNTC", "VERIFY", "ANYDIGIT",
              "ANYALPHA", "NOTDIGIT", "NOTALPHA", "ANYSPACE", "PRXMATCH", "DEQUOTE", "QUOTE", "INPUT", "INPUTN",
              "INPUTC", "CHAR", "FIRST", "SOUNDEX"}
_NUM_ARG = {"SUM", "MEAN", "MIN", "MAX", "N", "NMISS", "STD", "VAR", "ROUND", "INT", "FLOOR", "CEIL", "ABS", "MOD",
            "SQRT", "EXP", "LOG", "LOG10", "YEAR", "MONTH", "DAY", "QTR", "WEEKDAY", "DATEPART", "TIMEPART", "HOUR",
            "MINUTE", "SECOND", "INTCK", "INTNX", "YRDIF", "MDY", "DHMS", "HMS", "PUTN", "LAG", "DIF"}


def _var(e: A.Expr) -> str | None:
    return e.name.upper() if isinstance(e, A.Var) else None


def infer_usage(stmts: list[A.Stmt]) -> dict[str, str]:
    """{VAR: 'char'|'num'} with evidence from comparisons, operators and function arguments."""
    ev: dict[str, str] = {}
    order: list[str] = []

    def note(name: str | None, typ: str) -> None:
        if not name:
            return
        if name not in order:
            order.append(name)
        if typ == "char" or name not in ev:
            ev[name] = typ if ev.get(name) != "char" else "char"

    def expr(e, hint: str | None = None) -> None:
        if e is None:
            return
        if isinstance(e, A.Var):
            note(e.name.upper(), hint or ev.get(e.name.upper(), "num"))
            return
        if isinstance(e, A.Compare):
            lt = _lit_type(e.left)
            rt = _lit_type(e.right)
            expr(e.left, rt)
            expr(e.right, lt)
            return
        if isinstance(e, A.InList):
            t = next((_lit_type(i) for i in e.items if isinstance(i, A.Expr) and _lit_type(i)), None)
            expr(e.operand, t)
            return
        if isinstance(e, A.Binary):
            if e.op == "||":
                expr(e.left, "char")
                expr(e.right, "char")
            elif e.op in ("+", "-", "*", "/", "**", "MIN", "MAX"):
                expr(e.left, "num")
                expr(e.right, "num")
            else:
                expr(e.left)
                expr(e.right)
            return
        if isinstance(e, A.Unary):
            expr(e.operand, "num" if e.op in ("-", "+") else None)
            return
        if isinstance(e, A.Call):
            for i, a in enumerate(e.args):
                if isinstance(a, A.OfList):
                    for it in a.items:
                        if isinstance(it, A.VarName):
                            note(it.name.upper(), "num" if e.name in _NUM_ARG else ev.get(it.name.upper(), "num"))
                    continue
                if isinstance(a, A.FormatSpec):
                    continue
                if e.name in _CHAR_ARG1 and i == 0:
                    expr(a, "char")
                elif e.name in ("CATS", "CATX", "CAT", "CATT", "COALESCEC"):
                    expr(a, None)
                elif e.name == "PUT" and i == 0 and len(e.args) > 1 and isinstance(e.args[1], A.FormatSpec):
                    expr(a, "char" if e.args[1].spec.startswith("$") else "num")
                elif e.name in _NUM_ARG:
                    expr(a, "num")
                else:
                    expr(a)
            return
        for child in _children(e):
            expr(child)

    def stmt(s) -> None:
        if isinstance(s, A.Assign):
            expr(s.expr)
            if isinstance(s.target, A.Var):
                t = _lit_type(s.expr)
                if t:
                    note(s.target.name.upper(), t)
        elif isinstance(s, A.SumStmt):
            expr(s.expr, "num")
        elif isinstance(s, (A.If, A.SubsetIf, A.Where)):
            expr(s.cond)
            if isinstance(s, A.If):
                stmt(s.then)
                if s.else_ is not None:
                    stmt(s.else_)
        elif isinstance(s, (A.DoBlock, A.DoLoop, A.DoWhile, A.DoUntil)):
            if isinstance(s, (A.DoWhile, A.DoUntil)):
                expr(s.cond)
            for b in s.body:
                stmt(b)
        elif isinstance(s, A.Select):
            expr(s.expr)
            for w in s.whens:
                for v in w.values:
                    expr(v)
                stmt(w.stmt)
            if s.otherwise is not None:
                stmt(s.otherwise)
        elif isinstance(s, A.ByStmt):
            for n, _ in s.vars:
                note(n.upper(), ev.get(n.upper(), "num"))
        elif isinstance(s, A.CallStmt):
            for a in s.args:
                if isinstance(a, A.Expr) and not isinstance(a, A.OfList):
                    expr(a)

    for s in stmts:
        stmt(s)
    return {k: ev.get(k, "num") for k in order}


def _lit_type(e) -> str | None:
    if isinstance(e, A.Str) and not e.suffix:
        return "char"
    if isinstance(e, (A.Num, A.Missing)) or (isinstance(e, A.Str) and e.suffix in ("d", "dt", "t")):
        return "num"
    if isinstance(e, A.Call) and e.name in ("PUT", "PUTC", "PUTN", "CATS", "CATX", "CAT", "UPCASE", "LOWCASE", "STRIP",
                                            "TRIM", "SUBSTR", "SCAN", "COMPRESS", "TRANWRD", "LEFT", "PROPCASE"):
        return "char"
    return None


def _children(e):
    for attr in ("left", "right", "operand", "lo", "hi", "pattern", "sub"):
        x = getattr(e, attr, None)
        if isinstance(x, A.Expr):
            yield x


def inferred_schema(names_types: dict[str, str]) -> Schema:
    return Schema([Var(n, t, 200 if t == "char" else 8) for n, t in names_types.items()], known=False)
