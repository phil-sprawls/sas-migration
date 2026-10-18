"""SAS expression AST -> PySpark Column source code (typed, SAS semantics).

The compiler is type-directed: every result is a ``CE`` with its SAS type
(num | char | bool), maximum character length and constant value when known,
so the generator can choose the cheapest faithful translation (e.g. native
``==`` for character comparisons, null-aware ``S.lt`` for numerics).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..lang import ast as A
from ..lang.lexer import OPAQUE, RT_CLOSE, RT_OPEN
from ..runtime import dispatch as D
from ..runtime import pyfuncs

_NO = object()


class Unsupported(Exception):
    """A construct the deterministic converter will not translate (goes to LLM/manual)."""

    def __init__(self, message: str, line: int = 0):
        super().__init__(message)
        self.line = line


@dataclass
class CE:
    code: str
    type: str                     # num | char | bool
    length: int | None = None     # char: maximum length
    const: object = _NO           # python constant when known (float | None | str)
    padded: bool = False          # char value includes its trailing blank padding
    nonnull: bool = False         # num: can never be missing

    @property
    def is_const(self) -> bool:
        return self.const is not _NO


class Scope(Protocol):
    def var_ref(self, name: str, line: int) -> CE: ...
    def var_attr(self, name: str): ...
    def array_elements(self, name: str) -> list[str] | None: ...
    def first_last(self, which: str, var: str, line: int) -> CE: ...
    def lag(self, n: int, arg: CE, dif: bool, line: int) -> CE: ...
    def runtime_macro(self, name: str, want: str, line: int) -> CE: ...
    def param(self, name: str, kind: str, line: int) -> bool: ...   # a parameter-module variable (records its use)
    def user_format(self, name: str) -> str | None: ...
    def hazard(self, hid: str, detail: str, line: int) -> None: ...


def q(s: str) -> str:
    """Python string literal."""
    return repr(s)


def col(name: str) -> str:
    return f"F.col({q(name)})"


def num_lit(v) -> CE:
    if v is None:
        return CE("S.MISSING", "num", const=None)
    v = float(v)
    text = repr(v) if v == v else "float('nan')"
    return CE(text, "num", const=v, nonnull=True)


def str_lit(s: str) -> CE:
    return CE(q(s), "char", len(s) or 1, const=s, padded=True)


def split_param_filters(cond: A.Expr, is_param) -> tuple[A.Expr | None, list[str]]:
    """``where a and &filter``: the top-level AND conjuncts that are filter parameters (Spark SQL written by users,
    applied with ctx.param_filter) and the SAS condition that is left.  ``is_param(name)`` says (and records) whether
    a run-time macro value is a parameter-module variable.  A filter parameter under OR/NOT cannot be split off."""
    conj: list[A.Expr] = []

    def flat(e: A.Expr) -> None:
        if isinstance(e, A.Binary) and e.op == "AND":
            flat(e.left)
            flat(e.right)
        else:
            conj.append(e)

    flat(cond)
    names = [c.name.upper() for c in conj if isinstance(c, A.RtMacro) and is_param(c.name)]
    rest = [c for c in conj if not (isinstance(c, A.RtMacro) and c.name.upper() in names)]
    for c in rest:
        for n in _bool_operands(c):
            if is_param(n):
                raise Unsupported(f"filter parameter &{n} combined with OR/NOT: not converted yet", c.line)
    expr = None
    for c in rest:
        expr = c if expr is None else A.Binary("AND", expr, c, line=c.line)
    return expr, names


def _bool_operands(e: A.Expr) -> list[str]:
    """Run-time macro values used as whole conditions inside ``e`` (operands of AND/OR/NOT)."""
    if isinstance(e, A.RtMacro):
        return [e.name]
    if isinstance(e, A.Binary) and e.op in ("AND", "OR"):
        return _bool_operands(e.left) + _bool_operands(e.right)
    if isinstance(e, A.Unary) and e.op == "NOT":
        return _bool_operands(e.operand)
    return []


_RT_RE = re.compile(f"{re.escape(RT_OPEN)}([^{re.escape(RT_CLOSE)}]*){re.escape(RT_CLOSE)}")


def rt_text(value: str, line: int, register) -> str:
    """Python expression for literal text that embeds run-time macro values ("&r" after CALL SYMPUTX / SELECT INTO,
    a run-time parameter, &SYSDATE9): resolved by ``ctx.symget`` when the cell runs, as SAS resolves it when the
    step compiles.  ``register(name)`` records each name used."""
    parts, pos = [], 0
    for m in _RT_RE.finditer(value):
        name = m.group(1)
        if name.startswith(OPAQUE):
            raise Unsupported(f"value of %{name[1:]}(...) is only known at run time", line)
        register(name)
        if m.start() > pos:
            parts.append(q(value[pos:m.start()]))
        parts.append(f"ctx.symget({q(name.upper())})")
        pos = m.end()
    if pos < len(value):
        parts.append(q(value[pos:]))
    return " + ".join(parts) or "''"


def as_col(ce: CE) -> str:
    """Code that is guaranteed to be a Column (constants wrapped in F.lit)."""
    if ce.is_const:
        if ce.const is None:
            return "S.MISSING"
        if ce.type == "char":
            return f"F.lit({q(ce.const)})"
        return f"F.lit({ce.code})"
    return ce.code


def _paren(code: str) -> str:
    return code if code.startswith(("F.", "S.", "(")) or code.replace(".", "").replace("-", "").isdigit() else f"({code})"


# result-length rules for character functions (DATA step, SAS 9.4)
_LEN_OF_ARG1 = {"SUBSTR", "SUBSTRN", "UPCASE", "LOWCASE", "PROPCASE", "LEFT", "RIGHT", "TRIM", "TRIMN", "STRIP",
                "COMPRESS", "COMPBL", "TRANSLATE", "REVERSE", "DEQUOTE", "SCAN", "KSUBSTR", "KUPCASE", "KLEFT", "KTRIM"}
_LEN_200 = {"CAT", "CATS", "CATT", "CATX", "TRANWRD", "TRANSTRN", "REPEAT", "QUOTE", "COALESCEC", "PRXCHANGE",
            "IFC", "CHOOSEC", "VVALUE", "VLABEL", "TRANSLATE_200"}

NUMERIC_AGG_FUNCS = {"SUM": "sum", "MEAN": "mean", "MIN": "min", "MAX": "max", "N": "n", "NMISS": "nmiss",
                     "STD": "std", "VAR": "var", "RANGE": "range", "MEDIAN": "median"}
_SIMPLE_NUM1 = {"ABS": "abs", "SIGN": "sign", "SQRT": "sqrt", "EXP": "exp", "LOG": "log", "LOG10": "log10",
                "LOG2": "log2", "INT": "int_", "FLOOR": "floor", "CEIL": "ceil", "YEAR": "year", "MONTH": "month",
                "DAY": "day", "QTR": "qtr", "WEEKDAY": "weekday", "DATEPART": "datepart", "TIMEPART": "timepart",
                "HOUR": "hour", "MINUTE": "minute", "SECOND": "second", "INTZ": "int_"}
_ANY_FUNCS = {"ANYDIGIT", "ANYALPHA", "ANYALNUM", "ANYSPACE", "ANYUPPER", "ANYLOWER", "ANYPUNCT", "NOTDIGIT",
              "NOTALPHA", "NOTSPACE", "NOTALNUM"}


class ExprCompiler:
    def __init__(self, scope: Scope, *, where: bool = False):
        self.scope = scope
        self.where = where

    # ------------------------------------------------------------------ entry points
    def value(self, e: A.Expr) -> CE:
        ce = self._value(e)
        if ce.type == "bool":
            return CE(f"S.b2n({ce.code})", "num", nonnull=True)
        return ce

    def num(self, e: A.Expr) -> CE:
        ce = self.value(e)
        if ce.type == "char":
            self.scope.hazard("H-AUTOCONV", "character value used where a number is required", e.line)
            if ce.is_const:
                v = pyfuncs.input_value(ce.const, "BEST32.")
                return num_lit(v)
            return CE(f"S.char_to_num({ce.code})", "num")
        return ce

    def char(self, e: A.Expr) -> CE:
        ce = self.value(e)
        if ce.type == "num":
            self.scope.hazard("H-AUTOCONV", "numeric value used where text is required (BEST12. conversion)", e.line)
            if ce.is_const:
                return CE(q(pyfuncs.put(ce.const, "BEST12.")), "char", 12, const=pyfuncs.put(ce.const, "BEST12."), padded=True)
            return CE(f"S.num_to_char({ce.code})", "char", 12, padded=True)
        return ce

    def cond(self, e: A.Expr) -> str:
        """Boolean Column code that is never NULL."""
        ce = self._value(e)
        if ce.type == "bool":
            return ce.code
        if ce.type == "char":
            self.scope.hazard("H-AUTOCONV", "character value used as a condition", e.line)
            return f"S.truth(S.char_to_num({ce.code}))"
        if ce.is_const:
            return "F.lit(True)" if (ce.const is not None and ce.const != 0) else "F.lit(False)"
        if ce.nonnull:
            return f"({ce.code} != 0)"
        return f"S.truth({ce.code})"

    # ------------------------------------------------------------------ dispatch
    def _value(self, e: A.Expr) -> CE:
        m = getattr(self, "_e_" + type(e).__name__, None)
        if m is None:
            raise Unsupported(f"expression {type(e).__name__}", e.line)
        return m(e)

    def _e_Num(self, e: A.Num) -> CE:
        return num_lit(e.value)

    def _e_Str(self, e: A.Str) -> CE:
        if RT_OPEN in e.value:
            if e.suffix not in ("", "d", "dt", "t"):
                raise Unsupported(f"run-time macro value inside a '...'{e.suffix} literal", e.line)
            want = {"d": "date", "dt": "datetime", "t": "time"}.get(e.suffix, "char")
            text = rt_text(e.value, e.line, lambda nm: self.scope.runtime_macro(nm, want, e.line))
            if e.suffix:                       # '&start'd: parsed when the cell runs; an invalid value fails like SAS
                return CE(f"F.lit(S.{e.suffix}({text}))", "num", nonnull=True)
            # SAS sizes the literal after resolution, so no static length fits: never truncate the run-time value
            return CE(f"F.lit({text})", "char", 32767, padded=True)
        if e.suffix == "d":
            v = pyfuncs.input_value(e.value, "DATE11." if "-" in e.value else "DATE9.")
            if v is None:
                raise Unsupported(f"invalid date literal '{e.value}'d", e.line)
            return CE(f"S.d({q(e.value)})", "num", const=v, nonnull=True)
        if e.suffix == "dt":
            v = pyfuncs.input_value(e.value, "DATETIME40.")
            if v is None:
                raise Unsupported(f"invalid datetime literal '{e.value}'dt", e.line)
            return CE(f"S.dt({q(e.value)})", "num", const=v, nonnull=True)
        if e.suffix == "t":
            v = pyfuncs.input_value(e.value, "TIME20.")
            if v is None:
                raise Unsupported(f"invalid time literal '{e.value}'t", e.line)
            return CE(f"S.t({q(e.value)})", "num", const=v, nonnull=True)
        if e.suffix == "b":
            raise Unsupported("bit-testing constants", e.line)
        # SAS '' is a single blank
        return str_lit(e.value if e.value != "" else " ")

    def _e_Missing(self, e: A.Missing) -> CE:
        if e.kind not in (".", ".Z", "._"):
            self.scope.hazard("H-SPECIAL-MISSING", f"special missing {e.kind} treated as missing", e.line)
        return num_lit(None)

    def _e_Var(self, e: A.Var) -> CE:
        return self.scope.var_ref(e.name, e.line)

    def _e_FirstLast(self, e: A.FirstLast) -> CE:
        return self.scope.first_last(e.which, e.var, e.line)

    def _e_RtMacro(self, e: A.RtMacro) -> CE:
        return self.scope.runtime_macro(e.name, "num", e.line)

    def _e_Index(self, e: A.Index) -> CE:
        elems = self.scope.array_elements(e.array)
        if elems is None:
            raise Unsupported(f"unknown array {e.array}", e.line)
        if len(e.indices) != 1:
            raise Unsupported("multi-dimensional arrays", e.line)
        idx = self.num(e.indices[0])
        if idx.is_const and idx.const is not None:
            k = int(idx.const)
            if not 1 <= k <= len(elems):
                raise Unsupported(f"array subscript {k} out of range for {e.array}", e.line)
            return self.scope.var_ref(elems[k - 1], e.line)
        refs = [self.scope.var_ref(n, e.line) for n in elems]
        typ = refs[0].type
        arr = "F.array(" + ", ".join(as_col(r) for r in refs) + ")"
        code = f"F.when(({idx.code} >= 1) & ({idx.code} <= {len(elems)}), F.element_at({arr}, {idx.code}.cast('int')))"
        return CE(code, typ, max((r.length or 0) for r in refs) if typ == "char" else None)

    def _e_Unary(self, e: A.Unary) -> CE:
        if e.op == "NOT":
            return CE(f"~({self.cond(e.operand)})", "bool")
        v = self.num(e.operand)
        if e.op == "+":
            return v
        if v.is_const:
            return num_lit(None if v.const is None else -v.const)
        return CE(f"(-{_paren(v.code)})", "num", nonnull=v.nonnull)

    def _e_Binary(self, e: A.Binary) -> CE:
        op = e.op
        if op in ("AND", "OR"):
            a, b = self.cond(e.left), self.cond(e.right)
            return CE(f"({a}) {'&' if op == 'AND' else '|'} ({b})", "bool")
        if op == "||":
            return self._concat(e)
        a, b = self.num(e.left), self.num(e.right)
        if a.is_const and b.is_const:
            if a.const is None or b.const is None:
                return num_lit(None)
            try:
                if op == "+":
                    return num_lit(a.const + b.const)
                if op == "-":
                    return num_lit(a.const - b.const)
                if op == "*":
                    return num_lit(a.const * b.const)
                if op == "/":
                    return num_lit(None if b.const == 0 else a.const / b.const)
            except OverflowError:
                pass
        if op in ("+", "-", "*"):
            if a.is_const and a.const is None or b.is_const and b.const is None:
                return num_lit(None)
            left = as_col(a) if (a.is_const and b.is_const) else a.code
            return CE(f"({left} {op} {b.code})", "num", nonnull=a.nonnull and b.nonnull)
        if op == "/":
            if b.is_const and b.const not in (None, 0):
                return CE(f"({as_col(a) if a.is_const else a.code} / {b.code})", "num", nonnull=a.nonnull)
            self.scope.hazard("H-DIV0", "division (missing on zero divisor)", e.line)
            return CE(f"S.div({as_col(a)}, {as_col(b)})", "num")
        if op == "**":
            return CE(f"S.pow({as_col(a)}, {as_col(b)})", "num")
        if op == "MIN":
            return CE(f"S.minop({as_col(a)}, {as_col(b)})", "num")
        if op == "MAX":
            return CE(f"S.maxop({as_col(a)}, {as_col(b)})", "num")
        raise Unsupported(f"operator {op}", e.line)

    # ------------------------------------------------------------------ concatenation
    def _flatten_concat(self, e: A.Expr, out: list[A.Expr]) -> None:
        if isinstance(e, A.Binary) and e.op == "||":
            self._flatten_concat(e.left, out)
            self._flatten_concat(e.right, out)
        else:
            out.append(e)

    def _concat(self, e: A.Binary) -> CE:
        parts: list[A.Expr] = []
        self._flatten_concat(e, parts)
        codes = []
        total = 0
        consts = []
        for i, p in enumerate(parts):
            ce = self.char(p)
            last = i == len(parts) - 1
            if not last and not ce.padded and not ce.is_const and ce.length and isinstance(p, A.Var):
                self.scope.hazard("H-CONCAT-PAD", f"{p.name} is blank-padded to its length {ce.length} inside ||", p.line)
                ce = CE(f"S.pad({ce.code}, {ce.length})", "char", ce.length, padded=True)
            codes.append(as_col(ce) if not ce.is_const else q(ce.const))
            consts.append(ce.const if ce.is_const else _NO)
            total += ce.length or 200
        if all(c is not _NO for c in consts):
            s = "".join(consts)
            return CE(q(s), "char", len(s), const=s, padded=True)
        return CE("S.concat(" + ", ".join(codes) + ")", "char", total, padded=True)

    # ------------------------------------------------------------------ comparisons
    def _e_Compare(self, e: A.Compare) -> CE:
        return CE(self._compare(e.op, e.left, e.right, e.colon, e.line), "bool")

    def _compare(self, op: str, left: A.Expr, right: A.Expr, colon: bool, line: int) -> str:
        # missing-literal special cases
        if isinstance(right, A.Missing) or isinstance(left, A.Missing):
            if isinstance(left, A.Missing):
                left, right = right, left
                op = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE"}.get(op, op)
            v = self._value(left)
            if v.type == "bool":
                v = self.value(left)
            if v.type == "char":
                raise Unsupported("comparison of character value with numeric missing", line)
            c = v.code
            if v.nonnull:
                return {"EQ": "F.lit(False)", "NE": "F.lit(True)", "GT": "F.lit(True)", "GE": "F.lit(True)",
                        "LT": "F.lit(False)", "LE": "F.lit(False)"}[op]
            return {"EQ": f"{_paren(c)}.isNull()", "NE": f"{_paren(c)}.isNotNull()", "GT": f"{_paren(c)}.isNotNull()",
                    "GE": "F.lit(True)", "LT": "F.lit(False)", "LE": f"{_paren(c)}.isNull()"}[op]
        a, b = self._value(left), self._value(right)
        if a.type == "bool":
            a = self.value(left)
        if b.type == "bool":
            b = self.value(right)
        if a.type == "char" and b.type == "char":
            return self._char_compare(op, a, b, colon)
        if a.type == "char" or b.type == "char":
            if a.type == "char" and a.is_const:
                a = self.num(left)
            elif b.type == "char" and b.is_const:
                b = self.num(right)
            else:
                a, b = self.num(left), self.num(right)
        return self._num_compare(op, a, b, line)

    def _char_compare(self, op: str, a: CE, b: CE, colon: bool) -> str:
        def side(ce: CE) -> str:
            if ce.is_const:
                return q(str(ce.const).rstrip(" "))
            return ce.code
        if colon:
            if op not in ("EQ", "NE"):
                raise Unsupported("prefix comparison other than =: / ^=:")
            if b.is_const and not a.is_const:
                code = f"{_paren(a.code)}.startswith({side(b)})"
            else:
                code = f"S.prefix_eq({as_col(a)}, {as_col(b)})"
            return code if op == "EQ" else f"~({code})"
        pyop = {"EQ": "==", "NE": "!=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}[op]
        if a.is_const and b.is_const:
            x, y = str(a.const).rstrip(" "), str(b.const).rstrip(" ")
            res = {"EQ": x == y, "NE": x != y, "LT": x < y, "LE": x <= y, "GT": x > y, "GE": x >= y}[op]
            return "F.lit(True)" if res else "F.lit(False)"
        if a.is_const:
            flip = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE"}.get(op, op)
            return self._char_compare(flip, b, a, colon)
        bcode = side(b) if b.is_const else (f"F.rtrim({b.code})" if b.padded else b.code)
        acode = f"F.rtrim({a.code})" if a.padded else a.code
        return f"({_paren(acode)} {pyop} {bcode})"

    def _num_compare(self, op: str, a: CE, b: CE, line: int) -> str:
        if a.is_const and b.is_const:
            x, y = a.const, b.const
            return "F.lit(True)" if _sas_cmp(op, x, y) else "F.lit(False)"
        if a.is_const and not b.is_const:
            flip = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE"}.get(op, op)
            return self._num_compare(flip, b, a, line)
        if op in ("EQ", "NE") and a.nonnull and b.nonnull:
            return f"({a.code} {'==' if op == 'EQ' else '!='} {b.code})"
        if op == "EQ":
            return f"{_paren(a.code)}.eqNullSafe({as_col(b) if not b.is_const else b.code})"
        if op == "NE":
            return f"~{_paren(a.code)}.eqNullSafe({as_col(b) if not b.is_const else b.code})"
        if a.nonnull and b.nonnull:
            pyop = {"LT": "<", "LE": "<=", "GT": ">", "GE": ">="}[op]
            return f"({a.code} {pyop} {b.code})"
        if op in ("LT", "LE") and not a.nonnull:
            self.scope.hazard("H-MISS-CMP", "missing values satisfy '<' / '<=' in SAS", line)
        fn = {"LT": "lt", "LE": "le", "GT": "gt", "GE": "ge"}[op]
        return f"S.{fn}({a.code}, {b.code})"

    def _e_InList(self, e: A.InList) -> CE:
        v = self._value(e.operand)
        if v.type == "bool":
            v = self.value(e.operand)
        params = [it.name for it in e.items if isinstance(it, A.RtMacro)]   # in (&regions): a value-list parameter
        for n in params:
            if not self.scope.param(n, "values", e.line):
                raise Unsupported(f"IN list built from run-time macro variable &{n}", e.line)
        items = [it for it in e.items if not isinstance(it, A.RtMacro)]
        if v.type == "char":
            vals = []
            for it in items:
                if isinstance(it, A.InRange):
                    raise Unsupported("range in character IN list", e.line)
                c = self.char(it)
                if not c.is_const:
                    raise Unsupported("non-constant IN list item", e.line)
                vals.append(str(c.const).rstrip(" "))
            args = [q(x) for x in vals] + [f"*ctx.param_values({q(n)})" for n in params]
            code = f"{_paren(v.code)}.isin({', '.join(args)})"
        else:
            vals2: list = []
            for it in items:
                if isinstance(it, A.InRange):
                    lo, hi = self.num(it.lo), self.num(it.hi)
                    if not (lo.is_const and hi.is_const) or lo.const is None or hi.const is None:
                        raise Unsupported("non-constant IN range", e.line)
                    vals2.extend(float(k) for k in range(int(lo.const), int(hi.const) + 1))
                    continue
                c = self.num(it)
                if not c.is_const:
                    raise Unsupported("non-constant IN list item", e.line)
                vals2.append(c.const)
            args = ["None" if x is None else repr(float(x)) for x in vals2] + [f"*ctx.param_values({q(n)}, num=True)" for n in params]
            code = f"S.isin({v.code}, {', '.join(args)})"
        return CE(f"~({code})" if e.negated else code, "bool")

    def _e_Between(self, e: A.Between) -> CE:
        lo = self._compare("GE", e.operand, e.lo, False, e.line)
        hi = self._compare("LE", e.operand, e.hi, False, e.line)
        code = f"({lo}) & ({hi})"
        return CE(f"~({code})" if e.negated else code, "bool")

    def _e_IsMissing(self, e: A.IsMissing) -> CE:
        v = self.value(e.operand)
        code = f"({_paren(v.code)} == '')" if v.type == "char" else f"{_paren(v.code)}.isNull()"
        return CE(f"~({code})" if e.negated else code, "bool")

    def _e_Like(self, e: A.Like) -> CE:
        v, p = self.char(e.operand), self.char(e.pattern)
        if not p.is_const:
            raise Unsupported("non-constant LIKE pattern", e.line)
        code = f"{_paren(v.code)}.like({q(str(p.const).rstrip(' '))})"
        return CE(f"~({code})" if e.negated else code, "bool")

    def _e_Contains(self, e: A.Contains) -> CE:
        v, s = self.char(e.operand), self.char(e.sub)
        code = f"{_paren(v.code)}.contains({q(str(s.const)) if s.is_const else s.code})"
        return CE(f"~({code})" if e.negated else code, "bool")

    def _e_SoundsLike(self, e: A.SoundsLike) -> CE:
        a, b = self.char(e.left), self.char(e.right)
        return CE(f"(F.soundex({as_col(a)}) == F.soundex({as_col(b)}))", "bool")

    def _e_OfList(self, e: A.OfList) -> CE:
        raise Unsupported("OF variable list outside a function call", e.line)

    def _e_FormatSpec(self, e: A.FormatSpec) -> CE:
        raise Unsupported("format used as a value", e.line)

    # ------------------------------------------------------------------ calls
    def _args(self, args: list[A.Expr]) -> list[A.Expr]:
        """Expand OF lists into plain variable references."""
        out: list[A.Expr] = []
        for a in args:
            if isinstance(a, A.OfList):
                for name in self.scope.expand_var_items(a.items, a.line):
                    out.append(A.Var(name, line=a.line))
            else:
                out.append(a)
        return out

    def _lit_str_arg(self, e: A.Expr, what: str, line: int) -> str:
        c = self.char(e)
        if not c.is_const:
            raise Unsupported(f"{what} must be a literal", line)
        return str(c.const)

    def _e_Call(self, e: A.Call) -> CE:
        name = e.name.upper()
        line = e.line
        if name in ("LAG", "DIF") or (name[:3] in ("LAG", "DIF") and name[3:].isdigit()):
            n = int(name[3:]) if name[3:].isdigit() else 1
            if len(e.args) != 1:
                raise Unsupported(f"{name} expects one argument", line)
            return self.scope.lag(n, self.value(e.args[0]), name.startswith("DIF"), line)
        args = self._args(e.args)
        if name in NUMERIC_AGG_FUNCS:
            vals = [self.num(a) for a in args]
            if not vals:
                raise Unsupported(f"{name}() without arguments", line)
            return CE(f"S.{NUMERIC_AGG_FUNCS[name]}({', '.join(as_col(v) for v in vals)})", "num",
                      nonnull=name in ("N", "NMISS"))
        if name in _SIMPLE_NUM1:
            if len(args) != 1:
                raise Unsupported(f"{name} expects one argument", line)
            v = self.num(args[0])
            return CE(f"S.{_SIMPLE_NUM1[name]}({as_col(v)})", "num")
        handler = getattr(self, "_f_" + name.lower(), None)
        if handler is None:
            raise Unsupported(f"function {name}()", line)
        return handler(args, line)

    # numeric
    def _f_round(self, args, line):
        x = self.num(args[0])
        unit = self.num(args[1]) if len(args) > 1 else num_lit(1.0)
        self.scope.hazard("H-ROUND", "SAS rounding (half away from zero, decimal)", line)
        return CE(f"S.round({as_col(x)}, {unit.code})", "num")

    _f_roundz = _f_round

    def _f_mod(self, args, line):
        a, b = self.num(args[0]), self.num(args[1])
        return CE(f"S.mod({as_col(a)}, {b.code})", "num")

    _f_modz = _f_mod

    def _f_floorz(self, args, line):
        return CE(f"F.floor({as_col(self.num(args[0]))}).cast('double')", "num")

    def _f_ceilz(self, args, line):
        return CE(f"F.ceil({as_col(self.num(args[0]))}).cast('double')", "num")

    def _f_coalesce(self, args, line):
        vals = [self.num(a) for a in args]
        return CE(f"S.coalesce({', '.join(as_col(v) for v in vals)})", "num")

    def _f_coalescec(self, args, line):
        vals = [self.char(a) for a in args]
        return CE(f"S.coalescec({', '.join(as_col(v) for v in vals)})", "char", 200)

    def _f_cmiss(self, args, line):
        items = []
        for a in args:
            v = self.value(a)
            items.append(f"({as_col(v)}, {q('char' if v.type == 'char' else 'num')})")
        return CE(f"S.cmiss({', '.join(items)})", "num", nonnull=True)

    def _f_missing(self, args, line):
        v = self.value(args[0])
        if v.type == "char":
            return CE(f"({_paren(v.code)} == '')" if not v.padded else f"(F.trim({v.code}) == '')", "bool")
        return CE(f"{_paren(as_col(v))}.isNull()", "bool")

    def _f_ifn(self, args, line):
        c = self.cond(args[0])
        a, b = self.num(args[1]), self.num(args[2])
        code = f"F.when({c}, {as_col(a)}).otherwise({as_col(b)})"
        if len(args) > 3:
            m = self.num(args[3])
            raw = self.value(args[0])
            if raw.type == "num":
                code = f"F.when({_paren(raw.code)}.isNull(), {as_col(m)}).otherwise({code})"
        return CE(code, "num")

    def _f_ifc(self, args, line):
        c = self.cond(args[0])
        a, b = self.char(args[1]), self.char(args[2])
        return CE(f"F.when({c}, {as_col(a)}).otherwise({as_col(b)})", "char", 200)

    # dates
    def _f_today(self, args, line):
        self.scope.hazard("H-TODAY", "TODAY() is frozen by the context in validation runs", line)
        return CE("S.today()", "num", nonnull=True)

    _f_date = _f_today

    def _f_datetime(self, args, line):
        self.scope.hazard("H-TODAY", "DATETIME() is frozen by the context in validation runs", line)
        return CE("S.now()", "num", nonnull=True)

    def _f_time(self, args, line):
        return CE("S.timepart(S.now())", "num", nonnull=True)

    def _f_mdy(self, args, line):
        m, d, y = (self.num(a) for a in args[:3])
        return CE(f"S.mdy({as_col(m)}, {as_col(d)}, {as_col(y)})", "num")

    def _f_yyq(self, args, line):
        y, qt = self.num(args[0]), self.num(args[1])
        return CE(f"S.mdy(({as_col(qt)} - 1) * 3 + 1, 1.0, {as_col(y)})", "num")

    def _f_dhms(self, args, line):
        vals = [as_col(self.num(a)) for a in args[:4]]
        return CE(f"S.dhms({', '.join(vals)})", "num")

    def _f_hms(self, args, line):
        vals = [as_col(self.num(a)) for a in args[:3]]
        return CE(f"S.hms({', '.join(vals)})", "num")

    def _f_intnx(self, args, line):
        interval = self._lit_str_arg(args[0], "INTNX interval", line)
        start, n = self.num(args[1]), self.num(args[2])
        align = self._lit_str_arg(args[3], "INTNX alignment", line) if len(args) > 3 else "B"
        try:
            D.intnx_call(interval, align)
        except D.UnsupportedSpec as ex:
            raise Unsupported(str(ex), line) from ex
        return CE(f"S.intnx({q(interval.strip().lower())}, {as_col(start)}, {as_col(n)}, {q(align.strip().upper()[:1] or 'B')})", "num")

    def _f_intck(self, args, line):
        interval = self._lit_str_arg(args[0], "INTCK interval", line)
        a, b = self.num(args[1]), self.num(args[2])
        method = self._lit_str_arg(args[3], "INTCK method", line) if len(args) > 3 else "DISCRETE"
        try:
            D.intck_call(interval, method)
        except D.UnsupportedSpec as ex:
            raise Unsupported(str(ex), line) from ex
        self.scope.hazard("H-INTCK", "interval boundaries counted like SAS", line)
        meth = "C" if method.strip().upper().startswith("C") else "DISCRETE"
        return CE(f"S.intck({q(interval.strip().lower())}, {as_col(a)}, {as_col(b)}, {q(meth)})", "num")

    def _f_yrdif(self, args, line):
        a, b = self.num(args[0]), self.num(args[1])
        basis = self._lit_str_arg(args[2], "YRDIF basis", line) if len(args) > 2 else "ACT/ACT"
        try:
            D.yrdif_call(basis)
        except D.UnsupportedSpec as ex:
            raise Unsupported(str(ex), line) from ex
        return CE(f"S.yrdif({as_col(a)}, {as_col(b)}, {q(basis.strip().upper())})", "num")

    def _f_datdif(self, args, line):
        a, b = self.num(args[0]), self.num(args[1])
        return CE(f"({as_col(b)} - {as_col(a)})", "num")

    # characters
    def _char_arg(self, e: A.Expr) -> CE:
        return self.char(e)

    def _f_substr(self, args, line):
        s = self._char_arg(args[0])
        p = self.num(args[1])
        if len(args) > 2:
            n = self.num(args[2])
            return CE(f"S.substr({as_col(s)}, {p.code}, {n.code})", "char", s.length)
        return CE(f"S.substr({as_col(s)}, {p.code})", "char", s.length)

    def _f_substrn(self, args, line):
        s = self._char_arg(args[0])
        p, n = self.num(args[1]), self.num(args[2]) if len(args) > 2 else num_lit(32767)
        return CE(f"S.substrn({as_col(s)}, {p.code}, {n.code})", "char", s.length)

    def _simple_char(self, fname: str, args, keep_len=True):
        s = self._char_arg(args[0])
        return CE(f"S.{fname}({as_col(s)})", "char", s.length if keep_len else 200)

    def _f_upcase(self, args, line):
        return self._simple_char("upcase", args)

    def _f_lowcase(self, args, line):
        return self._simple_char("lowcase", args)

    def _f_propcase(self, args, line):
        s = self._char_arg(args[0])
        if len(args) > 1:
            d = self._lit_str_arg(args[1], "PROPCASE delimiters", line)
            return CE(f"S.propcase({as_col(s)}, {q(d)})", "char", s.length)
        return CE(f"S.propcase({as_col(s)})", "char", s.length)

    def _f_left(self, args, line):
        return self._simple_char("left", args)

    def _f_right(self, args, line):
        s = self._char_arg(args[0])
        if not s.length:
            raise Unsupported("RIGHT() of a value with unknown length", line)
        return CE(f"S.right({as_col(s)}, {s.length})", "char", s.length, padded=True)

    def _f_trim(self, args, line):
        self.scope.hazard("H-TRIM-BLANK", "TRIM returns one blank for a blank value", line)
        s = self._char_arg(args[0])
        return CE(f"S.trim({as_col(s)})", "char", s.length)

    def _f_trimn(self, args, line):
        return self._simple_char("trimn", args)

    def _f_strip(self, args, line):
        return self._simple_char("strip", args)

    def _f_compbl(self, args, line):
        return self._simple_char("compbl", args)

    def _f_reverse(self, args, line):
        s = self._char_arg(args[0])
        if not s.padded and s.length:
            s = CE(f"S.pad({s.code}, {s.length})", "char", s.length, padded=True)
        return CE(f"S.reverse({as_col(s)})", "char", s.length)

    def _f_compress(self, args, line):
        s = self._char_arg(args[0])
        chars = None
        mods = ""
        if len(args) > 1:
            chars = self._lit_str_arg(args[1], "COMPRESS characters", line)
            if chars == " " and isinstance(args[1], A.Str) and args[1].value == "":
                chars = " "
        if len(args) > 2:
            mods = self._lit_str_arg(args[2], "COMPRESS modifiers", line).strip()
        if s.padded and not s.is_const:
            pass
        args_code = [as_col(s)]
        if chars is not None or mods:
            args_code.append(q(chars) if chars is not None else "None")
        if mods:
            args_code.append(q(mods))
        return CE(f"S.compress({', '.join(args_code)})", "char", s.length)

    def _f_tranwrd(self, args, line):
        s, f_, t = (self._char_arg(a) for a in args[:3])
        return CE(f"S.tranwrd({as_col(s)}, {as_col(f_)}, {as_col(t)})", "char", 200)

    def _f_transtrn(self, args, line):
        s, f_, t = (self._char_arg(a) for a in args[:3])
        return CE(f"S.transtrn({as_col(s)}, {as_col(f_)}, {as_col(t)})", "char", 200)

    def _f_translate(self, args, line):
        self.scope.hazard("H-TRANSLATE-ARGS", "TRANSLATE(source, to, from) argument order", line)
        s = self._char_arg(args[0])
        rest = [as_col(self._char_arg(a)) for a in args[1:]]
        return CE(f"S.translate({as_col(s)}, {', '.join(rest)})", "char", s.length)

    def _f_index(self, args, line):
        s, t = self._char_arg(args[0]), self._char_arg(args[1])
        if isinstance(args[1], A.Var) and t.length and not t.padded:
            t = CE(f"S.pad({t.code}, {t.length})", "char", t.length, padded=True)
        return CE(f"S.index({as_col(s)}, {as_col(t)})", "num", nonnull=True)

    def _f_indexc(self, args, line):
        s = self._char_arg(args[0])
        rest = [as_col(self._char_arg(a)) for a in args[1:]]
        return CE(f"S.indexc({as_col(s)}, {', '.join(rest)})", "num", nonnull=True)

    def _f_verify(self, args, line):
        s = self._char_arg(args[0])
        rest = [as_col(self._char_arg(a)) for a in args[1:]]
        return CE(f"S.verify({as_col(s)}, {', '.join(rest)})", "num", nonnull=True)

    def _f_find(self, args, line):
        s, t = self._char_arg(args[0]), self._char_arg(args[1])
        mods, start = "", None
        for extra in args[2:]:
            v = self.value(extra)
            if v.type == "char":
                if not v.is_const:
                    raise Unsupported("FIND modifiers must be a literal", line)
                mods = str(v.const).strip()
            else:
                start = v
        parts = [as_col(s), as_col(t), q(mods)]
        if start is not None:
            parts.append(start.code)
        return CE(f"S.find({', '.join(parts)})", "num", nonnull=True)

    def _f_scan(self, args, line):
        s, n = self._char_arg(args[0]), self.num(args[1])
        delims = self._lit_str_arg(args[2], "SCAN delimiters", line) if len(args) > 2 else None
        mods = self._lit_str_arg(args[3], "SCAN modifiers", line).strip() if len(args) > 3 else ""
        parts = [as_col(s), n.code]
        if delims is not None or mods:
            parts.append(q(delims) if delims is not None else "None")
        if mods:
            parts.append(q(mods))
        return CE(f"S.scan({', '.join(parts)})", "char", s.length)

    def _f_countw(self, args, line):
        s = self._char_arg(args[0])
        delims = self._lit_str_arg(args[1], "COUNTW delimiters", line) if len(args) > 1 else None
        mods = self._lit_str_arg(args[2], "COUNTW modifiers", line).strip() if len(args) > 2 else ""
        parts = [as_col(s)]
        if delims is not None or mods:
            parts.append(q(delims) if delims is not None else "None")
        if mods:
            parts.append(q(mods))
        return CE(f"S.countw({', '.join(parts)})", "num", nonnull=True)

    def _f_count(self, args, line):
        s, t = self._char_arg(args[0]), self._char_arg(args[1])
        return CE(f"S.count({as_col(s)}, {as_col(t)})", "num", nonnull=True)

    def _f_countc(self, args, line):
        s, t = self._char_arg(args[0]), self._char_arg(args[1])
        return CE(f"S.countc({as_col(s)}, {as_col(t)})", "num", nonnull=True)

    def _f_length(self, args, line):
        return CE(f"S.length({as_col(self._char_arg(args[0]))})", "num", nonnull=True)

    def _f_lengthn(self, args, line):
        return CE(f"S.lengthn({as_col(self._char_arg(args[0]))})", "num", nonnull=True)

    def _f_lengthc(self, args, line):
        s = self._char_arg(args[0])
        if not s.length:
            raise Unsupported("LENGTHC of a value with unknown length", line)
        return num_lit(float(s.length))

    _f_lengthm = _f_lengthc

    def _f_repeat(self, args, line):
        s, n = self._char_arg(args[0]), self.num(args[1])
        return CE(f"S.repeat({as_col(s)}, {n.code})", "char", 200)

    def _cat(self, fname: str, args, sep: CE | None = None):
        parts = []
        for a in args:
            v = self.value(a)
            if v.type == "num":
                parts.append(f"S.num_for_cat({as_col(v)}{', catx=True' if fname == 'catx' else ''})")
            else:
                if fname == "cat" and isinstance(a, A.Var) and v.length and not v.padded:
                    v = CE(f"S.pad({v.code}, {v.length})", "char", v.length, padded=True)
                parts.append(as_col(v))
        head = [as_col(sep)] if sep is not None else []
        return CE(f"S.{fname}({', '.join(head + parts)})", "char", 200)

    def _f_cat(self, args, line):
        return self._cat("cat", args)

    def _f_cats(self, args, line):
        return self._cat("cats", args)

    def _f_catt(self, args, line):
        return self._cat("catt", args)

    def _f_catx(self, args, line):
        return self._cat("catx", args[1:], sep=self._char_arg(args[0]))

    def _f_quote(self, args, line):
        return CE(f"S.quote({as_col(self._char_arg(args[0]))})", "char", 200)

    def _f_dequote(self, args, line):
        return self._simple_char("dequote", args)

    def _f_char(self, args, line):
        s, n = self._char_arg(args[0]), self.num(args[1])
        return CE(f"S.substr({as_col(s)}, {n.code}, 1.0)", "char", 1)

    def _f_first(self, args, line):
        s = self._char_arg(args[0])
        return CE(f"S.substr({as_col(s)}, 1.0, 1.0)", "char", 1)

    def _f_byte(self, args, line):
        return CE(f"F.chr({as_col(self.num(args[0]))}.cast('long'))", "char", 1)

    def _f_rank(self, args, line):
        return CE(f"F.ascii({as_col(self._char_arg(args[0]))}).cast('double')", "num")

    def _f_prxmatch(self, args, line):
        pat = self._lit_str_arg(args[0], "PRXMATCH pattern", line)
        s = self._char_arg(args[1])
        return CE(f"S.prxmatch({q(pat)}, {as_col(s)})", "num", nonnull=True)

    def _f_prxchange(self, args, line):
        pat = self._lit_str_arg(args[0], "PRXCHANGE pattern", line)
        times = self.num(args[1])
        if not (times.is_const and times.const == -1):
            raise Unsupported("PRXCHANGE with a limited number of replacements", line)
        s = self._char_arg(args[2])
        return CE(f"S.prxchange({q(pat)}, -1, {as_col(s)})", "char", 200)

    # formats
    def _format_arg(self, e: A.Expr, line: int) -> str:
        if isinstance(e, A.FormatSpec):
            return e.spec
        c = self.char(e)
        if c.is_const:
            return str(c.const).strip()
        raise Unsupported("format name must be a literal", line)

    def _put(self, x: CE, spec: str, line: int) -> CE:
        dollar, name, w, d = pyfuncs.parse_format(spec)
        if pyfuncs.format_kind(spec) == "user" or (dollar and name not in ("", "CHAR", "UPCASE", "QUOTE")):
            fn_name = self.scope.user_format(("$" if dollar else "") + name)
            if fn_name is None:
                raise Unsupported(f"user-defined format {spec} (no PROC FORMAT definition available)", line)
            width = w
            code = f"F.call_function({q(fn_name)}, {as_col(x)})"
            if width:
                code = f"F.substring({code}, 1, {width})"
            return CE(code, "char", width or 200, padded=False)
        kind = "char" if x.type == "char" else "num"
        try:
            D.put_call(spec, kind)
        except D.UnsupportedSpec as ex:
            raise Unsupported(str(ex), line) from ex
        width = D.format_width(spec, kind)
        return CE(f"S.put({as_col(x)}, {q(spec.lower())}{', ' + q('char') if kind == 'char' else ''})", "char", width, padded=True)

    def _f_put(self, args, line):
        x = self.value(args[0])
        return self._put(x, self._format_arg(args[1], line), line)

    def _f_putn(self, args, line):
        return self._put(self.num(args[0]), self._format_arg(args[1], line), line)

    def _f_putc(self, args, line):
        return self._put(self.char(args[0]), self._format_arg(args[1], line), line)

    def _f_input(self, args, line):
        s = self.char(args[0])
        spec = self._format_arg(args[1], line)
        if pyfuncs.format_kind(spec) == "user":
            raise Unsupported(f"user-defined informat {spec}", line)
        try:
            call = D.input_call(spec)
        except D.UnsupportedSpec as ex:
            raise Unsupported(str(ex), line) from ex
        self.scope.hazard("H-INVALID-DATA", "invalid data becomes missing (no ANSI cast errors)", line)
        if call.returns == "char":
            _, _, w, _ = pyfuncs.parse_format(spec)
            return CE(f"S.input({as_col(s)}, {q(spec.lower())})", "char", w or s.length or 200)
        return CE(f"S.input({as_col(s)}, {q(spec.lower())})", "num")

    _f_inputn = _f_input
    _f_inputc = _f_input


def _sas_cmp(op: str, x, y) -> bool:
    def key(v):
        return (0, 0.0) if v is None else (1, float(v))
    a, b = key(x), key(y)
    return {"EQ": a == b, "NE": a != b, "LT": a < b, "LE": a <= b, "GT": a > b, "GE": a >= b}[op]


for _n in sorted(_ANY_FUNCS):
    def _make(fname):
        def h(self, args, line):
            return CE(f"S.{fname.lower()}({as_col(self.char(args[0]))})", "num", nonnull=True)
        return h
    setattr(ExprCompiler, "_f_" + _n.lower(), _make(_n))
