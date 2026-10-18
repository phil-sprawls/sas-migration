"""%EVAL / %SYSEVALF semantics.

Key SAS rules implemented:
* %EVAL does integer arithmetic; ``/`` truncates toward zero.  A non-integer
  operand in arithmetic is an error.
* Comparisons are numeric only when *both* operands are integers, otherwise
  they are character comparisons (so ``%eval(1.5 > 10)`` is a string compare).
* Operands may be empty (``%if &x = %then``) and may contain blanks.
* Logical results are 1/0.  NOT binds tighter than comparison (group I).
* ``IN`` / ``#`` require the MINOPERATOR option.
* %SYSEVALF does floating-point arithmetic and supports the conversion types
  BOOLEAN, CEIL, FLOOR, INTEGER.
"""
from __future__ import annotations

import math
import re

from .quoting import unmask


class EvalError(Exception):
    pass


_OPS = [
    "**", "<=", ">=", "^=", "~=", "¬=", "=", "<", ">", "+", "-", "*", "/", "(", ")", "&", "|", "^", "~", "¬", "#",
]
_MNEMONICS = {"EQ": "=", "NE": "^=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">=", "AND": "&", "OR": "|", "NOT": "^", "IN": "#"}
_INT_RE = re.compile(r"[+-]?\d+\Z")
_NUM_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\Z")


def _tokenize(expr: str, minoperator: bool) -> list[tuple[str, str]]:
    """Tokens: ('op', '+') / ('text', 'abc').  Masked (quoted) chars are always text."""
    toks: list[tuple[str, str]] = []
    i = 0
    n = len(expr)
    buf: list[str] = []

    def flush() -> None:
        if buf:
            txt = "".join(buf)
            if txt.strip(" \t\r\n"):
                # split mnemonic words out of text runs
                for kind, val in _split_mnemonics(txt, minoperator):
                    toks.append((kind, val))
            buf.clear()

    while i < n:
        ch = expr[i]
        matched = None
        for op in _OPS:
            if expr.startswith(op, i):
                matched = op
                break
        if matched:
            if matched == "#" and not minoperator:
                buf.append(ch)
                i += 1
                continue
            flush()
            toks.append(("op", {"~=": "^=", "¬=": "^=", "~": "^", "¬": "^"}.get(matched, matched)))
            i += len(matched)
            continue
        buf.append(ch)
        i += 1
    flush()
    return toks


def _split_mnemonics(txt: str, minoperator: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    words = re.split(r"(\s+)", txt)
    cur: list[str] = []
    for w in words:
        u = w.upper()
        if u in _MNEMONICS and (u != "IN" or minoperator):
            if "".join(cur).strip():
                out.append(("text", "".join(cur).strip()))
            cur = []
            out.append(("op", _MNEMONICS[u]))
        else:
            cur.append(w)
    if "".join(cur).strip():
        out.append(("text", "".join(cur).strip()))
    return out


class _Parser:
    """Precedence (high→low): prefix +,-,NOT, ** | * / | + - | comparisons, IN | AND | OR."""

    def __init__(self, toks, floating: bool):
        self.toks = toks
        self.i = 0
        self.floating = floating

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "")

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        v = self.or_()
        if self.peek()[0] != "eof":
            raise EvalError(f"unexpected {self.peek()[1]!r}")
        return v

    def or_(self):
        left = self.and_()
        while self.peek() == ("op", "|"):
            self.take()
            right = self.and_()
            left = 1 if (self._truth(left) or self._truth(right)) else 0
        return left

    def and_(self):
        left = self.cmp()
        while self.peek() == ("op", "&"):
            self.take()
            right = self.cmp()
            left = 1 if (self._truth(left) and self._truth(right)) else 0
        return left

    def cmp(self):
        left = self.add()
        while self.peek()[0] == "op" and self.peek()[1] in ("=", "^=", "<", "<=", ">", ">=", "#"):
            op = self.take()[1]
            if op == "#":
                items = self._in_list()
                left = 1 if any(self._compare(left, it) == 0 for it in items) else 0
                continue
            right = self.add()
            c = self._compare(left, right)
            left = 1 if {"=": c == 0, "^=": c != 0, "<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0}[op] else 0
        return left

    def _in_list(self):
        items = []
        if self.peek()[0] == "text":
            # MINDELIMITER default blank: "a b c"
            for part in self.take()[1].split():
                items.append(part)
        return items

    def add(self):
        left = self.mul()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            right = self.mul()
            a, b = self._num(left), self._num(right)
            left = a + b if op == "+" else a - b
        return left

    def mul(self):
        left = self.pow_()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            right = self.pow_()
            a, b = self._num(left), self._num(right)
            if op == "*":
                left = a * b
            else:
                if b == 0:
                    raise EvalError("division by zero in %EVAL")
                left = a / b if self.floating else int(a / b)
        return left

    def pow_(self):
        base = self.unary()
        if self.peek() == ("op", "**"):
            self.take()
            exp = self.pow_()
            a, b = self._num(base), self._num(exp)
            return a ** b
        return base

    def unary(self):
        t = self.peek()
        if t[0] == "op" and t[1] in ("+", "-"):
            self.take()
            v = self._num(self.unary())
            return v if t[1] == "+" else -v
        if t == ("op", "^"):
            self.take()
            v = self.unary()
            return 0 if self._truth(v) else 1
        if t == ("op", "("):
            self.take()
            v = self.or_()
            if self.take() != ("op", ")"):
                raise EvalError("missing ')'")
            return v
        if t[0] == "text":
            self.take()
            return t[1]
        # empty operand (e.g. "&x =" with x empty)
        return ""

    # -- value helpers
    def _is_num(self, v) -> bool:
        if isinstance(v, (int, float)):
            return True
        s = unmask(v).strip()
        if self.floating:
            return bool(_NUM_RE.match(s)) or s == "."
        return bool(_INT_RE.match(s))

    def _num(self, v):
        if isinstance(v, (int, float)):
            return v
        s = unmask(v).strip()
        if self.floating:
            if _NUM_RE.match(s):
                return float(s)
            if s == ".":
                return float("nan")
        elif _INT_RE.match(s):
            return int(s)
        raise EvalError(f"character operand {s!r} found where a numeric operand is required")

    def _compare(self, a, b) -> int:
        if self._is_num(a) and self._is_num(b):
            x, y = self._num(a), self._num(b)
            if isinstance(x, float) and math.isnan(x):
                return 0 if (isinstance(y, float) and math.isnan(y)) else -1
            if isinstance(y, float) and math.isnan(y):
                return 1
            return (x > y) - (x < y)
        sa = unmask(a if isinstance(a, str) else _fmt(a)).strip()
        sb = unmask(b if isinstance(b, str) else _fmt(b)).strip()
        return (sa > sb) - (sa < sb)

    def _truth(self, v) -> bool:
        if isinstance(v, (int, float)):
            return v != 0 and not (isinstance(v, float) and math.isnan(v))
        s = unmask(v).strip()
        if _NUM_RE.match(s):
            return float(s) != 0
        if s == "":
            raise EvalError("a character operand was found where a numeric operand is required (empty value)")
        raise EvalError(f"character operand {s!r} in a logical context")


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    return str(v)


def eval_int_expr(expr: str, *, minoperator: bool = False) -> int:
    toks = _tokenize(expr, minoperator)
    if not toks:
        raise EvalError("empty %EVAL expression")
    v = _Parser(toks, floating=False).parse()
    if isinstance(v, str):
        s = unmask(v).strip()
        if _INT_RE.match(s):
            return int(s)
        raise EvalError(f"%EVAL result {s!r} is not an integer")
    return int(v)


def eval_condition(expr: str, *, minoperator: bool = False) -> bool:
    toks = _tokenize(expr, minoperator)
    if not toks:
        raise EvalError("empty %IF condition")
    p = _Parser(toks, floating=False)
    v = p.parse()
    if isinstance(v, str):
        s = unmask(v).strip()
        if _INT_RE.match(s):
            return int(s) != 0
        # %IF with a non-numeric single operand is an error in SAS
        raise EvalError(f"character operand {s!r} where a numeric condition is required")
    return v != 0


def eval_int(text: str) -> int:
    s = unmask(text).strip()
    if _INT_RE.match(s):
        return int(s)
    return eval_int_expr(text)


def sysevalf(expr: str, conversion: str = "") -> str:
    toks = _tokenize(expr, False)
    v = _Parser(toks, floating=True).parse()
    if isinstance(v, str):
        s = unmask(v).strip()
        v = float(s) if _NUM_RE.match(s) else s
    conv = conversion.strip().upper()
    if conv == "BOOLEAN":
        return "0" if (v == 0 or (isinstance(v, float) and math.isnan(v))) else "1"
    if conv in ("CEIL", "FLOOR", "INTEGER"):
        f = {"CEIL": math.ceil, "FLOOR": math.floor, "INTEGER": math.trunc}[conv]
        return str(int(f(float(v))))
    if isinstance(v, float):
        if math.isnan(v):
            return "."
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.15g}"
    return str(v)
