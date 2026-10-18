"""SAS tokenizer.

Handles the parts of SAS lexing that regex-based scanners get wrong:

* ``* ... ;`` comments are statements, recognised only at statement start;
  ``/* ... */`` comments may appear anywhere; ``%* ... ;`` macro comments.
* Quoted literals with doubled-quote escapes and type suffixes:
  ``'01JAN2020'd``, ``'01JAN2020:00:00'dt``, ``'12:00't``, ``'my var'n``, ``'0A'x``.
* Missing-value literals ``.``, ``._``, ``.A`` - ``.Z`` versus member access
  (``lib.ds``, ``first.id``) versus numbers (``.5``).
* ``DATALINES;`` / ``CARDS;`` blocks, which are raw text, not code.

Tokens keep source offsets and a ``ws_before`` flag so parsers can reassemble
adjacency-sensitive constructs such as format specs (``date9.``, ``$char10.``,
``8.2``) without the lexer having to guess context.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    NAME = "NAME"
    NUMBER = "NUMBER"
    STRING = "STRING"
    MISSING = "MISSING"      # . ._ .A-.Z
    OP = "OP"
    SEMI = "SEMI"
    DATALINES = "DATALINES"  # raw in-stream data block
    RTMACRO = "RTMACRO"      # runtime macro-variable placeholder (see macro.processor)
    EOF = "EOF"


@dataclass(frozen=True, slots=True)
class Token:
    kind: Kind
    value: str          # NAME: as written; STRING: decoded content; OP: operator text
    start: int
    end: int
    line: int
    ws_before: bool = False
    suffix: str = ""    # STRING literal type: '', 'd', 'dt', 't', 'n', 'x', 'b'
    quote: str = ""     # STRING quote char

    @property
    def upper(self) -> str:
        return self.value.upper()

    def is_name(self, *names: str) -> bool:
        return self.kind is Kind.NAME and (not names or self.value.upper() in names)

    def is_op(self, *ops: str) -> bool:
        return self.kind is Kind.OP and (not ops or self.value in ops)

    def __repr__(self) -> str:  # compact, test-friendly
        extra = f"{self.suffix}" if self.suffix else ""
        return f"{self.kind.value}({self.value!r}{extra})@{self.line}"


# Longest operators first.
_OPERATORS = [
    "**", "||", "!!", "¦¦", "^=", "~=", "¬=", "<=", ">=", "<>", "><", "=*",
    "=", "<", ">", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}", ",", ":",
    "&", "|", "!", "^", "~", "¬", "$", "@", "#", "%", "?", ".", "¦",
]

_NAME_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAME_CHARS = _NAME_START | set("0123456789")
_DIGITS = set("0123456789")
_STRING_SUFFIXES = ("dt", "d", "t", "n", "x", "b")
_DATALINES_STMTS = {"DATALINES", "CARDS", "LINES", "DATALINES4", "CARDS4", "LINES4", "PARMCARDS", "PARMCARDS4"}

# Private-use characters bracketing a runtime macro-variable placeholder emitted by
# the macro processor:  name .
RT_OPEN = ""
RT_CLOSE = ""
OPAQUE = "?"   # RT placeholder prefix: a macro value that cannot be known before run time


class LexError(Exception):
    pass


class LineIndex:
    """Offset -> 1-based line number."""

    def __init__(self, text: str):
        self.starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.starts.append(i + 1)

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)


def tokenize(text: str, *, keep_comments: bool = False) -> list[Token]:
    return Lexer(text).run(keep_comments=keep_comments)


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.lines = LineIndex(text)
        self.comments: list[tuple[int, int]] = []

    def run(self, keep_comments: bool = False) -> list[Token]:
        text, n = self.text, self.n
        toks: list[Token] = []
        i = 0
        ws = True
        stmt_start = True
        while i < n:
            ch = text[i]
            # whitespace
            if ch in " \t\r\n\f\v\x00":
                i += 1
                ws = True
                continue
            # block comment
            if ch == "/" and text.startswith("/*", i):
                j = text.find("*/", i + 2)
                j = n if j < 0 else j + 2
                self.comments.append((i, j))
                i = j
                ws = True
                continue
            # macro comment %* ... ;
            if ch == "%" and text.startswith("%*", i):
                j = self._find_semi(i + 2)
                self.comments.append((i, j))
                i = j
                ws = True
                continue
            # statement comment * ... ;
            if ch == "*" and stmt_start:
                j = self._find_semi(i + 1)
                self.comments.append((i, j))
                i = j
                ws = True
                continue
            # runtime macro placeholder
            if ch == RT_OPEN:
                j = text.find(RT_CLOSE, i + 1)
                if j < 0:
                    raise LexError(f"unterminated runtime macro placeholder at line {self.lines.line_of(i)}")
                toks.append(Token(Kind.RTMACRO, text[i + 1:j], i, j + 1, self.lines.line_of(i), ws))
                i = j + 1
                ws = False
                stmt_start = False
                continue
            # strings
            if ch in "'\"":
                tok, i = self._string(i, ws)
                toks.append(tok)
                ws = False
                stmt_start = False
                continue
            # numbers (incl. leading-dot decimals)
            if ch in _DIGITS or (ch == "." and i + 1 < n and text[i + 1] in _DIGITS and not self._prev_is_qualifier(toks, ws)):
                tok, i = self._number(i, ws)
                toks.append(tok)
                ws = False
                stmt_start = False
                continue
            # missing-value literals
            if ch == "." and not self._prev_is_qualifier(toks, ws):
                j = i + 1
                if j < n and (text[j].isalpha() or text[j] == "_") and (j + 1 >= n or text[j + 1] not in _NAME_CHARS):
                    toks.append(Token(Kind.MISSING, "." + text[j].upper(), i, j + 1, self.lines.line_of(i), ws))
                    i = j + 1
                    ws = False
                    stmt_start = False
                    continue
                if j >= n or text[j] not in _NAME_CHARS:
                    toks.append(Token(Kind.MISSING, ".", i, j, self.lines.line_of(i), ws))
                    i = j
                    ws = False
                    stmt_start = False
                    continue
            # names
            if ch in _NAME_START:
                j = i + 1
                while j < n and text[j] in _NAME_CHARS:
                    j += 1
                toks.append(Token(Kind.NAME, text[i:j], i, j, self.lines.line_of(i), ws))
                i = j
                ws = False
                stmt_start = False
                continue
            # semicolon
            if ch == ";":
                toks.append(Token(Kind.SEMI, ";", i, i + 1, self.lines.line_of(i), ws))
                i += 1
                ws = False
                stmt_start = True
                if self._is_datalines_stmt(toks):
                    block, i = self._datalines(i, toks[-2].value.upper().endswith("4"))
                    toks.append(block)
                    ws = True
                continue
            # operators
            for op in _OPERATORS:
                if text.startswith(op, i):
                    toks.append(Token(Kind.OP, op, i, i + len(op), self.lines.line_of(i), ws))
                    i += len(op)
                    break
            else:
                # Unknown byte: keep it as a one-char operator so nothing is silently lost.
                toks.append(Token(Kind.OP, ch, i, i + 1, self.lines.line_of(i), ws))
                i += 1
            ws = False
            stmt_start = False
        toks.append(Token(Kind.EOF, "", n, n, self.lines.line_of(n), True))
        return toks

    # -- helpers -----------------------------------------------------------------
    def _find_semi(self, i: int) -> int:
        """End offset (exclusive) of a comment statement, honouring quotes."""
        text, n = self.text, self.n
        while i < n:
            ch = text[i]
            if ch == ";":
                return i + 1
            if ch in "'\"":
                j = i + 1
                while j < n:
                    if text[j] == ch:
                        if j + 1 < n and text[j + 1] == ch:
                            j += 2
                            continue
                        break
                    j += 1
                i = j + 1
                continue
            i += 1
        return n

    @staticmethod
    def _prev_is_qualifier(toks: list[Token], ws: bool) -> bool:
        """True when a '.' directly follows a name or ')' (lib.ds, first.x, a.*)."""
        if ws or not toks:
            return False
        p = toks[-1]
        return p.kind in (Kind.NAME, Kind.RTMACRO) or (p.kind is Kind.STRING and p.suffix == "n")

    def _string(self, i: int, ws: bool) -> tuple[Token, int]:
        text, n = self.text, self.n
        q = text[i]
        j = i + 1
        buf = []
        while True:
            if j >= n:
                raise LexError(f"unterminated string starting at line {self.lines.line_of(i)}")
            c = text[j]
            if c == q:
                if j + 1 < n and text[j + 1] == q:
                    buf.append(q)
                    j += 2
                    continue
                j += 1
                break
            buf.append(c)
            j += 1
        suffix = ""
        for s in _STRING_SUFFIXES:
            k = j + len(s)
            if text[j:k].lower() == s and (k >= n or text[k] not in _NAME_CHARS):
                suffix = s
                j = k
                break
        value = "".join(buf)
        if suffix == "n":
            return Token(Kind.NAME, value, i, j, self.lines.line_of(i), ws, suffix, q), j
        return Token(Kind.STRING, value, i, j, self.lines.line_of(i), ws, suffix, q), j

    def _number(self, i: int, ws: bool) -> tuple[Token, int]:
        text, n = self.text, self.n
        j = i
        while j < n and text[j] in _DIGITS:
            j += 1
        if j < n and text[j] == ".":
            j += 1
            while j < n and text[j] in _DIGITS:
                j += 1
        if j < n and text[j] in "eE":
            k = j + 1
            if k < n and text[k] in "+-":
                k += 1
            if k < n and text[k] in _DIGITS:
                j = k
                while j < n and text[j] in _DIGITS:
                    j += 1
        return Token(Kind.NUMBER, text[i:j], i, j, self.lines.line_of(i), ws), j

    @staticmethod
    def _is_datalines_stmt(toks: list[Token]) -> bool:
        # toks[-1] is ';'; datalines statement is exactly NAME ';' at statement start.
        if len(toks) < 2 or toks[-2].kind is not Kind.NAME:
            return False
        if toks[-2].value.upper() not in _DATALINES_STMTS:
            return False
        return len(toks) == 2 or toks[-3].kind is Kind.SEMI

    def _datalines(self, i: int, four: bool) -> tuple[Token, int]:
        text, n = self.text, self.n
        # data starts on the line after the DATALINES; statement
        nl = text.find("\n", i)
        start = n if nl < 0 else nl + 1
        pos = start
        while pos < n:
            nl = text.find("\n", pos)
            line_end = n if nl < 0 else nl
            line = text[pos:line_end]
            stripped = line.strip()
            if (four and stripped.startswith(";;;;")) or (not four and ";" in line):
                if four:
                    end_data = pos
                    after = pos + line.index(";;;;") + 4
                else:
                    end_data = pos
                    after = pos + line.index(";") + 1
                raw = text[start:end_data]
                return Token(Kind.DATALINES, raw, start, after, self.lines.line_of(start), True), after
            pos = line_end + 1
        return Token(Kind.DATALINES, text[start:n], start, n, self.lines.line_of(start), True), n


def statement_spans(tokens: list[Token]) -> list[tuple[int, int]]:
    """Index ranges [i, j) of statements (j points just past the ';')."""
    spans = []
    i = 0
    n = len(tokens)
    while i < n and tokens[i].kind is not Kind.EOF:
        j = i
        while j < n and tokens[j].kind not in (Kind.SEMI, Kind.EOF):
            if tokens[j].kind is Kind.DATALINES:
                break
            j += 1
        if j < n and tokens[j].kind is Kind.SEMI:
            j += 1
        if j < n and tokens[j].kind is Kind.DATALINES:
            j += 1
        if j == i:
            j += 1
        spans.append((i, j))
        i = j
    return spans
