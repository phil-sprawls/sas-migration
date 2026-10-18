"""Parser: expanded SAS text -> Program IR (steps, DATA-step statements, expressions)."""
from __future__ import annotations

import re

from . import ast as A
from .lexer import OPAQUE, Kind, Token, tokenize

DS_KEYWORDS = {
    "SET", "MERGE", "UPDATE", "MODIFY", "BY", "IF", "WHERE", "DO", "END", "SELECT", "WHEN", "OTHERWISE",
    "RETAIN", "KEEP", "DROP", "RENAME", "LENGTH", "FORMAT", "INFORMAT", "LABEL", "ATTRIB", "ARRAY", "OUTPUT",
    "DELETE", "RETURN", "STOP", "LEAVE", "CONTINUE", "CALL", "INPUT", "INFILE", "DATALINES", "CARDS",
    "LINES", "DATALINES4", "CARDS4", "LINES4", "PUT", "PUTLOG", "FILE", "ABORT", "ERROR", "LIST", "LOSTCARD",
    "MISSING", "GOTO", "GO", "LINK", "DECLARE", "DCL", "ELSE", "WINDOW", "DISPLAY", "TITLE", "FOOTNOTE",
    "OPTIONS", "OPTION", "LIBNAME", "FILENAME", "ODS", "RESETLINE", "REMOVE", "REPLACE",
}
GLOBAL_KEYWORDS = {
    "LIBNAME", "FILENAME", "OPTIONS", "OPTION", "ODS", "X", "ENDSAS", "DM", "GOPTIONS", "PAGE", "SKIP",
    "LOCK", "SASFILE", "CATNAME", "RESETLINE", "SYSECHO", "WINDOW", "NOTE",
}
_GLOBAL_PREFIX_RE = re.compile(r"^(TITLE|FOOTNOTE|AXIS|SYMBOL|LEGEND|PATTERN)\d*$")
INTERACTIVE_PROCS = {"SQL", "DATASETS", "REG", "GLM", "ANOVA", "CATALOG", "PLOT", "IML", "OPTMODEL", "FSEDIT", "GPLOT", "GCHART", "GREPLAY"}
FORMAT_ARG_FUNCS = {"PUT": 1, "PUTN": 1, "PUTC": 1, "INPUT": 1, "INPUTN": 1, "INPUTC": 1}
_FMT_SPEC_RE = re.compile(r"^\$?([A-Za-z_][A-Za-z0-9_]*)?\d*\.\d*$")
_CMP_OPS = {"=": "EQ", "^=": "NE", "~=": "NE", "¬=": "NE", "<": "LT", "<=": "LE", ">": "GT", ">=": "GE"}
_CMP_WORDS = {"EQ", "NE", "LT", "LE", "GT", "GE"}


class ParseError(Exception):
    def __init__(self, message: str, tok: Token):
        super().__init__(f"line {tok.line}: {message} (near {tok.value!r})")
        self.tok = tok


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0
        self.errors: list[str] = []
        self.arrays: set[str] = set()
        self._idle_at = -1
        self._idle = 0

    # ------------------------------------------------------------------ cursor
    def peek(self, k: int = 0) -> Token:
        if self.i != self._idle_at:
            self._idle_at, self._idle = self.i, 0
        else:
            self._idle += 1
            if self._idle > 100_000:   # a statement loop that cannot consume this token: fail the step, never hang
                self._idle = 0
                raise ParseError("parser made no progress on an unexpected token", self.toks[min(self.i, len(self.toks) - 1)])
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def advance(self) -> Token:
        t = self.toks[self.i]
        if t.kind is not Kind.EOF:
            self.i += 1
        return t

    def at_name(self, *names: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t.kind is Kind.NAME and (not names or t.value.upper() in names)

    def at_op(self, *ops: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t.kind is Kind.OP and (not ops or t.value in ops)

    def at_semi(self, k: int = 0) -> bool:
        return self.peek(k).kind is Kind.SEMI

    def at_eof(self) -> bool:
        return self.peek().kind is Kind.EOF

    def expect_op(self, op: str) -> Token:
        t = self.peek()
        if t.kind is Kind.OP and t.value == op:
            return self.advance()
        raise ParseError(f"expected {op!r}", t)

    def expect_semi(self) -> None:
        if self.at_semi():
            self.advance()
            return
        raise ParseError("expected ';'", self.peek())

    def skip_statement(self) -> list[Token]:
        """Consume through the next ';' and return the tokens before it."""
        out = []
        while not self.at_eof() and not self.at_semi():
            if self.peek().kind is Kind.DATALINES:
                break
            out.append(self.advance())
        if self.at_semi():
            self.advance()
        return out

    def text_of(self, toks: list[Token]) -> str:
        if not toks:
            return ""
        return self.text[toks[0].start:toks[-1].end]

    # ------------------------------------------------------------------ program
    def parse(self) -> A.Program:
        steps: list[A.Step] = []
        while not self.at_eof():
            t = self.peek()
            start_i = self.i
            try:
                if t.kind in (Kind.SEMI, Kind.DATALINES):
                    self.advance()
                    continue
                if t.kind is Kind.NAME and t.upper == "DATA" and not self.at_op("=", k=1):
                    steps.append(self.parse_data_step())
                elif t.kind is Kind.NAME and t.upper == "PROC":
                    steps.append(self.parse_proc())
                elif t.kind is Kind.NAME and t.upper == "__SAS2DBX_UNRESOLVED__":
                    toks = self.skip_statement()
                    msg = toks[1].value if len(toks) > 1 else "unresolved"
                    prev = steps[-1] if steps else None
                    if isinstance(prev, A.Unresolved) and not prev.message.startswith("parse error") \
                            and not self.text[prev.end:t.start].strip(" \t\r\n;"):
                        prev.message += "\n" + msg          # one unit of run-time macro logic
                        prev.end, prev.end_line = toks[-1].end, toks[-1].line
                    else:
                        steps.append(A.Unresolved(msg, start_line=t.line, end_line=toks[-1].line, start=t.start, end=toks[-1].end))
                elif t.kind is Kind.NAME and t.upper == "ENDSAS":
                    self.skip_statement()
                    steps.append(A.GlobalStmt("ENDSAS", "", start_line=t.line, end_line=t.line, start=t.start, end=t.end))
                    break
                elif t.kind is Kind.NAME and t.upper in ("RUN", "QUIT"):
                    self.skip_statement()
                else:
                    steps.append(self.parse_global())
            except (ParseError, RecursionError) as e:
                if isinstance(e, RecursionError):   # one oversized construct must not take the whole program down
                    e = ParseError(f"construct nested too deeply to parse (line {t.line})", t)
                self.errors.append(str(e))
                # recover: skip to the next step boundary; the skipped text becomes a visible unsupported step
                if self.i == start_i:
                    self.advance()
                self._recover()
                last = self.toks[max(self.i - 1, start_i)]
                steps.append(A.Unresolved(f"parse error: {e}", start_line=t.line, end_line=last.line, start=t.start, end=last.end))
        return A.Program(steps=steps, text=self.text, errors=self.errors)

    def _recover(self) -> None:
        while not self.at_eof():
            if self.at_semi():
                self.advance()
                if self.at_name("DATA", "PROC", "RUN", "QUIT") and not self.at_op("=", k=1):
                    if self.at_name("RUN", "QUIT"):
                        self.skip_statement()
                    return
                continue
            self.advance()

    def _at_step_boundary(self) -> bool:
        t = self.peek()
        if t.kind is Kind.EOF:
            return True
        if t.kind is not Kind.NAME:
            return False
        u = t.upper
        if u in ("DATA", "PROC") and not self.at_op("=", k=1):
            return True
        if u == "ENDSAS":
            return True
        return False

    def parse_global(self) -> A.GlobalStmt:
        t = self.peek()
        toks = self.skip_statement()
        kw = toks[0].value.upper() if toks else ""
        raw = self.text_of(toks[1:]) if len(toks) > 1 else ""
        end = toks[-1].end if toks else t.end
        return A.GlobalStmt(kw, raw, toks, start_line=t.line, end_line=self.peek(-1).line if self.i else t.line,
                            start=t.start, end=end)

    # ------------------------------------------------------------------ DATA step
    def parse_data_step(self) -> A.DataStep:
        first = self.advance()  # DATA
        self.arrays = set()
        outputs: list[A.DatasetRef] = []
        view = None
        while not self.at_semi() and not self.at_eof():
            if self.at_op("/"):
                self.advance()
                while not self.at_semi() and not self.at_eof():
                    opt = self.advance()
                    if opt.upper == "VIEW" and self.at_op("="):
                        self.advance()
                        view = self.advance().value
                break
            outputs.append(self.parse_dataset_ref())
        self.expect_semi()
        if not outputs:
            outputs = [A.DatasetRef(None, "DATA1", line=first.line)]
        stmts: list[A.Stmt] = []
        while True:
            if self._at_step_boundary():
                break
            if self.at_name("RUN", "QUIT"):
                self.skip_statement()
                break
            st = self.parse_ds_statement()
            if st is not None:
                stmts.append(st)
            if isinstance(st, A.DatalinesStmt):
                if self.at_name("RUN"):
                    self.skip_statement()
                break
        last = self.toks[self.i - 1] if self.i else first
        return A.DataStep(outputs, stmts, view, start_line=first.line, end_line=last.line, start=first.start, end=last.end)

    def parse_name(self) -> tuple[str, bool]:
        """A (possibly composite) name: NAME/RTMACRO tokens glued without whitespace."""
        t = self.peek()
        if t.kind not in (Kind.NAME, Kind.RTMACRO):
            raise ParseError("expected a name", t)
        parts = []
        dynamic = False
        while True:
            t = self.peek()
            if t.kind is Kind.NAME:
                parts.append(t.value)
            elif t.kind is Kind.RTMACRO:
                parts.append("{&" + t.value + "}")
                dynamic = True
            elif t.kind is Kind.NUMBER and parts and not t.ws_before:
                parts.append(t.value)
            else:
                break
            self.advance()
            nxt = self.peek()
            if nxt.ws_before or nxt.kind not in (Kind.NAME, Kind.RTMACRO, Kind.NUMBER):
                break
        return "".join(parts), dynamic

    def parse_dataset_ref(self) -> A.DatasetRef:
        t = self.peek()
        name, dyn = self.parse_name()
        libref = None
        if self.at_op(".") and not self.peek().ws_before:
            self.advance()
            libref = name
            name, dyn2 = self.parse_name()
            dyn = dyn or dyn2
        ref = A.DatasetRef(libref, name, line=t.line, dynamic=dyn)
        if self.at_op("("):
            self.advance()
            ref.options = self.parse_dataset_options()
            self.expect_op(")")
        return ref

    def parse_dataset_options(self) -> A.DatasetOptions:
        opts = A.DatasetOptions()
        while not self.at_op(")") and not self.at_eof() and not self.at_semi():
            t = self.advance()
            if t.kind is not Kind.NAME:
                raise ParseError("expected a data set option", t)
            key = t.upper
            if not self.at_op("="):
                opts.other[key] = ""
                continue
            self.advance()
            if key == "KEEP":
                opts.keep = self.parse_var_list(stop_at_option=True)
            elif key == "DROP":
                opts.drop = self.parse_var_list(stop_at_option=True)
            elif key == "RENAME":
                opts.rename.extend(self._parse_rename_pairs(paren=self.at_op("(")))
            elif key == "WHERE":
                self.expect_op("(")
                opts.where = self.parse_expr(where=True)
                self.expect_op(")")
            elif key == "IN":
                opts.in_var = self.advance().value
            elif key in ("OBS", "FIRSTOBS"):
                v = self.advance()
                val = None if v.upper == "MAX" else int(float(v.value))
                if key == "OBS":
                    opts.obs = val
                else:
                    opts.firstobs = val
            elif key == "LABEL":
                v = self.advance()
                opts.label = v.value
            else:
                vals = []
                depth = 0
                while not self.at_eof() and not self.at_semi():
                    if depth == 0 and (self.at_op(")") or (self.at_name() and self.at_op("=", k=1))):
                        break
                    tok = self.advance()
                    if tok.is_op("("):
                        depth += 1
                    elif tok.is_op(")"):
                        depth -= 1
                    vals.append(tok)
                opts.other[key] = self.text_of(vals)
        return opts

    def _parse_rename_pairs(self, paren: bool) -> list[tuple[str, str]]:
        pairs = []
        if paren:
            self.advance()
        while not self.at_eof() and not self.at_semi():
            if paren and self.at_op(")"):
                self.advance()
                break
            if not paren and (self.at_op(")") or (self.at_name() and self.at_op("=", k=1) and pairs and self._looks_like_option_after_rename())):
                break
            old = self.advance().value
            self.expect_op("=")
            new = self.advance().value
            pairs.append((old, new))
            if not paren:
                break
        return pairs

    def _looks_like_option_after_rename(self) -> bool:
        return self.peek().upper in ("KEEP", "DROP", "WHERE", "IN", "OBS", "FIRSTOBS", "LABEL", "RENAME")

    def parse_var_list(self, stop_at_option: bool = False, stop_names: tuple[str, ...] = ()) -> list[A.VarItem]:
        items: list[A.VarItem] = []
        while not self.at_eof() and not self.at_semi() and not self.at_op(")"):
            t = self.peek()
            if stop_at_option and t.kind is Kind.NAME and self.at_op("=", k=1):
                break
            if t.kind is Kind.NAME and t.upper in stop_names:
                break
            if t.kind is Kind.RTMACRO and not t.value.startswith(OPAQUE):
                prev = self.toks[self.i - 1] if self.i > 0 else None
                nxt = self.peek(1)
                glued = ((not t.ws_before and prev is not None and prev.kind in (Kind.NAME, Kind.NUMBER, Kind.RTMACRO))
                         or (not nxt.ws_before and nxt.kind in (Kind.NAME, Kind.NUMBER, Kind.RTMACRO)))
                if glued:              # part of a name built at run time (var_&i): not a list parameter
                    break
                self.advance()
                items.append(A.VarParam(t.value.upper()))
                continue
            if t.kind is not Kind.NAME:
                break
            self.advance()
            name = t.value
            u = name.upper()
            if u in ("_ALL_", "_NUMERIC_", "_CHARACTER_", "_CHAR_"):
                items.append(A.VarSpecial("_CHARACTER_" if u == "_CHAR_" else u))
                continue
            if self.at_op(":") and not self.peek().ws_before:
                self.advance()
                items.append(A.VarPrefix(name))
                continue
            if self.at_op("[", "{") and self.at_op("*", k=1):
                self.advance()
                self.advance()
                self.advance()
                items.append(A.ArrayStar(name))
                continue
            if self.at_op("-"):
                if self.at_op("-", k=1):
                    self.advance()
                    self.advance()
                    end = self.advance().value
                    items.append(A.VarRange(name, end, "positional"))
                    continue
                if self.at_name("NUMERIC", "CHARACTER", "CHAR", k=1) and self.at_op("-", k=2):
                    kind = "positional_num" if self.peek(1).upper == "NUMERIC" else "positional_char"
                    self.advance()
                    self.advance()
                    self.advance()
                    end = self.advance().value
                    items.append(A.VarRange(name, end, kind))
                    continue
                if self.peek(1).kind is Kind.NAME:
                    self.advance()
                    end = self.advance().value
                    items.append(A.VarRange(name, end, "numbered"))
                    continue
            items.append(A.VarName(name))
        return items

    # ------------------------------------------------------------------ statements
    def parse_ds_statement(self) -> A.Stmt | None:
        t = self.peek()
        if t.kind is Kind.SEMI:
            self.advance()
            return None
        if t.kind is Kind.DATALINES:
            self.advance()
            return A.DatalinesStmt(t.value, line=t.line)
        if t.kind is Kind.RTMACRO or (t.kind is Kind.NAME and t.upper not in DS_KEYWORDS) or (t.kind is Kind.NAME and self.at_op("=", k=1)):
            return self._parse_assignment_like()
        if t.kind is not Kind.NAME:
            toks = self.skip_statement()
            return A.UnknownStmt("?", self.text_of(toks), line=t.line)
        kw = t.upper
        if self.at_op("=", k=1):
            return self._parse_assignment_like()
        method = getattr(self, "_st_" + kw.lower(), None)
        if method is None:
            toks = self.skip_statement()
            return A.UnknownStmt(kw, self.text_of(toks[1:]), line=t.line)
        return method()

    def _parse_assignment_like(self) -> A.Stmt:
        t = self.peek()
        start = self.i
        # label:  stmt
        if t.kind is Kind.NAME and self.at_op(":", k=1) and not self.at_op("=", k=2):
            self.advance()
            self.advance()
            inner = self.parse_ds_statement()
            return A.LabeledStmt(t.value, inner or A.DoBlock([], line=t.line), line=t.line)
        # object.method(...) (hash objects) / unknown
        if t.kind is Kind.NAME and self.at_op(".", k=1) and not self.peek(1).ws_before:
            toks = self.skip_statement()
            return A.UnknownStmt("OBJECT_METHOD", self.text_of(toks), line=t.line)
        target = self.parse_primary(lvalue=True)
        if self.at_op("="):
            self.advance()
            expr = self.parse_expr()
            self._end_stmt()
            return A.Assign(target, expr, line=t.line)
        if self.at_op("+"):
            self.advance()
            expr = self.parse_expr()
            self._end_stmt()
            return A.SumStmt(target, expr, line=t.line)
        self.i = start
        toks = self.skip_statement()
        return A.UnknownStmt(t.value.upper(), self.text_of(toks), line=t.line)

    def _end_stmt(self) -> None:
        if not self.at_semi():
            raise ParseError("expected ';' at end of statement", self.peek())
        self.advance()

    def _st_if(self) -> A.Stmt:
        t = self.advance()
        cond = self.parse_expr()
        if self.at_name("THEN"):
            self.advance()
            then = self.parse_ds_statement() or A.DoBlock([], line=t.line)
            else_ = None
            if self.at_name("ELSE"):
                self.advance()
                else_ = self.parse_ds_statement() or A.DoBlock([], line=t.line)
            return A.If(cond, then, else_, line=t.line)
        self._end_stmt()
        return A.SubsetIf(cond, line=t.line)

    def _st_else(self) -> A.Stmt:
        t = self.advance()
        raise ParseError("ELSE without IF", t)

    def _st_where(self) -> A.Stmt:
        t = self.advance()
        also = False
        if self.at_name("ALSO") or (self.at_name("SAME") and self.at_name("AND", k=1)):
            if self.at_name("SAME"):
                self.advance()
            self.advance()
            also = True
        cond = self.parse_expr(where=True)
        self._end_stmt()
        return A.Where(cond, also, line=t.line)

    def _parse_block(self) -> list[A.Stmt]:
        body: list[A.Stmt] = []
        while True:
            if self.at_eof() or (self.at_name("RUN") and self.at_semi(k=1)) or self._at_step_boundary():
                raise ParseError("DO block without END", self.peek())
            if self.at_name("END") and self.at_semi(k=1):
                self.advance()
                self.advance()
                return body
            st = self.parse_ds_statement()
            if st is not None:
                body.append(st)

    def _st_do(self) -> A.Stmt:
        t = self.advance()
        if self.at_semi():
            self.advance()
            return A.DoBlock(self._parse_block(), line=t.line)
        if self.at_name("WHILE") and self.at_op("(", k=1):
            self.advance()
            self.advance()
            cond = self.parse_expr()
            self.expect_op(")")
            self._end_stmt()
            return A.DoWhile(cond, self._parse_block(), line=t.line)
        if self.at_name("UNTIL") and self.at_op("(", k=1):
            self.advance()
            self.advance()
            cond = self.parse_expr()
            self.expect_op(")")
            self._end_stmt()
            return A.DoUntil(cond, self._parse_block(), line=t.line)
        if self.at_name("OVER"):
            self.advance()
            arr = self.advance().value
            self._end_stmt()
            return A.DoLoop(A.Var("_I_"), [A.DoSpec(A.Num(1.0), A.Call("DIM", [A.Var(arr)]))], body=self._parse_block(), line=t.line)
        var = self.parse_primary(lvalue=True)
        self.expect_op("=")
        specs: list[A.DoSpec] = []
        while_ = until = None
        while True:
            start = self.parse_expr()
            stop = by = None
            if self.at_name("TO"):
                self.advance()
                stop = self.parse_expr()
            if self.at_name("BY"):
                self.advance()
                by = self.parse_expr()
            specs.append(A.DoSpec(start, stop, by))
            if self.at_op(","):
                self.advance()
                continue
            break
        if self.at_name("WHILE"):
            self.advance()
            self.expect_op("(")
            while_ = self.parse_expr()
            self.expect_op(")")
        if self.at_name("UNTIL"):
            self.advance()
            self.expect_op("(")
            until = self.parse_expr()
            self.expect_op(")")
        self._end_stmt()
        return A.DoLoop(var, specs, while_, until, self._parse_block(), line=t.line)

    def _st_select(self) -> A.Stmt:
        t = self.advance()
        expr = None
        if self.at_op("("):
            self.advance()
            expr = self.parse_expr()
            self.expect_op(")")
        self._end_stmt()
        whens: list[A.When] = []
        otherwise = None
        while True:
            if self.at_name("WHEN"):
                wt = self.advance()
                self.expect_op("(")
                values = [self.parse_expr()]
                while self.at_op(","):
                    self.advance()
                    values.append(self.parse_expr())
                self.expect_op(")")
                st = self.parse_ds_statement() or A.DoBlock([], line=wt.line)
                whens.append(A.When(values, st))
                continue
            if self.at_name("OTHERWISE"):
                ot = self.advance()
                otherwise = self.parse_ds_statement() or A.DoBlock([], line=ot.line)
                continue
            if self.at_name("END"):
                self.advance()
                self._end_stmt()
                break
            if self.at_eof() or self._at_step_boundary():
                raise ParseError("SELECT without END", self.peek())
            raise ParseError("expected WHEN, OTHERWISE or END in SELECT", self.peek())
        return A.Select(expr, whens, otherwise, line=t.line)

    _SET_OPTIONS = {"NOBS", "END", "POINT", "KEY", "INDSNAME", "OPEN", "CUROBS", "UNIQUE", "KEYRESET", "UPDATEMODE", "NOMISSINGCHECK"}

    def _parse_set_like(self, cls):
        t = self.advance()
        datasets: list[A.DatasetRef] = []
        options: dict[str, str] = {}
        while not self.at_semi() and not self.at_eof():
            if self.at_name() and self.peek().upper in self._SET_OPTIONS and (self.at_op("=", k=1) or self.peek().upper in ("UNIQUE", "UPDATEMODE")):
                key = self.advance().upper
                if self.at_op("="):
                    self.advance()
                    options[key] = self.advance().value
                else:
                    options[key] = ""
                continue
            datasets.append(self.parse_dataset_ref())
        self._end_stmt()
        return cls(datasets, options, line=t.line)

    def _st_set(self):
        return self._parse_set_like(A.SetStmt)

    def _st_merge(self):
        return self._parse_set_like(A.MergeStmt)

    def _st_update(self):
        return self._parse_set_like(A.UpdateStmt)

    def _st_modify(self):
        return self._parse_set_like(A.ModifyStmt)

    def _st_by(self):
        t = self.advance()
        vars_: list[tuple[str, bool]] = []
        notsorted = groupformat = False
        desc = False
        while not self.at_semi() and not self.at_eof():
            tok = self.advance()
            u = tok.upper
            if u == "DESCENDING":
                desc = True
                continue
            if u == "NOTSORTED":
                notsorted = True
                continue
            if u == "GROUPFORMAT":
                groupformat = True
                continue
            vars_.append((tok.value, desc))
            desc = False
        self._end_stmt()
        return A.ByStmt(vars_, notsorted, groupformat, line=t.line)

    def _parse_constant(self) -> A.Expr:
        t = self.peek()
        if t.is_op("-", "+") and self.peek(1).kind is Kind.NUMBER:
            self.advance()
            n = self.advance()
            v = float(n.value)
            return A.Num(-v if t.value == "-" else v, ("-" if t.value == "-" else "") + n.value, line=t.line)
        return self.parse_primary()

    def _st_retain(self):
        t = self.advance()
        items: list[tuple[list[A.VarItem], list[A.Expr]]] = []
        pending: list[A.VarItem] = []
        all_ = False
        while not self.at_semi() and not self.at_eof():
            p = self.peek()
            if p.kind is Kind.NAME:
                lst = self.parse_var_list()
                if any(isinstance(v, A.VarSpecial) and v.name == "_ALL_" for v in lst):
                    all_ = True
                pending.extend(lst)
                continue
            if p.is_op("("):
                self.advance()
                vals: list[A.Expr] = []
                while not self.at_op(")"):
                    if self.at_op(","):
                        self.advance()
                        continue
                    vals.append(self._parse_constant())
                self.advance()
                items.append((pending, vals))
                pending = []
                continue
            if p.kind in (Kind.NUMBER, Kind.STRING, Kind.MISSING) or p.is_op("-", "+"):
                val = self._parse_constant()
                items.append((pending, [val]))
                pending = []
                continue
            raise ParseError("unexpected token in RETAIN", p)
        if pending:
            items.append((pending, []))
        self._end_stmt()
        return A.RetainStmt(items, all_, line=t.line)

    def _st_keep(self):
        t = self.advance()
        v = self.parse_var_list()
        self._end_stmt()
        return A.KeepStmt(v, line=t.line)

    def _st_drop(self):
        t = self.advance()
        v = self.parse_var_list()
        self._end_stmt()
        return A.DropStmt(v, line=t.line)

    def _st_rename(self):
        t = self.advance()
        pairs = []
        while not self.at_semi() and not self.at_eof():
            old = self.advance().value
            self.expect_op("=")
            pairs.append((old, self.advance().value))
        self._end_stmt()
        return A.RenameStmt(pairs, line=t.line)

    def _st_length(self):
        t = self.advance()
        items: list[tuple[list[A.VarItem], bool, int]] = []
        default = None
        pending: list[A.VarItem] = []
        while not self.at_semi() and not self.at_eof():
            if self.at_name("DEFAULT") and self.at_op("=", k=1):
                self.advance()
                self.advance()
                default = int(float(self.advance().value))
                continue
            if self.at_op("$"):
                self.advance()
                n = self.peek()
                if n.kind is Kind.NUMBER:
                    self.advance()
                    items.append((pending, True, int(float(n.value))))
                else:
                    items.append((pending, True, 8))
                pending = []
                continue
            if self.peek().kind is Kind.NUMBER:
                n = self.advance()
                items.append((pending, False, int(float(n.value))))
                pending = []
                continue
            got = self.parse_var_list()
            if not got:     # nothing consumed (e.g. a run-time macro value): never loop on the same token
                raise ParseError("unexpected token in LENGTH", self.peek())
            pending.extend(got)
        if pending:
            items.append((pending, False, 8))
        self._end_stmt()
        return A.LengthStmt(items, default, line=t.line)

    def read_format_spec(self) -> str | None:
        """Read a format/informat spec made of adjacent tokens (date9. / 8.2 / $char10. / best.)."""
        start = self.i
        parts = []
        t = self.peek()
        if t.kind not in (Kind.NAME, Kind.NUMBER) and not t.is_op("$"):
            return None
        while True:
            t = self.peek()
            if parts and t.ws_before:
                break
            if t.kind in (Kind.NAME, Kind.NUMBER) or t.is_op("$", "."):
                parts.append(self.text[t.start:t.end])
                self.advance()
                if t.is_op(".") or (t.kind is Kind.NUMBER and "." in t.value):
                    # a spec ends after its '.', optionally followed by decimals
                    nxt = self.peek()
                    if nxt.kind is Kind.NUMBER and not nxt.ws_before and "." not in nxt.value:
                        parts.append(nxt.value)
                        self.advance()
                    break
                continue
            break
        spec = "".join(parts)
        if _FMT_SPEC_RE.match(spec) and "." in spec:
            return spec
        self.i = start
        return None

    def _parse_format_like(self, cls):
        t = self.advance()
        items: list[tuple[list[A.VarItem], str | None]] = []
        pending: list[A.VarItem] = []
        while not self.at_semi() and not self.at_eof():
            spec = self.read_format_spec()
            if spec is not None:
                items.append((pending, spec))
                pending = []
                continue
            if self.at_name("DEFAULT") and self.at_op("=", k=1):
                self.advance()
                self.advance()
                self.read_format_spec()
                continue
            lst = self._read_one_var_item()
            if lst is None:
                raise ParseError("unexpected token in FORMAT", self.peek())
            pending.append(lst)
        if pending:
            items.append((pending, None))
        self._end_stmt()
        return cls(items, line=t.line)

    def _read_one_var_item(self) -> A.VarItem | None:
        """Read exactly one variable-list item (used where formats are interleaved)."""
        t = self.peek()
        if t.kind is not Kind.NAME:
            return None
        save = self.i
        items = self.parse_var_list()
        if not items:
            self.i = save
            return None
        # parse_var_list may have consumed several items; re-wind to just the first
        self.i = save
        self.advance()
        first = items[0]
        if isinstance(first, A.VarRange):
            # consume the rest of the range
            if first.kind == "numbered":
                self.advance()
                self.advance()
            elif first.kind == "positional":
                self.advance()
                self.advance()
                self.advance()
            else:
                self.advance()
                self.advance()
                self.advance()
                self.advance()
        elif isinstance(first, A.VarPrefix):
            self.advance()
        elif isinstance(first, A.ArrayStar):
            self.advance()
            self.advance()
            self.advance()
        return first

    def _st_format(self):
        return self._parse_format_like(A.FormatStmt)

    def _st_informat(self):
        return self._parse_format_like(A.InformatStmt)

    def _st_label(self):
        t = self.advance()
        items = []
        while not self.at_semi() and not self.at_eof():
            name = self.advance().value
            self.expect_op("=")
            v = self.peek()
            if v.kind is Kind.STRING:
                self.advance()
                items.append((name, v.value))
            else:
                # unquoted label text runs until the next "name =" or ';'
                toks = []
                while not self.at_semi() and not (self.at_name() and self.at_op("=", k=1)):
                    toks.append(self.advance())
                items.append((name, self.text_of(toks)))
        self._end_stmt()
        return A.LabelStmt(items, line=t.line)

    def _st_attrib(self):
        t = self.advance()
        items: list[tuple[list[A.VarItem], dict[str, str]]] = []
        while not self.at_semi() and not self.at_eof():
            vars_: list[A.VarItem] = []
            while self.at_name() and not self.at_op("=", k=1) and self.peek().upper not in ("LENGTH", "FORMAT", "INFORMAT", "LABEL", "TRANSCODE"):
                item = self._read_one_var_item()
                if item is None:
                    break
                vars_.append(item)
            attrs: dict[str, str] = {}
            while self.at_name("LENGTH", "FORMAT", "INFORMAT", "LABEL", "TRANSCODE") and self.at_op("=", k=1):
                key = self.advance().upper
                self.advance()
                if key == "LENGTH":
                    if self.at_op("$"):
                        self.advance()
                        attrs["LENGTH"] = "$" + self.advance().value
                    else:
                        attrs["LENGTH"] = self.advance().value
                elif key in ("FORMAT", "INFORMAT"):
                    attrs[key] = self.read_format_spec() or self.advance().value
                elif key == "LABEL":
                    attrs["LABEL"] = self.advance().value
                else:
                    attrs[key] = self.advance().value
            if not vars_ and not attrs:
                raise ParseError("unexpected token in ATTRIB", self.peek())
            items.append((vars_, attrs))
        self._end_stmt()
        return A.AttribStmt(items, line=t.line)

    def _st_array(self):
        t = self.advance()
        name = self.advance().value
        self.arrays.add(name.upper())
        dims: list = []
        if self.at_op("{", "[", "("):
            close = {"{": "}", "[": "]", "(": ")"}[self.advance().value]
            while not self.at_op(close):
                if self.at_op(","):
                    self.advance()
                    continue
                if self.at_op("*"):
                    self.advance()
                    dims.append("*")
                    continue
                lo = self.advance()
                if self.at_op(":"):
                    self.advance()
                    hi = self.advance()
                    dims.append((int(float(lo.value)), int(float(hi.value))))
                else:
                    dims.append(int(float(lo.value)))
            self.advance()
        is_char = False
        length = None
        if self.at_op("$"):
            self.advance()
            is_char = True
            if self.peek().kind is Kind.NUMBER:
                length = int(float(self.advance().value))
        elif self.peek().kind is Kind.NUMBER:
            length = int(float(self.advance().value))
        temporary = False
        elements: list[A.VarItem] = []
        if self.at_name("_TEMPORARY_"):
            self.advance()
            temporary = True
        else:
            elements = self.parse_var_list()
        init: list[A.Expr] = []
        if self.at_op("("):
            self.advance()
            while not self.at_op(")") and not self.at_eof():
                if self.at_op(","):
                    self.advance()
                    continue
                init.append(self._parse_constant())
            self.advance()
        self._end_stmt()
        return A.ArrayStmt(name, dims, is_char, length, elements, init, temporary, line=t.line)

    def _st_output(self):
        t = self.advance()
        names = []
        while not self.at_semi() and not self.at_eof():
            nm, _ = self.parse_name()
            if self.at_op(".") and not self.peek().ws_before:
                self.advance()
                nm2, _ = self.parse_name()
                nm = nm + "." + nm2
            names.append(nm)
        self._end_stmt()
        return A.OutputStmt(names, line=t.line)

    def _simple(self, cls):
        t = self.advance()
        self._end_stmt()
        return cls(line=t.line)

    def _st_delete(self):
        return self._simple(A.DeleteStmt)

    def _st_return(self):
        return self._simple(A.ReturnStmt)

    def _st_stop(self):
        return self._simple(A.StopStmt)

    def _st_leave(self):
        return self._simple(A.LeaveStmt)

    def _st_continue(self):
        return self._simple(A.ContinueStmt)

    def _st_call(self):
        t = self.advance()
        name = self.advance().value.upper()
        args: list[A.Expr] = []
        if self.at_op("("):
            args = self.parse_call_args(name)
        self._end_stmt()
        return A.CallStmt(name, args, line=t.line)

    def _st_missing(self):
        t = self.advance()
        chars = [tok.value for tok in self.skip_statement()]
        return A.MissingStmt(chars, line=t.line)

    def _st_goto(self):
        t = self.advance()
        if self.at_name("TO"):
            self.advance()
        lbl = self.advance().value
        self._end_stmt()
        return A.GotoStmt(lbl, line=t.line)

    def _st_go(self):
        return self._st_goto()

    def _st_link(self):
        t = self.advance()
        lbl = self.advance().value
        self._end_stmt()
        return A.GotoStmt(lbl, link=True, line=t.line)

    def _raw_stmt(self, cls):
        t = self.advance()
        toks = self.skip_statement()
        return cls(self.text_of(toks), line=t.line)

    def _st_put(self):
        return self._raw_stmt(A.PutStmt)

    def _st_putlog(self):
        return self._raw_stmt(A.PutStmt)

    def _st_file(self):
        return self._raw_stmt(A.FileStmt)

    def _st_abort(self):
        return self._raw_stmt(A.AbortStmt)

    def _st_declare(self):
        return self._raw_stmt(A.DeclareStmt)

    def _st_dcl(self):
        return self._raw_stmt(A.DeclareStmt)

    def _st_infile(self):
        t = self.advance()
        toks = self.skip_statement()
        fileref = toks[0].value if toks else ""
        opts: dict[str, str] = {}
        k = 1
        while k < len(toks):
            key = toks[k].value.upper()
            if k + 2 < len(toks) + 1 and k + 1 < len(toks) and toks[k + 1].is_op("="):
                opts[key] = toks[k + 2].value if k + 2 < len(toks) else ""
                k += 3
            else:
                opts[key] = ""
                k += 1
        return A.InfileStmt(fileref, opts, line=t.line)

    def _st_input(self):
        t = self.advance()
        start = self.i
        items: list[A.InputItem] = []
        while not self.at_semi() and not self.at_eof():
            p = self.peek()
            if p.is_op("@", "#", "/"):
                self.advance()
                if p.value == "@" and self.at_op("@"):
                    self.advance()
                    items.append(A.InputItem("linehold", pointer="@@"))
                    continue
                if self.peek().kind is Kind.NUMBER or self.at_op("("):
                    v = self.advance()
                    items.append(A.InputItem("pointer", pointer=p.value + v.value))
                elif p.value == "@" and self.at_semi():
                    items.append(A.InputItem("linehold", pointer="@"))
                else:
                    items.append(A.InputItem("pointer", pointer=p.value))
                continue
            if p.kind is Kind.NAME:
                self.advance()
                item = A.InputItem("var", var=p.value)
                if self.at_op("$"):
                    self.advance()
                    item.is_char = True
                if self.at_op(":", "&", "~"):
                    item.modifier = self.advance().value
                spec = self.read_format_spec()
                if spec is not None:
                    if re.match(r"^\d+\.?$", spec) and not item.modifier and not spec.endswith("."):
                        pass
                    if re.match(r"^\d+$", spec.rstrip(".")) and not spec.endswith("."):
                        item.col_start = int(spec)
                    else:
                        item.informat = spec
                        if spec.startswith("$"):
                            item.is_char = True
                elif self.peek().kind is Kind.NUMBER:
                    c1 = self.advance()
                    item.col_start = int(float(c1.value))
                    if self.at_op("-"):
                        self.advance()
                        item.col_end = int(float(self.advance().value))
                items.append(item)
                continue
            self.advance()
        raw = self.text_of(self.toks[start:self.i])
        self._end_stmt()
        return A.InputStmt(items, raw, line=t.line)

    def _st_datalines(self):
        self.advance()
        self._end_stmt()
        t = self.peek()
        if t.kind is Kind.DATALINES:
            self.advance()
            return A.DatalinesStmt(t.value, line=t.line)
        return A.DatalinesStmt("", line=t.line)

    _st_cards = _st_lines = _st_datalines4 = _st_cards4 = _st_lines4 = _st_datalines

    def _st_end(self):
        t = self.advance()
        raise ParseError("END without DO/SELECT", t)

    # global statements allowed inside steps
    def _global_inside(self):
        t = self.peek()
        toks = self.skip_statement()
        return A.UnknownStmt(t.upper, self.text_of(toks[1:]), line=t.line)

    _st_title = _st_footnote = _st_options = _st_option = _st_libname = _st_filename = _st_ods = _global_inside
    _st_error = _st_list = _st_lostcard = _st_window = _st_display = _st_resetline = _global_inside
    _st_remove = _st_replace = _global_inside

    # ------------------------------------------------------------------ expressions
    def parse_expr(self, rbp: int = 0, where: bool = False) -> A.Expr:
        left = self.parse_prefix(where)
        while True:
            info = self._infix_info(where)
            if info is None:
                break
            lbp, kind = info
            if lbp <= rbp:
                break
            left = self._parse_infix(left, kind, lbp, where)
        return left

    def _infix_info(self, where: bool):
        t = self.peek()
        if t.kind is Kind.OP:
            v = t.value
            if v in ("|", "!", "¦"):
                return 10, "OR"
            if v == "&":
                return 20, "AND"
            if v in _CMP_OPS:
                return 30, "CMP"
            if v == "<>":
                return (30, "CMP") if where else (70, "MAX")
            if v == "><":
                return 70, "MIN"
            if v in ("^", "~", "¬") and self.at_name("IN", k=1):
                return 30, "NOTIN"
            if v in ("||", "!!", "¦¦"):
                return 40, "CONCAT"
            if v in ("+", "-"):
                return 50, v
            if v in ("*", "/"):
                return 60, v
            if v == "**":
                return 70, "POW"
            if where and v == "?":
                return 30, "CONTAINS"
            if v == "=*":
                return 30, "SOUNDS"
            return None
        if t.kind is Kind.NAME:
            u = t.upper
            if u == "OR":
                return 10, "OR"
            if u == "AND":
                return 20, "AND"
            if u in _CMP_WORDS:
                return 30, "CMP"
            if u == "IN":
                return 30, "IN"
            if u == "NOTIN":
                return 30, "NOTIN"
            if u == "NOT" and self.at_name("IN", "BETWEEN", "LIKE", "CONTAINS", k=1):
                return 30, "NOT_" + self.peek(1).upper
            if where and u in ("BETWEEN", "LIKE", "CONTAINS", "IS"):
                return 30, u
        return None

    def _parse_infix(self, left: A.Expr, kind: str, lbp: int, where: bool) -> A.Expr:
        t = self.advance()
        line = t.line
        if kind == "OR":
            return A.Binary("OR", left, self.parse_expr(lbp, where), line=line)
        if kind == "AND":
            return A.Binary("AND", left, self.parse_expr(lbp, where), line=line)
        if kind == "CMP":
            op = _CMP_OPS.get(t.value) if t.kind is Kind.OP else t.upper
            if t.value == "<>":
                op = "NE"
            colon = False
            if self.at_op(":") and not self.peek().ws_before:
                self.advance()
                colon = True
            right = self.parse_expr(lbp, where)
            node: A.Expr = A.Compare(op, left, right, colon, line=line)
            # chained comparison: a < b < c  ==  a<b and b<c
            while self._infix_info(where) == (30, "CMP"):
                t2 = self.advance()
                op2 = _CMP_OPS.get(t2.value) if t2.kind is Kind.OP else t2.upper
                if t2.value == "<>":
                    op2 = "NE"
                colon2 = False
                if self.at_op(":") and not self.peek().ws_before:
                    self.advance()
                    colon2 = True
                nxt = self.parse_expr(lbp, where)
                node = A.Binary("AND", node, A.Compare(op2, right, nxt, colon2, line=line), line=line)
                right = nxt
            return node
        if kind in ("IN", "NOTIN", "NOT_IN"):
            if kind == "NOT_IN" or (kind == "NOTIN" and t.kind is Kind.OP):
                self.advance()  # the IN after NOT / ^
            return A.InList(left, self._parse_in_items(), negated=kind != "IN", line=line)
        if kind in ("BETWEEN", "NOT_BETWEEN"):
            if kind == "NOT_BETWEEN":
                self.advance()
            lo = self.parse_expr(20, where)
            if not self.at_name("AND"):
                raise ParseError("BETWEEN requires AND", self.peek())
            self.advance()
            hi = self.parse_expr(20, where)
            return A.Between(left, lo, hi, negated=kind.startswith("NOT"), line=line)
        if kind in ("LIKE", "NOT_LIKE"):
            if kind == "NOT_LIKE":
                self.advance()
            return A.Like(left, self.parse_expr(lbp, where), negated=kind.startswith("NOT"), line=line)
        if kind in ("CONTAINS", "NOT_CONTAINS"):
            if kind == "NOT_CONTAINS":
                self.advance()
            return A.Contains(left, self.parse_expr(lbp, where), negated=kind.startswith("NOT"), line=line)
        if kind == "IS":
            neg = False
            if self.at_name("NOT"):
                self.advance()
                neg = True
            w = self.advance()
            if w.upper not in ("MISSING", "NULL"):
                raise ParseError("expected MISSING or NULL after IS", w)
            return A.IsMissing(left, neg, line=line)
        if kind == "SOUNDS":
            return A.SoundsLike(left, self.parse_expr(lbp, where), line=line)
        if kind == "CONCAT":
            return A.Binary("||", left, self.parse_expr(lbp, where), line=line)
        if kind in ("+", "-", "*", "/"):
            return A.Binary(kind, left, self.parse_expr(lbp, where), line=line)
        if kind in ("POW", "MIN", "MAX"):
            # group I operators associate right to left
            return A.Binary({"POW": "**", "MIN": "MIN", "MAX": "MAX"}[kind], left, self.parse_expr(lbp - 1, where), line=line)
        raise ParseError(f"unexpected operator {kind}", t)

    def _parse_in_items(self) -> list:
        self.expect_op("(")
        items: list = []
        while not self.at_op(")") and not self.at_eof():
            if self.at_op(","):
                self.advance()
                continue
            if self.peek().kind is Kind.RTMACRO:        # a value list supplied at run time: in (&regions)
                t = self.advance()
                items.append(A.RtMacro(t.value, line=t.line))
                continue
            v = self._parse_constant()
            if self.at_op(":"):
                self.advance()
                hi = self._parse_constant()
                items.append(A.InRange(v, hi))
            else:
                items.append(v)
        self.expect_op(")")
        return items

    def parse_prefix(self, where: bool) -> A.Expr:
        t = self.peek()
        if t.kind is Kind.OP:
            if t.value in ("-", "+"):
                self.advance()
                operand = self.parse_expr(69, where)
                if isinstance(operand, A.Num) and t.value == "-":
                    return A.Num(-operand.value, "-" + operand.text, line=t.line)
                return A.Unary(t.value, operand, line=t.line)
            if t.value in ("^", "~", "¬"):
                self.advance()
                return A.Unary("NOT", self.parse_expr(69, where), line=t.line)
            if t.value == "(":
                self.advance()
                e = self.parse_expr(0, where)
                self.expect_op(")")
                return e
        if t.kind is Kind.NAME and t.upper == "NOT" and "NOT" not in self.arrays:
            self.advance()
            return A.Unary("NOT", self.parse_expr(69, where), line=t.line)
        return self.parse_primary(where=where)

    def parse_primary(self, lvalue: bool = False, where: bool = False) -> A.Expr:
        t = self.peek()
        if t.kind is Kind.NUMBER:
            self.advance()
            return A.Num(float(t.value), t.value, line=t.line)
        if t.kind is Kind.STRING:
            self.advance()
            value = t.value
            if t.suffix == "x":
                try:
                    value = bytes.fromhex(t.value.replace(" ", "")).decode("latin-1")
                except ValueError:
                    pass
                return A.Str(value, "", t.quote, line=t.line)
            return A.Str(value, t.suffix, t.quote, line=t.line)
        if t.kind is Kind.MISSING:
            self.advance()
            return A.Missing(t.value, line=t.line)
        if t.kind is Kind.RTMACRO:
            self.advance()
            return A.RtMacro(t.value, line=t.line)
        if t.kind is Kind.OP and t.value == "(":
            self.advance()
            e = self.parse_expr(0, where)
            self.expect_op(")")
            return e
        if t.kind is not Kind.NAME:
            raise ParseError("expected an expression", t)
        self.advance()
        name = t.value
        u = name.upper()
        if u in ("FIRST", "LAST") and self.at_op(".") and not self.peek().ws_before and self.peek(1).kind is Kind.NAME:
            self.advance()
            v = self.advance()
            return A.FirstLast(u, v.value, line=t.line)
        if self.at_op("[", "{"):
            close = "]" if self.advance().value == "[" else "}"
            idx = [self.parse_expr()]
            while self.at_op(","):
                self.advance()
                idx.append(self.parse_expr())
            self.expect_op(close)
            return A.Index(name, idx, line=t.line)
        if self.at_op("("):
            if u in self.arrays:
                self.advance()
                idx = [self.parse_expr()]
                while self.at_op(","):
                    self.advance()
                    idx.append(self.parse_expr())
                self.expect_op(")")
                return A.Index(name, idx, line=t.line)
            args = self.parse_call_args(u)
            return A.Call(u, args, line=t.line)
        return A.Var(name, line=t.line)

    def parse_call_args(self, fname: str) -> list[A.Expr]:
        self.expect_op("(")
        args: list[A.Expr] = []
        pos = 0
        while not self.at_op(")") and not self.at_eof():
            if self.at_name("OF") and self.peek(1).kind is Kind.NAME:
                self.advance()
                args.append(A.OfList(self.parse_var_list(), line=self.peek().line))
            elif fname in FORMAT_ARG_FUNCS and pos == FORMAT_ARG_FUNCS[fname]:
                modifier = ""
                while self.at_op("?"):
                    modifier += self.advance().value
                spec = self.read_format_spec()
                if spec is None:
                    args.append(self.parse_expr())
                else:
                    args.append(A.FormatSpec(spec, modifier, line=self.peek().line))
            else:
                args.append(self.parse_expr())
            pos += 1
            if self.at_op(","):
                self.advance()
                continue
            if not self.at_op(")"):
                # SAS also allows blank-separated OF lists; anything else is an error
                raise ParseError(f"expected ',' or ')' in {fname}(...)", self.peek())
        self.expect_op(")")
        return args

    # ------------------------------------------------------------------ PROC
    def parse_proc(self) -> A.ProcStep:
        first = self.advance()  # PROC
        name_tok = self.advance()
        name = name_tok.value.upper()
        opt_toks = []
        while not self.at_semi() and not self.at_eof():
            opt_toks.append(self.advance())
        self._end_stmt()
        stmts: list[A.ProcStmt] = []
        interactive = name in INTERACTIVE_PROCS
        while True:
            if self._at_step_boundary():
                break
            t = self.peek()
            if t.kind is Kind.SEMI:
                self.advance()
                continue
            if t.kind is Kind.NAME and t.upper == "QUIT":
                self.skip_statement()
                break
            if t.kind is Kind.NAME and t.upper == "RUN" and (self.at_semi(k=1) or self.at_name("CANCEL", k=1)):
                self.skip_statement()
                if not interactive:
                    break
                continue
            if t.kind is Kind.DATALINES:
                self.advance()
                stmts.append(A.ProcStmt("DATALINES", t.value, [], t.line))
                continue
            toks = []
            while not self.at_semi() and not self.at_eof():
                if self.peek().kind is Kind.DATALINES:
                    break
                toks.append(self.advance())
            if self.at_semi():
                self.advance()
            kw = toks[0].value.upper() if toks and toks[0].kind is Kind.NAME else (toks[0].value if toks else "")
            raw = self.text_of(toks) if name == "SQL" else self.text_of(toks[1:])
            stmts.append(A.ProcStmt(kw, raw, toks, t.line))
            if kw in ("DATALINES", "CARDS", "LINES", "DATALINES4", "CARDS4") and self.peek().kind is Kind.DATALINES:
                d = self.advance()
                stmts.append(A.ProcStmt("DATALINES_BLOCK", d.value, [], d.line))
        last = self.toks[self.i - 1]
        return A.ProcStep(name, self.text_of(opt_toks), opt_toks, stmts, start_line=first.line, end_line=last.line,
                          start=first.start, end=last.end)


def parse(text: str) -> A.Program:
    return Parser(text).parse()


def parse_expression(text: str, where: bool = False) -> A.Expr:
    p = Parser(text)
    e = p.parse_expr(where=where)
    if not p.at_eof():
        raise ParseError("trailing tokens after expression", p.peek())
    return e
