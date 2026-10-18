"""Static SAS macro processor.

Expands open code the way the SAS word scanner + macro facility would for a
*given* symbol table, so that the generated SAS (what SAS would actually
compile) can be parsed and converted.  Things that can only be known at run
time - macro variables assigned by ``CALL SYMPUT[X]`` or ``SELECT ... INTO :``
- become placeholders (``RT_OPEN name RT_CLOSE``) in value positions, and
control flow that depends on them is reported as a diagnostic.

Every output character carries provenance (file, line, macro call stack) so
generated Databricks code can point back to the exact SAS source line.

Deliberate deviation (reported, never silent): macro calls and %LET inside
``* ... ;`` comment statements are *not* executed by default although SAS does
execute them (diagnostic ``H-MACRO-IN-COMMENT``); pass
``comment_macros="execute"`` to emulate SAS exactly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from ..lang.lexer import OPAQUE, RT_CLOSE, RT_OPEN

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REF_RUN_RE = re.compile(r"&[&A-Za-z0-9_.]*")
_WORD_BOUNDARY = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

from . import evaluator  # noqa: E402
from .quoting import MASK as _MASK, mask, unmask  # noqa: E402


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Origin:
    file: str
    line: int
    stack: tuple[str, ...] = ()

    def describe(self) -> str:
        loc = f"{os.path.basename(self.file)}:{self.line}"
        return loc if not self.stack else f"{loc} (via {' > '.join(self.stack)})"


@dataclass
class Diagnostic:
    code: str
    message: str
    origin: Origin
    severity: str = "warning"   # info | warning | error

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code} {self.origin.describe()}: {self.message}"


@dataclass
class MacroDef:
    name: str
    positional: list[str]
    keyword: dict[str, str]
    body: str
    origin: Origin
    options: frozenset[str] = frozenset()


@dataclass
class ExpandResult:
    text: str
    pieces: list[tuple[int, Origin]]          # (output offset where piece starts, origin)
    diagnostics: list[Diagnostic]
    macros: dict[str, MacroDef]
    symbols: dict[str, str]
    runtime_vars: set[str]
    includes: list[str]
    macro_calls: dict[str, int] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)   # macro variables referenced but never defined
    _line_cache: list[int] | None = None

    def origin_at(self, offset: int) -> Origin:
        """Origin of the output character at ``offset``."""
        import bisect

        starts = [p[0] for p in self.pieces]
        k = bisect.bisect_right(starts, offset) - 1
        if k < 0:
            return Origin("<unknown>", 1)
        start, org = self.pieces[k]
        # Count newlines between piece start and offset to advance the line.
        return Origin(org.file, org.line + self.text.count("\n", start, offset), org.stack)


class MacroError(Exception):
    pass


class _Return(Exception):
    """%RETURN inside a macro."""


class _Abort(Exception):
    """%ABORT."""


@dataclass
class _Src:
    """A text being scanned and where its first character came from."""

    text: str
    origin: Origin

    def origin_at(self, pos: int) -> Origin:
        return Origin(self.origin.file, self.origin.line + self.text.count("\n", 0, pos), self.origin.stack)


class _Out:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pieces: list[tuple[int, Origin]] = []
        self.length = 0

    def emit(self, text: str, origin: Origin) -> None:
        if not text:
            return
        if not self.pieces or self.pieces[-1][1] != origin or self._needs_new_piece(origin):
            self.pieces.append((self.length, origin))
        self.parts.append(text)
        self.length += len(text)

    def _needs_new_piece(self, origin: Origin) -> bool:
        # Same origin object but text emitted later may be from a later line; we always
        # start a new piece when the caller passes a fresh origin, so nothing to do.
        return False

    def text(self) -> str:
        return "".join(self.parts)


class _Sub:
    """Collects processed text without provenance (macro arguments, %LET values...)."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def emit(self, text: str, origin: Origin) -> None:  # noqa: ARG002 - same signature as _Out
        self.parts.append(text)

    def text(self) -> str:
        return "".join(self.parts)


_KEYWORD_STATEMENTS = {
    "LET", "PUT", "GLOBAL", "LOCAL", "IF", "DO", "END", "ELSE", "THEN", "MACRO", "MEND",
    "INCLUDE", "INC", "SYMDEL", "RETURN", "ABORT", "GOTO", "SYSEXEC", "SYSCALL", "SYSLPUT",
    "SYSRPUT", "WINDOW", "DISPLAY", "INPUT", "COPY", "SYSMACDELETE", "TO", "BY", "WHILE", "UNTIL",
}
_QUOTING_FUNCS = {"STR", "NRSTR", "QUOTE", "NRQUOTE", "BQUOTE", "NRBQUOTE", "SUPERQ", "UNQUOTE"}
_TEXT_FUNCS = {
    "EVAL", "SYSEVALF", "UPCASE", "QUPCASE", "LOWCASE", "QLOWCASE", "SUBSTR", "QSUBSTR", "SCAN",
    "QSCAN", "INDEX", "LENGTH", "SYSFUNC", "QSYSFUNC", "SYSGET", "SYMEXIST", "SYMGLOBL", "SYMLOCAL",
    "SYSMACEXIST", "CMPRES", "QCMPRES", "LEFT", "QLEFT", "TRIM", "QTRIM", "VERIFY", "SYSPROD",
    "DATATYP", "SYSMEXECDEPTH", "SYSMEXECNAME", "KUPCASE", "KSUBSTR", "KSCAN", "KINDEX", "KLENGTH",
}

DEFAULT_RUNTIME_VARS = {"SQLOBS", "SQLRC", "SQLOOPS", "SQLEXITCODE", "SYSNOBS", "SYSLAST", "SQLXRC", "SQLXMSG"}
# automatic variables set by the run itself (job date, user, host, -SYSPARM): never frozen at conversion time
RUNTIME_AUTOMATIC = {"SYSDATE", "SYSDATE9", "SYSDAY", "SYSTIME", "SYSPARM", "SYSUSERID", "SYSJOBID", "SYSHOSTNAME"}
_CLOCK_FUNCS = {"TODAY", "DATE", "DATETIME", "TIME"}
_AMP_REF_RE = re.compile(r"&([A-Za-z_][A-Za-z0-9_]*)")
_FMT_KEY_RE = re.compile(r"[A-Za-z]*\d*\.\d*")


class _Undecidable(Exception):
    """Macro control flow that cannot be decided at conversion time (run-time or undefined values)."""


class MacroProcessor:
    def __init__(
        self,
        *,
        include_paths: tuple[str, ...] | list[str] = (),
        symbols: dict[str, str] | None = None,
        now: datetime | None = None,
        runtime_vars: set[str] | None = None,
        comment_macros: str = "skip",      # 'skip' (default, flagged) | 'execute' (SAS-exact)
        max_steps: int = 2_000_000,
        include_reader=None,
        param_modules: set[str] | None = None,   # %INCLUDE'd parameter files (basenames): values come at run time
    ) -> None:
        self.include_paths = [p for p in include_paths]
        self.global_scope: dict[str, str] = {}
        self.local_stack: list[dict[str, str]] = []
        self.macros: dict[str, MacroDef] = {}
        self.diagnostics: list[Diagnostic] = []
        self.runtime_vars: set[str] = set(DEFAULT_RUNTIME_VARS) | RUNTIME_AUTOMATIC | {v.upper() for v in (runtime_vars or ())}
        self.runtime_prefixes: set[str] = set()
        self.unresolved: set[str] = set()          # referenced but defined nowhere (parameters supplied from outside?)
        self._unresolved_log: list[str] = []
        self.comment_macros = comment_macros
        self.includes: list[str] = []
        self.macro_calls: dict[str, int] = {}
        self._steps = 0
        self._max_steps = max_steps
        self._call_depth = 0
        self._exec_names: list[str] = []
        self._include_reader = include_reader or _read_file
        self.param_modules = {m.lower() for m in (param_modules or ())}
        self._out_for_scan: _Out | None = None
        self._scanned_upto = 0
        now = now or datetime(2026, 1, 1, 9, 0, 0)
        self._now = now
        self._init_automatic()
        for k, v in (symbols or {}).items():
            self.global_scope[k.upper()] = str(v)

    # ------------------------------------------------------------------ setup
    def _init_automatic(self) -> None:
        # RUNTIME_AUTOMATIC variables (dates, user, host, SYSPARM) are absent on purpose: SasContext supplies them
        auto = {
            "SYSSCP": "LIN X64",
            "SYSSCPL": "Linux",
            "SYSVER": "9.4",
            "SYSVLONG": "9.04.01M7P080520",
            "SYSVLONG4": "9.04.01M7P08052020",
            "SYSERR": "0",
            "SYSCC": "0",
            "SYSRC": "0",
            "SYSENCODING": "utf-8",
            "SYSPROCESSNAME": "Program",
            "SYSSITE": "0",
            "SYSMAXLONG": "9007199254740992",
            "SYSINDEX": "0",
            "SYSDSN": "_NULL_",
            "SYSFILRC": "0",
            "SYSLIBRC": "0",
            "SYSWARNINGTEXT": "",
            "SYSERRORTEXT": "",
            "SYSODSPATH": "WORK.TEMPLAT(UPDATE) SASUSER.TEMPLAT(READ) SASHELP.TMPLMST(READ)",
        }
        self.automatic = auto

    # ------------------------------------------------------------------ public
    def expand(self, text: str, file: str = "<input>") -> ExpandResult:
        out = _Out()
        self._out_for_scan = out
        src = _Src(text, Origin(file, 1))
        try:
            self._process(src, 0, len(text), out, stmt_start=True)
        except _Abort:
            self._diag("M-ABORT", "%ABORT encountered; expansion stopped", src.origin)
        result_text = unmask(out.text())
        return ExpandResult(
            text=result_text,
            pieces=out.pieces,
            diagnostics=self.diagnostics,
            macros=self.macros,
            symbols=dict(self.global_scope),
            runtime_vars=set(self.runtime_vars),
            includes=list(self.includes),
            macro_calls=dict(self.macro_calls),
            unresolved=set(self.unresolved),
        )

    # ------------------------------------------------------------------ symbols
    def lookup(self, name: str) -> str | None:
        key = name.upper()
        for scope in reversed(self.local_stack):
            if key in scope:
                return scope[key]
        if key in self.global_scope:
            return self.global_scope[key]
        if key == "SYSMACRONAME":
            return self._exec_names[-1] if self._exec_names else ""
        if key in self.automatic:
            return self.automatic[key]
        return None

    def assign(self, name: str, value: str) -> None:
        key = name.upper()
        for scope in reversed(self.local_stack):
            if key in scope:
                scope[key] = value
                return
        if key in self.global_scope or not self.local_stack:
            self.global_scope[key] = value
        else:
            self.local_stack[-1][key] = value

    def _make_runtime(self, name: str) -> None:
        key = name.upper()
        self.runtime_vars.add(key)
        for scope in [*self.local_stack, self.global_scope]:
            scope.pop(key, None)

    def _unresolved_since(self, mark: int, text: str = "") -> list[str]:
        """Undefined macro variables met since ``mark`` (a position in the log) or still present as '&name' in ``text``."""
        names = set(self._unresolved_log[mark:])
        names |= {m.group(1).upper() for m in _AMP_REF_RE.finditer(unmask(text)) if m.group(1).upper() in self.unresolved}
        return sorted(names)

    def _emit_undecidable(self, what: str, names: list[str], origin: Origin, out) -> None:
        """Macro logic that needs values nothing defines: a visible unsupported step, never a guessed branch."""
        refs = ", ".join("&" + n for n in names)
        self._diag("UNRESOLVED_MACRO_CONTROL", f"{what} depends on {refs}, which nothing defines", origin, "error")
        msg = (f"UNRESOLVED_MACRO_CONTROL at {origin.describe()}: {what} depends on {refs}, which nothing defines "
               "(define it with --define NAME=VALUE or declare a run-time parameter with --param NAME)")
        out.emit(f"\n__sas2dbx_unresolved__ '{msg.replace(chr(39), chr(39) * 2)}';\n", origin)

    def _is_runtime(self, name: str) -> bool:
        key = name.upper()
        if key in self.runtime_vars:
            return True
        self._scan_runtime_assignments()
        if key in self.runtime_vars:
            return True
        return any(key.startswith(p) for p in self.runtime_prefixes)

    _SYMPUT_RE = re.compile(r"call\s+symputx?\s*\(\s*(['\"])\s*([A-Za-z_][A-Za-z0-9_]*)", re.I)
    _SYMPUT_DYN_RE = re.compile(r"call\s+symputx?\s*\(\s*(['\"])\s*([A-Za-z_][A-Za-z0-9_]*)\s*\1\s*\|\|", re.I)
    _SYMPUT_CATS_RE = re.compile(r"call\s+symputx?\s*\(\s*cats?x?\s*\(\s*(['\"])\s*([A-Za-z_][A-Za-z0-9_]*)\s*\1", re.I)
    _INTO_RE = re.compile(r"\binto\s*((?::\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:-\s*:?\s*[A-Za-z_][A-Za-z0-9_]*)?\s*,?\s*)+)", re.I)

    def _scan_runtime_assignments(self) -> None:
        out = self._out_for_scan
        if out is None:
            return
        text = out.text()
        chunk = text[max(0, self._scanned_upto - 200):]
        self._scanned_upto = len(text)
        for m in self._SYMPUT_RE.finditer(chunk):
            self.runtime_vars.add(m.group(2).upper())
        for rx in (self._SYMPUT_DYN_RE, self._SYMPUT_CATS_RE):
            for m in rx.finditer(chunk):
                self.runtime_prefixes.add(m.group(2).upper())
        for m in self._INTO_RE.finditer(chunk):
            for name in re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", m.group(1)):
                stem = re.sub(r"\d+$", "", name.upper())
                self.runtime_vars.add(name.upper())
                if stem != name.upper():
                    self.runtime_prefixes.add(stem)
            self.runtime_vars.add("SQLOBS")

    # ------------------------------------------------------------------ diagnostics
    def _diag(self, code: str, message: str, origin: Origin, severity: str = "warning") -> None:
        self.diagnostics.append(Diagnostic(code, message, origin, severity))

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > self._max_steps:
            raise MacroError("macro expansion exceeded step budget (runaway %DO loop or recursion?)")

    # ------------------------------------------------------------------ main scanner
    def _process(self, src: _Src, i: int, end: int, out, *, stmt_start: bool, in_dquote: bool = False) -> bool:
        """Process src.text[i:end] into ``out``.  Returns statement-start state at the end."""
        text = src.text
        plain_start = i

        def flush(upto: int) -> None:
            if upto > plain_start:
                out.emit(text[plain_start:upto], src.origin_at(plain_start))

        while i < end:
            self._tick()
            ch = text[i]
            if not in_dquote and ch == "'":
                j = _skip_squote(text, i, end)
                i = j
                stmt_start = False
                continue
            if not in_dquote and ch == '"':
                flush(i)
                j = _find_dquote_end(text, i, end)
                out.emit('"', src.origin_at(i))
                self._process(src, i + 1, j - 1 if j - 1 > i else i + 1, out, stmt_start=False, in_dquote=True)
                out.emit('"', src.origin_at(j - 1))
                i = j
                plain_start = i
                stmt_start = False
                continue
            if not in_dquote and ch == "/" and text.startswith("/*", i):
                j = text.find("*/", i + 2, end)
                i = end if j < 0 else j + 2
                continue
            if ch == "%" and i + 1 < end and text[i + 1] == "*" and not in_dquote:
                flush(i)
                i = _find_semi(text, i + 2, end)
                plain_start = i
                continue
            if not in_dquote and ch == "*" and stmt_start:
                j = _find_semi(text, i + 1, end)
                body = text[i:j]
                if ("%" in body or "&" in body) and _has_macro_trigger(body):
                    if self.comment_macros == "execute":
                        flush(i)
                        out.emit("*", src.origin_at(i))
                        stmt_start = self._process(src, i + 1, j, out, stmt_start=False)
                        i = j
                        plain_start = i
                        continue
                    self._diag(
                        "H-MACRO-IN-COMMENT",
                        "macro trigger inside a '* ...;' comment statement; SAS would execute it, "
                        "sas2dbx treats it as a comment (use comment_macros='execute' for SAS-exact)",
                        src.origin_at(i),
                        "info",
                    )
                i = j
                stmt_start = True
                continue
            if ch == "%" and i + 1 < end and (text[i + 1].isalpha() or text[i + 1] == "_"):
                m = _NAME_RE.match(text, i + 1)
                name = m.group(0)
                flush(i)
                i, stmt_start = self._percent(src, i, m.end(), name, end, out, stmt_start, in_dquote)
                plain_start = i
                continue
            if ch == "&" and i + 1 < end and (text[i + 1].isalpha() or text[i + 1] in "_&"):
                m = _REF_RUN_RE.match(text, i)
                run = m.group(0)
                # Trailing '.'s are only part of the run if they terminate a reference.
                flush(i)
                resolved = self._resolve_run(run, src.origin_at(i))
                out.emit(resolved, src.origin_at(i))
                i = m.end()
                plain_start = i
                if resolved.strip():
                    stmt_start = resolved.rstrip().endswith(";")
                continue
            if ch == ";" and not in_dquote:
                stmt_start = True
                i += 1
                continue
            if not ch.isspace() and not in_dquote:
                stmt_start = False
            i += 1
        flush(end)
        return stmt_start

    # ------------------------------------------------------------------ & references
    def _resolve_run(self, run: str, origin: Origin) -> str:
        s = run
        for _ in range(64):
            if "&" not in s:
                return s
            parts: list[str] = []
            k = 0
            rescan = False
            progressed = False
            while k < len(s):
                c = s[k]
                if c != "&":
                    parts.append(c)
                    k += 1
                    continue
                if s.startswith("&&", k):
                    parts.append("&")
                    k += 2
                    rescan = True
                    progressed = True
                    continue
                m = _NAME_RE.match(s, k + 1)
                if not m:
                    parts.append("&")
                    k += 1
                    continue
                name = m.group(0)
                j = m.end()
                if j < len(s) and s[j] == ".":
                    j += 1
                val = self.lookup(name)
                if val is None:
                    if self._is_runtime(name):
                        parts.append(f"{RT_OPEN}{name.upper()}{RT_CLOSE}")
                        progressed = True
                    else:
                        if not rescan:
                            self.unresolved.add(name.upper())
                            self._unresolved_log.append(name.upper())
                            self._diag("M-UNRESOLVED", f"apparent symbolic reference &{name} not resolved", origin)
                        parts.append(s[k:j])
                else:
                    parts.append(val)
                    progressed = True
                k = j
            s = "".join(parts)
            if not rescan or not progressed:
                return s
        raise MacroError(f"too many resolution passes for {run!r}")

    def resolve_text(self, text: str, origin: Origin, *, stmt_start: bool = False) -> str:
        """Fully macro-process a fragment (arguments, %LET values, conditions)."""
        sub = _Sub()
        self._process(_Src(text, origin), 0, len(text), sub, stmt_start=stmt_start)
        return sub.text()

    # ------------------------------------------------------------------ % triggers
    def _percent(self, src, i, name_end, name, end, out, stmt_start, in_dquote):
        uname = name.upper()
        text = src.text
        origin = src.origin_at(i)
        if uname in _KEYWORD_STATEMENTS:
            return self._statement(src, i, name_end, uname, end, out, stmt_start)
        if uname in _QUOTING_FUNCS or uname in _TEXT_FUNCS:
            j = _skip_ws(text, name_end, end)
            if j < end and text[j] == "(":
                close = _match_paren(text, j, end)
                arg_raw = text[j + 1:close]
                value = self._function(uname, arg_raw, src, j + 1, origin)
                out.emit(value, origin)
                return close + 1, stmt_start if not value.strip() else value.rstrip().endswith(";")
            self._diag("M-SYNTAX", f"%{name} requires parentheses", origin, "error")
            out.emit(text[i:name_end], origin)
            return name_end, stmt_start
        if uname in self.macros:
            return self._call(src, i, name_end, self.macros[uname], end, out, stmt_start)
        if self._autocall(uname, origin):
            return self._call(src, i, name_end, self.macros[uname], end, out, stmt_start)
        # Unknown macro (autocall library not available) or literal text such as '%' in a title.
        self._diag("M-UNKNOWN-MACRO", f"%{name} is not defined (autocall macro not supplied?)", origin)
        out.emit(text[i:name_end], origin)
        return name_end, stmt_start

    # ---- statements
    def _statement(self, src, i, name_end, kw, end, out, stmt_start):
        text = src.text
        origin = src.origin_at(i)
        if kw == "LET":
            j = _find_stmt_end(text, name_end, end)
            body = text[name_end:j]
            eq = _find_top_level(body, "=")
            if eq < 0:
                self._diag("M-SYNTAX", "%LET without '='", origin, "error")
                return j + 1, stmt_start
            var = self.resolve_text(body[:eq], origin).strip()
            mark = len(self._unresolved_log)
            value = self.resolve_text(body[eq + 1:], src.origin_at(name_end + eq + 1))
            value = _strip_unmasked(value)
            deps = self._unresolved_since(mark, value)
            if not _NAME_RE.fullmatch(unmask(var).strip() or "_"):
                self._diag("M-SYNTAX", f"invalid macro variable name in %LET: {var!r}", origin, "error")
            elif deps and RT_OPEN + OPAQUE not in value:
                # leave it undefined: every later use is then flagged where it matters, never a frozen guess
                name = unmask(var).strip().upper()
                for scope in [*self.local_stack, self.global_scope]:
                    scope.pop(name, None)
                self.unresolved.add(name)
                self._diag("M-UNRESOLVED", f"&{name} depends on {', '.join('&' + d for d in deps)}, which nothing defines", origin)
            elif RT_OPEN + OPAQUE in value:
                name = unmask(var).strip().upper()
                self._make_runtime(name)
                stmt = " ".join(f"%let {unmask(var).strip()} = {body[eq + 1:].strip()}".split())
                msg = f"RUNTIME_LET at {origin.describe()}: {stmt}".replace("'", "''")
                out.emit(f"\n__sas2dbx_unresolved__ '{msg}';\n", origin)
            else:
                self.assign(unmask(var).strip(), value)
            return j + 1, stmt_start
        if kw in ("GLOBAL", "LOCAL"):
            j = _find_stmt_end(text, name_end, end)
            names = self.resolve_text(text[name_end:j], origin).split()
            for nm in names:
                if nm.upper() in ("/", "READONLY"):
                    continue
                if kw == "GLOBAL":
                    self.global_scope.setdefault(nm.upper(), "")
                elif self.local_stack:
                    self.local_stack[-1].setdefault(nm.upper(), "")
                else:
                    self.global_scope.setdefault(nm.upper(), "")
            return j + 1, stmt_start
        if kw == "PUT":
            j = _find_stmt_end(text, name_end, end)
            self._diag("M-PUT", unmask(self.resolve_text(text[name_end:j], origin)).strip(), origin, "info")
            return j + 1, stmt_start
        if kw == "SYMDEL":
            j = _find_stmt_end(text, name_end, end)
            for nm in self.resolve_text(text[name_end:j], origin).split():
                if nm.startswith("/"):
                    break
                self.global_scope.pop(nm.upper(), None)
            return j + 1, stmt_start
        if kw == "MACRO":
            return self._define(src, i, name_end, end), stmt_start
        if kw == "MEND":
            self._diag("M-SYNTAX", "%MEND without matching %MACRO", origin, "error")
            return _find_stmt_end(text, name_end, end) + 1, stmt_start
        if kw == "IF":
            return self._if(src, i, name_end, end, out, stmt_start)
        if kw == "DO":
            return self._do(src, i, name_end, end, out, stmt_start)
        if kw in ("INCLUDE", "INC"):
            return self._include(src, i, name_end, end, out, stmt_start)
        if kw == "RETURN":
            if self._call_depth:
                raise _Return()
            return _find_stmt_end(text, name_end, end) + 1, stmt_start
        if kw == "ABORT":
            raise _Abort()
        if kw in ("END", "ELSE", "THEN", "TO", "BY", "WHILE", "UNTIL"):
            self._diag("M-SYNTAX", f"%{kw} out of place", origin, "error")
            return name_end, stmt_start
        # %SYSEXEC, %SYSCALL, %GOTO, %WINDOW ... : not convertible statically.
        j = _find_stmt_end(text, name_end, end)
        self._diag("M-UNSUPPORTED", f"%{kw} is not supported by static expansion", origin, "error")
        return j + 1, stmt_start

    def _define(self, src: _Src, i: int, name_end: int, end: int) -> int:
        text = src.text
        origin = src.origin_at(i)
        j = _skip_ws(text, name_end, end)
        m = _NAME_RE.match(text, j)
        if not m:
            self._diag("M-SYNTAX", "%MACRO without a name", origin, "error")
            return _find_stmt_end(text, name_end, end) + 1
        mname = m.group(0)
        k = _skip_ws(text, m.end(), end)
        positional: list[str] = []
        keyword: dict[str, str] = {}
        if k < end and text[k] == "(":
            close = _match_paren(text, k, end)
            for raw in _split_top_level(text[k + 1:close], ","):
                raw = raw.strip()
                if not raw:
                    continue
                eq = raw.find("=")
                if eq >= 0:
                    keyword[raw[:eq].strip().upper()] = raw[eq + 1:].strip()
                else:
                    positional.append(raw.upper())
            k = close + 1
        stmt_end = _find_stmt_end(text, k, end)
        opts_text = text[k:stmt_end]
        options = frozenset(o.split("=")[0].strip().upper() for o in opts_text.replace("/", " ").split() if o.strip())
        body_start = stmt_end + 1
        mend = _find_mend(text, body_start, end)
        if mend is None:
            self._diag("M-SYNTAX", f"%MACRO {mname} has no matching %MEND", origin, "error")
            return end
        mend_start, after = mend
        self.macros[mname.upper()] = MacroDef(
            name=mname.upper(),
            positional=positional,
            keyword=keyword,
            body=text[body_start:mend_start],
            origin=src.origin_at(body_start),
            options=options,
        )
        return after

    def _call(self, src: _Src, i: int, name_end: int, mdef: MacroDef, end: int, out, stmt_start: bool):
        text = src.text
        origin = src.origin_at(i)
        args_raw: list[str] = []
        after = name_end
        j = _skip_ws(text, name_end, end)
        has_parens = j < end and text[j] == "("
        if has_parens and (mdef.positional or mdef.keyword or "PARMBUFF" in mdef.options or "PBUFF" in mdef.options):
            close = _match_paren(text, j, end)
            args_raw = _split_top_level(text[j + 1:close], ",")
            after = close + 1
            pbuff = text[j:close + 1]
        else:
            pbuff = ""
        local: dict[str, str] = {p: "" for p in mdef.positional}
        for k, default in mdef.keyword.items():
            local[k] = self.resolve_text(default, mdef.origin) if default else ""
        pos_idx = 0
        for raw in args_raw:
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)\Z", raw, re.S)
            if m and m.group(1).upper() in mdef.keyword or (m and m.group(1).upper() in mdef.positional and pos_idx >= len(mdef.positional)):
                local[m.group(1).upper()] = _strip_unmasked(self.resolve_text(m.group(2), origin))
                continue
            if m and m.group(1).upper() in mdef.positional:
                local[m.group(1).upper()] = _strip_unmasked(self.resolve_text(m.group(2), origin))
                continue
            if pos_idx < len(mdef.positional):
                local[mdef.positional[pos_idx]] = _strip_unmasked(self.resolve_text(raw, origin))
                pos_idx += 1
            elif raw.strip():
                if "PARMBUFF" not in mdef.options and "PBUFF" not in mdef.options:
                    self._diag("M-ARGS", f"too many positional arguments for %{mdef.name}", origin, "error")
        if "PARMBUFF" in mdef.options or "PBUFF" in mdef.options:
            local["SYSPBUFF"] = self.resolve_text(pbuff, origin)
        self.macro_calls[mdef.name] = self.macro_calls.get(mdef.name, 0) + 1
        if self._call_depth > 200:
            raise MacroError(f"macro recursion too deep at %{mdef.name}")
        self.local_stack.append(local)
        self._exec_names.append(mdef.name)
        self._call_depth += 1
        call_site = f"%{mdef.name.lower()}@{os.path.basename(origin.file)}:{origin.line}"
        body_src = _Src(mdef.body, Origin(mdef.origin.file, mdef.origin.line, origin.stack + (call_site,)))
        try:
            stmt_start = self._process(body_src, 0, len(mdef.body), out, stmt_start=stmt_start)
        except _Return:
            pass
        finally:
            self._call_depth -= 1
            self._exec_names.pop()
            self.local_stack.pop()
        return after, stmt_start

    def _autocall(self, uname: str, origin: Origin) -> bool:
        """SASAUTOS behaviour: compile <name>.sas from the autocall/include path on first use."""
        tried = getattr(self, "_autocall_tried", set())
        self._autocall_tried = tried
        if uname in tried:
            return uname in self.macros
        tried.add(uname)
        target = f"{uname.lower()}.sas"
        for d in self.include_paths:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower() == target:
                    path = os.path.join(d, f)
                    content = self._include_reader(path)
                    sink = _Sub()
                    self._process(_Src(content, Origin(path, 1, ("autocall",))), 0, len(content), sink, stmt_start=True)
                    if uname in self.macros:
                        self._diag("M-AUTOCALL", f"%{uname.lower()} compiled from {path}", origin, "info")
                        return True
        return False

    def _include(self, src: _Src, i: int, name_end: int, end: int, out, stmt_start: bool):
        text = src.text
        origin = src.origin_at(i)
        j = _find_stmt_end(text, name_end, end)
        spec = unmask(self.resolve_text(text[name_end:j], origin)).strip()
        m = re.match(r"""(['"])(.*?)\1""", spec)
        target = m.group(2) if m else spec.split()[0] if spec else ""
        self.includes.append(target)
        if os.path.basename(target.replace("\\", "/")).lower() in self.param_modules:
            # the parameter notebook supplies these values when the program runs: never today's file contents
            return j + 1, stmt_start
        path = self._find_include(target, src.origin.file)
        if path is None:
            self._diag("M-INCLUDE-MISSING", f"%INCLUDE {target!r} not found on include path", origin)
            return j + 1, stmt_start
        content = self._include_reader(path)
        inc_src = _Src(content, Origin(path, 1, src.origin.stack + (f"%include@{os.path.basename(origin.file)}:{origin.line}",)))
        stmt_start = self._process(inc_src, 0, len(content), out, stmt_start=True)
        out.emit("\n", origin)
        return j + 1, stmt_start

    def _find_include(self, target: str, current_file: str) -> str | None:
        if not target:
            return None
        norm = target.replace("\\", "/")
        base = os.path.basename(norm)
        cur_dir = os.path.dirname(current_file) if current_file and not current_file.startswith("<") else "."
        candidates = []
        for d in [cur_dir, *self.include_paths]:
            candidates.append(os.path.normpath(os.path.join(d, norm)))
            candidates.append(os.path.join(d, base))
        for c in candidates:
            if os.path.isfile(c):
                return c
            # case-insensitive match (SAS code from Windows)
            dirn = os.path.dirname(c) or "."
            if os.path.isdir(dirn):
                for f in os.listdir(dirn):
                    if f.lower() == os.path.basename(c).lower():
                        return os.path.join(dirn, f)
        return None

    # ---- %IF
    def _if(self, src: _Src, i: int, name_end: int, end: int, out, stmt_start: bool):
        text = src.text
        origin = src.origin_at(i)
        then_pos = _find_keyword(text, name_end, end, "THEN")
        if then_pos is None:
            self._diag("M-SYNTAX", "%IF without %THEN", origin, "error")
            return _find_stmt_end(text, name_end, end) + 1, stmt_start
        cond_raw = text[name_end:then_pos]
        then_start = then_pos + len("%then")
        action_end, then_action = self._parse_action(src, then_start, end)
        # optional %ELSE
        k = _skip_ws_comments(text, action_end, end)
        else_action = None
        after = action_end
        if text[k:k + 5].upper() == "%ELSE" and (k + 5 >= end or text[k + 5] not in _WORD_BOUNDARY):
            after, else_action = self._parse_action(src, k + 5, end)
        mark = len(self._unresolved_log)
        cond_text = self.resolve_text(cond_raw, origin)
        names = self._unresolved_since(mark, cond_text)
        if names and RT_OPEN not in cond_text:
            self._emit_undecidable(f"%IF {unmask(cond_raw).strip()}", names, origin, out)
            return after, stmt_start
        if RT_OPEN in cond_text:
            self._diag(
                "RUNTIME_MACRO_CONTROL",
                f"%IF condition depends on run-time macro variables: {unmask(cond_text).strip()}",
                origin,
                "error",
            )
            out.emit(f"\n__sas2dbx_unresolved__ 'RUNTIME_MACRO_CONTROL at {origin.describe()}';\n", origin)
            return after, stmt_start
        try:
            truth = evaluator.eval_condition(cond_text, minoperator=self._minoperator())
        except evaluator.EvalError as e:
            # SAS stops the macro on this error; picking a branch here would be a guess
            self._diag("M-EVAL", f"%IF condition {unmask(cond_text).strip()!r}: {e}", origin, "error")
            msg = f"MACRO_EVAL_ERROR at {origin.describe()}: %IF {unmask(cond_text).strip()}: {e}".replace("'", "''")
            out.emit(f"\n__sas2dbx_unresolved__ '{msg}';\n", origin)
            return after, stmt_start
        chosen = then_action if truth else else_action
        if chosen is not None:
            stmt_start = self._run_action(src, chosen, out, stmt_start)
        return after, stmt_start

    def _parse_action(self, src: _Src, start: int, end: int):
        """Parse a %THEN/%ELSE action.  Returns (end_index, action) where action is
        ('block', s, e) | ('if', s, e) | ('text', s, e) as offsets into src.text."""
        text = src.text
        k = _skip_ws(text, start, end)
        if text[k:k + 3].upper() == "%DO" and (k + 3 >= end or text[k + 3] not in _WORD_BOUNDARY):
            semi = _skip_ws(text, k + 3, end)
            if semi < end and text[semi] == ";":
                body_start = semi + 1
                end_pos = _find_matching_end(text, body_start, end)
                if end_pos is None:
                    return end, ("block", body_start, end)
                body_end, after = end_pos
                return after, ("block", body_start, body_end)
            # %DO with iteration as action: treat as a nested statement
            stmt_end = self._statement_extent(text, k, end)
            return stmt_end, ("if", k, stmt_end)
        if text[k:k + 3].upper() == "%IF" and (k + 3 >= end or text[k + 3] not in _WORD_BOUNDARY):
            stmt_end = self._statement_extent(text, k, end)
            return stmt_end, ("if", k, stmt_end)
        semi = _find_stmt_end(text, k, end)
        return semi + 1, ("text", k, semi)

    def _statement_extent(self, text: str, k: int, end: int) -> int:
        """Extent of a nested macro statement starting at k (for %ELSE %IF chains / %DO loops)."""
        upper = text[k:k + 3].upper()
        if upper == "%IF":
            then_pos = _find_keyword(text, k + 3, end, "THEN")
            if then_pos is None:
                return _find_stmt_end(text, k, end) + 1
            a_end, _ = self._parse_action(_Src(text, Origin("<x>", 1)), then_pos + 5, end)
            j = _skip_ws_comments(text, a_end, end)
            if text[j:j + 5].upper() == "%ELSE" and (j + 5 >= end or text[j + 5] not in _WORD_BOUNDARY):
                a_end, _ = self._parse_action(_Src(text, Origin("<x>", 1)), j + 5, end)
            return a_end
        # %DO ... ; body %END;
        semi = _find_stmt_end(text, k, end)
        found = _find_matching_end(text, semi + 1, end)
        return end if found is None else found[1]

    def _run_action(self, src: _Src, action, out, stmt_start: bool) -> bool:
        kind, s, e = action
        if kind in ("block", "text"):
            return self._process(src, s, e, out, stmt_start=stmt_start)
        # nested %IF / %DO statement
        return self._process(src, s, e, out, stmt_start=stmt_start)

    # ---- %DO
    def _do(self, src: _Src, i: int, name_end: int, end: int, out, stmt_start: bool):
        text = src.text
        origin = src.origin_at(i)
        k = _skip_ws(text, name_end, end)
        if k < end and text[k] == ";":
            found = _find_matching_end(text, k + 1, end)
            if found is None:
                self._diag("M-SYNTAX", "%DO without %END", origin, "error")
                return end, stmt_start
            body_end, after = found
            stmt_start = self._process(src, k + 1, body_end, out, stmt_start=stmt_start)
            return after, stmt_start
        head_end = _find_stmt_end(text, k, end)
        head = text[k:head_end]
        found = _find_matching_end(text, head_end + 1, end)
        if found is None:
            self._diag("M-SYNTAX", "%DO without %END", origin, "error")
            return end, stmt_start
        body_start = head_end + 1
        body_end, after = found
        hu = head.lstrip().upper()
        if hu.startswith("%WHILE") or hu.startswith("%UNTIL"):
            is_while = hu.startswith("%WHILE")
            p = head.find("(")
            q = head.rfind(")")
            cond_raw = head[p + 1:q] if p >= 0 and q > p else head.split(None, 1)[-1]
            iterations = 0
            try:
                while True:
                    iterations += 1
                    if iterations > 100_000:
                        raise MacroError("%DO %WHILE/%UNTIL exceeded 100000 iterations")
                    if is_while:
                        if not self._cond(cond_raw, origin):
                            break
                    stmt_start = self._process(src, body_start, body_end, out, stmt_start=stmt_start)
                    if not is_while and self._cond(cond_raw, origin):
                        break
            except _Undecidable as e:
                kind, names = e.args
                if kind == "runtime":
                    out.emit(f"\n__sas2dbx_unresolved__ 'RUNTIME_MACRO_CONTROL at {origin.describe()}';\n", origin)
                elif kind == "error":
                    msg = f"MACRO_EVAL_ERROR at {origin.describe()}: %DO {head.strip()}: {names[0]}".replace("'", "''")
                    out.emit(f"\n__sas2dbx_unresolved__ '{msg}';\n", origin)
                else:
                    self._emit_undecidable(f"%DO {head.strip()}", names, origin, out)
            return after, stmt_start
        # iterative: var = start %TO stop [%BY step] [%WHILE(...)|%UNTIL(...)]
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)\Z", head, re.S)
        if not m:
            self._diag("M-SYNTAX", f"unrecognised %DO form: {head.strip()!r}", origin, "error")
            return after, stmt_start
        var = m.group(1)
        rest = m.group(2)
        to_pos = _find_keyword(rest, 0, len(rest), "TO")
        if to_pos is None:
            self._diag("M-SYNTAX", "%DO without %TO", origin, "error")
            return after, stmt_start
        start_txt = rest[:to_pos]
        rest2 = rest[to_pos + 3:]
        by_pos = _find_keyword(rest2, 0, len(rest2), "BY")
        stop_txt = rest2 if by_pos is None else rest2[:by_pos]
        by_txt = "1" if by_pos is None else rest2[by_pos + 3:]
        mark = len(self._unresolved_log)
        try:
            start_v = evaluator.eval_int(self.resolve_text(start_txt, origin))
            stop_v = evaluator.eval_int(self.resolve_text(stop_txt, origin))
            by_v = evaluator.eval_int(self.resolve_text(by_txt, origin))
        except evaluator.EvalError as e:
            resolved = self.resolve_text(start_txt + " " + stop_txt + " " + by_txt, origin)
            names = self._unresolved_since(mark, resolved)
            if RT_OPEN in resolved:
                self._diag("RUNTIME_MACRO_CONTROL", f"%DO bounds depend on run-time macro variables: {unmask(resolved).strip()}", origin, "error")
                out.emit(f"\n__sas2dbx_unresolved__ 'RUNTIME_MACRO_CONTROL at {origin.describe()}';\n", origin)
            elif names:
                self._emit_undecidable(f"%DO {var} bounds", names, origin, out)
            else:
                self._diag("M-EVAL", f"%DO bounds: {e}", origin, "error")
                msg = f"MACRO_EVAL_ERROR at {origin.describe()}: %DO bounds: {e}".replace("'", "''")
                out.emit(f"\n__sas2dbx_unresolved__ '{msg}';\n", origin)
            return after, stmt_start
        if by_v == 0:
            self._diag("M-EVAL", "%DO %BY 0", origin, "error")
            return after, stmt_start
        v = start_v
        count = 0
        while (by_v > 0 and v <= stop_v) or (by_v < 0 and v >= stop_v):
            count += 1
            if count > 100_000:
                raise MacroError("%DO loop exceeded 100000 iterations")
            self.assign(var, str(v))
            stmt_start = self._process(src, body_start, body_end, out, stmt_start=stmt_start)
            cur = self.lookup(var)
            try:
                v = int(cur) + by_v
            except (TypeError, ValueError):
                v = v + by_v
        self.assign(var, str(v))
        return after, stmt_start

    def _cond(self, raw: str, origin: Origin) -> bool:
        mark = len(self._unresolved_log)
        text = self.resolve_text(raw, origin)
        if RT_OPEN in text:
            self._diag("RUNTIME_MACRO_CONTROL", f"loop condition depends on run-time values: {unmask(text).strip()}", origin, "error")
            raise _Undecidable("runtime", [])
        names = self._unresolved_since(mark, text)
        if names:
            raise _Undecidable("unresolved", names)
        try:
            return evaluator.eval_condition(text, minoperator=self._minoperator())
        except evaluator.EvalError as e:
            self._diag("M-EVAL", str(e), origin, "error")
            raise _Undecidable("error", [str(e)]) from e

    def _minoperator(self) -> bool:
        if self._exec_names:
            mdef = self.macros.get(self._exec_names[-1])
            if mdef and "MINOPERATOR" in mdef.options:
                return True
        return False

    # ---- functions
    def _function(self, fname: str, arg_raw: str, src: _Src, arg_offset: int, origin: Origin) -> str:
        if fname in ("NRSTR",):
            return mask(_process_escapes(arg_raw))
        if fname == "STR":
            resolved = self.resolve_text(_process_escapes_keep(arg_raw), origin)
            return mask(_unescape(resolved))
        if fname in ("QUOTE", "NRQUOTE", "BQUOTE", "NRBQUOTE"):
            resolved = self.resolve_text(arg_raw, origin) if fname in ("QUOTE", "BQUOTE") else arg_raw
            if fname in ("NRQUOTE", "NRBQUOTE"):
                resolved = self.resolve_text(arg_raw, origin)
            return mask(resolved)
        if fname == "SUPERQ":
            name = unmask(self.resolve_text(arg_raw, origin)).strip()
            val = self.lookup(name)
            if val is None:
                self._diag("M-UNRESOLVED", f"%SUPERQ({name}) refers to an undefined variable", origin)
                return ""
            return mask(val)
        if fname == "UNQUOTE":
            return unmask(self.resolve_text(arg_raw, origin))
        args_text = self.resolve_text(arg_raw, origin)
        quoted_result = fname.startswith("Q")
        if RT_OPEN in args_text:
            # %UPCASE/%SUBSTR/... of a run-time value: the result is only known at run time too
            self._diag("RUNTIME_MACRO_CONTROL", f"%{fname} applied to a run-time value", origin, "error")
            return f"{RT_OPEN}{OPAQUE}{fname}{RT_CLOSE}"
        try:
            result = self._text_function(fname, args_text, origin)
        except evaluator.EvalError as e:
            # never guess (an empty result would silently change the program): opaque until run time,
            # so the dependent %LET / step becomes visible unsupported work (RUNTIME_LET / codegen Unsupported)
            code = "RUNTIME_MACRO_CONTROL" if "run-time" in str(e) else "M-EVAL"
            self._diag(code, f"%{fname} could not be evaluated at conversion time: {e}", origin, "error")
            return f"{RT_OPEN}{OPAQUE}{fname}{RT_CLOSE}"
        return mask(result) if quoted_result else result

    def _text_function(self, fname: str, args_text: str, origin: Origin) -> str:
        f = fname[1:] if fname.startswith("Q") and fname[1:] in ("UPCASE", "LOWCASE", "SUBSTR", "SCAN", "SYSFUNC", "CMPRES", "LEFT", "TRIM") else fname
        if f.startswith("K") and f[1:] in ("UPCASE", "SUBSTR", "SCAN", "INDEX", "LENGTH"):
            f = f[1:]
        if f == "EVAL":
            if RT_OPEN in args_text:
                raise evaluator.EvalError("run-time macro variable in %EVAL")
            return str(evaluator.eval_int_expr(args_text, minoperator=self._minoperator()))
        if f == "SYSEVALF":
            parts = _split_top_level(args_text, ",")
            conv = unmask(parts[1]).strip() if len(parts) > 1 else ""
            return evaluator.sysevalf(parts[0], conv)
        if f == "UPCASE":
            return args_text.upper()
        if f == "LOWCASE":
            return args_text.lower()
        if f == "LENGTH":
            return str(len(unmask(args_text)))
        if f == "INDEX":
            a, b = (_split_top_level(args_text, ",") + [""])[:2]
            return str(unmask(a).find(unmask(b)) + 1)
        if f == "SUBSTR":
            parts = _split_top_level(args_text, ",")
            s = parts[0]
            pos = evaluator.eval_int(parts[1])
            if len(parts) > 2:
                n = evaluator.eval_int(parts[2])
                return s[pos - 1:pos - 1 + n]
            return s[pos - 1:]
        if f == "SCAN":
            parts = _split_top_level(args_text, ",")
            from ..runtime import pyfuncs

            delims = unmask(parts[2]) if len(parts) > 2 else None
            return pyfuncs.scan(unmask(parts[0]), evaluator.eval_int(parts[1]), delims) or ""
        if f in ("CMPRES",):
            return re.sub(r"\s+", " ", args_text).strip()
        if f == "LEFT":
            return args_text.lstrip()
        if f == "TRIM":
            return args_text.rstrip()
        if f == "VERIFY":
            a, b = (_split_top_level(args_text, ",") + [""])[:2]
            for idx, ch in enumerate(unmask(a)):
                if ch not in unmask(b):
                    return str(idx + 1)
            return "0"
        if f == "SYMEXIST":
            return "1" if self.lookup(unmask(args_text).strip()) is not None else "0"
        if f == "SYMGLOBL":
            return "1" if unmask(args_text).strip().upper() in self.global_scope else "0"
        if f == "SYMLOCAL":
            nm = unmask(args_text).strip().upper()
            return "1" if any(nm in s for s in self.local_stack) else "0"
        if f == "SYSMACEXIST":
            return "1" if unmask(args_text).strip().upper() in self.macros else "0"
        if f == "SYSGET":
            # the environment of the run, not of the converter (which may hold secrets): read when the job runs
            key = "SYSGET_" + re.sub(r"\W", "_", unmask(args_text).strip().upper())
            self.runtime_vars.add(key)
            return f"{RT_OPEN}{key}{RT_CLOSE}"
        if f == "SYSPROD":
            return "1"
        if f == "SYSMEXECDEPTH":
            return str(self._call_depth)
        if f == "SYSMEXECNAME":
            return self._exec_names[-1] if self._exec_names else "OPEN CODE"
        if f == "DATATYP":
            v = unmask(args_text).strip()
            try:
                float(v)
                return "NUMERIC"
            except ValueError:
                return "CHAR"
        if f == "SYSFUNC":
            return self._sysfunc(args_text, origin)
        raise evaluator.EvalError(f"%{fname} not implemented")

    def _sysfunc(self, args_text: str, origin: Origin) -> str:
        from ..runtime import pyfuncs

        parts = _split_top_level(args_text, ",")
        call = parts[0].strip()
        fmt = unmask(parts[1]).strip() if len(parts) > 1 else None
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*\Z", call, re.S)
        if not m:
            raise evaluator.EvalError(f"bad %SYSFUNC call {unmask(call)!r}")
        fn = m.group(1).upper()
        raw_args = [unmask(a).strip() for a in _split_top_level(m.group(2), ",")] if m.group(2).strip() else []
        if any(RT_OPEN in a for a in raw_args):
            raise evaluator.EvalError("run-time macro variable in %SYSFUNC")
        if fn in _CLOCK_FUNCS and not raw_args:
            # the date/time of the run, not of the conversion: SasContext computes SYSFUNC_<FN>[__<FORMAT>]
            if fmt and not _FMT_KEY_RE.fullmatch(fmt):
                raise evaluator.EvalError(f"run-time clock value %SYSFUNC({fn}()) with format {fmt}")
            key = f"SYSFUNC_{fn}" + (f"__{fmt.upper().rstrip('.').replace('.', '_')}" if fmt else "")
            self.runtime_vars.add(key)
            return f"{RT_OPEN}{key}{RT_CLOSE}"
        try:
            value = pyfuncs.sysfunc(fn, raw_args, now=self._now)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            raise evaluator.EvalError(f"%SYSFUNC({fn}): {e}") from e
        if fmt:
            return pyfuncs.put(value, fmt).strip() if isinstance(value, (int, float)) or value is None else pyfuncs.put(value, fmt)
        if isinstance(value, float):
            return pyfuncs.put(value, "BEST12.").strip()
        if value is None:
            return "."
        return str(value)


# ---------------------------------------------------------------------------
# raw-text scanning helpers
def _read_file(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _has_macro_trigger(s: str) -> bool:
    return bool(re.search(r"%[A-Za-z_]|&[A-Za-z_&]", s))


def _skip_squote(text: str, i: int, end: int) -> int:
    j = i + 1
    while j < end:
        if text[j] == "'":
            if j + 1 < end and text[j + 1] == "'":
                j += 2
                continue
            return j + 1
        j += 1
    return end


def _find_dquote_end(text: str, i: int, end: int) -> int:
    j = i + 1
    while j < end:
        if text[j] == '"':
            if j + 1 < end and text[j + 1] == '"':
                j += 2
                continue
            return j + 1
        j += 1
    return end


def _skip_ws(text: str, i: int, end: int) -> int:
    while i < end and text[i].isspace():
        i += 1
    return i


def _skip_ws_comments(text: str, i: int, end: int) -> int:
    while i < end:
        if text[i].isspace():
            i += 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            i = end if j < 0 else j + 2
        else:
            break
    return i


def _find_semi(text: str, i: int, end: int) -> int:
    """Index just past the next ';' (quotes respected)."""
    while i < end:
        ch = text[i]
        if ch == ";":
            return i + 1
        if ch == "'":
            i = _skip_squote(text, i, end)
            continue
        if ch == '"':
            i = _find_dquote_end(text, i, end)
            continue
        i += 1
    return end


def _find_stmt_end(text: str, i: int, end: int) -> int:
    """Index of the ';' ending a macro statement, honouring quotes, comments and
    parentheses (so %STR(;) inside a %LET value does not end it)."""
    depth = 0
    while i < end:
        ch = text[i]
        if ch == "'":
            i = _skip_squote(text, i, end)
            continue
        if ch == '"':
            i = _find_dquote_end(text, i, end)
            continue
        if ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            i = end if j < 0 else j + 2
            continue
        if ch == "%" and i + 1 < end and text[i + 1] in "();,'\"%":
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            return i
        i += 1
    return end


def _match_paren(text: str, i: int, end: int) -> int:
    """Index of the ')' matching text[i] == '('."""
    depth = 0
    j = i
    while j < end:
        ch = text[j]
        if ch == "'":
            j = _skip_squote(text, j, end)
            continue
        if ch == '"':
            j = _find_dquote_end(text, j, end)
            continue
        if ch == "%" and j + 1 < end and text[j + 1] in "()'\"%":
            j += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise MacroError("unbalanced parentheses in macro call")


def _split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in "'\"":
            j = _skip_squote(s, i, n) if ch == "'" else _find_dquote_end(s, i, n)
            cur.append(s[i:j])
            i = j
            continue
        if ch == "%" and i + 1 < n and s[i + 1] in "(),'\"%":
            cur.append(s[i:i + 2])
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _find_top_level(s: str, target: str) -> int:
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "'\"":
            i = _skip_squote(s, i, len(s)) if ch == "'" else _find_dquote_end(s, i, len(s))
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == target and depth == 0:
            return i
        i += 1
    return -1


def _find_keyword(text: str, i: int, end: int, kw: str) -> int | None:
    """Position of %KW at top level (not inside quotes/parentheses)."""
    target = "%" + kw.upper()
    depth = 0
    L = len(target)
    while i < end:
        ch = text[i]
        if ch == "'":
            i = _skip_squote(text, i, end)
            continue
        if ch == '"':
            i = _find_dquote_end(text, i, end)
            continue
        if ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            i = end if j < 0 else j + 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "%" and depth == 0 and text[i:i + L].upper() == target and (i + L >= end or text[i + L] not in _WORD_BOUNDARY):
            return i
        i += 1
    return None


def _find_matching_end(text: str, i: int, end: int) -> tuple[int, int] | None:
    """Find the %END; closing a %DO whose body starts at i.  Returns (body_end, after)."""
    depth = 1
    while i < end:
        ch = text[i]
        if ch == "'":
            i = _skip_squote(text, i, end)
            continue
        if ch == '"':
            i = _find_dquote_end(text, i, end)
            continue
        if ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            i = end if j < 0 else j + 2
            continue
        if ch == "%" and text.startswith("%*", i):
            i = _find_semi(text, i + 2, end)
            continue
        if ch == "%":
            word = text[i + 1:i + 4].upper()
            nxt = text[i + 3] if i + 3 < end else " "
            if word[:2] == "DO" and nxt not in _WORD_BOUNDARY:
                depth += 1
                i += 3
                continue
            nxt = text[i + 4] if i + 4 < end else " "
            if word == "END" and nxt not in _WORD_BOUNDARY:
                depth -= 1
                if depth == 0:
                    j = _skip_ws(text, i + 4, end)
                    after = j + 1 if j < end and text[j] == ";" else i + 4
                    return i, after
                i += 4
                continue
        i += 1
    return None


def _find_mend(text: str, i: int, end: int) -> tuple[int, int] | None:
    depth = 1
    while i < end:
        ch = text[i]
        if ch == "'":
            i = _skip_squote(text, i, end)
            continue
        if ch == '"':
            i = _find_dquote_end(text, i, end)
            continue
        if ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2, end)
            i = end if j < 0 else j + 2
            continue
        if ch == "%":
            if text[i:i + 6].upper() == "%MACRO" and (i + 6 >= end or text[i + 6] not in _WORD_BOUNDARY):
                depth += 1
                i += 6
                continue
            if text[i:i + 5].upper() == "%MEND" and (i + 5 >= end or text[i + 5] not in _WORD_BOUNDARY):
                depth -= 1
                if depth == 0:
                    j = _find_semi(text, i + 5, end)
                    return i, j
                i += 5
                continue
        i += 1
    return None


def _strip_unmasked(s: str) -> str:
    """Strip leading/trailing blanks that are not masked (quoted)."""
    return s.strip(" \t\r\n")


def _process_escapes(s: str) -> str:
    """%NRSTR: %' %" %( %) %% escapes -> literal characters."""
    return re.sub(r"%([()'\"%])", r"\1", s)


def _process_escapes_keep(s: str) -> str:
    # For %STR we must protect escaped chars from resolution: map them to masked forms.
    return re.sub(r"%([()'\"%])", lambda m: _MASK.get(m.group(1), m.group(1)), s)


def _unescape(s: str) -> str:
    return s
