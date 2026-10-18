"""PROC steps -> PySpark (runtime-backed)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..lang import ast as A
from ..lang.lexer import Kind
from ..lang.parser import ParseError, Parser
from ..runtime import pyfuncs
from ..semantic.schema import Schema, Var
from .datastep import StepCode
from .dsopts import SchemaScope, input_expr, output_apply, put_line
from .exprs import ExprCompiler, Unsupported, q, split_param_filters

DATASET_OPTIONS = {"DATA", "OUT", "DUPOUT", "OUTPUT", "BASE", "NEW", "OUTEST", "OUTSTAT", "OUT2", "OUTP"}
DISPLAY_ONLY = {"REPORT", "TABULATE", "GPLOT", "SGPLOT", "SGPANEL", "GCHART", "PLOT", "CHART",
                "PRINTTO", "TEMPLATE", "DOCUMENT", "OPTIONS", "SGRENDER", "GSLIDE", "CATALOG", "PWENCODE"}
# PROC statements whose operands are variable lists (a run-time value there is a column-list parameter)
_LIST_STATEMENTS = {"BY", "CLASS", "VAR", "TABLES", "TABLE", "ID", "KEEP", "DROP", "WEIGHT", "FREQ", "COPY"}


@dataclass
class ProcOptions:
    values: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)

    def ds(self, key: str) -> A.DatasetRef | None:
        v = self.values.get(key)
        return v if isinstance(v, A.DatasetRef) else None


def parse_options(text: str) -> ProcOptions:
    p = Parser(text)
    out = ProcOptions()
    while not p.at_eof():
        t = p.peek()
        if t.kind is not Kind.NAME:
            p.advance()
            continue
        p.advance()
        key = t.upper
        if p.at_op("="):
            p.advance()
            if key in DATASET_OPTIONS:
                out.values[key] = p.parse_dataset_ref()
            elif p.at_op("("):
                depth = 0
                toks = []
                while not p.at_eof():
                    x = p.advance()
                    toks.append(x)
                    if x.is_op("("):
                        depth += 1
                    elif x.is_op(")"):
                        depth -= 1
                        if depth == 0:
                            break
                out.values[key] = p.text_of(toks)
            else:
                v = p.advance()
                spec = None
                if v.kind in (Kind.NAME, Kind.NUMBER) and p.at_op(".") and not p.peek().ws_before:
                    p.advance()
                    spec = v.value + "."
                out.values[key] = spec or v.value
        else:
            out.flags.add(key)
    return out


def parse_by(stmt: A.ProcStmt) -> list[tuple[str, bool]]:
    vars_, desc = [], False
    for tok in stmt.tokens[1:]:
        u = tok.value.upper()
        if u == "DESCENDING":
            desc = True
            continue
        if u in ("NOTSORTED", "GROUPFORMAT"):
            continue
        vars_.append((tok.value, desc))
        desc = False
    return vars_


def where_lines(stmt: A.ProcStmt, schema: Schema, state, hazards, var: str = "df") -> list[str]:
    """A PROC WHERE statement: SAS conditions compiled; filter parameters (users' Spark SQL) via ctx.param_filter."""
    try:
        p = Parser(stmt.raw)
        e = p.parse_expr(where=True)
    except ParseError as ex:
        raise Unsupported(f"WHERE: {ex}", stmt.line) from ex
    rest, filters = split_param_filters(e, lambda n: state.note_param(n, "filter", schema))
    lines = []
    if rest is not None:
        lines.append(f"{var} = {var}.where({ExprCompiler(SchemaScope(schema, state, hazards), where=True).cond(rest)})")
    return lines + [f"{var} = ctx.param_filter({var}, {n!r})" for n in filters]


def canonical(schema: Schema, name: str) -> str:
    v = schema.get(name)
    if v is None:
        if not schema.known:
            return name.upper()
        raise Unsupported(f"variable {name} not found")
    return v.name


class ProcCompiler:
    def __init__(self, step: A.ProcStep, state):
        self.step = step
        self.state = state
        self.code = StepCode()
        self.lines = self.code.lines
        try:
            self.opts = parse_options(step.options_raw)
        except ParseError:
            self.opts = ProcOptions()

    def emit(self, s: str) -> None:
        self.lines.append(s)

    def emit_where(self, stmt: A.ProcStmt, schema: Schema) -> None:
        for ln in where_lines(stmt, schema, self.state, self.code.hazards):
            self.emit(ln)

    def _guard_runtime_tokens(self) -> None:
        """A run-time macro value where a handler reads names would be read as a column named after it, or dropped
        (class &vars -> no CLASS at all): only WHERE and PROC SORT's BY take one; anything else is not converted."""
        allowed = {"WHERE"} | ({"BY"} if self.step.name == "SORT" else set())
        for st in self.step.stmts:
            if st.keyword in allowed:
                continue
            for t in st.tokens:
                if t.kind is Kind.RTMACRO:
                    if st.keyword in _LIST_STATEMENTS:
                        self.state.note_param(t.value, "columns")
                    raise Unsupported(f"run-time macro value &{t.value} in PROC {self.step.name} {st.keyword}: "
                                      "not converted yet", st.line)

    def data_ref(self) -> A.DatasetRef:
        ref = self.opts.ds("DATA")
        if ref is not None:
            return ref
        last = self.state.last_dataset
        if last is None:
            raise Unsupported(f"PROC {self.step.name} without DATA= and no previous data set", self.step.start_line)
        lib, ds = last.split(".", 1)
        return A.DatasetRef(None if lib == "WORK" else lib, ds, line=self.step.start_line)

    def stmts(self, kw: str) -> list[A.ProcStmt]:
        return [s for s in self.step.stmts if s.keyword == kw]

    def compile(self) -> StepCode:
        name = self.step.name
        handler = getattr(self, "_proc_" + name.lower(), None)
        try:
            if handler is None:
                if name in DISPLAY_ONLY:
                    self.emit(f"# PROC {name}: report/display output only — no data set produced (review the report separately)")
                    self.code.status = "display"
                    return self.code
                raise Unsupported(f"PROC {name} is not supported by the rules", self.step.start_line)
            if name != "SQL":
                self._guard_runtime_tokens()
            handler()
        except ParseError as e:
            self.code.status = "unsupported"
            self.code.reasons.append(f"PROC {name}: {e}")
            self.code.lines.clear()
            return self.code
        except Unsupported as e:
            self.code.status = "unsupported"
            self.code.reasons.append(f"line {e.line}: {e}" if e.line else str(e))
            self.code.lines.clear()
        return self.code

    # ------------------------------------------------------------------ SORT
    def _proc_sort(self) -> None:
        data = self.data_ref()
        code, schema = input_expr(data, self.state, self.code.hazards)
        self.code.reads.append(data.key)
        by_stmts = self.stmts("BY")
        if not by_stmts:
            raise Unsupported("PROC SORT without BY", self.step.start_line)
        by_params = [t.value.upper() for t in by_stmts[0].tokens if t.kind is Kind.RTMACRO]
        for n in by_params:            # BY &vars: a column-list parameter, sorted by the run's list
            if not self.state.note_param(n, "columns"):
                raise Unsupported(f"BY list built from run-time macro variable &{n}", by_stmts[0].line)
        static_by = A.ProcStmt("BY", "", [t for t in by_stmts[0].tokens if t.kind is not Kind.RTMACRO], by_stmts[0].line)
        by = [(canonical(schema, n), d) for n, d in parse_by(static_by)]
        self.emit(f"df = {code}")
        for w in self.stmts("WHERE"):
            self.emit_where(w, schema)
        flags = self.opts.flags
        nodupkey = "NODUPKEY" in flags
        nodup = "NODUP" in flags or "NODUPRECS" in flags
        dupout = self.opts.ds("DUPOUT")
        by_code = "[" + ", ".join([f"({q(n)}, True)" if d else q(n) for n, d in by] + [f"*ctx.param_columns({n!r})" for n in by_params]) + "]"
        extra = ""
        if nodupkey:
            extra += ", nodupkey=True"
            self.code.hazards.append(("H-NODUPKEY", "first row of each key kept in input order", self.step.start_line))
        if nodup:
            extra += ", nodup=True"
        if dupout is not None:
            self.emit(f"df, dups = P.sort(df, by={by_code}{extra}, dupout=True)")
        else:
            self.emit(f"df = P.sort(df, by={by_code}{extra})")
        out = self.opts.ds("OUT") or A.DatasetRef(data.libref, data.name, line=data.line)
        lines, out_schema = output_apply(out, "df", schema, self.state, self.code.hazards)
        for ln in lines:
            self.emit(ln)
        self.emit(put_line(out, "df", out_schema))
        self.state.catalog.put(out.key, out_schema)
        self.code.writes.append(out.key)
        if dupout is not None:
            self.emit(put_line(dupout, "dups", schema))
            self.state.catalog.put(dupout.key, schema)
            self.code.writes.append(dupout.key)

    # ------------------------------------------------------------------ SQL
    def _proc_sql(self) -> None:
        from .sql import compile_proc_sql

        compile_proc_sql(self.step, self.state, self.code)
        if not self.code.writes and not any("ctx.symput(" in ln for ln in self.lines):
            self.code.status = "display" if any("spark.sql" in ln for ln in self.lines) else self.code.status

    # ------------------------------------------------------------------ DATASETS / DELETE / APPEND
    def _proc_datasets(self) -> None:
        lib = (self.opts.values.get("LIBRARY") or self.opts.values.get("LIB") or "WORK")
        lib = lib.upper() if isinstance(lib, str) else "WORK"
        for s in self.step.stmts:
            if s.keyword == "DELETE":
                for tok in s.tokens[1:]:
                    if tok.kind is Kind.NAME:
                        key = f"{lib}.{tok.value.upper()}"
                        self.emit(f"ctx.delete({q(key.lower())})")
                        self.state.catalog.drop(key)
            elif s.keyword == "CHANGE":
                toks = [t for t in s.tokens[1:] if t.kind is Kind.NAME]
                for i in range(0, len(toks) - 1, 2):
                    old, new = toks[i].value, toks[i + 1].value
                    self.emit(f"ctx.rename({q(f'{lib}.{old}'.lower())}, {q(new.lower())})")
                    sch = self.state.catalog.get(f"{lib}.{old.upper()}")
                    if sch is not None:
                        self.state.catalog.put(f"{lib}.{new.upper()}", sch)
                        self.state.catalog.drop(f"{lib}.{old.upper()}")
            elif s.keyword in ("KILL",) or "KILL" in self.opts.flags:
                self.emit(f"for _n in ctx.names({q(lib)}): ctx.delete({q(lib)} + '.' + _n)")
            elif s.keyword in ("MODIFY", "FORMAT", "LABEL", "RENAME", "ATTRIB", "INDEX", "CONTENTS", "SAVE", "AUDIT"):
                raise Unsupported(f"PROC DATASETS {s.keyword}", s.line)
        if "KILL" in self.opts.flags:
            self.emit(f"for _n in ctx.names({q(lib)}): ctx.delete({q(lib)} + '.' + _n)")
        if not self.lines:
            self.emit("# PROC DATASETS: nothing to do")

    def _proc_delete(self) -> None:
        ref = self.opts.ds("DATA")
        if ref is None:
            raise Unsupported("PROC DELETE without DATA=", self.step.start_line)
        self.emit(f"ctx.delete({q(ref.display.lower())})")
        self.state.catalog.drop(ref.key)

    def _proc_append(self) -> None:
        base = self.opts.ds("BASE") or self.opts.ds("OUT")
        new = self.opts.ds("DATA") or self.opts.ds("NEW")
        if base is None or new is None:
            raise Unsupported("PROC APPEND needs BASE= and DATA=", self.step.start_line)
        new_code, new_schema = input_expr(new, self.state, self.code.hazards)
        base_schema = self.state.catalog.get(base.key)
        if base_schema is None:
            self.emit(f"df = {new_code}")
            self.emit(put_line(base, "df", new_schema))
            self.state.catalog.put(base.key, new_schema)
        else:
            force = "FORCE" in self.opts.flags
            extra = [v.name for v in new_schema.vars if base_schema.get(v.name) is None]
            if extra and not force:
                self.code.hazards.append(("H-APPEND", f"variables {extra} not in BASE= are dropped (SAS requires FORCE)", self.step.start_line))
            cols = [v.name for v in base_schema.vars]
            self.emit(f"base = ctx.get({q(base.display.lower())}) if ctx.exists({q(base.display.lower())}) else None")
            self.emit(f"new = {new_code}")
            self.emit(f"df = new if base is None else P.set_concat([base, P.align(new, {cols!r})])")
            self.emit(put_line(base, "df", base_schema))
        self.code.reads.append(new.key)
        self.code.writes.append(base.key)


# ---------------------------------------------------------------------------
# statistical / reporting PROCs (methods attached to ProcCompiler below)
_FREQ_FLAGS = {"MISSING", "OUTPCT", "NOPRINT", "LIST", "SPARSE", "CHISQ", "EXACT", "FISHER", "NOCOL", "NOROW",
               "NOPERCENT", "NOCUM", "NOFREQ", "MISSPRINT", "NOWARN", "TOTPCT", "CUMCOL", "DEVIATION", "EXPECTED"}
_MEANS_STATS = {"N", "NMISS", "MEAN", "STD", "STDDEV", "VAR", "MIN", "MAX", "RANGE", "SUM", "USS", "CSS", "CV", "STDERR",
                "MEDIAN", "P1", "P5", "P10", "Q1", "P25", "P50", "Q3", "P75", "P90", "P95", "P99", "QRANGE", "SUMWGT",
                "LCLM", "UCLM", "SKEWNESS", "KURTOSIS"}


def _ods_outputs(step: A.ProcStep, state) -> dict[str, A.DatasetRef]:
    """ODS OUTPUT table=dataset(options) inside the step, plus a pending global ODS OUTPUT."""
    out: dict[str, A.DatasetRef] = dict(getattr(state, "pending_ods", {}) or {})
    state.pending_ods = {}
    for st in step.stmts:
        if st.keyword != "ODS":
            continue
        p = Parser(st.raw)
        if not p.at_name("OUTPUT"):
            continue
        p.advance()
        while not p.at_eof():
            name = p.advance().value
            if not p.at_op("="):
                break
            p.advance()
            out[name.upper()] = p.parse_dataset_ref()
    return out


def _parse_tables(stmt: A.ProcStmt) -> tuple[list[list[str]], set[str], dict]:
    """TABLES a b*c (x y)*z / options  ->  ([['A'], ['B','C'], ['X','Z'], ['Y','Z']], flags, {'OUT': ref})."""
    p = Parser(stmt.raw)
    terms: list[list[list[str]]] = []
    cur: list[list[str]] = []
    while not p.at_eof() and not p.at_op("/"):
        if p.at_op("*"):
            p.advance()
            continue
        if p.at_op("("):
            p.advance()
            group = []
            while not p.at_op(")"):
                group.append(p.advance().value)
            p.advance()
            factor = group
        else:
            factor = [p.advance().value]
        cur.append(factor)
        if not p.at_op("*"):
            terms.append(cur)
            cur = []
    flags: set[str] = set()
    opts: dict = {}
    if p.at_op("/"):
        p.advance()
        while not p.at_eof():
            t = p.advance()
            key = t.upper
            if p.at_op("="):
                p.advance()
                if key in ("OUT", "OUTEXPECT"):
                    opts[key] = p.parse_dataset_ref()
                else:
                    opts[key] = p.advance().value
            else:
                flags.add(key)
    requests = []
    for term in terms:
        for combo in itertools.product(*term):
            requests.append(list(combo))
    return requests, flags, opts


def _parse_output_stmt(stmt: A.ProcStmt, var: list[str]):
    """OUTPUT OUT=ds  MEAN= / MEAN(x)=m / MEAN=m1 m2  [/ AUTONAME]  -> (ref, {(stat,var): name}, stats, autoname)."""
    p = Parser(stmt.raw)
    ref = None
    names: dict[tuple[str, str], str] = {}
    stats: list[str] = []
    autoname = False
    pending: list[tuple[str, list[str]]] = []
    while not p.at_eof():
        if p.at_op("/"):
            p.advance()
            while not p.at_eof():
                if p.advance().upper == "AUTONAME":
                    autoname = True
            break
        t = p.advance()
        key = t.upper
        if key == "OUT" and p.at_op("="):
            p.advance()
            ref = p.parse_dataset_ref()
            continue
        if key in _MEANS_STATS:
            targets = list(var)
            if p.at_op("("):
                p.advance()
                targets = []
                while not p.at_op(")"):
                    targets.append(p.advance().value)
                p.advance()
            p.expect_op("=")
            given = []
            while p.peek().kind is Kind.NAME and not p.at_op("=", k=1) and not p.at_op("(", k=1) and p.peek().upper not in _MEANS_STATS | {"OUT"}:
                given.append(p.advance().value)
            stats.append(key)
            pending.append((key, targets, given))
            continue
        raise Unsupported(f"OUTPUT statement option {t.value}", stmt.line)
    for key, targets, given in pending:
        for i, v in enumerate(targets):
            if autoname and not given:
                names[(key, v.upper())] = f"{v}_{key.capitalize() if key not in ('N', 'NMISS', 'CV', 'CSS', 'USS') else key}"
            elif i < len(given):
                names[(key, v.upper())] = given[i]
            elif not given:
                names[(key, v.upper())] = v
    return ref, names, stats, autoname


def _proc_freq(self) -> None:
    data = self.data_ref()
    code, schema = input_expr(data, self.state, self.code.hazards)
    self.code.reads.append(data.key)
    self.emit(f"df = {code}")
    for w in self.stmts("WHERE"):
        self.emit_where(w, schema)
    by = [canonical(schema, n) for n, _ in parse_by(self.stmts("BY")[0])] if self.stmts("BY") else []
    weight = canonical(schema, self.stmts("WEIGHT")[0].tokens[1].value) if self.stmts("WEIGHT") else None
    formats = self._format_overrides(schema)
    ods = _ods_outputs(self.step, self.state)
    exact_fisher = any("FISHER" in s.raw.upper() for s in self.stmts("EXACT"))
    tables = self.stmts("TABLES") + self.stmts("TABLE")
    if not tables:
        tables_req = [[v.name] for v in schema.vars]
        tables_flags, tables_opts = set(), {}
        all_requests = [(r, tables_flags, tables_opts) for r in tables_req]
    else:
        all_requests = []
        for t in tables:
            reqs, flags, opts = _parse_tables(t)
            all_requests += [(r, flags, opts) for r in reqs]
    order = (self.opts.values.get("ORDER") or "internal").lower()
    results_by_table: dict[str, list[str]] = {}
    fmt_arg = ""
    if formats:
        fmt_arg = ", formats={" + ", ".join(f"{q(k)}: {v}" for k, v in formats.items()) + "}"
        self.code.hazards.append(("H-FMT-GROUP", "PROC FREQ groups on formatted values", self.step.start_line))
    for idx, (req, flags, opts) in enumerate(all_requests):
        cols = [canonical(schema, c) for c in req]
        if len(cols) > 2:
            raise Unsupported("PROC FREQ n-way tables (n > 2)", self.step.start_line)
        var = f"res{idx}"
        extra = []
        if by:
            extra.append(f"by={by!r}")
        if weight:
            extra.append(f"weight={weight!r}")
        if "MISSING" in flags:
            extra.append("missing=True")
        if "OUTPCT" in flags:
            extra.append("outpct=True")
        if order != "internal":
            extra.append(f"order={order!r}")
        self.emit(f"{var} = P.freq(df, {cols!r}{', ' + ', '.join(extra) if extra else ''}{fmt_arg})")
        if "OUT" in opts:
            ref = opts["OUT"]
            out_schema = Schema([schema.get(b) for b in by] + [schema.get(c) for c in cols]
                                + [Var("COUNT", "num", 8, label="Frequency Count"), Var("PERCENT", "num", 8, label="Percent of Total Frequency")]
                                + ([Var("PCT_ROW", "num", 8), Var("PCT_COL", "num", 8)] if "OUTPCT" in flags and len(cols) == 2 else []))
            lines, out_schema = output_apply(ref, f"{var}['OUT']", out_schema, self.state, self.code.hazards)
            self.emit(f"_out = {var}['OUT']")
            for ln in lines:
                self.emit(ln.replace(f"{var}['OUT']", "_out"))
            self.emit(put_line(ref, "_out", out_schema, with_meta=True))
            self.state.catalog.put(ref.key, out_schema)
            self.code.writes.append(ref.key)
        names = ["OneWayFreqs"] if len(cols) == 1 else ["CrossTabFreqs"]
        if len(cols) == 2 and ({"CHISQ", "EXACT", "FISHER"} & flags or exact_fisher):
            names += ["FishersExact", "ChiSq"]
        for n in names:
            results_by_table.setdefault(n.upper(), []).append(f"{var}[{q(n)}]")
    for table, ref in ods.items():
        srcs = results_by_table.get(table)
        if not srcs:
            raise Unsupported(f"ODS OUTPUT {table} from PROC FREQ", self.step.start_line)
        self.emit(f"_ods = P.set_concat([{', '.join(srcs)}])" if len(srcs) > 1 else f"_ods = {srcs[0]}")
        ods_schema = _ods_schema(table, schema, by, all_requests)
        lines, out_schema = output_apply(ref, "_ods", ods_schema, self.state, self.code.hazards)
        for ln in lines:
            self.emit(ln)
        self.emit(put_line(ref, "_ods", out_schema, with_meta=True))
        self.state.catalog.put(ref.key, out_schema)
        self.code.writes.append(ref.key)
    if not self.code.writes:
        self.emit("for _r in [" + ", ".join(f"res{i}" for i in range(len(all_requests))) + "]: P.display(_r.get('OneWayFreqs') or _r.get('CrossTabFreqs'))")
        self.code.status = "display"


def _ods_schema(table: str, schema: Schema, by: list[str], requests) -> Schema:
    bvars = [schema.get(b) for b in by]
    if table == "FISHERSEXACT":
        return Schema(bvars + [Var("Table", "char", 64), Var("Name1", "char", 8), Var("Label1", "char", 40),
                               Var("cValue1", "char", 16), Var("nValue1", "num", 8)])
    if table == "CHISQ":
        return Schema(bvars + [Var("Table", "char", 64), Var("Statistic", "char", 40), Var("DF", "num", 8),
                               Var("Value", "num", 8), Var("Prob", "num", 8)])
    if table == "ONEWAYFREQS":
        v = schema.get(requests[0][0][0])
        return Schema(bvars + [Var("Table", "char", 64), v, Var(f"F_{v.name}", "char", 200), Var("Frequency", "num", 8),
                               Var("Percent", "num", 8), Var("CumFrequency", "num", 8), Var("CumPercent", "num", 8)])
    if table == "CROSSTABFREQS":
        r, c = (schema.get(x) for x in requests[0][0][:2])
        return Schema(bvars + [Var("Table", "char", 64), r, c, Var("Frequency", "num", 8), Var("Percent", "num", 8)])
    raise Unsupported(f"ODS table {table}")


def _format_overrides(self, schema: Schema) -> dict[str, str]:
    """FORMAT statements inside the PROC -> grouping expressions (formatted values)."""
    out: dict[str, str] = {}
    for st in self.stmts("FORMAT"):
        p = Parser("format " + st.raw + ";")
        stmt = p.parse_ds_statement()
        for items, spec in stmt.items:
            for it in items:
                name = getattr(it, "name", None)
                if not name or spec is None:
                    continue
                v = schema.get(name)
                if v is None:
                    continue
                dollar, fname, _, _ = pyfuncs.parse_format(spec)
                user = self.state.user_formats.get(("$" if dollar else "") + fname)
                if user:
                    out[v.name] = f"F.call_function({q(user)}, F.col({q(v.name)}))"
                else:
                    out[v.name] = f"S.put(F.col({q(v.name)}), {q(spec.lower())}{', ' + q('char') if v.type == 'char' else ''})"
    return out


def _proc_means(self) -> None:
    data = self.data_ref()
    code, schema = input_expr(data, self.state, self.code.hazards)
    self.code.reads.append(data.key)
    self.emit(f"df = {code}")
    for w in self.stmts("WHERE"):
        self.emit_where(w, schema)
    by = [canonical(schema, n) for n, _ in parse_by(self.stmts("BY")[0])] if self.stmts("BY") else []
    class_ = []
    for st in self.stmts("CLASS") + self.stmts("CLASSES"):
        class_ += [canonical(schema, t.value) for t in st.tokens[1:] if t.kind is Kind.NAME and t.upper not in ("MISSING", "ORDER", "DESCENDING")]
    var = []
    for st in self.stmts("VAR"):
        var += [canonical(schema, t.value) for t in st.tokens[1:] if t.kind is Kind.NAME]
    if not var:
        var = [v.name for v in schema.vars if v.type == "num" and v.name not in by + class_]
    flags = self.opts.flags
    outs = self.stmts("OUTPUT")
    if self.stmts("WEIGHT") or self.stmts("FREQ") or self.stmts("ID"):
        raise Unsupported("PROC MEANS WEIGHT/FREQ/ID statements", self.step.start_line)
    if self.stmts("TYPES") or self.stmts("WAYS"):
        raise Unsupported("PROC MEANS TYPES/WAYS statements", self.step.start_line)
    vardef = (self.opts.values.get("VARDEF") or "DF").upper()
    qntldef = self.opts.values.get("QNTLDEF") or self.opts.values.get("PCTLDEF")
    if qntldef and str(qntldef) != "5":
        raise Unsupported(f"QNTLDEF={qntldef} (only the default 5 is implemented)", self.step.start_line)
    if any(s in flags for s in ("MEDIAN", "P1", "P5", "P10", "Q1", "P25", "Q3", "P75", "P90", "P95", "P99", "QRANGE")) or outs:
        self.code.hazards.append(("H-PCTL", "exact SAS QNTLDEF=5 percentiles", self.step.start_line))
    formats = self._format_overrides(schema)
    if not outs:
        stats = [s for s in flags if s in _MEANS_STATS] or ["N", "MEAN", "STD", "MIN", "MAX"]
        names = {(s, v.upper()): f"{v}_{s}" for s in stats for v in var}
        self.emit(f"res = P.means(df, var={var!r}, stats={stats!r}, class_={class_!r}, by={by!r}, nway=True, "
                  f"missing={'MISSING' in flags}, vardef={vardef!r}, names={names!r})")
        self.emit("P.display(res)")
        self.code.status = "display"
        return
    for st in outs:
        ref, names, stats, _ = _parse_output_stmt(st, var)
        if ref is None:
            raise Unsupported("OUTPUT without OUT=", st.line)
        if not stats:
            raise Unsupported("OUTPUT OUT= without statistic keywords (default _STAT_ layout)", st.line)
        fmt_arg = ""
        if formats:
            self.code.hazards.append(("H-FMT-GROUP", "class variables grouped on formatted values", st.line))
            fmt_arg = ", formats={" + ", ".join(f"{q(k)}: {v}" for k, v in formats.items() if k in class_) + "}"
        names_canon = {(s, canonical(schema, v)): n for (s, v), n in names.items()}
        self.emit(f"res = P.means(df, var={var!r}, stats={sorted(set(stats))!r}, class_={class_!r}, by={by!r}, "
                  f"nway={'NWAY' in flags}, missing={'MISSING' in flags}, vardef={vardef!r}, names={names_canon!r}{fmt_arg})")
        out_vars = [schema.get(b) for b in by] + [schema.get(c) for c in class_] + [Var("_TYPE_", "num", 8), Var("_FREQ_", "num", 8)]
        seen = set()
        for (s, v), n in names_canon.items():
            if n.upper() in seen:
                continue
            seen.add(n.upper())
            src = schema.get(v)
            out_vars.append(Var(n, "num", 8, src.format if src and s in ("MEAN", "MIN", "MAX", "MEDIAN", "SUM") else None, None,
                                src.label if src else None))
        lines, out_schema = output_apply(ref, "res", Schema(out_vars), self.state, self.code.hazards)
        for ln in lines:
            self.emit(ln)
        self.emit(put_line(ref, "res", out_schema, with_meta=True))
        self.state.catalog.put(ref.key, out_schema)
        self.code.writes.append(ref.key)


def _proc_transpose(self) -> None:
    data = self.data_ref()
    code, schema = input_expr(data, self.state, self.code.hazards)
    self.code.reads.append(data.key)
    self.emit(f"df = {code}")
    for w in self.stmts("WHERE"):
        self.emit_where(w, schema)
    by = [canonical(schema, n) for n, _ in parse_by(self.stmts("BY")[0])] if self.stmts("BY") else []
    id_ = canonical(schema, self.stmts("ID")[0].tokens[1].value) if self.stmts("ID") else None
    var = []
    for st in self.stmts("VAR"):
        var += [canonical(schema, t.value) for t in st.tokens[1:] if t.kind is Kind.NAME]
    if not var:
        var = [v.name for v in schema.vars if v.type == "num" and v.name not in by and v.name != id_]
    if self.stmts("COPY") or self.stmts("IDLABEL"):
        raise Unsupported("PROC TRANSPOSE COPY/IDLABEL", self.step.start_line)
    prefix = self.opts.values.get("PREFIX", "COL")
    name = self.opts.values.get("NAME", "_NAME_")
    label_col = self.opts.values.get("LABEL", "_LABEL_")
    labels = {v.name: v.label for v in (schema.get(x) for x in var) if v and v.label}
    self.emit(f"res = P.transpose(df, var={var!r}, by={by!r}, id_={id_!r}, prefix={prefix!r}, name={name!r}, "
              f"label={label_col!r}, labels={labels!r})")
    out = self.opts.ds("OUT")
    if out is None:
        raise Unsupported("PROC TRANSPOSE without OUT= (DATAn naming)", self.step.start_line)
    out_schema = Schema([schema.get(b) for b in by] + [Var(name, "char", 32)] + ([Var(label_col, "char", 256)] if labels else []), known=False)
    self.emit(put_line(out, "res", out_schema))
    self.code.hazards.append(("H-ORDER", "transposed column count depends on the data (schema known only at run time)", self.step.start_line))
    self.state.catalog.put(out.key, out_schema)
    self.code.writes.append(out.key)


def _proc_format(self) -> None:
    for st in self.step.stmts:
        if st.keyword != "VALUE":
            if st.keyword in ("INVALUE", "PICTURE"):
                raise Unsupported(f"PROC FORMAT {st.keyword}", st.line)
            continue
        name, kind, ranges, other = _parse_value_stmt(st)
        fn = "fmt_" + name.lower().lstrip("$") + ("_c" if kind == "char" else "")
        self.state.user_formats[name.upper()] = fn
        self.emit(f"P.define_format(spark, {q(fn)}, {q(kind)}, {ranges!r}, other={other!r})  # VALUE {name}")
    if "CNTLOUT" in self.opts.values or "CNTLIN" in self.opts.values:
        raise Unsupported("PROC FORMAT CNTLIN=/CNTLOUT=", self.step.start_line)
    if not self.lines:
        self.emit("# PROC FORMAT: no VALUE statements")


def _parse_value_stmt(st: A.ProcStmt):
    p = Parser(st.raw)
    is_char = False
    if p.at_op("$"):
        p.advance()
        is_char = True
    name = p.advance().value
    if p.at_op("("):
        while not p.at_op(")"):
            p.advance()
        p.advance()
    kind = "char" if is_char else "num"
    ranges: list = []
    other = None

    def bound():
        t = p.advance()
        if t.kind is Kind.NAME and t.upper in ("LOW", "LO"):
            return None
        if t.kind is Kind.NAME and t.upper in ("HIGH", "HI"):
            return None
        if t.kind is Kind.STRING:
            return t.value
        if t.is_op("-") and p.peek().kind is Kind.NUMBER:
            return -float(p.advance().value)
        if t.kind is Kind.NUMBER:
            return float(t.value)
        raise Unsupported(f"PROC FORMAT range value {t.value}", st.line)

    while not p.at_eof():
        specs = []
        while True:
            t = p.peek()
            if t.kind is Kind.NAME and t.upper == "OTHER":
                p.advance()
                specs.append(("other",))
            elif t.kind is Kind.MISSING or (t.kind is Kind.STRING and t.value.strip() == "" and is_char):
                p.advance()
                specs.append(("miss",))
            else:
                lo = bound()
                lo_ex = False
                if p.at_op("<") and p.at_op("-", k=1):
                    p.advance()
                    lo_ex = True
                if p.at_op("-"):
                    p.advance()
                    hi_ex = False
                    if p.at_op("<"):
                        p.advance()
                        hi_ex = True
                    hi = bound()
                    specs.append(("range", lo, hi, lo_ex, hi_ex))
                else:
                    specs.append(("eq", lo))
            if p.at_op(","):
                p.advance()
                continue
            break
        p.expect_op("=")
        lab = p.advance()
        label = lab.value
        for sp in specs:
            if sp[0] == "other":
                other = label
            elif sp[0] == "miss":
                ranges.append(("miss", label))
            elif sp[0] == "eq":
                ranges.append(("=", sp[1], label))
            else:
                ranges.append((sp[1], sp[2], sp[3], sp[4], label))
    return name, kind, ranges, other


def _proc_print(self) -> None:
    data = self.data_ref()
    code, schema = input_expr(data, self.state, self.code.hazards)
    self.emit(f"df = {code}")
    for w in self.stmts("WHERE"):
        self.emit_where(w, schema)
    var = []
    for st in self.stmts("VAR"):
        var += [canonical(schema, t.value) for t in st.tokens[1:] if t.kind is Kind.NAME]
    if var:
        self.emit(f"df = df.select(*{var!r}, '_sas_seq')")
    self.emit("P.display(df.orderBy('_sas_seq'))")
    self.code.status = "display"


def _proc_contents(self) -> None:
    data = self.data_ref()
    code, _ = input_expr(data, self.state, self.code.hazards)
    self.emit(f"P.display_schema({code})")
    self.code.status = "display"


ProcCompiler._proc_freq = _proc_freq
ProcCompiler._proc_means = _proc_means
ProcCompiler._proc_summary = _proc_means
ProcCompiler._proc_transpose = _proc_transpose
ProcCompiler._proc_format = _proc_format
ProcCompiler._proc_print = _proc_print
ProcCompiler._proc_contents = _proc_contents
ProcCompiler._format_overrides = _format_overrides
