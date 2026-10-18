"""DATA step -> PySpark.

Strategy (docs/ARCHITECTURE.md §5.5):

* compile-time PDV construction (variable order, type and length from first
  appearance, formats, labels, retention) exactly like the SAS compiler;
* Tier A: row-local statements become predicated column updates
  (``F.when(pred, new).otherwise(old)``) with IF/ELSE chains collapsed to CASE;
* Tier B: BY-group flags, LAG/DIF (incl. conditional LAG), _N_, END=,
  accumulators (sum statements reset at FIRST.x) and carried-forward retained
  values become window functions over the logical order ``_sas_seq``;
* everything else raises ``Unsupported`` with a reason so the step is routed to
  the LLM path or a manual TODO, never silently approximated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..lang import ast as A
from ..lang.lexer import OPAQUE
from ..runtime import pyfuncs
from ..semantic.schema import Schema, Var
from .exprs import CE, ExprCompiler, Unsupported, as_col, num_lit, q, split_param_filters

AUTO_VARS = {"_N_", "_ERROR_", "_IORC_", "_CMD_", "_MSG_", "_I_"}
COMPILE_ONLY = (A.LengthStmt, A.FormatStmt, A.InformatStmt, A.LabelStmt, A.AttribStmt, A.RetainStmt, A.KeepStmt,
                A.DropStmt, A.RenameStmt, A.ArrayStmt, A.ByStmt, A.MissingStmt)
MAX_UNROLL = 1000


@dataclass
class StepCode:
    lines: list[str] = field(default_factory=list)
    status: str = "converted"            # converted | unsupported
    reasons: list[str] = field(default_factory=list)
    hazards: list[tuple[str, str, int]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    tier: str = "A"


class PDV:
    def __init__(self) -> None:
        self.vars: dict[str, Var] = {}
        self.order: list[str] = []
        self.retained: set[str] = set()
        self.retain_init: dict[str, object] = {}
        self.auto: set[str] = set(AUTO_VARS)
        self.arrays: dict[str, list[str]] = {}
        self.temp_arrays: dict[str, list[object]] = {}
        self.from_input: set[str] = set()
        self.untyped: set[str] = set()      # placeholders (RETAIN/FORMAT before the defining statement)

    def get(self, name: str) -> Var | None:
        return self.vars.get(name.upper())

    def add(self, var: Var) -> Var:
        u = var.name.upper()
        if u not in self.vars:
            self.vars[u] = var
            self.order.append(u)
        return self.vars[u]

    def names(self) -> list[str]:
        return [self.vars[u].name for u in self.order]


def expand_items(items: list[A.VarItem], order: list[Var], arrays: dict[str, list[str]], line: int = 0) -> list[str]:
    names = [v.name for v in order]
    upper = [n.upper() for n in names]
    out: list[str] = []

    def add(n):
        if n.upper() not in {o.upper() for o in out}:
            out.append(n)

    for it in items:
        if isinstance(it, A.VarParam):
            raise Unsupported(f"column-list parameter &{it.name} in this position: not converted yet", line)
        if isinstance(it, A.VarName):
            add(names[upper.index(it.name.upper())] if it.name.upper() in upper else it.name)
        elif isinstance(it, A.VarPrefix):
            for n in names:
                if n.upper().startswith(it.prefix.upper()):
                    add(n)
        elif isinstance(it, A.VarSpecial):
            for v in order:
                if it.name == "_ALL_" or (it.name == "_NUMERIC_" and v.type == "num") or (it.name == "_CHARACTER_" and v.type == "char"):
                    add(v.name)
        elif isinstance(it, A.ArrayStar):
            if it.array.upper() not in arrays:
                raise Unsupported(f"unknown array {it.array}", line)
            for n in arrays[it.array.upper()]:
                add(n)
        elif isinstance(it, A.VarRange):
            if it.kind == "numbered":
                m1 = re.match(r"^(.*?)(\d+)$", it.start)
                m2 = re.match(r"^(.*?)(\d+)$", it.end)
                if not m1 or not m2 or m1.group(1).upper() != m2.group(1).upper():
                    raise Unsupported(f"invalid numbered range {it.start}-{it.end}", line)
                a, b = int(m1.group(2)), int(m2.group(2))
                width = len(m1.group(2)) if m1.group(2).startswith("0") else 0
                step = 1 if b >= a else -1
                for k in range(a, b + step, step):
                    n = f"{m1.group(1)}{str(k).zfill(width) if width else k}"
                    add(names[upper.index(n.upper())] if n.upper() in upper else n)
            else:
                if it.start.upper() not in upper or it.end.upper() not in upper:
                    raise Unsupported(f"positional range {it.start}--{it.end} needs known variables", line)
                i, j = upper.index(it.start.upper()), upper.index(it.end.upper())
                for v in order[min(i, j):max(i, j) + 1]:
                    if it.kind == "positional" or (it.kind == "positional_num" and v.type == "num") or (it.kind == "positional_char" and v.type == "char"):
                        add(v.name)
    return out


class _TypingScope:
    """Expression scope used while building the PDV (only types matter)."""

    def __init__(self, comp: "DataStepCompiler"):
        self.c = comp

    def var_ref(self, name, line):
        u = name.upper()
        if u == "_N_":
            return CE("x", "num", nonnull=True)
        if u in self.c.pdv.auto:
            return CE("x", "num", nonnull=True)
        v = self.c.pdv.get(name)
        if v is None:
            v = self.c.pdv.add(Var(name, "num", 8))
            self.c.uninitialized.add(name.upper())
        return CE("x", v.type, v.length if v.type == "char" else None)

    def var_attr(self, name):
        return self.c.pdv.get(name)

    def array_elements(self, name):
        return self.c.pdv.arrays.get(name.upper())

    def first_last(self, which, var, line):
        return CE("x", "bool")

    def lag(self, n, arg, dif, line):
        return CE("x", arg.type, arg.length)

    def runtime_macro(self, name, want, line):
        return CE("x", want)

    def param(self, name, kind, line):
        return name.upper() in self.c.state.param_specs

    def user_format(self, name):
        return self.c.state.user_formats.get(name.upper(), "fmt_x")

    def hazard(self, hid, detail, line):
        pass

    def expand_var_items(self, items, line):
        return expand_items(items, [self.c.pdv.vars[u] for u in self.c.pdv.order], self.c.pdv.arrays, line)


class _EmitScope:
    """Expression scope during code emission (columns present in the DataFrame)."""

    def __init__(self, comp: "DataStepCompiler"):
        self.c = comp

    def var_ref(self, name, line):
        c = self.c
        u = name.upper()
        if u in c.loop_consts:
            return num_lit(c.loop_consts[u])
        if u == "_N_":
            c.need_n = True
            return CE('F.col("_sas_n_")', "num", nonnull=True)
        if u == "_ERROR_":
            return num_lit(0.0)
        if u in c.in_vars or u in c.end_vars:
            return CE(f"F.col({q(c.auto_names[u])})", "num", nonnull=True)
        v = c.pdv.get(name)
        if v is None:
            raise Unsupported(f"unknown variable {name}", line)
        if v.name.upper() in c.present:
            return CE(f"F.col({q(v.name)})", v.type, v.length if v.type == "char" else None)
        return num_lit(None) if v.type == "num" else CE("F.lit('')", "char", v.length, const="")

    def var_attr(self, name):
        return self.c.pdv.get(name)

    def array_elements(self, name):
        if name.upper() in self.c.pdv.temp_arrays:
            raise Unsupported("_TEMPORARY_ array element access", 0)
        return self.c.pdv.arrays.get(name.upper())

    def first_last(self, which, var, line):
        c = self.c
        by_names = [b.upper() for b, _ in c.by]
        if var.upper() not in by_names:
            raise Unsupported(f"{which}.{var} without {var} in the BY statement", line)
        c.need_first_last = True
        return CE(f"F.col({q(f'_sas_{which.lower()}_{var.upper()}')})", "bool")

    def lag(self, n, arg, dif, line):
        return self.c.emit_lag(n, arg, dif, line)

    def runtime_macro(self, name, want, line):
        if name.startswith(OPAQUE):
            raise Unsupported(f"value of %{name[1:]}(...) is only known at run time", line)
        self.c.state.runtime_macros.add(name.upper())
        self.c.state.note_param(name, {"num": "number", "char": "text"}.get(want, want))
        self.c.hazard("H-RUNTIME-MACRO", f"&{name} resolved at run time from the context", line)
        if want == "char":
            return CE(f"F.lit(ctx.symget({q(name)}))", "char", 200)
        return CE(f"F.lit(ctx.symget_num({q(name)}))", "num")

    def param(self, name, kind, line):
        return self.c.state.note_param(name, kind)

    def user_format(self, name):
        return self.c.state.user_formats.get(name.upper())

    def hazard(self, hid, detail, line):
        self.c.hazard(hid, detail, line)

    def expand_var_items(self, items, line):
        return expand_items(items, [self.c.pdv.vars[u] for u in self.c.pdv.order], self.c.pdv.arrays, line)


class DataStepCompiler:
    def __init__(self, step: A.DataStep, state, source_of=None):
        self.step = step
        self.state = state
        self.source_of = source_of or (lambda line: "")
        self.pdv = PDV()
        self.code = StepCode()
        self.lines = self.code.lines
        self.uninitialized: set[str] = set()
        self.by: list[tuple[str, bool]] = []
        self.by_notsorted = False
        self.in_vars: dict[str, int] = {}
        self.end_vars: dict[str, str] = {}
        self.auto_names: dict[str, str] = {}
        self.present: set[str] = set()
        self.loop_consts: dict[str, float] = {}
        self.need_first_last = False
        self.need_n = False
        self.tmp = 0
        self.outputs_points: list[tuple[str | None, list[str], int]] = []
        self.assign_after_output = False
        self.keep_stmt: list[str] | None = None
        self.drop_stmt: set[str] = set()
        self.rename_stmt: dict[str, str] = {}
        self.special_retained: dict[str, dict] = {}
        self.skip_stmt_ids: set[int] = set()
        self.has_return = False
        self.input_stmt: A.Stmt | None = None
        self.attr_changed: set[str] = set()
        self.assigned_cols: set[str] = set()
        self.dynamic_input = False

    # ------------------------------------------------------------------ utilities
    def hazard(self, hid: str, detail: str, line: int) -> None:
        entry = (hid, detail, line)
        if entry not in self.code.hazards:
            self.code.hazards.append(entry)

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def new_tmp(self, prefix: str = "p") -> str:
        self.tmp += 1
        return f"_sas_{prefix}{self.state.next_id()}"

    def sas_text(self, line: int) -> str:
        t = self.source_of(line).strip()
        return t[:110] + ("…" if len(t) > 110 else "")

    def comment(self, line: int) -> str:
        t = self.sas_text(line)
        return f"  # L{line}: {t}" if t else f"  # L{line}"

    # ------------------------------------------------------------------ entry
    def compile(self) -> StepCode:
        try:
            self._find_input()
            self._build_pdv()
            self._analyze_retained()
            self._emit_input()
            self._emit_statements()
            self._emit_outputs()
        except Unsupported as e:
            self.code.status = "unsupported"
            self.code.reasons.append(f"line {e.line}: {e}" if e.line else str(e))
            self.code.lines = []
        return self.code

    # ------------------------------------------------------------------ input
    def _find_input(self) -> None:
        inputs = []

        def walk(stmts, top):
            for s in stmts:
                if isinstance(s, (A.SetStmt, A.MergeStmt, A.UpdateStmt, A.ModifyStmt, A.InputStmt, A.InfileStmt)):
                    inputs.append((s, top))
                for child in _children(s):
                    walk(child, False)

        walk(self.step.stmts, True)
        if not inputs:
            return
        reading = [s for s, _ in inputs if not isinstance(s, A.InfileStmt)]
        if len(reading) > 1:
            raise Unsupported("more than one SET/MERGE/INPUT statement (interleaved reading)", reading[1].line)
        s, top = next((x for x in inputs if not isinstance(x[0], A.InfileStmt)), inputs[0])
        if not top:
            raise Unsupported("SET/MERGE executed conditionally", s.line)
        if isinstance(s, (A.UpdateStmt, A.ModifyStmt)):
            raise Unsupported(f"{type(s).__name__.replace('Stmt', '').upper()} statement", s.line)
        if isinstance(s, (A.InputStmt, A.InfileStmt)):
            if not any(isinstance(x, A.DatalinesStmt) for x in self.step.stmts):
                raise Unsupported("INPUT from an external file (INFILE) — convert the source file to a table", s.line)
        if isinstance(s, A.SetStmt):
            for k in s.options:
                if k in ("POINT", "KEY"):
                    raise Unsupported(f"SET with {k}= (random access)", s.line)
        first_exec = next((x for x in self.step.stmts if not isinstance(x, COMPILE_ONLY)), None)
        if first_exec is not s and not isinstance(s, (A.InputStmt,)):
            # statements before SET run once per iteration before the read; supported only if compile-time
            pass
        self.input_stmt = s

    # ------------------------------------------------------------------ PDV
    def _input_schema(self, ref: A.DatasetRef) -> Schema:
        if ref.dynamic:
            raise Unsupported(f"data set name built from run-time macro variables: {ref.display}", ref.line)
        schema = self.state.catalog.get(ref.key)
        if schema is None or not schema.known:
            if not getattr(self.state, "infer_schemas", False):
                raise Unsupported(f"schema of {ref.display} is unknown (add it to the schema catalog)", ref.line)
            schema = self._infer_input(ref, schema)
        vars_ = [Var(v.name, v.type, v.length, v.format, v.informat, v.label) for v in schema.vars]
        o = ref.options
        if o.keep is not None:
            items, params = self._list_params(o.keep, ref.line)
            names = expand_items(items, vars_, {}, ref.line)
            keepset = {n.upper() for n in names}
            missing = keepset - {v.name.upper() for v in vars_}
            if missing:
                raise Unsupported(f"KEEP= variables not in {ref.display}: {sorted(missing)}", ref.line)
            if params:
                self.dynamic_input = True        # the run's column list decides which input columns exist
            else:
                vars_ = [v for v in vars_ if v.name.upper() in keepset]
        if o.drop is not None:
            items, params = self._list_params(o.drop, ref.line)
            dropset = {n.upper() for n in expand_items(items, vars_, {}, ref.line)}
            vars_ = [v for v in vars_ if v.name.upper() not in dropset]
            if params:
                self.dynamic_input = True
        for old, new in o.rename:
            for v in vars_:
                if v.name.upper() == old.upper():
                    v.name = new
        return Schema(vars_)

    def _infer_input(self, ref: A.DatasetRef, partial: Schema | None) -> Schema:
        """Schema for an input the catalog does not know: known columns + types inferred from usage."""
        from ..semantic.infer import infer_usage

        self.dynamic_input = True
        self.hazard("H-SCHEMA-INFERRED", f"{ref.display}: types inferred from usage; unseen columns pass through", ref.line)
        usage = infer_usage(self.step.stmts)
        assigned = _all_assigned(self.step.stmts)
        names: dict[str, Var] = {}
        for v in (partial.vars if partial else []):
            names[v.name.upper()] = v
        keep = ref.options.keep
        if keep is not None:
            for it in keep:
                n = getattr(it, "name", None)
                if n and n.upper() not in names:
                    names[n.upper()] = Var(n, usage.get(n.upper(), "num"), 200 if usage.get(n.upper()) == "char" else 8)
        for n, t in usage.items():
            if n in names or n in assigned or n in AUTO_VARS or n in self.pdv.arrays:
                continue
            if n.startswith("FIRST.") or n.startswith("LAST."):
                continue
            names[n] = Var(n, t, 200 if t == "char" else 8)
        for n, _ in self.by:
            if n.upper() not in names:
                names[n.upper()] = Var(n, usage.get(n.upper(), "num"), 200 if usage.get(n.upper()) == "char" else 8)
        return Schema(list(names.values()), known=False)

    def _build_pdv(self) -> None:
        ts = ExprCompiler(_TypingScope(self))

        def add_var(name, typ="num", length=8):
            v = self.pdv.get(name)
            if v is None:
                v = self.pdv.add(Var(name, typ, length if typ == "char" else 8))
            return v

        def walk(stmts):
            for s in stmts:
                if isinstance(s, (A.SetStmt, A.MergeStmt)):
                    for ref in s.datasets:
                        sch = self._input_schema(ref)
                        for v in sch.vars:
                            if self.pdv.get(v.name) is None:
                                self.pdv.add(Var(v.name, v.type, v.length, v.format, v.informat, v.label))
                            elif v.name.upper() in self.pdv.untyped:
                                existing = self.pdv.get(v.name)
                                existing.type, existing.length = v.type, v.length
                                existing.format = existing.format or v.format
                                existing.informat = existing.informat or v.informat
                                existing.label = existing.label or v.label
                                existing.name = v.name if existing.name.upper() == v.name.upper() else existing.name
                                self.pdv.untyped.discard(v.name.upper())
                            else:
                                existing = self.pdv.get(v.name)
                                if existing.type != v.type:
                                    raise Unsupported(f"variable {v.name} is character in one data set and numeric in another", s.line)
                                if v.type == "char" and v.length > existing.length:
                                    self.hazard("H-CHAR-LEN", f"{v.name}: length {existing.length} from the first data set truncates longer values", s.line)
                                if existing.format is None and v.format:
                                    existing.format = v.format
                                if existing.label is None and v.label:
                                    existing.label = v.label
                            self.pdv.from_input.add(v.name.upper())
                        if ref.options.in_var:
                            self.in_vars[ref.options.in_var.upper()] = 1
                            self.auto_names[ref.options.in_var.upper()] = ref.options.in_var
                            self.pdv.auto.add(ref.options.in_var.upper())
                    for k in ("END", "NOBS", "INDSNAME", "CUROBS"):
                        if k in s.options:
                            nm = s.options[k]
                            self.pdv.auto.add(nm.upper())
                            if k == "END":
                                self.end_vars[nm.upper()] = nm
                                self.auto_names[nm.upper()] = nm
                            elif k == "NOBS":
                                raise Unsupported("NOBS= option", s.line)
                elif isinstance(s, A.LengthStmt):
                    for items, is_char, length in s.items:
                        for n in self._expand(items, s.line, allow_new=True):
                            v = self.pdv.get(n)
                            if v is None:
                                self.pdv.add(Var(n, "char" if is_char else "num", length if is_char else 8))
                            elif v.name.upper() in self.pdv.untyped:
                                v.type, v.length = ("char", length) if is_char else ("num", 8)
                                self.pdv.untyped.discard(v.name.upper())
                            elif v.type == "char" and is_char and v.length != length:
                                if v.name.upper() in self.pdv.from_input or self._assigned_before(n):
                                    self.hazard("H-CHAR-LEN", f"LENGTH for {n} after its first use is ignored", s.line)
                                else:
                                    v.length = length
                elif isinstance(s, A.RetainStmt):
                    for items, inits in s.items:
                        names = self._expand(items, s.line, allow_new=True) if items else []
                        if s.all_ and not items:
                            names = self.pdv.names()
                        for idx, n in enumerate(names):
                            init = None
                            if len(inits) == 1:
                                init = inits[0]
                            elif idx < len(inits):
                                init = inits[idx]
                            if self.pdv.get(n) is None:
                                if init is not None and isinstance(init, A.Str):
                                    self.pdv.add(Var(n, "char", max(len(init.value), 1)))
                                else:
                                    self.pdv.add(Var(n, "num", 8))
                                    if init is None:
                                        self.pdv.untyped.add(n.upper())
                            self.pdv.retained.add(n.upper())
                            if init is not None:
                                self.pdv.retain_init[n.upper()] = init
                elif isinstance(s, A.ArrayStmt):
                    dims = s.dims
                    if any(isinstance(d, tuple) for d in dims) or len(dims) > 1:
                        raise Unsupported("multi-dimensional or explicitly bounded arrays", s.line)
                    if s.temporary:
                        n = dims[0] if dims and isinstance(dims[0], int) else len(s.init)
                        self.pdv.temp_arrays[s.name.upper()] = [None] * n
                        raise Unsupported("_TEMPORARY_ arrays", s.line)
                    if s.elements:
                        names = self._expand(s.elements, s.line, allow_new=True)
                    else:
                        if not dims or dims[0] == "*":
                            raise Unsupported("array without elements or dimension", s.line)
                        names = [f"{s.name}{k}" for k in range(1, int(dims[0]) + 1)]
                    for n in names:
                        if self.pdv.get(n) is None:
                            self.pdv.add(Var(n, "char" if s.is_char else "num", s.length or 8 if s.is_char else 8))
                    self.pdv.arrays[s.name.upper()] = [self.pdv.get(n).name for n in names]
                    if s.init:
                        raise Unsupported("array initial values", s.line)
                elif isinstance(s, (A.FormatStmt, A.InformatStmt)):
                    for items, spec in s.items:
                        for n in self._expand(items, s.line, allow_new=True):
                            existed = self.pdv.get(n) is not None
                            v = add_var(n, "char" if spec and spec.startswith("$") else "num")
                            if not existed:
                                self.pdv.untyped.add(v.name.upper())
                            self.attr_changed.add(v.name.upper())
                            if isinstance(s, A.FormatStmt):
                                v.format = spec.upper() if spec else None
                            else:
                                v.informat = spec.upper() if spec else None
                elif isinstance(s, A.LabelStmt):
                    for n, lab in s.items:
                        v = self.pdv.get(n)
                        if v is not None:
                            v.label = lab
                            self.attr_changed.add(v.name.upper())
                        else:
                            self.pending_labels = getattr(self, "pending_labels", {})
                            self.pending_labels[n.upper()] = lab
                elif isinstance(s, A.AttribStmt):
                    for items, attrs in s.items:
                        for n in self._expand(items, s.line, allow_new=True):
                            length = attrs.get("LENGTH")
                            is_char = bool(length and length.startswith("$")) or bool(attrs.get("FORMAT", "").startswith("$"))
                            ln = int(float(length.lstrip("$"))) if length else 8
                            v = self.pdv.get(n)
                            if v is None:
                                v = self.pdv.add(Var(n, "char" if is_char else "num", ln if is_char else 8))
                            if attrs.get("FORMAT"):
                                v.format = attrs["FORMAT"].upper()
                            if attrs.get("LABEL"):
                                v.label = attrs["LABEL"]
                            if attrs.get("INFORMAT"):
                                v.informat = attrs["INFORMAT"].upper()
                elif isinstance(s, A.ByStmt):
                    self.by = [(n, d) for n, d in s.vars]
                    self.by_notsorted = s.notsorted
                elif isinstance(s, A.Assign):
                    ce = ts.value(s.expr)
                    self._declare_target(s.target, ce, s.line)
                elif isinstance(s, A.SumStmt):
                    ts.num(s.expr)
                    if isinstance(s.var, A.Var):
                        v = add_var(s.var.name, "num")
                        self.pdv.retained.add(v.name.upper())
                    else:
                        raise Unsupported("sum statement into an array element", s.line)
                elif isinstance(s, A.If):
                    ts.cond(s.cond)
                    walk([s.then])
                    if s.else_ is not None:
                        walk([s.else_])
                elif isinstance(s, A.SubsetIf):
                    ts.cond(s.cond)
                elif isinstance(s, A.Where):
                    pass
                elif isinstance(s, A.DoBlock):
                    walk(s.body)
                elif isinstance(s, A.DoLoop):
                    if isinstance(s.var, A.Var):
                        add_var(s.var.name, "num")
                    for sp in s.specs:
                        ts.value(sp.start)
                        if sp.stop is not None:
                            ts.value(sp.stop)
                        if sp.by is not None:
                            ts.value(sp.by)
                    walk(s.body)
                elif isinstance(s, (A.DoWhile, A.DoUntil)):
                    ts.cond(s.cond)
                    walk(s.body)
                elif isinstance(s, A.Select):
                    if s.expr is not None:
                        ts.value(s.expr)
                    for w in s.whens:
                        for v in w.values:
                            ts.value(v)
                        walk([w.stmt])
                    if s.otherwise is not None:
                        walk([s.otherwise])
                elif isinstance(s, A.KeepStmt):
                    self.keep_stmt = (self.keep_stmt or []) + [("items", s.vars, s.line)]
                elif isinstance(s, A.DropStmt):
                    self._drop_items = getattr(self, "_drop_items", []) + [(s.vars, s.line)]
                elif isinstance(s, A.RenameStmt):
                    for o, n in s.pairs:
                        self.rename_stmt[o.upper()] = n
                elif isinstance(s, A.CallStmt):
                    for a in s.args:
                        if isinstance(a, A.OfList):
                            continue
                        try:
                            ts.value(a)
                        except Unsupported:
                            pass
                elif isinstance(s, A.LabeledStmt):
                    walk([s.stmt])
                elif isinstance(s, A.InputStmt):
                    for it in s.items:
                        if it.kind != "var":
                            continue
                        if self.pdv.get(it.var) is None:
                            is_char = it.is_char or bool(it.informat and it.informat.startswith("$"))
                            length = 8
                            if is_char and it.informat:
                                _, _, w, _ = pyfuncs.parse_format(it.informat)
                                length = w or 8
                            elif is_char and it.col_start is not None:
                                length = (it.col_end or it.col_start) - it.col_start + 1
                            self.pdv.add(Var(it.var, "char" if is_char else "num", length if is_char else 8,
                                             informat=(it.informat or "").upper() or None))
                        self.pdv.from_input.add(self.pdv.get(it.var).name.upper())

        walk(self.step.stmts)
        for n, lab in getattr(self, "pending_labels", {}).items():
            v = self.pdv.get(n)
            if v is not None:
                v.label = lab
                self.attr_changed.add(v.name.upper())
        for u in self.uninitialized:
            if u not in self.pdv.auto and u not in self.pdv.from_input and not self._ever_assigned(u):
                self.hazard("H-UNINIT", f"variable {self.pdv.vars[u].name} is never assigned (SAS: uninitialized)", self.step.start_line)

    def _expand(self, items, line, allow_new=False) -> list[str]:
        order = [self.pdv.vars[u] for u in self.pdv.order]
        return expand_items(items, order, self.pdv.arrays, line)

    def _assigned_before(self, name: str) -> bool:
        v = self.pdv.get(name)
        return v is not None and v.name.upper() in getattr(self, "_assigned", set())

    def _ever_assigned(self, upper: str) -> bool:
        found = False

        def walk(stmts):
            nonlocal found
            for s in stmts:
                if isinstance(s, A.Assign) and isinstance(s.target, A.Var) and s.target.name.upper() == upper:
                    found = True
                if isinstance(s, A.SumStmt) and isinstance(s.var, A.Var) and s.var.name.upper() == upper:
                    found = True
                if isinstance(s, A.DoLoop) and isinstance(s.var, A.Var) and s.var.name.upper() == upper:
                    found = True
                for ch in _children(s):
                    walk(ch)

        walk(self.step.stmts)
        return found

    def _declare_target(self, target: A.Expr, ce: CE, line: int) -> None:
        self._assigned = getattr(self, "_assigned", set())
        if isinstance(target, A.Var):
            v = self.pdv.get(target.name)
            if v is not None and v.name.upper() in self.pdv.untyped:
                v.type = "char" if ce.type == "char" else "num"
                v.length = (ce.length or 200) if v.type == "char" else 8
                self.pdv.untyped.discard(v.name.upper())
            if v is None:
                typ = "char" if ce.type == "char" else "num"
                length = (ce.length or 200) if typ == "char" else 8
                v = self.pdv.add(Var(target.name, typ, length))
                self.uninitialized.discard(target.name.upper())
            self._assigned.add(v.name.upper())
        elif isinstance(target, A.Index):
            if target.array.upper() not in self.pdv.arrays:
                raise Unsupported(f"assignment to unknown array {target.array}", line)
        elif isinstance(target, A.Call) and target.name == "SUBSTR":
            if not target.args or not isinstance(target.args[0], A.Var):
                raise Unsupported("SUBSTR pseudo-variable on an expression", line)
        else:
            raise Unsupported("unsupported assignment target", line)

    # ------------------------------------------------------------------ retained state (Tier B)
    def _analyze_retained(self) -> None:
        """Find retained variables whose value flows across rows and match supported patterns."""
        candidates = [u for u in self.pdv.retained if u not in self.pdv.from_input]
        for u in candidates:
            info = self._retained_usage(u)
            if not info["cross_row"]:
                continue
            pattern = self._match_accumulator(u, info) or self._match_carry_forward(u, info)
            if pattern is None:
                name = self.pdv.vars[u].name
                raise Unsupported(f"retained variable {name} carries values across rows in a pattern the rules do not cover", info["line"])
            self.special_retained[u] = pattern
            self.code.tier = "B"

    def _retained_usage(self, u: str) -> dict:
        """Collect top-level-ordered writes/reads of a retained variable."""
        writes: list[tuple[A.Stmt, A.Expr | None, list[A.Expr]]] = []   # (stmt, value expr, enclosing conds)
        reads_before_write = False
        assigned = False
        line = 0

        def refs(e: A.Expr | None) -> bool:
            return e is not None and u in _expr_vars(e)

        def walk(stmts, conds, uncond):
            nonlocal reads_before_write, assigned, line
            for s in stmts:
                if isinstance(s, A.Assign):
                    if refs(s.expr) and not (uncond and assigned):
                        reads_before_write = True
                        line = line or s.line
                    if isinstance(s.target, A.Var) and s.target.name.upper() == u:
                        writes.append((s, s.expr, list(conds)))
                        if uncond and not conds:
                            assigned = True
                elif isinstance(s, A.SumStmt):
                    if isinstance(s.var, A.Var) and s.var.name.upper() == u:
                        writes.append((s, s.expr, list(conds)))
                        reads_before_write = True
                        line = line or s.line
                    elif refs(s.expr):
                        reads_before_write = True
                elif isinstance(s, A.If):
                    if refs(s.cond) and not assigned:
                        reads_before_write = True
                        line = line or s.line
                    walk([s.then], conds + [s.cond], False)
                    if s.else_ is not None:
                        walk([s.else_], conds + [A.Unary("NOT", s.cond)], False)
                elif isinstance(s, A.SubsetIf):
                    if refs(s.cond) and not assigned:
                        reads_before_write = True
                        line = line or s.line
                elif isinstance(s, A.OutputStmt):
                    if not assigned:
                        reads_before_write = True
                        line = line or s.line
                else:
                    for ch in _children(s):
                        walk(ch, conds, False)

        walk(self.step.stmts, [], True)
        explicit_output = _has_output(self.step.stmts)
        if not explicit_output and not assigned and self._is_output_var(u):
            reads_before_write = True
        return {"writes": writes, "cross_row": reads_before_write, "line": line or self.step.start_line}

    def _is_output_var(self, u: str) -> bool:
        return u not in self.pdv.auto

    def _first_cond_var(self, cond: A.Expr) -> str | None:
        if isinstance(cond, A.FirstLast) and cond.which == "FIRST":
            return cond.var
        return None

    def _match_accumulator(self, u: str, info: dict):
        writes = info["writes"]
        sums = [w for w in writes if isinstance(w[0], A.SumStmt)]
        resets = [w for w in writes if isinstance(w[0], A.Assign)]
        if len(sums) != 1:
            # x = x + e  /  x = sum(x, e) as an accumulator
            accs = [w for w in resets if _accumulator_term(w[1], u) is not None]
            if len(accs) != 1:
                return None
            sums = accs
            resets = [w for w in resets if w is not accs[0]]
        reset_group = None
        reset_value = None
        for stmt, val, conds in resets:
            if len(conds) != 1 or self._first_cond_var(conds[0]) is None:
                return None
            if not isinstance(val, (A.Num, A.Missing)):
                return None
            g = self._first_cond_var(conds[0])
            if reset_group is not None and g.upper() != reset_group.upper():
                return None
            reset_group = g
            reset_value = None if isinstance(val, A.Missing) else val.value
            if stmt.line > sums[0][0].line:
                return None
            self.skip_stmt_ids.add(id(stmt))
        init = self.pdv.retain_init.get(u)
        if reset_group is None:
            if init is None:
                reset_value = 0.0 if isinstance(sums[0][0], A.SumStmt) else None
            elif isinstance(init, A.Num):
                reset_value = init.value
            else:
                return None
        stmt = sums[0][0]
        expr = stmt.expr if isinstance(stmt, A.SumStmt) else _accumulator_term(stmt.expr, u)
        return {"kind": "acc", "stmt": id(stmt), "expr": expr, "group": reset_group, "start": reset_value,
                "sumstmt": isinstance(stmt, A.SumStmt) or (isinstance(stmt.expr, A.Call) and stmt.expr.name == "SUM"),
                "line": stmt.line}

    def _match_carry_forward(self, u: str, info: dict):
        writes = info["writes"]
        if not writes or any(isinstance(w[0], A.SumStmt) for w in writes):
            return None
        reset_group = None
        reset_value = None
        events = []
        for stmt, val, conds in writes:
            if len(conds) == 1 and self._first_cond_var(conds[0]) is not None and isinstance(val, (A.Num, A.Missing, A.Str)):
                reset_group = self._first_cond_var(conds[0])
                reset_value = val
                self.skip_stmt_ids.add(id(stmt))
                continue
            if u in _expr_vars(val):
                return None
            events.append(stmt)
        if not events:
            return None
        init = self.pdv.retain_init.get(u)
        if reset_value is None and init is not None:
            reset_value = init
        return {"kind": "carry", "events": [id(s) for s in events], "last_event": id(events[-1]),
                "group": reset_group, "start": reset_value, "line": events[0].line, "event_preds": {}}

    # ------------------------------------------------------------------ emission: input
    def _ds_expr(self, ref: A.DatasetRef) -> str:
        self.code.reads.append(ref.key)
        code = f"ctx.get({q(ref.display.lower())})"
        o = ref.options
        sch = self.state.catalog.get(ref.key)
        vars_ = list(sch.vars) if sch else []
        if o.keep is not None:
            items, params = self._list_params(o.keep, ref.line)
            names = expand_items(items, vars_, {}, ref.line)
            if params:                   # KEEP=&vars: the run's column list (plus any listed names)
                code = f"ctx.param_keep({code}, {params[0]!r}, also={names!r})"
            else:
                code = f"P.keep({code}, {names!r})"
                vars_ = [v for v in vars_ if v.name.upper() in {n.upper() for n in names}]
        if o.drop is not None:
            items, params = self._list_params(o.drop, ref.line)
            names = expand_items(items, vars_, {}, ref.line)
            if names:
                code = f"P.drop({code}, {names!r})"
            for n in params:
                code = f"ctx.param_drop({code}, {n!r})"
        if o.rename:
            code = f"P.rename({code}, {[tuple(p) for p in o.rename]!r})"
        if o.where is not None:
            renamed = self._input_schema(ref)
            rest, filters = split_param_filters(o.where, lambda n: self.state.note_param(n, "filter", renamed))
            if rest is not None:
                scope = _ColumnScope(renamed, self)
                cond = ExprCompiler(scope, where=True).cond(rest)
                code = f"{code}.where({cond})"
            for n in filters:
                code = f"ctx.param_filter({code}, {n!r})"
        if o.firstobs is not None or o.obs is not None:
            code = f"P.obs({code}, firstobs={o.firstobs!r}, obs_={o.obs!r})"
        return code

    def _emit_input(self) -> None:
        s = self.input_stmt
        L = self.emit
        if s is None:
            L("df = spark.range(1).select(F.lit(0).cast('long').alias('_sas_seq'))  # no SET: one iteration")
        elif isinstance(s, A.SetStmt):
            if len(s.datasets) == 1 and not s.datasets[0].options.in_var:
                L(f"df = {self._ds_expr(s.datasets[0])}{self.comment(s.line)}")
            else:
                parts = [self._ds_expr(r) for r in s.datasets]
                in_vars = [r.options.in_var for r in s.datasets]
                by = self._by_arg() if self.by else None
                L(f"df = P.set_concat([{', '.join(parts)}], in_vars={in_vars!r}" + (f", by={by}" if by else "") + f"){self.comment(s.line)}")
        elif isinstance(s, A.MergeStmt):
            parts = [self._ds_expr(r) for r in s.datasets]
            in_vars = [r.options.in_var for r in s.datasets]
            if not self.by:
                self.hazard("H-ORDER", "MERGE without BY pairs rows by position", s.line)
            else:
                self.hazard("H-MERGE-M2M", "SAS match-merge semantics (not a SQL join)", s.line)
            L("df = P.merge([")
            for p in parts:
                L(f"    {p},")
            tail = f"], by={self._by_arg() if self.by else '[]'}"
            if any(in_vars):
                tail += f", in_vars={in_vars!r}"
            L(tail + ")" + self.comment(s.line))
        elif isinstance(s, (A.InputStmt, A.InfileStmt)):
            self._emit_datalines()
        else:
            raise Unsupported(f"input statement {type(s).__name__}", s.line)
        self.present = {u for u in self.pdv.from_input}
        # WHERE statements apply to the input before the PDV
        for w in [x for x in self.step.stmts if isinstance(x, A.Where)]:
            schema = Schema([self.pdv.vars[u] for u in self.pdv.order if u in self.pdv.from_input])
            rest, filters = split_param_filters(w.cond, lambda n: self.state.note_param(n, "filter", schema))
            if rest is not None:
                cond = ExprCompiler(_ColumnScope(schema, self), where=True).cond(rest)
                L(f"df = df.where({cond}){self.comment(w.line)}")
            for n in filters:          # users' Spark SQL, evaluated on the data as Databricks tables show it
                L(f"df = ctx.param_filter(df, {n!r}){self.comment(w.line)}")
        self._uses = _collect_uses(self.step.stmts)
        if self._uses["first_last"] or self._uses["n"] or self.end_vars:
            self.code.tier = "B"
        if self._uses["first_last"]:
            if not self.by:
                raise Unsupported("FIRST./LAST. without a BY statement", self.step.start_line)
            ns = ", notsorted=True" if self.by_notsorted else ""
            L(f"df = P.first_last(df, by={self._by_arg()}{ns})  # FIRST./LAST. flags")
        if self._uses["n"]:
            L("df = P.row_number(df, '_sas_n_')  # _N_")
        for u, name in self.end_vars.items():
            L(f"df = P.end_flag(df, {q(name)})  # END=")
        if _has_return(self.step.stmts):
            self.has_return = True
            L("df = df.withColumn('_sas_ret', F.lit(False))")

    def _by_arg(self) -> str:
        items = []
        for n, desc in self.by:
            v = self.pdv.get(n)
            nm = v.name if v else n
            items.append(f"({q(nm)}, True)" if desc else q(nm))
        return "[" + ", ".join(items) + "]"

    def _emit_datalines(self) -> None:
        from .datalines import compile_datalines

        rows_code = compile_datalines(self.step, self.pdv)
        for ln in rows_code:
            self.emit(ln)
        for v in self.pdv.vars.values():
            self.pdv.from_input.add(v.name.upper())

    # ------------------------------------------------------------------ emission: statements
    def _emit_statements(self) -> None:
        self.exprc = ExprCompiler(_EmitScope(self))
        self._stmts(self.step.stmts, None)

    def _stmts(self, stmts, pred):
        for s in stmts:
            self._stmt(s, pred)

    def _and(self, pred: str | None, cond: str) -> str:
        if pred is None:
            return cond
        if self.has_return:
            pass
        return f"({pred}) & ({cond})"

    def _active(self, pred: str | None) -> str | None:
        """Predicate including 'not yet returned'."""
        if self.has_return:
            return "~F.col('_sas_ret')" if pred is None else f"({pred}) & ~F.col('_sas_ret')"
        return pred

    def _freeze(self, cond: str) -> str:
        name = self.new_tmp("c")
        self.emit(f"df = df.withColumn({q(name)}, {cond})")
        return f"F.col({q(name)})"

    def _stmt(self, s: A.Stmt, pred: str | None) -> None:
        if id(s) in self.skip_stmt_ids:
            return
        saved = getattr(self, "_current_pred", None)
        self._current_pred = self._active(pred)
        try:
            self._stmt_inner(s, pred)
        finally:
            self._current_pred = saved

    def _stmt_inner(self, s: A.Stmt, pred: str | None) -> None:
        if isinstance(s, COMPILE_ONLY) or isinstance(s, (A.Where, A.SetStmt, A.MergeStmt, A.InputStmt, A.InfileStmt, A.DatalinesStmt)):
            return
        if isinstance(s, A.Assign):
            self._assign_stmt(s, pred)
        elif isinstance(s, A.SumStmt):
            self._sum_stmt(s, pred)
        elif isinstance(s, A.If):
            self._if(s, pred)
        elif isinstance(s, A.SubsetIf):
            schema = Schema([self.pdv.vars[u] for u in self.pdv.order if u in self.pdv.vars])
            rest, filters = split_param_filters(s.cond, lambda n: self.state.note_param(n, "filter", schema))
            ap = self._active(pred)
            if filters and ap is not None:
                raise Unsupported("a filter parameter in a conditional subsetting IF", s.line)
            if rest is not None:
                cond = self.exprc.cond(rest)
                self.emit(f"df = df.where({cond if ap is None else f'~({ap}) | ({cond})'}){self.comment(s.line)}")
            for n in filters:
                self.emit(f"df = ctx.param_filter(df, {n!r}){self.comment(s.line)}")
        elif isinstance(s, A.DoBlock):
            self._stmts(s.body, pred)
        elif isinstance(s, A.DoLoop):
            self._do_loop(s, pred)
        elif isinstance(s, (A.DoWhile, A.DoUntil)):
            raise Unsupported("DO WHILE/UNTIL loops", s.line)
        elif isinstance(s, A.Select):
            self._stmt(_select_to_if(s), pred)
        elif isinstance(s, A.OutputStmt):
            self._output(s, pred)
        elif isinstance(s, A.DeleteStmt):
            ap = self._active(pred)
            self.emit(f"df = df.where({'F.lit(False)' if ap is None else f'~({ap})'}){self.comment(s.line)}")
        elif isinstance(s, A.ReturnStmt):
            ap = self._active(pred)
            ret = "F.lit(True)" if ap is None else f"F.col('_sas_ret') | ({ap})"
            self.emit(f"df = df.withColumn('_sas_ret', {ret}){self.comment(s.line)}")
        elif isinstance(s, A.StopStmt):
            raise Unsupported("STOP statement", s.line)
        elif isinstance(s, A.CallStmt):
            self._call_stmt(s, pred)
        elif isinstance(s, A.PutStmt):
            self.emit(f"# PUT (SAS log output) omitted{self.comment(s.line)}")
        elif isinstance(s, (A.LeaveStmt, A.ContinueStmt)):
            raise Unsupported("LEAVE/CONTINUE", s.line)
        elif isinstance(s, A.UnknownStmt):
            if s.keyword in ("TITLE", "FOOTNOTE", "OPTIONS", "OPTION"):
                return
            raise Unsupported(f"statement {s.keyword} {s.raw[:40]}", s.line)
        elif isinstance(s, (A.GotoStmt, A.LabeledStmt)):
            raise Unsupported("GOTO/LINK/labels", s.line)
        elif isinstance(s, (A.FileStmt,)):
            raise Unsupported("FILE statement (writing external files)", s.line)
        elif isinstance(s, (A.DeclareStmt,)):
            raise Unsupported("DECLARE (hash/hiter objects)", s.line)
        elif isinstance(s, A.AbortStmt):
            raise Unsupported("ABORT statement", s.line)
        else:
            raise Unsupported(f"statement {type(s).__name__}", s.line)

    # assignment
    def _store(self, var: Var, ce: CE, line: int) -> str:
        """Code for the value as stored in ``var`` (type conversion, length truncation)."""
        if var.type == "num":
            if ce.type == "char":
                self.hazard("H-AUTOCONV", f"character value assigned to numeric {var.name}", line)
                if ce.is_const:
                    from ..runtime import pyfuncs
                    v = pyfuncs.input_value(ce.const, "BEST32.")
                    return as_col(num_lit(v))
                return f"S.char_to_num({ce.code})"
            if ce.type == "bool":
                return f"S.b2n({ce.code})"
            return as_col(ce)
        # character target
        if ce.type in ("num", "bool"):
            if ce.type == "bool":
                ce = CE(f"S.b2n({ce.code})", "num")
            self.hazard("H-AUTOCONV", f"numeric value assigned to character {var.name} (BEST12.)", line)
            ce = CE(f"S.num_to_char({as_col(ce)})", "char", 12, padded=True)
        if ce.is_const:
            text = str(ce.const)
            stored = text[:var.length].rstrip(" ")
            if len(text.rstrip(" ")) > var.length:
                self.hazard("H-CHAR-LEN", f"{text.rstrip()!r} truncated to {stored!r} (length of {var.name} is {var.length})", line)
            return f"F.lit({q(stored)})"
        if ce.length is not None and ce.length <= var.length:
            if not ce.padded and ce.code.startswith("F.col("):
                return ce.code
            return f"S.store({ce.code})"
        if ce.length is None or ce.length > var.length:
            self.hazard("H-CHAR-LEN", f"value may exceed length {var.length} of {var.name}; truncated like SAS", line)
        return f"S.store({ce.code}, {var.length})"

    def _old(self, var: Var) -> str:
        if var.name.upper() in self.present:
            return f"F.col({q(var.name)})"
        return "S.MISSING" if var.type == "num" else "F.lit('')"

    def _set(self, var: Var, value_code: str, pred: str | None, line: int, note: str | None = None) -> None:
        ap = self._active(pred)
        if ap is None:
            code = value_code
        else:
            code = f"F.when({ap}, {value_code}).otherwise({self._old(var)})"
            if var.name.upper() not in self.present and var.type == "num":
                code = f"F.when({ap}, {value_code})"
        self.emit(f"df = df.withColumn({q(var.name)}, {code}){self.comment(line) if note is None else note}")
        self.present.add(var.name.upper())
        self.assigned_cols.add(var.name.upper())
        if self.outputs_points:
            self.assign_after_output = True

    def _assign_stmt(self, s: A.Assign, pred):
        t = s.target
        if isinstance(t, A.Var):
            u = t.name.upper()
            if u in self.special_retained:
                self._special_assign(u, s, pred)
                return
            var = self.pdv.get(t.name)
            ce = self.exprc.value(s.expr)
            self._set(var, self._store(var, ce, s.line), pred, s.line)
        elif isinstance(t, A.Index):
            elems = self.pdv.arrays.get(t.array.upper())
            idx = self.exprc.num(t.indices[0])
            if idx.is_const and idx.const is not None:
                k = int(idx.const)
                if not 1 <= k <= len(elems):
                    raise Unsupported(f"array subscript {k} out of range", s.line)
                var = self.pdv.get(elems[k - 1])
                ce = self.exprc.value(s.expr)
                self._set(var, self._store(var, ce, s.line), pred, s.line)
            else:
                ce = self.exprc.value(s.expr)
                idx_code = self._freeze(idx.code)
                for k, n in enumerate(elems, start=1):
                    var = self.pdv.get(n)
                    p = self._and(pred, f"({idx_code} == {float(k)})")
                    self._set(var, self._store(var, ce, s.line), p, s.line)
        elif isinstance(t, A.Call) and t.name == "SUBSTR":
            var = self.pdv.get(t.args[0].name)
            if var is None:
                raise Unsupported(f"SUBSTR pseudo-variable on {t.args[0].name}, which is not defined yet", s.line)
            if var.type != "char":
                raise Unsupported("SUBSTR pseudo-variable on a numeric variable", s.line)
            pos = as_col(self.exprc.num(t.args[1]))
            ce = self.exprc.char(s.expr)
            n = as_col(self.exprc.num(t.args[2])) if len(t.args) > 2 else f"(F.lit({float(var.length)}) - {pos} + 1)"
            code = (f"S.store(F.overlay(S.pad({self._old(var)}, {var.length}), F.rpad({as_col(ce)}, ({n}).cast('int'), ' '), "
                    f"({pos}).cast('int'), ({n}).cast('int')), {var.length})")
            self._set(var, code, pred, s.line)
        else:
            raise Unsupported("assignment target", s.line)

    def _sum_stmt(self, s: A.SumStmt, pred):
        u = s.var.name.upper()
        if u in self.special_retained:
            self._special_assign(u, s, pred)
            return
        raise Unsupported(f"sum statement into {s.var.name} not recognised as an accumulator", s.line)

    def _window_for_group(self, group: str | None) -> str:
        if group is None:
            return "W.orderBy('_sas_seq').rowsBetween(W.unboundedPreceding, W.currentRow)"
        names = []
        for n, _ in self.by:
            v = self.pdv.get(n)
            names.append(v.name if v else n)
            if n.upper() == group.upper():
                break
        return f"W.partitionBy({', '.join(q(n) for n in names)}).orderBy('_sas_seq').rowsBetween(W.unboundedPreceding, W.currentRow)"

    def _special_assign(self, u: str, s, pred):
        info = self.special_retained[u]
        var = self.pdv.vars[u]
        ap = self._active(pred)
        if info["kind"] == "acc":
            if id(s) != info["stmt"]:
                raise Unsupported(f"unexpected write to accumulator {var.name}", s.line)
            if info["group"] is None:
                self.hazard("H-ORDER", f"running total {var.name} over the whole data set (single ordered window)", s.line)
            e = self.exprc.num(info["expr"])
            contrib = as_col(e) if info["sumstmt"] else as_col(e)
            if info["sumstmt"]:
                contrib = f"F.coalesce({contrib}, F.lit(0.0))"
            if ap is not None:
                contrib = f"F.when({ap}, {contrib}).otherwise(F.lit(0.0))"
            w = self._window_for_group(info["group"])
            start = info["start"]
            if info["sumstmt"]:
                base = "0.0" if start is None else repr(float(start))
                code = f"F.lit({base}) + F.sum({contrib}).over({w})"
            else:
                miss = f"F.when({ap}, {as_col(e)}.isNull()).otherwise(F.lit(False))" if ap else f"{as_col(e)}.isNull()"
                base = "S.MISSING" if start is None else f"F.lit({float(start)!r})"
                code = (f"F.when(F.max(F.when({miss}, 1).otherwise(0)).over({w}) == 1, S.MISSING)"
                        f".otherwise({base} + F.sum(F.coalesce({contrib}, F.lit(0.0))).over({w}))")
            self.emit(f"df = df.withColumn({q(var.name)}, {code}){self.comment(s.line)}")
            self.present.add(u)
            self.hazard("H-ORDER", f"retained accumulator {var.name} computed with an ordered window", s.line)
            return
        # carry forward
        events = info["events"]
        info["event_preds"][id(s)] = (ap, self._store(var, self.exprc.value(s.expr), s.line))
        if id(s) != info["last_event"]:
            return
        parts = []
        for sid in reversed(events):
            p, v = info["event_preds"][sid]
            parts.append(f"F.when({p if p is not None else 'F.lit(True)'}, F.struct({v}.alias('v')))")
        ev = parts[0] if len(parts) == 1 else f"F.coalesce({', '.join(parts)})"
        w = self._window_for_group(info["group"])
        start = info["start"]
        if start is None:
            start_code = "S.MISSING" if var.type == "num" else "F.lit('')"
        else:
            start_code = self._store(var, self.exprc.value(start), s.line)
        tmp = self.new_tmp("cf")
        self.emit(f"df = df.withColumn({q(tmp)}, F.last({ev}, ignorenulls=True).over({w}))  # retained {var.name}")
        self.emit(f"df = df.withColumn({q(var.name)}, F.when(F.col({q(tmp)}).isNull(), {start_code}).otherwise(F.col({q(tmp)}).getField('v'))){self.comment(s.line)}")
        self.present.add(u)
        self.hazard("H-ORDER", f"retained value {var.name} carried forward with an ordered window", s.line)

    # IF
    def _case_chain(self, s: A.If):
        """IF c1 THEN x=a; ELSE IF c2 THEN x=b; ... [ELSE x=z;] -> list of (cond, expr), else_expr."""
        target = None
        chain = []
        cur = s
        else_expr = None
        while True:
            then = cur.then
            if isinstance(then, A.DoBlock) and len(then.body) == 1:
                then = then.body[0]
            if not isinstance(then, A.Assign) or not isinstance(then.target, A.Var) or id(then) in self.skip_stmt_ids:
                return None
            if target is None:
                target = then.target.name.upper()
            elif then.target.name.upper() != target:
                return None
            chain.append((cur.cond, then))
            e = cur.else_
            if e is None:
                break
            if isinstance(e, A.DoBlock) and len(e.body) == 1:
                e = e.body[0]
            if isinstance(e, A.If):
                cur = e
                continue
            if isinstance(e, A.Assign) and isinstance(e.target, A.Var) and e.target.name.upper() == target and id(e) not in self.skip_stmt_ids:
                else_expr = e
                break
            return None
        if target in self.special_retained:
            return None
        if len(chain) == 1 and else_expr is None:
            return None
        return target, chain, else_expr

    def _if(self, s: A.If, pred):
        cc = self._case_chain(s)
        if cc is not None:
            target, chain, else_stmt = cc
            var = self.pdv.vars[target]
            parts = []
            for cond, st in chain:
                c = self.exprc.cond(cond)
                parts.append((c, self._store(var, self.exprc.value(st.expr), st.line)))
            code = "F.when(" + ").when(".join(f"{c}, {v}" for c, v in parts) + ")"
            if else_stmt is not None:
                code += f".otherwise({self._store(var, self.exprc.value(else_stmt.expr), else_stmt.line)})"
            elif var.name.upper() in self.present or var.type == "char":
                code += f".otherwise({self._old(var)})"
            ap = self._active(pred)
            if ap is not None:
                code = f"F.when({ap}, {code}).otherwise({self._old(var)})"
            self.emit(f"df = df.withColumn({q(var.name)}, {code}){self.comment(s.line)}")
            self.present.add(target)
            self.assigned_cols.add(target)
            if self.outputs_points:
                self.assign_after_output = True
            return
        cond = self.exprc.cond(s.cond)
        then_simple = _is_simple(s.then)
        else_simple = s.else_ is None or _is_simple(s.else_)
        cond_vars = _expr_vars(s.cond)
        modifies = _assigned_vars(s.then) & cond_vars
        must_freeze = not (then_simple and else_simple) or (s.else_ is not None and bool(modifies)) or _contains_lag(s.cond)
        if must_freeze:
            self.emit(f"# IF {self.sas_text(s.line)[:100]}")
            c = self._freeze(cond)
        else:
            c = cond
        self._stmt(s.then, self._and(pred, c))
        if s.else_ is not None:
            self._stmt(s.else_, self._and(pred, f"~({c})"))

    # DO loops
    def _do_loop(self, s: A.DoLoop, pred):
        if s.while_ is not None or s.until is not None:
            raise Unsupported("DO loop with WHILE/UNTIL", s.line)
        if not isinstance(s.var, A.Var):
            raise Unsupported("DO loop index that is not a simple variable", s.line)
        values: list[float] = []
        for sp in s.specs:
            start = self.exprc.num(sp.start)
            if sp.stop is None:
                if not start.is_const:
                    raise Unsupported("DO list with a non-constant value", s.line)
                values.append(start.const)
                continue
            stop = self.exprc.num(sp.stop)
            by = self.exprc.num(sp.by) if sp.by is not None else num_lit(1.0)
            if not (start.is_const and stop.is_const and by.is_const) or None in (start.const, stop.const, by.const):
                raise Unsupported("DO loop with data-dependent bounds", s.line)
            if by.const == 0:
                raise Unsupported("DO loop with BY 0", s.line)
            v = start.const
            while (by.const > 0 and v <= stop.const + 1e-12) or (by.const < 0 and v >= stop.const - 1e-12):
                values.append(v)
                v = v + by.const
                if len(values) > MAX_UNROLL:
                    raise Unsupported(f"DO loop with more than {MAX_UNROLL} iterations", s.line)
            final = v
        if not s.specs[-1].stop:
            final = values[-1] if values else None
        u = s.var.name.upper()
        saved = self.loop_consts.get(u)
        self.emit(f"# DO {s.var.name} unrolled ({len(values)} iterations){self.comment(s.line)}")
        for v in values:
            self.loop_consts[u] = v
            self._stmts(s.body, pred)
        if saved is None:
            self.loop_consts.pop(u, None)
        else:
            self.loop_consts[u] = saved
        var = self.pdv.get(s.var.name)
        if final is not None:
            self._set(var, as_col(num_lit(final)), pred, s.line, note=f"  # {var.name} after the loop")

    # OUTPUT
    def _output(self, s: A.OutputStmt, pred):
        targets = [n.upper() for n in s.datasets] or [o.display.upper() for o in self.step.outputs]
        valid = {o.display.upper(): o for o in self.step.outputs}
        for t in targets:
            if t not in valid and t.split(".")[-1] not in {k.split(".")[-1] for k in valid}:
                raise Unsupported(f"OUTPUT to {t} which is not listed on the DATA statement", s.line)
        ap = self._active(pred)
        k = len(self.outputs_points)
        cols = self._output_union()
        fields = []
        for v in cols:
            ref = _EmitScope(self).var_ref(v.name, s.line)
            fields.append(f"{as_col(ref)}.alias({q(v.name)})")
        name = f"_sas_o{k}"
        struct = f"F.struct({', '.join(fields)}, F.col('_sas_seq').alias('_sas_seq'))"
        self.emit(f"df = df.withColumn({q(name)}, {struct if ap is None else f'F.when({ap}, {struct})'}){self.comment(s.line)}")
        self.outputs_points.append((ap, targets, k))
        self.assign_after_output = False

    def _output_union(self) -> list[Var]:
        return [self.pdv.vars[u] for u in self.pdv.order if u not in self.pdv.auto]

    # CALL routines
    def _call_stmt(self, s: A.CallStmt, pred):
        n = s.name.upper()
        if n == "MISSING":
            names = []
            for a in s.args:
                if isinstance(a, A.OfList):
                    names.extend(self._expand(a.items, s.line))
                elif isinstance(a, A.Var):
                    names.append(a.name)
                else:
                    raise Unsupported("CALL MISSING argument", s.line)
            for nm in names:
                var = self.pdv.get(nm)
                self._set(var, "S.MISSING" if var.type == "num" else "F.lit('')", pred, s.line)
            return
        if n in ("SYMPUT", "SYMPUTX"):
            self._symput(s, pred)
            return
        if n in ("STREAMINIT", "RANUNI", "RANNOR", "RANBIN"):
            self.hazard("H-RANDOM", "random numbers cannot reproduce SAS row for row", s.line)
        raise Unsupported(f"CALL {n}", s.line)

    def _symput(self, s: A.CallStmt, pred):
        if len(s.args) < 2:
            raise Unsupported("CALL SYMPUT with fewer than 2 arguments", s.line)
        name = self.exprc.char(s.args[0])
        val = self.exprc.value(s.args[1])
        x = s.name.upper() == "SYMPUTX"
        if val.type == "num":
            fmt = f"S.put_best({as_col(val)}, 32)" if x else f"S.put_best({as_col(val)}, 12)"
            val_code = fmt
        else:
            val_code = as_col(val)
        if x:
            val_code = f"F.trim({val_code})"
        ap = self._active(pred)
        rows = f"df.where({ap})" if ap is not None else "df"
        if name.is_const:
            self.emit(f"_rows = {rows}.orderBy(F.desc('_sas_seq')).select({val_code}.alias('v')).limit(1).collect()"
                      f"{self.comment(s.line)}")
            self.emit(f"if _rows: ctx.symput({q(str(name.const).strip())}, _rows[0]['v'])")
        else:
            self.emit(f"for _r in {rows}.select(F.trim({as_col(name)}).alias('n'), {val_code}.alias('v'), '_sas_seq').orderBy('_sas_seq').collect():"
                      f"{self.comment(s.line)}")
            self.emit("    ctx.symput(_r['n'], _r['v'])")
        self.hazard("H-RUNTIME-MACRO", "CALL SYMPUT values are collected to the driver", s.line)

    # LAG
    def emit_lag(self, n: int, arg: CE, dif: bool, line: int) -> CE:
        """LAGn/DIFn.  SAS keeps one queue per LAG call that only advances when the call
        executes, so a conditional LAG looks back among the rows where it ran."""
        pred = getattr(self, "_current_pred", None)
        self.code.tier = "B"
        if self.by:
            self.hazard("H-LAG-BY", "LAG ignores BY groups (global order), as in SAS", line)
        tmp_arg = self.new_tmp("la")
        self.emit(f"df = df.withColumn({q(tmp_arg)}, {as_col(arg)})")
        res = self.new_tmp("lag")
        if pred is None:
            self.emit(f"df = df.withColumn({q(res)}, F.lag(F.col({q(tmp_arg)}), {n}).over(W.orderBy('_sas_seq'))){self.comment(line)}")
        else:
            self.hazard("H-LAG-COND", "conditional LAG: queue advances only where the condition holds", line)
            pcol = self._freeze(pred)
            self.emit(f"df = df.withColumn({q(res)}, F.when({pcol}, F.lag(F.col({q(tmp_arg)}), {n}).over("
                      f"W.partitionBy({pcol}).orderBy('_sas_seq')))){self.comment(line)}")
        code = f"F.col({q(res)})"
        if dif:
            return CE(f"(F.col({q(tmp_arg)}) - {code})", "num")
        return CE(code, arg.type, arg.length)

    # ------------------------------------------------------------------ outputs
    def _emit_outputs(self) -> None:
        L = self.emit
        outs = [o for o in self.step.outputs if not o.is_null]
        if not outs:
            if self.step.outputs and self.step.outputs[0].is_null:
                L("# DATA _NULL_: no data set is written")
            return
        explicit = bool(self.outputs_points)
        if explicit:
            single = len(self.outputs_points) == 1 and not self.assign_after_output
            if single:
                ap, targets, k = self.outputs_points[0]
                if ap is not None:
                    L(f"df = df.where(F.col('_sas_o{k}').isNotNull())")
                L(f"df = df.drop('_sas_o{k}')")
                srcs = {o.display.upper(): "df" for o in outs if self._targets_match(o, targets)}
            else:
                srcs = {}
                for o in outs:
                    ks = [k for _, t, k in self.outputs_points if self._targets_match(o, t)]
                    if not ks:
                        continue
                    var = f"df_{_pyname(o.name)}"
                    arr = ", ".join(f"F.col('_sas_o{k}')" for k in ks)
                    L(f"{var} = df.select(F.posexplode(F.array({arr})).alias('_sas_opos', '_sas_orow')).where(F.col('_sas_orow').isNotNull())")
                    L(f"{var} = P.reseq({var}.select('_sas_orow.*', '_sas_opos').withColumnRenamed('_sas_seq', '_sas_iseq'), F.col('_sas_iseq'), F.col('_sas_opos')).drop('_sas_iseq', '_sas_opos')")
                    srcs[o.display.upper()] = var
        else:
            srcs = {o.display.upper(): "df" for o in outs}
        for o in outs:
            src = srcs.get(o.display.upper())
            if src is None:
                L(f"df_{_pyname(o.name)} = df.limit(0)  # no OUTPUT statement targets {o.display}")
                src = f"df_{_pyname(o.name)}"
            self._write_output(o, src, from_snapshot=explicit and src != "df")

    def _targets_match(self, o: A.DatasetRef, targets: list[str]) -> bool:
        names = {t.upper() for t in targets}
        return o.display.upper() in names or o.name.upper() in {t.split(".")[-1] for t in names}

    def _output_vars(self, o: A.DatasetRef) -> tuple[list[Var], list[str]]:
        """The output's variables known at conversion, and the run-time column-list steps to apply to ``out``."""
        vars_ = self._output_union()
        ops: list[str] = []
        if self.keep_stmt:
            keep: list[str] = []
            params: list[str] = []
            for _, items, line in self.keep_stmt:
                static, ps = self._list_params(items, line)
                keep += expand_items(static, vars_, self.pdv.arrays, line)
                params += ps
            if params:                  # KEEP &vars: every variable is a candidate, the run's list decides
                ops.append(self._keep_op(params, keep, o.line))
            else:
                keepset = {k.upper() for k in keep}
                vars_ = [v for v in vars_ if v.name.upper() in keepset]
        for items, line in getattr(self, "_drop_items", []):
            static, ps = self._list_params(items, line)
            dropset = {d.upper() for d in expand_items(static, vars_, self.pdv.arrays, line)}
            vars_ = [v for v in vars_ if v.name.upper() not in dropset]
            ops += [f"out = ctx.param_drop(out, {n!r})" for n in ps]
        if o.options.keep is not None:
            static, ps = self._list_params(o.options.keep, o.line)
            keep = expand_items(static, vars_, self.pdv.arrays, o.line)
            if ps:
                ops.append(self._keep_op(ps, keep, o.line))
            else:
                keepset = {k.upper() for k in keep}
                vars_ = [v for v in vars_ if v.name.upper() in keepset]
        if o.options.drop is not None:
            static, ps = self._list_params(o.options.drop, o.line)
            dropset = {d.upper() for d in expand_items(static, vars_, self.pdv.arrays, o.line)}
            vars_ = [v for v in vars_ if v.name.upper() not in dropset]
            ops += [f"out = ctx.param_drop(out, {n!r})" for n in ps]
        return vars_, ops

    def _list_params(self, items: list, line: int) -> tuple[list, list[str]]:
        """A variable list's static items and its column-list parameters (&vars from a parameter module)."""
        params = [it.name for it in items if isinstance(it, A.VarParam)]
        for n in params:
            if not self.state.note_param(n, "columns"):
                raise Unsupported(f"variable list built from run-time macro variable &{n}", line)
        return [it for it in items if not isinstance(it, A.VarParam)], params

    @staticmethod
    def _keep_op(params: list[str], also: list[str], line: int) -> str:
        if len(params) > 1:
            raise Unsupported("more than one column-list parameter in one KEEP list", line)
        return f"out = ctx.param_keep(out, {params[0]!r}, also={also!r})"

    def _write_output(self, o: A.DatasetRef, src: str, from_snapshot: bool) -> None:
        L = self.emit
        vars_, ops = self._output_vars(o)
        renames = dict(self.rename_stmt)
        for old, new in o.options.rename:
            renames[old.upper()] = new
        if ops and renames:
            raise Unsupported("RENAME together with a column-list parameter", o.line)
        sel = []
        out_vars: list[Var] = []
        for v in vars_:
            new_name = renames.get(v.name.upper(), v.name)
            if from_snapshot or v.name.upper() in self.present or v.name.upper() in self.special_retained:
                c = f"F.col({q(v.name)})"
            else:
                c = "S.MISSING" if v.type == "num" else "F.lit('')"
            sel.append(f"{c}.alias({q(new_name)})" if (new_name != v.name or not c.startswith("F.col")) else c)
            out_vars.append(Var(new_name, v.type, v.length, v.format, v.informat, v.label))
        target = o.display.lower()
        if self.dynamic_input and not self.keep_stmt and o.options.keep is None:
            # IN= / END= variables are temporary in SAS: never written, also when unseen columns pass through
            temporary = [self.auto_names.get(u, u) for u in (*self.in_vars, *self.end_vars)]
            dropped = sorted(({v.name for v in self._output_union()} - {v.name for v in vars_}) | set(temporary))
            types = {v.name: v.type for v in vars_}
            ren = {v.name: renames[v.name.upper()] for v in vars_ if v.name.upper() in renames}
            L(f"out = P.select_pdv({src}, {types!r}, drop={dropped!r}" + (f", rename={ren!r}" if ren else "") + ")  # unseen input columns pass through")
        else:
            L(f"out = {src}.select(")
            for chunk in _chunks(sel, 4):
                L("    " + ", ".join(chunk) + ",")
            L("    '_sas_seq')")
        for op in ops:
            L(op)
        if o.options.where is not None:
            rest, filters = split_param_filters(o.options.where, lambda n: self.state.note_param(n, "filter", Schema(out_vars)))
            if rest is not None:
                scope = _ColumnScope(Schema(out_vars), self)
                cond = ExprCompiler(scope, where=True).cond(rest)
                L(f"out = out.where({cond})")
            for n in filters:
                L(f"out = ctx.param_filter(out, {n!r})")
        renamed_from = {renames.get(v.name.upper(), v.name).upper(): v.name.upper() for v in vars_}
        meta = {v.name: v.meta_dict() for v in out_vars
                if not self.dynamic_input or v.name.upper() not in self.pdv.from_input or v.name.upper() in self.assigned_cols
                if from_snapshot or renamed_from.get(v.name.upper(), v.name.upper()) not in self.pdv.from_input
                or renamed_from.get(v.name.upper(), v.name.upper()) in self.attr_changed
                or renamed_from.get(v.name.upper(), v.name.upper()) in self.assigned_cols}
        meta_arg = f", meta={_fmt_meta(meta)}" if meta else ""
        L(f"ctx.put({q(target)}, out{meta_arg}" + (f", label={o.options.label!r}" if o.options.label else "") + ")")
        self.code.writes.append(o.key)
        self.state.catalog.put(o.key, Schema(out_vars, o.options.label, known=not self.dynamic_input and not ops))


class _ColumnScope:
    """Scope over a known schema (WHERE= / WHERE statements / output WHERE=)."""

    def __init__(self, schema: Schema, comp: DataStepCompiler | None):
        self.schema = schema
        self.c = comp

    def var_ref(self, name, line):
        v = self.schema.get(name)
        if v is None:
            raise Unsupported(f"WHERE references {name}, which is not in the input", line)
        return CE(f"F.col({q(v.name)})", v.type, v.length if v.type == "char" else None)

    def var_attr(self, name):
        return self.schema.get(name)

    def array_elements(self, name):
        return None

    def first_last(self, which, var, line):
        raise Unsupported("FIRST./LAST. in WHERE", line)

    def lag(self, n, arg, dif, line):
        raise Unsupported("LAG in WHERE", line)

    def runtime_macro(self, name, want, line):
        if self.c is not None:
            return _EmitScope(self.c).runtime_macro(name, want, line)
        return CE(f"F.lit(ctx.symget_num({q(name)}))", "num")

    def param(self, name, kind, line):
        return self.c is not None and self.c.state.note_param(name, kind)

    def user_format(self, name):
        return self.c.state.user_formats.get(name.upper()) if self.c else None

    def hazard(self, hid, detail, line):
        if self.c is not None:
            self.c.hazard(hid, detail, line)

    def expand_var_items(self, items, line):
        return expand_items(items, self.schema.vars, {}, line)


# ---------------------------------------------------------------------------
# helpers
def _children(s: A.Stmt) -> list[list[A.Stmt]]:
    if isinstance(s, A.If):
        return [[s.then]] + ([[s.else_]] if s.else_ is not None else [])
    if isinstance(s, (A.DoBlock, A.DoLoop, A.DoWhile, A.DoUntil)):
        return [s.body]
    if isinstance(s, A.Select):
        return [[w.stmt] for w in s.whens] + ([[s.otherwise]] if s.otherwise is not None else [])
    if isinstance(s, A.LabeledStmt):
        return [[s.stmt]]
    return []


def _has_output(stmts) -> bool:
    for s in stmts:
        if isinstance(s, A.OutputStmt):
            return True
        if any(_has_output(c) for c in _children(s)):
            return True
    return False


def _has_return(stmts) -> bool:
    for s in stmts:
        if isinstance(s, A.ReturnStmt):
            return True
        if any(_has_return(c) for c in _children(s)):
            return True
    return False


def _contains_lag(e: A.Expr) -> bool:
    found = False

    def visit(x):
        nonlocal found
        if isinstance(x, A.Call) and (x.name.startswith("LAG") or x.name.startswith("DIF")):
            found = True
        for ch in _expr_children(x):
            visit(ch)

    visit(e)
    return found


def _expr_children(e):
    if isinstance(e, A.Unary):
        return [e.operand]
    if isinstance(e, (A.Binary, A.Compare)):
        return [e.left, e.right]
    if isinstance(e, A.Call):
        return [a for a in e.args if isinstance(a, A.Expr)]
    if isinstance(e, A.Index):
        return list(e.indices)
    if isinstance(e, A.InList):
        return [e.operand] + [i for i in e.items if isinstance(i, A.Expr)]
    if isinstance(e, A.Between):
        return [e.operand, e.lo, e.hi]
    if isinstance(e, (A.IsMissing,)):
        return [e.operand]
    if isinstance(e, (A.Like,)):
        return [e.operand, e.pattern]
    if isinstance(e, (A.Contains,)):
        return [e.operand, e.sub]
    if isinstance(e, A.SoundsLike):
        return [e.left, e.right]
    return []


def _expr_vars(e: A.Expr | None) -> set[str]:
    out: set[str] = set()

    def visit(x):
        if isinstance(x, A.Var):
            out.add(x.name.upper())
        elif isinstance(x, A.FirstLast):
            out.add(x.var.upper())
        elif isinstance(x, A.OfList):
            for it in x.items:
                if isinstance(it, A.VarName):
                    out.add(it.name.upper())
        for ch in _expr_children(x):
            visit(ch)

    if e is not None:
        visit(e)
    return out


def _assigned_vars(s: A.Stmt) -> set[str]:
    out: set[str] = set()

    def walk(st):
        if isinstance(st, A.Assign) and isinstance(st.target, A.Var):
            out.add(st.target.name.upper())
        if isinstance(st, A.SumStmt) and isinstance(st.var, A.Var):
            out.add(st.var.name.upper())
        for ch in _children(st):
            for x in ch:
                walk(x)

    walk(s)
    return out


def _is_simple(s: A.Stmt) -> bool:
    if isinstance(s, A.DoBlock):
        return len(s.body) == 1 and _is_simple(s.body[0])
    return isinstance(s, (A.Assign, A.DeleteStmt, A.OutputStmt, A.SubsetIf))


def _collect_uses(stmts) -> dict:
    uses = {"first_last": False, "n": False, "lag": False}

    def visit_e(e):
        if isinstance(e, A.FirstLast):
            uses["first_last"] = True
        if isinstance(e, A.Var) and e.name.upper() == "_N_":
            uses["n"] = True
        if isinstance(e, A.Call) and (e.name.startswith("LAG") or e.name.startswith("DIF")):
            uses["lag"] = True
        for ch in _expr_children(e):
            visit_e(ch)

    def walk(ss):
        for s in ss:
            for attr in ("cond", "expr", "target"):
                e = getattr(s, attr, None)
                if isinstance(e, A.Expr):
                    visit_e(e)
            if isinstance(s, A.DoLoop):
                for sp in s.specs:
                    for e in (sp.start, sp.stop, sp.by):
                        if e is not None:
                            visit_e(e)
            if isinstance(s, A.Select):
                if s.expr is not None:
                    visit_e(s.expr)
                for w in s.whens:
                    for v in w.values:
                        visit_e(v)
            if isinstance(s, A.CallStmt):
                for a in s.args:
                    if isinstance(a, A.Expr):
                        visit_e(a)
            for ch in _children(s):
                walk(ch)

    walk(stmts)
    return uses


def _select_to_if(s: A.Select) -> A.Stmt:
    tail: A.Stmt | None = s.otherwise
    for w in reversed(s.whens):
        if s.expr is not None:
            conds = [A.Compare("EQ", s.expr, v, line=s.line) for v in w.values]
        else:
            conds = list(w.values)
        cond = conds[0]
        for c in conds[1:]:
            cond = A.Binary("OR", cond, c, line=s.line)
        tail = A.If(cond, w.stmt, tail, line=s.line)
    return tail if tail is not None else A.DoBlock([], line=s.line)


def _accumulator_term(e: A.Expr | None, u: str) -> A.Expr | None:
    """x = x + e  -> e ;  x = sum(x, e) -> e."""
    if isinstance(e, A.Binary) and e.op == "+":
        if isinstance(e.left, A.Var) and e.left.name.upper() == u and u not in _expr_vars(e.right):
            return e.right
        if isinstance(e.right, A.Var) and e.right.name.upper() == u and u not in _expr_vars(e.left):
            return e.left
    if isinstance(e, A.Call) and e.name == "SUM" and len(e.args) == 2:
        a, b = e.args
        if isinstance(a, A.Var) and a.name.upper() == u and u not in _expr_vars(b):
            return b
        if isinstance(b, A.Var) and b.name.upper() == u and u not in _expr_vars(a):
            return a
    return None


def _all_assigned(stmts) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        out |= _assigned_vars(s)
    return out


def _pyname(name: str) -> str:
    return re.sub(r"\W", "_", name.lower())


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _fmt_meta(meta: dict) -> str:
    items = [f"{k!r}: {v!r}" for k, v in meta.items()]
    if len(items) <= 3:
        return "{" + ", ".join(items) + "}"
    return "{\n        " + ",\n        ".join(items) + ",\n    }"
