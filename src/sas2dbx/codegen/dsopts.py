"""Data set references with options (KEEP=/DROP=/RENAME=/WHERE=/OBS=) for inputs and outputs."""
from __future__ import annotations

from ..lang import ast as A
from ..semantic.schema import Schema, Var
from .exprs import CE, ExprCompiler, Unsupported, q


class SchemaScope:
    """Expression scope over a known schema (WHERE clauses of PROCs and data set options)."""

    def __init__(self, schema: Schema, state, hazards: list | None = None):
        self.schema = schema
        self.state = state
        self.hazards = hazards if hazards is not None else []

    def var_ref(self, name, line):
        v = self.schema.get(name)
        if v is None:
            raise Unsupported(f"variable {name} is not in the data set", line)
        return CE(f"F.col({q(v.name)})", v.type, v.length if v.type == "char" else None)

    def var_attr(self, name):
        return self.schema.get(name)

    def array_elements(self, name):
        return None

    def first_last(self, which, var, line):
        raise Unsupported("FIRST./LAST. outside a DATA step", line)

    def lag(self, n, arg, dif, line):
        raise Unsupported("LAG outside a DATA step", line)

    def runtime_macro(self, name, want, line):
        self.state.runtime_macros.add(name.upper())
        self.state.note_param(name, {"num": "number", "char": "text"}.get(want, want))
        if want == "char":
            return CE(f"F.lit(ctx.symget({q(name)}))", "char", 200)
        return CE(f"F.lit(ctx.symget_num({q(name)}))", "num")

    def param(self, name, kind, line):
        return self.state.note_param(name, kind)

    def user_format(self, name):
        return self.state.user_formats.get(name.upper())

    def hazard(self, hid, detail, line):
        e = (hid, detail, line)
        if e not in self.hazards:
            self.hazards.append(e)

    def expand_var_items(self, items, line):
        from .datastep import expand_items
        return expand_items(items, self.schema.vars, {}, line)


def known_schema(ref: A.DatasetRef, state) -> Schema:
    if ref.dynamic:
        raise Unsupported(f"data set name built from run-time macro variables: {ref.display}", ref.line)
    sch = state.catalog.get(ref.key)
    if sch is None:
        if not getattr(state, "infer_schemas", False):
            raise Unsupported(f"schema of {ref.display} is unknown (add it to the schema catalog)", ref.line)
        return Schema([], known=False)
    return sch


def input_expr(ref: A.DatasetRef, state, hazards: list) -> tuple[str, Schema]:
    """Code reading ``ref`` with its data set options applied, and the resulting schema."""
    from .datastep import expand_items

    sch = known_schema(ref, state)
    vars_ = [Var(v.name, v.type, v.length, v.format, v.informat, v.label) for v in sch.vars]
    code = f"ctx.get({q(ref.display.lower())})"
    o = ref.options
    if o.keep is not None:
        names = expand_items(o.keep, vars_, {}, ref.line)
        missing = {n.upper() for n in names} - {v.name.upper() for v in vars_}
        if missing:
            raise Unsupported(f"KEEP= variables not in {ref.display}: {sorted(missing)}", ref.line)
        names = [v.name for v in vars_ if v.name.upper() in {n.upper() for n in names}]
        code = f"P.keep({code}, {names!r})"
        vars_ = [v for v in vars_ if v.name.upper() in {n.upper() for n in names}]
    if o.drop is not None:
        names = expand_items(o.drop, vars_, {}, ref.line)
        code = f"P.drop({code}, {names!r})"
        vars_ = [v for v in vars_ if v.name.upper() not in {n.upper() for n in names}]
    if o.rename:
        code = f"P.rename({code}, {[tuple(p) for p in o.rename]!r})"
        for old, new in o.rename:
            for v in vars_:
                if v.name.upper() == old.upper():
                    v.name = new
    out_schema = Schema(vars_, sch.label, known=sch.known)
    if o.where is not None:
        cond = ExprCompiler(SchemaScope(out_schema, state, hazards), where=True).cond(o.where)
        code = f"{code}.where({cond})"
    if o.firstobs is not None or o.obs is not None:
        code = f"P.obs({code}, firstobs={o.firstobs!r}, obs_={o.obs!r})"
    return code, out_schema


def output_apply(ref: A.DatasetRef, var: str, schema: Schema, state, hazards: list) -> tuple[list[str], Schema]:
    """Lines applying output data set options to DataFrame variable ``var``; returns (lines, schema)."""
    from .datastep import expand_items

    lines = []
    vars_ = [Var(v.name, v.type, v.length, v.format, v.informat, v.label) for v in schema.vars]
    o = ref.options
    if o.keep is not None:
        names = expand_items(o.keep, vars_, {}, ref.line)
        keepset = {n.upper() for n in names}
        names = [v.name for v in vars_ if v.name.upper() in keepset]
        lines.append(f"{var} = P.keep({var}, {names!r})")
        vars_ = [v for v in vars_ if v.name.upper() in keepset]
    if o.drop is not None:
        names = expand_items(o.drop, vars_, {}, ref.line)
        lines.append(f"{var} = P.drop({var}, {names!r})")
        vars_ = [v for v in vars_ if v.name.upper() not in {n.upper() for n in names}]
    if o.rename:
        lines.append(f"{var} = P.rename({var}, {[tuple(p) for p in o.rename]!r})")
        for old, new in o.rename:
            for v in vars_:
                if v.name.upper() == old.upper():
                    v.name = new
    out = Schema(vars_, o.label or schema.label, known=schema.known)   # run-time columns stay run-time downstream
    if o.where is not None:
        cond = ExprCompiler(SchemaScope(out, state, hazards), where=True).cond(o.where)
        lines.append(f"{var} = {var}.where({cond})")
    return lines, out


def meta_literal(schema: Schema) -> str:
    items = [f"{v.name!r}: {v.meta_dict()!r}" for v in schema.vars]
    if len(items) <= 3:
        return "{" + ", ".join(items) + "}"
    return "{\n        " + ",\n        ".join(items) + ",\n    }"


def put_line(ref: A.DatasetRef, var: str, schema: Schema, order: str | None = None, with_meta: bool = False) -> str:
    """ctx.put(...).  Column metadata travels with the DataFrame; pass with_meta for new columns."""
    label = f", label={schema.label!r}" if schema.label else ""
    meta = f", meta={meta_literal(schema)}" if with_meta else ""
    return f"ctx.put({q(ref.display.lower())}, {var}{meta}{label})"
