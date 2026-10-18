"""PROC SQL -> Spark SQL with SAS semantics.

Pipeline per statement:

1. token-level pre-pass with the SAS lexer: data set references in FROM/JOIN
   (with data set options) become temp views, SAS literals ('01JAN2020'd, '.'),
   mnemonic operators, CALCULATED, IS MISSING, CONTAINS, INTO :macrovars and
   column modifiers (FORMAT= LENGTH= LABEL=) are rewritten or extracted;
2. sqlglot parse (Spark dialect);
3. AST transforms for SAS semantics: missing-value comparisons (``x = .``,
   null-safe equality, ``sas_lt``...), blank-aware aggregates over character
   columns, row-wise SUM/MIN/MAX, SAS function mapping, CALCULATED inlining,
   remerged summary statistics -> window aggregates, ``SELECT *`` duplicates;
4. pretty Spark SQL calling the same ``sas_*`` runtime functions as DATA steps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..lang import ast as A
from ..lang.lexer import OPAQUE, RT_CLOSE, RT_OPEN, Kind, tokenize
from ..lang.parser import ParseError, Parser
from ..runtime import dispatch as D
from ..runtime import pyfuncs
from ..semantic.schema import Schema, Var
from .dsopts import input_expr
from .exprs import Unsupported, q, rt_text

_FROM_END = {"WHERE", "GROUP", "HAVING", "ORDER", "UNION", "EXCEPT", "INTERSECT", "OUTER", "ON", "INNER", "LEFT",
             "RIGHT", "FULL", "CROSS", "NATURAL", "JOIN", "INTO", "SET", "VALUES", "AS", "CORR", "CORRESPONDING"}
_MNEMONIC = {"EQ": "=", "NE": "<>", "GT": ">", "LT": "<", "GE": ">=", "LE": "<="}
_MODIFIERS = {"FORMAT", "LENGTH", "LABEL", "INFORMAT", "TRANSCODE"}


@dataclass
class TableRef:
    view: str
    ref: A.DatasetRef
    alias: str


@dataclass
class Prepared:
    sql: str
    tables: list[TableRef] = field(default_factory=list)
    into: list[str] = field(default_factory=list)
    into_sep: str | None = None
    into_trimmed: bool = False
    into_range: tuple[str, int, int | None] | None = None
    modifiers: dict[int, dict[str, str]] = field(default_factory=dict)
    modifiers_by_name: dict[str, dict[str, str]] = field(default_factory=dict)
    calculated: set[str] = field(default_factory=set)


class SqlTranslator:
    def __init__(self, state, hazards: list):
        self.state = state
        self.hazards = hazards
        self.views: dict[str, TableRef] = {}
        self.functions: set[str] = set()
        self._n = 0
        self.rt_literals: list[str] = []   # Python expressions for literals holding run-time macro values

    def hazard(self, hid: str, detail: str, line: int = 0) -> None:
        e = (hid, detail, line)
        if e not in self.hazards:
            self.hazards.append(e)

    # ------------------------------------------------------------------ pre-pass
    def prepare(self, text: str, line: int) -> Prepared:
        toks = tokenize(text)
        p = Parser(text)
        out: list[str] = []
        pos = 0
        prep = Prepared("")
        i = 0
        depth = 0
        in_select = False
        select_depth = -1
        item_idx = 0
        n = len(toks)

        def emit_until(k: int) -> None:
            nonlocal pos
            out.append(text[pos:toks[k].start])
            pos = toks[k].start

        while i < n and toks[i].kind is not Kind.EOF:
            t = toks[i]
            u = t.value.upper() if t.kind is Kind.NAME else ""
            if t.is_op("("):
                depth += 1
            elif t.is_op(")"):
                depth -= 1
                if in_select and depth < select_depth:
                    in_select = False
            if t.kind is Kind.NAME and u == "SELECT":
                in_select, select_depth, item_idx = True, depth, 0
            elif t.kind is Kind.NAME and u in ("FROM", "INTO") and in_select and depth == select_depth:
                in_select = False
            elif in_select and depth == select_depth and t.is_op(","):
                item_idx += 1
            # data set references after FROM / JOIN
            if t.kind is Kind.NAME and u in ("FROM", "JOIN") and i + 1 < n and toks[i + 1].kind in (Kind.NAME, Kind.RTMACRO):
                k = i + 1
                while True:
                    emit_until(k)
                    p.i = k
                    try:
                        ref = p.parse_dataset_ref()
                    except ParseError as e:
                        raise Unsupported(f"PROC SQL table reference: {e}", line) from e
                    end_tok = p.i
                    alias = None
                    if p.at_name("AS") and p.peek(1).kind is Kind.NAME:
                        alias = p.peek(1).value
                        end_tok = p.i + 2
                    elif p.peek().kind is Kind.NAME and p.peek().upper not in _FROM_END:
                        alias = p.peek().value
                        end_tok = p.i + 1
                    view = self._view_for(ref, alias)
                    out.append(f"{view.view} AS {view.alias} ")
                    prep.tables.append(view)
                    pos = toks[end_tok].start if end_tok < n else len(text)
                    i = end_tok
                    if i < n and toks[i].is_op(",") and u == "FROM":
                        nxt = i + 1
                        if nxt < n and toks[nxt].kind in (Kind.NAME, Kind.RTMACRO):
                            emit_until(i)
                            out.append(", ")
                            pos = toks[nxt].start
                            k = nxt
                            continue
                    break
                continue
            # INTO :macro-variables
            if t.kind is Kind.NAME and u == "INTO" and i + 1 < n and toks[i + 1].is_op(":"):
                emit_until(i)
                j = i + 1
                names = []
                while j < n and toks[j].is_op(":"):
                    nm = toks[j + 1].value
                    j += 2
                    if j < n and toks[j].is_op("-"):
                        j += 1
                        if j < n and toks[j].is_op(":"):
                            hi = toks[j + 1].value
                            j += 2
                        else:
                            hi = None
                        m1 = re.match(r"^(.*?)(\d+)$", nm)
                        m2 = re.match(r"^(.*?)(\d+)$", hi) if hi else None
                        if not m1:
                            raise Unsupported(f"INTO range :{nm}-", line)
                        prep.into_range = (m1.group(1), int(m1.group(2)), int(m2.group(2)) if m2 else None)
                    else:
                        names.append(nm)
                    if j < n and toks[j].is_op(","):
                        j += 1
                        continue
                    break
                while j < n and toks[j].kind is Kind.NAME and toks[j].upper in ("SEPARATED", "BY", "TRIMMED", "NOTRIM"):
                    if toks[j].upper == "SEPARATED":
                        prep.into_sep = toks[j + 2].value
                        j += 3
                    else:
                        if toks[j].upper == "TRIMMED":
                            prep.into_trimmed = True
                        j += 1
                prep.into = names
                pos = toks[j].start
                i = j
                continue
            # column modifiers in the select list
            if in_select and depth == select_depth and t.kind is Kind.NAME and u in _MODIFIERS and i + 1 < n and toks[i + 1].is_op("="):
                emit_until(i)
                p.i = i + 2
                if u in ("FORMAT", "INFORMAT"):
                    val = p.read_format_spec() or p.advance().value
                else:
                    tok = p.advance()
                    val = tok.value
                    if u == "LENGTH" and tok.is_op("$"):
                        val = p.advance().value
                prep.modifiers.setdefault(item_idx, {})[u] = val
                pos = toks[p.i].start
                i = p.i
                continue
            if t.kind is Kind.NAME and u in ("PUT", "PUTN", "PUTC", "INPUT", "INPUTN", "INPUTC") and i + 1 < n and toks[i + 1].is_op("("):
                # find the top-level comma of this call and quote the format that follows it
                d0, j = 0, i + 1
                while j < n:
                    if toks[j].is_op("("):
                        d0 += 1
                    elif toks[j].is_op(")"):
                        d0 -= 1
                        if d0 == 0:
                            break
                    elif toks[j].is_op(",") and d0 == 1:
                        k = j + 1
                        while k < n and toks[k].is_op("?"):
                            k += 1
                        p.i = k
                        spec = p.read_format_spec()
                        if spec is not None and p.peek().is_op(")", ","):
                            emit_until(j + 1)
                            out.append(" " + q(spec))
                            pos = toks[p.i].start
                        break
                    j += 1
                i += 1
                continue
            if t.kind is Kind.NAME and u == "CALCULATED" and i + 1 < n:
                emit_until(i)
                nm = toks[i + 1].value
                prep.calculated.add(nm.upper())
                out.append(f"__calc__{nm}")
                pos = toks[i + 1].end
                i += 2
                continue
            if t.kind is Kind.STRING and RT_OPEN in t.value:
                # "&r" / "&start"d: a typed SQL literal filled in when the cell runs (escaped, never spliced as code)
                if t.suffix not in ("", "d", "dt", "t"):
                    raise Unsupported(f"run-time macro value inside a '...'{t.suffix} literal", line)
                emit_until(i)
                expr = rt_text(t.value, line, lambda nm: self.state.runtime_macros.add(nm.upper()))
                k = len(self.rt_literals)
                if t.suffix:
                    self.rt_literals.append(f"S.{t.suffix}({expr})")
                    out.append(f"__rtn__{k}")
                else:
                    self.rt_literals.append(f"S.sql_esc({expr})")
                    out.append(f"'__rts__{k}'")
                pos = t.end
                i += 1
                continue
            if t.kind is Kind.STRING and t.suffix in ("d", "dt", "t"):
                emit_until(i)
                spec = {"d": "DATE11." if "-" in t.value else "DATE9.", "dt": "DATETIME40.", "t": "TIME20."}[t.suffix]
                v = pyfuncs.input_value(t.value, spec)
                if v is None:
                    raise Unsupported(f"invalid SAS literal '{t.value}'{t.suffix}", line)
                out.append(f"{v!r}D")
                pos = t.end
                i += 1
                continue
            if t.kind is Kind.MISSING:
                emit_until(i)
                out.append("NULL")
                pos = t.end
                i += 1
                continue
            if t.kind is Kind.NAME and u in _MNEMONIC and i > 0 and not toks[i - 1].is_op(".") and not toks[i + 1].is_op("("):
                emit_until(i)
                out.append(_MNEMONIC[u])
                pos = t.end
                i += 1
                continue
            if t.kind is Kind.NAME and u == "IS" and i + 1 < n and toks[i + 1].upper in ("MISSING", "NOT"):
                if toks[i + 1].upper == "MISSING":
                    emit_until(i)
                    out.append("IS NULL")
                    pos = toks[i + 1].end
                    i += 2
                    continue
                if i + 2 < n and toks[i + 2].upper == "MISSING":
                    emit_until(i)
                    out.append("IS NOT NULL")
                    pos = toks[i + 2].end
                    i += 3
                    continue
            if (t.kind is Kind.NAME and u == "CONTAINS") or t.is_op("?"):
                emit_until(i)
                if i + 1 < n and toks[i + 1].kind is Kind.STRING:
                    lit = toks[i + 1].value.replace("%", "\\%").replace("_", "\\_")
                    out.append(f"LIKE {q('%' + lit + '%')}")
                    pos = toks[i + 1].end
                    i += 2
                    continue
                raise Unsupported("CONTAINS with a non-literal operand", line)
            if t.is_op("^=", "~=", "¬="):
                emit_until(i)
                out.append("<>")
                pos = t.end
                i += 1
                continue
            if t.is_op("^", "~", "¬"):
                emit_until(i)
                out.append(" NOT ")
                pos = t.end
                i += 1
                continue
            if t.is_op("&"):
                emit_until(i)
                out.append(" AND ")
                pos = t.end
                i += 1
                continue
            if t.is_op("|", "!"):
                emit_until(i)
                out.append(" OR ")
                pos = t.end
                i += 1
                continue
            if t.is_op("!!", "¦¦"):
                emit_until(i)
                out.append("||")
                pos = t.end
                i += 1
                continue
            if t.is_op("=*"):
                raise Unsupported("sounds-like operator =*", line)
            if t.kind is Kind.RTMACRO:
                if t.value.startswith(OPAQUE):
                    raise Unsupported(f"value of %{t.value[1:]}(...) is only known at run time", line)
                emit_until(i)
                out.append(f"__rt__{t.value}")
                pos = t.end
                self.state.runtime_macros.add(t.value.upper())
                i += 1
                continue
            i += 1
        out.append(text[pos:])
        prep.sql = "".join(out)
        return prep

    def _view_for(self, ref: A.DatasetRef, alias: str | None) -> TableRef:
        self._n += 1
        view = f"__t{self._n}"
        tr = TableRef(view, ref, alias or ref.name)
        self.views[view] = tr
        return tr

    # ------------------------------------------------------------------ schemas
    def table_schema(self, tr: TableRef) -> Schema:
        _, sch = input_expr(tr.ref, self.state, self.hazards)
        return sch

    # ------------------------------------------------------------------ translate
    def translate_query(self, prep: Prepared, line: int) -> tuple[exp.Expression, Schema]:
        try:
            tree = sqlglot.parse_one(prep.sql, read="spark")
        except sqlglot.errors.ParseError as e:
            raise Unsupported(f"PROC SQL syntax not understood: {str(e).splitlines()[0][:160]}", line) from e
        top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if top is not None:
            for idx, e in enumerate(top.expressions):
                if idx in prep.modifiers:
                    prep.modifiers_by_name[e.alias_or_name.upper()] = prep.modifiers[idx]
        tree = _Transformer(self, prep, line).run(tree)
        schema = self.output_schema(tree, prep)
        return tree, schema

    def output_schema(self, tree: exp.Expression, prep: Prepared) -> Schema:
        sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        out: list[Var] = []
        if sel is None:
            return Schema(out)
        scope = _scope_for(self, sel)
        for idx, e in enumerate(sel.expressions):
            name = e.alias_or_name
            if not name or name.startswith("_col"):
                name = f"_TEMA{idx + 1:03d}"
            typ, length, fmt, label = _infer(e.this if isinstance(e, exp.Alias) else e, scope)
            mods = prep.modifiers_by_name.get(name.upper(), {})
            if mods.get("LENGTH"):
                length = int(float(mods["LENGTH"].lstrip("$")))
            if mods.get("FORMAT"):
                fmt = mods["FORMAT"].upper()
            if mods.get("LABEL"):
                label = mods["LABEL"]
            out.append(Var(name, typ or "num", (length or 200) if typ == "char" else 8, fmt, None, label))
        return Schema(out)

    def sql_text(self, tree: exp.Expression) -> str:
        text = tree.sql(dialect="spark", pretty=True)
        for f in re.findall(r"\b(sas_[a-z0-9_]+)\(", text, re.I):
            self.functions.add(f.lower())
        text = re.sub(r"'__rts__(\d+)'", lambda m: "'{_rx_" + m.group(1) + "}'", text)
        text = re.sub(r"__rtn__(\d+)", lambda m: "{_rx_" + m.group(1) + "!r}D", text)
        return re.sub(r"__rt__([A-Za-z_][A-Za-z0-9_]*)", lambda m: "{" + f"_mv_{m.group(1).lower()}" + "}", text)


# ---------------------------------------------------------------------------
# scope & type inference
@dataclass
class _Scope:
    tables: dict[str, Schema]           # alias (upper) -> schema
    aliases: dict[str, exp.Expression]  # select-list alias (upper) -> expression

    def column(self, name: str, table: str | None):
        u = name.upper()
        if table:
            sch = self.tables.get(table.upper())
            return sch.get(u) if sch else None
        for sch in self.tables.values():
            v = sch.get(u)
            if v is not None:
                return v
        return None


def _from(sel: exp.Select):
    """FROM clause of a SELECT (sqlglot >= 30 stores it under 'from_')."""
    return sel.args.get("from_") or sel.args.get("from")


def _scope_for(tr: SqlTranslator, sel: exp.Select) -> _Scope:
    tables: dict[str, Schema] = {}
    frm = _from(sel)
    sources = ([frm.this] if frm is not None and frm.this is not None else []) + [j.this for j in sel.args.get("joins") or []]
    for src in sources:
        if isinstance(src, exp.Table):
            if src.name in tr.views:
                tables[src.alias_or_name.upper()] = tr.table_schema(tr.views[src.name])
        elif isinstance(src, exp.Subquery) and src.alias:
            inner = src.this if isinstance(src.this, exp.Select) else src.this.find(exp.Select)
            if inner is not None:
                tables[src.alias.upper()] = tr.output_schema(inner, Prepared(""))
    aliases = {e.alias.upper(): e.this for e in sel.expressions if isinstance(e, exp.Alias)}
    return _Scope(tables, aliases)


_NUM_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "STDDEV_SAMP", "VARIANCE", "VAR_SAMP", "ROW_NUMBER", "ABS"}
_CHAR_FUNCS = {"UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM", "SUBSTRING", "SUBSTR", "CONCAT", "CONCAT_WS", "LPAD", "RPAD"}


def _infer(e: exp.Expression, scope: _Scope):
    """(type, length, format, label) of an expression."""
    if isinstance(e, exp.Alias):
        return _infer(e.this, scope)
    if isinstance(e, exp.Column):
        v = scope.column(e.name, e.table or None)
        if v is not None:
            return v.type, v.length, v.format, v.label
        al = scope.aliases.get(e.name.upper())
        if al is not None and al is not e:
            return _infer(al, scope)
        return None, None, None, None
    if isinstance(e, exp.Literal):
        return ("char", max(len(e.this), 1), None, None) if e.is_string else ("num", 8, None, None)
    if isinstance(e, exp.Null):
        return None, None, None, None
    if isinstance(e, (exp.DPipe, exp.Concat)):
        lens = [_infer(x, scope)[1] or 200 for x in e.flatten()] if isinstance(e, exp.DPipe) else [200]
        return "char", sum(lens), None, None
    if isinstance(e, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.Neg)):
        return "num", 8, None, None
    if isinstance(e, (exp.Count, exp.Sum, exp.Avg, exp.Stddev, exp.Variance)):
        return "num", 8, None, None
    if isinstance(e, (exp.Min, exp.Max)):
        return _infer(e.this, scope)
    if isinstance(e, exp.Case):
        for w in e.args.get("ifs") or []:
            t = _infer(w.args.get("true"), scope)
            if t[0]:
                return t
        return _infer(e.args["default"], scope) if e.args.get("default") is not None else (None, None, None, None)
    if isinstance(e, exp.Coalesce):
        return _infer(e.this, scope)
    if isinstance(e, (exp.Paren,)):
        return _infer(e.this, scope)
    if isinstance(e, exp.Cast):
        return ("char", 200, None, None) if e.to.is_type("text", "varchar", "string") else ("num", 8, None, None)
    if isinstance(e, (exp.Upper, exp.Lower, exp.Trim, exp.Substring)):
        t = _infer(e.this, scope)
        return "char", t[1] or 200, None, None
    if isinstance(e, (exp.Predicate, exp.Not, exp.And, exp.Or, exp.Is)):
        return "num", 8, None, None
    if isinstance(e, exp.Window):
        return _infer(e.this, scope)
    if isinstance(e, (exp.Anonymous, exp.Func)):
        name = (e.name if isinstance(e, exp.Anonymous) else e.sql_name()).upper()
        if name.startswith("SAS_PUT") or name.startswith("SAS_FMT") or name.startswith("FMT_") or name in (
                "SAS_SCAN", "SAS_SCAN_M", "SAS_SUBSTR2", "SAS_SUBSTR3", "SAS_COMPRESS_RX", "SAS_TRANWRD", "SAS_TRANSLATE",
                "SAS_TRIM", "SAS_PROPCASE", "SAS_CATS_ARR", "SAS_CATX_ARR", "SAS_CAT_ARR", "SAS_CATT_ARR", "SAS_COALESCEC_ARR",
                "SAS_COMPBL", "SAS_QUOTE", "SAS_DEQUOTE", "SAS_RIGHT", "SAS_INPUT_CHAR", "SAS_REPEAT"):
            return "char", 200, None, None
        if name in _CHAR_FUNCS:
            return "char", 200, None, None
        return "num", 8, None, None
    return None, None, None, None


def _is_char(e: exp.Expression, scope: _Scope) -> bool:
    return _infer(e, scope)[0] == "char"


# ---------------------------------------------------------------------------
# AST transforms
class _Transformer:
    def __init__(self, tr: SqlTranslator, prep: Prepared, line: int):
        self.tr = tr
        self.prep = prep
        self.line = line

    def run(self, tree: exp.Expression) -> exp.Expression:
        for sel in reversed(list(tree.find_all(exp.Select))):
            self._select(sel)
        return tree

    # ------------------------------------------------------------------
    def _select(self, sel: exp.Select) -> None:
        self._alias_subqueries(sel)
        scope = _scope_for(self.tr, sel)
        self._inline_calculated(sel, scope)
        self._expand_star(sel, scope)
        # functions first (so later rules see sas_* calls); operators are not functions
        for node in reversed(list(sel.find_all(exp.Anonymous, exp.Func))):
            if node.find_ancestor(exp.Select) is not sel:
                continue
            if not isinstance(node, exp.Anonymous) and isinstance(node, (exp.Connector, exp.Unary, exp.Binary)) \
                    and not isinstance(node, exp.Mod):
                continue
            new = self._function(node, scope)
            if new is not None and new is not node:
                node.replace(new)
        for node in reversed(list(sel.find_all(exp.Div))):
            if node.find_ancestor(exp.Select) is not sel:
                continue
            b = node.expression
            if isinstance(b, exp.Literal) and not b.is_string and float(b.this) != 0:
                continue
            self.tr.hazard("H-DIV0", "division by zero is missing in SAS (try_divide)", self.line)
            node.replace(exp.Anonymous(this="try_divide", expressions=[node.this.copy(), b.copy()]))
        for node in reversed(list(sel.find_all(exp.Count, exp.Min, exp.Max))):
            if node.find_ancestor(exp.Select) is not sel:
                continue
            self._blank_aware_agg(node, scope)
        for node in reversed(list(sel.find_all(exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Is))):
            if node.find_ancestor(exp.Select) is not sel:
                continue
            new = self._comparison(node, scope)
            if new is not None and new is not node:
                node.replace(new)
        self._remerge(sel, scope)

    def _alias_subqueries(self, sel: exp.Select) -> None:
        """SAS allows unaliased subqueries in FROM; give them names so * can be expanded."""
        frm = _from(sel)
        sources = ([frm.this] if frm is not None and frm.this is not None else []) + [j.this for j in sel.args.get("joins") or []]
        for src in sources:
            if isinstance(src, exp.Subquery) and not src.alias:
                self.tr._n += 1
                src.set("alias", exp.TableAlias(this=exp.to_identifier(f"__s{self.tr._n}")))

    def _inline_calculated(self, sel: exp.Select, scope: _Scope) -> None:
        if not self.tr or not any(isinstance(c, exp.Column) and c.name.startswith("__calc__") for c in sel.find_all(exp.Column)):
            return
        for c in list(sel.find_all(exp.Column)):
            if c.name.startswith("__calc__"):
                target = c.name[len("__calc__"):].upper()
                e = scope.aliases.get(target)
                if e is None:
                    raise Unsupported(f"CALCULATED {target}: no such column alias", self.line)
                c.replace(exp.Paren(this=e.copy()))

    def _expand_star(self, sel: exp.Select, scope: _Scope) -> None:
        new_exprs = []
        changed = False
        seen: set[str] = set()
        for e in sel.expressions:
            if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
                changed = True
                table = e.table if isinstance(e, exp.Column) else None
                items = [(table.upper(), scope.tables[table.upper()])] if table and table.upper() in scope.tables else list(scope.tables.items())
                if any(not sch.known for _, sch in items) or not items:
                    new_exprs.append(e)
                    continue
                for alias, sch in items:
                    for v in sch.vars:
                        if v.name.upper() in seen:
                            self.tr.hazard("H-SQL-STAR-DUP", f"SELECT * : duplicate column {v.name} keeps the first table's value (SAS warns)", self.line)
                            continue
                        seen.add(v.name.upper())
                        new_exprs.append(exp.column(v.name, table=alias))
            else:
                seen.add(e.alias_or_name.upper())
                new_exprs.append(e)
        if changed:
            sel.set("expressions", new_exprs)

    def _blank_aware_agg(self, node: exp.Expression, scope: _Scope) -> None:
        arg = node.this
        inner = arg.this if isinstance(arg, exp.Distinct) else arg
        if isinstance(arg, exp.Distinct) and arg.expressions:
            inner = arg.expressions[0]
        if inner is None or isinstance(inner, exp.Star) or not _is_char(inner, scope):
            return
        self.tr.hazard("H-SQL-CHAR-AGG", "blank character values are missing in SQL aggregates", self.line)
        wrapped = exp.Nullif(this=inner.copy(), expression=exp.Literal.string(""))
        inner.replace(wrapped)
        if isinstance(node, (exp.Min, exp.Max)):
            node.replace(exp.Coalesce(this=node.copy(), expressions=[exp.Literal.string("")]))

    def _comparison(self, node: exp.Expression, scope: _Scope):
        if isinstance(node, exp.Is):
            if isinstance(node.expression, exp.Null) and _is_char(node.this, scope):
                return exp.EQ(this=node.this.copy(), expression=exp.Literal.string(""))
            return node
        a, b = node.this, node.expression
        a_null, b_null = isinstance(a, exp.Null), isinstance(b, exp.Null)
        if a_null or b_null:
            if a_null and b_null:
                return exp.true() if isinstance(node, (exp.EQ, exp.LTE, exp.GTE)) else exp.false()
            col = b if a_null else a
            kind = type(node)
            if a_null:
                kind = {exp.LT: exp.GT, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.GTE: exp.LTE}.get(kind, kind)
            if _is_char(col, scope):
                blank = exp.EQ(this=col.copy(), expression=exp.Literal.string(""))
                is_null, not_null = blank, exp.Not(this=blank.copy())
            else:
                is_null = exp.Is(this=col.copy(), expression=exp.Null())
                not_null = exp.Not(this=exp.Is(this=col.copy(), expression=exp.Null()))
            return {exp.EQ: is_null, exp.NEQ: not_null, exp.GT: not_null, exp.LTE: is_null,
                    exp.GTE: exp.true(), exp.LT: exp.false()}[kind]
        lit_a, lit_b = isinstance(a, exp.Literal), isinstance(b, exp.Literal)
        if lit_a and lit_b:
            return node
        char = _is_char(a, scope) or _is_char(b, scope)
        if isinstance(node, (exp.EQ, exp.NEQ)):
            if char:
                return node
            if node.find_ancestor(exp.Join) is not None:
                self.tr.hazard("H-SQL-JOIN-MISS", "PROC SQL joins match missing keys (null-safe equality)", self.line)
            eq = exp.NullSafeEQ(this=a.copy(), expression=b.copy())
            return eq if isinstance(node, exp.EQ) else exp.Not(this=exp.Paren(this=eq))
        if char:
            return node
        self.tr.hazard("H-MISS-CMP", "missing values sort below every number in SAS comparisons", self.line)
        fn = {exp.LT: "sas_lt", exp.LTE: "sas_le", exp.GT: "sas_gt", exp.GTE: "sas_ge"}[type(node)]
        return exp.Anonymous(this=fn, expressions=[_dbl(a.copy()), _dbl(b.copy())])

    # ------------------------------------------------------------------ functions
    def _function(self, node, scope: _Scope):
        if isinstance(node, exp.Anonymous):
            name = node.name.upper()
            if name.startswith("SAS_") or name.startswith("FMT_") or name in ("ROW_NUMBER", "MONOTONICALLY_INCREASING_ID", "ARRAY", "NULLIF", "LTRIM", "RTRIM"):
                return node
            h = getattr(self, "_fn_" + name.lower(), None)
            if h is None:
                raise Unsupported(f"function {name}() in PROC SQL", self.line)
            return h(list(node.expressions), scope)
        return self._builtin(node, scope)

    _PASS = tuple(c for c in (getattr(exp, n, None) for n in (
        "Count", "Avg", "Coalesce", "Abs", "Cast", "Upper", "Lower", "Nullif", "Case", "If", "Concat", "Stddev",
        "StddevSamp", "Variance", "VarianceSamp", "Array", "RowNumber", "Window", "Distinct", "TryCast")) if c is not None)

    def _builtin(self, node, scope: _Scope):
        args = [node.this] + list(node.expressions or []) if node.args.get("this") is not None else list(node.expressions or [])
        if isinstance(node, exp.Sum):
            return self._fn_sum(args, scope) if len(args) > 1 else node
        if isinstance(node, exp.Min):
            return self._fn_min(args, scope) if len(args) > 1 else node
        if isinstance(node, exp.Max):
            return self._fn_max(args, scope) if len(args) > 1 else node
        if isinstance(node, exp.Trim):
            if node.args.get("position") or node.args.get("expression"):
                raise Unsupported("TRIM(... FROM ...) in PROC SQL", self.line)
            return self._call("sas_trim", [node.this.copy()])
        if isinstance(node, exp.Substring):
            sargs = [node.this] + [x for x in (node.args.get("start"), node.args.get("length")) if x is not None]
            return self._fn_substr(sargs, scope)
        if isinstance(node, exp.Length):
            return self._call("sas_length", [node.this.copy()])
        if isinstance(node, exp.Round):
            rargs = [node.this] + ([node.args["decimals"]] if node.args.get("decimals") is not None else [])
            return self._fn_round(rargs, scope)
        if isinstance(node, exp.Mod):
            return self._call("sas_mod", [_dbl(node.this.copy()), _dbl(node.expression.copy())])
        if isinstance(node, (exp.Ln, exp.Log)):
            if isinstance(node, exp.Log) and node.args.get("expression") is not None:
                raise Unsupported("LOG with a base in PROC SQL", self.line)
            return self._call("sas_log", [_dbl(node.this.copy())])
        for cls, fn in ((exp.Sqrt, "sas_sqrt"), (exp.Exp, "sas_exp"), (exp.Floor, "sas_floor"), (exp.Ceil, "sas_ceil"),
                        (exp.Year, "sas_year"), (exp.Month, "sas_month"), (exp.Day, "sas_day")):
            if isinstance(node, cls):
                return self._call(fn, [_dbl(node.this.copy())])
        if isinstance(node, exp.Left) and node.args.get("expression") is None:
            return exp.Anonymous(this="ltrim", expressions=[node.this.copy()])
        if isinstance(node, exp.Repeat):
            return self._call("sas_repeat", [node.this.copy(), _dbl(node.args["times"].copy())])
        if hasattr(exp, "Translate") and isinstance(node, exp.Translate):
            return self._fn_translate([node.this] + [node.args.get("from"), node.args.get("to")], scope)
        if isinstance(node, self._PASS) or isinstance(node, (exp.AggFunc,)) and not isinstance(node, (exp.Sum, exp.Min, exp.Max)):
            return node
        raise Unsupported(f"function {node.sql_name()}() in PROC SQL", self.line)

    def _call(self, fn: str, args) -> exp.Expression:
        self.tr.functions.add(fn)
        return exp.Anonymous(this=fn, expressions=args)

    def _spec_call(self, spec: D.CallSpec, values) -> exp.Expression:
        def render(node):
            if isinstance(node, D.Arg):
                return values[node.i].copy()
            if isinstance(node, D.Lit):
                v = node.v
                if isinstance(v, bool):
                    return exp.true() if v else exp.false()
                if isinstance(v, (int, float)):
                    return exp.Literal.number(v)
                if v is None:
                    return exp.Null()
                return exp.Literal.string(str(v))
            return self._call(node.name, [render(a) for a in node.args])
        return render(spec)

    def _lit_str(self, e, what) -> str:
        if isinstance(e, exp.Literal) and e.is_string:
            return e.this
        if isinstance(e, exp.Column) and not e.table:
            return e.name   # unquoted format names such as best / date9 parsed as columns
        raise Unsupported(f"{what} must be a literal in PROC SQL", self.line)

    def _rowwise(self, fn: str, args, scope):
        arr = exp.Array(expressions=[_dbl(a.copy()) for a in args])
        return self._call(fn, [arr])

    def _fn_sum(self, args, scope):
        if len(args) == 1:
            return exp.Sum(this=args[0].copy())
        return self._rowwise("sas_sum_arr", args, scope)

    def _fn_min(self, args, scope):
        if len(args) == 1:
            return exp.Min(this=args[0].copy())
        return self._rowwise("sas_min_arr", args, scope)

    def _fn_max(self, args, scope):
        if len(args) == 1:
            return exp.Max(this=args[0].copy())
        return self._rowwise("sas_max_arr", args, scope)

    def _fn_mean(self, args, scope):
        if len(args) == 1:
            return exp.Avg(this=args[0].copy())
        return self._rowwise("sas_mean_arr", args, scope)

    _fn_avg = _fn_mean

    def _fn_n(self, args, scope):
        if len(args) == 1:
            return exp.Count(this=args[0].copy())
        return self._rowwise("sas_n_arr", args, scope)

    def _fn_freq(self, args, scope):
        return exp.Count(this=args[0].copy() if args else exp.Star())

    def _fn_nmiss(self, args, scope):
        if len(args) == 1:
            a = args[0]
            return exp.Sum(this=exp.Case(ifs=[exp.If(this=exp.Is(this=a.copy(), expression=exp.Null()), true=exp.Literal.number(1))],
                                         default=exp.Literal.number(0)))
        return self._rowwise("sas_nmiss_arr", args, scope)

    def _fn_std(self, args, scope):
        if len(args) == 1:
            return exp.Stddev(this=args[0].copy())
        return self._rowwise("sas_std_arr", args, scope)

    def _fn_var(self, args, scope):
        if len(args) == 1:
            return exp.Variance(this=args[0].copy())
        return self._rowwise("sas_var_arr", args, scope)

    def _fn_coalescec(self, args, scope):
        return self._call("sas_coalescec_arr", [exp.Array(expressions=[a.copy() for a in args])])

    def _fn_put(self, args, scope):
        spec = self._lit_str(args[1], "PUT format")
        if "." not in spec:
            spec += "."
        kind = "char" if _is_char(args[0], scope) else "num"
        if pyfuncs.format_kind(spec) == "user" or (spec.startswith("$") and pyfuncs.parse_format(spec)[1] not in ("", "CHAR", "UPCASE", "QUOTE")):
            fn = self.tr.state.user_formats.get(spec.rstrip(".0123456789").upper())
            if fn is None:
                raise Unsupported(f"user-defined format {spec} in PROC SQL", self.line)
            return self._call(fn, [args[0].copy()])
        try:
            return self._spec_call(D.put_call(spec, kind), [args[0] if kind == "char" else _dbl(args[0])])
        except D.UnsupportedSpec as e:
            raise Unsupported(str(e), self.line) from e

    def _fn_input(self, args, scope):
        spec = self._lit_str(args[1], "INPUT informat")
        if not spec.endswith(".") and "." not in spec:
            spec += "."
        return self._spec_call(D.input_call(spec), [args[0]])

    def _fn_scan(self, args, scope):
        delims = self._lit_str(args[2], "SCAN delimiters") if len(args) > 2 else None
        mods = self._lit_str(args[3], "SCAN modifiers") if len(args) > 3 else ""
        rx, m = D.scan_regex(delims, mods)
        return self._call("sas_scan_m" if m else "sas_scan", [args[0].copy(), _dbl(args[1].copy()), exp.Literal.string(rx)])

    def _fn_index(self, args, scope):
        return self._call("sas_index", [args[0].copy(), args[1].copy()])

    def _fn_compress(self, args, scope):
        chars = self._lit_str(args[1], "COMPRESS characters") if len(args) > 1 else None
        mods = self._lit_str(args[2], "COMPRESS modifiers") if len(args) > 2 else ""
        return self._call("sas_compress_rx", [args[0].copy(), exp.Literal.string(D.compress_regex(chars, mods))])

    def _fn_strip(self, args, scope):
        return exp.Trim(this=args[0].copy())

    def _fn_left(self, args, scope):
        return exp.Anonymous(this="ltrim", expressions=[args[0].copy()])

    def _fn_upcase(self, args, scope):
        return exp.Upper(this=args[0].copy())

    def _fn_lowcase(self, args, scope):
        return exp.Lower(this=args[0].copy())

    def _fn_propcase(self, args, scope):
        return self._call("sas_propcase", [args[0].copy(), exp.Literal.string(pyfuncs.PROPCASE_DEFAULT_DELIMS)])

    def _fn_substr(self, args, scope):
        if len(args) > 2:
            return self._call("sas_substr3", [args[0].copy(), _dbl(args[1].copy()), _dbl(args[2].copy())])
        return self._call("sas_substr2", [args[0].copy(), _dbl(args[1].copy())])

    def _fn_tranwrd(self, args, scope):
        return self._call("sas_tranwrd", [a.copy() for a in args[:3]])

    def _fn_translate(self, args, scope):
        self.tr.hazard("H-TRANSLATE-ARGS", "TRANSLATE(source, to, from) argument order", self.line)
        return self._call("sas_translate", [a.copy() for a in args[:3]])

    def _cat(self, fn, args, scope, sep=None):
        items = []
        for a in args:
            if _is_char(a, scope):
                items.append(a.copy())
            else:
                items.append(exp.Trim(this=self._call("sas_put_best", [_dbl(a.copy()), exp.Literal.number(12)])))
        arr = exp.Array(expressions=items)
        return self._call(fn, ([sep.copy()] if sep is not None else []) + [arr])

    def _fn_cats(self, args, scope):
        return self._cat("sas_cats_arr", args, scope)

    def _fn_catx(self, args, scope):
        return self._cat("sas_catx_arr", args[1:], scope, sep=args[0])

    def _fn_catt(self, args, scope):
        return self._cat("sas_catt_arr", args, scope)

    def _fn_cat(self, args, scope):
        return self._cat("sas_cat_arr", args, scope)

    def _fn_today(self, args, scope):
        return self._call("sas_today", [])

    _fn_date = _fn_today

    def _fn_datetime(self, args, scope):
        return self._call("sas_now", [])

    def _fn_mdy(self, args, scope):
        return self._call("sas_mdy", [_dbl(a.copy()) for a in args[:3]])

    def _simple(self, fn, args):
        return self._call(fn, [_dbl(a.copy()) for a in args])

    def _fn_year(self, args, scope):
        return self._simple("sas_year", args[:1])

    def _fn_month(self, args, scope):
        return self._simple("sas_month", args[:1])

    def _fn_day(self, args, scope):
        return self._simple("sas_day", args[:1])

    def _fn_qtr(self, args, scope):
        return self._simple("sas_qtr", args[:1])

    def _fn_weekday(self, args, scope):
        return self._simple("sas_weekday", args[:1])

    def _fn_datepart(self, args, scope):
        return self._simple("sas_datepart", args[:1])

    def _fn_timepart(self, args, scope):
        return self._simple("sas_timepart", args[:1])

    def _fn_int(self, args, scope):
        return self._simple("sas_int", args[:1])

    def _fn_round(self, args, scope):
        self.tr.hazard("H-ROUND", "SAS rounding (half away from zero, decimal)", self.line)
        unit = args[1].copy() if len(args) > 1 else exp.Literal.number(1)
        return self._call("sas_round", [_dbl(args[0].copy()), _dbl(unit)])

    def _fn_intnx(self, args, scope):
        interval = self._lit_str(args[0], "INTNX interval")
        align = self._lit_str(args[3], "INTNX alignment") if len(args) > 3 else "B"
        return self._spec_call(D.intnx_call(interval, align), [_dbl(args[1]), _dbl(args[2])])

    def _fn_intck(self, args, scope):
        interval = self._lit_str(args[0], "INTCK interval")
        method = self._lit_str(args[3], "INTCK method") if len(args) > 3 else "DISCRETE"
        return self._spec_call(D.intck_call(interval, method), [_dbl(args[1]), _dbl(args[2])])

    def _fn_yrdif(self, args, scope):
        basis = self._lit_str(args[2], "YRDIF basis") if len(args) > 2 else "ACT/ACT"
        return self._spec_call(D.yrdif_call(basis), [_dbl(args[0]), _dbl(args[1])])

    def _fn_missing(self, args, scope):
        a = args[0]
        if _is_char(a, scope):
            return exp.EQ(this=exp.Trim(this=a.copy()), expression=exp.Literal.string(""))
        return exp.Is(this=a.copy(), expression=exp.Null())

    def _fn_ifn(self, args, scope):
        return exp.Case(ifs=[exp.If(this=args[0].copy(), true=args[1].copy())], default=args[2].copy())

    _fn_ifc = _fn_ifn

    def _fn_monotonic(self, args, scope):
        self.tr.hazard("H-ORDER", "MONOTONIC() depends on physical row order", self.line)
        return exp.Window(this=exp.Anonymous(this="row_number", expressions=[]),
                          order=exp.Order(expressions=[exp.Ordered(this=exp.Anonymous(this="monotonically_increasing_id", expressions=[]))]))

    def _fn_length(self, args, scope):
        return self._call("sas_length", [args[0].copy()])

    def _fn_lengthn(self, args, scope):
        return self._call("sas_lengthn", [args[0].copy()])

    def _fn_find(self, args, scope):
        mods = self._lit_str(args[2], "FIND modifiers") if len(args) > 2 else ""
        return self._call("sas_find", [args[0].copy(), args[1].copy(), exp.Literal.string(mods), exp.Literal.number(1)])

    def _fn_coalesce(self, args, scope):
        return exp.Coalesce(this=args[0].copy(), expressions=[a.copy() for a in args[1:]])

    def _fn_abs(self, args, scope):
        return exp.Abs(this=args[0].copy())

    # ------------------------------------------------------------------ remerge
    def _remerge(self, sel: exp.Select, scope: _Scope) -> None:
        """SAS remerges aggregates onto detail rows when the select list has columns that are
        neither aggregated nor grouped; Spark rejects that query, so use window aggregates."""
        aggs = [a for a in sel.find_all(exp.AggFunc) if a.find_ancestor(exp.Select) is sel and a.find_ancestor(exp.Window) is None]
        if not aggs:
            return
        group = sel.args.get("group")
        group_exprs = []
        for g in (group.expressions if group else []):
            if isinstance(g, exp.Literal) and not g.is_string and g.this.isdigit():
                item = sel.expressions[int(g.this) - 1]
                group_exprs.append(item.this if isinstance(item, exp.Alias) else item)
            else:
                group_exprs.append(g)
        keys = {g.sql().upper() for g in group_exprs} | {g.name.upper() for g in group_exprs if isinstance(g, exp.Column)}
        aliases = {e.alias.upper(): e.this for e in sel.expressions if isinstance(e, exp.Alias)}
        for g in group_exprs:
            if isinstance(g, exp.Column) and g.name.upper() in aliases:
                keys.add(aliases[g.name.upper()].sql().upper())

        def covered(c: exp.Column) -> bool:
            return c.sql().upper() in keys or c.name.upper() in keys

        remerge = False
        for e in sel.expressions:
            inner = e.this if isinstance(e, exp.Alias) else e
            if inner.sql().upper() in keys:
                continue
            for c in inner.find_all(exp.Column):
                if c.find_ancestor(exp.AggFunc) is None and not covered(c):
                    remerge = True
        if not remerge:
            return
        self.tr.hazard("H-REMERGE", "summary statistics remerged onto detail rows (window aggregates)", self.line)
        partition = [g.copy() for g in group_exprs]
        for a in aggs:
            a.replace(exp.Window(this=a.copy(), partition_by=partition or None))
        sel.set("group", None)
        having = sel.args.get("having")
        if having is not None:
            sel.set("having", None)
            sub = sel.copy()
            outer = exp.select("*").from_(exp.Subquery(this=sub, alias=exp.TableAlias(this=exp.to_identifier("__remerge"))))
            sel.replace(outer.where(having.this.copy()))


def _is_rowwise(node) -> bool:
    return bool(getattr(node, "expressions", None))


def _dbl(e: exp.Expression) -> exp.Expression:
    if isinstance(e, exp.Literal) and not e.is_string:
        return exp.Cast(this=e, to=exp.DataType.build("double"))
    return e


# ---------------------------------------------------------------------------
# PROC SQL step
def compile_proc_sql(step: A.ProcStep, state, code) -> None:
    """Fill ``code`` (StepCode) for a PROC SQL step."""
    tr = SqlTranslator(state, code.hazards)
    L = code.lines.append
    for st in step.stmts:
        kw = st.keyword
        text = st.raw.strip()
        if not text:
            continue
        if kw in ("RESET", "VALIDATE", "DESCRIBE", "TITLE", "FOOTNOTE", "OPTIONS"):
            L(f"# PROC SQL {kw}: no effect on data")
            continue
        if kw == "DROP":
            m = re.match(r"(?is)drop\s+(table|view)\s+(.+)$", text)
            if not m:
                raise Unsupported(f"PROC SQL {text[:40]}", st.line)
            for name in [x.strip() for x in m.group(2).split(",")]:
                L(f"ctx.delete({q(name.lower())})")
                state.catalog.drop(_key(name))
            continue
        if kw == "CREATE":
            m = re.match(r"(?is)create\s+(table|view)\s+([A-Za-z_][\w.]*)\s*(\(.*?\))?\s+as\s+(.*)$", text)
            if not m:
                if re.match(r"(?is)create\s+(unique\s+)?index", text):
                    L("# CREATE INDEX: not needed on Delta (consider Z-ORDER / liquid clustering)")
                    continue
                raise Unsupported(f"PROC SQL {text[:60]}", st.line)
            target = m.group(2)
            if m.group(3):
                raise Unsupported("CREATE TABLE with data set options", st.line)
            body = m.group(4)
            code.writes.append(_key(target))
            _emit_query(tr, body, st.line, code, target=target)
            continue
        if kw == "SELECT":
            _emit_query(tr, text, st.line, code, target=None)
            continue
        if kw == "INSERT":
            m = re.match(r"(?is)insert\s+into\s+([A-Za-z_][\w.]*)\s*(\(([^)]*)\))?\s*(select\b.*)$", text)
            if not m:
                raise Unsupported("INSERT INTO ... VALUES / SET", st.line)
            target = m.group(1)
            cols = [c.strip() for c in m.group(3).split(",")] if m.group(3) else None
            _emit_query(tr, m.group(4), st.line, code, target=None, insert_into=(target, cols))
            continue
        if kw in ("DELETE", "UPDATE", "ALTER"):
            raise Unsupported(f"PROC SQL {kw}", st.line)
        raise Unsupported(f"PROC SQL statement {kw}", st.line)
    if tr.functions:
        code.lines.insert(0, f"S.ensure({', '.join(q(f) for f in sorted(tr.functions))})")


def _key(name: str) -> str:
    if "." in name:
        lib, ds = name.split(".", 1)
        return f"{lib.upper()}.{ds.upper()}"
    return f"WORK.{name.upper()}"


def _emit_query(tr: SqlTranslator, sql: str, line: int, code, target: str | None, insert_into=None) -> None:
    L = code.lines.append
    upper = sql.upper()
    if re.search(r"\bOUTER\s+UNION\b|\bUNION\s+CORR|\bEXCEPT\s+CORR|\bINTERSECT\s+CORR", upper):
        parts = re.split(r"(?is)\bouter\s+union\s+(?:all\s+)?corr(?:esponding)?\b|\bunion\s+(?:all\s+)?corr(?:esponding)?\b", sql)
        if len(parts) < 2 or re.search(r"\bEXCEPT\s+CORR|\bINTERSECT\s+CORR", upper):
            raise Unsupported("set operator with CORR other than (OUTER) UNION CORR", line)
        frames = []
        schema = None
        for part in parts:
            frame, sch = _query_frame(tr, part.strip(), line, code)
            frames.append(frame)
            schema = sch if schema is None else Schema(schema.vars + [v for v in sch.vars if schema.get(v.name) is None])
        all_distinct = not re.search(r"(?is)\bunion\s+all\s+corr|outer\s+union", sql)
        L(f"df = P.set_concat([{', '.join(frames)}])")
        if all_distinct:
            L("df = df.dropDuplicates([c for c in df.columns if c != '_sas_seq'])")
    else:
        frame, schema = _query_frame(tr, sql, line, code)
        L(f"df = {frame}")
    if target is not None:
        ref = A.DatasetRef(target.split(".")[0] if "." in target else None, target.split(".")[-1])
        L(f"ctx.put({q(target.lower())}, df, meta={_meta(schema)})")
        tr.state.catalog.put(ref.key, schema)
    elif insert_into is not None:
        tgt, cols = insert_into
        L(f"df = P.set_concat([ctx.get({q(tgt.lower())}), df])")
        L(f"ctx.put({q(tgt.lower())}, df)")
        code.writes.append(_key(tgt))
    else:
        pass


def _query_frame(tr: SqlTranslator, sql: str, line: int, code) -> tuple[str, Schema]:
    """Emit view registration + spark.sql(...) and return (expression, schema)."""
    prep = tr.prepare(sql, line)
    tree, schema = tr.translate_query(prep, line)
    filters, columns = _param_markers(tr, tree, prep, line)
    if columns:     # a column-list parameter: today's value gives the conversion-time schema, the run decides
        _, schema = tr.translate_query(tr.prepare(_with_defaults(sql, columns, tr.state), line), line)
        schema = Schema(schema.vars, schema.label, known=False)
    text = tr.sql_text(tree)
    for t in prep.tables:
        expr, _ = input_expr(t.ref, tr.state, code.hazards)
        for n in filters:           # users' Spark SQL on the data as Databricks tables show it (single input table)
            expr = f"ctx.param_filter({expr}, {n!r})"
        code.lines.append(f"{expr}.drop('_sas_seq').createOrReplaceTempView({q(t.view)})  # {t.ref.display}")
        code.reads.append(t.ref.key)
    literals = sorted({int(k) for k in re.findall(r"\{_rx_(\d+)", text)})
    has_rt = "{_mv_" in text or bool(literals)
    if has_rt:
        for nm in sorted(set(re.findall(r"\{_mv_([a-z0-9_]+)\}", text))):
            code.lines.append(f"_mv_{nm} = ctx.symget({q(nm.upper())})")
        for k in literals:
            code.lines.append(f"_rx_{k} = {tr.rt_literals[k]}")
    fstr = "f" if has_rt else ""
    sql_lit = fstr + '"""\n' + text.replace('"""', '\\"\\"\\"') + '\n"""'
    if prep.into or prep.into_range:
        code.lines.append(f"_rows = spark.sql({sql_lit}).collect()")
        _emit_into(prep, code)
        code.lines.append("ctx.symput('SQLOBS', str(len(_rows)))")
        return "None", schema
    order = bool(tree.args.get("order")) if isinstance(tree, exp.Select) else False
    seq = ".withColumn('_sas_seq', F.monotonically_increasing_id())"
    if "*" in text:
        seq = ".transform(P.dedupe_columns)" + seq
    return f"spark.sql({sql_lit}){seq}" + ("" if order else ""), schema


def _param_markers(tr: SqlTranslator, tree: exp.Expression, prep: Prepared, line: int) -> tuple[list[str], list[str]]:
    """Parameter-module variables in a query, by position (each use is recorded): a whole WHERE conjunct is a
    filter (removed here, applied to the single input table), an IN item a value list, a select / group by /
    order by item a column list (spliced in as names); anything else is a value."""
    where = tree.args.get("where") if isinstance(tree, exp.Select) else None
    filters: list[str] = []
    columns: list[str] = []
    for col in list(tree.find_all(exp.Column)):
        if not col.name.lower().startswith("__rt__"):
            continue
        key = col.name[len("__rt__"):].upper()
        if key not in tr.state.param_specs:
            continue
        node = col
        while isinstance(node.parent, exp.Paren):
            node = node.parent
        parent = node.parent
        if isinstance(parent, exp.In) and any(x is node for x in parent.expressions):
            tr.state.note_param(key, "values")
        elif where is not None and _top_conjunct(node, where):
            if len(prep.tables) != 1:
                raise Unsupported(f"filter parameter &{key} in a query over several tables: not converted yet", line)
            tr.state.note_param(key, "filter", tr.state.catalog.get(prep.tables[0].ref.key))
            filters.append(key)
            if parent is where:
                tree.set("where", None)
            else:
                parent.replace(parent.right if parent.left is node else parent.left)
        elif isinstance(parent, (exp.Select, exp.Group, exp.Ordered)):
            tr.state.note_param(key, "columns")
            columns.append(key)
        else:
            tr.state.note_param(key, "text")
    return filters, columns


def _top_conjunct(node: exp.Expression, where: exp.Expression) -> bool:
    p = node.parent
    while isinstance(p, (exp.And, exp.Paren)):
        p = p.parent
    return p is where


def _with_defaults(sql: str, names: list[str], state) -> str:
    """The query with today's values of column-list parameters, to derive its conversion-time schema."""
    for n in names:
        sql = sql.replace(f"{RT_OPEN}{n}{RT_CLOSE}", state.param_specs[n].default)
    return sql


def _emit_into(prep: Prepared, code) -> None:
    L = code.lines.append
    trimmed = "True" if prep.into_trimmed else "False"
    if prep.into_range is not None:
        stem, lo, hi = prep.into_range
        L(f"for _k, _r in enumerate(_rows[:{'None' if hi is None else hi - lo + 1}]):")
        L(f"    ctx.symput(f'{stem}{{_k + {lo}}}', P.into_text(_r[0], trimmed=True))")
        return
    if prep.into_sep is not None:
        for j, nm in enumerate(prep.into):
            L(f"ctx.symput({q(nm)}, {q(prep.into_sep)}.join(P.into_text(_r[{j}], trimmed=True) for _r in _rows))")
        return
    for j, nm in enumerate(prep.into):
        L(f"if _rows: ctx.symput({q(nm)}, P.into_text(_rows[0][{j}], trimmed={trimmed}))")


def _meta(schema: Schema) -> str:
    items = [f"{v.name!r}: {v.meta_dict()!r}" for v in schema.vars]
    return "{" + ", ".join(items) + "}" if len(items) <= 3 else "{\n        " + ",\n        ".join(items) + ",\n    }"
