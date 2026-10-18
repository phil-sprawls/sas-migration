"""SasContext: librefs, WORK, read/write boundaries, run-time macro variables, checkpoints.

Generated notebooks pass every dataset through the context so that

* librefs map to Unity Catalog schemas (or paths / SAS files for local runs),
* data entering the program is normalised to SAS space and data leaving it is
  converted back to idiomatic Delta types (DATE/TIMESTAMP_NTZ, NULL for blank),
* the validation harness can observe each step's output (checkpoints).
"""
from __future__ import annotations

import getpass
import glob
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import pyfuncs, sqlfuncs
from .meta import META_KEY, ColumnMeta, normalize_format, schema_meta

HIDDEN_PREFIX = "_sas_"


@dataclass
class LibSpec:
    kind: str                     # memory | uc | parquet | delta | sasfiles
    target: str = ""              # catalog.schema or directory
    readonly: bool = False


@dataclass
class _Entry:
    df: DataFrame
    meta: dict[str, ColumnMeta] = field(default_factory=dict)
    label: str | None = None


class SasContext:
    def __init__(
        self,
        spark: SparkSession,
        libnames: dict[str, str | dict | LibSpec] | None = None,
        *,
        as_of: str | date | None = None,
        char_missing_out: str = "null",       # how blank character values are written: 'null' | 'empty'
        dates_out: str = "date",              # 'date' (DATE/TIMESTAMP_NTZ) | 'sas' (keep SAS numbers)
        persist_order: bool = True,
        checkpoint_hook=None,
        materialize_work: bool = True,        # cut lineage after every step (like SAS writing WORK)
        macro_vars: dict[str, str] | None = None,
        write_outputs: bool = True,           # False: permanent outputs stay in memory (validation runs)
        sas_reader: str = "auto",             # .sas7bdat: auto (spark-sas7bdat if installed) | spark-sas7bdat | pyreadstat
    ) -> None:
        self.spark = spark
        self.libs: dict[str, LibSpec] = {"WORK": LibSpec("memory")}
        for k, v in (libnames or {}).items():
            self.libs[k.upper()] = _libspec(v)
        self.store: dict[str, _Entry] = {}
        self.char_missing_out = char_missing_out
        self.dates_out = dates_out
        self.persist_order = persist_order
        self.checkpoint_hook = checkpoint_hook
        self.materialize_work = materialize_work
        self.write_outputs = write_outputs
        self.mvars: dict[str, str] = {k.upper(): str(v) for k, v in (macro_vars or {}).items()}
        self.params: dict[str, object] = {}    # run-time parameters by name, typed by kind (load_params)
        self.sas_reader = sas_reader
        self.readers: dict[str, tuple[str, str]] = {}   # library/data set -> (file, reader used): evidence
        self.formats: dict[str, dict] = {}
        self.log: list[str] = []
        today = None
        self.started = datetime.now()          # SYSDATE/SYSTIME/TODAY(): when the run started (frozen by as_of)
        if as_of is not None:
            d = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
            today = pyfuncs.date_to_sas(d)
            self.started = datetime(d.year, d.month, d.day)
        sqlfuncs.register_clock(spark, today)

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, spark: SparkSession, config: dict | str) -> "SasContext":
        if isinstance(config, str):
            import yaml

            with open(config) as fh:
                config = yaml.safe_load(fh)
        cfg = dict(config or {})
        return cls(spark, cfg.pop("libnames", {}), **cfg)

    # ------------------------------------------------------------------ names
    @staticmethod
    def split(name: str) -> tuple[str, str]:
        n = name.strip()
        if "." in n:
            lib, ds = n.split(".", 1)
            return lib.upper(), ds.upper()
        return "WORK", n.upper()

    def key(self, name: str) -> str:
        lib, ds = self.split(name)
        return f"{lib}.{ds}"

    # ------------------------------------------------------------------ read
    def get(self, name: str) -> DataFrame:
        """Dataset in SAS space (with _sas_seq)."""
        lib, ds = self.split(name)
        k = f"{lib}.{ds}"
        if k in self.store:
            return self.store[k].df
        spec = self.libs.get(lib)
        if spec is None:
            raise KeyError(f"libref {lib} is not assigned (add it to the context's libnames)")
        if spec.kind == "memory":
            raise KeyError(f"dataset {k} does not exist")
        df, meta, label = self._load(spec, ds)
        df = self._normalize_in(df, meta)
        self.store[k] = _Entry(df, schema_meta(df.schema), label)
        return self.store[k].df

    def meta(self, name: str) -> dict[str, ColumnMeta]:
        self.get(name)
        return self.store[self.key(name)].meta

    def exists(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except KeyError:
            return False

    def _load(self, spec: LibSpec, ds: str):
        if spec.kind == "sasfiles":
            from .sasfile import read_sas_file

            path = _find_file(spec.target, ds, (".sas7bdat", ".xpt"))
            if path is None:
                raise KeyError(f"no SAS dataset {ds} in {spec.target}")
            df, meta, label, reader = read_sas_file(self.spark, path, self.sas_reader)
            self.readers[f"{spec.target}/{ds}"] = (path, reader)
            return df, meta, label
        if spec.kind == "uc":
            df = self.spark.table(f"{spec.target}.{ds.lower()}")
            return df, schema_meta(df.schema), None
        if spec.kind in ("parquet", "delta"):
            path = os.path.join(spec.target, ds.lower())
            df = self.spark.read.format(spec.kind).load(path)
            return df, schema_meta(df.schema), None
        raise KeyError(f"cannot read from library kind {spec.kind}")

    def _normalize_in(self, df: DataFrame, meta: dict[str, ColumnMeta]) -> DataFrame:
        """Delta/Spark types -> SAS space."""
        cols = []
        for f in df.schema.fields:
            name = f.name
            if name == "_sas_seq":
                continue
            m = meta.get(name.upper())
            dt = f.dataType
            c = F.col(f"`{name}`")
            if isinstance(dt, T.StringType):
                new = F.coalesce(F.rtrim(c), F.lit(""))
                m = m or ColumnMeta("char", 200)
            elif isinstance(dt, T.DateType):
                new = F.call_function("sas_from_date", c)
                m = m or ColumnMeta("num", 8, "DATE9.")
                if not m.format:
                    m.format = "DATE9."
            elif isinstance(dt, (T.TimestampType, T.TimestampNTZType)):
                new = F.call_function("sas_from_ts", c.cast("timestamp_ntz"))
                m = m or ColumnMeta("num", 8, "DATETIME20.")
                if not m.format:
                    m.format = "DATETIME20."
            elif isinstance(dt, T.BooleanType):
                new = F.when(c, 1.0).when(~c, 0.0)
                m = m or ColumnMeta("num", 8)
            elif isinstance(dt, T.NumericType):
                d = c.cast("double")
                new = F.when(F.isnan(d), F.lit(None).cast("double")).otherwise(d)
                m = m or ColumnMeta("num", 8)
            else:
                new = c
                m = m or ColumnMeta("char" if isinstance(dt, T.StringType) else "num", 8)
            cols.append(new.alias(name.upper(), metadata={META_KEY: m.to_dict()}))
        sqlfuncs.ensure(self.spark, ["sas_from_date", "sas_from_ts"])
        seq = F.col("_sas_seq") if "_sas_seq" in df.columns else F.monotonically_increasing_id()
        return df.select(*cols, seq.alias("_sas_seq"))

    # ------------------------------------------------------------------ write
    def put(self, name: str, df: DataFrame, meta: dict[str, dict] | None = None, *, label: str | None = None,
            order: list[str] | None = None, step: str | None = None) -> DataFrame:
        """Store a dataset produced by a step.  ``meta`` maps column -> SAS attributes."""
        lib, ds = self.split(name)
        k = f"{lib}.{ds}"
        if lib == "_NULL_" or ds == "_NULL_":
            return df
        df = _with_meta(df, meta or {})
        if order:
            df = df.orderBy(*order).withColumn("_sas_seq", F.monotonically_increasing_id())
        if "_sas_seq" not in df.columns:
            df = df.withColumn("_sas_seq", F.monotonically_increasing_id())
        visible = [c for c in df.columns if not c.startswith(HIDDEN_PREFIX)]
        df = df.select(*visible, "_sas_seq")
        if self.materialize_work:
            df = self.cut(df)
        spec = self.libs.get(lib)
        if spec is None:
            raise KeyError(f"libref {lib} is not assigned")
        self.store[k] = _Entry(df, schema_meta(df.schema), label)
        if spec.kind != "memory" and self.write_outputs:
            self._write(spec, ds, df)
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(step or "", k, df)
        return df

    def _write(self, spec: LibSpec, ds: str, df: DataFrame) -> None:
        if spec.readonly or spec.kind == "sasfiles":
            raise PermissionError(f"library {spec.target} is read-only")
        out = self.to_output(df)
        if spec.kind == "uc":
            out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{spec.target}.{ds.lower()}")
        else:
            out.write.mode("overwrite").format(spec.kind).save(os.path.join(spec.target, ds.lower()))

    def to_output(self, df: DataFrame) -> DataFrame:
        """SAS space -> idiomatic Delta types, driven by SAS formats."""
        meta = schema_meta(df.schema)
        typed = dict(self._typed_columns(df, keep_seq=self.persist_order))
        cols = []
        for f in df.schema.fields:
            if f.name not in typed:
                continue
            m = meta.get(f.name.upper())
            md = {META_KEY: m.to_dict()} if m is not None else {}
            if m is not None and m.label:
                md["comment"] = m.label
            cols.append(typed[f.name].alias(f.name, metadata=md))
        return df.select(*cols)

    def _typed_columns(self, df: DataFrame, keep_seq: bool = False) -> list[tuple[str, "F.Column"]]:
        """Each visible column as written to Delta (blank -> NULL, SAS dates -> DATE/TIMESTAMP by format)."""
        sqlfuncs.ensure(self.spark, ["sas_date"])
        meta = schema_meta(df.schema)
        out = []
        for f in df.schema.fields:
            if f.name.startswith(HIDDEN_PREFIX) and not (f.name == "_sas_seq" and keep_seq):
                continue
            c = F.col(f"`{f.name}`")
            m = meta.get(f.name.upper())
            if m is not None and m.type == "char":
                new = F.when(c == "", F.lit(None).cast("string")).otherwise(c) if self.char_missing_out == "null" else c
            elif m is not None and self.dates_out == "date" and m.kind == "date":
                new = F.call_function("sas_date", c)
            elif m is not None and self.dates_out == "date" and m.kind == "datetime":
                new = _sas_dt_to_ts(c)
            else:
                new = c
            out.append((f.name, new))
        return out

    # ------------------------------------------------------------------ run-time parameters (parameter notebook)
    def load_params(self, spec: dict[str, str], values: dict | None = None) -> None:
        """Load the program's run-time parameters and check every one before any data is read.

        ``spec`` (generated with the code) maps each parameter to its kind: text | number | date | filter (a
        Spark SQL condition) | values (a list of values) | columns (a list of column names).  ``values`` is the
        parameter notebook's PARAMETERS dict (via %run); a parameter missing there comes from values already set
        (validation runs: SasContext(macro_vars=...)).  All problems are reported together and nothing runs.
        """
        given = {str(k).upper(): v for k, v in (values or {}).items()}
        problems = []
        for name, kind in spec.items():
            key = name.upper()
            value = given[key] if key in given else self.mvars.get(key)
            if value is None:
                problems.append(f"{name.lower()}: no value (set it in the parameter notebook)")
                continue
            try:
                self._set_param(key, kind, value)
            except (TypeError, ValueError) as e:
                problems.append(f"{name.lower()}: {e}")
        if problems:
            raise ValueError("parameter check failed, nothing was run:\n  " + "\n  ".join(problems))

    def _set_param(self, key: str, kind: str, value) -> None:
        if kind == "filter":
            text = str(value).strip()
            if text:
                _check_condition(text)
            self.params[key] = text
            self.mvars[key] = text
        elif kind in ("values", "columns"):
            items = _as_list(value)
            if kind == "columns":
                bad = [c for c in items if not isinstance(c, str) or not _IDENT_RE.fullmatch(c)]
                if bad:
                    raise ValueError(f"not column names: {bad}")
                self.mvars[key] = ", ".join(items)
            else:
                self.mvars[key] = ", ".join(_sql_literal(x) for x in items)
            self.params[key] = items
        elif kind == "number":
            if isinstance(value, bool) or not _is_number(value):
                raise ValueError(f"expected a number, got {value!r}")
            self.params[key] = float(value)
            self.mvars[key] = str(value).strip()
        elif kind == "date":
            v = _date_value(value)
            self.params[key] = v
            self.mvars[key] = v.strftime("%d%b%Y").upper()
        else:
            if isinstance(value, (list, tuple, dict, set)):
                raise ValueError(f"expected text, got {type(value).__name__}")
            self.params[key] = str(value)
            self.mvars[key] = str(value)

    def _param(self, name: str):
        key = name.upper()
        if key not in self.params:
            raise KeyError(f"parameter {name.lower()} was not loaded (ctx.load_params runs first)")
        return self.params[key]

    def param_filter(self, df: DataFrame, name: str) -> DataFrame:
        """Rows that satisfy a filter parameter, a Spark SQL condition evaluated on the data as it appears in
        Databricks tables (DATE/TIMESTAMP columns, NULL for blank text); the SAS-space columns are returned."""
        text = self._param(name)
        if not text:
            return df
        typed = self._typed_columns(df)
        _check_condition(text, [n for n, _ in typed], name)
        hidden = [F.col(f"`{c}`").alias(f"__sas2dbx_{i}") for i, c in enumerate(df.columns)]
        view = df.select(*hidden, *[c.alias(n) for n, c in typed]).where(F.expr(text))
        try:
            _ = view.schema      # Spark Connect analyses lazily: report a bad filter now, before any data is read
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{name.lower()}: Spark cannot apply this filter: {str(e).splitlines()[0][:200]}") from e
        return view.select(*[F.col(f"__sas2dbx_{i}").alias(c, metadata=df.schema[c].metadata) for i, c in enumerate(df.columns)])

    def param_values(self, name: str, num: bool = False) -> list:
        """A value-list parameter for IN: floats (None = missing) or right-trimmed text."""
        items = self._param(name)
        if num:
            out = []
            for x in items:
                if x is None or str(x).strip() in ("", "."):
                    out.append(None)
                elif _is_number(x):
                    out.append(float(x))
                else:
                    raise ValueError(f"{name.lower()}: {x!r} is not a number")
            return out
        return [None if x is None else str(x).rstrip() for x in items]

    def param_columns(self, name: str) -> list[str]:
        return list(self._param(name))

    def param_keep(self, df: DataFrame, name: str, also: list[str] | tuple = ()) -> DataFrame:
        """KEEP with a column-list parameter: SAS order (the data's), unknown names are an error, not ignored."""
        want = {c.upper() for c in self.param_columns(name)} | {c.upper() for c in also}
        present = {c.upper() for c in df.columns}
        missing = sorted(want - present)
        if missing:
            raise ValueError(f"{name.lower()}: columns not in the data: {missing}; "
                             f"available: {sorted(c for c in df.columns if not c.startswith(HIDDEN_PREFIX))}")
        return df.select(*[c for c in df.columns if c.upper() in want or c.startswith(HIDDEN_PREFIX)])

    def param_drop(self, df: DataFrame, name: str) -> DataFrame:
        drop = {c.upper() for c in self.param_columns(name)}
        missing = sorted(drop - {c.upper() for c in df.columns})
        if missing:
            raise ValueError(f"{name.lower()}: columns not in the data: {missing}")
        return df.select(*[c for c in df.columns if c.upper() not in drop])

    def delete(self, name: str) -> None:
        self.store.pop(self.key(name), None)

    def rename(self, old: str, new: str) -> None:
        k = self.key(old)
        if k in self.store:
            lib, _ = self.split(old)
            self.store[f"{lib}.{self.split(new)[1]}"] = self.store.pop(k)

    def names(self, lib: str = "WORK") -> list[str]:
        p = lib.upper() + "."
        return sorted(k[len(p):] for k in self.store if k.startswith(p))

    # ------------------------------------------------------------------ macro variables
    @staticmethod
    def cut(df: DataFrame) -> DataFrame:
        """Materialise and drop the lineage (like SAS writing WORK): keeps Spark plans shallow - thousands of
        chained projections in one DATA step otherwise overflow the JVM stack during analysis."""
        try:
            return df.localCheckpoint(eager=True)
        except Exception as e:  # pragma: no cover - not supported on every compute type
            try:
                return df.cache()
            except Exception:   # serverless has no cache(): report the checkpoint's error (often the plan's own)
                raise e from None

    def symput(self, name: str, value) -> None:
        self.mvars[name.upper()] = "" if value is None else str(value)

    def symget(self, name: str) -> str:
        key = name.upper()
        if key in self.mvars:
            return self.mvars[key]
        value = self._automatic(key)
        if value is None:
            raise KeyError(f"macro variable &{name} was never assigned (a run-time parameter that was not supplied?)")
        return value

    def _automatic(self, key: str) -> str | None:
        """Values the SAS session knows only when it runs (macro.processor.RUNTIME_AUTOMATIC, %SYSGET, clock %SYSFUNCs)."""
        now = self.started
        if key == "SYSDATE9":
            return now.strftime("%d%b%Y").upper()
        if key == "SYSDATE":
            return now.strftime("%d%b%y").upper()
        if key == "SYSDAY":
            return now.strftime("%A")
        if key == "SYSTIME":
            return now.strftime("%H:%M")
        if key == "SYSPARM":
            return ""                          # SAS default when the run passes no -SYSPARM
        if key == "SYSUSERID":
            try:
                return str(self.spark.sql("SELECT current_user()").first()[0])
            except Exception:  # noqa: BLE001 - local Spark without current_user()
                return getpass.getuser()
        if key == "SYSJOBID":
            return str(os.getpid())
        if key == "SYSHOSTNAME":
            return socket.gethostname()
        if key.startswith("SYSGET_"):
            var = key[len("SYSGET_"):]
            hit = next((k for k in os.environ if k.upper() == var), None)
            return os.environ.get(var, os.environ.get(hit, "") if hit else "")
        if key.startswith("SYSFUNC_"):
            fn, _, fmt = key[len("SYSFUNC_"):].partition("__")
            seconds = now.hour * 3600 + now.minute * 60 + now.second
            value = {"TODAY": pyfuncs.date_to_sas(now.date()), "DATE": pyfuncs.date_to_sas(now.date()),
                     "DATETIME": pyfuncs.datetime_to_sas(now), "TIME": float(seconds)}.get(fn)
            if value is None:
                return None
            spec = (fmt.replace("_", ".") if "_" in fmt else fmt + ".") if fmt else "BEST12."
            return pyfuncs.put(value, spec).strip()
        return None

    def symget_num(self, name: str) -> float | None:
        v = self.symget(name).strip()
        if v in ("", "."):
            return None
        return float(v)


def _libspec(v) -> LibSpec:
    if isinstance(v, LibSpec):
        return v
    if isinstance(v, dict):
        return LibSpec(v.get("kind") or v.get("type") or "memory", v.get("target") or v.get("path") or "",
                       bool(v.get("readonly", False)))
    s = str(v)
    if s.lower() == "memory":
        return LibSpec("memory")
    for prefix in ("parquet:", "delta:", "sasfiles:", "uc:"):
        if s.lower().startswith(prefix):
            return LibSpec(prefix[:-1], s[len(prefix):])
    if os.path.isdir(s):
        return LibSpec("sasfiles", s)
    return LibSpec("uc", s)


def _find_file(directory: str, ds: str, exts) -> str | None:
    for ext in exts:
        for p in glob.glob(os.path.join(directory, "*" + ext)):
            if os.path.splitext(os.path.basename(p))[0].upper() == ds.upper():
                return p
    return None


def _with_meta(df: DataFrame, meta: dict[str, dict]) -> DataFrame:
    if not meta:
        return df
    upper = {k.upper(): v for k, v in meta.items()}
    cols = []
    for f in df.schema.fields:
        m = upper.get(f.name.upper())
        if m is None:
            cols.append(F.col(f"`{f.name}`"))
            continue
        cm = ColumnMeta(**{**{"type": "num", "length": 8}, **m}) if isinstance(m, dict) else m
        if cm.format:
            cm.format = normalize_format(cm.format)
        cols.append(F.col(f"`{f.name}`").alias(f.name, metadata={META_KEY: cm.to_dict()}))
    return df.select(*cols)


def _sas_dt_to_ts(c):
    secs = F.pmod(c, F.lit(86400.0))
    days = F.floor(c / 86400)
    d = F.call_function("sas_date", days.cast("double"))
    return F.when(c.isNull(), F.lit(None).cast("timestamp_ntz")).otherwise(
        F.call_function(
            "make_timestamp_ntz",
            F.year(d), F.month(d), F.dayofmonth(d),
            F.floor(secs / 3600).cast("int"), F.floor(F.pmod(secs, F.lit(3600.0)) / 60).cast("int"),
            F.pmod(secs, F.lit(60.0)).cast("decimal(18,6)"),
        )
    )


# ---------------------------------------------------------------------------- run-time parameter helpers
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ITEM_RE = re.compile(r"'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\"|([^\s,]+)")
_DENIED_RE = re.compile(r"\b(reflect|try_reflect|java_method)\s*\(", re.I)


def _as_list(value) -> list:
    """A list parameter: a Python list, or text written the SAS or SQL way ('EAST','WEST' / id region amount)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        out = []
        for m in _ITEM_RE.finditer(value):
            if m.group(1) is not None:
                out.append(m.group(1).replace("''", "'"))
            elif m.group(2) is not None:
                out.append(m.group(2).replace('""', '"'))
            else:
                out.append(m.group(3))
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value]
    raise ValueError(f"expected a list, got {type(value).__name__}")


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def _sql_literal(x) -> str:
    """A list value as a Spark SQL literal (data, never code)."""
    if x is None:
        return "NULL"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return repr(x)
    return "'" + str(x).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _date_value(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    v = pyfuncs.input_value(text, "DATE9.")
    if v is None:
        raise ValueError(f"expected a date such as 2024-01-31 or 31JAN2024, got {value!r}")
    return date(1960, 1, 1) + timedelta(days=int(v))


def _check_condition(text: str, columns: list[str] | None = None, name: str = "") -> None:
    """A filter parameter must be one Spark SQL condition over the data's columns; checked before any data is read."""
    if _DENIED_RE.search(text):
        raise ValueError("calls a function that is not allowed in a filter (reflect / java_method)")
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:  # pragma: no cover - runtime-only install: Spark's own analysis reports the problem
        return
    try:
        tree = sqlglot.parse_one(text, read="spark")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"not a valid Spark SQL condition: {str(e).splitlines()[0][:160]}") from e
    if tree is None or tree.find(exp.Select) is not None or isinstance(tree, (exp.Command, exp.DDL, exp.DML)):
        raise ValueError("a filter must be a single condition, not a query or statement")
    if columns is not None:
        known = {c.upper() for c in columns}
        unknown = sorted({c.name for c in tree.find_all(exp.Column) if c.name.upper() not in known})
        if unknown:
            raise ValueError(f"{name.lower()}: refers to columns not in the data: {unknown}; available: {sorted(columns)}")
