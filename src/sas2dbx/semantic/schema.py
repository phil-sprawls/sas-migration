"""Compile-time dataset schemas (what the converter knows about variables)."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

from ..runtime import pyfuncs


@dataclass
class Var:
    name: str
    type: str = "num"           # 'num' | 'char'
    length: int = 8
    format: str | None = None
    informat: str | None = None
    label: str | None = None

    @property
    def kind(self) -> str:
        if self.type == "char":
            return "char"
        k = pyfuncs.format_kind(self.format) if self.format else "num"
        return k if k in ("date", "datetime", "time") else "num"

    def meta_dict(self) -> dict:
        d = {"type": self.type, "length": self.length}
        if self.format:
            d["format"] = self.format
        if self.informat:
            d["informat"] = self.informat
        if self.label:
            d["label"] = self.label
        return d


@dataclass
class Schema:
    vars: list[Var] = field(default_factory=list)
    label: str | None = None
    known: bool = True          # False when the schema could not be determined statically

    def get(self, name: str) -> Var | None:
        u = name.upper()
        for v in self.vars:
            if v.name.upper() == u:
                return v
        return None

    def names(self) -> list[str]:
        return [v.name for v in self.vars]

    def copy(self) -> "Schema":
        return Schema([Var(**asdict(v)) for v in self.vars], self.label, self.known)


class Catalog:
    """Dataset key ('LIB.DS', WORK for one-level names) -> Schema."""

    def __init__(self) -> None:
        self.datasets: dict[str, Schema] = {}

    @staticmethod
    def key(libref: str | None, name: str) -> str:
        return f"{(libref or 'WORK').upper()}.{name.upper()}"

    def get(self, key: str) -> Schema | None:
        return self.datasets.get(key.upper())

    def put(self, key: str, schema: Schema) -> None:
        self.datasets[key.upper()] = schema

    def drop(self, key: str) -> None:
        self.datasets.pop(key.upper(), None)

    # -------------------------------------------------------------- persistence
    def to_json(self) -> str:
        return json.dumps({k: {"label": s.label, "vars": [asdict(v) for v in s.vars]} for k, s in self.datasets.items()}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Catalog":
        c = cls()
        for k, d in json.loads(text).items():
            c.put(k, Schema([Var(**v) for v in d["vars"]], d.get("label")))
        return c

    def harvest_dir(self, libref: str, directory: str) -> list[str]:
        """Read XPT / sas7bdat headers in a directory into the catalog."""
        import pyreadstat

        added = []
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".xpt", ".sas7bdat"):
                continue
            reader = pyreadstat.read_xport if ext == ".xpt" else pyreadstat.read_sas7bdat
            _, meta = reader(path, metadataonly=True)
            name = os.path.splitext(os.path.basename(path))[0]
            self.put(self.key(libref, name), schema_from_readstat(meta))
            added.append(name.upper())
        return added


    def harvest_tables(self, libref: str, directory: str) -> list[str]:
        """Read the schemas of Parquet/Delta tables in a directory (one sub-directory per table, e.g. from
        ``sas2dbx ingest``): the SAS attributes travel as column metadata in Spark's schema in the Parquet footer."""
        import pyarrow.parquet as pq

        added = []
        for table in sorted(glob.glob(os.path.join(directory, "*"))):
            files = sorted(glob.glob(os.path.join(table, "*.parquet")))
            if not os.path.isdir(table) or not files:
                continue
            md = pq.read_schema(files[0]).metadata or {}
            spark_schema = md.get(b"org.apache.spark.sql.parquet.row.metadata")
            if spark_schema is None:
                continue
            name = os.path.basename(table)
            self.put(self.key(libref, name), schema_from_spark(json.loads(spark_schema)))
            added.append(name.upper())
        return added

    def harvest_uc(self, libref: str, spark, schema: str) -> list[str]:
        """Read the schemas of the tables in a Unity Catalog schema (a ``uc:catalog.schema`` library), through a
        Spark session: column metadata carries the SAS attributes of tables from ``sas2dbx ingest``/``publish``."""
        added = []
        for row in spark.sql(f"SHOW TABLES IN {schema}").collect():
            if row["isTemporary"]:
                continue
            name = row["tableName"]
            self.put(self.key(libref, name), schema_from_spark(spark.table(f"{schema}.{name}").schema.jsonValue()))
            added.append(name.upper())
        return added


def schema_from_spark(schema: dict) -> Schema:
    """A Spark schema in JSON form; SAS attributes come from the column metadata that ingestion writes."""
    from ..runtime.meta import META_KEY

    vars_: list[Var] = []
    for f in schema["fields"]:
        if f["name"].startswith("_sas_"):
            continue
        m = (f.get("metadata") or {}).get(META_KEY) or {}
        typ = m.get("type") or ("char" if f["type"] == "string" else "num")
        fmt = m.get("format") or {"date": "DATE9.", "timestamp": "DATETIME20.", "timestamp_ntz": "DATETIME20."}.get(f["type"])
        vars_.append(Var(f["name"], typ, int(m.get("length") or (200 if typ == "char" else 8)), fmt, None, m.get("label")))
    return Schema(vars_)


def schema_from_readstat(meta) -> Schema:
    from ..runtime.meta import normalize_format

    vars_: list[Var] = []
    types = getattr(meta, "readstat_variable_types", {}) or {}
    fmts = getattr(meta, "original_variable_types", {}) or {}
    widths = getattr(meta, "variable_storage_width", {}) or {}
    labels = getattr(meta, "column_names_to_labels", {}) or {}
    for c in meta.column_names:
        is_char = types.get(c) == "string"
        fmt = normalize_format(fmts.get(c))
        if fmt and (is_char != fmt.startswith("$")):
            fmt = None
        if fmt and is_char and fmt == f"${widths.get(c)}.":
            fmt = None
        vars_.append(Var(c.upper(), "char" if is_char else "num", int(widths.get(c) or 8), fmt, None, labels.get(c) or None))
    return Schema(vars_, getattr(meta, "file_label", None) or None)
