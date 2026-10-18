"""SAS column metadata (type, length, format, informat, label) carried in Spark field metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import pyfuncs

META_KEY = "sas"


@dataclass
class ColumnMeta:
    type: str = "num"            # 'num' | 'char'
    length: int = 8
    format: str | None = None
    informat: str | None = None
    label: str | None = None

    @property
    def kind(self) -> str:
        """'char' | 'num' | 'date' | 'datetime' | 'time' (by format)."""
        if self.type == "char":
            return "char"
        k = pyfuncs.format_kind(self.format) if self.format else "num"
        return k if k in ("date", "datetime", "time") else "num"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ColumnMeta | None":
        if not d:
            return None
        return cls(**{k: d.get(k) for k in ("type", "length", "format", "informat", "label") if k in d})


def normalize_format(fmt: str | None) -> str | None:
    """'DATE9' -> 'DATE9.', 'BEST12' -> 'BEST12.', '$20' -> '$20.', '' -> None."""
    if fmt is None:
        return None
    f = str(fmt).strip()
    if not f or f.upper() in ("$", "."):
        return None
    if "." not in f:
        f = f + "."
    return f.upper()


def field_meta(field) -> ColumnMeta | None:
    md = getattr(field, "metadata", None) or {}
    return ColumnMeta.from_dict(md.get(META_KEY))


def schema_meta(schema) -> dict[str, ColumnMeta]:
    out = {}
    for f in schema.fields:
        m = field_meta(f)
        if m is not None:
            out[f.name.upper()] = m
    return out
