"""Macro quoting: masked characters (SAS 'delta characters').

Quoting functions (%STR, %NRSTR, %BQUOTE, %SUPERQ ...) replace special characters
with private-use code points so the scanner and %EVAL treat them as plain text.
They are restored by %UNQUOTE and on final output.
"""
from __future__ import annotations

_MASKABLE = ";,()+-*/<>=^~|&%'\"# \t\n¬"
_MASK_BASE = 0xE000
MASK = {c: chr(_MASK_BASE + i) for i, c in enumerate(_MASKABLE)}
UNMASK = {v: k for k, v in MASK.items()}
_UNMASK_TABLE = str.maketrans(UNMASK)
_MASK_TABLE = str.maketrans(MASK)


def mask(s: str) -> str:
    return s.translate(_MASK_TABLE)


def unmask(s: str) -> str:
    return s.translate(_UNMASK_TABLE)


def is_masked_char(c: str) -> bool:
    return c in UNMASK
