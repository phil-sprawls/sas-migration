"""Pure-Python reference implementations of SAS functions, formats and informats.

These are the *specification* of SAS semantics inside sas2dbx:

* the macro processor uses them for ``%SYSFUNC``;
* the Spark SQL runtime (``sas2dbx.runtime.sqlfuncs``) is differential-tested
  against them on random and edge-case inputs;
* the Tier-C PDV executor (roadmap) runs them row by row.

Conventions: numeric missing is ``None`` (NaN is treated as missing); character
values are Python ``str`` and SAS "blank" is ``''`` (trailing blanks are not
significant except where noted).  SAS dates are ``float`` days since
1960-01-01, datetimes are seconds since 1960-01-01T00:00:00, times are seconds.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

SAS_EPOCH = date(1960, 1, 1)
_EPOCH_ORD = SAS_EPOCH.toordinal()
SECONDS_PER_DAY = 86400.0
YEARCUTOFF = 1926

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SCAN_DEFAULT_DELIMS = " !$%&()*+,-./;<^|"
PROPCASE_DEFAULT_DELIMS = " /-(.\t"


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _num(x):
    return None if is_missing(x) else float(x)


# ---------------------------------------------------------------------------
# dates
def date_to_sas(d: date) -> float:
    return float(d.toordinal() - _EPOCH_ORD)


def sas_to_date(n) -> date | None:
    if is_missing(n):
        return None
    try:
        return date.fromordinal(int(math.floor(n)) + _EPOCH_ORD)
    except (ValueError, OverflowError):
        return None


def datetime_to_sas(dt: datetime) -> float:
    d = date_to_sas(dt.date())
    return d * SECONDS_PER_DAY + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def sas_to_datetime(n) -> datetime | None:
    if is_missing(n):
        return None
    days = math.floor(n / SECONDS_PER_DAY)
    d = sas_to_date(days)
    if d is None:
        return None
    secs = n - days * SECONDS_PER_DAY
    return datetime(d.year, d.month, d.day) + timedelta(seconds=secs)


def _days_in_month(y: int, m: int) -> int:
    if m == 12:
        return 31
    return (date(y, m + 1, 1) - date(y, m, 1)).days


def mdy(m, d, y):
    m, d, y = _num(m), _num(d), _num(y)
    if m is None or d is None or y is None:
        return None
    m, d, y = int(m), int(d), int(y)
    if 0 <= y < 100:
        y = _expand_year(y)
    if not (1 <= m <= 12) or y < 1 or y > 9999:
        return None
    if not (1 <= d <= _days_in_month(y, m)):
        return None
    return date_to_sas(date(y, m, d))


def _expand_year(yy: int, cutoff: int = YEARCUTOFF) -> int:
    century = cutoff - cutoff % 100
    y = century + yy
    if y < cutoff:
        y += 100
    return y


def year(n):
    d = sas_to_date(n)
    return None if d is None else float(d.year)


def month(n):
    d = sas_to_date(n)
    return None if d is None else float(d.month)


def day(n):
    d = sas_to_date(n)
    return None if d is None else float(d.day)


def qtr(n):
    d = sas_to_date(n)
    return None if d is None else float((d.month - 1) // 3 + 1)


def weekday(n):
    """1 = Sunday ... 7 = Saturday."""
    d = sas_to_date(n)
    return None if d is None else float((d.isoweekday() % 7) + 1)


def datepart(dt):
    dt = _num(dt)
    return None if dt is None else float(math.floor(dt / SECONDS_PER_DAY))


def timepart(dt):
    dt = _num(dt)
    return None if dt is None else float(dt - math.floor(dt / SECONDS_PER_DAY) * SECONDS_PER_DAY)


def dhms(d, h, mi, s):
    vals = [_num(v) for v in (d, h, mi, s)]
    if any(v is None for v in vals):
        return None
    d, h, mi, s = vals
    return d * SECONDS_PER_DAY + h * 3600 + mi * 60 + s


def hms(h, mi, s):
    vals = [_num(v) for v in (h, mi, s)]
    if any(v is None for v in vals):
        return None
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


def hour(t):
    t = _num(t)
    if t is None:
        return None
    return float(int((t % SECONDS_PER_DAY) // 3600)) if t >= 0 else float(int(t // 3600))


def minute(t):
    t = _num(t)
    return None if t is None else float(int((t % 3600) // 60))


def second(t):
    t = _num(t)
    return None if t is None else float(t % 60)


def today(now: datetime | None = None):
    return date_to_sas((now or datetime.now()).date())


# --- intervals --------------------------------------------------------------
_INTERVAL_RE = re.compile(r"^(DT)?([A-Z]+?)(\d*)(?:\.(\d+))?$")
_DATE_INTERVALS = {"DAY", "WEEK", "TENDAY", "SEMIMONTH", "MONTH", "QTR", "SEMIYEAR", "YEAR", "WEEKDAY"}
_TIME_INTERVALS = {"SECOND", "MINUTE", "HOUR"}


def parse_interval(interval: str):
    s = interval.strip().upper()
    m = _INTERVAL_RE.match(s)
    if not m:
        raise ValueError(f"invalid interval {interval!r}")
    dt, base, mult, shift = m.groups()
    if base not in _DATE_INTERVALS | _TIME_INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}")
    return bool(dt), base, int(mult) if mult else 1, int(shift) if shift else 1


def _period_index(base: str, d: date) -> int:
    """Index of the base period containing d (period 0 contains 1960-01-01 for months etc.)."""
    if base == "DAY":
        return d.toordinal() - _EPOCH_ORD
    if base == "WEEK":
        # SAS weeks start on Sunday; 1959-12-27 (SAS -5) is a Sunday.
        return (d.toordinal() - _EPOCH_ORD + 5) // 7
    if base == "MONTH":
        return d.year * 12 + d.month - 1
    if base == "QTR":
        return d.year * 4 + (d.month - 1) // 3
    if base == "SEMIYEAR":
        return d.year * 2 + (d.month - 1) // 6
    if base == "YEAR":
        return d.year
    if base == "SEMIMONTH":
        return (d.year * 12 + d.month - 1) * 2 + (0 if d.day <= 15 else 1)
    if base == "TENDAY":
        return (d.year * 12 + d.month - 1) * 3 + min((d.day - 1) // 10, 2)
    raise ValueError(base)


def _period_start(base: str, idx: int) -> date:
    if base == "DAY":
        return date.fromordinal(idx + _EPOCH_ORD)
    if base == "WEEK":
        return date.fromordinal(idx * 7 - 5 + _EPOCH_ORD)
    if base == "MONTH":
        return date(idx // 12, idx % 12 + 1, 1)
    if base == "QTR":
        return date(idx // 4, (idx % 4) * 3 + 1, 1)
    if base == "SEMIYEAR":
        return date(idx // 2, (idx % 2) * 6 + 1, 1)
    if base == "YEAR":
        return date(idx, 1, 1)
    if base == "SEMIMONTH":
        mi, half = divmod(idx, 2)
        return date(mi // 12, mi % 12 + 1, 1 if half == 0 else 16)
    if base == "TENDAY":
        mi, part = divmod(idx, 3)
        return date(mi // 12, mi % 12 + 1, 1 + 10 * part)
    raise ValueError(base)


_MONTH_BASED = {"MONTH": 1, "QTR": 3, "SEMIYEAR": 6, "YEAR": 12}


def _shift_months(base: str) -> int:
    return _MONTH_BASED.get(base, 0)


def _multi_index(base: str, mult: int, shift: int, d: date) -> int:
    """Index of the (possibly multiplied/shifted) interval containing d.

    Month-based intervals are aligned to 1960-01 (+ shift-1 months), as SAS does;
    WEEK.s shifts the start day (1 = Sunday)."""
    if base in _MONTH_BASED:
        span = _MONTH_BASED[base] * mult
        months = d.year * 12 + d.month - 1 - (1960 * 12 + shift - 1)
        return months // span
    if base == "WEEK":
        return (d.toordinal() - _EPOCH_ORD + 5 - (shift - 1)) // (7 * mult)
    if base == "DAY":
        return (d.toordinal() - _EPOCH_ORD) // mult
    return _period_index(base, d) // mult


def _multi_start(base: str, mult: int, shift: int, idx: int) -> date:
    if base in _MONTH_BASED:
        months = 1960 * 12 + shift - 1 + idx * _MONTH_BASED[base] * mult
        return date(months // 12, months % 12 + 1, 1)
    if base == "WEEK":
        return date.fromordinal(idx * 7 * mult - 5 + (shift - 1) + _EPOCH_ORD)
    if base == "DAY":
        return date.fromordinal(idx * mult + _EPOCH_ORD)
    return _period_start(base, idx * mult)


def _interval_end(base, mult, shift, idx) -> date:
    return _multi_start(base, mult, shift, idx + 1) - timedelta(days=1)


def intnx(interval: str, start, n, alignment: str = "B"):
    start, n = _num(start), _num(n)
    if start is None or n is None:
        return None
    is_dt, base, mult, shift = parse_interval(interval)
    n = int(math.trunc(n))
    align = (alignment or "B").strip().upper()[:1]
    if base in _TIME_INTERVALS:
        size = {"SECOND": 1, "MINUTE": 60, "HOUR": 3600}[base] * mult
        b = math.floor(start / size) + n
        if align == "E":
            return float((b + 1) * size - 1)
        if align == "M":
            return float(b * size + size / 2)
        return float(b * size)
    if is_dt:
        d = sas_to_date(math.floor(start / SECONDS_PER_DAY))
    else:
        d = sas_to_date(start)
    if d is None:
        return None
    if base == "WEEKDAY":
        res = _intnx_weekday(d, n)
    else:
        idx = _multi_index(base, mult, shift, d) + n
        b = _multi_start(base, mult, shift, idx)
        e = _interval_end(base, mult, shift, idx)
        if align == "E":
            res = e
        elif align == "M":
            res = b + timedelta(days=(e - b).days // 2)
        elif align == "S":
            if base in ("MONTH", "QTR", "SEMIYEAR", "YEAR") and mult == 1 and shift == 1:
                months_total = (d.year * 12 + d.month - 1) + n * _shift_months(base)
                y, m = divmod(months_total, 12)
                res = date(y, m + 1, min(d.day, _days_in_month(y, m + 1)))
            elif base == "DAY":
                res = d + timedelta(days=n * mult)
            elif base == "WEEK":
                res = d + timedelta(days=7 * n * mult)
            else:
                offset = (d - _multi_start(base, mult, shift, idx - n)).days
                res = min(b + timedelta(days=offset), e)
        else:
            res = b
    out = date_to_sas(res)
    return out * SECONDS_PER_DAY if is_dt else out


def _intnx_weekday(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    cur = d
    # SAS: a weekend start counts as the preceding Friday for positive n.
    count = abs(n)
    if count == 0:
        while cur.isoweekday() in (6, 7):
            cur -= timedelta(days=1)
        return cur
    while count:
        cur += timedelta(days=step)
        if cur.isoweekday() not in (6, 7):
            count -= 1
    return cur


def intck(interval: str, start, end, method: str = "DISCRETE"):
    start, end = _num(start), _num(end)
    if start is None or end is None:
        return None
    is_dt, base, mult, shift = parse_interval(interval)
    meth = (method or "D").strip().upper()[:1]
    if base in _TIME_INTERVALS:
        size = {"SECOND": 1, "MINUTE": 60, "HOUR": 3600}[base] * mult
        if meth == "C":
            return float(math.trunc((end - start) / size))
        return float(math.floor(end / size) - math.floor(start / size))
    if is_dt:
        start = math.floor(start / SECONDS_PER_DAY)
        end = math.floor(end / SECONDS_PER_DAY)
    d1, d2 = sas_to_date(start), sas_to_date(end)
    if d1 is None or d2 is None:
        return None
    if base == "WEEKDAY":
        return float(_weekdays_between(d1, d2))
    if meth == "C":
        # continuous: count whole intervals measured from the start date
        if d2 >= d1:
            k = 0
            while True:
                nxt = _continuous_add(base, mult, d1, k + 1)
                if nxt > d2:
                    return float(k)
                k += 1
        k = 0
        while True:
            prv = _continuous_add(base, mult, d1, k - 1)
            if prv < d2:
                return float(k)
            k -= 1
    return float(_multi_index(base, mult, shift, d2) - _multi_index(base, mult, shift, d1))


def _continuous_add(base, mult, d: date, k: int) -> date:
    if base == "DAY":
        return d + timedelta(days=k * mult)
    if base == "WEEK":
        return d + timedelta(days=7 * k * mult)
    months = {"MONTH": 1, "QTR": 3, "SEMIYEAR": 6, "YEAR": 12}.get(base)
    if months is None:
        raise ValueError(f"continuous INTCK not supported for {base}")
    total = d.year * 12 + d.month - 1 + k * months * mult
    y, m = divmod(total, 12)
    return date(y, m + 1, min(d.day, _days_in_month(y, m + 1)))


def _weekdays_between(d1: date, d2: date) -> int:
    sign = 1
    if d2 < d1:
        d1, d2, sign = d2, d1, -1
    count = 0
    cur = d1
    while cur < d2:
        cur += timedelta(days=1)
        if cur.isoweekday() not in (6, 7):
            count += 1
    return sign * count


def yrdif(start, end, basis: str = "ACT/ACT"):
    start, end = _num(start), _num(end)
    if start is None or end is None:
        return None
    b = basis.strip().upper()
    d1, d2 = sas_to_date(start), sas_to_date(end)
    if b == "AGE":
        # age in years: whole years plus fraction of the current year (SAS 'AGE' basis)
        sign = 1
        if d2 < d1:
            d1, d2, sign = d2, d1, -1
        years = d2.year - d1.year
        try:
            anniv = d1.replace(year=d1.year + years)
        except ValueError:  # Feb 29
            anniv = date(d1.year + years, 3, 1)
        if anniv > d2:
            years -= 1
            try:
                anniv = d1.replace(year=d1.year + years)
            except ValueError:
                anniv = date(d1.year + years, 3, 1)
        try:
            nxt = anniv.replace(year=anniv.year + 1)
        except ValueError:
            nxt = date(anniv.year + 1, 3, 1)
        frac = (d2 - anniv).days / (nxt - anniv).days
        return sign * (years + frac)
    if b in ("ACT/ACT", "ACTUAL"):
        if d1.year == d2.year:
            return (d2 - d1).days / (366 if _leap(d1.year) else 365)
        total = (date(d1.year + 1, 1, 1) - d1).days / (366 if _leap(d1.year) else 365)
        total += d2.year - d1.year - 1
        total += (d2 - date(d2.year, 1, 1)).days / (366 if _leap(d2.year) else 365)
        return total
    if b == "ACT/360":
        return (d2 - d1).days / 360.0
    if b == "ACT/365":
        return (d2 - d1).days / 365.0
    if b == "30/360":
        dd1 = min(d1.day, 30)
        dd2 = d2.day if not (d2.day == 31 and dd1 >= 30) else 30
        return ((d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (dd2 - dd1)) / 360.0
    raise ValueError(f"unsupported YRDIF basis {basis!r}")


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


# ---------------------------------------------------------------------------
# numeric functions
_FUZZ = 1e-12


def sas_round(x, unit=1.0):
    x, unit = _num(x), _num(unit)
    if x is None or unit is None:
        return None
    if unit <= 0:
        return None
    try:
        dx = Decimal(repr(x))
        du = Decimal(repr(unit))
        q = (dx / du).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return float(q * du)
    except InvalidOperation:
        return None


def rounde(x, unit=1.0):
    x, unit = _num(x), _num(unit)
    if x is None or unit is None or unit <= 0:
        return None
    dx, du = Decimal(repr(x)), Decimal(repr(unit))
    return float((dx / du).quantize(Decimal(1), rounding=ROUND_HALF_EVEN) * du)


def sas_int(x):
    x = _num(x)
    if x is None:
        return None
    r = round(x)
    if abs(x - r) <= _FUZZ * max(1.0, abs(x)):
        return float(r)
    return float(math.trunc(x))


def sas_floor(x):
    x = _num(x)
    if x is None:
        return None
    r = round(x)
    if abs(x - r) <= _FUZZ * max(1.0, abs(x)):
        return float(r)
    return float(math.floor(x))


def sas_ceil(x):
    x = _num(x)
    if x is None:
        return None
    r = round(x)
    if abs(x - r) <= _FUZZ * max(1.0, abs(x)):
        return float(r)
    return float(math.ceil(x))


def sas_mod(a, b):
    a, b = _num(a), _num(b)
    if a is None or b is None or b == 0:
        return None
    r = math.fmod(a, b)
    if abs(r) <= _FUZZ * max(1.0, abs(b)) or abs(abs(r) - abs(b)) <= _FUZZ * max(1.0, abs(b)):
        return 0.0
    return r


def _args(vals):
    return [float(v) for v in vals if not is_missing(v)]


def sas_sum(*vals):
    xs = _args(vals)
    return math.fsum(xs) if xs else None


def sas_mean(*vals):
    xs = _args(vals)
    return math.fsum(xs) / len(xs) if xs else None


def sas_min(*vals):
    xs = _args(vals)
    return min(xs) if xs else None


def sas_max(*vals):
    xs = _args(vals)
    return max(xs) if xs else None


def sas_n(*vals):
    return float(len(_args(vals)))


def sas_nmiss(*vals):
    return float(sum(1 for v in vals if is_missing(v)))


def sas_cmiss(*vals):
    return float(sum(1 for v in vals if is_missing(v) or (isinstance(v, str) and v.strip() == "")))


def sas_std(*vals):
    xs = _args(vals)
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sas_var(*vals):
    s = sas_std(*vals)
    return None if s is None else s * s


def sas_range(*vals):
    xs = _args(vals)
    return max(xs) - min(xs) if xs else None


def sas_median(*vals):
    xs = sorted(_args(vals))
    if not xs:
        return None
    k = len(xs)
    return xs[k // 2] if k % 2 else (xs[k // 2 - 1] + xs[k // 2]) / 2


def coalesce(*vals):
    for v in vals:
        if not is_missing(v):
            return float(v)
    return None


def coalescec(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


# ---------------------------------------------------------------------------
# character functions
def _c(s) -> str:
    return "" if s is None else str(s)


def scan(s, n, delims=None, modifiers: str = ""):
    s = _c(s)
    if is_missing(n):
        return ""
    n = int(n)
    if n == 0:
        return ""
    d = SCAN_DEFAULT_DELIMS if delims is None or delims == "" else delims
    mods = (modifiers or "").lower()
    if "a" in mods:
        d += "".join(chr(c) for c in range(65, 91)) + "".join(chr(c) for c in range(97, 123))
    if "d" in mods:
        d += "0123456789"
    if "s" in mods:
        d += " \t\n\r\f\v"
    if "m" in mods:
        words = re.split("[" + re.escape(d) + "]", s)
    else:
        words = [w for w in re.split("[" + re.escape(d) + "]+", s) if w != ""]
    if "m" not in mods:
        pass
    if abs(n) > len(words):
        return ""
    return words[n - 1] if n > 0 else words[n]


def countw(s, delims=None, modifiers: str = ""):
    s = _c(s)
    d = SCAN_DEFAULT_DELIMS if not delims else delims
    if "m" in (modifiers or "").lower():
        return float(len(re.split("[" + re.escape(d) + "]", s)))
    return float(len([w for w in re.split("[" + re.escape(d) + "]+", s) if w != ""]))


def _char_class(chars: str, mods: str) -> set[str]:
    cls = set(chars or "")
    m = mods.lower()
    if "a" in m:
        cls |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    if "d" in m:
        cls |= set("0123456789")
    if "s" in m:
        cls |= set(" \t\n\r\f\v")
    if "p" in m:
        cls |= set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    if "l" in m:
        cls |= set("abcdefghijklmnopqrstuvwxyz")
    if "u" in m:
        cls |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if "n" in m:
        cls |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
    if "f" in m:
        cls |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
    if "h" in m:
        cls |= {"\t"}
    if "c" in m:
        cls |= {chr(i) for i in range(32)} | {chr(127)}
    if "x" in m:
        cls |= set("0123456789ABCDEFabcdef")
    if "i" in m:
        cls |= {c.upper() for c in cls} | {c.lower() for c in cls}
    return cls


def compress(s, chars=None, modifiers: str = ""):
    s = _c(s)
    mods = modifiers or ""
    if chars is None and not mods:
        return s.replace(" ", "")
    cls = _char_class(chars if chars is not None else "", mods)
    if chars is None and not (set(mods.lower()) - set("kit")):
        cls |= {" "}
    if "t" in mods.lower():
        s = s.rstrip()
    if "k" in mods.lower():
        return "".join(ch for ch in s if ch in cls)
    return "".join(ch for ch in s if ch not in cls)


def compbl(s):
    return re.sub(" {2,}", " ", _c(s))


def tranwrd(s, target, replacement):
    s, target, replacement = _c(s), _c(target), _c(replacement)
    if target == "":
        target = " "
    if replacement == "":
        replacement = " "
    return s.replace(target, replacement)


def transtrn(s, target, replacement):
    s, target, replacement = _c(s), _c(target), _c(replacement)
    if target == "":
        return s
    return s.replace(target, replacement)


def translate(s, *pairs):
    s = _c(s)
    table: dict[str, str] = {}
    for k in range(0, len(pairs) - 1, 2):
        to, frm = _c(pairs[k]), _c(pairs[k + 1])
        for idx, ch in enumerate(frm):
            if ch not in table:
                table[ch] = to[idx] if idx < len(to) else " "
    return "".join(table.get(ch, ch) for ch in s)


def substr(s, pos, length=None):
    s = _c(s)
    if is_missing(pos):
        return ""
    p = int(pos)
    if p < 1:
        return ""
    if length is None:
        return s[p - 1:]
    if is_missing(length):
        return ""
    n = int(length)
    if n <= 0:
        return ""
    return s[p - 1:p - 1 + n]


def substrn(s, pos, length=None):
    s = _c(s)
    if is_missing(pos):
        return ""
    p = int(pos)
    if length is None:
        start = max(p, 1)
        return s[start - 1:]
    n = int(length)
    end = p + n
    start = max(p, 1)
    if end <= start:
        return ""
    return s[start - 1:end - 1]


def index(s, t):
    s, t = _c(s), _c(t)
    if t == "":
        t = " "
    return float(s.find(t) + 1)


def indexc(s, *chars):
    s = _c(s)
    pool = set("".join(_c(c) for c in chars))
    for i, ch in enumerate(s):
        if ch in pool:
            return float(i + 1)
    return 0.0


def indexw(s, word, delims=" "):
    s, word = _c(s), _c(word).strip()
    if not word:
        return 0.0
    pattern = r"(?:^|[" + re.escape(delims) + r"])(" + re.escape(word) + r")(?=$|[" + re.escape(delims) + r"])"
    m = re.search(pattern, s)
    return float(m.start(1) + 1) if m else 0.0


def find(s, sub, modifiers="", start=1):
    s, sub = _c(s), _c(sub)
    mods = (modifiers or "").lower() if isinstance(modifiers, str) else ""
    if isinstance(modifiers, (int, float)):
        start, mods = modifiers, ""
    if "t" in mods:
        s, sub = s.rstrip(), sub.rstrip()
    if "i" in mods:
        s, sub = s.lower(), sub.lower()
    st = int(start) if not is_missing(start) else 1
    if st >= 0:
        return float(s.find(sub, max(st - 1, 0)) + 1)
    pos = s.rfind(sub, 0, len(s) + st + len(sub))
    return float(pos + 1)


def verify(s, *chars):
    s = _c(s)
    pool = set("".join(_c(c) for c in chars))
    for i, ch in enumerate(s):
        if ch not in pool:
            return float(i + 1)
    return 0.0


def upcase(s):
    return _c(s).upper()


def lowcase(s):
    return _c(s).lower()


def propcase(s, delims=None):
    s = _c(s).lower()
    d = PROPCASE_DEFAULT_DELIMS if delims is None else delims
    out = []
    cap = True
    for ch in s:
        out.append(ch.upper() if cap else ch)
        cap = ch in d
    return "".join(out)


def strip(s):
    return _c(s).strip(" ")


def trim(s):
    t = _c(s).rstrip(" ")
    return t if t else " "


def trimn(s):
    return _c(s).rstrip(" ")


def left(s):
    s = _c(s)
    stripped = s.lstrip(" ")
    return stripped + " " * (len(s) - len(stripped))


def right(s, length=None):
    s = _c(s)
    L = len(s) if length is None else int(length)
    return s.rstrip(" ").rjust(L)


def length(s):
    t = _c(s).rstrip(" ")
    return float(len(t) if t else 1)


def lengthn(s):
    return float(len(_c(s).rstrip(" ")))


def lengthc(s):
    return float(len(_c(s)))


def repeat(s, n):
    return _c(s) * (int(n) + 1)


def reverse(s):
    return _c(s)[::-1]


def quote(s, q='"'):
    s = _c(s).rstrip(" ")
    return q + s.replace(q, q + q) + q


def dequote(s):
    s = _c(s)
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        q = s[0]
        return s[1:-1].replace(q + q, q)
    return s


def _cat_arg(v) -> str:
    if isinstance(v, (int, float)) or v is None:
        return put(v, "BEST12.").strip()
    return str(v)


def cat(*vals):
    return "".join(_cat_arg(v) for v in vals)


def cats(*vals):
    return "".join(_cat_arg(v).strip(" ") for v in vals)


def catt(*vals):
    return "".join(_cat_arg(v).rstrip(" ") for v in vals)


def catx(sep, *vals):
    parts = [_cat_arg(v).strip(" ") for v in vals]
    parts = [p for p in parts if p != "" and p != "."]
    return _c(sep).join(parts)


def count(s, sub, modifiers=""):
    s, sub = _c(s), _c(sub)
    if "t" in (modifiers or "").lower():
        sub = sub.rstrip()
    if "i" in (modifiers or "").lower():
        s, sub = s.lower(), sub.lower()
    if not sub:
        return 0.0
    return float(s.count(sub))


def countc(s, chars, modifiers=""):
    s = _c(s)
    cls = _char_class(_c(chars), modifiers or "")
    if "v" in (modifiers or "").lower():
        return float(sum(1 for ch in s if ch not in cls))
    return float(sum(1 for ch in s if ch in cls))


def _any(s, pred, start=1):
    s = _c(s)
    for i, ch in enumerate(s[int(start) - 1:], start=int(start)):
        if pred(ch):
            return float(i)
    return 0.0


def anydigit(s, start=1):
    return _any(s, str.isdigit, start)


def anyalpha(s, start=1):
    return _any(s, lambda c: c.isascii() and c.isalpha(), start)


def anyspace(s, start=1):
    return _any(s, str.isspace, start)


def anyalnum(s, start=1):
    return _any(s, lambda c: c.isascii() and c.isalnum(), start)


def anypunct(s, start=1):
    return _any(s, lambda c: c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", start)


def anyupper(s, start=1):
    return _any(s, str.isupper, start)


def anylower(s, start=1):
    return _any(s, str.islower, start)


def notdigit(s, start=1):
    return _any(s.rstrip() if isinstance(s, str) else s, lambda c: not c.isdigit(), start)


def notalpha(s, start=1):
    return _any(s.rstrip() if isinstance(s, str) else s, lambda c: not (c.isascii() and c.isalpha()), start)


def notspace(s, start=1):
    return _any(s, lambda c: not c.isspace(), start)


def _perl_regex(pattern: str):
    p = pattern.strip()
    flags = 0
    if p.startswith("s/") or p.startswith("m/") or p.startswith("/"):
        body = p[1:] if p.startswith("/") else p[2:]
        if p.startswith("s/"):
            m = re.match(r"(.*?)(?<!\\)/(.*?)(?<!\\)/([a-z]*)\Z", body, re.S)
            if not m:
                raise ValueError(f"invalid PRX substitution {pattern!r}")
            pat, rep, opts = m.groups()
            if "i" in opts:
                flags |= re.I
            return re.compile(pat, flags), re.sub(r"\$(\d)", r"\\\1", rep), "g" in opts or True
        m = re.match(r"(.*)(?<!\\)/([a-z]*)\Z", body, re.S)
        if not m:
            raise ValueError(f"invalid PRX pattern {pattern!r}")
        pat, opts = m.groups()
        if "i" in opts:
            flags |= re.I
        return re.compile(pat, flags), None, False
    return re.compile(p), None, False


def prxmatch(pattern, s):
    rx, _, _ = _perl_regex(_c(pattern))
    m = rx.search(_c(s))
    return float(m.start() + 1) if m else 0.0


def prxchange(pattern, times, s):
    rx, rep, _ = _perl_regex(_c(pattern))
    count_ = 0 if is_missing(times) or int(times) == -1 else int(times)
    return rx.sub(rep or "", _c(s), count=count_)


# ---------------------------------------------------------------------------
# formats (PUT)
_FMT_RE = re.compile(r"^(\$?)([A-Za-z_](?:[A-Za-z0-9_]*[A-Za-z_])?)?(\d*)\.(\d*)$")


def parse_format(spec: str):
    """'DATE9.' -> ('', 'DATE', 9, None); '8.2' -> ('', '', 8, 2); '$CHAR10.' -> ('$','CHAR',10,None)."""
    s = spec.strip()
    if "." not in s:
        s = s + "."
    m = _FMT_RE.match(s)
    if not m:
        raise ValueError(f"invalid format {spec!r}")
    dollar, name, w, d = m.groups()
    return dollar, (name or "").upper(), int(w) if w else None, int(d) if d else None


_DATE_FMT_DEFAULT_W = {
    "DATE": 7, "YYMMDD": 8, "MMDDYY": 8, "DDMMYY": 8, "MONYY": 5, "YEAR": 4, "MONTH": 2, "DAY": 2, "QTR": 1,
    "WEEKDATE": 29, "WORDDATE": 18, "WORDDATX": 18, "DOWNAME": 9, "MONNAME": 9, "JULIAN": 5, "E8601DA": 10,
    "IS8601DA": 10, "B8601DA": 8, "YYQ": 6, "YYMM": 7, "WEEKDAY": 1, "YYMMDDN": 8, "YYMMDDD": 8, "YYMMDDS": 8,
    "YYMMDDP": 8, "YYMMDDB": 8, "YYMMDDC": 8, "MMDDYYN": 8, "MMDDYYD": 8, "MMDDYYS": 8, "MMDDYYP": 8,
    "DDMMYYN": 8, "DDMMYYD": 8, "DDMMYYS": 8, "DDMMYYP": 8, "NLDATE": 20, "MMYY": 7, "YYMON": 7,
}
_DATETIME_FMT_DEFAULT_W = {"DATETIME": 16, "E8601DT": 19, "IS8601DT": 19, "B8601DT": 15, "DTDATE": 7,
                           "DATEAMPM": 19, "DTYYMMDD": 8, "DTMONYY": 5, "DTYEAR": 4, "DTWKDATX": 25,
                           "E8601DN": 10, "NLDATM": 30}
_TIME_FMT_DEFAULT_W = {"TIME": 8, "HHMM": 5, "TOD": 8, "HOUR": 2, "MMSS": 5, "E8601TM": 8, "IS8601TM": 8}
DATE_FORMATS = set(_DATE_FMT_DEFAULT_W)
DATETIME_FORMATS = set(_DATETIME_FMT_DEFAULT_W)
TIME_FORMATS = set(_TIME_FMT_DEFAULT_W)


def format_kind(spec: str | None) -> str:
    """'date' | 'datetime' | 'time' | 'char' | 'num' | 'user' for a format spec."""
    if not spec:
        return "num"
    try:
        dollar, name, _, _ = parse_format(spec)
    except ValueError:
        return "user"
    if dollar:
        return "char"
    if name in DATE_FORMATS:
        return "date"
    if name in DATETIME_FORMATS:
        return "datetime"
    if name in TIME_FORMATS:
        return "time"
    if name in ("", "BEST", "Z", "COMMA", "COMMAX", "DOLLAR", "DOLLARX", "PERCENT", "PERCENTN", "E", "F",
                "NEGPAREN", "BESTD", "PVALUE", "SSN", "HEX", "BINARY", "ROMAN", "WORDS", "WORDF", "NUMX",
                "FRACT", "OCTAL", "EURO", "EUROX", "NLNUM"):
        return "num"
    return "user"


def put(value, spec: str) -> str:
    dollar, name, w, d = parse_format(spec)
    if dollar or isinstance(value, str):
        s = _c(value)
        if name in ("", "CHAR", "F"):
            width = w if w is not None else max(len(s), 1)
            return s[:width].ljust(width) if name == "CHAR" else s[:width]
        if name == "UPCASE":
            width = w if w is not None else len(s)
            return s.upper()[:width]
        if name == "QUOTE":
            q = quote(s)
            return q if w is None else q[:w]
        return s if w is None else s[:w]
    x = _num(value)
    if name in DATE_FORMATS:
        return _put_date(x, name, w if w is not None else _DATE_FMT_DEFAULT_W[name])
    if name in DATETIME_FORMATS:
        return _put_datetime(x, name, w if w is not None else _DATETIME_FMT_DEFAULT_W[name], d)
    if name in TIME_FORMATS:
        return _put_time(x, name, w if w is not None else _TIME_FMT_DEFAULT_W[name], d)
    if name in ("BEST", "") and d is None and name == "BEST":
        return put_best(x, w or 12)
    if name in ("", "F"):
        if w is None:
            return put_best(x, 12)
        return put_wd(x, w, d or 0)
    if name == "Z":
        return put_z(x, w or 1, d or 0)
    if name in ("COMMA", "DOLLAR", "COMMAX", "DOLLARX"):
        return put_comma(x, w or 6, d or 0, dollar=name.startswith("DOLLAR"))
    if name in ("PERCENT", "PERCENTN"):
        return put_percent(x, w or 6, d or 0)
    if name == "E":
        return put_e(x, w or 12)
    if name == "BESTD":
        return put_best(x, w or 12)
    if name == "PVALUE":
        return put_pvalue(x, w or 6, d)
    raise ValueError(f"unsupported format {spec!r}")


def _missing_str(w: int) -> str:
    return ".".rjust(w)


def put_best(x, w: int = 12) -> str:
    if x is None:
        return _missing_str(w)
    if x == 0:
        return "0".rjust(w)
    if x == int(x) and abs(x) < 1e15:
        s = str(int(x))
        if len(s) <= w:
            return s.rjust(w)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    avail = w - len(sign)
    if ax >= 1:
        int_digits = len(str(int(ax)))
        if int_digits <= avail:
            decimals = max(min(avail - int_digits - 1, 15 - int_digits), 0)
            s = _fixed(ax, decimals)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            if len(s) <= avail:
                return (sign + s).rjust(w)
    else:
        # decimal notation if it shows at least 1 significant digit with reasonable precision
        decimals = avail - 2 if avail >= 3 else avail - 1
        lead_zeros = -int(math.floor(math.log10(ax))) - 1
        if decimals - lead_zeros >= min(3, decimals) and decimals > 0:
            sig = 15
            decimals = min(decimals, lead_zeros + sig)
            s = _fixed(ax, decimals).rstrip("0").rstrip(".")
            if s not in ("0", ""):
                if avail < len(s) and s.startswith("0."):
                    s = s[1:]
                if len(s) <= avail:
                    return (sign + s).rjust(w)
    return put_e_best(x, w)


def put_e_best(x, w: int) -> str:
    sign = "-" if x < 0 else ""
    ax = abs(x)
    exp = int(math.floor(math.log10(ax)))
    mant = ax / 10 ** exp
    exp_str = f"E{exp}" if exp >= 0 else f"E-{-exp}"
    room = w - len(sign) - len(exp_str)
    if room >= 3:
        dec = room - 2
        m = _fixed(mant, dec)
        if m.startswith("10"):
            exp += 1
            exp_str = f"E{exp}" if exp >= 0 else f"E-{-exp}"
            m = _fixed(ax / 10 ** exp, dec)
        m = m.rstrip("0").rstrip(".") if "." in m else m
        return (sign + m + exp_str).rjust(w)
    m = str(int(round(mant)))
    s = sign + m + exp_str
    return s.rjust(w) if len(s) <= w else "*" * w


def _fixed(x: float, decimals: int) -> str:
    """Round half away from zero on the shortest decimal representation."""
    q = Decimal(repr(x)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    s = format(q, "f")
    if s.startswith("-") and float(q) == 0:
        s = s[1:]
    return s


def put_wd(x, w: int, d: int) -> str:
    if x is None:
        return _missing_str(w)
    s = _fixed(x, d)
    if len(s) <= w:
        return s.rjust(w)
    return put_best(x, w)  # value does not fit: SAS switches to BESTw.


def put_z(x, w: int, d: int) -> str:
    if x is None:
        return _missing_str(w)
    s = _fixed(abs(x), d)
    sign = "-" if x < 0 and float(s) != 0 else ""
    if len(sign) + len(s) > w:
        return put_best(x, w)
    return sign + s.rjust(w - len(sign), "0")


def put_comma(x, w: int, d: int, dollar: bool = False) -> str:
    if x is None:
        return _missing_str(w)
    s = _fixed(abs(x), d)
    intpart, _, frac = s.partition(".")
    intpart = f"{int(intpart):,}"
    body = intpart + ("." + frac if d else "")
    body = ("$" if dollar else "") + body
    if x < 0 and float(s) != 0:
        body = "-" + body
    return body.rjust(w) if len(body) <= w else put_best(x, w)


def put_percent(x, w: int, d: int) -> str:
    if x is None:
        return _missing_str(w)
    v = x * 100
    s = _fixed(abs(v), d) + "%"
    if v < 0 and float(_fixed(abs(v), d)) != 0:
        s = "(" + s + ")"
    return s.rjust(w) if len(s) <= w else put_best(x, w)


def put_e(x, w: int) -> str:
    if x is None:
        return _missing_str(w)
    dec = max(w - 7, 0)
    s = f"{x:.{dec}E}"
    mant, exp = s.split("E")
    return (mant + "E" + exp[0] + exp[1:].zfill(3)).rjust(w)


def put_pvalue(x, w: int, d: int | None) -> str:
    if x is None:
        return _missing_str(w)
    d = d if d is not None else max(w - 2, 1)
    threshold = 10 ** (-d)
    if x < threshold:
        return ("<." + "0" * (d - 1) + "1").rjust(w)
    s = _fixed(x, d)
    if s.startswith("0."):
        s = s[1:]
    return s.rjust(w)


def _put_date(x, name: str, w: int) -> str:
    if x is None:
        return _missing_str(w)
    d = sas_to_date(x)
    if d is None:
        return "*" * w
    mon = MONTHS[d.month - 1]
    if name == "DATE":
        if w >= 11:
            s = f"{d.day:02d}-{mon}-{d.year:04d}"
        elif w >= 9:
            s = f"{d.day:02d}{mon}{d.year:04d}"
        elif w >= 7:
            s = f"{d.day:02d}{mon}{d.year % 100:02d}"
        elif w >= 5:
            s = f"{d.day:02d}{mon}"
        else:
            s = f"{d.day:02d}"
        return s.rjust(w)
    sep_map = {"N": "", "D": "-", "S": "/", "P": ".", "B": " ", "C": ":"}
    for base, order in (("YYMMDD", "ymd"), ("MMDDYY", "mdy"), ("DDMMYY", "dmy")):
        if name.startswith(base):
            suffix = name[len(base):]
            sep = sep_map.get(suffix, "-" if base == "YYMMDD" else "/")
            if suffix == "N":
                yw = 4 if w >= 8 else 2
            else:
                yw = 4 if w >= 10 else 2
            y = f"{d.year:04d}" if yw == 4 else f"{d.year % 100:02d}"
            parts = {"y": y, "m": f"{d.month:02d}", "d": f"{d.day:02d}"}
            if w < 6 and base == "YYMMDD":
                return (y + sep + parts["m"])[:w].rjust(w)
            return sep.join(parts[c] for c in order).rjust(w)
    if name == "MONYY":
        return (mon + (f"{d.year:04d}" if w >= 7 else f"{d.year % 100:02d}")).rjust(w)
    if name == "YEAR":
        return (f"{d.year:04d}" if w >= 4 else f"{d.year % 100:02d}").rjust(w)
    if name == "MONTH":
        return str(d.month).rjust(w)
    if name == "DAY":
        return str(d.day).rjust(w)
    if name == "QTR":
        return str((d.month - 1) // 3 + 1).rjust(w)
    if name == "WEEKDAY":
        return str((d.isoweekday() % 7) + 1).rjust(w)
    if name in ("E8601DA", "IS8601DA"):
        return d.isoformat().rjust(w)
    if name == "B8601DA":
        return d.strftime("%Y%m%d").rjust(w)
    if name == "WORDDATE":
        return f"{MONTH_NAMES[d.month - 1]} {d.day}, {d.year}".rjust(w)
    if name == "WORDDATX":
        return f"{d.day} {MONTH_NAMES[d.month - 1]} {d.year}".rjust(w)
    if name == "WEEKDATE":
        return f"{DAY_NAMES[d.isoweekday() % 7]}, {MONTH_NAMES[d.month - 1]} {d.day}, {d.year}".rjust(w)
    if name == "DOWNAME":
        return DAY_NAMES[d.isoweekday() % 7].rjust(w)
    if name == "MONNAME":
        return MONTH_NAMES[d.month - 1].rjust(w)
    if name == "JULIAN":
        doy = d.timetuple().tm_yday
        return (f"{d.year % 100:02d}{doy:03d}" if w < 7 else f"{d.year:04d}{doy:03d}").rjust(w)
    if name == "YYQ":
        q = (d.month - 1) // 3 + 1
        return (f"{d.year:04d}Q{q}" if w >= 6 else f"{d.year % 100:02d}Q{q}").rjust(w)
    if name == "YYMM":
        return f"{d.year:04d}M{d.month:02d}".rjust(w)
    if name == "MMYY":
        return f"{d.month:02d}M{d.year:04d}".rjust(w)
    if name == "YYMON":
        return f"{d.year:04d}{mon}".rjust(w)
    raise ValueError(f"unsupported date format {name}")


def _put_datetime(x, name: str, w: int, d: int | None) -> str:
    if x is None:
        return _missing_str(w)
    dt = sas_to_datetime(x)
    if dt is None:
        return "*" * w
    mon = MONTHS[dt.month - 1]
    if name == "DATETIME":
        frac = ""
        if d:
            frac = f"{(x % 1):.{d}f}"[1:]
        if w >= 18:
            s = f"{dt.day:02d}{mon}{dt.year:04d}:{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}{frac}"
        elif w >= 16:
            s = f"{dt.day:02d}{mon}{dt.year % 100:02d}:{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
        elif w >= 13:
            s = f"{dt.day:02d}{mon}{dt.year % 100:02d}:{dt.hour:02d}:{dt.minute:02d}"
        else:
            s = f"{dt.day:02d}{mon}{dt.year % 100:02d}"
        return s.rjust(w)
    if name in ("E8601DT", "IS8601DT"):
        s = dt.strftime("%Y-%m-%dT%H:%M:%S")
        if d:
            s += f"{(x % 1):.{d}f}"[1:]
        return s.rjust(w)
    if name == "B8601DT":
        return dt.strftime("%Y%m%dT%H%M%S").rjust(w)
    if name == "DTDATE":
        return _put_date(datepart(x), "DATE", w)
    if name == "DTYYMMDD":
        return _put_date(datepart(x), "YYMMDD", w)
    if name == "DTMONYY":
        return _put_date(datepart(x), "MONYY", w)
    if name == "DTYEAR":
        return _put_date(datepart(x), "YEAR", w)
    if name == "E8601DN":
        return _put_date(datepart(x), "E8601DA", w)
    if name == "DATEAMPM":
        h = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{dt.day:02d}{mon}{dt.year % 100:02d}:{h:02d}:{dt.minute:02d}:{dt.second:02d} {ampm}".rjust(w)
    raise ValueError(f"unsupported datetime format {name}")


def _put_time(x, name: str, w: int, d: int | None) -> str:
    if x is None:
        return _missing_str(w)
    neg = x < 0
    t = abs(x)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    if name in ("TIME", "TOD", "E8601TM", "IS8601TM"):
        if d:
            sec = f"{s:0{3 + d}.{d}f}"
        else:
            sec = f"{int(round(s)) if s - int(s) >= 0.5 else int(s):02d}"
        hh = f"{h:02d}" if name != "TIME" else str(h)
        if w < 8 and not d:
            out = f"{hh}:{m:02d}"
        else:
            out = f"{hh}:{m:02d}:{sec}"
        return (("-" if neg else "") + out).rjust(w)
    if name == "HHMM":
        mm = int(round((t % 3600) / 60))
        hh = h
        if mm == 60:
            hh, mm = hh + 1, 0
        return f"{hh}:{mm:02d}".rjust(w)
    if name == "HOUR":
        return str(int(round(t / 3600))).rjust(w)
    if name == "MMSS":
        return f"{int(t // 60)}:{int(t % 60):02d}".rjust(w)
    raise ValueError(f"unsupported time format {name}")


# ---------------------------------------------------------------------------
# informats (INPUT)
_MON_RE = "(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"


def input_value(text, spec: str):
    """INPUT(text, informat).  Returns float/None for numeric informats, str for $ informats."""
    dollar, name, w, d = parse_format(spec)
    s = _c(text)
    if w is not None:
        s = s[:w]
    if dollar:
        if name == "CHAR":
            return s
        return s.strip(" ") if name in ("", "UPCASE") else s.lstrip(" ")
    raw = s.strip()
    if raw == "" or raw == ".":
        return None
    if name in ("", "BEST", "F", "D", "E"):
        return _input_num(raw, d)
    if name in ("COMMA", "DOLLAR", "COMMAX", "DOLLARX", "NLNUM"):
        cleaned = re.sub(r"[,$\s]", "", raw)
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        v = _input_num(cleaned, d)
        return -v if (v is not None and neg) else v
    if name == "PERCENT":
        cleaned = raw.replace("%", "").replace(",", "")
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        v = _input_num(cleaned.strip("()"), None)
        if v is None:
            return None
        v = v / 100
        return -v if neg else v
    if name in ("YYMMDD", "E8601DA", "IS8601DA", "B8601DA", "YYMMDDN"):
        return _input_ymd(raw)
    if name == "MMDDYY":
        return _input_mdy(raw, "mdy")
    if name == "DDMMYY":
        return _input_mdy(raw, "dmy")
    if name == "DATE":
        return _input_date9(raw)
    if name in ("ANYDTDTE",):
        for fn in (_input_ymd, _input_date9, lambda r: _input_mdy(r, "mdy")):
            v = fn(raw)
            if v is not None:
                return v
        dt = _input_iso_datetime(raw)
        return None if dt is None else datepart(dt)
    if name == "MONYY":
        m = re.fullmatch(_MON_RE + r"[-/ ]?(\d{2}|\d{4})", raw.upper())
        if not m:
            return None
        y = int(m.group(2))
        y = _expand_year(y) if len(m.group(2)) == 2 else y
        return mdy(MONTHS.index(m.group(1)) + 1, 1, y)
    if name == "JULIAN":
        m = re.fullmatch(r"(\d{2}|\d{4})(\d{3})", raw)
        if not m:
            return None
        y = int(m.group(1))
        y = _expand_year(y) if len(m.group(1)) == 2 else y
        doy = int(m.group(2))
        if not 1 <= doy <= (366 if _leap(y) else 365):
            return None
        return date_to_sas(date(y, 1, 1)) + doy - 1
    if name in ("DATETIME",):
        return _input_datetime20(raw)
    if name in ("E8601DT", "IS8601DT", "ANYDTDTM", "B8601DT"):
        v = _input_iso_datetime(raw)
        if v is None and name == "ANYDTDTM":
            v = _input_datetime20(raw)
            if v is None:
                dv = input_value(raw, "ANYDTDTE.")
                v = None if dv is None else dv * SECONDS_PER_DAY
        return v
    if name in ("TIME", "E8601TM", "IS8601TM", "HHMMSS"):
        m = re.fullmatch(r"(-?\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?\s*(AM|PM)?", raw.upper())
        if not m:
            return None
        h, mi = int(m.group(1)), int(m.group(2))
        sec = float(m.group(3)) if m.group(3) else 0.0
        if m.group(4) == "PM" and h < 12:
            h += 12
        if m.group(4) == "AM" and h == 12:
            h = 0
        if mi > 59 or sec >= 60:
            return None
        return h * 3600 + mi * 60 + sec
    raise ValueError(f"unsupported informat {spec!r}")


_NUMBER_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\Z")


def _input_num(raw: str, d: int | None):
    s = raw.replace(" ", "")
    if not _NUMBER_RE.match(s):
        return None
    v = float(s)
    if d and "." not in s and "e" not in s.lower():
        v = v / (10 ** d)
    return v


def _input_ymd(raw: str):
    s = raw.strip()
    m = re.fullmatch(r"(\d{4}|\d{2})[-/.: ]?(\d{1,2})[-/.: ]?(\d{1,2})", s)
    if not m or (len(s) in (6, 8) and not s.isdigit() and not re.search(r"[-/.: ]", s)):
        return None
    if s.isdigit() and len(s) not in (6, 8):
        return None
    if s.isdigit():
        if len(s) == 8:
            y, mo, d = int(s[:4]), int(s[4:6]), int(s[6:])
        else:
            y, mo, d = _expand_year(int(s[:2])), int(s[2:4]), int(s[4:])
    else:
        y = int(m.group(1))
        if len(m.group(1)) == 2:
            y = _expand_year(y)
        mo, d = int(m.group(2)), int(m.group(3))
    return mdy(mo, d, y)


def _input_mdy(raw: str, order: str):
    s = raw.strip()
    if s.isdigit() and len(s) in (6, 8):
        a, b, y = s[:2], s[2:4], s[4:]
    else:
        m = re.fullmatch(r"(\d{1,2})[-/.: ](\d{1,2})[-/.: ](\d{2}|\d{4})", s)
        if not m:
            return None
        a, b, y = m.groups()
    yv = int(y)
    if len(y) == 2:
        yv = _expand_year(yv)
    mo, d = (int(a), int(b)) if order == "mdy" else (int(b), int(a))
    return mdy(mo, d, yv)


def _input_date9(raw: str):
    m = re.fullmatch(r"(\d{1,2})[-/ ]?" + _MON_RE + r"[-/ ]?(\d{2}|\d{4})", raw.strip().upper())
    if not m:
        return None
    d, mon, y = m.groups()
    yv = int(y)
    if len(y) == 2:
        yv = _expand_year(yv)
    return mdy(MONTHS.index(mon) + 1, int(d), yv)


def _input_datetime20(raw: str):
    m = re.fullmatch(r"(\d{1,2})" + _MON_RE + r"(\d{2}|\d{4})[: ](\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d*)?))?", raw.strip().upper())
    if not m:
        return None
    dd, mon, y, h, mi, s = m.groups()
    yv = int(y)
    if len(y) == 2:
        yv = _expand_year(yv)
    dv = mdy(MONTHS.index(mon) + 1, int(dd), yv)
    if dv is None:
        return None
    return dv * SECONDS_PER_DAY + int(h) * 3600 + int(mi) * 60 + (float(s) if s else 0.0)


def _input_iso_datetime(raw: str):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}(?:\.\d*)?))?", raw.strip())
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    dv = mdy(int(mo), int(d), int(y))
    if dv is None or int(h) > 23 or int(mi) > 59:
        return None
    return dv * SECONDS_PER_DAY + int(h) * 3600 + int(mi) * 60 + (float(s) if s else 0.0)


# ---------------------------------------------------------------------------
# %SYSFUNC dispatch
def _to_num_arg(a: str):
    a = a.strip()
    if a in ("", "."):
        return None
    try:
        return float(a)
    except ValueError:
        # date/time literal in %sysfunc args, e.g. '01jan2020'd
        m = re.fullmatch(r"(['\"])(.*)\1(d|dt|t)", a, re.I)
        if m:
            kind = m.group(3).lower()
            spec = {"d": "DATE9.", "dt": "DATETIME20.", "t": "TIME8."}[kind]
            return input_value(m.group(2), spec)
        raise ValueError(f"numeric argument expected, got {a!r}")


def sysfunc(fn: str, args: list[str], now: datetime | None = None):
    f = fn.upper()
    now = now or datetime.now()
    N = _to_num_arg
    if f in ("TODAY", "DATE"):
        return today(now)
    if f == "DATETIME":
        return datetime_to_sas(now)
    if f == "TIME":
        return float(now.hour * 3600 + now.minute * 60 + now.second)
    if f == "PUTN":
        return put(N(args[0]), args[1])
    if f == "PUTC":
        return put(args[0], args[1])
    if f == "INPUTN":
        return input_value(args[0], args[1])
    if f == "INPUTC":
        return input_value(args[0], args[1] if args[1].startswith("$") else "$" + args[1])
    if f == "INTNX":
        return intnx(args[0], N(args[1]), N(args[2]), args[3] if len(args) > 3 else "B")
    if f == "INTCK":
        return intck(args[0], N(args[1]), N(args[2]), args[3] if len(args) > 3 else "D")
    if f == "MDY":
        return mdy(N(args[0]), N(args[1]), N(args[2]))
    numeric_1 = {"YEAR": year, "MONTH": month, "DAY": day, "QTR": qtr, "WEEKDAY": weekday, "DATEPART": datepart,
                 "TIMEPART": timepart, "HOUR": hour, "MINUTE": minute, "SECOND": second, "INT": sas_int,
                 "CEIL": sas_ceil, "FLOOR": sas_floor, "ABS": lambda x: None if x is None else abs(x),
                 "SQRT": lambda x: None if x is None or x < 0 else math.sqrt(x),
                 "EXP": lambda x: None if x is None else math.exp(x),
                 "LOG": lambda x: None if x is None or x <= 0 else math.log(x),
                 "LOG10": lambda x: None if x is None or x <= 0 else math.log10(x)}
    if f in numeric_1:
        return numeric_1[f](N(args[0]))
    if f == "ROUND":
        return sas_round(N(args[0]), N(args[1]) if len(args) > 1 else 1.0)
    if f == "MOD":
        return sas_mod(N(args[0]), N(args[1]))
    agg = {"SUM": sas_sum, "MEAN": sas_mean, "MIN": sas_min, "MAX": sas_max, "N": sas_n, "NMISS": sas_nmiss}
    if f in agg:
        return agg[f](*[N(a) for a in args])
    char_1 = {"UPCASE": upcase, "LOWCASE": lowcase, "PROPCASE": propcase, "STRIP": strip, "TRIM": trimn,
              "LEFT": lambda s: s.lstrip(), "COMPBL": compbl, "REVERSE": reverse, "DEQUOTE": dequote,
              "QUOTE": quote}
    if f in char_1:
        return char_1[f](args[0] if args else "")
    if f == "LENGTH":
        return length(args[0] if args else "")
    if f == "LENGTHN":
        return lengthn(args[0] if args else "")
    if f == "COMPRESS":
        return compress(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else "")
    if f == "TRANWRD":
        return tranwrd(args[0], args[1], args[2])
    if f == "TRANSLATE":
        return translate(args[0], *args[1:])
    if f == "SCAN":
        return scan(args[0], N(args[1]), args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else "")
    if f in ("SUBSTR", "SUBSTRN"):
        fn_ = substr if f == "SUBSTR" else substrn
        return fn_(args[0], N(args[1]), N(args[2]) if len(args) > 2 else None)
    if f == "INDEX":
        return index(args[0], args[1])
    if f == "FIND":
        return find(args[0], args[1], *(args[2:3]))
    if f == "COUNTW":
        return countw(args[0], args[1] if len(args) > 1 else None)
    if f == "CATS":
        return cats(*args)
    if f == "CATX":
        return catx(args[0], *args[1:])
    if f == "CAT":
        return cat(*args)
    if f == "REPEAT":
        return repeat(args[0], N(args[1]))
    if f == "PRXMATCH":
        return prxmatch(args[0], args[1])
    if f == "PRXCHANGE":
        return prxchange(args[0], N(args[1]), args[2])
    if f in ("IFC", "IFN"):
        cond = N(args[0])
        pick = args[1] if (cond is not None and cond != 0) else args[2]
        if cond is None and len(args) > 3:
            pick = args[3]
        return pick if f == "IFC" else N(pick)
    if f == "GETOPTION":
        return {"YEARCUTOFF": str(YEARCUTOFF), "OBS": "MAX", "MISSING": "."}.get(args[0].upper(), "")
    if f in ("EXIST", "FILEEXIST", "FEXIST", "OPEN", "ATTRN", "VARNUM"):
        raise ValueError(f"%SYSFUNC({f}) depends on run-time data")
    raise ValueError(f"%SYSFUNC({f}) not implemented")
