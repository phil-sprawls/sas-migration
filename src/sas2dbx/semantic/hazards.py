"""Semantic hazard registry: where SAS and Spark disagree, and what sas2dbx does about it.

Single source of truth used by the analyzer (risk profile), code generator
(strategy + report), LLM prompts (context) and conformance tests (keyed by id).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hazard:
    id: str
    title: str
    sas: str
    spark: str
    handling: str
    severity: str = "high"      # high = silently wrong results if ignored


HAZARDS: dict[str, Hazard] = {}


def _h(id_, title, sas, spark, handling, severity="high"):
    HAZARDS[id_] = Hazard(id_, title, sas, spark, handling, severity)


_h("H-MISS-CMP", "Missing values in comparisons",
   "Numeric missing is smaller than every number: `x < 5` is TRUE for missing x, `. = .` is TRUE.",
   "NULL comparisons yield NULL, so rows silently drop out of filters and CASE branches.",
   "Null-aware sas_lt/le/gt/ge, <=> for equality; simplified where an operand cannot be missing.")
_h("H-MISS-ARITH", "Missing values in arithmetic vs. SAS functions",
   "Operators propagate missing; SUM()/MEAN()/MIN()/MAX() functions ignore missing.",
   "Operators propagate NULL (same); least/greatest ignore NULL but + does not.",
   "Operators kept; SAS statistical functions map to array-based runtime functions that skip NULL.", "medium")
_h("H-TRUTH", "Numeric truth values",
   "Any non-zero, non-missing number is true; comparisons return 1/0.",
   "Booleans are a separate type; NULL is neither true nor false.",
   "sas_truth()/b2n() conversions inserted by type inference.", "medium")
_h("H-ORDER", "Row order",
   "Data sets are ordered; FIRST./LAST., LAG, NODUPKEY, OBS=, _N_ and MERGE depend on it.",
   "DataFrames are unordered sets.",
   "Every data set carries _sas_seq (logical order) re-established after each order-changing step.")
_h("H-NODUPKEY", "PROC SORT NODUPKEY keeps the first row",
   "Keeps the first observation of each key in input order (EQUALS is the default).",
   "dropDuplicates keeps an arbitrary row.",
   "row_number() over (partition by keys order by _sas_seq) = 1.")
_h("H-MERGE-M2M", "Many-to-many MERGE",
   "Rows within a BY group are paired positionally; the shorter side's last row is retained.",
   "A join multiplies rows (Cartesian product within the key).",
   "P.merge pairs rows by (key, row-number) with SAS retention semantics.")
_h("H-MERGE-OVERLAP", "Variables present in several MERGE data sets",
   "The value from the data set read last (later in the MERGE statement) wins; exhausted data sets keep their last values.",
   "Join output has duplicate columns or picks an arbitrary side.",
   "P.merge selects the most recently read source per row.")
_h("H-MERGE-RETAIN", "Modifying a variable from a data set that is not re-read",
   "In one-to-many merges a changed value persists to later rows of the BY group.",
   "Vectorised code recomputes each row independently.",
   "Detected; escalated to Tier C / LLM.")
_h("H-UPDATE-DUPMASTER", "UPDATE with duplicate BY values in the master data set",
   "SAS warns and applies all transactions only to the first master observation of the BY group; "
   "the other duplicates are written unchanged after it.",
   "Window or join rewrites collapse the group to one row, or apply the transactions to every duplicate.",
   "UPDATE is routed to LLM/manual; golden case FRONTIER-UPDATE-02 and synthetic inputs check it.")
_h("H-CHAR-LEN", "Character length fixed at compile time",
   "Length comes from the first occurrence: `if a then x='Y'; else x='No';` stores 'N'.",
   "Strings are unbounded.",
   "Lengths tracked statically; values truncated with store(..., length) where they could exceed it.")
_h("H-CONCAT-PAD", "Concatenation keeps trailing blanks",
   "`a || b` includes a's padding up to its declared length.",
   "Strings carry no padding.",
   "Variables are padded to their declared length inside || (pad()).")
_h("H-BLANK", "Character missing is blank",
   "Blank and 'missing' are the same; blank keys match in joins and BY groups.",
   "'' and NULL differ; NULL keys never match.",
   "Character values are normalised to '' internally; NULL only at output boundaries.", "medium")
_h("H-ROUND", "ROUND semantics",
   "Rounds half away from zero on the decimal value with fuzzing (ROUND(2.675,.01)=2.68).",
   "Python rounds half-to-even; naive binary rounding gives 2.67.",
   "sas_round uses DECIMAL arithmetic on the shortest decimal representation.")
_h("H-DIV0", "Division by zero",
   "Result is missing with a NOTE.",
   "NULL in legacy mode but an ERROR under ANSI mode (Spark 4 / DBR 17+ default).",
   "try_divide / guarded runtime functions; tests run with ANSI on and off.")
_h("H-INVALID-DATA", "Invalid data in INPUT/conversions",
   "Invalid values become missing (with ?? silently).",
   "CAST raises under ANSI mode.",
   "Informats validate with regular expressions before any cast.")
_h("H-AUTOCONV", "Automatic character/numeric conversion",
   "Numbers used as text use BEST12. (right aligned); text used as numbers uses BEST32.",
   "Implicit casts differ (1.0 -> '1.0'; ANSI errors).",
   "Explicit num_to_char()/char_to_num() inserted; reported.")
_h("H-LAG-COND", "LAG inside a condition",
   "LAG's queue only updates when the call executes.",
   "A window lag always looks at the previous row.",
   "lag() over a window partitioned by the executing predicate.")
_h("H-LAG-BY", "LAG crosses BY groups",
   "LAG ignores BY groups unless reset explicitly.",
   "Window lag partitioned by key would differ.",
   "Global ordered window, exactly like SAS.", "medium")
_h("H-REMERGE", "PROC SQL remerging summary statistics",
   "Aggregates mixed with detail columns are remerged onto every row.",
   "Analysis error (non-aggregated column).",
   "Rewritten as window aggregates.")
_h("H-SQL-JOIN-MISS", "PROC SQL joins match missing keys",
   "Missing values compare equal in join conditions.",
   "NULL = NULL is not true; rows silently fail to join.",
   "<=> (null-safe equality) in ON/WHERE equalities.")
_h("H-SQL-CHAR-AGG", "Blank values in SQL aggregates",
   "COUNT(char) / MIN / MAX ignore blank (missing) character values.",
   "'' is a value and is counted.",
   "NULLIF(x, '') inside aggregates over character columns.")
_h("H-PCTL", "Percentile definition",
   "PROC MEANS/UNIVARIATE default QNTLDEF=5 (empirical distribution with averaging).",
   "percentile_approx is approximate; percentile uses linear interpolation (type 7).",
   "Exact QNTLDEF=5 implementation.")
_h("H-FMT-GROUP", "Grouping by formatted values",
   "PROC FREQ/MEANS/REPORT group on the formatted value.",
   "Group by raw value.",
   "Formats applied before grouping.")
_h("H-TRANSLATE-ARGS", "TRANSLATE argument order",
   "TRANSLATE(source, to, from); unmatched 'from' characters become blanks.",
   "translate(source, from, to) deletes unmatched characters.",
   "sas_translate swaps arguments and pads 'to'.")
_h("H-INTCK", "INTCK counts boundaries",
   "INTCK('MONTH','31JAN'd,'01FEB'd)=1 (discrete method).",
   "months_between returns ~0.03.",
   "Interval-index arithmetic identical to SAS (discrete and continuous).")
_h("H-SPECIAL-MISSING", "Special missing values (.A-.Z, ._)",
   "Distinct, ordered missing values.",
   "Single NULL.",
   "Collapsed to NULL; comparisons against .Z/. kept correct; equality tests between special missings flagged.", "medium")
_h("H-TRIM-BLANK", "TRIM of a blank value",
   "TRIM returns one blank, so trim(a)||trim(b) can start with a blank.",
   "rtrim returns ''.",
   "sas_trim returns ' ' for blank input.", "low")
_h("H-NOT-PRECEDENCE", "NOT binds tighter than comparisons",
   "`not x = 1` means `(not x) = 1` in a DATA step.",
   "NOT x = 1 means NOT (x = 1).",
   "Parser follows SAS precedence; reported where it changes meaning.", "medium")
_h("H-MACRO-IN-COMMENT", "Macro triggers inside * comments",
   "Macro calls and %LET inside `* ... ;` comments execute.",
   "n/a",
   "Treated as comments by default (intent), reported; SAS-exact mode available.", "medium")
_h("H-RUNTIME-MACRO", "Macro variables created at run time",
   "CALL SYMPUT / SELECT INTO values are substituted into later code.",
   "No equivalent text substitution.",
   "Converted to Python variables read from the context (ctx.symget), also inside quoted strings and date literals; "
   "SQL literals are escaped, never spliced in as code.", "medium")
_h("H-MACRO-EXTERNAL", "Macro variables supplied from outside the program",
   "Prompts, a parameter module (%INCLUDE), the autoexec or the scheduler set values the program never defines; an "
   "undefined reference stays as the text '&name' with only a WARNING.",
   "n/a",
   "Never frozen: steps using an undefined variable stay unsupported until it is declared - --param (read from the "
   "Databricks job parameter at run time; one notebook serves every value) or --define (fixed at conversion).")
_h("H-RUNTIME-AUTOMATIC", "Automatic macro variables of the run",
   "&SYSDATE9, &SYSDAY, &SYSTIME, &SYSUSERID, &SYSPARM, %SYSGET and %SYSFUNC(TODAY()) take the values of the session "
   "that runs the program.",
   "n/a",
   "Read from the context when the notebook runs (the clock frozen by as_of for parity runs), never from the "
   "converter's clock or environment; values derived from them by macro functions go to LLM/manual.", "medium")
_h("H-RANDOM", "Random number functions",
   "RANUNI/RAND streams depend on SAS's generator and seeds.",
   "Different generator.",
   "Flagged: results cannot match row for row; validate distributions instead.")
_h("H-TODAY", "Current date/time",
   "TODAY()/DATETIME() change between runs.",
   "Same.",
   "Clock frozen by the context (as_of) for reproducible parity runs.", "low")
_h("H-YEARCUTOFF", "Two-digit years",
   "YEARCUTOFF= (1926 in SAS 9.4; 1940 in recent Viya) decides the century.",
   "n/a",
   "Runtime uses 1926; make sure it matches the source system's option.", "medium")


_h("H-XPORT-ZERO", "XPORT (IBM float) zero decoding",
   "SAS reads an IBM-format zero in a transport file as 0.",
   "pandas.read_sas(format='xport') can decode it as 5.397605346934028e-79, silently breaking `x = 0` / `IN (0, ...)`.",
   "Loaders use pyreadstat (exact); never load XPT data with pandas.read_sas during migration.")


_h("H-SCHEMA-INFERRED", "Input schema inferred from usage",
   "Variable types come from the data set descriptor.",
   "No catalog entry: types were inferred from how the program uses each variable.",
   "Unknown columns pass through untouched; harvest the real schema (capture kit / Delta) and re-convert before sign-off.", "medium")


def get(hid: str) -> Hazard:
    return HAZARDS[hid]
