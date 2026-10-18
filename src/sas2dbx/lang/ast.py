"""IR for parsed SAS programs (post macro expansion)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# expressions
@dataclass
class Expr:
    line: int = field(default=0, kw_only=True, compare=False, repr=False)


@dataclass
class Num(Expr):
    value: float
    text: str = ""


@dataclass
class Str(Expr):
    value: str
    suffix: str = ""        # '' | 'd' | 'dt' | 't' | 'x' | 'b'
    quote: str = "'"


@dataclass
class Missing(Expr):
    kind: str = "."         # '.', '.A'..'.Z', '._'


@dataclass
class Var(Expr):
    name: str


@dataclass
class FirstLast(Expr):
    which: str              # FIRST | LAST
    var: str


@dataclass
class Call(Expr):
    name: str               # upper-case
    args: list["Expr"]


@dataclass
class Index(Expr):
    array: str
    indices: list["Expr"]


@dataclass
class Unary(Expr):
    op: str                 # '-', '+', 'NOT'
    operand: "Expr"


@dataclass
class Binary(Expr):
    op: str                 # '**','*','/','+','-','||','MIN','MAX','AND','OR'
    left: "Expr"
    right: "Expr"


@dataclass
class Compare(Expr):
    op: str                 # EQ NE LT LE GT GE
    left: "Expr"
    right: "Expr"
    colon: bool = False     # =: style prefix comparison


@dataclass
class InRange:
    lo: "Expr"
    hi: "Expr"


@dataclass
class InList(Expr):
    operand: "Expr"
    items: list[Union["Expr", InRange]]
    negated: bool = False


@dataclass
class Between(Expr):
    operand: "Expr"
    lo: "Expr"
    hi: "Expr"
    negated: bool = False


@dataclass
class IsMissing(Expr):
    operand: "Expr"
    negated: bool = False


@dataclass
class Like(Expr):
    operand: "Expr"
    pattern: "Expr"
    negated: bool = False


@dataclass
class Contains(Expr):
    operand: "Expr"
    sub: "Expr"
    negated: bool = False


@dataclass
class SoundsLike(Expr):
    left: "Expr"
    right: "Expr"


@dataclass
class OfList(Expr):
    items: list["VarItem"]


@dataclass
class RtMacro(Expr):
    """Run-time macro variable reference (&x where x comes from SYMPUT / INTO:)."""
    name: str


@dataclass
class FormatSpec(Expr):
    """A format/informat used as a function argument (PUT(x, date9.))."""
    spec: str
    modifier: str = ""      # '??' or '?' for INPUT


# ---------------------------------------------------------------------------
# variable lists
@dataclass
class VarName:
    name: str


@dataclass
class VarRange:
    start: str
    end: str
    kind: str = "numbered"  # numbered (x1-x5) | positional (a--z) | positional_num | positional_char


@dataclass
class VarPrefix:
    prefix: str


@dataclass
class VarSpecial:
    name: str               # _ALL_ _NUMERIC_ _CHARACTER_


@dataclass
class ArrayStar:
    array: str


@dataclass
class VarParam:
    """A variable list supplied at run time (&vars from a parameter module)."""
    name: str


VarItem = Union[VarName, VarRange, VarPrefix, VarSpecial, ArrayStar, VarParam]


# ---------------------------------------------------------------------------
# datasets
@dataclass
class DatasetOptions:
    keep: Optional[list[VarItem]] = None
    drop: Optional[list[VarItem]] = None
    rename: list[tuple[str, str]] = field(default_factory=list)
    where: Optional[Expr] = None
    obs: Optional[int] = None
    firstobs: Optional[int] = None
    in_var: Optional[str] = None
    label: Optional[str] = None
    other: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (self.keep is None and self.drop is None and not self.rename and self.where is None
                and self.obs is None and self.firstobs is None and self.in_var is None)


@dataclass
class DatasetRef:
    libref: Optional[str]
    name: str
    options: DatasetOptions = field(default_factory=DatasetOptions)
    line: int = 0
    dynamic: bool = False   # contains run-time macro placeholders

    @property
    def key(self) -> str:
        lib = (self.libref or "WORK").upper()
        return f"{lib}.{self.name.upper()}"

    @property
    def display(self) -> str:
        return f"{self.libref}.{self.name}" if self.libref else self.name

    @property
    def is_null(self) -> bool:
        return self.libref is None and self.name.upper() == "_NULL_"


# ---------------------------------------------------------------------------
# DATA step statements
@dataclass
class Stmt:
    line: int = field(default=0, kw_only=True, compare=False)


@dataclass
class Assign(Stmt):
    target: Expr            # Var | Index | Call(SUBSTR)
    expr: Expr


@dataclass
class SumStmt(Stmt):
    var: Expr               # Var | Index
    expr: Expr


@dataclass
class If(Stmt):
    cond: Expr
    then: Stmt
    else_: Optional[Stmt] = None


@dataclass
class SubsetIf(Stmt):
    cond: Expr


@dataclass
class Where(Stmt):
    cond: Expr
    also: bool = False


@dataclass
class DoBlock(Stmt):
    body: list[Stmt]


@dataclass
class DoSpec:
    start: Expr
    stop: Optional[Expr] = None
    by: Optional[Expr] = None


@dataclass
class DoLoop(Stmt):
    var: Expr
    specs: list[DoSpec]
    while_: Optional[Expr] = None
    until: Optional[Expr] = None
    body: list[Stmt] = field(default_factory=list)


@dataclass
class DoWhile(Stmt):
    cond: Expr
    body: list[Stmt]


@dataclass
class DoUntil(Stmt):
    cond: Expr
    body: list[Stmt]


@dataclass
class When:
    values: list[Expr]
    stmt: Stmt


@dataclass
class Select(Stmt):
    expr: Optional[Expr]
    whens: list[When]
    otherwise: Optional[Stmt] = None


@dataclass
class SetStmt(Stmt):
    datasets: list[DatasetRef]
    options: dict[str, str] = field(default_factory=dict)   # nobs end point key indsname open


@dataclass
class MergeStmt(Stmt):
    datasets: list[DatasetRef]
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class UpdateStmt(Stmt):
    datasets: list[DatasetRef]
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class ModifyStmt(Stmt):
    datasets: list[DatasetRef]
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class ByStmt(Stmt):
    vars: list[tuple[str, bool]]        # (name, descending)
    notsorted: bool = False
    groupformat: bool = False


@dataclass
class RetainStmt(Stmt):
    items: list[tuple[list[VarItem], list[Expr]]]   # vars with optional initial values
    all_: bool = False


@dataclass
class KeepStmt(Stmt):
    vars: list[VarItem]


@dataclass
class DropStmt(Stmt):
    vars: list[VarItem]


@dataclass
class RenameStmt(Stmt):
    pairs: list[tuple[str, str]]


@dataclass
class LengthStmt(Stmt):
    items: list[tuple[list[VarItem], bool, int]]    # (vars, is_char, length)
    default: Optional[int] = None


@dataclass
class FormatStmt(Stmt):
    items: list[tuple[list[VarItem], Optional[str]]]


@dataclass
class InformatStmt(Stmt):
    items: list[tuple[list[VarItem], Optional[str]]]


@dataclass
class LabelStmt(Stmt):
    items: list[tuple[str, str]]


@dataclass
class AttribStmt(Stmt):
    items: list[tuple[list[VarItem], dict[str, str]]]


@dataclass
class ArrayStmt(Stmt):
    name: str
    dims: list[Union[int, str, tuple[int, int]]]    # int | '*' | (lo, hi)
    is_char: bool
    length: Optional[int]
    elements: list[VarItem]
    init: list[Expr]
    temporary: bool = False


@dataclass
class OutputStmt(Stmt):
    datasets: list[str]


@dataclass
class DeleteStmt(Stmt):
    pass


@dataclass
class ReturnStmt(Stmt):
    pass


@dataclass
class StopStmt(Stmt):
    pass


@dataclass
class LeaveStmt(Stmt):
    pass


@dataclass
class ContinueStmt(Stmt):
    pass


@dataclass
class CallStmt(Stmt):
    name: str
    args: list[Expr]


@dataclass
class InputStmt(Stmt):
    items: list["InputItem"]
    raw: str = ""


@dataclass
class InputItem:
    kind: str                # var | pointer | linehold | group
    var: Optional[str] = None
    is_char: bool = False
    informat: Optional[str] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    modifier: str = ""       # ':' '&' '~'
    pointer: Optional[str] = None


@dataclass
class InfileStmt(Stmt):
    fileref: str
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class DatalinesStmt(Stmt):
    text: str


@dataclass
class PutStmt(Stmt):
    raw: str


@dataclass
class FileStmt(Stmt):
    raw: str


@dataclass
class GotoStmt(Stmt):
    label: str
    link: bool = False


@dataclass
class LabeledStmt(Stmt):
    label: str
    stmt: Stmt


@dataclass
class AbortStmt(Stmt):
    raw: str = ""


@dataclass
class MissingStmt(Stmt):
    chars: list[str]


@dataclass
class DeclareStmt(Stmt):
    raw: str


@dataclass
class UnknownStmt(Stmt):
    keyword: str
    raw: str


# ---------------------------------------------------------------------------
# steps
@dataclass
class Step:
    start_line: int = field(default=0, kw_only=True)
    end_line: int = field(default=0, kw_only=True)
    start: int = field(default=0, kw_only=True)     # offset into expanded text
    end: int = field(default=0, kw_only=True)


@dataclass
class DataStep(Step):
    outputs: list[DatasetRef]
    stmts: list[Stmt]
    view: Optional[str] = None


@dataclass
class ProcStmt:
    keyword: str
    raw: str                     # text after the keyword, without the ';'
    tokens: list = field(default_factory=list, repr=False)
    line: int = 0


@dataclass
class ProcStep(Step):
    name: str
    options_raw: str
    options_tokens: list = field(default_factory=list, repr=False)
    stmts: list[ProcStmt] = field(default_factory=list)


@dataclass
class GlobalStmt(Step):
    keyword: str
    raw: str
    tokens: list = field(default_factory=list, repr=False)


@dataclass
class Unresolved(Step):
    message: str


@dataclass
class Program:
    steps: list[Step]
    text: str                    # expanded source
    errors: list[str] = field(default_factory=list)
