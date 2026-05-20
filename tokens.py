from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import math, itertools
from environment import (
    Environment, Variable, NumericVar, ListVar, MatrixVar, StringVar, StatVar, WindowVar,
)
import purefunctions
import forms

@dataclass(eq=False)
class Token:
    code: bytes
    text: str
    bp: tuple | None = None # (left_bp, right_bp) for binary operators
    binary_op: Any = None   # (lhs, rhs) -> value
    unary_op: Any = None    # (operand) -> value  (prefix or postfix)
    postfix: bool = False   # True for postfix unary operators
    func: Any = None        # (ArgParser) -> value  for function tokens
    cmd: Any = None         # (ArgParser) -> None  for command tokens
    variable: Any = None    # Variable instance for storable typed variables
    nullary: Any = None     # (env) -> value  for read-only computed constants
    converter: Any = None   # (value) -> value  for ►DMS, ►Dec, ►Frac etc.

    # ── Token type predicates ──────────────────────────────────────────────────

    def is_numeric_var(self) -> bool:
        return 0x41 <= self.code[0] < 0x5c

    def is_list_var(self) -> bool:
        return self.code[0] == 0x5d

    def is_list_var_start(self) -> bool:
        return self.code[0] in {0x5d, 0xeb}

    def is_matrix_var(self) -> bool:
        return self.code[0] == 0x5c

    def is_string_var(self) -> bool:
        return self.code[0] == 0xaa

    def is_stat_var(self) -> bool:
        return self.code[0] == 0x62

    def is_window_var(self) -> bool:
        return self.code[0] == 0x63

    def is_digit(self) -> bool:
        return 0x30 <= self.code[0] <= 0x39

    def is_name_char(self):
        return self.is_numeric_var() or self.is_digit()

    def can_start_atom(self) -> bool:
        return (
            self.is_digit()
            or self.variable is not None
            or self.nullary is not None
            or self.func is not None
            or self in {DOT, L_PAREN, L_BRACE, L_BRACKET, QUOTE, NEG, LIST_PREFIX}
        )
    
    def __repr__(self):
        return f'{self.code.hex().upper()}:{self.text}'


EOF_TOKEN = Token(b'\x00', '<END-OF-INPUT>')

_SEEN: set[bytes] = set()

def _make_pure_func(f):
    def wrapper(a):
        return f(*a.parse_args())
    return wrapper

def _make_variable(code: bytes) -> Variable | None:
    b0 = code[0]
    if 0x41 <= b0 <= 0x5b:  # A–Z, θ
        return NumericVar(b0 - 0x41)
    if b0 == 0x5d:           # L1–L6
        return ListVar(code[1])
    if b0 == 0x5c:           # [A]–[J]
        return MatrixVar(code[1])
    if b0 == 0xaa:           # Str0–9
        return StringVar(code[1])
    if b0 == 0x62:           # stat vars
        return StatVar(code[1])
    if b0 == 0x63:           # window vars
        return WindowVar(code[1])
    return None


def token(
    code: bytes,
    text: str,
    *,
    bp: tuple[int, int] | None = None,
    binary_op=None,
    unary_op=None,
    postfix: bool = False,
    func: Callable | None = None,
    pure_func: Callable | None = None,
    cmd: Callable | None = None,
    variable: Variable | None = None,
    nullary: Callable | None = None,
    converter: Callable | None = None,
) -> Token:

    if code in _SEEN:
        raise ValueError(f'Duplicate token code: {code!r} ({text!r})')
    _SEEN.add(code)
    if pure_func is not None:
        if func is not None:
            raise ValueError(f'Token {text!r}: cannot set both func and pure_func')
        func = _make_pure_func(pure_func)
    if variable is None and nullary is None:
        variable = _make_variable(code)

    return Token(
        code, text,
        bp, binary_op, unary_op, postfix, func, cmd, variable, nullary, converter,
    )


# ── Named syntactic tokens (referenced by identity in the parser) ──────────────

# Structural / delimiter
STORE       = token(b'\x04', '→')
L_BRACKET   = token(b'\x06', '[')
R_BRACKET   = token(b'\x07', ']')
L_BRACE     = token(b'\x08', '{')
R_BRACE     = token(b'\x09', '}')
L_PAREN     = token(b'\x10', '(')
R_PAREN     = token(b'\x11', ')')
QUOTE       = token(b'\x2a', '"')
COMMA       = token(b'\x2b', ',')
DOT         = token(b'\x3a', '.')
COLON       = token(b'\x3e', ':')
NEWLINE     = token(b'\x3f', '\n')
PRGM        = token(b'\x5f', 'prgm')
ANS         = token(b'\x72', 'Ans')
NEG         = token(b'\xb0', '−', unary_op=purefunctions.neg)
DIM         = token(b'\xb5', 'dim(', pure_func=purefunctions.dim)
LIST_PREFIX = token(b'\xeb', '∟')

# Postfix operators
RAD       = token(b'\x0a', 'ʳ', postfix=True, unary_op=lambda x: x)
DEG       = token(b'\x0b', '°', postfix=True, unary_op=math.radians)
INV       = token(b'\x0c', '⁻¹', postfix=True, unary_op=purefunctions.inv)
SQ        = token(b'\x0d', '²', postfix=True, unary_op=lambda x: purefunctions.pow_(x, 2))
TRANSPOSE = token(b'\x0e', 'ᵀ', postfix=True, unary_op=purefunctions.transpose)
CUBE      = token(b'\x0f', '³', postfix=True, unary_op=lambda x: purefunctions.pow_(x, 3))
FACT      = token(b'\x2d', '!', postfix=True, unary_op=purefunctions.factorial)

# Binary operators
SCI_E = token(b'\x3b', 'ᴇ', bp=(65, 66), binary_op=lambda a, b: purefunctions.mul(a, 10 ** b))
OR    = token(b'\x3c', ' or ',  bp=(20, 21), binary_op=purefunctions.or_)
XOR   = token(b'\x3d', ' xor ', bp=(20, 21), binary_op=purefunctions.xor)
AND   = token(b'\x40', ' and ', bp=(30, 31), binary_op=purefunctions.and_)
EQ    = token(b'\x6a', '=', bp=(40, 41), binary_op=purefunctions.eq)
LT    = token(b'\x6b', '<', bp=(40, 41), binary_op=purefunctions.lt)
GT    = token(b'\x6c', '>', bp=(40, 41), binary_op=purefunctions.gt)
LE    = token(b'\x6d', '≤', bp=(40, 41), binary_op=purefunctions.le)
GE    = token(b'\x6e', '≥', bp=(40, 41), binary_op=purefunctions.ge)
NE    = token(b'\x6f', '≠', bp=(40, 41), binary_op=purefunctions.ne)
ADD   = token(b'\x70', '+', bp=(50, 51), binary_op=purefunctions.add)
SUB   = token(b'\x71', '-', bp=(50, 51), binary_op=purefunctions.sub)
MUL   = token(b'\x82', '*', bp=(60, 61), binary_op=purefunctions.mul)
DIV   = token(b'\x83', '/', bp=(60, 61), binary_op=purefunctions.div)
NPR   = token(b'\x94', 'nPr', bp=(60, 61), binary_op=purefunctions.npr)
NCR   = token(b'\x95', 'nCr', bp=(60, 61), binary_op=purefunctions.ncr)
RAND  = token(b'\xab', 'rand', nullary=Environment.rand)
POW   = token(b'\xf0', '^', bp=(70, 69), binary_op=purefunctions.pow)
XROOT = token(b'\xf1', '×√', bp=(60, 61), binary_op=lambda a, b: purefunctions.xth_root(b, a))

# ── Token list ─────────────────────────────────────────────────────────────────

TOKENS: list[Token] = [
    # One-byte tokens
    token(b'\x01', '►DMS',  converter=purefunctions.to_dms),
    token(b'\x02', '►Dec',  converter=purefunctions.to_dec),
    token(b'\x03', '►Frac', converter=purefunctions.to_frac),
    STORE,
    token(b'\x05', 'Boxplot'),
    L_BRACKET,
    R_BRACKET,
    L_BRACE,
    R_BRACE,
    RAD,
    DEG,
    INV,
    SQ,
    TRANSPOSE,
    CUBE,
    L_PAREN,
    R_PAREN,
    token(b'\x12', 'round(',   pure_func=purefunctions.round),
    token(b'\x13', 'pxl-Test('),
    token(b'\x14', 'augment('),
    token(b'\x15', 'rowSwap('),
    token(b'\x16', 'row+('),
    token(b'\x17', '*row('),
    token(b'\x18', '*row+('),
    token(b'\x19', 'max(',    pure_func=purefunctions.max),
    token(b'\x1a', 'min(',    pure_func=purefunctions.min),
    token(b'\x1b', 'R►Pr('),
    token(b'\x1c', 'R►Pθ('),
    token(b'\x1d', 'P►Rx('),
    token(b'\x1e', 'P►Ry('),
    token(b'\x1f', 'median(', pure_func=purefunctions.median),
    token(b'\x20', 'randM(', pure_func=purefunctions.rand_m),
    token(b'\x21', 'mean(',   pure_func=purefunctions.mean),
    token(b'\x22', 'solve('),
    token(b'\x23', 'seq(',    func=forms.seq),
    token(b'\x24', 'fnInt(',  func=forms.fnint),
    token(b'\x25', 'nDeriv(', func=forms.nderiv),
    token(b'\x27', 'fMin('),
    token(b'\x28', 'fMax('),
    token(b'\x29', ' '),
    QUOTE,
    COMMA,
    token(b'\x2c', '𝑖', nullary=lambda env: 1j),
    FACT,
    token(b'\x2e', 'CubicReg '),
    token(b'\x2f', 'QuartReg '),
    *[token(bytes([0x30 + i]), chr(0x30 + i)) for i in range(10)],
    DOT, SCI_E, OR, XOR, COLON, NEWLINE, AND,
    # Variables A–Z (0x41–0x5A)
    *[token(bytes([0x41 + i]), chr(0x41 + i)) for i in range(26)],
    token(b'\x5b', 'θ'),
    PRGM,
    token(b'\x64', 'Radian'),
    token(b'\x65', 'Degree'),
    token(b'\x66', 'Normal'),
    token(b'\x67', 'Sci'),
    token(b'\x68', 'Eng'),
    token(b'\x69', 'Float'),
    EQ,
    LT,
    GT,
    LE,
    GE,
    NE,
    ADD,
    SUB,
    ANS,
    token(b'\x73', 'Fix'),
    token(b'\x74', 'Horiz'),
    token(b'\x75', 'Full'),
    token(b'\x76', 'Func'),
    token(b'\x77', 'Param'),
    token(b'\x78', 'Polar'),
    token(b'\x79', 'Seq'),
    token(b'\x7a', 'IndpntAuto'),
    token(b'\x7b', 'IndpntAsk'),
    token(b'\x7c', 'DependAuto'),
    token(b'\x7d', 'DependAsk'),
    token(b'\x7f', '<squaremark>'),
    token(b'\x80', '<crossmark>'),
    token(b'\x81', '<dotmark>'),
    MUL,
    DIV,
    token(b'\x84', 'Trace'),
    token(b'\x85', 'ClrDraw'),
    token(b'\x86', 'ZStandard'),
    token(b'\x87', 'ZTrig'),
    token(b'\x88', 'ZBox'),
    token(b'\x89', 'Zoom In'),
    token(b'\x8a', 'Zoom Out'),
    token(b'\x8b', 'ZSquare'),
    token(b'\x8c', 'ZInteger'),
    token(b'\x8d', 'ZPrevious'),
    token(b'\x8e', 'ZDecimal'),
    token(b'\x8f', 'ZoomStat'),
    token(b'\x90', 'ZoomRcl'),
    token(b'\x91', 'PrintScreen'),
    token(b'\x92', 'ZoomSto'),
    token(b'\x93', 'Text('),
    NPR,
    NCR,
    token(b'\x96', 'FnOn '),
    token(b'\x97', 'FnOff '),
    token(b'\x98', 'StorePic '),
    token(b'\x99', 'RecallPic '),
    token(b'\x9a', 'StoreGDB '),
    token(b'\x9b', 'RecallGDB '),
    token(b'\x9c', 'Line('),
    token(b'\x9d', 'Vertical '),
    token(b'\x9e', 'Pt-On('),
    token(b'\x9f', 'Pt-Off('),
    token(b'\xa0', 'Pt-Change('),
    token(b'\xa1', 'Pxl-On('),
    token(b'\xa2', 'Pxl-Off('),
    token(b'\xa3', 'Pxl-Change('),
    token(b'\xa4', 'Shade('),
    token(b'\xa5', 'Circle('),
    token(b'\xa6', 'Horizontal '),
    token(b'\xa7', 'Tangent('),
    token(b'\xa8', 'DrawInv'),
    token(b'\xa9', 'DrawF'),
    RAND,
    token(b'\xac', 'π', nullary=lambda env: math.pi),
    token(b'\xad', 'getKey', nullary=Environment.get_key),
    token(b'\xae', "'"),
    token(b'\xaf', '?'),
    NEG,
    token(b'\xb1', 'int(',    pure_func=purefunctions.int_),
    token(b'\xb2', 'abs(',    pure_func=purefunctions.abs),
    token(b'\xb3', 'det(',    pure_func=purefunctions.det),
    token(b'\xb4', 'identity(', pure_func=purefunctions.identity),
    DIM,
    token(b'\xb6', 'sum(',    pure_func=purefunctions.sum),
    token(b'\xb7', 'prod(',   pure_func=purefunctions.prod),
    token(b'\xb8', 'not(',    pure_func=purefunctions.not_),
    token(b'\xb9', 'iPart(',  pure_func=purefunctions.i_part),
    token(b'\xba', 'fPart(',  pure_func=purefunctions.f_part),
    token(b'\xbc', '√(', pure_func=purefunctions.sqrt),
    token(b'\xbd', '³√(',   pure_func=purefunctions.cbrt),
    token(b'\xbe', 'ln(',     pure_func=purefunctions.ln),
    token(b'\xbf', 'e^(',     pure_func=purefunctions.exp),
    token(b'\xc0', 'log(',    pure_func=purefunctions.log),
    token(b'\xc1', '10^(',    pure_func=purefunctions.pow10),
    token(b'\xc2', 'sin(',    pure_func=purefunctions.sin),
    token(b'\xc3', 'sin⁻¹(',  pure_func=purefunctions.asin),
    token(b'\xc4', 'cos(',    pure_func=purefunctions.cos),
    token(b'\xc5', 'cos⁻¹(',  pure_func=purefunctions.acos),
    token(b'\xc6', 'tan(',    pure_func=purefunctions.tan),
    token(b'\xc7', 'tan⁻¹(',  pure_func=purefunctions.atan),
    token(b'\xc8', 'sinh(',   pure_func=purefunctions.sinh),
    token(b'\xc9', 'sinh⁻¹(', pure_func=purefunctions.asinh),
    token(b'\xca', 'cosh(',   pure_func=purefunctions.cosh),
    token(b'\xcb', 'cosh⁻¹(', pure_func=purefunctions.acosh),
    token(b'\xcc', 'tanh(',   pure_func=purefunctions.tanh),
    token(b'\xcd', 'tanh⁻¹(', pure_func=purefunctions.atanh),
    token(b'\xce', 'If '),
    token(b'\xcf', 'Then'),
    token(b'\xd0', 'Else'),
    token(b'\xd1', 'While '),
    token(b'\xd2', 'Repeat '),
    token(b'\xd3', 'For('),
    token(b'\xd4', 'End'),
    token(b'\xd5', 'Return'),
    token(b'\xd6', 'Lbl '),
    token(b'\xd7', 'Goto '),
    token(b'\xd8', 'Pause '),
    token(b'\xd9', 'Stop'),
    token(b'\xda', 'IS>('),
    token(b'\xdb', 'DS<('),
    token(b'\xdc', 'Input '),
    token(b'\xdd', 'Prompt '),
    token(b'\xde', 'Disp '),
    token(b'\xdf', 'DispGraph'),
    token(b'\xe0', 'Output('),
    token(b'\xe1', 'ClrHome'),
    token(b'\xe2', 'Fill('),
    token(b'\xe3', 'SortA('),
    token(b'\xe4', 'SortD('),
    token(b'\xe5', 'DispTable'),
    token(b'\xe6', 'Menu('),
    token(b'\xe7', 'Send('),
    token(b'\xe8', 'Get('),
    token(b'\xe9', 'PlotsOn'),
    token(b'\xea', 'PlotsOff'),
    LIST_PREFIX,
    token(b'\xec', 'Plot1('),
    token(b'\xed', 'Plot2('),
    token(b'\xee', 'Plot3('),
    POW,
    XROOT,
    token(b'\xf2', '1-Var Stats '),
    token(b'\xf3', '2-Var Stats '),
    token(b'\xf4', 'LinReg(a+bx) '),
    token(b'\xf5', 'ExpReg '),
    token(b'\xf6', 'LnReg '),
    token(b'\xf7', 'PwrReg '),
    token(b'\xf8', 'Med-Med '),
    token(b'\xf9', 'QuadReg '),
    token(b'\xfa', 'ClrList '),
    token(b'\xfb', 'ClrTable'),
    token(b'\xfc', 'Histogram'),
    token(b'\xfd', 'xyLine'),
    token(b'\xfe', 'Scatter'),
    token(b'\xff', 'LinReg(ax+b) '),

    # Two-byte: Matrix variables 0x5C xx
    *[token(bytes([0x5c, i]), f'[{chr(0x41 + i)}]') for i in range(10)],

    # Two-byte: List variables 0x5D xx
    *[token(bytes([0x5d, i]), f'L{chr(0x2081 + i)}') for i in range(0, 6)],

    # Two-byte: Y= equation variables 0x5E xx
    *[token(bytes([0x5e, 0x10 + i]), f'Y{chr(0x2080 + (i + 1) % 10)}') for i in range(10)], 
    *[token(bytes([0x5e, 0x20 + i]), f'{x}{chr(0x2080 + n)}ₜ') for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))],

    *[token(bytes([0x5e, 0x40 + i]), f'r{chr(0x2081 + i)}') for i in range(6)],

    token(b'\x5e\x80', 'u'),
    token(b'\x5e\x81', 'v'),
    token(b'\x5e\x82', 'w'),

    # Two-byte: Picture variables 0x60 xx
    *[token(bytes([0x60, i]), f'Pic{(i + 1) % 10}') for i in range(10)],

    # Two-byte: GDB variables 0x61 xx
    *[token(bytes([0x61, i]), f'GDB{(i + 1) % 10}') for i in range(10)],

    # Two-byte: String variables 0xAA xx (Str1=0x00 … Str9=0x08, Str0=0x09)
    *[token(bytes([0xaa, i]), f'Str{(i + 1) % 10}') for i in range(10)],

    # Two-byte: Statistics variables 0x62 xx
    token(b'\x62\x01', 'RegEq'),
    token(b'\x62\x02', 'n'),
    token(b'\x62\x03', 'x̄'),
    token(b'\x62\x04', 'Σx'),
    token(b'\x62\x05', 'Σx²'),
    token(b'\x62\x06', 'Sx'),
    token(b'\x62\x07', 'σx'),
    token(b'\x62\x08', 'minX'),
    token(b'\x62\x09', 'maxX'),
    token(b'\x62\x0a', 'minY'),
    token(b'\x62\x0b', 'maxY'),
    token(b'\x62\x0c', 'ȳ'),
    token(b'\x62\x0d', 'Σy'),
    token(b'\x62\x0e', 'Σy²'),
    token(b'\x62\x0f', 'Sy'),
    token(b'\x62\x10', 'σy'),
    token(b'\x62\x11', 'Σxy'),
    token(b'\x62\x12', 'r'),
    token(b'\x62\x13', 'Med'),
    token(b'\x62\x14', 'Q1'),
    token(b'\x62\x15', 'Q3'),
    token(b'\x62\x16', 'a'),
    token(b'\x62\x17', 'b'),
    token(b'\x62\x18', 'c'),
    token(b'\x62\x19', 'd'),
    token(b'\x62\x1a', 'e'),
    token(b'\x62\x1b', 'x₁'),
    token(b'\x62\x1c', 'x₂'),
    token(b'\x62\x1d', 'x₃'),
    token(b'\x62\x1e', 'y₁'),
    token(b'\x62\x1f', 'y₂'),
    token(b'\x62\x20', 'y₃'),
    token(b'\x62\x21', 'n'),
    token(b'\x62\x22', 'p'),
    token(b'\x62\x23', 'z'),
    token(b'\x62\x24', 't'),
    token(b'\x62\x25', 'χ²'),
    token(b'\x62\x26', 'F'),
    token(b'\x62\x27', 'df'),
    token(b'\x62\x28', 'p̂'),
    token(b'\x62\x29', 'p̂₁'),
    token(b'\x62\x2a', 'p̂₂'),
    token(b'\x62\x2b', 'x̄₁'),
    token(b'\x62\x2c', 'Sx₁'),
    token(b'\x62\x2d', 'n₁'),
    token(b'\x62\x2e', 'x̄₂'),
    token(b'\x62\x2f', 'Sx₂'),
    token(b'\x62\x30', 'n₂'),
    token(b'\x62\x31', 'Sxp'),
    token(b'\x62\x32', 'lower'),
    token(b'\x62\x33', 'upper'),
    token(b'\x62\x34', 's'),
    token(b'\x62\x35', 'r²'),
    token(b'\x62\x36', 'R²'),
    token(b'\x62\x37', 'Factor df'),
    token(b'\x62\x38', 'Factor SS'),
    token(b'\x62\x39', 'Factor MS'),
    token(b'\x62\x3a', 'Error df'),
    token(b'\x62\x3b', 'Error SS'),
    token(b'\x62\x3c', 'Error MS'),

    # Two-byte: Window / Finance variables 0x63 xx
    token(b'\x63\x02', 'Xscl'),
    token(b'\x63\x03', 'Yscl'),
    token(b'\x63\x0a', 'Xmin'),
    token(b'\x63\x0b', 'Xmax'),
    token(b'\x63\x0c', 'Ymin'),
    token(b'\x63\x0d', 'Ymax'),
    token(b'\x63\x0e', 'Tmin'),
    token(b'\x63\x0f', 'Tmax'),
    token(b'\x63\x10', 'θmin'),
    token(b'\x63\x11', 'θmax'),
    token(b'\x63\x1a', 'TblStart'),
    token(b'\x63\x1b', 'PlotStart'),
    token(b'\x63\x1d', 'nMax'),
    token(b'\x63\x1f', 'nMin'),
    token(b'\x63\x21', 'ΔTbl'),
    token(b'\x63\x22', 'Tstep'),
    token(b'\x63\x23', 'θstep'),
    token(b'\x63\x26', 'ΔX'),
    token(b'\x63\x27', 'ΔY'),
    token(b'\x63\x28', 'XFact'),
    token(b'\x63\x29', 'YFact'),
    token(b'\x63\x2b', 'N'),
    token(b'\x63\x2c', 'I%'),
    token(b'\x63\x2d', 'PV'),
    token(b'\x63\x2e', 'PMT'),
    token(b'\x63\x2f', 'FV'),
    token(b'\x63\x30', 'P/Y'),
    token(b'\x63\x31', 'C/Y'),
    token(b'\x63\x34', 'PlotStep'),
    token(b'\x63\x36', 'Xres'),

    # Two-byte: Graph format tokens 0x7E xx
    token(b'\x7e\x00', 'Sequential'),
    token(b'\x7e\x01', 'Simul'),
    token(b'\x7e\x02', 'PolarGC'),
    token(b'\x7e\x03', 'RectGC'),
    token(b'\x7e\x04', 'CoordOn'),
    token(b'\x7e\x05', 'CoordOff'),
    token(b'\x7e\x06', 'Connected'),
    token(b'\x7e\x07', 'Dot'),
    token(b'\x7e\x08', 'AxesOn'),
    token(b'\x7e\x09', 'AxesOff'),
    token(b'\x7e\x0a', 'GridOn'),
    token(b'\x7e\x0b', 'GridOff'),
    token(b'\x7e\x0c', 'LabelOn'),
    token(b'\x7e\x0d', 'LabelOff'),
    token(b'\x7e\x0e', 'Web'),
    token(b'\x7e\x0f', 'Time'),
    token(b'\x7e\x10', 'uvAxes'),
    token(b'\x7e\x11', 'vwAxes'),
    token(b'\x7e\x12', 'uwAxes'),

    # Two-byte: Miscellaneous tokens 0xBB xx
    token(b'\xbb\x00', 'npv('),
    token(b'\xbb\x01', 'irr('),
    token(b'\xbb\x02', 'bal('),
    token(b'\xbb\x03', 'Σprn('),
    token(b'\xbb\x04', 'ΣInt('),
    token(b'\xbb\x05', '►Nom('),
    token(b'\xbb\x06', '►Eff('),
    token(b'\xbb\x07', 'dbd('),
    token(b'\xbb\x08', 'lcm(',      pure_func=purefunctions.lcm),
    token(b'\xbb\x09', 'gcd(',      pure_func=purefunctions.gcd),
    token(b'\xbb\x0a', 'randInt(',  pure_func=purefunctions.randint),
    token(b'\xbb\x0b', 'randBin(', pure_func=purefunctions.rand_bin),
    token(b'\xbb\x0c', 'sub(',      pure_func=purefunctions.sub),
    token(b'\xbb\x0d', 'stdDev('),
    token(b'\xbb\x0e', 'variance('),
    token(b'\xbb\x0f', 'inString(', pure_func=purefunctions.in_string),
    token(b'\xbb\x10', 'normalcdf('),
    token(b'\xbb\x11', 'invNorm('),
    token(b'\xbb\x12', 'tcdf('),
    token(b'\xbb\x13', 'χ²cdf('),
    token(b'\xbb\x14', 'Fcdf('),
    token(b'\xbb\x15', 'binompdf('),
    token(b'\xbb\x16', 'binomcdf('),
    token(b'\xbb\x17', 'poissonpdf('),
    token(b'\xbb\x18', 'poissoncdf('),
    token(b'\xbb\x19', 'geometpdf('),
    token(b'\xbb\x1a', 'geometcdf('),
    token(b'\xbb\x1b', 'normalpdf('),
    token(b'\xbb\x1c', 'tpdf('),
    token(b'\xbb\x1d', 'χ²pdf('),
    token(b'\xbb\x1e', 'Fpdf('),
    token(b'\xbb\x1f', 'randNorm(', pure_func=purefunctions.randnorm),
    token(b'\xbb\x20', 'tvm_Pmt'),
    token(b'\xbb\x21', 'tvm_I%'),
    token(b'\xbb\x22', 'tvm_PV'),
    token(b'\xbb\x23', 'tvm_N'),
    token(b'\xbb\x24', 'tvm_FV'),
    token(b'\xbb\x25', 'conj(',   pure_func=purefunctions.conj),
    token(b'\xbb\x26', 'real(',   pure_func=purefunctions.real),
    token(b'\xbb\x27', 'imag(',   pure_func=purefunctions.imag),
    token(b'\xbb\x28', 'angle(',  pure_func=purefunctions.angle),
    token(b'\xbb\x29', 'cumSum(', pure_func=purefunctions.cum_sum),
    token(b'\xbb\x2a', 'expr('),
    token(b'\xbb\x2b', 'length(', pure_func=purefunctions.length),
    token(b'\xbb\x2c', 'ΔList(', pure_func=purefunctions.delta_list),
    token(b'\xbb\x2d', 'ref('),
    token(b'\xbb\x2e', 'rref('),
    token(b'\xbb\x2f', '►Rect'),
    token(b'\xbb\x30', '►Polar'),
    token(b'\xbb\x31', '𝑒', nullary=lambda env: math.e),
    token(b'\xbb\x32', 'SinReg '),
    token(b'\xbb\x33', 'Logistic '),
    token(b'\xbb\x34', 'LinRegTTest '),
    token(b'\xbb\x35', 'ShadeNorm('),
    token(b'\xbb\x36', 'Shade_t('),
    token(b'\xbb\x37', 'Shadeχ²('),
    token(b'\xbb\x38', 'ShadeF('),
    token(b'\xbb\x39', 'Matr►list(', cmd=forms.matr_to_list),
    token(b'\xbb\x3a', 'List►matr(', cmd=forms.list_to_matr),
    token(b'\xbb\x3b', 'Z-Test('),
    token(b'\xbb\x3c', 'T-Test'),
    token(b'\xbb\x3d', '2-SampZTest('),
    token(b'\xbb\x3e', '1-PropZTest('),
    token(b'\xbb\x3f', '2-PropZTest('),
    token(b'\xbb\x40', 'χ²-Test('),
    token(b'\xbb\x41', 'ZInterval '),
    token(b'\xbb\x42', '2-SampZInt('),
    token(b'\xbb\x43', '1-PropZInt('),
    token(b'\xbb\x44', '2-PropZInt('),
    token(b'\xbb\x45', 'GraphStyle('),
    token(b'\xbb\x46', '2-SampTTest '),
    token(b'\xbb\x47', '2-SampFTest '),
    token(b'\xbb\x48', 'TInterval '),
    token(b'\xbb\x49', '2-SampTInt '),
    token(b'\xbb\x4a', 'SetUpEditor '),
    token(b'\xbb\x4b', 'Pmt_End'),
    token(b'\xbb\x4c', 'Pmt_Bgn'),
    token(b'\xbb\x4d', 'Real'),
    token(b'\xbb\x4e', 're^θi'),
    token(b'\xbb\x4f', 'a+bi'),
    token(b'\xbb\x50', 'ExprOn'),
    token(b'\xbb\x51', 'ExprOff'),
    token(b'\xbb\x52', 'ClrAllLists'),
    token(b'\xbb\x53', 'GetCalc('),
    token(b'\xbb\x54', 'DelVar '),
    token(b'\xbb\x55', 'Equ►String('),
    token(b'\xbb\x56', 'String►Equ('),
    token(b'\xbb\x57', 'Clear Entries'),
    token(b'\xbb\x58', 'Select('),
    token(b'\xbb\x59', 'ANOVA('),
    token(b'\xbb\x5a', 'ModBoxplot'),
    token(b'\xbb\x5b', 'NormProbPlot'),
    token(b'\xbb\x64', 'G-T'),
    token(b'\xbb\x65', 'ZoomFit'),
    token(b'\xbb\x66', 'DiagnosticOn'),
    token(b'\xbb\x67', 'DiagnosticOff'),
    token(b'\xbb\x68', 'Archive '),
    token(b'\xbb\x69', 'UnArchive '),
    token(b'\xbb\x6a', 'Asm('),
    token(b'\xbb\x6b', 'AsmComp('),
    token(b'\xbb\x6c', 'AsmPrgm'),
    token(b'\xbb\x6d', '<compiledasm>'),
	token(b'\xbb\x6e', 'Á'),
	token(b'\xbb\x6f', 'À'),
	token(b'\xbb\x70', 'Â'),
	token(b'\xbb\x71', 'Ä'),
	token(b'\xbb\x72', 'á'),
	token(b'\xbb\x73', 'à'),
	token(b'\xbb\x74', 'â'),
	token(b'\xbb\x75', 'ä'),
	token(b'\xbb\x76', 'É'),
	token(b'\xbb\x77', 'È'),
	token(b'\xbb\x78', 'Ê'),
	token(b'\xbb\x79', 'Ë'),
	token(b'\xbb\x7a', 'é'),
	token(b'\xbb\x7b', 'è'),
	token(b'\xbb\x7c', 'ê'),
	token(b'\xbb\x7d', 'ë'),
	token(b'\xbb\x7f', 'Ì'),
	token(b'\xbb\x80', 'Î'),
	token(b'\xbb\x81', 'Ï'),
	token(b'\xbb\x82', 'í'),
	token(b'\xbb\x83', 'ì'),
	token(b'\xbb\x84', 'î'),
	token(b'\xbb\x85', 'ï'),
	token(b'\xbb\x86', 'Ó'),
	token(b'\xbb\x87', 'Ò'),
	token(b'\xbb\x88', 'Ô'),
	token(b'\xbb\x89', 'Ö'),
	token(b'\xbb\x8a', 'ó'),
	token(b'\xbb\x8b', 'ò'),
	token(b'\xbb\x8c', 'ô'),
	token(b'\xbb\x8d', 'ö'),
	token(b'\xbb\x8e', 'Ú'),
	token(b'\xbb\x8f', 'Ù'),
	token(b'\xbb\x90', 'Û'),
	token(b'\xbb\x91', 'Ü'),
	token(b'\xbb\x92', 'ú'),
	token(b'\xbb\x93', 'ù'),
	token(b'\xbb\x94', 'û'),
	token(b'\xbb\x95', 'ü'),
	token(b'\xbb\x96', 'Ç'),
	token(b'\xbb\x97', 'ç'),
	token(b'\xbb\x98', 'Ñ'),
	token(b'\xbb\x99', 'ñ'),
    token(b'\xbb\x9a', '´'),
    token(b'\xbb\x9b', '`'),
    token(b'\xbb\x9c', '¨'),
    token(b'\xbb\x9d', '¿'),
    token(b'\xbb\x9e', '¡'),
    token(b'\xbb\x9f', 'α'),
    token(b'\xbb\xa0', 'β'),
    token(b'\xbb\xa1', 'γ'),
    token(b'\xbb\xa2', 'Δ'),
    token(b'\xbb\xa3', 'δ'),
    token(b'\xbb\xa4', 'ε'),
    token(b'\xbb\xa5', 'λ'),
    token(b'\xbb\xa6', 'μ'),
    token(b'\xbb\xa7', 'π'),
    token(b'\xbb\xa8', 'ρ'),
    token(b'\xbb\xa9', 'Σ'),
    token(b'\xbb\xab', 'φ'),
    token(b'\xbb\xac', 'Ω'),
    token(b'\xbb\xad', 'ψ'),
    token(b'\xbb\xae', 'χ'),
    token(b'\xbb\xaf', 'F'),
	token(b'\xbb\xb0', 'a'),
	token(b'\xbb\xb1', 'b'),
	token(b'\xbb\xb2', 'c'),
	token(b'\xbb\xb3', 'd'),
	token(b'\xbb\xb4', 'e'),
	token(b'\xbb\xb5', 'f'),
	token(b'\xbb\xb6', 'g'),
	token(b'\xbb\xb7', 'h'),
	token(b'\xbb\xb8', 'i'),
	token(b'\xbb\xb9', 'j'),
	token(b'\xbb\xba', 'k'),
	token(b'\xbb\xbc', 'l'),
	token(b'\xbb\xbd', 'm'),
	token(b'\xbb\xbe', 'n'),
	token(b'\xbb\xbf', 'o'),
	token(b'\xbb\xc0', 'p'),
	token(b'\xbb\xc1', 'q'),
	token(b'\xbb\xc2', 'r'),
	token(b'\xbb\xc3', 's'),
	token(b'\xbb\xc4', 't'),
	token(b'\xbb\xc5', 'u'),
	token(b'\xbb\xc6', 'v'),
	token(b'\xbb\xc7', 'w'),
	token(b'\xbb\xc8', 'x'),
	token(b'\xbb\xc9', 'y'),
	token(b'\xbb\xca', 'z'),
    token(b'\xbb\xcb', 'σ'),
    token(b'\xbb\xcc', 'τ'),
    token(b'\xbb\xcd', 'Í'),
    token(b'\xbb\xce', 'GarbageCollect'),
    token(b'\xbb\xcf', '~'),
    token(b'\xbb\xd1', '@'),
    token(b'\xbb\xd2', '#'),
    token(b'\xbb\xd3', '$'),
    token(b'\xbb\xd4', '&'),
    token(b'\xbb\xd5', '`'),
    token(b'\xbb\xd6', ';'),
    token(b'\xbb\xd7', '\\'),
    token(b'\xbb\xd8', '|'),
    token(b'\xbb\xd9', '_'),
    token(b'\xbb\xda', '%'),
    token(b'\xbb\xdb', '…'),
    token(b'\xbb\xdc', '∠'),
    token(b'\xbb\xdd', 'ß'),
    token(b'\xbb\xde', 'x'),
    token(b'\xbb\xdf', 'T'),
	token(b'\xbb\xe0', '₀'),
	token(b'\xbb\xe1', '₁'),
	token(b'\xbb\xe2', '₂'),
	token(b'\xbb\xe3', '₃'),
	token(b'\xbb\xe4', '₄'),
	token(b'\xbb\xe5', '₅'),
	token(b'\xbb\xe6', '₆'),
	token(b'\xbb\xe7', '₇'),
	token(b'\xbb\xe8', '₈'),
	token(b'\xbb\xe9', '₉'),
	token(b'\xbb\xea', '₁₀'),
    token(b'\xbb\xeb', '←'),
    token(b'\xbb\xec', '→'),
    token(b'\xbb\xed', '↑'),
    token(b'\xbb\xee', '↓'),
    token(b'\xbb\xf0', 'x'),
    token(b'\xbb\xf1', '∫'),
    token(b'\xbb\xf2', '🡅'),
    token(b'\xbb\xf3', '🡇'),
    token(b'\xbb\xf4', '√'),
    token(b'\xbb\xf5', '<funcon>'),

    # Two-byte: TI-84+ extended tokens 0xEF xx
    token(b'\xef\x00', 'setDate('),
    token(b'\xef\x01', 'setTime('),
    token(b'\xef\x02', 'checkTmr('),
    token(b'\xef\x03', 'setDtFmt('),
    token(b'\xef\x04', 'setTmFmt('),
    token(b'\xef\x05', 'timeCnv('),
    token(b'\xef\x06', 'dayOfWk('),
    token(b'\xef\x07', 'getDtStr('),
    token(b'\xef\x08', 'getTmStr('),
    token(b'\xef\x09', 'getDate',   nullary=Environment.get_date),
    token(b'\xef\x0a', 'getTime',   nullary=Environment.get_time),
    token(b'\xef\x0b', 'startTmr',  nullary=Environment.start_tmr),
    token(b'\xef\x0c', 'getDtFmt',  nullary=Environment.get_dt_fmt),
    token(b'\xef\x0d', 'getTmFmt',  nullary=Environment.get_tm_fmt),
    token(b'\xef\x0e', 'isClockOn', nullary=Environment.is_clock_on),
    token(b'\xef\x0f', 'ClockOff'),
    token(b'\xef\x10', 'ClockOn'),
    token(b'\xef\x11', 'OpenLib('),
    token(b'\xef\x12', 'ExecLib'),
    token(b'\xef\x13', 'invT('),
    token(b'\xef\x14', 'χ²GOF-Test('),
    token(b'\xef\x15', 'LinRegTInt '),
    token(b'\xef\x16', 'Manual-Fit '),
    token(b'\xef\x17', 'ZQuadrant1'),
    token(b'\xef\x18', 'ZFrac1/2'),
    token(b'\xef\x19', 'ZFrac1/3'),
    token(b'\xef\x1a', 'ZFrac1/4'),
    token(b'\xef\x1b', 'ZFrac1/5'),
    token(b'\xef\x1c', 'ZFrac1/8'),
    token(b'\xef\x1d', 'ZFrac1/10'),
    token(b'\xef\x1e', '<mathprintbox>'),
    token(b'\xef\x30', '►n/d◄►Un/d'),
    token(b'\xef\x31', '►F◄►D'),
    token(b'\xef\x32', 'remainder(',    pure_func=purefunctions.remainder),
    token(b'\xef\x33', 'Σ(', func=forms.sigma),
    token(b'\xef\x34', 'logBASE(',      pure_func=purefunctions.logbase),
    token(b'\xef\x35', 'randIntNoRep(', pure_func=purefunctions.randintnotrep),
    token(b'\xef\x36', 'MATHPRINT'),
    token(b'\xef\x37', 'CLASSIC'),
    token(b'\xef\x38', 'n/d'),
    token(b'\xef\x39', 'Un/d'),
    token(b'\xef\x3a', 'AUTO'),
    token(b'\xef\x3b', 'DEC'),
    token(b'\xef\x3c', 'FRAC'),
    token(b'\xef\x3d', 'FRAC-APPROX'),
]


class TokenTable:

    def __init__(self, tokens):
        self._table: list[Token | list[Token | None] | None] = [None] * 256
        for token in tokens:
            b0 = token.code[0]
            if len(token.code) == 1:
                self._table[b0] = token
            else:
                sub = self._table[b0]
                if sub is None:
                    self._table[b0] = sub = []
                b1 = token.code[1]
                if b1 >= len(sub):
                    sub.extend([None] * (b1 + 1 - len(sub)))
                sub[b1] = token

    def __getitem__(self, code: int | Sequence[int]) -> Token:
        if isinstance(code, int):
            code = (code,)
        b0 = code[0]
        if len(code) == 1:
            table = self._table
            idx = b0
        else:
            sub = self._table[b0]
            if sub is None:
                raise KeyError(code)
            table = sub
            idx = code[1]
            
        token = table[idx] if idx < len(table) else None
        if token is None:
            raise KeyError(code)
        return token
    
    def __repr__(self):
        return repr(self._table)


TOKEN_TABLE = TokenTable(TOKENS)


if __name__ == '__main__':
    @dataclass
    class NullToken:
        code: bytes

    check = [None] * 0x100
    check_misc = [None] * 0xF6
    duplicate = set()

    for code in [
        b'\x00', b'\x26', b'\x5c', b'\x5d', b'\x5e', b'\x60', b'\x61', b'\x62', b'\x63', b'\x7e', b'\xaa', b'\xbb', b'\xef',
        b'\xbb\x5c', b'\xbb\x5d', b'\xbb\x5e', b'\xbb\x5f', b'\xbb\x60', b'\xbb\x61', b'\xbb\x62', b'\xbb\x63', b'\xbb\x7e', b'\xbb\xaa', b'\xbb\xbb', b'\xbb\xd0', b'\xbb\xef',
    ]:
        old_len = len(duplicate)
        duplicate.add(code)
        if old_len == len(duplicate):
            raise ValueError(f'Duplicate: {token}')
        if len(code) == 1:
            check[code[0]] = NullToken(code)
        elif code[0] == 0xBB:
            check_misc[code[1]] = NullToken(code)

    for token in TOKENS:
        old_len = len(duplicate)
        duplicate.add(token.code)
        if old_len == len(duplicate):
            raise ValueError(f'Duplicate: {token}')
        if len(token.code) == 1:
            check[token.code[0]] = token
        elif token.code[0] == 0xBB:
            check_misc[token.code[1]] = token

    for i, token in enumerate(check):
        if token is None:
            print('MISSING:', hex(i))

    for i, token in enumerate(check_misc):
        if token is None:
            print('MISSING:', hex(0xBB00 + i))

    print(TOKEN_TABLE)
