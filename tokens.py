from collections.abc import Callable, Sequence
from dataclasses import dataclass
import operator as op
from typing import Any
import math, itertools
from environment import (
	Environment, Variable, NumericVar, ListVar, MatrixVar, StringVar, StatVar, WindowVar,
)
import purefunctions as pf
import forms



@dataclass(slots=True, eq=False)
class Token:
	code: bytes
	ascii: str | None
	text: str
	bp: tuple[int, int] | None = None # (left_bp, right_bp) for binary operators
	operator:  Callable | None = None # (lhs, rhs) -> value
	postfix:   Callable | None = None # (operand) -> value (prefix or postfix)
	function:  Callable | None = None # (ArgParser) -> value for function tokens
	command:   Callable | None = None # (ArgParser) -> None for command tokens
	nullary:   Callable | None = None # (env) -> value for read-only computed constants
	converter: Callable | None = None # (value) -> value for ►DMS, ►Dec, ►Frac and others
	variable:  Callable | None = None # Variable instance for storable typed variables

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
			or self.function is not None
			or self in {DOT, L_PAREN, L_BRACE, L_BRACKET, QUOTE, NEG, LIST_PREFIX}
		)

	def __repr__(self):
		return f'{self.code.hex().upper()}:{self.text}'


# TODO: remove code after transition to flags
EOF_TOKEN = Token(b'\x00', None, '<END-OF-INPUT>')


ALL_TOKENS: list['Token'] = []
_SEEN: set[bytes] = set()

def _make_pure_func(f):
	def wrapper(a):
		return f(*a.parse_args())
	return wrapper

def _make_variable(code: bytes) -> Variable | None:
	b0 = code[0]
	if 0x41 <= b0 <= 0x5b: # A–Z, θ
		return NumericVar(b0 - 0x41)
	if b0 == 0x5d:         # L1–L6
		return ListVar(code[1])
	if b0 == 0x5c:         # [A]–[J]
		return MatrixVar(code[1])
	if b0 == 0xaa:         # Str0–9
		return StringVar(code[1])
	if b0 == 0x62:         # stat vars
		return StatVar(code[1])
	if b0 == 0x63:         # window vars
		return WindowVar(code[1])
	return None


def token(
	code: bytes,
	text: str = None,
	ascii: str = None,
	*,
	bp: tuple[int, int] | None = None,
	operator:  Callable | None = None,
	postfix:   Callable | None = None,
	func:      Callable | None = None,
	pure_func: Callable | None = None,
	cmd:       Callable | None = None,
	nullary:   Callable | None = None,
	converter: Callable | None = None,
	variable:  Variable | None = None,
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

	t = Token(code, ascii, text or ascii, bp, operator, postfix, func, cmd, nullary, converter, variable)
	ALL_TOKENS.append(t)
	return t


# ── 0x01–0x09: converters and delimiters ─────────────────────────────────────

TO_DMS      = token(b'\x01', '►DMS',  converter=pf.to_dms)
TO_DEC      = token(b'\x02', '►Dec',  converter=pf.to_dec)
TO_FRAC     = token(b'\x03', '►Frac', converter=pf.to_frac)
STORE       = token(b'\x04', '→')
Boxplot     = token(b'\x05', 'Boxplot')
L_BRACKET   = token(b'\x06', ascii='[')
R_BRACKET   = token(b'\x07', ascii=']')
L_BRACE     = token(b'\x08', ascii='{')
R_BRACE     = token(b'\x09', ascii='}')

# ── 0x0A–0x0F: postfix operators ─────────────────────────────────────────────

RAD         = token(b'\x0a', 'ʳ')           # postfix; needs env, handled specially
DEG         = token(b'\x0b', '°')           # ditto
INV         = token(b'\x0c', '¹',  postfix=pf.inv)
SQ          = token(b'\x0d', '²',  postfix=lambda x: x**2)
TRANSPOSE   = token(b'\x0e', 'ᵀ',  postfix=pf.transpose)
CUBE        = token(b'\x0f', '³',  postfix=lambda x: x**3)

# ── 0x10–0x2F: function tokens ────────────────────────────────────────────────

L_PAREN     = token(b'\x10', ascii='(')
R_PAREN     = token(b'\x11', ascii=')')
Round       = token(b'\x12', 'round(',       pure_func=pf.round)
PxlTest     = token(b'\x13', 'pxl-Test(')
augment     = token(b'\x14', 'augment(',     pure_func=pf.augment)
rowSwap     = token(b'\x15', 'rowSwap(',     pure_func=pf.rowswap)
rowPlus     = token(b'\x16', 'row+(',        pure_func=pf.row_plus)
timesRow    = token(b'\x17', '*row(',        pure_func=pf.times_row)
timesRowPlus= token(b'\x18', '*row+(',       pure_func=pf.times_row_plus)
Max         = token(b'\x19', 'max(',         pure_func=pf.max)
Min         = token(b'\x1a', 'min(',         pure_func=pf.min)
R_Pr        = token(b'\x1b', 'R►Pr(',        pure_func=pf.r_pr)
R_Ptheta    = token(b'\x1c', 'R►Pθ(',        pure_func=pf.r_ptheta)
P_Rx        = token(b'\x1d', 'P►Rx(',        pure_func=pf.p_rx)
P_Ry        = token(b'\x1e', 'P►Ry(',        pure_func=pf.p_ry)
median      = token(b'\x1f', 'median(',      pure_func=pf.median)
randM       = token(b'\x20', 'randM(',       pure_func=pf.rand_m)
mean        = token(b'\x21', 'mean(',        pure_func=pf.mean)
solve       = token(b'\x22', 'solve(')
seq         = token(b'\x23', 'seq(',         func=forms.seq)
fnInt       = token(b'\x24', 'fnInt(',       func=forms.fn_int)
nDeriv      = token(b'\x25', 'nDeriv(',      func=forms.n_deriv)
fMin        = token(b'\x27', 'fMin(')
fMax        = token(b'\x28', 'fMax(')
SPACE       = token(b'\x29', ascii=' ')
QUOTE       = token(b'\x2a', ascii='"')
COMMA       = token(b'\x2b', ascii=',')
IMAG_I      = token(b'\x2c', '𝑖', nullary=lambda env: 1j)
FACT        = token(b'\x2d', ascii='!',      postfix=pf.factorial)
CubicReg    = token(b'\x2e', 'CubicReg ')
QuartReg    = token(b'\x2f', 'QuartReg ')

# ── 0x30–0x39: digits ─────────────────────────────────────────────────────────

D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = DIGITS = tuple(
	token(bytes([0x30 + i]), ascii=chr(0x30 + i)) for i in range(10)
)

# ── 0x3A–0x3F: punctuation and low-precedence binary operators ────────────────

DOT         = token(b'\x3a', ascii='.')
SCI_E       = token(b'\x3b', 'ᴇ')
or_         = token(b'\x3c', ' or ',    bp=(20, 21), operator=pf.or_)
xor         = token(b'\x3d', ' xor ',   bp=(20, 21), operator=pf.xor)
COLON       = token(b'\x3e', ascii=':')
NEWLINE     = token(b'\x3f', ascii='\n')

# ── 0x40: and ─────────────────────────────────────────────────────────────────

and_        = token(b'\x40', ' and ',   bp=(30, 31), operator=pf.and_)

# ── 0x41–0x5B: numeric variables (A–Z, θ) ────────────────────────────────────

(VAR_A, VAR_B, VAR_C, VAR_D, VAR_E, VAR_F, VAR_G, VAR_H, VAR_I, VAR_J,
 VAR_K, VAR_L, VAR_M, VAR_N, VAR_O, VAR_P, VAR_Q, VAR_R, VAR_S, VAR_T,
 VAR_U, VAR_V, VAR_W, VAR_X, VAR_Y, VAR_Z) = LETTERS = tuple(
	token(bytes([0x41 + i]), ascii=chr(0x41 + i)) for i in range(26)
)
VAR_THETA = token(b'\x5b', 'θ')

# ── 0x5C xx: matrix variables ([A]–[J]) ──────────────────────────────────────

MAT_A, MAT_B, MAT_C, MAT_D, MAT_E, MAT_F, MAT_G, MAT_H, MAT_I, MAT_J = MATRICES = tuple(
	token(bytes([0x5c, i]), f'[{chr(0x41 + i)}]') for i in range(10)
)

# ── 0x5D xx: list variables (L1–L6) ──────────────────────────────────────────

L1, L2, L3, L4, L5, L6 = LISTS = tuple(
	token(bytes([0x5d, i]), f'L{chr(0x2081 + i)}') for i in range(6)
)

# ── 0x5E xx: equation and sequence variables ──────────────────────────────────

Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y0 = Y_EQUATIONS = tuple(
	token(bytes([0x5e, 0x10 + i]), f'Y{chr(0x2080 + (i + 1) % 10)}') for i in range(10)
)
X1t, Y1t, X2t, Y2t, X3t, Y3t, X4t, Y4t, X5t, Y5t, X6t, Y6t = PARAM_EQUATIONS = tuple(
	token(bytes([0x5e, 0x20 + i]), f'{x}{chr(0x2080 + n)}ₜ')
	for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))
)
r1, r2, r3, r4, r5, r6 = POLAR_EQUATIONS = tuple(
	token(bytes([0x5e, 0x40 + i]), f'r{chr(0x2081 + i)}') for i in range(6)
)
u_seq = token(b'\x5e\x80', '𝑢')
v_seq = token(b'\x5e\x81', '𝑣')
w_seq = token(b'\x5e\x82', '𝑤')

# ── 0x5F: program token ───────────────────────────────────────────────────────

prgm        = token(b'\x5f', 'prgm')

# ── 0x60 xx: picture variables (Pic1–Pic0) ───────────────────────────────────

PICTURES = tuple(token(bytes([0x60, i]), f'Pic{(i + 1) % 10}') for i in range(10))

# ── 0x61 xx: graph database variables (GDB1–GDB0) ────────────────────────────

GDBS     = tuple(token(bytes([0x61, i]), f'GDB{(i + 1) % 10}') for i in range(10))

# ── 0x62 xx: statistical result variables ─────────────────────────────────────

RegEq    = token(b'\x62\x01', 'RegEq')
StatN    = token(b'\x62\x02', 'n')
XBar     = token(b'\x62\x03', 'ẍ')
SumX     = token(b'\x62\x04', 'Σx')
SumX2    = token(b'\x62\x05', 'Σx²')
Sx       = token(b'\x62\x06', 'Sx')
SigmaX   = token(b'\x62\x07', 'σx')
MinX     = token(b'\x62\x08', 'minX')
MaxX     = token(b'\x62\x09', 'maxX')
MinY     = token(b'\x62\x0a', 'minY')
MaxY     = token(b'\x62\x0b', 'maxY')
YBar     = token(b'\x62\x0c', 'ȳ')
SumY     = token(b'\x62\x0d', 'Σy')
SumY2    = token(b'\x62\x0e', 'Σy²')
Sy       = token(b'\x62\x0f', 'Sy')
SigmaY   = token(b'\x62\x10', 'σy')
SumXY    = token(b'\x62\x11', 'Σxy')
StatR    = token(b'\x62\x12', 'r')
StatMed  = token(b'\x62\x13', 'Med')
Q1       = token(b'\x62\x14', 'Q1')
Q3       = token(b'\x62\x15', 'Q3')
StatA    = token(b'\x62\x16', 'a')
StatB    = token(b'\x62\x17', 'b')
StatC    = token(b'\x62\x18', 'c')
StatD    = token(b'\x62\x19', 'd')
StatE    = token(b'\x62\x1a', 'e')
StatX1   = token(b'\x62\x1b', 'x₁')
StatX2   = token(b'\x62\x1c', 'x₂')
StatX3   = token(b'\x62\x1d', 'x₃')
StatY1   = token(b'\x62\x1e', 'y₁')
StatY2   = token(b'\x62\x1f', 'y₂')
StatY3   = token(b'\x62\x20', 'y₃')
StatN2   = token(b'\x62\x21', '𝑛')
StatP    = token(b'\x62\x22', 'p')
StatZ    = token(b'\x62\x23', 'z')
StatT    = token(b'\x62\x24', 't')
ChiSqStat= token(b'\x62\x25', 'χ²')
StatF    = token(b'\x62\x26', '𝐅')
StatDF   = token(b'\x62\x27', 'df')
Phat     = token(b'\x62\x28', 'ṕ')
Phat1    = token(b'\x62\x29', 'ṕ₁')
Phat2    = token(b'\x62\x2a', 'ṕ₂')
XBar1    = token(b'\x62\x2b', 'ẍ₁')
Sx1      = token(b'\x62\x2c', 'Sx₁')
N1       = token(b'\x62\x2d', 'n₁')
XBar2    = token(b'\x62\x2e', 'ẍ₂')
Sx2      = token(b'\x62\x2f', 'Sx₂')
N2       = token(b'\x62\x30', 'n₂')
Sxp      = token(b'\x62\x31', 'Sxp')
Lower    = token(b'\x62\x32', 'lower')
Upper    = token(b'\x62\x33', 'upper')
StatS    = token(b'\x62\x34', 's')
R2       = token(b'\x62\x35', 'r²')
CapR2    = token(b'\x62\x36', 'R²')
FactorDF = token(b'\x62\x37', 'Factor df')
FactorSS = token(b'\x62\x38', 'Factor SS')
FactorMS = token(b'\x62\x39', 'Factor MS')
ErrorDF  = token(b'\x62\x3a', 'Error df')
ErrorSS  = token(b'\x62\x3b', 'Error SS')
ErrorMS  = token(b'\x62\x3c', 'Error MS')

# ── 0x63 xx: window and finance variables ─────────────────────────────────────

Xscl      = token(b'\x63\x02', 'Xscl')
Yscl      = token(b'\x63\x03', 'Yscl')
Xmin      = token(b'\x63\x0a', 'Xmin')
Xmax      = token(b'\x63\x0b', 'Xmax')
Ymin      = token(b'\x63\x0c', 'Ymin')
Ymax      = token(b'\x63\x0d', 'Ymax')
Tmin      = token(b'\x63\x0e', 'Tmin')
Tmax      = token(b'\x63\x0f', 'Tmax')
ThetaMin  = token(b'\x63\x10', 'θmin')
ThetaMax  = token(b'\x63\x11', 'θmax')
TblStart  = token(b'\x63\x1a', 'TblStart')
PlotStart = token(b'\x63\x1b', 'PlotStart')
NMax      = token(b'\x63\x1d', 'nMax')
NMin      = token(b'\x63\x1f', 'nMin')
DeltaTbl  = token(b'\x63\x21', 'ΔTbl')
Tstep     = token(b'\x63\x22', 'Tstep')
ThetaStep = token(b'\x63\x23', 'θstep')
DeltaX    = token(b'\x63\x26', 'ΔX')
DeltaY    = token(b'\x63\x27', 'ΔY')
XFact     = token(b'\x63\x28', 'XFact')
YFact     = token(b'\x63\x29', 'YFact')
FinN      = token(b'\x63\x2b', '𝐍')
FinI      = token(b'\x63\x2c', 'I%')
FinPV     = token(b'\x63\x2d', 'PV')
FinPMT    = token(b'\x63\x2e', 'PMT')
FinFV     = token(b'\x63\x2f', 'FV')
FinPY     = token(b'\x63\x30', 'P/Y')
FinCY     = token(b'\x63\x31', 'C/Y')
PlotStep  = token(b'\x63\x34', 'PlotStep')
Xres      = token(b'\x63\x36', 'Xres')

# ── 0x64–0x69: mode settings ──────────────────────────────────────────────────

Radian      = token(b'\x64', 'Radian')
Degree      = token(b'\x65', 'Degree')
Normal      = token(b'\x66', 'Normal')
Sci         = token(b'\x67', 'Sci')
Eng         = token(b'\x68', 'Eng')
Float       = token(b'\x69', 'Float')

# ── 0x6A–0x6F: comparison operators ──────────────────────────────────────────

EQ          = token(b'\x6a', ascii='=', bp=(40, 41), operator=op.eq)
LT          = token(b'\x6b', ascii='<', bp=(40, 41), operator=op.lt)
GT          = token(b'\x6c', ascii='>', bp=(40, 41), operator=op.gt)
LE          = token(b'\x6d', '≤',       bp=(40, 41), operator=op.le)
GE          = token(b'\x6e', '≥',       bp=(40, 41), operator=op.ge)
NE          = token(b'\x6f', '≠',       bp=(40, 41), operator=op.ne)

# ── 0x70–0x72: addition, subtraction, Ans ────────────────────────────────────

ADD         = token(b'\x70', ascii='+', bp=(50, 51), operator=op.add)
SUB         = token(b'\x71', ascii='-', bp=(50, 51), operator=op.sub)
Ans         = token(b'\x72', 'Ans',  nullary=Environment.get_ans, func=forms.ans_index_or_mul)

# ── 0x73–0x7D: mode settings (cont.) ─────────────────────────────────────────

Fix         = token(b'\x73', 'Fix')
Horiz       = token(b'\x74', 'Horiz')
Full        = token(b'\x75', 'Full')
Func        = token(b'\x76', 'Func')
Param       = token(b'\x77', 'Param')
Polar       = token(b'\x78', 'Polar')
Seq         = token(b'\x79', 'Seq')
IndpntAuto  = token(b'\x7a', 'IndpntAuto')
IndpntAsk   = token(b'\x7b', 'IndpntAsk')
DependAuto  = token(b'\x7c', 'DependAuto')
DependAsk   = token(b'\x7d', 'DependAsk')

# ── 0x7E xx: graph format settings ───────────────────────────────────────────

Sequential = token(b'\x7e\x00', 'Sequential')
Simul      = token(b'\x7e\x01', 'Simul')
PolarGC    = token(b'\x7e\x02', 'PolarGC')
RectGC     = token(b'\x7e\x03', 'RectGC')
CoordOn    = token(b'\x7e\x04', 'CoordOn')
CoordOff   = token(b'\x7e\x05', 'CoordOff')
Connected  = token(b'\x7e\x06', 'Connected')
Dot        = token(b'\x7e\x07', 'Dot')
AxesOn     = token(b'\x7e\x08', 'AxesOn')
AxesOff    = token(b'\x7e\x09', 'AxesOff')
GridOn     = token(b'\x7e\x0a', 'GridOn')
GridOff    = token(b'\x7e\x0b', 'GridOff')
LabelOn    = token(b'\x7e\x0c', 'LabelOn')
LabelOff   = token(b'\x7e\x0d', 'LabelOff')
Web        = token(b'\x7e\x0e', 'Web')
TimeAxis   = token(b'\x7e\x0f', 'Time')
uvAxes     = token(b'\x7e\x10', 'uvAxes')
vwAxes     = token(b'\x7e\x11', 'vwAxes')
uwAxes     = token(b'\x7e\x12', 'uwAxes')

# ── 0x7F–0x81: plot marks ─────────────────────────────────────────────────────

SQUARE_MARK = token(b'\x7f', '▫')
CROSS_MARK  = token(b'\x80', '﹢')
DOT_MARK    = token(b'\x81', '·')

# ── 0x82–0x83: multiplication and division ────────────────────────────────────

MUL         = token(b'\x82', ascii='*', bp=(60, 61), operator=op.mul)
DIV         = token(b'\x83', ascii='/', bp=(60, 61), operator=op.truediv)

# ── 0x84–0x93: graph, zoom, and draw commands ─────────────────────────────────

Trace       = token(b'\x84', 'Trace')
ClrDraw     = token(b'\x85', 'ClrDraw')
ZStandard   = token(b'\x86', 'ZStandard')
ZTrig       = token(b'\x87', 'ZTrig')
ZBox        = token(b'\x88', 'ZBox')
ZoomIn      = token(b'\x89', 'Zoom In')
ZoomOut     = token(b'\x8a', 'Zoom Out')
ZSquare     = token(b'\x8b', 'ZSquare')
ZInteger    = token(b'\x8c', 'ZInteger')
ZPrevious   = token(b'\x8d', 'ZPrevious')
ZDecimal    = token(b'\x8e', 'ZDecimal')
ZoomStat    = token(b'\x8f', 'ZoomStat')
ZoomRcl     = token(b'\x90', 'ZoomRcl')
PrintScreen = token(b'\x91', 'PrintScreen')
ZoomSto     = token(b'\x92', 'ZoomSto')
Text        = token(b'\x93', 'Text(')

# ── 0x94–0x95: permutation and combination ────────────────────────────────────

nPr         = token(b'\x94', 'nPr',     bp=(60, 61), operator=pf.npr)
nCr         = token(b'\x95', 'nCr',     bp=(60, 61), operator=pf.ncr)

# ── 0x96–0xA9: more draw commands ─────────────────────────────────────────────

FnOn        = token(b'\x96', 'FnOn ')
FnOff       = token(b'\x97', 'FnOff ')
StorePic    = token(b'\x98', 'StorePic ')
RecallPic   = token(b'\x99', 'RecallPic ')
StoreGDB    = token(b'\x9a', 'StoreGDB ')
RecallGDB   = token(b'\x9b', 'RecallGDB ')
Line        = token(b'\x9c', 'Line(')
Vertical    = token(b'\x9d', 'Vertical ')
PtOn        = token(b'\x9e', 'Pt-On(')
PtOff       = token(b'\x9f', 'Pt-Off(')
PtChange    = token(b'\xa0', 'Pt-Change(')
PxlOn       = token(b'\xa1', 'Pxl-On(')
PxlOff      = token(b'\xa2', 'Pxl-Off(')
PxlChange   = token(b'\xa3', 'Pxl-Change(')
Shade       = token(b'\xa4', 'Shade(')
Circle      = token(b'\xa5', 'Circle(')
Horizontal  = token(b'\xa6', 'Horizontal ')
Tangent     = token(b'\xa7', 'Tangent(')
DrawInv     = token(b'\xa8', 'DrawInv ')
DrawF       = token(b'\xa9', 'DrawF ')

# ── 0xAA xx: string variables (Str1–Str0) ────────────────────────────────────

STR_1, STR_2, STR_3, STR_4, STR_5, STR_6, STR_7, STR_8, STR_9, STR_0 = STRINGS = tuple(
	token(bytes([0xaa, i]), f'Str{(i + 1) % 10}') for i in range(10)
)

# ── 0xAB–0xAF: misc nullary constants ────────────────────────────────────────

rand        = token(b'\xab', 'rand', nullary=Environment.rand, pure_func=pf.rand_list)
pi          = token(b'\xac', 'π',    nullary=lambda env: math.pi)
getKey      = token(b'\xad', 'getKey', nullary=Environment.get_key)
APOS        = token(b'\xae', ascii="'")
QUESTION    = token(b'\xaf', ascii='?')

# ── 0xB0–0xBA: math functions ─────────────────────────────────────────────────

NEG         = token(b'\xb0', '−')
int_        = token(b'\xb1', 'int(',     pure_func=pf.int_)
abs         = token(b'\xb2', 'abs(',     pure_func=pf.abs)
det         = token(b'\xb3', 'det(',     pure_func=pf.det)
identity    = token(b'\xb4', 'identity(', pure_func=pf.identity)
dim         = token(b'\xb5', 'dim(',     pure_func=pf.dim)
sum         = token(b'\xb6', 'sum(',     pure_func=pf.sum)
prod        = token(b'\xb7', 'prod(',    pure_func=pf.prod)
Not         = token(b'\xb8', 'not(',     pure_func=pf.not_)
iPart       = token(b'\xb9', 'iPart(',   pure_func=pf.i_part)
fPart       = token(b'\xba', 'fPart(',   pure_func=pf.f_part)

# ── 0xBB xx: extended tokens ──────────────────────────────────────────────────

npv          = token(b'\xbb\x00', 'npv(')
irr          = token(b'\xbb\x01', 'irr(')
bal          = token(b'\xbb\x02', 'bal(')
SumPrn       = token(b'\xbb\x03', 'Σprn(')
SumInt       = token(b'\xbb\x04', 'ΣInt(')
ToNom        = token(b'\xbb\x05', '►Nom(')
ToEff        = token(b'\xbb\x06', '►Eff(')
dbd          = token(b'\xbb\x07', 'dbd(',           pure_func=pf.dbd)
lcm          = token(b'\xbb\x08', 'lcm(',           pure_func=pf.lcm)
gcd          = token(b'\xbb\x09', 'gcd(',           pure_func=pf.gcd)
randInt      = token(b'\xbb\x0a', 'randInt(',       pure_func=pf.rand_int)
randBin      = token(b'\xbb\x0b', 'randBin(',       pure_func=pf.rand_bin)
sub          = token(b'\xbb\x0c', 'sub(',           pure_func=pf.sub_string)
stdDev       = token(b'\xbb\x0d', 'stdDev(',        pure_func=pf.stddev)
variance     = token(b'\xbb\x0e', 'variance(',      pure_func=pf.variance)
inString     = token(b'\xbb\x0f', 'inString(',      pure_func=pf.in_string)
normalcdf    = token(b'\xbb\x10', 'normalcdf(',     pure_func=pf.normalcdf)
invNorm      = token(b'\xbb\x11', 'invNorm(',       pure_func=pf.invnorm)
tcdf         = token(b'\xbb\x12', 'tcdf(',          pure_func=pf.tcdf)
ChiSqCdf     = token(b'\xbb\x13', 'χ²cdf(',         pure_func=pf.chi2cdf)
Fcdf         = token(b'\xbb\x14', 'Fcdf(',          pure_func=pf.fcdf)
binompdf     = token(b'\xbb\x15', 'binompdf(',      pure_func=pf.binompdf)
binomcdf     = token(b'\xbb\x16', 'binomcdf(',      pure_func=pf.binomcdf)
poissonpdf   = token(b'\xbb\x17', 'poissonpdf(',    pure_func=pf.poissonpdf)
poissoncdf   = token(b'\xbb\x18', 'poissoncdf(',    pure_func=pf.poissoncdf)
geometpdf    = token(b'\xbb\x19', 'geometpdf(',     pure_func=pf.geometpdf)
geometcdf    = token(b'\xbb\x1a', 'geometcdf(',     pure_func=pf.geometcdf)
normalpdf    = token(b'\xbb\x1b', 'normalpdf(',     pure_func=pf.normalpdf)
tpdf         = token(b'\xbb\x1c', 'tpdf(',          pure_func=pf.tpdf)
ChiSqPdf     = token(b'\xbb\x1d', 'χ²pdf(',         pure_func=pf.chi2pdf)
Fpdf         = token(b'\xbb\x1e', 'Fpdf(',          pure_func=pf.fpdf)
randNorm     = token(b'\xbb\x1f', 'randNorm(',      pure_func=pf.rand_norm)
tvm_Pmt      = token(b'\xbb\x20', 'tvm_Pmt')
tvm_I        = token(b'\xbb\x21', 'tvm_I%')
tvm_PV       = token(b'\xbb\x22', 'tvm_PV')
tvm_N        = token(b'\xbb\x23', 'tvm_N')
tvm_FV       = token(b'\xbb\x24', 'tvm_FV')
conj         = token(b'\xbb\x25', 'conj(',          pure_func=pf.conj)
real         = token(b'\xbb\x26', 'real(',          pure_func=pf.real)
imag         = token(b'\xbb\x27', 'imag(',          pure_func=pf.imag)
angle        = token(b'\xbb\x28', 'angle(',         pure_func=pf.angle)
cumSum       = token(b'\xbb\x29', 'cumSum(',        pure_func=pf.cum_sum)
Expr         = token(b'\xbb\x2a', 'expr(',          func=forms.expr)
length       = token(b'\xbb\x2b', 'length(',        pure_func=pf.length)
DeltaList    = token(b'\xbb\x2c', 'ΔList(',         pure_func=pf.delta_list)
ref          = token(b'\xbb\x2d', 'ref(',           pure_func=pf.ref)
rref         = token(b'\xbb\x2e', 'rref(',          pure_func=pf.rref)
TO_RECT      = token(b'\xbb\x2f', '►Rect')
TO_POLAR     = token(b'\xbb\x30', '►Polar')
Euler_e      = token(b'\xbb\x31', '𝑒', nullary=lambda env: math.e)
SinReg       = token(b'\xbb\x32', 'SinReg ')
Logistic     = token(b'\xbb\x33', 'Logistic ')
LinRegTTest  = token(b'\xbb\x34', 'LinRegTTest ')
ShadeNorm    = token(b'\xbb\x35', 'ShadeNorm(')
Shade_t      = token(b'\xbb\x36', 'Shade_t(')
ShadeChiSq   = token(b'\xbb\x37', 'Shadeχ²(')
ShadeF       = token(b'\xbb\x38', 'ShadeF(')
MatrToList   = token(b'\xbb\x39', 'Matr►list(', cmd=forms.matr_to_list)
ListToMatr   = token(b'\xbb\x3a', 'List►matr(', cmd=forms.list_to_matr)
ZTest        = token(b'\xbb\x3b', 'Z-Test(')
TTest        = token(b'\xbb\x3c', 'T-Test')
TwoSampZTest = token(b'\xbb\x3d', '2-SampZTest(')
OnePropZTest = token(b'\xbb\x3e', '1-PropZTest(')
TwoPropZTest = token(b'\xbb\x3f', '2-PropZTest(')
ChiSqTest    = token(b'\xbb\x40', 'χ²-Test(')
ZInterval    = token(b'\xbb\x41', 'ZInterval ')
TwoSampZInt  = token(b'\xbb\x42', '2-SampZInt(')
OnePropZInt  = token(b'\xbb\x43', '1-PropZInt(')
TwoPropZInt  = token(b'\xbb\x44', '2-PropZInt(')
GraphStyle   = token(b'\xbb\x45', 'GraphStyle(')
TwoSampTTest = token(b'\xbb\x46', '2-SampTTest ')
TwoSampFTest = token(b'\xbb\x47', '2-SampFTest ')
TInterval    = token(b'\xbb\x48', 'TInterval ')
TwoSampTInt  = token(b'\xbb\x49', '2-SampTInt ')
SetUpEditor  = token(b'\xbb\x4a', 'SetUpEditor ')
Pmt_End      = token(b'\xbb\x4b', 'Pmt_End')
Pmt_Bgn      = token(b'\xbb\x4c', 'Pmt_Bgn')
RealMode     = token(b'\xbb\x4d', 'Real')
re_pow_Theta_i = token(b'\xbb\x4e', 're^θi')
a_plus_bi    = token(b'\xbb\x4f', 'a+bi')
ExprOn       = token(b'\xbb\x50', 'ExprOn')
ExprOff      = token(b'\xbb\x51', 'ExprOff')
ClrAllLists  = token(b'\xbb\x52', 'ClrAllLists')
GetCalc      = token(b'\xbb\x53', 'GetCalc(')
DelVar       = token(b'\xbb\x54', 'DelVar ')
EquToStr     = token(b'\xbb\x55', 'Equ►String(')
StrToEqu     = token(b'\xbb\x56', 'String►Equ(')
ClearEntries = token(b'\xbb\x57', 'Clear Entries')
Select       = token(b'\xbb\x58', 'Select(')
ANOVA        = token(b'\xbb\x59', 'ANOVA(')
ModBoxplot   = token(b'\xbb\x5a', 'ModBoxplot')
NormProbPlot = token(b'\xbb\x5b', 'NormProbPlot')
G_T          = token(b'\xbb\x64', 'G-T')
ZoomFit      = token(b'\xbb\x65', 'ZoomFit')
DiagnosticOn = token(b'\xbb\x66', 'DiagnosticOn')
DiagnosticOff= token(b'\xbb\x67', 'DiagnosticOff')
Archive      = token(b'\xbb\x68', 'Archive ')
UnArchive    = token(b'\xbb\x69', 'UnArchive ')
Asm          = token(b'\xbb\x6a', 'Asm(')
AsmComp      = token(b'\xbb\x6b', 'AsmComp(')
AsmPrgm      = token(b'\xbb\x6c', 'AsmPrgm')
A_acute      = token(b'\xbb\x6e', 'Á')
A_grave      = token(b'\xbb\x6f', 'À')
A_circum     = token(b'\xbb\x70', 'Â')
A_umlaut     = token(b'\xbb\x71', 'Ä')
a_acute      = token(b'\xbb\x72', 'á')
a_grave      = token(b'\xbb\x73', 'à')
a_circum     = token(b'\xbb\x74', 'â')
a_umlaut     = token(b'\xbb\x75', 'ä')
E_acute      = token(b'\xbb\x76', 'É')
E_grave      = token(b'\xbb\x77', 'È')
E_circum     = token(b'\xbb\x78', 'Ê')
E_umlaut     = token(b'\xbb\x79', 'Ë')
e_acute      = token(b'\xbb\x7a', 'é')
e_grave      = token(b'\xbb\x7b', 'è')
e_circum     = token(b'\xbb\x7c', 'ê')
e_umlaut     = token(b'\xbb\x7d', 'ë')
I_grave      = token(b'\xbb\x7f', 'Ì')
I_circum     = token(b'\xbb\x80', 'Î')
I_umlaut     = token(b'\xbb\x81', 'Ï')
i_acute      = token(b'\xbb\x82', 'í')
i_grave      = token(b'\xbb\x83', 'ì')
i_circum     = token(b'\xbb\x84', 'î')
i_umlaut     = token(b'\xbb\x85', 'ï')
O_acute      = token(b'\xbb\x86', 'Ó')
O_grave      = token(b'\xbb\x87', 'Ò')
O_circum     = token(b'\xbb\x88', 'Ô')
O_umlaut     = token(b'\xbb\x89', 'Ö')
o_acute      = token(b'\xbb\x8a', 'ó')
o_grave      = token(b'\xbb\x8b', 'ò')
o_circum     = token(b'\xbb\x8c', 'ô')
o_umlaut     = token(b'\xbb\x8d', 'ö')
U_acute      = token(b'\xbb\x8e', 'Ú')
U_grave      = token(b'\xbb\x8f', 'Ù')
U_circum     = token(b'\xbb\x90', 'Û')
U_umlaut     = token(b'\xbb\x91', 'Ü')
u_acute      = token(b'\xbb\x92', 'ú')
u_grave      = token(b'\xbb\x93', 'ù')
u_circum     = token(b'\xbb\x94', 'û')
u_umlaut     = token(b'\xbb\x95', 'ü')
C_cedilla    = token(b'\xbb\x96', 'Ç')
c_cedilla    = token(b'\xbb\x97', 'ç')
N_tilde      = token(b'\xbb\x98', 'Ñ')
n_tilde      = token(b'\xbb\x99', 'ñ')
ACUTE        = token(b'\xbb\x9a', '´')
GRAVE        = token(b'\xbb\x9b', 'ˋ')
DIAERESIS    = token(b'\xbb\x9c', '¨')
INV_QUESTION = token(b'\xbb\x9d', '¿')
INV_EXCLAIM  = token(b'\xbb\x9e', '¡')
alpha        = token(b'\xbb\x9f', 'α')
beta         = token(b'\xbb\xa0', 'β')
gamma        = token(b'\xbb\xa1', 'γ')
Delta        = token(b'\xbb\xa2', 'Δ')
delta        = token(b'\xbb\xa3', 'δ')
epsilon      = token(b'\xbb\xa4', 'ε')
lam          = token(b'\xbb\xa5', 'λ')   # lambda is a Python keyword
mu           = token(b'\xbb\xa6', 'μ')
pi_alt       = token(b'\xbb\xa7', '𝛑')
rho          = token(b'\xbb\xa8', 'ρ')
Sigma        = token(b'\xbb\xa9', 'Σ')
phi          = token(b'\xbb\xab', 'φ')
Omega        = token(b'\xbb\xac', 'Ω')
psi          = token(b'\xbb\xad', 'ψ')
chi          = token(b'\xbb\xae', 'χ')
digamma      = token(b'\xbb\xaf', '𝟊')
low_a, low_b, low_c, low_d, low_e, low_f, low_g, low_h, low_i, low_j, low_k = tuple(
	token(bytes([0xbb, 0xb0 + i]), ascii=chr(0x61 + i)) for i in range(11)
)
low_l, low_m, low_n, low_o, low_p, low_q, low_r, low_s, low_t, low_u, low_v, low_w, low_x, low_y, low_z = tuple(
	token(bytes([0xbb, 0xbc + i]), ascii=chr(0x6c + i)) for i in range(15)
)
sigma        = token(b'\xbb\xcb', 'σ')
tau          = token(b'\xbb\xcc', 'τ')
I_acute      = token(b'\xbb\xcd', 'Í')
GarbageCollect = token(b'\xbb\xce', 'GarbageCollect')
TILDE        = token(b'\xbb\xcf', ascii='~')
AT           = token(b'\xbb\xd1', ascii='@')
HASH         = token(b'\xbb\xd2', ascii='#')
DOLLAR       = token(b'\xbb\xd3', ascii='$')
AMPERSAND    = token(b'\xbb\xd4', ascii='&')
BACKTICK     = token(b'\xbb\xd5', ascii='`')
SEMICOLON    = token(b'\xbb\xd6', ascii=';')
BACKSLASH    = token(b'\xbb\xd7', ascii='\\')
PIPE         = token(b'\xbb\xd8', ascii='|')
UNDERSCORE   = token(b'\xbb\xd9', ascii='_')
PERCENT      = token(b'\xbb\xda', ascii='%', postfix=lambda x: x / 100)
ELLIPSIS     = token(b'\xbb\xdb', '…')
ANGLE_SYM    = token(b'\xbb\xdc', '∠')
SHARP_S      = token(b'\xbb\xdd', 'ß')
SUP_X        = token(b'\xbb\xde', 'ˣ')
SUB_T        = token(b'\xbb\xdf', 'ₜ')
SUB_0        = token(b'\xbb\xe0', '₀')
SUB_1        = token(b'\xbb\xe1', '₁')
SUB_2        = token(b'\xbb\xe2', '₂')
SUB_3        = token(b'\xbb\xe3', '₃')
SUB_4        = token(b'\xbb\xe4', '₄')
SUB_5        = token(b'\xbb\xe5', '₅')
SUB_6        = token(b'\xbb\xe6', '₆')
SUB_7        = token(b'\xbb\xe7', '₇')
SUB_8        = token(b'\xbb\xe8', '₈')
SUB_9        = token(b'\xbb\xe9', '₉')
SUB_10       = token(b'\xbb\xea', '⑽')
REV_CONVERT  = token(b'\xbb\xeb', '◄')
RIGHT_ARROW  = token(b'\xbb\xec', '🡆')
UP_ARROW     = token(b'\xbb\xed', '↑')
DOWN_ARROW   = token(b'\xbb\xee', '↓')
ITALIC_X     = token(b'\xbb\xf0', '𝑥')
INTEGRAL     = token(b'\xbb\xf1', '∫')
SCROLL_UP    = token(b'\xbb\xf2', '🡅')
SCROLL_DN    = token(b'\xbb\xf3', '🡇')
RADICAL      = token(b'\xbb\xf4', '√')
EQ_ON        = token(b'\xbb\xf5', '≛')

# ── 0xBC–0xCD: trig and transcendental functions ──────────────────────────────

sqrt        = token(b'\xbc', '√(',       pure_func=pf.sqrt)
cbrt        = token(b'\xbd', '³√(',      pure_func=pf.cbrt)
ln          = token(b'\xbe', 'ln(',      pure_func=pf.ln)
e_pow       = token(b'\xbf', '𝑒^(',     pure_func=pf.exp)
log         = token(b'\xc0', 'log(',     pure_func=pf.log)
pow10       = token(b'\xc1', '⑽^(',     pure_func=pf.pow10)
sin         = token(b'\xc2', 'sin(',     pure_func=pf.sin)
asin        = token(b'\xc3', 'sin¹(',   pure_func=pf.asin)
cos         = token(b'\xc4', 'cos(',     pure_func=pf.cos)
acos        = token(b'\xc5', 'cos¹(',   pure_func=pf.acos)
tan         = token(b'\xc6', 'tan(',     pure_func=pf.tan)
atan        = token(b'\xc7', 'tan¹(',   pure_func=pf.atan)
sinh        = token(b'\xc8', 'sinh(',    pure_func=pf.sinh)
asinh       = token(b'\xc9', 'sinh¹(',  pure_func=pf.asinh)
cosh        = token(b'\xca', 'cosh(',    pure_func=pf.cosh)
acosh       = token(b'\xcb', 'cosh¹(',  pure_func=pf.acosh)
tanh        = token(b'\xcc', 'tanh(',    pure_func=pf.tanh)
atanh       = token(b'\xcd', 'tanh¹(',  pure_func=pf.atanh)

# ── 0xCE–0xDB: control flow ───────────────────────────────────────────────────

If          = token(b'\xce', 'If ')
Then        = token(b'\xcf', 'Then')
Else        = token(b'\xd0', 'Else')
While       = token(b'\xd1', 'While ')
Repeat      = token(b'\xd2', 'Repeat ')
For         = token(b'\xd3', 'For(')
End         = token(b'\xd4', 'End')
Return      = token(b'\xd5', 'Return')
Lbl         = token(b'\xd6', 'Lbl ')
Goto        = token(b'\xd7', 'Goto ')
Pause       = token(b'\xd8', 'Pause ')
Stop        = token(b'\xd9', 'Stop')
IS_gt       = token(b'\xda', 'IS>(')
DS_lt       = token(b'\xdb', 'DS<(')

# ── 0xDC–0xE5: I/O commands ───────────────────────────────────────────────────

Input       = token(b'\xdc', 'Input ')
Prompt      = token(b'\xdd', 'Prompt ')
Disp        = token(b'\xde', 'Disp ')
DispGraph   = token(b'\xdf', 'DispGraph')
Output      = token(b'\xe0', 'Output(')
ClrHome     = token(b'\xe1', 'ClrHome')
Fill        = token(b'\xe2', 'Fill(')
SortA       = token(b'\xe3', 'SortA(')
SortD       = token(b'\xe4', 'SortD(')
DispTable   = token(b'\xe5', 'DispTable')

# ── 0xE6–0xEA: linking and plot commands ─────────────────────────────────────

Menu        = token(b'\xe6', 'Menu(')
Send        = token(b'\xe7', 'Send(')
Get         = token(b'\xe8', 'Get(')
PlotsOn     = token(b'\xe9', 'PlotsOn')
PlotsOff    = token(b'\xea', 'PlotsOff')

# ── 0xEB–0xEE: list prefix and plot commands ──────────────────────────────────

LIST_PREFIX = token(b'\xeb', '∟')
Plot1       = token(b'\xec', 'Plot1(')
Plot2       = token(b'\xed', 'Plot2(')
Plot3       = token(b'\xee', 'Plot3(')

# ── 0xEF xx: TI-84+ extended tokens ──────────────────────────────────────────

setDate      = token(b'\xef\x00', 'setDate(',    cmd=forms.set_date)
setTime      = token(b'\xef\x01', 'setTime(',    cmd=forms.set_time)
checkTmr     = token(b'\xef\x02', 'checkTmr(',   func=forms.check_tmr)
setDtFmt     = token(b'\xef\x03', 'setDtFmt(',   cmd=forms.set_dt_fmt)
setTmFmt     = token(b'\xef\x04', 'setTmFmt(',   cmd=forms.set_tm_fmt)
timeCnv      = token(b'\xef\x05', 'timeCnv(',    pure_func=pf.timecnv)
dayOfWk      = token(b'\xef\x06', 'dayOfWk(',    pure_func=pf.dayofwk)
getDtStr     = token(b'\xef\x07', 'getDtStr(',   func=forms.get_dt_str)
getTmStr     = token(b'\xef\x08', 'getTmStr(',   func=forms.get_tm_str)
getDate      = token(b'\xef\x09', 'getDate',     nullary=Environment.get_date)
getTime      = token(b'\xef\x0a', 'getTime',     nullary=Environment.get_time)
startTmr     = token(b'\xef\x0b', 'startTmr',   nullary=Environment.start_tmr)
getDtFmt     = token(b'\xef\x0c', 'getDtFmt',    nullary=Environment.get_dt_fmt)
getTmFmt     = token(b'\xef\x0d', 'getTmFmt',    nullary=Environment.get_tm_fmt)
isClockOn    = token(b'\xef\x0e', 'isClockOn',   nullary=Environment.is_clock_on)
ClockOff     = token(b'\xef\x0f', 'ClockOff',    cmd=Environment.clock_off)
ClockOn      = token(b'\xef\x10', 'ClockOn',     cmd=Environment.clock_on)
OpenLib      = token(b'\xef\x11', 'OpenLib(')
ExecLib      = token(b'\xef\x12', 'ExecLib')
invT         = token(b'\xef\x13', 'invT(',       pure_func=pf.invt)
ChiSqGofTest = token(b'\xef\x14', 'χ²GOF-Test(')
LinRegTInt   = token(b'\xef\x15', 'LinRegTInt ')
ManualFit    = token(b'\xef\x16', 'Manual-Fit ')
ZQuadrant1   = token(b'\xef\x17', 'ZQuadrant1')
ZFrac1_2     = token(b'\xef\x18', 'ZFrac1/2')
ZFrac1_3     = token(b'\xef\x19', 'ZFrac1/3')
ZFrac1_4     = token(b'\xef\x1a', 'ZFrac1/4')
ZFrac1_5     = token(b'\xef\x1b', 'ZFrac1/5')
ZFrac1_8     = token(b'\xef\x1c', 'ZFrac1/8')
ZFrac1_10    = token(b'\xef\x1d', 'ZFrac1/10')
MathprintBox = token(b'\xef\x1e', 'mathprintbox')
ToNd_UnD     = token(b'\xef\x30', '►n/d◄►Un/d')
ToF_D        = token(b'\xef\x31', '►F◄►D')
remainder    = token(b'\xef\x32', 'remainder(',  pure_func=pf.remainder)
SIGMA        = token(b'\xef\x33', 'Σ(',          func=forms.sigma)
logBASE      = token(b'\xef\x34', 'logBASE(',    pure_func=pf.log_base)
randIntNoRep = token(b'\xef\x35', 'randIntNoRep(', pure_func=pf.rand_int_no_rep)
MATHPRINT    = token(b'\xef\x36', 'MATHPRINT')
CLASSIC      = token(b'\xef\x37', 'CLASSIC')
N_D          = token(b'\xef\x38', 'n/d')
UN_D         = token(b'\xef\x39', 'Un/d')
AUTO         = token(b'\xef\x3a', 'AUTO')
DEC          = token(b'\xef\x3b', 'DEC')
FRAC         = token(b'\xef\x3c', 'FRAC')
FRAC_APPROX  = token(b'\xef\x3d', 'FRAC-APPROX')

# ── 0xF0–0xFF: power operators and regression commands ───────────────────────

POW         = token(b'\xf0', ascii='^', bp=(70, 69), operator=op.pow)
XTH_ROOT    = token(b'\xf1', 'ˣ√',      bp=(60, 61), operator=pf.xth_root)
OneVarStats  = token(b'\xf2', '1-Var Stats ')
TwoVarStats  = token(b'\xf3', '2-Var Stats ')
LinReg_abx   = token(b'\xf4', 'LinReg(a+bx) ')
ExpReg       = token(b'\xf5', 'ExpReg ')
LnReg        = token(b'\xf6', 'LnReg ')
PwrReg       = token(b'\xf7', 'PwrReg ')
MedMed       = token(b'\xf8', 'Med-Med ')
QuadReg      = token(b'\xf9', 'QuadReg ')
ClrList      = token(b'\xfa', 'ClrList ')
ClrTable     = token(b'\xfb', 'ClrTable')
Histogram    = token(b'\xfc', 'Histogram')
xyLine       = token(b'\xfd', 'xyLine')
Scatter      = token(b'\xfe', 'Scatter')
LinReg_axb   = token(b'\xff', 'LinReg(ax+b) ')


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


TOKEN_TABLE    = TokenTable(ALL_TOKENS)
_by_ascii      = {t.ascii: t for t in ALL_TOKENS if t.ascii}
_by_text       = {t.text: t for t in ALL_TOKENS} | _by_ascii

# Backward-compatible aliases
TOKENS         = ALL_TOKENS
ASCII          = _by_ascii
TOKENS_BY_TEXT = _by_text


if __name__ == '__main__':
	for token in sorted(ASCII.values(), key=lambda t: t.ascii):
		print(ord(token.ascii), token)
