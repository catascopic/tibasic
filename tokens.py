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
	char: str | None
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

	def is_digit(self) -> bool:
		return 0x30 <= self.code[0] <= 0x39

	def is_numeric_var(self) -> bool:
		return 0x41 <= self.code[0] < 0x5c

	def is_list_var(self) -> bool:
		return self.code[0] == 0x5d

	def is_matrix_var(self) -> bool:
		return self.code[0] == 0x5c

	def is_string_var(self) -> bool:
		return self.code[0] == 0xaa

	def is_stat_var(self) -> bool:
		return self.code[0] == 0x62

	def is_window_var(self) -> bool:
		return self.code[0] == 0x63

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


ALL_TOKENS: list[Token] = []
CHARS = {}

_SEEN: set[bytes] = set()

def _make_pure_func(f):
	def wrapper(a):
		return f(*a.parse_args())
	return wrapper


def token(
	code: bytes,
	text: str = None,
	char: str = None,
	*,
	bp:   tuple[int, int] | None = None,
	op:   Callable | None = None,
	post: Callable | None = None,
	func: Callable | None = None,
	pure: Callable | None = None,
	cmd:  Callable | None = None,
	res:  Callable | None = None,
	cnv:  Callable | None = None,
	var:  Variable | None = None,
) -> Token:

	if code in _SEEN:
		raise ValueError(f'Duplicate token code: {code!r} ({text!r})')
	_SEEN.add(code)
	if pure is not None:
		if func is not None:
			raise ValueError(f'Token {text!r}: cannot set both func and pure')
		func = _make_pure_func(pure)

	t = Token(code, char, text or char, bp, op, post, func, cmd, res, cnv, var)
	ALL_TOKENS.append(t)
	if char:
		CHARS[char] = t
	return t


token(b'\x01', '►DMS',  cnv=pf.to_dms)
token(b'\x02', '►Dec',  cnv=pf.to_dec)
token(b'\x03', '►Frac', cnv=pf.to_frac)

STORE = token(b'\x04', '→')

token(b'\x05', 'Boxplot')

L_BRACKET = token(b'\x06', char='[')
R_BRACKET = token(b'\x07', char=']')
L_BRACE   = token(b'\x08', char='{')
R_BRACE   = token(b'\x09', char='}')
RAD       = token(b'\x0a', 'ʳ')           # post; needs env, handled specially
DEG       = token(b'\x0b', char='°')      # ditto
INV       = token(b'\x0c', '¹',  post=pf.inv)
SQ        = token(b'\x0d', char='²',  post=lambda x: x**2)
TRANSPOSE = token(b'\x0e', 'ᵀ',  post=pf.transpose)
CUBE      = token(b'\x0f', char='³',  post=lambda x: x**3)
L_PAREN   = token(b'\x10', char='(')
R_PAREN   = token(b'\x11', char=')')

token(b'\x12', 'round(',       pure=pf.round)
token(b'\x13', 'pxl-Test(')
token(b'\x14', 'augment(',     pure=pf.augment)
token(b'\x15', 'rowSwap(',     pure=pf.rowswap)
token(b'\x16', 'row+(',        pure=pf.row_plus)
token(b'\x17', '*row(',        pure=pf.times_row)
token(b'\x18', '*row+(',       pure=pf.times_row_plus)
token(b'\x19', 'max(',         pure=pf.max)
token(b'\x1a', 'min(',         pure=pf.min)
token(b'\x1b', 'R►Pr(',        pure=pf.r_pr)
token(b'\x1c', 'R►Pθ(',        pure=pf.r_ptheta)
token(b'\x1d', 'P►Rx(',        pure=pf.p_rx)
token(b'\x1e', 'P►Ry(',        pure=pf.p_ry)
token(b'\x1f', 'median(',      pure=pf.median)
token(b'\x20', 'randM(',       pure=pf.rand_m)
token(b'\x21', 'mean(',        pure=pf.mean)
token(b'\x22', 'solve(')
token(b'\x23', 'seq(',         func=forms.seq)
token(b'\x24', 'fnInt(',       func=forms.fn_int)
token(b'\x25', 'nDeriv(',      func=forms.n_deriv)
token(b'\x27', 'fMin(')
token(b'\x28', 'fMax(')
token(b'\x29', char=' ')

QUOTE  = token(b'\x2a', char='"')
COMMA  = token(b'\x2b', char=',')
IMAG_I = token(b'\x2c', char='𝑖', res=lambda env: 1j)
FACT   = token(b'\x2d', char='!',      post=pf.factorial)

token(b'\x2e', 'CubicReg ')
token(b'\x2f', 'QuartReg ')

DIGITS = tuple(token(bytes([0x30 + i]), char=chr(0x30 + i)) for i in range(10))

DOT       = token(b'\x3a', char='.')
SCI_E     = token(b'\x3b', 'ᴇ')

token(b'\x3c', ' or ',    bp=(20, 21), op=pf.or_)
token(b'\x3d', ' xor ',   bp=(20, 21), op=pf.xor)

COLON     = token(b'\x3e', char=':')
NEWLINE   = token(b'\x3f', char='\n')

token(b'\x40', ' and ',   bp=(30, 31), op=pf.and_)

(
	VAR_A, VAR_B, VAR_C, VAR_D, VAR_E, VAR_F, VAR_G, VAR_H, VAR_I, VAR_J, 
	VAR_K, VAR_L, VAR_M, VAR_N, VAR_O, VAR_P, VAR_Q, VAR_R, VAR_S, VAR_T, 
	VAR_U, VAR_V, VAR_W, VAR_X, VAR_Y, VAR_Z) = LETTERS = tuple(
	token(bytes([0x41 + i]), char=chr(0x41 + i), var=NumericVar(i)) for i in range(26)
)

VAR_THETA = token(b'\x5b', char='θ', var=NumericVar(26))

# ── 0x5C xx: matrix variables ([A]–[J]) ──────────────────────────────────────

MATRICES = tuple(token(bytes([0x5c, i]), f'[{chr(0x41 + i)}]', var=MatrixVar(i)) for i in range(10))

# ── 0x5D xx: list variables (L1–L6) ──────────────────────────────────────────

LISTS = tuple(token(bytes([0x5d, i]), f'L{chr(0x2081 + i)}', var=ListVar(i)) for i in range(6))

# ── 0x5E xx: equation and sequence variables ──────────────────────────────────

Y_EQUATIONS = tuple(token(bytes([0x5e, 0x10 + i]), f'Y{chr(0x2080 + (i + 1) % 10)}') for i in range(10))
PARAM_EQUATIONS = tuple(
	token(bytes([0x5e, 0x20 + i]), f'{x}{chr(0x2080 + n)}ₜ') 
	for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))
)
POLAR_EQUATIONS = tuple(token(bytes([0x5e, 0x40 + i]), f'r{chr(0x2081 + i)}') for i in range(6))

token(b'\x5e\x80', '𝑢')
token(b'\x5e\x81', '𝑣')
token(b'\x5e\x82', '𝑤')

PRGM = token(b'\x5f', 'prgm')

# ── 0x60 xx: picture variables (Pic1–Pic0) ───────────────────────────────────

PICTURES = tuple(token(bytes([0x60, i]), f'Pic{(i + 1) % 10}') for i in range(10))

# ── 0x61 xx: graph database variables (GDB1–GDB0) ────────────────────────────

GDBS = tuple(token(bytes([0x61, i]), f'GDB{(i + 1) % 10}') for i in range(10))

# ── 0x62 xx: statistical result variables ─────────────────────────────────────

token(b'\x62\x01', 'RegEq')
token(b'\x62\x02', 'n')
token(b'\x62\x03', 'ẍ')
token(b'\x62\x04', 'Σx')
token(b'\x62\x05', 'Σx²')
token(b'\x62\x06', 'Sx')
token(b'\x62\x07', 'σx')
token(b'\x62\x08', 'minX')
token(b'\x62\x09', 'maxX')
token(b'\x62\x0a', 'minY')
token(b'\x62\x0b', 'maxY')
token(b'\x62\x0c', 'ȳ')
token(b'\x62\x0d', 'Σy')
token(b'\x62\x0e', 'Σy²')
token(b'\x62\x0f', 'Sy')
token(b'\x62\x10', 'σy')
token(b'\x62\x11', 'Σxy')
token(b'\x62\x12', 'r')
token(b'\x62\x13', 'Med')
token(b'\x62\x14', 'Q1')
token(b'\x62\x15', 'Q3')
token(b'\x62\x16', 'a')
token(b'\x62\x17', 'b')
token(b'\x62\x18', 'c')
token(b'\x62\x19', 'd')
token(b'\x62\x1a', 'e')
token(b'\x62\x1b', 'x₁')
token(b'\x62\x1c', 'x₂')
token(b'\x62\x1d', 'x₃')
token(b'\x62\x1e', 'y₁')
token(b'\x62\x1f', 'y₂')
token(b'\x62\x20', 'y₃')
token(b'\x62\x21', '𝑛')
token(b'\x62\x22', 'p')
token(b'\x62\x23', 'z')
token(b'\x62\x24', 't')
token(b'\x62\x25', 'χ²')
token(b'\x62\x26', '𝐅')
token(b'\x62\x27', 'df')
token(b'\x62\x28', 'ṕ')
token(b'\x62\x29', 'ṕ₁')
token(b'\x62\x2a', 'ṕ₂')
token(b'\x62\x2b', 'ẍ₁')
token(b'\x62\x2c', 'Sx₁')
token(b'\x62\x2d', 'n₁')
token(b'\x62\x2e', 'ẍ₂')
token(b'\x62\x2f', 'Sx₂')
token(b'\x62\x30', 'n₂')
token(b'\x62\x31', 'Sxp')
token(b'\x62\x32', 'lower')
token(b'\x62\x33', 'upper')
token(b'\x62\x34', 's')
token(b'\x62\x35', 'r²')
token(b'\x62\x36', 'R²')
token(b'\x62\x37', 'Factor df')
token(b'\x62\x38', 'Factor SS')
token(b'\x62\x39', 'Factor MS')
token(b'\x62\x3a', 'Error df')
token(b'\x62\x3b', 'Error SS')
token(b'\x62\x3c', 'Error MS')

# ── 0x63 xx: window and finance variables ─────────────────────────────────────

token(b'\x63\x02', 'Xscl', var=WindowVar(0))
token(b'\x63\x03', 'Yscl', var=WindowVar(1))
token(b'\x63\x0a', 'Xmin', var=WindowVar(2))
token(b'\x63\x0b', 'Xmax', var=WindowVar(3))
token(b'\x63\x0c', 'Ymin', var=WindowVar(4))
token(b'\x63\x0d', 'Ymax', var=WindowVar(5))
token(b'\x63\x0e', 'Tmin', var=WindowVar(6))
token(b'\x63\x0f', 'Tmax', var=WindowVar(7))
token(b'\x63\x10', 'θmin', var=WindowVar(8))
token(b'\x63\x11', 'θmax', var=WindowVar(9))
token(b'\x63\x1a', 'TblStart',  var=WindowVar(10))
token(b'\x63\x1b', 'PlotStart', var=WindowVar(11))
token(b'\x63\x1d', 'nMax', var=WindowVar(12))
token(b'\x63\x1f', 'nMin', var=WindowVar(13))
token(b'\x63\x21', 'ΔTbl', var=WindowVar(14))
token(b'\x63\x22', 'Tstep', var=WindowVar(15))
token(b'\x63\x23', 'θstep', var=WindowVar(16))
token(b'\x63\x26', 'ΔX', var=WindowVar(17))
token(b'\x63\x27', 'ΔY', var=WindowVar(18))
token(b'\x63\x28', 'XFact', var=WindowVar(19))
token(b'\x63\x29', 'YFact', var=WindowVar(20))
token(b'\x63\x2b', '𝐍')
token(b'\x63\x2c', 'I%')
token(b'\x63\x2d', 'PV')
token(b'\x63\x2e', 'PMT')
token(b'\x63\x2f', 'FV')
token(b'\x63\x30', 'P/Y')
token(b'\x63\x31', 'C/Y')
token(b'\x63\x34', 'PlotStep')
token(b'\x63\x36', 'Xres')


token(b'\x64', 'Radian')
token(b'\x65', 'Degree')
token(b'\x66', 'Normal')
token(b'\x67', 'Sci')
token(b'\x68', 'Eng')
token(b'\x69', 'Float')

EQ  = token(b'\x6a', char='=', bp=(40, 41), op=op.eq)
LT  = token(b'\x6b', char='<', bp=(40, 41), op=op.lt)
GT  = token(b'\x6c', char='>', bp=(40, 41), op=op.gt)
LE  = token(b'\x6d', char='≤', bp=(40, 41), op=op.le)
GE  = token(b'\x6e', char='≥', bp=(40, 41), op=op.ge)
NE  = token(b'\x6f', char='≠', bp=(40, 41), op=op.ne)
ADD = token(b'\x70', char='+', bp=(50, 51), op=op.add)
SUB = token(b'\x71', char='-', bp=(50, 51), op=op.sub)
ANS = token(b'\x72', 'Ans',  res=Environment.get_ans, func=forms.ans_index_or_mul)

token(b'\x73', 'Fix')
token(b'\x74', 'Horiz')
token(b'\x75', 'Full')
token(b'\x76', 'Func')
token(b'\x77', 'Param')
token(b'\x78', 'Polar')
token(b'\x79', 'Seq')
token(b'\x7a', 'IndpntAuto')
token(b'\x7b', 'IndpntAsk')
token(b'\x7c', 'DependAuto')
token(b'\x7d', 'DependAsk')

# ── 0x7E xx: graph format settings ───────────────────────────────────────────

token(b'\x7e\x00', 'Sequential')
token(b'\x7e\x01', 'Simul')
token(b'\x7e\x02', 'PolarGC')
token(b'\x7e\x03', 'RectGC')
token(b'\x7e\x04', 'CoordOn')
token(b'\x7e\x05', 'CoordOff')
token(b'\x7e\x06', 'Connected')
token(b'\x7e\x07', 'Dot')
token(b'\x7e\x08', 'AxesOn')
token(b'\x7e\x09', 'AxesOff')
token(b'\x7e\x0a', 'GridOn')
token(b'\x7e\x0b', 'GridOff')
token(b'\x7e\x0c', 'LabelOn')
token(b'\x7e\x0d', 'LabelOff')
token(b'\x7e\x0e', 'Web')
token(b'\x7e\x0f', 'Time')
token(b'\x7e\x10', 'uvAxes')
token(b'\x7e\x11', 'vwAxes')
token(b'\x7e\x12', 'uwAxes')

token(b'\x7f', '▫')
token(b'\x80', '﹢')
token(b'\x81', '·')

MUL = token(b'\x82', char='*', bp=(60, 61), op=op.mul)
DIV = token(b'\x83', char='/', bp=(60, 61), op=op.truediv)

token(b'\x84', 'Trace')
token(b'\x85', 'ClrDraw')
token(b'\x86', 'ZStandard')
token(b'\x87', 'ZTrig')
token(b'\x88', 'ZBox')
token(b'\x89', 'Zoom In')
token(b'\x8a', 'Zoom Out')
token(b'\x8b', 'ZSquare')
token(b'\x8c', 'ZInteger')
token(b'\x8d', 'ZPrevious')
token(b'\x8e', 'ZDecimal')
token(b'\x8f', 'ZoomStat')
token(b'\x90', 'ZoomRcl')
token(b'\x91', 'PrintScreen')
token(b'\x92', 'ZoomSto')
token(b'\x93', 'Text(')

NPR = token(b'\x94', 'nPr',     bp=(60, 61), op=pf.npr)
NCR = token(b'\x95', 'nCr',     bp=(60, 61), op=pf.ncr)

token(b'\x96', 'FnOn ')
token(b'\x97', 'FnOff ')
token(b'\x98', 'StorePic ')
token(b'\x99', 'RecallPic ')
token(b'\x9a', 'StoreGDB ')
token(b'\x9b', 'RecallGDB ')
token(b'\x9c', 'Line(')
token(b'\x9d', 'Vertical ')
token(b'\x9e', 'Pt-On(')
token(b'\x9f', 'Pt-Off(')
token(b'\xa0', 'Pt-Change(')
token(b'\xa1', 'Pxl-On(')
token(b'\xa2', 'Pxl-Off(')
token(b'\xa3', 'Pxl-Change(')
token(b'\xa4', 'Shade(')
token(b'\xa5', 'Circle(')
token(b'\xa6', 'Horizontal ')
token(b'\xa7', 'Tangent(')
token(b'\xa8', 'DrawInv ')
token(b'\xa9', 'DrawF ')

# ── 0xAA xx: string variables (Str1–Str0) ────────────────────────────────────

STRINGS = tuple(token(bytes([0xaa, i]), f'Str{(i + 1) % 10}', var=StringVar(i)) for i in range(10))

RAND = token(b'\xab', 'rand', res=Environment.rand, pure=pf.rand_list)
token(b'\xac', char='π', res=lambda env: math.pi)
token(b'\xad', 'getKey', res=Environment.get_key)
APOS = token(b'\xae', char="'")
token(b'\xaf', char='?')
NEG  = token(b'\xb0', '⁻')

token(b'\xb1', 'int(', pure=pf.int_)
token(b'\xb2', 'abs(', pure=pf.abs)
token(b'\xb3', 'det(', pure=pf.det)
token(b'\xb4', 'identity(', pure=pf.identity)

DIM = token(b'\xb5', 'dim(', pure=pf.dim)

token(b'\xb6', 'sum(', pure=pf.sum)
token(b'\xb7', 'prod(', pure=pf.prod)
token(b'\xb8', 'not(', pure=pf.not_)
token(b'\xb9', 'iPart(', pure=pf.i_part)
token(b'\xba', 'fPart(', pure=pf.f_part)

# ── 0xBB xx: extended tokens ──────────────────────────────────────────────────

token(b'\xbb\x00', 'npv(')
token(b'\xbb\x01', 'irr(')
token(b'\xbb\x02', 'bal(')
token(b'\xbb\x03', 'Σprn(')
token(b'\xbb\x04', 'ΣInt(')
token(b'\xbb\x05', '►Nom(')
token(b'\xbb\x06', '►Eff(')
token(b'\xbb\x07', 'dbd(',           pure=pf.dbd)
token(b'\xbb\x08', 'lcm(',           pure=pf.lcm)
token(b'\xbb\x09', 'gcd(',           pure=pf.gcd)
token(b'\xbb\x0a', 'randInt(',       pure=pf.rand_int)
token(b'\xbb\x0b', 'randBin(',       pure=pf.rand_bin)
token(b'\xbb\x0c', 'sub(',           pure=pf.sub_string)
token(b'\xbb\x0d', 'stdDev(',        pure=pf.stddev)
token(b'\xbb\x0e', 'variance(',      pure=pf.variance)
token(b'\xbb\x0f', 'inString(',      pure=pf.in_string)
token(b'\xbb\x10', 'normalcdf(',     pure=pf.normalcdf)
token(b'\xbb\x11', 'invNorm(',       pure=pf.invnorm)
token(b'\xbb\x12', 'tcdf(',          pure=pf.tcdf)
token(b'\xbb\x13', 'χ²cdf(',         pure=pf.chi2cdf)
token(b'\xbb\x14', 'Fcdf(',          pure=pf.fcdf)
token(b'\xbb\x15', 'binompdf(',      pure=pf.binompdf)
token(b'\xbb\x16', 'binomcdf(',      pure=pf.binomcdf)
token(b'\xbb\x17', 'poissonpdf(',    pure=pf.poissonpdf)
token(b'\xbb\x18', 'poissoncdf(',    pure=pf.poissoncdf)
token(b'\xbb\x19', 'geometpdf(',     pure=pf.geometpdf)
token(b'\xbb\x1a', 'geometcdf(',     pure=pf.geometcdf)
token(b'\xbb\x1b', 'normalpdf(',     pure=pf.normalpdf)
token(b'\xbb\x1c', 'tpdf(',          pure=pf.tpdf)
token(b'\xbb\x1d', 'χ²pdf(',         pure=pf.chi2pdf)
token(b'\xbb\x1e', 'Fpdf(',          pure=pf.fpdf)
token(b'\xbb\x1f', 'randNorm(',      pure=pf.rand_norm)
token(b'\xbb\x20', 'tvm_Pmt')
token(b'\xbb\x21', 'tvm_I%')
token(b'\xbb\x22', 'tvm_PV')
token(b'\xbb\x23', 'tvm_N')
token(b'\xbb\x24', 'tvm_FV')
token(b'\xbb\x25', 'conj(',          pure=pf.conj)
token(b'\xbb\x26', 'real(',          pure=pf.real)
token(b'\xbb\x27', 'imag(',          pure=pf.imag)
token(b'\xbb\x28', 'angle(',         pure=pf.angle)
token(b'\xbb\x29', 'cumSum(',        pure=pf.cum_sum)
token(b'\xbb\x2a', 'expr(',          func=forms.expr)
token(b'\xbb\x2b', 'length(',        pure=pf.length)
token(b'\xbb\x2c', 'ΔList(',         pure=pf.delta_list)
token(b'\xbb\x2d', 'ref(',           pure=pf.ref)
token(b'\xbb\x2e', 'rref(',          pure=pf.rref)
token(b'\xbb\x2f', '►Rect')
token(b'\xbb\x30', '►Polar')
token(b'\xbb\x31', char='𝑒', res=lambda env: math.e)
token(b'\xbb\x32', 'SinReg ')
token(b'\xbb\x33', 'Logistic ')
token(b'\xbb\x34', 'LinRegTTest ')
token(b'\xbb\x35', 'ShadeNorm(')
token(b'\xbb\x36', 'Shade_t(')
token(b'\xbb\x37', 'Shadeχ²(')
token(b'\xbb\x38', 'ShadeF(')
token(b'\xbb\x39', 'Matr►list(', cmd=forms.matr_to_list)
token(b'\xbb\x3a', 'List►matr(', cmd=forms.list_to_matr)
token(b'\xbb\x3b', 'Z-Test(')
token(b'\xbb\x3c', 'T-Test')
token(b'\xbb\x3d', '2-SampZTest(')
token(b'\xbb\x3e', '1-PropZTest(')
token(b'\xbb\x3f', '2-PropZTest(')
token(b'\xbb\x40', 'χ²-Test(')
token(b'\xbb\x41', 'ZInterval ')
token(b'\xbb\x42', '2-SampZInt(')
token(b'\xbb\x43', '1-PropZInt(')
token(b'\xbb\x44', '2-PropZInt(')
token(b'\xbb\x45', 'GraphStyle(')
token(b'\xbb\x46', '2-SampTTest ')
token(b'\xbb\x47', '2-SampFTest ')
token(b'\xbb\x48', 'TInterval ')
token(b'\xbb\x49', '2-SampTInt ')
token(b'\xbb\x4a', 'SetUpEditor ')
token(b'\xbb\x4b', 'Pmt_End')
token(b'\xbb\x4c', 'Pmt_Bgn')
token(b'\xbb\x4d', 'Real')
token(b'\xbb\x4e', 're^θi')
token(b'\xbb\x4f', 'a+bi')
token(b'\xbb\x50', 'ExprOn')
token(b'\xbb\x51', 'ExprOff')
token(b'\xbb\x52', 'ClrAllLists')
token(b'\xbb\x53', 'GetCalc(')
token(b'\xbb\x54', 'DelVar ')
token(b'\xbb\x55', 'Equ►String(')
token(b'\xbb\x56', 'String►Equ(')
token(b'\xbb\x57', 'Clear Entries')
token(b'\xbb\x58', 'Select(')
token(b'\xbb\x59', 'ANOVA(')
token(b'\xbb\x5a', 'ModBoxplot')
token(b'\xbb\x5b', 'NormProbPlot')
token(b'\xbb\x64', 'G-T')
token(b'\xbb\x65', 'ZoomFit')
token(b'\xbb\x66', 'DiagnosticOn')
token(b'\xbb\x67', 'DiagnosticOff')
token(b'\xbb\x68', 'Archive ')
token(b'\xbb\x69', 'UnArchive ')
token(b'\xbb\x6a', 'Asm(')
token(b'\xbb\x6b', 'AsmComp(')
token(b'\xbb\x6c', 'AsmPrgm')
token(b'\xbb\x6e', char='Á')
token(b'\xbb\x6f', char='À')
token(b'\xbb\x70', char='Â')
token(b'\xbb\x71', char='Ä')
token(b'\xbb\x72', char='á')
token(b'\xbb\x73', char='à')
token(b'\xbb\x74', char='â')
token(b'\xbb\x75', char='ä')
token(b'\xbb\x76', char='É')
token(b'\xbb\x77', char='È')
token(b'\xbb\x78', char='Ê')
token(b'\xbb\x79', char='Ë')
token(b'\xbb\x7a', char='é')
token(b'\xbb\x7b', char='è')
token(b'\xbb\x7c', char='ê')
token(b'\xbb\x7d', char='ë')
token(b'\xbb\x7f', char='Ì')
token(b'\xbb\x80', char='Î')
token(b'\xbb\x81', char='Ï')
token(b'\xbb\x82', char='í')
token(b'\xbb\x83', char='ì')
token(b'\xbb\x84', char='î')
token(b'\xbb\x85', char='ï')
token(b'\xbb\x86', char='Ó')
token(b'\xbb\x87', char='Ò')
token(b'\xbb\x88', char='Ô')
token(b'\xbb\x89', char='Ö')
token(b'\xbb\x8a', char='ó')
token(b'\xbb\x8b', char='ò')
token(b'\xbb\x8c', char='ô')
token(b'\xbb\x8d', char='ö')
token(b'\xbb\x8e', char='Ú')
token(b'\xbb\x8f', char='Ù')
token(b'\xbb\x90', char='Û')
token(b'\xbb\x91', char='Ü')
token(b'\xbb\x92', char='ú')
token(b'\xbb\x93', char='ù')
token(b'\xbb\x94', char='û')
token(b'\xbb\x95', char='ü')
token(b'\xbb\x96', char='Ç')
token(b'\xbb\x97', char='ç')
token(b'\xbb\x98', char='Ñ')
token(b'\xbb\x99', char='ñ')
token(b'\xbb\x9a', '´')
token(b'\xbb\x9b', 'ˋ')
token(b'\xbb\x9c', '¨')
token(b'\xbb\x9d', char='¿')
token(b'\xbb\x9e', char='¡')
token(b'\xbb\x9f', char='α')
token(b'\xbb\xa0', char='β')
token(b'\xbb\xa1', char='γ')
token(b'\xbb\xa2', char='Δ')
token(b'\xbb\xa3', char='δ')
token(b'\xbb\xa4', char='ε')
token(b'\xbb\xa5', char='λ')
token(b'\xbb\xa6', char='μ')
token(b'\xbb\xa7', '𝛑')  # alternate pi
token(b'\xbb\xa8', char='ρ')
token(b'\xbb\xa9', char='Σ')
token(b'\xbb\xab', char='φ')
token(b'\xbb\xac', char='Ω')
token(b'\xbb\xad', char='ψ')
token(b'\xbb\xae', char='χ')
token(b'\xbb\xaf', '𝟊')

for i in range(11):
	token(bytes([0xbb, 0xb0 + i]), char=chr(0x61 + i))
for i in range(15):
	token(bytes([0xbb, 0xbc + i]), char=chr(0x6c + i)) 

token(b'\xbb\xcb', char='σ')
token(b'\xbb\xcc', char='τ')
token(b'\xbb\xcd', char='Í')
token(b'\xbb\xce', 'GarbageCollect')
token(b'\xbb\xcf', char='~')
token(b'\xbb\xd1', char='@')
token(b'\xbb\xd2', char='#')
token(b'\xbb\xd3', char='$')
token(b'\xbb\xd4', char='&')
token(b'\xbb\xd5', char='`')
token(b'\xbb\xd6', char=';')
token(b'\xbb\xd7', char='\\')
token(b'\xbb\xd8', char='|')
token(b'\xbb\xd9', char='_')
token(b'\xbb\xda', char='%', post=lambda x: x / 100)
token(b'\xbb\xdb', char='…')
token(b'\xbb\xdc', char='∠')
token(b'\xbb\xdd', char='ß')
token(b'\xbb\xde', 'ˣ')
token(b'\xbb\xdf', 'ₜ')
token(b'\xbb\xe0', '₀')
token(b'\xbb\xe1', '₁')
token(b'\xbb\xe2', '₂')
token(b'\xbb\xe3', '₃')
token(b'\xbb\xe4', '₄')
token(b'\xbb\xe5', '₅')
token(b'\xbb\xe6', '₆')
token(b'\xbb\xe7', '₇')
token(b'\xbb\xe8', '₈')
token(b'\xbb\xe9', '₉')
token(b'\xbb\xea', '⑽')
token(b'\xbb\xeb', '◄')
token(b'\xbb\xec', '🡆')
token(b'\xbb\xed', '↑')
token(b'\xbb\xee', '↓')
token(b'\xbb\xf0', '𝑥')
token(b'\xbb\xf1', char='∫')
token(b'\xbb\xf2', '🡅')
token(b'\xbb\xf3', '🡇')
token(b'\xbb\xf4', '√')
token(b'\xbb\xf5', '≛')
token(b'\xbc', '√(',       pure=pf.sqrt)
token(b'\xbd', '³√(',      pure=pf.cbrt)
token(b'\xbe', 'ln(',      pure=pf.ln)
token(b'\xbf', '𝑒^(',     pure=pf.exp)
token(b'\xc0', 'log(',     pure=pf.log)
token(b'\xc1', '⑽^(',     pure=pf.pow10)
token(b'\xc2', 'sin(',     pure=pf.sin)
token(b'\xc3', 'sin¹(',   pure=pf.asin)
token(b'\xc4', 'cos(',     pure=pf.cos)
token(b'\xc5', 'cos¹(',   pure=pf.acos)
token(b'\xc6', 'tan(',     pure=pf.tan)
token(b'\xc7', 'tan¹(',   pure=pf.atan)
token(b'\xc8', 'sinh(',    pure=pf.sinh)
token(b'\xc9', 'sinh¹(',  pure=pf.asinh)
token(b'\xca', 'cosh(',    pure=pf.cosh)
token(b'\xcb', 'cosh¹(',  pure=pf.acosh)
token(b'\xcc', 'tanh(',    pure=pf.tanh)
token(b'\xcd', 'tanh¹(',  pure=pf.atanh)
token(b'\xce', 'If ')
token(b'\xcf', 'Then')
token(b'\xd0', 'Else')
token(b'\xd1', 'While ')
token(b'\xd2', 'Repeat ')
token(b'\xd3', 'For(')
token(b'\xd4', 'End')
token(b'\xd5', 'Return')
token(b'\xd6', 'Lbl ')
token(b'\xd7', 'Goto ')
token(b'\xd8', 'Pause ')
token(b'\xd9', 'Stop')
token(b'\xda', 'IS>(')
token(b'\xdb', 'DS<(')
token(b'\xdc', 'Input ')
token(b'\xdd', 'Prompt ')
token(b'\xde', 'Disp ')
token(b'\xdf', 'DispGraph')
token(b'\xe0', 'Output(')
token(b'\xe1', 'ClrHome')
token(b'\xe2', 'Fill(')
token(b'\xe3', 'SortA(')
token(b'\xe4', 'SortD(')
token(b'\xe5', 'DispTable')
token(b'\xe6', 'Menu(')
token(b'\xe7', 'Send(')
token(b'\xe8', 'Get(')
token(b'\xe9', 'PlotsOn')
token(b'\xea', 'PlotsOff')
LIST_PREFIX = token(b'\xeb', 'ᴸ')
token(b'\xec', 'Plot1(')
token(b'\xed', 'Plot2(')
token(b'\xee', 'Plot3(')

# ── 0xEF xx: TI-84+ extended tokens ──────────────────────────────────────────

token(b'\xef\x00', 'setDate(',    cmd=forms.set_date)
token(b'\xef\x01', 'setTime(',    cmd=forms.set_time)
token(b'\xef\x02', 'checkTmr(',   func=forms.check_tmr)
token(b'\xef\x03', 'setDtFmt(',   cmd=forms.set_dt_fmt)
token(b'\xef\x04', 'setTmFmt(',   cmd=forms.set_tm_fmt)
token(b'\xef\x05', 'timeCnv(',    pure=pf.timecnv)
token(b'\xef\x06', 'dayOfWk(',    pure=pf.dayofwk)
token(b'\xef\x07', 'getDtStr(',   func=forms.get_dt_str)
token(b'\xef\x08', 'getTmStr(',   func=forms.get_tm_str)
token(b'\xef\x09', 'getDate',     res=Environment.get_date)
token(b'\xef\x0a', 'getTime',     res=Environment.get_time)
token(b'\xef\x0b', 'startTmr',   res=Environment.start_tmr)
token(b'\xef\x0c', 'getDtFmt',    res=Environment.get_dt_fmt)
token(b'\xef\x0d', 'getTmFmt',    res=Environment.get_tm_fmt)
token(b'\xef\x0e', 'isClockOn',   res=Environment.is_clock_on)
token(b'\xef\x0f', 'ClockOff',    cmd=Environment.clock_off)
token(b'\xef\x10', 'ClockOn',     cmd=Environment.clock_on)
token(b'\xef\x11', 'OpenLib(')
token(b'\xef\x12', 'ExecLib')
token(b'\xef\x13', 'invT(',       pure=pf.invt)
token(b'\xef\x14', 'χ²GOF-Test(')
token(b'\xef\x15', 'LinRegTInt ')
token(b'\xef\x16', 'Manual-Fit ')
token(b'\xef\x17', 'ZQuadrant1')
token(b'\xef\x18', 'ZFrac1/2')
token(b'\xef\x19', 'ZFrac1/3')
token(b'\xef\x1a', 'ZFrac1/4')
token(b'\xef\x1b', 'ZFrac1/5')
token(b'\xef\x1c', 'ZFrac1/8')
token(b'\xef\x1d', 'ZFrac1/10')
token(b'\xef\x1e', 'mathprintbox')
token(b'\xef\x30', '►n/d◄►Un/d')
token(b'\xef\x31', '►F◄►D')
token(b'\xef\x32', 'remainder(', pure=pf.remainder)
token(b'\xef\x33', 'Σ(', func=forms.sigma)
token(b'\xef\x34', 'logBASE(', pure=pf.log_base)
token(b'\xef\x35', 'randIntNoRep(', pure=pf.rand_int_no_rep)
token(b'\xef\x36', 'MATHPRINT')
token(b'\xef\x37', 'CLASSIC')
token(b'\xef\x38', 'n/d')
token(b'\xef\x39', 'Un/d')
token(b'\xef\x3a', 'AUTO')
token(b'\xef\x3b', 'DEC')
token(b'\xef\x3c', 'FRAC')
token(b'\xef\x3d', 'FRAC-APPROX')

# ── 0xF0–0xFF: power operators and regression commands ───────────────────────

POW      = token(b'\xf0', char='^', bp=(70, 69), op=op.pow)
XTH_ROOT = token(b'\xf1', 'ˣ√',     bp=(60, 61), op=pf.xth_root)

token(b'\xf2', '1-Var Stats ')
token(b'\xf3', '2-Var Stats ')
token(b'\xf4', 'LinReg(a+bx) ')
token(b'\xf5', 'ExpReg ')
token(b'\xf6', 'LnReg ')
token(b'\xf7', 'PwrReg ')
token(b'\xf8', 'Med-Med ')
token(b'\xf9', 'QuadReg ')
token(b'\xfa', 'ClrList ')
token(b'\xfb', 'ClrTable')
token(b'\xfc', 'Histogram')
token(b'\xfd', 'xyLine')
token(b'\xfe', 'Scatter')
token(b'\xff', 'LinReg(ax+b) ')


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


TOKEN_TABLE = TokenTable(ALL_TOKENS)


if __name__ == '__main__':
	for token in sorted(ASCII.values(), key=lambda t: t.char):
		print(ord(token.char), token)
