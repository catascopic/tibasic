from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import math, itertools
from environment import Environment
import purefunctions

@dataclass(eq=False)
class Token:
	code: bytes
	text: str
	key: str | None = None
	alias: set[str] = frozenset()
	bp: tuple | None = None		# (left_bp, right_bp) for binary operators
	binary_op: Any = None		# (lhs, rhs) -> value
	unary_op: Any = None		# (operand) -> value  (prefix or postfix)
	postfix: bool = False		# True for postfix unary operators
	func: Any = None			# (parser) -> value  for function tokens
	cmd: Any = None				# (parser) -> None  for command tokens
	resolve: Any = None			# (env) -> value  for variables and nullary tokens
	store: Any = None			# (env, value) -> None  for writable variables

	# ── Token type predicates ──────────────────────────────────────────────────

	def is_real_var(self) -> bool:
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

	def is_digit(self) -> bool:
		return 0x30 <= self.code[0] <= 0x39

	def is_name_char(self):
		return self.is_real_var() or self.is_digit()

	def can_start_atom(self) -> bool:
		return (
			self.resolve is not None or self.func is not None or
			self.is_digit() or self.is_list_var() or self.is_matrix_var() or
			self.is_string_var() or self.is_stat_var() or self.is_window_var() or
			self in {DOT, L_PAREN, L_BRACE, QUOTE, NEG, ANS, LIST_PREFIX}
		)


EOF_TOKEN = Token(b'\x00', '')

_SEEN: set[bytes] = set()

def token(
	code: bytes,
	text: str,
	*,
	alt: str | set[str] | None = None,
	key: str | None = None,
	bp: tuple[int, int] | None = None,
	binary_op=None,
	unary_op=None,
	postfix: bool = False,
	func: Callable | None = None,
	pure_func: Callable | None = None,
	cmd: Callable[[Environment], None] | None = None,
	resolve: Callable[[Environment], Any] | None = None,
	store: Callable[[Environment], None] | None = None,
) -> Token:

	if code in _SEEN:
		raise ValueError(f"Duplicate token code: {code!r} ({text!r})")
	_SEEN.add(code)
	alias = {text.lower()}
	if isinstance(alt, str):
		alias.add(alt.lower())
	elif alt is not None:
		alias.update(a.lower() for a in alt)
	if func is None and pure_func is not None:
		func = lambda parser, _f=pure_func: _f(*parser.parse_args())
	return Token(
		code, text, key, alias,
		bp, binary_op, unary_op, postfix, func, cmd, resolve, store,
	)


# ── Named syntactic tokens (referenced by identity in the parser) ──────────────

# Structural / delimiter
STORE	   = token(b'\x04', "→", key='`', alt=("->", 'store'))
L_BRACKET   = token(b'\x06', "[", key='[')
R_BRACKET   = token(b'\x07', "]", key=']')
L_BRACE	 = token(b'\x08', "{", key='{')
R_BRACE	 = token(b'\x09', "}", key='}')
L_PAREN	 = token(b'\x10', "(", key='(')
R_PAREN	 = token(b'\x11', ")", key=')')
QUOTE	   = token(b'\x2a', '"', key='"')
COMMA	   = token(b'\x2b', ",", key=',')
DOT		 = token(b'\x3a', ".", key='.')
COLON	   = token(b'\x3e', ":", key=':')
NEWLINE	 = token(b'\x3f', "↵", alt="newline")
PRGM		= token(b'\x5f', "prgm")
ANS		 = token(b'\x72', "Ans", resolve=lambda env: env.ans, store=lambda env, value: setattr(env, 'ans', value))
NEG		 = token(b'\xb0', "−", alt=('~', "neg"), key='~', unary_op=lambda x: -x)
DIM      = token(b'\xb5', "dim(", pure_func=purefunctions.dim)
LIST_PREFIX = token(b'\xeb', "∟", alt="list-prefix", key='#')

# Postfix operators
RAD		 = token(b'\x0a', "ʳ", alt="rad", postfix=True, unary_op=lambda x: x)
DEG		 = token(b'\x0b', "°", alt="deg", postfix=True, unary_op=math.radians)
INV		 = token(b'\x0c', "⁻¹", alt=("inv", '^-1'), postfix=True, unary_op=lambda x: 1 / x)
SQ		  = token(b'\x0d', "²", alt="^2", postfix=True, unary_op=lambda x: x ** 2)
TRANSPOSE   = token(b'\x0e', "ᵀ", alt=("T", 'transpose'), postfix=True, unary_op=Environment.matrix_transpose)
CUBE		= token(b'\x0f', "³", alt="^3", postfix=True, unary_op=lambda x: x ** 3)
FACT		= token(b'\x2d', "!", key='!', postfix=True, unary_op=Environment.factorial)

# Binary operators
SCI_E	   = token(b'\x3b', "ᴇ", alt="E", bp=(65, 66), binary_op=lambda a, b: a * (10 ** b))
OR		  = token(b'\x3c', "or", bp=(20, 21), binary_op=lambda a, b: 1.0 if a or b else 0.0)
XOR		 = token(b'\x3d', "xor", bp=(20, 21), binary_op=lambda a, b: 1.0 if bool(a) != bool(b) else 0.0)
AND		 = token(b'\x40', "and", bp=(30, 31), binary_op=lambda a, b: 1.0 if a and b else 0.0)
EQ		  = token(b'\x6a', "=", key='=', bp=(40, 41), binary_op=lambda a, b: 1.0 if a == b else 0.0)
LT		  = token(b'\x6b', "<", key='<', bp=(40, 41), binary_op=lambda a, b: 1.0 if a < b else 0.0)
GT		  = token(b'\x6c', ">", key='>', bp=(40, 41), binary_op=lambda a, b: 1.0 if a > b else 0.0)
LE		  = token(b'\x6d', "≤", alt="<=", bp=(40, 41), binary_op=lambda a, b: 1.0 if a <= b else 0.0)
GE		  = token(b'\x6e', "≥", alt=">=", bp=(40, 41), binary_op=lambda a, b: 1.0 if a >= b else 0.0)
NE		  = token(b'\x6f', "≠", alt="!=", bp=(40, 41), binary_op=lambda a, b: 1.0 if a != b else 0.0)
ADD		 = token(b'\x70', "+", key='+', bp=(50, 51), binary_op=lambda a, b: a + b)
SUB		 = token(b'\x71', "-", key='-', bp=(50, 51), binary_op=lambda a, b: a - b)
MUL		 = token(b'\x82', "*", key='*', bp=(60, 61), binary_op=Environment.list_mul)
DIV		 = token(b'\x83', "/", key='/', bp=(60, 61), binary_op=lambda a, b: a / b)
NPR		 = token(b'\x94', "nPr", bp=(60, 61), binary_op=Environment.npr)
NCR		 = token(b'\x95', "nCr", bp=(60, 61), binary_op=Environment.ncr)
RAND     = token(b'\xab', "rand", resolve=Environment.rand, store=lambda env, value: env.set_random_seed(value))
POW		 = token(b'\xf0', "^", key='^', bp=(70, 69), binary_op=lambda a, b: a ** b)
XROOT	 = token(b'\xf1', "×√", alt="xroot", bp=(60, 61), binary_op=lambda a, b: b ** (1 / a))

# ── Token list ─────────────────────────────────────────────────────────────────

TOKENS: list[Token] = [
	# One-byte tokens
	token(b'\x01', "►DMS", alt='to-DMS'),
	token(b'\x02', "►Dec", alt='to-Dec'),
	token(b'\x03', "►Frac", alt='to-Frac'),
	STORE,
	token(b'\x05', "Boxplot"),
	L_BRACKET, R_BRACKET, L_BRACE, R_BRACE,
	RAD, DEG, INV, SQ, TRANSPOSE, CUBE,
	L_PAREN, R_PAREN,
	token(b'\x12', "round(",   pure_func=purefunctions.round),
	token(b'\x13', "pxl-Test("),
	token(b'\x14', "augment("),
	token(b'\x15', "rowSwap("),
	token(b'\x16', "row+("),
	token(b'\x17', "*row("),
	token(b'\x18', "*row+("),
	token(b'\x19', "max(",    pure_func=purefunctions.max),
	token(b'\x1a', "min(",    pure_func=purefunctions.min),
	token(b'\x1b', "R►Pr(", alt='R-to-Pr'),
	token(b'\x1c', "R►Pθ(", alt='R-to-P-theta'),
	token(b'\x1d', "P►Rx(", alt='R-to-Px'),
	token(b'\x1e', "P►Ry(", alt='R-to-Py'),
	token(b'\x1f', "median(", pure_func=purefunctions.median),
	token(b'\x20', "randM("),
	token(b'\x21', "mean(",   pure_func=purefunctions.mean),
	token(b'\x22', "solve("),
	token(b'\x23', "seq(",    func=lambda parser: parser.env.call_seq(parser)),
	token(b'\x24', "fnInt(",  func=lambda parser: parser.env.call_fnint(parser)),
	token(b'\x25', "nDeriv(", func=lambda parser: parser.env.call_nderiv(parser)),
	token(b'\x27', "fMin("),
	token(b'\x28', "fMax("),
	token(b'\x29', " ", key=' '),
	QUOTE,
	COMMA,
	token(b'\x2c', "𝑖", alt="imaginary", resolve=lambda env: 1j),
	FACT,
	token(b'\x2e', "CubicReg "),
	token(b'\x2f', "QuartReg "),
	*[token(bytes([0x30 + i]), chr(0x30 + i), key=chr(0x30 + i)) for i in range(10)],
	DOT, SCI_E, OR, XOR, COLON, NEWLINE, AND,
	# Variables A–Z (0x41–0x5A)
	*[token(bytes([0x41 + i]), chr(0x41 + i), key=chr(0x41 + i)) for i in range(26)],
	token(b'\x5b', "θ", alt="theta"),
	PRGM,
	token(b'\x64', "Radian"),
	token(b'\x65', "Degree"),
	token(b'\x66', "Normal"),
	token(b'\x67', "Sci"),
	token(b'\x68', "Eng"),
	token(b'\x69', "Float"),
	EQ,
	LT,
	GT,
	LE,
	GE,
	NE,
	ADD,
	SUB,
	ANS,
	token(b'\x73', "Fix"),
	token(b'\x74', "Horiz"),
	token(b'\x75', "Full"),
	token(b'\x76', "Func"),
	token(b'\x77', "Param"),
	token(b'\x78', "Polar"),
	token(b'\x79', "Seq"),
	token(b'\x7a', "IndpntAuto"),
	token(b'\x7b', "IndpntAsk"),
	token(b'\x7c', "DependAuto"),
	token(b'\x7d', "DependAsk"),
	token(b'\x7f', "<squaremark>", alt='square-mark'),
	token(b'\x80', "<crossmark>",alt='cross-mark'),
	token(b'\x81', "<dotmark>", alt='dot-mark'),
	MUL,
	DIV,
	token(b'\x84', "Trace"),
	token(b'\x85', "ClrDraw"),
	token(b'\x86', "ZStandard"),
	token(b'\x87', "ZTrig"),
	token(b'\x88', "ZBox"),
	token(b'\x89', "Zoom In", alt="ZoomIn"),
	token(b'\x8a', "Zoom Out", alt="ZoomOut"),
	token(b'\x8b', "ZSquare"),
	token(b'\x8c', "ZInteger"),
	token(b'\x8d', "ZPrevious"),
	token(b'\x8e', "ZDecimal"),
	token(b'\x8f', "ZoomStat"),
	token(b'\x90', "ZoomRcl"),
	token(b'\x91', "PrintScreen"),
	token(b'\x92', "ZoomSto"),
	token(b'\x93', "Text("),
	NPR,
	NCR,
	token(b'\x96', "FnOn "),
	token(b'\x97', "FnOff "),
	token(b'\x98', "StorePic "),
	token(b'\x99', "RecallPic "),
	token(b'\x9a', "StoreGDB "),
	token(b'\x9b', "RecallGDB "),
	token(b'\x9c', "Line("),
	token(b'\x9d', "Vertical "),
	token(b'\x9e', "Pt-On("),
	token(b'\x9f', "Pt-Off("),
	token(b'\xa0', "Pt-Change("),
	token(b'\xa1', "Pxl-On("),
	token(b'\xa2', "Pxl-Off("),
	token(b'\xa3', "Pxl-Change("),
	token(b'\xa4', "Shade("),
	token(b'\xa5', "Circle("),
	token(b'\xa6', "Horizontal "),
	token(b'\xa7', "Tangent("),
	token(b'\xa8', "DrawInv"),
	token(b'\xa9', "DrawF"),
	RAND,
	token(b'\xac', "π", alt="pi", resolve=lambda env: math.pi),
	token(b'\xad', "getKey", resolve=Environment.get_key),
	token(b'\xae', "'", alt="apostrophe", key="'"),
	token(b'\xaf', "?", key='?'),
	NEG,
	token(b'\xb1', "int(",    pure_func=purefunctions.int_),
	token(b'\xb2', "abs(",    pure_func=purefunctions.abs),
	token(b'\xb3', "det(",    pure_func=purefunctions.det),
	token(b'\xb4', "identity(", pure_func=purefunctions.identity),
	DIM,
	token(b'\xb6', "sum(",    pure_func=purefunctions.sum),
	token(b'\xb7', "prod(",   pure_func=purefunctions.prod),
	token(b'\xb8', "not(",    pure_func=purefunctions.not_),
	token(b'\xb9', "iPart(",  pure_func=purefunctions.ipart),
	token(b'\xba', "fPart(",  pure_func=purefunctions.fpart),
	token(b'\xbc', "√(",  alt=("sqrt(", 'squareroot'), pure_func=purefunctions.sqrt),
	token(b'\xbd', "³√(", alt=("cbrt(", 'cuberoot'),   pure_func=purefunctions.cbrt),
	token(b'\xbe', "ln(",     pure_func=purefunctions.ln),
	token(b'\xbf', "e^(",     pure_func=purefunctions.exp),
	token(b'\xc0', "log(",    pure_func=purefunctions.log),
	token(b'\xc1', "10^(",    pure_func=purefunctions.pow10),
	token(b'\xc2', "sin(",    pure_func=purefunctions.sin),
	token(b'\xc3', "sin⁻¹(", alt="arcsin(",  pure_func=purefunctions.asin),
	token(b'\xc4', "cos(",    pure_func=purefunctions.cos),
	token(b'\xc5', "cos⁻¹(", alt="arccos(",  pure_func=purefunctions.acos),
	token(b'\xc6', "tan(",    pure_func=purefunctions.tan),
	token(b'\xc7', "tan⁻¹(", alt="arctan(",  pure_func=purefunctions.atan),
	token(b'\xc8', "sinh(",   pure_func=purefunctions.sinh),
	token(b'\xc9', "sinh⁻¹(", alt="arcsinh(", pure_func=purefunctions.asinh),
	token(b'\xca', "cosh(",   pure_func=purefunctions.cosh),
	token(b'\xcb', "cosh⁻¹(", alt="arccosh(", pure_func=purefunctions.acosh),
	token(b'\xcc', "tanh(",   pure_func=purefunctions.tanh),
	token(b'\xcd', "tanh⁻¹(", alt="arctanh(", pure_func=purefunctions.atanh),
	token(b'\xce', "If "),
	token(b'\xcf', "Then"),
	token(b'\xd0', "Else"),
	token(b'\xd1', "While "),
	token(b'\xd2', "Repeat "),
	token(b'\xd3', "For("),
	token(b'\xd4', "End"),
	token(b'\xd5', "Return"),
	token(b'\xd6', "Lbl "),
	token(b'\xd7', "Goto "),
	token(b'\xd8', "Pause "),
	token(b'\xd9', "Stop"),
	token(b'\xda', "IS>("),
	token(b'\xdb', "DS<("),
	token(b'\xdc', "Input "),
	token(b'\xdd', "Prompt "),
	token(b'\xde', "Disp "),
	token(b'\xdf', "DispGraph"),
	token(b'\xe0', "Output("),
	token(b'\xe1', "ClrHome"),
	token(b'\xe2', "Fill("),
	token(b'\xe3', "SortA("),
	token(b'\xe4', "SortD("),
	token(b'\xe5', "DispTable"),
	token(b'\xe6', "Menu("),
	token(b'\xe7', "Send("),
	token(b'\xe8', "Get("),
	token(b'\xe9', "PlotsOn"),
	token(b'\xea', "PlotsOff"),
	LIST_PREFIX,
	token(b'\xec', "Plot1("),
	token(b'\xed', "Plot2("),
	token(b'\xee', "Plot3("),
	POW,
	XROOT,
	token(b'\xf2', "1-Var Stats "),
	token(b'\xf3', "2-Var Stats "),
	token(b'\xf4', "LinReg(a+bx) "),
	token(b'\xf5', "ExpReg "),
	token(b'\xf6', "LnReg "),
	token(b'\xf7', "PwrReg "),
	token(b'\xf8', "Med-Med "),
	token(b'\xf9', "QuadReg "),
	token(b'\xfa', "ClrList "),
	token(b'\xfb', "ClrTable"),
	token(b'\xfc', "Histogram"),
	token(b'\xfd', "xyLine"),
	token(b'\xfe', "Scatter"),
	token(b'\xff', "LinReg(ax+b) "),

	# Two-byte: Matrix variables 0x5C xx
	*[token(bytes([0x5c, i]), f"[{chr(0x41 + i)}]", alt=f"mat{chr(0x41 + i)}") for i in range(10)],

	# Two-byte: List variables 0x5D xx
	*[token(bytes([0x5d, i]), f"L{chr(0x2081 + i)}", alt=f"L{i + 1}") for i in range(0, 6)],

	# Two-byte: Y= equation variables 0x5E xx
	*[token(bytes([0x5e, 0x10 + i]), f"Y{chr(0x2080 + (i + 1) % 10)}") for i in range(10)], 
	*[token(bytes([0x5e, 0x20 + i]), f"{x}{chr(0x2080 + n)}ₜ") for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))],

	*[token(bytes([0x5e, 0x40 + i]), f"r{chr(0x2081 + i)}") for i in range(6)],

	token(b'\x5e\x80', "u", alt="sequence-u"),
	token(b'\x5e\x81', "v", alt="sequence-v"),
	token(b'\x5e\x82', "w", alt="sequence-w"),

	# Two-byte: Picture variables 0x60 xx
	*[token(bytes([0x60, i]), f"Pic{(i + 1) % 10}") for i in range(10)],

	# Two-byte: GDB variables 0x61 xx
	*[token(bytes([0x61, i]), f"GDB{(i + 1) % 10}") for i in range(10)],

	# Two-byte: String variables 0xAA xx (Str1=0x00 … Str9=0x08, Str0=0x09)
	*[token(bytes([0xaa, i]), f"Str{(i + 1) % 10}") for i in range(10)],

	# Two-byte: Statistics variables 0x62 xx
	token(b'\x62\x01', "RegEq"),
	token(b'\x62\x02', "n"),
	token(b'\x62\x03', "x̄", alt="x-mean"),
	token(b'\x62\x04', "Σx", alt="sum-x"),
	token(b'\x62\x05', "Σx²", alt="sum-x^2"),
	token(b'\x62\x06', "Sx"),
	token(b'\x62\x07', "σx", alt="sigma-x"),
	token(b'\x62\x08', "minX"),
	token(b'\x62\x09', "maxX"),
	token(b'\x62\x0a', "minY"),
	token(b'\x62\x0b', "maxY"),
	token(b'\x62\x0c', "ȳ", alt="y-mean"),
	token(b'\x62\x0d', "Σy", alt="sum-y"),
	token(b'\x62\x0e', "Σy²", alt="sum-y^2"),
	token(b'\x62\x0f', "Sy"),
	token(b'\x62\x10', "σy", alt="sigma-y"),
	token(b'\x62\x11', "Σxy", alt="sum-xy"),
	token(b'\x62\x12', "r"),
	token(b'\x62\x13', "Med"),
	token(b'\x62\x14', "Q1"),
	token(b'\x62\x15', "Q3"),
	token(b'\x62\x16', "a"),
	token(b'\x62\x17', "b"),
	token(b'\x62\x18', "c"),
	token(b'\x62\x19', "d"),
	token(b'\x62\x1a', "e"),
	token(b'\x62\x1b', "x₁", alt="x1"),
	token(b'\x62\x1c', "x₂", alt="x2"),
	token(b'\x62\x1d', "x₃", alt="x3"),
	token(b'\x62\x1e', "y₁", alt="y1"),
	token(b'\x62\x1f', "y₂", alt="y2"),
	token(b'\x62\x20', "y₃", alt="y3"),
	token(b'\x62\x21', "n"),
	token(b'\x62\x22', "p"),
	token(b'\x62\x23', "z"),
	token(b'\x62\x24', "t"),
	token(b'\x62\x25', "χ²", alt="chi2"),
	token(b'\x62\x26', "F"),
	token(b'\x62\x27', "df"),
	token(b'\x62\x28', "p̂", alt="p-hat"),
	token(b'\x62\x29', "p̂₁", alt="p-hat1"),
	token(b'\x62\x2a', "p̂₂", alt="p-hat2"),
	token(b'\x62\x2b', "x̄₁", alt="x-mean1"),
	token(b'\x62\x2c', "Sx₁", alt="Sx1"),
	token(b'\x62\x2d', "n₁", alt="n1"),
	token(b'\x62\x2e', "x̄₂", alt="x-mean2"),
	token(b'\x62\x2f', "Sx₂", alt="Sx2"),
	token(b'\x62\x30', "n₂", alt="n2"),
	token(b'\x62\x31', "Sxp"),
	token(b'\x62\x32', "lower"),
	token(b'\x62\x33', "upper"),
	token(b'\x62\x34', "s"),
	token(b'\x62\x35', "r²", alt="r^2"),
	token(b'\x62\x36', "R²", alt="R^2"),
	token(b'\x62\x37', "Factor df", alt="FactorDF"),
	token(b'\x62\x38', "Factor SS", alt="FactorSS"),
	token(b'\x62\x39', "Factor MS", alt="FactorMS"),
	token(b'\x62\x3a', "Error df", alt="ErrorDF"),
	token(b'\x62\x3b', "Error SS", alt="ErrorSS"),
	token(b'\x62\x3c', "Error MS", alt="ErrorMS"),

	# Two-byte: Window / Finance variables 0x63 xx
	token(b'\x63\x02', "Xscl"),
	token(b'\x63\x03', "Yscl"),
	token(b'\x63\x0a', "Xmin"),
	token(b'\x63\x0b', "Xmax"),
	token(b'\x63\x0c', "Ymin"),
	token(b'\x63\x0d', "Ymax"),
	token(b'\x63\x0e', "Tmin"),
	token(b'\x63\x0f', "Tmax"),
	token(b'\x63\x10', "θmin", alt="theta-min"),
	token(b'\x63\x11', "θmax", alt="theta-max"),
	token(b'\x63\x1a', "TblStart"),
	token(b'\x63\x1b', "PlotStart"),
	token(b'\x63\x1d', "nMax"),
	token(b'\x63\x1f', "nMin"),
	token(b'\x63\x21', "ΔTbl", alt="dTbl"),
	token(b'\x63\x22', "Tstep"),
	token(b'\x63\x23', "θstep", alt="theta-step"),
	token(b'\x63\x26', "ΔX", alt="dX"),
	token(b'\x63\x27', "ΔY", alt="dY"),
	token(b'\x63\x28', "XFact"),
	token(b'\x63\x29', "YFact"),
	token(b'\x63\x2b', "N"),
	token(b'\x63\x2c', "I%"),
	token(b'\x63\x2d', "PV"),
	token(b'\x63\x2e', "PMT"),
	token(b'\x63\x2f', "FV"),
	token(b'\x63\x30', "P/Y"),
	token(b'\x63\x31', "C/Y"),
	token(b'\x63\x34', "PlotStep"),
	token(b'\x63\x36', "Xres"),

	# Two-byte: Graph format tokens 0x7E xx
	token(b'\x7e\x00', "Sequential"),
	token(b'\x7e\x01', "Simul"),
	token(b'\x7e\x02', "PolarGC"),
	token(b'\x7e\x03', "RectGC"),
	token(b'\x7e\x04', "CoordOn"),
	token(b'\x7e\x05', "CoordOff"),
	token(b'\x7e\x06', "Connected"),
	token(b'\x7e\x07', "Dot"),
	token(b'\x7e\x08', "AxesOn"),
	token(b'\x7e\x09', "AxesOff"),
	token(b'\x7e\x0a', "GridOn"),
	token(b'\x7e\x0b', "GridOff"),
	token(b'\x7e\x0c', "LabelOn"),
	token(b'\x7e\x0d', "LabelOff"),
	token(b'\x7e\x0e', "Web"),
	token(b'\x7e\x0f', "Time"),
	token(b'\x7e\x10', "uvAxes"),
	token(b'\x7e\x11', "vwAxes"),
	token(b'\x7e\x12', "uwAxes"),

	# Two-byte: Miscellaneous tokens 0xBB xx
	token(b'\xbb\x00', "npv("),
	token(b'\xbb\x01', "irr("),
	token(b'\xbb\x02', "bal("),
	token(b'\xbb\x03', "Σprn(", alt="sum-prn"),
	token(b'\xbb\x04', "ΣInt(", alt="sum-int"),
	token(b'\xbb\x05', "►Nom(", alt="to-Nom"),
	token(b'\xbb\x06', "►Eff(", alt="to-Eff"),
	token(b'\xbb\x07', "dbd("),
	token(b'\xbb\x08', "lcm(",      pure_func=purefunctions.lcm),
	token(b'\xbb\x09', "gcd(",      pure_func=purefunctions.gcd),
	token(b'\xbb\x0a', "randInt(",  pure_func=purefunctions.randint),
	token(b'\xbb\x0b', "randBin("),
	token(b'\xbb\x0c', "sub(",      pure_func=purefunctions.sub),
	token(b'\xbb\x0d', "stdDev("),
	token(b'\xbb\x0e', "variance("),
	token(b'\xbb\x0f', "inString(", pure_func=purefunctions.instring),
	token(b'\xbb\x10', "normalcdf("),
	token(b'\xbb\x11', "invNorm("),
	token(b'\xbb\x12', "tcdf("),
	token(b'\xbb\x13', "χ²cdf(", alt="chi2cdf"),
	token(b'\xbb\x14', "Fcdf("),
	token(b'\xbb\x15', "binompdf("),
	token(b'\xbb\x16', "binomcdf("),
	token(b'\xbb\x17', "poissonpdf("),
	token(b'\xbb\x18', "poissoncdf("),
	token(b'\xbb\x19', "geometpdf("),
	token(b'\xbb\x1a', "geometcdf("),
	token(b'\xbb\x1b', "normalpdf("),
	token(b'\xbb\x1c', "tpdf("),
	token(b'\xbb\x1d', "χ²pdf(", alt="chi2pdf"),
	token(b'\xbb\x1e', "Fpdf("),
	token(b'\xbb\x1f', "randNorm(", pure_func=purefunctions.randnorm),
	token(b'\xbb\x20', "tvm_Pmt"),
	token(b'\xbb\x21', "tvm_I%"),
	token(b'\xbb\x22', "tvm_PV"),
	token(b'\xbb\x23', "tvm_N"),
	token(b'\xbb\x24', "tvm_FV"),
	token(b'\xbb\x25', "conj(",   pure_func=purefunctions.conj),
	token(b'\xbb\x26', "real(",   pure_func=purefunctions.real),
	token(b'\xbb\x27', "imag(",   pure_func=purefunctions.imag),
	token(b'\xbb\x28', "angle(",  pure_func=purefunctions.angle),
	token(b'\xbb\x29', "cumSum(", pure_func=purefunctions.cumsum),
	token(b'\xbb\x2a', "expr("),
	token(b'\xbb\x2b', "length(", pure_func=purefunctions.length),
	token(b'\xbb\x2c', "ΔList(",  alt="dList(", pure_func=purefunctions.delta_list),
	token(b'\xbb\x2d', "ref("),
	token(b'\xbb\x2e', "rref("),
	token(b'\xbb\x2f', "►Rect", alt="to-Rect"),
	token(b'\xbb\x30', "►Polar", alt="to-Polar"),
	token(b'\xbb\x31', "𝑒", resolve=lambda env: math.e),
	token(b'\xbb\x32', "SinReg "),
	token(b'\xbb\x33', "Logistic "),
	token(b'\xbb\x34', "LinRegTTest "),
	token(b'\xbb\x35', "ShadeNorm("),
	token(b'\xbb\x36', "Shade_t("),
	token(b'\xbb\x37', "Shadeχ²(", alt="shade-chi^2"),
	token(b'\xbb\x38', "ShadeF("),
	token(b'\xbb\x39', "Matr►list(", alt="Matr-to-list"),
	token(b'\xbb\x3a', "List►matr(", alt="List-to-matr"),
	token(b'\xbb\x3b', "Z-Test("),
	token(b'\xbb\x3c', "T-Test"),
	token(b'\xbb\x3d', "2-SampZTest("),
	token(b'\xbb\x3e', "1-PropZTest("),
	token(b'\xbb\x3f', "2-PropZTest("),
	token(b'\xbb\x40', "χ²-Test(", alt="chi^2-test"),
	token(b'\xbb\x41', "ZInterval "),
	token(b'\xbb\x42', "2-SampZInt("),
	token(b'\xbb\x43', "1-PropZInt("),
	token(b'\xbb\x44', "2-PropZInt("),
	token(b'\xbb\x45', "GraphStyle("),
	token(b'\xbb\x46', "2-SampTTest "),
	token(b'\xbb\x47', "2-SampFTest "),
	token(b'\xbb\x48', "TInterval "),
	token(b'\xbb\x49', "2-SampTInt "),
	token(b'\xbb\x4a', "SetUpEditor "),
	token(b'\xbb\x4b', "Pmt_End"),
	token(b'\xbb\x4c', "Pmt_Bgn"),
	token(b'\xbb\x4d', "Real"),
	token(b'\xbb\x4e', "re^θi", alt=('re^theta-i', "polar_complex")),
	token(b'\xbb\x4f', "a+bi", alt="rect_complex"),
	token(b'\xbb\x50', "ExprOn"),
	token(b'\xbb\x51', "ExprOff"),
	token(b'\xbb\x52', "ClrAllLists"),
	token(b'\xbb\x53', "GetCalc("),
	token(b'\xbb\x54', "DelVar "),
	token(b'\xbb\x55', "Equ►String(", alt="Equ-to-Str"),
	token(b'\xbb\x56', "String►Equ(", alt="Str-to-Equ"),
	token(b'\xbb\x57', "Clear Entries", alt="ClearEntries"),
	token(b'\xbb\x58', "Select("),
	token(b'\xbb\x59', "ANOVA("),
	token(b'\xbb\x5a', "ModBoxplot"),
	token(b'\xbb\x5b', "NormProbPlot"),
	token(b'\xbb\x64', "G-T"),
	token(b'\xbb\x65', "ZoomFit"),
	token(b'\xbb\x66', "DiagnosticOn"),
	token(b'\xbb\x67', "DiagnosticOff"),
	token(b'\xbb\x68', "Archive "),
	token(b'\xbb\x69', "UnArchive "),
	token(b'\xbb\x6a', "Asm("),
	token(b'\xbb\x6b', "AsmComp("),
	token(b'\xbb\x6c', "AsmPrgm"),
	token(b'\xbb\x6d', "<compiledasm>"),

	# Accented Latin characters (0xBB6E–0xBB99; 0xBB7E unused — uppercase I-acute absent)
	*[token(bytes([0xBB, b]), ch, alt=name)
		for b, ch, name in [
			(0x6e, "Á", "A-acute"),	(0x6f, "À", "A-grave"),	(0x70, "Â", "A-circumflex"), (0x71, "Ä", "A-umlaut"),
			(0x72, "á", "a-acute"),	(0x73, "à", "a-grave"),	(0x74, "â", "a-circumflex"), (0x75, "ä", "a-umlaut"),
			(0x76, "É", "E-acute"),	(0x77, "È", "E-grave"),	(0x78, "Ê", "E-circumflex"), (0x79, "Ë", "E-umlaut"),
			(0x7a, "é", "e-acute"),	(0x7b, "è", "e-grave"),	(0x7c, "ê", "e-circumflex"), (0x7d, "ë", "e-umlaut"),
			(0x7f, "Ì", "I-grave"),	(0x80, "Î", "I-circumflex"),(0x81, "Ï", "I-umlaut"),
			(0x82, "í", "i-acute"),	(0x83, "ì", "i-grave"),	(0x84, "î", "i-circumflex"), (0x85, "ï", "i-umlaut"),
			(0x86, "Ó", "O-acute"),	(0x87, "Ò", "O-grave"),	(0x88, "Ô", "O-circumflex"), (0x89, "Ö", "O-umlaut"),
			(0x8a, "ó", "o-acute"),	(0x8b, "ò", "o-grave"),	(0x8c, "ô", "o-circumflex"), (0x8d, "ö", "o-umlaut"),
			(0x8e, "Ú", "U-acute"),	(0x8f, "Ù", "U-grave"),	(0x90, "Û", "U-circumflex"), (0x91, "Ü", "U-umlaut"),
			(0x92, "ú", "u-acute"),	(0x93, "ù", "u-grave"),	(0x94, "û", "u-circumflex"), (0x95, "ü", "u-umlaut"),
			(0x96, "Ç", "C-cedilla"),  (0x97, "ç", "c-cedilla"),
			(0x98, "Ñ", "N-tilde"),	(0x99, "ñ", "n-tilde"),
		]],

	token(b'\xbb\x9a', "´", alt="acute-accent"),
	token(b'\xbb\x9b', "`", alt="grave-accent"),
	token(b'\xbb\x9c', "¨", alt="umlaut-accent"),
	token(b'\xbb\x9d', "¿", alt="?-inverted"),
	token(b'\xbb\x9e', "¡", alt="!-inverted"),
	token(b'\xbb\x9f', "α", alt="alpha"),
	token(b'\xbb\xa0', "β", alt="beta"),
	token(b'\xbb\xa1', "γ", alt="gamma"),
	token(b'\xbb\xa2', "Δ", alt="Delta"),
	token(b'\xbb\xa3', "δ", alt="delta"),
	token(b'\xbb\xa4', "ε", alt="epsilon"),
	token(b'\xbb\xa5', "λ", alt="lambda"),
	token(b'\xbb\xa6', "μ", alt="mu"),
	token(b'\xbb\xa7', "π", alt="pi-non-math"),
	token(b'\xbb\xa8', "ρ", alt="rho"),
	token(b'\xbb\xa9', "Σ", alt="Sigma"),
	token(b'\xbb\xab', "φ", alt="phi"),
	token(b'\xbb\xac', "Ω", alt="Omega"),
	token(b'\xbb\xad', "ψ", alt="psi"),
	token(b'\xbb\xae', "χ", alt="chi"),
	token(b'\xbb\xaf', "F"),

	# Lowercase letters a–k (0xBBB0–0xBBBA; 0xBBBB is unused)
	*[token(bytes([0xBB, 0xB0 + i]), chr(0x61 + i), key=chr(0x61 + i)) for i in range(11)],
	# Lowercase letters l–z (0xBBBC–0xBBCA)
	*[token(bytes([0xBB, 0xBC + i]), chr(0x6C + i), key=chr(0x6C + i)) for i in range(15)],

	token(b'\xbb\xcb', "σ", alt="sigma"),
	token(b'\xbb\xcc', "τ", alt="tau"),
	token(b'\xbb\xcd', "Í", alt="I-acute"),
	token(b'\xbb\xce', "GarbageCollect"),
	token(b'\xbb\xcf', "~", alt="tilde"),
	token(b'\xbb\xd1', "@", alt="at-sign",   key='@'),
	token(b'\xbb\xd2', "#", alt="hash",   key='#'),
	token(b'\xbb\xd3', "$", alt="dollar",	key='$'),
	token(b'\xbb\xd4', "&", alt="ampersand", key='&'),
	token(b'\xbb\xd5', "`",alt="backtick"),
	token(b'\xbb\xd6', ";", alt="semicolon", key=';'),
	token(b'\xbb\xd7', "\\", alt="backslash", key='\\'),
	token(b'\xbb\xd8', "|", alt="pipe",  key='|'),
	token(b'\xbb\xd9', "_", alt="underscore",key='_'),
	token(b'\xbb\xda', "%", alt="percent",   key='%'),
	token(b'\xbb\xdb', "…", alt="ellipsis"),
	token(b'\xbb\xdc', "∠", alt="angle"),
	token(b'\xbb\xdd', "ß", alt="sharp-s"),
	token(b'\xbb\xde', "x", alt="superscript-x"),
	token(b'\xbb\xdf', "T", alt="subscript-t"),

	*[token(bytes([0xBB, 0xE0 + i]), chr(0x2080 + i), alt=f"subscript-{i}") for i in range(10)],
	token(b'\xbb\xea', "₁₀", alt=f"subscript-10"),

	token(b'\xbb\xeb', "←", alt="left-arrow"),
	token(b'\xbb\xec', "→", alt="right-arrow"),
	token(b'\xbb\xed', "↑", alt="up-arrow"),
	token(b'\xbb\xee', "↓", alt="down-arrow"),
	token(b'\xbb\xf0', "x"),
	token(b'\xbb\xf1', "∫", alt="integral"),
	token(b'\xbb\xf2', "🡅"),
	token(b'\xbb\xf3', "🡇"),
	token(b'\xbb\xf4', "√", alt="root"),
	token(b'\xbb\xf5', "<funcon>", alt='function-on'),

	# Two-byte: TI-84+ extended tokens 0xEF xx
	token(b'\xef\x00', "setDate("),
	token(b'\xef\x01', "setTime("),
	token(b'\xef\x02', "checkTmr("),
	token(b'\xef\x03', "setDtFmt("),
	token(b'\xef\x04', "setTmFmt("),
	token(b'\xef\x05', "timeCnv("),
	token(b'\xef\x06', "dayOfWk("),
	token(b'\xef\x07', "getDtStr("),
	token(b'\xef\x08', "getTmStr("),
	token(b'\xef\x09', "getDate",   resolve=Environment.get_date),
	token(b'\xef\x0a', "getTime",   resolve=Environment.get_time),
	token(b'\xef\x0b', "startTmr",  resolve=Environment.start_tmr),
	token(b'\xef\x0c', "getDtFmt",  resolve=Environment.get_dt_fmt),
	token(b'\xef\x0d', "getTmFmt",  resolve=Environment.get_tm_fmt),
	token(b'\xef\x0e', "isClockOn", resolve=Environment.is_clock_on),
	token(b'\xef\x0f', "ClockOff"),
	token(b'\xef\x10', "ClockOn"),
	token(b'\xef\x11', "OpenLib("),
	token(b'\xef\x12', "ExecLib"),
	token(b'\xef\x13', "invT("),
	token(b'\xef\x14', "χ²GOF-Test(", alt="chi^2-GOF-Test"),
	token(b'\xef\x15', "LinRegTInt "),
	token(b'\xef\x16', "Manual-Fit "),
	token(b'\xef\x17', "ZQuadrant1"),
	token(b'\xef\x18', "ZFrac1/2"),
	token(b'\xef\x19', "ZFrac1/3"),
	token(b'\xef\x1a', "ZFrac1/4"),
	token(b'\xef\x1b', "ZFrac1/5"),
	token(b'\xef\x1c', "ZFrac1/8"),
	token(b'\xef\x1d', "ZFrac1/10"),
	token(b'\xef\x1e', "<mathprintbox>"),
	token(b'\xef\x30', "►n/d◄►Un/d"),
	token(b'\xef\x31', "►F◄►D"),
	token(b'\xef\x32', "remainder(",    pure_func=purefunctions.remainder),
	token(b'\xef\x33', "Σ(",  alt='sigma(', func=lambda parser: parser.env.call_sigma(parser)),
	token(b'\xef\x34', "logBASE(",      pure_func=purefunctions.logbase),
	token(b'\xef\x35', "randIntNoRep(", pure_func=purefunctions.randintnotrep),
	token(b'\xef\x36', "MATHPRINT"),
	token(b'\xef\x37', "CLASSIC"),
	token(b'\xef\x38', "n/d"),
	token(b'\xef\x39', "Un/d"),
	token(b'\xef\x3a', "AUTO"),
	token(b'\xef\x3b', "DEC"),
	token(b'\xef\x3c', "FRAC"),
	token(b'\xef\x3d', "FRAC-APPROX"),
]

# Assign resolve/store for variable tokens (done post-creation to capture token identity)
for _t in TOKENS:
	if _t.resolve is not None or _t.store is not None:
		continue  # already set on named constants (ANS, RAND, etc.)
	if _t.is_real_var():
		_t.resolve = lambda env, t=_t: env.reals.get(t, 0.0)
		_t.store   = lambda env, value, t=_t: env.reals.__setitem__(t, value)
	elif _t.is_list_var():
		_t.resolve = lambda env, t=_t: env.lists[t]
		_t.store   = lambda env, value, t=_t: env.lists.__setitem__(t, value)
	elif _t.is_matrix_var():
		_t.resolve = lambda env, t=_t: env.matrices[t]
		_t.store   = lambda env, value, t=_t: env.matrices.__setitem__(t, value)
	elif _t.is_string_var():
		_t.resolve = lambda env, t=_t: env.strings[t]
		_t.store   = lambda env, value, t=_t: env.strings.__setitem__(t, value)
	elif _t.is_stat_var():
		_t.resolve = lambda env, t=_t: env.stat[t]
	elif _t.is_window_var():
		_t.resolve = lambda env, t=_t: env.window[t]
		_t.store   = lambda env, value, t=_t: env.window.__setitem__(t, value)


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
			raise ValueError(f"Duplicate: {token}")
		if len(code) == 1:
			check[code[0]] = NullToken(code)
		elif code[0] == 0xBB:
			check_misc[code[1]] = NullToken(code)

	for token in TOKENS:
		old_len = len(duplicate)
		duplicate.add(token.code)
		if old_len == len(duplicate):
			raise ValueError(f"Duplicate: {token}")
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

	for token in sorted(TOKENS, key=lambda t: t.code):
		print(token.code.hex(), token.display.decode('latin-1'))


