import enum
import itertools
from collections.abc import Sequence
from dataclasses import dataclass


class TokenFlags(enum.Flag):
	FUNCTION    = enum.auto()   # takes ArgParser, returns a value
	COMMAND     = enum.auto()   # takes ArgParser, returns None (side-effect only)
	NULLARY     = enum.auto()   # constant / computed value, no args
	POSTFIX     = enum.auto()   # unary postfix operator
	CONVERTER   = enum.auto()   # ►DMS, ►Dec, ►Frac — postfix type conversion
	OPENS_PAREN = enum.auto()   # opens a ( context in capture() (functions + L_PAREN)


_F   = TokenFlags.FUNCTION | TokenFlags.OPENS_PAREN
_CMD = TokenFlags.COMMAND
_N   = TokenFlags.NULLARY
_NF  = TokenFlags.NULLARY | TokenFlags.FUNCTION | TokenFlags.OPENS_PAREN
_PFX = TokenFlags.POSTFIX
_CVT = TokenFlags.CONVERTER


@dataclass(slots=True, eq=False)
class Token:
	code:  bytes
	ascii: str | None
	text:  str
	bp:    tuple[int, int] | None = None
	flags: TokenFlags = TokenFlags(0)

	# ── Token type predicates ──────────────────────────────────────────────────

	def is_numeric_var(self) -> bool:
		return 0x41 <= self.code[0] <= 0x5b

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
			or self.is_numeric_var()
			or self.is_list_var()
			or self.is_matrix_var()
			or self.is_string_var()
			or self.is_stat_var()
			or self.is_window_var()
			or TokenFlags.FUNCTION in self.flags
			or TokenFlags.NULLARY in self.flags
			or self in {DOT, L_PAREN, L_BRACE, L_BRACKET, QUOTE, NEG, LIST_PREFIX}
		)

	def __repr__(self):
		return f'{self.code.hex().upper()}:{self.text}'


EOF_TOKEN = Token(b'\x00', None, '<END-OF-INPUT>')

_SEEN: set[bytes] = set()


def token(
	code:  bytes,
	text:  str = None,
	ascii: str = None,
	*,
	bp:    tuple[int, int] | None = None,
	flags: TokenFlags = TokenFlags(0),
) -> Token:
	if code in _SEEN:
		raise ValueError(f'Duplicate token code: {code!r} ({text!r})')
	_SEEN.add(code)
	return Token(code, ascii, text or ascii, bp, flags)


# ── Named syntactic tokens (referenced by identity in the parser) ──────────────

DIGITS   = [token(bytes([0x30 + i]), ascii=chr(0x30 + i)) for i in range(10)]
LETTERS  = [token(bytes([0x41 + i]), ascii=chr(0x41 + i)) for i in range(26)]
THETA    = token(b'\x5b', 'θ')
MATRICES = [token(bytes([0x5c, i]), f'[{chr(0x41 + i)}]') for i in range(10)]
LISTS    = [token(bytes([0x5d, i]), f'L{chr(0x2081 + i)}') for i in range(6)]
PICTURES = [token(bytes([0x60, i]), f'Pic{(i + 1) % 10}') for i in range(10)]
GDBS     = [token(bytes([0x61, i]), f'GDB{(i + 1) % 10}') for i in range(10)]
STRINGS  = [token(bytes([0xaa, i]), f'Str{(i + 1) % 10}') for i in range(10)]


# Structural / delimiter
STORE       = token(b'\x04', '→')
L_BRACKET   = token(b'\x06', ascii='[')
R_BRACKET   = token(b'\x07', ascii=']')
L_BRACE     = token(b'\x08', ascii='{')
R_BRACE     = token(b'\x09', ascii='}')
L_PAREN     = token(b'\x10', ascii='(', flags=TokenFlags.OPENS_PAREN)
R_PAREN     = token(b'\x11', ascii=')')
QUOTE       = token(b'\x2a', ascii='"')
COMMA       = token(b'\x2b', ascii=',')
DOT         = token(b'\x3a', ascii='.')
COLON       = token(b'\x3e', ascii=':')
NEWLINE     = token(b'\x3f', ascii='\n')
PRGM        = token(b'\x5f', 'prgm')
ANS         = token(b'\x72', 'Ans',  flags=_NF)
NEG         = token(b'\xb0', ascii='−')
APOS        = token(b'\xae', ascii="'")
DIM         = token(b'\xb5', 'dim(', flags=_F)
LIST_PREFIX = token(b'\xeb', '∟')

# Postfix operators (RAD/DEG handled specially in parser via eat_if)
RAD         = token(b'\x0a', 'ʳ')
DEG         = token(b'\x0b', '°')
INV         = token(b'\x0c', '¹',  flags=_PFX)
SQ          = token(b'\x0d', '²',  flags=_PFX)
TRANSPOSE   = token(b'\x0e', 'ᵀ',  flags=_PFX)
CUBE        = token(b'\x0f', '³',  flags=_PFX)
FACT        = token(b'\x2d', ascii='!', flags=_PFX)

# Binary operators
SCI_E       = token(b'\x3b', 'ᴇ')
OR          = token(b'\x3c', ' or ',    bp=(20, 21))
XOR         = token(b'\x3d', ' xor ',   bp=(20, 21))
AND         = token(b'\x40', ' and ',   bp=(30, 31))
EQ          = token(b'\x6a', ascii='=', bp=(40, 41))
LT          = token(b'\x6b', ascii='<', bp=(40, 41))
GT          = token(b'\x6c', ascii='>', bp=(40, 41))
LE          = token(b'\x6d', '≤',       bp=(40, 41))
GE          = token(b'\x6e', '≥',       bp=(40, 41))
NE          = token(b'\x6f', '≠',       bp=(40, 41))
ADD         = token(b'\x70', ascii='+', bp=(50, 51))
SUB         = token(b'\x71', ascii='-', bp=(50, 51))
MUL         = token(b'\x82', ascii='*', bp=(60, 61))
DIV         = token(b'\x83', ascii='/', bp=(60, 61))
NPR         = token(b'\x94', 'nPr',     bp=(60, 61))
NCR         = token(b'\x95', 'nCr',     bp=(60, 61))
RAND        = token(b'\xab', 'rand',    flags=_NF)
POW         = token(b'\xf0', ascii='^', bp=(70, 69))
XTH_ROOT    = token(b'\xf1', 'ˣ√',      bp=(60, 61))

# ── Token list ─────────────────────────────────────────────────────────────────

TOKENS: list[Token] = [
	# One-byte tokens
	token(b'\x01', '►DMS',   flags=_CVT),
	token(b'\x02', '►Dec',   flags=_CVT),
	token(b'\x03', '►Frac',  flags=_CVT),
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
	token(b'\x12', 'round(',    flags=_F),
	token(b'\x13', 'pxl-Test('),
	token(b'\x14', 'augment(',  flags=_F),
	token(b'\x15', 'rowSwap(',  flags=_F),
	token(b'\x16', 'row+(',     flags=_F),
	token(b'\x17', '*row(',     flags=_F),
	token(b'\x18', '*row+(',    flags=_F),
	token(b'\x19', 'max(',      flags=_F),
	token(b'\x1a', 'min(',      flags=_F),
	token(b'\x1b', 'R►Pr(',    flags=_F),
	token(b'\x1c', 'R►Pθ(',   flags=_F),
	token(b'\x1d', 'P►Rx(',    flags=_F),
	token(b'\x1e', 'P►Ry(',    flags=_F),
	token(b'\x1f', 'median(',   flags=_F),
	token(b'\x20', 'randM(',    flags=_F),
	token(b'\x21', 'mean(',     flags=_F),
	token(b'\x22', 'solve('),
	token(b'\x23', 'seq(',      flags=_F),
	token(b'\x24', 'fnInt(',    flags=_F),
	token(b'\x25', 'nDeriv(',   flags=_F),
	token(b'\x27', 'fMin('),
	token(b'\x28', 'fMax('),
	token(b'\x29', ascii=' '),
	QUOTE,
	COMMA,
	token(b'\x2c', '𝑖',         flags=_N),
	FACT,
	token(b'\x2e', 'CubicReg '),
	token(b'\x2f', 'QuartReg '),
	*DIGITS,
	DOT,
	SCI_E,
	OR,
	XOR,
	COLON,
	NEWLINE,
	AND,
	*LETTERS,
	THETA,
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
	token(b'\x7f', '▫'),
	token(b'\x80', '﹢'),
	token(b'\x81', '·'),
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
	token(b'\xa8', 'DrawInv '),
	token(b'\xa9', 'DrawF '),
	RAND,
	token(b'\xac', 'π',          flags=_N),
	token(b'\xad', 'getKey',     flags=_N),
	APOS,
	token(b'\xaf', ascii='?'),
	NEG,
	token(b'\xb1', 'int(',       flags=_F),
	token(b'\xb2', 'abs(',       flags=_F),
	token(b'\xb3', 'det(',       flags=_F),
	token(b'\xb4', 'identity(',  flags=_F),
	DIM,
	token(b'\xb6', 'sum(',       flags=_F),
	token(b'\xb7', 'prod(',      flags=_F),
	token(b'\xb8', 'not(',       flags=_F),
	token(b'\xb9', 'iPart(',     flags=_F),
	token(b'\xba', 'fPart(',     flags=_F),
	token(b'\xbc', '√(',         flags=_F),
	token(b'\xbd', '³√(',        flags=_F),
	token(b'\xbe', 'ln(',        flags=_F),
	token(b'\xbf', '𝑒^(',        flags=_F),
	token(b'\xc0', 'log(',       flags=_F),
	token(b'\xc1', '⑽^(',        flags=_F),
	token(b'\xc2', 'sin(',       flags=_F),
	token(b'\xc3', 'sin¹(',      flags=_F),
	token(b'\xc4', 'cos(',       flags=_F),
	token(b'\xc5', 'cos¹(',      flags=_F),
	token(b'\xc6', 'tan(',       flags=_F),
	token(b'\xc7', 'tan¹(',      flags=_F),
	token(b'\xc8', 'sinh(',      flags=_F),
	token(b'\xc9', 'sinh¹(',     flags=_F),
	token(b'\xca', 'cosh(',      flags=_F),
	token(b'\xcb', 'cosh¹(',     flags=_F),
	token(b'\xcc', 'tanh(',      flags=_F),
	token(b'\xcd', 'tanh¹(',     flags=_F),
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
	token(b'\xe2', 'Fill(',      flags=_CMD),
	token(b'\xe3', 'SortA(',     flags=_CMD),
	token(b'\xe4', 'SortD(',     flags=_CMD),
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
	XTH_ROOT,
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
	*MATRICES,
	*LISTS,
	*[token(bytes([0x5e, 0x10 + i]), f'Y{chr(0x2080 + (i + 1) % 10)}') for i in range(10)],
	*[token(bytes([0x5e, 0x20 + i]), f'{x}{chr(0x2080 + n)}ₜ') for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))],
	*[token(bytes([0x5e, 0x40 + i]), f'r{chr(0x2081 + i)}') for i in range(6)],
	token(b'\x5e\x80', '𝑢'),
	token(b'\x5e\x81', '𝑣'),
	token(b'\x5e\x82', '𝑤'),
	*PICTURES,
	*GDBS,
	*STRINGS,
	token(b'\x62\x01', 'RegEq'),
	token(b'\x62\x02', 'n'),
	token(b'\x62\x03', 'ẍ'),
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
	token(b'\x62\x21', '𝑛'),
	token(b'\x62\x22', 'p'),
	token(b'\x62\x23', 'z'),
	token(b'\x62\x24', 't'),
	token(b'\x62\x25', 'χ²'),
	token(b'\x62\x26', '𝐅'),
	token(b'\x62\x27', 'df'),
	token(b'\x62\x28', 'ṕ'),
	token(b'\x62\x29', 'ṕ₁'),
	token(b'\x62\x2a', 'ṕ₂'),
	token(b'\x62\x2b', 'ẍ₁'),
	token(b'\x62\x2c', 'Sx₁'),
	token(b'\x62\x2d', 'n₁'),
	token(b'\x62\x2e', 'ẍ₂'),
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
	token(b'\x63\x2b', '𝐍'),
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
	token(b'\xbb\x07', 'dbd(',            flags=_F),
	token(b'\xbb\x08', 'lcm(',            flags=_F),
	token(b'\xbb\x09', 'gcd(',            flags=_F),
	token(b'\xbb\x0a', 'randInt(',        flags=_F),
	token(b'\xbb\x0b', 'randBin(',        flags=_F),
	token(b'\xbb\x0c', 'sub(',            flags=_F),
	token(b'\xbb\x0d', 'stdDev(',         flags=_F),
	token(b'\xbb\x0e', 'variance(',       flags=_F),
	token(b'\xbb\x0f', 'inString(',       flags=_F),
	token(b'\xbb\x10', 'normalcdf(',      flags=_F),
	token(b'\xbb\x11', 'invNorm(',        flags=_F),
	token(b'\xbb\x12', 'tcdf(',           flags=_F),
	token(b'\xbb\x13', 'χ²cdf(',          flags=_F),
	token(b'\xbb\x14', 'Fcdf(',           flags=_F),
	token(b'\xbb\x15', 'binompdf(',       flags=_F),
	token(b'\xbb\x16', 'binomcdf(',       flags=_F),
	token(b'\xbb\x17', 'poissonpdf(',     flags=_F),
	token(b'\xbb\x18', 'poissoncdf(',     flags=_F),
	token(b'\xbb\x19', 'geometpdf(',      flags=_F),
	token(b'\xbb\x1a', 'geometcdf(',      flags=_F),
	token(b'\xbb\x1b', 'normalpdf(',      flags=_F),
	token(b'\xbb\x1c', 'tpdf(',           flags=_F),
	token(b'\xbb\x1d', 'χ²pdf(',          flags=_F),
	token(b'\xbb\x1e', 'Fpdf(',           flags=_F),
	token(b'\xbb\x1f', 'randNorm(',       flags=_F),
	token(b'\xbb\x20', 'tvm_Pmt'),
	token(b'\xbb\x21', 'tvm_I%'),
	token(b'\xbb\x22', 'tvm_PV'),
	token(b'\xbb\x23', 'tvm_N'),
	token(b'\xbb\x24', 'tvm_FV'),
	token(b'\xbb\x25', 'conj(',           flags=_F),
	token(b'\xbb\x26', 'real(',           flags=_F),
	token(b'\xbb\x27', 'imag(',           flags=_F),
	token(b'\xbb\x28', 'angle(',          flags=_F),
	token(b'\xbb\x29', 'cumSum(',         flags=_F),
	token(b'\xbb\x2a', 'expr(',           flags=_F),
	token(b'\xbb\x2b', 'length(',         flags=_F),
	token(b'\xbb\x2c', 'ΔList(',          flags=_F),
	token(b'\xbb\x2d', 'ref(',            flags=_F),
	token(b'\xbb\x2e', 'rref(',           flags=_F),
	token(b'\xbb\x2f', '►Rect'),
	token(b'\xbb\x30', '►Polar'),
	token(b'\xbb\x31', '𝑒',              flags=_N),
	token(b'\xbb\x32', 'SinReg '),
	token(b'\xbb\x33', 'Logistic '),
	token(b'\xbb\x34', 'LinRegTTest '),
	token(b'\xbb\x35', 'ShadeNorm('),
	token(b'\xbb\x36', 'Shade_t('),
	token(b'\xbb\x37', 'Shadeχ²('),
	token(b'\xbb\x38', 'ShadeF('),
	token(b'\xbb\x39', 'Matr►list(',     flags=_CMD),
	token(b'\xbb\x3a', 'List►matr(',     flags=_CMD),
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
	# token(b'\xbb\x6d', 'compiled asm'),
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
	token(b'\xbb\x9b', 'ˋ'),
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
	token(b'\xbb\xa7', '𝛑'),
	token(b'\xbb\xa8', 'ρ'),
	token(b'\xbb\xa9', 'Σ'),
	token(b'\xbb\xab', 'φ'),
	token(b'\xbb\xac', 'Ω'),
	token(b'\xbb\xad', 'ψ'),
	token(b'\xbb\xae', 'χ'),
	token(b'\xbb\xaf', '𝟊'),
	token(b'\xbb\xb0', ascii='a'),
	token(b'\xbb\xb1', ascii='b'),
	token(b'\xbb\xb2', ascii='c'),
	token(b'\xbb\xb3', ascii='d'),
	token(b'\xbb\xb4', ascii='e'),
	token(b'\xbb\xb5', ascii='f'),
	token(b'\xbb\xb6', ascii='g'),
	token(b'\xbb\xb7', ascii='h'),
	token(b'\xbb\xb8', ascii='i'),
	token(b'\xbb\xb9', ascii='j'),
	token(b'\xbb\xba', ascii='k'),
	token(b'\xbb\xbc', ascii='l'),
	token(b'\xbb\xbd', ascii='m'),
	token(b'\xbb\xbe', ascii='n'),
	token(b'\xbb\xbf', ascii='o'),
	token(b'\xbb\xc0', ascii='p'),
	token(b'\xbb\xc1', ascii='q'),
	token(b'\xbb\xc2', ascii='r'),
	token(b'\xbb\xc3', ascii='s'),
	token(b'\xbb\xc4', ascii='t'),
	token(b'\xbb\xc5', ascii='u'),
	token(b'\xbb\xc6', ascii='v'),
	token(b'\xbb\xc7', ascii='w'),
	token(b'\xbb\xc8', ascii='x'),
	token(b'\xbb\xc9', ascii='y'),
	token(b'\xbb\xca', ascii='z'),
	token(b'\xbb\xcb', 'σ'),
	token(b'\xbb\xcc', 'τ'),
	token(b'\xbb\xcd', 'Í'),
	token(b'\xbb\xce', 'GarbageCollect'),
	token(b'\xbb\xcf', ascii='~'),
	token(b'\xbb\xd1', ascii='@'),
	token(b'\xbb\xd2', ascii='#'),
	token(b'\xbb\xd3', ascii='$'),
	token(b'\xbb\xd4', ascii='&'),
	token(b'\xbb\xd5', ascii='`'),
	token(b'\xbb\xd6', ascii=';'),
	token(b'\xbb\xd7', ascii='\\'),
	token(b'\xbb\xd8', ascii='|'),
	token(b'\xbb\xd9', ascii='_'),
	token(b'\xbb\xda', ascii='%',  flags=_PFX),
	token(b'\xbb\xdb', '…'),
	token(b'\xbb\xdc', '∠'),
	token(b'\xbb\xdd', 'ß'),
	token(b'\xbb\xde', 'ˣ'),
	token(b'\xbb\xdf', 'ₜ'),
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
	token(b'\xbb\xea', '⑽'),
	token(b'\xbb\xeb', '◄'),
	token(b'\xbb\xec', '🡆'),
	token(b'\xbb\xed', '↑'),
	token(b'\xbb\xee', '↓'),
	token(b'\xbb\xf0', '𝑥'),
	token(b'\xbb\xf1', '∫'),
	token(b'\xbb\xf2', '🡅'),
	token(b'\xbb\xf3', '🡇'),
	token(b'\xbb\xf4', '√'),
	token(b'\xbb\xf5', '≛'),

	# Two-byte: TI-84+ extended tokens 0xEF xx
	token(b'\xef\x00', 'setDate(',    flags=_CMD),
	token(b'\xef\x01', 'setTime(',    flags=_CMD),
	token(b'\xef\x02', 'checkTmr(',   flags=_F),
	token(b'\xef\x03', 'setDtFmt(',   flags=_CMD),
	token(b'\xef\x04', 'setTmFmt(',   flags=_CMD),
	token(b'\xef\x05', 'timeCnv(',    flags=_F),
	token(b'\xef\x06', 'dayOfWk(',    flags=_F),
	token(b'\xef\x07', 'getDtStr(',   flags=_F),
	token(b'\xef\x08', 'getTmStr(',   flags=_F),
	token(b'\xef\x09', 'getDate',     flags=_N),
	token(b'\xef\x0a', 'getTime',     flags=_N),
	token(b'\xef\x0b', 'startTmr',    flags=_N),
	token(b'\xef\x0c', 'getDtFmt',    flags=_N),
	token(b'\xef\x0d', 'getTmFmt',    flags=_N),
	token(b'\xef\x0e', 'isClockOn',   flags=_N),
	token(b'\xef\x0f', 'ClockOff',    flags=_CMD),
	token(b'\xef\x10', 'ClockOn',     flags=_CMD),
	token(b'\xef\x11', 'OpenLib('),
	token(b'\xef\x12', 'ExecLib'),
	token(b'\xef\x13', 'invT(',       flags=_F),
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
	token(b'\xef\x1e', 'mathprintbox'),
	token(b'\xef\x30', '►n/d◄►Un/d'),
	token(b'\xef\x31', '►F◄►D'),
	token(b'\xef\x32', 'remainder(',  flags=_F),
	token(b'\xef\x33', 'Σ(',          flags=_F),
	token(b'\xef\x34', 'logBASE(',    flags=_F),
	token(b'\xef\x35', 'randIntNoRep(', flags=_F),
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

ASCII = {t.ascii: t for t in TOKENS if t.ascii}

TOKENS_BY_TEXT = {t.text: t for t in TOKENS} | ASCII


if __name__ == '__main__':
	for i in range(32, 127):
		print(i, TOKENS_BY_TEXT[chr(i)])
