from collections.abc import Callable, Sequence
from io import BytesIO
from typing import Any
import math, itertools
from titoken import Token
from environment import Environment
import purefunctions as pf
import operators as ops
import envfunctions as ef
import forms
import modes
import draw



_TABLE: list[Token | list[Token | None] | None] = [None] * 256

ALL_TOKENS: list[Token] = []
CHARS = {}

def get_token(code: int | Sequence[int]) -> Token:
	if isinstance(code, int):
		code = code.to_bytes(1 + (code > 0xFF))
	item = _TABLE[code[0]]
	if len(code) > 1:
		item = item[code[1]] if isinstance(item, list) and code[1] < len(item) else None
	if not isinstance(item, Token):
		raise KeyError(code)
	return item


def _set_token(token: Token):
	b0 = token.code[0]
	if len(token.code) == 1:
		tbl = _TABLE
		idx = b0
	else:
		tbl = _TABLE[b0]
		idx = token.code[1]
		if tbl is None:
			_TABLE[b0] = tbl = []
		if idx >= len(tbl):
			tbl.extend([None] * (idx + 1 - len(tbl)))

	if (dup := tbl[idx]) is not None:
		raise ValueError(f"Duplicate token: {token} vs. {dup}")
	tbl[idx] = token


def read_token(f: BytesIO) -> Token:
	first = f.read(1)
	code = first + f.read(1) if isinstance(_TABLE[first[0]], list) else first
	try:
		return get_token(code)
	except KeyError:
		raise ValueError(f"Invalid token code: 0x{int.from_bytes(code):0{2 * len(code)}X}")


def _make_accessor(table: str, index: int):
	return lambda env: getattr(env, table)[index]


def token(
	code: int,
	text: str = None,
	char: str = None,
	*,
	bp:   tuple[int, int] | None = None,
	op:   Callable | None = None,
	post: Callable | None = None,
	func: Callable | None = None,
	cmd:  Callable | None = None,
	res:  Callable | None = None,
	cnv:  Callable | None = None,
	var:  Callable | None = None,
) -> Token:
	text = text or char
	t = Token(code.to_bytes(1 + (code > 0xFF)), char, text, bp, op, post, func, cmd, res, cnv, var)
	_set_token(t)
	ALL_TOKENS.append(t)
	if char:
		CHARS[char] = t
	return t


token(0x01, '►DMS')
token(0x02, '►Dec')
token(0x03, '►Frac')

STORE = token(0x04, '→')

token(0x05, 'Boxplot')

L_BRACKET = token(0x06, char='[')
R_BRACKET = token(0x07, char=']')
L_BRACE   = token(0x08, char='{')
R_BRACE   = token(0x09, char='}')
RAD       = token(0x0A, 'ʳ')       # post needs env, handled specially
DEG       = token(0x0B, char='°')  # ditto
INV       = token(0x0C, '¹',       post=ops.inv)
SQ        = token(0x0D, char='²',  post=lambda x: x**2)
TRANSPOSE = token(0x0E, 'ᵀ',       post=ops.transpose)
CUBE      = token(0x0F, char='³',  post=lambda x: x**3)
L_PAREN   = token(0x10, char='(')
R_PAREN   = token(0x11, char=')')

token(0x12, 'round(',       func=pf.round)
token(0x13, 'pxl-Test(', func=draw.pxl_test)
token(0x14, 'augment(',     func=pf.augment)
token(0x15, 'rowSwap(',     func=pf.row_swap)
token(0x16, 'row+(',        func=pf.row_plus)
token(0x17, '*row(',        func=pf.times_row)
token(0x18, '*row+(',       func=pf.times_row_plus)
token(0x19, 'max(',         func=pf.max)
token(0x1A, 'min(',         func=pf.min)
token(0x1B, 'R►Pr(',        func=ef.rect_to_polar_radius)
token(0x1C, 'R►Pθ(',        func=ef.rect_to_polar_angle)
token(0x1D, 'P►Rx(',        func=ef.polar_to_rect_x)
token(0x1E, 'P►Ry(',        func=ef.polar_to_rect_y)
token(0x1F, 'median(',      func=pf.median)
token(0x20, 'randM(',       func=pf.rand_m)
token(0x21, 'mean(',        func=pf.mean)
token(0x22, 'solve(')
token(0x23, 'seq(',         func=forms.seq)
token(0x24, 'fnInt(',       func=forms.fn_int)
token(0x25, 'nDeriv(',      func=forms.n_deriv)
token(0x27, 'fMin(')
token(0x28, 'fMax(')
token(0x29, char=' ')

QUOTE  = token(0x2A, char='"')
COMMA  = token(0x2B, char=',')
IMAG_I = token(0x2C, char='𝑖', res=lambda env: 1j)
FACT   = token(0x2D, char='!',      post=ops.factorial)

token(0x2E, 'CubicReg ')
token(0x2F, 'QuartReg ')

DIGITS = tuple(token(0x30 + i, char=chr(0x30 + i)) for i in range(10))

DOT       = token(0x3A, char='.')
SCI_E     = token(0x3B, 'ᴇ')

token(0x3C, ' or ',    bp=(20, 21), op=ops.or_)
token(0x3D, ' xor ',   bp=(20, 21), op=ops.xor)

COLON     = token(0x3E, char=':')
NEWLINE   = token(0x3F, char='\n')

token(0x40, ' and ',   bp=(30, 31), op=ops.and_)

LETTERS = tuple([
	*(token(0x41 + i, char=chr(0x41 + i), var=_make_accessor('numerics', i)) for i in range(26)),
	token(0x5B, char='θ', var=_make_accessor('numerics', 26)),
])

# ── 0x5C xx: matrix variables ([A]–[J]) ──────────────────────────────────────

MATRICES = tuple(token(0x5C00 | i, f'[{chr(0x41 + i)}]', var=_make_accessor('matrices', i)) for i in range(10))

# ── 0x5D xx: list variables (L1–L6) ──────────────────────────────────────────

LISTS = tuple(token(0x5D00 | i, f'L{chr(0x2081 + i)}', var=_make_accessor('lists', i)) for i in range(6))

# ── 0x5E xx: equation and sequence variables ──────────────────────────────────

FUNCTION   = tuple(token(
	0x5E10 + i,
	f'Y{chr(0x2080 + (i + 1) % 10)}', 
	var=_make_accessor('function', i)
) for i in range(10))

PARAMETRIC = tuple(token(
	0x5E20 + i, 
	f'{'XY'[i % 2]}{chr(0x2081 + i // 2)}ₜ', 
	var=_make_accessor('parametric', i)
) for i in range(12))

POLAR      = tuple(token(
	0x5E40 + i, 
	f'r{chr(0x2081 + i)}', 
	var=_make_accessor('polar', i)
) for i in range(6))

SEQUENCE   = tuple(token(
	0x5E80 + i, 
	chr(0x1D462 + i), 
	var=_make_accessor('sequence', i)
) for i in range(3))

PRGM = token(0x5F, 'prgm', cmd=forms.prgm)

# ── 0x60 xx: picture variables (Pic1–Pic0) ───────────────────────────────────

PICTURES = tuple(token(0x6000 + i, f'Pic{(i + 1) % 10}') for i in range(10))

# ── 0x61 xx: graph database variables (GDB1–GDB0) ────────────────────────────

GDBS = tuple(token(0x6100 | i, f'GDB{(i + 1) % 10}') for i in range(10))

# ── 0x62 xx: statistical result variables ─────────────────────────────────────

token(0x6201, 'RegEq')
token(0x6202, 'n')
token(0x6203, 'ẍ')
token(0x6204, 'Σx')
token(0x6205, 'Σx²')
token(0x6206, 'Sx')
token(0x6207, 'σx')
token(0x6208, 'minX')
token(0x6209, 'maxX')
token(0x620A, 'minY')
token(0x620B, 'maxY')
token(0x620C, 'ȳ')
token(0x620D, 'Σy')
token(0x620E, 'Σy²')
token(0x620F, 'Sy')
token(0x6210, 'σy')
token(0x6211, 'Σxy')
token(0x6212, 'r')
token(0x6213, 'Med')
token(0x6214, 'Q1')
token(0x6215, 'Q3')
token(0x6216, 'a')
token(0x6217, 'b')
token(0x6218, 'c')
token(0x6219, 'd')
token(0x621A, 'e')
token(0x621B, 'x₁')
token(0x621C, 'x₂')
token(0x621D, 'x₃')
token(0x621E, 'y₁')
token(0x621F, 'y₂')
token(0x6220, 'y₃')
REC_N = token(0x6221, '𝑛', var=lambda env: env.n)
token(0x6222, 'p')
token(0x6223, 'z')
token(0x6224, 't')
token(0x6225, 'χ²')
token(0x6226, '𝐅')
token(0x6227, 'df')
token(0x6228, 'ṕ')
token(0x6229, 'ṕ₁')
token(0x622A, 'ṕ₂')
token(0x622B, 'ẍ₁')
token(0x622C, 'Sx₁')
token(0x622D, 'n₁')
token(0x622E, 'ẍ₂')
token(0x622F, 'Sx₂')
token(0x6230, 'n₂')
token(0x6231, 'Sxp')
token(0x6232, 'lower')
token(0x6233, 'upper')
token(0x6234, 's')
token(0x6235, 'r²')
token(0x6236, 'R²')
token(0x6237, 'Factor df')
token(0x6238, 'Factor SS')
token(0x6239, 'Factor MS')
token(0x623A, 'Error df')
token(0x623B, 'Error SS')
token(0x623C, 'Error MS')

# ── 0x63 xx: window and finance variables ─────────────────────────────────────

def _window_accessor(attr: str):
	return lambda env: getattr(env.window, attr)

token(0x6302, 'Xscl',      var=_window_accessor('xscl'))
token(0x6303, 'Yscl',      var=_window_accessor('yscl'))
token(0x630A, 'Xmin',      var=_window_accessor('xmin'))
token(0x630B, 'Xmax',      var=_window_accessor('xmax'))
token(0x630C, 'Ymin',      var=_window_accessor('ymin'))
token(0x630D, 'Ymax',      var=_window_accessor('ymax'))
token(0x630E, 'Tmin',      var=_window_accessor('tmin'))
token(0x630F, 'Tmax',      var=_window_accessor('tmax'))
token(0x6310, 'θmin',      var=_window_accessor('theta_min'))
token(0x6311, 'θmax',      var=_window_accessor('theta_max'))
token(0x631A, 'TblStart',  var=_window_accessor('tbl_start'))
token(0x631B, 'PlotStart', var=_window_accessor('plot_start'))
token(0x631D, 'nMax',      var=_window_accessor('n_max'))
token(0x631F, 'nMin',      var=_window_accessor('n_min'))
token(0x6321, 'ΔTbl',      var=_window_accessor('delta_tbl'))
token(0x6322, 'Tstep',     var=_window_accessor('tstep'))
token(0x6323, 'θstep',     var=_window_accessor('theta_step'))
token(0x6326, 'ΔX',        var=_window_accessor('delta_x'))
token(0x6327, 'ΔY',        var=_window_accessor('delta_y'))
token(0x6328, 'XFact',     var=_window_accessor('x_fact'))
token(0x6329, 'YFact',     var=_window_accessor('y_fact'))
token(0x6334, 'PlotStep',  var=_window_accessor('plot_step'))
token(0x6336, 'Xres',      var=_window_accessor('xres'))
token(0x632B, '𝐍',   var=lambda env: env.n_tvm)
token(0x632C, 'I%',  var=lambda env: env.i_pct)
token(0x632D, 'PV',  var=lambda env: env.pv)
token(0x632E, 'PMT', var=lambda env: env.pmt)
token(0x632F, 'FV',  var=lambda env: env.fv)
token(0x6330, 'P/Y', var=lambda env: env.py)
token(0x6331, 'C/Y', var=lambda env: env.cy)


token(0x64, 'Radian', cmd=modes.radian)
token(0x65, 'Degree', cmd=modes.degree)
token(0x66, 'Normal', cmd=modes.normal)
token(0x67, 'Sci',    cmd=modes.sci)
token(0x68, 'Eng',    cmd=modes.eng)
token(0x69, 'Float',  cmd=modes.float_)

EQ  = token(0x6A, char='=', bp=(40, 41), op=ops.eq)
LT  = token(0x6B, char='<', bp=(40, 41), op=ops.lt)
GT  = token(0x6C, char='>', bp=(40, 41), op=ops.gt)
LE  = token(0x6D, char='≤', bp=(40, 41), op=ops.le)
GE  = token(0x6E, char='≥', bp=(40, 41), op=ops.ge)
NE  = token(0x6F, char='≠', bp=(40, 41), op=ops.ne)
ADD = token(0x70, char='+', bp=(50, 51), op=ops.add)
SUB = token(0x71, char='-', bp=(50, 51), op=ops.sub)
ANS = token(0x72, 'Ans',  res=Environment.get_ans, func=forms.ans_index_or_mul)

token(0x73, 'Fix',   cmd=modes.fix)
token(0x74, 'Horiz')
token(0x75, 'Full')
token(0x76, 'Func',  cmd=modes.func)
token(0x77, 'Param', cmd=modes.param)
token(0x78, 'Polar', cmd=modes.polar)
token(0x79, 'Seq',   cmd=modes.seq)
token(0x7A, 'IndpntAuto')
token(0x7B, 'IndpntAsk')
token(0x7C, 'DependAuto')
token(0x7D, 'DependAsk')

# ── 0x7E xx: graph format settings ───────────────────────────────────────────

token(0x7E00, 'Sequential', cmd=modes.sequential)
token(0x7E01, 'Simul',      cmd=modes.simul)
token(0x7E02, 'PolarGC',    cmd=modes.polar_gc)
token(0x7E03, 'RectGC',     cmd=modes.rect_gc)
token(0x7E04, 'CoordOn',    cmd=modes.coord_on)
token(0x7E05, 'CoordOff',   cmd=modes.coord_off)
token(0x7E06, 'Connected',  cmd=modes.connected)
token(0x7E07, 'Dot',        cmd=modes.dot)
token(0x7E08, 'AxesOn',     cmd=modes.axes_on)
token(0x7E09, 'AxesOff',    cmd=modes.axes_off)
token(0x7E0A, 'GridOn',     cmd=modes.grid_on)
token(0x7E0B, 'GridOff',    cmd=modes.grid_off)
token(0x7E0C, 'LabelOn',    cmd=modes.label_on)
token(0x7E0D, 'LabelOff',   cmd=modes.label_off)
token(0x7E0E, 'Web')
token(0x7E0F, 'Time')
token(0x7E10, 'uvAxes')
token(0x7E11, 'vwAxes')
token(0x7E12, 'uwAxes')

token(0x7F, '▫')
token(0x80, '﹢')
token(0x81, '·')

MUL = token(0x82, char='*', bp=(60, 61), op=ops.mul)
DIV = token(0x83, char='/', bp=(60, 61), op=ops.div)

token(0x84, 'Trace')
token(0x85, 'ClrDraw', cmd=draw.clr_draw)
token(0x86, 'ZStandard')
token(0x87, 'ZTrig')
token(0x88, 'ZBox')
token(0x89, 'Zoom In')
token(0x8A, 'Zoom Out')
token(0x8B, 'ZSquare')
token(0x8C, 'ZInteger')
token(0x8D, 'ZPrevious')
token(0x8E, 'ZDecimal')
token(0x8F, 'ZoomStat')
token(0x90, 'ZoomRcl')
token(0x91, 'PrintScreen')
token(0x92, 'ZoomSto')
token(0x93, 'Text(')

NPR = token(0x94, 'nPr',     bp=(60, 61), op=ops.npr)
NCR = token(0x95, 'nCr',     bp=(60, 61), op=ops.ncr)

token(0x96, 'FnOn ')
token(0x97, 'FnOff ')
token(0x98, 'StorePic ')
token(0x99, 'RecallPic ')
token(0x9A, 'StoreGDB ')
token(0x9B, 'RecallGDB ')
token(0x9C, 'Line(')
token(0x9D, 'Vertical ')
token(0x9E, 'Pt-On(', cmd=draw.pt_on)
token(0x9F, 'Pt-Off(',    cmd=draw.pt_off)
token(0xA0, 'Pt-Change(', cmd=draw.pt_change)
token(0xA1, 'Pxl-On(',     cmd=draw.pxl_on)
token(0xA2, 'Pxl-Off(',    cmd=draw.pxl_off)
token(0xA3, 'Pxl-Change(', cmd=draw.pxl_change)
token(0xA4, 'Shade(')
token(0xA5, 'Circle(')
token(0xA6, 'Horizontal ')
token(0xA7, 'Tangent(')
token(0xA8, 'DrawInv ')
token(0xA9, 'DrawF ')

# ── 0xAA xx: string variables (Str1–Str0) ────────────────────────────────────

STRINGS = tuple(token(0xAA00 | i, f'Str{(i + 1) % 10}', var=_make_accessor('strings', i)) for i in range(10))

RAND = token(0xAB, 'rand', res=Environment.rand, func=pf.rand_list)
token(0xAC, char='π', res=lambda env: math.pi)
token(0xAD, 'getKey', res=Environment.get_key)
APOS = token(0xAE, char="'")
token(0xAF, char='?')
NEG  = token(0xB0, '⁻')

token(0xB1, 'int(', func=pf.int_)
token(0xB2, 'abs(', func=pf.abs)
token(0xB3, 'det(', func=pf.det)
token(0xB4, 'identity(', func=pf.identity)

DIM = token(0xB5, 'dim(', func=forms.dim)

token(0xB6, 'sum(', func=pf.sum)
token(0xB7, 'prod(', func=pf.prod)
token(0xB8, 'not(', func=pf.not_)
token(0xB9, 'iPart(', func=pf.i_part)
token(0xBA, 'fPart(', func=pf.f_part)

# ── 0xBB xx: extended tokens ──────────────────────────────────────────────────

token(0xBB00, 'npv(',  			func=pf.npv)
token(0xBB01, 'irr(',  			func=pf.irr)
token(0xBB02, 'bal(',  			func=ef.bal)
token(0xBB03, 'Σprn(', 			func=ef.sigma_prn)
token(0xBB04, 'ΣInt(', 			func=ef.sigma_int)
token(0xBB05, '►Nom(', 			func=pf.nom)
token(0xBB06, '►Eff(', 			func=pf.eff)
token(0xBB07, 'dbd(',           func=pf.dbd)
token(0xBB08, 'lcm(',           func=pf.lcm)
token(0xBB09, 'gcd(',           func=pf.gcd)
token(0xBB0A, 'randInt(',       func=pf.rand_int)
token(0xBB0B, 'randBin(',       func=pf.rand_bin)
token(0xBB0C, 'sub(',           func=pf.sub)
token(0xBB0D, 'stdDev(',        func=pf.stddev)
token(0xBB0E, 'variance(',      func=pf.variance)
token(0xBB0F, 'inString(',      func=pf.in_string)
token(0xBB10, 'normalcdf(',     func=pf.normalcdf)
token(0xBB11, 'invNorm(',       func=pf.inv_norm)
token(0xBB12, 'tcdf(',          func=pf.tcdf)
token(0xBB13, 'χ²cdf(',         func=pf.chi_sq_cdf)
token(0xBB14, 'Fcdf(',          func=pf.fcdf)
token(0xBB15, 'binompdf(',      func=pf.binompdf)
token(0xBB16, 'binomcdf(',      func=pf.binomcdf)
token(0xBB17, 'poissonpdf(',    func=pf.poissonpdf)
token(0xBB18, 'poissoncdf(',    func=pf.poissoncdf)
token(0xBB19, 'geometpdf(',     func=pf.geometpdf)
token(0xBB1A, 'geometcdf(',     func=pf.geometcdf)
token(0xBB1B, 'normalpdf(',     func=pf.normalpdf)
token(0xBB1C, 'tpdf(',          func=pf.tpdf)
token(0xBB1D, 'χ²pdf(',         func=pf.chi_sq_pdf)
token(0xBB1E, 'Fpdf(',          func=pf.f_pdf)
token(0xBB1F, 'randNorm(',      func=pf.rand_norm)
token(0xBB20, 'tvm_Pmt')
token(0xBB21, 'tvm_I%')
token(0xBB22, 'tvm_PV')
token(0xBB23, 'tvm_N')
token(0xBB24, 'tvm_FV')
token(0xBB25, 'conj(',          func=pf.conj)
token(0xBB26, 'real(',          func=pf.real_)
token(0xBB27, 'imag(',          func=pf.imag)
token(0xBB28, 'angle(',         func=pf.angle)
token(0xBB29, 'cumSum(',        func=pf.cum_sum)
token(0xBB2A, 'expr(',          func=ef.expr)
token(0xBB2B, 'length(',        func=pf.length)
token(0xBB2C, 'ΔList(',         func=pf.delta_list)
token(0xBB2D, 'ref(',           func=pf.ref)
token(0xBB2E, 'rref(',          func=pf.rref)
token(0xBB2F, '►Rect')
token(0xBB30, '►Polar')
token(0xBB31, char='𝑒', res=lambda env: math.e)
token(0xBB32, 'SinReg ')
token(0xBB33, 'Logistic ')
token(0xBB34, 'LinRegTTest ')
token(0xBB35, 'ShadeNorm(')
token(0xBB36, 'Shade_t(')
token(0xBB37, 'Shadeχ²(')
token(0xBB38, 'ShadeF(')
token(0xBB39, 'Matr►list(', cmd=forms.matr_to_list)
token(0xBB3A, 'List►matr(', cmd=forms.list_to_matr)
token(0xBB3B, 'Z-Test(')
token(0xBB3C, 'T-Test')
token(0xBB3D, '2-SampZTest(')
token(0xBB3E, '1-PropZTest(')
token(0xBB3F, '2-PropZTest(')
token(0xBB40, 'χ²-Test(')
token(0xBB41, 'ZInterval ')
token(0xBB42, '2-SampZInt(')
token(0xBB43, '1-PropZInt(')
token(0xBB44, '2-PropZInt(')
token(0xBB45, 'GraphStyle(')
token(0xBB46, '2-SampTTest ')
token(0xBB47, '2-SampFTest ')
token(0xBB48, 'TInterval ')
token(0xBB49, '2-SampTInt ')
token(0xBB4A, 'SetUpEditor ', cmd=forms.set_up_editor)
token(0xBB4B, 'Pmt_End')
token(0xBB4C, 'Pmt_Bgn')
token(0xBB4D, 'Real',   cmd=modes.real)
token(0xBB4E, 're^θi',  cmd=modes.re_theta_i)
token(0xBB4F, 'a+bi',   cmd=modes.a_plus_bi)
token(0xBB50, 'ExprOn',  cmd=modes.expr_on)
token(0xBB51, 'ExprOff', cmd=modes.expr_off)
token(0xBB52, 'ClrAllLists', cmd=forms.clr_all_lists)
token(0xBB53, 'GetCalc(')
token(0xBB54, 'DelVar ', cmd=forms.del_var)
token(0xBB55, 'Equ►String(', cmd=forms.equ_to_string)
token(0xBB56, 'String►Equ(', cmd=forms.string_to_equ)
token(0xBB57, 'Clear Entries')
token(0xBB58, 'Select(')
token(0xBB59, 'ANOVA(')
token(0xBB5A, 'ModBoxplot')
token(0xBB5B, 'NormProbPlot')
token(0xBB64, 'G-T')
token(0xBB65, 'ZoomFit')
token(0xBB66, 'DiagnosticOn',  cmd=modes.diagnostic_on)
token(0xBB67, 'DiagnosticOff', cmd=modes.diagnostic_off)
token(0xBB68, 'Archive ')
token(0xBB69, 'UnArchive ')
token(0xBB6A, 'Asm(')
token(0xBB6B, 'AsmComp(')
token(0xBB6C, 'AsmPrgm')
token(0xBB6D, 'compiled asm')
token(0xBB6E, char='Á')
token(0xBB6F, char='À')
token(0xBB70, char='Â')
token(0xBB71, char='Ä')
token(0xBB72, char='á')
token(0xBB73, char='à')
token(0xBB74, char='â')
token(0xBB75, char='ä')
token(0xBB76, char='É')
token(0xBB77, char='È')
token(0xBB78, char='Ê')
token(0xBB79, char='Ë')
token(0xBB7A, char='é')
token(0xBB7B, char='è')
token(0xBB7C, char='ê')
token(0xBB7D, char='ë')
token(0xBB7F, char='Ì')
token(0xBB80, char='Î')
token(0xBB81, char='Ï')
token(0xBB82, char='í')
token(0xBB83, char='ì')
token(0xBB84, char='î')
token(0xBB85, char='ï')
token(0xBB86, char='Ó')
token(0xBB87, char='Ò')
token(0xBB88, char='Ô')
token(0xBB89, char='Ö')
token(0xBB8A, char='ó')
token(0xBB8B, char='ò')
token(0xBB8C, char='ô')
token(0xBB8D, char='ö')
token(0xBB8E, char='Ú')
token(0xBB8F, char='Ù')
token(0xBB90, char='Û')
token(0xBB91, char='Ü')
token(0xBB92, char='ú')
token(0xBB93, char='ù')
token(0xBB94, char='û')
token(0xBB95, char='ü')
token(0xBB96, char='Ç')
token(0xBB97, char='ç')
token(0xBB98, char='Ñ')
token(0xBB99, char='ñ')
token(0xBB9A, '´')
token(0xBB9B, 'ˋ')
token(0xBB9C, '¨')
token(0xBB9D, char='¿')
token(0xBB9E, char='¡')
token(0xBB9F, char='α')
token(0xBBA0, char='β')
token(0xBBA1, char='γ')
token(0xBBA2, char='Δ')
token(0xBBA3, char='δ')
token(0xBBA4, char='ε')
token(0xBBA5, char='λ')
token(0xBBA6, char='μ')
token(0xBBA7, '𝛑')  # alternate pi
token(0xBBA8, char='ρ')
token(0xBBA9, char='Σ')
token(0xBBAB, char='φ')
token(0xBBAC, char='Ω')
token(0xBBAD, char='ψ')
token(0xBBAE, char='χ')
token(0xBBAF, '𝟊')
token(0xBBB0, char='a')
token(0xBBB1, char='b')
token(0xBBB2, char='c')
token(0xBBB3, char='d')
token(0xBBB4, char='e')
token(0xBBB5, char='f')
token(0xBBB6, char='g')
token(0xBBB7, char='h')
token(0xBBB8, char='i')
token(0xBBB9, char='j')
token(0xBBBA, char='k')
token(0xBBBC, char='l')
token(0xBBBD, char='m')
token(0xBBBE, char='n')
token(0xBBBF, char='o')
token(0xBBC0, char='p')
token(0xBBC1, char='q')
token(0xBBC2, char='r')
token(0xBBC3, char='s')
token(0xBBC4, char='t')
token(0xBBC5, char='u')
token(0xBBC6, char='v')
token(0xBBC7, char='w')
token(0xBBC8, char='x')
token(0xBBC9, char='y')
token(0xBBCA, char='z')
token(0xBBCB, char='σ')
token(0xBBCC, char='τ')
token(0xBBCD, char='Í')
token(0xBBCE, 'GarbageCollect')
token(0xBBCF, char='~')
token(0xBBD1, char='@')
token(0xBBD2, char='#')
token(0xBBD3, char='$')
token(0xBBD4, char='&')
token(0xBBD5, char='`')
token(0xBBD6, char=';')
token(0xBBD7, char='\\')
token(0xBBD8, char='|')
token(0xBBD9, char='_')
token(0xBBDA, char='%', post=lambda x: x / 100)
token(0xBBDB, char='…')
token(0xBBDC, char='∠')
token(0xBBDD, char='ß')
token(0xBBDE, 'ˣ')
token(0xBBDF, 'ₜ')
token(0xBBE0, '₀')
token(0xBBE1, '₁')
token(0xBBE2, '₂')
token(0xBBE3, '₃')
token(0xBBE4, '₄')
token(0xBBE5, '₅')
token(0xBBE6, '₆')
token(0xBBE7, '₇')
token(0xBBE8, '₈')
token(0xBBE9, '₉')
token(0xBBEA, '⑽')
token(0xBBEB, '◄')
token(0xBBEC, '🡆')
token(0xBBED, '↑')
token(0xBBEE, '↓')
token(0xBBF0, '𝑥')
token(0xBBF1, char='∫')
token(0xBBF2, '🡅')
token(0xBBF3, '🡇')
token(0xBBF4, '√')
token(0xBBF5, '≛')
SQRT = token(0xBC, '√(', func=ef.sqrt)
CBRT = token(0xBD, '³√(', func=pf.cbrt)
token(0xBE, 'ln(',		func=ef.ln)
token(0xBF, '𝑒^(',		func=pf.exp)
token(0xC0, 'log(',		func=ef.log)
token(0xC1, '⑽^(',		func=pf.pow10)
token(0xC2, 'sin(',		func=ef.sin)
token(0xC3, 'sin¹(',	func=ef.asin)
token(0xC4, 'cos(',		func=ef.cos)
token(0xC5, 'cos¹(',	func=ef.acos)
token(0xC6, 'tan(',		func=ef.tan)
token(0xC7, 'tan¹(',	func=ef.atan)
token(0xC8, 'sinh(',	func=pf.sinh)
token(0xC9, 'sinh¹(',	func=pf.asinh)
token(0xCA, 'cosh(',	func=pf.cosh)
token(0xCB, 'cosh¹(',	func=ef.acosh)
token(0xCC, 'tanh(',	func=pf.tanh)
token(0xCD, 'tanh¹(',	func=ef.atanh)
IF     = token(0xCE, 'If ',    cmd=forms.if_cmd)
THEN   = token(0xCF, 'Then',   cmd=forms.then_cmd)
ELSE   = token(0xD0, 'Else',   cmd=forms.else_cmd)
WHILE  = token(0xD1, 'While ', cmd=forms.while_cmd)
REPEAT = token(0xD2, 'Repeat ', cmd=forms.repeat_cmd)
FOR    = token(0xD3, 'For(',   cmd=forms.for_cmd)
END    = token(0xD4, 'End',    cmd=forms.end_cmd)
token(0xD5, 'Return',          cmd=forms.return_cmd)
LBL    = token(0xD6, 'Lbl ',   cmd=forms.lbl_cmd)
token(0xD7, 'Goto ',           cmd=forms.goto_cmd)
token(0xD8, 'Pause ')
token(0xD9, 'Stop',            cmd=forms.stop_cmd)
token(0xDA, 'IS>(',            cmd=forms.is_gt_cmd)
token(0xDB, 'DS<(',            cmd=forms.ds_lt_cmd)
token(0xDC, 'Input ')
token(0xDD, 'Prompt ')
token(0xDE, 'Disp ', cmd=forms.disp)
token(0xDF, 'DispGraph')
token(0xE0, 'Output(')
token(0xE1, 'ClrHome')
token(0xE2, 'Fill(', cmd=forms.fill)
token(0xE3, 'SortA(', cmd=forms.sort_a)
token(0xE4, 'SortD(', cmd=forms.sort_d)
token(0xE5, 'DispTable')
token(0xE6, 'Menu(')
token(0xE7, 'Send(')
token(0xE8, 'Get(')
token(0xE9, 'PlotsOn')
token(0xEA, 'PlotsOff')
LIST_PREFIX = token(0xEB, 'ᴸ')
token(0xEC, 'Plot1(')
token(0xED, 'Plot2(')
token(0xEE, 'Plot3(')

# ── 0xEF xx: TI-84+ extended tokens ──────────────────────────────────────────

token(0xEF00, 'setDate(',    cmd=ef.set_date)
token(0xEF01, 'setTime(',    cmd=ef.set_time)
token(0xEF02, 'checkTmr(',   func=ef.check_tmr)
token(0xEF03, 'setDtFmt(',   cmd=ef.set_dt_fmt)
token(0xEF04, 'setTmFmt(',   cmd=ef.set_tm_fmt)
token(0xEF05, 'timeCnv(',    func=pf.time_cnv)
token(0xEF06, 'dayOfWk(',    func=pf.dayofwk)
token(0xEF07, 'getDtStr(',   func=ef.get_dt_str)
token(0xEF08, 'getTmStr(',   func=ef.get_tm_str)
token(0xEF09, 'getDate',     res=Environment.get_date)
token(0xEF0A, 'getTime',     res=Environment.get_time)
token(0xEF0B, 'startTmr',    res=Environment.start_tmr)
token(0xEF0C, 'getDtFmt',    res=Environment.get_dt_fmt)
token(0xEF0D, 'getTmFmt',    res=Environment.get_tm_fmt)
token(0xEF0E, 'isClockOn',   res=Environment.is_clock_on)
token(0xEF0F, 'ClockOff', cmd=modes.clock_off)
token(0xEF10, 'ClockOn',  cmd=modes.clock_on)
token(0xEF11, 'OpenLib(')
token(0xEF12, 'ExecLib')
token(0xEF13, 'invT(',       func=pf.inv_t)
token(0xEF14, 'χ²GOF-Test(')
token(0xEF15, 'LinRegTInt ')
token(0xEF16, 'Manual-Fit ')
token(0xEF17, 'ZQuadrant1')
token(0xEF18, 'ZFrac1/2')
token(0xEF19, 'ZFrac1/3')
token(0xEF1A, 'ZFrac1/4')
token(0xEF1B, 'ZFrac1/5')
token(0xEF1C, 'ZFrac1/8')
token(0xEF1D, 'ZFrac1/10')
token(0xEF1E, 'mathprintbox')
token(0xEF30, '►n/d◄►Un/d')
token(0xEF31, '►F◄►D')
token(0xEF32, 'remainder(', func=pf.remainder)
token(0xEF33, 'Σ(', func=forms.sigma)
token(0xEF34, 'logBASE(', func=ef.log_base)
token(0xEF35, 'randIntNoRep(', func=pf.rand_int_no_rep)
token(0xEF36, 'MATHPRINT')
token(0xEF37, 'CLASSIC')
token(0xEF38, 'n/d')
token(0xEF39, 'Un/d')
token(0xEF3A, 'AUTO')
token(0xEF3B, 'DEC')
token(0xEF3C, 'FRAC')
token(0xEF3D, 'FRAC-APPROX')

# ── 0xF0–0xFF: power operators and regression commands ───────────────────────

POW      = token(0xF0, char='^', bp=(70, 69), op=ops.power)
XTH_ROOT = token(0xF1, 'ˣ√',     bp=(60, 61), op=ops.xth_root)

token(0xF2, '1-Var Stats ')
token(0xF3, '2-Var Stats ')
token(0xF4, 'LinReg(a+bx) ')
token(0xF5, 'ExpReg ')
token(0xF6, 'LnReg ')
token(0xF7, 'PwrReg ')
token(0xF8, 'Med-Med ')
token(0xF9, 'QuadReg ')
token(0xFA, 'ClrList ', cmd=forms.clr_list)
token(0xFB, 'ClrTable')
token(0xFC, 'Histogram')
token(0xFD, 'xyLine')
token(0xFE, 'Scatter')
token(0xFF, 'LinReg(ax+b) ')


if __name__ == '__main__':
	print(len(ALL_TOKENS))
