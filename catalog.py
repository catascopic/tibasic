import math
from collections.abc import Callable
from io import BytesIO

from environment import Environment
from parser import ArgParser
from preparse import special_func
from titoken import Token
from accessors import (
	NumericVar, MatrixVar, ListVar, StringVar,
	ComputedAccessor, RandAccessor,
	WindowVar, XresVar, IntWindowVar, FactorWindowVar, DeltaWindowVar,
	EnvVar, TableVar, EquationVar, ParEquationVar, SequenceVar, SequenceInitialVar,
)
import commands as cmds
import distributions as dist
import draw
import finance
import matrix
import modecmds
import operators as ops
import tilist
import timath
import tistring
import titime
import titoken as tk
import zoom


from preparse import special_func

@special_func
def _PLACEHOLDER(args):
	while args.has_next:
		args.equation_var()
	args.end_cmd()


_TABLE: list[Token | list[Token | None] | None] = [None] * 256

ALL_TOKENS: list[Token] = []

# TEXT_INPUT: characters accepted when converting a Python string into tokens
# (TiString.from_str).  Populated at definition time for tokens marked
# `typeable=True` — these are the tokens whose display character is a legitimate,
# directly-typeable equivalent (letters, digits, punctuation, Greek, ...).  Keyed
# by the token's decoded text.
TEXT_INPUT: dict[str, Token] = {}

def get_token(code: int) -> Token:
	if code <= 0xFF:
		item = _TABLE[code]
	else:
		sub = _TABLE[code >> 8]
		lo = code & 0xFF
		item = sub[lo] if isinstance(sub, list) and lo < len(sub) else None
	if not isinstance(item, Token):
		raise KeyError(code)
	return item


def _set_token(token: Token):
	code = token.code
	if code <= 0xFF:
		tbl = _TABLE
		idx = code
	else:
		b0 = code >> 8
		idx = code & 0xFF
		tbl = _TABLE[b0]
		if tbl is None:
			_TABLE[b0] = tbl = []
		if idx >= len(tbl):
			tbl.extend([None] * (idx + 1 - len(tbl)))

	if (dup := tbl[idx]) is not None:
		raise ValueError(f"Duplicate token: {token} vs. {dup}")
	tbl[idx] = token


def read_token(f: BytesIO) -> Token:
	b0 = f.read(1)[0]
	code = (b0 << 8) | f.read(1)[0] if isinstance(_TABLE[b0], list) else b0
	try:
		return get_token(code)
	except KeyError:
		raise ValueError(f"Invalid token code: 0x{code:0{4 if code > 0xFF else 2}X}")




def token(
	code: int,
	display: bytes,
	*,
	typeable: bool = False,
	bp:   tuple[int, int] | None = None,
	op:   Callable | None = None,
	post: Callable | None = None,
	func: Callable | None = None,
	cmd:  Callable | None = None,
	res:  Callable | None = None,
	cnv:  Callable | None = None,
	var=None,
) -> Token:
	# `res` (a 0-arg query: π, getKey, …) and `var` (an Accessor) describe how the
	# token reaches the environment; both collapse to a single Accessor.
	accessor = None
	if res is not None:
		accessor = ComputedAccessor(res)
	elif var is not None:
		accessor = var
	t = Token(code, display, bp, op, post, func, cmd, accessor, cnv)
	_set_token(t)
	ALL_TOKENS.append(t)
	if typeable:
		TEXT_INPUT[t.text] = t
	return t


token(0x01, b'\x05DMS')   # ►DMS
token(0x02, b'\x05Dec')   # ►Dec
token(0x03, b'\x05Frac')  # ►Frac
token(tk.STORE, b'\x1c')  # →
token(0x05, b'Boxplot')
token(tk.L_BRACKET, b'\xc1',            typeable=True)  # The '[' symbol (θ steals this place in the charset)
token(tk.R_BRACKET, b']',               typeable=True)
token(tk.L_BRACE, b'{',                 typeable=True)
token(tk.R_BRACE, b'}',                 typeable=True)
token(tk.RAD, b'\x15')                                    # post needs env, handled specially
token(tk.DEG, b'\x14',                  typeable=True)  # ditto
token(0x0C, b'\x11',                    post=ops.inv)   # ¹
token(0x0D, b'\x12',                    post=lambda x: x**2, typeable=True)  # ²
token(0x0E, b'\x16',                    post=ops.transpose)                  # ᵀ
token(0x0F, b'\xd5',                    post=lambda x: x**3, typeable=True)  # ³
token(tk.L_PAREN, b'(',                 typeable=True)
token(tk.R_PAREN, b')',                 typeable=True)
token(0x12, b'round(',                  func=timath.round)
token(0x13, b'pxl-Test(',               func=draw.pxl_test)
token(0x14, b'augment(',                func=tilist.augment)
token(0x15, b'rowSwap(',                func=matrix.row_swap)
token(0x16, b'row+(',                   func=matrix.row_plus)
token(0x17, b'*row(',                   func=matrix.times_row)
token(0x18, b'*row+(',                  func=matrix.times_row_plus)
token(0x19, b'max(',                    func=timath.max)
token(0x1A, b'min(',                    func=timath.min)
token(0x1B, b'R\x05Pr(',                func=timath.rect_to_polar_radius)  # R►Pr(
token(0x1C, b'R\x05P\x5b(',             func=timath.rect_to_polar_angle)   # R►Pθ(
token(0x1D, b'P\x05Rx(',                func=timath.polar_to_rect_x)       # P►Rx(
token(0x1E, b'P\x05Ry(',                func=timath.polar_to_rect_y)       # P►Ry(
token(0x1F, b'median(',                 func=tilist.median)
token(0x20, b'randM(',                  func=matrix.rand_m)
token(0x21, b'mean(',                   func=tilist.mean)
token(0x22, b'solve(',                  func=timath.solve)
token(0x23, b'seq(',                    func=tilist.seq)
token(0x24, b'fnInt(',                  func=timath.fn_int)
token(0x25, b'nDeriv(',                 func=timath.n_deriv)
token(0x27, b'fMin(',                   func=timath.f_min)
token(0x28, b'fMax(',                   func=timath.f_max)
token(0x29, b' ',                       typeable=True)
token(tk.QUOTE, b'"',                   typeable=True)
token(tk.COMMA, b',',                   typeable=True)
token(0x2C, b'\xd7',                    res=lambda env: 1j, typeable=True)  # 𝑖
token(0x2D, b'!',                       post=ops.factorial, typeable=True)
token(0x2E, b'CubicReg ')
token(0x2F, b'QuartReg ')

# 0 - 9
for _i in range(10):
	token(0x30 + _i, bytes([0x30 + _i]), typeable=True)

token(tk.DOT, b'.',                     typeable=True)
token(tk.SCI_E, b'\x1b')  # ᴇ
token(0x3C, b' or ',                    bp=(20, 21), op=ops.or_)
token(0x3D, b' xor ',                   bp=(20, 21), op=ops.xor)
token(tk.COLON, b':',                   typeable=True)
token(tk.NEWLINE, b'\xd6',              typeable=True)
token(0x40, b' and ',                   bp=(30, 31), op=ops.and_)

# A - Z, θ
for _i in range(26):
	token(0x41 + _i, bytes([0x41 + _i]), var=NumericVar(chr(0x41 + _i)), typeable=True)

token(0x5B, b'\x5b', var=NumericVar('theta'), typeable=True)

# [A] - [J]
for _i in range(10):
	token(0x5C00 | _i, bytes([0xC1, 0x41 + _i, 0x5D]), var=MatrixVar(chr(0x41 + _i)))

# L₁ - L₆
for _i in range(6):
	token(0x5D00 | _i, bytes([0x4C, 0x81 + _i]), var=ListVar(_i))

# Y₁ - Y₀  (Function mode)
for _i in range(10):
	token(0x5E10 + _i, bytes([0x59, 0x80 + (_i + 1) % 10]), var=EquationVar('function', _i))

# X₁ₜ/Y₁ₜ - X₆ₜ/Y₆ₜ  (Parametric mode: pairs share one index)
for _i in range(6):
	token(0x5E20 + _i * 2,     bytes([0x58, 0x81 + _i, 0x0D]), var=ParEquationVar(_i, half='x'))
	token(0x5E20 + _i * 2 + 1, bytes([0x59, 0x81 + _i, 0x0D]), var=ParEquationVar(_i, half='y'))

# r₁ - r₆  (Polar mode)
for _i in range(6):
	token(0x5E40 + _i, bytes([0x72, 0x81 + _i]), var=EquationVar('polar', _i))

# 𝑢, 𝑣, 𝑤  (Sequence mode)
for _i in range(3):
	token(0x5E80 + _i, bytes([0x02 + _i]), var=SequenceVar(_i))

token(0x5F, b'prgm', cmd=cmds.prgm)

# Pic1 - Pic0
for _i in range(10):
	token(0x6000 + _i, b'Pic' + bytes([0x30 + (_i + 1) % 10]))

# GDB1 - GDB 0
for _i in range(10):
	token(0x6100 | _i, b'GDB' + bytes([0x30 + (_i + 1) % 10]))

token(0x6201, b'RegEq')
token(0x6202, b'n')
token(0x6203, b'\xcb')       # ẍ
token(0x6204, b'\xc6x')      # Σx
token(0x6205, b'\xc6x\x12')  # Σx²
token(0x6206, b'Sx')
token(0x6207, b'\xc7x')      # σx
token(0x6208, b'minX')
token(0x6209, b'maxX')
token(0x620A, b'minY')
token(0x620B, b'maxY')
token(0x620C, b'\xcc')       # ȳ
token(0x620D, b'\xc6y')      # Σy
token(0x620E, b'\xc6y\x12')  # Σy²
token(0x620F, b'Sy')
token(0x6210, b'\xc7y')      # σy
token(0x6211, b'\xc6xy')     # Σxy
token(0x6212, b'r')
token(0x6213, b'Med')
token(0x6214, b'Q1')
token(0x6215, b'Q3')
token(0x6216, b'a')
token(0x6217, b'b')
token(0x6218, b'c')
token(0x6219, b'd')
token(0x621A, b'e')
token(0x621B, b'x\x81')     # x₁
token(0x621C, b'x\x82')     # x₂
token(0x621D, b'x\x83')     # x₃
token(0x621E, b'y\x81')     # y₁
token(0x621F, b'y\x82')     # y₂
token(0x6220, b'y\x83')     # y₃
token(0x6221, b'\x01',                  var=EnvVar('n'))      # 𝑛
token(0x6222, b'p')
token(0x6223, b'z')
token(0x6224, b't')
token(0x6225, b'\xd9\x12')  # χ²
token(0x6226, b'\xda')      # Mathematical Bold Capital F, known as "Stat F" in the tibasicdev docs
token(0x6227, b'df')
token(0x6228, b'\xd8')      # ṕ
token(0x6229, b'\xd8\x81')  # ṕ₁
token(0x622A, b'\xd8\x82')  # ṕ₂
token(0x622B, b'\xcb\x81')  # ẍ₁
token(0x622C, b'Sx\x81')    # Sx₁
token(0x622D, b'n\x81')     # n₁
token(0x622E, b'\xcb\x82')  # ẍ₂
token(0x622F, b'Sx\x82')    # Sx₂
token(0x6230, b'n\x82')     # n₂
token(0x6231, b'Sxp')
token(0x6232, b'lower')
token(0x6233, b'upper')
token(0x6234, b's')
token(0x6235, b'r\x12')     # r²
token(0x6236, b'R\x12')     # R²
token(0x6237, b'df')        # Factor df
token(0x6238, b'SS')        # Factor SS
token(0x6239, b'MS')        # Factor MS
token(0x623A, b'df')        # Error df
token(0x623B, b'SS')        # Error SS
token(0x623C, b'MS')        # Error MS

token(0x6300, b'ZXscl',                 var=WindowVar('zxscl'))
token(0x6301, b'ZYscl',                 var=WindowVar('zyscl'))
token(0x6302, b'Xscl',                  var=WindowVar('xscl'))
token(0x6303, b'Yscl',                  var=WindowVar('yscl'))
token(0x6304, b'\x02(nMin)',            var=SequenceInitialVar(0))   # u(nMin)
token(0x6305, b'\x03(nMin)',            var=SequenceInitialVar(1))   # v(nMin)
token(0x6306, b'\x02(n-1)')             # u(n-1)
token(0x6307, b'\x03(n-1)')             # v(n-1)
token(0x6308, b'Z\x02(nMin)',           var=SequenceInitialVar(0))  # Zu(nMin) — same slot, zoom-window alias
token(0x6309, b'Z\x03(nMin)',           var=SequenceInitialVar(1))  # Zv(nMin)
token(0x630A, b'Xmin',                  var=WindowVar('xmin'))
token(0x630B, b'Xmax',                  var=WindowVar('xmax'))
token(0x630C, b'Ymin',                  var=WindowVar('ymin'))
token(0x630D, b'Ymax',                  var=WindowVar('ymax'))
token(0x630E, b'Tmin',                  var=WindowVar('tmin'))
token(0x630F, b'Tmax',                  var=WindowVar('tmax'))
token(0x6310, b'\x5bmin',               var=WindowVar('theta_min'))    # θmin
token(0x6311, b'\x5bmax',               var=WindowVar('theta_max'))    # θmax
token(0x6312, b'ZXmin',                 var=WindowVar('zxmin'))
token(0x6313, b'ZXmax',                 var=WindowVar('zxmax'))
token(0x6314, b'ZYmin',                 var=WindowVar('zymin'))
token(0x6315, b'ZYmax',                 var=WindowVar('zymax'))
token(0x6316, b'Z\x5bmin',              var=WindowVar('ztheta_min'))   # Zθmin
token(0x6317, b'Z\x5bmax',              var=WindowVar('ztheta_max'))   # Zθmax
token(0x6318, b'ZTmin',                 var=WindowVar('ztmin'))
token(0x6319, b'ZTmax',                 var=WindowVar('ztmax'))
token(0x631A, b'TblStart',              var=TableVar('tbl_start'))
token(0x631B, b'PlotStart',             var=WindowVar('plot_start'))
token(0x631C, b'ZPlotStart',            var=WindowVar('zplot_start'))
token(0x631D, b'nMax',                  var=IntWindowVar('n_max'))
token(0x631E, b'ZnMax',                 var=IntWindowVar('zn_max'))
token(0x631F, b'nMin',                  var=IntWindowVar('n_min'))
token(0x6320, b'ZnMin',                 var=IntWindowVar('zn_min'))
token(0x6321, b'\xbeTbl',               var=TableVar('delta_tbl'))       # ΔTbl
token(0x6322, b'Tstep',                 var=WindowVar('tstep'))
token(0x6323, b'\x5bstep',              var=WindowVar('theta_step'))    # θstep
token(0x6324, b'ZTstep',                var=WindowVar('ztstep'))
token(0x6325, b'Z\x5bstep',             var=WindowVar('ztheta_step'))  # Zθstep
token(0x6326, b'\xbeX',                 var=DeltaWindowVar('xmin', 'xmax', 94))  # ΔX
token(0x6327, b'\xbeY',                 var=DeltaWindowVar('ymin', 'ymax', 62))  # ΔY
token(0x6328, b'XFact',                 var=FactorWindowVar('x_fact'))
token(0x6329, b'YFact',                 var=FactorWindowVar('y_fact'))
token(0x632A, b'TblInput')
token(0x632B, b'\xdd',                  var=EnvVar('n_tvm'))     # 𝐍
token(0x632C, b'I%',                    var=EnvVar('i_pct'))
token(0x632D, b'PV',                    var=EnvVar('pv'))
token(0x632E, b'PMT',                   var=EnvVar('pmt'))
token(0x632F, b'FV',                    var=EnvVar('fv'))
token(0x6330, b'P/Y',                   var=EnvVar('py'))
token(0x6331, b'C/Y',                   var=EnvVar('cy'))
token(0x6332, b'\x04(nMin)', var=SequenceInitialVar(2))   # w(nMin)
token(0x6333, b'Z\x04(nMin)', var=SequenceInitialVar(2))  # Zw(nMin)
token(0x6334, b'PlotStep',              var=WindowVar('plot_step'))
token(0x6335, b'ZPlotStep',             var=WindowVar('zplot_step'))
token(0x6336, b'Xres',                  var=XresVar('xres'))
token(0x6337, b'ZXres',                 var=XresVar('zxres'))
token(0x6338, b'TraceStep')


token(0x64, b'Radian',                  cmd=modecmds.radian)
token(0x65, b'Degree',                  cmd=modecmds.degree)
token(0x66, b'Normal',                  cmd=modecmds.normal)
token(0x67, b'Sci',                     cmd=modecmds.sci)
token(0x68, b'Eng',                     cmd=modecmds.eng)
token(0x69, b'Float',                   cmd=modecmds.float_)
token(0x6A, b'=',                       bp=(40, 41), op=ops.eq,  typeable=True)
token(0x6B, b'<',                       bp=(40, 41), op=ops.lt,  typeable=True)
token(0x6C, b'>',                       bp=(40, 41), op=ops.gt,  typeable=True)
token(0x6D, b'\x17',                    bp=(40, 41), op=ops.le,  typeable=True)  # ≤
token(0x6E, b'\x19',                    bp=(40, 41), op=ops.ge,  typeable=True)  # ≥
token(0x6F, b'\x18',                    bp=(40, 41), op=ops.ne,  typeable=True)  # ≠
token(0x70, b'+',                       bp=(50, 51), op=ops.add, typeable=True)
token(0x71, b'-',                       bp=(50, 51), op=ops.sub, typeable=True)
token(tk.ANS, b'Ans')                   # Handled as special case in parser
token(0x73, b'Fix',                     cmd=modecmds.fix)
token(0x74, b'Horiz')
token(0x75, b'Full')
token(0x76, b'Func',                    cmd=modecmds.func)
token(0x77, b'Param',                   cmd=modecmds.param)
token(0x78, b'Polar',                   cmd=modecmds.polar)
token(0x79, b'Seq',                     cmd=modecmds.seq)
token(0x7A, b'IndpntAuto')
token(0x7B, b'IndpntAsk')
token(0x7C, b'DependAuto')
token(0x7D, b'DependAsk')


token(0x7E00, b'Sequential',            cmd=modecmds.sequential)
token(0x7E01, b'Simul',                 cmd=modecmds.simul)
token(0x7E02, b'PolarGC',               cmd=modecmds.polar_gc)
token(0x7E03, b'RectGC',                cmd=modecmds.rect_gc)
token(0x7E04, b'CoordOn',               cmd=modecmds.coord_on)
token(0x7E05, b'CoordOff',              cmd=modecmds.coord_off)
token(0x7E06, b'Connected',             cmd=modecmds.connected)
token(0x7E07, b'Dot',                   cmd=modecmds.dot)
token(0x7E08, b'AxesOn',                cmd=modecmds.axes_on)
token(0x7E09, b'AxesOff',               cmd=modecmds.axes_off)
token(0x7E0A, b'GridOn',                cmd=modecmds.grid_on)
token(0x7E0B, b'GridOff',               cmd=modecmds.grid_off)
token(0x7E0C, b'LabelOn',               cmd=modecmds.label_on)
token(0x7E0D, b'LabelOff',              cmd=modecmds.label_off)
token(0x7E0E, b'Web')
token(0x7E0F, b'Time')
token(0x7E10, b'uvAxes')
token(0x7E11, b'vwAxes')
token(0x7E12, b'uwAxes')

token(0x7F, b'\x0a')    # ▫
token(0x80, b'\x0b')  # ﹢
token(0x81, b'\x0c')  # ·
token(0x82, b'*',                       bp=(60, 61), op=ops.mul, typeable=True)
token(0x83, b'/',                       bp=(60, 61), op=ops.div, typeable=True)
token(0x84, b'Trace')
token(0x85, b'ClrDraw',                 cmd=draw.clr_draw)
token(0x86, b'ZStandard',               cmd=zoom.z_standard)
token(0x87, b'ZTrig',                   cmd=zoom.z_trig)
token(0x88, b'ZBox')
token(0x89, b'Zoom In',                 cmd=zoom.zoom_in)
token(0x8A, b'Zoom Out',                cmd=zoom.zoom_out)
token(0x8B, b'ZSquare',                 cmd=zoom.z_square)
token(0x8C, b'ZInteger',                cmd=zoom.z_integer)
token(0x8D, b'ZPrevious')
token(0x8E, b'ZDecimal',                cmd=zoom.z_decimal)
token(0x8F, b'ZoomStat')
token(0x90, b'ZoomRcl',                 cmd=cmds.zoom_rcl)
token(0x91, b'PrintScreen')
token(0x92, b'ZoomSto',                 cmd=cmds.zoom_sto)
token(0x93, b'Text(',                   cmd=draw.text)
token(0x94, b'nPr',                     bp=(60, 61), op=ops.npr)
token(0x95, b'nCr',                     bp=(60, 61), op=ops.ncr)
token(0x96, b'FnOn ', cmd=cmds.fn_on)
token(0x97, b'FnOff ', cmd=cmds.fn_off)
token(0x98, b'StorePic ')
token(0x99, b'RecallPic ')
token(0x9A, b'StoreGDB ')
token(0x9B, b'RecallGDB ')
token(0x9C, b'Line(',                   cmd=draw.line)
token(0x9D, b'Vertical ',               cmd=draw.vertical)
token(0x9E, b'Pt-On(',                  cmd=draw.pt_on)
token(0x9F, b'Pt-Off(',                 cmd=draw.pt_off)
token(0xA0, b'Pt-Change(',              cmd=draw.pt_change)
token(0xA1, b'Pxl-On(',                 cmd=draw.pxl_on)
token(0xA2, b'Pxl-Off(',                cmd=draw.pxl_off)
token(0xA3, b'Pxl-Change(',             cmd=draw.pxl_change)
token(0xA4, b'Shade(',                  cmd=draw.shade)
token(0xA5, b'Circle(',                 cmd=draw.circle)
token(0xA6, b'Horizontal ',             cmd=draw.horizontal)
token(0xA7, b'Tangent(',                cmd=draw.tangent)
token(0xA8, b'DrawInv ',                cmd=draw.draw_inv)
token(0xA9, b'DrawF ',                  cmd=draw.draw_f)

# Str1 - Str0
for _i in range(10):
	token(0xAA00 | _i, b'Str' + bytes([0x30 + (_i + 1) % 10]), var=StringVar(_i))

token(tk.RAND, b'rand',                 var=RandAccessor(), func=timath.rand_list)
token(0xAC, b'\xc4',                    res=lambda env: math.pi, typeable=True)  # π
token(0xAD, b'getKey',                  res=Environment.get_key)
token(tk.APOS, b"'",                    typeable=True)
token(0xAF, b'?',                       typeable=True)
token(tk.NEG, b'\x1a')  # ⁻
token(0xB1, b'int(',                    func=timath.int_)
token(0xB2, b'abs(',                    func=timath.abs)
token(0xB3, b'det(',                    func=matrix.det)
token(0xB4, b'identity(',               func=matrix.identity)
token(tk.DIM, b'dim(',                  func=tilist.dim)
token(0xB6, b'sum(',                    func=tilist.sum)
token(0xB7, b'prod(',                   func=tilist.prod)
token(0xB8, b'not(',                    func=timath.not_)
token(0xB9, b'iPart(',                  func=timath.i_part)
token(0xBA, b'fPart(',                  func=timath.f_part)


token(0xBB00, b'npv(',  			    func=finance.npv)
token(0xBB01, b'irr(',  			    func=finance.irr)
token(0xBB02, b'bal(',  			    func=finance.bal)
token(0xBB03, b'\xc6prn(', 			    func=finance.sigma_prn)  # Σprn(
token(0xBB04, b'\xc6Int(', 			    func=finance.sigma_int)  # ΣInt(
token(0xBB05, b'\x05Nom(', 			    func=finance.nom)  # ►Nom(
token(0xBB06, b'\x05Eff(', 			    func=finance.eff)  # ►Eff(
token(0xBB07, b'dbd(',                  func=finance.dbd)
token(0xBB08, b'lcm(',                  func=timath.lcm)
token(0xBB09, b'gcd(',                  func=timath.gcd)
token(0xBB0A, b'randInt(',              func=timath.rand_int)
token(0xBB0B, b'randBin(',              func=timath.rand_bin)
token(0xBB0C, b'sub(',                  func=tistring.sub)
token(0xBB0D, b'stdDev(',               func=tilist.stddev)
token(0xBB0E, b'variance(',             func=tilist.variance)
token(0xBB0F, b'inString(',             func=tistring.in_string)
token(0xBB10, b'normalcdf(',            func=dist.normalcdf)
token(0xBB11, b'invNorm(',              func=dist.inv_norm)
token(0xBB12, b'tcdf(',                 func=dist.tcdf)
token(0xBB13, b'\xd9\x12cdf(',          func=dist.chi_sq_cdf)  # χ²cdf(
token(0xBB14, b'Fcdf(',                 func=dist.fcdf)
token(0xBB15, b'binompdf(',             func=dist.binompdf)
token(0xBB16, b'binomcdf(',             func=dist.binomcdf)
token(0xBB17, b'poissonpdf(',           func=dist.poissonpdf)
token(0xBB18, b'poissoncdf(',           func=dist.poissoncdf)
token(0xBB19, b'geometpdf(',            func=dist.geometpdf)
token(0xBB1A, b'geometcdf(',            func=dist.geometcdf)
token(0xBB1B, b'normalpdf(',            func=dist.normalpdf)
token(0xBB1C, b'tpdf(',                 func=dist.tpdf)
token(0xBB1D, b'\xd9\x12pdf(',          func=dist.chi_sq_pdf)  # χ²pdf(
token(0xBB1E, b'Fpdf(',                 func=dist.f_pdf)
token(0xBB1F, b'randNorm(',             func=timath.rand_norm)
token(0xBB20, b'tvm_Pmt',               res=finance.tvm_pmt_value,   func=finance.tvm_pmt)
token(0xBB21, b'tvm_I%',                res=finance.tvm_i_pct_value, func=finance.tvm_i_pct)
token(0xBB22, b'tvm_PV',                res=finance.tvm_pv_value,    func=finance.tvm_pv)
token(0xBB23, b'tvm_N',                 res=finance.tvm_n_value,     func=finance.tvm_n)
token(0xBB24, b'tvm_FV',                res=finance.tvm_fv_value,    func=finance.tvm_fv)
token(0xBB25, b'conj(',                 func=timath.conj)
token(0xBB26, b'real(',                 func=timath.real)
token(0xBB27, b'imag(',                 func=timath.imag)
token(0xBB28, b'angle(',                func=timath.angle)
token(0xBB29, b'cumSum(',               func=tilist.cum_sum)
token(0xBB2A, b'expr(',                 func=tistring.expr)
token(0xBB2B, b'length(',               func=tistring.length)
token(0xBB2C, b'\xbeList(',             func=tilist.delta_list)  # ΔList(
token(0xBB2D, b'ref(',                  func=matrix.ref)
token(0xBB2E, b'rref(',                 func=matrix.rref)
token(0xBB2F, b'\x05Rect')  # ►Rect
token(0xBB30, b'\x05Polar')  # ►Polar
token(0xBB31, b'\xdb',                  res=lambda env: math.e, typeable=True)  # 𝑒
token(0xBB32, b'SinReg ')
token(0xBB33, b'Logistic ')
token(0xBB34, b'LinRegTTest ')
token(0xBB35, b'ShadeNorm(',            cmd=draw.shade_norm)
token(0xBB36, b'Shade_t(',              cmd=draw.shade_t)
token(0xBB37, b'Shade\xd9\x12(',        cmd=draw.shade_chi_sq)    # Shadeχ²(
token(0xBB38, b'Shade\xda(',            cmd=draw.shade_f)         # Shade𝐅(
token(0xBB39, b'Matr\x05list(',         cmd=matrix.matr_to_list)  # Matr►list(
token(0xBB3A, b'List\x05matr(',         cmd=matrix.list_to_matr)  # List►matr(
token(0xBB3B, b'Z-Test(')
token(0xBB3C, b'T-Test')
token(0xBB3D, b'2-SampZTest(')
token(0xBB3E, b'1-PropZTest(')
token(0xBB3F, b'2-PropZTest(')
token(0xBB40, b'\xd9\x12-Test(')  # χ²-Test(
token(0xBB41, b'ZInterval ')
token(0xBB42, b'2-SampZInt(')
token(0xBB43, b'1-PropZInt(')
token(0xBB44, b'2-PropZInt(')
token(0xBB45, b'GraphStyle(')
token(0xBB46, b'2-SampTTest ')
token(0xBB47, b'2-SampFTest ')
token(0xBB48, b'TInterval ')
token(0xBB49, b'2-SampTInt ')
token(0xBB4A, b'SetUpEditor ',          cmd=tilist.set_up_editor)
token(0xBB4B, b'Pmt_End')
token(0xBB4C, b'Pmt_Bgn')
token(0xBB4D, b'Real',                  cmd=modecmds.real)
token(0xBB4E, b're^\x5bi',              cmd=modecmds.re_theta_i)  # re^θi
token(0xBB4F, b'a+bi',                  cmd=modecmds.a_plus_bi)
token(0xBB50, b'ExprOn',                cmd=modecmds.expr_on)
token(0xBB51, b'ExprOff',               cmd=modecmds.expr_off)
token(0xBB52, b'ClrAllLists',           cmd=tilist.clr_all_lists)
token(0xBB53, b'GetCalc(')
token(0xBB54, b'DelVar ',               cmd=cmds.del_var)
token(0xBB55, b'Equ\x05String(',        cmd=tistring.equ_to_string)  # Equ►String(
token(0xBB56, b'String\x05Equ(',        cmd=tistring.string_to_equ)  # String►Equ(
token(0xBB57, b'Clear Entries')
token(0xBB58, b'Select(')
token(0xBB59, b'ANOVA(')
token(0xBB5A, b'ModBoxplot')
token(0xBB5B, b'NormProbPlot')
token(0xBB64, b'G-T')
token(0xBB65, b'ZoomFit',               cmd=zoom.zoom_fit)
token(0xBB66, b'DiagnosticOn',          cmd=modecmds.diagnostic_on)
token(0xBB67, b'DiagnosticOff',         cmd=modecmds.diagnostic_off)
token(0xBB68, b'Archive ')
token(0xBB69, b'UnArchive ')
token(0xBB6A, b'Asm(')
token(0xBB6B, b'AsmComp(')
token(0xBB6C, b'?')  # "compiled asm" token, displays as '?'
token(0xBB6D, b'compiled asm')
token(0xBB6E, b'\x8a',                  typeable=True)  # Á
token(0xBB6F, b'\x8b',                  typeable=True)  # À
token(0xBB70, b'\x8c',                  typeable=True)  # Â
token(0xBB71, b'\x8d',                  typeable=True)  # Ä
token(0xBB72, b'\x8e',                  typeable=True)  # á
token(0xBB73, b'\x8f',                  typeable=True)  # à
token(0xBB74, b'\x90',                  typeable=True)  # â
token(0xBB75, b'\x91',                  typeable=True)  # ä
token(0xBB76, b'\x92',                  typeable=True)  # É
token(0xBB77, b'\x93',                  typeable=True)  # È
token(0xBB78, b'\x94',                  typeable=True)  # Ê
token(0xBB79, b'\x95',                  typeable=True)  # Ë
token(0xBB7A, b'\x96',                  typeable=True)  # é
token(0xBB7B, b'\x97',                  typeable=True)  # è
token(0xBB7C, b'\x98',                  typeable=True)  # ê
token(0xBB7D, b'\x99',                  typeable=True)  # ë
token(0xBB7F, b'\x9b',                  typeable=True)  # Ì
token(0xBB80, b'\x9c',                  typeable=True)  # Î
token(0xBB81, b'\x9d',                  typeable=True)  # Ï
token(0xBB82, b'\x9e',                  typeable=True)  # í
token(0xBB83, b'\x9f',                  typeable=True)  # ì
token(0xBB84, b'\xa0',                  typeable=True)  # î
token(0xBB85, b'\xa1',                  typeable=True)  # ï
token(0xBB86, b'\xa2',                  typeable=True)  # Ó
token(0xBB87, b'\xa3',                  typeable=True)  # Ò
token(0xBB88, b'\xa4',                  typeable=True)  # Ô
token(0xBB89, b'\xa5',                  typeable=True)  # Ö
token(0xBB8A, b'\xa6',                  typeable=True)  # ó
token(0xBB8B, b'\xa7',                  typeable=True)  # ò
token(0xBB8C, b'\xa8',                  typeable=True)  # ô
token(0xBB8D, b'\xa9',                  typeable=True)  # ö
token(0xBB8E, b'\xaa',                  typeable=True)  # Ú
token(0xBB8F, b'\xab',                  typeable=True)  # Ù
token(0xBB90, b'\xac',                  typeable=True)  # Û
token(0xBB91, b'\xad',                  typeable=True)  # Ü
token(0xBB92, b'\xae',                  typeable=True)  # ú
token(0xBB93, b'\xaf',                  typeable=True)  # ù
token(0xBB94, b'\xb0',                  typeable=True)  # û
token(0xBB95, b'\xb1',                  typeable=True)  # ü
token(0xBB96, b'\xb2',                  typeable=True)  # Ç
token(0xBB97, b'\xb3',                  typeable=True)  # ç
token(0xBB98, b'\xb4',                  typeable=True)  # Ñ
token(0xBB99, b'\xb5',                  typeable=True)  # ñ
token(0xBB9A, b'\xb6')                                  # ´
token(0xBB9B, b'\xb7')                                  # Modifier Letter Grave Accent (differentiates from backtick)
token(0xBB9C, b'\xb8')                                  # ¨
token(0xBB9D, b'\xb9',                  typeable=True)  # ¿
token(0xBB9E, b'\xba',                  typeable=True)  # ¡
token(0xBB9F, b'\xbb',                  typeable=True)  # α
token(0xBBA0, b'\xbc',                  typeable=True)  # β
token(0xBBA1, b'\xbd',                  typeable=True)  # γ
token(0xBBA2, b'\xbe',                  typeable=True)  # Δ
token(0xBBA3, b'\xbf',                  typeable=True)  # δ
token(0xBBA4, b'\xc0',                  typeable=True)  # ε
token(0xBBA5, b'\xc2',                  typeable=True)  # λ
token(0xBBA6, b'\xc3',                  typeable=True)  # μ
token(0xBBA7, b'\xc4')                                  # π (homoglyph of 0xAC without syntactic meaning)
token(0xBBA8, b'\xc5',                  typeable=True)  # ρ
token(0xBBA9, b'\xc6',                  typeable=True)  # Σ
token(0xBBAB, b'\xc9',                  typeable=True)  # φ
token(0xBBAC, b'\xca',                  typeable=True)  # Ω
token(0xBBAD, b'\xd8',                  typeable=True)  # ṕ (homoglyph of 0x6228 without syntactic meaning)
token(0xBBAE, b'\xd9',                  typeable=True)  # χ
token(0xBBAF, b'\x0f')                                  # Mathematical Bold Capital Digamma, known as "Hexadecimal F" in the tibasicdev docs
token(0xBBB0, b'a',                     typeable=True)
token(0xBBB1, b'b',                     typeable=True)
token(0xBBB2, b'c',                     typeable=True)
token(0xBBB3, b'd',                     typeable=True)
token(0xBBB4, b'e',                     typeable=True)
token(0xBBB5, b'f',                     typeable=True)
token(0xBBB6, b'g',                     typeable=True)
token(0xBBB7, b'h',                     typeable=True)
token(0xBBB8, b'i',                     typeable=True)
token(0xBBB9, b'j',                     typeable=True)
token(0xBBBA, b'k',                     typeable=True)
token(0xBBBC, b'l',                     typeable=True)
token(0xBBBD, b'm',                     typeable=True)
token(0xBBBE, b'n',                     typeable=True)
token(0xBBBF, b'o',                     typeable=True)
token(0xBBC0, b'p',                     typeable=True)
token(0xBBC1, b'q',                     typeable=True)
token(0xBBC2, b'r',                     typeable=True)
token(0xBBC3, b's',                     typeable=True)
token(0xBBC4, b't',                     typeable=True)
token(0xBBC5, b'u',                     typeable=True)
token(0xBBC6, b'v',                     typeable=True)
token(0xBBC7, b'w',                     typeable=True)
token(0xBBC8, b'x',                     typeable=True)
token(0xBBC9, b'y',                     typeable=True)
token(0xBBCA, b'z',                     typeable=True)
token(0xBBCB, b'\xc7',                  typeable=True)  # σ
token(0xBBCC, b'\xc8',                  typeable=True)  # τ
token(0xBBCD, b'\x9a',                  typeable=True)  # Í
token(0xBBCE, b'GarbageCollect')
token(0xBBCF, b'~',                     typeable=True)
token(0xBBD1, b'@',                     typeable=True)
token(0xBBD2, b'#',                     typeable=True)
token(0xBBD3, b'\xf2',                  typeable=True)  # $
token(0xBBD4, b'&',                     typeable=True)
token(0xBBD5, b'`',                     typeable=True)  # Backtick (confusing because the symbol is officially called Grave Accent)
token(0xBBD6, b';',                     typeable=True)
token(0xBBD7, b'\\',                    typeable=True)
token(0xBBD8, b'|',                     typeable=True)
token(0xBBD9, b'_',                     typeable=True)
token(0xBBDA, b'%',                     typeable=True, post=lambda x: x / 100)
token(0xBBDB, b'\xce',                  typeable=True)  # …
token(0xBBDC, b'\x13',                  typeable=True)  # ∠
token(0xBBDD, b'\xf4',                  typeable=True)  # ß
token(0xBBDE, b'\xcd')                                  # ˣ
token(0xBBDF, b'\r')                                    # ₜ
token(0xBBE0, b'\x80')                                  # ₀
token(0xBBE1, b'\x81')                                  # ₁
token(0xBBE2, b'\x82')                                  # ₂
token(0xBBE3, b'\x83')                                  # ₃
token(0xBBE4, b'\x84')                                  # ₄
token(0xBBE5, b'\x85')                                  # ₅
token(0xBBE6, b'\x86')                                  # ₆
token(0xBBE7, b'\x87')                                  # ₇
token(0xBBE8, b'\x88')                                  # ₈
token(0xBBE9, b'\x89')                                  # ₉
token(0xBBEA, b'\x1d')                                  # ⑽
token(0xBBEB, b'\xcf')                                  # ◄
token(0xBBEC, b'\x05')                                  # ►
token(0xBBED, b'\x1e')                                  # ↑
token(0xBBEE, b'\x1f')                                  # ↓
token(0xBBF0, b'\x09')                                  # ×
token(0xBBF1, b'\x08',                  typeable=True)  # ∫
token(0xBBF2, b'\x06')                                  # 🡅
token(0xBBF3, b'\x07')                                  # 🡇
token(0xBBF4, b'\x10')                                  # √
token(0xBBF5, b'\x7f')                                  # ≛


token(0xBC, b'\x10(',                   func=timath.sqrt)   # √(
token(0xBD, b'\x0e\x10(',               func=timath.cbrt)   # 𝟑√(
token(0xBE, b'ln(',		                func=timath.ln)
token(0xBF, b'\xdb^(',		            func=timath.exp)    # 𝑒^(
token(0xC0, b'log(',		            func=timath.log)
token(0xC1, b'\x1d^(',		            func=timath.pow10)  # ⑽^(
token(0xC2, b'sin(',		            func=timath.sin)
token(0xC3, b'sin\x11(',	            func=timath.asin)   # sin¹(
token(0xC4, b'cos(',		            func=timath.cos)
token(0xC5, b'cos\x11(',	            func=timath.acos)   # cos¹(
token(0xC6, b'tan(',		            func=timath.tan)
token(0xC7, b'tan\x11(',	            func=timath.atan)   # tan¹(
token(0xC8, b'sinh(',	                func=timath.sinh)
token(0xC9, b'sinh\x11(',	            func=timath.asinh)  # sinh¹(
token(0xCA, b'cosh(',	                func=timath.cosh)
token(0xCB, b'cosh\x11(',	            func=timath.acosh)  # cosh¹(
token(0xCC, b'tanh(',	                func=timath.tanh)
token(0xCD, b'tanh\x11(',	            func=timath.atanh)  # tanh¹(
token(tk.IF, b'If ',                    cmd=cmds.if_cmd)
token(tk.THEN, b'Then',                 cmd=cmds.then_cmd)
token(tk.ELSE, b'Else',                 cmd=cmds.else_cmd)
token(tk.WHILE, b'While ',              cmd=cmds.while_cmd)
token(tk.REPEAT, b'Repeat ',            cmd=cmds.repeat_cmd)
token(tk.FOR, b'For(',                  cmd=cmds.for_cmd)
token(tk.END, b'End',                   cmd=cmds.end_cmd)
token(0xD5, b'Return',                  cmd=cmds.return_cmd)
token(tk.LBL, b'Lbl ',                  cmd=cmds.lbl_cmd)
token(0xD7, b'Goto ',                   cmd=cmds.goto_cmd)
token(0xD8, b'Pause ',                  cmd=cmds.pause_cmd)
token(0xD9, b'Stop',                    cmd=cmds.stop_cmd)
token(0xDA, b'IS>(',                    cmd=cmds.is_gt_cmd)
token(0xDB, b'DS<(',                    cmd=cmds.ds_lt_cmd)
token(0xDC, b'Input ',                  cmd=cmds.input_cmd)
token(0xDD, b'Prompt ',                 cmd=cmds.prompt_cmd)
token(0xDE, b'Disp ',                   cmd=cmds.disp)
token(0xDF, b'DispGraph',               cmd=cmds.disp_graph)
token(0xE0, b'Output(',                 cmd=cmds.output)
token(0xE1, b'ClrHome',                 cmd=cmds.clr_home)
token(0xE2, b'Fill(',                   cmd=tilist.fill)
token(0xE3, b'SortA(',                  cmd=tilist.sort_a)
token(0xE4, b'SortD(',                  cmd=tilist.sort_d)
token(0xE5, b'DispTable',               cmd=cmds.disp_table)
token(0xE6, b'Menu(',                   cmd=cmds.menu_cmd)
token(0xE7, b'Send(')
token(0xE8, b'Get(')
token(0xE9, b'PlotsOn', cmd=_PLACEHOLDER)
token(0xEA, b'PlotsOff', cmd=_PLACEHOLDER)
token(tk.LIST_PREFIX, b'\xdc')  # ᴸ
token(0xEC, b'Plot1(')
token(0xED, b'Plot2(')
token(0xEE, b'Plot3(')


token(0xEF00, b'setDate(',              cmd=titime.set_date)
token(0xEF01, b'setTime(',              cmd=titime.set_time)
token(0xEF02, b'checkTmr(',             func=titime.check_tmr)
token(0xEF03, b'setDtFmt(',             cmd=titime.set_dt_fmt)
token(0xEF04, b'setTmFmt(',             cmd=titime.set_tm_fmt)
token(0xEF05, b'timeCnv(',              func=titime.time_cnv)
token(0xEF06, b'dayOfWk(',              func=titime.dayofwk)
token(0xEF07, b'getDtStr(',             func=titime.get_dt_str)
token(0xEF08, b'getTmStr(',             func=titime.get_tm_str)
token(0xEF09, b'getDate',               res=Environment.get_date)
token(0xEF0A, b'getTime',               res=Environment.get_time)
token(0xEF0B, b'startTmr',              res=Environment.start_tmr)
token(0xEF0C, b'getDtFmt',              res=Environment.get_dt_fmt)
token(0xEF0D, b'getTmFmt',              res=Environment.get_tm_fmt)
token(0xEF0E, b'isClockOn',             res=Environment.is_clock_on)
token(0xEF0F, b'ClockOff',              cmd=modecmds.clock_off)
token(0xEF10, b'ClockOn',               cmd=modecmds.clock_on)
token(0xEF11, b'OpenLib(')
token(0xEF12, b'ExecLib')
token(0xEF13, b'invT(',                 func=dist.inv_t)
token(0xEF14, b'\xd9\x12GOF-Test(')  # χ²GOF-Test(
token(0xEF15, b'LinRegTInt ')
token(0xEF16, b'Manual-Fit ')
token(0xEF17, b'ZQuadrant1')
token(0xEF18, b'ZFrac1/2')
token(0xEF19, b'ZFrac1/3')
token(0xEF1A, b'ZFrac1/4')
token(0xEF1B, b'ZFrac1/5')
token(0xEF1C, b'ZFrac1/8')
token(0xEF1D, b'ZFrac1/10')
token(0xEF1E, b'mathprintbox')
token(0xEF30, b'\x05n/d\xcf\x05Un/d')  # ►n/d◄►Un/d
token(0xEF31, b'\x05F\xcf\x05D')  # ►F◄►D
token(0xEF32, b'remainder(',            func=timath.remainder)
token(0xEF33, b'\xc6(',                 func=timath.sigma)  # Σ(
token(0xEF34, b'logBASE(',              func=timath.log_base)
token(0xEF35, b'randIntNoRep(',         func=timath.rand_int_no_rep)
token(0xEF36, b'MATHPRINT')
token(0xEF37, b'CLASSIC')
token(0xEF38, b'n/d')
token(0xEF39, b'Un/d')
token(0xEF3A, b'AUTO')
token(0xEF3B, b'DEC')
token(0xEF3C, b'FRAC')
token(0xEF3D, b'FRAC-APPROX')


token(0xF0, b'^',                       bp=(70, 70), op=ops.power, typeable=True)
token(0xF1, b'\xcd\x10',                bp=(60, 61), op=ops.xth_root)  # ˣ√
token(0xF2, b'1-Var Stats ')
token(0xF3, b'2-Var Stats ')
token(0xF4, b'LinReg(a+bx) ')
token(0xF5, b'ExpReg ')
token(0xF6, b'LnReg ')
token(0xF7, b'PwrReg ')
token(0xF8, b'Med-Med ')
token(0xF9, b'QuadReg ')
token(0xFA, b'ClrList ',                cmd=tilist.clr_list)
token(0xFB, b'ClrTable')
token(0xFC, b'Histogram')
token(0xFD, b'xyLine')
token(0xFE, b'Scatter')
token(0xFF, b'LinReg(ax+b) ')


if __name__ == '__main__':
	print(len(ALL_TOKENS))
