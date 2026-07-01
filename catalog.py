import math
from io import BytesIO

from core import require_real, TiList, TiMatrix
from environment import Environment
from parser import ArgParser
from preparse import special_func
from tokenbase import Token, Flag
import tokenbase as tok
from tokentypes import *
import prgmcmds as prgm
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
import zoom


@special_func
def _PLACEHOLDER(args):
	while args.has_next:
		args.equation_var()
	args.end_cmd()


def _generate():
	yield Token(0x01, b'\x05DMS')   # ►DMS
	yield Token(0x02, b'\x05Dec')   # ►Dec
	yield Token(0x03, b'\x05Frac')  # ►Frac
	yield Token(tok.STORE, b'\x1c')  # →
	yield Token(0x05, b'Boxplot')
	yield Token(tok.L_BRACKET, b'\xc1', Flag.ATOM_START, char='[')
	yield Token(tok.R_BRACKET, b']',                char=']')
	yield Token(tok.L_BRACE, b'{', Flag.ATOM_START, char='{')
	yield Token(tok.R_BRACE, b'}',                  char='}')
	yield Token(tok.RAD, b'\x15')
	yield Token(tok.DEG, b'\x14',                   char='°')
	yield PostfixToken(0x0C, b'\x11',               ops.inv)
	yield PostfixToken(0x0D, b'\x12',               ops.squared)
	yield PostfixToken(0x0E, b'\x16',               ops.transpose)
	yield PostfixToken(0x0F, b'\xd5',               ops.cubed)
	yield Token(tok.L_PAREN, b'(', Flag.ATOM_START, char='(')
	yield Token(tok.R_PAREN, b')',                  char=')')
	yield FunctionToken(0x12, b'round(',            timath.round)
	yield FunctionToken(0x13, b'pxl-Test(',         draw.pxl_test)
	yield FunctionToken(0x14, b'augment(',          tilist.augment)
	yield FunctionToken(0x15, b'rowSwap(',          matrix.row_swap)
	yield FunctionToken(0x16, b'row+(',             matrix.row_plus)
	yield FunctionToken(0x17, b'*row(',             matrix.times_row)
	yield FunctionToken(0x18, b'*row+(',            matrix.times_row_plus)
	yield FunctionToken(0x19, b'max(',              timath.max)
	yield FunctionToken(0x1A, b'min(',              timath.min)
	yield FunctionToken(0x1B, b'R\x05Pr(',          timath.rect_to_polar_radius)   # R►Pr(
	yield FunctionToken(0x1C, b'R\x05P\x5b(',       timath.rect_to_polar_angle)    # R►Pθ(
	yield FunctionToken(0x1D, b'P\x05Rx(',          timath.polar_to_rect_x)        # P►Rx(
	yield FunctionToken(0x1E, b'P\x05Ry(',          timath.polar_to_rect_y)        # P►Ry(
	yield FunctionToken(0x1F, b'median(',           tilist.median)
	yield FunctionToken(0x20, b'randM(',            matrix.rand_m)
	yield FunctionToken(0x21, b'mean(',             tilist.mean)
	yield FunctionToken(0x22, b'solve(',            timath.solve)
	yield FunctionToken(0x23, b'seq(',              tilist.seq)
	yield FunctionToken(0x24, b'fnInt(',            timath.fn_int)
	yield FunctionToken(0x25, b'nDeriv(',           timath.n_deriv)
	yield FunctionToken(0x27, b'fMin(',             timath.f_min)
	yield FunctionToken(0x28, b'fMax(',             timath.f_max)
	yield Token(0x29, b' ',                         char=' ')
	yield Token(tok.QUOTE, b'"', Flag.ATOM_START,   char='"')
	yield Token(tok.COMMA, b',',                    char=',')
	yield ConstantToken(0x2C, b'\xd7', 1j,          char='𝑖')
	yield PostfixToken(0x2D, b'!',                  ops.factorial, char='!')
	yield Token(0x2E, b'CubicReg ')
	yield Token(0x2F, b'QuartReg ')

	for i in range(10):
		ch = 0x30 + i
		yield Token(ch, bytes([ch]), Flag.DIGIT, char=chr(ch))

	yield Token(tok.DOT, b'.', Flag.ATOM_START,     char='.')
	yield Token(tok.SCI_E, b'\x1b', Flag.ATOM_START)                    # ᴇ
	yield OperatorToken(0x3C, b' or ',              ops.or_,   (20, 21))
	yield OperatorToken(0x3D, b' xor ',             ops.xor,   (20, 21))
	yield Token(tok.COLON, b':',                    char=':')
	yield Token(tok.NEWLINE, b'\xd6',               char='\n')
	yield OperatorToken(0x40, b' and ',             ops.and_,  (30, 31))

	for i in range(27):
		yield LetterToken(i)

	for i in range(10):
		yield MatrixToken(i)

	for i in range(6):
		yield ListToken(i)

	for i in range(10):
		yield FuncEquationToken(i)

	for i in range(6):
		yield ParEquationToken(i, 'x')
		yield ParEquationToken(i, 'y')

	for i in range(6):
		yield PolarEquationToken(i)

	for i in range(3):
		yield SequenceToken(i)

	yield CommandToken(0x5F, b'prgm', prgm.prgm)

	for i in range(10):
		yield PicToken(i)

	for i in range(10):
		yield Token(0x6100 | i, b'GDB' + bytes([0x30 + (i + 1) % 10]), Flag.GDB)

	yield Token(0x6201, b'RegEq')
	yield Token(0x6202, b'n')
	yield Token(0x6203, b'\xcb')                                         # ẍ
	yield Token(0x6204, b'\xc6x')                                        # Σx
	yield Token(0x6205, b'\xc6x\x12')                                    # Σx²
	yield Token(0x6206, b'Sx')
	yield Token(0x6207, b'\xc7x')                                        # σx
	yield Token(0x6208, b'minX')
	yield Token(0x6209, b'maxX')
	yield Token(0x620A, b'minY')
	yield Token(0x620B, b'maxY')
	yield Token(0x620C, b'\xcc')                                         # ȳ
	yield Token(0x620D, b'\xc6y')                                        # Σy
	yield Token(0x620E, b'\xc6y\x12')                                    # Σy²
	yield Token(0x620F, b'Sy')
	yield Token(0x6210, b'\xc7y')                                        # σy
	yield Token(0x6211, b'\xc6xy')                                       # Σxy
	yield Token(0x6212, b'r')
	yield Token(0x6213, b'Med')
	yield Token(0x6214, b'Q1')
	yield Token(0x6215, b'Q3')
	yield Token(0x6216, b'a')
	yield Token(0x6217, b'b')
	yield Token(0x6218, b'c')
	yield Token(0x6219, b'd')
	yield Token(0x621A, b'e')
	yield Token(0x621B, b'x\x81')                                        # x₁
	yield Token(0x621C, b'x\x82')                                        # x₂
	yield Token(0x621D, b'x\x83')                                        # x₃
	yield Token(0x621E, b'y\x81')                                        # y₁
	yield Token(0x621F, b'y\x82')                                        # y₂
	yield Token(0x6220, b'y\x83')                                        # y₃
	yield RealToken(0x6221, b'\x01', 'n', Flag.NUMERIC)                  # 𝑛 (sequence index)
	yield Token(0x6222, b'p')
	yield Token(0x6223, b'z')
	yield Token(0x6224, b't')
	yield Token(0x6225, b'\xd9\x12')                                     # χ²
	yield Token(0x6226, b'\xda')                                         # 𝐅 (stat F)
	yield Token(0x6227, b'df')
	yield Token(0x6228, b'\xd8')                                         # ṕ (p-hat)
	yield Token(0x6229, b'\xd8\x81')                                     # ṕ₁
	yield Token(0x622A, b'\xd8\x82')                                     # ṕ₂
	yield Token(0x622B, b'\xcb\x81')                                     # ẍ₁
	yield Token(0x622C, b'Sx\x81')                                       # Sx₁
	yield Token(0x622D, b'n\x81')                                        # n₁
	yield Token(0x622E, b'\xcb\x82')                                     # ẍ₂
	yield Token(0x622F, b'Sx\x82')                                       # Sx₂
	yield Token(0x6230, b'n\x82')                                        # n₂
	yield Token(0x6231, b'Sxp')
	yield Token(0x6232, b'lower')
	yield Token(0x6233, b'upper')
	yield Token(0x6234, b's')
	yield Token(0x6235, b'r\x12')                                        # r²
	yield Token(0x6236, b'R\x12')                                        # R²
	yield Token(0x6237, b'df')                                           # Factor df
	yield Token(0x6238, b'SS')                                           # Factor SS
	yield Token(0x6239, b'MS')                                           # Factor MS
	yield Token(0x623A, b'df')                                           # Error df
	yield Token(0x623B, b'SS')                                           # Error SS
	yield Token(0x623C, b'MS')                                           # Error MS

	yield WindowToken(0x6300, b'ZXscl',             'zxscl')
	yield WindowToken(0x6301, b'ZYscl',             'zyscl')
	yield WindowToken(0x6302, b'Xscl',              'xscl')
	yield WindowToken(0x6303, b'Yscl',              'yscl')
	yield SequenceInitialToken(0x6304, b'\x02(nMin)', 0)                 # 𝑢(nMin)
	yield SequenceInitialToken(0x6305, b'\x03(nMin)', 1)                 # 𝑣(nMin)
	yield Token(0x6306, b'\x02(n-1)')                                    # 𝑢(n-1) read-only
	yield Token(0x6307, b'\x03(n-1)')                                    # 𝑣(n-1) read-only
	yield SequenceInitialToken(0x6308, b'Z\x02(nMin)', 0)                # Z𝑢(nMin)
	yield SequenceInitialToken(0x6309, b'Z\x03(nMin)', 1)                # Z𝑣(nMin)
	yield WindowToken(0x630A, b'Xmin',              'xmin')
	yield WindowToken(0x630B, b'Xmax',              'xmax')
	yield WindowToken(0x630C, b'Ymin',              'ymin')
	yield WindowToken(0x630D, b'Ymax',              'ymax')
	yield WindowToken(0x630E, b'Tmin',              'tmin')
	yield WindowToken(0x630F, b'Tmax',              'tmax')
	yield WindowToken(0x6310, b'\x5bmin',           'theta_min')         # θmin
	yield WindowToken(0x6311, b'\x5bmax',           'theta_max')         # θmax
	yield WindowToken(0x6312, b'ZXmin',             'zxmin')
	yield WindowToken(0x6313, b'ZXmax',             'zxmax')
	yield WindowToken(0x6314, b'ZYmin',             'zymin')
	yield WindowToken(0x6315, b'ZYmax',             'zymax')
	yield WindowToken(0x6316, b'Z\x5bmin',          'ztheta_min')        # Zθmin
	yield WindowToken(0x6317, b'Z\x5bmax',          'ztheta_max')        # Zθmax
	yield WindowToken(0x6318, b'ZTmin',             'ztmin')
	yield WindowToken(0x6319, b'ZTmax',             'ztmax')
	yield TableToken(0x631A, b'TblStart',           'tbl_start')
	yield WindowToken(0x631B, b'PlotStart',         'plot_start')
	yield WindowToken(0x631C, b'ZPlotStart',        'zplot_start')
	yield IntWindowToken(0x631D, b'nMax',           'n_max')
	yield IntWindowToken(0x631E, b'ZnMax',          'zn_max')
	yield IntWindowToken(0x631F, b'nMin',           'n_min')
	yield IntWindowToken(0x6320, b'ZnMin',          'zn_min')
	yield TableToken(0x6321, b'\xbeTbl',            'delta_tbl')         # ΔTbl
	yield WindowToken(0x6322, b'Tstep',             'tstep')
	yield WindowToken(0x6323, b'\x5bstep',          'theta_step')        # θstep
	yield WindowToken(0x6324, b'ZTstep',            'ztstep')
	yield WindowToken(0x6325, b'Z\x5bstep',         'ztheta_step')       # Zθstep
	yield DeltaWindowToken(0x6326, b'\xbeX', 'xmin', 'xmax', 94)         # ΔX
	yield DeltaWindowToken(0x6327, b'\xbeY', 'ymin', 'ymax', 62)         # ΔY
	yield FactorWindowToken(0x6328, b'XFact',       'x_fact')
	yield FactorWindowToken(0x6329, b'YFact',       'y_fact')
	yield Token(0x632A, b'TblInput')
	yield RealToken(0x632B, b'\xdd',                'n_tvm')             # N (finance)
	yield RealToken(0x632C, b'I%',                  'i_pct')             # I%
	yield RealToken(0x632D, b'PV',                  'pv')                # PV
	yield RealToken(0x632E, b'PMT',                 'pmt')               # PMT
	yield RealToken(0x632F, b'FV',                  'fv')                # FV
	yield PaymentsPerYearToken(0x6330, b'P/Y',      'py')                # P/Y → also copies to C/Y
	yield RealToken(0x6331, b'C/Y',                 'cy')                # C/Y (independent)
	yield SequenceInitialToken(0x6332, b'\x04(nMin)', 2)                 # 𝑤(nMin)
	yield SequenceInitialToken(0x6333, b'Z\x04(nMin)', 2)                # Z𝑤(nMin)
	yield WindowToken(0x6334, b'PlotStep',          'plot_step')
	yield WindowToken(0x6335, b'ZPlotStep',         'zplot_step')
	yield XresToken(0x6336, b'Xres',                'xres')
	yield XresToken(0x6337, b'ZXres',               'zxres')
	yield Token(0x6338, b'TraceStep')

	yield CommandToken(0x64, b'Radian',             modecmds.radian)
	yield CommandToken(0x65, b'Degree',             modecmds.degree)
	yield CommandToken(0x66, b'Normal',             modecmds.normal)
	yield CommandToken(0x67, b'Sci',                modecmds.sci)
	yield CommandToken(0x68, b'Eng',                modecmds.eng)
	yield CommandToken(0x69, b'Float',              modecmds.float_)
	yield OperatorToken(0x6A, b'=',                 ops.eq,    (40, 41), char='=')
	yield OperatorToken(0x6B, b'<',                 ops.lt,    (40, 41), char='<')
	yield OperatorToken(0x6C, b'>',                 ops.gt,    (40, 41), char='>')
	yield OperatorToken(0x6D, b'\x17',              ops.le,    (40, 41), char='≤')
	yield OperatorToken(0x6E, b'\x19',              ops.ge,    (40, 41), char='≥')
	yield OperatorToken(0x6F, b'\x18',              ops.ne,    (40, 41), char='≠')
	yield OperatorToken(0x70, b'+',                 ops.add,   (50, 51), char='+')
	yield OperatorToken(0x71, b'-',                 ops.sub,   (50, 51), char='-')
	yield AnsToken(tok.ANS, b'Ans', Flag.ATOM_START)
	yield CommandToken(0x73, b'Fix',                modecmds.fix)
	yield Token(0x74, b'Horiz')
	yield Token(0x75, b'Full')
	yield CommandToken(0x76, b'Func',               modecmds.func)
	yield CommandToken(0x77, b'Param',              modecmds.param)
	yield CommandToken(0x78, b'Polar',              modecmds.polar)
	yield CommandToken(0x79, b'Seq',                modecmds.seq)
	yield Token(0x7A, b'IndpntAuto')
	yield Token(0x7B, b'IndpntAsk')
	yield Token(0x7C, b'DependAuto')
	yield Token(0x7D, b'DependAsk')

	yield CommandToken(0x7E00, b'Sequential',       modecmds.sequential)
	yield CommandToken(0x7E01, b'Simul',            modecmds.simul)
	yield CommandToken(0x7E02, b'PolarGC',          modecmds.polar_gc)
	yield CommandToken(0x7E03, b'RectGC',           modecmds.rect_gc)
	yield CommandToken(0x7E04, b'CoordOn',          modecmds.coord_on)
	yield CommandToken(0x7E05, b'CoordOff',         modecmds.coord_off)
	yield CommandToken(0x7E06, b'Connected',        modecmds.connected)
	yield CommandToken(0x7E07, b'Dot',              modecmds.dot)
	yield CommandToken(0x7E08, b'AxesOn',           modecmds.axes_on)
	yield CommandToken(0x7E09, b'AxesOff',          modecmds.axes_off)
	yield CommandToken(0x7E0A, b'GridOn',           modecmds.grid_on)
	yield CommandToken(0x7E0B, b'GridOff',          modecmds.grid_off)
	yield CommandToken(0x7E0C, b'LabelOn',          modecmds.label_on)
	yield CommandToken(0x7E0D, b'LabelOff',         modecmds.label_off)
	yield Token(0x7E0E, b'Web')
	yield Token(0x7E0F, b'Time')
	yield Token(0x7E10, b'uvAxes')
	yield Token(0x7E11, b'vwAxes')
	yield Token(0x7E12, b'uwAxes')

	yield Token(0x7F, b'\x0a')                                           # ▫
	yield Token(0x80, b'\x0b')                                           # ﹢
	yield Token(0x81, b'\x0c')                                           # ·
	yield OperatorToken(0x82, b'*',                 ops.mul,   (60, 61), char='*')
	yield OperatorToken(0x83, b'/',                 ops.div,   (60, 61), char='/')
	yield Token(0x84, b'Trace')
	yield CommandToken(0x85, b'ClrDraw',            draw.clr_draw)
	yield CommandToken(0x86, b'ZStandard',          zoom.z_standard)
	yield CommandToken(0x87, b'ZTrig',              zoom.z_trig)
	yield Token(0x88, b'ZBox')
	yield CommandToken(0x89, b'Zoom In',            zoom.zoom_in)
	yield CommandToken(0x8A, b'Zoom Out',           zoom.zoom_out)
	yield CommandToken(0x8B, b'ZSquare',            zoom.z_square)
	yield CommandToken(0x8C, b'ZInteger',           zoom.z_integer)
	yield Token(0x8D, b'ZPrevious')
	yield CommandToken(0x8E, b'ZDecimal',           zoom.z_decimal)
	yield Token(0x8F, b'ZoomStat')
	yield CommandToken(0x90, b'ZoomRcl',            zoom.zoom_rcl)
	yield Token(0x91, b'PrintScreen')
	yield CommandToken(0x92, b'ZoomSto',            zoom.zoom_sto)
	yield CommandToken(0x93, b'Text(',              draw.text)
	yield OperatorToken(0x94, b'nPr',               ops.npr,   (60, 61))
	yield OperatorToken(0x95, b'nCr',               ops.ncr,   (60, 61))
	yield CommandToken(0x96, b'FnOn ',              modecmds.fn_on)
	yield CommandToken(0x97, b'FnOff ',             modecmds.fn_off)
	yield CommandToken(0x98, b'StorePic ',          draw.store_pic)
	yield CommandToken(0x99, b'RecallPic ',         draw.recall_pic)
	yield Token(0x9A, b'StoreGDB ')
	yield Token(0x9B, b'RecallGDB ')
	yield CommandToken(0x9C, b'Line(',              draw.line)
	yield CommandToken(0x9D, b'Vertical ',          draw.vertical)
	yield CommandToken(0x9E, b'Pt-On(',             draw.pt_on)
	yield CommandToken(0x9F, b'Pt-Off(',            draw.pt_off)
	yield CommandToken(0xA0, b'Pt-Change(',         draw.pt_change)
	yield CommandToken(0xA1, b'Pxl-On(',            draw.pxl_on)
	yield CommandToken(0xA2, b'Pxl-Off(',           draw.pxl_off)
	yield CommandToken(0xA3, b'Pxl-Change(',        draw.pxl_change)
	yield CommandToken(0xA4, b'Shade(',             draw.shade)
	yield CommandToken(0xA5, b'Circle(',            draw.circle)
	yield CommandToken(0xA6, b'Horizontal ',        draw.horizontal)
	yield CommandToken(0xA7, b'Tangent(',           draw.tangent)
	yield CommandToken(0xA8, b'DrawInv ',           draw.draw_inv)
	yield CommandToken(0xA9, b'DrawF ',             draw.draw_f)

	for i in range(10):
		yield StringToken(i)

	yield timath.RandToken(tok.RAND, b'rand')
	yield ConstantToken(0xAC, b'\xc4',              math.pi)  # π
	yield ComputedToken(0xAD, b'getKey',            Environment.get_key)
	yield Token(tok.APOS, b"'",                     char="'")
	yield Token(0xAF, b'?',                         char='?')
	yield Token(tok.NEG, b'\x1a', Flag.ATOM_START)  # negative sign
	yield FunctionToken(0xB1, b'int(',              timath.int_)
	yield FunctionToken(0xB2, b'abs(',              timath.abs)
	yield FunctionToken(0xB3, b'det(',              matrix.det)
	yield FunctionToken(0xB4, b'identity(',         matrix.identity)
	yield FunctionToken(tok.DIM, b'dim(',           tilist.dim)
	yield FunctionToken(0xB6, b'sum(',              tilist.sum)
	yield FunctionToken(0xB7, b'prod(',             tilist.prod)
	yield FunctionToken(0xB8, b'not(',              timath.not_)
	yield FunctionToken(0xB9, b'iPart(',            timath.i_part)
	yield FunctionToken(0xBA, b'fPart(',            timath.f_part)

	yield FunctionToken(0xBB00, b'npv(',            finance.npv)
	yield FunctionToken(0xBB01, b'irr(',            finance.irr)
	yield FunctionToken(0xBB02, b'bal(',            finance.bal)
	yield FunctionToken(0xBB03, b'\xc6prn(',        finance.sigma_prn)   # Σprn(
	yield FunctionToken(0xBB04, b'\xc6Int(',        finance.sigma_int)   # ΣInt(
	yield FunctionToken(0xBB05, b'\x05Nom(',        finance.nom)         # ►Nom(
	yield FunctionToken(0xBB06, b'\x05Eff(',        finance.eff)         # ►Eff(
	yield FunctionToken(0xBB07, b'dbd(',            finance.dbd)
	yield FunctionToken(0xBB08, b'lcm(',            timath.lcm)
	yield FunctionToken(0xBB09, b'gcd(',            timath.gcd)
	yield FunctionToken(0xBB0A, b'randInt(',        timath.rand_int)
	yield FunctionToken(0xBB0B, b'randBin(',        timath.rand_bin)
	yield FunctionToken(0xBB0C, b'sub(',            tistring.sub)
	yield FunctionToken(0xBB0D, b'stdDev(',         tilist.stddev)
	yield FunctionToken(0xBB0E, b'variance(',       tilist.variance)
	yield FunctionToken(0xBB0F, b'inString(',       tistring.in_string)
	yield FunctionToken(0xBB10, b'normalcdf(',      dist.normalcdf)
	yield FunctionToken(0xBB11, b'invNorm(',        dist.inv_norm)
	yield FunctionToken(0xBB12, b'tcdf(',           dist.tcdf)
	yield FunctionToken(0xBB13, b'\xd9\x12cdf(',    dist.chi_sq_cdf)    # χ²cdf(
	yield FunctionToken(0xBB14, b'Fcdf(',           dist.fcdf)
	yield FunctionToken(0xBB15, b'binompdf(',       dist.binompdf)
	yield FunctionToken(0xBB16, b'binomcdf(',       dist.binomcdf)
	yield FunctionToken(0xBB17, b'poissonpdf(',     dist.poissonpdf)
	yield FunctionToken(0xBB18, b'poissoncdf(',     dist.poissoncdf)
	yield FunctionToken(0xBB19, b'geometpdf(',      dist.geometpdf)
	yield FunctionToken(0xBB1A, b'geometcdf(',      dist.geometcdf)
	yield FunctionToken(0xBB1B, b'normalpdf(',      dist.normalpdf)
	yield FunctionToken(0xBB1C, b'tpdf(',           dist.tpdf)
	yield FunctionToken(0xBB1D, b'\xd9\x12pdf(',    dist.chi_sq_pdf)    # χ²pdf(
	yield FunctionToken(0xBB1E, b'Fpdf(',           dist.f_pdf)
	yield FunctionToken(0xBB1F, b'randNorm(',       timath.rand_norm)
	yield finance.TvmToken(0xBB20, b'tvm_Pmt',      finance.tvm_pmt_value,   finance.tvm_pmt)
	yield finance.TvmToken(0xBB21, b'tvm_I%',       finance.tvm_i_pct_value, finance.tvm_i_pct)
	yield finance.TvmToken(0xBB22, b'tvm_PV',       finance.tvm_pv_value,    finance.tvm_pv)
	yield finance.TvmToken(0xBB23, b'tvm_N',        finance.tvm_n_value,     finance.tvm_n)
	yield finance.TvmToken(0xBB24, b'tvm_FV',       finance.tvm_fv_value,    finance.tvm_fv)
	yield FunctionToken(0xBB25, b'conj(',           timath.conj)
	yield FunctionToken(0xBB26, b'real(',           timath.real)
	yield FunctionToken(0xBB27, b'imag(',           timath.imag)
	yield FunctionToken(0xBB28, b'angle(',          timath.angle)
	yield FunctionToken(0xBB29, b'cumSum(',         tilist.cum_sum)
	yield FunctionToken(0xBB2A, b'expr(',           tistring.expr)
	yield FunctionToken(0xBB2B, b'length(',         tistring.length)
	yield FunctionToken(0xBB2C, b'\xbeList(',       tilist.delta_list)   # ΔList(
	yield FunctionToken(0xBB2D, b'ref(',            matrix.ref)
	yield FunctionToken(0xBB2E, b'rref(',           matrix.rref)
	yield Token(0xBB2F, b'\x05Rect')   # ►Rect
	yield Token(0xBB30, b'\x05Polar')  # ►Polar
	yield ConstantToken(0xBB31, b'\xdb', math.e)  # 𝑒
	yield Token(0xBB32, b'SinReg ')
	yield Token(0xBB33, b'Logistic ')
	yield Token(0xBB34, b'LinRegTTest ')
	yield CommandToken(0xBB35, b'ShadeNorm(',       draw.shade_norm)
	yield CommandToken(0xBB36, b'Shade_t(',         draw.shade_t)
	yield CommandToken(0xBB37, b'Shade\xd9\x12(',   draw.shade_chi_sq)   # Shadeχ²(
	yield CommandToken(0xBB38, b'Shade\xda(',       draw.shade_f)        # Shade𝐅(
	yield CommandToken(0xBB39, b'Matr\x05list(',    matrix.matr_to_list) # Matr►list(
	yield CommandToken(0xBB3A, b'List\x05matr(',    matrix.list_to_matr) # List►matr(
	yield Token(0xBB3B, b'Z-Test(')
	yield Token(0xBB3C, b'T-Test')
	yield Token(0xBB3D, b'2-SampZTest(')
	yield Token(0xBB3E, b'1-PropZTest(')
	yield Token(0xBB3F, b'2-PropZTest(')
	yield Token(0xBB40, b'\xd9\x12-Test(')  # χ²-Test(
	yield Token(0xBB41, b'ZInterval ')
	yield Token(0xBB42, b'2-SampZInt(')
	yield Token(0xBB43, b'1-PropZInt(')
	yield Token(0xBB44, b'2-PropZInt(')
	yield Token(0xBB45, b'GraphStyle(')
	yield Token(0xBB46, b'2-SampTTest ')
	yield Token(0xBB47, b'2-SampFTest ')
	yield Token(0xBB48, b'TInterval ')
	yield Token(0xBB49, b'2-SampTInt ')
	yield CommandToken(0xBB4A, b'SetUpEditor ',     tilist.set_up_editor)
	yield Token(0xBB4B, b'Pmt_End')
	yield Token(0xBB4C, b'Pmt_Bgn')
	yield CommandToken(0xBB4D, b'Real',             modecmds.real)
	yield CommandToken(0xBB4E, b're^\x5bi',         modecmds.re_theta_i) # re^θi
	yield CommandToken(0xBB4F, b'a+bi',             modecmds.a_plus_bi)
	yield CommandToken(0xBB50, b'ExprOn',           modecmds.expr_on)
	yield CommandToken(0xBB51, b'ExprOff',          modecmds.expr_off)
	yield CommandToken(0xBB52, b'ClrAllLists',      tilist.clr_all_lists)
	yield Token(0xBB53, b'GetCalc(')
	yield CommandToken(0xBB54, b'DelVar ',          prgm.del_var)
	yield CommandToken(0xBB55, b'Equ\x05String(',   tistring.equ_to_string)  # Equ►String(
	yield CommandToken(0xBB56, b'String\x05Equ(',   tistring.string_to_equ)  # String►Equ(
	yield Token(0xBB57, b'Clear Entries')
	yield Token(0xBB58, b'Select(')
	yield Token(0xBB59, b'ANOVA(')
	yield Token(0xBB5A, b'ModBoxplot')
	yield Token(0xBB5B, b'NormProbPlot')
	yield Token(0xBB64, b'G-T')
	yield CommandToken(0xBB65, b'ZoomFit',          zoom.zoom_fit)
	yield CommandToken(0xBB66, b'DiagnosticOn',     modecmds.diagnostic_on)
	yield CommandToken(0xBB67, b'DiagnosticOff',    modecmds.diagnostic_off)
	yield Token(0xBB68, b'Archive ')
	yield Token(0xBB69, b'UnArchive ')
	yield Token(0xBB6A, b'Asm(')
	yield Token(0xBB6B, b'AsmComp(')
	yield Token(0xBB6C, b'AsmPrgm')
	yield Token(0xBB6D, b'?')  # compiled asm token
	yield Token(0xBB6E, b'\x8a', char='Á')
	yield Token(0xBB6F, b'\x8b', char='À')
	yield Token(0xBB70, b'\x8c', char='Â')
	yield Token(0xBB71, b'\x8d', char='Ä')
	yield Token(0xBB72, b'\x8e', char='á')
	yield Token(0xBB73, b'\x8f', char='à')
	yield Token(0xBB74, b'\x90', char='â')
	yield Token(0xBB75, b'\x91', char='ä')
	yield Token(0xBB76, b'\x92', char='É')
	yield Token(0xBB77, b'\x93', char='È')
	yield Token(0xBB78, b'\x94', char='Ê')
	yield Token(0xBB79, b'\x95', char='Ë')
	yield Token(0xBB7A, b'\x96', char='é')
	yield Token(0xBB7B, b'\x97', char='è')
	yield Token(0xBB7C, b'\x98', char='ê')
	yield Token(0xBB7D, b'\x99', char='ë')
	yield Token(0xBB7F, b'\x9b', char='Ì')
	yield Token(0xBB80, b'\x9c', char='Î')
	yield Token(0xBB81, b'\x9d', char='Ï')
	yield Token(0xBB82, b'\x9e', char='í')
	yield Token(0xBB83, b'\x9f', char='ì')
	yield Token(0xBB84, b'\xa0', char='î')
	yield Token(0xBB85, b'\xa1', char='ï')
	yield Token(0xBB86, b'\xa2', char='Ó')
	yield Token(0xBB87, b'\xa3', char='Ò')
	yield Token(0xBB88, b'\xa4', char='Ô')
	yield Token(0xBB89, b'\xa5', char='Ö')
	yield Token(0xBB8A, b'\xa6', char='ó')
	yield Token(0xBB8B, b'\xa7', char='ò')
	yield Token(0xBB8C, b'\xa8', char='ô')
	yield Token(0xBB8D, b'\xa9', char='ö')
	yield Token(0xBB8E, b'\xaa', char='Ú')
	yield Token(0xBB8F, b'\xab', char='Ù')
	yield Token(0xBB90, b'\xac', char='Û')
	yield Token(0xBB91, b'\xad', char='Ü')
	yield Token(0xBB92, b'\xae', char='ú')
	yield Token(0xBB93, b'\xaf', char='ù')
	yield Token(0xBB94, b'\xb0', char='û')
	yield Token(0xBB95, b'\xb1', char='ü')
	yield Token(0xBB96, b'\xb2', char='Ç')
	yield Token(0xBB97, b'\xb3', char='ç')
	yield Token(0xBB98, b'\xb4', char='Ñ')
	yield Token(0xBB99, b'\xb5', char='ñ')
	yield Token(0xBB9A, b'\xb6')  # acute accent
	yield Token(0xBB9B, b'\xb7')  # grave accent
	yield Token(0xBB9C, b'\xb8')  # diaresis
	yield Token(0xBB9D, b'\xb9', char='¿')
	yield Token(0xBB9E, b'\xba', char='¡')
	yield Token(0xBB9F, b'\xbb', char='α')
	yield Token(0xBBA0, b'\xbc', char='β')
	yield Token(0xBBA1, b'\xbd', char='γ')
	yield Token(0xBBA2, b'\xbe', char='Δ')
	yield Token(0xBBA3, b'\xbf', char='δ')
	yield Token(0xBBA4, b'\xc0', char='ε')
	yield Token(0xBBA5, b'\xc2', char='λ')
	yield Token(0xBBA6, b'\xc3', char='μ')
	yield Token(0xBBA7, b'\xc4')  # π homoglyph
	yield Token(0xBBA8, b'\xc5', char='ρ')
	yield Token(0xBBA9, b'\xc6', char='Σ')
	yield Token(0xBBAB, b'\xc9', char='φ')
	yield Token(0xBBAC, b'\xca', char='Ω')
	yield Token(0xBBAD, b'\xd8')  # 0xD8 homoglyph
	yield Token(0xBBAE, b'\xd9', char='χ')
	yield Token(0xBBAF, b'\x0f')
	yield Token(0xBBB0, b'a', char='a')
	yield Token(0xBBB1, b'b', char='b')
	yield Token(0xBBB2, b'c', char='c')
	yield Token(0xBBB3, b'd', char='d')
	yield Token(0xBBB4, b'e', char='e')
	yield Token(0xBBB5, b'f', char='f')
	yield Token(0xBBB6, b'g', char='g')
	yield Token(0xBBB7, b'h', char='h')
	yield Token(0xBBB8, b'i', char='i')
	yield Token(0xBBB9, b'j', char='j')
	yield Token(0xBBBA, b'k', char='k')
	yield Token(0xBBBC, b'l', char='l')
	yield Token(0xBBBD, b'm', char='m')
	yield Token(0xBBBE, b'n', char='n')
	yield Token(0xBBBF, b'o', char='o')
	yield Token(0xBBC0, b'p', char='p')
	yield Token(0xBBC1, b'q', char='q')
	yield Token(0xBBC2, b'r', char='r')
	yield Token(0xBBC3, b's', char='s')
	yield Token(0xBBC4, b't', char='t')
	yield Token(0xBBC5, b'u', char='u')
	yield Token(0xBBC6, b'v', char='v')
	yield Token(0xBBC7, b'w', char='w')
	yield Token(0xBBC8, b'x', char='x')
	yield Token(0xBBC9, b'y', char='y')
	yield Token(0xBBCA, b'z', char='z')
	yield Token(0xBBCB, b'\xc7', char='σ')
	yield Token(0xBBCC, b'\xc8', char='τ')
	yield Token(0xBBCD, b'\x9a', char='Í')
	yield Token(0xBBCE, b'GarbageCollect')
	yield Token(0xBBCF, b'~', char='~')
	yield Token(0xBBD1, b'@', char='@')
	yield Token(0xBBD2, b'#', char='#')
	yield Token(0xBBD3, b'\xf2', char='$')
	yield Token(0xBBD4, b'&', char='&')
	yield Token(0xBBD5, b'`', char='`')
	yield Token(0xBBD6, b';', char=';')
	yield Token(0xBBD7, b'\\', char='\\')
	yield Token(0xBBD8, b'|', char='|')
	yield Token(0xBBD9, b'_', char='_')
	yield PostfixToken(0xBBDA, b'%', ops.percent, char='%')
	yield Token(0xBBDB, b'\xce')  # …
	yield Token(0xBBDC, b'\x13')  # ∠
	yield Token(0xBBDD, b'\xf4')  # ß
	yield Token(0xBBDE, b'\xcd')  # ˣ
	yield Token(0xBBDF, b'\x0d')  # ₜ
	yield Token(0xBBE0, b'\x80')  # ₀
	yield Token(0xBBE1, b'\x81')  # ₁
	yield Token(0xBBE2, b'\x82')  # ₂
	yield Token(0xBBE3, b'\x83')  # ₃
	yield Token(0xBBE4, b'\x84')  # ₄
	yield Token(0xBBE5, b'\x85')  # ₅
	yield Token(0xBBE6, b'\x86')  # ₆
	yield Token(0xBBE7, b'\x87')  # ₇
	yield Token(0xBBE8, b'\x88')  # ₈
	yield Token(0xBBE9, b'\x89')  # ₉
	yield Token(0xBBEA, b'\x1d')  # ⑽
	yield Token(0xBBEB, b'\xcf')  # ◄
	yield Token(0xBBEC, b'\x05')  # ►
	yield Token(0xBBED, b'\x1e')  # ↑
	yield Token(0xBBEE, b'\x1f')  # ↓
	yield Token(0xBBF0, b'\x09')  # ×
	yield Token(0xBBF1, b'\x08')  # ∫
	yield Token(0xBBF2, b'\x06')  # 🡅
	yield Token(0xBBF3, b'\x07')  # 🡇
	yield Token(0xBBF4, b'\x10')  # √
	yield Token(0xBBF5, b'\x7f')  # ≛

	yield FunctionToken(0xBC, b'\x10(',             timath.sqrt)         # √(
	yield FunctionToken(0xBD, b'\x0e\x10(',         timath.cbrt)         # 𝟑√(
	yield FunctionToken(0xBE, b'ln(',               timath.ln)
	yield FunctionToken(0xBF, b'\xdb^(',            timath.exp)          # 𝑒^(
	yield FunctionToken(0xC0, b'log(',              timath.log)
	yield FunctionToken(0xC1, b'\x1d^(',            timath.pow10)        # ⑽^(
	yield FunctionToken(0xC2, b'sin(',              timath.sin)
	yield FunctionToken(0xC3, b'sin\x11(',          timath.asin)         # sin⁻¹(
	yield FunctionToken(0xC4, b'cos(',              timath.cos)
	yield FunctionToken(0xC5, b'cos\x11(',          timath.acos)         # cos⁻¹(
	yield FunctionToken(0xC6, b'tan(',              timath.tan)
	yield FunctionToken(0xC7, b'tan\x11(',          timath.atan)         # tan⁻¹(
	yield FunctionToken(0xC8, b'sinh(',             timath.sinh)
	yield FunctionToken(0xC9, b'sinh\x11(',         timath.asinh)        # sinh⁻¹(
	yield FunctionToken(0xCA, b'cosh(',             timath.cosh)
	yield FunctionToken(0xCB, b'cosh\x11(',         timath.acosh)        # cosh⁻¹(
	yield FunctionToken(0xCC, b'tanh(',             timath.tanh)
	yield FunctionToken(0xCD, b'tanh\x11(',         timath.atanh)        # tanh⁻¹(
	yield CommandToken(tok.IF, b'If ',              prgm.if_cmd)
	yield CommandToken(tok.THEN, b'Then',           prgm.then_cmd)
	yield CommandToken(tok.ELSE, b'Else',           prgm.else_cmd)
	yield CommandToken(tok.WHILE, b'While ',        prgm.while_cmd)
	yield CommandToken(tok.REPEAT, b'Repeat ',      prgm.repeat_cmd)
	yield CommandToken(tok.FOR, b'For(',            prgm.for_cmd)
	yield CommandToken(tok.END, b'End',             prgm.end_cmd)
	yield CommandToken(0xD5, b'Return',             prgm.return_cmd)
	yield CommandToken(tok.LBL, b'Lbl ',            prgm.lbl_cmd)
	yield CommandToken(0xD7, b'Goto ',              prgm.goto_cmd)
	yield CommandToken(0xD8, b'Pause ',             prgm.pause_cmd)
	yield CommandToken(0xD9, b'Stop',               prgm.stop_cmd)
	yield CommandToken(0xDA, b'IS>(',               prgm.is_gt_cmd)
	yield CommandToken(0xDB, b'DS<(',               prgm.ds_lt_cmd)
	yield CommandToken(0xDC, b'Input ',             prgm.input_cmd)
	yield CommandToken(0xDD, b'Prompt ',            prgm.prompt_cmd)
	yield CommandToken(0xDE, b'Disp ',              prgm.disp)
	yield CommandToken(0xDF, b'DispGraph',          prgm.disp_graph)
	yield CommandToken(0xE0, b'Output(',            prgm.output)
	yield CommandToken(0xE1, b'ClrHome',            prgm.clr_home)
	yield CommandToken(0xE2, b'Fill(',              tilist.fill)
	yield CommandToken(0xE3, b'SortA(',             tilist.sort_a)
	yield CommandToken(0xE4, b'SortD(',             tilist.sort_d)
	yield CommandToken(0xE5, b'DispTable',          prgm.disp_table)
	yield CommandToken(0xE6, b'Menu(',              prgm.menu_cmd)
	yield Token(0xE7, b'Send(')
	yield Token(0xE8, b'Get(')
	yield CommandToken(0xE9, b'PlotsOn',            _PLACEHOLDER)
	yield CommandToken(0xEA, b'PlotsOff',           _PLACEHOLDER)
	yield Token(tok.LIST_PREFIX, b'\xdc', Flag.ATOM_START)
	yield Token(0xEC, b'Plot1(')
	yield Token(0xED, b'Plot2(')
	yield Token(0xEE, b'Plot3(')

	yield CommandToken(0xEF00, b'setDate(',         titime.set_date)
	yield CommandToken(0xEF01, b'setTime(',         titime.set_time)
	yield FunctionToken(0xEF02, b'checkTmr(',       titime.check_tmr)
	yield CommandToken(0xEF03, b'setDtFmt(',        titime.set_dt_fmt)
	yield CommandToken(0xEF04, b'setTmFmt(',        titime.set_tm_fmt)
	yield FunctionToken(0xEF05, b'timeCnv(',        titime.time_cnv)
	yield FunctionToken(0xEF06, b'dayOfWk(',        titime.dayofwk)
	yield FunctionToken(0xEF07, b'getDtStr(',       titime.get_dt_str)
	yield FunctionToken(0xEF08, b'getTmStr(',       titime.get_tm_str)
	yield ComputedToken(0xEF09, b'getDate',         Environment.get_date)
	yield ComputedToken(0xEF0A, b'getTime',         Environment.get_time)
	yield ComputedToken(0xEF0B, b'startTmr',        Environment.start_tmr)
	yield ComputedToken(0xEF0C, b'getDtFmt',        Environment.get_dt_fmt)
	yield ComputedToken(0xEF0D, b'getTmFmt',        Environment.get_tm_fmt)
	yield ComputedToken(0xEF0E, b'isClockOn',       Environment.is_clock_on)
	yield CommandToken(0xEF0F, b'ClockOff',         modecmds.clock_off)
	yield CommandToken(0xEF10, b'ClockOn',          modecmds.clock_on)
	yield Token(0xEF11, b'OpenLib(')
	yield Token(0xEF12, b'ExecLib')
	yield FunctionToken(0xEF13, b'invT(',           dist.inv_t)
	yield Token(0xEF14, b'\xd9\x12GOF-Test(')                            # χ²GOF-Test(
	yield Token(0xEF15, b'LinRegTInt ')
	yield Token(0xEF16, b'Manual-Fit ')
	yield Token(0xEF17, b'ZQuadrant1')
	yield Token(0xEF18, b'ZFrac1/2')
	yield Token(0xEF19, b'ZFrac1/3')
	yield Token(0xEF1A, b'ZFrac1/4')
	yield Token(0xEF1B, b'ZFrac1/5')
	yield Token(0xEF1C, b'ZFrac1/8')
	yield Token(0xEF1D, b'ZFrac1/10')
	yield Token(0xEF1E, b'mathprintbox')
	yield Token(0xEF30, b'\x05n/d\xcf\x05Un/d')                          # ►n/d◄►Un/d
	yield Token(0xEF31, b'\x05F\xcf\x05D')                               # ►F◄►D
	yield FunctionToken(0xEF32, b'remainder(',      timath.remainder)
	yield FunctionToken(0xEF33, b'\xc6(',           timath.sigma)         # Σ(
	yield FunctionToken(0xEF34, b'logBASE(',        timath.log_base)
	yield FunctionToken(0xEF35, b'randIntNoRep(',   timath.rand_int_no_rep)
	yield Token(0xEF36, b'MATHPRINT')
	yield Token(0xEF37, b'CLASSIC')
	yield Token(0xEF38, b'n/d')
	yield Token(0xEF39, b'Un/d')
	yield Token(0xEF3A, b'AUTO')
	yield Token(0xEF3B, b'DEC')
	yield Token(0xEF3C, b'FRAC')
	yield Token(0xEF3D, b'FRAC-APPROX')

	yield OperatorToken(0xF0, b'^',                 ops.power,    (70, 70), char='^')
	yield OperatorToken(0xF1, b'\xcd\x10',          ops.xth_root, (60, 61))  # ˣ√
	yield Token(0xF2, b'1-Var Stats ')
	yield Token(0xF3, b'2-Var Stats ')
	yield Token(0xF4, b'LinReg(a+bx) ')
	yield Token(0xF5, b'ExpReg ')
	yield Token(0xF6, b'LnReg ')
	yield Token(0xF7, b'PwrReg ')
	yield Token(0xF8, b'Med-Med ')
	yield Token(0xF9, b'QuadReg ')
	yield CommandToken(0xFA, b'ClrList ',           tilist.clr_list)
	yield Token(0xFB, b'ClrTable')
	yield Token(0xFC, b'Histogram')
	yield Token(0xFD, b'xyLine')
	yield Token(0xFE, b'Scatter')
	yield Token(0xFF, b'LinReg(ax+b) ')


# TOKEN TABLES

ALL_TOKENS: list[Token] = []
CHAR_TABLE: dict[str, Token] = {}
_TABLE: list[Token | list[Token | None] | None] = [None] * 256


def _build():
	for token in _generate():
		ALL_TOKENS.append(token)
		if token.char:
			CHAR_TABLE[token.char] = token
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


_build()


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


def read_token(f: BytesIO) -> Token:
	b0 = f.read(1)[0]
	code = (b0 << 8) | f.read(1)[0] if isinstance(_TABLE[b0], list) else b0
	try:
		return get_token(code)
	except KeyError:
		raise ValueError(f"Invalid token code: 0x{code:0{4 if code > 0xFF else 2}X}")


if __name__ == '__main__':
	for token in ALL_TOKENS:
		print(token)
