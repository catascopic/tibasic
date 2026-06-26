import math
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO

from environment import Environment
from parser import ArgParser
from preparse import special_func
from titoken import Token, TokenKind
from accessors import (
	Accessor, NumericVar, MatrixVar, ListVar, StringVar,
	Computed, Constant,
	WindowVar, XresVar, IntWindowVar, FactorWindowVar, DeltaWindowVar,
	EnvVar, TableVar, FuncEquationVar, ParEquationVar, PolarEquationVar, SequenceVar, SequenceInitialVar,
)
import prgmcmds as cmds
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


@special_func
def _PLACEHOLDER(args):
	while args.has_next:
		args.equation_var()
	args.end_cmd()


# ── Behavior-carrying token subclasses ──────────────────────────────────────────
# These live here, not in titoken, so they can reach the operator/command modules and
# ArgParser.  Each owns its parse hook(s) and carries only the field(s) it needs, so
# there are no nullable callable bags and no way to wire two roles onto one token.

class FunctionToken(Token):
	
	def __init__(self, code, display, func)
		super().__(code, display, Flag.FUNCTION)
		self._func = func

	def function(self, parser):
		return self._func.call_with_parser(ArgParser(parser))

	# TODO: Flag.FUNCTION signifies start-of-atom as well as opens-parentheses


class CommandToken(Token):

	def __init__(self, code, display, cmd)
		super().__(code, display, Flag.COMMAND)
		self._cmd = cmd

	def command(self, parser):
		self._cmd.call_with_parser(ArgParser(parser))
		
	# Note: Some commands use parentheses, but since they can't be nested, the parser doesn't really care.


class OperatorToken(Token):

	def __init__(self, code, display, char=None, op, bp)
		super().__(code, display, Flag.INFIX, char=char)
		self._op = op
		self._bp = bp

	@property
	def infix_bp(self):
		return self._bp[0]

	def parse_infix(self, parser, lhs):
		rhs = parser.parse_expr(self._bp[1])
		return parser.env.guard_real((lhs, rhs), self._op(lhs, rhs))


class PostfixToken(Token):

	def __init__(self, code, display, op)
		super().__(code, display, Flag.POSTFIX)
		self._op = op

	def apply_postfix(self, value):
		return self._op(value)


class Accessor(ABC, Token):

	def _get(self, env):
		raise NotImplementedError(f"{type(self).__name__} does not support _get")

	def _set(self, env, value):
		raise NotImplementedError(f"{type(self).__name__} does not support _set")

	def resolve(self, env):
		value = self._get(env)
		if value is None:
			raise UndefinedError(f"{self} is not defined")
		return value

	def store(self, env, value):
		raise TiSyntaxError(f"Cannot store to {self}")

	def delete(self, env):
		raise TiSyntaxError(f"Cannot delete {self}")

	def reference(self, env) -> Reference:
		return Reference(env, self)


class LetterToken(Deletable, Accessor):

	def __init__(self, index):
		ch = 0x41 + index
		super().__init__(ch, bytes([ch]), TokenKind.NUMERIC, char=chr(ch) if ch != 0x91 else 'θ'):
		self.index = index

	def _get(self, env):
		return env.numerics[self.index]

	def _set(self, env, value):
		return env.numerics[self.index] = value

	def resolve(self, env):
		value = self._get(env)
		if value is None:
			value = 0.0
			self._set(env, value)
		return value

	def store(self, env, value):
		self._set(env, require_num(value))


class MatrixToken(Deletable, Accessor):

	def __init__(self, index):
		super().__init__(0x5C00 | index, bytes([0xC1, 0x41 + index, 0x5D]), Flag.MATRIX | Flag.INVOKABLE):
		self._index = index

	def _get(self, env):
		return env.matrices[self._index]

	def _set(self, env, value):
		env.matrices[self._index] = value

	def store(self, env, value):
		self._set(env, require_matrix(value).copy())

	def invoke(self, arg_parser):
		row = py_int(arg_parser.expr(), InvalidDimError)
		col = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[(row, col)]


class ListToken(Deletable, Accessor):

	def __init__(self, index: int):
		super().__init__(0x5D00 | index, bytes([0x4C, 0x81 + index]), TokenKind.LIST | Flag.INVOKABLE)
		self._index = index

	def _get(self, env):
		return env.lists[self._index]

	def _set(self, env, value):
		env.lists[self._index] = value

	def resolve(self, env):
		value = super().resolve(env)
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		self._set(env, require_list(value).copy())

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]


class EquationVar(Deletable, Accessor):
	mode: GraphMode = None
	# In theory, the graph_var should be derivable from the mode?
	graph_var = None

	def _get(self, env):
		return getattr(env, self.mode)[self.index].equation

	def _set(self, env, value):
		getattr(env, self.mode)[self.index].equation = value

	def resolve(self, env):
		return super().resolve(env).eval(env)

	def invoke(self, arg_parser):
		# The '(' is already eaten: Y1(x) composes by resolving with X set to x.
		x = arg_parser.expr()
		arg_parser.end_func()
		env = arg_parser.env
		with scoped_numeric(env, self.graph_var, require_num(x)):
			return self.resolve(env)

	def store(self, env, value):
		self._set(env, _normalize_eq(value))
		getattr(env, self.mode)[self.index].selected = True
		if env.graph_mode is self.mode:
			env.graph.valid = False


class FuncEquationVar(EquationVar):
	mode = GraphMode.FUNC
	graph_var = 'X'

	def __init__(self, index):
		return super().__init__(0x5E10 + index, bytes([0x59, 0x80 + (index + 1) % 10]), Flag.EQUATION | Flag.INVOKABLE)
		self.index = index
		# TODO: how to reduce code duplication of setting index as well as EQUATION|INVOKABLE?

	# Maybe something like this instead of a constructor?
	# I'm not sure I like it...
	@staticmethod
	def _format(index):
		return 0x5E10 + index, bytes([0x59, 0x80 + (index + 1) % 10])


class RealVariable(Deletable, Accessor):
	def __init__(self, code, display, name):
		super().__init__(code, display, Flag.REAL)
		self._name = name

	def _get(self, env):
		return getattr(env, self._name)

	def _set(self, env, value):
		setattr(env, self._name, value)

	def store(self, env, value):
		self._set(env, require_real(value))


class StringToken(Deletable, Accessor):
	def __init__(self, index: int):
		super().__init__(0xAA00 | index, b'Str' + bytes([0x30 + (index + 1) % 10]))
		self.index = index

	def _get(self, env):
		return env.strings[self.index]

	def _set(self, env, value):
		env.strings[self.index] = value

	def store(self, env, value):
		self._set(env, require_string(value))


def generate():
	yield Token(0x01, b'\x05DMS')   # ►DMS
	yield Token(0x02, b'\x05Dec')   # ►Dec
	yield Token(0x03, b'\x05Frac')  # ►Frac
	yield Token(tk.STORE, b'\x1c')  # → (Store operator)
	yield Token(0x05, b'Boxplot')
	yield Token(tk.L_BRACKET, b'\xc1',            char='[')  # (θ steals this place in the displayet)
	yield Token(tk.R_BRACKET, b']',               char=']')
	yield Token(tk.L_BRACE, b'{',                 char='{')
	yield Token(tk.R_BRACE, b'}',                 char='}')
	yield Token(tk.RAD, b'\x15')                             #  postfix needs env, handled specially.
	yield Token(tk.DEG, b'\x14',                  char='°')  #  postfix also handled specially
	yield PostfixToken(0x0C, b'\x11',             ops.inv)
	yield PostfixToken(0x0D, b'\x12',             lambda x: x**2)
	yield PostfixToken(0x0E, b'\x16',             ops.transpose)
	yield PostfixToken(0x0F, b'\xd5',             lambda x: x**3)
	yield Token(tk.L_PAREN, b'(',                 char='(')
	yield Token(tk.R_PAREN, b')',                 char=')')
	yield FunctionToken(0x12, b'round(',          timath.round)
	yield FunctionToken(0x13, b'pxl-Test(',       draw.pxl_test)
	yield FunctionToken(0x14, b'augment(',        tilist.augment)
	yield FunctionToken(0x15, b'rowSwap(',        matrix.row_swap)
	yield FunctionToken(0x16, b'row+(',           matrix.row_plus)
	yield FunctionToken(0x17, b'*row(',           matrix.times_row)
	yield FunctionToken(0x18, b'*row+(',          matrix.times_row_plus)
	yield FunctionToken(0x19, b'max(',            timath.max)
	yield FunctionToken(0x1A, b'min(',            timath.min)
	yield FunctionToken(0x1B, b'R\x05Pr(',        timath.rect_to_polar_radius)  # R►Pr(
	yield FunctionToken(0x1C, b'R\x05P\x5b(',     timath.rect_to_polar_angle)   # R►Pθ(
	yield FunctionToken(0x1D, b'P\x05Rx(',        timath.polar_to_rect_x)       # P►Rx(
	yield FunctionToken(0x1E, b'P\x05Ry(',        timath.polar_to_rect_y)       # P►Ry(
	yield FunctionToken(0x1F, b'median(',         tilist.median)
	yield FunctionToken(0x20, b'randM(',          matrix.rand_m)
	yield FunctionToken(0x21, b'mean(',           tilist.mean)
	yield FunctionToken(0x22, b'solve(',          timath.solve)
	yield FunctionToken(0x23, b'seq(',            tilist.seq)
	yield FunctionToken(0x24, b'fnInt(',          timath.fn_int)
	yield FunctionToken(0x25, b'nDeriv(',         timath.n_deriv)
	yield FunctionToken(0x27, b'fMin(',           timath.f_min)
	yield FunctionToken(0x28, b'fMax(',           timath.f_max)
	yield token(0x29, b' ',                       char=' ')
	yield token(tk.QUOTE, b'"',                   char='"')
	yield token(tk.COMMA, b',',                   char=',')
	yield variable_token(0x2C, b'\xd7',           Constant(1j), char='𝑖')
	yield PostfixToken(0x2D, b'!',                ops.factorial, char='!')
	yield token(0x2E, b'CubicReg ')
	yield token(0x2F, b'QuartReg ')
	
	# 0 - 9
	for i in range(10):
		ch = 0x30 + i
		yield Token(ch, bytes([ch]), Flag.DIGIT, char=chr(ch))

	yield Token(tk.DOT, b'.', char='.')
	yield Token(tk.SCI_E, b'\x1b')  # ᴇ
	yield OperatorToken(0x3C, b' or ',            ops.or_, (20, 21))
	yield OperatorToken(0x3D, b' xor ',           ops.xor, (20, 21))
	yield Token(tk.COLON, b':', char=':')
	yield Token(tk.NEWLINE, b'\xd6', char='\n')
	yield OperatorToken(0x40, b' and ',           ops.and_, (30, 31))

	# A - Z, θ
	for i in range(27): 
		yield LetterToken(i)
	
	# [A] - [J]
	for i in range(10):
		yield MatrixToken(i)

	# L₁ - L₆
	for i in range(6):
		yield ListToken(i)

	# Y₁ - Y₀  (Function mode)
	for i in range(10):
		yield FuncEquationToken(i)

	# TODO: FIX THESE!

	# X₁ₜ/Y₁ₜ - X₆ₜ/Y₆ₜ  (Parametric mode: pairs share one index)
	for i in range(12):
		index = i // 2
		variable_token(0x5E20 + i, bytes([0x58, 0x81 + index, 0x0D]), ParEquationVar(index, half='x'), kind=TokenKind.EQUATION)
		variable_token(0x5E20 + i, bytes([0x59, 0x81 + index, 0x0D]), ParEquationVar(index, half='y'), kind=TokenKind.EQUATION)

	# r₁ - r₆  (Polar mode)
	for i in range(6):
		variable_token(0x5E40 + i, bytes([0x72, 0x81 + i]), PolarEquationVar(i), kind=TokenKind.EQUATION)

	# 𝑢, 𝑣, 𝑤  (Sequence mode)
	for i in range(3):
		variable_token(0x5E80 + i, bytes([0x02 + i]), SequenceVar(i), kind=TokenKind.SEQUENCE)


	yield CommandToken(0x5F, b'prgm', cmds.prgm)

	# Pic1 - Pic0
	for i in range(10):
		yield Token(0x6000 + i, b'Pic' + bytes([0x30 + (i + 1) % 10]), Flag.PIC)

	# GDB1 - GDB 0
	for i in range(10):
		yield Token(0x6100 | i, b'GDB' + bytes([0x30 + (i + 1) % 10]), Flag.GDB)


	yield Token(0x6201, b'RegEq')
	yield Token(0x6202, b'n')
	yield Token(0x6203, b'\xcb')       # ẍ
	yield Token(0x6204, b'\xc6x')      # Σx
	yield Token(0x6205, b'\xc6x\x12')  # Σx²
	yield Token(0x6206, b'Sx')
	yield Token(0x6207, b'\xc7x')      # σx
	yield Token(0x6208, b'minX')
	yield Token(0x6209, b'maxX')
	yield Token(0x620A, b'minY')
	yield Token(0x620B, b'maxY')
	yield Token(0x620C, b'\xcc')       # ȳ
	yield Token(0x620D, b'\xc6y')      # Σy
	yield Token(0x620E, b'\xc6y\x12')  # Σy²
	yield Token(0x620F, b'Sy')
	yield Token(0x6210, b'\xc7y')      # σy
	yield Token(0x6211, b'\xc6xy')     # Σxy
	yield Token(0x6212, b'r')
	yield Token(0x6213, b'Med')
	yield Token(0x6214, b'Q1')
	yield Token(0x6215, b'Q3')
	yield Token(0x6216, b'a')
	yield Token(0x6217, b'b')
	yield Token(0x6218, b'c')
	yield Token(0x6219, b'd')
	yield Token(0x621A, b'e')
	yield Token(0x621B, b'x\x81')     # x₁
	yield Token(0x621C, b'x\x82')     # x₂
	yield Token(0x621D, b'x\x83')     # x₃
	yield Token(0x621E, b'y\x81')     # y₁
	yield Token(0x621F, b'y\x82')     # y₂
	yield Token(0x6220, b'y\x83')     # y₃
	yield RealVariable(0x6221, b'\x01', 'n')  # 𝑛 ("recursive n" for sequence equations)
	yield Token(0x6222, b'p')
	yield Token(0x6223, b'z')
	yield Token(0x6224, b't')
	yield Token(0x6225, b'\xd9\x12')  # χ²
	yield Token(0x6226, b'\xda')      # Mathematical Bold Capital F, known as "Stat F" in the tibasicdev docs
	yield Token(0x6227, b'df')
	yield Token(0x6228, b'\xd8')      # ṕ (should be p-hat; this is an approximation)
	yield Token(0x6229, b'\xd8\x81')  # ṕ₁
	yield Token(0x622A, b'\xd8\x82')  # ṕ₂
	yield Token(0x622B, b'\xcb\x81')  # ẍ₁ (should be x-bar; this is an approximation)
	yield Token(0x622C, b'Sx\x81')    # Sx₁
	yield Token(0x622D, b'n\x81')     # n₁
	yield Token(0x622E, b'\xcb\x82')  # ẍ₂
	yield Token(0x622F, b'Sx\x82')    # Sx₂
	yield Token(0x6230, b'n\x82')     # n₂
	yield Token(0x6231, b'Sxp')
	yield Token(0x6232, b'lower')
	yield Token(0x6233, b'upper')
	yield Token(0x6234, b's')
	yield Token(0x6235, b'r\x12')     # r²
	yield Token(0x6236, b'R\x12')     # R²
	yield Token(0x6237, b'df')        # Factor df
	yield Token(0x6238, b'SS')        # Factor SS
	yield Token(0x6239, b'MS')        # Factor MS
	yield Token(0x623A, b'df')        # Error df
	yield Token(0x623B, b'SS')        # Error SS
	yield Token(0x623C, b'MS')        # Error MS
	

variable_token(0x6300, b'ZXscl', WindowVar('zxscl'))
variable_token(0x6301, b'ZYscl', WindowVar('zyscl'))
variable_token(0x6302, b'Xscl', WindowVar('xscl'))
variable_token(0x6303, b'Yscl', WindowVar('yscl'))
variable_token(0x6304, b'\x02(nMin)', SequenceInitialVar(0))   # 𝑢(nMin)
variable_token(0x6305, b'\x03(nMin)', SequenceInitialVar(1))   # 𝑣(nMin)
token(0x6306, b'\x02(n-1)')             # 𝑢(n-1)
token(0x6307, b'\x03(n-1)')             # 𝑣(n-1)
variable_token(0x6308, b'Z\x02(nMin)', SequenceInitialVar(0))  # Z𝑢(nMin) — same slot, zoom-window alias
variable_token(0x6309, b'Z\x03(nMin)', SequenceInitialVar(1))  # Z𝑣(nMin)
variable_token(0x630A, b'Xmin', WindowVar('xmin'))
variable_token(0x630B, b'Xmax', WindowVar('xmax'))
variable_token(0x630C, b'Ymin', WindowVar('ymin'))
variable_token(0x630D, b'Ymax', WindowVar('ymax'))
variable_token(0x630E, b'Tmin', WindowVar('tmin'))
variable_token(0x630F, b'Tmax', WindowVar('tmax'))
variable_token(0x6310, b'\x5bmin', WindowVar('theta_min'))    # θmin
variable_token(0x6311, b'\x5bmax', WindowVar('theta_max'))    # θmax
variable_token(0x6312, b'ZXmin', WindowVar('zxmin'))
variable_token(0x6313, b'ZXmax', WindowVar('zxmax'))
variable_token(0x6314, b'ZYmin', WindowVar('zymin'))
variable_token(0x6315, b'ZYmax', WindowVar('zymax'))
variable_token(0x6316, b'Z\x5bmin', WindowVar('ztheta_min'))   # Zθmin
variable_token(0x6317, b'Z\x5bmax', WindowVar('ztheta_max'))   # Zθmax
variable_token(0x6318, b'ZTmin', WindowVar('ztmin'))
variable_token(0x6319, b'ZTmax', WindowVar('ztmax'))
variable_token(0x631A, b'TblStart', TableVar('tbl_start'))
variable_token(0x631B, b'PlotStart', WindowVar('plot_start'))
variable_token(0x631C, b'ZPlotStart', WindowVar('zplot_start'))
variable_token(0x631D, b'nMax', IntWindowVar('n_max'))
variable_token(0x631E, b'ZnMax', IntWindowVar('zn_max'))
variable_token(0x631F, b'nMin', IntWindowVar('n_min'))
variable_token(0x6320, b'ZnMin', IntWindowVar('zn_min'))
variable_token(0x6321, b'\xbeTbl', TableVar('delta_tbl'))     # ΔTbl
variable_token(0x6322, b'Tstep', WindowVar('tstep'))
variable_token(0x6323, b'\x5bstep', WindowVar('theta_step'))   # θstep
variable_token(0x6324, b'ZTstep', WindowVar('ztstep'))
variable_token(0x6325, b'Z\x5bstep', WindowVar('ztheta_step'))  # Zθstep
variable_token(0x6326, b'\xbeX', DeltaWindowVar('xmin', 'xmax', 94))  # ΔX
variable_token(0x6327, b'\xbeY', DeltaWindowVar('ymin', 'ymax', 62))  # ΔY
variable_token(0x6328, b'XFact', FactorWindowVar('x_fact'))
variable_token(0x6329, b'YFact', FactorWindowVar('y_fact'))
token(0x632A, b'TblInput')
variable_token(0x632B, b'\xdd', EnvVar('n_tvm'))     # 𝐍
variable_token(0x632C, b'I%', EnvVar('i_pct'))
variable_token(0x632D, b'PV', EnvVar('pv'))
variable_token(0x632E, b'PMT', EnvVar('pmt'))
variable_token(0x632F, b'FV', EnvVar('fv'))
variable_token(0x6330, b'P/Y', EnvVar('py'))
variable_token(0x6331, b'C/Y', EnvVar('cy'))
variable_token(0x6332, b'\x04(nMin)', SequenceInitialVar(2))  # 𝑤(nMin)
variable_token(0x6333, b'Z\x04(nMin)', SequenceInitialVar(2))  # Z𝑤(nMin)
variable_token(0x6334, b'PlotStep', WindowVar('plot_step'))
variable_token(0x6335, b'ZPlotStep', WindowVar('zplot_step'))
variable_token(0x6336, b'Xres', XresVar('xres'))
variable_token(0x6337, b'ZXres', XresVar('zxres'))
token(0x6338, b'TraceStep')


command_token(0x64, b'Radian',                  modecmds.radian)
command_token(0x65, b'Degree',                  modecmds.degree)
command_token(0x66, b'Normal',                  modecmds.normal)
command_token(0x67, b'Sci',                     modecmds.sci)
command_token(0x68, b'Eng',                     modecmds.eng)
command_token(0x69, b'Float',                   modecmds.float_)
yield OperatorToken(0x6A, b'=',       char='=', ops.eq,  (40, 41))
yield OperatorToken(0x6B, b'<',       char='<', ops.lt,  (40, 41))
yield OperatorToken(0x6C, b'>',       char='>', ops.gt,  (40, 41))
yield OperatorToken(0x6D, b'\x17',    char='≤', ops.le,  (40, 41))
yield OperatorToken(0x6E, b'\x19',    char='≥', ops.ge,  (40, 41))
yield OperatorToken(0x6F, b'\x18',    char='≠', ops.ne,  (40, 41))
yield OperatorToken(0x70, b'+',       char='+', ops.add, (50, 51))
yield OperatorToken(0x71, b'-',       char='-', ops.sub, (50, 51))
token(tk.ANS, b'Ans')                           # Handled as special case in parser
command_token(0x73, b'Fix',                     modecmds.fix)
token(0x74, b'Horiz')
token(0x75, b'Full')
command_token(0x76, b'Func',                    modecmds.func)
command_token(0x77, b'Param',                   modecmds.param)
command_token(0x78, b'Polar',                   modecmds.polar)
command_token(0x79, b'Seq',                     modecmds.seq)
token(0x7A, b'IndpntAuto')
token(0x7B, b'IndpntAsk')
token(0x7C, b'DependAuto')
token(0x7D, b'DependAsk')


command_token(0x7E00, b'Sequential',            modecmds.sequential)
command_token(0x7E01, b'Simul',                 modecmds.simul)
command_token(0x7E02, b'PolarGC',               modecmds.polar_gc)
command_token(0x7E03, b'RectGC',                modecmds.rect_gc)
command_token(0x7E04, b'CoordOn',               modecmds.coord_on)
command_token(0x7E05, b'CoordOff',              modecmds.coord_off)
command_token(0x7E06, b'Connected',             modecmds.connected)
command_token(0x7E07, b'Dot',                   modecmds.dot)
command_token(0x7E08, b'AxesOn',                modecmds.axes_on)
command_token(0x7E09, b'AxesOff',               modecmds.axes_off)
command_token(0x7E0A, b'GridOn',                modecmds.grid_on)
command_token(0x7E0B, b'GridOff',               modecmds.grid_off)
command_token(0x7E0C, b'LabelOn',               modecmds.label_on)
command_token(0x7E0D, b'LabelOff',              modecmds.label_off)
token(0x7E0E, b'Web')
token(0x7E0F, b'Time')
token(0x7E10, b'uvAxes')
token(0x7E11, b'vwAxes')
token(0x7E12, b'uwAxes')

token(0x7F, b'\x0a')  # ▫
token(0x80, b'\x0b')  # ﹢
token(0x81, b'\x0c')  # ·
yield OperatorToken(0x82, b'*', char='*', ops.mul, (60, 61))
yield OperatorToken(0x83, b'/', char='/', ops.div, (60, 61))
token(0x84, b'Trace')
command_token(0x85, b'ClrDraw',                 draw.clr_draw)
command_token(0x86, b'ZStandard',               zoom.z_standard)
command_token(0x87, b'ZTrig',                   zoom.z_trig)
token(0x88, b'ZBox')
command_token(0x89, b'Zoom In',                 zoom.zoom_in)
command_token(0x8A, b'Zoom Out',                zoom.zoom_out)
command_token(0x8B, b'ZSquare',                 zoom.z_square)
command_token(0x8C, b'ZInteger',                zoom.z_integer)
token(0x8D, b'ZPrevious')
command_token(0x8E, b'ZDecimal',                zoom.z_decimal)
token(0x8F, b'ZoomStat')
command_token(0x90, b'ZoomRcl',                 zoom.zoom_rcl)
token(0x91, b'PrintScreen')
command_token(0x92, b'ZoomSto',                 zoom.zoom_sto)
command_token(0x93, b'Text(',                   draw.text)
yield OperatorToken(0x94, b'nPr', ops.npr, (60, 61))
yield OperatorToken(0x95, b'nCr', ops.ncr, (60, 61))
command_token(0x96, b'FnOn ',                   modecmds.fn_on)
command_token(0x97, b'FnOff ',                  modecmds.fn_off)
token(0x98, b'StorePic ')
token(0x99, b'RecallPic ')
token(0x9A, b'StoreGDB ')
token(0x9B, b'RecallGDB ')
command_token(0x9C, b'Line(',                   draw.line)
command_token(0x9D, b'Vertical ',               draw.vertical)
command_token(0x9E, b'Pt-On(',                  draw.pt_on)
command_token(0x9F, b'Pt-Off(',                 draw.pt_off)
command_token(0xA0, b'Pt-Change(',              draw.pt_change)
command_token(0xA1, b'Pxl-On(',                 draw.pxl_on)
command_token(0xA2, b'Pxl-Off(',                draw.pxl_off)
command_token(0xA3, b'Pxl-Change(',             draw.pxl_change)
command_token(0xA4, b'Shade(',                  draw.shade)
command_token(0xA5, b'Circle(',                 draw.circle)
command_token(0xA6, b'Horizontal ',             draw.horizontal)
command_token(0xA7, b'Tangent(',                draw.tangent)
command_token(0xA8, b'DrawInv ',                draw.draw_inv)
command_token(0xA9, b'DrawF ',                  draw.draw_f)

# Str1 - Str0
for i in range(10):
	yield StringToken(i)

variable_token(tk.RAND, b'rand', timath.RandAccessor())
variable_token(0xAC, b'\xc4', Constant(math.pi))  # π
variable_token(0xAD, b'getKey', Computed(Environment.get_key))
token(tk.APOS, b"'", char="'")
token(0xAF,    b'?', char='?')
token(tk.NEG, b'\x1a')  # ⁻
function_token(0xB1, b'int(',                    timath.int_)
function_token(0xB2, b'abs(',                    timath.abs)
function_token(0xB3, b'det(',                    matrix.det)
function_token(0xB4, b'identity(',               matrix.identity)
function_token(tk.DIM, b'dim(',                  tilist.dim)
function_token(0xB6, b'sum(',                    tilist.sum)
function_token(0xB7, b'prod(',                   tilist.prod)
function_token(0xB8, b'not(',                    timath.not_)
function_token(0xB9, b'iPart(',                  timath.i_part)
function_token(0xBA, b'fPart(',                  timath.f_part)


function_token(0xBB00, b'npv(',  			    finance.npv)
function_token(0xBB01, b'irr(',  			    finance.irr)
function_token(0xBB02, b'bal(',  			    finance.bal)
function_token(0xBB03, b'\xc6prn(', 			    finance.sigma_prn)  # Σprn(
function_token(0xBB04, b'\xc6Int(', 			    finance.sigma_int)  # ΣInt(
function_token(0xBB05, b'\x05Nom(', 			    finance.nom)  # ►Nom(
function_token(0xBB06, b'\x05Eff(', 			    finance.eff)  # ►Eff(
function_token(0xBB07, b'dbd(',                  finance.dbd)
function_token(0xBB08, b'lcm(',                  timath.lcm)
function_token(0xBB09, b'gcd(',                  timath.gcd)
function_token(0xBB0A, b'randInt(',              timath.rand_int)
function_token(0xBB0B, b'randBin(',              timath.rand_bin)
function_token(0xBB0C, b'sub(',                  tistring.sub)
function_token(0xBB0D, b'stdDev(',               tilist.stddev)
function_token(0xBB0E, b'variance(',             tilist.variance)
function_token(0xBB0F, b'inString(',             tistring.in_string)
function_token(0xBB10, b'normalcdf(',            dist.normalcdf)
function_token(0xBB11, b'invNorm(',              dist.inv_norm)
function_token(0xBB12, b'tcdf(',                 dist.tcdf)
function_token(0xBB13, b'\xd9\x12cdf(',          dist.chi_sq_cdf)  # χ²cdf(
function_token(0xBB14, b'Fcdf(',                 dist.fcdf)
function_token(0xBB15, b'binompdf(',             dist.binompdf)
function_token(0xBB16, b'binomcdf(',             dist.binomcdf)
function_token(0xBB17, b'poissonpdf(',           dist.poissonpdf)
function_token(0xBB18, b'poissoncdf(',           dist.poissoncdf)
function_token(0xBB19, b'geometpdf(',            dist.geometpdf)
function_token(0xBB1A, b'geometcdf(',            dist.geometcdf)
function_token(0xBB1B, b'normalpdf(',            dist.normalpdf)
function_token(0xBB1C, b'tpdf(',                 dist.tpdf)
function_token(0xBB1D, b'\xd9\x12pdf(',          dist.chi_sq_pdf)  # χ²pdf(
function_token(0xBB1E, b'Fpdf(',                 dist.f_pdf)
function_token(0xBB1F, b'randNorm(',             timath.rand_norm)
variable_token(0xBB20, b'tvm_Pmt', finance.TvmAccessor(finance.tvm_pmt_value,   finance.tvm_pmt))
variable_token(0xBB21, b'tvm_I%', finance.TvmAccessor(finance.tvm_i_pct_value, finance.tvm_i_pct))
variable_token(0xBB22, b'tvm_PV', finance.TvmAccessor(finance.tvm_pv_value,    finance.tvm_pv))
variable_token(0xBB23, b'tvm_N', finance.TvmAccessor(finance.tvm_n_value,     finance.tvm_n))
variable_token(0xBB24, b'tvm_FV', finance.TvmAccessor(finance.tvm_fv_value,    finance.tvm_fv))
function_token(0xBB25, b'conj(',                 timath.conj)
function_token(0xBB26, b'real(',                 timath.real)
function_token(0xBB27, b'imag(',                 timath.imag)
function_token(0xBB28, b'angle(',                timath.angle)
function_token(0xBB29, b'cumSum(',               tilist.cum_sum)
function_token(0xBB2A, b'expr(',                 tistring.expr)
function_token(0xBB2B, b'length(',               tistring.length)
function_token(0xBB2C, b'\xbeList(',             tilist.delta_list)  # ΔList(
function_token(0xBB2D, b'ref(',                  matrix.ref)
function_token(0xBB2E, b'rref(',                 matrix.rref)
token(0xBB2F, b'\x05Rect')  # ►Rect
token(0xBB30, b'\x05Polar')  # ►Polar
variable_token(0xBB31, b'\xdb', Constant(math.e))  # 𝑒
token(0xBB32, b'SinReg ')
token(0xBB33, b'Logistic ')
token(0xBB34, b'LinRegTTest ')
command_token(0xBB35, b'ShadeNorm(',            draw.shade_norm)
command_token(0xBB36, b'Shade_t(',              draw.shade_t)
command_token(0xBB37, b'Shade\xd9\x12(',        draw.shade_chi_sq)    # Shadeχ²(
command_token(0xBB38, b'Shade\xda(',            draw.shade_f)         # Shade𝐅(
command_token(0xBB39, b'Matr\x05list(',         matrix.matr_to_list)  # Matr►list(
command_token(0xBB3A, b'List\x05matr(',         matrix.list_to_matr)  # List►matr(
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
command_token(0xBB4A, b'SetUpEditor ',          tilist.set_up_editor)
token(0xBB4B, b'Pmt_End')
token(0xBB4C, b'Pmt_Bgn')
command_token(0xBB4D, b'Real',                  modecmds.real)
command_token(0xBB4E, b're^\x5bi',              modecmds.re_theta_i)  # re^θi
command_token(0xBB4F, b'a+bi',                  modecmds.a_plus_bi)
command_token(0xBB50, b'ExprOn',                modecmds.expr_on)
command_token(0xBB51, b'ExprOff',               modecmds.expr_off)
command_token(0xBB52, b'ClrAllLists',           tilist.clr_all_lists)
token(0xBB53, b'GetCalc(')
command_token(0xBB54, b'DelVar ',               cmds.del_var)
command_token(0xBB55, b'Equ\x05String(',        tistring.equ_to_string)  # Equ►String(
command_token(0xBB56, b'String\x05Equ(',        tistring.string_to_equ)  # String►Equ(
token(0xBB57, b'Clear Entries')
token(0xBB58, b'Select(')
token(0xBB59, b'ANOVA(')
token(0xBB5A, b'ModBoxplot')
token(0xBB5B, b'NormProbPlot')
token(0xBB64, b'G-T')
command_token(0xBB65, b'ZoomFit',               zoom.zoom_fit)
command_token(0xBB66, b'DiagnosticOn',          modecmds.diagnostic_on)
command_token(0xBB67, b'DiagnosticOff',         modecmds.diagnostic_off)
token(0xBB68, b'Archive ')
token(0xBB69, b'UnArchive ')
token(0xBB6A, b'Asm(')
token(0xBB6B, b'AsmComp(')
token(0xBB6C, b'AsmPrgm')
token(0xBB6D, b'?')  # "compiled asm" token, displays as '?'
Token(0xBB6E, b'\x8a', char='Á')
Token(0xBB6F, b'\x8b', char='À')
Token(0xBB70, b'\x8c', char='Â')
Token(0xBB71, b'\x8d', char='Ä')
Token(0xBB72, b'\x8e', char='á')
Token(0xBB73, b'\x8f', char='à')
Token(0xBB74, b'\x90', char='â')
Token(0xBB75, b'\x91', char='ä')
Token(0xBB76, b'\x92', char='É')
Token(0xBB77, b'\x93', char='È')
Token(0xBB78, b'\x94', char='Ê')
Token(0xBB79, b'\x95', char='Ë')
Token(0xBB7A, b'\x96', char='é')
Token(0xBB7B, b'\x97', char='è')
Token(0xBB7C, b'\x98', char='ê')
Token(0xBB7D, b'\x99', char='ë')
Token(0xBB7F, b'\x9b', char='Ì')
Token(0xBB80, b'\x9c', char='Î')
Token(0xBB81, b'\x9d', char='Ï')
Token(0xBB82, b'\x9e', char='í')
Token(0xBB83, b'\x9f', char='ì')
Token(0xBB84, b'\xa0', char='î')
Token(0xBB85, b'\xa1', char='ï')
Token(0xBB86, b'\xa2', char='Ó')
Token(0xBB87, b'\xa3', char='Ò')
Token(0xBB88, b'\xa4', char='Ô')
Token(0xBB89, b'\xa5', char='Ö')
Token(0xBB8A, b'\xa6', char='ó')
Token(0xBB8B, b'\xa7', char='ò')
Token(0xBB8C, b'\xa8', char='ô')
Token(0xBB8D, b'\xa9', char='ö')
Token(0xBB8E, b'\xaa', char='Ú')
Token(0xBB8F, b'\xab', char='Ù')
Token(0xBB90, b'\xac', char='Û')
Token(0xBB91, b'\xad', char='Ü')
Token(0xBB92, b'\xae', char='ú')
Token(0xBB93, b'\xaf', char='ù')
Token(0xBB94, b'\xb0', char='û')
Token(0xBB95, b'\xb1', char='ü')
Token(0xBB96, b'\xb2', char='Ç')
Token(0xBB97, b'\xb3', char='ç')
Token(0xBB98, b'\xb4', char='Ñ')
Token(0xBB99, b'\xb5', char='ñ')
Token(0xBB9A, b'\xb6')  # ´
Token(0xBB9B, b'\xb7')  # Modifier Letter Grave Accent (differentiates from backtick)
Token(0xBB9C, b'\xb8')  # ¨
Token(0xBB9D, b'\xb9', char='¿')
Token(0xBB9E, b'\xba', char='¡')
Token(0xBB9F, b'\xbb', char='α')
Token(0xBBA0, b'\xbc', char='β')
Token(0xBBA1, b'\xbd', char='γ')
Token(0xBBA2, b'\xbe', char='Δ')
Token(0xBBA3, b'\xbf', char='δ')
Token(0xBBA4, b'\xc0', char='ε')
Token(0xBBA5, b'\xc2', char='λ')
Token(0xBBA6, b'\xc3', char='μ')
Token(0xBBA7, b'\xc4')  # π (homoglyph of 0xAC without syntactic meaning)
Token(0xBBA8, b'\xc5', char='ρ')
Token(0xBBA9, b'\xc6', char='Σ')
Token(0xBBAB, b'\xc9', char='φ')
Token(0xBBAC, b'\xca', char='Ω')
Token(0xBBAD, b'\xd8')  # ṕ (homoglyph of 0x6228 without syntactic meaning)
Token(0xBBAE, b'\xd9', char='χ')
Token(0xBBAF, b'\x0f')  # Mathematical Bold Capital Digamma, known as "Hexadecimal F" in the tibasicdev docs
token(0xBBB0, b'a', char='a')
token(0xBBB1, b'b', char='b')
token(0xBBB2, b'c', char='c')
token(0xBBB3, b'd', char='d')
token(0xBBB4, b'e', char='e')
token(0xBBB5, b'f', char='f')
token(0xBBB6, b'g', char='g')
token(0xBBB7, b'h', char='h')
token(0xBBB8, b'i', char='i')
token(0xBBB9, b'j', char='j')
token(0xBBBA, b'k', char='k')
token(0xBBBC, b'l', char='l')
token(0xBBBD, b'm', char='m')
token(0xBBBE, b'n', char='n')
token(0xBBBF, b'o', char='o')
token(0xBBC0, b'p', char='p')
token(0xBBC1, b'q', char='q')
token(0xBBC2, b'r', char='r')
token(0xBBC3, b's', char='s')
token(0xBBC4, b't', char='t')
token(0xBBC5, b'u', char='u')
token(0xBBC6, b'v', char='v')
token(0xBBC7, b'w', char='w')
token(0xBBC8, b'x', char='x')
token(0xBBC9, b'y', char='y')
token(0xBBCA, b'z', char='z')
token(0xBBCB, b'\xc7', char='σ')
token(0xBBCC, b'\xc8', char='τ')
token(0xBBCD, b'\x9a', char='Í')
token(0xBBCE, b'GarbageCollect')
token(0xBBCF, b'~', char='~')
token(0xBBD1, b'@', char='@')
token(0xBBD2, b'#', char='#')
token(0xBBD3, b'\xf2', char='$')
token(0xBBD4, b'&', char='&')
token(0xBBD5, b'`', char='`')  # Backtick (confusing because the symbol is officially called Grave Accent)
token(0xBBD6, b';', char=';')
token(0xBBD7, b'\\', char='\\')
token(0xBBD8, b'|', char='|')
token(0xBBD9, b'_', char='_')
postfix_token(0xBBDA, b'%', lambda x: x / 100, char='%')
token(0xBBDB, b'\xce')  # …
token(0xBBDC, b'\x13')  # ∠
token(0xBBDD, b'\xf4')  # ß
token(0xBBDE, b'\xcd')  # ˣ
token(0xBBDF, b'\x0d')  # ₜ
token(0xBBE0, b'\x80')  # ₀
token(0xBBE1, b'\x81')  # ₁
token(0xBBE2, b'\x82')  # ₂
token(0xBBE3, b'\x83')  # ₃
token(0xBBE4, b'\x84')  # ₄
token(0xBBE5, b'\x85')  # ₅
token(0xBBE6, b'\x86')  # ₆
token(0xBBE7, b'\x87')  # ₇
token(0xBBE8, b'\x88')  # ₈
token(0xBBE9, b'\x89')  # ₉
token(0xBBEA, b'\x1d')  # ⑽
token(0xBBEB, b'\xcf')  # ◄
token(0xBBEC, b'\x05')  # ►
token(0xBBED, b'\x1e')  # ↑
token(0xBBEE, b'\x1f')  # ↓
token(0xBBF0, b'\x09')  # ×
token(0xBBF1, b'\x08')  # ∫
token(0xBBF2, b'\x06')  # 🡅
token(0xBBF3, b'\x07')  # 🡇
token(0xBBF4, b'\x10')  # √
token(0xBBF5, b'\x7f')  # ≛


function_token(0xBC, b'\x10(',                  timath.sqrt)   # √(
function_token(0xBD, b'\x0e\x10(',              timath.cbrt)   # 𝟑√(
function_token(0xBE, b'ln(',		            timath.ln)
function_token(0xBF, b'\xdb^(',		            timath.exp)    # 𝑒^(
function_token(0xC0, b'log(',		            timath.log)
function_token(0xC1, b'\x1d^(',		            timath.pow10)  # ⑽^(
function_token(0xC2, b'sin(',		            timath.sin)
function_token(0xC3, b'sin\x11(',	            timath.asin)   # sin¹(
function_token(0xC4, b'cos(',		            timath.cos)
function_token(0xC5, b'cos\x11(',	            timath.acos)   # cos¹(
function_token(0xC6, b'tan(',		            timath.tan)
function_token(0xC7, b'tan\x11(',	            timath.atan)   # tan¹(
function_token(0xC8, b'sinh(',	                timath.sinh)
function_token(0xC9, b'sinh\x11(',	            timath.asinh)  # sinh¹(
function_token(0xCA, b'cosh(',	                timath.cosh)
function_token(0xCB, b'cosh\x11(',	            timath.acosh)  # cosh¹(
function_token(0xCC, b'tanh(',	                timath.tanh)
function_token(0xCD, b'tanh\x11(',	            timath.atanh)  # tanh¹(
command_token(tk.IF, b'If ',                    cmds.if_cmd)
command_token(tk.THEN, b'Then',                 cmds.then_cmd)
command_token(tk.ELSE, b'Else',                 cmds.else_cmd)
command_token(tk.WHILE, b'While ',              cmds.while_cmd)
command_token(tk.REPEAT, b'Repeat ',            cmds.repeat_cmd)
command_token(tk.FOR, b'For(',                  cmds.for_cmd)
command_token(tk.END, b'End',                   cmds.end_cmd)
command_token(0xD5, b'Return',                  cmds.return_cmd)
command_token(tk.LBL, b'Lbl ',                  cmds.lbl_cmd)
command_token(0xD7, b'Goto ',                   cmds.goto_cmd)
command_token(0xD8, b'Pause ',                  cmds.pause_cmd)
command_token(0xD9, b'Stop',                    cmds.stop_cmd)
command_token(0xDA, b'IS>(',                    cmds.is_gt_cmd)
command_token(0xDB, b'DS<(',                    cmds.ds_lt_cmd)
command_token(0xDC, b'Input ',                  cmds.input_cmd)
command_token(0xDD, b'Prompt ',                 cmds.prompt_cmd)
command_token(0xDE, b'Disp ',                   cmds.disp)
command_token(0xDF, b'DispGraph',               cmds.disp_graph)
command_token(0xE0, b'Output(',                 cmds.output)
command_token(0xE1, b'ClrHome',                 cmds.clr_home)
command_token(0xE2, b'Fill(',                   tilist.fill)
command_token(0xE3, b'SortA(',                  tilist.sort_a)
command_token(0xE4, b'SortD(',                  tilist.sort_d)
command_token(0xE5, b'DispTable',               cmds.disp_table)
command_token(0xE6, b'Menu(',                   cmds.menu_cmd)
token(0xE7, b'Send(')
token(0xE8, b'Get(')
command_token(0xE9, b'PlotsOn', _PLACEHOLDER)
command_token(0xEA, b'PlotsOff', _PLACEHOLDER)
token(tk.LIST_PREFIX, b'\xdc')  # ᴸ
token(0xEC, b'Plot1(')
token(0xED, b'Plot2(')
token(0xEE, b'Plot3(')


command_token(0xEF00, b'setDate(',              titime.set_date)
command_token(0xEF01, b'setTime(',              titime.set_time)
function_token(0xEF02, b'checkTmr(',             titime.check_tmr)
command_token(0xEF03, b'setDtFmt(',             titime.set_dt_fmt)
command_token(0xEF04, b'setTmFmt(',             titime.set_tm_fmt)
function_token(0xEF05, b'timeCnv(',              titime.time_cnv)
function_token(0xEF06, b'dayOfWk(',              titime.dayofwk)
function_token(0xEF07, b'getDtStr(',             titime.get_dt_str)
function_token(0xEF08, b'getTmStr(',             titime.get_tm_str)
variable_token(0xEF09, b'getDate', Computed(Environment.get_date))
variable_token(0xEF0A, b'getTime', Computed(Environment.get_time))
variable_token(0xEF0B, b'startTmr', Computed(Environment.start_tmr))
variable_token(0xEF0C, b'getDtFmt', Computed(Environment.get_dt_fmt))
variable_token(0xEF0D, b'getTmFmt', Computed(Environment.get_tm_fmt))
variable_token(0xEF0E, b'isClockOn', Computed(Environment.is_clock_on))
command_token(0xEF0F, b'ClockOff',              modecmds.clock_off)
command_token(0xEF10, b'ClockOn',               modecmds.clock_on)
token(0xEF11, b'OpenLib(')
token(0xEF12, b'ExecLib')
function_token(0xEF13, b'invT(',                 dist.inv_t)
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
function_token(0xEF32, b'remainder(',            timath.remainder)
function_token(0xEF33, b'\xc6(',                 timath.sigma)  # Σ(
function_token(0xEF34, b'logBASE(',              timath.log_base)
function_token(0xEF35, b'randIntNoRep(',         timath.rand_int_no_rep)
token(0xEF36, b'MATHPRINT')
token(0xEF37, b'CLASSIC')
token(0xEF38, b'n/d')
token(0xEF39, b'Un/d')
token(0xEF3A, b'AUTO')
token(0xEF3B, b'DEC')
token(0xEF3C, b'FRAC')
token(0xEF3D, b'FRAC-APPROX')


OperatorToken(0xF0, b'^', char='^', ops.power, (70, 70))
OperatorToken(0xF1, b'\xcd\x10', ops.xth_root, (60, 61))  # ˣ√
token(0xF2, b'1-Var Stats ')
token(0xF3, b'2-Var Stats ')
token(0xF4, b'LinReg(a+bx) ')
token(0xF5, b'ExpReg ')
token(0xF6, b'LnReg ')
token(0xF7, b'PwrReg ')
token(0xF8, b'Med-Med ')
token(0xF9, b'QuadReg ')
command_token(0xFA, b'ClrList ',                tilist.clr_list)
token(0xFB, b'ClrTable')
token(0xFC, b'Histogram')
token(0xFD, b'xyLine')
token(0xFE, b'Scatter')
token(0xFF, b'LinReg(ax+b) ')


ALL_TOKENS: list[Token] = []

CHAR_TABLE: dict[str, Token] = {}

_TABLE: list[Token | list[Token | None] | None] = [None] * 256

def _init():
	for token in _generate():
		CHAR_TABLE.append(token)

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

_init()


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
	print(len(ALL_TOKENS))
