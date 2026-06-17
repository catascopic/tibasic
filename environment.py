
import math
import random
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Any, ClassVar

from core import TiList, is_complex_val
from core import Variable, NumericVariable, RealVariable, ListVariable, UserList, MatrixVariable, StringVariable, EquationVariable, require_real, py_int
from errors import TiError, DataTypeError, DomainError, IllegalNestError, InvalidCommandError, InvalidDimError, UndefinedError, NonRealAnsError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder
from graph import Graph
from homescreen import HomeScreen
from terminal import ScriptedConsole


class Environment:

	def __init__(self):
		# VARIABLES
		self.numerics   = [NumericVariable()  for _ in range(27)]  # A–Z, θ
		self.lists      = [ListVariable()     for _ in range(6)]   # L1–L6
		self.matrices   = [MatrixVariable()   for _ in range(10)]  # [A]–[J]
		self.strings    = [StringVariable()   for _ in range(10)]  # Str1–Str0
		self.function   = [EquationVariable() for _ in range(10)]  # Y1–Y0
		self.parametric = [EquationVariable() for _ in range(12)]  # X1T–Y6T
		self.polar      = [EquationVariable() for _ in range(6)]   # r1–r6
		self.sequence   = [EquationVariable() for _ in range(3)]   # u, v, w
		self.user_lists = {}
		# self.stat        = [None] * 0x3D # stat vars
		# self.window      = [None] * 0x37 # window vars
		self.n = NumericVariable()
		self.ans = 0
		# MODES
		self.angle_mode    = AngleMode.RAD
		self.number_mode   = NumberMode.NORMAL
		self.fix_digits    = None          # None = Float, 0–9 = Fix N
		self.graph_mode    = GraphMode.FUNC
		self.complex_mode  = ComplexMode.REAL
		self.draw_mode     = DrawMode.CONNECTED
		self.graph_order   = GraphOrder.SEQUENTIAL
		self.coord_on      = True
		self.polar_gc      = False
		self.axes_on       = True
		self.grid_on       = False
		self.label_on      = False
		self.expr_on       = True
		self.diagnostic_on = False
		self.dt_fmt        = 1
		self.tm_fmt        = 12
		self.clock_on      = True
		# Window / graphing variables (Xscl, Xmin, Xmax, …)
		self.window = WindowVars()
		# LCD pixel buffer (used by Pxl-/Pt-/Line/etc. drawing commands)
		self.graph = Graph()
		# Home screen (16×8 char grid) and the I/O frontend that renders it.
		self.home = HomeScreen()
		self.console = ScriptedConsole()
		# TVM finance variables (used by bal(, ΣPrn(, ΣInt(, tvm_Pmt, etc.)
		self.n_tvm = RealVariable()   # 𝐍 (number of payments)
		self.i_pct = RealVariable()   # I% (interest rate per period, as percentage)
		self.pv    = RealVariable()   # PV (present value)
		self.pmt   = RealVariable()   # PMT (payment amount)
		self.fv    = RealVariable()   # FV (future value)
		self.py    = RealVariable(1.0)  # P/Y (payments per year)
		self.cy    = RealVariable(1.0)  # C/Y (compounding periods per year)
		# Programs
		self.programs: dict[str, list] = {}  # name -> token list for stored programs
		self.program_stack: deque = deque()  # currently executing programs (innermost last)
		# Internal data
		self._datetime_offset = timedelta(0)  # virtual_time = system_time + offset
		self._nest_depth: dict[object, int] = defaultdict(lambda: 0)  # tracks nesting depth for ILLEGAL NEST guards

	# ── Graph variables ──────────────────────────────────────────────────────────
	# X and Y are ordinary numeric variables (letters 'X' and 'Y'); the graph and
	# drawing commands (DrawF, DrawInv, Tangent, …) read and update them by name.

	@property
	def x(self) -> Variable:
		return self.numerics[23]

	@property
	def y(self) -> Variable:
		return self.numerics[24]

	def run(self, tokens):
		"""
		Runs a string of tokens as if from the "home screen".
		"""
		# TODO: Should there be some kind of flag that makes newline characters raise an error?
		# On the calculator, it's impossible to get a newline character on the home screen (arguably that's what Enter does)
		# And actually, maybe if you treat NEWLINE as pressing Enter, everything works as intended
		from parser import Parser
		try:
			Parser(tokens, self).run()
		except StopSignal:
			pass
		self.console.finish(self.home)

	def run_program(self, prgm_name: str):
		"""Runs a stored program. To simulate running a program as you would on a calculator from the home screen, use Environment.run([PRGM, ...])."""
		try:
			prgm_code = self.programs[prgm_name]
		except KeyError:
			raise UndefinedError(f"Program not found: {prgm_name!r}")
		from program import Program
		Program(prgm_code, self).run()

	def to_rad(self, x):
		"""Convert x from the current angle mode to radians (for trig input)."""
		return x * (math.pi / 180) if self.angle_mode is AngleMode.DEG else x

	def from_rad(self, r):
		"""Convert r (radians) to the current angle mode (for inverse trig output)."""
		return r * (180 / math.pi) if self.angle_mode is AngleMode.DEG else r

	def from_deg(self, x):
		"""Convert x (in degrees) to the current angle mode (for DMS literals)."""
		return x * (math.pi / 180) if self.angle_mode is AngleMode.RAD else x
	
	@property
	def real_only(self):
		return self.complex_mode is ComplexMode.REAL

	def guard_real(self, inputs, result):
		"""Raise NonRealAnsError if real-mode is active and a real-input operation produced a complex result."""
		if self.real_only and is_complex_val(result) and not any(is_complex_val(x) for x in inputs):
			raise NonRealAnsError(repr(result))
		return result

	def set_random_seed(self, value):
		random.seed(require_real(value))

	# ── Virtual clock ────────────────────────────────────────────────────────────

	def now(self) -> datetime:
		"""Current datetime adjusted by any offset set via setDate/setTime."""
		return datetime.now() + self._datetime_offset

	def set_date(self, year, month, day):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(py_int(year), py_int(month), py_int(day), v.hour, v.minute, v.second)
		self._datetime_offset = new_v - now

	def set_time(self, hour, minute, second):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(v.year, v.month, v.day, py_int(hour), py_int(minute), py_int(second))
		self._datetime_offset = new_v - now

	# ── Nullary helpers (used by nullary= fields in tokens) ──────────────────────

	def get_date(self):
		d = self.now()
		return TiList([float(d.year), float(d.month), float(d.day)])

	def get_time(self):
		t = self.now()
		return TiList([float(t.hour), float(t.minute), float(t.second)])

	def start_tmr(self):
		return float(int(self.now().timestamp()))

	def get_dt_fmt(self):
		return float(self.dt_fmt)

	def get_tm_fmt(self):
		return float(self.tm_fmt)

	def is_clock_on(self):
		return float(self.clock_on)

	def get_key(self):
		return float(self.console.read_key())

	def rand(self):
		return random.random()

	@contextmanager
	def nest_guard(self, func: object, max_depth: int = 0):
		if self._nest_depth[func] > max_depth:
			raise IllegalNestError(func)
		self._nest_depth[func] += 1
		try:
			yield
		finally:
			self._nest_depth[func] -= 1

	def _iter_values(self):
		# Variable display names, reconstructed from each storage list's index so
		# this doesn't depend on catalog's token tables.  Order/spelling mirror the
		# TI charset: A–Z then θ; L₁–L₆ (subscript digits); [A]–[J]; Str1–Str0.
		numeric_names = [chr(0x41 + i) for i in range(26)] + ['θ']
		named = (
			zip(numeric_names, self.numerics),
			((f"L{chr(0x2081 + i)}", var) for i, var in enumerate(self.lists)),
			((f"[{chr(0x41 + i)}]",  var) for i, var in enumerate(self.matrices)),
			((f"Str{(i + 1) % 10}",  var) for i, var in enumerate(self.strings)),
		)
		for group in named:
			for name, var in group:
				if var.value is not None:
					yield name, var.value
		for name, lst in self.user_lists.items():
			yield f"${name}", lst
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name}= {int(value) if isinstance(value, float) and value.is_integer() else value!r}")

	def __repr__(self):
		return f"ENV({','.join(f"{name}={value!r}" for name, value in self._iter_values())})"

	def current_program(self):
		"""Return the innermost currently-executing Program.

		Raises InvalidCommandError if called outside a program (e.g. from the
		home screen), matching the calculator's ERR:INVALID for control-flow
		commands like Return, Goto, and End.
		"""
		if not self.program_stack:
			raise InvalidCommandError("This command cannot be used outside a program")
		return self.program_stack[-1]

	def print_screen(self):
		pass


# ── Window variables ──────────────────────────────────────────────────────────

class WindowVars:
	"""Named storage for all TI-84 window/graphing variables."""

	def __init__(self):
		self.xscl       = RealVariable(1.0)
		self.yscl       = RealVariable(1.0)
		self.xmin       = RealVariable(-10.0)
		self.xmax       = RealVariable(10.0)
		self.ymin       = RealVariable(-10.0)
		self.ymax       = RealVariable(10.0)
		self.tmin       = RealVariable()
		self.tmax       = RealVariable()
		self.theta_min  = RealVariable()
		self.theta_max  = RealVariable()
		self.tbl_start  = RealVariable()
		self.plot_start = RealVariable(1.0)
		self.n_max      = RealVariable(10.0)
		self.n_min      = RealVariable(1.0)
		self.delta_tbl  = RealVariable(1.0)
		self.tstep      = RealVariable()
		self.theta_step = RealVariable()
		self.delta_x    = RealVariable()
		self.delta_y    = RealVariable()
		self.x_fact     = RealVariable(4.0)
		self.y_fact     = RealVariable(4.0)
		self.plot_step  = RealVariable(1.0)
		self.xres       = RealVariable(1.0)


class ReturnSignal(Exception):
	"""Raised by Return to exit the current sub-program and return to the caller."""


class StopSignal(Exception):
	"""Raised by Stop to terminate all program execution immediately."""
