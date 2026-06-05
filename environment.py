from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Any, ClassVar, TYPE_CHECKING

from tiobjects import (
	TiList, TiMatrix, TiString, TiEquation, 
	require_num, require_real, require_int, require_list, require_matrix, require_str, require_equation,
)
from errors import TiError, DataTypeError, DomainError, IllegalNestError, InvalidDimError, UndefinedError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder
from signals import StopSignal


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
		self.key_code = 0
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
		self.window = [
			RealVariable(1),    # [0]  Xscl
			RealVariable(1),    # [1]  Yscl
			RealVariable(-10),  # [2]  Xmin
			RealVariable(10),   # [3]  Xmax
			RealVariable(-10),  # [4]  Ymin
			RealVariable(10),   # [5]  Ymax
			RealVariable(),     # [6]  Tmin
			RealVariable(),     # [7]  Tmax
			RealVariable(),     # [8]  θmin
			RealVariable(),     # [9]  θmax
			RealVariable(),     # [10] TblStart
			RealVariable(1),    # [11] PlotStart
			RealVariable(10),   # [12] nMax
			RealVariable(1),    # [13] nMin
			RealVariable(1),    # [14] ΔTbl
			RealVariable(),     # [15] Tstep
			RealVariable(),     # [16] θstep
			RealVariable(),     # [17] ΔX
			RealVariable(),     # [18] ΔY
			RealVariable(4),    # [19] XFact
			RealVariable(4),    # [20] YFact
		]
		# TVM finance variables (used by bal(, ΣPrn(, ΣInt(, tvm_Pmt, etc.)
		self.n_tvm = RealVariable()   # 𝐍 (number of payments)
		self.i_pct = RealVariable()   # I% (interest rate per period, as percentage)
		self.pv    = RealVariable()   # PV (present value)
		self.pmt   = RealVariable()   # PMT (payment amount)
		self.fv    = RealVariable()   # FV (future value)
		self.py    = RealVariable(1)  # P/Y (payments per year)
		self.cy    = RealVariable(1)  # C/Y (compounding periods per year)
		# Programs
		self.programs: dict[str, list] = {}  # name -> token list for stored programs
		self.program_stack: deque = deque()  # currently executing programs (innermost last)
		# Internal data
		self._datetime_offset = timedelta(0)  # virtual_time = system_time + offset
		self._nest_depth: dict[object, int] = defaultdict(lambda: 0)  # tracks nesting depth for ILLEGAL NEST guards


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

	def set_random_seed(self, value):
		random.seed(require_int(value))

	# ── Virtual clock ────────────────────────────────────────────────────────────

	def _now(self) -> datetime:
		"""Current datetime adjusted by any offset set via setDate/setTime."""
		return datetime.now() + self._datetime_offset

	def set_date(self, year, month, day):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(require_int(year), require_int(month), require_int(day), v.hour, v.minute, v.second)
		self._datetime_offset = new_v - now

	def set_time(self, hour, minute, second):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(v.year, v.month, v.day, require_int(hour), require_int(minute), require_int(second))
		self._datetime_offset = new_v - now

	def check_tmr(self, start):
		return int(self._now().timestamp()) - int(require_real(start))

	def set_dt_fmt(self, fmt):
		fmt = require_int(fmt)
		if fmt not in {1, 2, 3}:
			raise DomainError(f"setDtFmt: expected 1, 2, or 3; got {fmt}")
		self.dt_fmt = fmt

	def set_tm_fmt(self, fmt):
		fmt = require_int(fmt)
		if fmt not in {12, 24}:
			raise DomainError(f"setTmFmt: expected 12 or 24; got {fmt}")
		self.tm_fmt = fmt

	def get_dt_str(self, fmt):
		fmt = require_int(fmt)
		if fmt not in {1, 2, 3}:
			raise DomainError(f"getDtStr: invalid format {fmt}")
		return TiString.from_str(self._now().strftime(['%m/%d/%y', '%d/%m/%y', '%y/%m/%d'][fmt - 1]))

	def get_tm_str(self, fmt):
		fmt = require_int(fmt)
		now = self._now()
		if fmt == 24:
			time_str = now.strftime('%H:%M')
		elif fmt == 12:
			time_str = now.strftime('%I:%M %p').lstrip('0')
		else:
			raise DomainError(f"getTmStr: invalid format {fmt}")
		return TiString.from_str(time_str)

	# ── Nullary helpers (used by nullary= fields in tokens) ──────────────────────

	def get_ans(self):
		return self.ans

	def get_date(self):
		d = self._now()
		return TiList([d.year, d.month, d.day])

	def get_time(self):
		t = self._now()
		return TiList([t.hour, t.minute, t.second])

	def start_tmr(self):
		return int(self._now().timestamp())

	def get_dt_fmt(self):
		return self.dt_fmt

	def get_tm_fmt(self):
		return self.tm_fmt

	def is_clock_on(self):
		return self.clock_on

	def get_key(self):
		return self.key_code

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
		from catalog import LETTERS, LISTS, MATRICES, STRINGS
		for tok in (*LETTERS, *LISTS, *MATRICES, *STRINGS):
			value = tok.variable(self).value
			if value is not None:
				yield tok.text, value
		for name, lst in self.user_lists.items():
			yield f"${name}", lst
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name}= {int(value) if isinstance(value, float) and value.is_integer() else value!r}")

	def __repr__(self):
		return f"ENV({','.join(f"{name}={value!r}" for name, value in self._iter_values())})"

	@property
	def current_program(self):
		"""The innermost currently-executing Program, or None if running interactively."""
		return self.program_stack[-1] if self.program_stack else None

	def print_screen(self):
		pass


# ── Variable hierarchy ────────────────────────────────────────────────────────────

class Variable(ABC):
	"""Abstract base for all TI-BASIC variable types.

	Concrete subclasses fall into two storage models:
	  - Instance-stored (NumericVariable and its descendants, ListVariable, etc.):
	    value lives in self.value; NumericVariable provides __init__(default=None)
	    which subclasses inherit.  Other instance-stored classes have no __init__
	    and therefore never accept a constructor default — intentional, since lists,
	    matrices, strings and equations are always initialised to undefined (None).
	  - Proxy-stored (UserList): value lives in env.user_lists[name]; the class
	    manages its own __init__ and exposes value as a property.  It inherits from
	    Variable without calling super().__init__() because there is no super
	    __init__ to call.
	"""
	value = None  # class-level sentinel; instance writes shadow it

	def resolve(self) -> Any:
		"""Called when the user references a variable."""
		if self.value is None:
			raise UndefinedError(f"Undefined variable: {self}")
		return self.value

	def store(self, new_value) -> None:
		"""Called when the user stores a variable."""
		self.value = self.normalize(new_value)

	@abstractmethod
	def normalize(self, value) -> Any:
		"""Validates and coerces a value before storage. For structured types, returns a defensive copy."""
		pass
	
	def __repr__(self):
		return f"{type(self).__name__}({self.value})"


class NumericVariable(Variable):
	"""Numeric (real or complex) variable.  Accepts an optional constructor default
	which RealVariable and other subclasses inherit — the only Variable subclasses
	that are ever constructed with a non-None starting value."""

	def __init__(self, default=None):
		self.value = default

	def resolve(self):
		if self.value is None:
			self.value = 0
		return self.value

	def normalize(self, value):
		return require_num(value)
		
	@contextmanager
	def scoped(self):
		saved = self.resolve()
		try:
			yield
		finally:
			self.value = saved


class RealVariable(NumericVariable):
	def normalize(self, value):
		return require_real(value)
	
class ListVariable(Variable):
	def resolve(self):
		lst = super().resolve()
		if not lst.data:
			raise InvalidDimError("list is empty")
		return lst

	def normalize(self, value):
		return require_list(value).copy()

class MatrixVariable(Variable):
	def normalize(self, value):
		return require_matrix(value).copy()

class StringVariable(Variable):
	def normalize(self, value):
		return require_str(value)

class EquationVariable(Variable):
	def normalize(self, value):
		if isinstance(value, TiEquation):
			return value
		if isinstance(value, TiString):
			return TiEquation(value.tokens)
		raise DataTypeError(f"Expected equation or string; got {value}")


class UserList(Variable):

	def __init__(self, env: Environment, name: str):
		self.lookup = env.user_lists
		self.name = name
		# Don't call super().__init__() — value is managed via the property below.

	@property
	def value(self):
		return self.lookup.get(self.name)

	@value.setter
	def value(self, new_value):
		if new_value is None:
			self.lookup.pop(self.name, None)
		else:
			self.lookup[self.name] = new_value

	def normalize(self, value) -> Any:
		return require_list(value).copy()

	def resolve(self) -> Any:
		try:
			lst = self.lookup[self.name]
		except KeyError:
			raise UndefinedError(f"User list {self.name!r} is not defined")
		if not lst.data:
			raise InvalidDimError("list is empty")
		return lst
