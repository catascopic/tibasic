from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Any, ClassVar, TYPE_CHECKING

from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_matrix, require_str, require_equation
from errors import TiError, DataTypeError, DomainError, IllegalNestError, UndefinedError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder
from signals import StopSignal


class Environment:

	def __init__(self):
		# VARIABLES
		self.numerics      = [None] * 27   # A–Z, θ
		self.lists         = [None] * 6    # L1–L6
		self.user_lists    = {}            # ᴸNAME lists
		self.matrices      = [None] * 10   # [A]–[J]
		self.strings       = [None] * 10   # Str1–Str0
		self.function      = [None] * 10   # Y1–Y0
		self.parametric    = [None] * 12   # X1T–Y6T
		self.polar         = [None] * 6    # r1–r6
		self.sequence      = [None] * 3    # u, v, w
		# self.stat        = [None] * 0x3D # stat vars
		# self.window      = [None] * 0x37 # window vars
		self.n = None
		self.ans = 0
		self.key_code = 0
		# MODES
		self.angle_mode    = AngleMode.RAD
		self.number_mode   = NumberMode.NORMAL
		self.fix_digits    = None           # None = Float, 0–9 = Fix N
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
		# TVM finance variables (used by bal(, ΣPrn(, ΣInt(, tvm_Pmt, etc.)
		self.n_tvm = 0   # 𝐍 (number of payments)
		self.i_pct = 0   # I% (interest rate per period, as percentage)
		self.pv    = 0   # PV (present value)
		self.pmt   = 0   # PMT (payment amount)
		self.fv    = 0   # FV (future value)
		self.py    = 1   # P/Y (payments per year)
		self.cy    = 1   # C/Y (compounding periods per year)
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
	def scoped(self, variable):
		saved = variable.resolve(self)
		try:
			yield
		finally:
			variable.set(self, saved)

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
			v = tok.variable.get(self)
			if v is not None:
				yield tok.char if tok.char else tok.text, v
		for name, lst in self.user_lists.items():
			yield f"ᴸ{name}", lst
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name:8}= {value!r}")

	def __repr__(self):
		return f"ENV({';'.join(f"{name}={value!r}" for name, value in self._iter_values())})"

	@property
	def current_program(self):
		"""The innermost currently-executing Program, or None if running interactively."""
		return self.program_stack[-1] if self.program_stack else None

	def print_screen(self):
		pass


# ── Variable hierarchy ────────────────────────────────────────────────────────────

class Variable(ABC):

	@abstractmethod
	def get(self, env: Environment) -> Any | None:
		"""Directly gets the variable. May return None."""
		pass

	@abstractmethod
	def set(self, env: Environment, value: Any) -> None:
		"""Directly sets the variable with no type enforcement. Internal use only."""
		pass

	def resolve(self, env: Environment) -> Any:
		"""Called when the user references a variable."""
		val = self.get(env)
		if val is None:
			raise UndefinedError(f"Undefined variable: {self}")
		return val

	def store(self, env, value) -> None:
		"""Called when the user stores a variable."""
		self.set(env, self.normalize(value))

	@abstractmethod
	def normalize(self, value) -> Any:
		"""Ensures that the stored data is the right type. For structured data, creates a defensive copy."""
		pass

	def delete(self, env: Environment) -> None:
		"""Called when the user executes DelVar."""
		raise DataTypeError(f"Cannot delete {self}")


class DefaultZeroMixin:
	def resolve(self, env):
		val = self.get(env)
		if val is None:
			self.set(env, 0)
			return 0
		return val


class OffsetVariable(Variable):
	"""Variable stored by integer index in one of the environment's flat arrays."""
	array_attr: ClassVar[str]

	def __init__(self, index: int):
		self.index = index

	def get(self, env):
		return getattr(env, self.array_attr)[self.index]

	def set(self, env, value):
		getattr(env, self.array_attr)[self.index] = value

	def delete(self, env):
		self.set(env, None)


class NumericVariable(DefaultZeroMixin, OffsetVariable):
	array_attr = 'numerics'
	def normalize(self, value): return require_num(value)
	def __repr__(self):         return f"Var<{chr(65 + self.index) if self.index < 26 else 'θ'}>"

class ListVariable(OffsetVariable):
	array_attr = 'lists'
	def normalize(self, value): return require_list(value).copy()
	def __repr__(self):         return f"Var<L{self.index + 1}>"

class MatrixVariable(OffsetVariable):
	array_attr = 'matrices'
	def normalize(self, value): return require_matrix(value).copy()
	def __repr__(self):         return f"Var<[{chr(65 + self.index)}]>"

class StringVariable(OffsetVariable):
	array_attr = 'strings'
	def normalize(self, value): return require_str(value)
	def __repr__(self):         return f"Var<Str{(self.index + 1) % 10}>"


class EquationVariable(OffsetVariable):

	def normalize(self, value):
		if instanceof(value, TiEquation):
			return value
		if instanceof(value, TiString):
			return TiEquation(value.tokens)
		raise DataTypeError(f"Expected equation or string; got {value}")


class FunctionVariable(EquationVariable):
	array_attr = 'function'
	def __repr__(self): return f"FunctionVar<Y{(self.index + 1) % 10}>"

class ParametricVariable(EquationVariable):
	array_attr = 'parametric'
	def __repr__(self): return f"ParametricVar<{'XY'[self.index % 2]}{1 + self.index // 2}T>"

class PolarVariable(EquationVariable):
	array_attr = 'polar'
	def __repr__(self): return f"PolarVar<r{(self.index + 1)}>"

class SequenceVariable(EquationVariable):
	array_attr = 'sequence'
	def __repr__(self): return f"SequenceVar<{'uvw'[self.index]}>"


class RealVariable(DefaultZeroMixin, Variable):

	def __init__(self, name: str):
		self.name = name

	def get(self, env):
		return getattr(env, self.name)

	def set(self, env, value):
		setattr(env, self.name, value)

	def delete(self, env):
		self.set(env, None)

	def normalize(self, value):
		return require_real(value)

	def __repr__(self):
		return f"RealVar({self.name})"


class UserListVariable(Variable):

	def __init__(self, name: str):
		self.name = name

	def get(self, env):
		return env.user_lists.get(self.name)

	def set(self, env, value):
		env.user_lists[self.name] = value

	def delete(self, env):
		env.user_lists.pop(self.name, None)
		
	def normalize(self, value):
		return require_list(value).copy()

	def __repr__(self):
		return f"UserListVar<{self.name}>"


class WindowVariable:
	def __init__(self, index):
		pass  # TODO
