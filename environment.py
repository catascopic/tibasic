from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from itertools import repeat
from typing import Any, ClassVar, TYPE_CHECKING

from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_matrix, require_str, require_equation
from errors import TiError, DataTypeError, DomainError, IllegalNestError, UndefinedError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder
from signals import StopSignal


# ── Variable hierarchy ────────────────────────────────────────────────────────────

class Variable(ABC):
	"""Self-contained variable: holds its own value and provides get/set/delete
	without an env parameter.  Instances live in env's arrays; Token.variable
	is a callable (env) -> Variable that looks up the right slot.
	"""

	@abstractmethod
	def get(self) -> Any:
		"""Return the current value.  May auto-initialise or raise UndefinedError."""
		pass

	@abstractmethod
	def set(self, value: Any) -> None:
		pass

	def delete(self) -> None:
		raise DataTypeError("Cannot delete this variable")

	def get_unsafe(self) -> Any:
		"""Return the raw stored value without auto-init or error; None means absent."""
		return None


class NumericVar(Variable):
	"""Numeric variable (A–Z, θ).  Auto-initialises to 0 when read while absent."""
	__slots__ = ('_value', 'name')

	def __init__(self, name: str) -> None:
		self._value: float | None = None
		self.name = name

	def __str__(self) -> str:
		return self.name

	def get(self) -> float:
		if self._value is None:
			self._value = 0.0
		return self._value

	def set(self, value: Any) -> None:
		self._value = require_num(value)

	def delete(self) -> None:
		self._value = None

	def get_unsafe(self) -> float | None:
		return self._value


class ListVar(Variable):
	"""Standard list variable (L1–L6).  Stores a Python list as the backing store;
	get() returns a TiList sharing that list (in-place mutations propagate back);
	set() copies on write.  None data means absent/deleted.
	"""
	__slots__ = ('data', 'name')

	def __init__(self, name: str) -> None:
		self.data: list | None = None
		self.name = name

	def __str__(self) -> str:
		return self.name

	def __len__(self) -> int:
		return 0 if self.data is None else len(self.data)

	def get(self) -> TiList:
		if self.data is None:
			raise UndefinedError(self.name)
		return TiList(self.data)

	def set(self, value: Any) -> None:
		self.data = list(require_list(value).data)

	def delete(self) -> None:
		self.data = None

	def get_unsafe(self) -> TiList | None:
		return TiList(self.data) if self.data is not None else None

	def set_dim(self, value: Any) -> None:
		"""Resize this list in-place (used by dim(→ and ClrList)."""
		from tiobjects import require_int_dim
		n = require_int_dim(value)
		if self.data is None:
			self.data = []
		dim = len(self.data)
		if n < dim:
			del self.data[n:]
		elif n > dim:
			self.data.extend(repeat(0, n - dim))


class MatrixVar(Variable):
	"""Matrix variable ([A]–[J]).  Stores the TiMatrix directly."""
	__slots__ = ('_value', 'name')

	def __init__(self, name: str) -> None:
		self._value: TiMatrix | None = None
		self.name = name

	def __str__(self) -> str:
		return self.name

	@property
	def data(self):
		"""Convenience accessor for tests: env.matrices[i].data."""
		return self._value.data if self._value is not None else None

	def get(self) -> TiMatrix:
		if self._value is None:
			raise UndefinedError(self.name)
		return self._value

	def set(self, value: Any) -> None:
		self._value = require_matrix(value).copy()

	def delete(self) -> None:
		self._value = None

	def get_unsafe(self) -> TiMatrix | None:
		return self._value


class StringVar(Variable):
	"""String variable (Str1–Str0)."""
	__slots__ = ('_value', 'name')

	def __init__(self, name: str) -> None:
		self._value: TiString | None = None
		self.name = name

	def __str__(self) -> str:
		return self.name

	def get(self) -> TiString:
		if self._value is None:
			raise UndefinedError(self.name)
		return self._value

	def set(self, value: Any) -> None:
		self._value = require_str(value)

	def delete(self) -> None:
		self._value = None

	def get_unsafe(self) -> TiString | None:
		return self._value


class EquationVar(Variable):
	"""Equation variable (Y1–Y0, X1T–Y6T, r1–r6, u/v/w).
	Holds a reference to the environment's raw storage list and an index into it.
	"""
	__slots__ = ('_storage', 'index', 'name')

	def __init__(self, storage: list, name: str, index: int) -> None:
		self._storage = storage
		self.index = index
		self.name = name

	def __str__(self) -> str:
		return self.name

	def get(self) -> Any:
		val = self._storage[self.index]
		if val is None:
			raise UndefinedError(self.name)
		return val

	def set(self, value: Any) -> None:
		self._storage[self.index] = require_equation(value)

	def delete(self) -> None:
		self._storage[self.index] = None

	def get_unsafe(self) -> Any:
		return self._storage[self.index]


class StatVar(Variable):
	"""Stat result variable (read-only; None before a stat function has run)."""
	__slots__ = ('_storage', 'index')

	def __init__(self, storage: list, index: int) -> None:
		self._storage = storage
		self.index = index

	def get(self) -> Any:
		return self._storage[self.index]  # may be None

	def set(self, value: Any) -> None:
		raise DataTypeError("Stat variables are read-only")

	def get_unsafe(self) -> Any:
		return self._storage[self.index]


class WindowVar(Variable):
	"""Window/table variable (Xmin, Xmax, etc.)."""
	__slots__ = ('_storage', 'index', 'name')

	def __init__(self, storage: list, name: str, index: int) -> None:
		self._storage = storage
		self.index = index
		self.name = name

	def __str__(self) -> str:
		return self.name

	def get(self) -> float:
		val = self._storage[self.index]
		if val is None:
			raise UndefinedError(self.name)
		return val

	def set(self, value: Any) -> None:
		self._storage[self.index] = require_real(value)

	def get_unsafe(self) -> float | None:
		return self._storage[self.index]


class RealVar(Variable):
	"""Scalar env attribute variable (𝑵, I%, PV, PMT, FV, P/Y, C/Y, 𝑛).
	Holds a back-reference to env so it can use getattr/setattr.
	"""
	__slots__ = ('_env', 'name')

	def __init__(self, env: Environment, name: str) -> None:
		self._env = env
		self.name = name

	def __str__(self) -> str:
		return self.name

	def get(self) -> float:
		val = getattr(self._env, self.name)
		if val is None:
			setattr(self._env, self.name, 0)
			return 0
		return val

	def set(self, value: Any) -> None:
		setattr(self._env, self.name, require_real(value))

	def get_unsafe(self) -> float | None:
		return getattr(self._env, self.name)


class UserListVar(Variable):
	"""User-named list variable (ᴸNAME).
	Holds a reference to env.user_lists so it can look up / create entries by name.
	Created at parse time by _parse_user_list_var(); not pre-allocated in env.
	"""
	__slots__ = ('_storage', 'name')

	def __init__(self, storage: dict, name: str) -> None:
		self._storage = storage
		self.name = name

	def __str__(self) -> str:
		return f"ᴸ{self.name}"

	def __len__(self) -> int:
		lst = self._storage.get(self.name)
		return 0 if lst is None else len(lst)

	def get(self) -> TiList:
		lst = self._storage.get(self.name)
		if lst is None:
			raise UndefinedError(str(self))
		return lst

	def set(self, value: Any) -> None:
		self._storage[self.name] = require_list(value).copy()

	def delete(self) -> None:
		self._storage.pop(self.name, None)

	def get_unsafe(self) -> TiList | None:
		return self._storage.get(self.name)


# ── Window variable name list (matches catalog WindowVar indices 0–20) ────────────

_WINDOW_VAR_NAMES = [
	'Xscl', 'Yscl', 'Xmin', 'Xmax', 'Ymin', 'Ymax',
	'Tmin', 'Tmax', 'θmin', 'θmax',
	'TblStart', 'PlotStart', 'nMax', 'nMin', 'ΔTbl',
	'Tstep', 'θstep', 'ΔX', 'ΔY', 'XFact', 'YFact',
]


# ── Environment ───────────────────────────────────────────────────────────────

class Environment:

	def __init__(self):
		# ── Raw storage (for types whose Variable objects hold references) ──────
		# Equations
		self._y_eq_storage     = [None] * 10
		self._param_eq_storage = [None] * 12
		self._polar_eq_storage = [None] * 6
		self._seq_eq_storage   = [None] * 3
		# Window / stat
		self.window = [None] * 0x37
		self.stat   = [None] * 0x3D

		# ── Pre-allocated variable objects ────────────────────────────────────
		_num_names = [chr(65 + i) for i in range(26)] + ['θ']
		self.numerics = [NumericVar(n) for n in _num_names]

		self.lists    = [ListVar(f'L{i + 1}') for i in range(6)]

		self.matrices = [MatrixVar(f'[{chr(65 + i)}]') for i in range(10)]

		self.strings  = [StringVar(f'Str{(i + 1) % 10}') for i in range(10)]

		_y_names = [f'Y{(i + 1) % 10}' for i in range(10)]
		self.y_equations = [
			EquationVar(self._y_eq_storage, _y_names[i], i) for i in range(10)
		]

		import itertools as _it
		_param_names = [
			f'{x}{chr(0x2080 + n)}ₜ'
			for n, x in _it.product(range(1, 7), 'XY')
		]
		self.param_equations = [
			EquationVar(self._param_eq_storage, _param_names[i], i) for i in range(12)
		]

		self.polar_equations = [
			EquationVar(self._polar_eq_storage, f'r{i + 1}', i) for i in range(6)
		]

		self.seq_equations = [
			EquationVar(self._seq_eq_storage, name, i)
			for i, name in enumerate(['𝑢', '𝑣', '𝑤'])
		]

		self.window_vars = [
			WindowVar(self.window, name, i) for i, name in enumerate(_WINDOW_VAR_NAMES)
		]

		# RealVar objects for scalar finance / misc attributes
		self._rv_n     = RealVar(self, 'n')
		self._rv_n_tvm = RealVar(self, 'n_tvm')
		self._rv_i_pct = RealVar(self, 'i_pct')
		self._rv_pv    = RealVar(self, 'pv')
		self._rv_pmt   = RealVar(self, 'pmt')
		self._rv_fv    = RealVar(self, 'fv')
		self._rv_py    = RealVar(self, 'py')
		self._rv_cy    = RealVar(self, 'cy')

		# ── Scalar variables ─────────────────────────────────────────────────
		self.user_lists: dict[str, TiList] = {}
		self.n = None
		self.ans = 0
		self.key_code = 0
		# Finance
		self.n_tvm = 0
		self.i_pct = 0
		self.pv    = 0
		self.pmt   = 0
		self.fv    = 0
		self.py    = 1
		self.cy    = 1
		# MODES
		self.angle_mode    = AngleMode.RAD
		self.number_mode   = NumberMode.NORMAL
		self.fix_digits    = None
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
		# Programs
		self.programs: dict[str, list] = {}
		self.program_stack: deque = deque()
		# Internal
		self._datetime_offset = timedelta(0)
		self._nest_depth: dict[object, int] = defaultdict(lambda: 0)


	def run(self, tokens):
		from parser import Parser
		try:
			Parser(tokens, self).run()
		except StopSignal:
			pass

	def run_program(self, prgm_name: str):
		try:
			prgm_code = self.programs[prgm_name]
		except KeyError:
			raise UndefinedError(f"Program not found: {prgm_name!r}")
		from program import Program
		Program(prgm_code, self).run()

	def to_rad(self, x):
		return x * (math.pi / 180) if self.angle_mode is AngleMode.DEG else x

	def from_rad(self, r):
		return r * (180 / math.pi) if self.angle_mode is AngleMode.DEG else r

	def from_deg(self, x):
		return x * (math.pi / 180) if self.angle_mode is AngleMode.RAD else x

	def set_random_seed(self, value):
		random.seed(require_int(value))

	# ── Virtual clock ────────────────────────────────────────────────────────────

	def _now(self) -> datetime:
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

	# ── Nullary helpers ─────────────────────────────────────────────────────────

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
	def scoped_var(self, variable: Variable):
		"""Save and restore a variable's value around a block."""
		saved = variable.get_unsafe()
		try:
			yield
		finally:
			if saved is None:
				variable.delete()
			else:
				variable.set(saved)

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
			v = tok.variable(self).get_unsafe()
			if v is not None:
				yield tok.char if tok.char else tok.text, v
		for name, lst in self.user_lists.items():
			yield f"ᴸ{name}", lst
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name:8}= {value!r}")

	def __repr__(self):
		return f"ENV({';'.join(f'{name}={value!r}' for name, value in self._iter_values())})"

	@property
	def current_program(self):
		return self.program_stack[-1] if self.program_stack else None

	def print_screen(self):
		pass
