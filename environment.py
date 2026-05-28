from collections import defaultdict
import math
import random

from contextlib import contextmanager
from datetime import datetime, date, timedelta

from tiobjects import TiList, TiMatrix, TiString, require_num, require_real, require_int, require_list, require_matrix, require_str
from errors import DataTypeError, DomainError, IllegalNestError


class _VarArray:
	"""Array-backed variable store with integer indexing."""
	__slots__ = ('_data', '_formatter')

	def __init__(self, size, default, formatter):
		self._data = [default] * size
		self._formatter = formatter

	def __getitem__(self, idx: int):
		return self._data[idx]

	def __setitem__(self, idx: int, value):
		self._data[idx] = value

	def iter_values(self):
		for i, x in enumerate(self._data):
			if x is not None:
				yield self._formatter(i), x



class _NumericVarArray(_VarArray):
	def __init__(self):
		super().__init__(27, None, lambda n: chr(65+n) if n < 26 else 'θ')

	def __getitem__(self, idx: int):
		val = self._data[idx]
		if val is None:
			val = self._data[idx] = 0
		return val


class Environment:

	def __init__(self):
		self.numerics   = _NumericVarArray()                               # A–Z, θ
		self.lists      = _VarArray(6,  None, lambda n: f"L{n+1}")         # L1–L6
		self.matrices   = _VarArray(10, None, lambda n: f"[{chr(65+n)}]")  # [A]–[J]
		self.strings    = _VarArray(10, None, lambda n: f"Str{n+1}")       # Str0–9
		self.stat       = _VarArray(0x3D, None, repr)                      # stat vars
		self.window     = _VarArray(0x37, None, repr)                      # window vars
		self.user_lists = {}                                               # ᴸNAME lists
		self.n = None
		self.ans              = 0
		self.angle_mode       = 'RAD'
		self.dt_fmt           = 1
		self.tm_fmt           = 12
		self.clock_on         = True
		self.key_code         = 0
		self._datetime_offset = timedelta(0)  # virtual_time = system_time + offset
		self._nest_depth: dict[object, int] = defaultdict(lambda: 0)  # tracks nesting depth for ILLEGAL NEST guards

	def to_radians(self, x):
		return x / (180 / math.pi) if self.angle_mode == 'RAD' else x

	def to_degrees(self, x):
		return x / (math.pi / 180) if self.angle_mode == 'DEG' else x

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

	def clock_on(self):
		self.clock_on = True

	def clock_off(self):
		self.clock_on = False

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
	def scoped_var(self, variable):
		saved = variable.get(self)
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
		for field in ('numerics', 'lists', 'matrices', 'strings'):
			yield from getattr(self, field).iter_values()
		for name, lst in self.user_lists.items():
			yield f"ᴸ{name}", value
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name:8}= {value!r}")
	
	def __repr__(self):
		return f"ENV({';'.join(f"{name}={value!r}" for name, value in self._iter_values())})"

	def print_screen(self):
		pass


# ── Variable hierarchy ────────────────────────────────────────────────────────────

class Variable:
	"""Base class for typed, storable token variables."""
	def get(self, env): ...
	def set(self, env, value): ...


class NumericVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.numerics[self._idx]

	def set(self, env, value):
		env.numerics[self._idx] = require_num(value)


class RealVar(Variable):
	__slots__ = ('_name',)

	def __init__(self, name: int):
		self._name = name

	def get(self, env):
		val = getattr(env, name)
		if val is None:
			setattr(env, name, 0)
			val = 0
		return val

	def set(self, env, value):
		return setattr(env, name, require_real(value))


class ListVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.lists[self._idx]

	def set(self, env, value):
		env.lists[self._idx] = require_list(value)


class UserListVar(Variable):
	__slots__ = ('_name',)

	def __init__(self, name: str):
		self._name = name

	def get(self, env):
		return env.user_lists[self._name]

	def set(self, env, value):
		env.user_lists[self._name] = require_list(value)


class MatrixVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.matrices[self._idx]

	def set(self, env, value):
		env.matrices[self._idx] = require_matrix(value)


class StringVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.strings[self._idx]

	def set(self, env, value):
		env.strings[self._idx] = require_str(value)


class StatVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.stat[self._idx]

	def set(self, env, value):
		raise DataTypeError("Stat variables are read-only")


class WindowVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.window[self._idx]

	def set(self, env, value):
		env.window[self._idx] = require_real(value)
