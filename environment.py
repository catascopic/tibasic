import math
import random

from datetime import datetime, date
from tiobjects import TiList, TiMatrix, TiString, TiTypeError, require_num, require_list, require_matrix, require_str


class _VarArray:
	"""Array-backed variable store with integer indexing."""
	__slots__ = ('_data','_formatter')

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

	def __repr__(self):
		return ', '.join(iter_values())


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
		raise TiTypeError("Stat variables are read-only")


class WindowVar(Variable):
	__slots__ = ('_idx',)

	def __init__(self, idx: int):
		self._idx = idx

	def get(self, env):
		return env.window[self._idx]

	def set(self, env, value):
		if isinstance(value, complex):
			raise TiTypeError("Cannot store complex number in window variable")
		env.window[self._idx] = float(value)


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
		self.stat       = _VarArray(0x3D, None, repr)                         # stat vars
		self.window     = _VarArray(0x37, None, repr)                         # window vars
		self.user_lists = {}                                               # ∟NAME lists
		self.ans        = 0
		self.dt_fmt     = 1
		self.tm_fmt     = 12
		self.clock_on   = True
		self.key_code   = 0

	def set_random_seed(self, value):
		random.seed(value)

	# ── Nullary helpers (used by nullary= fields in tokens) ──────────────────────

	def get_ans(self):
		return self.ans

	def get_date(self):
		t = date.today()
		return TiList([t.year, t.month, t.day])

	def get_time(self):
		t = datetime.now()
		return TiList([t.hour, t.minute, t.second])

	def start_tmr(self):
		return int(datetime.now().timestamp())

	def get_dt_fmt(self):
		return self.dt_fmt

	def get_tm_fmt(self):
		return self.tm_fmt

	def is_clock_on(self):
		return int(self.clock_on)

	def get_key(self):
		return self.key_code

	def rand(self):
		return random.random()

	def iter_values(self):
		for field in ('numerics', 'lists', 'matrices', 'strings'):
			yield from getattr(self, field).iter_values()
		for name, lst in self.user_lists.items():
			yield f"∟{name}", value
		yield "Ans", self.ans

	def dump(self):
		for name, value in self.iter_values():
			print(f"{name:8}= {value}")