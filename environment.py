import math
import random

from datetime import datetime, date
from tiobjects import TiList, TiMatrix, TiTypeError


class _VarArray:
	"""Array-backed variable store with transparent Token-to-index mapping."""
	__slots__ = ('_data', '_byte', '_offset')

	def __init__(self, size, default, *, byte=1, offset=0):
		self._data   = [default] * size
		self._byte   = byte
		self._offset = offset

	def __getitem__(self, token):
		return self._data[token.code[self._byte] - self._offset]

	def __setitem__(self, token, value):
		self._data[token.code[self._byte] - self._offset] = value

	def __repr__(self):
		return repr(self._data)


# ── Variable hierarchy ────────────────────────────────────────────────────────────

class Variable:
	"""Base class for typed, storable token variables."""
	def get(self, env): ...
	def set(self, env, value): ...


class RealVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.reals[self._token]

	def set(self, env, value):
		if isinstance(value, complex):
			raise TiTypeError("Cannot store complex number in real variable")
		env.reals[self._token] = float(value)


class ListVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.lists[self._token]

	def set(self, env, value):
		if not isinstance(value, TiList):
			raise TiTypeError(f"Expected a list, got {type(value).__name__}")
		env.lists[self._token] = value


class UserListVar(Variable):
	__slots__ = ('_name',)

	def __init__(self, name: str):
		self._name = name

	def get(self, env):
		return env.user_lists[self._name]

	def set(self, env, value):
		if not isinstance(value, TiList):
			raise TiTypeError(f"Expected a list, got {type(value).__name__}")
		env.user_lists[self._name] = value


class MatrixVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.matrices[self._token]

	def set(self, env, value):
		if not isinstance(value, TiMatrix):
			raise TiTypeError(f"Expected a matrix, got {type(value).__name__}")
		env.matrices[self._token] = value


class StringVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.strings[self._token]

	def set(self, env, value):
		if not isinstance(value, str):
			raise TiTypeError(f"Expected a string, got {type(value).__name__}")
		env.strings[self._token] = value


class StatVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.stat[self._token]

	def set(self, env, value):
		raise TiTypeError("Stat variables are read-only")


class WindowVar(Variable):
	__slots__ = ('_token',)

	def __init__(self, token):
		self._token = token

	def get(self, env):
		return env.window[self._token]

	def set(self, env, value):
		if isinstance(value, complex):
			raise TiTypeError("Cannot store complex number in window variable")
		env.window[self._token] = float(value)


class AnsVar(Variable):
	def get(self, env):
		return env.ans

	def set(self, env, value):
		env.ans = value


ANS_VAR = AnsVar()


class Environment:

	def __init__(self):
		self.reals      = _VarArray(27,    0,    byte=0, offset=0x41)  # A–Z, θ
		self.lists      = _VarArray(6,     None, byte=1)               # L1–L6
		self.matrices   = _VarArray(10,    None, byte=1)               # [A]–[J]
		self.strings    = _VarArray(10,    "",   byte=1)               # Str0–9
		self.stat       = _VarArray(0x3D,  0,    byte=1)               # stat vars
		self.window     = _VarArray(0x37,  0,    byte=1)               # window vars
		self.user_lists = {}                                           # ∟NAME lists
		self.ans        = 0
		self.dt_fmt     = 1
		self.tm_fmt     = 12
		self.clock_on   = True
		self.key_code   = 0

	def set_random_seed(self, value):
		random.seed(float(value))

	# ── Nullary helpers (used by nullary= fields in tokens) ──────────────────────

	def get_date(self):
		t = date.today()
		return TiList([float(t.year), float(t.month), float(t.day)])

	def get_time(self):
		t = datetime.now()
		return TiList([float(t.hour), float(t.minute), float(t.second)])

	def start_tmr(self):
		return float(int(datetime.now().timestamp()))

	def get_dt_fmt(self):
		return float(self.dt_fmt)

	def get_tm_fmt(self):
		return float(self.tm_fmt)

	def is_clock_on(self):
		return 1.0 if self.clock_on else 0.0

	def get_key(self):
		return float(self.key_code)

	def rand(self):
		return random.random()
