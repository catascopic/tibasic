import cmath
import math
import random

from datetime import datetime, date


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


class _ListRef:
	"""Bound reference to a list variable — captures storage and key."""
	__slots__ = ('_store', '_key')

	def __init__(self, store, key):
		self._store = store
		self._key   = key

	def get(self):
		return self._store[self._key]

	def set(self, value):
		self._store[self._key] = value


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

	# ── Nullary resolve helpers (used by resolve= fields in tokens) ─────────────

	def get_date(self):
		t = date.today()
		return [float(t.year), float(t.month), float(t.day)]

	def get_time(self):
		t = datetime.now()
		return [float(t.hour), float(t.minute), float(t.second)]

	def start_tmr(self):
		return float(int(datetime.now().timestamp()))

	def get_dt_fmt(self):
		return self.dt_fmt

	def get_tm_fmt(self):
		return self.tm_fmt

	def is_clock_on(self):
		return 1 if self.clock_on else 0

	def get_key(self):
		return self.key_code

	def rand(self):
		return random.random()
