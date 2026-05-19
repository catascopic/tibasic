import math, cmath, random, datetime as _dt


class Environment:

	def __init__(self):
		self.ans = 0.0
		self.reals = {}
		self.lists = {}
		self.matrices = {}
		self.strings = {}
		self.stat = {}
		self.window = {}

	# ── Nullary resolve helpers (used by resolve= fields in tokens) ─────────────

	def get_date(self):
		t = _dt.date.today()
		return [float(t.year), float(t.month), float(t.day)]

	def get_time(self):
		t = _dt.datetime.now()
		return [float(t.hour), float(t.minute), float(t.second)]

	def start_tmr(self):
		return float(int(_dt.datetime.now().timestamp()))

	def get_dt_fmt(self):
		return getattr(self, '_dtfmt', 1.0)

	def get_tm_fmt(self):
		return getattr(self, '_tmfmt', 12.0)

	def is_clock_on(self):
		return 1.0

	def get_key(self):
		return getattr(self, '_getkey', 0.0)

	def rand(self):
		return random.random()

