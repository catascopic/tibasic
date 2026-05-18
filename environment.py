import math, cmath, random, datetime as _dt


class Environment:

	# ── Static helpers (used as binary_op / unary_op by tokens) ────────────────

	@staticmethod
	def factorial(n: float) -> float:
		if n < 0 or n != int(n):
			raise ValueError("Argument to ! must be a non-negative integer")
		return float(math.factorial(int(n)))

	@staticmethod
	def ncr(n: float, r: float) -> float:
		return float(math.comb(int(n), int(r)))

	@staticmethod
	def npr(n: float, r: float) -> float:
		return float(math.perm(int(n), int(r)))

	@staticmethod
	def list_mul(a, b):
		if isinstance(a, list) and not isinstance(a[0], list):
			return [x * b for x in a]
		if isinstance(b, list) and not isinstance(b[0], list):
			return [a * x for x in b]
		return a * b

	@staticmethod
	def matrix_transpose(m):
		return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]

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

	# ── Custom-parse functions ──────────────────────────────────────────────────

	def call_seq(self, parser) -> list:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var_tok = parser.advance()
		if not var_tok.is_real_var():
			raise ValueError("seq: second arg must be a variable")
		var = var_tok.text
		parser.expect(COMMA)
		start = parser.parse_expr()
		parser.expect(COMMA)
		end = parser.parse_expr()
		step = 1.0
		if parser.eat_if(COMMA):
			step = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		result = []
		n = start
		while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
			self.reals[var] = n
			result.append(thunk.eval())
			n += step
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return result

	def call_sigma(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var_tok = parser.advance()
		if not var_tok.is_real_var():
			raise ValueError("Σ: second arg must be a variable")
		var = var_tok.text
		parser.expect(COMMA)
		start = int(parser.parse_expr())
		parser.expect(COMMA)
		end = int(parser.parse_expr())
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		total = 0.0
		for i in range(start, end + 1):
			self.reals[var] = float(i)
			total += thunk.eval()
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return total

	def call_nderiv(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var = parser.advance().text
		parser.expect(COMMA)
		val = parser.parse_expr()
		h = 1e-5
		if parser.eat_if(COMMA):
			h = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		self.reals[var] = val + h
		fwd = thunk.eval()
		self.reals[var] = val - h
		bwd = thunk.eval()
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return (fwd - bwd) / (2 * h)

	def call_fnint(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var = parser.advance().text
		parser.expect(COMMA)
		a = parser.parse_expr()
		parser.expect(COMMA)
		b = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		n = 1000
		h = (b - a) / n
		def f(x):
			self.reals[var] = x
			return thunk.eval()
		total = f(a) + f(b)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(a + i * h)
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return total * h / 3
