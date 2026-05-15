import math, cmath, random, datetime as _dt


class Environment:

	# ── Static helpers (used by binary_op / unary_op fields in tokens) ─────────

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
	def ti_int(x: float) -> float:
		"""Truncation toward zero — TI's int() behaviour."""
		return math.floor(x) if x >= 0 else math.ceil(x)

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

	# ── Nullary / rand ──────────────────────────────────────────────────────────

	def rand(self):
		return random.random()

	# ── Function implementations ────────────────────────────────────────────────

	@staticmethod
	def round(a, b=9):
		return round(a)

	@staticmethod
	def max(*a):
		return max(a) if len(a) > 1 else max(a[0])

	@staticmethod
	def min(*a):
		return min(a) if len(a) > 1 else min(a[0])

	@staticmethod
	def median(a, b=None):
		return float(sorted(a)[len(a) // 2])

	@staticmethod
	def mean(a, b=None):
		return sum(a) / len(a) if b is None else sum(x * w for x, w in zip(a, b)) / sum(b)

	@staticmethod
	def int_(x):
		return Environment.ti_int(x)

	@staticmethod
	def abs(a):
		return abs(a)

	@staticmethod
	def det(m):
		return float(sum(m[i][i] for i in range(len(m))))

	@staticmethod
	def identity(n):
		n = int(n)
		return [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]

	@staticmethod
	def dim(a):
		if isinstance(a, list) and not isinstance(a[0], list):
			return float(len(a))
		return [float(len(a)), float(len(a[0]))]

	@staticmethod
	def sum(a):
		return sum(a)

	@staticmethod
	def prod(a):
		return math.prod(a)

	@staticmethod
	def not_(a):
		return float(not a)

	@staticmethod
	def ipart(x):
		return math.trunc(x)

	@staticmethod
	def fpart(x):
		return x - Environment.ti_int(x)

	@staticmethod
	def sqrt(a):
		return cmath.sqrt(a) if a < 0 else math.sqrt(a)

	@staticmethod
	def cbrt(a):
		return -(-a) ** (1/3) if a < 0 else a ** (1/3)

	@staticmethod
	def ln(a):
		return cmath.log(a)

	@staticmethod
	def exp(a):
		return cmath.exp(a)

	@staticmethod
	def log(a):
		return cmath.log10(a)

	@staticmethod
	def pow10(a):
		return 10 ** a

	@staticmethod
	def sin(x): return math.sin(x)

	@staticmethod
	def asin(x): return math.asin(x)

	@staticmethod
	def cos(x): return math.cos(x)

	@staticmethod
	def acos(x): return math.acos(x)

	@staticmethod
	def tan(x): return math.tan(x)

	@staticmethod
	def atan(x): return math.atan(x)

	@staticmethod
	def sinh(x): return math.sinh(x)

	@staticmethod
	def asinh(x): return math.asinh(x)

	@staticmethod
	def cosh(x): return math.cosh(x)

	@staticmethod
	def acosh(x): return math.acosh(x)

	@staticmethod
	def tanh(x): return math.tanh(x)

	@staticmethod
	def atanh(x): return math.atanh(x)

	@staticmethod
	def lcm(a, b):
		return math.lcm(int(a), int(b))

	@staticmethod
	def gcd(a, b):
		return math.gcd(int(a), int(b))

	@staticmethod
	def randint(low, high, count=1):
		if count == 1:
			return random.randint(low, high)
		return [random.randint(low, high) for _ in range(count)]

	@staticmethod
	def sub(s, start, length):
		return s[int(start) - 1 : int(start) - 1 + int(length)]

	@staticmethod
	def instring(s, sub):
		return float(s.find(sub) + 1) if sub in s else 0.0

	@staticmethod
	def randnorm(mu, sigma):
		return random.gauss(mu, sigma)

	@staticmethod
	def conj(a):
		return complex(a.real, -a.imag)

	@staticmethod
	def real(a): return a.real

	@staticmethod
	def imag(a): return a.imag

	@staticmethod
	def angle(a):
		return cmath.phase(a)

	@staticmethod
	def cumsum(a):
		return [sum(a[:i+1]) for i in range(len(a))]

	@staticmethod
	def length(s):
		return float(len(s))

	@staticmethod
	def delta_list(a):
		return [a[i+1] - a[i] for i in range(len(a) - 1)]

	@staticmethod
	def remainder(a, b):
		return float(int(a) % int(b))

	@staticmethod
	def logbase(a, b):
		return math.log(a) / math.log(b)

	@staticmethod
	def randintnotrep(a, b, n=None):
		return random.sample(range(int(a), int(b) + 1), int(b - a + 1) if n is None else int(n))

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
