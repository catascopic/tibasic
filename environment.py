import math, cmath, random, datetime as _dt



def autoparse(func):
	def wrapper(self, parser):
		return func(self, *parser.parse_args())
	return wrapper


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

	# ── Static resolve helpers (used by resolve= fields in tokens) ──────────────

	@staticmethod
	def get_date(env):
		t = _dt.date.today()
		return [float(t.year), float(t.month), float(t.day)]

	@staticmethod
	def get_time(env):
		t = _dt.datetime.now()
		return [float(t.hour), float(t.minute), float(t.second)]

	@staticmethod
	def start_tmr(env):
		return float(int(_dt.datetime.now().timestamp()))

	@staticmethod
	def get_dt_fmt(env):
		return getattr(env, '_dtfmt', 1.0)

	@staticmethod
	def get_tm_fmt(env):
		return getattr(env, '_tmfmt', 12.0)

	@staticmethod
	def is_clock_on(env):
		return 1.0

	@staticmethod
	def get_key(env):
		return getattr(env, '_getkey', 0.0)

	# ── Nullary / rand ──────────────────────────────────────────────────────────

	def rand(self):
		return random.random()

	# ── Autoparse function implementations ─────────────────────────────────────

	@autoparse
	def round(self, a, b=9):
		return round(a)

	@autoparse
	def max(self, *a):
		return max(a) if len(a) > 1 else max(a[0])

	@autoparse
	def min(self, *a):
		return min(a) if len(a) > 1 else min(a[0])

	@autoparse
	def median(self, a, b=None):
		return float(sorted(a)[len(a) // 2])

	@autoparse
	def mean(self, a, b=None):
		return sum(a) / len(a) if b is None else sum(x * w for x, w in zip(a, b)) / sum(b)

	@autoparse
	def int(self, x):
		return Environment.ti_int(x)

	@autoparse
	def abs(self, a):
		return abs(a)

	@autoparse
	def det(self, m):
		return float(sum(m[i][i] for i in range(len(m))))

	@autoparse
	def identity(self, n):
		n = int(n)
		return [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]

	@autoparse
	def dim(self, a):
		if isinstance(a, list) and not isinstance(a[0], list):
			return float(len(a))
		return [float(len(a)), float(len(a[0]))]

	@autoparse
	def sum(self, a):
		return sum(a)

	@autoparse
	def prod(self, a):
		return math.prod(a)

	@autoparse
	def not_(self, a):
		return float(not a)

	@autoparse
	def ipart(self, x):
		return math.trunc(x)

	@autoparse
	def fpart(self, x):
		return x - Environment.ti_int(x)

	@autoparse
	def sqrt(self, a):
		return cmath.sqrt(a) if a < 0 else math.sqrt(a)

	@autoparse
	def cbrt(self, a):
		return -(-a) ** (1/3) if a < 0 else a ** (1/3)

	@autoparse
	def ln(self, a):
		return cmath.log(a)

	@autoparse
	def exp(self, a):
		return cmath.exp(a)

	@autoparse
	def log(self, a):
		return cmath.log10(a)

	@autoparse
	def pow10(self, a):
		return 10 ** a

	@autoparse
	def sin(self, x):
		return math.sin(x)

	@autoparse
	def asin(self, x):
		return math.asin(x)

	@autoparse
	def cos(self, x):
		return math.cos(x)

	@autoparse
	def acos(self, x):
		return math.acos(x)

	@autoparse
	def tan(self, x):
		return math.tan(x)

	@autoparse
	def atan(self, x):
		return math.atan(x)

	@autoparse
	def sinh(self, x):
		return math.sinh(x)

	@autoparse
	def asinh(self, x):
		return math.asinh(x)

	@autoparse
	def cosh(self, x):
		return math.cosh(x)

	@autoparse
	def acosh(self, x): return math.acosh(x)

	@autoparse
	def tanh(self, x): return math.tanh(x)

	@autoparse
	def atanh(self, x): return math.atanh(x)

	@autoparse
	def lcm(self, a, b):
		return math.lcm(int(a), int(b))

	@autoparse
	def gcd(self, a, b):
		return math.gcd(int(a), int(b))

	@autoparse
	def randint(self, low, high, count=1):
		if count == 1:
			return random.randint(low, high)
		return [random.randint(low, high) for _ in range(count)]

	@autoparse
	def sub(self, s, start, length):
		return s[int(start) - 1 : int(start) - 1 + int(length)]

	@autoparse
	def instring(self, s, sub):
		return float(s.find(sub) + 1) if sub in s else 0.0

	@autoparse
	def randnorm(self, mu, sigma):
		return random.gauss(mu, sigma)

	@autoparse
	def conj(self, a):
		return complex(a.real, -a.imag)

	@autoparse
	def real(self, a):
		return a.real

	@autoparse
	def imag(self, a):
		return a.imag

	@autoparse
	def angle(self, a):
		return cmath.phase(a)

	@autoparse
	def cumsum(self, a):
		return [sum(a[:i+1]) for i in range(len(a))]

	@autoparse
	def length(self, s):
		return float(len(s))

	@autoparse
	def delta_list(self, a):
		return [a[i+1] - a[i] for i in range(len(a) - 1)]

	@autoparse
	def remainder(self, a, b):
		return float(int(a) % int(b))

	@autoparse
	def logbase(self, a, b):
		return math.log(a) / math.log(b)

	@autoparse
	def randintnotrep(self, a, b, n=None):
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


