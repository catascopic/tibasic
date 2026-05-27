import cmath
import builtins
import math
import operator
import random
import sys
from datetime import date
from fractions import Fraction
from functools import wraps
from itertools import accumulate, pairwise, chain, repeat, batched
from math import prod
from numbers import Number
from tiobjects import (
	TiList, TiMatrix,
	require_num, require_real,
	TiString, require_list, require_matrix, require_str, require_int,
)



# ── Decorators ────────────────────────────────────────────────────────────────────

def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply


def vectorized(func):
	"""Allow a scalar function to accept TiList arguments, applying element-wise."""
	@wraps(func)
	def apply(*args):
		len_check = set()
		vec = []
		for a in args:
			if isinstance(a, TiList):
				len_check.add(len(a))
				vec.append(a)
			else:
				vec.append(repeat(require_num(a)))
		if not len_check:
			return func(*args)
		if len(len_check) == 1:
			return TiList([func(*v) for v in zip(*vec)])
		raise ValueError(f"Dim mismatch: {len_check}")
	return apply


def vectorized_with_matrix(func):
	"""Like vectorized, but also applies element-wise to a TiMatrix first argument."""
	vec = vectorized(func)
	@wraps(func)
	def apply(a, *args):
		if isinstance(a, TiMatrix):
			return a.transform(lambda x: func(x, *args))
		return vec(a, *args)
	return apply


# ── Logical operators ────────────────────────────────────────────────────────────

@vectorized
def and_(a, b):
	return int(bool(require_real(a)) and bool(require_real(b)))

@vectorized
def or_(a, b):
	return int(bool(require_real(a)) or bool(require_real(b)))

@vectorized
def xor(a, b):
	return int(bool(require_real(a)) ^ bool(require_real(b)))


def inv(x):
	if isinstance(x, TiMatrix):
		return x.inv()
	return 1 / x


# ── dim ───────────────────────────────────────────────────────────────────────────

def dim(value):
	if isinstance(value, TiList):
		return len(value)
	if isinstance(value, TiMatrix):
		return TiList([value.rows, value.cols])
	raise ValueError(f"Invalid type: {type(value).__name__}; required: list or matrix")


# ── Numeric functions ────────────────────────────────────────────────────────────

@vectorized
def not_(x):
	return int(not require_real(x))

@vectorized_with_matrix
@handle_complex
def i_part(x):
	return math.trunc(x)

@vectorized_with_matrix
@handle_complex
def int_(x):
	return math.floor(x)

@vectorized_with_matrix
@handle_complex
def f_part(x):
	return x - math.trunc(x)

@vectorized
def sqrt(x):
	return cmath.sqrt(x) if isinstance(x, complex) or x < 0 else math.sqrt(x)

@vectorized
def cbrt(x):
	return cmath.exp(cmath.log(x) / 3) if isinstance(x, complex) else math.cbrt(x)

@vectorized
def xth_root(n, x):
	return cmath.exp(cmath.log(x) / n) if isinstance(x, complex) or x < 0 else x ** (1 / n)


def cum_sum(lst):
	if isinstance(lst, TiMatrix):
		cols = lst.cols
		rows = lst.rows
		# TODO: can be simplified with zip?
		return TiMatrix([
			[builtins.sum(lst.data[rr][c] for rr in range(r + 1))
				for c in range(cols)]
			for r in range(rows)
		])
	return TiList(list(accumulate(require_list(lst))))


def delta_list(lst):
	return TiList([b - a for a, b in pairwise(require_list(lst))])


def augment(a, b):
	if isinstance(a, TiList) and isinstance(b, TiList):
		return TiList(a.data + b.data)
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if a.rows != b.rows:
			raise ValueError(f"Row count mismatch: {a.rows} vs {b.rows}")
		return TiMatrix([r1 + r2 for r1, r2 in zip(a.data, b.data)])
	raise ValueError("augment: both args must be lists or both must be matrices")


@vectorized
def real(x):
	return x.real if isinstance(x, complex) else x

@vectorized
def imag(x):
	return x.imag if isinstance(x, complex) else 0

@vectorized
def conj(a):
	return complex(a.real, -a.imag) if isinstance(a, complex) else a

# Technically works on matrices, but since matrices can't store complex numbers, the result is all 0s.
# TiBasicDev thinks this is basically a bug, so I'm not implementing it in order to discourage it.
# (If you want a matrix of all 0s, you can just do 0[A].)
@vectorized
def angle(a):
	return cmath.phase(a)


# ── Converters (►DMS, ►Dec, ►Frac) ─────────────────────────────────────────────

def to_dms(x):
	x = require_real(x)
	neg = x < 0
	x = abs(x)
	deg = int(x)
	rem = (x - deg) * 60
	mins = int(rem)
	secs = (rem - mins) * 60
	sign = "-" if neg else ""
	return f"{sign}{deg}°{mins}'{_repr_num(secs)}\""

def to_dec(x):
	return require_real(x)

def to_frac(x):
	x = require_real(x)
	f = Fraction(x).limit_denominator(10000)
	return float(f) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ── String functions ────────────────────────────────────────────────────────────

def in_string(string, substring, start=1):
	v = require_str(string).tokens
	s = require_str(substring).tokens
	start = require_int(start) - 1
	for i in range(start, len(v) - len(s) + 1):
		if v[i:i + len(s)] == s:
			return i + 1
	return 0


def length(string):
	return len(require_str(string))


def sub_string(*args):
	if len(args) == 1:
		return require_num(args[0]) / 100
	if len(args) == 3:
		string, start, length = args
		require_str(string)
		start = require_int(start)
		length = require_int(length)
		if length < 1:
			raise ValueError(f"sub: length must be ≥ 1, got {length}")
		if not (1 <= start <= len(string) - length + 1):
			raise ValueError(f"sub: index out of range")
		return TiString(string.tokens[start - 1 : start - 1 + length])
	raise ValueError(f"Invalid arguments: {args}")

# ── Aggregate / statistics ───────────────────────────────────────────────────────

def variance(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		n = len(lst)
		if n < 2:
			raise ValueError("stdDev: need at least 2 elements")
		m = mean(lst)
		return builtins.sum((x - m) ** 2 for x in lst) / (n - 1)
	require_list(freqlist)
	if len(lst) != len(freqlist):
		raise ValueError("stdDev: dim mismatch")
	m = mean(lst, freqlist)
	total_w = builtins.sum(freqlist)
	if total_w <= 1:
		raise ValueError("stdDev: total frequency must be > 1")
	return builtins.sum(w * (x - m) ** 2 for x, w in zip(lst, freqlist)) / (total_w - 1)


def stddev(lst, freqlist=None):
	return math.sqrt(variance(lst, freqlist))


@vectorized_with_matrix
def round(a, b=9):
	return builtins.round(a, require_int(b))


def _minmax(fn, a, b):
	if b is None:
		return fn(require_list(a))
	if isinstance(a, TiList) and isinstance(b, TiList):
		if len(a) != len(b):
			raise ValueError(f"{fn.__name__}: dim mismatch ({len(a)} vs {len(b)})")
		return TiList([fn(x, y) for x, y in zip(a, b)])
	if isinstance(a, Number) and isinstance(b, Number):
		return fn(a, b)
	raise ValueError(f"{fn.__name__}: both args must be the same type (both numeric or both list)")


def max(a, b=None):
	return _minmax(builtins.max, a, b)


def min(a, b=None):
	return _minmax(builtins.min, a, b)


def median(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		s = sorted(lst)
		n = len(s)
		if n == 0:
			raise ValueError("median: empty list")
		mid = n // 2
		return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

	require_list(freqlist)
	if len(lst) != len(freqlist):
		raise ValueError("median: dim mismatch")
	pairs = sorted(zip(lst, freqlist), key=lambda p: p[0])
	total = builtins.sum(require_int(f) for _, f in pairs)
	if total <= 0:
		raise ValueError("median: total frequency must be positive")

	def nth(n):
		count = 0
		for value, freq in pairs:
			count += int(freq)
			if n < count:
				return value

	if total % 2:
		return nth(total // 2)
	return (nth(total // 2 - 1) + nth(total // 2)) / 2


def mean(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		return builtins.sum(lst) / len(lst)
	require_list(freqlist)
	return builtins.sum(x * w for x, w in zip(lst, freqlist)) / builtins.sum(freqlist)


@vectorized_with_matrix
def abs(a):
	return builtins.abs(a)


def det(mat):
	require_matrix(mat)
	n = mat.rows
	if n == 0 or n != mat.cols:
		raise ValueError(f"det requires a square matrix, got {mat.rows}×{mat.cols}")
	m = [row.copy() for row in mat.data]
	sign = 1.0
	for col in range(n):
		pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
		if pivot is None:
			return 0.0
		if pivot != col:
			m[col], m[pivot] = m[pivot], m[col]
			sign = -sign
		for row in range(col + 1, n):
			if m[row][col] != 0:
				f = m[row][col] / m[col][col]
				for j in range(col, n):
					m[row][j] -= f * m[col][j]
	return sign * math.prod(m[i][i] for i in range(n))


def identity(n):
	n = require_int(n)
	return TiMatrix([[1 if r == c else 0 for c in range(n)] for r in range(n)])


def transpose(mat):
	require_matrix(mat)
	return TiMatrix([[mat.data[r][c] for r in range(mat.rows)] for c in range(mat.cols)])


def sum(lst, start=None, end=None):
	data = require_list(lst).data
	if start is None:
		return builtins.sum(data)
	start = require_int(start)
	end = require_int(end) if end is not None else len(data)
	if not (1 <= start <= end <= len(data)):
		raise ValueError(f"sum: index out of range (start={start}, end={end}, dim={len(data)})")
	return builtins.sum(data[start - 1 : end])


def prod(lst, start=None, end=None):
	data = require_list(lst).data
	if start is None:
		return math.prod(data)
	start = require_int(start)
	end = require_int(end) if end is not None else len(data)
	if not (1 <= start <= end <= len(data)):
		raise ValueError(f"prod: index out of range (start={start}, end={end}, dim={len(data)})")
	return math.prod(data[start - 1 : end])


# ── Transcendental functions ────────────────────────────────────────────────────

@vectorized
def pow10(a):
	return 10 ** a


def _make_dispatch(name, real_fn, cpx_fn):
	def fn(x):
		return cpx_fn(x) if isinstance(x, complex) else real_fn(x)
	fn.__name__ = fn.__qualname__ = name
	return vectorized(fn)

for _name, _real_fn, _cpx_fn in [
	('sin',   math.sin,   cmath.sin),
	('asin',  math.asin,  cmath.asin),
	('cos',   math.cos,   cmath.cos),
	('acos',  math.acos,  cmath.acos),
	('tan',   math.tan,   cmath.tan),
	('atan',  math.atan,  cmath.atan),
	('sinh',  math.sinh,  cmath.sinh),
	('asinh', math.asinh, cmath.asinh),
	('cosh',  math.cosh,  cmath.cosh),
	('acosh', math.acosh, cmath.acosh),
	('tanh',  math.tanh,  cmath.tanh),
	('atanh', math.atanh, cmath.atanh),
	('ln',    math.log,   cmath.log),
	('exp',   math.exp,   cmath.exp),
	('log',   math.log10, cmath.log10),
	('log_base', math.log, cmath.log),
]:
	globals()[_name] = _make_dispatch(_name, _real_fn, _cpx_fn)


# ── Integer / combinatorics ─────────────────────────────────────────────────────

def factorial(n):
	n = require_int(n)
	if n < 0:
		raise ValueError("Argument to ! must be a non-negative integer")
	return math.factorial(n)

def ncr(n, r):
	return math.comb(require_int(n), require_int(r))

def npr(n, r):
	return math.perm(require_int(n), require_int(r))

@vectorized
def lcm(a, b):
	return math.lcm(require_int(a), require_int(b))

@vectorized
def gcd(a, b):
	return math.gcd(require_int(a), require_int(b))

@vectorized
def remainder(a, b):
	return require_int(a) % require_int(b)

# ── Random ──────────────────────────────────────────────────────────────────────

def rand_list(n):
	return TiList([random.random() for _ in range(require_int(n))])


def rand_int(low, high, count=1):
	low, high = require_int(low), require_int(high)
	if count == 1:
		return random.randint(low, high)
	return TiList([random.randint(low, high) for _ in range(require_int(count))])


def rand_norm(mu, sigma, n=None):
	if n is None:
		return random.gauss(mu, sigma)
	return TiList([random.gauss(mu, sigma) for _ in range(require_int(n))])


def rand_int_no_rep(a, b):
	lst = list(range(require_int(a), require_int(b) + 1))
	random.shuffle(lst)
	return TiList(lst)


# ── Matrix row operations ────────────────────────────────────────────────────

def row_swap(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row1, mat.get_row(row2))
	result.set_row(row2, mat.get_row(row1))
	return result


def row_plus(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [a + b for a, b in zip(mat.get_row(row2), mat.get_row(row1))])
	return result


def times_row(factor, mat, row):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row, [factor * x for x in mat.get_row(row)])
	return result


def times_row_plus(factor, mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [factor * a + b for a, b in zip(mat.get_row(row1), mat.get_row(row2))])
	return result


# ── ref / rref ───────────────────────────────────────────────────────────────

def ref(mat):
	require_matrix(mat)
	if mat.rows > mat.cols:
		raise ValueError(f"ref: matrix must have at least as many columns as rows")
	m = [row.copy() for row in mat.data]
	rows, cols = mat.rows, mat.cols
	pivot_row = 0
	for col in range(cols):
		if pivot_row >= rows:
			break
		# Find pivot
		pivot = next((r for r in range(pivot_row, rows) if m[r][col] != 0), None)
		if pivot is None:
			continue
		m[pivot_row], m[pivot] = m[pivot], m[pivot_row]
		p = m[pivot_row][col]
		m[pivot_row] = [x / p for x in m[pivot_row]]
		for r in range(pivot_row + 1, rows):
			if m[r][col] != 0:
				f = m[r][col]
				m[r] = [m[r][k] - f * m[pivot_row][k] for k in range(cols)]
		pivot_row += 1
	return TiMatrix(m)


def rref(mat):
	require_matrix(mat)
	if mat.rows > mat.cols:
		raise ValueError(f"rref: matrix must have at least as many columns as rows")
	m = [row.copy() for row in mat.data]
	rows, cols = mat.rows, mat.cols
	pivot_row = 0
	for col in range(cols):
		if pivot_row >= rows:
			break
		pivot = next((r for r in range(pivot_row, rows) if m[r][col] != 0), None)
		if pivot is None:
			continue
		m[pivot_row], m[pivot] = m[pivot], m[pivot_row]
		p = m[pivot_row][col]
		m[pivot_row] = [x / p for x in m[pivot_row]]
		for r in range(rows):
			if r != pivot_row and m[r][col] != 0:
				f = m[r][col]
				m[r] = [m[r][k] - f * m[pivot_row][k] for k in range(cols)]
		pivot_row += 1
	return TiMatrix(m)


# ── Coordinate conversions ───────────────────────────────────────────────────

@vectorized
def r_pr(x, y):
	return math.hypot(require_real(x), require_real(y))


@vectorized
def r_ptheta(x, y):
	return math.atan2(require_real(y), require_real(x))


@vectorized
def p_rx(r, theta):
	return require_real(r) * math.cos(require_real(theta))


@vectorized
def p_ry(r, theta):
	return require_real(r) * math.sin(require_real(theta))


# ── randM / randBin ──────────────────────────────────────────────────────────

def rand_m(rows, cols):
	rows = require_int(rows)
	cols = require_int(cols)
	if not (1 <= rows <= 99) or not (1 <= cols <= 99):
		raise ValueError("randM: dimensions must be 1-99")
	# Per spec: entries are successive randInt(-9,9) calls filled bottom-right to top-left
	data = [random.randint(-9, 9) for _ in range(rows * cols)]
	return TiMatrix([list(row) for row in batched(reversed(data), cols)])


def rand_bin(n, p, simulations=None):
	n = require_int(n)
	if not (0 <= p <= 1):
		raise ValueError("randBin: p must be in [0, 1]")
	if simulations is None:
		return builtins.sum(1 for _ in range(n) if random.random() < p)
	return TiList([builtins.sum(1 for _ in range(n) if random.random() < p) for _ in range(require_int(simulations))])


# ── Date / time utilities ────────────────────────────────────────────────────

def timecnv(seconds):
	"""Convert a number of seconds into {days, hours, minutes, seconds}."""
	seconds = require_int(seconds)
	sign = -1 if seconds < 0 else 1
	s, secs = divmod(abs(seconds), 60)
	s, minutes = divmod(s, 60)
	days, hours = divmod(s, 24)
	return sign * TiList([days, hours, minutes, secs])


def dayofwk(year, month, day):
	"""Day of week: 1=Sunday, 2=Monday, …, 7=Saturday."""
	d = date(require_int(year), require_int(month), require_int(day))
	return d.isoweekday() % 7 + 1


def _parse_dbd_date(d):
	"""Parse a TI Finance date float into a date object.

	Two formats (can be mixed in the same dbd call):
	  MM.DDYY  — integer part is month (1–12); decimal encodes 4 digits DDYY
	  DDMM.YY  — integer part is DDMM (≥100); decimal encodes 2 digits YY
	YY 00–49 → 2000–2049; 50–99 → 1950–1999.
	ERR:DOMAIN if the integer part is 13–99, or the decimal has too many digits.
	"""
	d = require_real(d)
	int_part = int(d)
	frac_part = d - int_part

	if int_part <= 12:
		# MM.DDYY: 4 decimal digits expected
		raw = frac_part * 10000
		ddyy = builtins.round(raw)
		if abs(raw - ddyy) > 1e-6:
			raise ValueError(f"dbd: too many decimal places in MM.DDYY date {d!r}")
		month = int_part
		day, yy = divmod(ddyy, 100)
	elif int_part >= 100:
		# DDMM.YY: 2 decimal digits expected
		raw = frac_part * 100
		yy  = builtins.round(raw)
		if abs(raw - yy) > 1e-6:
			raise ValueError(f"dbd: too many decimal places in DDMM.YY date {d!r}")
		day, month = divmod(int_part, 100)
	else:
		raise ValueError(f"dbd: invalid date {d!r} (integer part {int_part} is ambiguous: must be ≤12 or ≥100)")

	year = (2000 if yy < 50 else 1900) + yy
	return date(year, month, day)


@vectorized
def dbd(date1: float, date2: float):
	"""Days between two dates in TI Finance format (MM.DDYY or DDMM.YY)."""
	return (_parse_dbd_date(date2) - _parse_dbd_date(date1)).days


# ── Probability distributions ────────────────────────────────────────────────

def _regularized_inc_gamma(a, x):
	"""Lower regularized incomplete gamma function P(a, x) via series."""
	if x < 0:
		raise ValueError("x must be >= 0")
	if x == 0:
		return 0.0
	# Use series representation for x < a+1, continued fraction otherwise
	if x < a + 1:
		# Series
		term = 1.0 / a
		total = term
		for k in range(1, 300):
			term *= x / (a + k)
			total += term
			if builtins.abs(term) < builtins.abs(total) * 1e-15:
				break
		return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
	else:
		# Continued fraction (Lentz)
		FPMIN = 1e-300
		b = x + 1 - a
		c = 1 / FPMIN
		d = 1 / b
		h = d
		for i in range(1, 300):
			an = -i * (i - a)
			b += 2
			d = an * d + b
			if builtins.abs(d) < FPMIN:
				d = FPMIN
			c = b + an / c
			if builtins.abs(c) < FPMIN:
				c = FPMIN
			d = 1 / d
			delta = d * c
			h *= delta
			if builtins.abs(delta - 1) < 1e-15:
				break
		return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _inc_beta(a, b, x):
	"""Regularized incomplete beta function I_x(a,b)."""
	if x < 0 or x > 1:
		raise ValueError("x must be in [0,1]")
	if x == 0:
		return 0.0
	if x == 1:
		return 1.0
	lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
	# Use symmetry relation if needed for convergence
	if x > (a + 1) / (a + b + 2):
		return 1.0 - _inc_beta(b, a, 1 - x)
	# Continued fraction
	FPMIN = 1e-300
	qab = a + b
	qap = a + 1
	qam = a - 1
	c = 1.0
	d = 1.0 - qab * x / qap
	if builtins.abs(d) < FPMIN:
		d = FPMIN
	d = 1 / d
	h = d
	for m in range(1, 300):
		m2 = 2 * m
		aa = m * (b - m) * x / ((qam + m2) * (a + m2))
		d = 1 + aa * d
		if builtins.abs(d) < FPMIN:
			d = FPMIN
		c = 1 + aa / c
		if builtins.abs(c) < FPMIN:
			c = FPMIN
		d = 1 / d
		h *= d * c
		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
		d = 1 + aa * d
		if builtins.abs(d) < FPMIN:
			d = FPMIN
		c = 1 + aa / c
		if builtins.abs(c) < FPMIN:
			c = FPMIN
		d = 1 / d
		delta = d * c
		h *= delta
		if builtins.abs(delta - 1) < 1e-15:
			break
	return math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * h / a


def normalpdf(x, mu=0, sigma=1):
	require_real(x)
	z = (x - mu) / sigma
	return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def normalcdf(lower, upper, mu=0, sigma=1):
	require_real(lower)
	require_real(upper)
	def _cdf(z):
		return 0.5 * (1 + math.erf(z / math.sqrt(2)))
	return _cdf((upper - mu) / sigma) - _cdf((lower - mu) / sigma)


def inv_norm(p, mu=0, sigma=1):
	require_real(p)
	if p <= 0:
		return -1e99
	if p >= 1:
		return 1e99
	# Rational approximation (Abramowitz & Stegun / Beasley-Springer-Moro)
	def _inv_std(q):
		# Beasley-Springer-Moro algorithm
		a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
		b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
		c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
			 0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
			 0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
		if 0.08 <= q <= 0.92:
			r = q - 0.5
			s = r * r
			return r * (a[0] + s * (a[1] + s * (a[2] + s * a[3]))) / \
			       (1 + s * (b[0] + s * (b[1] + s * (b[2] + s * b[3]))))
		else:
			r = math.sqrt(-math.log(q if q < 0.5 else 1 - q))
			x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
			return x if q >= 0.5 else -x
	z = _inv_std(p)
	return mu + sigma * z


def tpdf(t, df):
	require_real(t)
	require_real(df)
	log_coeff = math.lgamma((df + 1) / 2) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2)
	return math.exp(log_coeff - (df + 1) / 2 * math.log(1 + t * t / df))


def tcdf(lower, upper, df):
	require_real(lower)
	require_real(upper)
	require_real(df)
	def _t_cdf(x, v):
		if x == 0:
			return 0.5
		z = v / (v + x * x)
		ib = _inc_beta(v / 2, 0.5, z)
		if x > 0:
			return 1 - 0.5 * ib
		else:
			return 0.5 * ib
	return _t_cdf(upper, df) - _t_cdf(lower, df)


def invt(p, df):
	require_real(p)
	require_real(df)
	if p <= 0:
		return -1e99
	if p >= 1:
		return 1e99
	# Newton's method starting from normal approximation
	x = inv_norm(p)
	for _ in range(50):
		fx = tcdf(-1e99, x, df) - p
		fpx = tpdf(x, df)
		# TODO: calculator doesn't go below 1e-99
		if builtins.abs(fpx) < 1e-300:
			break
		dx = fx / fpx
		x -= dx
		if builtins.abs(dx) < 1e-12:
			break
	return x


def chi_sq_pdf(x, df):
	require_real(x)
	require_real(df)
	if x <= 0:
		return 0.0
	k = df
	return math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))


def chi_sq_cdf(lower, upper, df):
	require_real(lower)
	require_real(upper)
	require_real(df)
	def _cdf(x, k):
		if x <= 0:
			return 0.0
		return _regularized_inc_gamma(k / 2, x / 2)
	return _cdf(upper, df) - _cdf(lower, df)


def f_pdf(x, df1, df2):
	require_real(x)
	require_real(df1)
	require_real(df2)
	if x <= 0:
		return 0.0
	d1, d2 = df1, df2
	log_num = (d1 / 2) * math.log(d1 * x) + (d2 / 2) * math.log(d2) - ((d1 + d2) / 2) * math.log(d1 * x + d2)
	log_den = math.log(x) + math.lgamma(d1 / 2) + math.lgamma(d2 / 2) - math.lgamma((d1 + d2) / 2)
	return math.exp(log_num - log_den)


def fcdf(lower, upper, df1, df2):
	require_real(lower)
	require_real(upper)
	require_real(df1)
	require_real(df2)
	def _cdf(x, d1, d2):
		if x <= 0:
			return 0.0
		z = d1 * x / (d1 * x + d2)
		return _inc_beta(d1 / 2, d2 / 2, z)
	return _cdf(upper, df1, df2) - _cdf(lower, df1, df2)


def binompdf(n, p, k=None):
	n = require_int(n)
	require_real(p)
	if k is None:
		return TiList([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
	k = require_int(k)
	return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binomcdf(n, p, k=None):
	n = require_int(n)
	require_real(p)
	if k is None:
		acc = 0.0
		result = []
		for i in range(n + 1):
			acc += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
			result.append(acc)
		return TiList(result)
	k = require_int(k)
	return builtins.sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def poissonpdf(lam, k):
	require_real(lam)
	k = require_int(k)
	return math.exp(-lam) * lam ** k / math.factorial(k)


def poissoncdf(lam, k):
	require_real(lam)
	k = require_int(k)
	return builtins.sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


def geometpdf(p, n):
	require_real(p)
	n = require_int(n)
	return p * (1 - p) ** (n - 1)


def geometcdf(p, n):
	require_real(p)
	n = require_int(n)
	return 1 - (1 - p) ** n
