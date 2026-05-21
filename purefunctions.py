import cmath
import builtins
import math
import operator
import random
import sys
from functools import wraps
from itertools import accumulate, pairwise, chain, repeat, batched
from math import prod
from tiobjects import (
	TiList, TiMatrix,
	require_num, require_real,
	require_list, require_matrix, require_str, require_int,
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


# ── Comparison operators ──────────────────────────────────────────────────────────

@vectorized
def eq(a, b):
	return int(require_real(a) == require_real(b))

@vectorized
def ne(a, b):
	return int(require_real(a) != require_real(b))

@vectorized
def lt(a, b):
	return int(require_real(a) < require_real(b))

@vectorized
def gt(a, b):
	return int(require_real(a) > require_real(b))

@vectorized
def le(a, b):
	return int(require_real(a) <= require_real(b))

@vectorized
def ge(a, b):
	return int(require_real(a) >= require_real(b))


# ── Arithmetic operators ──────────────────────────────────────────────────────────

@vectorized_with_matrix
def neg(x):
	return -x


_vec_add = vectorized(operator.add)
_vec_sub = vectorized(operator.sub)
_vec_mul = vectorized(operator.mul)
_vec_pow = vectorized(operator.pow)


def add(a, b):
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if (a.rows, a.cols) != (b.rows, b.cols):
			raise ValueError(f"add: dim mismatch {a.rows}×{a.cols} vs {b.rows}×{b.cols}")
		return TiMatrix([[a.inner[r][c] + b.inner[r][c] for c in range(a.cols)] for r in range(a.rows)])
	if isinstance(a, str) or isinstance(b, str):
		return a + b
	return _vec_add(a, b)


def sub(a, b):
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if (a.rows, a.cols) != (b.rows, b.cols):
			raise ValueError(f"sub: dim mismatch {a.rows}×{a.cols} vs {b.rows}×{b.cols}")
		return TiMatrix([[a.inner[r][c] - b.inner[r][c] for c in range(a.cols)] for r in range(a.rows)])
	return _vec_sub(a, b)


def mul(a, b):
	if isinstance(a, TiMatrix):
		return a @ b if isinstance(b, TiMatrix) else a.transform(lambda x: x * b)
	if isinstance(b, TiMatrix):
		return b.transform(lambda x: a * x)
	return _vec_mul(a, b)


@vectorized
def div(a, b):
	return a / b


def pow(a, b):
	if isinstance(a, TiMatrix):
		return a ** b
	return _vec_pow(a, b)


def _matrix_inv(m):
	n = m.rows
	if m.cols != n:
		raise ValueError(f"inv: matrix must be square, got {m.rows}×{m.cols}")
	aug = [m.inner[r].copy() + [1 if r == c else 0 for c in range(n)] for r in range(n)]
	for col in range(n):
		pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
		aug[col], aug[pivot] = aug[pivot], aug[col]
		p = aug[col][col]
		if abs(p) < 1e-12:
			raise ValueError("inv: matrix is singular")
		aug[col] = [x / p for x in aug[col]]
		for r in range(n):
			if r != col:
				f = aug[r][col]
				aug[r] = [aug[r][k] - f * aug[col][k] for k in range(2 * n)]
	return TiMatrix([row[n:] for row in aug])


def inv(x):
	if isinstance(x, TiMatrix):
		return _matrix_inv(x)
	return div(1, x)


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
def xth_root(x, n):
	return cmath.exp(cmath.log(x) / n) if isinstance(x, complex) or x < 0 else x ** (1 / n)


def cum_sum(lst):
	if isinstance(lst, TiMatrix):
		cols = lst.cols
		rows = lst.rows
		return TiMatrix([
			[builtins.sum(lst.inner[rr][c] for rr in range(r + 1))
			 for c in range(cols)]
			for r in range(rows)
		])
	return TiList(list(accumulate(require_list(lst))))


def delta_list(lst):
	return TiList([b - a for a, b in pairwise(require_list(lst))])


def augment(a, b):
	if isinstance(a, TiList) and isinstance(b, TiList):
		return TiList(a.inner + b.inner)
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if a.rows != b.rows:
			raise ValueError(f"Row count mismatch: {a.rows} vs {b.rows}")
		return TiMatrix([r1 + r2 for r1, r2 in zip(a.inner, b.inner)])
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

# Technically works on matrices, but since matrices can't store complex numbers, the result is all 0s
@vectorized
def angle(a):
	return cmath.phase(a)


# ── Sorting / filling ────────────────────────────────────────────────────────────

def sort_a(lst, *dep, reverse=False):
	inner = require_list(lst).inner
	indices = sorted(range(len(inner)), key=lambda i: inner[i], reverse=reverse)
	lst.inner = [inner[i] for i in indices]
	for d in dep:
		d.inner = [d.inner[i] for i in indices]


def sort_d(lst, *dep):
	sort_a(lst, *dep, reverse=True)


def fill(lst, x):
	require_num(x)
	if isinstance(lst, TiList):
		for i in range(len(lst.inner)):
			lst.inner[i] = x
	elif isinstance(lst, TiMatrix):
		for row in lst.inner:
			for i in range(len(row)):
				row[i] = x
	else:
		raise ValueError(f"fill: expected list or matrix, got {type(lst).__name__}")


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
	from fractions import Fraction
	x = require_real(x)
	f = Fraction(x).limit_denominator(10000)
	return float(f) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ── String functions ────────────────────────────────────────────────────────────

def in_string(value, substring):
	return require_str(value).find(require_str(substring)) + 1


def length(value):
	return len(require_str(value))


def sub(value, start, length):
	if isinstance(value, Number):
		return value / 100
	require_str(value)
	start, length = int(start), int(length)
	if length < 1:
		raise ValueError(f"sub: length must be ≥ 1, got {length}")
	if not (1 <= start <= len(value) - length + 1):
		raise ValueError(f"sub: index out of range")
	return value[start - 1 : start - 1 + length]

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
	m = [row.copy() for row in mat.inner]
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
	return TiMatrix([[mat.inner[r][c] for r in range(mat.rows)] for c in range(mat.cols)])


def sum(lst, start=None, end=None):
	inner = require_list(lst).inner
	if start is None:
		return builtins.sum(inner)
	start = require_int(start)
	end = require_int(end) if end is not None else len(inner)
	if not (1 <= start <= end <= len(inner)):
		raise ValueError(f"sum: index out of range (start={start}, end={end}, dim={len(inner)})")
	return builtins.sum(inner[start - 1 : end])


def prod(lst, start=None, end=None):
	inner = require_list(lst).inner
	if start is None:
		return math.prod(inner)
	start = require_int(start)
	end = require_int(end) if end is not None else len(inner)
	if not (1 <= start <= end <= len(inner)):
		raise ValueError(f"prod: index out of range (start={start}, end={end}, dim={len(inner)})")
	return math.prod(inner[start - 1 : end])


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
	('logbase',   math.log, cmath.log),
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


def randint(low, high, count=1):
	low, high = require_int(low), require_int(high)
	if count == 1:
		return random.randint(low, high)
	return TiList([random.randint(low, high) for _ in range(require_int(count))])


def randnorm(mu, sigma, n=None):
	if n is None:
		return random.gauss(mu, sigma)
	return TiList([random.gauss(mu, sigma) for _ in range(require_int(n))])


def randintnotrep(a, b):
	lst = list(range(require_int(a), require_int(b) + 1))
	random.shuffle(lst)
	return TiList(lst)


# ── Matrix row operations ────────────────────────────────────────────────────

def rowswap(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	r1, r2 = result.get_row(row1), result.get_row(row2)
	result.set_row(row1, r2)
	result.set_row(row2, r1)
	return result


def row_plus(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [a + b for a, b in zip(result.get_row(row2), result.get_row(row1))])
	return result


def times_row(factor, mat, row):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row, [factor * x for x in result.get_row(row)])
	return result


def times_row_plus(factor, mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [a + factor * b for a, b in zip(result.get_row(row2), result.get_row(row1))])
	return result


# ── ref / rref ───────────────────────────────────────────────────────────────

def ref(mat):
	require_matrix(mat)
	if mat.rows > mat.cols:
		raise ValueError(f"ref: matrix must have at least as many columns as rows")
	m = [row.copy() for row in mat.inner]
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
	m = [row.copy() for row in mat.inner]
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


# ── Matrix/list conversions ──────────────────────────────────────────────────

def matr_list(mat, *args):
	"""matr_list(mat, list1, list2, ...) or matr_list(mat, col_num, list)"""
	require_matrix(mat)
	if len(args) == 2 and isinstance(args[0], (int, float)):
		# Single-column extraction: matr_list(mat, col#, list_var)
		col = require_int(args[0])
		if not (1 <= col <= mat.cols):
			raise ValueError(f"matr_list: column {col} out of range")
		target = args[1]
		require_list(target)
		target.inner = [mat.inner[r][col - 1] for r in range(mat.rows)]
		return
	# Store successive columns into the provided lists
	for i, lst in enumerate(args):
		if i >= mat.cols:
			break
		require_list(lst)
		lst.inner = [mat.inner[r][i] for r in range(mat.rows)]


def list_matr(mat, *lists):
	"""list_matr(list1, list2, ..., mat) — the last arg is the matrix variable."""
	# Note: in TI-BASIC, List►matr(list1,...,mat) stores to mat
	require_matrix(mat)
	if not lists:
		raise ValueError("list_matr: need at least one list")
	cols = len(lists)
	max_rows = builtins.max(len(require_list(lst)) for lst in lists)
	mat.inner = [
		[lists[c].inner[r] if r < len(lists[c]) else 0.0 for c in range(cols)]
		for r in range(max_rows)
	]


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


def invnorm(p, mu=0, sigma=1):
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
	x = invnorm(p)
	for _ in range(50):
		fx = tcdf(-1e99, x, df) - p
		fpx = tpdf(x, df)
		if builtins.abs(fpx) < 1e-300:
			break
		dx = fx / fpx
		x -= dx
		if builtins.abs(dx) < 1e-12:
			break
	return x


def chi2pdf(x, df):
	require_real(x)
	require_real(df)
	if x <= 0:
		return 0.0
	k = df
	return math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))


def chi2cdf(lower, upper, df):
	require_real(lower)
	require_real(upper)
	require_real(df)
	def _cdf(x, k):
		if x <= 0:
			return 0.0
		return _regularized_inc_gamma(k / 2, x / 2)
	return _cdf(upper, df) - _cdf(lower, df)


def fpdf(x, df1, df2):
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
