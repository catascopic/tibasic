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
	TiList, TiMatrix, TiString,
	require_num, require_real, require_int,
	require_list, require_matrix, require_str, py_int,
)
from errors import (
	DataTypeError, DimMismatchError, InvalidDimError,
	DomainError, StatError, ArgumentError,
)
from decorators import pure_func, pure_vectorized, vectorized


# ── Helpers ───────────────────────────────────────────────────────────────────

def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply


def matrix_vectorized(func):
	"""Like pure_vectorized, but also applies element-wise to a TiMatrix first argument."""
	vec = vectorized(func)
	@wraps(func)
	def apply(a, *args):
		if isinstance(a, TiMatrix):
			return a.transform(lambda x: func(x, *args))
		return vec(a, *args)
	return pure_func(apply)


@pure_vectorized
def not_(x):
	return float(not require_real(x))

##################
# MAIN FUNCTIONS #
##################

@pure_vectorized
def pow10(x):
	return 10 ** require_num(x)

@pure_vectorized
def exp(x):
	require_num(x)
	return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

##################
# MATH FUNCTIONS #
##################

@pure_vectorized
def cbrt(x):
	require_num(x)
	if isinstance(x, complex):
		if x == 0:
			return 0
		return cmath.exp(cmath.log(x) / 3)
	return math.cbrt(x)

@matrix_vectorized
def abs(x):
	return builtins.abs(require_num(x))

@matrix_vectorized
def round(x, decimals=9):
	return builtins.round(require_num(x), py_int(decimals))

@matrix_vectorized
@handle_complex
def i_part(x):
	return float(math.trunc(require_num(x)))

@matrix_vectorized
@handle_complex
def f_part(x):
	return x - math.trunc(require_num(x))

@matrix_vectorized
@handle_complex
def int_(x):
	return float(math.floor(require_num(x)))

def _minmax(fn, a, b):
	if b is None:
		return fn(require_list(a))
	if isinstance(a, TiList) and isinstance(b, TiList):
		if len(a) != len(b):
			raise DimMismatchError(f"{fn.__name__}: dim mismatch ({len(a)} vs {len(b)})")
		return TiList([fn(x, y) for x, y in zip(a, b)])
	if isinstance(a, Number) and isinstance(b, Number):
		return fn(a, b)
	raise DataTypeError(f"{fn.__name__}: both args must be the same type (both numeric or both list)")

@pure_func
def min(a, b=None):
	return _minmax(builtins.min, a, b)

@pure_func
def max(a, b=None):
	return _minmax(builtins.max, a, b)

@pure_vectorized
def lcm(a, b):
	return float(math.lcm(py_int(a), py_int(b)))

@pure_vectorized
def gcd(a, b):
	return float(math.gcd(py_int(a), py_int(b)))

@pure_vectorized
def remainder(a, b):
	require_int(a)
	require_int(b)
	if a < 0:
		raise DomainError(f"a must be non-negative but got {a}")
	if b < 1:
		raise DomainError(f"b must be positive but got {b}")
	return a % b

@pure_vectorized
def conj(x):
	require_num(x)
	return complex(x.real, -x.imag) if isinstance(x, complex) else x

@pure_vectorized
def real(x):
	require_num(x)
	return x.real if isinstance(x, complex) else x

@pure_vectorized
def imag(x):
	require_num(x)
	return x.imag if isinstance(x, complex) else 0

# Technically works on matrices, but since matrices can't store complex numbers, the result is all 0s.
# TiBasicDev thinks this is basically a bug, so I'm not implementing it in order to discourage it.
# (If you want a matrix of all 0s, you can just do 0[A].)
@pure_vectorized
def angle(x):
	require_num(x)
	return cmath.phase(x)

@pure_func
def rand_list(n):
	return TiList([random.random() for _ in range(py_int(n))])

@vectorized
def _rand_int_single(low, high):
	if low > high:
		raise DomainError(f"randInt: low must be ≤ high, got {low} > {high}")
	return float(random.randint(py_int(low), py_int(high)))

@pure_func
def rand_int(low, high, n=None):
	if n is None:
		return _rand_int_single(low, high)
	low = py_int(low)
	high = py_int(high)
	if low > high:
		raise DomainError(f"randInt: low must be ≤ high, got {low} > {high}")
	n = py_int(n)
	return TiList([float(random.randint(low, high)) for _ in range(n)])

@pure_func
def rand_norm(mu, sigma, n=None):
	require_real(mu)
	require_real(sigma)
	if n is None:
		return random.gauss(mu, sigma)
	return TiList([random.gauss(mu, sigma) for _ in range(py_int(n))])

@pure_func
def rand_bin(n, p, simulations=None):
	n = py_int(n)
	if not (0 <= p <= 1):
		raise DomainError("randBin: p must be in [0, 1]")
	if n <= 0:
		raise DomainError("randBin: n must be positive")
	if simulations is None:
		return builtins.sum(1 for _ in range(n) if random.random() < p)
	simulations = py_int(simulations)
	return TiList([builtins.sum(1 for _ in range(n) if random.random() < p) for _ in range(simulations)])

@pure_func
def rand_int_no_rep(low, high):
	lst = list(range(py_int(low), py_int(high) + 1))
	random.shuffle(lst)
	return TiList(lst)

##################
# LIST FUNCTIONS #
##################

@pure_func
def cum_sum(obj):
	if isinstance(obj, TiMatrix):
		cols = obj.cols
		rows = obj.rows
		# TODO: can be simplified with zip?
		return TiMatrix([
			[builtins.sum(obj.data[rr][c] for rr in range(r + 1))
				for c in range(cols)]
			for r in range(rows)
		])
	if isinstance(obj, TiList):
		return TiList(list(accumulate(obj.data)))
	raise DataTypeError(f"Expected list or matrix; got {obj}")

@pure_func
def delta_list(lst):
	return TiList([b - a for a, b in pairwise(require_list(lst))])

@pure_func
def augment(a, b):
	if isinstance(a, TiList) and isinstance(b, TiList):
		return TiList(a.data + b.data)
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if a.rows != b.rows:
			raise DimMismatchError(f"Row count mismatch: {a.rows} vs {b.rows}")
		return TiMatrix([r1 + r2 for r1, r2 in zip(a.data, b.data)])
	raise DataTypeError(f"augment: both args must be lists or both must be matrices; got {a}, {b}")

@pure_func
def mean(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		return builtins.sum(lst) / len(lst)
	require_list(freqlist)
	return builtins.sum(x * w for x, w in zip(lst, freqlist)) / builtins.sum(freqlist)

@pure_func
def median(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		sorted_data = sorted(lst)
		n = len(sorted_data)
		mid = n // 2
		return sorted_data[mid] if n % 2 else (sorted_data[mid - 1] + sorted_data[mid]) / 2

	require_list(freqlist)
	if len(lst) != len(freqlist):
		raise DimMismatchError("median: dim mismatch")
	pairs = sorted(zip(lst, freqlist), key=lambda p: p[0])
	total = builtins.sum(require_int(f) for _, f in pairs)
	if total <= 0:
		raise StatError("median: total frequency must be positive")

	def nth(n):
		count = 0
		for value, freq in pairs:
			count += int(freq)
			if n < count:
				return value

	if total % 2:
		return nth(total // 2)
	return (nth(total // 2 - 1) + nth(total // 2)) / 2

@pure_func
def sum(lst, start=None, end=None):
	data = require_list(lst).data
	if start is None:
		return builtins.sum(data)

	start = py_int(start)
	end = len(data) if end is None else py_int(end)
	if not (1 <= start <= end <= len(data)):
		raise InvalidDimError(f"sum: index out of range (start={start}, end={end}, dim={len(data)})")

	return builtins.sum(data[start - 1 : end])

@pure_func
def prod(lst, start=None, end=None):
	data = require_list(lst).data
	if start is None:
		return math.prod(data)

	start = py_int(start)
	end = len(data) if end is None else py_int(end)
	if not (1 <= start <= end <= len(data)):
		raise InvalidDimError(f"prod: index out of range (start={start}, end={end}, dim={len(data)})")

	return math.prod(data[start - 1 : end])

@pure_func
def variance(lst, freqlist=None):
	require_list(lst)
	if freqlist is None:
		n = len(lst)
		if n < 2:
			raise StatError("stdDev: need at least 2 elements")
		m = mean(lst)
		return builtins.sum((x - m) ** 2 for x in lst) / (n - 1)
	require_list(freqlist)
	if len(lst) != len(freqlist):
		raise DimMismatchError("stdDev: dim mismatch")
		
	m = mean(lst, freqlist)
	total_w = builtins.sum(freqlist)
	if total_w <= 1:
		raise StatError("stdDev: total frequency must be > 1")

	return builtins.sum(w * (x - m) ** 2 for x, w in zip(lst, freqlist)) / (total_w - 1)

@pure_func
def stddev(lst, freqlist=None):
	return math.sqrt(variance(lst, freqlist))

####################
# MATRIX FUNCTIONS #
####################

@pure_func
def det(mat):
	require_matrix(mat)
	n = mat.rows
	if n == 0 or n != mat.cols:
		raise InvalidDimError(f"det requires a square matrix, got {mat.rows}×{mat.cols}")
	work = [row.copy() for row in mat.data]
	sign = 1.0
	for col in range(n):
		pivot = next((r for r in range(col, n) if work[r][col] != 0), None)
		if pivot is None:
			return 0.0
		if pivot != col:
			work[col], work[pivot] = work[pivot], work[col]
			sign = -sign
		for row in range(col + 1, n):
			if work[row][col] != 0:
				factor = work[row][col] / work[col][col]
				for j in range(col, n):
					work[row][j] -= factor * work[col][j]

	return sign * math.prod(work[i][i] for i in range(n))

@pure_func
def identity(n):
	size = py_int(n)
	return TiMatrix([[float(r == c) for c in range(size)] for r in range(size)])

@pure_func
def rand_m(rows, cols):
	rows = py_int(rows)
	cols = py_int(cols)
	if not (1 <= rows <= 99) or not (1 <= cols <= 99):
		raise InvalidDimError("randM: dimensions must be 1-99")

	# Per spec: entries are successive randInt(-9,9) calls filled bottom-right to top-left
	data = [float(random.randint(-9, 9)) for _ in range(rows * cols)]
	return TiMatrix([list(row) for row in batched(reversed(data), cols)])

def _row_reduce(mat, get_range):
	require_matrix(mat)
	if mat.rows > mat.cols:
		raise InvalidDimError(f"ref/rref: matrix must have at least as many columns as rows")
	result = [row.copy() for row in mat.data]
	pivot_row = 0
	for col in range(mat.cols):
		if pivot_row >= mat.rows:
			break
		try:
			swap_row = next((r for r in range(pivot_row, mat.rows) if result[r][col] != 0))
		except StopIteration:
			continue

		result[pivot_row], result[swap_row] = result[swap_row], result[pivot_row]
		pivot = result[pivot_row]
		scale = pivot[col]
		for k in range(mat.cols):
			pivot[k] /= scale
		for r in get_range(pivot_row, mat.rows):
			if r != pivot_row and result[r][col] != 0:
				pivot = result[pivot_row]
				current = result[r]
				factor = current[col]
				for k in range(mat.cols):
					current[k] -= factor * pivot[k]

		pivot_row += 1
	return TiMatrix(result)

@pure_func
def ref(mat):
	return _row_reduce(mat, lambda pivot_row, rows: range(pivot_row + 1, rows))

@pure_func
def rref(mat):
	return _row_reduce(mat, lambda pivot_row, rows: range(rows))

@pure_func
def row_swap(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row1, mat.get_row(row2))
	result.set_row(row2, mat.get_row(row1))
	return result

@pure_func
def row_plus(mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [a + b for a, b in zip(mat.get_row(row2), mat.get_row(row1))])
	return result

@pure_func
def times_row(factor, mat, row):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row, [factor * x for x in mat.get_row(row)])
	return result

@pure_func
def times_row_plus(factor, mat, row1, row2):
	require_matrix(mat)
	result = mat.copy()
	result.set_row(row2, [factor * a + b for a, b in zip(mat.get_row(row1), mat.get_row(row2))])
	return result


####################
# STRING FUNCTIONS #
####################

@pure_func
def length(string):
	return len(require_str(string))

@pure_func
def in_string(string, substring, start=1):
	v = require_str(string).tokens
	s = require_str(substring).tokens
	start = py_int(start)
	for i in range(start - 1, len(v) - len(s) + 1):
		if v[i:i + len(s)] == s:
			return i + 1
	return 0

@pure_func
def sub(*args):
	# DO NOT REMOVE THIS!
	# This is a weird feature of sub(, but it's true: with a single numeric
	# argument, sub( divides it by 100 like the undocumented % operator.
	if len(args) == 1:
		return require_num(args[0]) / 100
	if len(args) == 3:
		string, start, length = args
		require_str(string)
		start = py_int(start)
		length = py_int(length)
		if length < 1:
			raise DomainError(f"sub: length must be ≥ 1, got {length}")
		if not (1 <= start <= len(string) - length + 1):
			raise InvalidDimError(f"sub: index out of range")
		return TiString(string.tokens[start - 1 : start + length - 1])
	raise ArgumentError(f"Invalid arguments: {args}")

###########
# FINANCE #
###########

@pure_func
def time_cnv(seconds):
	"""Convert a number of seconds into {days, hours, minutes, seconds}."""
	require_int(seconds)
	sign = -1 if seconds < 0 else 1
	remaining, secs = divmod(builtins.abs(seconds), 60)
	remaining, minutes = divmod(remaining, 60)
	days, hours = divmod(remaining, 24)
	return sign * TiList([days, hours, minutes, secs])

@pure_func
def dayofwk(year, month, day):
	"""Day of week: 1=Sunday, 2=Monday, …, 7=Saturday."""
	try:
		d = date(py_int(year), py_int(month), py_int(day))
	except ValueError as e:
		raise DomainError(f"dayOfWk: invalid date ({year}/{month}/{day})") from e
	return float(d.isoweekday() % 7 + 1)

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
		if builtins.abs(raw - ddyy) > 1e-6:
			raise DomainError(f"dbd: too many decimal places in MM.DDYY date {d!r}")
		month = int_part
		day, yy = divmod(ddyy, 100)
	elif int_part >= 100:
		# DDMM.YY: 2 decimal digits expected
		raw = frac_part * 100
		yy  = builtins.round(raw)
		if builtins.abs(raw - yy) > 1e-6:
			raise DomainError(f"dbd: too many decimal places in DDMM.YY date {d!r}")
		day, month = divmod(int_part, 100)
	else:
		raise DomainError(f"dbd: invalid date {d!r} (integer part {int_part} is ambiguous: must be ≤12 or ≥100)")

	year = (2000 if yy < 50 else 1900) + yy
	try:
		return date(year, month, day)
	except ValueError as e:
		raise DomainError(f"dbd: invalid date ({year}/{month}/{day})") from e

@pure_vectorized
def dbd(date1: float, date2: float):
	"""Days between two dates in TI Finance format (MM.DDYY or DDMM.YY)."""
	return (_parse_dbd_date(date2) - _parse_dbd_date(date1)).days

def _expand_cash_flows(cflist, cffreq):
	"""Expand a cash flow list with optional frequencies into a flat list."""
	require_list(cflist)
	if cffreq is None:
		return list(cflist)
	require_list(cffreq)
	if len(cflist) != len(cffreq):
		raise DimMismatchError("npv/irr: CFList and CFFreq must have the same dimension")
	result = []
	for cf, freq in zip(cflist, cffreq):
		require_int(freq)
		if freq < 1:
			raise DomainError("npv/irr: frequencies must be positive integers")
		result.extend([cf] * int(freq))
	return result

@pure_func
def npv(rate, cf0, cflist, cffreq=None):
	"""Net present value: CF0 + Σ CFj·(1+rate/100)^-j over expanded cash flows."""
	rate  = require_real(rate)
	cf0   = require_real(cf0)
	flows = _expand_cash_flows(cflist, cffreq)
	if rate == 0:
		return cf0 + builtins.sum(flows)
	r = 1 + rate / 100
	return cf0 + builtins.sum(cf * r ** -j for j, cf in enumerate(flows, 1))

@pure_func
def irr(cf0, cflist, cffreq=None):
	"""Internal rate of return: the rate (%) at which NPV equals zero."""
	require_real(cf0)
	flows = _expand_cash_flows(cflist, cffreq)
	all_flows = [cf0] + flows

	def _f(rate):
		if builtins.abs(rate) < 1e-10:
			return builtins.sum(all_flows)
		r = 1 + rate / 100
		return builtins.sum(cf * r ** -j for j, cf in enumerate(all_flows))

	def _df(rate):
		r = 1 + rate / 100
		return builtins.sum(-j / 100 * cf * r ** (-j - 1) for j, cf in enumerate(all_flows))

	for start in (10.0, 50.0, 100.0, 1.0, 200.0):
		rate = float(start)
		for _ in range(200):
			f  = _f(rate)
			df = _df(rate)
			if builtins.abs(df) < 1e-15:
				break
			step = f / df
			rate -= step
			if builtins.abs(step) < 1e-10 and builtins.abs(f) < 1e-6:
				break
		if rate > 1e-8 and builtins.abs(_f(rate)) < 1e-4:
			return rate

	raise DomainError("irr: no positive real solution found (ERR:NO SIGN CHG)")

@pure_vectorized
def eff(nom, cp):
	"""►Eff(: convert nominal interest rate to effective interest rate."""
	require_real(nom)
	require_real(cp)
	if cp <= 0:
		raise DomainError("►Eff: compounding periods must be positive")
	if cp == 1:
		return nom
	if nom <= -100:
		raise DomainError("►Eff: nominal rate must be > -100%")
	return 100 * ((1 + nom / (100 * cp)) ** cp - 1)

@pure_vectorized
def nom(eff_rate, cp):
	"""►Nom(: convert effective interest rate to nominal interest rate."""
	require_real(eff_rate)
	require_real(cp)
	if cp <= 0:
		raise DomainError("►Nom: compounding periods must be positive")
	if cp == 1:
		return eff_rate
	if eff_rate <= -100:
		raise DomainError("►Nom: effective rate must be > -100%")
	return 100 * cp * ((eff_rate / 100 + 1) ** (1 / cp) - 1)


#################
# DISTRIBUTIONS #
#################

def _regularized_inc_gamma(a, x):
	"""Lower regularized incomplete gamma function P(a, x) via series."""
	if x < 0:
		raise DomainError("x must be >= 0")
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
		raise DomainError("x must be in [0,1]")
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

@pure_func
def normalpdf(x, mu=0, sigma=1):
	require_real(x)
	require_real(mu)
	require_real(sigma)
	if sigma == 0:
		raise DomainError("normalpdf: sigma must be non-zero")
	z = (x - mu) / sigma
	return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

@pure_func
def normalcdf(lower, upper, mu=0, sigma=1):
	require_real(lower)
	require_real(upper)
	require_real(mu)
	require_real(sigma)
	if sigma == 0:
		raise DomainError("normalcdf: sigma must be non-zero")
	def _cdf(z):
		return 0.5 * (1 + math.erf(z / math.sqrt(2)))
	return _cdf((upper - mu) / sigma) - _cdf((lower - mu) / sigma)

@pure_func
def inv_norm(p, mu=0, sigma=1):
	require_real(p)
	require_real(mu)
	require_real(sigma)
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

@pure_func
def inv_t(p, df):
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

@pure_func
def tpdf(t, df):
	require_real(t)
	require_real(df)
	log_coeff = math.lgamma((df + 1) / 2) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2)
	return math.exp(log_coeff - (df + 1) / 2 * math.log(1 + t * t / df))

@pure_func
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

@pure_func
def chi_sq_pdf(x, df):
	require_real(x)
	require_real(df)
	if x <= 0:
		return 0.0
	k = df
	return math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))

@pure_func
def chi_sq_cdf(lower, upper, df):
	require_real(lower)
	require_real(upper)
	require_real(df)
	def _cdf(x, k):
		if x <= 0:
			return 0.0
		return _regularized_inc_gamma(k / 2, x / 2)
	return _cdf(upper, df) - _cdf(lower, df)

@pure_func
def f_pdf(x, df1, df2):
	require_real(x)
	require_real(df1)
	require_real(df2)
	if x <= 0:
		return 0.0
	log_num = (df1 / 2) * math.log(df1 * x) + (df2 / 2) * math.log(df2) - ((df1 + df2) / 2) * math.log(df1 * x + df2)
	log_den = math.log(x) + math.lgamma(df1 / 2) + math.lgamma(df2 / 2) - math.lgamma((df1 + df2) / 2)
	return math.exp(log_num - log_den)

@pure_func
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

@pure_func
def binompdf(n, p, k=None):
	n = py_int(n)
	require_real(p)
	if k is None:
		return TiList([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
	return math.comb(n, py_int(k)) * p ** k * (1 - p) ** (n - k)

@pure_func
def binomcdf(n, p, k=None):
	n = py_int(n)
	require_real(p)
	if k is None:
		acc = 0
		result = []
		for i in range(n + 1):
			acc += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
			result.append(acc)
		return TiList(result)

	return float(builtins.sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(py_int(k) + 1)))

@pure_func
def poissonpdf(lam, k):
	require_real(lam)
	k = py_int(k)
	return math.exp(-lam) * lam ** k / math.factorial(k)

@pure_func
def poissoncdf(lam, k):
	require_real(lam)
	k = py_int(k)
	return builtins.sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))

@pure_func
def geometpdf(p, n):
	require_real(p)
	n = require_int(n)
	return p * (1 - p) ** (n - 1)

@pure_func
def geometcdf(p, n):
	require_real(p)
	n = require_int(n)
	return 1 - (1 - p) ** n

###########
# CATALOG #
###########

@pure_vectorized
def sinh(x):
	return math.sinh(require_real(x))

@pure_vectorized
def cosh(x):
	return math.cosh(require_real(x))

@pure_vectorized
def tanh(x):
	return math.tanh(require_real(x))

@pure_vectorized
def asinh(x):
	return math.asinh(require_real(x))
