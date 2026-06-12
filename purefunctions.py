import cmath
import builtins
import decimal as _decimal
import math
import random
from datetime import date
from functools import wraps
from numbers import Number
from tiobjects import TiList, require_list
from core import require_num, require_real, require_int, py_int
from errors import (
	DataTypeError, DimMismatchError, InvalidDimError,
	DomainError,
)
from preparse import preparse_func, Real, Vectorized, VectorizedReal, MatrixVectorized, AnyValue
from decorators import vectorize


# ── Helpers ───────────────────────────────────────────────────────────────────

def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply


@preparse_func
def not_(x: VectorizedReal):
	return float(not x)

##################
# MAIN FUNCTIONS #
##################

@preparse_func
def pow10(x: Vectorized):
	return 10 ** x

@preparse_func
def exp(x: Vectorized):
	return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

##################
# MATH FUNCTIONS #
##################

@preparse_func
def cbrt(x: Vectorized):
	if isinstance(x, complex):
		if x == 0:
			return 0
		return cmath.exp(cmath.log(x) / 3)
	return math.cbrt(x)

@preparse_func
def abs(x: MatrixVectorized):
	return builtins.abs(x)

def _ti_round(x: float, decimals: int) -> float:
	"""Round half away from zero — TI-84 behavior.

	Python's built-in round() uses banker's rounding (half-to-even), so
	round(0.5) == 0.  TI always rounds 0.5 up (away from zero), so we use
	decimal.ROUND_HALF_UP.  str(x) gives the shortest round-trip representation,
	which matches the ~10 significant digits TI operates with internally.
	"""
	quant = _decimal.Decimal(10) ** -decimals
	return float(_decimal.Decimal(str(x)).quantize(quant, rounding=_decimal.ROUND_HALF_UP))


@preparse_func
def round(x: MatrixVectorized, decimals: Real = 9):
	n = py_int(decimals)
	if isinstance(x, complex):
		return complex(_ti_round(x.real, n), _ti_round(x.imag, n))
	return _ti_round(x, n)

@preparse_func
@handle_complex
def i_part(x: MatrixVectorized):
	return float(math.trunc(x))

@preparse_func
@handle_complex
def f_part(x: MatrixVectorized):
	return x - math.trunc(x)

@preparse_func
@handle_complex
def int_(x: MatrixVectorized):
	return float(math.floor(x))


def _minmax(func, a, b):
	if b is None:
		return func(require_list(a))
	
	if isinstance(a, TiList) and isinstance(b, TiList):
		if len(a) != len(b):
			raise DimMismatchError(f"{func.__name__}: dim mismatch ({len(a)} vs {len(b)})")
		return TiList([func(x, y) for x, y in zip(a, b)])
		
	if isinstance(a, Number) and isinstance(b, Number):
		return func(a, b)

	raise DataTypeError(f"{func.__name__}: both args must be the same type (both numeric or both list)")

@preparse_func
def min(a: AnyValue, b: AnyValue = None):
	return _minmax(builtins.min, a, b)

@preparse_func
def max(a: AnyValue, b: AnyValue = None):
	return _minmax(builtins.max, a, b)

@preparse_func
def lcm(a: VectorizedReal, b: VectorizedReal) -> float:
	return float(math.lcm(py_int(a), py_int(b)))

@preparse_func
def gcd(a: VectorizedReal, b: VectorizedReal) -> float:
	return float(math.gcd(py_int(a), py_int(b)))

@preparse_func
def remainder(a: VectorizedReal, b: VectorizedReal):
	a = require_int(a)
	b = require_int(b)
	if a < 0:
		raise DomainError(f"a must be non-negative but got {a}")
	if b < 1:
		raise DomainError(f"b must be positive but got {b}")
	return a % b

@preparse_func
def conj(x: Vectorized):
	return complex(x.real, -x.imag) if isinstance(x, complex) else x

@preparse_func
def real_(x: Vectorized):
	return x.real if isinstance(x, complex) else x

@preparse_func
def imag(x: Vectorized):
	return x.imag if isinstance(x, complex) else 0

# Technically works on matrices, but since matrices can't store complex numbers, the result is all 0s.
# TiBasicDev thinks this is basically a bug, so I'm not implementing it in order to discourage it.
# (If you want a matrix of all 0s, you can just do 0[A].)
@preparse_func
def angle(x: Vectorized):
	return cmath.phase(x)

@preparse_func
def rand_list(n: Real):
	return TiList([random.random() for _ in range(py_int(n))])

@vectorize
def _rand_int_single(low, high):
	return float(random.randint(py_int(low), py_int(high)))

@preparse_func
def rand_int(low: AnyValue, high: AnyValue, n: Real = 1.0):
	if isinstance(low, TiList) or isinstance(high, TiList):
		if n != 1.0:
			raise DataTypeError("randInt: list arguments cannot be combined with n > 1")
		return _rand_int_single(low, high)
	require_real(low)
	require_real(high)
	if low > high:
		raise DomainError(f"randInt: low must be ≤ high, got {low} > {high}")
	if n == 1:
		return _rand_int_single(low, high)
	low = py_int(low)
	high = py_int(high)
	n = py_int(n)
	return TiList([float(random.randint(low, high)) for _ in range(n)])

@preparse_func
def rand_norm(mu: Real, sigma: Real, n: Real = None):
	if n is None:
		return random.gauss(mu, sigma)
	return TiList([random.gauss(mu, sigma) for _ in range(py_int(n))])

@preparse_func
def rand_bin(n: Real, p: Real, simulations: Real = None):
	n = py_int(n)
	if not (0 <= p <= 1):
		raise DomainError("randBin: p must be in [0, 1]")
	if n <= 0:
		raise DomainError("randBin: n must be positive")
	if simulations is None:
		return builtins.sum(1 for _ in range(n) if random.random() < p)
	simulations = py_int(simulations)
	return TiList([builtins.sum(1 for _ in range(n) if random.random() < p) for _ in range(simulations)])

@preparse_func
def rand_int_no_rep(low: Real, high: Real):
	lst = list(range(py_int(low), py_int(high) + 1))
	random.shuffle(lst)
	return TiList(lst)

###########
# FINANCE #
###########

@preparse_func
def time_cnv(seconds: Real):
	"""Convert a number of seconds into {days, hours, minutes, seconds}."""
	seconds = py_int(seconds)
	sign = -1 if seconds < 0 else 1
	remaining, secs = divmod(builtins.abs(seconds), 60)
	remaining, minutes = divmod(remaining, 60)
	days, hours = divmod(remaining, 24)
	return sign * TiList([days, hours, minutes, secs])

@preparse_func
def dayofwk(year: Real, month: Real, day: Real):
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

@preparse_func
def dbd(date1: VectorizedReal, date2: VectorizedReal):
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

@preparse_func
def npv(rate: Real, cf0: Real, cflist: TiList, cffreq: TiList = None):
	"""Net present value: CF0 + Σ CFj·(1+rate/100)^-j over expanded cash flows."""
	flows = _expand_cash_flows(cflist, cffreq)
	if rate == 0:
		return cf0 + builtins.sum(flows)
	r = 1 + rate / 100
	return cf0 + builtins.sum(cf * r ** -j for j, cf in enumerate(flows, 1))

@preparse_func
def irr(cf0: Real, cflist: TiList, cffreq: TiList = None):
	"""Internal rate of return: the rate (%) at which NPV equals zero."""
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

@preparse_func
def eff(nom: VectorizedReal, cp: VectorizedReal):
	"""►Eff(: convert nominal interest rate to effective interest rate."""
	if cp <= 0:
		raise DomainError("►Eff: compounding periods must be positive")
	if cp == 1:
		return nom
	if nom <= -100:
		raise DomainError("►Eff: nominal rate must be > -100%")
	return 100 * ((1 + nom / (100 * cp)) ** cp - 1)

@preparse_func
def nom(eff_rate: VectorizedReal, cp: VectorizedReal):
	"""►Nom(: convert effective interest rate to nominal interest rate."""
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

@preparse_func
def normalpdf(x: Real, mu: Real = 0, sigma: Real = 1):
	if sigma == 0:
		raise DomainError("normalpdf: sigma must be non-zero")
	z = (x - mu) / sigma
	return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

@preparse_func
def normalcdf(lower: Real, upper: Real, mu: Real = 0, sigma: Real = 1):
	if sigma == 0:
		raise DomainError("normalcdf: sigma must be non-zero")
	def _cdf(z):
		return 0.5 * (1 + math.erf(z / math.sqrt(2)))
	return _cdf((upper - mu) / sigma) - _cdf((lower - mu) / sigma)

@preparse_func
def inv_norm(p: Real, mu: Real = 0, sigma: Real = 1):
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

@preparse_func
def inv_t(p: Real, df: Real):
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

@preparse_func
def tpdf(t: Real, df: Real):
	log_coeff = math.lgamma((df + 1) / 2) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2)
	return math.exp(log_coeff - (df + 1) / 2 * math.log(1 + t * t / df))

@preparse_func
def tcdf(lower: Real, upper: Real, df: Real):
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

@preparse_func
def chi_sq_pdf(x: Real, df: Real):
	if x <= 0:
		return 0.0
	k = df
	return math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))

@preparse_func
def chi_sq_cdf(lower: Real, upper: Real, df: Real):
	def _cdf(x, k):
		if x <= 0:
			return 0.0
		return _regularized_inc_gamma(k / 2, x / 2)
	return _cdf(upper, df) - _cdf(lower, df)

@preparse_func
def f_pdf(x: Real, df1: Real, df2: Real):
	if x <= 0:
		return 0.0
	log_num = (df1 / 2) * math.log(df1 * x) + (df2 / 2) * math.log(df2) - ((df1 + df2) / 2) * math.log(df1 * x + df2)
	log_den = math.log(x) + math.lgamma(df1 / 2) + math.lgamma(df2 / 2) - math.lgamma((df1 + df2) / 2)
	return math.exp(log_num - log_den)

@preparse_func
def fcdf(lower: Real, upper: Real, df1: Real, df2: Real):
	def _cdf(x, d1, d2):
		if x <= 0:
			return 0.0
		z = d1 * x / (d1 * x + d2)
		return _inc_beta(d1 / 2, d2 / 2, z)
	return _cdf(upper, df1, df2) - _cdf(lower, df1, df2)

@preparse_func
def binompdf(n: Real, p: Real, k: Real = None):
	n = py_int(n)
	if k is None:
		return TiList([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
	k = py_int(k)
	return math.comb(n, k) * p ** k * (1 - p) ** (n - k)

@preparse_func
def binomcdf(n: Real, p: Real, k: Real = None):
	n = py_int(n)
	if k is None:
		acc = 0
		result = []
		for i in range(n + 1):
			acc += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
			result.append(acc)
		return TiList(result)

	return float(builtins.sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(py_int(k) + 1)))

@preparse_func
def poissonpdf(lam: Real, k: Real):
	k = py_int(k)
	return math.exp(-lam) * lam ** k / math.factorial(k)

@preparse_func
def poissoncdf(lam: Real, k: Real):
	k = py_int(k)
	return builtins.sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))

@preparse_func
def geometpdf(p: Real, n: Real):
	n = py_int(n)
	return p * (1 - p) ** (n - 1)

@preparse_func
def geometcdf(p: Real, n: Real):
	n = py_int(n)
	return 1 - (1 - p) ** n

###########
# CATALOG #
###########

@preparse_func
def sinh(x: VectorizedReal):
	return math.sinh(x)

@preparse_func
def cosh(x: VectorizedReal):
	return math.cosh(x)

@preparse_func
def tanh(x: VectorizedReal):
	return math.tanh(x)

@preparse_func
def asinh(x: VectorizedReal):
	return math.asinh(x)
