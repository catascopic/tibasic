"""Probability distributions — the DISTR and DISTR DRAW menus.

DISTR:
  1: normalpdf(    2: normalcdf(    3: invNorm(    4: invT(
  5: tpdf(         6: tcdf(         7: χ²pdf(      8: χ²cdf(
  9: Fpdf(         0: Fcdf(         A: binompdf(   B: binomcdf(
  C: poissonpdf(   D: poissoncdf(   E: geometpdf(  F: geometcdf(
"""
from __future__ import annotations

import builtins
import math

from core import TiList
from core import require_int, py_int
from preparse import preparse_func, Real
from errors import DomainError


# ── Numerical helpers ─────────────────────────────────────────────────────────

def _regularized_inc_gamma(a, x):
	"""Lower regularized incomplete gamma function P(a, x) via series."""
	if x < 0:
		raise DomainError("x must be >= 0")
	if x == 0:
		return 0.0
	if x < a + 1:
		term = 1.0 / a
		total = term
		for k in range(1, 300):
			term *= x / (a + k)
			total += term
			if builtins.abs(term) < builtins.abs(total) * 1e-15:
				break
		return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
	else:
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
	if x > (a + 1) / (a + b + 2):
		return 1.0 - _inc_beta(b, a, 1 - x)
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


# ── Normal distribution ───────────────────────────────────────────────────────

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
	def _inv_std(q):
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
	return mu + sigma * _inv_std(p)


# ── Student's t distribution ──────────────────────────────────────────────────

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
def inv_t(p: Real, df: Real):
	if p <= 0:
		return -1e99
	if p >= 1:
		return 1e99
	x = inv_norm(p)
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


# ── Chi-squared distribution ──────────────────────────────────────────────────

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


# ── F distribution ────────────────────────────────────────────────────────────

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


# ── Binomial distribution ─────────────────────────────────────────────────────

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


# ── Poisson distribution ──────────────────────────────────────────────────────

@preparse_func
def poissonpdf(lam: Real, k: Real):
	k = py_int(k)
	return math.exp(-lam) * lam ** k / math.factorial(k)

@preparse_func
def poissoncdf(lam: Real, k: Real):
	k = py_int(k)
	return builtins.sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


# ── Geometric distribution ────────────────────────────────────────────────────

@preparse_func
def geometpdf(p: Real, n: Real):
	n = py_int(n)
	return p * (1 - p) ** (n - 1)

@preparse_func
def geometcdf(p: Real, n: Real):
	n = py_int(n)
	return 1 - (1 - p) ** n
