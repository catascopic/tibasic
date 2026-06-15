"""Math functions organized by calculator menu placement.

Covered menus / button groups:
  Dedicated keys : sin, cos, tan, asin, acos, atan, √(, ln, log, e^(, 10^(
  MATH           : ³√(, nDeriv(, fnInt(, Σ(, logBASE(
  NUM            : abs(, round(, iPart(, fPart(, int(, min(, max(, lcm(, gcd(, remainder(
  CPX            : conj(, real(, imag(, angle(
  ANGLE          : R►Pr(, R►Pθ(, P►Rx(, P►Ry(
  PRB            : randList, randInt(, randNorm(, randBin(, randIntNoRep(
  TEST LOGIC     : not(
  CATALOG        : sinh, cosh, tanh, asinh, acosh, atanh
"""

import builtins
import cmath
import decimal as _decimal
import math
import random
from functools import wraps
from numbers import Number

from core import TiList, TiMatrix, require_list
from core import require_real, require_int, require_num, py_int
from preparse import preparse_func, Real, Vectorized, VectorizedReal, MatrixVectorized, AnyValue, Thunk, NumericVar, Env
from core import vectorize
from errors import DataTypeError, DimMismatchError, DomainError


# ── Shared helpers ────────────────────────────────────────────────────────────

def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply

def _inv_trig(func, env, x):
	try:
		return env.from_rad(func(x))
	except ValueError:
		raise DomainError(f"{func.__name__}: argument out of domain: {x}")


# ── Dedicated keys ────────────────────────────────────────────────────────────

@preparse_func
def sin(env: Env, x: VectorizedReal):
	return math.sin(env.to_rad(x))

@preparse_func
def cos(env: Env, x: VectorizedReal):
	return math.cos(env.to_rad(x))

@preparse_func
def tan(env: Env, x: VectorizedReal):
	return math.tan(env.to_rad(x))

@preparse_func
def asin(env: Env, x: VectorizedReal):
	return _inv_trig(math.asin, env, x)

@preparse_func
def acos(env: Env, x: VectorizedReal):
	return _inv_trig(math.acos, env, x)

@preparse_func
def atan(env: Env, x: VectorizedReal):
	return _inv_trig(math.atan, env, x)

@preparse_func
def sqrt(x: Vectorized):
	if isinstance(x, complex):
		return cmath.sqrt(x)
	if x >= 0:
		return math.sqrt(x)
	return cmath.sqrt(x)

@preparse_func
def ln(x: Vectorized):
	if isinstance(x, complex):
		return cmath.log(x)
	if x > 0:
		return math.log(x)
	if x == 0:
		raise DomainError("ln: undefined for 0")
	return cmath.log(x)

@preparse_func
def log(x: Vectorized):
	if isinstance(x, complex):
		return cmath.log10(x)
	if x > 0:
		return math.log10(x)
	if x == 0:
		raise DomainError("log: undefined for 0")
	return cmath.log10(x)

@preparse_func
def exp(x: Vectorized):
	return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

@preparse_func
def pow10(x: Vectorized):
	return 10 ** x


# ── MATH menu ─────────────────────────────────────────────────────────────────
# MATH 4: ³√(

@preparse_func
def cbrt(x: Vectorized):
	if isinstance(x, complex):
		if x == 0:
			return 0
		return cmath.exp(cmath.log(x) / 3)
	return math.cbrt(x)

# MATH 8: nDeriv(

@preparse_func
def n_deriv(env: Env, formula: Thunk, var: NumericVar, val: Real, h: Real = 0.001) -> float:
	with env.nest_guard(n_deriv, max_depth=1), var.scoped():
		var.value = val + h
		fwd = formula.eval()
		var.value = val - h
		bwd = formula.eval()
	return (fwd - bwd) / (2 * h)

# MATH 9: fnInt(

_K15_NODES = [
	0.0,                0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
	0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126
]
_K15_WEIGHTS = [
	0.2094821410847278, 0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
	0.1406532597155259, 0.1047900103222502, 0.0630920926299786, 0.0229353220105292
]
_G7_WEIGHTS = [
	0.4179591836734694, None, 0.3818300505051189, None,
	0.2797053914892767, None, 0.1294849661688697, None
]

def _gk15(f, lo, hi):
	mid = (lo + hi) / 2
	half = (hi - lo) / 2
	k15 = g7 = 0
	for i, x in enumerate(_K15_NODES):
		for sign in ([1] if x == 0 else [1, -1]):
			fx = f(mid + sign * x * half)
			k15 += _K15_WEIGHTS[i] * fx
			if _G7_WEIGHTS[i] is not None:
				g7 += _G7_WEIGHTS[i] * fx
	k15 *= half
	g7  *= half
	return k15, builtins.abs(k15 - g7)

def _adaptive_gk15(f, lo, hi, tol, depth=0):
	k15, err = _gk15(f, lo, hi)
	if err <= tol or depth >= 50:
		return k15
	mid = (lo + hi) / 2
	return (
		_adaptive_gk15(f, lo, mid, tol / 2, depth + 1) +
		_adaptive_gk15(f, mid, hi, tol / 2, depth + 1)
	)

@preparse_func
def fn_int(env: Env, formula: Thunk, var: NumericVar, lo: Real, hi: Real, tol: Real = 1e-5) -> float:
	with env.nest_guard('fnInt'), var.scoped():
		def f(x):
			var.value = x
			return formula.eval()
		return _adaptive_gk15(f, lo, hi, tol)

# MATH 0: Σ(

@preparse_func
def sigma(env: Env, formula: Thunk, var: NumericVar, start: Real, end: Real) -> float:
	total = 0
	n = start
	with env.nest_guard(sigma), var.scoped():
		while n <= end:
			var.value = n
			total += formula.eval()
			n += 1
	return total

# MATH A: logBASE(

@preparse_func
def log_base(x: Vectorized, base: Vectorized):
	if isinstance(x, complex) or isinstance(base, complex):
		return cmath.log(x, base)
	if base <= 0 or base == 1:
		raise DomainError(f"logBASE: base must be positive and ≠ 1, got {base}")
	if x == 0:
		raise DomainError("logBASE: undefined for x=0")
	if x > 0:
		return math.log(x, base)
	return cmath.log(x, base)


# ── NUM menu ──────────────────────────────────────────────────────────────────
# NUM 1: abs(  (also CPX 5: abs()

@preparse_func
def abs(x: MatrixVectorized):
	return builtins.abs(x)

# NUM 2: round(

def _ti_round(x: float, decimals: int) -> float:
	"""Round half away from zero (TI-84 behavior; Python uses banker's rounding)."""
	quant = _decimal.Decimal(10) ** -decimals
	return float(_decimal.Decimal(str(x)).quantize(quant, rounding=_decimal.ROUND_HALF_UP))

@preparse_func
def round(x: MatrixVectorized, decimals: Real = 9):
	n = py_int(decimals)
	if isinstance(x, complex):
		return complex(_ti_round(x.real, n), _ti_round(x.imag, n))
	return _ti_round(x, n)

# NUM 3: iPart(

@preparse_func
@handle_complex
def i_part(x: MatrixVectorized):
	return float(math.trunc(x))

# NUM 4: fPart(

@preparse_func
@handle_complex
def f_part(x: MatrixVectorized):
	return x - math.trunc(x)

# NUM 5: int(

@preparse_func
@handle_complex
def int_(x: MatrixVectorized):
	return float(math.floor(x))

# NUM 6: min(  /  NUM 7: max(

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

# NUM 8: lcm(

@preparse_func
def lcm(a: VectorizedReal, b: VectorizedReal) -> float:
	return float(math.lcm(py_int(a), py_int(b)))

# NUM 9: gcd(

@preparse_func
def gcd(a: VectorizedReal, b: VectorizedReal) -> float:
	return float(math.gcd(py_int(a), py_int(b)))

# NUM 0: remainder(

@preparse_func
def remainder(a: VectorizedReal, b: VectorizedReal):
	a = require_int(a)
	b = require_int(b)
	if a < 0:
		raise DomainError(f"a must be non-negative but got {a}")
	if b < 1:
		raise DomainError(f"b must be positive but got {b}")
	return a % b


# ── CPX menu ──────────────────────────────────────────────────────────────────
# CPX 1: conj(

@preparse_func
def conj(x: Vectorized):
	return complex(x.real, -x.imag) if isinstance(x, complex) else x

# CPX 2: real(

@preparse_func
def real(x: Vectorized):
	return x.real if isinstance(x, complex) else x

# CPX 3: imag(

@preparse_func
def imag(x: Vectorized):
	return x.imag if isinstance(x, complex) else 0

# CPX 4: angle(
# Note: matrix support is intentionally omitted (see comment in original code).

@preparse_func
def angle(x: Vectorized):
	return cmath.phase(x)


# ── ANGLE menu ────────────────────────────────────────────────────────────────

@preparse_func
def rect_to_polar_radius(env: Env, x: VectorizedReal, y: VectorizedReal):
	return math.hypot(x, y)

@preparse_func
def rect_to_polar_angle(env: Env, x: VectorizedReal, y: VectorizedReal):
	return env.from_rad(math.atan2(y, x))

@preparse_func
def polar_to_rect_x(env: Env, r: VectorizedReal, theta: VectorizedReal):
	return r * math.cos(env.to_rad(theta))

@preparse_func
def polar_to_rect_y(env: Env, r: VectorizedReal, theta: VectorizedReal):
	return r * math.sin(env.to_rad(theta))


# ── PRB menu ──────────────────────────────────────────────────────────────────
# (PRB 1: rand is a token with a special res= handler in catalog.py, not a function here)

@preparse_func
def rand_list(n: Real):
	return TiList([random.random() for _ in range(py_int(n))])

@vectorize
def _rand_int_single(low, high):
	return float(random.randint(py_int(low), py_int(high)))

# PRB 5: randInt(

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

# PRB 6: randNorm(

@preparse_func
def rand_norm(mu: Real, sigma: Real, n: Real = None):
	if n is None:
		return random.gauss(mu, sigma)
	return TiList([random.gauss(mu, sigma) for _ in range(py_int(n))])

# PRB 7: randBin(

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

# PRB 8: randIntNoRep(

@preparse_func
def rand_int_no_rep(low: Real, high: Real):
	lst = list(range(py_int(low), py_int(high) + 1))
	random.shuffle(lst)
	return TiList(lst)


# ── Hyperbolic trig (CATALOG) ─────────────────────────────────────────────────

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

@preparse_func
def acosh(x: VectorizedReal):
	if x >= 1:
		return math.acosh(x)
	return cmath.acosh(x)

@preparse_func
def atanh(x: VectorizedReal):
	if builtins.abs(x) < 1:
		return math.atanh(x)
	if builtins.abs(x) == 1:
		raise DomainError("atanh: undefined for ±1")
	return cmath.atanh(x)


# ── TEST LOGIC menu ────────────────────────────────────────────────────────────

@preparse_func
def not_(x: VectorizedReal):
	return float(not x)



if __name__ == '__main__':
	print(round.schema)

