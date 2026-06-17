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
import math
import random
from functools import wraps
from numbers import Number

from core import TiList, TiMatrix, require_list
from core import require_real, require_int, require_num, py_int
from preparse import preparse_func, Real, Vectorized, VectorizedReal, MatrixVectorized, AnyValue, Thunk, NumericVar, Env
from core import vectorize
from errors import (
	DataTypeError, DimMismatchError, DomainError, InvalidDimError,
	DivideByZeroError, NonRealAnsError, TiOverflowError,
	BoundError, BadGuessError, NoSignChangeError, ToleranceError,
)


def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(x, *args):
		return complex(func(x.real, *args), func(x.imag, *args)) if isinstance(x, complex) else func(x, *args)
	return apply


def _inv_trig(func, env, x):
	try:
		return env.from_rad(func(x))
	except ValueError:
		raise DomainError(f"{func.__name__}: argument out of domain: {x}")


# NOT (the only LOGIC command that's actually a function)

@preparse_func
def not_(x: VectorizedReal):
	return float(not x)


# DEDICATED KEYS

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


# MATH MENU

@preparse_func
def cbrt(x: Vectorized):
	if isinstance(x, complex):
		if x == 0:
			return 0.0
		return cmath.exp(cmath.log(x) / 3)
	return math.cbrt(x)


@preparse_func
def n_deriv(env: Env, formula: Thunk, var: NumericVar, val: Real, h: Real = 0.001) -> float:
	with env.nest_guard(n_deriv, max_depth=1), var.scoped():
		var.value = val + h
		fwd = formula.eval()
		var.value = val - h
		bwd = formula.eval()
	return (fwd - bwd) / (2 * h)


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


@preparse_func
def sigma(env: Env, formula: Thunk, var: NumericVar, start: Real, end: Real) -> float:
	total = 0.0
	n = start
	with env.nest_guard(sigma), var.scoped():
		while n <= end:
			var.value = n
			total += formula.eval()
			n += 1
	return total


# ── Root finding and optimization (solve(, fMin(, fMax() ──────────────────────
# All three drive a real-valued f(var) built from the user's expression, sweeping
# `var` while leaving its stored value untouched (var.scoped()).

def _real_of(env, formula, var):
	"""Build f(x): set `var` to x, evaluate the expression, require a real result."""
	def f(x):
		var.value = x
		return require_real(formula.eval())
	return f


def _brentq(f, a, b, tol=1e-12, max_iter=100):
	"""Brent's method: the root of f in a bracket [a, b] where f(a), f(b) differ in
	sign.  Raises NoSignChangeError if they don't."""
	fa, fb = f(a), f(b)
	if fa == 0.0:
		return a
	if fb == 0.0:
		return b
	if fa * fb > 0.0:
		raise NoSignChangeError("solve: no sign change over the bracket")
	if builtins.abs(fa) < builtins.abs(fb):
		a, b, fa, fb = b, a, fb, fa
	c, fc, d = a, fa, a
	mflag = True
	for _ in range(max_iter):
		if fb == 0.0 or builtins.abs(b - a) < tol:
			return b
		if fa != fc and fb != fc:  # inverse quadratic interpolation
			s = (
				a * fb * fc / ((fa - fb) * (fa - fc)) + 
				b * fa * fc / ((fb - fa) * (fb - fc)) + 
				c * fa * fb / ((fc - fa) * (fc - fb))
			)
		else:  # secant
			s = b - fb * (b - a) / (fb - fa)
		lo, hi = builtins.min((3 * a + b) / 4, b), builtins.max((3 * a + b) / 4, b)
		if (not lo <= s <= hi
		    or (mflag and builtins.abs(s - b) >= builtins.abs(b - c) / 2)
		    or (not mflag and builtins.abs(s - b) >= builtins.abs(c - d) / 2)
		    or (mflag and builtins.abs(b - c) < tol)
		    or (not mflag and builtins.abs(c - d) < tol)
		):
			s = (a + b) / 2  # fall back to bisection
			mflag = True
		else:
			mflag = False
		fs = f(s)
		d, c, fc = c, b, fb
		if fa * fs < 0:
			b, fb = s, fs
		else:
			a, fa = s, fs
		if builtins.abs(fa) < builtins.abs(fb):
			a, b, fa, fb = b, a, fb, fa
	return b


# _bracket_root tuning: the outward search starts with a step of _STEP_FRAC·|guess|
# (but at least _MIN_STEP, so a guess of 0 still moves) and grows it by _GROWTH each
# sweep, until a sign change is bracketed or both sides are exhausted.
_BRACKET_STEP_FRAC = 0.01
_BRACKET_MIN_STEP  = 1e-4
_BRACKET_GROWTH    = 1.6


def _bracket_root(f, guess, lo, hi, max_steps=2000):
	"""Expand outward from `guess` within [lo, hi] until f brackets a root.

	Returns a (left, right) bracket where f changes sign (or touches zero).  A
	probe that overflows or leaves the function's domain caps that side.
	"""
	def value(x):
		# The function value at x, or None when the expression can't yield a real
		# value there (overflow, ln/√ of a negative → NonRealAnsError in Real mode,
		# 1/0, out-of-domain) — None caps the search in that direction.  Genuine
		# errors (undefined var, non-scalar result) propagate.
		try:
			return f(x)
		except (OverflowError, DomainError, DivideByZeroError, NonRealAnsError, TiOverflowError):
			return None

	f_guess = value(guess)
	if f_guess is None:
		raise BadGuessError(f"solve: function is undefined at the guess {guess}")
	if f_guess == 0:
		return guess, guess

	def expand(pos, f_pos, bound, direction):
		"""Take one step of the current size from `pos` toward `bound`.

		Returns (bracket, new_pos, new_f, done): `bracket` is the (left, right)
		pair once a root is crossed or touched (else None); `done` is True when
		this side is spent — the bound is reached or a wall (unevaluable probe).
		"""
		if pos == bound:
			return None, pos, f_pos, True
		probe = builtins.min(pos + step, bound) if direction > 0 else builtins.max(pos - step, bound)
		f_probe = value(probe)
		if f_probe is None:   # hit a wall
			return None, pos, f_pos, True
		if f_probe == 0 or f_probe * f_pos < 0:  # bracketed a root
			bracket = (pos, probe) if direction > 0 else (probe, pos)
			return bracket, probe, f_probe, True
		return None, probe, f_probe, probe == bound

	left = right = guess
	f_left = f_right = f_guess
	left_done = right_done = False
	step = builtins.max(builtins.abs(guess) * _BRACKET_STEP_FRAC, _BRACKET_MIN_STEP)
	for _ in range(max_steps):
		if not right_done:
			bracket, right, f_right, right_done = expand(right, f_right, hi, +1)
			if bracket:
				return bracket
		if not left_done:
			bracket, left, f_left, left_done = expand(left, f_left, lo, -1)
			if bracket:
				return bracket
		if left_done and right_done:
			break
		step *= _BRACKET_GROWTH
	raise NoSignChangeError("solve: no sign change found near the guess")


def _solve_bounds(bounds: TiList):
	data = bounds.data
	if len(data) != 2:
		raise InvalidDimError("solve: bounds must be a list {lower, upper}")
	a, b = data
	require_real(a)
	require_real(b)	
	return (a, b) if a <= b else (b, a)


@preparse_func
def solve(env: Env, formula: Thunk, var: NumericVar, guess: Real, bounds: TiList = None) -> float:
	lo, hi = (-1e99, 1e99) if bounds is None else _solve_bounds(bounds)
	if not lo <= guess <= hi:
		raise BadGuessError(f"solve: guess {guess} is outside the bounds [{lo}, {hi}]")
	with env.nest_guard('solve'), var.scoped():
		f = _real_of(env, formula, var)
		a, b = _bracket_root(f, guess, lo, hi)
		return _brentq(f, a, b)


_GOLDEN   = 0.3819660112501051      # (3 − √5) / 2, the golden-section fraction
_SQRT_EPS = 1.4901161193847656e-08  # √(machine epsilon), Brent's relative tolerance floor


def _brent_minimize(f, a, b, xatol, max_eval=500):
	"""Brent's bounded minimization on [a, b] (SciPy's fminbound); returns the argmin.

	Combines golden-section search with parabolic interpolation, tracking the three
	best points seen so far: `best`, then `second`, then `third`.  `delta` is the
	step about to be taken and `last_span` the one taken last (used to decide
	whether a parabolic step is worth trying).  Raises ToleranceError if it can't
	converge within max_eval evaluations.
	"""
	best = second = third = a + _GOLDEN * (b - a)
	f_best = f_second = f_third = f(best)
	delta = last_span = 0.0
	mid = 0.5 * (a + b)
	tol1 = _SQRT_EPS * builtins.abs(best) + xatol / 3.0
	tol2 = 2.0 * tol1
	evals = 1
	while builtins.abs(best - mid) > tol2 - 0.5 * (b - a):
		use_golden = True
		if builtins.abs(last_span) > tol1:  # try a parabolic step
			term = (best - second) * (f_best - f_third)
			denom = (best - third) * (f_best - f_second)
			numer = (best - third) * denom - (best - second) * term
			denom = 2.0 * (denom - term)
			if denom > 0.0:
				numer = -numer
			denom = builtins.abs(denom)
			prev_span, last_span = last_span, delta
			if builtins.abs(numer) < builtins.abs(0.5 * denom * prev_span) and denom * (a - best) < numer < denom * (b - best):
				delta = numer / denom
				step_to = best + delta
				if step_to - a < tol2 or b - step_to < tol2:
					delta = math.copysign(tol1, mid - best)
				use_golden = False
		if use_golden:  # golden-section step
			last_span = (a - best) if best >= mid else (b - best)
			delta = _GOLDEN * last_span
		direction = math.copysign(1.0, delta) if delta != 0 else 1.0
		trial = best + direction * builtins.max(builtins.abs(delta), tol1)
		f_trial = f(trial)
		evals += 1
		if f_trial <= f_best:  # new incumbent
			if trial >= best:
				a = best
			else:
				b = best
			third, f_third = second, f_second
			second, f_second = best, f_best
			best, f_best = trial, f_trial
		else:  # tighten the bracket
			if trial < best:
				a = trial
			else:
				b = trial
			if f_trial <= f_second or second == best:
				third, f_third = second, f_second
				second, f_second = trial, f_trial
			elif f_trial <= f_third or third == best or third == second:
				third, f_third = trial, f_trial
		mid = 0.5 * (a + b)
		tol1 = _SQRT_EPS * builtins.abs(best) + xatol / 3.0
		tol2 = 2.0 * tol1
		if evals >= max_eval:
			raise ToleranceError("fMin/fMax: could not meet the requested tolerance")
	return best


def _optimize(env, formula, var, lo, hi, tol, sign, func):
	if lo > hi:
		raise BoundError(f"{func.__name__}: lower bound {lo} > upper bound {hi}")
	if tol == 0:
		raise DomainError(f"{func.__name__}: tolerance cannot be 0")
	with env.nest_guard(func), var.scoped():
		f = _real_of(env, formula, var)
		return _brent_minimize(lambda x: sign * f(x), lo, hi, builtins.abs(tol))


@preparse_func
def f_min(env: Env, formula: Thunk, var: NumericVar, lo: Real, hi: Real, tol: Real = 1e-5) -> float:
	return _optimize(env, formula, var, lo, hi, tol, 1.0, f_min)


@preparse_func
def f_max(env: Env, formula: Thunk, var: NumericVar, lo: Real, hi: Real, tol: Real = 1e-5) -> float:
	return _optimize(env, formula, var, lo, hi, tol, -1.0, f_max)


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


@preparse_func
def abs(x: MatrixVectorized):
	return builtins.abs(x)


@preparse_func
@handle_complex
def round(x: MatrixVectorized, decimals: Real = 9.0):
	# Should this use Decimal to round?
	mag = 10 ** decimals
	return math.floor(x * mag + 0.5) / mag


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
def real(x: Vectorized):
	return x.real if isinstance(x, complex) else x


@preparse_func
def imag(x: Vectorized):
	return x.imag if isinstance(x, complex) else 0.0


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
		return float(builtins.sum(1 for _ in range(n) if random.random() < p))
	simulations = py_int(simulations)
	return TiList([float(builtins.sum(1 for _ in range(n) if random.random() < p)) for _ in range(simulations)])


@preparse_func
def rand_int_no_rep(low: Real, high: Real):
	lst = [float(x) for x in range(py_int(low), py_int(high) + 1)]
	random.shuffle(lst)
	return TiList(lst)


# ANGLE MENU

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


# CATALOG (hyperbolic trig functions)

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
