import cmath
import builtins
import decimal as _decimal
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
from preparse import preparse_func, Real, ListOrMatrix, Vectorized, VectorizedReal, MatrixVectorized, AnyValue
from decorators import forms_func, vectorize

#####################
# DEDICATED BUTTONS #
#####################

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
	if isinstance(x, complex) or x < 0:
		return cmath.sqrt(x)
	return math.sqrt(x)

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
def pow10(x: Vectorized):
	return 10 ** x

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
def exp(x: Vectorized):
	return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

###############
# MATH / MATH #
###############

@preparse_func
def cbrt(x: Vectorized):
	if isinstance(x, complex):
		if x == 0:
			return 0
		return cmath.exp(cmath.log(x) / 3)
	return math.cbrt(x)

# TODO: fMin(

# TODO: fMax(

@preparse_func
def n_deriv(env: Env, formula: Thunk, var: NumericVar, val: Real, h: Real = 0.001) -> float:
	with env.nest_guard(n_deriv, max_depth=1), var.scoped():
		var.value = val + h
		fwd = formula.eval()
		var.value = val - h
		bwd = formula.eval()
	return (fwd - bwd) / (2 * h)

# G7K15 nodes (positive half + 0) and weights on [-1, 1]
_K15_NODES = [
	0.0,                0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
	0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126
]
_K15_WEIGHTS = [
	0.2094821410847278, 0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
	0.1406532597155259, 0.1047900103222502, 0.0630920926299786, 0.0229353220105292
]
# G7 uses nodes at indices 0, 2, 4, 6 (every other Kronrod node)
_G7_WEIGHTS  = [
	0.4179591836734694, None, 0.3818300505051189, None,
	0.2797053914892767, None, 0.1294849661688697, None
]

def _gk15(f, lo, hi):
	"""Apply G7K15 to [lo, hi]; return (k15_estimate, error)."""
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
	return k15, abs(k15 - g7)

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
	total = 0
	n = start
	with env.nest_guard(sigma), var.scoped():
		while n <= end:
			var.value = n
			total += formula.eval()
			n += 1
	return total

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

##############
# MATH / NUM #
##############

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

##############
# MATH / CPX #
##############

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

###############
# MATH / PROB #
###############

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

#################
# ANGLE / ANGLE #
#################

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

##############
# PRGM / CTL #
##############

@preparse_cmd
def if_cmd(env: Env, cond: Real):
	env.current_program().begin_if(bool(cond))

@no_arg_command
def then_cmd(a: ArgParser):
	"""Then without a preceding If: always a syntax error."""
	raise TiSyntaxError("Then without If")

@no_arg_command
def else_cmd(env):
	"""If we encounter Else this way, always skip the block.
	(Else blocks are only executed when encountered while skipping an If-Then block.)"""
	env.current_program().begin_else()

@preparse_cmd_func
def for_cmd(env: Env, var: NumericVar, start: Real, end: Real, step: Real = 1.0):
	env.current_program().begin_for(var, start, end, step)

@preparse_cmd
def while_cmd(env: Env, condition: Thunk):
	env.current_program().begin_while(condition)

@preparse_cmd
def repeat_cmd(env: Env, condition: Thunk):
	env.current_program().begin_repeat(condition)

@no_arg_command
def end_cmd(env):
	env.current_program().end_block()

@preparse_cmd
def lbl_cmd(env: Env, name: LabelName):
	"""Lbl is a no-op at runtime; just verify the syntax and that we're in a program."""
	env.current_program()  # raises if not in a program

@preparse_cmd
def goto_cmd(env: Env, name: LabelName):
	env.current_program().goto(name)

@preparse_cmd_func
def is_gt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_program().is_gt(var, threshold)

@preparse_cmd_func
def ds_lt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_program().ds_lt(var, threshold)

@preparse_cmd
def prgm(env: Env, name: ProgramName):
	env.run_program(name)

@no_arg_command
def return_cmd(env):
	env.current_program()  # raises if not in a program
	raise ReturnSignal()

@no_arg_command
def stop_cmd(env):
	env.current_program()  # raises if not in a program
	raise StopSignal()

@preparse_bunch
def del_var(var: AnyVar):
	"""DelVar variable — clear one variable without consuming the statement separator.

	end=NONE leaves the parser untouched (no finalizer), so DelVar bunches with
	whatever follows: DelVar ADelVar B and DelVar ADisp X are both valid on the
	same line.  Does not update Ans.
	"""
	var.value = None

#############
# PRGM / IO #
#############

@forms_func
def disp(args: ArgParser):
	if args.has_next:
		while True:
			print(args.expr())
			if not args.has_next:
				break
	else:
		pass  # args.env.focus_home()
	args.end_cmd()
	
########
# DRAW #
########

# See draw.py

#################
# DISTR / DISTR #
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

################
# DISTR / DRAW #
################

# shade_norm(
# shade_t
# shade_chi_sq
# shade_f
# See draw.py

#################
# MATRIX / MATH #
#################

@preparse_func
def det(mat: TiMatrix):
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

@preparse_func
def identity(n: Real):
	size = py_int(n)
	return TiMatrix([[float(r == c) for c in range(size)] for r in range(size)])

@preparse_func
def rand_m(rows: Real, cols: Real):
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
		raise InvalidDimError(f"ref/rref: TiMatrix must have at least as many columns as rows")
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

@preparse_func
def ref(mat: TiMatrix):
	return _row_reduce(mat, lambda pivot_row, rows: range(pivot_row + 1, rows))

@preparse_func
def rref(mat: TiMatrix):
	return _row_reduce(mat, lambda pivot_row, rows: range(rows))

@preparse_func
def row_swap(mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row1, mat.get_row(row2))
	result.set_row(row2, mat.get_row(row1))
	return result

@preparse_func
def row_plus(mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row2, [a + b for a, b in zip(mat.get_row(row2), mat.get_row(row1))])
	return result

@preparse_func
def times_row(factor: Real, mat: TiMatrix, row: Real):
	result = mat.copy()
	result.set_row(row, [factor * x for x in mat.get_row(row)])
	return result

@preparse_func
def times_row_plus(factor: Real, mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row2, [factor * a + b for a, b in zip(mat.get_row(row1), mat.get_row(row2))])
	return result
