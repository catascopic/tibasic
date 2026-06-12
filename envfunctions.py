from __future__ import annotations
import cmath
import math
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

import operators
from preparse import preparse_func, Real, Vectorized, VectorizedReal, Thunk, NumericVar, Env
from decorators import forms_func, TiCall
from environment import Environment
from errors import (
	TiSyntaxError, DomainError,
	DataTypeError, IncrementError, UndefinedError,
)
from modes import ComplexMode
from tiobjects import TiList, TiMatrix, TiString
from core import py_int


def _inv_trig(func, env, x):
	try:
		return env.from_rad(func(x))
	except ValueError:
		raise DomainError(f"{func.__name__}: argument out of domain: {x}")


##################
# MAIN FUNCTIONS #
##################

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

####################
# MATH FUNCTIONS   #
####################

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

####################
# ANGLE FUNCTIONS  #
####################

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

############
# FINANCE  #
############

def _bal(env, n, roundvalue=None):
	"""Balance after n payments, using TVM variables from env."""
	r = env.i_pct.resolve() / 100
	pv = env.pv.resolve()
	pmt = env.pmt.resolve()
	if roundvalue is not None:
		b = pv
		for _ in range(n):
			b = round(b * (1 + r) + pmt, roundvalue)
		return b
	if r == 0:
		return pv + pmt * n
	return pv * (1 + r) ** n + pmt * ((1 + r) ** n - 1) / r


@preparse_func
def bal(env: Env, n: Real, roundvalue: Real = None):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	n = py_int(n)
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	if roundvalue is not None:
		roundvalue = py_int(roundvalue)
	return _bal(env, n, roundvalue)


@preparse_func
def sigma_prn(env: Env, n1: Real, n2: Real, roundvalue: Real = None):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	n1 = py_int(n1)
	n2 = py_int(n2)
	if roundvalue is not None:
		roundvalue = py_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)


@preparse_func
def sigma_int(env: Env, n1: Real, n2: Real, roundvalue: Real = None):
	"""ΣInt(n1,n2[,roundvalue]) — interest paid from payment n1 through n2."""
	n1 = py_int(n1)
	n2 = py_int(n2)
	if roundvalue is not None:
		roundvalue = py_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣInt: payment numbers must be positive")
	sprn = _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)
	return (n2 - n1 + 1) * env.pmt.resolve() - sprn


####################
# CALCULUS / SUMS  #
####################

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
def seq(env: Env, formula: Thunk, var: NumericVar, start: Real, end: Real, step: Real = 1) -> TiList:
	n = start
	result = []
	if step == 0:
		raise IncrementError("seq: step cannot be zero")

	if step > 0:
		if start > end + 1e-10:
			raise IncrementError(f"seq: step is positive but start ({start}) > end ({end})")
		op = operator.le
		end += 1e-10
	else:
		if start < end - 1e-10:
			raise IncrementError(f"seq: step is negative but start ({start}) < end ({end})")
		op = operator.ge
		end -= 1e-10

	with env.nest_guard(seq), var.scoped():
		while op(n, end):
			var.value = n
			result.append(formula.eval())
			n += step

	return TiList(result)


####################
# ANS / DIMENSIONS #
####################

@forms_func
def ans_index_or_mul(args: ArgParser):
	ans = args.env.ans
	if isinstance(ans, TiList):
		(index,) = args.parse_indices(1)
		return ans[index]
	if isinstance(ans, TiMatrix):
		return ans[args.parse_indices(2)]
	b = args.expr()
	args.end_func()
	return operators.mul(ans, b)

# You'd think dim could be a pure function, right? For a while, it was.
# But then I realized empty lists are illegal everywhere, with a single exception.
# dim( reads a variable's stored dimension rather than its value: it accesses
# .value (raw storage) instead of .resolve(), so dim(L1) returns 0 for an empty
# list where a bare reference to L1 would raise InvalidDimError.  This mirrors
# the calculator, where dim( works on both sides of → for the same reason.

@forms_func
def dim(args: ArgParser):
	if args.peek().is_list_start():
		var = args.list_var()
		val = var.value
		if val is None:
			raise UndefinedError(f"Undefined list variable")
		args.end_func()
		return len(val)

	value = args.expr()
	args.end_func()
	if isinstance(value, TiList):
		return len(value)
	# Don't need direct matrix variable access because empty matrices aren't possible
	if isinstance(value, TiMatrix):
		return TiList([value.rows, value.cols])
	raise DataTypeError(f"dim: expected list or matrix; got {value}")


###########
# CATALOG #
###########

@preparse_func
def acosh(x: VectorizedReal):
	if x >= 1:
		return math.acosh(x)
	return cmath.acosh(x)

@preparse_func
def atanh(x: VectorizedReal):
	if abs(x) < 1:
		return math.atanh(x)
	if abs(x) == 1:
		raise DomainError(f"atanh: undefined for ±1")
	return cmath.atanh(x)


class set_time_wrapper(TiCall):
	def call_with_parser(self, args: ArgParser):
		values = args.parse_args()
		args.end_paren_cmd()
		return self(args.env, *values)


set_date   = set_time_wrapper(Environment.set_date)
set_time   = set_time_wrapper(Environment.set_time)
set_dt_fmt = set_time_wrapper(Environment.set_dt_fmt)
set_tm_fmt = set_time_wrapper(Environment.set_tm_fmt)

@preparse_func
def check_tmr(env: Env):
	return Environment.check_tmr(env)

@preparse_func
def get_dt_str(env: Env):
	return Environment.get_dt_str(env)

@preparse_func
def get_tm_str(env: Env):
	return Environment.get_tm_str(env)
