from __future__ import annotations
import cmath
import math
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from argspec import env, expr as expr_spec, optional, numeric, vectorized, PassEnv
from decorators import preparse, preparse_vectorized, TiCall, FUNC
from environment import Environment
from errors import TiSyntaxError, DomainError, NonRealAnsError
from modes import ComplexMode
from tiobjects import require_num, require_real, require_int, require_str, py_int


def trig(func):
	"""Decorator for trig functions: vectorize and convert input from current angle mode."""
	@wraps(func)
	def apply(env, x):
		return func(env.to_rad(require_real(x)))
	return preparse_vectorized(env, expr_spec)(apply)

def inv_trig(func):
	"""Decorator for inverse trig functions: vectorize, convert output to current angle mode,
	and raise DomainError for out-of-domain inputs."""
	@wraps(func)
	def apply(env, x):
		try:
			return env.from_rad(func(require_real(x)))
		except ValueError:
			raise DomainError(f"{func.__name__}: argument out of domain: {x}")
	return preparse_vectorized(env, expr_spec)(apply)


##################
# MAIN FUNCTIONS #
##################

@trig
def sin(x):
	return math.sin(x)

@trig
def cos(x):
	return math.cos(x)

@trig
def tan(x):
	return math.tan(x)

@inv_trig
def asin(x):
	return math.asin(x)

@inv_trig
def acos(x):
	return math.acos(x)

@inv_trig
def atan(x):
	return math.atan(x)

@preparse(FUNC)
def sqrt(env: PassEnv, x: vectorized[numeric]):
	require_num(x)
	if isinstance(x, complex):
		return cmath.sqrt(x)
	if x >= 0:
		return math.sqrt(x)
	if env.real_only:
		raise NonRealAnsError(f"√({x}): non-real result")
	return cmath.sqrt(x)

@preparse_vectorized(env, expr_spec)
def ln(env, x):
	require_num(x)
	if isinstance(x, complex):
		return cmath.log(x)
	if x > 0:
		return math.log(x)
	if x == 0:
		raise DomainError("ln: undefined for 0")
	if env.real_only:
		raise NonRealAnsError(f"ln({x}): non-real result")
	return cmath.log(x)

@preparse_vectorized(env, expr_spec)
def log(env, x):
	require_num(x)
	if isinstance(x, complex):
		return cmath.log10(x)
	if x > 0:
		return math.log10(x)
	if x == 0:
		raise DomainError("log: undefined for 0")
	if env.real_only:
		raise NonRealAnsError(f"log({x}): non-real result")
	return cmath.log10(x)

####################
# MATH FUNCTIONS   #
####################

@preparse_vectorized(env, expr_spec, expr_spec)
def log_base(env, x, base):
	require_num(x)
	require_num(base)
	if isinstance(x, complex) or isinstance(base, complex):
		return cmath.log(x, base)
	if base <= 0 or base == 1:
		raise DomainError(f"logBASE: base must be positive and ≠ 1, got {base}")
	if x == 0:
		raise DomainError("logBASE: undefined for x=0")
	if x > 0:
		return math.log(x, base)
	if env.real_only:
		raise NonRealAnsError(f"logBASE({x}, {base}): non-real result")
	return cmath.log(x, base)

####################
# ANGLE FUNCTIONS  #
####################

@preparse_vectorized(env, expr_spec, expr_spec)
def rect_to_polar_radius(env, x, y):
	return math.hypot(require_real(x), require_real(y))

@preparse_vectorized(env, expr_spec, expr_spec)
def rect_to_polar_angle(env, x, y):
	return env.from_rad(math.atan2(require_real(y), require_real(x)))

@preparse_vectorized(env, expr_spec, expr_spec)
def polar_to_rect_x(env, r, theta):
	return require_real(r) * math.cos(env.to_rad(require_real(theta)))

@preparse_vectorized(env, expr_spec, expr_spec)
def polar_to_rect_y(env, r, theta):
	return require_real(r) * math.sin(env.to_rad(require_real(theta)))

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


@preparse(env, expr_spec, optional(expr_spec))
def bal(env, n, roundvalue=None):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	n = py_int(n)
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	if roundvalue is not None:
		roundvalue = py_int(roundvalue)
	return _bal(env, n, roundvalue)


@preparse(env, expr_spec, expr_spec, optional(expr_spec))
def sigma_prn(env, n1, n2, roundvalue=None):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	n1 = py_int(n1)
	n2 = py_int(n2)
	if roundvalue is not None:
		roundvalue = py_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)


@preparse(env, expr_spec, expr_spec, optional(expr_spec))
def sigma_int(env, n1, n2, roundvalue=None):
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
# STRING FUNCTIONS #
####################

@preparse(env, expr_spec)
def expr(env, string):
	"""Evaluate a TiString as a TI-BASIC expression."""
	from parser import Parser
	require_str(string)
	with env.nest_guard(expr):
		p = Parser(string.tokens, env)
		result = p.parse_expr()
		if p.has_next:
			raise TiSyntaxError(f"expr: evaluated string must contain a single expression; got: {string!r}")
		return result


###########
# CATALOG #
###########

@preparse_vectorized(env, expr_spec)
def acosh(env, x):
	require_real(x)
	if x >= 1:
		return math.acosh(x)
	if env.real_only:
		raise NonRealAnsError(f"acosh({x}): non-real result")
	return cmath.acosh(x)

@preparse_vectorized(env, expr_spec)
def atanh(env, x):
	require_real(x)
	if abs(x) < 1:
		return math.atanh(x)
	if abs(x) == 1:
		raise DomainError(f"atanh: undefined for ±1")
	if env.real_only:
		raise NonRealAnsError(f"tanh⁻¹({x}): non-real result")
	return cmath.atanh(x)


class set_time_wrapper(TiCall):
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_paren_cmd()
		return self(a.env, *args)


set_date   = set_time_wrapper(Environment.set_date)
set_time   = set_time_wrapper(Environment.set_time)
set_dt_fmt = set_time_wrapper(Environment.set_dt_fmt)
set_tm_fmt = set_time_wrapper(Environment.set_tm_fmt)

check_tmr  = preparse(env)(Environment.check_tmr)
get_dt_str = preparse(env)(Environment.get_dt_str)
get_tm_str = preparse(env)(Environment.get_tm_str)
