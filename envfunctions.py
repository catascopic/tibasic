from __future__ import annotations
import cmath
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from argspec import expr as expr_spec, numeric, real, integer, vectorized, PassEnv
from decorators import preparse, TiCall, FUNC
from environment import Environment
from errors import TiSyntaxError, DomainError, NonRealAnsError
from modes import ComplexMode
from tiobjects import require_str


def trig(func):
	"""Decorator for trig functions: vectorize and convert input from current angle mode."""
	@preparse(FUNC)
	def apply(env: PassEnv, x: vectorized[real]):
		return func(env.to_rad(x))
	apply.__name__ = func.__name__
	return apply

def inv_trig(func):
	"""Decorator for inverse trig functions: vectorize, convert output to current angle mode,
	and raise DomainError for out-of-domain inputs."""
	@preparse(FUNC)
	def apply(env: PassEnv, x: vectorized[real]):
		try:
			return env.from_rad(func(x))
		except ValueError:
			raise DomainError(f"{func.__name__}: argument out of domain: {x}")
	apply.__name__ = func.__name__
	return apply


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
	if isinstance(x, complex):
		return cmath.sqrt(x)
	if x >= 0:
		return math.sqrt(x)
	if env.real_only:
		raise NonRealAnsError(f"√({x}): non-real result")
	return cmath.sqrt(x)

@preparse(FUNC)
def ln(env: PassEnv, x: vectorized[numeric]):
	if isinstance(x, complex):
		return cmath.log(x)
	if x > 0:
		return math.log(x)
	if x == 0:
		raise DomainError("ln: undefined for 0")
	if env.real_only:
		raise NonRealAnsError(f"ln({x}): non-real result")
	return cmath.log(x)

@preparse(FUNC)
def log(env: PassEnv, x: vectorized[numeric]):
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

@preparse(FUNC)
def log_base(env: PassEnv, x: vectorized[numeric], base: vectorized[numeric]):
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

@preparse(FUNC)
def rect_to_polar_radius(env: PassEnv, x: vectorized[real], y: vectorized[real]):
	return math.hypot(x, y)

@preparse(FUNC)
def rect_to_polar_angle(env: PassEnv, x: vectorized[real], y: vectorized[real]):
	return env.from_rad(math.atan2(y, x))

@preparse(FUNC)
def polar_to_rect_x(env: PassEnv, r: vectorized[real], theta: vectorized[real]):
	return r * math.cos(env.to_rad(theta))

@preparse(FUNC)
def polar_to_rect_y(env: PassEnv, r: vectorized[real], theta: vectorized[real]):
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


@preparse(FUNC)
def bal(env: PassEnv, n: integer, roundvalue: integer = None):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	n = int(n)
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	if roundvalue is not None:
		roundvalue = int(roundvalue)
	return _bal(env, n, roundvalue)


@preparse(FUNC)
def sigma_prn(env: PassEnv, n1: integer, n2: integer, roundvalue: integer = None):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	n1 = int(n1)
	n2 = int(n2)
	if roundvalue is not None:
		roundvalue = int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)


@preparse(FUNC)
def sigma_int(env: PassEnv, n1: integer, n2: integer, roundvalue: integer = None):
	"""ΣInt(n1,n2[,roundvalue]) — interest paid from payment n1 through n2."""
	n1 = int(n1)
	n2 = int(n2)
	if roundvalue is not None:
		roundvalue = int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣInt: payment numbers must be positive")
	sprn = _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)
	return (n2 - n1 + 1) * env.pmt.resolve() - sprn


####################
# STRING FUNCTIONS #
####################

@preparse(FUNC)
def expr(env: PassEnv, string: expr_spec):
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

@preparse(FUNC)
def acosh(env: PassEnv, x: vectorized[real]):
	if x >= 1:
		return math.acosh(x)
	if env.real_only:
		raise NonRealAnsError(f"acosh({x}): non-real result")
	return cmath.acosh(x)

@preparse(FUNC)
def atanh(env: PassEnv, x: vectorized[real]):
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

@preparse(FUNC)
def check_tmr(env: PassEnv):
	return Environment.check_tmr(env)

@preparse(FUNC)
def get_dt_str(env: PassEnv):
	return Environment.get_dt_str(env)

@preparse(FUNC)
def get_tm_str(env: PassEnv):
	return Environment.get_tm_str(env)
