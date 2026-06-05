from __future__ import annotations
import cmath
import math
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from decorators import env_func, env_vectorized, TiCall
from environment import Environment
from errors import TiSyntaxError, DomainError, NonRealAnsError
from modes import ComplexMode
from tiobjects import require_num, require_real, require_int, require_str


# ── Trig / coordinate helpers ─────────────────────────────────────────────────

def trig(func):
	"""Decorator for trig functions: vectorize and convert input from current angle mode."""
	@wraps(func)
	def apply(env, x):
		return func(env.to_rad(require_real(x)))
	return env_vectorized(apply)

def inv_trig(func):
	"""Decorator for inverse trig functions: vectorize, convert output to current angle mode,
	and raise DomainError for out-of-domain inputs."""
	@wraps(func)
	def apply(env, x):
		try:
			return env.from_rad(func(require_real(x)))
		except ValueError:
			raise DomainError(f"{func.__name__}: argument out of domain: {x}")
	return env_vectorized(apply)


# ── Trig functions ────────────────────────────────────────────────────────────

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


# ── Logarithms / roots (env-aware for ComplexMode) ───────────────────────────

@env_vectorized
def sqrt(env, x):
	require_num(x)
	if isinstance(x, complex):
		return cmath.sqrt(x)
	if x >= 0:
		return math.sqrt(x)
	if env.real_only:
		raise NonRealAnsError(f"√({x}): non-real result")
	return cmath.sqrt(x)

@env_vectorized
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

@env_vectorized
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

@env_vectorized
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

@env_vectorized
def acosh(env, x):
	require_real(x)
	if x >= 1:
		return math.acosh(x)
	if env.real_only:
		raise NonRealAnsError(f"acosh({x}): non-real result")
	return cmath.acosh(x)

@env_vectorized
def atanh(env, x):
	require_real(x)
	if abs(x) < 1:
		return math.atanh(x)
	if abs(x) == 1:
		raise DomainError(f"atanh: undefined for ±1")
	if env.real_only:
		raise NonRealAnsError(f"tanh⁻¹({x}): non-real result")
	return cmath.atanh(x)


# ── Coordinate conversions ────────────────────────────────────────────────────

@env_vectorized
def rect_to_polar_radius(env, x, y):
	return math.hypot(require_real(x), require_real(y))

@env_vectorized
def rect_to_polar_angle(env, x, y):
	return env.from_rad(math.atan2(require_real(y), require_real(x)))

@env_vectorized
def polar_to_rect_x(env, r, theta):
	return require_real(r) * math.cos(env.to_rad(require_real(theta)))

@env_vectorized
def polar_to_rect_y(env, r, theta):
	return require_real(r) * math.sin(env.to_rad(require_real(theta)))


# ── Amortization: bal(, ΣPrn(, ΣInt( ─────────────────────────────────────────

def _bal(env, n, roundvalue=None):
	"""Balance after n payments, using TVM variables from env."""
	r = env.i_pct.resolve() / 100
	pv = env.pv.resolve()
	pmt = env.pmt.resolve()
	if roundvalue is not None:
		roundvalue = int(roundvalue)
		b = pv
		for _ in range(int(n)):
			b = round(b * (1 + r) + pmt, roundvalue)
		return b
	if r == 0:
		return pv + pmt * n
	return pv * (1 + r) ** n + pmt * ((1 + r) ** n - 1) / r


@env_func
def bal(env, n, roundvalue=None):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	require_int(n)
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	return _bal(env, n, roundvalue)


@env_func
def sigma_prn(env, n1, n2, roundvalue=None):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	require_int(n1)
	require_int(n2)
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)


@env_func
def sigma_int(env, n1, n2, roundvalue=None):
	"""ΣInt(n1,n2[,roundvalue]) — interest paid from payment n1 through n2."""
	require_int(n1)
	require_int(n2)
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣInt: payment numbers must be positive")
	sprn = _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)
	return (n2 - n1 + 1) * env.pmt.resolve() - sprn


# ── expr( ─────────────────────────────────────────────────────────────────────

@env_func
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


# ── Clock / date-time ─────────────────────────────────────────────────────────


class set_time_wrapper(TiCall):
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_paren_cmd()
		return self(a.env, *args)


set_date   = set_time_wrapper(Environment.set_date)
set_time   = set_time_wrapper(Environment.set_time)
set_dt_fmt = set_time_wrapper(Environment.set_dt_fmt)
set_tm_fmt = set_time_wrapper(Environment.set_tm_fmt)

check_tmr  = env_func(Environment.check_tmr)
get_dt_str = env_func(Environment.get_dt_str)
get_tm_str = env_func(Environment.get_tm_str)
