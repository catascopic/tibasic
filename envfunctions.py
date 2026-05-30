from __future__ import annotations
import math
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from decorators import env_func, env_vectorized, forms_func
from environment import Environment
from errors import DomainError
from tiobjects import require_real, require_int, require_str


# ── Trig / coordinate helpers ─────────────────────────────────────────────────

def trig(func):
	"""Decorator for trig functions: vectorize and convert input from current angle mode."""
	@wraps(func)
	def apply(env, x):
		return func(env.to_rad(require_real(x)))
	return env_vectorized(apply)

def inv_trig(func):
	"""Decorator for inverse trig functions: vectorize and convert output to current angle mode."""
	@wraps(func)
	def apply(env, x):
		return env.from_rad(func(require_real(x)))
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

def _bal(env, m: int, roundvalue=None):
	"""Balance after m payments, using TVM variables from env."""
	r = env.i_pct / 100
	pv = env.pv
	pmt = env.pmt
	if roundvalue is not None:
		b = pv
		for _ in range(m):
			b = round(b * (1 + r) + pmt, roundvalue)
		return b
	if r == 0:
		return pv + pmt * m
	return pv * (1 + r) ** m + pmt * ((1 + r) ** m - 1) / r


@env_func
def bal(env, n, roundvalue=None):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	n = require_int(n)
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	return _bal(env, n, roundvalue)


@env_func
def sigma_prn(env, n1, n2, roundvalue=None):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	n1 = require_int(n1)
	n2 = require_int(n2)
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)


@env_func
def sigma_int(env, n1, n2, roundvalue=None):
	"""ΣInt(n1,n2[,roundvalue]) — interest paid from payment n1 through n2."""
	n1 = require_int(n1)
	n2 = require_int(n2)
	if roundvalue is not None:
		roundvalue = require_int(roundvalue)
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣInt: payment numbers must be positive")
	sprn = _bal(env, n2, roundvalue) - _bal(env, n1 - 1, roundvalue)
	return (n2 - n1 + 1) * env.pmt - sprn


# ── expr( ─────────────────────────────────────────────────────────────────────

@env_func
def expr(env, string):
	"""Evaluate a TiString as a TI-BASIC expression."""
	from parser import Parser
	string = require_str(string)
	with env.nest_guard(expr):
		return Parser(string.tokens, env).parse_expr()


# ── Clock / date-time ─────────────────────────────────────────────────────────

set_date   = env_func(Environment.set_date)
set_time   = env_func(Environment.set_time)
check_tmr  = env_func(Environment.check_tmr)
set_dt_fmt = env_func(Environment.set_dt_fmt)
set_tm_fmt = env_func(Environment.set_tm_fmt)
get_dt_str = env_func(Environment.get_dt_str)
get_tm_str = env_func(Environment.get_tm_str)
