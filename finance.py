"""Finance functions — the FINANCE menu and related TVM helpers.

  npv(      irr(      bal(      ΣPrn(     ΣInt(
  ►Nom(     ►Eff(     dbd(
  tvm_Pmt   tvm_I%    tvm_PV    tvm_N     tvm_FV
"""

import builtins
import math
from datetime import date

from accessors import Accessor
from core import TiList, require_list
from core import require_real, require_int, py_int
from preparse import preparse_func, Real, VectorizedReal, Env
from errors import DomainError, DimMismatchError


def _parse_dbd_date(d):
	"""Parse a TI Finance date float into a date object.

	Two formats (can be mixed in the same dbd call):
	  MM.DDYY  — integer part is month (1–12); decimal encodes 4 digits DDYY
	  DDMM.YY  — integer part is DDMM (≥100); decimal encodes 2 digits YY
	YY 00–49 → 2000–2049; 50–99 → 1950–1999.
	"""
	d = require_real(d)
	int_part = int(d)
	frac_part = d - int_part

	if int_part <= 12:
		raw = frac_part * 10000
		ddyy = builtins.round(raw)
		if builtins.abs(raw - ddyy) > 1e-6:
			raise DomainError(f"dbd: too many decimal places in MM.DDYY date {d!r}")
		month = int_part
		day, yy = divmod(ddyy, 100)
	elif int_part >= 100:
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
	return float((_parse_dbd_date(date2) - _parse_dbd_date(date1)).days)


def _expand_cash_flows(cflist, cffreq):
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


def _bal(env, n, roundvalue=None):
	"""Balance after n payments, using TVM variables from env."""
	r = env.i_pct / 100
	pv = env.pv
	pmt = env.pmt
	if roundvalue is not None:
		b = pv
		for _ in range(n):
			b = builtins.round(b * (1 + r) + pmt, roundvalue)
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
	return (n2 - n1 + 1) * env.pmt - sprn


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


# ── TVM solver (tvm_Pmt, tvm_I%, tvm_PV, tvm_N, tvm_FV) ───────────────────────
# Each solves the time-value-of-money equation for one variable given the other
# six.  All five share the "bare-or-called" token shape: used bare (tvm_Pmt) they
# read the stored finance variables; called (tvm_Pmt(N,I%,PV,FV,P/Y,C/Y)) the
# optional arguments first overwrite those variables, then the value is computed.
# The bare form is wired as the token's `nullary`, the called form as its
# `function` — see Token in titoken.py.
#
# Like _bal above, I% is the rate *per period* (periodic rate = I%/100); P/Y and
# C/Y are accepted and stored for signature compatibility but are not folded into
# the rate, and payments are end-of-period.  The governing equation is
#     0 = PV + PMT·(1 − (1+r)^(−N))/r + FV·(1+r)^(−N)     (r ≠ 0)
#     0 = PV + PMT·N + FV                                  (r = 0)
# The computed result is returned but not stored back into its own variable.


def _factors(r: float, n: float) -> tuple[float, float]:
	"""Discount factor v=(1+r)^(−N) and end-of-period annuity factor a=(1−v)/r.

	At r=0 the annuity factor's removable singularity is replaced by its limit, N
	(with v=1), so the three closed-form solvers below need no separate r=0 branch.
	"""
	if r == 0:
		return 1.0, n
	v = (1 + r) ** -n
	return v, (1 - v) / r


def tvm_pmt_value(env) -> float:
	"""Payment amount from the stored finance variables (the bare tvm_Pmt)."""
	v, a = _factors(env.i_pct / 100, env.n_tvm)
	if a == 0:
		raise DomainError("tvm_Pmt: N must be nonzero")
	return -(env.pv + env.fv * v) / a


def tvm_pv_value(env) -> float:
	"""Present value from the stored finance variables (the bare tvm_PV)."""
	v, a = _factors(env.i_pct / 100, env.n_tvm)
	return -(env.pmt * a + env.fv * v)


def tvm_fv_value(env) -> float:
	"""Future value from the stored finance variables (the bare tvm_FV)."""
	v, a = _factors(env.i_pct / 100, env.n_tvm)
	return -(env.pv + env.pmt * a) / v


def tvm_n_value(env) -> float:
	"""Number of payment periods from the stored finance variables (the bare tvm_N)."""
	r = env.i_pct / 100
	pv, pmt, fv = env.pv, env.pmt, env.fv
	if r == 0:
		if pmt == 0:
			raise DomainError("tvm_N: PMT must be nonzero when I%=0")
		return -(pv + fv) / pmt
	# (1+r)^N = (PMT − FV·r) / (PMT + PV·r); the ratio must be positive to take a log.
	num = pmt - fv * r
	den = pmt + pv * r
	if den == 0 or num / den <= 0:
		raise DomainError("tvm_N: no solution for these values")
	return math.log(num / den) / math.log(1 + r)


def tvm_i_pct_value(env) -> float:
	"""Periodic interest rate (percent) from the stored finance variables (bare tvm_I%).

	I% has no closed form, so the TVM residual is bracketed (scanning outward from
	just above −100% per period, widening the step) and then bisected.
	"""
	n, pv, pmt, fv = env.n_tvm, env.pv, env.pmt, env.fv

	def f(r):
		if builtins.abs(r) < 1e-12:
			return pv + pmt * n + fv
		v = (1 + r) ** -n
		return pv + pmt * (1 - v) / r + fv * v

	# Start just inside the rate that would overflow (1+r)^(−N); for the usual N>0
	# that is exp(−700/N)−1, comfortably below any real financial rate.
	r_min = math.exp(-700 / n) - 1 if n > 0 else -0.999999
	prev_r, prev_f = r_min, f(r_min)
	step = 1e-4
	bracket = None
	while prev_r < 100:
		r = prev_r + step
		fr = f(r)
		if prev_f == 0:
			return prev_r * 100
		if (prev_f < 0) != (fr < 0):
			bracket = (prev_r, r)
			break
		prev_r, prev_f, step = r, fr, step * 1.05
	if bracket is None:
		raise DomainError("tvm_I%: no solution found (ERR:NO SIGN CHG)")

	lo, hi = bracket
	f_lo = f(lo)
	for _ in range(200):
		mid = (lo + hi) / 2
		f_mid = f(mid)
		if builtins.abs(f_mid) < 1e-12 or (hi - lo) < 1e-15:
			return mid * 100
		if (f_lo < 0) != (f_mid < 0):
			hi = mid
		else:
			lo, f_lo = mid, f_mid
	return (lo + hi) / 2 * 100


def _apply_tvm(env, *, n=None, i_pct=None, pv=None, pmt=None, fv=None, py=None, cy=None) -> None:
	"""Overwrite the stored finance variables with any explicitly-passed arguments."""
	for value, attr in (
		(n, 'n_tvm'), (i_pct, 'i_pct'), (pv, 'pv'),
		(pmt, 'pmt'), (fv, 'fv'), (py, 'py'), (cy, 'cy'),
	):
		if value is not None:
			setattr(env, attr, require_real(value))


@preparse_func
def tvm_pmt(env: Env, n: Real = None, i_pct: Real = None, pv: Real = None, fv: Real = None, py: Real = None, cy: Real = None):
	"""tvm_Pmt([N,I%,PV,FV,P/Y,C/Y]) — payment amount."""
	_apply_tvm(env, n=n, i_pct=i_pct, pv=pv, fv=fv, py=py, cy=cy)
	return tvm_pmt_value(env)


@preparse_func
def tvm_i_pct(env: Env, n: Real = None, pv: Real = None, pmt: Real = None, fv: Real = None, py: Real = None, cy: Real = None):
	"""tvm_I%([N,PV,PMT,FV,P/Y,C/Y]) — interest rate per period (percent)."""
	_apply_tvm(env, n=n, pv=pv, pmt=pmt, fv=fv, py=py, cy=cy)
	return tvm_i_pct_value(env)


@preparse_func
def tvm_pv(env: Env, n: Real = None, i_pct: Real = None, pmt: Real = None, fv: Real = None, py: Real = None, cy: Real = None):
	"""tvm_PV([N,I%,PMT,FV,P/Y,C/Y]) — present value."""
	_apply_tvm(env, n=n, i_pct=i_pct, pmt=pmt, fv=fv, py=py, cy=cy)
	return tvm_pv_value(env)


@preparse_func
def tvm_n(env: Env, i_pct: Real = None, pv: Real = None, pmt: Real = None, fv: Real = None, py: Real = None, cy: Real = None):
	"""tvm_N([I%,PV,PMT,FV,P/Y,C/Y]) — number of payment periods."""
	_apply_tvm(env, i_pct=i_pct, pv=pv, pmt=pmt, fv=fv, py=py, cy=cy)
	return tvm_n_value(env)


@preparse_func
def tvm_fv(env: Env, n: Real = None, i_pct: Real = None, pv: Real = None, pmt: Real = None, py: Real = None, cy: Real = None):
	"""tvm_FV([N,I%,PV,PMT,P/Y,C/Y]) — future value."""
	_apply_tvm(env, n=n, i_pct=i_pct, pv=pv, pmt=pmt, py=py, cy=cy)
	return tvm_fv_value(env)


class TvmAccessor(Accessor):
	"""Bare-or-called accessor for tvm_Pmt, tvm_I%, tvm_PV, tvm_N, tvm_FV.

	Bare (no '('): resolves the stored finance variables via `_value_fn`.
	Called (with '('): updates the stored variables then resolves, via `_solver`.
	"""

	invocable = True

	def __init__(self, value_fn, solver):
		self._value_fn = value_fn
		self._solver = solver

	def resolve(self, env):
		return self._value_fn(env)

	def invoke(self, args):
		return self._solver.call_with_parser(args)
