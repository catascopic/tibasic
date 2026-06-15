"""Finance functions — the FINANCE menu and related TVM helpers.

  npv(      irr(      bal(      ΣPrn(     ΣInt(
  ►Nom(     ►Eff(     dbd(
"""

import builtins
from datetime import date

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
	return (_parse_dbd_date(date2) - _parse_dbd_date(date1)).days


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
	r = env.i_pct.resolve() / 100
	pv = env.pv.resolve()
	pmt = env.pmt.resolve()
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
	return (n2 - n1 + 1) * env.pmt.resolve() - sprn


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
