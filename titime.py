"""Time and date functions."""
import builtins
from datetime import date

from core import TiList, TiString, py_int
from preparse import preparse_func, preparse_cmd_func, Real, Env
from environment import Environment
from errors import DomainError


@preparse_func
def time_cnv(seconds: Real):
	"""Convert a number of seconds into {days, hours, minutes, seconds}."""
	seconds = py_int(seconds)
	sign = -1 if seconds < 0 else 1
	remaining, secs = divmod(builtins.abs(seconds), 60)
	remaining, minutes = divmod(remaining, 60)
	days, hours = divmod(remaining, 24)
	return sign * TiList([float(days), float(hours), float(minutes), float(secs)])


@preparse_func
def dayofwk(year: Real, month: Real, day: Real):
	"""Day of week: 1=Sunday, 2=Monday, …, 7=Saturday."""
	try:
		d = date(py_int(year), py_int(month), py_int(day))
	except ValueError as e:
		raise DomainError(f"dayOfWk: invalid date ({year}/{month}/{day})") from e
	return float(d.isoweekday() % 7 + 1)


@preparse_cmd_func
def set_date(env: Env, year: Real, month: Real, day: Real):
	env.set_date(year, month, day)


@preparse_cmd_func
def set_time(env: Env, hour: Real, minute: Real, second: Real):
	env.set_time(hour, minute, second)


@preparse_cmd_func
def set_dt_fmt(env: Env, fmt: Real):
	fmt = py_int(fmt)
	if fmt not in {1, 2, 3}:
		raise DomainError(f"setDtFmt: expected 1, 2, or 3; got {fmt}")
	env.dt_fmt = fmt


@preparse_cmd_func
def set_tm_fmt(env: Env, fmt: Real):
	fmt = py_int(fmt)
	if fmt not in {12, 24}:
		raise DomainError(f"setTmFmt: expected 12 or 24; got {fmt}")
	env.tm_fmt = fmt


@preparse_func
def check_tmr(env: Env, start: Real):
	return float(int(env.now().timestamp()) - int(start))


@preparse_func
def get_dt_str(env: Env, fmt: Real):
	fmt = py_int(fmt)
	if fmt not in {1, 2, 3}:
		raise DomainError(f"getDtStr: invalid format {fmt}")
	return TiString.from_str(env.now().strftime(['%m/%d/%y', '%d/%m/%y', '%y/%m/%d'][fmt - 1]))


@preparse_func
def get_tm_str(env: Env, fmt: Real):
	fmt = py_int(fmt)
	now = env.now()
	if fmt == 24:
		time_str = now.strftime('%H:%M')
	elif fmt == 12:
		time_str = now.strftime('%I:%M %p').lstrip('0')
	else:
		raise DomainError(f"getTmStr: invalid format {fmt}")
	return TiString.from_str(time_str)
