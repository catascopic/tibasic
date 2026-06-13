"""Time and date functions."""
import builtins
from datetime import date

from parser import ArgParser

from core import TiList
from core import py_int
from preparse import preparse_func, Real, Env
from preparse import TiCall
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
	return sign * TiList([days, hours, minutes, secs])


@preparse_func
def dayofwk(year: Real, month: Real, day: Real):
	"""Day of week: 1=Sunday, 2=Monday, …, 7=Saturday."""
	try:
		d = date(py_int(year), py_int(month), py_int(day))
	except ValueError as e:
		raise DomainError(f"dayOfWk: invalid date ({year}/{month}/{day})") from e
	return float(d.isoweekday() % 7 + 1)


class _set_time_wrapper(TiCall):
	def call_with_parser(self, args: ArgParser):
		values = args.parse_args()
		args.end_paren_cmd()
		return self(args.env, *values)


set_date   = _set_time_wrapper(Environment.set_date)
set_time   = _set_time_wrapper(Environment.set_time)
set_dt_fmt = _set_time_wrapper(Environment.set_dt_fmt)
set_tm_fmt = _set_time_wrapper(Environment.set_tm_fmt)


@preparse_func
def check_tmr(env: Env):
	return Environment.check_tmr(env)

@preparse_func
def get_dt_str(env: Env):
	return Environment.get_dt_str(env)

@preparse_func
def get_tm_str(env: Env):
	return Environment.get_tm_str(env)
