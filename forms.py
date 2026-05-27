from __future__ import annotations
import operator
import purefunctions
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from tiobjects import TiList, TiMatrix, require_real, require_str
from environment import Variable, Environment


def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	args = a.parse_args()
	if isinstance(ans, TiMatrix):
		return ans[args]
	if len(args) != 1:
		raise ValueError(f"Too many arguments: {args}")
	(arg,) = args
	if isinstance(ans, TiList):
		return ans[arg]
	return ans * arg


def seq(a: ArgParser) -> TiList:
	formula = a.thunk()
	var = a.real_var()
	start = a.expr()
	end = a.expr()
	step = a.expr(optional=True, default=1)
	a.end()
	n = start
	result = []
	if step == 0:
		raise ValueError("seq: step cannot be zero")
	if step > 0:
		if start > end + 1e-10:
			raise ValueError(f"seq: step is positive but start ({start}) > end ({end})")
		op = operator.le
		end += 1e-10
	else:
		if start < end - 1e-10:
			raise ValueError(f"seq: step is negative but start ({start}) < end ({end})")
		op = operator.ge
		end -= 1e-10
	variable = var.variable
	with a.env.nest_guard(seq), a.env.scoped_var(variable):
		while op(n, end):
			variable.set(a.env, n)
			result.append(formula.eval())
			n += step
	return TiList(result)


def sigma(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.real_var()
	start = a.expr()
	end = a.expr()
	a.end()
	total = 0
	n = start
	variable = var.variable
	with a.env.nest_guard(sigma), a.env.scoped_var(variable):
		while n <= end:
			variable.set(a.env, n)
			total += formula.eval()
			n += 1
	return total


def n_deriv(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.real_var()
	val = a.expr()
	h = a.expr(optional=True, default=0.001)
	a.end()
	variable = var.variable
	with a.env.nest_guard(n_deriv, max_depth=1), a.env.scoped_var(variable):
		variable.set(a.env, val + h)
		fwd = formula.eval()
		variable.set(a.env, val - h)
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


def fn_int(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.real_var()
	lo = a.expr()
	hi = a.expr()
	tol = a.expr(optional=True, default=1e-5)
	a.end()
	variable = var.variable
	with a.env.nest_guard('fnInt'), a.env.scoped_var(variable):
		def f(x):
			variable.set(a.env, x)
			return formula.eval()
		return _adaptive_gk15(f, lo, hi, tol)


def matr_to_list(a: ArgParser) -> None:
	mat = a.expr()
	if not isinstance(mat, TiMatrix):
		raise ValueError("Matr►list: first argument must be a matrix")
	if a.next_is_list_var():
		list_refs = [a.list_var()]
		while a.has_next_arg():
			list_refs.append(a.list_var())
		a.end()
		for col, ref in enumerate(list_refs):
			ref.set(a.env, TiList([mat.data[r][col] for r in range(mat.rows)]))
	else:
		col = int(a.expr()) - 1
		ref = a.list_var()
		a.end()
		ref.set(a.env, TiList([mat.data[r][col] for r in range(mat.rows)]))


def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(a.expr())
		if not a.has_next_arg():
			raise ValueError("List►matr: expected matrix variable as last argument")
		# TODO: let ArgParser peek?
		# if a.peek().is_matrix_var():
		if a.next_is_matrix_var():
			mat_var = a.matrix_var()
			break
	a.end()
	cols = len(list_vals)
	rows = max(len(lst) for lst in list_vals)
	# TODO: can we use zip with a default value?
	mat_var.set(a.env, TiMatrix([
		[list_vals[c].data[r] if r < len(list_vals[c]) else 0 for c in range(cols)]
		for r in range(rows)
	]))


# ── Sorting / filling ────────────────────────────────────────────────────────────

# THESE ARE COMMANDS, NOT FUNCTIONS!

def _sort(a: ArgParser, reverse: bool):
	main = a.list_var().get(a.env)
	deps = []
	while a.has_next_arg():
		deps.append(a.list_var().get(a.env))
	a.end()
	if not deps:
		sort(main, reverse=reverse)
	else:
		data = main.data
		indices = sorted(range(len(data)), key=lambda i: data[i], reverse=reverse)
		lst.data = [data[i] for i in indices]
		for d in deps:
			d.data = [d.data[i] for i in indices]


def sort_a(a: ArgParser):
	_sort(a, True)


def sort_d(a: ArgParser):
	_sort(a, False)


def fill(a: ArgParser):
	var = a.var().get(a.env)
	x = require_real(a.expr())
	a.end()
	if isinstance(lst, TiList):
		for i in range(len(lst.data)):
			lst.data[i] = x
	elif isinstance(lst, TiMatrix):
		for row in lst.data:
			for i in range(len(row)):
				row[i] = x
	else:
		raise ValueError(f"fill: expected list or matrix, got {type(lst).__name__}")


# ── env_func decorator ────────────────────────────────────────────────────────

def env_func(func):
	"""Convert an (env, arg1, ...) function into an ArgParser handler."""
	@wraps(func)
	def wrapper(a: ArgParser):
		return func(a.env, *a.parse_args())
	return wrapper


# ── expr( ─────────────────────────────────────────────────────────────────────

def expr(a: ArgParser):
	"""Evaluate a TiString as a TI-BASIC expression."""
	from parser import Parser
	string = require_str(a.expr())
	a.end()
	with a.env.nest_guard(expr):
		return Parser(string.tokens, a.env).parse_expr()


# ── Clock / date-time commands and functions ──────────────────────────────────

@env_func
def set_date(env, year, month, day):
	env.set_date(year, month, day)

@env_func
def set_time(env, hour, minute, second):
	env.set_time(hour, minute, second)

@env_func
def check_tmr(env, start):
	return env.check_tmr(start)

@env_func
def set_dt_fmt(env, fmt):
	env.set_dt_fmt(fmt)

@env_func
def set_tm_fmt(env, fmt):
	env.set_tm_fmt(fmt)

@env_func
def get_dt_str(env, fmt):
	return env.get_dt_str(fmt)

@env_func
def get_tm_str(env, fmt):
	return env.get_tm_str(fmt)
