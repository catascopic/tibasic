from __future__ import annotations
import operator
import purefunctions
from functools import wraps
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from tiobjects import TiList, TiMatrix, require_real, require_int, require_str, require_list
from environment import Variable, Environment
from errors import DomainError, DataTypeError, ArgumentError, IncrementError, InvalidDimError


def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	args = a.parse_args()
	if isinstance(ans, TiMatrix):
		return ans[args]
	if len(args) != 1:
		raise ArgumentError(f"Too many arguments: {args}")
	(arg,) = args
	if isinstance(ans, TiList):
		return ans[arg]
	return ans * arg


def seq(a: ArgParser) -> TiList:
	formula = a.thunk()
	var = a.numeric_var()
	start = a.expr()
	end = a.expr()
	step = a.expr(optional=True, default=1)
	a.end()
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
	variable = var.variable
	with a.env.nest_guard(seq), a.env.scoped_var(variable):
		while op(n, end):
			variable.set(a.env, n)
			result.append(formula.eval())
			n += step
	return TiList(result)


def sigma(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
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
	var = a.numeric_var()
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
	var = a.numeric_var()
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
		raise DataTypeError("Matr►list: first argument must be a matrix")
	if a.peek().is_list_start():
		list_refs = [a.list_var()]
		while a.has_next():
			list_refs.append(a.list_var())
		for ref, col_data in zip(list_refs, zip(*mat.data)):
			ref.set(a.env, TiList(list(col_data)))
	else:
		col = require_int(a.expr()) - 1
		if not (0 <= col < mat.cols):
			raise InvalidDimError(
				f"Matr►list: column {col + 1} out of range for {mat.rows}×{mat.cols} matrix"
			)
		ref = a.list_var()
		ref.set(a.env, TiList([mat.data[r][col] for r in range(mat.rows)]))


def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(require_list(a.expr()))
		if not a.has_next():
			raise ArgumentError("List►matr: expected matrix variable as last argument")
		if a.peek().is_matrix_var():
			mat_var = a.matrix_var()
			break
	mat_var.set(a.env, TiMatrix([
		list(row) for row in zip_longest(*[lst.data for lst in list_vals], fillvalue=0)
	]))


# ── Amortization: bal(, ΣPrn(, ΣInt( ────────────────────────────────────────────

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


def bal(a: ArgParser):
	"""bal(n[,roundvalue]) — remaining balance after n payments."""
	n = require_int(a.expr())
	if n < 0:
		raise DomainError("bal: n must be non-negative")
	roundvalue = require_int(a.expr()) if a.has_next() else None
	return _bal(a.env, n, roundvalue)


def sigma_prn(a: ArgParser):
	"""ΣPrn(n1,n2[,roundvalue]) — principal paid from payment n1 through n2."""
	n1 = require_int(a.expr())
	n2 = require_int(a.expr())
	roundvalue = require_int(a.expr()) if a.has_next() else None
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣPrn: payment numbers must be positive")
	return _bal(a.env, n2, roundvalue) - _bal(a.env, n1 - 1, roundvalue)


def sigma_int(a: ArgParser):
	"""ΣInt(n1,n2[,roundvalue]) — interest paid from payment n1 through n2."""
	n1 = require_int(a.expr())
	n2 = require_int(a.expr())
	roundvalue = require_int(a.expr()) if a.has_next() else None
	if n1 < 1 or n2 < 0:
		raise DomainError("ΣInt: payment numbers must be positive")
	sprn = _bal(a.env, n2, roundvalue) - _bal(a.env, n1 - 1, roundvalue)
	return (n2 - n1 + 1) * a.env.pmt - sprn


# ── Sorting / filling ────────────────────────────────────────────────────────────

# THESE ARE COMMANDS, NOT FUNCTIONS!

def _sort(a: ArgParser, reverse: bool):
	main = a.list_var().get(a.env)
	deps = []
	while a.has_next():
		deps.append(a.list_var().get(a.env))
	if not deps:
		main.data.sort(reverse=reverse)
	else:
		data = main.data
		indices = sorted(range(len(data)), key=lambda i: data[i], reverse=reverse)
		main.data = [data[i] for i in indices]
		for d in deps:
			d.data = [d.data[i] for i in indices]


def sort_a(a: ArgParser):
	_sort(a, False)  # ascending


def sort_d(a: ArgParser):
	_sort(a, True)   # descending


def fill(a: ArgParser):
	# Fill(value, listname) or Fill(value, matrixname) — value comes first
	x = require_real(a.expr())
	if a.peek().is_matrix_var():
		lst = a.matrix_var().get(a.env)
		a.end()
		for row in lst.data:
			for i in range(len(row)):
				row[i] = x
	else:
		lst = a.list_var().get(a.env)
		a.end()
		for i in range(len(lst.data)):
			lst.data[i] = x


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
