from __future__ import annotations
import operator
import purefunctions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Parser, ParseError
	from tokens import Token

from contextlib import contextmanager
from tiobjects import TiList, TiMatrix, require_real
from environment import Variable


@contextmanager
def _scoped_var(env, variable):
	saved = variable.get(env)
	try:
		yield
	finally:
		variable.set(env, saved)


def _parse_method(method):
	def wrapper(self, optional=False, default=None):
		return self._arg(lambda: method(self), optional, default)
	return wrapper


class ArgParser:
	"""Stateful helper for parsing comma-separated function arguments."""

	def __init__(self, parser: Parser):
		self._parser = parser
		self._first = True

	def _sep(self):
		if self._first:
			self._first = False
		else:
			self._parser.expect_comma()

	def _arg(self, parse_fn, optional=False, default=None):
		if optional:
			if self._first:
				if self._parser.at_end() or self._parser.peek_is_rparen():
					return default
				self._first = False
				return parse_fn()
			if not self._parser.eat_if_comma():
				return default
			return parse_fn()
		self._sep()
		return parse_fn()

	@_parse_method
	def expr(self):
		return self._parser.parse_expr()

	@_parse_method
	def thunk(self):
		return self._parser.capture()

	@_parse_method
	def real_var(self) -> Token:
		tok = self._parser.advance()
		if not tok.is_numeric_var():
			raise ValueError(f"Expected a real variable, got {tok.text!r}")
		return tok

	@_parse_method
	def list_var(self) -> Variable:
		return self._parser.parse_list_var()

	@_parse_method
	def matrix_var(self) -> Variable:
		return self._parser.parse_matrix_var()

	@_parse_method
	def var(self) -> Variable:
		return self._parser.parse_any_var()

	def end(self):
		self._parser.eat_if_rparen()

	@property
	def env(self):
		return self._parser.env

	def has_next_arg(self) -> bool:
		return self._parser.peek_is_comma()

	def parse_args(self) -> list:
		args = []
		if not self._parser.at_end() and not self._parser.peek_is_rparen():
			args.append(self.expr())
			while self.has_next_arg():
				args.append(self.expr())
		self.end()
		return args

	def next_is_matrix_var(self) -> bool:
		pos = self._parser.pos + 1
		return pos < len(self._parser.tokens) and self._parser.tokens[pos].is_matrix_var()

	def next_is_list_var(self) -> bool:
		pos = self._parser.pos + 1
		return pos < len(self._parser.tokens) and self._parser.tokens[pos].is_list_var_start()


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
	return purefunctions.mul(ans, arg)


def seq(a: ArgParser) -> TiList:
	formula = a.thunk()
	var = a.real_var()
	start = a.expr()
	end = a.expr()
	step = a.expr(optional=True, default=1)
	a.end()
	n = start
	result = []
	if step > 0:
		op = operator.le
		end += 1e-10
	else:
		op = operator.ge
		end -= 1e-10
	variable = var.variable
	with _scoped_var(a.env, variable):
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
	with _scoped_var(a.env, variable):
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
	with _scoped_var(a.env, variable):
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
	with _scoped_var(a.env, variable):
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


# ── Clock / date-time commands and functions ──────────────────────────────────

def set_date(a: ArgParser) -> None:
	a.env.set_date(a.expr(), a.expr(), a.expr())
	a.end()

def set_time(a: ArgParser) -> None:
	a.env.set_time(a.expr(), a.expr(), a.expr())
	a.end()

def check_tmr(a: ArgParser):
	start = a.expr()
	a.end()
	return a.env.check_tmr(start)

def set_dt_fmt(a: ArgParser) -> None:
	a.env.set_dt_fmt(a.expr())
	a.end()

def set_tm_fmt(a: ArgParser) -> None:
	a.env.set_tm_fmt(a.expr())
	a.end()

def get_dt_str(a: ArgParser):
	fmt = a.expr()
	a.end()
	return a.env.get_dt_str(fmt)

def get_tm_str(a: ArgParser):
	fmt = a.expr()
	a.end()
	return a.env.get_tm_str(fmt)
