from __future__ import annotations
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Parser, ParseError
	from tokens import Token

from contextlib import contextmanager
from tiobjects import TiList, TiMatrix
from environment import Variable


@contextmanager
def _scoped_var(env, var):
	saved = env.reals[var]
	try:
		yield env.reals
	finally:
		env.reals[var] = saved


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
		if not tok.is_real_var():
			raise ValueError(f"Expected a real variable, got {tok.text!r}")
		return tok

	@_parse_method
	def list_var(self) -> Variable:
		return self._parser.parse_list_var_key()

	@_parse_method
	def matrix_var(self) -> Variable:
		return self._parser.parse_matrix_var_key()

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
	with _scoped_var(a.env, var) as reals:
		while op(n, end):
			reals[var] = n
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
	with _scoped_var(a.env, var) as reals:
		while n <= end:
			reals[var] = n
			total += formula.eval()
			n += 1
	return total


def nderiv(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.real_var()
	val = a.expr()
	h = a.expr(optional=True, default=0.001)
	a.end()
	with _scoped_var(a.env, var) as reals:
		reals[var] = val + h
		fwd = formula.eval()
		reals[var] = val - h
		bwd = formula.eval()
	return (fwd - bwd) / (2 * h)


def fnint(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.real_var()
	lo = a.expr()
	hi = a.expr()
	tol = a.expr(optional=True, default=1e-5)
	# TODO: use tol
	a.end()
	n = 1000
	h = (hi - lo) / n
	with _scoped_var(a.env, var) as reals:
		def f(x):
			reals[var] = x
			return formula.eval()
		total = f(lo) + f(hi)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(lo + i * h)
	return total * h / 3


def matr_to_list(a: ArgParser) -> None:
	mat = a.expr()
	if not isinstance(mat, TiMatrix):
		raise ValueError("Matr►list: first argument must be a matrix")
	# TODO: second argument can also be a number, in which case we only get that column
	list_refs = [a.list_var()]
	while a.has_next_arg():
		list_refs.append(a.list_var())
	a.end()
	for col, ref in enumerate(list_refs):
		ref.set(a.env, TiList([mat.inner[r][col] for r in range(mat.rows)]))


def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(a.expr())
		if not a.has_next_arg():
			raise ValueError("List►matr: expected matrix variable as last argument")
		# TODO:
		# if a.peek().is_matrix_var():
		if a.next_is_matrix_var():
			mat_var = a.matrix_var()
			break
	a.end()
	cols = len(list_vals)
	rows = max(len(lst) for lst in list_vals)
	# TODO: can we use zip with a default value?
	mat_var.set(a.env, TiMatrix([
		[list_vals[c].inner[r] if r < len(list_vals[c]) else 0.0 for c in range(cols)]
		for r in range(rows)
	]))
