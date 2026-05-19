from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Parser, ParseError
	from tokens import Token

from contextlib import contextmanager
from purefunctions import TiList, TiMatrix


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
	def list_var(self):
		return self._parser.parse_list_var_key()

	@_parse_method
	def matrix_var(self):
		return self._parser.parse_matrix_var_key()

	def finish(self):
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
		self.finish()
		return args

	def next_is_matrix_var(self) -> bool:
		pos = self._parser.pos + 1
		return pos < len(self._parser.tokens) and self._parser.tokens[pos].is_matrix_var()


def seq(a: ArgParser) -> TiList:
	thunk = a.thunk()
	var = a.real_var()
	start = a.expr()
	end = a.expr()
	step = a.expr(optional=True, default=1)
	a.finish()
	env = a.env
	result = []
	with _scoped_var(env, var) as reals:
		n = start
		while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
			reals[var] = n
			result.append(thunk.eval())
			n += step
	return TiList(result)


def sigma(a: ArgParser) -> float:
	thunk = a.thunk()
	var = a.real_var()
	start = int(a.expr())
	end = int(a.expr())
	a.finish()
	env = a.env
	total = 0
	with _scoped_var(env, var) as reals:
		for i in range(start, end + 1):
			reals[var] = float(i)
			total += thunk.eval()
	return total


def nderiv(a: ArgParser) -> float:
	thunk = a.thunk()
	var = a.real_var()
	val = a.expr()
	h = a.expr(optional=True, default=1e-5)
	a.finish()
	env = a.env
	with _scoped_var(env, var) as reals:
		reals[var] = val + h
		fwd = thunk.eval()
		reals[var] = val - h
		bwd = thunk.eval()
	return (fwd - bwd) / (2 * h)


def fnint(a: ArgParser) -> float:
	thunk = a.thunk()
	var = a.real_var()
	lo = a.expr()
	hi = a.expr()
	a.finish()
	env = a.env
	n = 1000
	h = (hi - lo) / n
	with _scoped_var(env, var) as reals:
		def f(x):
			reals[var] = x
			return thunk.eval()
		total = f(lo) + f(hi)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(lo + i * h)
	return total * h / 3


def matr_to_list(a: ArgParser) -> None:
	mat = a.expr()
	if not isinstance(mat, TiMatrix):
		raise ValueError("Matr►list: first argument must be a matrix")
	keys = [a.list_var()]
	while a.has_next_arg():
		keys.append(a.list_var())
	a.finish()
	for col, ref in enumerate(keys):
		ref.set(TiList([mat.inner[r][col] for r in range(mat.rows)]))


def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(a.expr())
		if not a.has_next_arg():
			raise ValueError("List►matr: expected matrix variable as last argument")
		if a.next_is_matrix_var():
			mat_key = a.matrix_var()
			break
	a.finish()
	cols = len(list_vals)
	rows = max(len(lst) for lst in list_vals)
	a.env.matrices[mat_key] = TiMatrix([
		[list_vals[c].inner[r] if r < len(list_vals[c]) else 0.0 for c in range(cols)]
		for r in range(rows)
	])
