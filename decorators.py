from __future__ import annotations
from functools import partial, wraps, update_wrapper
from itertools import repeat
from typing import TYPE_CHECKING

from tiobjects import TiList
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


def _call_vectorized(func, args):
	"""Core vectorize logic: broadcast func(*args) element-wise over any TiList args."""
	len_check = set()
	vec = []
	for a in args:
		if isinstance(a, TiList):
			len_check.add(len(a))
			vec.append(a)
		else:
			vec.append(repeat(a))
	if not len_check:
		return func(*args)
	if len(len_check) == 1:
		return TiList([func(*v) for v in zip(*vec)])
	raise DimMismatchError(f"Dim mismatch: {len_check}")


def vectorized(func):
	"""Standalone decorator: broadcast f(*args) element-wise over TiList arguments."""
	@wraps(func)
	def apply(*args):
		return _call_vectorized(func, args)
	return apply


def _vectorized_env(func):
	"""Wraps func(env, *args) so that *args are vectorized while env is held fixed."""
	@wraps(func)
	def apply(env, *args):
		return _call_vectorized(partial(func, env), args)
	return apply


class TiFunction:
	"""Wraps a calculator function with a plain-value __call__ and a parser interface."""

	def __init__(self, func, env=False):
		self._func = func
		self._env = env
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)

	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		return self(a.env, *args) if self._env else self(*args)


class FormsFunction:
	"""Wraps a custom-parsing function with call_with_parser passing ArgParser directly."""

	def __init__(self, func):
		self._func = func
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)

	def call_with_parser(self, a: ArgParser):
		return self._func(a)


class NoParen:
	"""Wraps a no-paren command: passes a CommandArgParser to the body.

	CommandArgParser does not eat a closing ) after arguments and checks
	COLON/NEWLINE/EOF (not )) for optional-arg termination.  End-of-statement
	validation is left to Parser.run(), which already calls expect(EOF_TOKEN).
	"""

	def __init__(self, func):
		self._func = func
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)

	def call_with_parser(self, a: ArgParser):
		from parser import CommandArgParser   # lazy: avoids circular import
		return self._func(CommandArgParser(a._parser))


def forms_func(func):
	"""Forms function: receives ArgParser directly for custom parsing."""
	return FormsFunction(func)


def no_paren_func(func):
	"""No-paren command: receives CommandArgParser; end-of-statement is validated automatically."""
	return NoParen(func)


def pure_func(func):
	"""TiFunction: parse args, call func(*args)."""
	return TiFunction(func)


def pure_vectorized(func):
	"""TiFunction: parse args, vectorize, call func(*args)."""
	return TiFunction(vectorized(func))


def env_func(func):
	"""TiFunction: parse args, call func(env, *args)."""
	return TiFunction(func, env=True)


def env_vectorized(func):
	"""TiFunction: parse args, vectorize, call func(env, *args)."""
	return TiFunction(_vectorized_env(func), env=True)
