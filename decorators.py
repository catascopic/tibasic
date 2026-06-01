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
		a.end_func()
		return self(a.env, *args) if self._env else self(*args)


class cmd_env_func(TiFunction):
	"""Decorator for env-method paren commands (setDate, setTime, etc.).

	Like TiFunction(env=True) but calls end_paren_cmd() instead of end_func(),
	so the trailing statement separator is eaten after the closing ).
	"""

	def __init__(self, func):
		super().__init__(func, env=True)

	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		result = self(a.env, *args)
		a.end_paren_cmd()
		return result


class FormsFunction:
	"""Wraps a custom-parsing function with call_with_parser passing ArgParser directly."""

	def __init__(self, func):
		self._func = func
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)

	def call_with_parser(self, a: ArgParser):
		return self._func(a)


def forms_func(func):
	"""Receives ArgParser directly for custom parsing."""
	return FormsFunction(func)


def no_paren_func(func):
	"""No-paren command: receives ArgParser directly; must call a.end_cmd() to close."""
	return FormsFunction(func)


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


def cmd_env_func(func):
	"""For env-method commands: parse args, call func(env, *args), eat separator."""
	return TiCmdFunc(func, env=True)
