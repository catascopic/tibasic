from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial, wraps, update_wrapper
from itertools import repeat
from typing import Any, TYPE_CHECKING

from tiobjects import TiList
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


def _call_vectorized(func: Callable, args: tuple) -> Any:
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


def vectorized(func: Callable) -> Callable:
	@wraps(func)
	def apply(*args: Any) -> Any:
		return _call_vectorized(func, args)
	return apply


def _vectorized_with_env(func: Callable) -> Callable:
	@wraps(func)
	def apply(env: Any, *args: Any) -> Any:
		return _call_vectorized(partial(func, env), args)
	return apply


class TiCall(ABC):

	def __init__(self, func: Callable) -> None:
		self.func = func
		update_wrapper(self, func)

	def __call__(self, *args: Any) -> Any:
		return self.func(*args)

	@abstractmethod
	def call_with_parser(self, a: ArgParser) -> Any:
		pass


class pure_func(TiCall):
	"""Decorator for pure math functions that don't need access to the environment."""
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_func()
		return self(*args)


class env_func(TiCall):
	"""Decorator for functions that need access to the environment."""
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_func()
		return self(a.env, *args)


class forms_func(TiCall):
	"""Decorator for functions/commands that do their own parsing."""
	def call_with_parser(self, a: ArgParser):
		return self.func(a)


def pure_vectorized(func):
	"""Same as pure_func, but also vectorized."""
	return pure_func(vectorized(func))


def env_vectorized(func):
	"""Same as env_func, but also vectorized."""
	return env_func(_vectorized_with_env(func))


class nullary_command(TiCall):
	"""Decorator for no-arg commands that consume the statement separator.

	The decorated function receives only the environment.  The decorator itself
	calls no_args() (raises TiSyntaxError if anything follows) then end().
	Use for Normal, Float, Radian, etc.
	"""
	def call_with_parser(self, a: Any) -> None:
		a.no_args()
		a.end()
		self.func(a.env)


class nullary_bunch(TiCall):
	"""Decorator for no-arg commands that do NOT consume the separator.

	Does not check for surplus tokens — because the next token may be the start
	of the following command (e.g. ClockOnClockOn).  Simply runs the function
	and returns, leaving the parser exactly where it is so the main loop
	picks up the next statement naturally.
	Use for ClockOn, ClockOff, etc.
	"""
	def call_with_parser(self, a: Any) -> None:
		self.func(a.env)
