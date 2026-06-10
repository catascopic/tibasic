from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps, update_wrapper
from itertools import repeat
from numbers import Number
from typing import Any, TYPE_CHECKING

from tiobjects import TiList, TiMatrix
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


# ── Whole-function vectorization ─────────────────────────────────────────────
# `vectorize` (used by operators.py and @preparse) maps a function over every
# TiList/Number argument uniformly, broadcasting scalars.  `matrix_vectorize`
# additionally maps over a single TiMatrix argument via its transform.

def call_vectorized(func: Callable, args: tuple) -> Any:
	len_check = set()
	vec = []
	for a in args:
		if isinstance(a, TiList):
			len_check.add(len(a))
			vec.append(a)
		elif isinstance(a, Number):
			vec.append(repeat(a))
		else:
			return func(*args)
	if not len_check:
		return func(*args)
	if len(len_check) == 1:
		return TiList([func(*v) for v in zip(*vec)])
	raise DimMismatchError(f"Dim mismatch: {len_check}")


def vectorize(func: Callable) -> Callable:
	@wraps(func)
	def apply(*args: Any) -> Any:
		return call_vectorized(func, args)
	return apply


def matrix_vectorize(func: Callable) -> Callable:
	"""Like vectorize, but a single TiMatrix argument is mapped element-wise too.

	A matrix in any slot is mapped via its transform, holding the other arguments
	constant; the recursion then handles those scalar elements (and any TiList
	arguments) through call_vectorized.  @preparse guarantees at most one
	matrix-capable parameter, so scanning for the first TiMatrix is unambiguous.
	"""
	@wraps(func)
	def apply(*args: Any) -> Any:
		for i, a in enumerate(args):
			if isinstance(a, TiMatrix):
				return a.transform(lambda x, j=i: apply(*args[:j], x, *args[j + 1:]))
		return call_vectorized(func, args)
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
	def call_with_parser(self, args: ArgParser):
		values = args.parse_args()
		args.end_func()
		return self(*values)


class env_func(TiCall):
	"""Decorator for functions that need access to the environment."""
	def call_with_parser(self, args: ArgParser):
		values = args.parse_args()
		args.end_func()
		return self(args.env, *values)


class forms_func(TiCall):
	"""Decorator for functions/commands that do their own parsing."""
	def call_with_parser(self, args: ArgParser):
		return self.func(args)


class no_arg_command(TiCall):
	"""Decorator for no-arg commands that consume the statement separator.

	The decorated function receives only the environment.  The decorator itself
	calls no_args() (raises TiSyntaxError if anything follows) then end().
	Use for Normal, Float, Radian, etc.
	"""
	def call_with_parser(self, args: Any) -> None:
		args.no_args()
		args.end_cmd()
		self.func(args.env)


class no_arg_bunch(TiCall):
	"""Decorator for no-arg commands that do NOT consume the separator.

	Does not check for surplus tokens — because the next token may be the start
	of the following command (e.g. ClockOnClockOn).  Simply runs the function
	and returns, leaving the parser exactly where it is so the main loop
	picks up the next statement naturally.
	Use for ClockOn, ClockOff, etc.
	"""
	def call_with_parser(self, args: Any) -> None:
		self.func(args.env)
