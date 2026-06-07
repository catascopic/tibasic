from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from functools import partial, wraps, update_wrapper
from itertools import repeat
from numbers import Number
from typing import Any, TYPE_CHECKING

from tiobjects import TiList
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


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


def vectorized(func: Callable) -> Callable:
	@wraps(func)
	def apply(*args: Any) -> Any:
		return call_vectorized(func, args)
	return apply


def _vectorized_with_env(func: Callable) -> Callable:
	@wraps(func)
	def apply(env: Any, *args: Any) -> Any:
		return call_vectorized(partial(func, env), args)
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


class Finalize(Enum):
	"""Which ArgParser end method PreparsedFunc calls after parsing.

	  FUNC      — end_func()       expression functions; eats ), no separator
	  CMD       — end_cmd()        no-paren commands; eats trailing separator
	  CMD_FUNC  — end_paren_cmd()  paren commands; eats ) + separator
	  NONE      — (nothing)        leaves the parser untouched so the command
	                               bunches with whatever follows (e.g. DelVar)

	The members are re-exported as bare module-level names (FUNC, CMD, …) so
	call sites can write `@preparse(..., end=CMD)` after a plain
	`from decorators import CMD`.
	"""
	FUNC = auto()
	CMD = auto()
	CMD_FUNC = auto()
	NONE = auto()


# Bare aliases so callers can `from decorators import FUNC, CMD, CMD_FUNC, NONE`.
FUNC, CMD, CMD_FUNC, NONE = Finalize

# Maps each Finalize mode to the ArgParser end method name (None = no call).
_END_METHOD = {
	Finalize.FUNC: 'end_func',
	Finalize.CMD: 'end_cmd',
	Finalize.CMD_FUNC: 'end_paren_cmd',
	Finalize.NONE: None,
}


class PreparsedFunc(TiCall):
	"""Wraps a plain core function with a declarative arg schema (see argspec.py).

	The schema is a sequence of ArgSpec values, one per core parameter.  An
	`env` spec injects ArgParser.env in that slot without consuming a token;
	every other spec parses from the token stream via the named parse method.

	`end` is a Finalize member controlling which ArgParser end method is called
	after parsing (FUNC end_func, CMD end_cmd, CMD_FUNC end_paren_cmd, NONE none).

	If vectorize=True the core maps over TiList arguments.  When the schema
	carries an `env` slot, env is threaded through unchanged rather than being
	mapped over (via _vectorized_with_env).

	The core stays a plain function, so functions remain callable from other
	Python code (composability) via TiCall.__call__.
	"""
	def __init__(
		self,
		core: Callable,
		schema: tuple,
		vectorize: bool = False,
		end: Finalize = Finalize.FUNC,
	) -> None:
		if vectorize:
			has_env = any(s.method == 'env' for s in schema)
			func = _vectorized_with_env(core) if has_env else vectorized(core)
		else:
			func = core
		super().__init__(func)
		self.schema = schema
		self.end_method = _END_METHOD[end]

	def call_with_parser(self, a: ArgParser):
		args = a.take(*self.schema)
		if self.end_method is not None:
			getattr(a, self.end_method)()
		return self.func(*args)


def preparse(*schema, end: Finalize = Finalize.FUNC):
	"""Declarative-schema decorator for functions called once per invocation.

	`end` is a Finalize member selecting the ArgParser end method (default FUNC):
	  FUNC      — end_func()       expression functions; does not eat separator
	  CMD       — end_cmd()        no-paren commands; eats trailing separator
	  CMD_FUNC  — end_paren_cmd()  paren commands;    eats ) + separator
	  NONE      — (nothing)        leaves the parser untouched (e.g. DelVar)

	    @preparse(env, expr, expr, end=CMD_FUNC)
	    def pxl_on(env, row, col): ...

	    @preparse(expr, optional(expr))
	    def round(x, n=9): ...
	"""
	def decorator(core: Callable) -> PreparsedFunc:
		return PreparsedFunc(core, schema, end=end)
	return decorator


def preparse_vectorized(*schema):
	"""Like preparse, but maps over TiList arguments (env, if present, is
	threaded through rather than vectorized over).

	    @preparse_vectorized(expr)
	    def sinh(x): ...

	    @preparse_vectorized(env, expr)
	    def some_env_math(env, x): ...
	"""
	def decorator(core: Callable) -> PreparsedFunc:
		return PreparsedFunc(core, schema, vectorize=True)
	return decorator


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
		a.end_cmd()
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
