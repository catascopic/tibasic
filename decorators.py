from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from functools import wraps, update_wrapper
from itertools import repeat
from numbers import Number
from typing import Any, TYPE_CHECKING

from tiobjects import TiList, TiMatrix
from errors import DimMismatchError
from argspec import schema_from_signature

if TYPE_CHECKING:
	from parser import ArgParser


# ── Per-parameter vectorization ──────────────────────────────────────────────
# Vectorization is driven by per-spec flags: a `vectorized[...]` parameter maps
# over a TiList in its slot; a `matrix_vectorized[...]` parameter also maps over
# a TiMatrix.  `vec` / `mat` are the sets of such positions in the arg tuple.

def map_vectorized(func: Callable, args: tuple, vec: frozenset, mat: frozenset) -> Any:
	# A single matrix in a matrix-capable slot is mapped via transform; every
	# other argument (including the other matrix-slot scalars) is held constant.
	for i in mat:
		if i < len(args) and isinstance(args[i], TiMatrix):
			return args[i].transform(
				lambda x, j=i: func(*args[:j], x, *args[j + 1:])
			)
	# Otherwise map over TiLists in the vectorized slots, broadcasting the rest.
	lengths: set[int] = set()
	cols = []
	for i, a in enumerate(args):
		if i in vec and isinstance(a, TiList):
			lengths.add(len(a))
			cols.append(a)
		else:
			cols.append(repeat(a))
	if not lengths:
		return func(*args)
	if len(lengths) != 1:
		raise DimMismatchError(f"Dim mismatch: {lengths}")
	return TiList([func(*row) for row in zip(*cols)])


def _make_vectorized(core: Callable, vec: frozenset, mat: frozenset) -> Callable:
	@wraps(core)
	def apply(*args: Any) -> Any:
		return map_vectorized(core, args, vec, mat)
	return apply


# ── Whole-function vectorization ─────────────────────────────────────────────
# The `vectorized` decorator (used by operators.py) maps a function over every
# TiList/Number argument uniformly, broadcasting scalars.

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
	"""
	FUNC = auto()
	CMD = auto()
	CMD_FUNC = auto()
	NONE = auto()


# Maps each Finalize mode to the ArgParser end method name (None = no call).
_END_METHOD = {
	Finalize.FUNC: 'end_func',
	Finalize.CMD: 'end_cmd',
	Finalize.CMD_FUNC: 'end_paren_cmd',
	Finalize.NONE: None,
}


class PreparsedFunc(TiCall):
	"""Wraps a plain core function with a declarative arg schema (see argspec.py).

	The schema is read from the core's type annotations via
	`schema_from_signature`: one ArgSpec per parameter.  An `env` spec injects
	ArgParser.env in that slot without consuming a token; every other spec parses
	from the token stream via the named parse method.

	`end` is a Finalize member controlling which ArgParser end method is called
	after parsing (FUNC end_func, CMD end_cmd, CMD_FUNC end_paren_cmd, NONE none).

	Vectorization is per-parameter: any spec flagged `vectorize` maps over a
	TiList in its slot, and a `matrix` spec also maps over a TiMatrix.

	The core stays a plain function, so functions remain callable from other
	Python code (composability) via TiCall.__call__.
	"""
	def __init__(
		self,
		core: Callable,
		end: Finalize = Finalize.FUNC,
	) -> None:
		schema = schema_from_signature(core)

		# Each parse method guards its argument's true type at the token boundary,
		# so the core needs no value-validation wrapper — vectorization (if any) is
		# the only wrapper, mapping a vectorized slot's list/matrix onto the scalar
		# core element-wise.
		#
		vec = frozenset(i for i, s in enumerate(schema) if s.vectorize)
		mat = frozenset(i for i, s in enumerate(schema) if s.matrix)
		func = _make_vectorized(core, vec, mat) if vec else core

		super().__init__(func)
		self.schema = schema
		self.end_method = _END_METHOD[end]

	def call_with_parser(self, a: ArgParser):
		args = a.take(*self.schema)
		if self.end_method is not None:
			getattr(a, self.end_method)()
		return a.env.guard_real(args, self.func(*args))


def preparse(end: Finalize | None = None):
	"""Return a decorator that wraps a core function as a PreparsedFunc.

	Prefer the named aliases below over calling this directly:
	  preparse_func     — expression functions  (end_func, eats `)`              )
	  preparse_cmd      — no-paren commands     (end_cmd, eats trailing separator)
	  preparse_cmd_func — paren commands        (end_paren_cmd, eats `) + sep`   )
	  preparse_bunch    — bunching commands     (no finalizer, e.g. DelVar       )

	Use this directly only when you need to select a Finalize mode dynamically.
	"""
	final_end = Finalize.FUNC if end is None else end

	def decorator(core: Callable) -> PreparsedFunc:
		return PreparsedFunc(core, end=final_end)
	return decorator


# Named decorator aliases — preferred over calling preparse() directly.
preparse_func     = preparse(Finalize.FUNC)
preparse_cmd      = preparse(Finalize.CMD)
preparse_cmd_func = preparse(Finalize.CMD_FUNC)
preparse_bunch    = preparse(Finalize.NONE)


class no_arg_command(TiCall):
	"""Decorator for no-arg commands that consume the statement separator.

	The decorated function receives only the environment.  The decorator itself
	calls no_args() (raises TiSyntaxError if anything follows) then end().
	Use for Normal, Float, Radian, etc.
	"""
	def call_with_parser(self, a: Any) -> None:
		a.no_args()
		a.end_cmd()
		self.func(a.env)


class no_arg_bunch(TiCall):
	"""Decorator for no-arg commands that do NOT consume the separator.

	Does not check for surplus tokens — because the next token may be the start
	of the following command (e.g. ClockOnClockOn).  Simply runs the function
	and returns, leaving the parser exactly where it is so the main loop
	picks up the next statement naturally.
	Use for ClockOn, ClockOff, etc.
	"""
	def call_with_parser(self, a: Any) -> None:
		self.func(a.env)
