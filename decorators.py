from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from functools import partial, wraps, update_wrapper
from itertools import repeat
from numbers import Number
from typing import Any, TYPE_CHECKING

from tiobjects import TiList, TiMatrix, require_num, require_real, require_int
from errors import DimMismatchError
from argspec import _as_spec, schema_from_signature

if TYPE_CHECKING:
	from parser import ArgParser


# ── Per-parameter value validation ───────────────────────────────────────────
# Value validation is a schema concern, not the parser's: ArgParser only extracts
# values, and @preparse wraps the core so each argument is checked by its slot's
# validator just before the call.  Because this wrapper sits *inside* the
# vectorization wrapper, a vectorized slot's validator sees one scalar element at
# a time — exactly where "is this element real/integer?" belongs.

_VALIDATORS: dict[str, Callable] = {
	'numeric': require_num,
	'real': require_real,
	'integer': require_int,
}


def _make_validated(core: Callable, schema: tuple) -> Callable:
	"""Wrap *core* so each positional argument is passed through its slot's value
	validator first.  Returns *core* unchanged when no slot declares a validator.

	Validators run on scalars.  For a variadic slot (which holds a list of values
	in one slot) the validator is mapped over the elements."""
	slots = tuple((_VALIDATORS.get(s.validate), s.variadic) for s in schema)
	if not any(v for v, _ in slots):
		return core

	@wraps(core)
	def apply(*args: Any) -> Any:
		checked = []
		for (validate, variadic), a in zip(slots, args):
			if validate is None:
				checked.append(a)
			elif variadic:
				checked.append([validate(x) for x in a])
			else:
				checked.append(validate(a))
		checked.extend(args[len(slots):])
		return core(*checked)
	return apply


# ── Per-parameter vectorization (new) ────────────────────────────────────────
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


# ── Whole-function vectorization (legacy) ────────────────────────────────────
# Used by preparse_vectorized / pure_vectorized / env_vectorized: maps over every
# TiList/Number argument uniformly, threading env through untouched.

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

	The schema is a tuple of ArgSpec values, one per core parameter.  It may be
	passed explicitly (legacy positional form) or, when `schema` is None, read
	from the core's type annotations via `schema_from_signature`.  An `env` spec
	injects ArgParser.env in that slot without consuming a token; every other
	spec parses from the token stream via the named parse method.

	`end` is a Finalize member controlling which ArgParser end method is called
	after parsing (FUNC end_func, CMD end_cmd, CMD_FUNC end_paren_cmd, NONE none).

	Vectorization is per-parameter: any spec flagged `vectorize` maps over a
	TiList in its slot, and a `matrix` spec also maps over a TiMatrix.  (The
	legacy `vectorize=True` flag, set by preparse_vectorized, instead maps the
	whole function over every TiList/Number argument, threading env through.)

	The core stays a plain function, so functions remain callable from other
	Python code (composability) via TiCall.__call__.
	"""
	def __init__(
		self,
		core: Callable,
		schema: tuple | None = None,
		vectorize: bool = False,
		end: Finalize = Finalize.FUNC,
	) -> None:
		if schema is None:
			schema = schema_from_signature(core)
		else:
			schema = tuple(_as_spec(s) for s in schema)

		# Validation wraps the core innermost; vectorization (if any) wraps that,
		# so a vectorized slot's validator runs per element.
		validated = _make_validated(core, schema)

		if vectorize:
			# Legacy whole-function vectorization (preparse_vectorized).
			has_env = any(s.method == 'env' for s in schema)
			func = _vectorized_with_env(validated) if has_env else vectorized(validated)
		else:
			vec = frozenset(i for i, s in enumerate(schema) if s.vectorize)
			mat = frozenset(i for i, s in enumerate(schema) if s.matrix)
			func = _make_vectorized(validated, vec, mat) if vec else validated

		super().__init__(func)
		self.schema = schema
		self.end_method = _END_METHOD[end]

	def call_with_parser(self, a: ArgParser):
		args = a.take(*self.schema)
		if self.end_method is not None:
			getattr(a, self.end_method)()
		return self.func(*args)


def preparse(*args, end: Finalize | None = None):
	"""Declarative decorator for functions/commands called once per invocation.

	The schema comes from the core's parameter annotations (see argspec.py):

	    @preparse(FUNC)
	    def gcd(a: vectorized[numeric], b: vectorized[numeric]) -> float: ...

	    @preparse(CMD_FUNC)
	    def pxl_on(env: PassEnv, row: expr, col: expr) -> None: ...

	The Finalize mode may be given positionally (as above) or via `end=`.  It
	selects the ArgParser end method (default FUNC):
	  FUNC      — end_func()       expression functions; does not eat separator
	  CMD       — end_cmd()        no-paren commands; eats trailing separator
	  CMD_FUNC  — end_paren_cmd()  paren commands;    eats ) + separator
	  NONE      — (nothing)        leaves the parser untouched (e.g. DelVar)

	A legacy positional schema is still accepted; in that case the schema is
	taken from the arguments rather than the annotations:

	    @preparse(expr, optional(expr))
	    def round(x, n=9): ...
	"""
	schema = []
	for a in args:
		if isinstance(a, Finalize):
			if end is not None and end is not a:
				raise TypeError("preparse: end mode given both positionally and via end=")
			end = a
		else:
			schema.append(a)
	final_end = Finalize.FUNC if end is None else end

	def decorator(core: Callable) -> PreparsedFunc:
		return PreparsedFunc(core, tuple(schema) or None, end=final_end)
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
