"""Low-level runtime types shared across the interpreter.

These types form a foundational layer that several heavyweight modules
(parser, program, environment, argspec/preparse) all need to reference — most
often as type hints.  Keeping them here, rather than inside parser.py or
environment.py, lets those modules import the *types* without importing each
other, avoiding circular-import knots.

Contents:
  - The Variable hierarchy (storable, typed calculator variables) plus UserList.
  - Thunk — a captured, lazily-evaluated token slice.

Token lives in titoken.py and the control-flow signals in signals.py; both are
already dependency-free, so they stay where they are.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from tiobjects import (
	TiString, TiEquation,
	require_num, require_real, require_list, require_matrix, require_str,
)
from errors import DataTypeError, InvalidDimError, UndefinedError
from titoken import Token

if TYPE_CHECKING:
	from environment import Environment


# ── Variable hierarchy ────────────────────────────────────────────────────────

class Variable(ABC):
	"""Abstract base for all TI-BASIC variable types.

	Concrete subclasses fall into two storage models:
	  - Instance-stored (NumericVariable and its descendants, ListVariable, etc.):
	    value lives in self.value; NumericVariable provides __init__(default=None)
	    which subclasses inherit.  Other instance-stored classes have no __init__
	    and therefore never accept a constructor default — intentional, since lists,
	    matrices, strings and equations are always initialised to undefined (None).
	  - Proxy-stored (UserList): value lives in env.user_lists[name]; the class
	    manages its own __init__ and exposes value as a property.  It inherits from
	    Variable without calling super().__init__() because there is no super
	    __init__ to call.
	"""
	value = None  # class-level sentinel; instance writes shadow it

	def resolve(self) -> Any:
		"""Called when the user references a variable."""
		if self.value is None:
			raise UndefinedError(f"Undefined {type(self).__name__}")
		return self.value

	def store(self, new_value) -> None:
		"""Called when the user stores a variable."""
		self.value = self.normalize(new_value)

	@abstractmethod
	def normalize(self, value) -> Any:
		"""Validates and coerces a value before storage. For structured types, returns a defensive copy."""
		pass

	def __repr__(self):
		return f"{type(self).__name__}({self.value})"


class NumericVariable(Variable):
	"""Numeric (real or complex) variable.  Accepts an optional constructor default
	which RealVariable and other subclasses inherit — the only Variable subclasses
	that are ever constructed with a non-None starting value."""

	def __init__(self, default=None):
		self.value = default

	def resolve(self):
		if self.value is None:
			self.value = 0
		return self.value

	def normalize(self, value):
		return require_num(value)

	@contextmanager
	def scoped(self):
		saved = self.resolve()
		try:
			yield
		finally:
			self.value = saved


class RealVariable(NumericVariable):
	def normalize(self, value):
		return require_real(value)

class ListVariable(Variable):
	def resolve(self):
		lst = super().resolve()
		if not lst.data:
			raise InvalidDimError("empty list")
		return lst

	def normalize(self, value):
		return require_list(value).copy()

	def store(self, new_value) -> None:
		was_complex = self.value is not None and self.value.is_complex
		self.value = self.normalize(new_value)
		if was_complex:
			self.value._upgrade_to_complex()

class MatrixVariable(Variable):
	def normalize(self, value):
		return require_matrix(value).copy()

class StringVariable(Variable):
	def normalize(self, value):
		return require_str(value)

class EquationVariable(Variable):
	def normalize(self, value):
		if isinstance(value, TiEquation):
			return value
		if isinstance(value, TiString):
			return TiEquation(value.tokens)
		raise DataTypeError(f"Expected equation or string; got {value}")


class UserList(ListVariable):

	def __init__(self, env: Environment, name: str):
		self.lookup = env.user_lists
		self.name = name
		# Don't call super().__init__() — value is managed via the property below.

	@property
	def value(self):
		return self.lookup.get(self.name)

	@value.setter
	def value(self, new_value):
		if new_value is None:
			self.lookup.pop(self.name, None)
		else:
			self.lookup[self.name] = new_value

	def resolve(self) -> Any:
		try:
			lst = self.lookup[self.name]
		except KeyError:
			raise UndefinedError(f"User list {self.name!r} is not defined")
		if not lst.data:
			raise InvalidDimError("empty list")
		return lst


# ── Thunk ─────────────────────────────────────────────────────────────────────

@dataclass
class Thunk:
	"""A captured slice of tokens plus the environment to evaluate them in.

	`eval()` re-parses the tokens on demand, which is how deferred-evaluation
	commands (While/Repeat conditions) re-test their expression each iteration.
	Parser is imported lazily inside eval() to avoid a core ⇄ parser import cycle.
	"""
	tokens: list[Token]
	env: Environment

	def eval(self):
		from parser import Parser
		parser = Parser(self.tokens, self.env)
		value = parser.parse_expr()
		if parser.has_next:
			# This is a ValueError, not a TiError, because the parser should always get this right
			raise ValueError(f"Expected end of Thunk; remaining: {parser.tokens[parser.pos:]}")
		return value
