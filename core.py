"""Low-level runtime types and scalar validators shared across the interpreter.

Deliberately kept free of all project-internal imports so that nothing in the
dependency graph can create a cycle back to this file.

Contents:
  - Scalar guard functions (require_num, require_real, require_int, py_int).
  - The Variable hierarchy — abstract base and numeric/real subclasses only.
    Collection-typed subclasses (ListVariable, MatrixVariable, StringVariable,
    EquationVariable, UserList) live in their respective domain modules.
  - Thunk — a captured, lazily-evaluated token slice.

Token lives in titoken.py and control-flow signals in signals.py; both are
already dependency-free, so they stay where they are.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Number
from typing import Any, TypeVar, TYPE_CHECKING

from errors import DataTypeError, DomainError, UndefinedError
from titoken import Token

if TYPE_CHECKING:
	from environment import Environment


# ── Scalar validators ────────────────────────────────────────────────────────

_T = TypeVar('_T')

def _require_type(value: Any, tp: type[_T], exc_cls=DataTypeError) -> _T:
	if not isinstance(value, tp):
		raise exc_cls(f"Invalid value: {value!r}; required: {tp.__name__}")
	return value

def require_num(value: Any, exc_cls=DataTypeError) -> Number:
	return _require_type(value, Number, exc_cls)

def require_real(value: Any, exc_cls=DataTypeError) -> float:
	require_num(value, exc_cls)
	if isinstance(value, complex):
		raise exc_cls(f"Expected real number, got complex: {value}")
	return value

def require_int(value: Any, exc_cls=DomainError) -> float:
	require_real(value, exc_cls)
	if not value.is_integer():
		raise exc_cls(f"Expected integer, got {value}")
	return value

def py_int(value: Any, exc_cls=DomainError) -> int:
	"""Validate that value is a whole number, then return it as a Python int.
	Use when passing a TI value to a Python API that requires int (range, math.comb, etc.).
	For TI-level validation only, use require_int."""
	return int(require_int(value, exc_cls))

def repr_num(value: Number) -> str:
	return repr(int(value) if not isinstance(value, complex) and value.is_integer() else value)


# ── Variable hierarchy ───────────────────────────────────────────────────────

class Variable(ABC):
	"""Abstract base for all TI-BASIC variable types.

	Collection-typed subclasses (ListVariable, MatrixVariable, StringVariable,
	EquationVariable, UserList) live in their respective domain modules.
	"""
	value = None  # class-level sentinel; instance writes shadow it

	def resolve(self) -> Any:
		if self.value is None:
			raise UndefinedError(f"Undefined {type(self).__name__}")
		return self.value

	def store(self, new_value) -> None:
		self.value = self.normalize(new_value)

	@abstractmethod
	def normalize(self, value) -> Any:
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
			raise ValueError(f"Expected end of Thunk; remaining: {parser.tokens[parser.pos:]}")
		return value
