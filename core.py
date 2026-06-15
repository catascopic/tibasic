"""Runtime value types, validators, and the Variable hierarchy.

Imports only from errors and titoken so nothing in the graph can cycle back here.
"""
from __future__ import annotations

import builtins
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import repeat
from numbers import Number
from typing import Any, TypeVar, TYPE_CHECKING

from errors import (
	DataTypeError, DimMismatchError, DomainError,
	InvalidDimError, SingularMatrixError, TiMemoryError, UndefinedError,
)
from titoken import Token

if TYPE_CHECKING:
	from environment import Environment


# ── Scalar validators ─────────────────────────────────────────────────────────

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

def _get_dim(value: Any) -> int:
	return py_int(value, InvalidDimError)


# ── TiList ────────────────────────────────────────────────────────────────────

class TiList:
	__slots__ = ('data', 'is_complex')

	def __init__(self, data: list[Number] | None = None) -> None:
		self.data = [] if data is None else data
		if not all(isinstance(e, Number) for e in self.data):
			raise ValueError(self.data)
		self.is_complex = False
		if any(isinstance(e, complex) for e in self.data):
			self._upgrade_to_complex()

	@classmethod
	def alloc(cls, size: Number) -> TiList:
		return cls(list(repeat(0.0, _get_dim(size))))

	def _check_index(self, index):
		if not (1 <= index <= len(self)):
			raise InvalidDimError(f"out of bounds: {index}; dim: {len(self)}")
		return index

	def __getitem__(self, index: int) -> Number:
		return self.data[self._check_index(index) - 1]

	def __setitem__(self, index: int, value: Number) -> None:
		if isinstance(value, complex):
			self._upgrade_to_complex()
		elif self.is_complex:
			value = complex(value)

		if index == len(self) + 1:
			self.data.append(value)
		else:
			self.data[self._check_index(index) - 1] = value

	def __len__(self) -> int:
		return len(self.data)

	def __iter__(self) -> Iterator:
		return iter(self.data)

	def __neg__(self) -> TiList:
		return TiList([-e for e in self.data])

	def set_dim(self, value: Any) -> None:
		value = _get_dim(value)
		dim = len(self)
		if value < dim:
			del self.data[value:]
		elif value > dim:
			fill_value = 0+0j if self.is_complex else 0.0
			self.data.extend(repeat(fill_value, value - dim))

	def _upgrade_to_complex(self) -> None:
		"""Promote every element to complex and set the is_complex flag. No-op if already complex."""
		if not self.is_complex:
			self.is_complex = True
			self.data[:] = [complex(e) for e in self.data]

	def clear(self):
		self.data.clear()
		self.is_complex = False

	def copy(self) -> TiList:
		result = TiList()
		result.data = self.data.copy()
		result.is_complex = self.is_complex
		return result

	def __repr__(self) -> str:
		return f"{{{','.join(repr_num(e) for e in self.data)}}}"


def _vectorize_op(op: Callable) -> Callable:
	@wraps(op)
	def list_op(self: TiList, other: Any) -> TiList:
		if isinstance(other, TiList):
			if len(self) != len(other):
				raise DimMismatchError(f"Dim mismatch: {len(self)} vs {len(other)}")
			return TiList([op(a, b) for a, b in zip(self, other)])
		if isinstance(other, Number):
			return TiList([op(a, other) for a in self.data])
		return NotImplemented
	return list_op


for _name, _op in [
	('__add__', operator.add),
	('__radd__', operator.add),
	('__sub__', operator.sub),
	('__rsub__', lambda a, b: b - a),
	('__mul__', operator.mul),
	('__rmul__', operator.mul),
	('__truediv__', operator.truediv),
	('__rtruediv__', lambda a, b: b / a),
	('__pow__', pow),
	('__rpow__', lambda a, b: b ** a),
]:
	setattr(TiList, _name, _vectorize_op(_op))


# ── TiMatrix ──────────────────────────────────────────────────────────────────

def _check_valid_dim(value: Any) -> tuple[int, int]:
	require_list(value)
	if len(value) != 2:
		raise InvalidDimError(f"Matrix dimensions must be 2 elements, but got {value}")
	rows, cols = value
	rows = _get_dim(rows)
	cols = _get_dim(cols)
	if not (1 <= rows <= 99):
		raise InvalidDimError(f"Required: 1 <= rows <= 99; got {rows}")
	if not (1 <= cols <= 99):
		raise InvalidDimError(f"Required: 1 <= cols <= 99; got {cols}")
	return rows, cols


class TiMatrix:
	__slots__ = ('data',)

	def __init__(self, data: list[list[float]] | None = None) -> None:
		self.data = [] if data is None else data
		for row in self.data:
			if not all(isinstance(i, float) for i in row):
				raise ValueError(self.data)

	@classmethod
	def alloc(cls, dim_list: TiList) -> TiMatrix:
		rows, cols = _check_valid_dim(dim_list)
		return cls([list(repeat(0.0, cols)) for r in range(rows)])

	@property
	def rows(self) -> int:
		return len(self.data)

	@property
	def cols(self) -> int:
		return len(self.data[0]) if self.data else 0

	def _check_index(self, index: Any) -> tuple[int, int]:
		row_index, col_index = index
		if not (1 <= row_index <= self.rows):
			raise InvalidDimError(f"{row_index=}")
		if not (1 <= col_index <= self.cols):
			raise InvalidDimError(f"{col_index=}")
		return row_index, col_index

	def __getitem__(self, index: Any) -> float:
		row_index, col_index = self._check_index(index)
		return self.data[row_index - 1][col_index - 1]

	def __setitem__(self, index: Any, value: Any) -> None:
		row_index, col_index = self._check_index(index)
		require_real(value)
		self.data[row_index - 1][col_index - 1] = value

	def set_dim(self, dim_list: TiList) -> None:
		new_rows, new_cols = _check_valid_dim(dim_list)
		self.data = [[
			self.data[r][c] if r < self.rows and c < self.cols else 0.0
			for c in range(new_cols)
		] for r in range(new_rows)]

	def get_row(self, r: Any) -> list:
		n = _get_dim(r)
		if not (1 <= n <= self.rows):
			raise InvalidDimError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		return self.data[n - 1]

	def set_row(self, r: Any, row: list) -> None:
		n = _get_dim(r)
		if not (1 <= n <= self.rows):
			raise InvalidDimError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		self.data[n - 1] = row

	def transform(self, func: Callable) -> TiMatrix:
		return TiMatrix([[func(x) for x in row] for row in self.data])

	def transform_zip(self, other: TiMatrix, func: Callable) -> TiMatrix:
		if self.rows != other.rows or self.cols != other.cols:
			raise DimMismatchError(
				"Operation not allowed on matrices with different sizes: "
				f"{self.rows}×{self.cols} vs. {other.rows}×{other.cols}"
			)
		return TiMatrix([
			[func(a, b) for a, b in zip(row_a, row_b)]
			for row_a, row_b in zip(self.data, other.data)
		])

	def __add__(self, other: Any) -> TiMatrix:
		if isinstance(other, TiMatrix):
			return self.transform_zip(other, operator.add)
		return NotImplemented

	def __sub__(self, other: Any) -> TiMatrix:
		if isinstance(other, TiMatrix):
			return self.transform_zip(other, operator.sub)
		return NotImplemented

	def __matmul__(self, other: TiMatrix) -> TiMatrix:
		if self.cols != other.rows:
			raise DimMismatchError(f"Dimension mismatch: ({self.rows}×{self.cols}) @ ({other.rows}×{other.cols})")
		other_cols = list(zip(*other.data))
		return TiMatrix([
			[builtins.sum(a * b for a, b in zip(row, col)) for col in other_cols]
			for row in self.data
		])

	def __mul__(self, other: Any) -> TiMatrix:
		if isinstance(other, float):
			return self.transform(lambda x: x * other)
		if isinstance(other, TiMatrix):
			return self @ other
		return NotImplemented

	def __rmul__(self, other: Any) -> TiMatrix:
		if isinstance(other, float):
			return self.transform(lambda x: other * x)
		return NotImplemented

	def __pow__(self, n: Any) -> TiMatrix:
		if not isinstance(n, float):
			return NotImplemented
		require_int(n)
		if self.rows != self.cols:
			raise DomainError(f"Matrix power requires a square matrix, got {self.rows}×{self.cols}")
		if n < 0:
			raise DomainError("Negative matrix power not supported")
		bits = int(n)
		size = self.rows
		result = TiMatrix([[float(r == c) for c in range(size)] for r in range(size)])
		base = self
		while bits > 0:
			if bits & 1:
				result = result @ base
			base = base @ base
			bits >>= 1
		return result

	def __eq__(self, other: object) -> bool:
		if isinstance(other, TiMatrix):
			return self.data == other.data
		raise DataTypeError(f"Cannot compare matrix with {type(other).__name__}")

	def __neg__(self) -> TiMatrix:
		return self.transform(operator.neg)

	def inv(self) -> TiMatrix:
		n = self.rows
		if self.cols != n:
			raise InvalidDimError(f"inv: matrix must be square, got {self.rows}×{self.cols}")

		aug = [row.copy() + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(self.data)]
		for col in range(n):
			pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
			aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
			pivot = aug[col][col]
			if abs(pivot) < 1e-12:
				raise SingularMatrixError("inv: matrix is singular")

			for j in range(2 * n):
				aug[col][j] /= pivot
			for r in range(n):
				if r == col:
					continue
				factor = aug[r][col]
				for j in range(2 * n):
					aug[r][j] -= factor * aug[col][j]

		return TiMatrix([row[n:] for row in aug])

	def transpose(self):
		return TiMatrix([[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)])

	def copy(self) -> TiMatrix:
		return TiMatrix([row.copy() for row in self.data])

	def __repr__(self) -> str:
		return '[' + ''.join('[' + ' '.join(repr_num(x) for x in row) + ']' for row in self.data) + ']'


# ── TiString / TiEquation ─────────────────────────────────────────────────────

class TiString:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens

	@classmethod
	def from_str(cls, s: str) -> TiString:
		"""Create a TiString from a plain Python string."""
		from catalog import TEXT_INPUT
		return cls([TEXT_INPUT[c] for c in s])

	def __len__(self) -> int:
		return len(self.tokens)

	def __add__(self, other: Any) -> TiString:
		if isinstance(other, TiString):
			if not self.tokens or not other.tokens:
				raise InvalidDimError(f"Cannot concatenate with an empty string (God knows why)!")
			return TiString(self.tokens + other.tokens)
		raise DataTypeError(f"Expected string but got {other}")

	def __eq__(self, other: object) -> bool:
		if isinstance(other, TiString):
			return self.tokens == other.tokens
		raise DataTypeError(f"Expected string but got {other}")

	def __str__(self) -> str:
		return ''.join(t.text for t in self.tokens)

	def __repr__(self) -> str:
		return '"' + str(self) + '"'


class TiEquation:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens

	def eval(self, env: Environment) -> Any:
		from parser import Parser
		from titoken import EOF_CODE
		try:
			parser = Parser(self.tokens, env)
			value = parser.parse_expr()
			parser.expect(EOF_CODE)
			return value
		except RecursionError:
			raise TiMemoryError("Equation recursion overflow (ERR:MEMORY)")

	def __str__(self) -> str:
		return ''.join(t.text for t in self.tokens)

	def __repr__(self) -> str:
		return '"' + str(self) + '"'


# ── Type guards ───────────────────────────────────────────────────────────────

def is_complex_val(x) -> bool:
	"""Return True if x is a complex scalar or a TiList containing complex elements."""
	return isinstance(x, complex) or (isinstance(x, TiList) and x.is_complex)

def require_list(value: Any) -> TiList:
	return _require_type(value, TiList)

def require_real_list(value: Any, exc_cls=DataTypeError) -> TiList:
	"""A TiList whose elements are all real (rejects a complex list)."""
	require_list(value)
	if value.is_complex:
		raise exc_cls(f"Expected real list, got complex: {value}")
	return value

def require_complex_list(value: Any, exc_cls=DataTypeError) -> TiList:
	"""A TiList that holds complex elements (rejects a real list and any scalar)."""
	require_list(value)
	if not value.is_complex:
		raise exc_cls(f"Expected complex list, got real: {value}")
	return value

def require_matrix(value: Any) -> TiMatrix:
	return _require_type(value, TiMatrix)

def require_string(value: Any) -> TiString:
	return _require_type(value, TiString)

def require_equation(value: Any) -> TiEquation:
	return _require_type(value, TiEquation)

def require_list_or_matrix(value: Any, exc_cls=DataTypeError) -> Any:
	"""A TiList or TiMatrix (the polymorphic shape accepted by cumSum(, augment(...)."""
	if isinstance(value, (TiList, TiMatrix)):
		return value
	raise exc_cls(f"Expected a list or matrix, got {value!r}")

def require_vectorizable(value: Any, exc_cls=DataTypeError) -> Any:
	"""A numeric scalar or a TiList — a slot that maps element-wise over a list."""
	if isinstance(value, (Number, TiList)):
		return value
	raise exc_cls(f"Expected a number or list, got {value!r}")

def require_vectorizable_real(value: Any, exc_cls=DataTypeError) -> Any:
	"""A real scalar or a real TiList (rejects complex scalars and complex lists)."""
	if is_complex_val(value):
		raise exc_cls(f"Expected real number or list, got complex: {value}")
	if isinstance(value, (Number, TiList)):
		return value
	raise exc_cls(f"Expected a number or list, got {value!r}")

def require_matrix_vectorizable(value: Any, exc_cls=DataTypeError) -> Any:
	"""A numeric scalar, a TiList, or a TiMatrix — maps over a list or matrix."""
	if isinstance(value, (Number, TiList, TiMatrix)):
		return value
	raise exc_cls(f"Expected a number, list, or matrix, got {value!r}")


# ── Vectorization ─────────────────────────────────────────────────────────────
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


# ── Variable hierarchy ────────────────────────────────────────────────────────

class Variable(ABC):
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


class UserList(ListVariable):

	def __init__(self, env: Environment, name: str):
		self.lookup = env.user_lists
		self.name = name

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


class MatrixVariable(Variable):
	def normalize(self, value):
		return require_matrix(value).copy()


class StringVariable(Variable):
	def normalize(self, value):
		return require_string(value)


class EquationVariable(Variable):
	def normalize(self, value):
		if isinstance(value, TiString):
			return TiEquation(value.tokens)
		if isinstance(value, TiEquation):
			return value
		raise DataTypeError(f"Expected equation or string, got {value!r}")


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
