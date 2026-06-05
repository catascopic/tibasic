from __future__ import annotations

import builtins
import operator
from collections.abc import Callable, Iterator
from functools import wraps
from itertools import repeat
from numbers import Number
from typing import Any, TypeVar, TYPE_CHECKING

from errors import (
	DataTypeError, DimMismatchError, InvalidDimError,
	SingularMatrixError, DomainError
)

if TYPE_CHECKING:
	from titoken import Token
	from environment import Environment


def repr_num(value: Number) -> str:
	return repr(int(value) if not isinstance(value, complex) and value.is_integer() else value)


# ── Guard functions ───────────────────────────────────────────────────────────────

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

def require_int(value: Any, exc_cls=DomainError) -> int:
	require_real(value, exc_cls)
	if not value.is_integer():
		raise exc_cls(f"Expected integer, got {value}")
	return value

def require_list(value: Any) -> TiList:
	return _require_type(value, TiList)

def require_nonempty_list(value: Any) -> TiList:
	lst = require_list(value)
	if not lst.data:
		raise InvalidDimError("list is empty")
	return lst

def require_matrix(value: Any) -> TiMatrix:
	return _require_type(value, TiMatrix)

def require_str(value: Any) -> TiString:
	return _require_type(value, TiString)


def _get_dim(value: Any) -> int:
	require_int(value, InvalidDimError)
	return int(value)


class TiList:
	__slots__ = ('data',)

	def __init__(self, data: list[Number] | None = None) -> None:
		self.data = [] if data is None else data
		if not all(isinstance(i, Number) for i in self.data):
			raise ValueError(self.data)

	@classmethod
	def alloc(cls, size: Number) -> TiList:
		return cls(list(repeat(0.0, _get_dim(size))))

	def _check_index(self, index):
		# Hold off on _get_dim because __setitem__ has these decoupled
		if not (1 <= index <= len(self)):
			raise InvalidDimError(f"out of bounds: {index}; dim: {len(self)}")
		return index

	def __getitem__(self, index: Number) -> Number:
		return self.data[self._check_index(_get_dim(index)) - 1]

	def __setitem__(self, index: Number, value: Number) -> None:
		index = _get_dim(index)
		if index == len(self) + 1:
			self.data.append(value)
		else:
			self.data[self._check_index(index) - 1] = value

	def __len__(self) -> int:
		return len(self.data)

	def __iter__(self) -> Iterator:
		return iter(self.data)

	def __neg__(self) -> TiList:
		return TiList([-a for a in self.data])

	def set_dim(self, value: Any) -> None:
		value = _get_dim(value)
		dim = len(self)
		if value < dim:
			del self.data[value:]
		elif value > dim:
			self.data.extend(repeat(0, value - dim))

	def copy(self) -> TiList:
		return TiList(self.data.copy())

	def __repr__(self) -> str:
		return f"{{{','.join(repr_num(i) for i in self)}}}"


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


for name, op in [
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
	setattr(TiList, name, _vectorize_op(op))


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
		if len(index) != 2:
			raise ArgumentError(f"Matrix index must have 2 elements but got {index}")
		row_index, col_index = index
		row_index = _get_dim(row_index)
		col_index = _get_dim(col_index)
		if not (1 <= row_index <= self.rows):
			raise InvalidDimError(f"{row_index=}")
		if not (1 <= col_index <= self.cols):
			raise InvalidDimError(f"{col_index=}")
		return row_index, col_index

	def __getitem__(self, index: Any) -> float:
		row_index, col_index = self._check_index(index)
		return self.data[row_index - 1][col_index - 1]

	def __setitem__(self, index: Any, value: Any) -> None:
		require_real(value)
		row_index, col_index = self._check_index(index)
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

	# get __ne__ for free

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

	def copy(self) -> TiMatrix:
		return TiMatrix([row.copy() for row in self.data])

	def __repr__(self) -> str:
		return '[' + ''.join('[' + ' '.join(repr_num(x) for x in row) + ']' for row in self.data) + ']'
		# widths = [max(len(repr_num(row[c])) for row in self.data) for c in range(len(self.data[0]))]
		# return f"[{'\n'.join([f"[{' '.join(f'{repr_num(x):{widths[c]}}' for c, x in enumerate(row))}]" for row in self.data])}]"


class TiString:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens

	@classmethod
	def from_str(cls, s: str) -> TiString:
		"""Create a TiString from a plain Python string."""
		from catalog import CHARS
		return cls([CHARS[c] for c in s])

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
		from parser import Parser, EOF_TOKEN
		parser = Parser(self.tokens, env)
		value = parser.parse_expr()
		parser.expect(EOF_TOKEN)
		return value

	def __repr__(self) -> str:
		return ''.join(t.text for t in self.tokens)


def require_equation(value: Any) -> TiEquation:
	return _require_type(value, TiEquation)
