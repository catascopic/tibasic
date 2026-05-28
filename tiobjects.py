import builtins
import operator
from itertools import repeat
from numbers import Number

from errors import (
	DataTypeError, DimMismatchError, InvalidDimError,
	SingularMatrixError, DomainError,
)


def repr_num(value):
	return repr(int(value) if not isinstance(value, complex) and value.is_integer() else value)


class DMS(float):
	"""A float in decimal degrees that displays as degrees°minutes'seconds\"."""
	def __repr__(self):
		total = abs(self)
		sign = '-' if self < 0 else ''
		d = int(total)
		m_total = (total - d) * 60
		m = int(m_total)
		s = (m_total - m) * 60
		return f"{sign}{d}°{m}'{repr_num(s)}\""


# ── Guard functions ───────────────────────────────────────────────────────────────

def _require_type(value, tp):
	if not isinstance(value, tp):
		raise DataTypeError(f"Invalid value: {value!r}; required: {tp.__name__}")
	return value

def require_num(value):
	return _require_type(value, Number)

def require_real(value):
	require_num(value)
	if isinstance(value, complex):
		raise DataTypeError(f"Expected real number, got complex: {value}")
	return value

def require_int(value):
	require_real(value)
	if not value.is_integer():
		raise DataTypeError(f"Expected integer, got {value}")
	return int(value)

def require_list(value):
	return _require_type(value, TiList)

def require_matrix(value):
	return _require_type(value, TiMatrix)

def require_str(value):
	return _require_type(value, TiString)


class TiList:
	__slots__ = ('data',)

	def __init__(self, data=None):
		self.data = [] if data is None else data

	@classmethod
	def alloc(cls, value):
		return cls(list(repeat(0, require_int(value))))

	def __getitem__(self, index):
		if index != int(index) or not (1 <= index <= len(self)):
			raise InvalidDimError(f"{index=}")
		return self.data[int(index) - 1]

	def __setitem__(self, index, value):
		require_num(value)
		if index == len(self) + 1:
			self.data.append(value)
		elif index != int(index) or not (1 <= index <= len(self)):
			raise InvalidDimError(f"out of bounds: {index}; dim: {len(self)}")
		else:
			self.data[int(index) - 1] = value

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		return iter(self.data)

	def __neg__(self):
		return TiList([-a for a in self.data])

	def set_dim(self, value):
		value = require_int(value)
		dim = len(self)
		if value < dim:
			del self.data[value:]
		elif value > dim:
			self.data.extend(repeat(0, value - dim))

	def copy(self):
		return TiList(self.data.copy())

	def __repr__(self):
		return f"{{{','.join(repr_num(i) for i in self)}}}"


def _vectorize_op(op):
	def list_op(self, other):
		if isinstance(other, TiList):
			if len(self) != len(other):
				raise DimMismatchError(f"Dim mismatch: {len(self)} vs {len(other)}")
			return TiList([op(a, b) for a, b in zip(self, other)])
		return TiList([op(a, other) for a in self.data])
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
	('__eq__', operator.eq),
	('__ne__', operator.ne),
	('__lt__', operator.lt),
	('__gt__', operator.gt),
	('__le__', operator.le),
	('__ge__', operator.ge),
]:
	setattr(TiList, name, _vectorize_op(op))


def _check_valid_dim(value):
	require_list(value)
	if len(value) != 2:
		raise InvalidDimError(f"Matrix dimensions must be 2 elements, but got {value}")
	rows, cols = value
	rows = require_int(rows)
	cols = require_int(cols)
	if not (1 <= rows <= 99):
		raise InvalidDimError(f"Required: 1 <= rows <= 99; got {rows}")
	if not (1 <= cols <= 99):
		raise InvalidDimError(f"Required: 1 <= cols <= 99; got {cols}")
	return rows, cols


class TiMatrix:
	__slots__ = ('data',)

	def __init__(self, data=None):
		self.data = [] if data is None else data

	@classmethod
	def alloc(cls, dim_list):
		rows, cols = _check_valid_dim(dim_list)
		return cls([list(repeat(0, cols)) for r in range(rows)])

	@property
	def rows(self):
		return len(self.data)

	@property
	def cols(self):
		return len(self.data[0]) if self.data else 0

	def _check_index(self, index):
		if len(index) != 2:
			raise ArgumentError(f"Matrix index must have 2 elements but got {index}")
		row_index, col_index = index
		row_index = require_int(row_index)
		col_index = require_int(col_index)
		if not (1 <= row_index <= self.rows):
			raise InvalidDimError(f"{row_index=}")
		if not (1 <= col_index <= self.cols):
			raise InvalidDimError(f"{col_index=}")
		return row_index, col_index

	def __getitem__(self, index):
		row_index, col_index = self._check_index(index)
		return self.data[int(row_index) - 1][int(col_index) - 1]

	def __setitem__(self, index, value):
		require_real(value)
		row_index, col_index = self._check_index(index)
		self.data[int(row_index) - 1][int(col_index) - 1] = value

	def set_dim(self, dim_list: TiList):
		new_rows, new_cols = _check_valid_dim(dim_list)
		self.data = [[
			self.data[r][c] if r < self.rows and c < self.cols else 0.0
			for c in range(new_cols)
		] for r in range(new_rows)]

	def get_row(self, r) -> list:
		n = require_int(r)
		if not (1 <= n <= self.rows):
			raise InvalidDimError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		return self.data[n - 1]

	def set_row(self, r, row: list) -> None:
		n = require_int(r)
		if not (1 <= n <= self.rows):
			raise InvalidDimError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		self.data[n - 1] = row

	def transform(self, func):
		return TiMatrix([[func(x) for x in row] for row in self.data])

	def transform_zip(self, other, func):
		if self.rows != other.rows or self.cols != other.cols:
			raise DimMismatchError(
				"Operation not allowed on matrices with different sizes: "
				f"{self.rows}×{self.cols} vs. {other.rows}×{other.cols}"
			)
		return TiMatrix([
			[func(a, b) for a, b in zip(row_a, row_b)]
			for row_a, row_b in zip(self.data, other.data)
		])

	def __add__(self, other):
		return self.transform_zip(other, operator.add) if isinstance(other, TiMatrix) else self.transform(lambda x: x + other)

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		return self.transform_zip(other, operator.sub) if isinstance(other, TiMatrix) else self.transform(lambda x: x - other)

	def __rsub__(self, other):
		return self.transform(lambda x: other - x)

	def __matmul__(self, other):
		if self.cols != other.rows:
			raise DimMismatchError(f"Dimension mismatch: ({self.rows}×{self.cols}) @ ({other.rows}×{other.cols})")
		other_cols = list(zip(*other.data))
		return TiMatrix([
			[builtins.sum(a * b for a, b in zip(row, col)) for col in other_cols]
			for row in self.data
		])

	def __mul__(self, other):
		if isinstance(other, Number):
			return self.transform(lambda x: x * other)
		if isinstance(other, TiMatrix):
			return self @ other
		return NotImplemented

	def __rmul__(self, other):
		if isinstance(other, Number):
			return self.transform(lambda x: other * x)
		return NotImplemented

	def __pow__(self, n):
		n = require_int(n)
		if self.rows != self.cols:
			raise DomainError(f"Matrix power requires a square matrix, got {self.rows}×{self.cols}")
		if n < 0:
			raise DomainError("Negative matrix power not supported")
		size = self.rows
		result = TiMatrix([[int(r == c) for c in range(size)] for r in range(size)])
		base = self
		while n > 0:
			if n & 1:
				result = result @ base
			base = base @ base
			n >>= 1
		return result

	def __eq__(self, other):
		if isinstance(other, TiMatrix):
			return self.data == other.data
		raise DataTypeError(f"Cannot compare matrix with {type(other).__name__}")

	# get __ne__ for free

	def __neg__(self):
		return self.transform(operator.neg)

	def inv(self):
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

	def copy(self):
		return TiMatrix([row.copy() for row in self.data])

	def __repr__(self):
		return '[' + ''.join('[' + ' '.join(repr_num(x) for x in row) + ']' for row in self.data) + ']'
		# widths = [max(len(repr_num(row[c])) for row in self.data) for c in range(len(self.data[0]))]
		# return f"[{'\n'.join([f"[{' '.join(f'{repr_num(x):{widths[c]}}' for c, x in enumerate(row))}]" for row in self.data])}]"


class TiString:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list['Token']):
		self.tokens = tokens

	@classmethod
	def from_str(cls, s: str) -> 'TiString':
		"""Create a TiString from a plain Python string."""
		from tokens import CHARS
		return cls([CHARS[c] for c in s])

	def __len__(self):
		return len(self.tokens)

	def __add__(self, other):
		if isinstance(other, TiString):
			if not self.tokens or not other.tokens:
				raise InvalidDimError(f"Cannot concatenate with an empty string (God knows why)!")
			return TiString(self.tokens + other.tokens)
		raise DataTypeError(f"Expected string but got {other}")

	def __eq__(self, other):
		if isinstance(other, TiString):
			return self.tokens == other.tokens
		raise DataTypeError(f"Expected string but got {other}")

	def __str__(self):
		return ''.join(t.text for t in self.tokens)

	def __repr__(self):
		return '"' + str(self) + '"'
