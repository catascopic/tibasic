import builtins
import operator
from itertools import repeat
from numbers import Number


class TiTypeError(Exception):
	pass


def repr_num(value):
	return repr(int(value) if value.is_integer() else value)


# ── Guard functions ───────────────────────────────────────────────────────────────

def _require_type(value, tp):
	if not isinstance(value, tp):
		raise ValueError(f"Invalid value: {value!r}; required: {tp.__name__}")
	return value

def require_num(value):
	return _require_type(value, Number)

def require_real(value):
	require_num(value)
	if isinstance(value, complex):
		raise ValueError(f"Expected real number, got complex: {value}")
	return value

def require_int(value):
	require_real(value)
	if not value.is_integer():
		raise ValueError(f"Expected integer, got {value}")
	return int(value)

def require_list(value):
	return _require_type(value, TiList)

def require_matrix(value):
	return _require_type(value, TiMatrix)

def require_str(value):
	return _require_type(value, TiString)


class TiList:
	__slots__ = ('inner',)

	def __init__(self, data=None):
		self.inner = [] if data is None else data

	def __getitem__(self, index):
		if index != int(index) or not (1 <= index <= len(self)):
			raise IndexError(f"{index=}")
		return self.inner[int(index) - 1]

	def __setitem__(self, index, value):
		if index == len(self) + 1:
			self.inner.append(value)
		elif index != int(index) or not (1 <= index <= len(self)):
			raise ValueError(f"out of bounds: {index}; dim: {len(self)}")
		else:
			self.inner[int(index) - 1] = value

	def __len__(self):
		return len(self.inner)

	def __iter__(self):
		return iter(self.inner)
	
	def __neg__(self):
		return TiList([-a for a in self.inner])

	def set_dim(self, value):
		new_dim = int(value)
		dim = len(self)
		if new_dim < dim:
			del self.inner[new_dim:]
		elif new_dim > dim:
			self.inner.extend(repeat(0, new_dim - dim))

	def copy(self):
		return TiList(self.inner.copy())

	def __repr__(self):
		return f"{{{','.join(repr_num(i) for i in self)}}}"


def _vectorize_op(op):
	def list_op(self, other):
		if isinstance(other, TiList):
			return TiList([op(a, b) for a, b in zip(self, other, strict=True)])
		return TiList([op(a, other) for a in self.inner])
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


def _check_valid_dim(rows, cols):
	rows = require_int(rows)
	cols = require_int(cols)
	if not (1 <= rows <= 99):
		raise IndexError(f"{rows=}")
	if not (1 <= cols <= 99):
		raise IndexError(f"{cols=}")
	return rows, cols


class TiMatrix:
	__slots__ = ('inner',)

	def __init__(self, data=None):
		self.inner = [] if data is None else data

	@property
	def rows(self):
		return len(self.inner)

	@property
	def cols(self):
		return len(self.inner[0]) if self.inner else 0
		
	def _check_index(self, index):
		if len(index) != 2:
			raise ValueError(f"Matrix index must have 2 elements but got {index}")
		row_index, col_index = index
		row_index = require_int(row_index)
		col_index = require_int(col_index)
		if not (1 <= row_index <= self.rows):
			raise IndexError(f"{row_index=}")
		if not (1 <= col_index <= self.cols):
			raise IndexError(f"{col_index=}")
		return row_index, col_index

	def __getitem__(self, index):
		row_index, col_index = self._check_index(index)
		return self.inner[int(row_index) - 1][int(col_index) - 1]

	def __setitem__(self, index, value):
		row_index, col_index = self._check_index(index)
		self.inner[int(row_index) - 1][int(col_index) - 1] = value

	def set_dim(self, dim_list: TiList):
		new_rows, new_cols = _check_valid_dim(*dim_list.inner)
		self.inner = [
			[
				self.inner[r][c] if r < self.rows and c < self.cols else 0.0
				for c in range(new_cols)
			] for r in range(new_rows)
		]

	def get_row(self, r) -> list:
		n = _check_int(r)
		if not (1 <= n <= self.rows):
			raise IndexError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		return self.inner[n - 1].copy()

	def set_row(self, r, row: list) -> None:
		n = _check_int(r)
		if not (1 <= n <= self.rows):
			raise IndexError(f"row {r} out of range for {self.rows}×{self.cols} matrix")
		self.inner[n - 1] = row

	def transform(self, func):
		return TiMatrix([[func(x) for x in row] for row in self.inner])
	
	def __add__(self, other):
		raise NotImplementedError
	
	def __radd__(self, other):
		raise NotImplementedError

	def __sub__(self, other):
		raise NotImplementedError
	
	def __rsub__(self, other):
		raise NotImplementedError

	def __mul__(self, other):
		if isinstance(other, Number):
			return self.transform(lambda i: i + other)
		if not isinstance(other, TiMatrix):
			raise ValueError(f"Cannot multiply matrix by {type(other).__name__}")
		if self.cols != other.rows:
			raise ValueError(f"Dimension mismatch: ({self.rows}×{self.cols}) @ ({other.rows}×{other.cols})")
		return TiMatrix([
			[
				builtins.sum(
					self.inner[r][k] * other.inner[k][c] for k in range(self.cols)
				) for c in range(other.cols)
			] for r in range(self.rows)
		])
	
	def __rmul__(self, other):
		raise NotImplementedError

	def __pow__(self, n):
		n = require_int(n)
		if self.rows != self.cols:
			raise ValueError(f"Matrix power requires a square matrix, got {self.rows}×{self.cols}")
		if n < 0:
			raise ValueError("Negative matrix power not supported")
		size = self.rows
		result = TiMatrix([[1.0 if r == c else 0.0 for c in range(size)] for r in range(size)])
		base = self.copy()
		while n > 0:
			if n & 1:
				result = result * base
			base = base * base
			n >>= 1
		return result
	
	def __eq__(self, other):
		raise NotImplementedError
	
	# get __ne__ for free
	
	def __neg__(self):
		return transform(operator.neg)
	
	def inv(self):
		n = self.rows
		if self.cols != n:
			raise ValueError(f"inv: matrix must be square, got {m.rows}×{m.cols}")
		
		aug = [row.copy() + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(self.inner)]
		for col in range(n):
			pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
			aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
			pivot = aug[col][col]
			if abs(pivot) < 1e-12:
				raise ValueError("inv: matrix is singular")
				
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
		return TiMatrix([row.copy() for row in self.inner])

	def __repr__(self):
		return '[' + ''.join('[' + ' '.join(repr_num(x) for x in row) + ']' for row in self.inner) + ']'
		# widths = [max(len(repr_num(row[c])) for row in self.inner) for c in range(len(self.inner[0]))]
		# return f"[{'\n'.join([f"[{' '.join(f'{repr_num(x):{widths[c]}}' for c, x in enumerate(row))}]" for row in self.inner])}]"


class TiString:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list['Token']):
		self.tokens = tokens

	def __len__(self):
		return len(self.tokens)

	def __add__(self, other):
		if isinstance(other, TiString):
			return TiString(self.tokens + other.tokens)
		raise ValueError(f"Expected string but got {other}")

	def __eq__(self, other):
		if isinstance(other, TiString):
			return self.tokens == other.tokens
		raise ValueError(f"Expected string but got {other}")

	def __str__(self):
		return ''.join(t.text for t in self.tokens)

	def __repr__(self):
		return '"' + str(self) + '"'
