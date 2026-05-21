import builtins
from itertools import repeat
from numbers import Number


class TiTypeError(Exception):
	pass


def _repr_num(value):
	int_value = int(value)
	return repr(int_value if int_value == value else value)


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
	int_value = int(value)
	if value != int_value:
		raise ValueError(f"Expected integer, got {value}")
	return int_value

def require_list(value):
	return _require_type(value, TiList)

def require_matrix(value):
	return _require_type(value, TiMatrix)

def require_str(value):
	return _require_type(value, TiString)


class TiString:
	__slots__ = ('tokens',)

	def __init__(self, tokens: list):
		self.tokens = tokens

	def __len__(self):
		return len(self.tokens)

	def __eq__(self, other):
		if isinstance(other, TiString):
			return len(self.tokens) == len(other.tokens) and all(
				a.code == b.code for a, b in zip(self.tokens, other.tokens)
			)
		return NotImplemented

	def __str__(self):
		return ''.join(t.text for t in self.tokens)

	def __repr__(self):
		return '"' + str(self) + '"'


class TiList:

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

	def __repr__(self):
		return f"{{{','.join(_repr_num(i) for i in self)}}}"

	def set_dim(self, value):
		new_dim = int(value)
		dim = len(self)
		if new_dim < dim:
			del self.inner[new_dim:]
		elif new_dim > dim:
			self.inner.extend(repeat(0, new_dim - dim))

	def copy(self):
		return TiList(self.inner.copy())


def _check_valid_dim(rows, cols):
	rows = require_int(rows)
	cols = require_int(cols)
	if not (1 <= rows <= 99):
		raise IndexError(f"{rows=}")
	if not (1 <= cols <= 99):
		raise IndexError(f"{cols=}")
	return rows, cols


class TiMatrix:

	def __init__(self, data=None):
		self.inner = [] if data is None else data

	@property
	def rows(self):
		return len(self.inner)

	@property
	def cols(self):
		return len(self.inner[0]) if self.inner else 0
		
	def _check_index(self, index):
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

	def __repr__(self):
		return '[' + ''.join('[' + ' '.join(_repr_num(x) for x in row) + ']' for row in self.inner) + ']'
		# widths = [max(len(_repr_num(row[c])) for row in self.inner) for c in range(len(self.inner[0]))]
		# return f"[{'\n'.join([f"[{' '.join(f'{_repr_num(x):{widths[c]}}' for c, x in enumerate(row))}]" for row in self.inner])}]"

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

	def __matmul__(self, other):
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
				result = result @ base
			base = base @ base
			n >>= 1
		return result

	def copy(self):
		return TiMatrix([row.copy() for row in self.inner])
