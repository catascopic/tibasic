import cmath
import builtins
import math
import operator
import random
import sys
from functools import wraps
from itertools import accumulate, pairwise, chain, repeat, batched
from math import prod
from numbers import Number


# ── Formatting ──────────────────────────────────────────────────────────────────

def _repr_num(value):
	int_value = int(value)
	return repr(int_value if int_value == value else value)


# ── Type helpers ────────────────────────────────────────────────────────────────

def _require_type(value, tp):
	if not isinstance(value, tp):
		raise ValueError(f"Invalid type: {value!r}; required: {tp.__name__}")
	return value

def _require_num(value):
	return _require_type(value, Number)

def _require_real(value):
	if isinstance(value, complex) and value.imag != 0:
		raise ValueError(f"Expected real number, got complex: {value}")
	return value.real if isinstance(value, complex) else value

def _require_list(value):
	return _require_type(value, TiList)

def _require_matrix(value):
	return _require_type(value, TiMatrix)

def _require_str(value):
	return _require_type(value, str)

def _require_int(value):
	value = _require_real(value)
	if value != int(value):
		raise ValueError(f"Expected integer, got {value}")
	return int(value)


# ── Logical operators ────────────────────────────────────────────────────────────

def and_(a, b):
	return float(bool(_require_real(a)) and bool(_require_real(b)))

def or_(a, b):
	return float(bool(_require_real(a)) or bool(_require_real(b)))

def xor(a, b):
	return float(bool(_require_real(a)) ^ bool(_require_real(b)))


# ── Comparison operators ──────────────────────────────────────────────────────────

def eq(a, b): return float(_require_real(a) == _require_real(b))
def ne(a, b): return float(_require_real(a) != _require_real(b))
def lt(a, b): return float(_require_real(a) < _require_real(b))
def gt(a, b): return float(_require_real(a) > _require_real(b))
def le(a, b): return float(_require_real(a) <= _require_real(b))
def ge(a, b): return float(_require_real(a) >= _require_real(b))


# ── TiList ────────────────────────────────────────────────────────────────────────

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

	def __neg__(self):
		return TiList([-x for x in self])

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


def make_binary_list_op(op):
	def list_op(self, other):
		if isinstance(other, TiList):
			return TiList([op(a, b) for a, b in zip(self, other, strict=True)])
		return TiList([op(a, other) for a in self])
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
	('__eq__', lambda a, b: float(a == b)),
	('__ne__', lambda a, b: float(a != b)),
	('__lt__', lambda a, b: float(a < b)),
	('__gt__', lambda a, b: float(a > b)),
	('__le__', lambda a, b: float(a <= b)),
	('__ge__', lambda a, b: float(a >= b)),
]:
	setattr(TiList, _name, make_binary_list_op(_op))


# ── TiMatrix ──────────────────────────────────────────────────────────────────────

class TiMatrix:

	def __init__(self, data=None):
		self.inner = [] if data is None else data

	@property
	def rows(self):
		return len(self.inner)

	@property
	def cols(self):
		return len(self.inner[0]) if self.inner else 0

	def __getitem__(self, index):
		row_index, col_index = index
		if row_index != int(row_index) or not (1 <= row_index <= self.rows):
			raise IndexError(f"{row_index=}")
		if col_index != int(col_index) or not (1 <= col_index <= self.cols):
			raise IndexError(f"{col_index=}")
		return self.inner[int(row_index) - 1][int(col_index) - 1]

	def __setitem__(self, index, value):
		row_index, col_index = index
		if row_index != int(row_index) or not (1 <= row_index <= self.rows):
			raise IndexError(f"{row_index=}")
		if col_index != int(col_index) or not (1 <= col_index <= self.cols):
			raise IndexError(f"{col_index=}")
		self.inner[int(row_index) - 1][int(col_index) - 1] = value

	def __len__(self):
		return self.rows

	def __iter__(self):
		return iter(self.inner)

	def __repr__(self):
		return '[' + ''.join('[' + ' '.join(_repr_num(x) for x in row) + ']' for row in self.inner) + ']'

	def set_dim(self, dim_list):
		new_rows, new_cols = int(dim_list[0]), int(dim_list[1])
		self.inner = [
			[self.inner[r][c] if r < self.rows and c < self.cols else 0.0
			 for c in range(new_cols)]
			for r in range(new_rows)
		]

	def transform(self, func):
		return TiMatrix([[func(x) for x in row] for row in self.inner])

	def __neg__(self):
		return self.transform(operator.neg)

	def __matmul__(self, other):
		if not isinstance(other, TiMatrix):
			raise ValueError(f"Cannot multiply matrix by {type(other).__name__}")
		if self.cols != other.rows:
			raise ValueError(f"Dimension mismatch: ({self.rows}×{self.cols}) @ ({other.rows}×{other.cols})")
		return TiMatrix([
			[builtins.sum(self.inner[r][k] * other.inner[k][c] for k in range(self.cols))
			 for c in range(other.cols)]
			for r in range(self.rows)
		])

	def __pow__(self, n):
		n = _require_int(n)
		if self.rows != self.cols:
			raise ValueError(f"Matrix power requires a square matrix, got {self.rows}×{self.cols}")
		if n < 0:
			raise ValueError("Negative matrix power not supported")
		result = identity(self.rows)
		base = self.copy()
		while n > 0:
			if n & 1:
				result = result @ base
			base = base @ base
			n >>= 1
		return result

	def __mul__(self, other):
		if isinstance(other, TiMatrix):
			return self @ other
		return self.transform(lambda x: x * other)

	def __rmul__(self, other):
		if isinstance(other, TiMatrix):
			return other @ self
		return self.transform(lambda x: other * x)

	def copy(self):
		return TiMatrix([row.copy() for row in self.inner])


def make_matrix_binary_op(op):
	def mat_op(self, other):
		_require_matrix(other)
		if (self.rows, self.cols) != (other.rows, other.cols):
			raise ValueError(f"Dim mismatch: {self.rows}×{self.cols} vs {other.rows}×{other.cols}")
		return TiMatrix([[op(self.inner[r][c], other.inner[r][c]) for c in range(self.cols)] for r in range(self.rows)])
	return mat_op


for _name, _op in [
	('__add__', operator.add),
	('__sub__', operator.sub),
]:
	setattr(TiMatrix, _name, make_matrix_binary_op(_op))


# ── Decorators ────────────────────────────────────────────────────────────────────

def handle_complex(func):
	"""Apply a real-valued func separately to the real and imaginary parts."""
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply


def vectorized(func):
	"""Allow a scalar function to accept TiList arguments, applying element-wise."""
	@wraps(func)
	def apply(*args):
		len_check = set()
		vec = []
		for a in args:
			if isinstance(a, TiList):
				len_check.add(len(a))
				vec.append(a)
			else:
				vec.append(repeat(_require_num(a)))
		if not len_check:
			return func(*args)
		if len(len_check) == 1:
			return TiList([func(*v) for v in zip(*vec)])
		raise ValueError(f"Dim mismatch: {len_check}")
	return apply


def vectorized_with_matrix(func):
	"""Like vectorized, but also applies element-wise to a TiMatrix first argument."""
	vec = vectorized(func)
	@wraps(func)
	def apply(a, *args):
		if isinstance(a, TiMatrix):
			return a.transform(lambda x: func(x, *args))
		return vec(a, *args)
	return apply


# ── dim ───────────────────────────────────────────────────────────────────────────

def dim(value):
	if isinstance(value, TiList):
		return len(value)
	if isinstance(value, TiMatrix):
		return TiList([value.rows, value.cols])
	raise ValueError(f"Invalid type: {type(value).__name__}; required: list or matrix")


# ── Numeric functions ────────────────────────────────────────────────────────────

@vectorized
def not_(x):
	return float(not _require_real(x))


@vectorized_with_matrix
@handle_complex
def i_part(x):
	return math.trunc(x)


@vectorized_with_matrix
@handle_complex
def int_(x):
	return math.floor(x)


@vectorized_with_matrix
@handle_complex
def f_part(x):
	return x - math.trunc(x)


@vectorized
def sqrt(x):
	return cmath.sqrt(x) if isinstance(x, complex) or x < 0 else math.sqrt(x)


@vectorized
def cbrt(x):
	return cmath.exp(cmath.log(x) / 3) if isinstance(a, complex) else math.cbrt(x)


@vectorized
def xth_root(x, n):
	return cmath.exp(cmath.log(x) / n) if isinstance(x, complex) or x < 0 else math.log(x) / n


def cum_sum(lst):
	return TiList(list(accumulate(_require_list(lst))))


def delta_list(lst):
	return TiList([b - a for a, b in pairwise(_require_list(lst))])


def augment(a, b):
	if isinstance(a, TiList) and isinstance(b, TiList):
		return TiList(a.inner + b.inner)
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if a.rows != b.rows:
			raise ValueError(f"Row count mismatch: {a.rows} vs {b.rows}")
		return TiMatrix([r1 + r2 for r1, r2 in zip(a.inner, b.inner)])
	raise ValueError("augment: both args must be lists or both must be matrices")


@vectorized
def real(x):
	return x.real if isinstance(x, complex) else x


@vectorized
def imag(x):
	return x.imag if isinstance(x, complex) else 0


@vectorized
def conj(a):
	return complex(a.real, -a.imag) if isinstance(a, complex) else a


@vectorized
def angle(a):
	return cmath.phase(a)


# ── Sorting / filling ────────────────────────────────────────────────────────────

def sort_a(lst, *dep, reverse=False):
	inner = _require_list(lst).inner
	indices = sorted(range(len(inner)), key=lambda i: inner[i], reverse=reverse)
	lst.inner = [inner[i] for i in indices]
	for d in dep:
		d.inner = [d.inner[i] for i in indices]


def sort_d(lst, *dep):
	sort_a(lst, *dep, reverse=True)


def fill(lst, x):
	_require_num(x)
	if isinstance(lst, TiList):
		for i in range(len(lst.inner)):
			lst.inner[i] = x
	elif isinstance(lst, TiMatrix):
		for row in lst.inner:
			for i in range(len(row)):
				row[i] = x
	else:
		raise ValueError(f"fill: expected list or matrix, got {type(lst).__name__}")


# ── String functions ────────────────────────────────────────────────────────────

def in_string(value, needle):
	pos = _require_str(value).find(needle)
	return float(pos + 1) if pos >= 0 else 0.0


def length(value):
	return float(len(_require_str(value)))


def sub(value, start, length):
	if isinstance(value, Number):
		return value / 100
	_require_str(value)
	start, length = int(start), int(length)
	if length < 1:
		raise ValueError(f"sub: length must be ≥ 1, got {length}")
	if not (1 <= start <= len(value) - length + 1):
		raise ValueError(f"sub: index out of range")
	return value[start - 1 : start - 1 + length]

# ── Aggregate / statistics ───────────────────────────────────────────────────────

@vectorized_with_matrix
def round(a, b=9):
	return builtins.round(a, int(b))

def _minmax(fn, a, b):
	if b is None:
		return fn(_require_list(a))
	if isinstance(a, TiList) and isinstance(b, TiList):
		if len(a) != len(b):
			raise ValueError(f"{fn.__name__}: dim mismatch ({len(a)} vs {len(b)})")
		return TiList([fn(x, y) for x, y in zip(a, b)])
	if isinstance(a, Number) and isinstance(b, Number):
		return fn(a, b)
	raise ValueError(f"{fn.__name__}: both args must be the same type (both numeric or both list)")


def max(a, b=None):
	return _minmax(builtins.max, a, b)


def min(a, b=None):
	return _minmax(builtins.min, a, b)


def median(lst, freqlist=None):
	_require_list(lst)
	if freqlist is None:
		s = sorted(lst)
		n = len(s)
		if n == 0:
			raise ValueError("median: empty list")
		mid = n // 2
		return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2

	_require_list(freqlist)
	if len(lst) != len(freqlist):
		raise ValueError("median: dim mismatch")
	pairs = sorted(zip(lst, freqlist), key=lambda p: p[0])
	total = builtins.sum(_require_int(f) for _, f in pairs)
	if total <= 0:
		raise ValueError("median: total frequency must be positive")

	def nth(k):
		count = 0
		for value, freq in pairs:
			count += int(freq)
			if k < count:
				return value

	if total % 2:
		return float(nth(total // 2))
	return (nth(total // 2 - 1) + nth(total // 2)) / 2


def mean(lst, freqlist=None):
	_require_list(lst)
	if freqlist is None:
		return builtins.sum(lst) / len(lst)
	_require_list(freqlist)
	return builtins.sum(x * w for x, w in zip(lst, freqlist)) / builtins.sum(freqlist)


@vectorized_with_matrix
def abs(a):
	return builtins.abs(a)


def det(mat):
	_require_matrix(mat)
	n = mat.rows
	if n == 0 or n != mat.cols:
		raise ValueError(f"det requires a non-empty square matrix, got {mat.rows}×{mat.cols}")
	m = [row.copy() for row in mat.inner]
	sign = 1.0
	for col in range(n):
		pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
		if pivot is None:
			return 0.0
		if pivot != col:
			m[col], m[pivot] = m[pivot], m[col]
			sign = -sign
		for row in range(col + 1, n):
			if m[row][col] != 0:
				f = m[row][col] / m[col][col]
				for j in range(col, n):
					m[row][j] -= f * m[col][j]
	return sign * math.prod(m[i][i] for i in range(n))


def identity(n):
	n = int(n)
	return TiMatrix([[1 if r == c else 0 for c in range(n)] for r in range(n)])


def transpose(m):
	_require_matrix(m)
	return TiMatrix([[m.inner[r][c] for r in range(m.rows)] for c in range(m.cols)])


def sum(lst):
	return builtins.sum(_require_list(lst))


def prod(lst):
	return math.prod(_require_list(lst))


# ── Transcendental functions ────────────────────────────────────────────────────

@vectorized
def pow10(a):
	return 10 ** a

@vectorized
def logbase(a, b):
	return cmath.log(a, b) if isinstance(a, complex) else math.log(a, b)


def _make_dispatch(name, real_fn, cpx_fn):
	def fn(x):
		return cpx_fn(x) if isinstance(x, complex) else real_fn(x)
	fn.__name__ = fn.__qualname__ = name
	return vectorized(fn)

for _name, _mf, _cf in [
	('sin',   math.sin,   cmath.sin),
	('asin',  math.asin,  cmath.asin),
	('cos',   math.cos,   cmath.cos),
	('acos',  math.acos,  cmath.acos),
	('tan',   math.tan,   cmath.tan),
	('atan',  math.atan,  cmath.atan),
	('sinh',  math.sinh,  cmath.sinh),
	('asinh', math.asinh, cmath.asinh),
	('cosh',  math.cosh,  cmath.cosh),
	('acosh', math.acosh, cmath.acosh),
	('tanh',  math.tanh,  cmath.tanh),
	('atanh', math.atanh, cmath.atanh),
	('ln',    math.log,   cmath.log),
	('exp',   math.exp,   cmath.exp),
	('log',   math.log10, cmath.log10),
]:
	globals()[_name] = _make_dispatch(_name, _mf, _cf)


# ── Integer / combinatorics ─────────────────────────────────────────────────────

def factorial(n):
	n = _require_int(n)
	if n < 0:
		raise ValueError("Argument to ! must be a non-negative integer")
	return float(math.factorial(n))

def ncr(n, r):
	return float(math.comb(_require_int(n), _require_int(r)))

def npr(n, r):
	return float(math.perm(_require_int(n), _require_int(r)))

@vectorized
def lcm(a, b):
	return float(math.lcm(_require_int(a), _require_int(b)))

@vectorized
def gcd(a, b):
	return float(math.gcd(_require_int(a), _require_int(b)))

@vectorized
def remainder(a, b):
	return float(_require_int(a) % _require_int(b))

# ── Random ──────────────────────────────────────────────────────────────────────

def randint(low, high, count=1):
	low, high = _require_int(low), _require_int(high)
	if count == 1:
		return float(random.randint(low, high))
	return TiList([float(random.randint(low, high)) for _ in range(_require_int(count))])


def randnorm(mu, sigma):
	return random.gauss(mu, sigma)


def randintnotrep(a, b, n=None):
	a, b = _require_int(a), _require_int(b)
	count = b - a + 1 if n is None else _require_int(n)
	return TiList([float(x) for x in random.sample(range(a, b + 1), count)])
