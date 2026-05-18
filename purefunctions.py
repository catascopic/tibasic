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


def _repr_num(value):
	int_value = int(value)
	return repr(int_value if int_value == value else value


def _require_type(value, tp):
	if not isinstance(value, tp):
		raise ValueError(f"Invalid type: {value}; requred: {tp}")
	return value

def _require_num(value):
	return(value, Number)

def _require_list(value):
	return(value, TiList)

def _require_str(value):
	return(value, str)


def and_(a, b):
	return float(bool(a) & bool(b))

def or_(a, b):
	return float(bool(a) | bool(b))

def xor(a, b):
	return float(bool(a) ^ bool(b))


class TiList:

	def __init__(self, data=()):
		self.inner = list(data)
	
	def __getitem__(self, index):
		if index != int(index) or not(1 <= index <= len(self)):
			raise IndexError(f"{index=}")
		return self.inner[int(index) - 1]

	def __setitem__(self, index, value):
		if index == len(self) + 1:
			self.inner.append(value)
		elif index != int(index) or not(1 <= index <= len(self)):
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
			return TiList(op(a, b) for a, b in zip(self, other, strict=True))
		return TiList(op(a, other) for a in self)
	return list_op


for name, op in [
	('__add__', operator.add),
	('__radd__', operator.add),
	('__sub__', operator.sub),
	('__rsub__', lambda a, b: b - a),
	('__mul__', operator.mul),
	('__rmul__', operator.mul),
	# Can I use Fractions here?
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
	setattr(TiList, name, make_binary_list_op(op))


def make_unary_list_op(op):
	def list_op(self, op=op):
		return TiList(op(a) for a in self)
	return list_op


for name, op in [
	('__neg__', operator.neg),
	('__abs__', abs),
	('__trunc__', math.trunc),
]:
	setattr(TiList, name, make_unary_list_op(op))


class TiMatrix:

	def __init__(self, data=None):
		self.inner = [] if data is None else data
	
	def __getitem__(self, index):
		row_index, col_index = index
		if row_index != int(row_index) or not(1 <= row_index <= len(self.inner)):
			raise IndexError(f"{row_index=}")
		col = self.inner[int(index) - 1]
		if col_index != int(col_index) or not(1 <= col_index <= len(col)):
			raise IndexError(f"{col_index=}")
		return col[col_index]
	
	def __setitem__(self, index, value):
		pass  # TODO
	
	def __len__(self):
		rows = len(self.inner)
		return TiList[rows, len(self.inner[0]) if rows else 0]
	
	def transform(self, func):
		return TiMatrix([
			[func(i) for i in row]
			for row in inner
		])

	def __mul__(self, other):
		if isinstance(other, TiMatrix):
			return self.inner @ other.inner
		return self.transform(lambda i: i * other)
		
	def __rmul__(self, other):
		if isinstance(other, TiMatrix):
			return other.inner @ self.inner
		return self.transform(lambda i: other * i)

	def __repr__(self):
		raise NotImplementedError
	
	def set_dim(self, value):
		pass  # TODO

	def copy(self):
		return TiMatrix([row.copy() for row in self.inner])


def handle_complex(func):
	@wraps(func)
	def apply(a):
		return complex(func(a.real), func(a.imag)) if isinstance(a, complex) else func(a)
	return apply


def vectorized(func):
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
			return TiList(func(*v) for v in zip(*vec))
		raise ValueError(f"Dim mismatch: {len_check}")
		
	return apply


def dim(value):
	if not isinstance(value, (TiList, TiMatrix)):
		raise ValueError(f"Invalid type: {value}; requred: list or matrix")
	return len(value)


@vectorized
def not_(num):
	return float(num == 0)


@vectorized 
@handle_complex
def i_part(num):
	return math.trunc(num)


@vectorized
@handle_complex
def int_(num):
	return math.floor(num)


@vectorized
@handle_complex
def f_part(num):
	return num - math.trunc(num)


@vectorized
@handle_complex
def sqrt(num):
	return math.sqrt(num)


@vectorized
@handle_complex
def cbrt(a):
	return math.cbrt(a)


def cum_sum(lst):
	return TiList(accumulate(_require_list(lst)))


def delta_list(lst):
	return TiList([b - a for a, b in pairwise(_require_list(lst))])


def augment(lst1, lst2):
	if isinstance(lst1, TiList) and if isinstance(lst2, TiList):
		return TiList(chain(lst1, lst2))
	if isinstance(lst1, TiMatrix) and if isinstance(lst2, TiMatrix):
		pass  # TODO
	raise ValueError(f"Invalid type: {value}; requred: list or matrix")


@vectorized
def real(num):
	return num.real if isinstance(num, complex) else num


@vectorized
def imag(num):
	return num.imag if isinstance(num, complex) else 0


def sort_a(lst, *dep, *, reverse=False):
	inner = _require_list(lst).inner
	indices = sorted(range(len(inner)), key=lambda i: inner[i], reverse=reverse)
	lst.inner = [inner[i] for i in indices]
	for d in dep:
		d.inner = [d.inner[i] for i in indices]


def sort_d(lst, *dep):
	sortA(lst, *dep, reverse=True)


def fill(lst, num):
	_require_num(num)
	if isinstance(lst, TiList):
		inner = _require_list(lst).inner
		for i in len(inner):
			inner[i] = num
	elif isinstance(lst, TiMatrix):
		pass  # TODO
	else:
		raise ValueError(f"Invalid type: {value}; requred: list or matrix")


def in_string(value, t):
	return _require_str(value).find(t) + 1


def length(value):
	return len(_require_str(value))


def sub(value, start, length):
	if isinstance(lst, Number):
		return value / 100
	if isinstance(lst, str):
		if length < 1:
			raise ValueError(length)
		if not(1 <= start <= len(value) - length + 1):
			raise ValueError(value, start, length)
		return value[start - 1 : start - 1 + length]
	raise ValueError(f"Invalid type: {value}; requred: string or number")

@vectorized
def round(a, b=9):
	return builtins.round(a)

@vectorized
def max(a, b):
	return builtins.max(a, b)


@vectorized
def min(*a):
	return builtins.min(a, b)


def median(lst, freqlist=None):
	_require_list(lst)
	if freqlist is None:
		return sorted(a)[len(a) // 2]
	_require_list(freqlist)
	# TODO


def mean(lst, freqlist=None):
	_require_list(lst)
	if freqlist is None:
		return builtins.sum(lst) / len(lst)
	_require_list(freqlist)
	return builtins.sum(x * w for x, w in zip(lst, freqlist)) / builtins.sum(freqlist)


@vectorized
def abs(a):
	return builtins.abs(a)


def det(mat):
	_require_matrix(mat)
	# TODO


def identity(n):
	n = _int(n)
	return TiMatrix([[1 if r == c else 0 for c in range(n)] for r in range(n)])


def sum(a):
	return builtins.sum(a)


def prod(a):
	return math.prod(a)


@vectorized
def ln(a):
	return cmath.log(a)

@vectorized
def exp(a):
	return cmath.exp(a)

@vectorized
def log(a):
	return cmath.log10(a)

@vectorized
def pow10(a):
	return 10 ** a

@vectorized
def sin(x): 
	return math.sin(x)

@vectorized
def asin(x):
	return math.asin(x)

@vectorized
def cos(x):
	return math.cos(x)

@vectorized
def acos(x):
	return math.acos(x)

@vectorized
def tan(x):
	return math.tan(x)

@vectorized
def atan(x):
	return math.atan(x)

@vectorized
def sinh(x):
	return math.sinh(x)

@vectorized
def asinh(x):
	return math.asinh(x)

@vectorized
def cosh(x):
	return math.cosh(x)

@vectorized
def acosh(x):
	return math.acosh(x)

@vectorized
def tanh(x):
	return math.tanh(x)

@vectorized
def atanh(x):
	return math.atanh(x)

@vectorized
def lcm(a, b):
	return math.lcm(_require_int(a), _require_int(b))

@vectorized
def gcd(a, b):
	return math.gcd(_require_int(a), _require_int(b))

def randint(low, high, count=1):
	if count == 1:
		return random.randint(low, high)
	return [random.randint(low, high) for _ in range(_int(count))]

def randnorm(mu, sigma):
	return random.gauss(mu, sigma)

@vectorized
def conj(a):
	return complex(a.real, -a.imag)

def angle(a):
	return cmath.phase(a)

@vectorized
def remainder(a, b):
	return float(_int(a) % _int(b))

@vectorized
def logbase(a, b):
	return math.log(a) / math.log(b)

def randintnotrep(a, b, n=None):
	return random.sample(range(_int(a), _int(b) + 1), _int(b - a + 1) if n is None else _int(n))
