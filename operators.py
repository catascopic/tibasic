import cmath
import math

from decorators import vectorized
from functools import wraps
from numbers import Number
from errors import DataTypeError, DomainError, DivideByZeroError
from tiobjects import TiMatrix, require_real, require_int, require_num, require_matrix, py_int


# ── Pure comparison operators ────────────────────────────────────────────────
# Return 1.0/0.0; @vectorized handles TiList element-wise iteration.
# Type checking is delegated to the operands' magic methods:
#   - eq/ne: TiMatrix.__eq__ handles mat==mat (returns bool) and mat==other (raises DataTypeError)
#   - lt/gt/le/ge: TypeError from missing __lt__ etc. is converted to DataTypeError


def comparison(func):
	@wraps(func)
	def apply(a, b):
		try:
			return float(func(a, b))
		except TypeError:
			raise DataTypeError(f"'{func.__name__}' not supported between {type(a).__name__} and {type(b).__name__}")
	return vectorized(apply)

@comparison
def eq(a, b):
	if isinstance(a, complex) != isinstance(b, complex):
		raise DataTypeError("Cannot compare complex and real number with = or ≠")
	return a == b

@comparison
def ne(a, b):
	if isinstance(a, complex) != isinstance(b, complex):
		raise DataTypeError("Cannot compare complex and real number with = or ≠")
	return a != b

@comparison
def lt(a, b): return a < b

@comparison
def le(a, b): return a <= b

@comparison
def gt(a, b): return a > b

@comparison
def ge(a, b): return a >= b


# ── Pure arithmetic operators ────────────────────────────────────────────────
# TiList broadcasting is handled by TiList's magic methods (__add__, etc.).

def add(a, b):
	try:
		return a + b
	except TypeError:
		raise DataTypeError(f"cannot add {a} and {b}")

def sub(a, b):
	try:
		return a - b
	except TypeError:
		raise DataTypeError(f"cannot subtract {a} and {b}")

def mul(a, b):
	try:
		return a * b
	except TypeError:
		raise DataTypeError(f"cannot multiply {a} by {b}")

def div(a, b):
	try:
		return a / b
	except ZeroDivisionError:
		raise DivideByZeroError
	except TypeError:
		raise DataTypeError(f"cannot divide {a} by {b}")


# ── Pure logical operators ────────────────────────────────────────────────────

def logical(func):
	@wraps(func)
	def apply(a, b):
		return float(func(bool(require_real(a)), bool(require_real(b))))
	return vectorized(apply)

@logical
def and_(a, b): return a & b

@logical
def or_(a, b):  return a | b

@logical
def xor(a, b):  return a ^ b


# ── Pure combinatorics operators ─────────────────────────────────────────────

@vectorized
def ncr(n, r):
	try:
		return math.comb(py_int(n), py_int(r))
	except ValueError:
		raise DomainError(f"nCr: invalid arguments ({n}, {r})")

@vectorized
def npr(n, r):
	try:
		return math.perm(py_int(n), py_int(r))
	except ValueError:
		raise DomainError(f"nPr: invalid arguments ({n}, {r})")


# ── Env-aware binary operators ────────────────────────────────────────────────
# These may return complex numbers; the parser checks env.real_only after calling them.


def power(base, exp):
	"""^ operator — returns complex when the result is non-real."""
	try:
		result = base ** exp
	except ValueError:
		return cmath.exp(cmath.log(complex(base)) * exp)
	except TypeError:
		raise DataTypeError(f"{base}^{exp} not supported")
	return result


def xth_root(x, n):
	"""ˣ√ operator — returns complex when the result is non-real."""
	if isinstance(n, TiMatrix):
		raise DataTypeError(f"xth root not supported for matrix")
	try:
		return n ** (1 / x)
	except ValueError:
		return cmath.exp(cmath.log(n) / x)
	except TypeError:
		raise DataTypeError(f"{x}th root of {n} not supported")


# ── Postfix operators ────────────────────────────────────────────────────────

def inv(x):
	if isinstance(x, TiMatrix):
		return x.inv()
	try:
		return 1 / x
	except ZeroDivisionError:
		raise DivideByZeroError("Division by zero")

def transpose(mat):
	require_matrix(mat)
	return TiMatrix([[mat.data[r][c] for r in range(mat.rows)] for c in range(mat.cols)])

@vectorized
def factorial(n):
	"""! — factorial (via gamma function for non-negative reals)."""
	try:
		return math.gamma(require_real(n) + 1)
	except ValueError:
		raise DomainError(f"factorial: undefined for {n} (negative integer)")
