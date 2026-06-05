import cmath
import functools
import math

from decorators import vectorized, pure_op, op_vectorized
from numbers import Number
from errors import DataTypeError, DomainError, DivideByZeroError, NonRealAnsError
from modes import ComplexMode
from tiobjects import TiMatrix, require_real, require_int, require_num, require_matrix


# ── Pure comparison operators ────────────────────────────────────────────────
# Return 1.0/0.0; @vectorized handles TiList element-wise iteration.
# Type checking is delegated to the operands' magic methods:
#   - eq/ne: TiMatrix.__eq__ handles mat==mat (returns bool) and mat==other (raises DataTypeError)
#   - lt/gt/le/ge: TypeError from missing __lt__ etc. is converted to DataTypeError


def _type_err(op, a, b):
	raise DataTypeError(f"'{op}' not supported between {type(a).__name__} and {type(b).__name__}")


@pure_op
@vectorized
def eq(a, b): return float(a == b)

@pure_op
@vectorized
def ne(a, b): return float(a != b)

@pure_op
@vectorized
def lt(a, b):
	try: return float(a < b)
	except TypeError: _type_err('<', a, b)

@pure_op
@vectorized
def le(a, b):
	try: return float(a <= b)
	except TypeError: _type_err('≤', a, b)

@pure_op
@vectorized
def gt(a, b):
	try: return float(a > b)
	except TypeError: _type_err('>', a, b)

@pure_op
@vectorized
def ge(a, b):
	try: return float(a >= b)
	except TypeError: _type_err('≥', a, b)


# ── Pure arithmetic operators ────────────────────────────────────────────────
# TiList broadcasting is handled by TiList's magic methods (__add__, etc.).

@pure_op
def add(a, b):
	try:
		return a + b
	except TypeError:
		_type_err('+', a, b)

@pure_op
def sub(a, b):
	try:
		return a - b
	except TypeError:
		_type_err('-', a, b)

@pure_op
def mul(a, b):
	try:
		return a * b
	except TypeError:
		_type_err('*', a, b)

@pure_op
def div(a, b):
	try:
		return a / b
	except ZeroDivisionError:
		raise DivideByZeroError("Division by zero")
	except TypeError:
		_type_err('/', a, b)


# ── Pure logical operators ────────────────────────────────────────────────────

def logical(func):
	functools.wraps(func)
	def apply(a, b):
		return float(func(bool(require_real(a)), bool(require_real(b))))
	return pure_op(vectorized(apply))


@logical
def and_(a, b): return a & b

@logical
def or_(a, b):  return a | b

@logical
def xor(a, b):  return a ^ b


# ── Pure combinatorics operators ─────────────────────────────────────────────

@pure_op
@vectorized
def ncr(n, r):
	try:
		return math.comb(require_int(n), require_int(r))
	except ValueError:
		raise DomainError(f"nCr: invalid arguments ({n}, {r})")

@pure_op
@vectorized
def npr(n, r):
	try:
		return math.perm(require_int(n), require_int(r))
	except ValueError:
		raise DomainError(f"nPr: invalid arguments ({n}, {r})")


# ── Env-aware binary operators ────────────────────────────────────────────────
# These need env to check ComplexMode; they are NOT wrapped with @pure_op.

@op_vectorized
def _power(base, exp, env):
	try:
		result = base ** exp
	except ValueError:
		if env.real_only:
			raise NonRealAnsError("Non-real result")
		return cmath.exp(cmath.log(complex(base)) * exp)
	except TypeError:
		_type_err('^', base, exp)
	return result

def power(base, exp, env):
	"""^ operator — checks ComplexMode before returning a non-real result."""
	if isinstance(base, TiMatrix) and not isinstance(exp, Number):
		_type_err('^', base, exp)
	return _power(base, exp, env)


@op_vectorized
def xth_root(n, x, env):
	"""ˣ√ operator — checks ComplexMode before returning a non-real result."""
	require_num(x)
	try:
		return x ** (1 / n)
	except ValueError:
		if env.real_only:
			raise NonRealAnsError("Non-real result")
		return cmath.exp(cmath.log(x) / n)


# ── Postfix operators ────────────────────────────────────────────────────────

def inv(x):
	"""¹ — multiplicative inverse or matrix inverse."""
	if isinstance(x, TiMatrix):
		return x.inv()
	try:
		return 1 / x
	except ZeroDivisionError:
		raise DivideByZeroError("Division by zero")

def transpose(mat):
	"""ᵀ — matrix transpose."""
	require_matrix(mat)
	return TiMatrix([[mat.data[r][c] for r in range(mat.rows)] for c in range(mat.cols)])

@vectorized
def factorial(n):
	"""! — factorial (via gamma function for non-negative reals)."""
	try:
		return math.gamma(require_real(n) + 1)
	except ValueError:
		raise DomainError(f"factorial: undefined for {n} (negative integer)")
