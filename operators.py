import cmath
import math

from decorators import vectorized, pure_op, op_vectorized
from errors import DomainError, NonRealAnsError
from modes import ComplexMode
from tiobjects import TiList, TiMatrix, require_real, require_int, require_num, require_matrix


# ── Pure comparison operators ────────────────────────────────────────────────
# TiList broadcasting is handled by TiList's magic methods (__eq__, __lt__, etc.),
# which already produce float elements.  _as_float converts scalar bool → float.

def _as_float(x):
	"""Convert a scalar comparison result (bool) to float; pass TiList through."""
	return x if isinstance(x, TiList) else float(x)

@pure_op
def eq(a, b): return _as_float(a == b)

@pure_op
def ne(a, b): return _as_float(a != b)

@pure_op
def lt(a, b): return _as_float(a < b)

@pure_op
def le(a, b): return _as_float(a <= b)

@pure_op
def gt(a, b): return _as_float(a > b)

@pure_op
def ge(a, b): return _as_float(a >= b)


# ── Pure arithmetic operators ────────────────────────────────────────────────
# TiList broadcasting is handled by TiList's magic methods (__add__, etc.).

@pure_op
def add(a, b): return a + b

@pure_op
def sub(a, b): return a - b

@pure_op
def mul(a, b): return a * b

@pure_op
def div(a, b): return a / b


# ── Pure logical operators ────────────────────────────────────────────────────

@pure_op
@vectorized
def and_(a, b): return int(bool(require_real(a)) and bool(require_real(b)))

@pure_op
@vectorized
def or_(a, b): return int(bool(require_real(a)) or bool(require_real(b)))

@pure_op
@vectorized
def xor(a, b): return int(bool(require_real(a)) ^ bool(require_real(b)))


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
def power(base, exp, env):
	"""^ operator — checks ComplexMode before returning a non-real result."""
	if isinstance(base, TiMatrix):
		return base ** exp
	try:
		result = base ** exp
	except ValueError:
		# Negative base with fractional exponent → complex result.
		result = cmath.exp(cmath.log(complex(base)) * exp)
	if isinstance(result, complex) and env.complex_mode is ComplexMode.REAL:
		raise NonRealAnsError("Non-real result")
	return result


@op_vectorized
def xth_root(n, x, env):
	"""ˣ√ operator — checks ComplexMode before returning a non-real result."""
	require_num(x)
	if isinstance(x, complex) or x < 0:
		if env.complex_mode is ComplexMode.REAL:
			raise NonRealAnsError("Non-real result")
		return cmath.exp(cmath.log(x) / n)
	return x ** (1 / n)


# ── Postfix operators ────────────────────────────────────────────────────────

def inv(x):
	"""¹ — multiplicative inverse or matrix inverse."""
	if isinstance(x, TiMatrix):
		return x.inv()
	return 1 / x

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
