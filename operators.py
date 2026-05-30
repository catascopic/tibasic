import cmath
import math

from decorators import vectorized
from errors import DomainError
from tiobjects import TiMatrix, require_real, require_int, require_num, require_matrix


# ── Logical operators ────────────────────────────────────────────────────────

@vectorized
def and_(a, b):
	return int(bool(require_real(a)) and bool(require_real(b)))

@vectorized
def or_(a, b):
	return int(bool(require_real(a)) or bool(require_real(b)))

@vectorized
def xor(a, b):
	return int(bool(require_real(a)) ^ bool(require_real(b)))


# ── Postfix operators ────────────────────────────────────────────────────────

def inv(x):
	if isinstance(x, TiMatrix):
		return x.inv()
	return 1 / x

def transpose(mat):
	require_matrix(mat)
	return TiMatrix([[mat.data[r][c] for r in range(mat.rows)] for c in range(mat.cols)])

@vectorized
def factorial(n):
	try:
		return math.gamma(require_real(n) + 1)
	except ValueError:
		raise DomainError(f"factorial: undefined for {n} (negative integer)")


# ── Binary operators ─────────────────────────────────────────────────────────

@vectorized
def ncr(n, r):
	try:
		return math.comb(require_int(n), require_int(r))
	except ValueError:
		raise DomainError(f"nCr: invalid arguments ({n}, {r})")

@vectorized
def npr(n, r):
	try:
		return math.perm(require_int(n), require_int(r))
	except ValueError:
		raise DomainError(f"nPr: invalid arguments ({n}, {r})")

@vectorized
def xth_root(n, x):
	require_num(x)
	return cmath.exp(cmath.log(x) / n) if isinstance(x, complex) or x < 0 else x ** (1 / n)
