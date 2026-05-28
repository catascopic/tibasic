"""Test suite for the TI-BASIC interpreter."""

import math
import pytest
from pytest import approx

import purefunctions as pf
from environment import Environment
from errors import (
	TiSyntaxError, IllegalNestError, DomainError, DimMismatchError,
	StatError, IncrementError, DataTypeError, InvalidDimError, ArgumentError,
)
from parser import parse_line
import tokens
from tokens import (
	ALL_TOKENS, Token, get_token,
	XTH_ROOT, INV, SQ, TRANSPOSE,
	LISTS, MATRICES, STRINGS, VAR_A, VAR_B,
)
from tiobjects import TiList, TiMatrix, TiString

L1 = LISTS[0]
L2 = LISTS[1]
MAT_A = MATRICES[0]
STR_1 = STRINGS[0]

# ── Helpers ───────────────────────────────────────────────────────────────────


lookup = {_t.text: _t for _t in ALL_TOKENS}
lookup['~'] = tokens.NEG
lookup['@'] = tokens.STORE
lookup['e'] = tokens.SCI_E
lookup['$'] = tokens.LIST_PREFIX
lookup['i'] = tokens.IMAG_I
for i, ls in enumerate(tokens.LISTS, start=1):
	lookup[f"L{i}"] = ls
for name, value in vars(tokens).items():
	if isinstance(value, Token):
		lookup[name] = value


def _iter_chars(obj):
	for c in str(obj):
		yield lookup[c]

def _iter_tokens(line):
	for obj in line:
		if isinstance(obj, Token):
			yield obj
		elif isinstance(obj, (int, float)):
			yield from _iter_chars(str(obj))
		elif isinstance(obj, str):
			for seg in obj.split():
				try:
					yield lookup[seg]
				except KeyError:
					yield from _iter_chars(seg)
		else:
			yield get_token(obj)

def toks(*line) -> list[Token]:
	"""Build a token list.
	- Token: used directly
	- number: tokenised digit-by-digit
	- str in token table: that token
	- other str: each character looked up individually
	"""
	return list(_iter_tokens(line))

def calc(*items, env: Environment | None = None):
	"""Evaluate a token sequence and return Ans."""
	if env is None:
		env = Environment()
	parse_line(toks(*items), env)
	return env.ans


@pytest.fixture
def env():
	return Environment()

@pytest.fixture
def deg():
	e = Environment()
	e.angle_mode = 'DEG'
	return e


def approx_mat(matrix, **kwargs):
	return [pytest.approx(row, **kwargs) for row in matrix]


# ── Arithmetic ────────────────────────────────────────────────────────────────

class TestArithmetic:
	def test_add(self):          assert calc('1+2') == 3
	def test_sub(self):          assert calc('5-3') == 2
	def test_mul(self):          assert calc('3*4') == 12
	def test_div(self):          assert calc('7/2') == 3.5
	def test_pow(self):          assert calc('2^10') == 1024
	def test_negation(self):     assert calc('~5') == -5
	def test_sq_postfix(self):   assert calc(7, SQ) == 49
	def test_xroot(self):        assert calc(4, XTH_ROOT, 256) == approx(4)
	def test_sci_e(self):        assert calc('1e3') == 1000
	def test_implicit_mul(self): assert calc('2(3+4)') == 14

	def test_precedence_mul_over_add(self):
		assert calc('2+3*4') == 14

	def test_precedence_parens(self):
		assert calc('(2+3)*4') == 20

	def test_pow_right_assoc(self):
		# 2^3^2 = 2^(3^2) = 2^9 = 512 (right-associative)
		assert calc('2^3^2') == 512


# ── Scientific notation (ᴇ) ───────────────────────────────────────────────────

class TestSciE:
	# ── Positive cases ────────────────────────────────────────────────────────

	def test_infix_basic(self):
		# 1ᴇ3 = 1000
		assert calc('1e3') == 1000

	def test_prefix_basic(self):
		# ᴇ3 = 10^3 = 1000  (no left operand → implicit 1)
		assert calc('e3') == 1000

	def test_infix_zero_exp(self):
		# 5ᴇ0 = 5
		assert calc('5e0') == 5

	def test_infix_decimal_exp(self):
		# 1ᴇ1.5 = 10^1.5
		assert calc('1e1.5') == approx(10 ** 1.5)

	def test_infix_multi_digit_exp(self):
		# 2ᴇ10 = 2048 (not 2*(10^1)*0 or anything weird)
		assert calc('2e10') == approx(2 * 10 ** 10)

	def test_dms_e_in_minutes(self):
		# 1°2ᴇ2'3" is valid: minutes may use ᴇ notation → 1 + 200/60 + 3/3600
		assert calc("1°2e2'3\"") == approx(1 + 200 / 60 + 3 / 3600)

	def test_dms_sci_min(self):
		assert calc("1e1°30'") == approx(10.5)
	
	def test_dms_sci_min(self):
		assert calc('1e~1°2\'3"') == approx(.1341666666)
	
	def test_leading_sci_dms(self):
		assert calc("e1°30'") == approx(10.5)

	def test_expr_sci_dms(self):
		assert calc("(1+1)e1°30'") == approx(21)
	
	def test_dms_expr_error(self):
		with pytest.raises(TiSyntaxError):
			calc('(1e~1)°2\'3"')

	def test_precedence_over_add(self):
		# 2 + 3ᴇ2 = 2 + 300 = 302  (ᴇ binds tighter than +)
		assert calc('2+3e2') == 302

	def test_precedence_over_mul(self):
		# 2 * 3ᴇ2 = 2 * 300 = 600  (ᴇ binds tighter than *)
		assert calc('2*3e2') == 600

	def test_neg_before_sci_e(self):
		# ~1ᴇ3 = ~1000  (negation of the whole scientific-notation number)
		assert calc('~1e3') == -1000

	def test_pow_rhs_is_sci_e(self):
		# 2^3ᴇ2: ᴇ binds tighter than ^, so exponent is 3ᴇ2=300 → 2^300
		assert calc('2^3e2') == approx(2 ** 300)

	def test_in_larger_expression(self):
		# (1ᴇ3 + 1ᴇ2) = 1100
		assert calc('(1e3+1e2)') == 1100

	# ── Negative cases ────────────────────────────────────────────────────────

	def test_rejects_paren_expr(self):
		# 1ᴇ(3) — parenthesised expression is not a numeric literal
		with pytest.raises(TiSyntaxError):
			calc('1e(3)')

	def test_rejects_variable(self):
		# 1ᴇA — variable is not a numeric literal
		with pytest.raises(TiSyntaxError):
			calc('1eA')

	def test_rejects_expression_rhs(self):
		# 1ᴇ2+1 must parse as (1ᴇ2)+1 = 101, not 1ᴇ(2+1) = 1000
		# (confirms the RHS stops at the literal boundary)
		assert calc('1e2+1') == 101

	def test_rejects_ans_as_exponent(self):
		# 1ᴇAns — Ans is not a numeric literal
		with pytest.raises(TiSyntaxError):
			calc('1e', 'Ans')
	
	def test_rejects_double_sci(self):
		with pytest.raises(TiSyntaxError):
			calc('1e1e1')
	
	def test_multi_degrees(self, deg):
		assert calc("1e2°2'°2", env=deg) == approx(200 + (2/30))
			
	def test_infix_negative_exp(self):
		# 1ᴇ~3 = 0.001
		assert calc('1e~3') == approx(0.001)

	def test_prefix_negative_exp(self):
		# ᴇ~3 = 10^~3 = 0.001
		assert calc('e~3') == approx(0.001)

	def test_negative_exp_decimal(self):
		# 1ᴇ~1.5 = 10^~1.5
		assert calc('1e~1.5') == approx(10 ** -1.5)

	def test_negative_exp_in_expression(self):
		# 2 + 3ᴇ~2 = 2 + 0.03 = 2.03
		assert calc('2+3e~2') == approx(2.03)

	def test_neg_literal_neg_exp(self):
		# ~2ᴇ~3 = ~0.002
		assert calc('~2e~3') == approx(-0.002)

	def test_rejects_double_neg_exp(self):
		# 1ᴇ~~3 — two negations is actually a valid literal
		assert calc('e~~3') == approx(1000)


# ── Numeric functions ─────────────────────────────────────────────────────────

class TestNumericFunctions:
	def test_abs_pos(self):       assert pf.abs(5) == 5
	def test_abs_neg(self):       assert pf.abs(-5) == 5
	def test_abs_complex(self):   assert pf.abs(3+4j) == approx(5)
	def test_round_2dp(self):     assert pf.round(3.14159, 2) == approx(3.14)
	def test_round_default(self): assert pf.round(1/3) == approx(1/3)
	def test_i_part_pos(self):    assert pf.i_part(3.9) == 3
	def test_i_part_neg(self):    assert pf.i_part(-3.9) == -3   # toward zero
	def test_f_part_pos(self):    assert pf.f_part(3.7) == approx(0.7)
	def test_f_part_neg(self):    assert pf.f_part(-3.7) == approx(-0.7)
	def test_int_floor_pos(self): assert pf.int_(3.9) == 3
	def test_int_floor_neg(self): assert pf.int_(-3.1) == -4     # floor, not truncate
	def test_sqrt(self):          assert pf.sqrt(9) == approx(3)
	def test_sqrt_negative(self): assert pf.sqrt(-1) == approx(1j)
	def test_cbrt(self):          assert pf.cbrt(8) == approx(2)
	def test_ln(self):            assert pf.ln(math.e) == approx(1)
	def test_log(self):           assert pf.log(100) == approx(2)
	def test_exp(self):           assert pf.exp(0) == approx(1)
	def test_pow10(self):         assert pf.pow10(3) == approx(1000)
	def test_not_false(self):     assert pf.not_(0) == 1
	def test_not_true(self):      assert pf.not_(5) == 0


class TestTrig:
	def test_sin(self):   assert pf.sin(math.pi / 6) == approx(0.5)
	def test_cos(self):   assert pf.cos(0) == approx(1)
	def test_tan(self):   assert pf.tan(math.pi / 4) == approx(1)
	def test_asin(self):  assert pf.asin(0.5) == approx(math.pi / 6)
	def test_acos(self):  assert pf.acos(1) == approx(0)
	def test_atan(self):  assert pf.atan(1) == approx(math.pi / 4)
	def test_sinh(self):  assert pf.sinh(0) == approx(0)
	def test_cosh(self):  assert pf.cosh(0) == approx(1)
	def test_tanh(self):  assert pf.tanh(0) == approx(0)
	def test_asinh(self): assert pf.asinh(0) == approx(0)
	def test_acosh(self): assert pf.acosh(1) == approx(0)
	def test_atanh(self): assert pf.atanh(0) == approx(0)


class TestLogic:
	def test_and_tt(self):  assert pf.and_(1, 1) == 1
	def test_and_tf(self):  assert pf.and_(1, 0) == 0
	def test_or_ff(self):   assert pf.or_(0, 0) == 0
	def test_or_tf(self):   assert pf.or_(1, 0) == 1
	def test_xor_tt(self):  assert pf.xor(1, 1) == 0
	def test_xor_tf(self):  assert pf.xor(1, 0) == 1


# ── Combinatorics ─────────────────────────────────────────────────────────────

class TestCombinatorics:
	def test_factorial(self):  assert pf.factorial(5) == 120
	def test_ncr(self):        assert pf.ncr(10, 3) == 120
	def test_npr(self):        assert pf.npr(5, 3) == 60
	def test_lcm(self):        assert pf.lcm(12, 8) == 24
	def test_gcd(self):        assert pf.gcd(12, 8) == 4
	def test_remainder(self):  assert pf.remainder(17, 5) == 2

	# Through the parser (NPR/NCR are binary operators; FACT is postfix)
	def test_fact_parser(self): assert calc('5!') == 120
	def test_npr_parser(self):  assert calc(5, 'nPr', 3) == 60
	def test_ncr_parser(self):  assert calc(5, 'nCr', 3) == 10


# ── List operations ───────────────────────────────────────────────────────────

class TestListOperations:
	def test_augment(self):
		assert list(pf.augment(TiList([1, 2]), TiList([3, 4]))) == [1, 2, 3, 4]

	def test_cum_sum(self):
		assert list(pf.cum_sum(TiList([1, 2, 3, 4]))) == [1, 3, 6, 10]

	def test_delta_list(self):
		assert list(pf.delta_list(TiList([1, 3, 6, 10]))) == [2, 3, 4]

	def test_sum_full(self):
		assert pf.sum(TiList([1, 2, 3, 4, 5])) == 15

	def test_sum_partial(self):
		assert pf.sum(TiList([1, 2, 3, 4, 5]), 2, 4) == 9

	def test_prod(self):
		assert pf.prod(TiList([1, 2, 3, 4])) == 24

	def test_mean(self):
		assert pf.mean(TiList([1, 2, 3, 4, 5])) == approx(3)

	def test_median_odd(self):
		assert pf.median(TiList([3, 1, 4, 1, 5])) == 3

	def test_median_even(self):
		assert pf.median(TiList([1, 2, 3, 4])) == 2.5

	def test_max_list(self):
		assert pf.max(TiList([3, 1, 4, 1, 5, 9])) == 9

	def test_min_list(self):
		assert pf.min(TiList([3, 1, 4, 1, 5, 9])) == 1

	def test_max_two_scalars(self):
		assert pf.max(3, 7) == 7

	def test_variance(self):
		# known result: sum of squared deviations / (n-1)
		assert pf.variance(TiList([2, 4, 4, 4, 5, 5, 7, 9])) == approx(32 / 7)

	def test_stddev(self):
		assert pf.stddev(TiList([2, 4, 4, 4, 5, 5, 7, 9])) == approx(math.sqrt(32 / 7))

	def test_dim_list(self):
		assert pf.dim(TiList([1, 2, 3])) == 3

	def test_vectorized_add(self):
		assert list(TiList([1, 2, 3]) + TiList([4, 5, 6])) == [5, 7, 9]

	def test_vectorized_scalar(self):
		assert list(TiList([2, 4, 6]) / 2) == [1, 2, 3]


# ── Empty-list behaviour ──────────────────────────────────────────────────────

class TestEmptyList:
	"""
	{} is a syntax error.  An empty TiList (created programmatically) causes
	InvalidDimError in every aggregate function except dim(), which returns 0.
	0→dim(L₁) is also rejected with InvalidDimError.
	"""

	_empty = TiList([])   # created directly, not via the parser

	# ── Parser rejects {} ─────────────────────────────────────────────────────

	def test_empty_literal_raises(self):
		with pytest.raises(TiSyntaxError):
			calc('{}')

	def test_empty_literal_unclosed_raises(self):
		# { with nothing before implicit close is equally invalid
		with pytest.raises(TiSyntaxError):
			calc('{')

	# ── dim() is the one exception ────────────────────────────────────────────

	def test_dim_empty_returns_zero(self):
		assert pf.dim(self._empty) == 0

	def test_store_dim_zero(self, env):
		calc('{1,2,3@', 'L1', env=env)
		calc('0@', 'dim(', 'L1', env=env)
		assert env.lists[0].data == []

	# ── Aggregate functions ───────────────────────────────────────────────────

	def test_sum_empty(self):
		with pytest.raises(InvalidDimError):
			pf.sum(self._empty)

	def test_prod_empty(self):
		with pytest.raises(InvalidDimError):
			pf.prod(self._empty)

	def test_mean_empty(self):
		with pytest.raises(InvalidDimError):
			pf.mean(self._empty)

	def test_median_empty(self):
		with pytest.raises(InvalidDimError):
			pf.median(self._empty)

	def test_max_empty(self):
		with pytest.raises(InvalidDimError):
			pf.max(self._empty)

	def test_min_empty(self):
		with pytest.raises(InvalidDimError):
			pf.min(self._empty)

	def test_variance_empty(self):
		with pytest.raises(InvalidDimError):
			pf.variance(self._empty)

	def test_stddev_empty(self):
		with pytest.raises(InvalidDimError):
			pf.stddev(self._empty)

	def test_cum_sum_empty(self):
		with pytest.raises(InvalidDimError):
			pf.cum_sum(self._empty)

	def test_delta_list_empty(self):
		with pytest.raises(InvalidDimError):
			pf.delta_list(self._empty)

	# ── augment ───────────────────────────────────────────────────────────────

	def test_augment_empty_left(self):
		with pytest.raises(InvalidDimError):
			pf.augment(self._empty, TiList([1, 2]))

	def test_augment_empty_right(self):
		with pytest.raises(InvalidDimError):
			pf.augment(TiList([1, 2]), self._empty)

	def test_augment_both_empty(self):
		with pytest.raises(InvalidDimError):
			pf.augment(self._empty, self._empty)


# ── Stat functions with freq_list ─────────────────────────────────────────────

class TestStatWithFreqList:
	"""mean, median, variance, stddev all accept an optional freq_list second arg."""

	# ── mean ──────────────────────────────────────────────────────────────────

	def test_mean_uniform(self):
		# Uniform weights → same result as plain mean
		assert pf.mean(TiList([1, 2, 3]), TiList([1, 1, 1])) == approx(2.0)

	def test_mean_weighted(self):
		# [0,0,0,10] → mean = 10/4 = 2.5
		assert pf.mean(TiList([0, 10]), TiList([3, 1])) == approx(2.5)

	def test_mean_integer_counts(self):
		# [1,1,1,2,3,3] → mean = (3+2+6)/6 = 11/6
		assert pf.mean(TiList([1, 2, 3]), TiList([3, 1, 2])) == approx(11 / 6)

	# ── median ────────────────────────────────────────────────────────────────

	def test_median_odd_total(self):
		# Expanded: [10,10,20,30,30] → middle element is 20
		assert pf.median(TiList([10, 20, 30]), TiList([2, 1, 2])) == 20

	def test_median_even_total(self):
		# Expanded: [10,10,30,30] → (10+30)/2 = 20
		assert pf.median(TiList([10, 30]), TiList([2, 2])) == approx(20.0)

	def test_median_unsorted_input(self):
		# Must sort by value: {3:1,1:2,2:1} → [1,1,2,3] → (1+2)/2 = 1.5
		assert pf.median(TiList([3, 1, 2]), TiList([1, 2, 1])) == approx(1.5)

	def test_median_uniform_matches_plain(self):
		plain    = pf.median(TiList([1, 2, 3, 4, 5]))
		weighted = pf.median(TiList([1, 2, 3, 4, 5]), TiList([1, 1, 1, 1, 1]))
		assert weighted == approx(plain)

	def test_median_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			pf.median(TiList([1, 2, 3]), TiList([1, 1]))

	# ── variance ──────────────────────────────────────────────────────────────

	def test_variance_weighted(self):
		# mean=1; 3*(0-1)² + 1*(4-1)² = 3+9=12; 12/(4-1) = 4.0
		assert pf.variance(TiList([0, 4]), TiList([3, 1])) == approx(4.0)

	def test_variance_uniform_matches_plain(self):
		plain    = pf.variance(TiList([2, 4, 6]))
		weighted = pf.variance(TiList([2, 4, 6]), TiList([1, 1, 1]))
		assert weighted == approx(plain)

	def test_variance_total_freq_le_one(self):
		# total freq = 1 → denominator (n-1) = 0
		with pytest.raises(StatError, match="total frequency"):
			pf.variance(TiList([5]), TiList([1]))

	def test_variance_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			pf.variance(TiList([1, 2, 3]), TiList([1, 1]))

	# ── stddev ────────────────────────────────────────────────────────────────

	def test_stddev_weighted(self):
		assert pf.stddev(TiList([0, 4]), TiList([3, 1])) == approx(2.0)

	def test_stddev_uniform_matches_plain(self):
		plain    = pf.stddev(TiList([2, 4, 4, 4, 5, 5, 7, 9]))
		weighted = pf.stddev(TiList([2, 4, 4, 4, 5, 5, 7, 9]), TiList([1, 1, 1, 1, 1, 1, 1, 1]))
		assert weighted == approx(plain)

	def test_stddev_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			pf.stddev(TiList([1, 2]), TiList([1]))


# ── Matrix operations ─────────────────────────────────────────────────────────

class TestMatrixOperations:
	def test_det_2x2(self):
		assert pf.det(TiMatrix([[1, 2], [3, 4]])) == approx(-2)

	def test_det_identity(self):
		assert pf.det(pf.identity(4)) == approx(1)

	def test_identity(self):
		mat = pf.identity(3)
		assert mat.data == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

	def test_transpose(self):
		result = pf.transpose(TiMatrix([[1, 2, 3], [4, 5, 6]]))
		assert result.data == [[1, 4], [2, 5], [3, 6]]

	def test_matmul(self):
		a = TiMatrix([[1, 2], [3, 4]])
		b = TiMatrix([[5, 6], [7, 8]])
		assert (a * b).data == [[19, 22], [43, 50]]

	def test_inv_roundtrip(self):
		mat = TiMatrix([[1, 2], [3, 4]])
		product = mat * mat.inv()
		assert product.data == approx_mat([[1, 0], [0, 1]])

	def test_dim_matrix(self):
		assert list(pf.dim(TiMatrix([[1, 2, 3], [4, 5, 6]]))) == [2, 3]

	def test_augment_matrix(self):
		a = TiMatrix([[1, 2], [3, 4]])
		b = TiMatrix([[5], [6]])
		assert pf.augment(a, b).data == [[1, 2, 5], [3, 4, 6]]

	def test_rref_solve(self):
		# 2x + y = 5, x - y = 1  →  x=2, y=1
		result = pf.rref(TiMatrix([[2, 1, 5], [1, -1, 1]]))
		assert result.data == approx_mat([[1, 0, 2], [0, 1, 1]])


class TestMatrixRowOps:
	def setup_method(self):
		self.mat = TiMatrix([[1, 2], [3, 4], [5, 6]])

	def test_row_swap(self):
		result = pf.row_swap(self.mat, 1, 3)
		assert result.data == [[5, 6], [3, 4], [1, 2]]
		assert self.mat.data == [[1, 2], [3, 4], [5, 6]]  # original unchanged

	def test_row_plus(self):
		result = pf.row_plus(self.mat, 1, 2)
		assert result.data == [[1, 2], [4, 6], [5, 6]]

	def test_times_row(self):
		result = pf.times_row(3, self.mat, 1)
		assert result.data == [[3, 6], [3, 4], [5, 6]]

	def test_times_row_plus(self):
		result = pf.times_row_plus(2, self.mat, 1, 2)
		assert result.data == [[1, 2], [5, 8], [5, 6]]


# ── Matr►list( / List►matr( ───────────────────────────────────────────────────

class TestMatrToList:
	def test_single_column_first(self, env):
		# Matr►list([A], 1, L1) extracts column 1 into L1
		calc('Matr►list(', '[[1,2][3,4][5,6]],1,', 'L1', env=env)
		assert L1.variable.get(env).data == [1, 3, 5]

	def test_single_column_second(self, env):
		# Matr►list([A], 2, L1) extracts column 2
		calc('Matr►list( [[1,2][3,4][5,6]],2,', L1, env=env)
		assert L1.variable.get(env).data == [2, 4, 6]

	def test_multi_list_all_columns(self, env):
		# Matr►list([A], L1, L2) — each list gets one column
		calc('[[1,2][3,4]]@', MAT_A, env=env)
		calc('Matr►list( [A] , L1 , L2', env=env)
		assert L1.variable.get(env).data == [1, 3]
		assert L2.variable.get(env).data == [2, 4]

	def test_multi_list_single_column_matrix(self, env):
		# 1-column matrix: one list var is enough
		calc('[[7][8][9', env=env)
		calc('Matr►list(', 'Ans', ',', 'L1', env=env)
		assert L1.variable.get(env).data == [7, 8, 9]

	def test_column_out_of_range(self):
		with pytest.raises(InvalidDimError):
			calc('Matr►list(', '[[1,2][3,4]],3,', L1)

	def test_too_many_list_args(self, env):
		# 2-column matrix but 3 list destinations → InvalidDimError
		with pytest.raises(InvalidDimError):
			calc('Matr►list( [[1,2][3,4]], L1 , L2 , L3 )')

	def test_non_matrix_raises(self, env):
		with pytest.raises(DataTypeError):
			calc('Matr►list(', '{1,2,3},1,', L1, env=env)


class TestListToMatr:
	def test_basic_two_lists(self, env):
		# List►matr({1,3},{2,4},[A]) → [[1,2],[3,4]]  (lists become columns)
		calc('List►matr(', '{1,3},{2,4},', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[1, 2], [3, 4]]

	def test_single_list(self, env):
		calc('List►matr(', '{5,6,7},', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[5], [6], [7]]

	def test_unequal_lengths_pads_zero(self, env):
		# Shorter list gets zero-padded to match the longest
		calc('List►matr(', '{1,2,3},{4,5},', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[1, 4], [2, 5], [3, 0]]

	def test_roundtrip_with_matr_to_list(self, env):
		# Store a matrix, round-trip through List►matr
		calc('Matr►list(', '[[10,20][30,40]],', L1, ',', L2, env=env)
		calc('List►matr(', L1, ',', L2, ',', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[10, 20], [30, 40]]

	def test_non_list_raises(self, env):
		# Scalar where a list is expected → DataTypeError
		with pytest.raises(DataTypeError):
			calc('List►matr(', '5,{1,2},', MAT_A, env=env)

	def test_no_matrix_raises(self, env):
		# Missing matrix destination → ArgumentError
		with pytest.raises(ArgumentError):
			calc('List►matr(', '{1,2},{3,4}', env=env)


# ── User-named lists (ᴸNAME) ─────────────────────────────────────────────────

class TestUserLists:
	"""User-named lists stored and read back via the $ (LIST_PREFIX) proxy.
	'$AB' is tokenised as LIST_PREFIX + VAR_A + VAR_B, which the parser reads
	as the user list named 'AB'.
	"""

	# ── Basic store / retrieve ────────────────────────────────────────────────

	def test_store_and_retrieve_with_prefix(self, env):
		calc('{1,2,3}@$AB', env=env)
		assert env.user_lists['AB'].data == [1, 2, 3]

	def test_store_bare_name(self, env):
		# When the value is already a list, →NAME (no ᴸ) stores as a user list
		calc('{4,5}@AB', env=env)
		assert env.user_lists['AB'].data == [4, 5]

	def test_single_char_name(self, env):
		calc('{7,8}@$Z', env=env)
		assert env.user_lists['Z'].data == [7, 8]
		
	def test_with_numbers(self, env):
		calc('{1,2@A1234', env=env)
		assert env.user_lists['A1234'].data == [1, 2]

	def test_overwrite(self, env):
		calc('{1,2,3}@$AB', env=env)
		calc('{9,8}@$AB', env=env)
		assert env.user_lists['AB'].data == [9, 8]
	
	def test_fail_leading_number(self):
		with pytest.raises(TiSyntaxError):
			calc('{1}@$7A')

	# ── Indexing ──────────────────────────────────────────────────────────────

	def test_index_read(self, env):
		calc('{10,20,30}@$AB', env=env)
		assert calc('$AB(2)', env=env) == 20

	def test_index_first_element(self, env):
		calc('{10,20,30}@$AB', env=env)
		assert calc('$AB(1)', env=env) == 10

	def test_index_write(self, env):
		calc('{1,2,3}@$AB', env=env)
		calc('99@$AB(2)', env=env)
		assert env.user_lists['AB'].data == [1, 99, 3]

	# ── Arithmetic — behaves the same as L1–L6 ───────────────────────────────

	def test_scalar_div(self, env):
		calc('{2,4,6}@$AB', env=env)
		assert list(calc('$AB/2', env=env)) == [1, 2, 3]

	def test_add_two_user_lists(self, env):
		calc('{1,2,3}@$AB', env=env)
		calc('{4,5,6}@$CD', env=env)
		assert list(calc('$AB+$CD', env=env)) == [5, 7, 9]

	def test_add_user_list_and_regular_list(self, env):
		calc('{1,2,3}@$AB', env=env)
		calc('{4,5,6}@', L1, env=env)
		assert list(calc('$AB+', L1, env=env)) == [5, 7, 9]

	def test_dim_mismatch_raises(self, env):
		calc('{1,2}@$AB', env=env)
		calc('{3,4,5}@$CD', env=env)
		with pytest.raises(DimMismatchError):
			calc('$AB+$CD', env=env)

	# ── Aggregate functions ───────────────────────────────────────────────────

	def test_dim(self, env):
		calc('{1,2,3,4}@$AB', env=env)
		assert calc('dim(', '$AB', env=env) == 4

	def test_sum(self, env):
		calc('{1,2,3}@$AB', env=env)
		assert calc('sum(', '$AB', env=env) == 6

	def test_max(self, env):
		calc('{3,1,4,1,5}@$AB', env=env)
		assert calc('max(', '$AB', env=env) == 5

	def test_augment_two_user_lists(self, env):
		calc('{1,2}@$AB', env=env)
		calc('{3,4}@$CD', env=env)
		assert list(calc('augment(', '$AB,$CD', env=env)) == [1, 2, 3, 4]

	def test_cum_sum(self, env):
		calc('{1,2,3}@$AB', env=env)
		assert list(calc('cumSum(', '$AB', env=env)) == [1, 3, 6]

	def test_seq_result_stored_in_user_list(self, env):
		calc('seq(', 'X,X,1,5)@$AB', env=env)
		assert list(calc('$AB', env=env)) == [1, 2, 3, 4, 5]

	# ── Matr►list / List►matr ────────────────────────────────────────────────

	def test_matr_to_list_single_column(self, env):
		# Matr►list([A], 1, ᴸAB) — extract column 1 into the user list
		calc('[[1,2][3,4][5,6]]@', MAT_A, env=env)
		calc('Matr►list(', MAT_A, ',1,$AB', env=env)
		assert env.user_lists['AB'].data == [1, 3, 5]

	def test_matr_to_list_multi(self, env):
		# Matr►list([A], ᴸAB, ᴸCD) — extract each column into a user list
		calc('[[1,2][3,4]]@', MAT_A, env=env)
		calc('Matr►list(', MAT_A, ',$AB,$CD', env=env)
		assert env.user_lists['AB'].data == [1, 3]
		assert env.user_lists['CD'].data == [2, 4]

	def test_matr_to_list_mixed(self, env):
		# Matr►list([A], L1, ᴸAB) — one regular list, one user list
		calc('[[1,2][3,4][5,6]]@', MAT_A, env=env)
		calc('Matr►list(', MAT_A, ',', L1, ',$AB', env=env)
		assert L1.variable.get(env).data == [1, 3, 5]
		assert env.user_lists['AB'].data == [2, 4, 6]

	def test_list_to_matr_from_user_lists(self, env):
		# List►matr(ᴸAB, ᴸCD, [A])
		calc('{1,3,5}@$AB', env=env)
		calc('{2,4,6}@$CD', env=env)
		calc('List►matr(', '$AB,$CD,', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[1, 2], [3, 4], [5, 6]]

	def test_list_to_matr_mixed(self, env):
		# Mix a regular list and a user list as sources
		calc('{1,3,5}@', L1, env=env)
		calc('{2,4,6}@$AB', env=env)
		calc('List►matr(', L1, ',$AB,', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[1, 2], [3, 4], [5, 6]]

	def test_roundtrip(self, env):
		# Store a matrix → Matr►list → List►matr → should recover original
		calc('[[10,20][30,40]]@', MAT_A, env=env)
		calc('Matr►list(', MAT_A, ',$AB,$CD', env=env)
		calc('List►matr(', '$AB,$CD,', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[10, 20], [30, 40]]
	
	def test_six_char(self):
		with pytest.raises(TiSyntaxError):
			calc('{1}@$ABCDEF')

# ── Complex numbers ───────────────────────────────────────────────────────────

class TestComplex:
	def test_real(self):         assert pf.real(3+4j) == 3
	def test_imag(self):         assert pf.imag(3+4j) == 4
	def test_conj(self):         assert pf.conj(3+4j) == 3-4j
	def test_angle(self):        assert pf.angle(1j) == approx(math.pi / 2)
	def test_real_on_real(self): assert pf.real(5) == 5
	def test_imag_on_real(self): assert pf.imag(5) == 0


# ── Coordinate conversions ────────────────────────────────────────────────────

class TestCoordinates:
	def test_r_pr(self):
		assert pf.r_pr(3, 4) == approx(5)

	def test_r_ptheta(self):
		assert pf.r_ptheta(1, 0) == approx(0)

	def test_p_rx(self):
		assert pf.p_rx(5, 0) == approx(5)

	def test_p_ry(self):
		assert pf.p_ry(5, math.pi / 2) == approx(5)

	def test_roundtrip(self):
		# (r, θ) → (x, y) → r
		r, theta = 5, math.pi / 3
		x = pf.p_rx(r, theta)
		y = pf.p_ry(r, theta)
		assert pf.r_pr(x, y) == approx(r)
		assert pf.r_ptheta(x, y) == approx(theta)


# ── Probability distributions ─────────────────────────────────────────────────

class TestDistributions:
	def test_normalcdf_median(self):
		assert pf.normalcdf(-1e99, 0) == approx(0.5, rel=1e-4)

	def test_normalcdf_68_rule(self):
		assert pf.normalcdf(-1, 1) == approx(0.6827, rel=1e-3)

	def test_inv_norm_median(self):
		assert pf.inv_norm(0.5) == approx(0, abs=1e-6)

	def test_inv_norm_roundtrip(self):
		assert pf.normalcdf(-1e99, pf.inv_norm(0.9)) == approx(0.9, rel=1e-4)

	def test_normalpdf_peak(self):
		# PDF peaks at x=μ with value 1/sqrt(2π)
		assert pf.normalpdf(0) == approx(1 / math.sqrt(2 * math.pi))

	def test_binompdf(self):
		# P(X=5) for Binomial(10, 0.5) = C(10,5)/2^10
		assert pf.binompdf(10, 0.5, 5) == approx(252 / 1024)

	def test_binomcdf_all(self):
		assert pf.binomcdf(10, 0.5, 10) == approx(1)

	def test_poissonpdf(self):
		assert pf.poissonpdf(3, 3) == approx(math.exp(-3) * 27 / 6)

	def test_poissoncdf_all(self):
		assert pf.poissoncdf(3, 50) == approx(1)

	def test_geometpdf_first(self):
		# P(X=1) = p
		assert pf.geometpdf(0.3, 1) == approx(0.3)

	def test_geometcdf(self):
		assert pf.geometcdf(0.5, 1) == approx(0.5)

	def test_tcdf_symmetric(self):
		# t-distribution is symmetric; CDF(-∞, 0) = 0.5
		assert pf.tcdf(-1e9, 0, df=10) == approx(0.5, rel=1e-4)

	def test_chi_sq_cdf_zero(self):
		assert pf.chi_sq_cdf(0, 0, df=5) == approx(0, abs=1e-6)

	def test_invt_roundtrip(self):
		from purefunctions import invt
		assert pf.tcdf(-1e9, invt(0.9, 10), 10) == approx(0.9, rel=1e-4)


# ── String functions ──────────────────────────────────────────────────────────

class TestStrings:
	def test_length(self):
		assert pf.length(TiString.from_str("HELLO")) == 5

	def test_length_empty(self):
		assert pf.length(TiString.from_str("")) == 0

	def test_in_string_found(self):
		assert pf.in_string(TiString.from_str("HELLO"), TiString.from_str("ELL")) == 2

	def test_in_string_not_found(self):
		assert pf.in_string(TiString.from_str("HELLO"), TiString.from_str("XYZ")) == 0

	def test_in_string_with_start(self):
		assert pf.in_string(TiString.from_str("ABAB"), TiString.from_str("AB"), 3) == 3

	def test_sub(self):
		result = pf.sub(TiString.from_str("HELLO"), 2, 3)
		assert str(result) == "ELL"


# ── Date / time ───────────────────────────────────────────────────────────────

class TestDateTime:
	def test_timecnv(self):
		assert list(pf.timecnv(3661)) == [0, 1, 1, 1]

	def test_timecnv_days(self):
		# 1 day + 1 hr + 1 min + 1 sec = 86400+3600+60+1 = 90061
		assert list(pf.timecnv(90061)) == [1, 1, 1, 1]

	def test_timecnv_negative(self):
		assert list(pf.timecnv(-3661)) == [0, -1, -1, -1]

	def test_dayofwk_wednesday(self):
		assert pf.dayofwk(2024, 12, 25) == 4   # Wednesday

	def test_dayofwk_sunday(self):
		assert pf.dayofwk(2023, 1, 1) == 1     # Sunday

	def test_dbd_mmddyy(self):
		# MM.DDYY: Dec 25 → Dec 31 2024
		assert pf.dbd(12.2524, 12.3124) == 6

	def test_dbd_negative(self):
		assert pf.dbd(12.3124, 12.2524) == -6

	def test_dbd_ddmmyy_leap(self):
		# DDMM.YY: Jan 17 1996 → Jan 17 1997 (1996 is a leap year → 366 days)
		assert pf.dbd(1701.96, 1701.97) == 366

	def test_dbd_mmddyy_leap(self):
		# MM.DDYY same dates — formats can be mixed or used separately
		assert pf.dbd(1.1796, 1.1797) == 366

	def test_dbd_mixed_formats(self):
		# Doc example: dbd(612.07, 2512.07) = 19
		# DDMM.YY: 612.07 → Dec 6 2007; 2512.07 → Dec 25 2007
		assert pf.dbd(612.07, 2512.07) == 19

	def test_dbd_mmddyy_doc_example(self):
		# Doc example: dbd(1.0207, 1.0107) = -1
		# MM.DDYY: Jan 2 2007 → Jan 1 2007
		assert pf.dbd(1.0207, 1.0107) == -1

	def test_dbd_too_many_decimals_mmddyy(self):
		# 5 decimal places in MM.DDYY → ERR:DOMAIN
		with pytest.raises(DomainError, match="too many decimal places"):
			pf.dbd(1.01075, 1.0107)

	def test_dbd_too_many_decimals_ddmmyy(self):
		# 3 decimal places in DDMM.YY → ERR:DOMAIN
		with pytest.raises(DomainError, match="too many decimal places"):
			pf.dbd(1701.961, 1701.97)

	def test_dbd_ambiguous_integer(self):
		# Integer part 13–99 is invalid
		with pytest.raises(DomainError, match="ambiguous"):
			pf.dbd(50.0101, 51.0101)

	def test_setdate_getdate(self):
		e = Environment()
		e.set_date(2008, 7, 4)
		assert list(e.get_date()) == [2008, 7, 4]

	def test_settime_gettime(self):
		e = Environment()
		e.set_time(14, 30, 2)
		assert list(e.get_time()) == [14, 30, approx(2, abs=1)]  # seconds omitted: may drift by 1 across a wall-clock tick

	def test_dt_str_fmt1(self):
		e = Environment()
		e.set_date(2006, 6, 15)
		assert str(e.get_dt_str(1)) == "06/15/06"

	def test_dt_str_fmt2(self):
		e = Environment()
		e.set_date(2005, 12, 25)
		assert str(e.get_dt_str(2)) == "25/12/05"

	def test_dt_str_fmt3(self):
		e = Environment()
		e.set_date(2009, 2, 20)
		assert str(e.get_dt_str(3)) == "09/02/20"

	def test_tm_str_24h(self):
		e = Environment()
		e.set_time(2, 30, 5)
		assert str(e.get_tm_str(24)) == "02:30"

	def test_tm_str_12h_pm(self):
		e = Environment()
		e.set_time(14, 30, 5)
		assert str(e.get_tm_str(12)) == "2:30 PM"

	def test_tm_str_12h_am(self):
		e = Environment()
		e.set_time(9, 5, 0)
		assert str(e.get_tm_str(12)) == "9:05 AM"

	def test_check_tmr(self):
		e = Environment()
		start = e.start_tmr()
		assert 0 <= e.check_tmr(start) <= 2


# ── Parser features ───────────────────────────────────────────────────────────

class TestParserFeatures:
	def test_variable_store_retrieve(self, env):
		calc('3@A', env=env)
		assert calc('A', env=env) == 3

	def test_ans(self, env):
		calc(5, env=env)
		calc('Ans', '+1', env=env)
		assert env.ans == 6

	def test_colon_separator(self, env):
		assert calc('3@A:A*2', env=env) == 6

	def test_list_literal(self):
		assert list(calc('{1,2,3')) == [1, 2, 3]

	def test_list_index(self, env):
		calc('{1,2,3@', L1, env=env)
		assert calc(L1, '(2', env=env) == 2

	def test_matrix_literal(self):
		result = calc('[[1,2][3,4]]')
		assert isinstance(result, TiMatrix)
		assert result.data == [[1, 2], [3, 4]]

	def test_matrix_index(self, env):
		calc('[[1,2][3,4]]@', MAT_A, env=env)
		assert calc(MAT_A, '(2,1', env=env) == 3

	def test_string_literal(self, env):
		calc('"HI"', env=env)
		assert str(env.ans) == "HI"

	def test_dms_degree_in_rad_mode(self):
		# 90° in radian mode = π/2
		assert calc('90°') == approx(math.pi / 2)

	def test_dms_degree_in_deg_mode(self, deg):
		# 90° in degree mode = 90 (no conversion)
		assert calc('90°', env=deg) == 90

	def test_dms_literal_minutes(self):
		# 1°30' = 1.5 decimal degrees (DMS literals always return decimal degrees, no mode conversion)
		assert calc("1°30'") == approx(1.5)

	def test_dms_literal_seconds(self):
		# 0°0'36" = 0.01 decimal degrees (36/3600 = 0.01)
		assert calc('0°0\'36"') == approx(0.01)
	
	def test_dms_min_error(self):
		with pytest.raises(TiSyntaxError):
			calc('1°(2+2)\'3"')
			
	def test_neg_min_not_allowed(self):
		# Direct negation (~) cannot start a DMS component
		with pytest.raises(TiSyntaxError):
			calc("1°~30'")

	def test_dms_neg_exp_in_minutes(self):
		# 1°2e~1' — negative ᴇ exponent in minutes is valid: 2e~1 = 0.2 min
		assert calc("1°2e~1'") == approx(1 + 0.2 / 60)

	def test_dms_prefix_e_neg_exp_in_minutes(self):
		# 1°e~1'3" — prefix ᴇ with negative exponent as minutes: e~1 = 0.1 min
		assert calc("1°e~1'3\"") == approx(1 + 0.1 / 60 + 3 / 3600)

	def test_dms_prefix_e_minutes(self):
		# 1°e1'3" — bare ᴇ1 (= 10) as minutes: 1 + 10/60 + 3/3600
		assert calc("1°e1'3\"") == approx(1 + 10 / 60 + 3 / 3600)

	def test_dms_trailing_sci_errors(self):
		# 1°30'e2 — ᴇ immediately after a DMS literal is invalid
		with pytest.raises(TiSyntaxError):
			calc("1°30'e2")

	def test_dms_neg_sci_literal(self):
		# ~1e1°30' = ~(1e1°30') = -10.5
		assert calc("~1e1°30'") == approx(-10.5)

	def test_dms_lit_sci(self):
		# 2e1°30': literal 2e1 = 20, then DMS 20°30' → 20.5
		assert calc("2e1°30'") == approx(20.5)

	def test_dms_any_minutes(self):
		# 1°60' is valid — no range restriction on minutes
		assert calc("1°60'") == approx(2)

	def test_dms_any_seconds(self):
		# 1°0'60" is valid — no range restriction on seconds
		assert calc("1°0'60\"") == approx(1 + 60 / 3600)

	def test_expr(self, env):
		# expr("1+2") evaluates the string as code
		calc('expr(', '"1+2"', env=env)
		assert env.ans == approx(3)

	def test_inv_postfix(self):
		# [[1,2][3,4]]¹ gives the inverse
		result = calc('[[1,2][3,4]]', INV)
		assert result.data == approx_mat([[-2, 1], [1.5, -0.5]])
		
	def test_transpose_postfix(self):
		result = calc('[[1,2][3,4]]', TRANSPOSE)
		assert result.data == [[1, 3], [2, 4]]


# ── rand ─────────────────────────────────────────────────────────────────────────

class TestRand:
	def test_rand_no_parens_in_range(self):
		# bare rand produces a single float in [0, 1)
		result = calc('rand')
		assert isinstance(result, float)
		assert 0 <= result < 1

	def test_rand_with_parens_returns_list(self):
		# rand(5) returns a TiList of 5 floats
		result = calc('rand', '(5)')
		assert isinstance(result, TiList)
		assert len(result) == 5
		assert all(0 <= x < 1 for x in result)

	def test_rand_with_parens_no_close(self):
		# Trailing ) may be omitted
		result = calc('rand', '(3')
		assert isinstance(result, TiList)
		assert len(result) == 3

	def test_rand_seed_reproducible(self, env):
		# Store a seed → rand, then same seed → rand again; must match
		parse_line(toks('1@', 'rand'), env)
		parse_line(toks('rand'), env)
		first = env.ans
		parse_line(toks('1@', 'rand'), env)
		parse_line(toks('rand'), env)
		assert env.ans == first

	def test_rand_implicit_multiply(self, env):
		# 2rand  ≡  2 * rand()  — result must be in [0, 2)
		parse_line(toks('1@', 'rand'), env)   # fix seed
		parse_line(toks('rand'), env)
		single = env.ans
		parse_line(toks('1@', 'rand'), env)   # reset seed
		parse_line(toks(2, 'rand'), env)          # implicit multiply
		assert env.ans == approx(2 * single)

	def test_rand_int(self):
		# randInt(1,6) returns an integer value in [1, 6]
		result = calc('randInt(', '1,6)')
		assert result == int(result)
		assert 1 <= result <= 6

	def test_rand_int_list(self):
		# randInt(1,6,10) returns a TiList of 10 ints
		result = calc('randInt(', '1,6,10)')
		assert isinstance(result, TiList)
		assert len(result) == 10
		assert all(1 <= x <= 6 for x in result)

	def test_rand_norm(self):
		# randNorm(0,1) returns a float (no guaranteed range, just check type)
		result = calc('randNorm(', '0,1)')
		assert isinstance(result, float)

	def test_rand_norm_list(self):
		# randNorm(0,1,5) returns a TiList of 5 floats
		result = calc('randNorm(', '0,1,5)')
		assert isinstance(result, TiList)
		assert len(result) == 5


# ── Colon-separated statements ────────────────────────────────────────────────

class TestColonStatements:
	def test_colon_ans_is_last(self, env):
		# 1→A:2  →  Ans=2, A=1
		calc('1@A', ':', 2, env=env)
		assert env.ans == 2
		assert VAR_A.variable.get(env) == 1

	def test_colon_store_then_read(self, env):
		# 5→A:A*3  →  Ans=15
		assert calc('5@A:A*3', env=env) == 15

	def test_colon_two_stores(self, env):
		# 1→A:3→B  →  A=1, B=3, Ans=3
		calc('1@A:3@B', env=env)
		assert VAR_A.variable.get(env) == 1
		assert VAR_B.variable.get(env) == 3
		assert env.ans == 3

	def test_colon_three_segments(self, env):
		# 1:2:3  →  Ans=3
		assert calc('1:2:3', env=env) == 3

	def test_colon_ans_carries_across(self, env):
		# 7:Ans+1  →  Ans=8  (Ans from segment 1 is visible in segment 2)
		assert calc('7:', 'Ans', '+1', env=env) == 8

	def test_colon_store_does_not_clobber_a(self, env):
		# 1→A:2  →  A must still be 1 after Ans becomes 2
		calc('1@A:2', env=env)
		assert calc('A', env=env) == 1

	def test_colon_list_then_index(self, env):
		# {10,20,30}→L₁:L₁(2)  →  Ans=20
		assert calc('{10,20,30@', L1, ':', L1, '(2', env=env) == 20


# ── Implicit delimiter closing ────────────────────────────────────────────────

class TestImplicitClose:
	def test_unclosed_paren(self):
		# (1+2  →  3 (trailing ) omitted)
		assert calc('(1+2') == 3

	def test_unclosed_list(self):
		# {1,2,3  →  TiList [1,2,3]
		result = calc('{1,2,3')
		assert list(result) == [1, 2, 3]

	def test_unclosed_matrix(self):
		# [[1,2][3,4  →  2×2 matrix (both ] omitted)
		result = calc('[[1,2][3,4')
		assert isinstance(result, TiMatrix)
		assert result.data == [[1, 2], [3, 4]]

	def test_unclosed_matrix_single_element(self):
		# [[1  →  1×1 matrix
		result = calc('[[1')
		assert isinstance(result, TiMatrix)
		assert result.data == [[1]]

	def test_unclosed_matrix_then_colon_index(self, env):
		# [[1:Ans(1,1  →  first segment produces [[1]], second indexes it → 1
		parse_line(toks('[[1:', 'Ans', '(1,1'), env)
		assert env.ans == 1

	def test_unclosed_list_then_colon_sum(self, env):
		# {1,2,3:sum(Ans  →  Ans=6
		assert calc('{1,2,3:', 'sum(', 'Ans', env=env) == 6

	def test_unclosed_fn_args(self):
		# max(3,7  →  7 (trailing ) omitted)
		assert calc('max(', '3,7') == 7

	def test_nested_unclosed(self):
		# abs(~(3+4  →  7
		assert calc('abs(', '~(3+4') == 7


# ── Storing to dim( ───────────────────────────────────────────────────────────

class TestStoreDim:
	def test_store_dim_list_create(self, env):
		# 5→dim(L₁)  →  L₁ becomes {0,0,0,0,0}
		calc('5@', 'dim(', L1, env=env)
		assert L1.variable.get(env).data == [0, 0, 0, 0, 0]

	def test_store_dim_list_expand(self, env):
		# {1,2,3}→L₁ : 5→dim(L₁)  →  L₁ = {1,2,3,0,0}
		calc('{1,2,3@', L1, env=env)
		calc('5@', 'dim(', L1, env=env)
		assert L1.variable.get(env).data == [1, 2, 3, 0, 0]

	def test_store_dim_list_shrink(self, env):
		# {1,2,3,4,5}→L₁ : 3→dim(L₁)  →  L₁ = {1,2,3}
		calc('{1,2,3,4,5@', L1, env=env)
		calc('3@', 'dim(', L1, env=env)
		assert L1.variable.get(env).data == [1, 2, 3]

	def test_store_dim_matrix_create(self, env):
		# {2,3}→dim([A])  →  [A] becomes 2×3 of zeros
		calc('{2,3@', 'dim(', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == 2 * [3 * [0]]

	def test_store_dim_matrix_resize_preserves(self, env):
		# Build [[1,2][3,4]], then resize to 3×3; original values survive, new cells = 0
		calc('[[1,2][3,4@', MAT_A, env=env)
		calc('{3,3@', 'dim(', MAT_A, env=env)
		assert MAT_A.variable.get(env).data == [[1, 2, 0], [3, 4, 0], [0, 0, 0]]

	def test_dim_read_list(self, env):
		# dim({1,2,3,4}) = 4  (reading, not storing)
		result = calc('dim(', '{1,2,3,4}')
		assert result == 4

	def test_dim_read_matrix(self, env):
		# dim([[1,2,3][4,5,6]]) = {2,3}
		result = calc('dim(', '[[1,2,3][4,5,6')
		assert result.data == [2, 3]


# ── Nesting and combinations ──────────────────────────────────────────────────

class TestNesting:
	def test_sum_of_seq(self):
		# sum(seq(X²,X,1,5))  =  1+4+9+16+25 = 55
		assert calc('sum(', 'seq(', 'X^2,X,1,5') == approx(55)

	def test_seq_with_step(self):
		# seq(X,X,1,9,2)  =  {1,3,5,7,9}
		assert list(calc('seq(', 'X,X,1,9,2')) == approx([1, 3, 5, 7, 9])

	def test_seq_negative_step(self):
		# seq(X,X,5,1,~1)  =  {5,4,3,2,1}
		assert list(calc('seq(', 'X,X,5,1,~1')) == approx([5, 4, 3, 2, 1])

	def test_sigma(self):
		# Σ(X,X,1,10)  =  55
		assert calc('Σ(', 'X,X,1,10') == approx(55)

	def test_sigma_formula(self):
		# Σ(X²,X,1,4)  =  1+4+9+16 = 30
		assert calc('Σ(', 'X^2,X,1,4') == approx(30)

	def test_nderiv(self):
		# nDeriv(X²,X,3) ≈ 6  (derivative of x² at x=3)
		assert calc('nDeriv(', 'X^2,X,3') == approx(6, rel=1e-4)

	def test_fnint(self):
		# fnInt(X²,X,0,3) ≈ 9  (∫₀³ x² dx = 9)
		assert calc('fnInt(', 'X^2,X,0,3') == approx(9, rel=1e-4)

	def test_abs_of_neg_expr(self):
		# abs(~(3+4))  =  7
		assert calc('abs(', '~(3+4') == 7

	def test_max_of_list_expr(self):
		# max({3,1,4,1,5})  =  5
		assert calc('max(', '{3,1,4,1,5') == 5

	def test_nested_arithmetic_functions(self):
		# round(1/6, 3)  =  0.167
		assert calc('round(', '1/6,3') == approx(0.167)

	def test_list_arithmetic_then_sum(self, env):
		# {1,2,3}*2  =  {2,4,6}, then sum({2,4,6}) = 12
		calc('{1,2,3}*2@', L1, env=env)
		assert calc('sum(', L1, env=env) == 12

	def test_matrix_power_then_det(self):
		# det([[1,1][0,1]]²)  =  det([[1,2][0,1]])  =  1
		assert calc('det(', '[[1,1][0,1]]^2') == approx(1)

	def test_string_concat_then_length(self, env):
		# "AB"+"CD" stored in Str1, then length(Str1) = 4
		calc('"AB"+"CD"@', 'Str1', env=env)
		assert calc('length(', 'Str1', env=env) == 4

	def test_cumsum_then_max(self):
		# max(cumSum({1,2,3,4}))  =  max({1,3,6,10})  =  10
		assert calc('max(', 'cumSum(', '{1,2,3,4') == 10

	def test_expr_evaluates_string(self, env):
		# Build "2+3" dynamically as a string stored in Str1, then expr(Str1) = 5
		calc('"2+3"@', 'Str1', env=env)
		assert calc('expr(', 'Str1', env=env) == approx(5)

	def test_ans_index_or_mul_list(self, env):
		# {10,20,30}→Ans  (via plain eval), then Ans(2)  =  20
		parse_line(toks('{10,20,30}'), env)
		parse_line(toks('Ans', '(2)'), env)
		assert env.ans == 20

	def test_ans_index_or_mul_scalar(self, env):
		# 7→Ans, then Ans(3)  =  21  (scalar * 3)
		parse_line(toks(7), env)
		parse_line(toks('Ans', '(3)'), env)
		assert env.ans == 21

	def test_ans_index_matrix(self, env):
		# [[1,2][3,4]]→Ans, then Ans(2,1) = 3
		parse_line(toks('[[1,2][3,4'), env)
		parse_line(toks('Ans', '(2,1'), env)
		assert env.ans == 3

	def test_seq_preserves_variable(self, env):
		# X=99 before seq; seq restores X=99 afterward
		parse_line(toks('99@X'), env)
		parse_line(toks('seq(', 'X,X,1,3'), env)
		parse_line(toks('X'), env)
		assert env.ans == 99


# ── Illegal nesting (ERR:ILLEGAL NEST) ───────────────────────────────────────

class TestIllegalNest:
	"""Each restricted function raises IllegalNestError if nested beyond its limit."""

	def test_seq_no_self_nest(self, env):
		# seq( inside its own formula → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			parse_line(toks('seq(', 'seq(', 'X,X,1,2),X,1,3)'), env)

	def test_seq_allows_normal_nesting(self, env):
		# sum(seq(...)) is fine — only seq inside seq is forbidden
		parse_line(toks('sum(', 'seq(', 'X,X,1,4))'), env)
		assert env.ans == approx(10)

	def test_sigma_no_self_nest(self, env):
		# Σ( inside its own formula → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			parse_line(toks('Σ(', 'Σ(', 'X,X,1,2),X,1,3'), env)

	def test_fnint_no_self_nest(self, env):
		# fnInt( inside its own integrand → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			parse_line(toks('fnInt(', 'fnInt(', 'X,X,0,1),X,0,1'), env)

	def test_nderiv_one_level_ok(self, env):
		# nDeriv( inside nDeriv( once is allowed
		parse_line(toks('nDeriv(', 'nDeriv(', 'X²,X,X),X,1'), env)
		assert env.ans == approx(2, rel=1e-3)

	def test_nderiv_two_levels_raises(self, env):
		# nDeriv( inside nDeriv( inside nDeriv( → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			parse_line(toks('nDeriv(', 'nDeriv(', 'nDeriv(', 'X,X,X),X,X),X,1'), env)

	def test_expr_no_self_nest(self, env):
		# expr( evaluating a string that itself calls expr( → ERR:ILLEGAL NEST
		# Directly store TiString([expr(, Str1, )]) in Str1 — evaluating it calls expr again
		STR_1.variable.set(env, TiString(toks('expr(', 'Str1', ')')))
		with pytest.raises(IllegalNestError):
			calc('expr(', 'Str1', env=env)

	def test_expr_nest_depth_resets(self, env):
		# After a successful expr( call, the guard is back to 0 — can call again
		calc('"1+2"@', 'Str1', env=env)
		assert calc('expr(', 'Str1', env=env) == 3
		assert calc('expr(', 'Str1', env=env) == 3   # second call — must not raise


# ── Thunk capture: commas inside nested delimiters ────────────────────────────

class TestThunkCapture:
	"""Verify that _capture_subgroup and _capture_opener correctly skip over
	commas inside list literals, matrix literals, and string literals so they
	are not mistaken for argument separators."""

	def test_list_literal_in_seq_formula(self, env):
		# seq(sum({1,2,X}),X,1,3) — commas inside {} must not split the thunk
		parse_line(toks('seq(', 'sum(', '{1,2,X}),X,1,3'), env)
		assert env.ans.data == [4, 5, 6]

	def test_matrix_literal_in_seq_formula(self, env):
		# seq(sum({1,2,X}),X,1,3) — commas inside {} must not split the thunk
		parse_line(toks('seq(', 'det(', '[[1,2][X,4]]),X,1,3'), env)
		assert env.ans.data == [2, 0, -2]

	def test_multi_arg_func_in_seq_formula(self, env):
		# seq(max(X,10), X, 8, 12) — commas inside max(...) must not split the thunk
		parse_line(toks('seq(', 'max(', 'X,10),X,8,12)'), env)
		assert env.ans.data == [10, 10, 10, 11, 12]

	def test_string_literal_in_seq_formula(self, env):
		# seq(length("a,b"), X, 1, 3) — the comma in the string must not split the thunk
		# "a,b" has length 3; result should be {3,3,3}
		parse_line(toks('seq(', 'length(', '"a,b"),X,1,3'), env)
		assert env.ans.data == [3, 3, 3]

	def test_colon_inside_thunk_raises(self, env):
		# seq(X:5, X, 1, 3) — colon crosses a statement boundary; rejected at capture time
		with pytest.raises(TiSyntaxError, match="arguments"):
			parse_line(toks('seq(', 'X:5,X,1,3)'), env)

	def test_store_inside_thunk_raises(self, env):
		# store inside a formula is a statement-level construct; rejected at capture time
		with pytest.raises(TiSyntaxError, match="arguments"):
			parse_line(toks('seq(', '5@A,X,1,3)'), env)


class TestSeqIncrement:
	"""seq( raises a clear error when start/end/step are inconsistent."""

	def test_zero_step_raises(self, env):
		with pytest.raises(IncrementError, match="zero"):
			calc('seq( X,X,1,5,0', env=env)

	def test_positive_step_start_after_end_raises(self, env):
		with pytest.raises(IncrementError, match="start.*end|end.*start"):
			calc('seq( X,X,5,1', env=env)

	def test_negative_step_start_before_end_raises(self, env):
		with pytest.raises(IncrementError, match="start.*end|end.*start"):
			calc('seq( X,X,1,5,~1', env=env)

	def test_equal_start_end_is_fine(self, env):
		assert list(calc('seq( X,X,3,3', env=env)) == [3]

	def test_negative_step_descending_is_fine(self, env):
		assert list(calc('seq( X,X,3,1,~1', env=env)) == [3, 2, 1]


class TestCompleXor:
	
	def test_xor(self, env):
		calc('55@A:99@B', env=env)
		calc('int( log( 2) INV log( max( {A,B', env=env)
		calc('2^ cumSum( binomcdf( Ans ,0', env=env)
		assert calc('sum( Ans .5(1= abs( int( 2 fPart( Ans INV (A+Bi', env=env) == 84

	def test_xor2(self, env):
		calc('55@A:99@B', env=env)
		calc('seq( 2^N,N,8,1,~1@ L1', env=env).data == [256, 128, 64, 32, 16, 8, 4, 2]
		calc('.5 sum( L1 *(1= abs( int( 2 fPart( (A+Bi)/ L1 @F', env=env)
		assert calc('F', env=env) == 84
