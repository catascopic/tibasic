"""Test suite for the TI-BASIC interpreter."""

import math
import pytest
from pytest import approx

import purefunctions as pf
from environment import Environment
from parser import parse_line, ParseError
from tokens import (
	TOKENS, Token,
	STORE, COMMA, QUOTE, COLON, DOT, NEG, DEG, APOS, SCI_E,
	ADD, SUB, MUL, DIV, POW, XROOT, FACT, NPR, NCR,
	EQ, LT, GT, LE, GE, NE, AND, OR, XOR,
	L_PAREN, R_PAREN, L_BRACE, R_BRACE, L_BRACKET, R_BRACKET,
	ANS, INV, SQ, TRANSPOSE, RAND, DIM,
)
from tiobjects import TiList, TiMatrix, TiString


# ── Helpers ───────────────────────────────────────────────────────────────────

_by_text = {t.text: t for t in reversed(TOKENS)}

def T(text: str) -> Token:
	"""Look up a token by its display text."""
	return _by_text[text]

def toks(*items) -> list[Token]:
	"""Build a token list.
	- Token     → used directly
	- int ≥ 0  → tokenised digit-by-digit
	- str in token table → that token
	- other str → each character looked up individually
	"""
	result = []
	for item in items:
		if isinstance(item, Token):
			result.append(item)
		elif isinstance(item, int) and item >= 0:
			for c in str(item):
				result.append(_by_text[c])
		elif isinstance(item, str):
			tok = _by_text.get(item)
			if tok is not None:
				result.append(tok)
			else:
				for c in item:
					result.append(_by_text[c])
	return result

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


# ── Arithmetic ────────────────────────────────────────────────────────────────

class TestArithmetic:
	def test_add(self):          assert calc(1, ADD, 2) == 3.0
	def test_sub(self):          assert calc(5, SUB, 3) == 2.0
	def test_mul(self):          assert calc(3, MUL, 4) == 12.0
	def test_div(self):          assert calc(7, DIV, 2) == 3.5
	def test_pow(self):          assert calc(2, POW, 10) == 1024.0
	def test_negation(self):     assert calc(NEG, 5) == -5.0
	def test_sq_postfix(self):   assert calc(7, SQ) == 49.0
	def test_xroot(self):        assert calc(3, XROOT, 8) == approx(2)
	def test_sci_e(self):        assert calc(1, SCI_E, 3) == 1000.0
	def test_implicit_mul(self): assert calc(2, L_PAREN, 3, ADD, 4, R_PAREN) == 14.0

	def test_precedence_mul_over_add(self):
		assert calc(2, ADD, 3, MUL, 4) == 14.0

	def test_precedence_parens(self):
		assert calc(L_PAREN, 2, ADD, 3, R_PAREN, MUL, 4) == 20.0

	def test_pow_right_assoc(self):
		# 2^3^2 = 2^(3^2) = 2^9 = 512 (right-associative)
		assert calc(2, POW, 3, POW, 2) == 512.0


# ── Scientific notation (ᴇ) ───────────────────────────────────────────────────

class TestSciE:
	# ── Positive cases ────────────────────────────────────────────────────────

	def test_infix_basic(self):
		# 1ᴇ3 = 1000
		assert calc(1, SCI_E, 3) == 1000.0

	def test_prefix_basic(self):
		# ᴇ3 = 10^3 = 1000  (no left operand → implicit 1)
		assert calc(SCI_E, 3) == 1000.0

	def test_infix_zero_exp(self):
		# 5ᴇ0 = 5
		assert calc(5, SCI_E, 0) == 5.0

	def test_infix_decimal_exp(self):
		# 1ᴇ1.5 = 10^1.5
		assert calc(1, SCI_E, 1, DOT, 5) == approx(10 ** 1.5)

	def test_infix_multi_digit_exp(self):
		# 2ᴇ10 = 2048 (not 2*(10^1)*0 or anything weird)
		assert calc(2, SCI_E, 10) == approx(2 * 10 ** 10)

	def test_infix_dms_exp(self):
		# 1ᴇ1°30' — exponent is 1.5 decimal degrees → 10^1.5
		assert calc(1, SCI_E, 1, DEG, 30, APOS) == approx(10 ** 1.5)

	def test_precedence_over_add(self):
		# 2 + 3ᴇ2 = 2 + 300 = 302  (ᴇ binds tighter than +)
		assert calc(2, ADD, 3, SCI_E, 2) == 302.0

	def test_precedence_over_mul(self):
		# 2 * 3ᴇ2 = 2 * 300 = 600  (ᴇ binds tighter than *)
		assert calc(2, MUL, 3, SCI_E, 2) == 600.0

	def test_neg_before_sci_e(self):
		# −1ᴇ3 = −1000  (negation of the whole scientific-notation number)
		assert calc(NEG, 1, SCI_E, 3) == -1000.0

	def test_pow_rhs_is_sci_e(self):
		# 2^3ᴇ2: ᴇ binds tighter than ^, so exponent is 3ᴇ2=300 → 2^300
		assert calc(2, POW, 3, SCI_E, 2) == approx(2 ** 300)

	def test_in_larger_expression(self):
		# (1ᴇ3 + 1ᴇ2) = 1100
		assert calc(L_PAREN, 1, SCI_E, 3, ADD, 1, SCI_E, 2, R_PAREN) == 1100.0

	# ── Negative cases ────────────────────────────────────────────────────────

	def test_rejects_paren_expr(self):
		# 1ᴇ(3) — parenthesised expression is not a numeric literal
		with pytest.raises(ParseError):
			calc(1, SCI_E, L_PAREN, 3, R_PAREN)

	def test_rejects_variable(self):
		# 1ᴇA — variable is not a numeric literal
		with pytest.raises(ParseError):
			calc(1, SCI_E, 'A')

	def test_rejects_expression_rhs(self):
		# 1ᴇ2+1 must parse as (1ᴇ2)+1 = 101, not 1ᴇ(2+1) = 1000
		# (confirms the RHS stops at the literal boundary)
		assert calc(1, SCI_E, 2, ADD, 1) == 101.0

	def test_rejects_ans_as_exponent(self):
		# 1ᴇAns — Ans is not a numeric literal
		with pytest.raises(ParseError):
			calc(1, SCI_E, ANS)

	def test_infix_negative_exp(self):
		# 1ᴇ−3 = 0.001
		assert calc(1, SCI_E, NEG, 3) == approx(0.001)

	def test_prefix_negative_exp(self):
		# ᴇ−3 = 10^−3 = 0.001
		assert calc(SCI_E, NEG, 3) == approx(0.001)

	def test_negative_exp_decimal(self):
		# 1ᴇ−1.5 = 10^−1.5
		assert calc(1, SCI_E, NEG, 1, DOT, 5) == approx(10 ** -1.5)

	def test_negative_exp_in_expression(self):
		# 2 + 3ᴇ−2 = 2 + 0.03 = 2.03
		assert calc(2, ADD, 3, SCI_E, NEG, 2) == approx(2.03)

	def test_neg_literal_neg_exp(self):
		# −2ᴇ−3 = −0.002
		assert calc(NEG, 2, SCI_E, NEG, 3) == approx(-0.002)

	def test_rejects_double_neg_exp(self):
		# 1ᴇ−−3 — two negations is not a valid literal
		with pytest.raises(ParseError):
			calc(1, SCI_E, NEG, NEG, 3)


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
	def test_fact_parser(self): assert calc(5, FACT) == 120.0
	def test_npr_parser(self):  assert calc(5, NPR, 3) == 60.0
	def test_ncr_parser(self):  assert calc(5, NCR, 3) == 10.0


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
		assert list(TiList([2, 4, 6]) / 2) == [1.0, 2.0, 3.0]


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
		assert product.data[0] == approx([1, 0])
		assert product.data[1] == approx([0, 1])

	def test_dim_matrix(self):
		assert list(pf.dim(TiMatrix([[1, 2, 3], [4, 5, 6]]))) == [2, 3]

	def test_augment_matrix(self):
		a = TiMatrix([[1, 2], [3, 4]])
		b = TiMatrix([[5], [6]])
		assert pf.augment(a, b).data == [[1, 2, 5], [3, 4, 6]]

	def test_rref_solve(self):
		# 2x + y = 5, x - y = 1  →  x=2, y=1
		result = pf.rref(TiMatrix([[2, 1, 5], [1, -1, 1]]))
		assert result.data[0] == approx([1, 0, 2])
		assert result.data[1] == approx([0, 1, 1])


class TestMatrixRowOps:
	def setup_method(self):
		self.mat = TiMatrix([[1, 2], [3, 4], [5, 6]])

	def test_rowswap(self):
		result = pf.rowswap(self.mat, 1, 3)
		assert result.data[0] == [5, 6]
		assert result.data[2] == [1, 2]
		assert self.mat.data[0] == [1, 2]   # original unchanged

	def test_row_plus(self):
		result = pf.row_plus(self.mat, 1, 2)
		assert result.data[1] == [4, 6]     # row2 += row1

	def test_times_row(self):
		result = pf.times_row(3, self.mat, 1)
		assert result.data[0] == [3, 6]

	def test_times_row_plus(self):
		result = pf.times_row_plus(2, self.mat, 1, 2)
		assert result.data[1] == [5, 8]     # row2 += 2*row1


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

	def test_invnorm_median(self):
		assert pf.invnorm(0.5) == approx(0, abs=1e-6)

	def test_invnorm_roundtrip(self):
		assert pf.normalcdf(-1e99, pf.invnorm(0.9)) == approx(0.9, rel=1e-4)

	def test_normalpdf_peak(self):
		# PDF peaks at x=μ with value 1/sqrt(2π)
		assert pf.normalpdf(0) == approx(1 / math.sqrt(2 * math.pi))

	def test_binompdf(self):
		# P(X=5) for Binomial(10, 0.5) = C(10,5)/2^10
		assert pf.binompdf(10, 0.5, 5) == approx(252 / 1024)

	def test_binomcdf_all(self):
		assert pf.binomcdf(10, 0.5, 10) == approx(1.0)

	def test_poissonpdf(self):
		assert pf.poissonpdf(3, 3) == approx(math.exp(-3) * 27 / 6)

	def test_poissoncdf_all(self):
		assert pf.poissoncdf(3, 50) == approx(1.0)

	def test_geometpdf_first(self):
		# P(X=1) = p
		assert pf.geometpdf(0.3, 1) == approx(0.3)

	def test_geometcdf(self):
		assert pf.geometcdf(0.5, 1) == approx(0.5)

	def test_tcdf_symmetric(self):
		# t-distribution is symmetric; CDF(-∞, 0) = 0.5
		assert pf.tcdf(-1e9, 0, df=10) == approx(0.5, rel=1e-4)

	def test_chi2cdf_zero(self):
		assert pf.chi2cdf(0, 0, df=5) == approx(0.0, abs=1e-6)

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

	def test_sub_string(self):
		result = pf.sub_string(TiString.from_str("HELLO"), 2, 3)
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
		assert pf.dbd(12.2524, 12.3124) == 6.0

	def test_dbd_negative(self):
		assert pf.dbd(12.3124, 12.2524) == -6.0

	def test_dbd_ddmmyy_leap(self):
		# DDMM.YY: Jan 17 1996 → Jan 17 1997 (1996 is a leap year → 366 days)
		assert pf.dbd(1701.96, 1701.97) == 366.0

	def test_dbd_mmddyy_leap(self):
		# MM.DDYY same dates — formats can be mixed or used separately
		assert pf.dbd(1.1796, 1.1797) == 366.0

	def test_dbd_mixed_formats(self):
		# Doc example: dbd(612.07, 2512.07) = 19
		# DDMM.YY: 612.07 → Dec 6 2007; 2512.07 → Dec 25 2007
		assert pf.dbd(612.07, 2512.07) == 19.0

	def test_dbd_mmddyy_doc_example(self):
		# Doc example: dbd(1.0207, 1.0107) = -1
		# MM.DDYY: Jan 2 2007 → Jan 1 2007
		assert pf.dbd(1.0207, 1.0107) == -1.0

	def test_dbd_too_many_decimals_mmddyy(self):
		# 5 decimal places in MM.DDYY → ERR:DOMAIN
		with pytest.raises(ValueError, match="too many decimal places"):
			pf.dbd(1.01075, 1.0107)

	def test_dbd_too_many_decimals_ddmmyy(self):
		# 3 decimal places in DDMM.YY → ERR:DOMAIN
		with pytest.raises(ValueError, match="too many decimal places"):
			pf.dbd(1701.961, 1701.97)

	def test_dbd_ambiguous_integer(self):
		# Integer part 13–99 is invalid
		with pytest.raises(ValueError, match="ambiguous"):
			pf.dbd(50.0101, 51.0101)

	def test_setdate_getdate(self):
		e = Environment()
		e.set_date(2020, 6, 15)
		assert list(e.get_date()) == [2020, 6, 15]

	def test_settime_gettime(self):
		e = Environment()
		e.set_time(14, 30, 0)
		assert list(e.get_time()) == [14, 30, 0]

	def test_dt_str_fmt1(self):
		e = Environment()
		e.set_date(2020, 6, 15)
		assert str(e.get_dt_str(1)) == "6/15/20"

	def test_dt_str_fmt2(self):
		e = Environment()
		e.set_date(2020, 6, 15)
		assert str(e.get_dt_str(2)) == "15/6/20"

	def test_dt_str_fmt3(self):
		e = Environment()
		e.set_date(2020, 6, 15)
		assert str(e.get_dt_str(3)) == "20/6/15"

	def test_tm_str_24h(self):
		e = Environment()
		e.set_time(14, 30, 5)
		assert str(e.get_tm_str(24)) == "14:30:05"

	def test_tm_str_12h_pm(self):
		e = Environment()
		e.set_time(14, 30, 5)
		assert str(e.get_tm_str(12)) == "2:30:05PM"

	def test_tm_str_12h_am(self):
		e = Environment()
		e.set_time(9, 5, 0)
		assert str(e.get_tm_str(12)) == "9:05:00AM"

	def test_check_tmr(self):
		e = Environment()
		start = e.start_tmr()
		assert 0 <= e.check_tmr(start) <= 2


# ── Parser features ───────────────────────────────────────────────────────────

class TestParserFeatures:
	def test_variable_store_retrieve(self, env):
		parse_line(toks(3, STORE, 'A'), env)
		parse_line(toks('A'), env)
		assert env.ans == 3.0

	def test_ans(self, env):
		parse_line(toks(5), env)
		parse_line(toks(ANS, ADD, 1), env)
		assert env.ans == 6.0

	def test_colon_separator(self, env):
		parse_line(toks(3, STORE, 'A', COLON, 'A', MUL, 2), env)
		assert env.ans == 6.0

	def test_list_literal(self):
		result = calc(L_BRACE, 1, COMMA, 2, COMMA, 3, R_BRACE)
		assert list(result) == [1.0, 2.0, 3.0]

	def test_list_index(self, env):
		l1 = T('L₁')
		parse_line(toks(L_BRACE, 1, COMMA, 2, COMMA, 3, R_BRACE, STORE, l1), env)
		parse_line(toks(l1, L_PAREN, 2, R_PAREN), env)
		assert env.ans == 2.0

	def test_matrix_literal(self):
		result = calc(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		              L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET)
		assert isinstance(result, TiMatrix)
		assert result.data == [[1.0, 2.0], [3.0, 4.0]]

	def test_matrix_index(self, env):
		ma = T('[A]')
		parse_line(toks(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		               L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET, STORE, ma), env)
		parse_line(toks(ma, L_PAREN, 2, COMMA, 1, R_PAREN), env)
		assert env.ans == 3.0

	def test_string_literal(self, env):
		parse_line([QUOTE, T('H'), T('I'), QUOTE], env)
		assert str(env.ans) == "HI"

	def test_dms_degree_in_rad_mode(self):
		# 90° in radian mode = π/2
		assert calc(90, DEG) == approx(math.pi / 2)

	def test_dms_degree_in_deg_mode(self, deg):
		# 90° in degree mode = 90 (no conversion)
		assert calc(90, DEG, env=deg) == 90.0

	def test_dms_literal_minutes(self):
		# 1°30' = 1.5 decimal degrees (DMS literals always return decimal degrees, no mode conversion)
		assert calc(1, DEG, 30, APOS) == approx(1.5)

	def test_dms_literal_seconds(self):
		# 0°0'36" = 0.01 decimal degrees (36/3600 = 0.01)
		assert calc(0, DEG, 0, APOS, 36, QUOTE) == approx(0.01)

	def test_expr(self, env):
		# expr("1+2") evaluates the string as code
		parse_line([T('expr('), QUOTE, T('1'), ADD, T('2'), QUOTE], env)
		assert env.ans == approx(3)

	def test_inv_postfix(self):
		# [[1,2][3,4]]¹ gives the inverse
		mat_toks = toks(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		                L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET)
		result = calc(*mat_toks, INV)
		assert isinstance(result, TiMatrix)
		assert result.data[0][0] == approx(-2)
		assert result.data[1][1] == approx(-0.5)

	def test_transpose_postfix(self):
		mat_toks = toks(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		                L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET)
		result = calc(*mat_toks, TRANSPOSE)
		assert result.data == [[1.0, 3.0], [2.0, 4.0]]


# ── rand ─────────────────────────────────────────────────────────────────────────

class TestRand:
	def test_rand_no_parens_in_range(self):
		# bare rand produces a single float in [0, 1)
		result = calc(RAND)
		assert isinstance(result, float)
		assert 0.0 <= result < 1.0

	def test_rand_with_parens_returns_list(self):
		# rand(5) returns a TiList of 5 floats
		result = calc(RAND, L_PAREN, 5, R_PAREN)
		assert isinstance(result, TiList)
		assert len(result) == 5
		assert all(0.0 <= x < 1.0 for x in result)

	def test_rand_with_parens_no_close(self):
		# Trailing ) may be omitted
		result = calc(RAND, L_PAREN, 3)
		assert isinstance(result, TiList)
		assert len(result) == 3

	def test_rand_seed_reproducible(self, env):
		# Store a seed → rand, then same seed → rand again; must match
		parse_line(toks(1, STORE, RAND), env)
		parse_line(toks(RAND), env)
		first = env.ans
		parse_line(toks(1, STORE, RAND), env)
		parse_line(toks(RAND), env)
		assert env.ans == first

	def test_rand_implicit_multiply(self, env):
		# 2rand  ≡  2 * rand()  — result must be in [0, 2)
		parse_line(toks(1, STORE, RAND), env)   # fix seed
		parse_line(toks(RAND), env)
		single = env.ans
		parse_line(toks(1, STORE, RAND), env)   # reset seed
		parse_line(toks(2, RAND), env)          # implicit multiply
		assert env.ans == approx(2 * single)

	def test_rand_int(self):
		# randInt(1,6) returns an integer value in [1, 6]
		result = calc('randInt(', 1, COMMA, 6, R_PAREN)
		assert result == int(result)
		assert 1 <= result <= 6

	def test_rand_int_list(self):
		# randInt(1,6,10) returns a TiList of 10 ints
		result = calc('randInt(', 1, COMMA, 6, COMMA, 10, R_PAREN)
		assert isinstance(result, TiList)
		assert len(result) == 10
		assert all(1 <= x <= 6 for x in result)

	def test_rand_norm(self):
		# randNorm(0,1) returns a float (no guaranteed range, just check type)
		result = calc('randNorm(', 0, COMMA, 1, R_PAREN)
		assert isinstance(result, float)

	def test_rand_norm_list(self):
		# randNorm(0,1,5) returns a TiList of 5 floats
		result = calc('randNorm(', 0, COMMA, 1, COMMA, 5, R_PAREN)
		assert isinstance(result, TiList)
		assert len(result) == 5


# ── Colon-separated statements ────────────────────────────────────────────────

class TestColonStatements:
	def test_colon_ans_is_last(self, env):
		# 1→A:2  →  Ans=2, A=1
		parse_line(toks(1, STORE, 'A', COLON, 2), env)
		assert env.ans == 2.0
		assert T('A').variable.get(env) == 1.0

	def test_colon_store_then_read(self, env):
		# 5→A:A*3  →  Ans=15
		parse_line(toks(5, STORE, 'A', COLON, 'A', MUL, 3), env)
		assert env.ans == 15.0

	def test_colon_two_stores(self, env):
		# 1→A:3→B  →  A=1, B=3, Ans=3
		parse_line(toks(1, STORE, 'A', COLON, 3, STORE, 'B'), env)
		assert T('A').variable.get(env) == 1.0
		assert T('B').variable.get(env) == 3.0
		assert env.ans == 3.0

	def test_colon_three_segments(self, env):
		# 1:2:3  →  Ans=3
		parse_line(toks(1, COLON, 2, COLON, 3), env)
		assert env.ans == 3.0

	def test_colon_ans_carries_across(self, env):
		# 7:Ans+1  →  Ans=8  (Ans from segment 1 is visible in segment 2)
		parse_line(toks(7, COLON, ANS, ADD, 1), env)
		assert env.ans == 8.0

	def test_colon_store_does_not_clobber_a(self, env):
		# 1→A:2  →  A must still be 1 after Ans becomes 2
		parse_line(toks(1, STORE, 'A', COLON, 2), env)
		parse_line(toks('A'), env)
		assert env.ans == 1.0

	def test_colon_list_then_index(self, env):
		# {10,20,30}→L₁:L₁(2)  →  Ans=20
		l1 = T('L₁')
		parse_line(toks(L_BRACE, 10, COMMA, 20, COMMA, 30, R_BRACE, STORE, l1,
		                COLON, l1, L_PAREN, 2, R_PAREN), env)
		assert env.ans == 20.0


# ── Implicit delimiter closing ────────────────────────────────────────────────

class TestImplicitClose:
	def test_unclosed_paren(self):
		# (1+2  →  3 (trailing ) omitted)
		assert calc(L_PAREN, 1, ADD, 2) == 3.0

	def test_unclosed_list(self):
		# {1,2,3  →  TiList [1,2,3]
		result = calc(L_BRACE, 1, COMMA, 2, COMMA, 3)
		assert list(result) == [1.0, 2.0, 3.0]

	def test_unclosed_matrix(self):
		# [[1,2][3,4  →  2×2 matrix (both ] omitted)
		result = calc(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		              L_BRACKET, 3, COMMA, 4)
		assert isinstance(result, TiMatrix)
		assert result.data == [[1.0, 2.0], [3.0, 4.0]]

	def test_unclosed_matrix_single_element(self):
		# [[1  →  1×1 matrix
		result = calc(L_BRACKET, L_BRACKET, 1)
		assert isinstance(result, TiMatrix)
		assert result.data == [[1.0]]

	def test_unclosed_matrix_then_colon_index(self, env):
		# [[1:Ans(1,1  →  first segment produces [[1]], second indexes it → 1.0
		parse_line(toks(L_BRACKET, L_BRACKET, 1, COLON, ANS, L_PAREN, 1, COMMA, 1), env)
		assert env.ans == 1.0

	def test_unclosed_list_then_colon_sum(self, env):
		# {1,2,3:sum(Ans  →  Ans=6
		parse_line(toks(L_BRACE, 1, COMMA, 2, COMMA, 3, COLON, T('sum('), ANS), env)
		assert env.ans == 6.0

	def test_unclosed_fn_args(self):
		# max(3,7  →  7 (trailing ) omitted)
		assert calc(T('max('), 3, COMMA, 7) == 7.0

	def test_nested_unclosed(self):
		# abs(−(3+4  →  7
		assert calc(T('abs('), NEG, L_PAREN, 3, ADD, 4) == 7.0


# ── Storing to dim( ───────────────────────────────────────────────────────────

class TestStoreDim:
	def test_store_dim_list_create(self, env):
		# 5→dim(L₁)  →  L₁ becomes {0,0,0,0,0}
		l1 = T('L₁')
		parse_line(toks(5, STORE, DIM, l1), env)
		lst = l1.variable.get(env)
		assert len(lst) == 5
		assert all(x == 0 for x in lst)

	def test_store_dim_list_expand(self, env):
		# {1,2,3}→L₁ : 5→dim(L₁)  →  L₁ = {1,2,3,0,0}
		l1 = T('L₁')
		parse_line(toks(L_BRACE, 1, COMMA, 2, COMMA, 3, R_BRACE, STORE, l1), env)
		parse_line(toks(5, STORE, DIM, l1), env)
		lst = l1.variable.get(env)
		assert len(lst) == 5
		assert list(lst)[:3] == [1.0, 2.0, 3.0]
		assert list(lst)[3:] == [0, 0]

	def test_store_dim_list_shrink(self, env):
		# {1,2,3,4,5}→L₁ : 3→dim(L₁)  →  L₁ = {1,2,3}
		l1 = T('L₁')
		parse_line(toks(L_BRACE, 1, COMMA, 2, COMMA, 3, COMMA, 4, COMMA, 5, R_BRACE, STORE, l1), env)
		parse_line(toks(3, STORE, DIM, l1), env)
		assert len(l1.variable.get(env)) == 3

	def test_store_dim_matrix_create(self, env):
		# {2,3}→dim([A])  →  [A] becomes 2×3 of zeros
		ma = T('[A]')
		parse_line(toks(L_BRACE, 2, COMMA, 3, R_BRACE, STORE, DIM, ma), env)
		mat = ma.variable.get(env)
		assert mat.rows == 2
		assert mat.cols == 3
		assert all(mat.data[r][c] == 0 for r in range(2) for c in range(3))

	def test_store_dim_matrix_resize_preserves(self, env):
		# Build [[1,2][3,4]], then resize to 3×3; original values survive, new cells = 0
		ma = T('[A]')
		parse_line(toks(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		                L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET, STORE, ma), env)
		parse_line(toks(L_BRACE, 3, COMMA, 3, R_BRACE, STORE, DIM, ma), env)
		mat = ma.variable.get(env)
		assert mat.rows == 3 and mat.cols == 3
		assert mat.data[0][0] == 1.0
		assert mat.data[1][1] == 4.0
		assert mat.data[2][2] == 0.0

	def test_dim_read_list(self, env):
		# dim({1,2,3,4}) = 4  (reading, not storing)
		result = calc(DIM, L_BRACE, 1, COMMA, 2, COMMA, 3, COMMA, 4, R_BRACE)
		assert result == 4.0

	def test_dim_read_matrix(self, env):
		# dim([[1,2,3][4,5,6]]) = {2,3}
		result = calc(DIM, L_BRACKET, L_BRACKET, 1, COMMA, 2, COMMA, 3, R_BRACKET,
		              L_BRACKET, 4, COMMA, 5, COMMA, 6, R_BRACKET, R_BRACKET)
		assert list(result) == [2.0, 3.0]


# ── Nesting and combinations ──────────────────────────────────────────────────

class TestNesting:
	def test_sum_of_seq(self):
		# sum(seq(X²,X,1,5))  =  1+4+9+16+25 = 55
		X = T('X')
		result = calc(T('sum('), T('seq('), X, SQ, COMMA, X, COMMA, 1, COMMA, 5, R_PAREN)
		assert result == approx(55)

	def test_seq_with_step(self):
		# seq(X,X,1,9,2)  =  {1,3,5,7,9}
		X = T('X')
		result = calc(T('seq('), X, COMMA, X, COMMA, 1, COMMA, 9, COMMA, 2, R_PAREN)
		assert list(result) == approx([1, 3, 5, 7, 9])

	def test_seq_negative_step(self):
		# seq(X,X,5,1,−1)  =  {5,4,3,2,1}
		X = T('X')
		result = calc(T('seq('), X, COMMA, X, COMMA, 5, COMMA, 1, COMMA, NEG, 1, R_PAREN)
		assert list(result) == approx([5, 4, 3, 2, 1])

	def test_sigma(self):
		# Σ(X,X,1,10)  =  55
		X = T('X')
		result = calc(T('Σ('), X, COMMA, X, COMMA, 1, COMMA, 10, R_PAREN)
		assert result == approx(55)

	def test_sigma_formula(self):
		# Σ(X²,X,1,4)  =  1+4+9+16 = 30
		X = T('X')
		result = calc(T('Σ('), X, SQ, COMMA, X, COMMA, 1, COMMA, 4, R_PAREN)
		assert result == approx(30)

	def test_nderiv(self):
		# nDeriv(X²,X,3)  ≈  6  (derivative of x² at x=3)
		X = T('X')
		result = calc(T('nDeriv('), X, SQ, COMMA, X, COMMA, 3, R_PAREN)
		assert result == approx(6.0, rel=1e-4)

	def test_fnint(self):
		# fnInt(X²,X,0,3)  ≈  9  (∫₀³ x² dx = 9)
		X = T('X')
		result = calc(T('fnInt('), X, SQ, COMMA, X, COMMA, 0, COMMA, 3, R_PAREN)
		assert result == approx(9.0, rel=1e-4)

	def test_abs_of_neg_expr(self):
		# abs(−(3+4))  =  7
		assert calc(T('abs('), NEG, L_PAREN, 3, ADD, 4, R_PAREN, R_PAREN) == 7.0

	def test_max_of_list_expr(self):
		# max({3,1,4,1,5})  =  5
		result = calc(T('max('), L_BRACE, 3, COMMA, 1, COMMA, 4, COMMA, 1, COMMA, 5, R_BRACE, R_PAREN)
		assert result == 5.0

	def test_nested_arithmetic_functions(self):
		# round(1/6, 3)  =  0.167
		assert calc(T('round('), 1, DIV, 6, COMMA, 3) == approx(0.167)

	def test_list_arithmetic_then_sum(self, env):
		# {1,2,3}*2  =  {2,4,6}, then sum({2,4,6}) = 12
		parse_line(toks(L_BRACE, 1, COMMA, 2, COMMA, 3, R_BRACE, MUL, 2, STORE, T('L₁')), env)
		parse_line(toks(T('sum('), T('L₁'), R_PAREN), env)
		assert env.ans == 12.0

	def test_matrix_power_then_det(self):
		# det([[1,1][0,1]]²)  =  det([[1,2][0,1]])  =  1
		mat_toks = toks(L_BRACKET, L_BRACKET, 1, COMMA, 1, R_BRACKET,
		                L_BRACKET, 0, COMMA, 1, R_BRACKET, R_BRACKET)
		result = calc(T('det('), *mat_toks, POW, 2, R_PAREN)
		assert result == approx(1.0)

	def test_string_concat_then_length(self, env):
		# "AB"+"CD" stored in Str1, then length(Str1) = 4
		str1 = T('Str1')
		parse_line([QUOTE, T('A'), T('B'), QUOTE, ADD, QUOTE, T('C'), T('D'), QUOTE, STORE, str1], env)
		parse_line([T('length('), str1, R_PAREN], env)
		assert env.ans == 4.0

	def test_cumsum_then_max(self):
		# max(cumSum({1,2,3,4}))  =  max({1,3,6,10})  =  10
		result = calc(T('max('), T('cumSum('), L_BRACE, 1, COMMA, 2, COMMA, 3, COMMA, 4, R_BRACE, R_PAREN)
		assert result == 10.0

	def test_expr_evaluates_string(self, env):
		# Build "2+3" dynamically as a string stored in Str1, then expr(Str1) = 5
		str1 = T('Str1')
		parse_line([QUOTE, T('2'), ADD, T('3'), QUOTE, STORE, str1], env)
		parse_line([T('expr('), str1, R_PAREN], env)
		assert env.ans == approx(5.0)

	def test_ans_index_or_mul_list(self, env):
		# {10,20,30}→Ans  (via plain eval), then Ans(2)  =  20
		parse_line(toks(L_BRACE, 10, COMMA, 20, COMMA, 30, R_BRACE), env)
		parse_line(toks(ANS, L_PAREN, 2, R_PAREN), env)
		assert env.ans == 20.0

	def test_ans_index_or_mul_scalar(self, env):
		# 7→Ans, then Ans(3)  =  21  (scalar * 3)
		parse_line(toks(7), env)
		parse_line(toks(ANS, L_PAREN, 3, R_PAREN), env)
		assert env.ans == 21.0

	def test_ans_index_matrix(self, env):
		# [[1,2][3,4]]→Ans, then Ans(2,1) = 3
		parse_line(toks(L_BRACKET, L_BRACKET, 1, COMMA, 2, R_BRACKET,
		                L_BRACKET, 3, COMMA, 4, R_BRACKET, R_BRACKET), env)
		parse_line(toks(ANS, L_PAREN, 2, COMMA, 1, R_PAREN), env)
		assert env.ans == 3.0

	def test_seq_preserves_variable(self, env):
		# X=99 before seq; seq restores X=99 afterward
		parse_line(toks(99, STORE, 'X'), env)
		X = T('X')
		parse_line(toks(T('seq('), X, COMMA, X, COMMA, 1, COMMA, 3, R_PAREN), env)
		parse_line(toks(T('X')), env)
		assert env.ans == 99.0


# ── Illegal nesting (ERR:ILLEGAL NEST) ───────────────────────────────────────

class TestIllegalNest:
	"""Each restricted function raises ValueError if nested beyond its limit."""

	def test_seq_no_self_nest(self, env):
		# seq( inside its own formula → ERR:ILLEGAL NEST
		X = T('X')
		with pytest.raises(ValueError, match="ILLEGAL NEST"):
			parse_line(toks(
				T('seq('), T('seq('), X, COMMA, X, COMMA, 1, COMMA, 2, R_PAREN,
				COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
			), env)

	def test_seq_allows_normal_nesting(self, env):
		# sum(seq(...)) is fine — only seq inside seq is forbidden
		X = T('X')
		parse_line(toks(T('sum('), T('seq('), X, COMMA, X, COMMA, 1, COMMA, 4, R_PAREN, R_PAREN), env)
		assert env.ans == approx(10)

	def test_sigma_no_self_nest(self, env):
		# Σ( inside its own formula → ERR:ILLEGAL NEST
		X = T('X')
		with pytest.raises(ValueError, match="ILLEGAL NEST"):
			parse_line(toks(
				T('Σ('), T('Σ('), X, COMMA, X, COMMA, 1, COMMA, 2, R_PAREN,
				COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
			), env)

	def test_fnint_no_self_nest(self, env):
		# fnInt( inside its own integrand → ERR:ILLEGAL NEST
		X = T('X')
		with pytest.raises(ValueError, match="ILLEGAL NEST"):
			parse_line(toks(
				T('fnInt('), T('fnInt('), X, COMMA, X, COMMA, 0, COMMA, 1, R_PAREN,
				COMMA, X, COMMA, 0, COMMA, 1, R_PAREN
			), env)

	def test_nderiv_one_level_ok(self, env):
		# nDeriv( inside nDeriv( once is allowed
		X = T('X')
		parse_line(toks(
			T('nDeriv('), T('nDeriv('), X, SQ, COMMA, X, COMMA, X, R_PAREN,
			COMMA, X, COMMA, 1, R_PAREN
		), env)
		assert env.ans == approx(2.0, rel=1e-3)

	def test_nderiv_two_levels_raises(self, env):
		# nDeriv( inside nDeriv( inside nDeriv( → ERR:ILLEGAL NEST
		X = T('X')
		with pytest.raises(ValueError, match="ILLEGAL NEST"):
			parse_line(toks(
				T('nDeriv('),
				T('nDeriv('), T('nDeriv('), X, COMMA, X, COMMA, X, R_PAREN,
				COMMA, X, COMMA, X, R_PAREN,
				COMMA, X, COMMA, 1, R_PAREN
			), env)

	def test_expr_no_self_nest(self, env):
		# expr( evaluating a string that itself calls expr( → ERR:ILLEGAL NEST
		str1 = T('Str1')
		# Directly store TiString([expr(, Str1, )]) in Str1 — evaluating it calls expr again
		str1.variable.set(env, TiString([T('expr('), str1, R_PAREN]))
		with pytest.raises(ValueError, match="ILLEGAL NEST"):
			parse_line([T('expr('), str1, R_PAREN], env)

	def test_expr_nest_depth_resets(self, env):
		# After a successful expr( call, the guard is back to 0 — can call again
		str1 = T('Str1')
		parse_line([QUOTE, T('1'), ADD, T('2'), QUOTE, STORE, str1], env)
		parse_line([T('expr('), str1, R_PAREN], env)
		assert env.ans == approx(3.0)
		parse_line([T('expr('), str1, R_PAREN], env)   # second call — must not raise
		assert env.ans == approx(3.0)


# ── Thunk capture: commas inside nested delimiters ────────────────────────────

class TestThunkCapture:
	"""Verify that _capture_subgroup and _capture_opener correctly skip over
	commas inside list literals, matrix literals, and string literals so they
	are not mistaken for argument separators."""

	def test_list_literal_in_seq_formula(self, env):
		# seq({1,2,3}(X), X, 1, 3) — commas inside {} must not split the thunk
		# {1,2,3}(X) indexes the list at position X: result is {1,2,3}
		X = T('X')
		parse_line(toks(
			T('seq('), L_BRACE, 1, COMMA, 2, COMMA, 3, R_BRACE, L_PAREN, X, R_PAREN,
			COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
		), env)
		assert env.ans == TiList([1, 2, 3])

	def test_multi_arg_func_in_seq_formula(self, env):
		# seq(max(X,10), X, 8, 12) — commas inside max(...) must not split the thunk
		X = T('X')
		parse_line(toks(
			T('seq('), T('max('), X, COMMA, 10, R_PAREN,
			COMMA, X, COMMA, 8, COMMA, 12, R_PAREN
		), env)
		assert env.ans == TiList([10, 10, 10, 11, 12])

	def test_string_literal_in_seq_formula(self, env):
		# seq(length("a,b"), X, 1, 3) — the comma in the string must not split the thunk
		# "a,b" has length 3; result should be {3,3,3}
		X = T('X')
		parse_line(toks(
			T('seq('), T('length('), QUOTE, T('a'), T(','), T('b'), QUOTE, R_PAREN,
			COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
		), env)
		assert env.ans == TiList([3, 3, 3])

	def test_colon_inside_thunk_raises(self, env):
		# seq(X:5, X, 1, 3) — colon crosses a statement boundary; rejected at capture time
		X = T('X')
		with pytest.raises(ParseError, match="arguments"):
			parse_line(toks(
				T('seq('), X, COLON, 5, COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
			), env)

	def test_store_inside_thunk_raises(self, env):
		# store inside a formula is a statement-level construct; rejected at capture time
		X = T('X')
		with pytest.raises(ParseError, match="arguments"):
			parse_line(toks(
				T('seq('), X, STORE, T('A'), COMMA, X, COMMA, 1, COMMA, 3, R_PAREN
			), env)


class TestSeqIncrement:
	"""seq( raises a clear error when start/end/step are inconsistent."""

	def test_zero_step_raises(self, env):
		X = T('X')
		with pytest.raises(ValueError, match="zero"):
			parse_line(toks(T('seq('), X, COMMA, X, COMMA, 1, COMMA, 5, COMMA, 0, R_PAREN), env)

	def test_positive_step_start_after_end_raises(self, env):
		X = T('X')
		with pytest.raises(ValueError, match="start.*end|end.*start"):
			parse_line(toks(T('seq('), X, COMMA, X, COMMA, 5, COMMA, 1, R_PAREN), env)

	def test_negative_step_start_before_end_raises(self, env):
		X = T('X')
		with pytest.raises(ValueError, match="start.*end|end.*start"):
			parse_line(toks(T('seq('), X, COMMA, X, COMMA, 1, COMMA, 5, COMMA, NEG, 1, R_PAREN), env)

	def test_equal_start_end_is_fine(self, env):
		X = T('X')
		parse_line(toks(T('seq('), X, COMMA, X, COMMA, 3, COMMA, 3, R_PAREN), env)
		assert env.ans == TiList([3])

	def test_negative_step_descending_is_fine(self, env):
		X = T('X')
		parse_line(toks(T('seq('), X, COMMA, X, COMMA, 3, COMMA, 1, COMMA, NEG, 1, R_PAREN), env)
		assert env.ans == TiList([3, 2, 1])
