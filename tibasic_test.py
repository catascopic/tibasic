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
	ANS, INV, SQ, TRANSPOSE,
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

	def test_dbd(self):
		assert pf.dbd(12.2524, 12.3124) == 6.0     # Dec 25 → Dec 31 2024

	def test_dbd_negative(self):
		assert pf.dbd(12.3124, 12.2524) == -6.0    # reversed

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
