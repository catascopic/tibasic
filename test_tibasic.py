"""Test suite for the TI-BASIC interpreter."""

import math
import pytest
from pytest import approx, mark

import purefunctions as pf
from environment import Environment
from modes import AngleMode
from errors import (
	TiSyntaxError, IllegalNestError, DomainError, DimMismatchError,
	StatError, IncrementError, DataTypeError, InvalidDimError, ArgumentError,
	UndefinedError,
)
import catalog
from titoken import Token
from catalog import ALL_TOKENS, get_token, NEWLINE
from tiobjects import TiList, TiMatrix, TiString


# ── Helpers ───────────────────────────────────────────────────────────────────


lookup = {_t.text.strip().replace(' ', '_'): _t for _t in ALL_TOKENS}
lookup['~'] = catalog.NEG
lookup['@'] = catalog.STORE
lookup['e'] = catalog.SCI_E
lookup['$'] = catalog.LIST_PREFIX
lookup['i'] = catalog.IMAG_I
for i, _tok in enumerate(catalog.LISTS, start=1):
	lookup[f"L{i}"] = _tok
for i, _tok in enumerate(catalog.FUNCTION, start=1):
	lookup[f"Y{i}"] = _tok
for name, value in vars(catalog).items():
	if isinstance(value, Token):
		lookup[name] = value


def toks(code) -> list[Token]:
	tokens = []

	def append_line(line):
		for seg in line.strip().split():
			try:
				tokens.append(lookup[seg])
			except KeyError:
				for c in str(seg):
					tokens.append(lookup[c])

	lines = code.strip().splitlines()
	if lines:
		append_line(lines[0])
		for line in lines[1:]:
			tokens.append(NEWLINE)
			append_line(line)

	return tokens


def calc(items, env: Environment | None = None):
	"""Evaluate a token sequence and return Ans."""
	if env is None:
		env = Environment()
	env.run(toks(items))
	return env.ans


def run(src: str, env: Environment | None = None) -> Environment:
	"""Run a token sequence and return the environment (for side-effect inspection)."""
	if env is None:
		env = Environment()
	env.run(toks(src))
	return env


def var(env: Environment, name: str):
	"""Read any variable by its TI-BASIC name (e.g. 'A', '[A]', 'L1', 'Str1').

	Returns the raw stored value (None if never written), bypassing auto-init
	side-effects and UndefinedError.  Intended for test assertions only.
	"""
	if name.startswith('$'):
		return env.user_lists[name[1:]]
	tok = lookup[name]
	if tok.variable is None:
		raise TypeError(f"{name!r} is not a variable")
	return tok.variable(env).value


@pytest.fixture
def deg():
	e = Environment()
	e.angle_mode = AngleMode.DEG
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
	def test_sq_postfix(self):   assert calc('7 SQ') == 49
	def test_xroot(self):        assert calc('4 XTH_ROOT 256') == approx(4)
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
			calc('1e Ans')

	def test_rejects_double_sci(self):
		with pytest.raises(TiSyntaxError):
			calc('1e1e1')

	def test_multi_degrees(self, deg):
		assert calc("1e2°2'°2", deg) == approx(200 + (2/30))

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


# ── Logic ─────────────────────────────────────────────────────────────────────

class TestLogic:
	def test_and_tt(self):  assert calc('1 and 1') == 1
	def test_and_tf(self):  assert calc('1 and 0') == 0
	def test_or_ff(self):   assert calc('0 or 0') == 0
	def test_or_tf(self):   assert calc('1 or 0') == 1
	def test_xor_tt(self):  assert calc('1 xor 1') == 0
	def test_xor_tf(self):  assert calc('1 xor 0') == 1


# ── Combinatorics ─────────────────────────────────────────────────────────────

class TestCombinatorics:
	def test_factorial(self):  assert calc('5!') == 120
	def test_ncr(self):        assert calc('10 nCr 3') == 120
	def test_npr(self):        assert calc('5 nPr 3') == 60
	def test_lcm(self):        assert calc('lcm( 12,8') == 24
	def test_gcd(self):        assert calc('gcd( 12,8') == 4
	def test_remainder(self):  assert calc('remainder( 17,5') == 2


# ── Comparison operators ───────────────────────────────────────────────────────

class TestComparisonOps:
	"""Comparison operators: scalar float return type and list vectorization."""

	# Return type must be float (not bool or int)
	def test_true_is_float(self):   assert type(calc('1=1')) is float
	def test_false_is_float(self):  assert type(calc('1=2')) is float

	# Scalar results
	def test_eq_true(self):   assert calc('2=2')  == 1.0
	def test_eq_false(self):  assert calc('1=2')  == 0.0
	def test_ne_true(self):   assert calc('1≠2')  == 1.0
	def test_ne_false(self):  assert calc('2≠2')  == 0.0
	def test_lt_true(self):   assert calc('1<2')  == 1.0
	def test_lt_false(self):  assert calc('2<1')  == 0.0
	def test_le_equal(self):  assert calc('2≤2')  == 1.0
	def test_le_less(self):   assert calc('1≤2')  == 1.0
	def test_le_false(self):  assert calc('3≤2')  == 0.0
	def test_gt_true(self):   assert calc('3>2')  == 1.0
	def test_gt_false(self):  assert calc('1>2')  == 0.0
	def test_ge_equal(self):  assert calc('2≥2')  == 1.0
	def test_ge_greater(self): assert calc('3≥2') == 1.0
	def test_ge_false(self):  assert calc('1≥2')  == 0.0

	# List vectorization
	def test_eq_list_scalar(self):
		assert calc('{1,2,3}=2').data == [0.0, 1.0, 0.0]

	def test_ne_list_scalar(self):
		assert calc('{1,2,3}≠2').data == [1.0, 0.0, 1.0]

	def test_lt_list_scalar(self):
		assert calc('{1,2,3}<2').data == [1.0, 0.0, 0.0]

	def test_le_list_scalar(self):
		assert calc('{1,2,3}≤2').data == [1.0, 1.0, 0.0]

	def test_gt_list_scalar(self):
		assert calc('{1,2,3}>2').data == [0.0, 0.0, 1.0]

	def test_ge_list_scalar(self):
		assert calc('{1,2,3}≥2').data == [0.0, 1.0, 1.0]

	def test_eq_list_list(self):
		assert calc('{1,2}={1,3}').data == [1.0, 0.0]

	def test_eq_scalar_list(self):
		assert calc('2={2,1}').data == [1.0, 0.0]

	def test_eq_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('{1,2}={1,2,3}')


class TestLogicVectorized:
	"""Logical operators (and/or/xor) vectorized over lists."""

	def test_and_list_scalar(self):
		assert list(calc('{1,0,1} and 1')) == [1, 0, 1]

	def test_and_list_list(self):
		assert list(calc('{1,0} and {1,1}')) == [1, 0]

	def test_or_list_scalar(self):
		assert list(calc('{1,0} or 0')) == [1, 0]

	def test_or_list_list(self):
		assert list(calc('{1,0} or {0,1}')) == [1, 1]

	def test_xor_list_list(self):
		assert list(calc('{1,0} xor {0,1}')) == [1, 1]

	def test_xor_list_scalar(self):
		assert list(calc('{1,0,1} xor 1')) == [0, 1, 0]


class TestCombinatoricsVectorized:
	"""nCr and nPr vectorized over lists."""

	def test_ncr_list_scalar(self):
		assert list(calc('{5,6} nCr 2')) == [10, 15]

	def test_npr_list_scalar(self):
		assert list(calc('{4,5} nPr 2')) == [12, 20]

	def test_ncr_scalar_list(self):
		assert list(calc('6 nCr {1,2,3}')) == [6, 15, 20]


class TestOpVectorized:
	"""op_vectorized operators (^ and ˣ√) with list operands."""

	def test_power_list_exponent(self):
		assert calc('{4,9}^0.5').data == approx([2.0, 3.0])

	def test_power_list_base(self):
		assert calc('2^{2,3,4}').data == approx([4.0, 8.0, 16.0])

	def test_power_list_list(self):
		assert calc('{2,3}^{3,2}').data == approx([8.0, 9.0])

	def test_xth_root_list_radicand(self):
		# 2 ˣ√ {4, 9} = {√4, √9} = {2, 3}
		assert calc('2 XTH_ROOT {4,9}').data == approx([2.0, 3.0])

	def test_xth_root_list_degree(self):
		# {2,3} ˣ√ 8 = {√8, ∛8} = {2√2, 2}
		assert calc('{2,3} XTH_ROOT 8').data == approx([8 ** 0.5, 2.0])


# ── List and matrix arithmetic with both operands non-scalar ──────────────────

class TestListArithmetic:
	"""Arithmetic operators with two list operands (via TiList magic methods)."""

	def test_add_list_list(self):
		assert calc('{1,2,3}+{4,5,6}').data == [5, 7, 9]

	def test_sub_list_list(self):
		assert calc('{5,3,1}-{1,1,1}').data == [4, 2, 0]

	def test_mul_list_list(self):
		assert calc('{2,3,4}*{5,6,7}').data == [10, 18, 28]

	def test_div_list_list(self):
		assert calc('{6,9,12}/{2,3,4}').data == approx([3.0, 3.0, 3.0])

	def test_add_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('{1,2}+{1,2,3}')

	def test_mul_list_scalar(self):
		assert calc('{1,2,3}*2').data == [2, 4, 6]

	def test_add_scalar_list(self):
		# scalar on left — exercises __radd__
		assert calc('10+{1,2,3}').data == [11, 12, 13]

	def test_sub_scalar_list(self):
		# scalar on left — exercises __rsub__
		assert calc('10-{1,2,3}').data == [9, 8, 7]


class TestMatrixArithmetic:
	"""Arithmetic operators with two matrix operands."""

	def test_add_matrix_matrix(self):
		assert calc('[[1,2][3,4]]+[[5,6][7,8]]').data == [[6, 8], [10, 12]]

	def test_sub_matrix_matrix(self):
		assert calc('[[5,6][7,8]]-[[1,2][3,4]]').data == [[4, 4], [4, 4]]

	def test_mul_matrix_scalar(self):
		assert calc('[[1,2][3,4]]*2').data == [[2, 4], [6, 8]]

	def test_mul_scalar_matrix(self):
		# scalar * matrix — exercises __rmul__
		assert calc('3*[[1,2][3,4]]').data == [[3, 6], [9, 12]]

	def test_mul_matrix_matrix(self):
		# matrix * matrix is matrix multiplication (not element-wise)
		# [[1,2][3,4]] * [[1,0][0,1]] = [[1,2][3,4]]
		assert calc('[[1,2][3,4]]*[[1,0][0,1]]').data == [[1, 2], [3, 4]]

	def test_add_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('[[1,2]]+[[1,2][3,4]]')


# ── String concatenation ───────────────────────────────────────────────────────

class TestStringConcat:
	"""String + String via TiString.__add__."""

	def test_concat_result(self):
		assert str(calc('"HELLO"+"WORLD"')) == 'HELLOWORLD'

	def test_concat_empty_left_raises(self):
		# TI-84 raises ERR:INVALID DIM when either operand is an empty string
		with pytest.raises(InvalidDimError):
			calc('""+\"ABC"')

	def test_concat_empty_right_raises(self):
		with pytest.raises(InvalidDimError):
			calc('"ABC"+""')

	def test_concat_stored_and_retrieved(self):
		env = run('"FOO"+"BAR"@ Str1')
		assert str(var(env, 'Str1')) == 'FOOBAR'


# ── Parser features ───────────────────────────────────────────────────────────

class TestParserFeatures:
	def test_variable_store_retrieve(self):
		env = run('3@A')
		assert calc('A', env) == 3

	def test_ans(self):
		env = run('5')
		run('Ans +1', env)
		assert env.ans == 6

	def test_colon_separator(self):
		env = Environment()
		assert calc('3@A:A*2', env) == 6

	def test_list_literal(self):
		assert calc('{1,2,3').data == [1, 2, 3]

	def test_list_index(self):
		env = run('{1,2,3@ L1')
		assert calc('L1 (2', env) == 2

	def test_matrix_literal(self):
		result = calc('[[1,2][3,4]]')
		assert isinstance(result, TiMatrix)
		assert result.data == [[1, 2], [3, 4]]

	def test_matrix_index(self):
		env = run('[[1,2][3,4]]@ [A]')
		assert calc('[A] (2,1', env) == 3

	def test_string_literal(self):
		env = run('"HI"')
		assert str(env.ans) == "HI"

	def test_dms_degree_in_rad_mode(self):
		# 90° in radian mode = π/2
		assert calc('90°') == approx(math.pi / 2)

	def test_dms_degree_in_deg_mode(self, deg):
		# 90° in degree mode = 90 (no conversion)
		assert calc('90°', deg) == 90

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

	def test_expr(self):
		# expr("1+2") evaluates the string as code
		env = run('expr( "1+2"')
		assert env.ans == approx(3)

	def test_inv_postfix(self):
		# [[1,2][3,4]]¹ gives the inverse
		result = calc('[[1,2][3,4]] INV')
		assert result.data == approx_mat([[-2, 1], [1.5, -0.5]])

	def test_transpose_postfix(self):
		result = calc('[[1,2][3,4]] TRANSPOSE')
		assert result.data == [[1, 3], [2, 4]]


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

	def test_unclosed_matrix_then_colon_index(self):
		# [[1:Ans(1,1  →  first segment produces [[1]], second indexes it → 1
		env = run('[[1: Ans (1,1')
		assert env.ans == 1

	def test_unclosed_list_then_colon_sum(self):
		# {1,2,3:sum(Ans  →  Ans=6
		env = Environment()
		assert calc('{1,2,3: sum( Ans', env) == 6

	def test_unclosed_fn_args(self):
		# max(3,7  →  7 (trailing ) omitted)
		assert calc('max( 3,7') == 7

	def test_nested_unclosed(self):
		# abs(~(3+4  →  7
		assert calc('abs( ~(3+4') == 7


# ── Colon-separated statements ────────────────────────────────────────────────

class TestColonStatements:
	def test_colon_ans_is_last(self):
		# 1→A:2  →  Ans=2, A=1
		env = run('1@A :2')
		assert env.ans == 2
		assert var(env, 'A') == 1

	def test_colon_store_then_read(self):
		# 5→A:A*3  →  Ans=15
		env = Environment()
		assert calc('5@A:A*3', env) == 15

	def test_colon_two_stores(self):
		# 1→A:3→B  →  A=1, B=3, Ans=3
		env = run('1@A:3@B')
		assert var(env, 'A') == 1
		assert var(env, 'B') == 3
		assert env.ans == 3

	def test_colon_three_segments(self):
		# 1:2:3  →  Ans=3
		env = Environment()
		assert calc('1:2:3', env) == 3

	def test_colon_ans_carries_across(self):
		# 7:Ans+1  →  Ans=8  (Ans from segment 1 is visible in segment 2)
		env = Environment()
		assert calc('7: Ans +1', env) == 8

	def test_colon_store_does_not_clobber_a(self):
		# 1→A:2  →  A must still be 1 after Ans becomes 2
		env = run('1@A:2')
		assert calc('A', env) == 1

	def test_colon_list_then_index(self):
		# {10,20,30}→L₁:L₁(2)  →  Ans=20
		env = Environment()
		assert calc('{10,20,30@ L1 : L1 (2', env) == 20


# ── Thunk capture: commas inside nested delimiters ────────────────────────────

class TestThunkCapture:
	"""Verify that _capture_subgroup and _capture_opener correctly skip over
	commas inside list literals, matrix literals, and string literals so they
	are not mistaken for argument separators."""

	def test_list_literal_in_seq_formula(self):
		# seq(sum({1,2,X}),X,1,3) — commas inside {} must not split the thunk
		env = run('seq( sum( {1,2,X}),X,1,3')
		assert env.ans.data == [4, 5, 6]

	def test_matrix_literal_in_seq_formula(self):
		# seq(sum({1,2,X}),X,1,3) — commas inside {} must not split the thunk
		env = run('seq( det( [[1,2][X,4]]),X,1,3')
		assert env.ans.data == [2, 0, -2]

	def test_multi_arg_func_in_seq_formula(self):
		# seq(max(X,10), X, 8, 12) — commas inside max(...) must not split the thunk
		env = run('seq( max( X,10),X,8,12)')
		assert env.ans.data == [10, 10, 10, 11, 12]

	def test_string_literal_in_seq_formula(self):
		# seq(length("a,b"), X, 1, 3) — the comma in the string must not split the thunk
		# "a,b" has length 3; result should be {3,3,3}
		env = run('seq( length( "a,b"),X,1,3')
		assert env.ans.data == [3, 3, 3]

	def test_colon_inside_thunk_raises(self):
		# seq(X:5, X, 1, 3) — colon terminates the statement; seq sees only 'X' as its
		# formula and then fails on the missing variable argument (ERR:ARGUMENT on hardware)
		with pytest.raises(ArgumentError):
			calc('seq( X:5,X,1,3)')

	def test_store_inside_thunk_raises(self):
		# store inside a formula is a statement-level construct; rejected at capture time
		with pytest.raises(TiSyntaxError):
			calc('seq( 5@A,X,1,3)')

	def test_store_cannot_be_quoted_in_thunk(self):
		with pytest.raises(TiSyntaxError):
			calc('seq( length( "5@A"),X,1,3)')


class TestSeqIncrement:
	"""seq( raises a clear error when start/end/step are inconsistent."""

	def test_zero_step_raises(self):
		with pytest.raises(IncrementError):
			calc('seq( X,X,1,5,0')

	def test_positive_step_start_after_end_raises(self):
		with pytest.raises(IncrementError):
			calc('seq( X,X,5,1')

	def test_negative_step_start_before_end_raises(self):
		with pytest.raises(IncrementError):
			calc('seq( X,X,1,5,~1')

	def test_equal_start_end_is_fine(self):
		assert calc('seq( X,X,3,3').data == [3]

	def test_negative_step_descending_is_fine(self):
		assert calc('seq( X,X,3,1,~1').data == [3, 2, 1]


# ── Storing to dim( ───────────────────────────────────────────────────────────

class TestStoreDim:
	def test_store_dim_list_create(self):
		# 5→dim(L₁)  →  L₁ becomes {0,0,0,0,0}
		env = run('5@ dim( L1')
		assert var(env, 'L1').data == [0, 0, 0, 0, 0]

	def test_store_dim_list_expand(self):
		# {1,2,3}→L₁ : 5→dim(L₁)  →  L₁ = {1,2,3,0,0}
		env = run('{1,2,3@ L1')
		run('5@ dim( L1', env)
		assert var(env, 'L1').data == [1, 2, 3, 0, 0]

	def test_store_dim_list_shrink(self):
		# {1,2,3,4,5}→L₁ : 3→dim(L₁)  →  L₁ = {1,2,3}
		env = run('{1,2,3,4,5@ L1')
		run('3@ dim( L1', env)
		assert var(env, 'L1').data == [1, 2, 3]

	def test_store_dim_matrix_create(self):
		# {2,3}→dim([A])  →  [A] becomes 2×3 of zeros
		env = run('{2,3@ dim( [A]')
		assert var(env, '[A]').data == 2 * [3 * [0]]

	def test_store_dim_matrix_resize_preserves(self):
		# Build [[1,2][3,4]], then resize to 3×3; original values survive, new cells = 0
		env = run('[[1,2][3,4@ [A]')
		run('{3,3@ dim( [A]', env)
		assert var(env, '[A]').data == [[1, 2, 0], [3, 4, 0], [0, 0, 0]]

	def test_dim_read_list(self):
		# dim({1,2,3,4}) = 4  (reading, not storing)
		result = calc('dim( {1,2,3,4}')
		assert result == 4

	def test_dim_read_matrix(self):
		# dim([[1,2,3][4,5,6]]) = {2,3}
		result = calc('dim( [[1,2,3][4,5,6')
		assert result.data == [2, 3]

	def test_store_invalid_dim_doesnt_create_list(self):
		env = Environment()
		with pytest.raises(InvalidDimError):
			calc('1.5 @ dim( $BAD', env)
		assert env.user_lists == {}

	def test_store_bad_datatype_dim_doesnt_create_matrix(self):
		env = Environment()
		with pytest.raises(DataTypeError):
			calc('5 @ dim( [A]', env)
		assert var(env, '[A]') is None

	def test_store_invalid_dim_doesnt_create_matrix(self):
		env = Environment()
		with pytest.raises(InvalidDimError):
			calc('{0,2} @ dim( [A]', env)
		assert var(env, '[A]') is None


# ── Undefined variable behavior ───────────────────────────────────────────────

class TestUndefinedVars:
	"""Numeric vars default to 0; all other var types raise UndefinedError."""

	# ── Numeric: always 0 ─────────────────────────────────────────────────────

	def test_numeric_var_defaults_to_zero(self):
		assert calc('A') == 0

	# ── Lists ─────────────────────────────────────────────────────────────────

	def test_read_undefined_list(self):
		with pytest.raises(UndefinedError):
			calc('L1')

	def test_store_index_1_undefined_list(self):
		# Storing to L₁(1) when L₁ is undefined auto-creates it.
		env = run('7@ L1 (1')
		assert var(env, 'L1').data == [7]

	def test_store_index_gt1_undefined_list(self):
		# Storing to L₁(2) when L₁ is undefined is out of range.
		with pytest.raises(InvalidDimError):
			calc('7@ L1 (2')

	def test_store_index2_doesnt_create(self):
		env = Environment()
		with pytest.raises(InvalidDimError):
			calc('1@$BAD(2', env)
		assert env.user_lists == {}

	# ── Matrices ─────────────────────────────────────────────────────────────

	def test_read_undefined_matrix(self):
		with pytest.raises(UndefinedError):
			calc('[A]')

	def test_store_indexed_undefined_matrix(self):
		# Unlike lists, matrices do not auto-create on indexed store.
		with pytest.raises(UndefinedError):
			calc('1@ [A] (1,1')

	# ── Strings ──────────────────────────────────────────────────────────────

	def test_read_undefined_string(self):
		with pytest.raises(UndefinedError):
			calc('Str1')

	# ── User-named lists ─────────────────────────────────────────────────────

	def test_read_undefined_user_list(self):
		with pytest.raises(UndefinedError):
			calc('$FOO')

	def test_store_index_1_undefined_user_list(self):
		env = run('7@ $FOO (1')
		assert env.user_lists['FOO'].data == [7]

	def test_store_index_gt1_undefined_user_list(self):
		with pytest.raises(InvalidDimError):
			calc('7@ $FOO (2')


# ── Copy vars ────────────────────────────────────────────────────────────────

class TestCopyVars:

	def test_copy_string(self):
		env = run('"ABC"@ Str1')
		run('Str1 @ Str2', env)
		assert str(var(env, 'Str1')) == 'ABC'
		assert str(var(env, 'Str2')) == 'ABC'

	def test_copy_list(self):
		env = run('{1,2,3}@ L1')
		run('L1 @ L2', env)
		assert var(env, 'L1').data == [1, 2, 3]
		assert var(env, 'L2').data == [1, 2, 3]
		run('4@ L1 (1', env)
		assert var(env, 'L1').data == [4, 2, 3]
		assert var(env, 'L2').data == [1, 2, 3]

	def test_copy_matrix(self):
		env = run('[[1,2][3,4]]@ [A]')
		run('[A] @ [B]', env)
		assert var(env, '[A]').data == [[1, 2], [3, 4]]
		assert var(env, '[B]').data == [[1, 2], [3, 4]]
		run('5@ [A] (2,1', env)
		assert var(env, '[A]').data == [[1, 2], [5, 4]]
		assert var(env, '[B]').data == [[1, 2], [3, 4]]


# ── Store data type ───────────────────────────────────────────────────────────

class TestDataType:

	def test_store_wrong_data_type(self):
		with pytest.raises(DataTypeError):
			calc('"A"@A')
		with pytest.raises(DataTypeError):
			calc('[[1]]@A')
		# {1}@A does work, stores to user list

		with pytest.raises(DataTypeError):
			calc('1@ L1')
		with pytest.raises(DataTypeError):
			calc('[[1]]@ L1')
		with pytest.raises(DataTypeError):
			calc('"A"@ L1')

		with pytest.raises(DataTypeError):
			calc('1@ [A]')
		with pytest.raises(DataTypeError):
			calc('{1}@ [A]')
		with pytest.raises(DataTypeError):
			calc('"A"@ [A]')

		with pytest.raises(DataTypeError):
			calc('1@ Str1')
		with pytest.raises(DataTypeError):
			calc('{1}@ Str1')
		with pytest.raises(DataTypeError):
			calc('[[1]]@ Str1')

		with pytest.raises(DataTypeError):
			calc('1@ Y1')
		with pytest.raises(DataTypeError):
			calc('{1}@ Y1')
		with pytest.raises(DataTypeError):
			calc('[[1]]@ Y1')

	def test_str_var_to_equ(self):
		env = run('"X" @ Str1')
		run('Str1 @ Y1', env)
		assert var(env, 'Str1').tokens == toks('X')

	def test_str_to_equ(self):
		env = run('"X"@ Y1')
		assert var(env, 'Y1').tokens == toks('X')

	def test_equ_to_str(self):
		env = run('"X"@ Y1')
		with pytest.raises(DataTypeError):
			calc('Y1 @ Str1', env)
	
	def test_plus(self):
		assert calc('1+1') == 2
		assert calc('1+{1,2}').data == [2, 3]
		assert calc('1+[[1][2]]').data == [[2], [3]]
		with pytest.raises(DataTypeError):
			calc('1+"A"')
		
		assert calc('{1,2}+1').data == [2, 3]
		assert calc('{1,2}+{3,4}').data == [4, 6]
		with pytest.raises(DataTypeError):
			calc('{1,2}+[[1][2]]')
		with pytest.raises(DataTypeError):
			calc('{1,2}+"A"')
		
		assert calc('[[1][2]]+3').data == [[4], [5]]
		with pytest.raises(DataTypeError):
			assert calc('[[1][2]]+{3,4}')
		assert calc('[[1][2]]+[[3][4]]').data == [[4], [6]]
		with pytest.raises(DataTypeError):
			calc('[[1][2]]+"A"')
		
		with pytest.raises(DataTypeError):
			assert calc('"A"+1')
		with pytest.raises(DataTypeError):
			assert calc('"A"+{1,2}')
		with pytest.raises(DataTypeError):
			assert calc('"A"+[[1][2]]')
		assert str(calc('"A"+"B"')) == 'AB'


# ── Nesting and combinations ──────────────────────────────────────────────────

class TestNesting:
	def test_sum_of_seq(self):
		# sum(seq(X²,X,1,5))  =  1+4+9+16+25 = 55
		assert calc('sum( seq( X^2,X,1,5') == approx(55)

	def test_seq_with_step(self):
		# seq(X,X,1,9,2)  =  {1,3,5,7,9}
		assert calc('seq( X,X,1,9,2').data == approx([1, 3, 5, 7, 9])

	def test_seq_negative_step(self):
		# seq(X,X,5,1,~1)  =  {5,4,3,2,1}
		assert calc('seq( X,X,5,1,~1').data == approx([5, 4, 3, 2, 1])

	def test_sigma(self):
		# Σ(X,X,1,10)  =  55
		assert calc('Σ( X,X,1,10') == approx(55)

	def test_sigma_formula(self):
		# Σ(X²,X,1,4)  =  1+4+9+16 = 30
		assert calc('Σ( X^2,X,1,4') == approx(30)

	def test_nderiv(self):
		# nDeriv(X²,X,3) ≈ 6  (derivative of x² at x=3)
		assert calc('nDeriv( X^2,X,3') == approx(6, rel=1e-4)

	def test_fnint(self):
		# fnInt(X²,X,0,3) ≈ 9  (∫₀³ x² dx = 9)
		assert calc('fnInt( X^2,X,0,3') == approx(9, rel=1e-4)

	def test_abs_of_neg_expr(self):
		# abs(~(3+4))  =  7
		assert calc('abs( ~(3+4') == 7

	def test_max_of_list_expr(self):
		# max({3,1,4,1,5})  =  5
		assert calc('max( {3,1,4,1,5') == 5

	def test_nested_arithmetic_functions(self):
		# round(1/6, 3)  =  0.167
		assert calc('round( 1/6,3') == approx(0.167)

	def test_list_arithmetic_then_sum(self):
		# {1,2,3}*2  =  {2,4,6}, then sum({2,4,6}) = 12
		env = run('{1,2,3}*2@ L1')
		assert calc('sum( L1', env) == 12

	def test_matrix_power_then_det(self):
		# det([[1,1][0,1]]²)  =  det([[1,2][0,1]])  =  1
		assert calc('det( [[1,1][0,1]]^2') == approx(1)

	def test_string_concat_then_length(self):
		# "AB"+"CD" stored in Str1, then length(Str1) = 4
		env = run('"AB"+"CD"@ Str1')
		assert calc('length( Str1', env) == 4

	def test_cumsum_then_max(self):
		# max(cumSum({1,2,3,4}))  =  max({1,3,6,10})  =  10
		assert calc('max( cumSum( {1,2,3,4') == 10

	def test_expr_evaluates_string(self):
		# Build "2+3" dynamically as a string stored in Str1, then expr(Str1) = 5
		env = run('"2+3"@ Str1')
		assert calc('expr( Str1', env) == approx(5)

	def test_ans_index_or_mul_list(self):
		# {10,20,30}→Ans  (via plain eval), then Ans(2)  =  20
		env = run('{10,20,30}')
		run('Ans (2)', env)
		assert env.ans == 20

	def test_ans_index_or_mul_scalar(self):
		# 7→Ans, then Ans(3)  =  21  (scalar * 3)
		env = run('7')
		run('Ans (3)', env)
		assert env.ans == 21

	def test_ans_index_matrix(self):
		# [[1,2][3,4]]→Ans, then Ans(2,1) = 3
		env = run('[[1,2][3,4')
		run('Ans (2,1', env)
		assert env.ans == 3

	def test_seq_preserves_variable(self):
		# X=99 before seq; seq restores X=99 afterward
		env = run('99@X')
		run('seq( X,X,1,3', env)
		run('X', env)
		assert env.ans == 99


# ── Illegal nesting (ERR:ILLEGAL NEST) ───────────────────────────────────────

class TestIllegalNest:
	"""Each restricted function raises IllegalNestError if nested beyond its limit."""

	def test_seq_no_self_nest(self):
		# seq( inside its own formula → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			calc('seq( seq( X,X,1,2),X,1,3)')

	def test_seq_allows_normal_nesting(self):
		# sum(seq(...)) is fine — only seq inside seq is forbidden
		env = run('sum( seq( X,X,1,4))')
		assert env.ans == approx(10)

	def test_sigma_no_self_nest(self):
		# Σ( inside its own formula → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			calc('Σ( Σ( X,X,1,2),X,1,3')

	def test_fnint_no_self_nest(self):
		# fnInt( inside its own integrand → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			calc('fnInt( fnInt( X,X,0,1),X,0,1')

	def test_nderiv_one_level_ok(self):
		# nDeriv( inside nDeriv( once is allowed
		env = run('nDeriv( nDeriv( X²,X,X),X,1')
		assert env.ans == approx(2, rel=1e-3)

	def test_nderiv_two_levels_raises(self):
		# nDeriv( inside nDeriv( inside nDeriv( → ERR:ILLEGAL NEST
		with pytest.raises(IllegalNestError):
			calc('nDeriv( nDeriv( nDeriv( X,X,X),X,X),X,1')

	def test_expr_no_self_nest(self):
		# expr( evaluating a string that itself calls expr( → ERR:ILLEGAL NEST
		env = run('" expr( Str1 )"@ Str1')
		with pytest.raises(IllegalNestError):
			calc('expr( Str1', env)

	def test_expr_nest_depth_resets(self):
		# After a successful expr( call, the guard is back to 0 — can call again
		env = run('"1+2"@ Str1')
		assert calc('expr( Str1', env) == 3
		assert calc('expr( Str1', env) == 3   # second call — must not raise

	def test_expr_explicit_close(self):
		env = run('expr( "2+2@A')
		assert var(env, 'A') == 4


# ── Syntax ────────────────────────────────────────────────────────────────────

class TestSyntax:

	def test_clockon_bunch(self):
		# ClockOn is a weird command that doesn't end the line
		calc('ClockOn ClockOn')

	def test_consecutive_commands_without_separator_raises(self):
		# Two commands with no COLON or NEWLINE between them must be a syntax error.
		with pytest.raises(TiSyntaxError):
			calc('Normal Float')


# ── Complex xor ──────────────────────────────────────────────────────────────

class TestCompleXor:

	def test_xor(self):
		env = run('55@A:99@B')
		run('int( log( 2) INV log( max( {A,B', env)
		run('2^ cumSum( binomcdf( Ans ,0', env)
		assert calc('sum( Ans .5(1= abs( int( 2 fPart( Ans INV (A+Bi', env) == 84

	def test_xor2(self):
		env = run('55@A:99@B')
		run('seq( 2^N,N,8,1,~1@ L1', env)
		run('.5 sum( L1 *(1= abs( int( 2 fPart( (A+Bi)/ L1 @F', env)
		assert calc('F', env) == 84


# ── Vars (A–Z variable storage) ───────────────────────────────────────────────

class TestVars:
	def test_all_num_vars(self):
		env = run("""
		1@A
		2@B
		3@C
		4@D
		5@E
		6@F
		7@G
		8@H
		9@I
		10@J
		11@K
		12@L
		13@M
		14@N
		15@O
		16@P
		17@Q
		18@R
		19@S
		20@T
		21@U
		22@V
		23@W
		24@X
		25@Y
		26@Z
		27@θ
		""")
		assert calc('QWERTYUIOPASDFGHJKLZXCVBNMθ', env) == approx(math.factorial(27))
