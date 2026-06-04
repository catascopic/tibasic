"""Tests for specific TI-BASIC built-in functions."""

import math
import pytest
from pytest import approx

from environment import Environment
from modes import AngleMode, ComplexMode
from errors import (
	TiSyntaxError, DomainError, DimMismatchError,
	DataTypeError, InvalidDimError, ArgumentError, NonRealAnsError,
)
import catalog
from tiobjects import TiList, TiMatrix, TiString
from test_tibasic import toks, calc, run, var, approx_mat


@pytest.fixture
def deg():
	e = Environment()
	e.angle_mode = AngleMode.DEG
	return e


# ── Numeric functions ─────────────────────────────────────────────────────────

class TestNumericFunctions:
	def test_abs_pos(self):       assert calc('abs( 5') == 5
	def test_abs_neg(self):       assert calc('abs( ~5') == 5
	def test_abs_complex(self):   assert calc('abs( 3+4i') == approx(5)
	def test_round_2dp(self):     assert calc('round( 3.14159,2') == approx(3.14)
	def test_round_default(self): assert calc(f'round( {1/3}') == approx(1/3)
	def test_i_part_pos(self):    assert calc('iPart( 3.9') == 3
	def test_i_part_neg(self):    assert calc('iPart( ~3.9') == -3   # toward zero
	def test_f_part_pos(self):    assert calc('fPart( 3.7') == approx(0.7)
	def test_f_part_neg(self):    assert calc('fPart( ~3.7') == approx(-0.7)
	def test_int_floor_pos(self): assert calc('int( 3.9') == 3
	def test_int_floor_neg(self): assert calc('int( ~3.1') == -4     # floor, not truncate
	def test_sqrt(self):                  assert calc('SQRT 9') == approx(3)
	def test_sqrt_negative_complex(self):
		env = Environment(); env.complex_mode = ComplexMode.A_PLUS_BI
		assert calc('SQRT ~1', env) == approx(1j)
	def test_sqrt_negative_real(self):    pytest.raises(NonRealAnsError, calc, 'SQRT ~1')
	def test_cbrt(self):                  assert calc('CBRT 8') == approx(2)
	def test_ln(self):                    assert calc(f'ln( {math.e}') == approx(1)
	def test_ln_negative_complex(self):
		env = Environment(); env.complex_mode = ComplexMode.A_PLUS_BI
		assert calc('ln( ~1', env) == approx(1j * math.pi)
	def test_ln_negative_real(self):      pytest.raises(NonRealAnsError, calc, 'ln( ~1')
	def test_ln_zero(self):               pytest.raises(DomainError, calc, 'ln( 0')
	def test_log(self):                   assert calc('log( 100') == approx(2)
	def test_log_negative_real(self):     pytest.raises(NonRealAnsError, calc, 'log( ~1')
	def test_log_zero(self):              pytest.raises(DomainError, calc, 'log( 0')
	def test_exp(self):           assert calc('𝑒^( 0') == approx(1)
	def test_pow10(self):         assert calc('⑽^( 3') == approx(1000)
	def test_not_false(self):     assert calc('not( 0') == 1
	def test_not_true(self):      assert calc('not( 5') == 0


# ── Trig ──────────────────────────────────────────────────────────────────────

class TestTrig:
	# RAD mode (default)
	def test_sin(self):   assert calc(f'sin( {math.pi / 6}') == approx(0.5)
	def test_cos(self):   assert calc(f'cos( 0') == approx(1)
	def test_tan(self):   assert calc(f'tan( {math.pi / 4}') == approx(1)
	def test_asin(self):  assert calc(f'sin¹( 0.5') == approx(math.pi / 6)
	def test_acos(self):  assert calc(f'cos¹( 1') == approx(0)
	def test_atan(self):  assert calc(f'tan¹( 1') == approx(math.pi / 4)
	# DEG mode
	def test_sin_deg(self, deg):  assert calc('sin( 30',   deg) == approx(0.5)
	def test_cos_deg(self, deg):  assert calc('cos( 0',    deg) == approx(1)
	def test_tan_deg(self, deg):  assert calc('tan( 45',   deg) == approx(1)
	def test_asin_deg(self, deg): assert calc('sin¹( 0.5', deg) == approx(30)
	def test_acos_deg(self, deg): assert calc('cos¹( 1',   deg) == approx(0)
	def test_atan_deg(self, deg): assert calc('tan¹( 1',   deg) == approx(45)
	# Hyperbolics stay in purefunctions, no angle mode
	def test_sinh(self):  assert calc('sinh( 0') == approx(0)
	def test_cosh(self):  assert calc('cosh( 0') == approx(1)
	def test_tanh(self):  assert calc('tanh( 0') == approx(0)
	def test_asinh(self): assert calc('sinh¹( 0') == approx(0)
	def test_acosh(self): assert calc('cosh¹( 1') == approx(0)
	def test_atanh(self): assert calc('tanh¹( 0') == approx(0)


# ── List operations ───────────────────────────────────────────────────────────

class TestListOperations:

	def test_store_index_1_undefined_list(self):
		env = run('1@ L1 (1')
		assert var(env, 'L1').data == [1]

	def test_store_index_2_undefined_list(self):
		with pytest.raises(InvalidDimError):
			calc('1@ L1 (2')

	def test_augment(self):
		assert list(calc('augment( {1,2},{3,4}')) == [1, 2, 3, 4]

	def test_cum_sum(self):
		assert list(calc('cumSum( {1,2,3,4}')) == [1, 3, 6, 10]

	def test_delta_list(self):
		assert list(calc('ΔList( {1,3,6,10}')) == [2, 3, 4]

	def test_sum_full(self):
		assert calc('sum( {1,2,3,4,5}') == 15

	def test_sum_partial(self):
		assert calc('sum( {1,2,3,4,5},2,4') == 9

	def test_prod(self):
		assert calc('prod( {1,2,3,4}') == 24

	def test_mean(self):
		assert calc('mean( {1,2,3,4,5}') == approx(3)

	def test_median_odd(self):
		assert calc('median( {3,1,4,1,5}') == 3

	def test_median_even(self):
		assert calc('median( {1,2,3,4}') == 2.5

	def test_max_list(self):
		assert calc('max( {3,1,4,1,5,9}') == 9

	def test_min_list(self):
		assert calc('min( {3,1,4,1,5,9}') == 1

	def test_max_two_scalars(self):
		assert calc('max( 3,7') == 7

	def test_variance(self):
		# known result: sum of squared deviations / (n-1)
		assert calc('variance( {2,4,4,4,5,5,7,9}') == approx(32 / 7)

	def test_stddev(self):
		assert calc('stdDev( {2,4,4,4,5,5,7,9}') == approx(math.sqrt(32 / 7))

	def test_dim_list(self):
		assert calc('dim( {1,2,3}') == 3

	def test_vectorized_add(self):
		assert list(TiList([1, 2, 3]) + TiList([4, 5, 6])) == [5, 7, 9]

	def test_vectorized_scalar(self):
		assert list(TiList([2, 4, 6]) / 2) == [1, 2, 3]


# ── Matrix operations ─────────────────────────────────────────────────────────

class TestMatrixOperations:
	def test_det_2x2(self):
		assert calc('det( [[1,2][3,4]]') == approx(-2)

	def test_det_identity(self):
		assert calc('det( identity( 4') == approx(1)

	def test_identity(self):
		result = calc('identity( 3')
		assert result.data == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

	def test_transpose(self):
		result = calc('[[1,2,3][4,5,6]] TRANSPOSE')
		assert result.data == [[1, 4], [2, 5], [3, 6]]

	def test_matmul(self):
		assert calc('[[1,2][3,4]]*[[5,6][7,8]]').data == [[19, 22], [43, 50]]

	def test_inv_roundtrip(self):
		result = calc('[[1,2][3,4]]*[[1,2][3,4]] INV')
		assert result.data == approx_mat([[1, 0], [0, 1]])

	def test_dim_matrix(self):
		assert list(calc('dim( [[1,2,3][4,5,6]]')) == [2, 3]

	def test_augment_matrix(self):
		assert calc('augment( [[1,2][3,4]],[[5][6]]').data == [[1, 2, 5], [3, 4, 6]]

	def test_rref_solve(self):
		# 2x + y = 5, x - y = 1  →  x=2, y=1
		result = calc('rref( [[2,1,5][1,~1,1]]')
		assert result.data == approx_mat([[1, 0, 2], [0, 1, 1]])


class TestMatrixRowOps:
	def test_row_swap(self):
		result = calc('rowSwap( [[1,2][3,4][5,6]],1,3')
		assert result.data == [[5, 6], [3, 4], [1, 2]]

	def test_row_plus(self):
		result = calc('row+( [[1,2][3,4][5,6]],1,2')
		assert result.data == [[1, 2], [4, 6], [5, 6]]

	def test_times_row(self):
		result = calc('*row( 3,[[1,2][3,4][5,6]],1')
		assert result.data == [[3, 6], [3, 4], [5, 6]]

	def test_times_row_plus(self):
		result = calc('*row+( 2,[[1,2][3,4][5,6]],1,2')
		assert result.data == [[1, 2], [5, 8], [5, 6]]


# ── Matr►list( / List►matr( ───────────────────────────────────────────────────

class TestMatrToList:
	def test_single_column_first(self):
		# Matr►list([A], 1, L1) extracts column 1 into L1
		env = run('Matr►list( [[1,2][3,4][5,6]],1, L1')
		assert var(env, 'L1').data == [1, 3, 5]

	def test_single_column_second(self):
		# Matr►list([A], 2, L1) extracts column 2
		env = run('Matr►list( [[1,2][3,4][5,6]],2, L1')
		assert var(env, 'L1').data == [2, 4, 6]

	def test_multi_list_all_columns(self):
		# Matr►list([A], L1, L2) — each list gets one column
		env = run('[[1,2][3,4]]@ [A]')
		run('Matr►list( [A] , L1 , L2', env)
		assert var(env, 'L1').data == [1, 3]
		assert var(env, 'L2').data == [2, 4]

	def test_multi_list_single_column_matrix(self):
		# 1-column matrix: one list var is enough
		env = run('[[7][8][9')
		run('Matr►list( Ans , L1', env)
		assert var(env, 'L1').data == [7, 8, 9]

	def test_column_out_of_range(self):
		with pytest.raises(InvalidDimError):
			calc('Matr►list( [[1,2][3,4]],3, L1')

	def test_fewer_lists_than_columns(self):
		# 2-column matrix, only 1 list destination → only column 1 is filled
		env = run('Matr►list( [[1,2][3,4]], L1')
		assert var(env, 'L1').data == [1, 3]
		assert var(env, 'L2') is None  # L2 untouched

	def test_extra_lists_ignored(self):
		# 2-column matrix but 3 list destinations → extra list is simply ignored
		env = run('Matr►list( [[1,2][3,4]], L1 , L2 , L3 )')
		assert var(env, 'L1').data == [1, 3]
		assert var(env, 'L2').data == [2, 4]
		assert var(env, 'L3') is None  # L3 never written

	def test_non_matrix_raises(self):
		with pytest.raises(DataTypeError):
			calc('Matr►list( {1,2,3},1, L1')


class TestListToMatr:
	def test_basic_two_lists(self):
		# List►matr({1,3},{2,4},[A]) → [[1,2],[3,4]]  (lists become columns)
		env = run('List►matr( {1,3},{2,4}, [A]')
		assert var(env, '[A]').data == [[1, 2], [3, 4]]

	def test_single_list(self):
		env = run('List►matr( {5,6,7}, [A]')
		assert var(env, '[A]').data == [[5], [6], [7]]

	def test_unequal_lengths_pads_zero(self):
		# Shorter list gets zero-padded to match the longest
		env = run('List►matr( {1,2,3},{4,5}, [A]')
		assert var(env, '[A]').data == [[1, 4], [2, 5], [3, 0]]

	def test_roundtrip_with_matr_to_list(self):
		# Store a matrix, round-trip through List►matr
		env = run('Matr►list( [[10,20][30,40]], L1 , L2')
		run('List►matr( L1 , L2 , [A]', env)
		assert var(env, '[A]').data == [[10, 20], [30, 40]]

	def test_non_list_raises(self):
		# Scalar where a list is expected → DataTypeError
		with pytest.raises(DataTypeError):
			calc('List►matr( 5,{1,2}, [A]')

	def test_no_matrix_raises(self):
		# Missing matrix destination → ArgumentError
		with pytest.raises(ArgumentError):
			calc('List►matr( {1,2},{3,4}')


# ── User-named lists (ᴸNAME) ─────────────────────────────────────────────────

class TestUserLists:
	"""User-named lists stored and read back via the $ (LIST_PREFIX) proxy.
	'$AB' is tokenised as LIST_PREFIX + VAR_A + VAR_B, which the parser reads
	as the user list named 'AB'.
	"""

	# ── Basic store / retrieve ────────────────────────────────────────────────

	def test_store_and_retrieve_with_prefix(self):
		env = run('{1,2,3}@$AB')
		assert env.user_lists['AB'].data == [1, 2, 3]

	def test_store_bare_name(self):
		# When the value is already a list, →NAME (no ᴸ) stores as a user list
		env = run('{4,5}@AB')
		assert env.user_lists['AB'].data == [4, 5]

	def test_single_char_name(self):
		env = run('{7,8}@$Z')
		assert env.user_lists['Z'].data == [7, 8]

	def test_with_numbers(self):
		env = run('{1,2@A1234')
		assert env.user_lists['A1234'].data == [1, 2]

	def test_overwrite(self):
		env = run('{1,2,3}@$AB')
		run('{9,8}@$AB', env)
		assert env.user_lists['AB'].data == [9, 8]

	def test_fail_leading_number(self):
		with pytest.raises(TiSyntaxError):
			calc('{1}@$7A')

	def test_store_index_1_undefined_user_list(self):
		env = run('1@$FOO(1')
		assert env.user_lists['FOO'].data == [1]

	def test_store_index_2_undefined_user_list(self):
		with pytest.raises(InvalidDimError):
			calc('1@$FOO(2')

	# ── Indexing ──────────────────────────────────────────────────────────────

	def test_index_read(self):
		env = run('{10,20,30}@$AB')
		assert calc('$AB(2)', env) == 20

	def test_index_first_element(self):
		env = run('{10,20,30}@$AB')
		assert calc('$AB(1)', env) == 10

	def test_index_write(self):
		env = run('{1,2,3}@$AB')
		run('99@$AB(2)', env)
		assert env.user_lists['AB'].data == [1, 99, 3]

	# ── Arithmetic — behaves the same as L1–L6 ───────────────────────────────

	def test_scalar_div(self):
		env = run('{2,4,6}@$AB')
		assert calc('$AB/2', env).data == [1, 2, 3]

	def test_add_two_user_lists(self):
		env = run('{1,2,3}@$AB')
		run('{4,5,6}@$CD', env)
		assert calc('$AB+$CD', env).data == [5, 7, 9]

	def test_add_user_list_and_regular_list(self):
		env = run('{1,2,3}@$AB')
		run('{4,5,6}@ L1', env)
		assert calc('$AB+ L1', env).data == [5, 7, 9]

	def test_dim_mismatch_raises(self):
		env = run('{1,2}@$AB')
		run('{3,4,5}@$CD', env)
		with pytest.raises(DimMismatchError):
			calc('$AB+$CD', env)

	# ── Aggregate functions ───────────────────────────────────────────────────

	def test_dim(self):
		env = run('{1,2,3,4}@$AB')
		assert calc('dim( $AB', env) == 4

	def test_sum(self):
		env = run('{1,2,3}@$AB')
		assert calc('sum( $AB', env) == 6

	def test_max(self):
		env = run('{3,1,4,1,5}@$AB')
		assert calc('max( $AB', env) == 5

	def test_augment_two_user_lists(self):
		env = run('{1,2}@$AB')
		run('{3,4}@$CD', env)
		assert calc('augment( $AB,$CD', env).data == [1, 2, 3, 4]

	def test_cum_sum(self):
		env = run('{1,2,3}@$AB')
		assert calc('cumSum( $AB', env).data == [1, 3, 6]

	def test_seq_result_stored_in_user_list(self):
		env = run('seq( X,X,1,5)@$AB')
		assert calc('$AB', env).data == [1, 2, 3, 4, 5]

	# ── Matr►list / List►matr ────────────────────────────────────────────────

	def test_matr_to_list_single_column(self):
		# Matr►list([A], 1, ᴸAB) — extract column 1 into the user list
		env = run('[[1,2][3,4][5,6]]@ [A]')
		run('Matr►list( [A] ,1,$AB', env)
		assert env.user_lists['AB'].data == [1, 3, 5]

	def test_matr_to_list_multi(self):
		# Matr►list([A], ᴸAB, ᴸCD) — extract each column into a user list
		env = run('[[1,2][3,4]]@ [A]')
		run('Matr►list( [A] ,$AB,$CD', env)
		assert env.user_lists['AB'].data == [1, 3]
		assert env.user_lists['CD'].data == [2, 4]

	def test_matr_to_list_mixed(self):
		# Matr►list([A], L1, ᴸAB) — one regular list, one user list
		env = run('[[1,2][3,4][5,6]]@ [A]')
		run('Matr►list( [A] , L1 ,$AB', env)
		assert var(env, 'L1').data == [1, 3, 5]
		assert env.user_lists['AB'].data == [2, 4, 6]

	def test_list_to_matr_from_user_lists(self):
		# List►matr(ᴸAB, ᴸCD, [A])
		env = run('{1,3,5}@$AB')
		run('{2,4,6}@$CD', env)
		run('List►matr( $AB,$CD, [A]', env)
		assert var(env, '[A]').data == [[1, 2], [3, 4], [5, 6]]

	def test_list_to_matr_mixed(self):
		# Mix a regular list and a user list as sources
		env = run('{1,3,5}@ L1')
		run('{2,4,6}@$AB', env)
		run('List►matr( L1 ,$AB, [A]', env)
		assert var(env, '[A]').data == [[1, 2], [3, 4], [5, 6]]

	def test_roundtrip(self):
		# Store a matrix → Matr►list → List►matr → should recover original
		env = run('[[10,20][30,40]]@ [A]')
		run('Matr►list( [A] ,$AB,$CD', env)
		run('List►matr( $AB,$CD, [A] ', env)
		assert var(env, '[A]').data == [[10, 20], [30, 40]]

	def test_six_char(self):
		with pytest.raises(TiSyntaxError):
			calc('{1}@$ABCDEF')


# ── ►Eff( / ►Nom( ────────────────────────────────────────────────────────────

class TestEffNom:
	# ── ►Eff( ─────────────────────────────────────────────────────────────────

	def test_eff_example_from_docs(self):
		# ►Eff(7.5, 12) → 7.763259886  (the docs say 7.663 but that's a typo)
		assert calc('►Eff( 7.5,12') == approx(7.763259886, rel=1e-8)

	def test_eff_annual(self):
		# 1 compounding period is a pass-through
		assert calc('►Eff( 10,1') == 10

	def test_eff_quarterly(self):
		assert calc('►Eff( 8,4') == approx(100 * ((1 + 0.08 / 4) ** 4 - 1), rel=1e-12)

	def test_eff_zero_rate(self):
		assert calc('►Eff( 0,12') == approx(0)

	def test_eff_domain_zero_periods(self):
		with pytest.raises(DomainError):
			calc('►Eff( 10,0')

	def test_eff_domain_negative_periods(self):
		with pytest.raises(DomainError):
			calc('►Eff( 10,~1')

	def test_eff_domain_rate_at_minus_100(self):
		with pytest.raises(DomainError):
			calc('►Eff( ~100,12')

	def test_eff_domain_rate_below_minus_100(self):
		with pytest.raises(DomainError):
			calc('►Eff( ~200,12')

	def test_eff_minus_100_with_one_period_is_passthrough(self):
		# cp=1 is always a pass-through, even for extreme rates
		assert calc('►Eff( ~100,1') == -100

	# ── ►Nom( ─────────────────────────────────────────────────────────────────

	def test_nom_example_from_docs(self):
		# ►Nom(10, 12) → 9.568968515
		assert calc('►Nom( 10,12') == approx(9.568968515, rel=1e-8)

	def test_nom_annual(self):
		# 1 compounding period is a pass-through
		assert calc('►Nom( 10,1') == 10

	def test_nom_quarterly(self):
		assert calc('►Nom( 8,4') == approx(100 * 4 * ((1.08 ** (1/4)) - 1), rel=1e-12)

	def test_nom_zero_rate(self):
		assert calc('►Nom( 0,12') == approx(0)

	def test_nom_domain_zero_periods(self):
		with pytest.raises(DomainError):
			calc('►Nom( 10,0')

	def test_nom_domain_negative_periods(self):
		with pytest.raises(DomainError):
			calc('►Nom( 10,~1')

	def test_nom_domain_rate_at_minus_100(self):
		with pytest.raises(DomainError):
			calc('►Nom( ~100,12')

	def test_nom_domain_rate_below_minus_100(self):
		with pytest.raises(DomainError):
			calc('►Nom( ~200,12')

	def test_nom_minus_100_with_one_period_is_passthrough(self):
		assert calc('►Nom( ~100,1') == -100

	# ── Roundtrip ─────────────────────────────────────────────────────────────

	def test_roundtrip_eff_then_nom(self):
		# ►Eff then ►Nom should recover the original rate
		x = calc('►Eff( 7.5,12')
		assert calc(f'►Nom( {x},12') == approx(7.5, rel=1e-10)

	def test_roundtrip_nom_then_eff(self):
		x = calc('►Nom( 10,12')
		assert calc(f'►Eff( {x},12') == approx(10, rel=1e-10)


# ── npv( / irr( / bal( / ΣPrn( / ΣInt( ──────────────────────────────────────

class TestNpv:
	def test_basic_no_freq(self):
		# npv(5, 500, {1250,1333,1575,1100,1900}) — example from docs
		r = 1.05
		expected = 500 + 1250/r + 1333/r**2 + 1575/r**3 + 1100/r**4 + 1900/r**5
		assert calc('npv( 5,500,{1250,1333,1575,1100,1900}') == approx(expected, rel=1e-10)

	def test_zero_rate(self):
		# At 0% all cash flows just sum
		assert calc('npv( 0,50,{100,200,300}') == approx(650)

	def test_with_freq(self):
		# npv(8, 0, {200,300}, {2,3}) same as npv(8, 0, {200,200,300,300,300})
		with_freq = calc('npv( 8,0,{200,300},{2,3}')
		flat      = calc('npv( 8,0,{200,200,300,300,300}')
		assert with_freq == approx(flat, rel=1e-10)

	def test_freq_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('npv( 5,0,{100,200},{1}')

	def test_data_type_error_on_complex_rate(self):
		with pytest.raises(DataTypeError):
			calc('npv( 1i,0,{100}')

	def test_single_cash_flow(self):
		# npv(10, -100, {110}) = -100 + 110/1.1 = 0
		assert calc('npv( 10,~100,{110}') == approx(0, abs=1e-10)


class TestIrr:
	def test_simple_one_period(self):
		# Invest 100, get 110 back → IRR = 10%
		assert calc('irr( ~100,{110}') == approx(10.0, rel=1e-6)

	def test_two_periods(self):
		# npv(irr) = 0 check
		rate = calc('irr( ~1000,{500,600}')
		assert calc(f'npv( {rate},~1000,{{500,600}}') == approx(0, abs=1e-4)

	def test_with_freq(self):
		# Same cash flow with and without frequencies
		assert calc('irr( ~100,{110},{1}') == approx(10.0, rel=1e-6)

	def test_no_positive_solution_raises(self):
		# All positive cash flows → no sign change → no real positive IRR
		with pytest.raises(DomainError):
			calc('irr( 100,{100,100}')

	def test_npv_is_zero_at_irr(self):
		rate = calc('irr( ~900,{300,400,500}')
		assert calc(f'npv( {rate},~900,{{300,400,500}}') == approx(0, abs=1e-3)


class TestBal:
	"""Uses a 30-year mortgage: PV=100000, I%=8/12 per month, N=360, PMT≈-733.76."""

	@pytest.fixture
	def mortgage(self):
		env = Environment()
		env.pv.value    = 100_000
		env.i_pct.value = 8 / 12           # monthly rate as percentage
		env.n_tvm.value = 360
		# Exact PMT for zero FV
		r = env.i_pct.value / 100
		env.pmt.value = -env.pv.value * r / (1 - (1 + r) ** -env.n_tvm.value)
		return env

	def test_bal_zero_is_pv(self, mortgage):
		assert calc('bal( 0', mortgage) == approx(mortgage.pv.value)

	def test_bal_180(self, mortgage):
		# After 15 years (180 payments) — docs quote ~$76781.55
		assert calc('bal( 180', mortgage) == approx(76781.55, rel=1e-3)

	def test_bal_360_is_zero(self, mortgage):
		# After all payments the loan is paid off
		assert calc('bal( 360', mortgage) == approx(0, abs=0.01)

	def test_bal_negative_raises(self, mortgage):
		with pytest.raises(DomainError):
			calc('bal( ~1', mortgage)

	def test_bal_zero_interest(self):
		env = Environment()
		env.pv.value    = 1200
		env.i_pct.value = 0
		env.pmt.value   = -100
		assert calc('bal( 6', env) == approx(600)

	def test_bal_with_rounding(self, mortgage):
		# With roundvalue=2, result should still be roughly correct
		result = calc('bal( 180,2', mortgage)
		assert result == approx(76781.55, rel=0.01)


class TestSigmaPrn:
	@pytest.fixture
	def mortgage(self):
		env = Environment()
		env.pv.value    = 100_000
		env.i_pct.value = 8 / 12
		env.n_tvm.value = 360
		r = env.i_pct.value / 100
		env.pmt.value = -env.pv.value * r / (1 - (1 + r) ** -env.n_tvm.value)
		return env

	def test_sigma_prn_first_60(self, mortgage):
		# Docs quote ≈ -$4930.14 for first 5 years
		assert calc('Σprn( 1,60', mortgage) == approx(-4930.14, rel=1e-2)

	def test_sigma_prn_equals_bal_difference(self, mortgage):
		# ΣPrn(n1,n2) = bal(n2) - bal(n1-1)
		from envfunctions import _bal
		sprn = calc('Σprn( 13,24', mortgage)
		expected = _bal(mortgage, 24) - _bal(mortgage, 12)
		assert sprn == approx(expected, rel=1e-10)

	def test_sigma_prn_full_term(self, mortgage):
		# Principal paid over all 360 payments should equal -PV
		sprn = calc('Σprn( 1,360', mortgage)
		assert sprn == approx(-mortgage.pv.value, rel=1e-6)


class TestSigmaInt:
	@pytest.fixture
	def mortgage(self):
		env = Environment()
		env.pv.value    = 100_000
		env.i_pct.value = 8 / 12
		env.n_tvm.value = 360
		r = env.i_pct.value / 100
		env.pmt.value = -env.pv.value * r / (1 - (1 + r) ** -env.n_tvm.value)
		return env

	def test_sigma_int_first_60(self, mortgage):
		# Docs quote ≈ -$39095.73 for first 5 years
		assert calc('ΣInt( 1,60', mortgage) == approx(-39095.73, rel=1e-2)

	def test_prn_plus_int_equals_total_payments(self, mortgage):
		# ΣPrn + ΣInt should equal total PMT paid
		n1, n2 = 1, 60
		sprn = calc('Σprn( 1,60', mortgage)
		sint = calc('ΣInt( 1,60', mortgage)
		total_pmt = n2 * mortgage.pmt.value  # 60 payments
		assert sprn + sint == approx(total_pmt, rel=1e-8)

	def test_sigma_int_full_term(self, mortgage):
		# Total interest = total paid - principal = 360*PMT - (-PV) = 360*PMT + PV
		sint = calc('ΣInt( 1,360', mortgage)
		expected = 360 * mortgage.pmt.value + mortgage.pv.value  # negative (outflow)
		assert sint == approx(expected, rel=1e-6)


# ── Complex numbers ───────────────────────────────────────────────────────────

class TestComplex:
	def test_real(self):         assert calc('real( 3+4i') == 3
	def test_imag(self):         assert calc('imag( 3+4i') == 4
	def test_conj(self):         assert calc('conj( 3+4i') == 3-4j
	def test_angle(self):        assert calc('angle( i') == approx(math.pi / 2)
	def test_real_on_real(self): assert calc('real( 5') == 5
	def test_imag_on_real(self): assert calc('imag( 5') == 0


# ── Coordinate conversions ────────────────────────────────────────────────────

class TestCoordinates:
	def test_r_pr(self):
		assert calc('R►Pr( 3,4') == approx(5)

	def test_r_ptheta(self):
		assert calc('R►Pθ( 1,0') == approx(0)

	def test_p_rx(self):
		assert calc('P►Rx( 5,0') == approx(5)

	def test_p_ry(self):
		assert calc(f'P►Ry( 5,{math.pi / 2}') == approx(5)

	def test_roundtrip(self):
		r, theta = 5, math.pi / 3
		x = calc(f'P►Rx( {r},{theta}')
		y = calc(f'P►Ry( {r},{theta}')
		assert calc(f'R►Pr( {x},{y}') == approx(r)
		assert calc(f'R►Pθ( {x},{y}') == approx(theta)

	def test_r_ptheta_deg(self, deg): assert calc('R►Pθ( 1,0',  deg) == approx(0)
	def test_p_rx_deg(self, deg):    assert calc('P►Rx( 5,0',  deg) == approx(5)
	def test_p_ry_deg(self, deg):    assert calc('P►Ry( 5,90', deg) == approx(5)


# ── Probability distributions ─────────────────────────────────────────────────

class TestDistributions:
	def test_normalcdf_median(self):
		assert calc('normalcdf( ~1e99,0') == approx(0.5, rel=1e-4)

	def test_normalcdf_68_rule(self):
		assert calc('normalcdf( ~1,1') == approx(0.6827, rel=1e-3)

	def test_inv_norm_median(self):
		assert calc('invNorm( 0.5') == approx(0, abs=1e-6)

	def test_inv_norm_roundtrip(self):
		x = calc('invNorm( 0.9')
		assert calc(f'normalcdf( ~1e99,{x}') == approx(0.9, rel=1e-4)

	def test_normalpdf_peak(self):
		# PDF peaks at x=μ with value 1/sqrt(2π)
		assert calc('normalpdf( 0') == approx(1 / math.sqrt(2 * math.pi))

	def test_binompdf(self):
		# P(X=5) for Binomial(10, 0.5) = C(10,5)/2^10
		assert calc('binompdf( 10,0.5,5') == approx(252 / 1024)

	def test_binomcdf_all(self):
		assert calc('binomcdf( 10,0.5,10') == approx(1)

	def test_poissonpdf(self):
		assert calc('poissonpdf( 3,3') == approx(math.exp(-3) * 27 / 6)

	def test_poissoncdf_all(self):
		assert calc('poissoncdf( 3,50') == approx(1)

	def test_geometpdf_first(self):
		# P(X=1) = p
		assert calc('geometpdf( 0.3,1') == approx(0.3)

	def test_geometcdf(self):
		assert calc('geometcdf( 0.5,1') == approx(0.5)

	def test_tcdf_symmetric(self):
		# t-distribution is symmetric; CDF(-∞, 0) = 0.5
		assert calc('tcdf( ~1e9,0,10') == approx(0.5, rel=1e-4)

	def test_chi_sq_cdf_zero(self):
		assert calc('χ²cdf( 0,0,5') == approx(0, abs=1e-6)

	def test_invt_roundtrip(self):
		x = calc('invT( 0.9,10')
		assert calc(f'tcdf( ~1e9,{x},10') == approx(0.9, rel=1e-4)


# ── String functions ──────────────────────────────────────────────────────────

class TestStrings:
	def test_length(self):
		assert calc('length( "HELLO"') == 5

	def test_length_empty(self):
		assert calc('length( ""') == 0

	def test_in_string_found(self):
		assert calc('inString( "HELLO","ELL"') == 2

	def test_in_string_not_found(self):
		assert calc('inString( "HELLO","XYZ"') == 0

	def test_in_string_with_start(self):
		assert calc('inString( "ABAB","AB",3') == 3

	def test_sub(self):
		result = calc('sub( "HELLO",2,3')
		assert str(result) == "ELL"


# ── Date / time ───────────────────────────────────────────────────────────────

class TestDateTime:
	def test_timecnv(self):
		assert list(calc('timeCnv( 3661')) == [0, 1, 1, 1]

	def test_timecnv_days(self):
		# 1 day + 1 hr + 1 min + 1 sec = 86400+3600+60+1 = 90061
		assert list(calc('timeCnv( 90061')) == [1, 1, 1, 1]

	def test_timecnv_negative(self):
		assert list(calc('timeCnv( ~3661')) == [0, -1, -1, -1]

	def test_dayofwk_wednesday(self):
		assert calc('dayOfWk( 2024,12,25') == 4   # Wednesday

	def test_dayofwk_sunday(self):
		assert calc('dayOfWk( 2023,1,1') == 1     # Sunday

	def test_dbd_mmddyy(self):
		# MM.DDYY: Dec 25 → Dec 31 2024
		assert calc('dbd( 12.2524,12.3124') == 6

	def test_dbd_negative(self):
		assert calc('dbd( 12.3124,12.2524') == -6

	def test_dbd_ddmmyy_leap(self):
		# DDMM.YY: Jan 17 1996 → Jan 17 1997 (1996 is a leap year → 366 days)
		assert calc('dbd( 1701.96,1701.97') == 366

	def test_dbd_mmddyy_leap(self):
		# MM.DDYY same dates — formats can be mixed or used separately
		assert calc('dbd( 1.1796,1.1797') == 366

	def test_dbd_mixed_formats(self):
		# Doc example: dbd(612.07, 2512.07) = 19
		# DDMM.YY: 612.07 → Dec 6 2007; 2512.07 → Dec 25 2007
		assert calc('dbd( 612.07,2512.07') == 19

	def test_dbd_mmddyy_doc_example(self):
		# Doc example: dbd(1.0207, 1.0107) = -1
		# MM.DDYY: Jan 2 2007 → Jan 1 2007
		assert calc('dbd( 1.0207,1.0107') == -1

	def test_dbd_too_many_decimals_mmddyy(self):
		# 5 decimal places in MM.DDYY → ERR:DOMAIN
		with pytest.raises(DomainError):
			calc('dbd( 1.01075,1.0107')

	def test_dbd_too_many_decimals_ddmmyy(self):
		# 3 decimal places in DDMM.YY → ERR:DOMAIN
		with pytest.raises(DomainError):
			calc('dbd( 1701.961,1701.97')

	def test_dbd_ambiguous_integer(self):
		# Integer part 13–99 is invalid
		with pytest.raises(DomainError):
			calc('dbd( 50.0101,51.0101')

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


# ── rand ─────────────────────────────────────────────────────────────────────────

class TestRand:
	def test_rand_no_parens_in_range(self):
		# bare rand produces a single float in [0, 1)
		result = calc('rand')
		assert isinstance(result, float)
		assert 0 <= result < 1

	def test_rand_with_parens_returns_list(self):
		# rand(5) returns a TiList of 5 floats
		result = calc('rand (5)')
		assert isinstance(result, TiList)
		assert len(result) == 5
		assert all(0 <= x < 1 for x in result)

	def test_rand_with_parens_no_close(self):
		# Trailing ) may be omitted
		result = calc('rand (3')
		assert isinstance(result, TiList)
		assert len(result) == 3

	def test_rand_seed_reproducible(self):
		# Store a seed → rand, then same seed → rand again; must match
		env = run('1@ rand')
		run('rand', env)
		first = env.ans
		run('1@ rand', env)
		run('rand', env)
		assert env.ans == first

	def test_rand_implicit_multiply(self):
		# 2rand  ≡  2 * rand()  — result must be in [0, 2)
		env = run('1@ rand')   # fix seed
		run('rand', env)
		single = env.ans
		run('1@ rand', env)   # reset seed
		run('2 rand', env)          # implicit multiply
		assert env.ans == approx(2 * single)

	def test_rand_int(self):
		# randInt(1,6) returns an integer value in [1, 6]
		result = calc('randInt( 1,6)')
		assert result == int(result)
		assert 1 <= result <= 6

	def test_rand_int_list(self):
		# randInt(1,6,10) returns a TiList of 10 ints
		result = calc('randInt( 1,6,10)')
		assert isinstance(result, TiList)
		assert len(result) == 10
		assert all(1 <= x <= 6 for x in result)

	def test_rand_norm(self):
		# randNorm(0,1) returns a float (no guaranteed range, just check type)
		result = calc('randNorm( 0,1)')
		assert isinstance(result, float)

	def test_rand_norm_list(self):
		# randNorm(0,1,5) returns a TiList of 5 floats
		result = calc('randNorm( 0,1,5)')
		assert isinstance(result, TiList)
		assert len(result) == 5


# ── matrix_vectorized functions ────────────────────────────────────────────────

class TestMatrixVectorized:
	"""Functions decorated with @matrix_vectorized applied to a TiMatrix."""

	def test_ipart_matrix(self):
		assert calc('iPart( [[1.7,~1.7]]').data == [[1, -1]]

	def test_int_matrix(self):
		# int( uses floor: floor(-1.7) = -2
		assert calc('int( [[1.7,~1.7]]').data == [[1, -2]]

	def test_fpart_matrix(self):
		assert calc('fPart( [[1.7,2.3]]').data == approx_mat([[0.7, 0.3]])

	def test_round_matrix(self):
		assert calc('round( [[1.567,2.345]],2').data == approx_mat([[1.57, 2.35]])

	def test_abs_matrix(self):
		assert calc('abs( [[~3,4]]').data == [[3, 4]]

	def test_abs_matrix_2x2(self):
		assert calc('abs( [[~1,2][3,~4]]').data == [[1, 2], [3, 4]]
