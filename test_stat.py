"""Tests for TI-BASIC statistical functions."""

import math
import pytest
from pytest import approx

import purefunctions as pf
import tilist
from environment import Environment
from errors import (
	TiSyntaxError, StatError, DimMismatchError, InvalidDimError,
)
from tiobjects import TiList
from test_tibasic import calc, run, var


# ── Stat unit tests (plain-value interface) ───────────────────────────────────

class TestStatUnit:
	"""Direct unit tests for statistical functions via TiFunction.__call__.

	These complement the calc()-based integration tests by isolating the
	numerical algorithms from the parser and token machinery.
	"""

	def _lst(self, *values):
		return TiList(list(values))

	# ── mean ──────────────────────────────────────────────────────────────────

	def test_mean_basic(self):
		assert tilist.mean(self._lst(1, 2, 3, 4, 5)) == approx(3.0)

	def test_mean_weighted(self):
		# [0, 0, 0, 10] → 10/4 = 2.5
		assert tilist.mean(self._lst(0, 10), self._lst(3, 1)) == approx(2.5)

	def test_mean_single(self):
		assert tilist.mean(self._lst(7)) == approx(7.0)

	# ── variance ──────────────────────────────────────────────────────────────

	def test_variance_known(self):
		# Classic dataset; sample variance = 32/7
		assert tilist.variance(self._lst(2, 4, 4, 4, 5, 5, 7, 9)) == approx(32 / 7)

	def test_variance_two_elements(self):
		# mean=2; ((1-2)²+(3-2)²)/(2-1) = 2
		assert tilist.variance(self._lst(1, 3)) == approx(2.0)

	def test_variance_weighted(self):
		# mean=1; 3*(0-1)² + 1*(4-1)² = 12; 12/(4-1) = 4.0
		assert tilist.variance(self._lst(0, 4), self._lst(3, 1)) == approx(4.0)

	# ── stddev ────────────────────────────────────────────────────────────────

	def test_stddev_known(self):
		assert tilist.stddev(self._lst(2, 4, 4, 4, 5, 5, 7, 9)) == approx(math.sqrt(32 / 7))

	def test_stddev_is_sqrt_variance(self):
		lst = self._lst(3, 7, 7, 19)
		assert tilist.stddev(lst) == approx(math.sqrt(tilist.variance(lst)))

	def test_stddev_weighted_is_sqrt_variance(self):
		lst, freq = self._lst(0, 4), self._lst(3, 1)
		assert tilist.stddev(lst, freq) == approx(math.sqrt(tilist.variance(lst, freq)))

	# ── median ────────────────────────────────────────────────────────────────

	def test_median_odd(self):
		assert tilist.median(self._lst(3, 1, 4, 1, 5)) == approx(3.0)

	def test_median_even(self):
		assert tilist.median(self._lst(1, 2, 3, 4)) == approx(2.5)

	def test_median_weighted_odd_total(self):
		# Expanded: [10, 10, 20, 30, 30] → middle is 20
		assert tilist.median(self._lst(10, 20, 30), self._lst(2, 1, 2)) == approx(20.0)

	def test_median_weighted_even_total(self):
		# Expanded: [10, 10, 30, 30] → (10+30)/2 = 20
		assert tilist.median(self._lst(10, 30), self._lst(2, 2)) == approx(20.0)

	# ── normalcdf: 68–95–99.7 rule ────────────────────────────────────────────

	def test_normalcdf_full_range(self):
		assert pf.normalcdf(float('-inf'), float('inf')) == approx(1.0, rel=1e-6)

	def test_normalcdf_median(self):
		assert pf.normalcdf(float('-inf'), 0) == approx(0.5, rel=1e-6)

	def test_normalcdf_one_sigma(self):
		assert pf.normalcdf(-1, 1) == approx(0.6827, rel=1e-3)

	def test_normalcdf_two_sigma(self):
		assert pf.normalcdf(-2, 2) == approx(0.9545, rel=1e-3)

	def test_normalcdf_three_sigma(self):
		assert pf.normalcdf(-3, 3) == approx(0.9973, rel=1e-3)

	# ── tcdf ──────────────────────────────────────────────────────────────────

	def test_tcdf_symmetric(self):
		# t distribution is symmetric around 0
		assert pf.tcdf(float('-inf'), 0, 10) == approx(0.5, rel=1e-6)

	def test_tcdf_full_range(self):
		assert pf.tcdf(float('-inf'), float('inf'), 10) == approx(1.0, rel=1e-6)

	def test_tcdf_converges_to_normal(self):
		# At large df the t-distribution approaches normal; ±1.96 ≈ 95%
		assert pf.tcdf(-1.96, 1.96, 1000) == approx(0.95, rel=1e-2)


# ── Empty-list behaviour ──────────────────────────────────────────────────────

class TestEmptyList:
	"""
	{} is a syntax error.  An empty TiList (via 0→dim(L₁)) causes
	InvalidDimError in every aggregate function except dim(), which returns 0.
	0→dim(L₁) is also rejected with InvalidDimError.
	"""

	@pytest.fixture
	def empty(self):
		"""L1 = empty list, L2 = {1,2}."""
		env = run('{1@ L1')
		run('0@ dim( L1', env)
		run('{1,2@ L2', env)
		return env

	# ── Parser rejects {} ─────────────────────────────────────────────────────

	def test_empty_literal_raises(self):
		with pytest.raises(TiSyntaxError):
			calc('{}')

	def test_empty_literal_unclosed_raises(self):
		# { with nothing before implicit close is equally invalid
		with pytest.raises(TiSyntaxError):
			calc('{')

	# ── dim() is the one exception ────────────────────────────────────────────

	def test_dim_empty_returns_zero(self, empty):
		assert calc('dim( L1', empty) == 0

	def test_store_dim_zero(self):
		env = run('{1,2,3@ L1')
		run('0@ dim( L1', env)
		assert var(env, 'L1').data == []

	# ── Aggregate functions ───────────────────────────────────────────────────

	def test_sum_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('sum( L1', empty)

	def test_prod_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('prod( L1', empty)

	def test_mean_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('mean( L1', empty)

	def test_median_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('median( L1', empty)

	def test_max_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('max( L1', empty)

	def test_min_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('min( L1', empty)

	def test_variance_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('variance( L1', empty)

	def test_stddev_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('stdDev( L1', empty)

	def test_cum_sum_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('cumSum( L1', empty)

	def test_delta_list_empty(self, empty):
		with pytest.raises(InvalidDimError):
			calc('ΔList( L1', empty)

	# ── augment ───────────────────────────────────────────────────────────────

	def test_augment_empty_left(self, empty):
		with pytest.raises(InvalidDimError):
			calc('augment( L1 , L2', empty)

	def test_augment_empty_right(self, empty):
		with pytest.raises(InvalidDimError):
			calc('augment( L2 , L1', empty)

	def test_augment_both_empty(self, empty):
		run('0@ dim( L2', empty)
		with pytest.raises(InvalidDimError):
			calc('augment( L1 , L2', empty)


# ── Stat functions with freq_list ─────────────────────────────────────────────

class TestStatWithFreqList:
	"""mean, median, variance, stddev all accept an optional freq_list second arg."""

	# ── mean ──────────────────────────────────────────────────────────────────

	def test_mean_uniform(self):
		# Uniform weights → same result as plain mean
		assert calc('mean( {1,2,3},{1,1,1}') == approx(2.0)

	def test_mean_weighted(self):
		# [0,0,0,10] → mean = 10/4 = 2.5
		assert calc('mean( {0,10},{3,1}') == approx(2.5)

	def test_mean_integer_counts(self):
		# [1,1,1,2,3,3] → mean = (3+2+6)/6 = 11/6
		assert calc('mean( {1,2,3},{3,1,2}') == approx(11 / 6)

	# ── median ────────────────────────────────────────────────────────────────

	def test_median_odd_total(self):
		# Expanded: [10,10,20,30,30] → middle element is 20
		assert calc('median( {10,20,30},{2,1,2}') == 20

	def test_median_even_total(self):
		# Expanded: [10,10,30,30] → (10+30)/2 = 20
		assert calc('median( {10,30},{2,2}') == approx(20.0)

	def test_median_unsorted_input(self):
		# Must sort by value: {3:1,1:2,2:1} → [1,1,2,3] → (1+2)/2 = 1.5
		assert calc('median( {3,1,2},{1,2,1}') == approx(1.5)

	def test_median_uniform_matches_plain(self):
		plain    = calc('median( {1,2,3,4,5}')
		weighted = calc('median( {1,2,3,4,5},{1,1,1,1,1}')
		assert weighted == approx(plain)

	def test_median_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('median( {1,2,3},{1,1}')

	# ── variance ──────────────────────────────────────────────────────────────

	def test_variance_weighted(self):
		# mean=1; 3*(0-1)² + 1*(4-1)² = 3+9=12; 12/(4-1) = 4.0
		assert calc('variance( {0,4},{3,1}') == approx(4.0)

	def test_variance_uniform_matches_plain(self):
		plain    = calc('variance( {2,4,6}')
		weighted = calc('variance( {2,4,6},{1,1,1}')
		assert weighted == approx(plain)

	def test_variance_total_freq_le_one(self):
		# total freq = 1 → denominator (n-1) = 0
		with pytest.raises(StatError):
			calc('variance( {5},{1}')

	def test_variance_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('variance( {1,2,3},{1,1}')

	# ── stddev ────────────────────────────────────────────────────────────────

	def test_stddev_weighted(self):
		assert calc('stdDev( {0,4},{3,1}') == approx(2.0)

	def test_stddev_uniform_matches_plain(self):
		plain    = calc('stdDev( {2,4,4,4,5,5,7,9}')
		weighted = calc('stdDev( {2,4,4,4,5,5,7,9},{1,1,1,1,1,1,1,1}')
		assert weighted == approx(plain)

	def test_stddev_dim_mismatch(self):
		with pytest.raises(DimMismatchError):
			calc('stdDev( {1,2},{1}')
