"""Tests for memory-management and list commands:
DelVar, SetUpEditor, ClrList, ClrAllLists, SortA(, SortD(, Fill(.
"""

import pytest

from environment import Environment
from errors import TiSyntaxError
from tiobjects import TiList, TiMatrix
from tibasic_test import toks, var, calc


def run(src: str, env: Environment | None = None) -> Environment:
	if env is None:
		env = Environment()
	env.run(toks(src))
	return env


class TestDelVar:

	def test_clears_numeric(self):
		env = run("""
		42@A
		DelVar A
		""")
		assert var(env, 'A') is None

	def test_numeric_autoinits_to_zero_after_delvar(self):
		env = run('42@A')
		run('DelVar A', env)
		assert calc('A', env) == 0

	def test_does_not_update_ans(self):
		env = run('42@A')
		run('DelVar A', env)
		assert env.ans == 42

	def test_clears_standard_list(self):
		env = run("""
		{1,2,3}@ L1
		DelVar L1
		""")
		assert var(env, 'L1') is None

	def test_clears_user_list(self):
		env = run('{9,8}@FOO')
		assert env.user_lists.get('FOO') is not None
		run('DelVar $FOO', env)
		assert env.user_lists.get('FOO') is None

	def test_clears_matrix(self):
		env = run('[[1,2][3,4]]@ [A]')
		run('DelVar [A]', env)
		assert var(env, '[A]') is None

	def test_bunching_chain(self):
		env = run("""
		1@A
		2@B
		DelVar A DelVar B
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') is None

	def test_bunching_with_subsequent_statement(self):
		# DelVar A then 99@B on the same line — B must still be set
		env = run('1@A')
		run('DelVar A 99@B', env)
		assert var(env, 'A') is None
		assert var(env, 'B') == 99


class TestSetUpEditor:

	def test_no_args_creates_all_six_default_lists(self):
		env = run('SetUpEditor')
		assert all(env.lists[i] is not None for i in range(6))

	def test_no_args_default_lists_are_empty(self):
		env = run('SetUpEditor')
		assert all(len(env.lists[i]) == 0 for i in range(6))

	def test_no_args_preserves_existing_data(self):
		env = run('{1,2,3}@ L1')
		run('SetUpEditor', env)
		assert var(env, 'L1').data == [1, 2, 3]

	def test_with_standard_list(self):
		env = run('SetUpEditor L3')
		assert var(env, 'L3') is not None   # L3 created
		assert var(env, 'L1') is None       # L1 untouched

	def test_with_bare_name_creates_user_list(self):
		env = run('SetUpEditor SAVE')
		assert env.user_lists.get('SAVE') is not None

	def test_with_prefix_name_creates_user_list(self):
		env = run('SetUpEditor $DATA')
		assert env.user_lists.get('DATA') is not None

	def test_does_not_overwrite_existing_user_list(self):
		env = run('{7,8,9}@SAVE')
		run('SetUpEditor SAVE', env)
		assert env.user_lists['SAVE'].data == [7, 8, 9]

	def test_multiple_args(self):
		env = run('SetUpEditor L1 , L3 , SAVE')
		assert var(env, 'L1') is not None
		assert var(env, 'L2') is None       # L2 not requested
		assert var(env, 'L3') is not None
		assert env.user_lists.get('SAVE') is not None


# ── ClrList ───────────────────────────────────────────────────────────────────

class TestClrList:

	def test_clears_existing_list_to_empty(self):
		env = run("""
		{1,2,3}@ L1
		ClrList L1
		""")
		assert var(env, 'L1') is not None   # still defined
		assert len(var(env, 'L1')) == 0     # but empty

	def test_dim_returns_zero_after_clrlist(self):
		env = run("""
		{1,2,3}@ L1
		ClrList L1
		""")
		assert calc('dim( L1', env) == 0

	def test_nonexistent_list_is_silent_noop(self):
		env = run('ClrList L2')           # L2 was never set
		assert var(env, 'L2') is None       # still None, not created

	def test_multiple_lists(self):
		env = run("""
		{1,2}@ L1
		{3,4}@ L2
		ClrList L1 , L2
		""")
		assert len(var(env, 'L1')) == 0
		assert len(var(env, 'L2')) == 0

	def test_clears_user_list(self):
		env = run('{5,6,7}@FOO')
		run('ClrList $ FOO', env)
		assert env.user_lists.get('FOO') is not None
		assert len(env.user_lists['FOO']) == 0

	def test_nonexistent_mixed_with_existing(self):
		# L1 exists, L3 does not — L3 must remain None, L1 must be cleared
		env = run('{9,9,9}@ L1')
		run('ClrList L1 , L3', env)
		assert len(var(env, 'L1')) == 0
		assert var(env, 'L3') is None


# ── ClrAllLists ───────────────────────────────────────────────────────────────

class TestClrAllLists:

	def test_clears_all_defined_standard_lists(self):
		env = run("""
		{1,2}@ L1
		{3,4}@ L2
		ClrAllLists
		""")
		assert len(var(env, 'L1')) == 0
		assert len(var(env, 'L2')) == 0

	def test_leaves_undefined_slots_as_none(self):
		env = run("""
		{1,2}@ L1
		ClrAllLists
		""")
		assert var(env, 'L1') is not None   # was defined, now empty
		assert var(env, 'L2') is None       # was never defined

	def test_clears_user_lists(self):
		env = run('{5,6}@FOO')
		run('ClrAllLists', env)
		assert len(env.user_lists['FOO']) == 0

	def test_no_args(self):
		with pytest.raises(TiSyntaxError):
			run('ClrAllLists 5')


class TestSortA:

	def test_sorts_ascending(self):
		env = run("""
		{3,1,4,1,5}@ L1
		SortA( L1
		""")
		assert var(env, 'L1').data == [1, 1, 3, 4, 5]

	def test_already_sorted_unchanged(self):
		env = run("""
		{1,2,3}@ L1
		SortA( L1
		""")
		assert var(env, 'L1').data == [1, 2, 3]

	def test_dependent_list_reordered_consistently(self):
		env = run("""
		{3,1,2}@ L1
		{1,2,3}@ L2
		SortA( L1 , L2
		""")
		assert var(env, 'L1').data == [1, 2, 3]
		assert var(env, 'L2').data == [2, 3, 1]

	def test_two_dependent_lists(self):
		env = run("""
		{3,1,2}@ L1
		{1,2,3}@ L2
		{20,30,10}@ L3
		SortA( L1 , L2 , L3
		""")
		assert var(env, 'L1').data == [1, 2, 3]
		assert var(env, 'L2').data == [2, 3, 1]
		assert var(env, 'L3').data == [30, 10, 20]


class TestSortD:

	def test_sorts_descending(self):
		env = run("""
		{3,1,4,1,5}@ L1
		SortD( L1
		""")
		assert var(env, 'L1').data == [5, 4, 3, 1, 1]

	def test_dependent_list_reordered_consistently(self):
		env = run("""
		{1,3,2}@ L1
		{4,5,6}@ L2
		SortD( L1 , L2
		""")
		assert var(env, 'L1').data == [3, 2, 1]
		assert var(env, 'L2').data == [5, 6, 4]


class TestFill:

	def test_fill_list(self):
		env = run("""
		{1,2,3}@ L1
		Fill( 7, L1
		""")
		assert var(env, 'L1').data == [7, 7, 7]

	def test_fill_list_with_zero(self):
		env = run("""
		{5,6,7}@ L1
		Fill( 0, L1
		""")
		assert var(env, 'L1').data == [0, 0, 0]

	def test_fill_matrix(self):
		env = run("""
		[[1,2][3,4]]@ [A]
		Fill( 9, [A]
		""")
		assert var(env, '[A]').data == [[9, 9], [9, 9]]
