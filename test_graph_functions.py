"""Graph-function on/off selection: storing selects, and FnOn/FnOff toggle."""
import pytest

from environment import Environment
from modes import GraphMode
from errors import DomainError
from core import TiEquation
from test_tibasic import run

FUNC, PAR, POL, SEQ = GraphMode.FUNC, GraphMode.PAR, GraphMode.POL, GraphMode.SEQ


def selected(env, mode):
	return [f.selected for f in env.graph_functions.groups[mode]]


class TestDefaults:
	def test_all_start_deselected(self):
		env = Environment()
		assert not any(flag for m in GraphMode for flag in selected(env, m))

	def test_layout_counts(self):
		env = Environment()
		assert (len(selected(env, FUNC)), len(selected(env, PAR)),
		        len(selected(env, POL)), len(selected(env, SEQ))) == (10, 6, 6, 3)


class TestStoreSelects:
	def test_storing_selects_that_function(self):
		env = run('"2X"@ Y1')
		assert selected(env, FUNC)[0]
		assert not any(selected(env, FUNC)[1:])

	def test_store_still_records_the_equation(self):
		env = run('"2X"@ Y1')
		assert isinstance(env.function[0].value, TiEquation)

	def test_selection_survives_wrong_mode(self):
		# Storing Y1 while in Polar mode still selects it — for when you return to Func.
		env = Environment()
		env.graph_mode = POL
		run('"2X"@ Y1', env)
		assert selected(env, FUNC)[0]
		assert not any(selected(env, POL))      # nothing polar got selected

	def test_parametric_pair_shares_one_flag(self):
		x_env = run('"T"@ X1t')                  # store the X half
		y_env = run('"T"@ Y1t')                  # store the Y half
		assert x_env.graph_functions.groups[PAR][0].selected
		assert y_env.graph_functions.groups[PAR][0].selected


class TestFnOnOff:
	def test_no_args_selects_all(self):
		assert all(selected(run('FnOn'), FUNC))

	def test_no_args_deselects_all(self):
		env = Environment()
		run('FnOn', env)
		run('FnOff', env)
		assert not any(selected(env, FUNC))

	def test_select_by_number(self):
		env = run('FnOn 2')
		assert selected(env, FUNC)[1] and not selected(env, FUNC)[0]

	def test_select_multiple(self):
		s = selected(run('FnOn 1,3'), FUNC)
		assert s[0] and s[2] and not s[1]

	def test_deselect_specific(self):
		env = Environment()
		run('FnOn', env)
		run('FnOff 1', env)
		s = selected(env, FUNC)
		assert not s[0] and all(s[1:])

	def test_zero_is_the_tenth_function(self):
		assert selected(run('FnOn 0'), FUNC)[9]

	def test_acts_on_current_mode_only(self):
		env = Environment()
		env.graph_mode = POL
		run('FnOn 1', env)
		assert selected(env, POL)[0]
		assert not any(selected(env, FUNC))

	def test_number_out_of_range_raises(self):
		env = Environment()
		env.graph_mode = POL                     # only 6 functions
		with pytest.raises(DomainError):
			run('FnOn 7', env)
