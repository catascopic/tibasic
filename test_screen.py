"""Which screen is displayed (Screen) and when the graph re-plots.

Plotting itself is a stub, so these assert the state machine: the active screen
after each command, and that env.regraph() fires at the right moments (spied)."""
import pytest

from environment import Environment
from modes import Screen
from program import Program
from terminal import ScriptedConsole
from test_tibasic import run, toks

HOME, GRAPH, TABLE = Screen.HOME, Screen.GRAPH, Screen.TABLE


def spy_regraph(env):
	"""Replace env.regraph with a counter; return the (mutable) call list."""
	calls = []
	env.regraph = lambda: calls.append(1)
	return calls


def graph_env():
	"""An environment already showing the graph (as if a draw/DispGraph ran)."""
	env = Environment()
	env.screen = GRAPH
	return env


class TestHomeScreenCommands:
	def test_default_is_home(self):
		assert Environment().screen is HOME

	def test_disp_shows_home(self):
		env = graph_env()
		run('Disp 5', env)
		assert env.screen is HOME

	def test_output_shows_home(self):
		env = graph_env()
		run('Output( 1,1,5', env)
		assert env.screen is HOME

	def test_input_shows_home(self):
		env = graph_env()
		env.console = ScriptedConsole(inputs=['5'])
		run('Input X', env)
		assert env.screen is HOME

	def test_clrhome_does_not_change_screen(self):
		env = graph_env()
		run('ClrHome', env)
		assert env.screen is GRAPH

	def test_pause_does_not_change_screen(self):
		# A program can pause while the graph is showing.
		env = graph_env()
		env.console = ScriptedConsole()
		Program(toks('Pause')).run(env)
		assert env.screen is GRAPH


class TestGraphDisplay:
	def test_dispgraph_shows_graph_and_regraphs(self):
		env = Environment()
		calls = spy_regraph(env)
		run('DispGraph', env)
		assert env.screen is GRAPH and len(calls) == 1

	def test_disptable_switches_to_table(self):
		env = graph_env()
		run('DispTable', env)
		assert env.screen is TABLE


class TestClrDraw:
	def test_on_home_clears_without_regraph(self):
		env = Environment()                 # screen HOME
		calls = spy_regraph(env)
		run('ClrDraw', env)
		assert env.screen is HOME and calls == []

	def test_on_graph_regraphs(self):
		env = graph_env()
		calls = spy_regraph(env)
		run('ClrDraw', env)
		assert env.screen is GRAPH and len(calls) == 1


class TestDrawingDisplaysGraph:
	def test_draw_from_home_displays_and_regraphs(self):
		env = Environment()                 # screen HOME
		calls = spy_regraph(env)
		run('Pxl-On( 3,5', env)
		assert env.screen is GRAPH and len(calls) == 1   # regraphed on the transition

	def test_draw_while_on_graph_does_not_regraph(self):
		env = graph_env()
		calls = spy_regraph(env)
		run('Pxl-On( 3,5', env)
		assert env.screen is GRAPH and calls == []       # already up: no re-plot

	def test_pxl_test_is_a_query_not_a_draw(self):
		# Reading a pixel must not display the graph.
		env = Environment()                 # screen HOME
		run('pxl-Test( 3,5', env)
		assert env.screen is HOME
