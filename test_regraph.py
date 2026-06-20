"""Function plotting: Environment.regraph draws the selected, defined Y= functions
onto the graph (Func mode only), reusing the same tracer as DrawF."""
import pytest

from environment import Environment
from modes import GraphMode, DrawMode
from test_tibasic import run, var


def total_pixels(graph):
	return sum(sum(row) for row in graph.buffer)


class TestDispGraphPlots:
	def test_plots_selected_function(self):
		# Y1=X² is selected by the store; DispGraph plots the parabola.
		env = run('"X²"@ Y1')
		run('DispGraph', env)
		assert env.graph.get(31, 47)             # vertex at the centre pixel
		assert total_pixels(env.graph) > 0

	def test_matches_drawf_of_same_expression(self):
		# A plotted Y1=X² is pixel-for-pixel the same as DrawF X² on a blank graph.
		env_plot = run('"X²"@ Y1')
		run('DispGraph', env_plot)
		env_draw = run('DrawF X²')
		assert env_plot.graph.buffer == env_draw.graph.buffer

	def test_two_functions_both_plot(self):
		# Storing selects each; DispGraph plots the union of Y1=X² and Y2=X.
		env = run('"X²"@ Y1')
		run('"X"@ Y2', env)
		run('DispGraph', env)
		# Y2=X passes through (62,0) and (0,94); Y1=X² hits the vertex.
		assert env.graph.get(31, 47)
		assert env.graph.get(62, 0) and env.graph.get(0, 94)

	def test_leaves_x_at_last_sample(self):
		# Like DrawF, plotting exits with X holding the last column's x (Xmax).
		env = run('"X²"@ Y1')
		run('DispGraph', env)
		assert var(env, 'X') == 10

	def test_honors_dot_mode(self):
		# Y1=X stays on screen every column → Dot mode lights exactly one pixel/column.
		env = run('"X"@ Y1')
		env.draw_mode = DrawMode.DOT
		run('DispGraph', env)
		assert total_pixels(env.graph) == 95     # MAX_COL + 1 columns


class TestSelectionAndDefinition:
	def test_deselected_function_not_plotted(self):
		env = run('"X²"@ Y1')
		run('FnOff', env)                        # turn Y1 off
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_undefined_function_not_plotted(self):
		# FnOn selects all ten, but none are defined → nothing to plot.
		env = Environment()
		run('FnOn', env)
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_only_func_mode_plots(self):
		# Y1 is selected and defined, but a non-Function mode plots nothing (parametric/
		# polar/sequence graphing isn't implemented) — the graph just comes up blank.
		env = run('"X²"@ Y1')
		env.graph_mode = GraphMode.POL
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0


class TestRegraphClears:
	def test_regraph_starts_from_a_blank_graph(self):
		# A manual draw made while on the graph is erased by the next full redraw.
		env = run('"X²"@ Y1')
		run('DispGraph', env)
		run('Pxl-On( 0,0', env)                  # a stray pixel on top of the curve
		assert env.graph.get(0, 0)
		run('DispGraph', env)                    # redraw from the functions
		assert not env.graph.get(0, 0)
		assert env.graph.get(31, 47)             # the curve is still there

	def test_drawing_brings_up_the_function_underneath(self):
		# Y1=0 is a horizontal line at row 31; drawing from the home screen displays
		# the graph (plotting Y1) first, then lays the pixel on top.
		env = run('"0"@ Y1')                      # screen is HOME after a store
		run('Pxl-On( 5,5', env)
		assert env.graph.get(5, 5)               # the drawn pixel
		assert env.graph.get(31, 0) and env.graph.get(31, 94)   # Y1=0 underneath
