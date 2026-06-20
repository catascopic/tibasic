"""Function plotting: Environment.regraph draws the selected, defined functions onto
the graph, reusing the shared tracers in plot.py.  Func/parametric/polar are plotted;
sequence isn't yet."""
import math

import pytest

from environment import Environment
from modes import GraphMode, DrawMode
from test_tibasic import run, var


def total_pixels(graph):
	return sum(sum(row) for row in graph.buffer)


def lit(env):
	"""Set of (row, col) pixels that are on, across the drawable region."""
	return {(r, c) for r in range(63) for c in range(95) if env.graph.get(r, c)}


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


def _par_env(x_expr, y_expr, tmin=0.0, tmax=10.0, tstep=0.1):
	"""Parametric env: store X1T/Y1T (selecting the pair) and set the T sweep."""
	env = run(f'"{x_expr}"@ X1t')
	run(f'"{y_expr}"@ Y1t', env)
	env.graph_mode = GraphMode.PAR
	env.window.tmin.value = tmin
	env.window.tmax.value = tmax
	env.window.tstep.value = tstep
	return env


class TestParametric:
	def test_plots_the_path(self):
		# X1T=T, Y1T=T over T∈[0,10] traces y=x in the first quadrant on a ±10 window:
		# (0,0)→centre pixel, (10,10)→top-right corner.
		env = _par_env('T', 'T')
		run('DispGraph', env)
		assert env.graph.get(31, 47)
		assert env.graph.get(0, 94)
		assert not env.graph.get(62, 0)          # the third-quadrant arm is never swept

	def test_both_halves_required(self):
		# X1T defined but Y1T undefined → the pair doesn't plot.
		env = run('"T"@ X1t')
		env.graph_mode = GraphMode.PAR
		env.window.tmin.value, env.window.tmax.value, env.window.tstep.value = 0, 10, 0.1
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_zero_tstep_plots_nothing(self):
		# Tstep≤0 graphs nothing rather than looping forever.
		env = _par_env('T', 'T', tstep=0)
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_leaves_t_at_last_sample(self):
		env = _par_env('T', 'T', tmin=0, tmax=5, tstep=1)
		run('DispGraph', env)
		assert var(env, 'T') == 5


class TestPolar:
	def _circle_env(self, r_expr='5', step=0.02):
		env = run(f'"{r_expr}"@ r1')
		env.graph_mode = GraphMode.POL
		env.window.theta_min.value = 0.0
		env.window.theta_max.value = 2 * math.pi   # default angle mode is radians
		env.window.theta_step.value = step
		return env

	def test_constant_r_is_a_circle(self):
		# r=5 traces a circle of radius 5 about the origin: an outline, centre unlit,
		# reaching both sides of the centre column.
		env = self._circle_env()
		run('DispGraph', env)
		assert total_pixels(env.graph) > 0
		assert not env.graph.get(31, 47)                       # hollow centre
		assert any(env.graph.get(31, c) for c in range(48, 95))  # right of centre
		assert any(env.graph.get(31, c) for c in range(0, 47))   # left of centre

	def test_respects_angle_mode_degrees(self):
		# In Degree mode the same θ window (0..2π≈6.28°) sweeps only a tiny arc, so the
		# far-left side of the circle is never reached.
		env = self._circle_env()
		from modes import AngleMode
		env.angle_mode = AngleMode.DEG
		run('DispGraph', env)
		assert not any(env.graph.get(31, c) for c in range(0, 24))


class TestSequenceNotYet:
	def test_sequence_mode_plots_nothing(self):
		# Sequence graphing isn't implemented; the graph just comes up cleared.
		env = Environment()
		env.graph_mode = GraphMode.SEQ
		run('FnOn', env)
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0
