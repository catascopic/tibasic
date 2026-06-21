"""Function plotting: Environment.regraph draws the selected, defined functions onto
the graph, reusing the shared tracers in graph.py.  Func/parametric/polar/sequence
are all plotted; the axes (drawn by regraph when axes_on) have their own tests in
TestAxes at the bottom."""
import math

import pytest

from environment import Environment as _RealEnvironment
from errors import DomainError
from modes import GraphMode, DrawMode
from test_tibasic import run as _run, var


# Most tests here check the plotted curves in isolation, so they default the axes
# off (regraph draws axes + Xscl/Yscl ticks underneath when axes_on).  TestAxes
# turns them back on to check the axes themselves.

def Environment(*args, **kwargs):
	env = _RealEnvironment(*args, **kwargs)
	env.axes_on = False
	return env


def run(src, env=None):
	return _run(src, env if env is not None else Environment())


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


def _seq_env(formula=None, initial=None, nmin=1, nmax=10):
	"""Seq-mode env: optionally store u's recurrence and u(nMin) initial list, set n range."""
	env = Environment()
	if formula is not None:
		run(f'"{formula}"@ u', env)
	if initial is not None:
		run(f'{{{",".join(map(str, initial))}}}@ u ( nMin )', env)
	env.graph_mode = GraphMode.SEQ
	env.window.n_min.value = nmin
	env.window.n_max.value = nmax
	env.window.plot_start.value = nmin
	env.window.plot_step.value = 1
	return env


class TestSequenceEval:
	def test_explicit_formula(self):
		# u(n)=2n needs no initial values: it just reads the current n.
		env = _seq_env('2n')
		assert [env.eval_sequence(0, k) for k in range(1, 6)] == [2, 4, 6, 8, 10]

	def test_recursive_fibonacci(self):
		# u(n)=u(n-1)+u(n-2), u(nMin)={1,1}: the classic two-term recurrence.
		env = _seq_env('u(n-1)+u(n-2)', initial=[1, 1])
		assert [env.eval_sequence(0, k) for k in range(1, 9)] == [1, 1, 2, 3, 5, 8, 13, 21]

	def test_initial_values_are_chronological(self):
		# Element i of u(nMin) is the term at nMin+i (earliest first).
		env = _seq_env(initial=[10, 20], nmin=1)
		assert env.eval_sequence(0, 1) == 10
		assert env.eval_sequence(0, 2) == 20

	def test_scalar_initial_value(self):
		# A single-term recurrence takes a scalar u(nMin) (wrapped as a 1-element list).
		env = _seq_env('2u(n-1)', initial=[3])
		assert [env.eval_sequence(0, k) for k in range(1, 5)] == [3, 6, 12, 24]

	def test_cross_reference_between_sequences(self):
		# v may reference u: v(n)=u(n-1)+1 with u(n)=n.
		env = Environment()
		run('"n"@ u', env)
		run('"u(n-1)+1"@ v', env)
		assert env.eval_sequence(1, 5) == 5     # u(4)+1

	def test_self_reference_raises(self):
		env = _seq_env('u(n)')
		with pytest.raises(DomainError):
			env.eval_sequence(0, 5)

	def test_index_below_nmin_raises(self):
		env = _seq_env('n', nmin=3)
		with pytest.raises(DomainError):
			env.eval_sequence(0, 2)

	def test_store_selects_the_sequence(self):
		# Storing the recurrence selects u, like any equation store.
		env = run('"2n"@ u')
		assert env.graph_functions.groups[GraphMode.SEQ][0].selected


class TestSequencePlot:
	def _window(self, env, lo=0, hi=11):
		env.window.xmin.value, env.window.xmax.value = lo, hi
		env.window.ymin.value, env.window.ymax.value = lo, hi

	def test_time_plot_draws_points(self):
		# u(n)=n over n=1..10: a Time plot rising from lower-left toward upper-right.
		env = _seq_env('n')
		self._window(env)
		run('DispGraph', env)
		assert total_pixels(env.graph) > 0

	def test_plotstep_thins_the_plot(self):
		# PlotStep=2 plots half as many terms as PlotStep=1 (counted in Dot mode).
		env1 = _seq_env('n'); self._window(env1)
		env1.draw_mode = DrawMode.DOT
		run('DispGraph', env1)

		env2 = _seq_env('n'); self._window(env2)
		env2.draw_mode = DrawMode.DOT
		env2.window.plot_step.value = 2
		run('DispGraph', env2)

		assert 0 < total_pixels(env2.graph) < total_pixels(env1.graph)

	def test_unselected_not_plotted(self):
		env = _seq_env('n'); self._window(env)
		run('FnOff', env)
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_undefined_sequence_mode_is_blank(self):
		# Seq mode with nothing defined just comes up cleared.
		env = Environment()
		env.graph_mode = GraphMode.SEQ
		run('FnOn', env)
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_leaves_n_at_last_term(self):
		env = _seq_env('n', nmax=7)
		self._window(env)
		run('DispGraph', env)
		assert var(env, 'n') == 7


class TestAxes:
	"""regraph draws the axes (with Xscl/Yscl ticks) when axes_on is set.

	On the default ±10 window the origin lands at pixel (row 31, col 47): the x-axis
	is row 31, the y-axis is column 47.  Ticks sit one pixel toward the positive side
	— row 30 above the x-axis, column 48 right of the y-axis.
	"""

	@pytest.fixture
	def env(self):
		e = Environment()
		e.axes_on = True
		return e

	def test_axes_drawn_on_dispgraph(self, env):
		run('DispGraph', env)
		assert all(env.graph.get(31, c) for c in range(95))   # x-axis is the full row
		assert all(env.graph.get(r, 47) for r in range(63))   # y-axis is the full column

	def test_x_axis_ticks_sit_above_axis(self, env):
		# Xscl=1, so x=5 (column 71) and x=-5 (column 24) get a tick on row 30.
		run('DispGraph', env)
		assert env.graph.get(30, 71)
		assert env.graph.get(30, 24)
		assert not env.graph.get(30, 49)   # column 49 is between ticks → blank

	def test_y_axis_ticks_sit_right_of_axis(self, env):
		# Yscl=1, so y=5 (row 16) and y=-5 (row 47) get a tick in column 48.
		run('DispGraph', env)
		assert env.graph.get(16, 48)
		assert env.graph.get(47, 48)

	def test_scl_controls_tick_spacing(self, env):
		# With Xscl=5 only multiples of 5 are ticked: x=5 keeps its tick, x=1 loses it.
		env.window.xscl.value = 5
		run('DispGraph', env)
		assert env.graph.get(30, 71)        # x=5 still ticked
		assert not env.graph.get(30, 52)    # x=1 (column 52) no longer ticked

	def test_zero_scl_draws_no_ticks(self, env):
		# Xscl=0 must not hang or divide by zero — the axis is bare.
		env.window.xscl.value = 0
		run('DispGraph', env)
		assert all(env.graph.get(31, c) for c in range(95))   # axis still there
		assert not env.graph.get(30, 71)   # but the x=5 tick is gone
		assert not env.graph.get(30, 24)   # and the x=-5 tick is gone

	def test_axes_off_leaves_blank(self, env):
		env.axes_on = False
		run('DispGraph', env)
		assert total_pixels(env.graph) == 0

	def test_axes_redrawn_each_display(self, env):
		run('DispGraph', env)
		run('Pxl-On( 5,5', env)             # a stray pixel off the axes
		assert env.graph.get(5, 5)
		run('DispGraph', env)               # redraw: axes return, stray is cleared
		assert not env.graph.get(5, 5)
		assert all(env.graph.get(31, c) for c in range(95))

	def test_x_axis_absent_when_zero_offscreen(self, env):
		# Shift the window so y=0 is below it: no x-axis row, but the y-axis remains.
		env.window.ymin.value, env.window.ymax.value = 1, 11
		run('DispGraph', env)
		assert not any(all(env.graph.get(r, c) for c in range(95)) for r in range(63))
		assert all(env.graph.get(r, 47) for r in range(63))   # y-axis still drawn
