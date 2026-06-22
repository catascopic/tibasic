"""Zoom commands (zoom.py): each adjusts the graphing window, then redisplays."""
import math

import pytest
from pytest import approx

from environment import Environment
from catalog import get_token
from errors import WindowRangeError
from modes import AngleMode, GraphMode
from test_tibasic import run, var


def _cmd(env, code):
	"""Run a single command token by code (for tokens like 'Zoom In' that contain a
	space and so can't be spelled as one whitespace-delimited piece for the tokenizer)."""
	env.submit([get_token(code)])


def _win(env):
	w = env.window
	return (w.xmin.resolve(), w.xmax.resolve(), w.ymin.resolve(), w.ymax.resolve())


class TestZStandard:
	def test_resets_bounds_and_scales(self):
		env = Environment()
		env.window.xmin.value, env.window.xmax.value = 3, 50
		env.window.xscl.value = 7
		run('ZStandard', env)
		assert _win(env) == (-10, 10, -10, 10)
		assert env.window.xscl.resolve() == 1
		assert env.window.yscl.resolve() == 1

	def test_func_mode_sets_xres(self):
		env = Environment()
		env.window.xres.value = 5
		run('ZStandard', env)
		assert env.window.xres.resolve() == 1

	def test_param_mode_sets_t_range_radians(self):
		env = Environment()
		env.graph_mode = GraphMode.PAR
		run('ZStandard', env)
		assert env.window.tmin.resolve() == 0
		assert env.window.tmax.resolve() == approx(2 * math.pi)
		assert env.window.tstep.resolve() == approx(math.pi / 24)

	def test_param_mode_t_range_degrees(self):
		env = Environment()
		env.angle_mode = AngleMode.DEG
		env.graph_mode = GraphMode.PAR
		run('ZStandard', env)
		assert env.window.tmax.resolve() == approx(360)
		assert env.window.tstep.resolve() == approx(7.5)

	def test_seq_mode_sets_n_range(self):
		env = Environment()
		env.graph_mode = GraphMode.SEQ
		run('ZStandard', env)
		assert env.window.n_min.resolve() == 1
		assert env.window.n_max.resolve() == 10
		assert env.window.plot_start.resolve() == 1
		assert env.window.plot_step.resolve() == 1


class TestZDecimal:
	def test_window(self):
		env = run('ZDecimal')
		assert _win(env) == (-4.7, 4.7, -3.1, 3.1)
		assert env.window.xscl.resolve() == 1 and env.window.yscl.resolve() == 1

	def test_pixels_are_tenths(self):
		# The point of ZDecimal: ΔX = ΔY = 0.1.
		env = run('ZDecimal')
		assert env.window.delta_x.resolve() == approx(0.1)
		assert env.window.delta_y.resolve() == approx(0.1)


class TestZTrig:
	def test_radians(self):
		env = run('ZTrig')
		assert env.window.xmin.resolve() == approx(-47 / 24 * math.pi)
		assert env.window.xmax.resolve() == approx(47 / 24 * math.pi)
		assert env.window.xscl.resolve() == approx(math.pi / 2)
		assert (env.window.ymin.resolve(), env.window.ymax.resolve()) == (-4, 4)

	def test_degrees(self):
		env = Environment()
		env.angle_mode = AngleMode.DEG
		run('ZTrig', env)
		assert env.window.xmin.resolve() == approx(-352.5)
		assert env.window.xmax.resolve() == approx(352.5)
		assert env.window.xscl.resolve() == approx(90)


class TestZInteger:
	def test_unit_pixels_and_scl(self):
		env = run('ZInteger')                       # default centre (0,0)
		assert _win(env) == (-47, 47, -31, 31)
		assert env.window.delta_x.resolve() == approx(1)
		assert env.window.delta_y.resolve() == approx(1)
		assert env.window.xscl.resolve() == 10 and env.window.yscl.resolve() == 10

	def test_recenters_on_rounded_center(self):
		env = Environment()
		env.window.xmin.value, env.window.xmax.value = 4.8, 5.8   # centre 5.3 → 5
		env.window.ymin.value, env.window.ymax.value = 2.1, 3.1   # centre 2.6 → 3
		run('ZInteger', env)
		assert _win(env) == (5 - 47, 5 + 47, 3 - 31, 3 + 31)


class TestZSquare:
	def test_equalizes_pixel_size(self):
		# Wide window (ΔX > ΔY): the y-axis grows so ΔX == ΔY, x is untouched.
		env = Environment()
		env.window.xmin.value, env.window.xmax.value = -20, 20
		env.window.ymin.value, env.window.ymax.value = -10, 10
		run('ZSquare', env)
		assert env.window.delta_x.resolve() == approx(env.window.delta_y.resolve())
		assert (env.window.xmin.resolve(), env.window.xmax.resolve()) == (-20, 20)
		assert env.window.ymin.resolve() == approx(-env.window.ymax.resolve())  # centred

	def test_never_shrinks(self):
		# Tall window (ΔY > ΔX): the x-axis grows; the window only gets bigger.
		env = Environment()
		env.window.xmin.value, env.window.xmax.value = -1, 1
		env.window.ymin.value, env.window.ymax.value = -10, 10
		run('ZSquare', env)
		assert env.window.delta_x.resolve() == approx(env.window.delta_y.resolve())
		assert env.window.xmax.resolve() > 1                       # x expanded
		assert (env.window.ymin.resolve(), env.window.ymax.resolve()) == (-10, 10)


class TestZoomInOut:
	def test_zoom_in_halves_quarters_by_fact(self):
		env = Environment()                          # default ±10, XFact=YFact=4
		_cmd(env, 0x89)                              # Zoom In
		assert _win(env) == (-2.5, 2.5, -2.5, 2.5)

	def test_zoom_out_grows_by_fact(self):
		env = Environment()
		_cmd(env, 0x8A)                              # Zoom Out
		assert _win(env) == (-40, 40, -40, 40)

	def test_in_then_out_is_identity(self):
		env = Environment()
		env.window.xmin.value, env.window.xmax.value = -6, 14   # centre 4
		_cmd(env, 0x89)
		_cmd(env, 0x8A)
		assert env.window.xmin.resolve() == approx(-6)
		assert env.window.xmax.resolve() == approx(14)

	def test_respects_xfact(self):
		env = Environment()
		env.window.x_fact.value = 2
		_cmd(env, 0x89)                              # width 20 / 2 → ±5
		assert (env.window.xmin.resolve(), env.window.xmax.resolve()) == (-5, 5)
		assert (env.window.ymin.resolve(), env.window.ymax.resolve()) == (-2.5, 2.5)  # YFact still 4

	def test_keeps_center(self):
		env = Environment()
		env.window.ymin.value, env.window.ymax.value = 0, 20     # centre 10
		_cmd(env, 0x89)
		assert (env.window.ymin.resolve() + env.window.ymax.resolve()) / 2 == approx(10)


class TestZoomFit:
	def test_func_fits_y_keeps_x(self):
		# Y1=X² over [-10,10]: Y range becomes [0,100]; X is left alone.
		env = run('"X²"@ Y1')
		run('ZoomFit', env)
		assert (env.window.xmin.resolve(), env.window.xmax.resolve()) == (-10, 10)
		assert env.window.ymin.resolve() == approx(0)
		assert env.window.ymax.resolve() == approx(100)

	def test_func_uses_only_selected(self):
		# Y1=X (deselected) shouldn't widen the fit; only Y2=X² counts.
		env = run('"X"@ Y1')
		run('"X²"@ Y2', env)
		run('FnOff 1', env)
		run('ZoomFit', env)
		assert env.window.ymin.resolve() == approx(0)   # would be -10 if Y1 counted

	def test_constant_function_raises(self):
		env = run('"5"@ Y1')
		with pytest.raises(WindowRangeError):
			run('ZoomFit', env)

	def test_no_function_raises(self):
		env = Environment()
		with pytest.raises(WindowRangeError):
			run('ZoomFit', env)

	def test_parametric_fits_x_and_y(self):
		env = run('"T"@ X1t')
		run('"T"@ Y1t', env)
		env.graph_mode = GraphMode.PAR
		env.window.tmin.value, env.window.tmax.value, env.window.tstep.value = 0.0, 10.0, 1.0
		run('ZoomFit', env)
		assert env.window.xmin.resolve() == approx(0)
		assert env.window.xmax.resolve() == approx(10)
		assert env.window.ymin.resolve() == approx(0)
		assert env.window.ymax.resolve() == approx(10)


class TestDisplaysGraph:
	def test_zoom_shows_the_graph(self):
		# A zoom brings up the graph screen (and regraphs), like the calculator.
		from modes import Screen
		env = Environment()
		run('ZStandard', env)
		assert env.screen is Screen.GRAPH
