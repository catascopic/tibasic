"""Zoom commands (the ZOOM menu): each changes the graphing window, then redisplays
the graph with the new window.

  ZStandard  reset window to defaults (mode-aware)
  ZDecimal   friendly window, ΔX=ΔY=0.1
  ZTrig      trig-friendly window (ΔX = π/24 rad or 7.5°)
  ZInteger   integer-coordinate window, ΔX=ΔY=1, Xscl=Yscl=10, recentred
  ZSquare    grow one axis so ΔX=ΔY (a true square window)
  Zoom In    shrink the window about its centre by XFact/YFact
  Zoom Out   grow the window about its centre by XFact/YFact
  ZoomFit    fit Ymin/Ymax (and X for non-Func modes) to the graphed functions

Only Xmin/Xmax/Ymin/Ymax (and, where noted, the scales / Tmax / etc.) change; ΔX and
ΔY follow as they are derived from the bounds.

The real calculator raises ERR:INVALID if these run outside a program; this emulator
treats all input as program-like, so that distinction isn't modeled.  ZoomStat (fit
to stat plots) is not here — it needs StatPlot, which isn't implemented.
"""
import math

from preparse import no_arg_command
from modes import AngleMode, GraphMode
from errors import WindowRangeError
from graph import (
	MAX_ROW, MAX_COL, _param_values,
	sample_function, sample_parametric, sample_polar, sample_sequence,
)


def _set_x(env, xmin, xmax):
	env.window.xmin.value, env.window.xmax.value = float(xmin), float(xmax)


def _set_y(env, ymin, ymax):
	env.window.ymin.value, env.window.ymax.value = float(ymin), float(ymax)


# ── Fixed / friendly windows ─────────────────────────────────────────────────

@no_arg_command
def z_standard(env):
	"""ZStandard — reset the window to its defaults for the current graph mode."""
	w = env.window
	_set_x(env, -10, 10)
	_set_y(env, -10, 10)
	w.xscl.value, w.yscl.value = 1.0, 1.0
	# 2π / π⁄24 in Radian mode become 360 / 7.5 in Degree mode (Tmax, Tstep, θ…).
	full = 2 * math.pi if env.angle_mode is AngleMode.RAD else 360.0
	step = math.pi / 24 if env.angle_mode is AngleMode.RAD else 7.5
	mode = env.graph_mode
	if mode is GraphMode.FUNC:
		w.xres.value = 1.0
	elif mode is GraphMode.PAR:
		w.tmin.value, w.tmax.value, w.tstep.value = 0.0, full, step
	elif mode is GraphMode.POL:
		w.theta_min.value, w.theta_max.value, w.theta_step.value = 0.0, full, step
	elif mode is GraphMode.SEQ:
		w.n_min.value, w.n_max.value = 1.0, 10.0
		w.plot_start.value, w.plot_step.value = 1.0, 1.0
	env.display_graph()


@no_arg_command
def z_decimal(env):
	"""ZDecimal — friendly window where adjacent pixels differ by 0.1."""
	w = env.window
	_set_x(env, -4.7, 4.7)
	_set_y(env, -3.1, 3.1)
	w.xscl.value, w.yscl.value = 1.0, 1.0
	env.display_graph()


@no_arg_command
def z_trig(env):
	"""ZTrig — trig-friendly window: ΔX is π/24 (Radian) or 7.5° (Degree)."""
	w = env.window
	if env.angle_mode is AngleMode.RAD:
		_set_x(env, -47 / 24 * math.pi, 47 / 24 * math.pi)
		w.xscl.value = math.pi / 2
	else:
		_set_x(env, -352.5, 352.5)
		w.xscl.value = 90.0
	_set_y(env, -4, 4)
	w.yscl.value = 1.0
	env.display_graph()


@no_arg_command
def z_integer(env):
	"""ZInteger — recentre on the (rounded) current centre with ΔX=ΔY=1, Xscl=Yscl=10."""
	w = env.window
	xc = round((w.xmin.resolve() + w.xmax.resolve()) / 2)
	yc = round((w.ymin.resolve() + w.ymax.resolve()) / 2)
	_set_x(env, xc - MAX_COL // 2, xc + MAX_COL // 2)   # span 94 → ΔX = 1
	_set_y(env, yc - MAX_ROW // 2, yc + MAX_ROW // 2)   # span 62 → ΔY = 1
	w.xscl.value, w.yscl.value = 10.0, 10.0
	env.display_graph()


# ── Reshaping the current window ─────────────────────────────────────────────

@no_arg_command
def z_square(env):
	"""ZSquare — grow the narrower axis so ΔX=ΔY, keeping the centre and the scales."""
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	# Match the larger pixel size, so the window only ever grows (never shrinks).
	delta = max((xmax - xmin) / MAX_COL, (ymax - ymin) / MAX_ROW)
	xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
	_set_x(env, xc - delta * MAX_COL / 2, xc + delta * MAX_COL / 2)
	_set_y(env, yc - delta * MAX_ROW / 2, yc + delta * MAX_ROW / 2)
	env.display_graph()


def _scale(env, fx, fy):
	"""Scale the window about its centre by factors fx (width) and fy (height)."""
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
	hx, hy = (xmax - xmin) / 2 * fx, (ymax - ymin) / 2 * fy
	_set_x(env, xc - hx, xc + hx)
	_set_y(env, yc - hy, yc + hy)
	env.display_graph()


@no_arg_command
def zoom_in(env):
	"""Zoom In — shrink the window about its centre: width÷XFact, height÷YFact."""
	_scale(env, 1 / env.window.x_fact.resolve(), 1 / env.window.y_fact.resolve())


@no_arg_command
def zoom_out(env):
	"""Zoom Out — grow the window about its centre: width×XFact, height×YFact."""
	_scale(env, env.window.x_fact.resolve(), env.window.y_fact.resolve())


# ── ZoomFit (fit the window to the graphed functions) ────────────────────────

def _func_y_values(env):
	"""Every finite Y of the selected Func-mode equations across Xmin..Xmax."""
	w = env.window
	xmin = w.xmin.resolve()
	delta = (w.xmax.resolve() - xmin) / MAX_COL
	ys = []
	for func in env.graph_functions.groups[GraphMode.FUNC]:
		eq = func.equations[0].value
		if func.selected and eq is not None:
			f = sample_function(env, lambda eq=eq: eq.eval(env))
			ys += [y for i in range(MAX_COL + 1) if (y := f(xmin + i * delta)) is not None]
	return ys


def _collect(point, start, stop, step):
	"""Every finite (x, y) of a parameter-swept curve."""
	return [xy for t in _param_values(start, stop, step) if (xy := point(t)) is not None]


def _swept_points(env):
	"""Every finite (x, y) of the selected curves in the current non-Func mode."""
	w = env.window
	mode = env.graph_mode
	pts = []
	if mode is GraphMode.PAR:
		start, stop, step = w.tmin.resolve(), w.tmax.resolve(), w.tstep.resolve()
		for func in env.graph_functions.groups[GraphMode.PAR]:
			x_eq, y_eq = func.equations[0].value, func.equations[1].value
			if func.selected and x_eq is not None and y_eq is not None:
				point = sample_parametric(env, lambda e=x_eq: e.eval(env), lambda e=y_eq: e.eval(env))
				pts += _collect(point, start, stop, step)
	elif mode is GraphMode.POL:
		start, stop, step = w.theta_min.resolve(), w.theta_max.resolve(), w.theta_step.resolve()
		for func in env.graph_functions.groups[GraphMode.POL]:
			r_eq = func.equations[0].value
			if func.selected and r_eq is not None:
				point = sample_polar(env, lambda e=r_eq: e.eval(env))
				pts += _collect(point, start, stop, step)
	elif mode is GraphMode.SEQ:
		start, stop, step = w.plot_start.resolve(), w.n_max.resolve(), w.plot_step.resolve()
		with env._sequence_pass():
			for index, func in enumerate(env.graph_functions.groups[GraphMode.SEQ]):
				if func.selected and func.equations[0].value is not None:
					pts += _collect(sample_sequence(env, index), start, stop, step)
	return pts


@no_arg_command
def zoom_fit(env):
	"""ZoomFit — set the window to the smallest one holding every plotted point.

	In Func mode only Ymin/Ymax change (fit over the current Xmin..Xmax); in the
	parameter-swept modes both X and Y are fit over the range of T, θ, or n.  Raises
	ERR:WINDOW RANGE when there's nothing to fit or the range collapses to a point.
	"""
	if env.graph_mode is GraphMode.FUNC:
		ys = _func_y_values(env)
		if not ys or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no Y range to fit")
		_set_y(env, min(ys), max(ys))
	else:
		pts = _swept_points(env)
		xs = [x for x, _ in pts]
		ys = [y for _, y in pts]
		if not pts or min(xs) == max(xs) or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no range to fit")
		_set_x(env, min(xs), max(xs))
		_set_y(env, min(ys), max(ys))
	env.display_graph()
