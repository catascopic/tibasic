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
from modes import AngleMode
from graph import MAX_ROW, MAX_COL


def _set_x(env, xmin, xmax):
	env.window.xmin = float(xmin)
	env.window.xmax = float(xmax)


def _set_y(env, ymin, ymax):
	env.window.ymin = float(ymin)
	env.window.ymax = float(ymax)


# ── Fixed / friendly windows ─────────────────────────────────────────────────

@no_arg_command
def z_standard(env):
	"""ZStandard — reset the window to its defaults for the current graph mode."""
	_set_x(env, -10, 10)
	_set_y(env, -10, 10)
	env.window.xscl = 1.0
	env.window.yscl = 1.0
	env.graph_mode_handler.standard_window(env)   # mode-specific vars: Xres / T… / θ… / n…
	env.display_graph()


@no_arg_command
def z_decimal(env):
	"""ZDecimal — friendly window where adjacent pixels differ by 0.1."""
	_set_x(env, -4.7, 4.7)
	_set_y(env, -3.1, 3.1)
	env.window.xscl = 1.0
	env.window.yscl = 1.0
	env.display_graph()


@no_arg_command
def z_trig(env):
	"""ZTrig — trig-friendly window: ΔX is π/24 (Radian) or 7.5° (Degree)."""
	w = env.window
	if env.angle_mode is AngleMode.RAD:
		_set_x(env, -47 / 24 * math.pi, 47 / 24 * math.pi)
		w.xscl = math.pi / 2
	else:
		_set_x(env, -352.5, 352.5)
		w.xscl = 90.0
	_set_y(env, -4, 4)
	w.yscl = 1.0
	env.display_graph()


@no_arg_command
def z_integer(env):
	"""ZInteger — recentre on the (rounded) current centre with ΔX=ΔY=1, Xscl=Yscl=10."""
	w = env.window
	xc = round((w.xmin + w.xmax) / 2)
	yc = round((w.ymin + w.ymax) / 2)
	_set_x(env, xc - MAX_COL // 2, xc + MAX_COL // 2)   # span 94 → ΔX = 1
	_set_y(env, yc - MAX_ROW // 2, yc + MAX_ROW // 2)   # span 62 → ΔY = 1
	w.xscl, w.yscl = 10.0, 10.0
	env.display_graph()


# ── Reshaping the current window ─────────────────────────────────────────────

@no_arg_command
def z_square(env):
	"""ZSquare — grow the narrower axis so ΔX=ΔY, keeping the centre and the scales."""
	w = env.window
	xmin, xmax = w.xmin, w.xmax
	ymin, ymax = w.ymin, w.ymax
	# Match the larger pixel size, so the window only ever grows (never shrinks).
	delta = max((xmax - xmin) / MAX_COL, (ymax - ymin) / MAX_ROW)
	xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
	_set_x(env, xc - delta * MAX_COL / 2, xc + delta * MAX_COL / 2)
	_set_y(env, yc - delta * MAX_ROW / 2, yc + delta * MAX_ROW / 2)
	env.display_graph()


def _scale(env, fx, fy):
	"""Scale the window about its centre by factors fx (width) and fy (height)."""
	w = env.window
	xmin, xmax = w.xmin, w.xmax
	ymin, ymax = w.ymin, w.ymax
	xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
	hx, hy = (xmax - xmin) / 2 * fx, (ymax - ymin) / 2 * fy
	_set_x(env, xc - hx, xc + hx)
	_set_y(env, yc - hy, yc + hy)
	env.display_graph()


@no_arg_command
def zoom_in(env):
	"""Zoom In — shrink the window about its centre: width÷XFact, height÷YFact."""
	_scale(env, 1 / env.window.x_fact, 1 / env.window.y_fact)


@no_arg_command
def zoom_out(env):
	"""Zoom Out — grow the window about its centre: width×XFact, height×YFact."""
	_scale(env, env.window.x_fact, env.window.y_fact)


# ── ZoomFit (fit the window to the graphed functions) ────────────────────────

@no_arg_command
def zoom_fit(env):
	"""ZoomFit — set the window to the smallest one holding every plotted point.

	The per-mode extent (and the ERR:WINDOW RANGE check) lives on the graph-mode
	handler: Func fits only Ymin/Ymax over the current Xmin..Xmax, while the
	parameter-swept modes fit both axes over the range of T, θ, or n (see graphmodes.py).
	"""
	xmin, xmax, ymin, ymax = env.graph_mode_handler.fit_bounds(env)
	if xmin is not None:           # None → leave the x-range alone (Func mode)
		_set_x(env, xmin, xmax)
	_set_y(env, ymin, ymax)
	env.display_graph()
