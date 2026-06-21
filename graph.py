"""Shared graph-plotting math: graph⇄pixel coordinate mapping, line rasterization,
and curve tracing.

Two subsystems use this: the Draw commands (DrawF/DrawInv/Tangent/Shade…, see draw.py)
and the function grapher (Environment.regraph, which plots the selected Y= functions).
Everything here takes the environment as a parameter and reads only env.window /
env.graph / env.draw_mode, so this module stays free of the
preparse⇄environment import cycle (it imports nothing from either)."""

import math

from errors import (
	DataTypeError, DivideByZeroError, DomainError, IncrementError,
	NonRealAnsError, TiOverflowError, SingularMatrixError,
)
from modes import DrawMode


# Pxl- commands address a narrower region than the full 64×96 LCD:
# rows 0–62 (63 rows) and columns 0–94 (95 columns), inclusive.
# The graph screen (used by point/graph commands) spans the same region.
# 95 columns → 94 intervals; 63 rows → 62 intervals.
MAX_ROW = 62
MAX_COL = 94

# Guard zone: how far off-screen a coordinate may be before a curve tracer clamps
# it.  Clamping both endpoints keeps Bresenham bounded near vertical asymptotes (or
# a parametric point that shoots far off screen) while still letting the visible
# stretch of the connecting line draw correctly.
_GUARD = MAX_ROW + MAX_COL   # generous but finite

# Errors the calculator silently swallows while evaluating a graphed expression:
# the offending point is dropped rather than aborting the command.
_GRAPH_ERRORS = (
	DataTypeError, DivideByZeroError, DomainError, IncrementError,
	NonRealAnsError, TiOverflowError, SingularMatrixError,
)


def _round_half_up(value: float) -> int:
	"""Round to the nearest integer with ties going up (x.5 → x+1).

	The calculator rounds graph→pixel coordinates this way; Python's built-in
	round() uses banker's rounding (round-half-to-even), which differs at .5.
	"""
	return math.floor(value + 0.5)


def _x_to_col(env, x: float) -> int:
	w = env.window
	xmin = w.xmin.resolve()
	xmax = w.xmax.resolve()
	return _round_half_up((x - xmin) * MAX_COL / (xmax - xmin))


def _y_to_row(env, y: float) -> int:
	w = env.window
	ymin = w.ymin.resolve()
	ymax = w.ymax.resolve()
	return _round_half_up((ymax - y) * MAX_ROW / (ymax - ymin))


def _col_to_x(env, col: float) -> float:
	"""Inverse of _x_to_col: the graph x-coordinate at the centre of a pixel column."""
	w = env.window
	xmin = w.xmin.resolve()
	xmax = w.xmax.resolve()
	return xmin + col * (xmax - xmin) / MAX_COL


def _row_to_y(env, row: float) -> float:
	"""Inverse of _y_to_row: the graph y-coordinate at the centre of a pixel row."""
	w = env.window
	ymin = w.ymin.resolve()
	ymax = w.ymax.resolve()
	return ymax - row * (ymax - ymin) / MAX_ROW


def _graph_to_pixel(env, x: float, y: float) -> tuple[int, int]:
	"""Translate graph coordinates to (row, column), without bounds checking.

	Used by Line( so Bresenham's can run end-to-end and clip per pixel.
	"""
	return _y_to_row(env, y), _x_to_col(env, x)


def _point_to_pixel(env, x: float, y: float):
	"""Translate graph coordinates to (row, column), or None if off-screen.

	Used by Pt- commands, which draw nothing when the point is out of range.
	"""
	row, col = _graph_to_pixel(env, x, y)
	return (row, col) if 0 <= row <= MAX_ROW and 0 <= col <= MAX_COL else None


def _bresenham(r0: int, c0: int, r1: int, c1: int):
	"""Yield (row, col) for every pixel on the line from (r0,c0) to (r1,c1)."""
	dr = abs(r1 - r0)
	dc = abs(c1 - c0)
	sr = 1 if r1 > r0 else -1
	sc = 1 if c1 > c0 else -1
	err = dr - dc
	while True:
		yield r0, c0
		if r0 == r1 and c0 == c1:
			break
		e2 = 2 * err
		if e2 > -dc:
			err -= dc
			r0 += sr
		if e2 < dr:
			err += dr
			c0 += sc


def _in_bounds(row, col):
	return 0 <= row <= MAX_ROW and 0 <= col <= MAX_COL


def _clip_segment(r0, c0, r1, c1):
	"""Liang–Barsky clip of a segment to the screen rectangle [0,MAX_ROW]×[0,MAX_COL].

	Returns integer endpoints (r0,c0,r1,c1) of the visible portion, or None if the
	segment lies entirely outside.  Clipping first keeps Bresenham bounded even when
	a near-vertical connecting line (e.g. across an asymptote) spans millions of rows.
	"""
	dr = r1 - r0
	dc = c1 - c0
	u1 = 0.0
	u2 = 1.0
	for pi, qi in ((-dc, c0), (dc, MAX_COL - c0), (-dr, r0), (dr, MAX_ROW - r0)):
		if pi == 0:
			if qi < 0:
				return None  # parallel to a border and outside it
		else:
			t = qi / pi
			if pi < 0:
				if t > u2:
					return None
				u1 = max(u1, t)
			else:
				if t < u1:
					return None
				u2 = min(u2, t)
	return (
		round(r0 + u1 * dr),
		round(c0 + u1 * dc),
		round(r0 + u2 * dr),
		round(c0 + u2 * dc),
	)


def _plot_segment(env, r0: int, c0: int, r1: int, c1: int, on: bool = True) -> None:
	"""Clip a segment to the screen, then draw the visible portion with Bresenham."""
	clipped = _clip_segment(r0, c0, r1, c1)
	if clipped is None:
		return
	for r, c in _bresenham(*clipped):
		if _in_bounds(r, c):
			env.graph.set(r, c, on)


def sample_function(env, evaluate):
	"""Return f(t): set X to t, call *evaluate*, store the result in Y, return it.

	*evaluate* is a zero-argument callable that reads the current X and returns the
	corresponding Y — e.g. a Thunk's `.eval` (DrawF) or `lambda: equation.eval(env)`
	(the function grapher).

	X and Y are deliberately left holding their last values when the caller
	finishes — DrawF/DrawInv/Tangent and a regular graph all "exit with the last
	coordinate stored".

	Any complex (or otherwise non-real) result is skipped outright — the calculator
	does not graph points whose Y value is complex, even if the imaginary part
	happens to be zero — as is any point whose evaluation raises one of the errors
	the calculator silently drops mid-graph.
	"""

	def f(x):
		env.x.value = x
		try:
			y = evaluate()
		except _GRAPH_ERRORS:
			return None
		if not isinstance(y, float):
			return None
		env.y.value = y
		return y

	return f


def trace_curve(env, f, inv: bool = False, on: bool = True) -> None:
	"""Sample f along one screen axis and plot the resulting curve.

	inv=False: iterate pixel columns; f maps graph-x → graph-y  (DrawF / Y= graphing).
	inv=True:  iterate pixel rows;    f maps graph-y → graph-x  (DrawInv).
	Consecutive points are joined with line segments in Connected mode, or drawn
	as single pixels in Dot mode.  Points that f skips (None) break the curve.

	ΔX (or ΔY) is computed once before the loop, matching the TI-84's behaviour
	of storing it as a window variable rather than recomputing per sample.
	"""
	connected = env.draw_mode is DrawMode.CONNECTED
	prev = None
	w = env.window
	if not inv:
		xmin = w.xmin.resolve()
		delta = (w.xmax.resolve() - xmin) / MAX_COL   # ΔX, computed once
		span = MAX_COL
		def to_indep(i):
			return xmin + i * delta

		def to_pixel(v, col):
			return (_y_to_row(env, v), col)

	else:
		ymax = w.ymax.resolve()
		delta = (ymax - w.ymin.resolve()) / MAX_ROW    # ΔY, computed once
		span = MAX_ROW
		def to_indep(i):
			return ymax - i * delta

		def to_pixel(v, row):
			return (row, _x_to_col(env, v))

	# Off-screen points (e.g. row=63 when MAX_ROW=62) are left unclamped to a pixel
	# but bounded by _GUARD, so the Bresenham path through the visible region is
	# identical to what the TI produces by running Bresenham end-to-end with
	# per-pixel clipping.
	for i in range(span + 1):
		value = f(to_indep(i))
		if value is None:
			prev = None
			continue
		row, col = to_pixel(value, i)
		curr = (
			max(-_GUARD, min(MAX_ROW + _GUARD, row)),
			max(-_GUARD, min(MAX_COL + _GUARD, col)),
		)
		if connected and prev is not None:
			for r, c in _bresenham(*prev, *curr):
				if _in_bounds(r, c):
					env.graph.set(r, c, on)

		elif _in_bounds(row, col):
			env.graph.set(row, col, on)

		prev = curr


def _param_values(start: float, stop: float, step: float):
	"""Yield parameter values start, start+step, … up to (and including) stop.

	Used to sweep T (parametric) and θ (polar).  Empty if step ≤ 0 — the calculator
	requires a positive Tstep/θstep and graphs nothing otherwise (rather than hang).
	The small epsilon keeps a stop that should land exactly on a step (e.g. Tmax=2π
	with a step that divides it) from being dropped to floating-point error.
	"""
	if step <= 0:
		return
	count = math.floor((stop - start) / step + 1e-9)
	for i in range(count + 1):
		yield start + i * step


def sample_parametric(env, eval_x, eval_y):
	"""Return point(t): set T to t, evaluate the X and Y equations, and return the
	(x, y) graph coordinates — or None if either is undefined, non-real, or raises a
	swallowed graph error.  Both halves must be defined for the point to plot.

	T is set before either equation is evaluated; X and Y are left holding the last
	point, mirroring how the calculator exits a graph with the trace coordinates set.
	"""

	def point(t):
		env.t.value = t
		try:
			x = eval_x()
			y = eval_y()
		except _GRAPH_ERRORS:
			return None
		if not (isinstance(x, float) and isinstance(y, float)):
			return None
		env.x.value = x
		env.y.value = y
		return (x, y)

	return point


def sample_polar(env, eval_r):
	"""Return point(θ): set θ to θ, evaluate r=f(θ), and return the (x, y) graph
	coordinates of (r, θ) in the current angle mode — or None if r is undefined,
	non-real, or raises a swallowed graph error.

	θ is in the current angle mode (as the θmin/θmax/θstep window values are), so it
	is converted to radians for the polar→rectangular conversion; the equation itself
	already honors the angle mode when it evaluates.  θ, X, and Y are left holding the
	last point.
	"""

	def point(theta):
		env.theta.value = theta
		try:
			r = eval_r()
		except _GRAPH_ERRORS:
			return None
		if not isinstance(r, float):
			return None
		angle = env.to_rad(theta)
		x = r * math.cos(angle)
		y = r * math.sin(angle)
		env.x.value = x
		env.y.value = y
		return (x, y)

	return point


def sample_sequence(env, index):
	"""Return point(n): evaluate sequence u/v/w (index 0/1/2) at term n and return the
	(n, value) graph coordinates for a Time plot — or None if the term is undefined,
	non-real, or raises a swallowed graph error.  n, X, and Y are left at the last point.
	"""

	def point(n):
		try:
			y = env.eval_sequence(index, n)
		except _GRAPH_ERRORS:
			return None
		if not isinstance(y, float):
			return None
		env.n.value = float(round(n))
		env.x.value = float(n)
		env.y.value = y
		return (float(n), y)

	return point


def trace_parametric(env, point, start: float, stop: float, step: float, on: bool = True) -> None:
	"""Sweep a parameter from start to stop by step, plotting point(t) → (x, y) graph
	coordinates (or None to break the curve).  Consecutive points are joined with line
	segments in Connected mode, or drawn as single pixels in Dot mode.

	This is the parameter-driven analogue of trace_curve: parametric and polar graphs
	are a path traced as the parameter advances, rather than one y per pixel column.
	"""
	connected = env.draw_mode is DrawMode.CONNECTED
	prev = None
	for t in _param_values(start, stop, step):
		xy = point(t)
		if xy is None:
			prev = None
			continue
		row, col = _graph_to_pixel(env, xy[0], xy[1])
		curr = (
			max(-_GUARD, min(MAX_ROW + _GUARD, row)),
			max(-_GUARD, min(MAX_COL + _GUARD, col)),
		)
		if connected and prev is not None:
			for r, c in _bresenham(*prev, *curr):
				if _in_bounds(r, c):
					env.graph.set(r, c, on)

		elif _in_bounds(row, col):
			env.graph.set(row, col, on)

		prev = curr


def _axis_ticks(lo: float, hi: float, scl: float):
	"""Yield the tick coordinates in [lo, hi] at integer multiples of scl.

	Nothing is produced when scl ≤ 0 (the calculator draws no tick marks rather
	than dividing by zero).  Ticks are stepped as k·scl, not by repeated addition,
	so they don't drift; the epsilon keeps an endpoint that should land exactly on
	a multiple from being dropped to floating-point error.
	"""
	if scl <= 0:
		return
	k = math.ceil(lo / scl - 1e-9)
	while True:
		value = k * scl
		if value > hi + 1e-9:
			return
		yield value
		k += 1


def draw_axes(env, on: bool = True) -> None:
	"""Draw the x- and y-axes, with Xscl/Yscl tick marks, onto the graph.

	Each axis is drawn only when its zero line falls within the window.  A tick is a
	single pixel on the positive side of the axis — one row above the x-axis (toward
	+y) and one column right of the y-axis (toward +x) — at every multiple of Xscl /
	Yscl.  A tick at the origin coincides with the crossing axis line, so it is
	harmless; ticks (or an axis edge) that fall off-screen are simply skipped.
	"""
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	axis_row = _y_to_row(env, 0.0)   # the x-axis lies on row(y=0)
	axis_col = _x_to_col(env, 0.0)   # the y-axis lies on col(x=0)

	if 0 <= axis_row <= MAX_ROW:
		for col in range(MAX_COL + 1):
			env.graph.set(axis_row, col, on)
		if axis_row - 1 >= 0:                 # tick marks above the x-axis
			for x in _axis_ticks(xmin, xmax, w.xscl.resolve()):
				col = _x_to_col(env, x)
				if 0 <= col <= MAX_COL:
					env.graph.set(axis_row - 1, col, on)

	if 0 <= axis_col <= MAX_COL:
		for row in range(MAX_ROW + 1):
			env.graph.set(row, axis_col, on)
		if axis_col + 1 <= MAX_COL:           # tick marks right of the y-axis
			for y in _axis_ticks(ymin, ymax, w.yscl.resolve()):
				row = _y_to_row(env, y)
				if 0 <= row <= MAX_ROW:
					env.graph.set(row, axis_col + 1, on)
