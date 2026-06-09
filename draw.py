import math
from numbers import Number

import purefunctions as pf
from argspec import PassEnv, complex_list, real, thunk
from decorators import no_arg_command, preparse_cmd, preparse_cmd_func
from errors import DataTypeError, DivideByZeroError, DomainError, IncrementError, NonRealAnsError, TiOverflowError, SingularMatrixError
from modes import DrawMode
from screen import Screen
from tiobjects import TiEquation, py_int

# Pxl- commands address a narrower region than the full 64×96 LCD:
# rows 0–62 (63 rows) and columns 0–94 (95 columns), inclusive.
# The graph screen (used by point/graph commands) spans the same region.
# 95 columns → 94 intervals; 63 rows → 62 intervals.
MAX_ROW = 62
MAX_COL = 94

# Pt-On/Off/Change mark pixel offsets (Δrow, Δcol) relative to centre.
# mark 2/6 = 3×3 filled box (9 pixels)
# mark 3/7 = 3×3 cross / plus sign (5 pixels)
# anything else = dot (1 pixel)
_CROSS  = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_BOX    = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_MARK_OFFSETS = {2.0: _BOX, 3.0: _CROSS, 6.0: _BOX, 7.0: _CROSS}


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


def _validate(row, col):
	"""Check the Pxl-addressable range, then return Python ints for screen indexing."""
	if not _in_bounds(row, col):
		raise DomainError(f"Pixel out of range: row={row}, column={col}")
	return py_int(row), py_int(col)


@preparse_cmd_func
def pxl_on(env: PassEnv, row: real, col: real) -> None:
	env.screen.set(*_validate(row, col))


@preparse_cmd_func
def pxl_off(env: PassEnv, row: real, col: real) -> None:
	env.screen.set_off(*_validate(row, col))


@preparse_cmd_func
def pxl_change(env: PassEnv, row: real, col: real) -> None:
	env.screen.toggle(*_validate(row, col))


@preparse_cmd_func
def pxl_test(env: PassEnv, row: real, col: real) -> float:
	return float(env.screen.get(*_validate(row, col)))


@no_arg_command
def clr_draw(env) -> None:
	env.screen.clear()


def _pt_action(env, x, y, mark, action) -> None:
	row, col = _graph_to_pixel(env, x, y)
	if _in_bounds(row, col):
		try:
			points = _MARK_OFFSETS[mark]
		except KeyError:
			action(env.screen, row, col)
		else:
			for dr, dc in points:
				r = row + dr
				c = col + dc
				if _in_bounds(r, c):
					action(env.screen, r, c)


@preparse_cmd_func
def pt_on(env: PassEnv, x: real, y: real, mark: real = 1.0) -> None:
	_pt_action(env, x, y, mark, Screen.set)


@preparse_cmd_func
def pt_off(env: PassEnv, x: real, y: real, mark: real = 1.0) -> None:
	_pt_action(env, x, y, mark, Screen.set_off)


@preparse_cmd_func
def pt_change(env: PassEnv, x: real, y: real, mark: real = 1.0) -> None:
	_pt_action(env, x, y, mark, Screen.toggle)


@preparse_cmd
def vertical(env: PassEnv, x: real) -> None:
	"""Vertical X — draw a full-height line at graph x-coordinate X."""
	col = _x_to_col(env, x)
	if 0 <= col <= MAX_COL:
		for row in range(MAX_ROW + 1):
			env.screen.set(row, col, True)


@preparse_cmd
def horizontal(env: PassEnv, y: real) -> None:
	"""Horizontal Y — draw a full-width line at graph y-coordinate Y."""
	row = _y_to_row(env, y)
	if 0 <= row <= MAX_ROW:
		for col in range(MAX_COL + 1):
			env.screen.set(row, col, True)


@preparse_cmd_func
def line(env: PassEnv, x1: real, y1: real, x2: real, y2: real, erase: real = 1) -> None:
	"""Line(X1,Y1,X2,Y2[,erase]) — draw (or erase) a line between two graph points.

	erase=0 turns pixels off; any other value (default 1) turns them on.
	Off-screen pixels are clipped silently.
	"""
	on = (erase != 0)
	r0, c0 = _graph_to_pixel(env, x1, y1)
	r1, c1 = _graph_to_pixel(env, x2, y2)
	for r, c in _bresenham(r0, c0, r1, c1):
		if _in_bounds(r, c):
			env.screen.set(r, c, on)


@preparse_cmd_func
def circle(env: PassEnv, x: real, y: real, r: real, _fast: complex_list = None) -> None:
	"""Circle(X,Y,r[,{i}]) — draw a circle (or ellipse) at graph (X,Y) with graph radius r.

	The optional 4th argument enables the 'fast circle' routine on real hardware
	(Bresenham 8-fold symmetry); the calculator requires it to be a complex list
	(e.g. {i}).  We validate it for fidelity but ignore its value — we always use
	the parametric approach, which handles non-square windows correctly.
	Negative radius is treated as its absolute value.  Off-screen pixels are clipped.
	"""
	w = env.window
	xmin = w.xmin.resolve()
	xmax = w.xmax.resolve()
	ymin = w.ymin.resolve()
	ymax = w.ymax.resolve()
	cy, cx = _graph_to_pixel(env, x, y)
	rx = abs(r) * MAX_COL / (xmax - xmin)
	ry = abs(r) * MAX_ROW / (ymax - ymin)
	# Step finely enough that no pixel is skipped (~4 steps per pixel of circumference).
	n = max(8, math.ceil(4 * math.pi * max(rx, ry)) + 1)
	for i in range(n):
		theta = 2 * math.pi * i / n
		col = cx + _round_half_up(rx * math.cos(theta))
		row = cy - _round_half_up(ry * math.sin(theta))
		if _in_bounds(row, col):
			env.screen.set(row, col)


# ── Function graphing (DrawF / DrawInv) and distribution shading ────────────────

# Errors the calculator silently swallows while evaluating a graphed expression:
# the offending point is dropped rather than aborting the command.


def _function_sampler(env, formula):
	"""Return f(t): set X to t, evaluate *formula*, store the result in Y, return it.

	X and Y are deliberately left holding their last values when the caller
	finishes — DrawF/DrawInv/Tangent all "exit with the last coordinate stored".

	Any complex result is skipped outright — the calculator does not graph points
	whose Y value is complex, even if the imaginary part happens to be zero.
	"""

	def f(x):
		env.x.value = x
		try:
			y = formula.eval()
		except (
			DataTypeError, DivideByZeroError, DomainError, IncrementError, 
			NonRealAnsError, TiOverflowError, SingularMatrixError
		):
			return None
		if not isinstance(y, float):
			return None
		env.y.value = y
		return y

	return f


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
			env.screen.set(r, c, on)


def _trace_curve(env, f, inv: bool = False, on: bool = True) -> None:
	"""Sample f along one screen axis and plot the resulting curve.

	axis='x': iterate pixel columns; f maps graph-x → graph-y  (DrawF).
	axis='y': iterate pixel rows;    f maps graph-y → graph-x  (DrawInv).
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

	# Guard zone: how far off-screen a coordinate may be before we clamp it.
	# Clamping both endpoints keeps Bresenham bounded near vertical asymptotes
	# while leaving ordinarily off-screen points (e.g. row=63 when MAX_ROW=62)
	# unclamped so the Bresenham path through the visible region is identical to
	# what the TI produces by running Bresenham end-to-end with per-pixel clipping.

	_GUARD = MAX_ROW + MAX_COL   # generous but finite
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
					env.screen.set(r, c, on)

		elif _in_bounds(row, col):
			env.screen.set(row, col, on)

		prev = curr


def _shade_under(env, f, lo: float, hi: float) -> None:
	"""Fill the area between the curve y=f(x) and the x-axis for lo ≤ x ≤ hi."""
	axis_row = _y_to_row(env, 0.0)
	for col in range(MAX_COL + 1):
		x = _col_to_x(env, col)
		if x < lo or x > hi:
			continue
		y = f(x)
		if y is None:
			continue
		top, bot = sorted((_y_to_row(env, y), axis_row))
		for row in range(max(top, 0), min(bot, MAX_ROW) + 1):
			env.screen.set(row, col)


@preparse_cmd
def draw_f(env: PassEnv, formula: thunk) -> None:
	"""DrawF expr — graph an expression in X as Y=f(X) (Func mode, regardless of mode)."""
	_trace_curve(env, _function_sampler(env, formula))


@preparse_cmd
def draw_inv(env: PassEnv, formula: thunk) -> None:
	"""DrawInv expr — graph the inverse of expr: X becomes vertical, Y horizontal."""
	_trace_curve(env, _function_sampler(env, formula), inv=True)


@preparse_cmd_func
def shade_norm(env: PassEnv, lower: real, upper: real, mu: real = 0, sigma: real = 1) -> None:
	"""ShadeNorm(lower,upper[,μ,σ]) — draw the normal curve, shade the interval's area."""
	f = lambda x: pf.normalpdf(x, mu, sigma)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_t(env: PassEnv, lower: real, upper: real, df: real) -> None:
	"""Shade_t(lower,upper,df) — draw the Student-t curve, shade the interval's area."""
	f = lambda x: pf.tpdf(x, df)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_chi_sq(env: PassEnv, lower: real, upper: real, df: real) -> None:
	"""Shadeχ²(lower,upper,df) — draw the chi-square curve, shade the interval's area."""
	f = lambda x: pf.chi_sq_pdf(x, df)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_f(env: PassEnv, lower: real, upper: real, df1: real, df2: real) -> None:
	"""ShadeF(lower,upper,df1,df2) — draw the F curve, shade the interval's area."""
	f = lambda x: pf.f_pdf(x, df1, df2)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


def _shade_pixel(pattern: int, patres: int, row: int, col: int) -> bool:
	"""Return True if (row, col) falls on a shading line for the given pattern/resolution.

	pattern 1 — vertical lines:            col % patres == 0
	pattern 2 — horizontal lines:          row % patres == 0
	pattern 3 — negative-slope 45° lines:  (row - col) % patres == 0
	pattern 4 — positive-slope 45° lines:  (row + col) % patres == 0

	patres=1 makes every pixel eligible → solid fill for all patterns.
	"""
	if pattern == 2:
		return row % patres == 0
	elif pattern == 3:
		return (row - col) % patres == 0
	elif pattern == 4:
		return (row + col) % patres == 0
	else:                          # pattern 1 (default) or out-of-range
		return col % patres == 0


@preparse_cmd_func
def shade(env: PassEnv, lower: thunk, upper: thunk,
          xleft: real = None, xright: real = None,
          pattern: real = 1, patres: real = 1) -> None:
	"""Shade(lowerfunc,upperfunc[,Xleft,Xright,pattern,patres]) — shade between two curves.

	Draws both boundary curves on the graph, then fills the region where
	lowerfunc(X) < upperfunc(X), restricted to Xleft ≤ X ≤ Xright (defaulting to
	the window's Xmin/Xmax).

	pattern (1–4) selects the shading line direction:
	  1 = vertical (default)   2 = horizontal
	  3 = negative-slope 45°   4 = positive-slope 45°

	patres (1–8) is the spacing between shading lines: 1 fills every pixel
	(solid), 2 every second, … 8 every eighth.
	"""
	w = env.window
	lo = w.xmin.resolve() if xleft is None else xleft
	hi = w.xmax.resolve() if xright is None else xright
	flo = _function_sampler(env, lower)
	fhi = _function_sampler(env, upper)
	pat = max(1, min(4, py_int(pattern)))
	res = max(1, py_int(patres))
	# Draw both boundary curves.
	_trace_curve(env, flo)
	_trace_curve(env, fhi)
	# Fill the region between them.
	for col in range(0, MAX_COL + 1):
		x = _col_to_x(env, col)
		if x < lo or x > hi:
			continue
		ylo = flo(x)
		yhi = fhi(x)
		if ylo is None or yhi is None or ylo > yhi:
			continue
		top = _y_to_row(env, yhi)   # upper function → smaller row number
		bot = _y_to_row(env, ylo)   # lower function → larger row number
		for row in range(max(top, 0), min(bot, MAX_ROW) + 1):
			if _shade_pixel(pat, res, row, col):
				env.screen.set(row, col, True)


def _numeric_derivative(f, x: float, h: float = 1e-3):
	"""Central-difference slope of f at x, or None if either sample is undefined.

	Uses h=0.001 to match the calculator's nDeriv default tolerance, which is the
	same routine Tangent( uses to find the slope of the tangent line.
	"""
	fp = f(x + h)
	fm = f(x - h)
	if fp is None or fm is None:
		return None
	return (fp - fm) / (2 * h)


@preparse_cmd_func
def tangent(env: PassEnv, formula: thunk, value: real) -> None:
	"""Tangent(expr,value) — graph expr and draw the line tangent to it at X=value.

	The slope is found numerically (central difference, matching nDeriv), and the
	tangent line is drawn across the full window from Xmin to Xmax.
	"""
	f = _function_sampler(env, formula)
	_trace_curve(env, f)
	m = _numeric_derivative(f, value)
	y0 = f(value)                 # evaluated last so X/Y exit holding the tangent point
	if m is None or y0 is None:
		return
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	tan = lambda x: y0 + m * (x - value)
	r0, c0 = _graph_to_pixel(env, xmin, tan(xmin))
	r1, c1 = _graph_to_pixel(env, xmax, tan(xmax))
	_plot_segment(env, r0, c0, r1, c1, True)
