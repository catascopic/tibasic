from __future__ import annotations

import math

from argspec import PassEnv, expr, integer, real
from decorators import no_arg_command, preparse_cmd, preparse_cmd_func
from errors import DomainError

# Pxl- commands address a narrower region than the full 64×96 LCD:
# rows 0–62 (63 rows) and columns 0–94 (95 columns), inclusive.
MAX_ROW = 62
MAX_COL = 94

# The graph screen (used by point/graph commands) spans the same region.
# 95 columns → 94 intervals; 63 rows → 62 intervals.
GRAPH_COL_SPAN = 94
GRAPH_ROW_SPAN = 62

# Pt-On/Off/Change mark pixel offsets (Δrow, Δcol) relative to centre.
# mark 2/6 = 3×3 filled box (9 pixels)
# mark 3/7 = 3×3 cross / plus sign (5 pixels)
# anything else = dot (1 pixel)
_BOX    = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
_CROSS  = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_DOT    = ((0, 0),)
_MARK_OFFSETS = {2: _BOX, 6: _BOX, 3: _CROSS, 7: _CROSS}


def _mark_offsets(mark: int):
	return _MARK_OFFSETS.get(mark, _DOT)


def _round_half_up(value: float) -> int:
	"""Round to the nearest integer with ties going up (x.5 → x+1).

	The calculator rounds graph→pixel coordinates this way; Python's built-in
	round() uses banker's rounding (round-half-to-even), which differs at .5.
	"""
	return math.floor(value + 0.5)


def _x_to_col(env, x: float) -> int:
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	return _round_half_up((x - xmin) * GRAPH_COL_SPAN / (xmax - xmin))


def _y_to_row(env, y: float) -> int:
	w = env.window
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	return _round_half_up((ymax - y) * GRAPH_ROW_SPAN / (ymax - ymin))


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


def _apply_mark(screen, row: int, col: int, mark: int, action) -> None:
	"""Apply *action(r, c)* to each pixel in the mark shape, clipping to screen bounds."""
	for dr, dc in _mark_offsets(mark):
		r, c = row + dr, col + dc
		if 0 <= r <= MAX_ROW and 0 <= c <= MAX_COL:
			action(r, c)


def _validate(row: float, col: float) -> tuple[int, int]:
	"""Check the Pxl-addressable range, then return Python ints for screen indexing."""
	if not (0 <= row <= MAX_ROW and 0 <= col <= MAX_COL):
		raise DomainError(f"Pixel out of range: row={row}, column={col}")
	return int(row), int(col)


@preparse_cmd_func
def pxl_on(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, True)


@preparse_cmd_func
def pxl_off(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, False)


@preparse_cmd_func
def pxl_change(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.toggle(row, col)


@preparse_cmd_func
def pxl_test(env: PassEnv, row: integer, col: integer) -> float:
	row, col = _validate(row, col)
	return float(env.screen.get(row, col))


@no_arg_command
def clr_draw(env) -> None:
	env.screen.clear()


@preparse_cmd_func
def pt_on(env: PassEnv, x: real, y: real, mark: integer = 1) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		row, col = pixel
		_apply_mark(env.screen, row, col, int(mark),
		            lambda r, c: env.screen.set(r, c, True))


@preparse_cmd_func
def pt_off(env: PassEnv, x: real, y: real, mark: integer = 1) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		row, col = pixel
		_apply_mark(env.screen, row, col, int(mark),
		            lambda r, c: env.screen.set(r, c, False))


@preparse_cmd_func
def pt_change(env: PassEnv, x: real, y: real, mark: integer = 1) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		row, col = pixel
		_apply_mark(env.screen, row, col, int(mark), env.screen.toggle)


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
def circle(env: PassEnv, x: real, y: real, r: real, _fast: expr = None) -> None:
	"""Circle(X,Y,r[,{i}]) — draw a circle (or ellipse) at graph (X,Y) with graph radius r.

	The optional 4th argument enables the 'fast circle' routine on real hardware
	(Bresenham 8-fold symmetry); it is accepted here and silently ignored — we
	always use the parametric approach, which handles non-square windows correctly.
	Negative radius is treated as its absolute value.  Off-screen pixels are clipped.
	"""
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	cy, cx = _graph_to_pixel(env, x, y)              # (row, col)
	rx = abs(r) * GRAPH_COL_SPAN / (xmax - xmin)   # pixel semi-axis, horizontal
	ry = abs(r) * GRAPH_ROW_SPAN / (ymax - ymin)    # pixel semi-axis, vertical
	# Step finely enough that no pixel is skipped (~4 steps per pixel of circumference).
	n = max(8, math.ceil(4 * math.pi * max(rx, ry)) + 1)
	for i in range(n):
		theta = 2 * math.pi * i / n
		c   = cx + _round_half_up(rx * math.cos(theta))
		row = cy - _round_half_up(ry * math.sin(theta))
		if 0 <= row <= MAX_ROW and 0 <= c <= MAX_COL:
			env.screen.set(row, c)


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
		if 0 <= r <= MAX_ROW and 0 <= c <= MAX_COL:
			env.screen.set(r, c, on)
