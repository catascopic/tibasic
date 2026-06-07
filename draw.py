from __future__ import annotations

import math

from argspec import PassEnv, integer, real
from decorators import no_arg_command, preparse_cmd_func
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


def _point_to_pixel(env, x: float, y: float):
	"""Translate graph coordinates (x, y) to (row, column).

	Returns None if the point falls outside the visible graph screen, in which
	case the calculator simply draws nothing (no error).
	"""
	w = env.window
	xmin, xmax = w.xmin.resolve(), w.xmax.resolve()
	ymin, ymax = w.ymin.resolve(), w.ymax.resolve()
	# Multiply before dividing: it keeps exact half-integers exact (e.g.
	# 3*94/188 == 1.5), whereas dividing first can yield 1.4999… and round wrong.
	col = _round_half_up((x - xmin) * GRAPH_COL_SPAN / (xmax - xmin))
	row = _round_half_up((ymax - y) * GRAPH_ROW_SPAN / (ymax - ymin))
	if 0 <= row <= MAX_ROW and 0 <= col <= MAX_COL:
		return row, col
	return None


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
