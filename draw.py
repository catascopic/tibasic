from __future__ import annotations

import math

from argspec import PassEnv, integer, numeric, real
from decorators import nullary_command, preparse, CMD_FUNC
from errors import DomainError
from tiobjects import py_int

# Pxl- commands address a narrower region than the full 64×96 LCD:
# rows 0–62 (63 rows) and columns 0–94 (95 columns), inclusive.
MAX_ROW = 62
MAX_COL = 94

# The graph screen (used by point/graph commands) spans the same region.
# 95 columns → 94 intervals; 63 rows → 62 intervals.
GRAPH_COL_SPAN = 94
GRAPH_ROW_SPAN = 62


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


def _validate(row, col) -> tuple[int, int]:
	"""Coerce a (row, column) pair to ints and check the Pxl- addressable range."""
	row = py_int(row)
	col = py_int(col)
	if not (0 <= row <= MAX_ROW and 0 <= col <= MAX_COL):
		raise DomainError(f"Pixel out of range: row={row}, column={col}")
	return row, col


@preparse(CMD_FUNC)
def pxl_on(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, True)


@preparse(CMD_FUNC)
def pxl_off(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, False)


@preparse(CMD_FUNC)
def pxl_change(env: PassEnv, row: integer, col: integer) -> None:
	row, col = _validate(row, col)
	env.screen.toggle(row, col)


@preparse(CMD_FUNC)
def pxl_test(env: PassEnv, row: integer, col: integer) -> float:
	row, col = _validate(row, col)
	return float(env.screen.get(row, col))


@nullary_command
def clr_draw(env) -> None:
	env.screen.clear()


@preparse(CMD_FUNC)
def pt_on(env: PassEnv, x: real, y: real) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		env.screen.set(*pixel, True)


@preparse(CMD_FUNC)
def pt_off(env: PassEnv, x: real, y: real) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		env.screen.set(*pixel, False)


@preparse(CMD_FUNC)
def pt_change(env: PassEnv, x: real, y: real) -> None:
	pixel = _point_to_pixel(env, x, y)
	if pixel is not None:
		env.screen.toggle(*pixel)
