from __future__ import annotations

from decorators import env_func
from errors import DomainError
from tiobjects import py_int

# Pxl- commands address a narrower region than the full 64×96 LCD:
# rows 0–62 (63 rows) and columns 0–94 (95 columns), inclusive.
MAX_ROW = 62
MAX_COL = 94


def _validate(row, col) -> tuple[int, int]:
	"""Coerce a (row, column) pair to ints and check the Pxl- addressable range."""
	row = py_int(row)
	col = py_int(col)
	if not (0 <= row <= MAX_ROW and 0 <= col <= MAX_COL):
		raise DomainError(f"Pixel out of range: row={row}, column={col}")
	return row, col


@env_func
def pxl_on(env, row, col) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, True)


@env_func
def pxl_off(env, row, col) -> None:
	row, col = _validate(row, col)
	env.screen.set(row, col, False)


@env_func
def pxl_change(env, row, col) -> None:
	row, col = _validate(row, col)
	env.screen.toggle(row, col)


@env_func
def pxl_test(env, row, col) -> float:
	row, col = _validate(row, col)
	return float(env.screen.get(row, col))
