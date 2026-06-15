"""Matrix types, variables, validators, and functions."""

import builtins
import math
import random
from itertools import accumulate, batched

from core import TiList, TiMatrix, require_list, require_matrix
from core import require_real, require_int, py_int
from preparse import preparse_func, preparse_cmd_func, forms_func, Real, MatrixVar
from errors import DataTypeError, DimMismatchError, InvalidDimError, ArgumentError

from parser import ArgParser


@preparse_func
def det(mat: TiMatrix):
	n = mat.rows
	if n == 0 or n != mat.cols:
		raise InvalidDimError(f"det requires a square matrix, got {mat.rows}×{mat.cols}")
	work = [row.copy() for row in mat.data]
	sign = 1.0
	for col in range(n):
		pivot = next((r for r in range(col, n) if work[r][col] != 0), None)
		if pivot is None:
			return 0.0
		if pivot != col:
			work[col], work[pivot] = work[pivot], work[col]
			sign = -sign
		for row in range(col + 1, n):
			if work[row][col] != 0:
				factor = work[row][col] / work[col][col]
				for j in range(col, n):
					work[row][j] -= factor * work[col][j]

	return sign * math.prod(work[i][i] for i in range(n))


@preparse_func
def identity(n: Real):
	size = py_int(n)
	return TiMatrix([[float(r == c) for c in range(size)] for r in range(size)])


@preparse_func
def rand_m(rows: Real, cols: Real):
	rows = py_int(rows)
	cols = py_int(cols)
	if not (1 <= rows <= 99) or not (1 <= cols <= 99):
		raise InvalidDimError("randM: dimensions must be 1-99")

	# Per spec: entries are successive randInt(-9,9) calls filled bottom-right to top-left
	data = [float(random.randint(-9, 9)) for _ in range(rows * cols)]
	return TiMatrix([list(row) for row in batched(reversed(data), cols)])


def _row_reduce(mat, get_range):
	require_matrix(mat)
	if mat.rows > mat.cols:
		raise InvalidDimError("ref/rref: TiMatrix must have at least as many columns as rows")
	result = [row.copy() for row in mat.data]
	pivot_row = 0
	for col in range(mat.cols):
		if pivot_row >= mat.rows:
			break
		try:
			swap_row = next((r for r in range(pivot_row, mat.rows) if result[r][col] != 0))
		except StopIteration:
			continue

		result[pivot_row], result[swap_row] = result[swap_row], result[pivot_row]
		pivot = result[pivot_row]
		scale = pivot[col]
		for k in range(mat.cols):
			pivot[k] /= scale
		for r in get_range(pivot_row, mat.rows):
			if r != pivot_row and result[r][col] != 0:
				pivot = result[pivot_row]
				current = result[r]
				factor = current[col]
				for k in range(mat.cols):
					current[k] -= factor * pivot[k]

		pivot_row += 1
	return TiMatrix(result)


@preparse_func
def ref(mat: TiMatrix):
	return _row_reduce(mat, lambda pivot_row, rows: range(pivot_row + 1, rows))


@preparse_func
def rref(mat: TiMatrix):
	return _row_reduce(mat, lambda pivot_row, rows: range(rows))


@preparse_func
def row_swap(mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row1, mat.get_row(row2))
	result.set_row(row2, mat.get_row(row1))
	return result


@preparse_func
def row_plus(mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row2, [a + b for a, b in zip(mat.get_row(row2), mat.get_row(row1))])
	return result


@preparse_func
def times_row(factor: Real, mat: TiMatrix, row: Real):
	result = mat.copy()
	result.set_row(row, [factor * x for x in mat.get_row(row)])
	return result


@preparse_func
def times_row_plus(factor: Real, mat: TiMatrix, row1: Real, row2: Real):
	result = mat.copy()
	result.set_row(row2, [factor * a + b for a, b in zip(mat.get_row(row1), mat.get_row(row2))])
	return result


@forms_func
def list_to_matr(args: ArgParser) -> None:
	from itertools import zip_longest
	list_vals = []
	while True:
		list_vals.append(require_list(args.expr()))
		if not args.has_next:
			raise ArgumentError("List►matr: expected matrix variable as last argument")
		if args.peek().is_matrix_var():
			mat_var = args.matrix_var()
			break
	mat_var.value = TiMatrix([list(row) for row in zip_longest(*(lst.data for lst in list_vals), fillvalue=0.0)])
	args.end_paren_cmd()


@forms_func
def matr_to_list(args: ArgParser) -> None:
	mat = args.expr()
	if not isinstance(mat, TiMatrix):
		raise DataTypeError("Matr►list: first argument must be a matrix")
	if args.peek().is_list_start():
		list_vars = [args.list_var()]
		while args.has_next:
			list_vars.append(args.list_var())
		for var, col_data in zip(list_vars, zip(*mat.data)):
			var.value = TiList(list(col_data))
	else:
		col = py_int(args.expr()) - 1
		if not (0 <= col < mat.cols):
			raise InvalidDimError(
				f"Matr►list: column {col + 1} out of range for {mat.rows}×{mat.cols} matrix"
			)
		args.list_var().value = TiList([mat.data[r][col] for r in range(mat.rows)])
	args.end_paren_cmd()
