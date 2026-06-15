"""List types, variables, validators, and functions."""

import builtins
import math
import operator
from itertools import accumulate, pairwise
from numbers import Number

from core import TiList, TiMatrix, is_complex_val, require_list, require_real_list, require_complex_list, require_vectorizable, require_vectorizable_real, require_real, require_int, py_int
from preparse import preparse_func, preparse_cmd, preparse_cmd_func, Real, Env, Thunk, NumericVar, ListVar, ListVarPrefixOptional, forms_func, no_arg_command
from errors import DataTypeError, DimMismatchError, InvalidDimError, IncrementError, StatError, UndefinedError
from preparse import preparse_func, preparse_cmd, preparse_cmd_func, forms_func, no_arg_command, Real, Env, Thunk, NumericVar, ListVar, ListOrMatrix, ListVarPrefixOptional
from parser import ArgParser


def _sort(main_var, dep_vars, reverse: bool):
	main = main_var.resolve()
	deps = [v.resolve() for v in dep_vars]

	for d in deps:
		if len(d) != len(main):
			raise DimMismatchError(f"SortA/SortD: dependent list length {len(d)} doesn't match {len(main)}")

	if not deps:
		main.data.sort(reverse=reverse)
	else:
		data = main.data
		indices = sorted(range(len(data)), key=lambda i: data[i], reverse=reverse)
		main.data = [data[i] for i in indices]
		for d in deps:
			d.data = [d.data[i] for i in indices]


@preparse_cmd_func
def sort_a(main_var: ListVar, *dep_vars: ListVar):
	_sort(main_var, dep_vars, False)


@preparse_cmd_func
def sort_d(main_var: ListVar, *dep_vars: ListVar):
	_sort(main_var, dep_vars, True)


# You'd think dim could be a pure function, right? For a while, it was.
# But then I realized empty lists are illegal everywhere, with a single exception.
# dim( reads a variable's stored dimension rather than its value: it accesses
# .value (raw storage) instead of .resolve(), so dim(L1) returns 0 for an empty
# list where a bare reference to L1 would raise InvalidDimError.  This mirrors
# the calculator, where dim( works on both sides of → for the same reason.
@forms_func
def dim(args: ArgParser):
	if args.peek().is_list_start():
		var = args.list_var()
		val = var.value
		if val is None:
			raise UndefinedError("Undefined list variable")
		args.end_func()
		return len(val)

	value = args.expr()
	args.end_func()
	if isinstance(value, TiList):
		return len(value)
	if isinstance(value, TiMatrix):
		return TiList([value.rows, value.cols])
	raise DataTypeError(f"dim: expected list or matrix; got {value}")


@forms_func
def fill(args: ArgParser):
	fill_value = require_real(args.expr())
	if args.peek().is_matrix_var():
		lst = args.matrix_var().resolve()
		args.end_paren_cmd()
		for row in lst.data:
			for i in range(len(row)):
				row[i] = fill_value
	elif args.peek().is_list_start():
		lst = args.list_var().resolve()
		args.end_paren_cmd()
		for i in range(len(lst.data)):
			lst.data[i] = fill_value
	else:
		raise DataTypeError("Fill(: expected a list or matrix variable")


@preparse_func
def seq(env: Env, formula: Thunk, var: NumericVar, start: Real, end: Real, step: Real = 1) -> TiList:
	n = start
	result = []
	if step == 0:
		raise IncrementError("seq: step cannot be zero")

	if step > 0:
		if start > end + 1e-10:
			raise IncrementError(f"seq: step is positive but start ({start}) > end ({end})")
		op = operator.le
		end += 1e-10
	else:
		if start < end - 1e-10:
			raise IncrementError(f"seq: step is negative but start ({start}) < end ({end})")
		op = operator.ge
		end -= 1e-10

	with env.nest_guard(seq), var.scoped():
		while op(n, end):
			var.value = n
			result.append(formula.eval())
			n += step

	return TiList(result)


@preparse_func
def cum_sum(obj: ListOrMatrix):
	if isinstance(obj, TiMatrix):
		cols = obj.cols
		rows = obj.rows
		return TiMatrix([
			[builtins.sum(obj.data[rr][c] for rr in range(r + 1))
				for c in range(cols)]
			for r in range(rows)
		])
	if isinstance(obj, TiList):
		return TiList(list(accumulate(obj.data)))
	raise DataTypeError(f"Expected list or matrix; got {obj}")


@preparse_func
def augment(a: ListOrMatrix, b: ListOrMatrix):
	if isinstance(a, TiList) and isinstance(b, TiList):
		return TiList(a.data + b.data)
	if isinstance(a, TiMatrix) and isinstance(b, TiMatrix):
		if a.rows != b.rows:
			raise DimMismatchError(f"Row count mismatch: {a.rows} vs {b.rows}")
		return TiMatrix([r1 + r2 for r1, r2 in zip(a.data, b.data)])
	raise DataTypeError(f"augment: both args must be lists or both must be matrices; got {a}, {b}")


@preparse_func
def mean(lst: TiList, freqlist: TiList = None):
	if freqlist is None:
		return builtins.sum(lst) / len(lst)
	return builtins.sum(x * w for x, w in zip(lst, freqlist)) / builtins.sum(freqlist)


@preparse_func
def median(lst: TiList, freqlist: TiList = None):
	if freqlist is None:
		sorted_data = sorted(lst)
		n = len(sorted_data)
		mid = n // 2
		return sorted_data[mid] if n % 2 else (sorted_data[mid - 1] + sorted_data[mid]) / 2

	if len(lst) != len(freqlist):
		raise DimMismatchError("median: dim mismatch")
	pairs = sorted(zip(lst, freqlist), key=lambda p: p[0])
	total = builtins.sum(require_int(f) for _, f in pairs)
	if total <= 0:
		raise StatError("median: total frequency must be positive")

	def nth(n):
		count = 0
		for value, freq in pairs:
			count += int(freq)
			if n < count:
				return value

	if total % 2:
		return nth(total // 2)
	return (nth(total // 2 - 1) + nth(total // 2)) / 2


@preparse_func
def delta_list(lst: TiList):
	return TiList([b - a for a, b in pairwise(lst)])


@preparse_func
def sum(lst: TiList, start: Real = None, end: Real = None):
	data = lst.data
	if start is None:
		return builtins.sum(data)

	start = py_int(start)
	end = len(data) if end is None else py_int(end)
	if not (1 <= start <= end <= len(data)):
		raise InvalidDimError(f"sum: index out of range (start={start}, end={end}, dim={len(data)})")

	return builtins.sum(data[start - 1 : end])


@preparse_func
def prod(lst: TiList, start: Real = None, end: Real = None):
	data = lst.data
	if start is None:
		return math.prod(data)

	start = py_int(start)
	end = len(data) if end is None else py_int(end)
	if not (1 <= start <= end <= len(data)):
		raise InvalidDimError(f"prod: index out of range (start={start}, end={end}, dim={len(data)})")

	return math.prod(data[start - 1 : end])


@preparse_func
def variance(lst: TiList, freqlist: TiList = None):
	if freqlist is None:
		n = len(lst)
		if n < 2:
			raise StatError("stdDev: need at least 2 elements")
		m = mean(lst)
		return builtins.sum((x - m) ** 2 for x in lst) / (n - 1)
	if len(lst) != len(freqlist):
		raise DimMismatchError("stdDev: dim mismatch")

	m = mean(lst, freqlist)
	total_w = builtins.sum(freqlist)
	if total_w <= 1:
		raise StatError("stdDev: total frequency must be > 1")

	return builtins.sum(w * (x - m) ** 2 for x, w in zip(lst, freqlist)) / (total_w - 1)


@preparse_func
def stddev(lst: TiList, freqlist: TiList = None):
	return math.sqrt(variance(lst, freqlist))


@preparse_cmd
def clr_list(first: ListVar, *rest_vars: ListVar):
	"""ClrList list[, list, ...] — clear each named list to empty; silently skip nonexistent lists."""
	for var in (first, *rest_vars):
		lst = var.value
		if lst is not None:
			lst.clear()


@no_arg_command
def clr_all_lists(env):
	"""ClrAllLists — set every defined list (L1–L6 and user lists) to empty."""
	for list_var in env.lists:
		if list_var.value is not None:
			list_var.value.clear()
	for lst in env.user_lists.values():
		lst.clear()


@preparse_cmd
def set_up_editor(env: Env, *list_vars: ListVarPrefixOptional):
	"""SetUpEditor [list, ...] — ensure lists exist, creating empty ones as needed."""
	if list_vars:
		for var in list_vars:
			if var.value is None:
				var.value = TiList([])
	else:
		for var in env.lists:
			if var.value is None:
				var.value = TiList([])
