from __future__ import annotations
import operator
from functools import wraps
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

import operators
from argspec import env, expr, thunk, numeric_var, list_var, list_var_prefix_optional, equation_var, string_var, label_name, program_name, optional, rest
from decorators import forms_func, preparse, nullary_command
from errors import DataTypeError, ArgumentError, IncrementError, InvalidDimError, DimMismatchError, UndefinedError, TiSyntaxError
from signals import ReturnSignal, StopSignal
from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_str, py_int


@forms_func
def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	if isinstance(ans, TiList):
		(index,) = a.parse_indices(1)
		return ans[index]
	if isinstance(ans, TiMatrix):
		return ans[a.parse_indices(2)]
	b = a.expr()
	a.end_func()
	return operators.mul(ans, b)

##################
# MATH FUNCTIONS #
##################

@preparse(env, thunk, numeric_var, expr, expr)
def sigma(env, formula, var, start, end) -> float:
	total = 0
	n = start
	with env.nest_guard(sigma), var.scoped():
		while n <= end:
			var.value = n
			total += formula.eval()
			n += 1
	return total

@preparse(env, thunk, numeric_var, expr, optional(expr, 0.001))
def n_deriv(env, formula, var, val, h) -> float:
	with env.nest_guard(n_deriv, max_depth=1), var.scoped():
		var.value = val + h
		fwd = formula.eval()
		var.value = val - h
		bwd = formula.eval()
	return (fwd - bwd) / (2 * h)

# G7K15 nodes (positive half + 0) and weights on [-1, 1]
_K15_NODES = [
	0.0,                0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
	0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126
]
_K15_WEIGHTS = [
	0.2094821410847278, 0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
	0.1406532597155259, 0.1047900103222502, 0.0630920926299786, 0.0229353220105292
]
# G7 uses nodes at indices 0, 2, 4, 6 (every other Kronrod node)
_G7_WEIGHTS  = [
	0.4179591836734694, None, 0.3818300505051189, None,
	0.2797053914892767, None, 0.1294849661688697, None
]

def _gk15(f, lo, hi):
	"""Apply G7K15 to [lo, hi]; return (k15_estimate, error)."""
	mid = (lo + hi) / 2
	half = (hi - lo) / 2
	k15 = g7 = 0
	for i, x in enumerate(_K15_NODES):
		for sign in ([1] if x == 0 else [1, -1]):
			fx = f(mid + sign * x * half)
			k15 += _K15_WEIGHTS[i] * fx
			if _G7_WEIGHTS[i] is not None:
				g7 += _G7_WEIGHTS[i] * fx
	k15 *= half
	g7  *= half
	return k15, abs(k15 - g7)

def _adaptive_gk15(f, lo, hi, tol, depth=0):
	k15, err = _gk15(f, lo, hi)
	if err <= tol or depth >= 50:
		return k15
	mid = (lo + hi) / 2
	return (
		_adaptive_gk15(f, lo, mid, tol / 2, depth + 1) +
		_adaptive_gk15(f, mid, hi, tol / 2, depth + 1)
	)

@preparse(env, thunk, numeric_var, expr, expr, optional(expr, 1e-5))
def fn_int(env, formula, var, lo, hi, tol) -> float:
	with env.nest_guard('fnInt'), var.scoped():
		def f(x):
			var.value = x
			return formula.eval()
		return _adaptive_gk15(f, lo, hi, tol)


##################
# LIST FUNCTIONS #
##################

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

@preparse(list_var, rest(list_var), finalize='paren_cmd')
def sort_a(main_var, dep_vars):
	_sort(main_var, dep_vars, False)

@preparse(list_var, rest(list_var), finalize='paren_cmd')
def sort_d(main_var, dep_vars):
	_sort(main_var, dep_vars, True)

# You'd think dim could be a pure function, right? For a while, it was.
# But then I realized empty lists are illegal everywhere, with a single exception.
# dim( reads a variable's stored dimension rather than its value: it accesses
# .value (raw storage) instead of .resolve(), so dim(L1) returns 0 for an empty
# list where a bare reference to L1 would raise InvalidDimError.  This mirrors
# the calculator, where dim( works on both sides of → for the same reason.

@forms_func
def dim(a: ArgParser):
	if a.peek().is_list_start():
		var = a.list_var()
		val = var.value
		if val is None:
			raise UndefinedError(f"Undefined list variable")
		a.end_func()
		return len(val)

	value = a.expr()
	a.end_func()
	if isinstance(value, TiList):
		return len(value)
	# Don't need direct matrix variable access because empty matrices aren't possible
	if isinstance(value, TiMatrix):
		return TiList([value.rows, value.cols])
	raise DataTypeError(f"dim: expected list or matrix; got {value}")

@forms_func
def fill(a: ArgParser):
	fill_value = require_real(a.expr())
	if a.peek().is_matrix_var():
		lst = a.matrix_var().resolve()
		a.end_paren_cmd()
		for row in lst.data:
			for i in range(len(row)):
				row[i] = fill_value
	elif a.peek().is_list_start():
		lst = a.list_var().resolve()
		a.end_paren_cmd()
		for i in range(len(lst.data)):
			lst.data[i] = fill_value
	else:
		raise DataTypeError("Fill(: expected a list or matrix variable")

@preparse(env, thunk, numeric_var, expr, expr, optional(expr, 1))
def seq(env, formula, var, start, end, step) -> TiList:
	start = require_real(start)
	end = require_real(end)
	step = require_real(step)
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

@forms_func
def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(require_list(a.expr()))
		if not a.has_next:
			raise ArgumentError("List►matr: expected matrix variable as last argument")
		if a.peek().is_matrix_var():
			mat_var = a.matrix_var()
			break
	mat_var.value = TiMatrix([list(row) for row in zip_longest(*(lst.data for lst in list_vals), fillvalue=0.0)])
	a.end_paren_cmd()

@forms_func
def matr_to_list(a: ArgParser) -> None:
	mat = a.expr()
	if not isinstance(mat, TiMatrix):
		raise DataTypeError("Matr►list: first argument must be a matrix")
	if a.peek().is_list_start():
		list_vars = [a.list_var()]
		while a.has_next:
			list_vars.append(a.list_var())
		for var, col_data in zip(list_vars, zip(*mat.data)):
			var.value = TiList(list(col_data))
	else:
		col = py_int(a.expr()) - 1
		if not (0 <= col < mat.cols):
			raise InvalidDimError(
				f"Matr►list: column {col + 1} out of range for {mat.rows}×{mat.cols} matrix"
			)
		a.list_var().value = TiList([mat.data[r][col] for r in range(mat.rows)])
	a.end_paren_cmd()


##############
# STATISTICS #
##############

@preparse(list_var, rest(list_var), finalize='cmd')
def clr_list(first, rest_vars):
	"""ClrList list[, list, ...] — clear each named list to empty; silently skip nonexistent lists."""
	for var in [first] + rest_vars:
		lst = var.value
		if lst is not None:
			lst.clear()

@nullary_command
def clr_all_lists(env):
	"""ClrAllLists — set every defined list (L1–L6 and user lists) to empty."""
	for list_var in env.lists:
		if list_var.value is not None:
			list_var.value.clear()
	for lst in env.user_lists.values():
		lst.clear()

@preparse(env, rest(list_var_prefix_optional), finalize='cmd')
def set_up_editor(env, list_vars):
	"""SetUpEditor [list, ...] — ensure lists exist, creating empty ones as needed.

	With no arguments, ensures L1–L6 all exist (the default list editor columns).
	With arguments, ensures each named list exists (standard or user-defined).
	Does not modify lists that already contain data.
	"""
	if list_vars:
		for var in list_vars:
			if var.value is None:
				var.value = TiList([])
	else:
		for var in env.lists:
			if var.value is None:
				var.value = TiList([])

###########
# STRINGS #
###########

@preparse(equation_var, string_var, finalize='paren_cmd')
def equ_to_string(equ_var, str_var) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	str_var.value = TiString(equ_var.resolve().tokens)

@preparse(expr, equation_var, finalize='paren_cmd')
def string_to_equ(string, equ_var) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	equ_var.value = TiEquation(require_str(string).tokens)

############
# PROGRAMS #
############

@preparse(env, expr, finalize='cmd')
def if_cmd(env, cond):
	env.current_program().begin_if(bool(cond))

@forms_func
def then_cmd(a: ArgParser):
	"""Then without a preceding If: always a syntax error."""
	raise TiSyntaxError("Then without If")

@nullary_command
def else_cmd(env):
	"""If we encounter Else this way, always skip the block.
	(Else blocks are only executed when encountered while skipping an If-Then block.)"""
	env.current_program().begin_else()

@preparse(env, numeric_var, expr, expr, optional(expr, 1.0), finalize='paren_cmd')
def for_cmd(env, var, start, end, step):
	env.current_program().begin_for(
		var, require_real(start), require_real(end), require_real(step)
	)

@preparse(env, thunk, finalize='cmd')
def while_cmd(env, condition):
	env.current_program().begin_while(condition)

@preparse(env, thunk, finalize='cmd')
def repeat_cmd(env, condition):
	env.current_program().begin_repeat(condition)

@nullary_command
def end_cmd(env):
	env.current_program().end_block()

@preparse(env, label_name, finalize='cmd')
def lbl_cmd(env, name):
	"""Lbl is a no-op at runtime; just verify the syntax and that we're in a program."""
	env.current_program()  # raises if not in a program

@preparse(env, label_name, finalize='cmd')
def goto_cmd(env, name):
	env.current_program().goto(name)

@preparse(env, numeric_var, expr, finalize='paren_cmd')
def is_gt_cmd(env, var, threshold):
	env.current_program().is_gt(var, require_real(threshold))

@preparse(env, numeric_var, expr, finalize='paren_cmd')
def ds_lt_cmd(env, var, threshold):
	env.current_program().ds_lt(var, require_real(threshold))

@preparse(env, program_name, finalize='cmd')
def prgm(env, name):
	env.run_program(name)

@nullary_command
def return_cmd(env):
	env.current_program()  # raises if not in a program
	raise ReturnSignal()

@nullary_command
def stop_cmd(env):
	env.current_program()  # raises if not in a program
	raise StopSignal()

@forms_func
def del_var(a):
	"""DelVar variable — clear one variable without consuming the statement separator.

	Bunches: DelVar ADelVar B and DelVar ADisp X are both valid on the same line.
	Does not update Ans.
	"""
	a.any_var().value = None
	# No end_cmd() call; leaves separator in place for next command

@forms_func
def disp(a: ArgParser):
	if a.has_next:
		while True:
			print(a.expr())
			if not a.has_next:
				break
	else:
		pass  # a.env.focus_home()
	a.end_cmd()
