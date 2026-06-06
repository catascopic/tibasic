from __future__ import annotations
import operator
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

import operators
from decorators import forms_func, nullary_command, TiCall
from errors import DataTypeError, ArgumentError, IncrementError, InvalidDimError, DimMismatchError, UndefinedError, TiSyntaxError
from signals import ReturnSignal, StopSignal
from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_str, py_int


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
def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	if isinstance(ans, TiList):
		(index,) = a.parse_indices(1)
		return ans[index]
	if isinstance(ans, TiMatrix):
		return ans[a.parse_indices(2)]
	b = a.expr()
	a.end_func()
	return operators.mul.fn(ans, b)

@forms_func
def seq(a: ArgParser) -> TiList:
	formula = a.thunk()
	var = a.numeric_var()
	start = require_real(a.expr())
	end = require_real(a.expr())
	step = require_real(a.expr(optional=True, default=1))
	a.end_func()
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

	with a.env.nest_guard(seq), var.scoped():
		while op(n, end):
			var.value = n
			result.append(formula.eval())
			n += step

	return TiList(result)

@forms_func
def sigma(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
	start = a.expr()
	end = a.expr()
	a.end_func()
	total = 0
	n = start
	with a.env.nest_guard(sigma), var.scoped():
		while n <= end:
			var.value = n
			total += formula.eval()
			n += 1
	return total

@forms_func
def n_deriv(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
	val = a.expr()
	h = a.expr(optional=True, default=0.001)
	a.end_func()
	with a.env.nest_guard(n_deriv, max_depth=1), var.scoped():
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

@forms_func
def fn_int(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
	lo = a.expr()
	hi = a.expr()
	tol = a.expr(optional=True, default=1e-5)
	a.end_func()
	with a.env.nest_guard('fnInt'), var.scoped():
		def f(x):
			var.value = x
			return formula.eval()
		return _adaptive_gk15(f, lo, hi, tol)

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

def _sort(a: ArgParser, reverse: bool):
	main = a.list_var().resolve()
	deps = []
	while a.has_next:
		deps.append(a.list_var().resolve())
	a.end_paren_cmd()
	
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

@forms_func
def sort_a(a: ArgParser):
	_sort(a, False)

@forms_func
def sort_d(a: ArgParser):
	_sort(a, True)

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

@forms_func
def clr_list(a):
	"""ClrList list[, list, ...] — clear each named list to empty; silently skip nonexistent lists."""
	while True:
		lst = a.list_var().value
		if lst is not None:
			lst.clear()
		if not a.has_next:
			break
	a.end_cmd()

@nullary_command
def clr_all_lists(env):
	"""ClrAllLists — set every defined list (L1–L6 and user lists) to empty."""
	for list_var in env.lists:
		if list_var.value is not None:
			list_var.value.clear()
	for lst in env.user_lists.values():
		lst.clear()

@forms_func
def del_var(a):
	"""DelVar variable — clear one variable without consuming the statement separator.

	Bunches: DelVar ADelVar B and DelVar ADisp X are both valid on the same line.
	Does not update Ans.
	"""
	a.any_var().value = None
	# No end_cmd() call; leaves separator in place for next command

@forms_func
def set_up_editor(a):
	"""SetUpEditor [list, ...] — ensure lists exist, creating empty ones as needed.

	With no arguments, ensures L1–L6 all exist (the default list editor columns).
	With arguments, ensures each named list exists (standard or user-defined).
	Does not modify lists that already contain data.
	"""
	if a.has_next:
		while True:
			var = a.list_var_prefix_optional()
			if var.value is None:
				var.value = TiList([])
			if not a.has_next:
				break
	else:
		for list_var in a.env.lists:
			if list_var.value is None:
				list_var.value = TiList([])

	a.end_cmd()

@forms_func
def equ_to_string(a: ArgParser) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	equ_var = a.equation_var()
	str_var = a.string_var()
	a.end_paren_cmd()
	str_var.value = TiString(equ_var.resolve().tokens)

@forms_func
def string_to_equ(a: ArgParser) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	string = require_str(a.expr())
	equ_var = a.equation_var()
	a.end_paren_cmd()
	equ_var.value = TiEquation(string.tokens)

####################
# PROGRAM COMMANDS #
####################

class program_command(TiCall):
	"""Decorator for no-arg commands that require a running program.

	Calls no_args() + end(), then passes the current Program to the function.
	Raises InvalidCommandError automatically if called outside a program.
	Use for Return, Stop, Else, End, etc.
	"""
	def call_with_parser(self, a: ArgParser) -> None:
		a.no_args()
		a.end_cmd()
		self.func(a.current_program())

@forms_func
def if_cmd(a: ArgParser):
	cond = bool(a.expr())
	a.end_cmd()
	a.current_program().begin_if(cond)

@forms_func
def then_cmd(a: ArgParser):
	"""Then without a preceding If: always a syntax error."""
	raise TiSyntaxError("Then without If")

@program_command
def else_cmd(prgm):
	"""If we encounter Else this way, always skip the block.
	(Else blocks are only executed when encountered while skipping an If-Then block.)"""
	prgm.begin_else()

@forms_func
def while_cmd(a: ArgParser):
	condition = a.thunk()
	a.end_cmd()
	a.current_program().begin_while(condition)

@forms_func
def repeat_cmd(a: ArgParser):
	condition = a.thunk()
	a.end_cmd()
	a.current_program().begin_repeat(condition)

@forms_func
def for_cmd(a: ArgParser):
	var   = a.numeric_var()
	start = require_real(a.expr())
	end   = require_real(a.expr())
	step  = require_real(a.expr(optional=True, default=1.0))
	a.end_paren_cmd()
	a.current_program().begin_for(var, start, end, step)

@program_command
def end_cmd(prgm):
	prgm.end_block()

@forms_func
def lbl_cmd(a: ArgParser):
	"""Lbl is a no-op at runtime; just verify the syntax."""
	a.label_name()
	a.end_cmd()
	a.current_program()  # raises if not in a program

@forms_func
def goto_cmd(a: ArgParser):
	name = a.label_name()
	a.end_cmd()
	a.current_program().goto(name)

@program_command
def return_cmd(prgm):
	raise ReturnSignal()

@program_command
def stop_cmd(prgm):
	raise StopSignal()

@forms_func
def is_gt_cmd(a: ArgParser):
	var = a.numeric_var()
	threshold = require_real(a.expr())
	a.end_paren_cmd()
	a.current_program().is_gt(var, threshold)

@forms_func
def ds_lt_cmd(a: ArgParser):
	var = a.numeric_var()
	threshold = require_real(a.expr())
	a.end_paren_cmd()
	a.current_program().ds_lt(var, threshold)

@forms_func
def prgm(a: ArgParser):
	prgm_name = a.program_name()
	a.end_cmd()
	a.env.run_program(prgm_name)

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
