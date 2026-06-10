from __future__ import annotations
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from preparse import (
	preparse_cmd, preparse_cmd_func, preparse_bunch,
	Thunk, NumericVar, ListVar, ListVarPrefixOptional, EquationVar, StringVar,
	LabelName, ProgramName, AnyVar, Real, Env,
)
from decorators import forms_func, no_arg_command
from errors import DataTypeError, ArgumentError, InvalidDimError, DimMismatchError, TiSyntaxError
from signals import ReturnSignal, StopSignal
from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_real, require_list, require_str, py_int


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

@preparse_cmd_func
def sort_a(main_var: ListVar, *dep_vars: ListVar):
	_sort(main_var, dep_vars, False)

@preparse_cmd_func
def sort_d(main_var: ListVar, *dep_vars: ListVar):
	_sort(main_var, dep_vars, True)

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

@forms_func
def list_to_matr(args: ArgParser) -> None:
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


##############
# STATISTICS #
##############

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

@preparse_cmd_func
def equ_to_string(equ_var: EquationVar, str_var: StringVar) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	str_var.value = TiString(equ_var.resolve().tokens)

@preparse_cmd_func
def string_to_equ(string: TiString, equ_var: EquationVar) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	equ_var.value = TiEquation(require_str(string).tokens)

############
# PROGRAMS #
############

@preparse_cmd
def if_cmd(env: Env, cond: Real):
	env.current_program().begin_if(bool(cond))

@no_arg_command
def then_cmd(a: ArgParser):
	"""Then without a preceding If: always a syntax error."""
	raise TiSyntaxError("Then without If")

@no_arg_command
def else_cmd(env):
	"""If we encounter Else this way, always skip the block.
	(Else blocks are only executed when encountered while skipping an If-Then block.)"""
	env.current_program().begin_else()

@preparse_cmd_func
def for_cmd(env: Env, var: NumericVar, start: Real, end: Real, step: Real = 1.0):
	env.current_program().begin_for(var, start, end, step)

@preparse_cmd
def while_cmd(env: Env, condition: Thunk):
	env.current_program().begin_while(condition)

@preparse_cmd
def repeat_cmd(env: Env, condition: Thunk):
	env.current_program().begin_repeat(condition)

@no_arg_command
def end_cmd(env):
	env.current_program().end_block()

@preparse_cmd
def lbl_cmd(env: Env, name: LabelName):
	"""Lbl is a no-op at runtime; just verify the syntax and that we're in a program."""
	env.current_program()  # raises if not in a program

@preparse_cmd
def goto_cmd(env: Env, name: LabelName):
	env.current_program().goto(name)

@preparse_cmd_func
def is_gt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_program().is_gt(var, threshold)

@preparse_cmd_func
def ds_lt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_program().ds_lt(var, threshold)

@preparse_cmd
def prgm(env: Env, name: ProgramName):
	env.run_program(name)

@no_arg_command
def return_cmd(env):
	env.current_program()  # raises if not in a program
	raise ReturnSignal()

@no_arg_command
def stop_cmd(env):
	env.current_program()  # raises if not in a program
	raise StopSignal()

@preparse_bunch
def del_var(var: AnyVar):
	"""DelVar variable — clear one variable without consuming the statement separator.

	end=NONE leaves the parser untouched (no finalizer), so DelVar bunches with
	whatever follows: DelVar ADelVar B and DelVar ADisp X are both valid on the
	same line.  Does not update Ans.
	"""
	var.value = None

@forms_func
def disp(args: ArgParser):
	if args.has_next:
		while True:
			print(args.expr())
			if not args.has_next:
				break
	else:
		pass  # args.env.focus_home()
	args.end_cmd()
