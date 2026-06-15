from itertools import zip_longest

from parser import ArgParser

from preparse import (
	preparse_cmd, preparse_cmd_func, preparse_bunch,
	Thunk, NumericVar, LabelName, ProgramName, AnyVar, Real, Env,
)
from environment import ReturnSignal, StopSignal
from preparse import special_func, no_arg_command
from errors import TiSyntaxError

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

@special_func
def disp(args: ArgParser):
	if args.has_next:
		while True:
			print(args.expr())
			if not args.has_next:
				break
	else:
		pass  # args.env.focus_home()
	args.end_cmd()
