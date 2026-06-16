from parser import ArgParser, Parser
from titoken import QUOTE

from preparse import (
	preparse_cmd, preparse_cmd_func, preparse_bunch,
	Thunk, NumericVar, LabelName, ProgramName, AnyVar, Real, Env,
)
from environment import ReturnSignal, StopSignal
from preparse import special_func, no_arg_command
from core import TiString, py_int, require_string
from numberformat import ti83_format
from errors import TiSyntaxError, DataTypeError, DomainError

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

def _home_text(value) -> str:
	"""The home-screen string for a value (a real number or a string).

	Complex/list/matrix values aren't handled yet — they raise, matching Text('s
	current scope; expand here when Disp needs them.
	"""
	if isinstance(value, float):
		return ti83_format(value)
	if isinstance(value, TiString):
		return str(value)
	raise DataTypeError(f"Disp: expected a real number or string, got {type(value).__name__}")


@special_func
def disp(args: ArgParser):
	"""Disp [value[,value...]] — append each value to the home screen (no args just
	re-renders it)."""
	home = args.env.home
	while args.has_next:
		home.disp(_home_text(args.expr()))
	args.end_cmd()
	args.env.console.update(home)


@special_func
def output(args: ArgParser):
	"""Output(row, col, value) — write value at a fixed 1-indexed home-screen cell."""
	row = py_int(args.expr())
	col = py_int(args.expr())
	value = args.expr()
	args.end_paren_cmd()
	if not (1 <= row <= args.env.home.ROWS and 1 <= col <= args.env.home.COLS):
		raise DomainError(f"Output(: position out of range: ({row}, {col})")
	args.env.home.output(row - 1, col - 1, _home_text(value))
	args.env.console.update(args.env.home)


@no_arg_command
def clr_home(env):
	"""ClrHome — clear the home screen and reset the Disp cursor."""
	env.home.clear()
	env.console.update(env.home)


@special_func
def pause_cmd(args: ArgParser):
	"""Pause [value] — show value (Disp-style), then block until Enter.

	The optional value is stored to Ans (a real quirk: no other command does this).
	ERR:INVALID outside a program, like Goto/Return/Stop.
	"""
	env = args.env
	env.current_program()       # raises ERR:INVALID outside a program
	if args.has_next:
		value = args.expr()
		env.home.disp(_home_text(value))
		env.ans = value
	args.end_cmd()
	env.console.pause(env.home)


def _eval_input(text: str, env) -> object:
	"""Tokenize text the user typed and evaluate it as one TI expression (like expr()."""
	try:
		tokens = TiString.from_str(text).tokens
	except KeyError as bad:
		raise TiSyntaxError(f"Input: unsupported character {bad} in {text!r}")
	parser = Parser(tokens, env)
	value = parser.parse_expr()
	if parser.has_next:
		raise TiSyntaxError(f"Input: expected a single expression, got {text!r}")
	return value


@special_func
def input_cmd(args: ArgParser):
	"""Input ["prompt",] var — read an expression from the console and store it in var.

	The bare graph-cursor form (Input with no arguments) isn't supported yet.
	"""
	if not args.has_next:
		raise TiSyntaxError("Input: the graph-cursor form is not supported yet")
	prompt = '?'
	if args.peek().code == QUOTE:        # Input "prompt", var
		prompt = str(args.expr())
	var = args.any_var()
	args.end_cmd()
	var.store(_eval_input(args.env.console.read_value(prompt), args.env))


@special_func
def menu_cmd(args: ArgParser):
	"""Menu("title","opt1",lbl1[,...,"opt7",lbl7]) — show a menu, Goto the choice.

	Like Goto, this branches, so it can only run inside a program.  The 1-to-7
	option count isn't checked explicitly: expr()'s missing-argument error covers
	zero options, and end_paren_cmd's leftover-token error covers more than seven.
	"""
	program = args.env.current_program()       # raises ERR:INVALID outside a program
	title = require_string(args.expr())
	options: list[str] = []
	labels: list[str] = []
	for _ in range(7):
		options.append(require_string(args.expr()))
		labels.append(args.label_name())
		if not args.has_next:
			break
	args.end_paren_cmd()

	program.goto(labels[args.env.console.choose(title, options)])
