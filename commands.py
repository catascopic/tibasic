from parser import ArgParser, Parser
from titoken import QUOTE

from preparse import (
	preparse_cmd, preparse_cmd_func, preparse_bunch,
	Thunk, NumericVar, LabelName, ProgramName, AnyVar, Real, Env, AnyValue,
)
from environment import ReturnSignal, StopSignal
from preparse import special_func, no_arg_command
from core import TiString, StringVariable, py_int, require_string
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
	try:
		prgm_code = env.programs[name]
	except KeyError:
		raise UndefinedError(f"Program not found: {name!r}")
	from program import Program
	Program(prgm_code, env).run()

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
	var.delete()

@special_func
def disp(args: ArgParser):
	"""Disp [value[,value...]] — show each value (no args just re-renders).  Every TI
	data type is supported; how it's laid out is the device's call (the home screen
	right-aligns numbers/lists/matrices, a matrix one line per row, strings left)."""
	io = args.env.io
	if not args.has_next:
		io.refresh()
	while args.has_next:
		io.disp(args.expr())
	args.end_cmd()


@special_func
def output(args: ArgParser):
	"""Output(row, col, value) — write value at a fixed 1-indexed cell.

	The value is rendered linearly (as you'd type it): lists/matrices use comma
	separators and a matrix is inlined onto the single starting position, wrapping
	across cells from there."""
	row = py_int(args.expr())
	col = py_int(args.expr())
	value = args.expr()
	args.end_paren_cmd()
	args.env.io.output(row, col, value)


@no_arg_command
def clr_home(env):
	"""ClrHome — clear the text screen."""
	env.io.clear_home()


@no_arg_command
def zoom_sto(env):
	"""ZoomSto — save the current window into the Zoom memory."""
	env.zoom_store()


@no_arg_command
def zoom_rcl(env):
	"""ZoomRcl — restore the window saved by the last ZoomSto."""
	env.zoom_recall()


@preparse_cmd
def pause_cmd(env: Env, value: AnyValue = None):
	"""Pause [value] — show value, then block until the user continues.

	The device renders and waits; the home-screen device makes a too-big list/matrix
	scrollable with the arrow keys (see ScrollView), but Pause doesn't decide that.
	The optional value is stored to Ans (a real quirk: no other command does this).
	ERR:INVALID outside a program, like Goto/Return/Stop.
	"""
	env.current_program()       # raises ERR:INVALID outside a program
	if value is not None:
		env.ans = value
	env.io.pause(value)


def _tokenize_input(text: str) -> list:
	"""Convert console-typed text to tokens, restricted to TI's typeable set.

	This is the same restriction the real keypad imposes — there's no key for
	`sin(` or similar — so it doubles as validation: a character with no
	typeable token raises before anything is shown or evaluated.
	"""
	try:
		return TiString.from_str(text).tokens
	except KeyError as bad:
		raise TiSyntaxError(f"Input: unsupported character {bad} in {text!r}")


def _eval_input(tokens: list, env) -> object:
	"""Evaluate a typed token sequence as one TI expression (like expr()."""
	parser = Parser(tokens, env)
	value = parser.parse_expr()
	if parser.has_next:
		text = ''.join(t.text for t in tokens)
		raise TiSyntaxError(f"Input: expected a single expression, got {text!r}")
	return value


def _input_one(env, prompt: str, var) -> None:
	"""Read one value for `var` and store it.  Shared by Input and Prompt.

	The device reads the text and (if it manages a screen) commits the typed line to
	it; this just validates and stores.

	A string variable takes the typed text as a literal string, verbatim — no
	quotes, no expression evaluation (you can't type a quote-enclosed string
	expression on the real keypad input line either).  Every other variable
	type evaluates the typed text as an expression, same as expr(.

	Empty input is rejected: an entry that's blank (or only whitespace) re-prompts
	rather than storing anything, so neither Input nor Prompt can yield a value.
	"""
	while not (text := env.io.read_value(prompt).strip()):
		pass
	tokens = _tokenize_input(text)
	if isinstance(var, StringVariable):
		var.store(TiString(tokens))
	else:
		var.store(_eval_input(tokens, env))


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
	_input_one(args.env, prompt, var)


@special_func
def prompt_cmd(args: ArgParser):
	"""Prompt var[,var...] — Input each variable in turn, with an implicit
	"NAME=?" prompt instead of a custom one."""
	pending = []
	while args.has_next:
		name = args.peek().text
		pending.append((name, args.any_var()))
	args.end_cmd()
	env = args.env
	for name, var in pending:
		_input_one(env, f'{name}=?', var)


@special_func
def menu_cmd(args: ArgParser):
	"""Menu("title","opt1",lbl1[,...,"opt7",lbl7]) — show a menu, Goto the choice.

	Like Goto, this branches, so it can only run inside a program.  The 1-to-7
	option count isn't checked explicitly: expr()'s missing-argument error covers
	zero options, and end_paren_cmd's leftover-token error covers more than seven.
	"""
	program = args.env.current_program()       # raises ERR:INVALID outside a program
	title = str(require_string(args.expr()))
	options: list[str] = []
	labels: list[str] = []
	for _ in range(7):
		options.append(str(require_string(args.expr())))
		labels.append(args.label_name())
		if not args.has_next:
			break
	args.end_paren_cmd()

	program.goto(labels[args.env.io.menu(title, options)])
