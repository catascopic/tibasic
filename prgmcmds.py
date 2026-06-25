import time

from parser import ArgParser, Parser
from titoken import QUOTE, ANS, EOF_CODE, COLON, NEWLINE

from preparse import (
	preparse_cmd, preparse_cmd_func, preparse_bunch,
	Thunk, NumericVar, LabelName, ProgramName, AnyVar, Real, Env, AnyValue,
)
from accessors import Reference
from environment import ReturnSignal, StopSignal
from preparse import special_func, no_arg_command
from core import TiString, py_int, require_string
from errors import TiSyntaxError, UndefinedError, DomainError
from tiformat import output_text
from menuscreen import MenuScreen
from modes import Screen

############
# PROGRAMS #
############

@preparse_cmd
def if_cmd(env: Env, cond: Real):
	env.current_execution().begin_if(bool(cond))

@no_arg_command
def then_cmd(a: ArgParser):
	"""Then without a preceding If: always a syntax error."""
	raise TiSyntaxError("Then without If")

@no_arg_command
def else_cmd(env):
	"""If we encounter Else this way, always skip the block.
	(Else blocks are only executed when encountered while skipping an If-Then block.)"""
	env.current_execution().begin_else()

@preparse_cmd_func
def for_cmd(env: Env, var: NumericVar, start: Real, end: Real, step: Real = 1.0):
	env.current_execution().begin_for(var, start, end, step)

@preparse_cmd
def while_cmd(env: Env, condition: Thunk):
	env.current_execution().begin_while(condition)

@preparse_cmd
def repeat_cmd(env: Env, condition: Thunk):
	env.current_execution().begin_repeat(condition)

@no_arg_command
def end_cmd(env):
	env.current_execution().end_block()

@preparse_cmd
def lbl_cmd(env: Env, name: LabelName):
	"""Lbl is a no-op at runtime; just verify the syntax and that we're in a program."""
	env.current_execution()  # raises if not in a program

@preparse_cmd
def goto_cmd(env: Env, name: LabelName):
	env.current_execution().goto(name)

@preparse_cmd_func
def is_gt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_execution().is_gt(var, threshold)

@preparse_cmd_func
def ds_lt_cmd(env: Env, var: NumericVar, threshold: Real):
	env.current_execution().ds_lt(var, threshold)

@preparse_cmd
def prgm(env: Env, name: ProgramName):
	try:
		program = env.programs[name]
	except KeyError:
		raise UndefinedError(f"Program not found: {name!r}")
	program.run(env)

@no_arg_command
def return_cmd(env):
	env.current_execution()  # raises if not in a program
	raise ReturnSignal()

@no_arg_command
def stop_cmd(env):
	env.current_execution()  # raises if not in a program
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
	env = args.env
	env.screen = Screen.HOME            # Disp brings up the home screen
	if not args.has_next:
		env.console.present()           # bare Disp just re-presents the screen
	while args.has_next:
		env.home.disp(args.expr())
		env.console.present()
	args.end_cmd()
	time.sleep(0.05)


@preparse_cmd_func
def output(env: Env, row: Real, col: Real, value: AnyValue):
	"""Output(row, col, value) — write value at a fixed 1-indexed cell.

	The value is rendered linearly (as you'd type it): lists/matrices use comma
	separators and a matrix is inlined onto the single starting position, wrapping
	across cells from there."""
	env.screen = Screen.HOME  # Output( brings up the home screen
	r, c = py_int(row), py_int(col)
	if not (1 <= r <= env.home.ROWS and 1 <= c <= env.home.COLS):
		raise DomainError(f"Output(: position out of range: ({r}, {c})")
	env.home.output(r - 1, c - 1, output_text(value))
	env.console.present()


@no_arg_command
def clr_home(env):
	"""ClrHome — clear the text screen.  Does not change which screen is displayed."""
	env.home.clear()
	env.console.present()


@no_arg_command
def disp_graph(env):
	"""DispGraph — display the graph screen, re-plotting the active functions."""
	env.display_graph()
	env.console.present()


@no_arg_command
def disp_table(env):
	"""DispTable — switch to the table screen (the table itself isn't implemented)."""
	env.screen = Screen.TABLE


@preparse_cmd
def pause_cmd(env: Env, value: AnyValue = None):
	"""Pause [value] — show value, then block until the user continues.

	The device renders and waits; the home-screen device makes a too-big list/matrix
	scrollable with the arrow keys (see ScrollView), but Pause doesn't decide that.
	The optional value is stored to Ans (a real quirk: no other command does this).
	ERR:INVALID outside a program, like Goto/Return/Stop.
	"""
	env.current_execution()       # raises ERR:INVALID outside a program
	if value is not None:
		env.ans = value
	env.console.pause(value)


def _input_one(env, prompt: TiString | None, var: Reference, raw_string: bool = False) -> None:
	"""Read one value for `var` and store it.  Shared by Input and Prompt.

	When `raw_string` is True (Input only), a string variable stores the typed
	text verbatim — quotes are not required and are included literally if typed.
	Otherwise (Prompt, or non-string vars) the typed text is evaluated as an
	expression, same as expr(.
	"""
	env.screen = Screen.HOME            # Input/Prompt bring up the home screen
	try:
		tokens = env.console.read_tokens(prompt)
	except KeyError as bad:
		# A character the calculator's keypad couldn't type (e.g. a multi-char
		# function name); the keypad-restricted consoles raise this on tokenizing.
		raise TiSyntaxError(f"Input: unsupported character {bad}")
	# Mirror the entry onto the home grid (the console paints it on present); the
	# prompt's own bytes precede the typed bytes, exactly as on the calculator.
	if tokens:
		prompt_bytes = b''.join(t.display for t in prompt.tokens) if prompt is not None else b''
		env.home.echo(prompt_bytes + b''.join(t.display for t in tokens))
		env.console.present()
	if raw_string and var.accessor.kind == 'string':
		var.store(TiString(tokens))
	else:
		parser = Parser(tokens, env)
		value = parser.parse_expr()
		if parser.has_next:
			raise TiSyntaxError(f"Input: expected a single expression, got {tokens}")
		var.store(value)


_SUB = 0xBB0C  # sub( — the one string function the calculator accepts in a prompt

def _prompt_starter(tok) -> bool:
	"""Whether `tok` begins an Input/Prompt display string rather than the target var.

	The calculator decides purely from the first token: a string literal, a string
	variable, Ans, or sub( introduce a display string; anything else (a numeric/list/
	matrix variable, a number, '(' …) means there's no prompt and this token is the
	storage target itself.

	That's the whole check — once the first token says "prompt", the rest is parsed
	as an ordinary expression.  So a display string can't *start* with anything but
	those four, but is otherwise unrestricted: "A"+5 fails at evaluation (can't add a
	number to a string), and the clock functions getDtStr(/getTmStr( — which the real
	calculator rejects from a prompt — are simply allowed here."""
	return tok.code in (QUOTE, ANS, _SUB) or tok.is_string_var()


@special_func
def input_cmd(args: ArgParser):
	"""Input [strexpr,] var — read from the console and store in var.

	The optional display string is a string expression that begins with a string
	literal, string variable, Ans, or sub( (see _prompt_starter), followed by a comma
	and the target variable.  When the first argument isn't one of those (a numeric/
	list/matrix variable, a number, …) there's no prompt and it is the target; the
	bare string-variable form `Input Str1` likewise stores without a prompt.

	For string variables the typed text is stored verbatim (no quotes needed; any
	typed quotes are included literally).  The bare graph-cursor form (no arguments)
	isn't supported yet.
	"""
	if not args.has_next:
		raise TiSyntaxError("Input: the graph-cursor form is not supported yet")
	q_tok = TiString.from_str('?').tokens[0]
	env = args.env

	if _prompt_starter(args.peek()):
		if args.peek().is_string_var() and args.peek_next().code in {EOF_CODE, COLON, NEWLINE}:
			# Input Str1 with nothing after — lone string var is the target
			prompt = TiString([q_tok])
			var = args.any_var_or_user_list()
		else:
			# Prompt: QUOTE/ANS/SUB are never targets; StrN here starts a longer expression
			prompt_str = require_string(args.expr())
			prompt = TiString(list(prompt_str.tokens) + [q_tok])
			var = args.any_var_or_user_list()
	else:
		prompt = TiString([q_tok])
		var = args.any_var_or_user_list()

	args.end_cmd()
	_input_one(env, prompt, var, raw_string=True)


@special_func
def prompt_cmd(args: ArgParser):
	"""Prompt var[,var...] — Input each variable in turn, with an implicit
	"NAME=?" prompt instead of a custom one."""
	pending = []
	while args.has_next:
		tok = args.peek()
		pending.append((tok, args.any_var_or_user_list()))
	args.end_cmd()
	env = args.env
	eq_token = TiString.from_str('=').tokens[0]
	q_token  = TiString.from_str('?').tokens[0]
	for tok, var in pending:
		prompt = TiString([tok, eq_token, q_token])
		_input_one(env, prompt, var)


@special_func
def menu_cmd(args: ArgParser):
	"""Menu("title","opt1",lbl1[,...,"opt7",lbl7]) — show a menu, Goto the choice.

	Like Goto, this branches, so it can only run inside a program.  The 1-to-7
	option count isn't checked explicitly: expr()'s missing-argument error covers
	zero options, and end_paren_cmd's leftover-token error covers more than seven.
	"""
	program = args.env.current_execution()       # raises ERR:INVALID outside a program
	title = str(require_string(args.expr()))
	options: list[str] = []
	labels: list[str] = []
	for _ in range(7):
		options.append(str(require_string(args.expr())))
		labels.append(args.label_name())
		if not args.has_next:
			break
	args.end_paren_cmd()

	# Put the menu up as a transient modal screen over whatever was showing; the
	# console drives the highlight and returns the chosen index.  Restore the prior
	# screen and clear the modal before branching to the chosen label.
	env = args.env
	prev_screen = env.screen
	env.menu = MenuScreen(title, options)
	env.screen = Screen.MENU
	try:
		index = env.console.menu()
	finally:
		env.menu = None
		env.screen = prev_screen
	program.goto(labels[index])
