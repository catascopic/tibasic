from __future__ import annotations
import operator
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from decorators import forms_func, no_paren_func
from errors import DomainError, DataTypeError, ArgumentError, IncrementError, InvalidDimError, TiSyntaxError
from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_str


@forms_func
def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	args = a.parse_args()
	if isinstance(ans, TiMatrix):
		return ans[args]
	if len(args) != 1:
		raise ArgumentError(f"Too many arguments: {args}")
	(arg,) = args
	if isinstance(ans, TiList):
		return ans[arg]
	return ans * arg


@forms_func
def seq(a: ArgParser) -> TiList:
	formula = a.thunk()
	var = a.numeric_var()
	start = a.expr()
	end = a.expr()
	step = a.expr(optional=True, default=1)
	a.end()
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
	variable = var.variable
	with a.env.nest_guard(seq), a.env.scoped_var(variable):
		while op(n, end):
			variable.set(a.env, n)
			result.append(formula.eval())
			n += step
	return TiList(result)


@forms_func
def sigma(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
	start = a.expr()
	end = a.expr()
	a.end()
	total = 0
	n = start
	variable = var.variable
	with a.env.nest_guard(sigma), a.env.scoped_var(variable):
		while n <= end:
			variable.set(a.env, n)
			total += formula.eval()
			n += 1
	return total


@forms_func
def n_deriv(a: ArgParser) -> float:
	formula = a.thunk()
	var = a.numeric_var()
	val = a.expr()
	h = a.expr(optional=True, default=0.001)
	a.end()
	variable = var.variable
	with a.env.nest_guard(n_deriv, max_depth=1), a.env.scoped_var(variable):
		variable.set(a.env, val + h)
		fwd = formula.eval()
		variable.set(a.env, val - h)
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
	a.end()
	variable = var.variable
	with a.env.nest_guard('fnInt'), a.env.scoped_var(variable):
		def f(x):
			variable.set(a.env, x)
			return formula.eval()
		return _adaptive_gk15(f, lo, hi, tol)


# ── Matr►list( and List►matr( ────────────────────────────────────────────────

@forms_func
def matr_to_list(a: ArgParser) -> None:
	mat = a.expr()
	if not isinstance(mat, TiMatrix):
		raise DataTypeError("Matr►list: first argument must be a matrix")
	if a.peek().is_list_start():
		list_refs = [a.list_var()]
		while a.has_next():
			list_refs.append(a.list_var())
		for ref, col_data in zip(list_refs, zip(*mat.data)):
			ref.set(a.env, TiList(list(col_data)))
	else:
		col = require_int(a.expr()) - 1
		if not (0 <= col < mat.cols):
			raise InvalidDimError(
				f"Matr►list: column {col + 1} out of range for {mat.rows}×{mat.cols} matrix"
			)
		ref = a.list_var()
		ref.set(a.env, TiList([mat.data[r][col] for r in range(mat.rows)]))


@forms_func
def list_to_matr(a: ArgParser) -> None:
	list_vals = []
	while True:
		list_vals.append(require_list(a.expr()))
		if not a.has_next():
			raise ArgumentError("List►matr: expected matrix variable as last argument")
		if a.peek().is_matrix_var():
			mat_var = a.matrix_var()
			break
	mat_var.set(a.env, TiMatrix([
		list(row) for row in zip_longest(*[lst.data for lst in list_vals], fillvalue=0)
	]))


# ── SortA(, SortD(, Fill( ─────────────────────────────────────────────────────
# These are commands (cmd=), not functions — called as cmd(ArgParser) directly.

def _sort(a: ArgParser, reverse: bool):
	main = a.list_var().get(a.env)
	deps = []
	while a.has_next():
		deps.append(a.list_var().get(a.env))
	if not deps:
		main.data.sort(reverse=reverse)
	else:
		data = main.data
		indices = sorted(range(len(data)), key=lambda i: data[i], reverse=reverse)
		main.data = [data[i] for i in indices]
		for d in deps:
			d.data = [d.data[i] for i in indices]


def sort_a(a: ArgParser):
	_sort(a, False)  # ascending


def sort_d(a: ArgParser):
	_sort(a, True)   # descending


def fill(a: ArgParser):
	# Fill(value, listname) or Fill(value, matrixname) — value comes first
	x = require_real(a.expr())
	if a.peek().is_matrix_var():
		lst = a.matrix_var().get(a.env)
		a.end()
		for row in lst.data:
			for i in range(len(row)):
				row[i] = x
	else:
		lst = a.list_var().get(a.env)
		a.end()
		for i in range(len(lst.data)):
			lst.data[i] = x


# ── Equ►String( and String►Equ( ──────────────────────────────────────────────

@forms_func
def equ_to_string(a: ArgParser) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	equ_var = a.equation_var()
	str_var = a.string_var()
	equ = equ_var.get(a.env)
	str_var.set(a.env, TiString(list(equ.tokens)))


@forms_func
def string_to_equ(a: ArgParser) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	string = require_str(a.expr())
	equ_var = a.equation_var()
	equ_var.set(a.env, TiEquation(list(string.tokens)))


# ── Control flow ──────────────────────────────────────────────────────────────

def _for_continues(val: float, end_val: float, step: float) -> bool:
	"""True if the For loop body should execute for the given variable value."""
	if step > 0:
		return val <= end_val + 1e-10
	elif step < 0:
		return val >= end_val - 1e-10
	else:
		raise IncrementError("For: step cannot be zero")


@no_paren_func
def if_cmd(a):
	"""If condition — execute or skip the next statement (or delegate to Then)."""
	cond = bool(a.expr())
	a.end()
	p = a._parser
	# Peek past the trailing separator to see whether the next statement is Then
	saved = p.pos
	p.eat_statement_sep()
	next_tok = p.peek()
	p.pos = saved
	from tokens import THEN
	if next_tok is THEN:
		prog = p.env.current_program
		if prog is None:
			raise TiSyntaxError("Then without enclosing program")
		prog._pending_if_result = cond
	elif not cond:
		# One-line If: condition is False → skip the following statement
		p.eat_statement_sep()
		p.skip_statement()


@no_paren_func
def then_cmd(a):
	"""Then — begin the body of a conditional block."""
	a.end()
	p = a._parser
	prog = p.env.current_program
	if prog is None:
		raise TiSyntaxError("Then without enclosing program")
	result = prog._pending_if_result
	prog._pending_if_result = None
	if result is None:
		raise TiSyntaxError("Then without If")
	from program import ThenBlock
	if result:
		prog.push_block(ThenBlock())
	else:
		# Condition was False: skip to Else or End
		from tokens import ELSE
		found = p.scan_block_end(also_stop_at_else=True)
		if found is ELSE:
			# Execute the else-body; End will pop this block
			prog.push_block(ThenBlock())
		# If found is END, nothing more to do


@no_paren_func
def else_cmd(a):
	"""Else — skip the else-body (we just finished executing the then-body)."""
	a.end()
	p = a._parser
	prog = p.env.current_program
	if prog is None:
		raise TiSyntaxError("Else without enclosing program")
	from program import ThenBlock
	block = prog.pop_block()
	if not isinstance(block, ThenBlock):
		raise TiSyntaxError("Else without matching Then")
	p.scan_block_end()


@no_paren_func
def while_cmd(a):
	"""While condition — loop while condition is True."""
	p = a._parser
	cond_start = p.pos
	val = bool(a.expr())
	cond_end = p.pos
	a.end()
	if val:
		from parser import Thunk
		from program import WhileBlock
		thunk = Thunk(p.tokens[cond_start:cond_end], p.env)
		prog = p.env.current_program
		prog.push_block(WhileBlock(pos=p.pos, condition=thunk))
	else:
		p.scan_block_end()


@no_paren_func
def repeat_cmd(a):
	"""Repeat condition — loop until condition is True (body executes at least once)."""
	p = a._parser
	cond_start = p.pos
	a.expr()  # parse but discard — condition is only checked at End
	cond_end = p.pos
	a.end()
	from parser import Thunk
	from program import RepeatBlock
	thunk = Thunk(p.tokens[cond_start:cond_end], p.env)
	prog = p.env.current_program
	prog.push_block(RepeatBlock(pos=p.pos, condition=thunk))


@forms_func
def for_cmd(a):
	"""For(var, start, end[, step]) — iterate a numeric variable over a range."""
	from program import ForBlock
	var_tok = a.numeric_var()
	variable = var_tok.variable
	start   = require_real(a.expr())
	end_val = require_real(a.expr())
	step    = require_real(a.expr(optional=True, default=1.0))
	a.end()
	p = a._parser
	prog = p.env.current_program
	variable.set(p.env, start)
	if _for_continues(start, end_val, step):
		prog.push_block(ForBlock(pos=p.pos, var=variable, end_val=end_val, step=step))
	else:
		p.scan_block_end()


@no_paren_func
def end_cmd(a):
	"""End — close the innermost active block (For / While / Repeat / Then)."""
	a.end()
	p = a._parser
	prog = p.env.current_program
	if prog is None:
		raise TiSyntaxError("End without enclosing program")
	from program import ForBlock, WhileBlock, RepeatBlock, ThenBlock
	block = prog.pop_block()
	if isinstance(block, ForBlock):
		new_val = block.var.get(p.env) + block.step
		block.var.set(p.env, new_val)
		if _for_continues(new_val, block.end_val, block.step):
			p.pos = block.pos  # jump back to the separator before the loop body
	elif isinstance(block, WhileBlock):
		if block.condition.eval():
			p.pos = block.pos
	elif isinstance(block, RepeatBlock):
		if not block.condition.eval():
			p.pos = block.pos  # condition False → keep looping
		# condition True → fall through (exit loop)
	# ThenBlock: just pop and fall through


@no_paren_func
def lbl_cmd(a):
	"""Lbl name — mark a label; no-op at runtime (consumed as an identifier)."""
	a._parser.parse_label_name()


@no_paren_func
def goto_cmd(a):
	"""Goto name — jump to the named label in the current program."""
	p = a._parser
	prog = p.env.current_program
	if prog is None:
		raise TiSyntaxError("Goto outside program")
	name = p.parse_label_name()
	prog.goto(name)


@no_paren_func
def return_cmd(a):
	"""Return — exit the current sub-program and return to the caller."""
	a.end()
	from errors import ReturnSignal
	raise ReturnSignal()


@no_paren_func
def stop_cmd(a):
	"""Stop — terminate all program execution immediately."""
	a.end()
	from errors import StopSignal
	raise StopSignal()


@forms_func
def is_gt_cmd(a):
	"""IS>(var, value) — increment var; skip the next statement if var > value."""
	var_tok = a.numeric_var()
	variable = var_tok.variable
	threshold = require_real(a.expr())
	a.end()
	p = a._parser
	new_val = require_real(variable.get(p.env)) + 1
	variable.set(p.env, new_val)
	if new_val > threshold:
		p.eat_statement_sep()
		p.skip_statement()


@forms_func
def ds_lt_cmd(a):
	"""DS<(var, value) — decrement var; skip the next statement if var < value."""
	var_tok = a.numeric_var()
	variable = var_tok.variable
	threshold = require_real(a.expr())
	a.end()
	p = a._parser
	new_val = require_real(variable.get(p.env)) - 1
	variable.set(p.env, new_val)
	if new_val < threshold:
		p.eat_statement_sep()
		p.skip_statement()
