from __future__ import annotations
import operator
from itertools import zip_longest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import ArgParser

from decorators import forms_func
from errors import DomainError, DataTypeError, ArgumentError, IncrementError, InvalidDimError, UndefinedError, TiSyntaxError
from signals import ReturnSignal, StopSignal
from tiobjects import TiList, TiMatrix, TiString, TiEquation, require_num, require_real, require_int, require_list, require_str


@forms_func
def ans_index_or_mul(a: ArgParser):
	ans = a.env.ans
	args = a.parse_args()
	a.end_func()
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
	a.end_func()
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
	a.end_func()
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
	a.end_func()
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
	a.end_paren_cmd()


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
	a.end_paren_cmd()


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
	a.end_paren_cmd()


def sort_a(a: ArgParser):
	_sort(a, False)  # ascending


def sort_d(a: ArgParser):
	_sort(a, True)   # descending


def fill(a: ArgParser):
	# Fill(value, listname) or Fill(value, matrixname) — value comes first
	x = require_real(a.expr())
	if a.peek().is_matrix_var():
		lst = a.matrix_var().get(a.env)
		a.end_paren_cmd()
		for row in lst.data:
			for i in range(len(row)):
				row[i] = x
	else:
		lst = a.list_var().get(a.env)
		a.end_paren_cmd()
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
	a.end_paren_cmd()


@forms_func
def string_to_equ(a: ArgParser) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	string = require_str(a.expr())
	equ_var = a.equation_var()
	equ_var.set(a.env, TiEquation(list(string.tokens)))
	a.end_paren_cmd()


# ── Control flow ──────────────────────────────────────────────────────────────

@forms_func
def if_cmd(a: ArgParser):
	"""If condition — execute or skip the next statement (or delegate to Then)."""
	cond = bool(a.expr())
	a.end_cmd()
	a.current_program('If').begin_if(cond)


@forms_func
def then_cmd(a: ArgParser):
	"""Then without a preceding If — always a syntax error."""
	a.end_cmd()
	raise TiSyntaxError("Then without If")


@forms_func
def else_cmd(a: ArgParser):
	"""Else — skip the else-body (we just finished executing the then-body)."""
	a.end_cmd()
	a.current_program('Else').begin_else()


@forms_func
def while_cmd(a: ArgParser):
	"""While condition — loop while condition is True."""
	thunk = a.thunk()
	a.end_cmd()
	a.current_program('While').begin_while(thunk)


@forms_func
def repeat_cmd(a: ArgParser):
	"""Repeat condition — loop until condition is True (body executes at least once)."""
	thunk = a.thunk()
	a.end_cmd()
	a.current_program('Repeat').begin_repeat(thunk)


@forms_func
def for_cmd(a: ArgParser):
	"""For(var, start, end[, step]) — iterate a numeric variable over a range."""
	var_tok = a.numeric_var()
	start   = require_real(a.expr())
	end_val = require_real(a.expr())
	step    = require_real(a.expr(optional=True, default=1.0))
	a.end_paren_cmd()
	a.current_program('For(').begin_for(var_tok.variable, start, end_val, step)


@forms_func
def end_cmd(a: ArgParser):
	"""End — close the innermost active block (For / While / Repeat / Then)."""
	a.end_cmd()
	a.current_program('End').end_block()


@forms_func
def lbl_cmd(a: ArgParser):
	"""Lbl name — mark a label; no-op at runtime."""
	a.label_name()
	a.end_cmd()


@forms_func
def goto_cmd(a: ArgParser):
	"""Goto name — jump to the named label in the current program."""
	name = a.label_name()
	a.end_cmd()
	a.current_program('Goto').goto(name)


@forms_func
def return_cmd(a: ArgParser):
	"""Return — exit the current sub-program and return to the caller."""
	a.end_cmd()
	raise ReturnSignal()


@forms_func
def stop_cmd(a: ArgParser):
	"""Stop — terminate all program execution immediately."""
	a.end_cmd()
	raise StopSignal()


@forms_func
def is_gt_cmd(a: ArgParser):
	"""IS>(var, value) — increment var; skip the next statement if var > value."""
	var_tok = a.numeric_var()
	threshold = require_real(a.expr())
	a.end_paren_cmd()
	a.current_program('IS>(').is_gt(var_tok.variable, threshold)


@forms_func
def ds_lt_cmd(a: ArgParser):
	"""DS<(var, value) — decrement var; skip the next statement if var < value."""
	var_tok = a.numeric_var()
	threshold = require_real(a.expr())
	a.end_paren_cmd()
	a.current_program('DS<(').ds_lt(var_tok.variable, threshold)


@forms_func
def prgm(a: ArgParser):
	"""prgm NAME — execute the stored sub-program named NAME."""
	name = a.program_name()
	a.end_cmd()
	try:
		prgm_code = a.env.programs[name]
	except KeyError:
		raise UndefinedError(f"Program not found: {name!r}")
	from program import Program
	Program(prgm_code, a.env).run()
