from core import TiString, TiEquation, require_num, require_string, py_int
from preparse import preparse_func, preparse_cmd_func, forms_func, Real, Env, StringVar, EquationVar
from errors import DomainError, InvalidDimError, TiSyntaxError
from parser import ArgParser, Parser


@preparse_func
def length(string: TiString):
	return len(string)

@preparse_func
def in_string(string: TiString, substring: TiString, start: Real = 1.0):
	v = string.tokens
	s = substring.tokens
	start = py_int(start)
	for i in range(start - 1, len(v) - len(s) + 1):
		if v[i:i + len(s)] == s:
			return i + 1

	return 0

@forms_func
def sub(args: ArgParser):
	first = args.expr()
	if not args.has_next:
		# sub( with a single numeric arg divides by 100, like the undocumented % operator.
		args.end_func()
		return require_num(first) / 100

	string = require_string(first)
	start = py_int(args.expr())
	length_val = py_int(args.expr())
	args.end_func()

	if length_val < 1:
		raise DomainError(f"sub: length must be ≥ 1, got {length_val}")
	if not (1 <= start <= len(string) - length_val + 1):
		raise InvalidDimError("sub: index out of range")
	return TiString(string.tokens[start - 1 : start + length_val - 1])

@preparse_func
def expr(env: Env, string: TiString):
	"""Evaluate a TiString as a TI-BASIC expression."""
	with env.nest_guard(expr):
		p = Parser(string.tokens, env)
		result = p.parse_expr()
		if p.has_next:
			raise TiSyntaxError(f"expr: evaluated string must contain a single expression; got: {string!r}")
		return result

@preparse_cmd_func
def equ_to_string(equ_var: EquationVar, str_var: StringVar) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	str_var.value = TiString(equ_var.resolve().tokens)

@preparse_cmd_func
def string_to_equ(string: TiString, equ_var: EquationVar) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	equ_var.value = TiEquation(require_string(string).tokens)
