"""String and equation types, variables, validators, and functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core import TiString, TiEquation, require_str
from core import py_int
from preparse import preparse_func, preparse_cmd_func, Real, Env, StringVar, EquationVar
from errors import DomainError, InvalidDimError, ArgumentError, TiSyntaxError

if TYPE_CHECKING:
	from parser import ArgParser
	from environment import Environment


# ── String functions ──────────────────────────────────────────────────────────

@preparse_func
def length(string: TiString):
	return len(string)


@preparse_func
def in_string(string: TiString, substring: TiString, start: Real = 1):
	v = string.tokens
	s = substring.tokens
	start = py_int(start)
	for i in range(start - 1, len(v) - len(s) + 1):
		if v[i:i + len(s)] == s:
			return i + 1
	return 0


from preparse import forms_func


@forms_func
def sub(args: ArgParser):
	# sub( with a single numeric arg divides by 100, like the undocumented % operator.
	values = args.parse_args()
	args.end_func()
	if len(values) == 1:
		from core import require_num
		return require_num(values[0]) / 100
	if len(values) == 3:
		string, start, length_val = values
		require_str(string)
		start = py_int(start)
		length_val = py_int(length_val)
		if length_val < 1:
			raise DomainError(f"sub: length must be ≥ 1, got {length_val}")
		if not (1 <= start <= len(string) - length_val + 1):
			raise InvalidDimError("sub: index out of range")
		return TiString(string.tokens[start - 1 : start + length_val - 1])
	raise ArgumentError(f"Invalid arguments: {values}")


@preparse_func
def expr(env: Env, string: TiString):
	"""Evaluate a TiString as a TI-BASIC expression."""
	from parser import Parser
	with env.nest_guard(expr):
		p = Parser(string.tokens, env)
		result = p.parse_expr()
		if p.has_next:
			raise TiSyntaxError(f"expr: evaluated string must contain a single expression; got: {string!r}")
		return result


# ── String commands ───────────────────────────────────────────────────────────

@preparse_cmd_func
def equ_to_string(equ_var: EquationVar, str_var: StringVar) -> None:
	"""Equ►String(equvar, strvar) — copy the equation's tokens into a string variable."""
	str_var.value = TiString(equ_var.resolve().tokens)


@preparse_cmd_func
def string_to_equ(string: TiString, equ_var: EquationVar) -> None:
	"""String►Equ(str_expr, equvar) — parse a string value into an equation variable."""
	equ_var.value = TiEquation(require_str(string).tokens)
