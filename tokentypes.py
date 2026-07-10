"""Concrete accessor *tokens* — the variable tokens that read/write the environment.

An `Accessor` is a `Token` (see tokenbase) that knows *how* to get/set one symbol — a
numeric variable, a list, π, rand, a window setting.  The token IS the accessor:
`resolve`/`store`/`invoke` take the env as a parameter (tokens are env-less singletons
built once in the catalog), so a variable token serves directly as an expression atom
or a storage target.

This module sits below `catalog` and `parser`: it imports `core`/`graph`/`modes`/
`tokenbase` but never the parser.

`Accessor`, `Deletable`, and `Reference` themselves live in `tokenbase` (the leaf); this
module only provides the concrete subclasses.
"""
from contextlib import contextmanager

from core import TiList, TiMatrix, TiString, TiEquation
from core import require_num, require_matrix, require_list, require_string, require_real, require_int, py_int
from errors import TiSyntaxError, UndefinedError, InvalidDimError, DomainError, DataTypeError
from graph import eval_sequence
from modes import GraphMode
from parser import ArgParser
from tokenbase import Token, Deletable, VariableToken, TokenFileVar, Reference, Flag
import tokenbase as tk

__all__ = [
    # Behavior-carrying
    'FunctionToken', 'CommandToken', 'OperatorToken', 'PostfixToken', 'AnsToken',
    # Numeric / constant
    'DigitToken', 'LetterToken', 'RealToken', 'PaymentsPerYearToken', 'ConstantToken', 'ComputedToken',
    # Aggregate variables
    'MatrixToken', 'ListToken', 'StringToken', 'PicToken',
    # Equation / sequence
    'EquationToken', 'FuncEquationToken', 'ParEquationToken', 'PolarEquationToken',
    'SequenceToken', 'SequenceInitialToken',
    # Window / table
    'WindowToken', 'XresToken', 'IntWindowToken', 'FactorWindowToken', 'DeltaWindowToken',
    'TableToken',
]


class DigitToken(Token):
	def __init__(self, value: int):
		ch = 0x30 + value
		super().__init__(ch, bytes([ch]), Flag.DIGIT, char=chr(ch))
		self.value = value


def _window_affects_graph(env, attr: str) -> bool:
	"""True if `attr` is a window variable whose value affects the current graph mode."""
	return attr in env.graph_mode_handler.window_attrs


def _normalize_eq(value) -> TiEquation:
	"""Coerce a TiString or TiEquation to TiEquation; raise DataTypeError otherwise."""
	if isinstance(value, TiString):
		return TiEquation(value.tokens)
	if isinstance(value, TiEquation):
		return value
	raise DataTypeError(f"Expected equation or string, got {value!r}")


@contextmanager
def scoped_numeric(env, name: str, value):
	"""Temporarily set numeric variable `name` to `value`, restoring it on exit."""
	saved = getattr(env.numerics, name)
	setattr(env.numerics, name, value)
	try:
		yield
	finally:
		setattr(env.numerics, name, saved)



class FunctionToken(Token):
	def __init__(self, code, display, func):
		super().__init__(code, display, Flag.FUNCTION)
		self._func = func

	def parse_prefix(self, parser):
		return self._func.call_with_parser(ArgParser(parser))

	def __repr__(self):
		return f"{super().__repr__()} <{self._func.__module__}.{self._func.__qualname__}>"


class CommandToken(Token):
	def __init__(self, code, display, cmd):
		super().__init__(code, display, Flag.COMMAND)
		self._cmd = cmd

	def run_statement(self, parser):
		self._cmd.call_with_parser(ArgParser(parser))

	def __repr__(self):
		return f"{super().__repr__()} <{self._cmd.__module__}.{self._cmd.__qualname__}>"


class OperatorToken(Token):
	def __init__(self, code, display, op, bp, *, char=None):
		super().__init__(code, display, Flag.INFIX, char=char)
		self._op = op
		self._bp = bp

	def infix_bp(self):
		return self._bp[0]

	def parse_infix(self, parser, lhs):
		rhs = parser.parse_expr(self._bp[1])
		return parser.env.guard_real((lhs, rhs), self._op(lhs, rhs))

	def __repr__(self):
		return f"{super().__repr__()} <{self._op.__module__}.{self._op.__qualname__}>"


class PostfixToken(Token):
	def __init__(self, code, display, op, *, char=None):
		super().__init__(code, display, Flag.POSTFIX, char=char)
		self._op = op

	def apply_postfix(self, value):
		return self._op(value)

	def __repr__(self):
		return f"{super().__repr__()} <{self._op.__module__}.{self._op.__qualname__}>"


class ConstantToken(VariableToken):
	"""A constant π / 𝑒 / 𝑖 — read-only; storing is an error (inherited)."""

	def __init__(self, code: int, display: bytes, value, char: str | None = None):
		super().__init__(code, display, char=char)
		self.value = value

	def resolve(self, env):
		return self.value


class ComputedToken(VariableToken):
	"""A read-only 0-arg computed query (getKey, getDate, …): `resolve(env)` calls func."""

	def __init__(self, code: int, display: bytes, func):
		super().__init__(code, display)
		self.func = func

	def resolve(self, env):
		return self.func(env)


class LetterToken(Deletable, TokenFileVar, VariableToken):
	"""A real/complex variable A–Z, θ — an index into env.numerics (0–25, θ = 26).

	An undefined numeric reads as 0 (and is initialized to 0 on first resolve), matching
	the calculator; `store` accepts any number (real or complex).
	"""

	def __init__(self, index: int):
		code = 0x41 + index           # A–Z = 0x41–0x5A, θ = 0x5B
		super().__init__(code, bytes([code]), Flag.NUMERIC, char=(chr(code) if index < 26 else 'θ'))
		self.index = index

	def get(self, env):
		return env.numerics[self.index]

	def set(self, env, value):
		env.numerics[self.index] = value

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			value = 0.0
			self.set(env, value)
		return value

	def store(self, env, value):
		self.set(env, require_num(value))


class MatrixToken(Deletable, TokenFileVar, VariableToken):
	"""A matrix variable [A]–[J] — an index into env.matrices (0–9)."""

	def __init__(self, index: int):
		super().__init__(0x5C00 | index, bytes([0xC1, 0x41 + index, 0x5D]), Flag.MATRIX | Flag.INVOKABLE)
		self.index = index

	def get(self, env):
		return env.matrices[self.index]

	def set(self, env, value):
		env.matrices[self.index] = value

	def store(self, env, value):
		self.set(env, require_matrix(value).copy())

	def invoke(self, arg_parser):
		row = py_int(arg_parser.expr(), InvalidDimError)
		col = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[(row, col)]


class PicToken(Deletable, TokenFileVar, Token):
	"""A picture Pic1–Pic9, Pic0 (token 0x6000+i) — a slot in env.pics.

	An Accessor (via TokenFileVar) with only the storage half implemented
	(get/set/delete/name_tokens/name_bytes), and not a VariableToken: pictures
	aren't expression atoms or Store-arrow targets — only usable with StorePic/
	RecallPic (and DelVar to clear the slot).  Accessor's raising resolve/store/
	invoke defaults and Token's flag-based can_start_atom/parse_prefix keep it out
	of expressions, matching the calculator.  `index` is the env.pics slot index.
	"""

	def __init__(self, index: int):
		Token.__init__(self, 0x6000 | index, b'Pic' + bytes([0x30 + (index + 1) % 10]), Flag.PIC)
		self.index = index

	def get(self, env):
		return env.pics[self.index]

	def set(self, env, value):
		env.pics[self.index] = value

	def name_tokens(self) -> list:
		return [self]   # like a VariableToken, the token spells itself


class ListToken(Deletable, TokenFileVar, VariableToken):
	"""A built-in list L1–L6 — a 0-based slot in env.lists (TiList | None)."""

	def __init__(self, index: int):
		super().__init__(0x5D00 | index, bytes([0x4C, 0x81 + index]), Flag.LIST | Flag.INVOKABLE)
		self.index = index

	def get(self, env):
		return env.lists[self.index]

	def set(self, env, value):
		env.lists[self.index] = value

	def resolve(self, env):
		value = super().resolve(env)      # errors if undefined
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		self.set(env, require_list(value).copy())

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]


class StringToken(Deletable, TokenFileVar, VariableToken):
	"""A string variable Str1–Str0 — a 0-based slot in env.strings (TiString | None).

	Input/Prompt recognize a string target through its token's `is_string_var()` and
	store the raw typed text rather than evaluating it as an expression.
	"""

	def __init__(self, index: int):
		super().__init__(0xAA00 | index, b'Str' + bytes([0x30 + (index + 1) % 10]), Flag.STRING)
		self.index = index

	def get(self, env):
		return env.strings[self.index]

	def set(self, env, value):
		env.strings[self.index] = value

	def store(self, env, value):
		self.set(env, require_string(value))


class EquationToken(Deletable, TokenFileVar, VariableToken):
	"""A graph equation (Y1–Y0, X1T/Y1T–X6T/Y6T, r1–r6).

	An equation is not one of TI's runtime value types, so reading it as a value
	(`resolve`) evaluates the formula at the current graph variable.  `Y1(x)` is function
	composition — resolve with X temporarily set to x.  `store` normalises a
	string/equation and selects the function.  `mode` is a GraphMode whose value doubles
	as the Environment attribute holding the equation collection.
	"""

	mode: GraphMode = None
	graph_var = None

	def __init__(self, code: int, display: bytes, index: int):
		super().__init__(code, display, Flag.EQUATION | Flag.INVOKABLE)
		self.index = index

	def get(self, env):
		return getattr(env, self.mode)[self.index].equation

	def set(self, env, value):
		getattr(env, self.mode)[self.index].equation = value

	def resolve(self, env):
		return super().resolve(env).eval(env)

	def invoke(self, arg_parser):
		# The '(' is already eaten: Y1(x) composes by resolving with the graph var set to x.
		x = arg_parser.expr()
		arg_parser.end_func()
		env = arg_parser.env
		with scoped_numeric(env, self.graph_var, require_num(x)):
			return self.resolve(env)

	def store(self, env, value):
		self.set(env, _normalize_eq(value))
		getattr(env, self.mode)[self.index].selected = True
		if env.graph_mode is self.mode:
			env.graph.valid = False


class FuncEquationToken(EquationToken):
	mode = GraphMode.FUNC
	graph_var = 'X'

	def __init__(self, index: int):
		super().__init__(0x5E10 + index, bytes([0x59, 0x80 + (index + 1) % 10]), index)


class ParEquationToken(EquationToken):
	"""One half of a parametric pair (XnT or YnT); both halves share the pair index."""

	mode = GraphMode.PAR
	graph_var = 'T'

	def __init__(self, index: int, half: str):
		is_y = half == 'y'
		code = 0x5E20 + index * 2 + (1 if is_y else 0)
		display = bytes([0x59 if is_y else 0x58, 0x81 + index, 0x0D])
		super().__init__(code, display, index)
		self.half = half

	def get(self, env):
		return getattr(env.parametric[self.index], self.half)

	def set(self, env, value):
		setattr(env.parametric[self.index], self.half, value)


class PolarEquationToken(EquationToken):
	mode = GraphMode.POL
	graph_var = 'theta'

	def __init__(self, index: int):
		super().__init__(0x5E40 + index, bytes([0x72, 0x81 + index]), index)


class SequenceToken(EquationToken):
	"""A recursive sequence variable 𝑢/𝑣/𝑤.  `resolve` evaluates at the current n;
	`u(n)` evaluates at an explicit index."""

	mode = GraphMode.SEQ

	def __init__(self, index: int):
		super().__init__(0x5E80 + index, bytes([0x02 + index]), index)

	def resolve(self, env):
		return eval_sequence(env, self.index, env.n)

	def invoke(self, arg_parser):
		n = arg_parser.expr()
		arg_parser.end_func()
		return eval_sequence(arg_parser.env, self.index, n)


class SequenceInitialToken(Deletable, VariableToken):
	"""The u(nMin)/v(nMin)/w(nMin) token — stores/reads the initial-value list."""

	def __init__(self, code: int, display: bytes, index: int):
		super().__init__(code, display)
		self.index = index

	def get(self, env):
		return env.sequence[self.index].initial

	def set(self, env, value):
		env.sequence[self.index].initial = value

	def store(self, env, value):
		if isinstance(value, TiList):
			if len(value.data) > 2:
				raise InvalidDimError("u/v/w(nMin) list may have at most 2 elements")
		else:
			value = TiList([require_real(value)])
		if value != self.get(env) and env.graph_mode is GraphMode.SEQ:
			env.graph.valid = False
		self.set(env, value)


class RealToken(VariableToken):
	"""A plain real variable stored directly on env (TVM variables, stat n, …).  Takes an
	explicit `flags` so stat 𝑛 can be classified NUMERIC while TVM vars stay unflagged."""

	def __init__(self, code: int, display: bytes, attr: str, flags: Flag = Flag(0)):
		super().__init__(code, display, flags)
		self.attr = attr

	def get(self, env):
		return getattr(env, self.attr)

	def set(self, env, value):
		setattr(env, self.attr, value)

	def store(self, env, value):
		self.set(env, require_real(value))


class AnsToken(Token):
	"""Ans — reads env.ans, with optional index into the result when it's a list/matrix."""

	def parse_prefix(self, parser):
		ans = parser.env.ans
		if parser.peek().code == tk.L_PAREN:
			if isinstance(ans, TiList):
				parser.advance()
				return ans[parser.parse_list_index()]
			if isinstance(ans, TiMatrix):
				parser.advance()
				return ans[parser.parse_matrix_indices()]
		return ans


class PaymentsPerYearToken(RealToken):
	"""P/Y — storing also copies the value to C/Y (one-way coupling)."""

	def store(self, env, value):
		super().store(env, value)
		env.cy = value



class WindowToken(VariableToken):
	"""A plain real-valued window variable (Xmin, ZXmin, Tmax, …) — a named float attr on
	env.window.  Storing a value that affects the current graph invalidates it."""

	def __init__(self, code: int, display: bytes, attr: str):
		super().__init__(code, display, Flag.WINDOW_VAR)
		self.attr = attr

	def get(self, env):
		return getattr(env.window, self.attr)

	def set(self, env, value):
		setattr(env.window, self.attr, value)

	def _store_check_valid(self, env, value):
		if _window_affects_graph(env, self.attr) and value != self.get(env):
			env.graph.valid = False
		self.set(env, value)

	def store(self, env, value):
		self._store_check_valid(env, require_real(value))


class XresToken(WindowToken):
	"""Xres — function-graph resolution; a whole number 1–8."""

	def store(self, env, value):
		require_int(value)
		if not (1 <= value <= 8):
			raise DomainError(f"Xres must be an integer 1-8, got {value:g}")
		self._store_check_valid(env, value)


class IntWindowToken(WindowToken):
	"""nMin, nMax — window variables constrained to whole numbers."""

	def store(self, env, value):
		self._store_check_valid(env, require_int(value))


class FactorWindowToken(WindowToken):
	"""XFact, YFact — zoom-in/out scaling factors; must be ≥ 1."""

	def store(self, env, value):
		v = require_real(value)
		if v < 1:
			raise DomainError(f"Zoom factor must be ≥ 1, got {v:g}")
		self.set(env, v)


class DeltaWindowToken(VariableToken):
	"""ΔX / ΔY — computed from the window bounds; not stored directly.

	resolve: (hi − lo) / divisions.  store(δ): adjusts the hi bound, leaving lo fixed.
	"""

	def __init__(self, code: int, display: bytes, lo_attr: str, hi_attr: str, divisions: int):
		super().__init__(code, display, Flag.WINDOW_VAR)
		self.lo_attr = lo_attr
		self.hi_attr = hi_attr
		self.divisions = divisions

	def resolve(self, env):
		lo = getattr(env.window, self.lo_attr)
		hi = getattr(env.window, self.hi_attr)
		return (hi - lo) / self.divisions

	def store(self, env, value):
		delta = require_real(value)
		lo = getattr(env.window, self.lo_attr)
		new_hi = lo + self.divisions * delta
		if _window_affects_graph(env, self.hi_attr) and new_hi != getattr(env.window, self.hi_attr):
			env.graph.valid = False
		setattr(env.window, self.hi_attr, new_hi)


class TableToken(VariableToken):
	"""A plain real variable stored on env.table (TblStart, ΔTbl)."""

	def __init__(self, code: int, display: bytes, attr: str):
		super().__init__(code, display)
		self.attr = attr

	def get(self, env):
		return getattr(env.table, self.attr)

	def set(self, env, value):
		setattr(env.table, self.attr, value)

	def store(self, env, value):
		self.set(env, require_real(value))
