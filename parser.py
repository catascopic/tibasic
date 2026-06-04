from dataclasses import dataclass
from numbers import Number

from tiobjects import TiList, TiMatrix, TiString, require_num, require_real
from titoken import Token, EOF_TOKEN
from catalog import (
	STORE, COMMA, DOT, NEG, COLON, NEWLINE,
	L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN, QUOTE,
	SCI_E, DEG, RAD, APOS,
	LIST_PREFIX, RAND, DIM,
	IF, THEN, ELSE, FOR, WHILE, REPEAT, END,
)
from environment import Environment, Variable, UserList
from errors import TiError, TiSyntaxError, ArgumentError, DataTypeError, InvalidCommandError, UndefinedError


@dataclass
class Thunk:
	tokens: list[Token]
	env: Environment

	def eval(self):
		parser = Parser(self.tokens, self.env)
		value = parser.parse_expr()
		if parser.has_next:
			# This is a ValueError, not a TiError, because the parser should always get this right
			raise ValueError(f"Expected end of Thunk; remaining: {parser.tokens[parser.pos:]}")
		return value


class Parser:

	def __init__(self, tokens: list[Token], env: Environment):
		self.tokens = tokens
		self.pos = 0
		self.env = env
		self._struct_depth = 0

	def __repr__(self):
		return f"tokens={self.tokens}, pos={self.pos}"

	# ── Primitives ─────────────────────────────────────────────────────────────

	@property
	def has_next(self):
		return self.pos < len(self.tokens)

	def peek(self) -> Token:
		return self.tokens[self.pos] if self.has_next else EOF_TOKEN

	def advance(self) -> Token:
		if self.has_next:
			t = self.tokens[self.pos]
			self.pos += 1
			return t
		raise TiSyntaxError("Unexpected end of input")

	def eat_if(self, tok) -> bool:
		"""Consume the next token if it matches.

		tok may be a single Token (identity check) or a set/frozenset of Tokens
		(membership check).  Returns True and advances iff the token matched.
		"""
		t = self.peek()
		matched = t in tok if isinstance(tok, (set, frozenset)) else t is tok
		if matched:
			self.pos += 1
			return True
		return False

	def expect(self, tok: Token) -> None:
		if self.peek() is not tok:
			raise TiSyntaxError(f"Expected {tok}, got {self.peek()}")
		self.pos += 1

	def end_statement(self):
		if self.eat_if({COLON, NEWLINE}):
			return
		if self.has_next:
			raise TiSyntaxError(f"Expected end of statement; got {self.peek()}")

	def close_delimiter(self, expected: Token) -> bool:
		"""Consume expected closer, or implicitly close at statement boundaries.
		Raises ParseError if a stray ) is found."""
		if self.eat_if(expected):
			return True
		if self.peek() is R_PAREN:
			raise TiSyntaxError(f"Mismatched delimiter: expected {expected}, got ')'")
		return False

	# ── Sub-parsers ────────────────────────────────────────────────────────────

	def _parse_digits(self, first: Token) -> Number:
		"""Parse a bare numeric literal with no DMS or ᴇ handling."""
		num = [first.char]
		while self.peek_digit_or_dot():
			num.append(self.advance().char)
		try:
			return float(''.join(num))
		except ValueError:
			raise TiSyntaxError(f"Bad numeric literal: {num!r}")

	def _parse_sci_exp(self) -> Number:
		"""Parse the exponent of an ᴇ expression: an optional ~ followed by a numeric literal."""
		sign = 1
		while self.eat_if(NEG):
			sign *= -1
		if not self.peek_digit_or_dot():
			raise TiSyntaxError("ᴇ requires a numeric literal exponent")
		return sign * self._parse_digits(self.advance())

	def _peek_dms_start(self) -> bool:
		"""Return True if the next token can start a DMS minutes/seconds component."""
		t = self.peek()
		return t.is_digit() or t in {DOT, SCI_E}

	def _parse_dms_component(self) -> Number:
		"""Parse one DMS component (minutes or seconds): digits[ᴇ exp] or prefix ᴇexp."""
		if self.eat_if(SCI_E):
			return 10 ** self._parse_sci_exp()
		value = self._parse_digits(self.advance())
		if self.eat_if(SCI_E):
			value = value * 10 ** self._parse_sci_exp()
		return value

	def _parse_dms_num(self, value: Number) -> Number:
		"""If ° follows, apply DMS (°min'sec") or deg→rad. Otherwise return value unchanged.

		DMS is valid directly after a numeric literal or ᴇ expression; plain deg→rad
		is valid anywhere. Callers that represent an ᴇ context call this after ᴇ;
		parse_expr's bare ° branch handles the non-ᴇ case with to_radians only.
		"""
		if not self.eat_if(DEG):
			return value
		if self._peek_dms_start():
			# DMS form: degrees°minutes'seconds"
			# Components may be digits, digits+ᴇ, or a bare prefix ᴇ (e.g. 1°E1'3")
			minutes = self._parse_dms_component()
			self.expect(APOS)
			seconds = 0
			if self._peek_dms_start():
				seconds = self._parse_dms_component()
				self.expect(QUOTE)
			result = value + minutes / 60 + seconds / 3600
			if self.peek() is SCI_E:
				raise TiSyntaxError("ᴇ cannot follow a DMS literal")
			return result
		return self.env.from_deg(value)

	def parse_num_literal(self, first: Token) -> Number:
		"""Parse a numeric literal: digits, optional ᴇ exponent, optional DMS/° suffix."""
		value = self._parse_digits(first)
		if self.eat_if(SCI_E):
			value = value * 10 ** self._parse_sci_exp()
			if self.peek() is SCI_E:
				raise TiSyntaxError("Cannot chain ᴇ notation (e.g. 1ᴇ1ᴇ1 is invalid)")
		return self._parse_dms_num(value)

	def parse_string_literal(self) -> TiString:
		"""Opening \" already consumed. Reads until closing \", STORE, NEWLINE, or EOF.
		Colons are valid string content; newlines implicitly terminate the string."""
		tokens = []
		while not self.peek() in {STORE, NEWLINE, EOF_TOKEN}:
			if self.eat_if(QUOTE):
				break
			tokens.append(self.advance())
		return TiString(tokens)

	def parse_list_literal(self) -> TiList:
		"""{ already consumed."""
		self._struct_depth += 1
		items = [require_num(self.parse_expr())]
		while self.eat_if(COMMA):
			items.append(require_num(self.parse_expr()))
		self.close_delimiter(R_BRACE)
		self._struct_depth -= 1
		return TiList(items)

	def parse_matrix_literal(self) -> TiMatrix:
		"""Opening [ already consumed; reads one or more [row] blocks."""
		self._struct_depth += 1
		rows = []
		while True:
			self.expect(L_BRACKET)
			row = []
			while True:
				row.append(require_real(self.parse_expr()))
				if not self.eat_if(COMMA):
					break
			rows.append(row)
			if len(row) != len(rows[0]):
				raise TiSyntaxError(f"Unequal matrix rows: {rows}")
			if not self.close_delimiter(R_BRACKET):
				break
			self.eat_if(COMMA)  # comma is completely optional between rows
			if self.peek() is not L_BRACKET:
				break
		self.close_delimiter(R_BRACKET)
		self._struct_depth -= 1
		return TiMatrix(rows)

	def capture(self) -> Thunk:
		"""Return a Thunk for the tokens up to the next top-level COMMA or R_PAREN.
		Tracks open delimiters on a stack so interior commas in nested
		groups (function calls, {…}, [[…]], "…") are not mistaken for
		argument separators.

		NEWLINE implicitly closes all open delimiters (including inside strings).
		COLON is a statement separator outside strings but is valid string content.
		STORE (→) is always an error inside a formula argument."""
		start = self.pos
		stack: list[Token] = []
		in_string = False
		while self.has_next:
			t = self.peek()
			if t is STORE:
				raise TiSyntaxError(f"Unexpected STORE in formula")
			if t is NEWLINE:
				break
			if in_string:
				if t is QUOTE:
					in_string = False

			elif t is COLON or (not stack and t in {COMMA, R_PAREN}):
				break
			elif t is QUOTE:
				in_string = True
			elif stack and t is stack[-1]:
				stack.pop()
			elif t.function is not None or t is L_PAREN:
				stack.append(R_PAREN)
			elif t is L_BRACE:
				stack.append(R_BRACE)
			elif t is L_BRACKET:
				stack.append(R_BRACKET)
			self.advance()
		return Thunk(self.tokens[start:self.pos], self.env)

	def peek_digit_or_dot(self) -> bool:
		t = self.peek()
		return t.is_digit() or t is DOT

	def parse_label_name(self) -> str:
		"""Read up to 2 alphanumeric characters as a label name."""
		t = self.advance()
		if not t.is_name_char():
			raise TiSyntaxError("Expected a label")
		label = t.char
		if self.peek().is_name_char():
			label += self.advance().char

		return label

	def read_name(self, limit) -> str:
		"""Read alphanumeric tokens as an identifier (prgm, user list, etc.)."""
		t = self.advance()
		if not t.is_numeric_var():
			raise TiSyntaxError("Expected a name")
		chars = [t.char]
		while self.peek().is_name_char():
			chars.append(self.advance().char)
		name = ''.join(chars)
		if len(name) > limit:
			raise TiSyntaxError(f"Name to long; limit {limit} chars but got: {name}")
		return name

	# ── Atom parser ────────────────────────────────────────────────────────────

	def _call_function(self, t: Token):
		"""Dispatch a token function via its call_with_parser interface."""
		return t.function.call_with_parser(ArgParser(self))

	def parse_atom(self):
		if self.peek() in {COLON, NEWLINE, EOF_TOKEN}:
			raise TiSyntaxError("Expected an expression")

		t = self.advance()

		if t.is_digit() or t is DOT:
			return self.parse_num_literal(t)

		if t is QUOTE:
			return self.parse_string_literal()

		if t is L_BRACE:
			if self._struct_depth > 0:
				raise TiSyntaxError("List literal not allowed inside a list or matrix")
			return self.parse_list_literal()

		if t is L_BRACKET:
			if self._struct_depth > 0:
				raise TiSyntaxError("Matrix literal not allowed inside a list or matrix")
			return self.parse_matrix_literal()

		if t is L_PAREN:
			val = self.parse_expr()
			self.eat_if(R_PAREN)
			return val

		# special case: this is the only prefix unary operator
		if t is NEG:
			return -self.parse_expr(65)

		# ᴇ with no left operand: treat as 10^rhs  (e.g. ᴇ3 = 1000, ᴇ~3 = 0.001)
		if t is SCI_E:
			return self._parse_dms_num(10 ** self._parse_sci_exp())

		# Nullary constants (π, e, rand, Ans, getDate, etc.)
		# Checked before function so tokens with both can dispatch on whether ( follows.
		if t.nullary is not None:
			if self.peek() is L_PAREN and t.function is not None:
				self.advance()
				return self._call_function(t)
			return t.nullary(self.env)

		if t.function is not None:
			return self._call_function(t)

		if t.is_list_var():
			return self.parse_list_atom(t.variable(self.env))

		if t is LIST_PREFIX:
			return self.parse_list_atom(self.parse_user_list())

		if t.is_matrix_var():
			val = t.variable(self.env).resolve()
			if self.eat_if(L_PAREN):
				val = val[self.parse_row_col()]
				self.eat_if(R_PAREN)
			return val

		if t.variable is not None:
			return t.variable(self.env).resolve()

		raise TiSyntaxError(f"Unexpected token in expression: {t}")

	def parse_list_atom(self, var):
		value = var.resolve()
		if self.eat_if(L_PAREN):
			value = value[self.parse_expr()]
			self.eat_if(R_PAREN)
		return value

	def parse_row_col(self):
		row = self.parse_expr()
		self.expect(COMMA)
		col = self.parse_expr()
		self.eat_if(R_PAREN)
		return row, col

	# ── Pratt expression parser ────────────────────────────────────────────────

	def parse_expr(self, min_bp: int = 0):
		lhs = self.parse_atom()

		while True:

			# Angle-mode conversions (need env access, so handled here rather than as token postfixes)
			# DMS (°minutes'seconds") is handled inside parse_num_literal, so ° here is
			# always the plain degrees→radians conversion, valid on any expression result.
			if self.eat_if(DEG):
				lhs = self.env.from_deg(lhs)
				continue
			if self.eat_if(RAD):
				lhs = self.env.from_rad(lhs)
				continue

			t = self.peek()

			# Postfix operators
			# make a .eat_if_flag method if I make flags for tokens after decoupling from functions
			if t.postfix:
				self.advance()
				lhs = t.postfix(lhs)
				continue

			# Explicit binary operator
			if t.bp is not None:
				left_bp, right_bp = t.bp
				if left_bp <= min_bp:
					break
				self.advance()
				lhs = t.operator(lhs, self.parse_expr(right_bp), self.env)
				continue

			# Implicit multiplication
			if _can_start_atom(t):
				if 60 <= min_bp:
					break
				lhs = lhs * self.parse_expr(61)
				continue

			break

		return lhs

	# ── Store target parser ────────────────────────────────────────────────────

	def parse_store(self, value):
		if self.peek().is_numeric_var() and isinstance(value, TiList):
			self.env.user_lists[self.read_name(5)] = value
			return

		t = self.advance()

		if t.is_list_var():
			self.parse_store_list(t.variable(self.env), value)

		elif t is LIST_PREFIX:
			self.parse_store_list(self.parse_user_list(), value)

		elif t.is_matrix_var():
			var = t.variable(self.env)
			if self.eat_if(L_PAREN):
				var.resolve()[self.parse_row_col()] = value
				self.eat_if(R_PAREN)
			else:
				var.store(value)

		elif t.variable is not None:
			t.variable(self.env).store(value)

		elif t is DIM:
			self.parse_store_dim(value)

		elif t is RAND:
			self.env.set_random_seed(value)

		else:
			raise TiSyntaxError(f"Invalid store target: {t}")

	def parse_store_list(self, var: Variable, value):
		if self.eat_if(L_PAREN):			
			if var.value is None:
				lst = TiList()
				lst[self.parse_expr()] = value
				var.value = lst
			else:
				var.value[self.parse_expr()] = value
			self.eat_if(R_PAREN)
		else:
			var.store(value)

	def parse_store_dim(self, value):
		t = self.peek()
		if t.is_list_var() or t is LIST_PREFIX:
			var = self.parse_list_var()
			if var.value is None:
				var.value = TiList.alloc(value)
			else:
				var.value.set_dim(value)
		elif t.is_matrix_var():
			self.advance()
			var = t.variable(self.env)
			if var.value is None:
				var.value = TiMatrix.alloc(value)
			else:
				var.value.set_dim(value)
		else:
			raise TiSyntaxError(f"Invalid store-to-dim target: {t}")
		self.eat_if(R_PAREN)

	def parse_list_var(self):
		t = self.advance()
		if t is LIST_PREFIX:
			return self.parse_user_list()
		if t.is_list_var():
			return t.variable(self.env)
		raise TiSyntaxError(f"Expected a list variable, got {t}")

	def parse_user_list(self):
		return UserList(self.env, self.read_name(5))

	# SKIPPING

	def skip_statement(self) -> None:
		"""Advance past one statement without executing it, consuming the trailing separator.

		After this call, pos is at the start of the next statement (or EOF).
		Respects string literals — a NEWLINE inside a string closes the string
		and also terminates the statement.  If called at a separator (empty
		statement), that separator is the entire statement and is consumed.
		"""
		while self.has_next:
			t = self.advance()
			if t in {COLON, NEWLINE}:
				return
			if t is QUOTE:
				while self.has_next:
					t = self.advance()
					if t in {QUOTE, STORE}:
						break
					if t is NEWLINE:
						return
		# statement skipped by reaching EOF

	def skip_block(self, else_mode: bool = False) -> Token:
		"""Scan forward to the matching End (or Else if *else_mode* is True).

		Processes the stream statement-by-statement via skip_statement.
		FOR/WHILE/REPEAT always open a new block; THEN opens one only when the
		immediately preceding non-empty statement was IF — a bare Then (without a
		preceding If) is transparent to the depth counter.  At depth 0 the scan
		stops at END (always) or ELSE (when *else_mode*).  Leaves pos just past
		the stopping token.  Raises TiSyntaxError if no match is found.
		"""
		depth = 0
		prev_if = False
		start_pos = self.pos
		while self.has_next:
			t = self.advance()
			if t is THEN:
				if prev_if:
					depth += 1
			elif t in {FOR, WHILE, REPEAT}:
				depth += 1
			elif t is END or (else_mode and t is ELSE):
				if depth == 0:
					self.end_statement()
					return t
				depth -= 1
			self.skip_statement()
			prev_if = t is IF

		raise TiSyntaxError("Unmatched block: End not found", pos=self.pos)

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def _exec_statement(self):
		"""Execute one statement and its trailing separator.

		An empty statement (bare COLON or NEWLINE) is consumed and returns
		immediately — this also handles post-jump residue from Goto/loops.
		"""
		if self.eat_if({COLON, NEWLINE}):
			return

		try:
			if self.peek().command is not None:
				self.advance().command.call_with_parser(ArgParser(self))
			else:
				value = self.parse_expr()
				if self.eat_if(STORE):
					self.parse_store(value)
				elif self.peek().converter is not None:
					value = self.advance().converter(value)
				self.env.ans = value
				self.end_statement()
		except TiError as e:
			if e.pos is None:
				e.pos = self.pos - 1
			raise

	def run(self):
		"""Execute all statements in the token stream until EOF."""
		try:
			while self.has_next:
				self._exec_statement()
		except TiError as e:
			# Only the parser that owns the token stream for e.pos displays the
			# location; once shown, mark the error so enclosing parsers (e.g. the
			# caller of a `prgm` invocation, whose tokens are unrelated) neither
			# re-display nor index out of range.
			if e.pos is not None and not e.located and 0 <= e.pos < len(self.tokens):
				loc = [t.text for t in self.tokens]
				loc[e.pos] = f"[![{loc[e.pos]}]!]"
				print(''.join(loc))
				e.located = True
			raise


def _can_start_atom(t: Token) -> bool:
	return (
		t.is_digit() or t.variable or t.nullary or t.function
		or t in {L_PAREN, L_BRACE, L_BRACKET, QUOTE, DOT, SCI_E, NEG, LIST_PREFIX}
	)


def _parse_arg(method):
	def wrapper(self, optional=False, default=None):
		return self._arg(lambda: method(self), optional, default)
	return wrapper


class ArgParser:
	"""Stateful helper for parsing comma-separated function arguments.

	Uses a trailing-comma model: after each argument is parsed, the following
	COMMA (if present) is consumed immediately.  This means peek() between
	argument calls always shows the first token of the *next* argument rather
	than a COMMA, which lets callers dispatch on argument type before consuming.

	The caller is responsible for finalization via one of three end methods:
	  - end_func()      — eat ) + check no surplus  (expression functions)
	  - end_paren_cmd() — eat ) + check no surplus + eat separator  (paren commands)
	  - end_cmd()       — check at statement end + eat separator  (no-paren commands)
	"""

	def __init__(self, parser: Parser):
		self._parser = parser
		self._next = parser.peek() not in {COLON, NEWLINE, EOF_TOKEN}

	def _arg(self, parse_fn, optional=False, default=None):
		if not self._next:
			if optional:
				return default
			raise ArgumentError("Missing argument: expected comma before next argument")
		val = parse_fn()
		self._next = self._parser.eat_if(COMMA)
		return val

	@_parse_arg
	def expr(self):
		return self._parser.parse_expr()

	@_parse_arg
	def thunk(self):
		return self._parser.capture()

	@_parse_arg
	def numeric_var(self) -> Variable:
		t = self._parser.advance()
		if not t.is_numeric_var():
			raise DataTypeError(f"Expected a numeric variable, got {t}")
		return t.variable(self.env)

	@_parse_arg
	def list_var(self) -> Variable:
		return self._parser.parse_list_var()

	@_parse_arg
	def matrix_var(self) -> Variable:
		t = self._parser.advance()
		if t.is_matrix_var():
			return t.variable(self.env)
		raise DataTypeError(f"Expected a matrix variable, got {t}")

	@_parse_arg
	def string_var(self) -> Variable:
		t = self._parser.advance()
		if t.is_string_var():
			return t.variable(self.env)
		raise DataTypeError(f"Expected a string variable, got {t}")

	@_parse_arg
	def equation_var(self) -> Variable:
		t = self._parser.advance()
		if t.is_equation_var():
			return t.variable(self.env)
		raise DataTypeError(f"Expected an equation variable, got {t}")

	@_parse_arg
	def list_var_prefix_optional(self) -> Variable:
		"""Read a list variable: L1–L6, ᴸNAME, or a bare user-list name without the ᴸ prefix.

		SetUpEditor accepts all three forms; ordinary list contexts require the prefix.
		"""
		if self.peek().is_numeric_var():
			return UserList(self.env, self._parser.read_name(5))
		return self._parser.parse_list_var()

	def any_var(self) -> Variable:
		"""Read any variable reference: numeric, list, matrix, string, equation, or user list."""
		t = self._parser.advance()
		if t.variable is not None:
			return t.variable(self.env)
		if t is LIST_PREFIX:
			return self._parser.parse_user_list()
		raise TiSyntaxError(f"Expected a variable, got {t}")

	@_parse_arg
	def label_name(self) -> str:
		"""Read up to 2 alphanumeric characters as a label name (for Lbl / Goto)."""
		return self._parser.parse_label_name()

	@_parse_arg
	def program_name(self) -> str:
		"""Read up to 8 alphanumeric characters as a program name (for prgm)."""
		return self._parser.read_name(8)

	def current_program(self):
		"""Return the innermost currently-executing Program.

		Raises InvalidCommandError if called from outside a program (e.g. from
		the home screen).  The stack trace identifies the calling command.
		"""
		prog = self.env.current_program
		if prog is None:
			raise InvalidCommandError("This command cannot be used outside a program")
		return prog

	def no_args(self) -> None:
		"""Raise TiSyntaxError if any tokens follow on this statement.

		For no-arg commands (Normal, Return, End, etc.) that should reject
		trailing tokens outright.  Raises TiSyntaxError rather than ArgumentError
		because stray tokens here are a syntactic mistake, not a wrong argument count.
		"""
		if self._next:
			raise TiSyntaxError(f"Command takes no arguments, but got: {self.peek()}")

	def end_func(self):
		"""Consume the closing ) and validate no surplus arguments remain.

		For expression functions called from within parse_expr(): eats the
		optional closing ) (TI-BASIC allows implicit close), then checks that
		no surplus arguments follow.  Does not eat the statement separator.
		"""
		if self._next:
			raise ArgumentError(f"Too many arguments: unexpected {self.peek()}")
		self._parser.eat_if(R_PAREN)

	def end_paren_cmd(self):
		"""Consume ), validate no surplus args, and eat the trailing statement separator.

		For paren commands (e.g. For(, IS>(, Matr►list(): eats the optional
		closing ), then eats the COLON/NEWLINE that follows, completing the statement.
		"""
		self.end_func()
		self._parser.end_statement()

	def end_cmd(self):
		"""Validate we are at a statement boundary and eat the trailing separator.

		For no-paren commands (e.g. If, While, Goto, End): checks that nothing
		unexpected follows the command's arguments, then consumes the COLON/NEWLINE
		(or does nothing at EOF).
		"""
		if self._next:
			raise ArgumentError(f"Too many arguments: unexpected {self.peek()}")
		self._parser.end_statement()

	@property
	def env(self):
		return self._parser.env

	@property
	def has_next(self) -> bool:
		return self._next

	def parse_args(self) -> list:
		args = [self.expr()]
		while self._next:
			args.append(self.expr())
		return args

	def peek(self):
		return self._parser.peek()



if __name__ == '__main__':
	from tibasic_test import toks, calc

	env = Environment()

	def test(*line):
		tokens = toks(*line)
		print('>>', ''.join(t.text for t in tokens))
		env.run(tokens)
		print('<<', env.ans)

	env.angle_mode = 'DEG'

	test('55@A:99@B')
	test('int( log( 2) INV log( max( A,B')
	# test('2^ cumSum( binomcdf( Ans ,0')
	# test('sum( Ans .5(1= abs( int( 2 fPart( Ans INV (A+Bi')

	# test('55@A:99@B')
	# test('seq( 2^N,N,8,1,~1@ L1')
	# test('.5 sum( L1 *(1= abs( int( 2 fPart( (A+Bi)/ L1')

	# test("1E2°1E2'")
	# test("1E2+(1E2/60)")
	# test('1@A')
	# test('1E~1°2\'3"')
	# test("1°~30'")
	# test('List►matr( {1,2},{3,4}, [A]')
	# test('⑽^( {1,10')
	# test('5°')
	# test('5°5\'5"')
	# test('2^3',SCI_E,2)
	# test('"a',STORE,'Str1')
	# test('1°ʳ')
	# test('([[1,2][3,4@ [A]')
	# test('[A]','+[[5,6],[7,8]]')
	# test('[A]','^4')
	# test('[A]')
	# test('length( "  or ')
	# test('{1,2,3}',SCI_E,'{1,2,3}')
	# test('[[2','dim(','{1,2,3]')
	# test('rand (5')
	# test('[[1:[[','Ans','(1,1')
	# test('{5,5',STORE,'dim(',(0x5C,0))
	# test('{5',STORE,LIST_PREFIX,'AB')
	# test(0,STORE,'dim(',(0x5D,0))
	# test(1,STORE,'A:3',STORE,'B')
	# test('[[1,2.5],[π,4]]')
	# test('randM(','3,4')
	# test('Ans',STORE,(0x5C,0))
	# test(3,STORE,(0x5C,0),'(2,1')
	# print(env.numerics)
	env.dump()
