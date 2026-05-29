from dataclasses import dataclass
from numbers import Number

from tiobjects import TiList, TiMatrix, TiString, require_num, require_real
from tokens import (
	Token,
	STORE, L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN, QUOTE, 
	COMMA, DOT, NEG, COLON, NEWLINE, PRGM,
	LIST_PREFIX, RAND, DIM, SCI_E, DEG, RAD, APOS,
)
from environment import Environment, Variable, UserListVar
from errors import TiSyntaxError, ArgumentError, DataTypeError


EOF_TOKEN = Token(b'\x00', None, '<END-OF-INPUT>')


@dataclass
class Thunk:
	tokens: list[Token]
	env: Environment

	def eval(self):
		parser = Parser(self.tokens, self.env)
		value = parser.parse_expr()
		parser.expect(EOF_TOKEN)
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

	def peek(self) -> Token:
		return self.tokens[self.pos] if self.pos < len(self.tokens) else EOF_TOKEN

	def advance(self) -> Token:
		t = self.peek()
		if t is EOF_TOKEN:
			raise TiSyntaxError("Unexpected end of input")
		self.pos += 1
		return t

	def eat_if(self, tok: Token) -> bool:
		if self.peek() is tok:
			self.pos += 1
			return True
		return False

	def expect(self, tok: Token) -> None:
		if self.peek() is not tok:
			raise TiSyntaxError(f"Expected {tok}, got {self.peek()}")
		self.pos += 1

	def at_end(self) -> bool:
		return self.peek() is EOF_TOKEN

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
		"""Opening \" already consumed. Reads until the next \" or end of line."""
		tokens = []
		while not (self.at_end() or self.peek() is STORE):
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
		argument separators. Raises on COLON, NEWLINE, and STORE — statement-
		level tokens that must not appear inside a formula argument."""
		start = self.pos
		stack: list[Token] = []   # expected closers, innermost on top
		in_string = False
		while self.pos < len(self.tokens):
			t = self.tokens[self.pos]
			if in_string:
				if t is QUOTE:
					in_string = False
			elif t in {STORE, COLON, NEWLINE}:
				raise TiSyntaxError(f"Unexpected {t} inside function arguments")
			elif t is QUOTE:
				in_string = True
			elif not stack and t in {COMMA, R_PAREN}:
				break
			elif stack and t is stack[-1]:
				stack.pop()
			elif t.function is not None or t is L_PAREN:
				stack.append(R_PAREN)
			elif t is L_BRACE:
				stack.append(R_BRACE)
			elif t is L_BRACKET:
				stack.append(R_BRACKET)
			self.pos += 1
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

	def _read_name(self, limit) -> str:
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

	def parse_atom(self):
		if self.at_end():
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
				return t.function(ArgParser(self))
			return t.nullary(self.env)

		if t.function is not None:
			return t.function(ArgParser(self))

		if t.is_list_var():
			return self.parse_list_atom(t.variable)

		if t is LIST_PREFIX:
			return self.parse_list_atom(self._parse_user_list_var())
		
		if t.is_matrix_var():
			val = t.variable.get(self.env)
			if self.eat_if(L_PAREN):
				val = val[self.parse_row_col()]
				self.eat_if(R_PAREN)
			return val

		if t.variable is not None:
			return t.variable.get(self.env)
			
		raise TiSyntaxError(f"Unexpected token in expression: {t}")

	def parse_list_atom(self, var):
		val = var.get(self.env)
		if self.eat_if(L_PAREN):
			val = val[self.parse_expr()]
			self.eat_if(R_PAREN)
		return val

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
				lhs = t.operator(lhs, self.parse_expr(right_bp))
				continue

			# Implicit multiplication
			if t.can_start_atom():
				if 60 <= min_bp:
					break
				lhs = lhs * self.parse_expr(61)
				continue

			break

		return lhs

	# ── Store target parser ────────────────────────────────────────────────────

	def parse_store(self, value):
		if self.peek().is_numeric_var() and isinstance(value, TiList):
			self.env.user_lists[self._read_name(5)] = value
			return
				
		t = self.advance()

		if t.is_list_var():
			self.parse_store_list(t.variable, value)

		elif t is LIST_PREFIX:
			self.parse_store_list(self._parse_user_list_var(), value)

		elif t.is_matrix_var():
			if self.eat_if(L_PAREN):
				mat = t.variable.get(self.env)
				if mat is None:
					raise UndefinedError(f"Undefined matrix: {t}")
				mat[self.parse_row_col()] = value
				self.eat_if(R_PAREN)
			else:
				t.variable.set(self.env, value)
		
		elif t.variable is not None:
			t.variable.set(self.env, value)

		elif t is DIM:
			self.parse_store_dim(value)
			
		elif t is RAND:
			self.env.set_random_seed(value)

		else:
			raise TiSyntaxError(f"Invalid store target: {t}")

	def parse_store_list(self, var: Variable, value):
		if self.eat_if(L_PAREN):
			lst = var.get(self.env)
			if lst is None:
				lst = TiList()
				var.set(self.env, lst)
			lst[self.parse_expr()] = value
			self.eat_if(R_PAREN)
		else:
			var.set(self.env, value)

	def parse_store_dim(self, value):
		t = self.peek()
		if t.is_list_var() or t is LIST_PREFIX:
			var = self.parse_list_var()
			lst = var.get(self.env)
			if lst is None:
				var.set(self.env, TiList.alloc(value))
			else:
				lst.set_dim(value)
		elif t.is_matrix_var():
			self.advance()
			mat = t.variable.get(self.env)
			if mat is None:
				t.variable.set(self.env, TiMatrix.alloc(value))
			else:
				mat.set_dim(value)
		else:
			raise TiSyntaxError(f"Invalid store-to-dim target: {t}")

	def parse_list_var(self):
		t = self.advance()
		if t is LIST_PREFIX:
			return self._parse_user_list_var()
		if t.is_list_var():
			return t.variable
		raise TiSyntaxError(f"Expected a list variable, got {t}")
	
	def _parse_user_list_var(self):
		return UserListVar(self._read_name(5))

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		while True:
			if self.at_end():
				return

			if self.eat_if(PRGM):
				name = self._read_name(8)
				val = self.env.programs[name].execute()

			elif self.peek().command is not None:
				self.advance().command(ArgParser(self))

			else:
				value = self.parse_expr()
				if self.eat_if(STORE):
					self.parse_store(value)
				elif self.peek().converter is not None:
					value = self.advance().converter(value)
				self.env.ans = value
				
			if not self.eat_if(COLON):
				break

		self.expect(EOF_TOKEN)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_line(tokens: list[Token], env: Environment):
	"""Parse and evaluate a single program line."""
	Parser(tokens, env).parse_statement()



def _parse_method(method):
	def wrapper(self, optional=False, default=None):
		return self._arg(lambda: method(self), optional, default)
	return wrapper


class ArgParser:
	"""Stateful helper for parsing comma-separated function arguments.

	Uses a trailing-comma model: after each argument is parsed, the following
	COMMA (if present) is consumed immediately.  This means peek() between
	argument calls always shows the first token of the *next* argument rather
	than a COMMA, which lets callers dispatch on argument type before consuming.
	"""

	def __init__(self, parser: Parser):
		self._parser = parser
		self._next = True  # True = next arg token is already exposed (no leading comma to eat)

	def _arg(self, parse_fn, optional=False, default=None):
		if not self._next:
			if optional:
				return default
			raise ArgumentError("Missing argument: expected comma before next argument")
		if optional and (self._parser.at_end() or self._parser.peek() is R_PAREN):
			return default
		val = parse_fn()
		self._next = self._parser.eat_if(COMMA)
		if not self._next:
			self._parser.eat_if(R_PAREN)
		return val

	@_parse_method
	def expr(self):
		return self._parser.parse_expr()

	@_parse_method
	def thunk(self):
		return self._parser.capture()

	@_parse_method
	def numeric_var(self) -> Token:
		t = self._parser.advance()
		if not t.is_numeric_var():
			raise DataTypeError(f"Expected a numeric variable, got {t}")
		return t

	@_parse_method
	def list_var(self) -> Variable:
		return self._parser.parse_list_var()

	@_parse_method
	def matrix_var(self) -> Variable:
		t = self._parser.advance()
		if t.is_matrix_var():
			return t.variable
		raise DataTypeError(f"Expected a matrix variable, got {t}")

	def end(self):
		"""Assert no surplus arguments remain.  The closing ) is already consumed
		by the trailing-comma logic in _arg, so this is purely a validation call."""
		if self._next:
			raise ArgumentError(f"Too many arguments: unexpected {self.peek()}")

	@property
	def env(self):
		return self._parser.env

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
	from tokens import INV

	env = Environment()

	def test(*line):
		tokens = toks(*line)
		print('>>', ''.join(t.text for t in tokens))
		parse_line(tokens, env)
		print('<<', env.ans)

	env.angle_mode = 'DEG'

	
	# test('55@A:99@B')
	# test('int( log( 2) INV log( max( {A,B')
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
