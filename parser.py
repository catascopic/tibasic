from dataclasses import dataclass

from tiobjects import TiList, TiMatrix, TiString, require_num, require_real
from tokens import (
	Token, EOF_TOKEN,
	STORE, L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN,
	QUOTE, COMMA, DOT, COLON, NEWLINE, PRGM, ANS, NEG, LIST_PREFIX,
	RAND, DIM, SCI_E, DEG, RAD, APOS
)
from environment import Environment, Variable, UserListVar
from forms import ArgParser

class ParseError(Exception):
	pass


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
			raise ParseError("Unexpected end of input")
		self.pos += 1
		return t

	# TODO: remane to try_eat?
	def eat_if(self, tok: Token) -> bool:
		if self.peek() is tok:
			self.pos += 1
			return True
		return False

	def expect(self, tok: Token) -> None:
		if self.peek() is not tok:
			raise ParseError(f"Expected {tok}, got {self.peek()}")
		self.pos += 1

	def at_end(self) -> bool:
		return self.peek() is EOF_TOKEN

	# Wrappers used by ArgParser so it doesn't need to import token objects
	def expect_comma(self):
		self.expect(COMMA)
	def eat_if_comma(self) -> bool:
		return self.eat_if(COMMA)
	def eat_if_rparen(self):
		self.eat_if(R_PAREN)
	def peek_is_comma(self) -> bool:
		return self.peek() is COMMA
	def peek_is_rparen(self) -> bool:
		return self.peek() is R_PAREN

	def close_delimiter(self, expected: Token) -> bool:
		"""Consume expected closer, or implicitly close at statement boundaries.
		Raises ParseError if a stray ) is found."""
		if self.eat_if(expected):
			return True
		if self.peek() is R_PAREN:
			raise ParseError(f"Mismatched delimiter: expected {expected.text!r}, got ')'")
		return False

	# ── Sub-parsers ────────────────────────────────────────────────────────────

	def _parse_digits(self, first: Token) -> float:
		"""Parse a bare numeric literal with no DMS handling."""
		num = [first.text]
		while True:
			t = self.peek()
			if not (t.is_digit() or t is DOT):
				break
			num.append(self.advance().text)
		try:
			return float(''.join(num))
		except ValueError:
			raise ParseError(f"Bad numeric literal: {num!r}")

	def parse_num_literal(self, first: Token) -> float:
		value = self._parse_digits(first)
		if self.eat_if(DEG):
			if self.peek_digit_or_dot():
				minutes = self._parse_digits(self.advance())
				self.expect(APOS)
				seconds = 0
				if self.peek_digit_or_dot():
					seconds = self._parse_digits(self.advance())
					self.expect(QUOTE)
				value = value + minutes / 60 + seconds / 3600
			else:
				value = self.env.to_radians(value)
		return value

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
		items = []
		if not self.eat_if(R_BRACE):
			items.append(self.parse_expr())
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
				raise ValueError(f"Unequal matrix rows: {rows}")
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
				raise ParseError(f"Unexpected {t.text!r} inside function arguments")
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

	def parse_sci_e_exp(self) -> float:
		"""Parse the exponent of a ᴇ expression: an optional − followed by a numeric literal."""
		neg = self.eat_if(NEG)
		if not self.peek_digit_or_dot():
			raise ParseError("ᴇ requires a numeric literal exponent")
		exp = self.parse_num_literal(self.advance())
		return -exp if neg else exp

	def parse_label_name(self) -> str:
		"""Read up to 2 alphanumeric characters as a label name."""
		t = self.advance()
		if not t.is_name_char():
			raise ParseError("Expected a label")
		label = t.text
		if self.peek().is_name_char():
			label += self.advance().text

		return label

	def _read_name(self) -> str:
		"""Read alphanumeric tokens as an identifier (prgm, user list, etc.)."""
		t = self.advance()
		if not t.is_numeric_var():
			raise ParseError("Expected a name")
		name = [t.text]
		while self.peek().is_name_char():
			name.append(self.advance().text)
		return ''.join(name)

	# ── Atom parser ────────────────────────────────────────────────────────────

	def parse_atom(self):
		if self.at_end():
			raise ParseError("Expected an expression")

		t = self.advance()

		if t.is_digit() or t is DOT:
			return self.parse_num_literal(t)
		if t is QUOTE:
			return self.parse_string_literal()
		if t is L_BRACE:
			if self._struct_depth > 0:
				raise ParseError("List literal not allowed inside a list or matrix")
			return self.parse_list_literal()
		if t is L_BRACKET:
			if self._struct_depth > 0:
				raise ParseError("Matrix literal not allowed inside a list or matrix")
			return self.parse_matrix_literal()
		if t is L_PAREN:
			val = self.parse_expr()
			self.close_delimiter(R_PAREN)
			return val

		# special case: this is the only prefix unary operator
		if t is NEG:
			return -self.parse_expr(65)

		# ᴇ with no left operand: treat as 10^rhs  (e.g. ᴇ3 = 1000, ᴇ−3 = 0.001)
		if t is SCI_E:
			return 10 ** self.parse_sci_e_exp()

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
			return self.parse_list_atom(UserListVar(self._read_name()))
		
		if t.is_matrix_var():
			val = t.variable.get(self.env)
			if self.eat_if(L_PAREN):
				val = val[self.parse_row_col()]
				self.eat_if(R_PAREN)
			return val

		if t.variable is not None:
			return t.variable.get(self.env)
			
		raise ParseError(f"Unexpected token in expression: {t}")

	def parse_list_atom(self, var):
		val = var.get(self.env)
		if self.eat_if(L_PAREN):
			val = val[self.parse_expr()]
			self.close_delimiter(R_PAREN)
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
			if self.eat_if(DEG):
				lhs = self.env.to_radians(lhs)
				continue
			if self.eat_if(RAD):
				lhs = self.env.to_degrees(lhs)
				continue

			# ᴇ (scientific notation): RHS must be a numeric literal, not a general expression.
			# No min_bp guard needed — the RHS is always a bare literal, never a sub-expression,
			# so ᴇ binds maximally tight (tighter than ^) with no Pratt conflict to worry about.
			if self.eat_if(SCI_E):
				lhs = lhs * 10 ** self.parse_sci_e_exp()
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
			self.env.user_lists[self._read_name()] = value
			return
				
		t = self.advance()

		if t.is_list_var():
			self.parse_store_list(t.variable, value)

		elif t is LIST_PREFIX:
			self.parse_store_list(UserListVar(self._read_name()), value)

		elif t.is_matrix_var():
			if self.eat_if(L_PAREN):
				mat = t.variable.get(self.env)
				if mat is None:
					raise ValueError(f"Undefined matrix: {t.text}")
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
			raise ParseError(f"Invalid store target: {t}")

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
				lst = TiList([])
				var.set(self.env, lst)
			lst.set_dim(value)
		elif t.is_matrix_var():
			self.advance()
			mat = t.variable.get(self.env)
			if mat is None:
				mat = TiMatrix([])
				t.variable.set(self.env, mat)
			mat.set_dim(value)
		else:
			raise ParseError(f"Invalid store-to-dim target: {t}")

	# ── Variable key parsers ──────────────────────────────────────────────────────

	def parse_list_var(self) -> Variable:
		t = self.advance()
		if t.is_list_var():
			return t.variable
		if t is LIST_PREFIX:
			return UserListVar(self._read_name())
		raise ParseError(f"Expected a list variable, got {t.text!r}")

	def parse_matrix_var(self) -> Variable:
		t = self.advance()
		if t.is_matrix_var():
			return t.variable
		raise ParseError(f"Expected a matrix variable, got {t.text!r}")

	def parse_any_var(self) -> Variable:
		t = self.advance()
		if t is LIST_PREFIX:
			return UserListVar(self._read_name())
		if t.variable:
			return t.variable
		raise ParseError(f"Expected a variable, got {t.text!r}")

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		while True:
			if self.at_end():
				return

			if self.eat_if(PRGM):
				name = self._read_name()
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



if __name__ == '__main__':
	from tokens import *

	env = Environment()
	digits = TOKENS[0x2E:0x37]
	str_to_token = {t.text: t for t in reversed(TOKENS)}

	def test(*line):
		tokens = []
		for obj in line:
			if isinstance(obj, Token):
				tokens.append(obj)
			elif isinstance(obj, int):
				for c in str(obj):
					tokens.append(str_to_token[c])
			elif isinstance(obj, str):
				try:
					tokens.append(str_to_token[obj])
				except KeyError:
					for c in obj:
						tokens.append(str_to_token[c])
			else:
				tokens.append(TOKEN_TABLE[obj])

		print('>>', ''.join(t.text for t in tokens))
		parse_line(tokens, env)
		print('<<', env.ans)

	env.angle_mode = 'DEG'
	
	test('{1,2,3}(2)')
	# test('⑽^(', '{1,10')
	# test('5°')
	# test('5°5\'5"')
	# test('2^3',SCI_E,2)
	# test('"a',STORE,'Str1')
	# test('1°ʳ')
	# test('([[1,2][3,4',STORE,'[A]')
	# test('[A]','+[[5,6],[7,8]]')
	# test('[A]','^4')
	# test('[A]')
	# test('length(', '"', ' or ')
	# test('{1,2,3}',SCI_E,'{1,2,3}')
	# test('[[2','dim(','{1,2,3]')
	# test('rand', '(5')
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
