from dataclasses import dataclass

from tiobjects import TiList, TiMatrix
from tokens import (
	Token, EOF_TOKEN,
	STORE, L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN,
	QUOTE, COMMA, DOT, COLON, NEWLINE, PRGM, ANS, NEG, LIST_PREFIX,
	RAND, DIM, SCI_E
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
		return Parser(self.tokens, self.env).parse_expr()



class Parser:

	def __init__(self, tokens: list[Token], env: Environment):
		self.tokens = tokens
		self.pos = 0
		self.env = env

	# ── Primitives ─────────────────────────────────────────────────────────────

	def peek(self) -> Token:
		return self.tokens[self.pos] if self.pos < len(self.tokens) else EOF_TOKEN

	def advance(self) -> Token:
		t = self.peek()
		if t is EOF_TOKEN:
			raise ParseError("Unexpected end of input")
		self.pos += 1
		return t

	def eat_if(self, tok: Token) -> bool:
		if self.peek() is tok:
			self.pos += 1
			return True
		return False

	def expect(self, tok: Token) -> None:
		if self.peek() is not tok:
			raise ParseError(f"Expected {tok.text!r}, got {self.peek().text!r}")
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

	# ── Sub-parsers ────────────────────────────────────────────────────────────

	def parse_num_literal(self, first: Token) -> float:
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

	def parse_string_literal(self) -> str:
		"""Opening \" already consumed. Reads until the next \" or end of line."""
		chars = []
		while not self.at_end() and self.peek() is not QUOTE:
			chars.append(self.advance().text)
		self.eat_if(QUOTE)  # closing " is optional
		return "".join(chars)

	def parse_list_literal(self) -> TiList:
		"""{ already consumed."""
		items = []
		if not self.eat_if(R_BRACE):
			items.append(self.parse_expr())
			while self.eat_if(COMMA):
				items.append(self.parse_expr())
			self.eat_if(R_BRACE)
		return TiList(items)

	def parse_matrix_literal(self) -> TiMatrix:
		"""Opening [ already consumed; reads one or more [row] blocks."""
		rows = []
		while self.peek() is L_BRACKET:
			self.advance()
			row = [self.parse_expr()]
			while self.eat_if(COMMA):
				row.append(self.parse_expr())
			self.eat_if(R_BRACKET)
			rows.append(row)
		self.eat_if(R_BRACKET)
		return TiMatrix(rows)

	def parse_args(self) -> list:
		"""Comma-separated expressions until ) or end of line. Consumes )."""
		args = []
		if not self.at_end() and self.peek() is not R_PAREN:
			args.append(self.parse_expr())
			while self.eat_if(COMMA):
				args.append(self.parse_expr())
		self.eat_if(R_PAREN)
		return args

	def _capture_subgroup(self, out: list[Token]) -> None:
		"""Collect tokens into out until a top-level comma or unmatched ), recursing into sub-groups."""
		while self.pos < len(self.tokens):
			t = self.tokens[self.pos]
			if t is COMMA or t is R_PAREN:
				break
			out.append(t)
			self.pos += 1
			if t.func is not None or t is L_PAREN:
				self._capture_subgroup(out)
				if self.pos < len(self.tokens) and self.tokens[self.pos] is R_PAREN:
					out.append(self.tokens[self.pos])
					self.pos += 1

	def capture(self) -> Thunk:
		"""Return a Thunk for the tokens up to the next top-level comma or )."""
		out: list[Token] = []
		self._capture_subgroup(out)
		return Thunk(out, self.env)

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
			return self.parse_list_literal()
		if t is L_BRACKET:
			return self.parse_matrix_literal()
		if t is L_PAREN:
			val = self.parse_expr()
			self.eat_if(R_PAREN)
			return val

		if t is NEG:
			return t.unary_op(self.parse_expr(65))

		# ᴇ with no left operand: treat as 10^rhs  (e.g. ᴇ10 = 10^10)
		if t is SCI_E:
			return SCI_E.binary_op(1.0, self.parse_expr(SCI_E.bp[1]))

		if t is LIST_PREFIX:
			return self.parse_list_atom(UserListVar(self._read_name()).get(self.env))

		# Function call
		if t.func is not None:
			return t.func(ArgParser(self))

		# Nullary constants (π, e, rand, Ans, getDate, etc.)
		if t.nullary is not None:
			return t.nullary(self.env)

		# Typed variables
		if t.variable is not None:
			val = t.variable.get(self.env)
			if t.is_list_var():
				return self.parse_list_atom(val)
			if t.is_matrix_var() and self.peek() is L_PAREN:
				return val[self.parse_row_col()]
			return val

		raise ParseError(f"Unexpected token in expression: {t.text!r}")

	def parse_list_atom(self, val):
		if self.peek() is L_PAREN:
			self.advance()
			idx = self.parse_expr()
			self.eat_if(R_PAREN)
			return val[idx]  # TiList uses 1-based indexing
		return val

	def parse_row_col(self):
		self.advance()
		row = self.parse_expr()
		self.expect(COMMA)
		col = self.parse_expr()
		self.eat_if(R_PAREN)
		return int(row) - 1, int(col) - 1

	# ── Pratt expression parser ────────────────────────────────────────────────

	def parse_expr(self, min_bp: int = 0):
		lhs = self.parse_atom()

		while True:
			t = self.peek()

			# Postfix operators
			if t.postfix:
				if 80 <= min_bp:
					break
				self.advance()
				lhs = t.unary_op(lhs)
				continue

			# Explicit binary operator
			if t.bp is not None:
				left_bp, right_bp = t.bp
				if left_bp <= min_bp:
					break
				self.advance()
				lhs = t.binary_op(lhs, self.parse_expr(right_bp))
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

	def parse_store_target(self, value):
		t = self.advance()

		if t.is_list_var():
			self.parse_store_list(t.variable, value)

		elif t is LIST_PREFIX:
			self.parse_store_list(UserListVar(self._read_name()), value)

		elif t is DIM:
			self.parse_store_dim(value)

		elif t.is_matrix_var():
			if self.eat_if(L_PAREN):
				t.variable.get(self.env)[self.parse_row_col()] = value
			else:
				t.variable.set(self.env, value)

		elif t is RAND:
			self.env.set_random_seed(value)

		elif t.variable is not None:
			t.variable.set(self.env, value)

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
			self.parse_list_var_key().get(self.env).set_dim(value)
		elif t.is_matrix_var():
			self.advance()
			t.variable.get(self.env).set_dim(value)
		else:
			raise ParseError(f"Invalid store-to-dim target: {t}")

	# ── Variable key parsers ──────────────────────────────────────────────────────

	def parse_list_var_key(self) -> Variable:
		t = self.advance()
		if t.is_list_var():
			return t.variable
		if t is LIST_PREFIX:
			return UserListVar(self._read_name())
		raise ParseError(f"Expected a list variable, got {t.text!r}")

	def parse_matrix_var_key(self) -> Variable:
		t = self.advance()
		if t.is_matrix_var():
			return t.variable
		raise ParseError(f"Expected a matrix variable, got {t.text!r}")

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		if self.at_end():
			return

		t = self.peek()
		
		# prgm subprogram call
		if t is PRGM:
			self.advance()
			name = self._read_name()
			val = self.env.programs[name].execute()

		# Command tokens dispatch via token.cmd
		elif t.cmd is not None:
			self.advance()
			t.cmd(ArgParser(self))
			
		# Expression statement, optionally followed by → target or converter
		else:
			value = self.parse_expr()
			if self.eat_if(STORE):
				self.parse_store_target(value)
			elif self.peek().converter is not None:
				value = self.advance().converter(value)
			self.env.ans = value


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_line(tokens: list[Token], env: dict):
	"""Parse and evaluate a single program line. Returns a StatementResult."""
	Parser(tokens, env).parse_statement()



if __name__ == '__main__':
	from tokens import *

	d = TOKENS[0x2E:0x37]
	lookup = {t.code: t for t in TOKENS}
	def tok(hi, low=None):
		return lookup[bytes([hi] if low is None else [hi, low])]

	env = Environment()
	parse_line([L_BRACE, d[3], COMMA, d[1], d[0], STORE, tok(0x5d, 0)], env)
	# parse_line([d[3], STORE, tok(0x5d, 0)], env)
	# parse_line([d[3], STORE, tok(0x5d, 0), L_PAREN, d[1]], env)
	print(env.lists)
