from __future__ import annotations
import math, cmath, random
from tokens import (
	Token, EOF_TOKEN,
	STORE, L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN,
	QUOTE, COMMA, DOT, COLON, NEWLINE, PRGM, ANS, NEG, LIST_PREFIX,
)
from results import (
	ExprResult, StoreResult, Thunk,
	RealVarTarget, ListVarTarget, UserListTarget, MatrixVarTarget,
	StringVarTarget, ListElementTarget, MatrixElementTarget, AnsTarget,
)

class ParseError(Exception):
	pass

# ── Parser ─────────────────────────────────────────────────────────────────────

class Parser:
	def __init__(self, tokens: list[Token], env: dict):
		self.tokens = tokens
		self.pos    = 0
		self.env    = env

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

	# ── Sub-parsers ────────────────────────────────────────────────────────────

	def parse_num_literal(self) -> float:
		num = []
		while self.peek().is_digit() or self.peek() is DOT:
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

	def parse_list_literal(self) -> list:
		"""{ already consumed."""
		items = []
		if not self.eat_if(R_BRACE):
			items.append(self.parse_expr())
			while self.eat_if(COMMA):
				items.append(self.parse_expr())
			self.eat_if(R_BRACE)
		return items

	def parse_matrix_literal(self) -> list[list]:
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
		return rows

	def parse_args(self) -> list:
		"""Comma-separated expressions until ) or end of line. Consumes )."""
		args = []
		if not self.at_end() and self.peek() is not R_PAREN:
			args.append(self.parse_expr())
			while self.eat_if(COMMA):
				args.append(self.parse_expr())
		self.eat_if(R_PAREN)
		return args

	def _grab_group(self, out: list) -> None:
		"""Collect tokens into out until a top-level comma or unmatched ), recursing into sub-groups."""
		while self.pos < len(self.tokens):
			t = self.tokens[self.pos]
			if t is COMMA or t is R_PAREN:
				break
			out.append(t)
			self.pos += 1
			if t.is_function() or t is L_PAREN:
				self._grab_group(out)
				if self.pos < len(self.tokens) and self.tokens[self.pos] is R_PAREN:
					out.append(self.tokens[self.pos])
					self.pos += 1

	def grab_until_comma(self) -> Thunk:
		"""Return a Thunk for the tokens up to the next top-level comma or )."""
		out: list = []
		self._grab_group(out)
		return Thunk(out, self.env)

	def parse_label_name(self) -> str:
		"""Read up to 2 alphanumeric characters as a label name."""
		name = ""
		while len(name) < 2:
			ch = self.peek().text
			if len(ch) == 1 and (ch.isalnum() or ch == 'θ'):
				name += ch; self.advance()
			else:
				break
		if not name:
			raise ParseError("Expected a label name")
		return name

	def _read_name(self) -> str:
		"""Read alphanumeric tokens as an identifier (prgm, user list, etc.)."""
		name = ""
		while True:
			ch = self.peek().text
			if len(ch) == 1 and (ch.isalnum() or ch == 'θ'):
				name += ch; self.advance()
			else:
				break
		if not name:
			raise ParseError("Expected a name")
		return name

	# ── Atom parser ────────────────────────────────────────────────────────────

	def _check_int_index(self, val) -> int:
		if val != int(val):
			raise ParseError(f"Index must be an integer, got {val}")
		return int(val)

	def parse_atom(self):
		t = self.peek()
		if self.at_end():
			raise ParseError("Expected an expression")

		# Numeric literal (multi-token — don't advance here)
		if t.is_digit() or t is DOT:
			return self.parse_num_literal()

		self.advance()

		if t is QUOTE:       return self.parse_string_literal()
		if t is L_BRACE:     return self.parse_list_literal()
		if t is L_BRACKET:   return self.parse_matrix_literal()

		if t is L_PAREN:
			val = self.parse_expr()
			self.eat_if(R_PAREN)
			return val

		if t is NEG:
			return t.unary_op_fn(self.parse_expr(65))

		# Ans — special: may be indexed when it holds a list
		if t is ANS:
			val = self.env.get('Ans', 0.0)
			if isinstance(val, list) and self.peek() is L_PAREN:
				self.advance()
				idx = self._check_int_index(self.parse_expr())
				self.eat_if(R_PAREN)
				return val[idx - 1]
			return val

		# ∟ user-defined list prefix
		if t is LIST_PREFIX:
			name = self._read_name()
			val  = self.env.get(f'∟{name}', [])
			if self.peek() is L_PAREN:
				self.advance()
				idx = self._check_int_index(self.parse_expr())
				self.eat_if(R_PAREN)
				return val[idx - 1]
			return val

		# prgm subprogram call
		if t is PRGM:
			raise NotImplementedError(f"prgm{self._read_name()}")

		# Function call
		if t.call_fn is not None:
			return t.call_fn(self)

		# Nullary (0-arg, no parentheses): π, e, 𝑖, rand, getKey, etc.
		if t.value_fn is not None:
			return t.value_fn(self.env)

		# Variables — list/matrix vars support (index) access
		if t.is_real_var():
			return self.env.setdefault(t.text, 0.0)

		if t.is_list_var():
			val = self.env.get(t.text, [])
			if self.peek() is L_PAREN:
				self.advance()
				idx = self._check_int_index(self.parse_expr())
				self.eat_if(R_PAREN)
				return val[idx - 1]
			return val

		if t.is_matrix_var():
			val = self.env.get(t.text, [[]])
			if self.peek() is L_PAREN:
				self.advance()
				row = self._check_int_index(self.parse_expr())
				self.expect(COMMA)
				col = self._check_int_index(self.parse_expr())
				self.eat_if(R_PAREN)
				return val[row - 1][col - 1]
			return val

		if t.is_string_var() or t.is_stat_var():
			return self.env.get(t.text, 0.0)

		raise ParseError(f"Unexpected token in expression: {t.text!r}")

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
				lhs = t.unary_op_fn(lhs)
				continue

			# Explicit binary operator
			if t.bp is not None:
				left_bp, right_bp = t.bp
				if left_bp <= min_bp:
					break
				self.advance()
				lhs = t.binary_op_fn(lhs, self.parse_expr(right_bp))
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

	def parse_store_target(self):
		t = self.advance()

		if t is ANS:
			return AnsTarget()

		if t.is_real_var():
			return RealVarTarget(t.text)

		if t.is_list_var():
			if self.eat_if(L_PAREN):
				idx = self.parse_expr()
				self.eat_if(R_PAREN)
				return ListElementTarget(t.code, idx)
			return ListVarTarget(t.code)

		if t.is_matrix_var():
			if self.eat_if(L_PAREN):
				row = self.parse_expr()
				self.expect(COMMA)
				col = self.parse_expr()
				self.eat_if(R_PAREN)
				return MatrixElementTarget(t.code, row, col)
			return MatrixVarTarget(t.code)

		if t.is_string_var():
			return StringVarTarget(t.code)

		if t is LIST_PREFIX:
			name = self._read_name()
			if self.eat_if(L_PAREN):
				idx = self.parse_expr()
				self.eat_if(R_PAREN)
				return ListElementTarget(name, idx)
			return UserListTarget(name)

		raise ParseError(f"Invalid store target: {t.text!r}")

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		if self.at_end():
			return None

		t = self.peek()

		# Command tokens dispatch via token.execute_fn
		if t.execute_fn is not None:
			self.advance()
			return t.execute_fn(self)

		# Expression statement, optionally followed by → target
		value = self.parse_expr()
		if self.eat_if(STORE):
			return StoreResult(value, self.parse_store_target())
		return ExprResult(value)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_line(tokens: list[Token], env: dict):
	"""Parse and evaluate a single program line. Returns a StatementResult."""
	return Parser(tokens, env).parse_statement()
