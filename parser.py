from __future__ import annotations
import math, cmath, random
from operations import EXEC_CALLS
from tokens import Token
from results import (
	ExprResult, StoreResult,
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

	def peek(self) -> Token | None:
		# there is no whitespace to skip
		return self.tokens[self.pos] if self.pos < len(self.tokens) else None

	def advance(self) -> Token:
		t = self.peek()
		if t is None:
			raise ParseError("Unexpected end of input")
		self.pos += 1
		return t

	def eat_if(self, code: bytes) -> bool:
		if self.peek() is not None and self.peek().code == code:
			self.pos += 1
			return True
		return False

	def expect(self, code: bytes) -> Token:
		t = self.peek()
		if t is None or t.code != code:
			raise ParseError(f"Expected {code!r}, got {t!r}")
		self.pos += 1

	def at_end(self) -> bool:
		return self.pos >= len(self.tokens)

	# ── Sub-parsers ────────────────────────────────────────────────────────────

	def parse_num_literal(self) -> float:
		num = []
		while self.peek() is not None and (self.peek().is_digit() or self.peek().code == b'\x3a'):
			num.append(self.advance().text)
		try:
			return float(''.join(num))
		except ValueError:
			raise ParseError(f"Bad numeric literal: {num!r}")

	def parse_string_literal(self) -> str:
		"""Opening \" already consumed. Reads until the next \" or end of line."""
		chars = []
		while self.peek() is not None and self.peek().code != b'\x2a':
			chars.append(self.advance().text)
		self.eat_if(b'\x2a')  # closing " is optional
		return "".join(chars)

	def parse_list_literal(self) -> list:
		"""{ already consumed."""
		items = []
		if not self.eat_if(b'\x09'):  # }
			items.append(self.parse_expr())
			while self.eat_if(b'\x2b'):  # ,
				items.append(self.parse_expr())
			self.eat_if(b'\x09')
		return items

	def parse_matrix_literal(self) -> list[list]:
		"""Opening [ already consumed; reads one or more [row] blocks."""
		rows = []
		while self.peek() is not None and self.peek().code == b'\x06':  # [
			self.advance()
			row = [self.parse_expr()]
			while self.eat_if(b'\x2b'):
				row.append(self.parse_expr())
			self.eat_if(b'\x07')  # ]
			rows.append(row)
		self.eat_if(b'\x07')
		return rows

	def parse_args(self) -> list:
		"""Comma-separated expressions until ) or end of line. Consumes )."""
		args = []
		if self.peek() is not None and self.peek().code != b'\x11':  # )
			args.append(self.parse_expr())
			while self.eat_if(b'\x2b'):  # ,
				args.append(self.parse_expr())
		self.eat_if(b'\x11')
		return args

	def grab_until_comma(self) -> list[Token]:
		"""Return tokens up to the next top-level comma or ), without evaluating."""
		depth = 0
		result = []
		while self.pos < len(self.tokens):
			t = self.tokens[self.pos]
			if t.is_function() or t.code == b'\x10':  # ( or function-open
				depth += 1
			elif t.code == b'\x11':  # )
				if depth == 0:
					break
				depth -= 1
			elif t.code == b'\x2b' and depth == 0:  # ,
				break
			result.append(t)
			self.pos += 1
		return result

	def eval_token_list(self, tokens: list[Token]):
		"""Re-evaluate a captured token list as an expression (used by seq, Σ, etc.)."""
		return Parser(tokens, self.env).parse_expr()

	def parse_label_name(self) -> str:
		"""Read up to 2 alphanumeric characters as a label name."""
		name = ""
		while len(name) < 2 and self.peek() is not None:
			ch = self.peek().text
			if len(ch) == 1 and (ch.isalnum() or ch == 'θ'):
				name += ch; self.advance()
			else:
				break
		if not name:
			raise ParseError("Expected a label name")
		return name

	def _read_name(self, max_len: int) -> str:
		"""Read up to max_len alphanumeric tokens as an identifier (prgm, user list)."""
		name = ""
		while len(name) < max_len and self.peek() is not None:
			ch = self.peek().text
			if len(ch) == 1 and (ch.isalnum() or ch == 'θ'):
				name += ch; self.advance()
			else:
				break
		if not name:
			raise ParseError("Expected a name")
		return name

	# ── Atom parser ────────────────────────────────────────────────────────────

	def parse_atom(self):
		t = self.peek()
		if t is None:
			raise ParseError("Expected an expression")

		# Numeric literal (multi-token — don't advance here)
		if t.is_digit() or t.code == b'\x3a':  # digit or '.'
			return self.parse_num_literal()

		self.advance()

		if t.code == b'\x2a':  # "
			return self.parse_string_literal()
		if t.code == b'\x08':  # {
			return self.parse_list_literal()
		if t.code == b'\x06':  # [
			return self.parse_matrix_literal()

		if t.code == b'\x10':  # (
			val = self.parse_expr()
			self.eat_if(b'\x11')
			return val

		if t.code == b'\xb0':  # − (negation)
			return t.unary_op(self.parse_expr(65))

		# Constants
		if t.code == b'\xac':
			return math.pi
		if t.code == b'\xbb\x31':
			return math.e
		if t.code == b'\x2c':
			return 1j

		# Ans — special: may be followed by (index) if Ans is a list
		if t.code == b'\x72':
			val = self.env.get('Ans', 0.0)
			if isinstance(val, list) and self.peek() is not None and self.peek().code == b'\x10':
				self.advance()
				idx = int(self.parse_expr())
				self.eat_if(b'\x11')
				return val[idx - 1]
			return val

		if t.code == b'\xad':
			return self.env.get('_getkey', 0.0)  # getKey

		# ∟ user-defined list prefix
		if t.code == b'\xeb':
			name = self._read_name(5)
			val  = self.env.get(f'∟{name}', [])
			if self.peek() is not None and self.peek().code == b'\x10':
				self.advance()
				idx = int(self.parse_expr());  self.eat_if(b'\x11')
				return val[idx - 1]
			return val

		# prgm subprogram call
		if t.code == b'\x5f':
			raise NotImplementedError(f"prgm{self._read_name(8)}")

		# Function call (token.call drives arg parsing via parse_args / grab_until_comma)
		if t.is_function():
			return t.call(self)

		# Variables — look up in env; list/matrix vars support (index) access
		if t.is_real_var():
			return self.env.get(t.text, 0.0)

		if t.is_list_var():
			val = self.env.get(t.text, [])
			if self.peek() is not None and self.peek().code == b'\x10':
				self.advance()
				idx = int(self.parse_expr());  self.eat_if(b'\x11')
				return val[idx - 1]
			return val

		if t.is_matrix_var():
			val = self.env.get(t.text, [[]])
			if self.peek() is not None and self.peek().code == b'\x10':
				self.advance()
				row = int(self.parse_expr())
				self.expect(b'\x2b')
				col = int(self.parse_expr())
				self.eat_if(b'\x11')
				return val[row - 1][col - 1]
			return val

		if t.is_string_var() or t.is_stat_var():
			return self.env.get(t.text, 0.0)

		raise ParseError(f"Unexpected token in expression: {t.text!r}")

	# ── Pratt expression parser ────────────────────────────────────────────────

	_POSTFIX = {b'\x0d', b'\x0f', b'\x0c', b'\x2d', b'\x0e', b'\x0b', b'\x0a'}

	_BP: dict[bytes, tuple[int, int]] = {
		b'\x04': (5,  5),   # → store (right side parsed as lvalue, not via bp)
		b'\x3c': (20, 21),  # or
		b'\x3d': (20, 21),  # xor
		b'\x40': (30, 31),  # and
		b'\x6a': (40, 41),  # =
		b'\x6b': (40, 41),  # <
		b'\x6c': (40, 41),  # >
		b'\x6d': (40, 41),  # ≤
		b'\x6e': (40, 41),  # ≥
		b'\x6f': (40, 41),  # ≠
		b'\x70': (50, 51),  # +
		b'\x71': (50, 51),  # -
		b'\x82': (60, 61),  # *
		b'\x83': (60, 61),  # /
		b'\x94': (60, 61),  # nPr
		b'\x95': (60, 61),  # nCr
		b'\xf1': (60, 61),  # ×√
		b'\xf0': (70, 69),  # ^ right-assoc
	}
	_IMPL_MUL_BP = (60, 61)
	_POSTFIX_BP  = 80

	def parse_expr(self, min_bp: int = 0):
		lhs = self.parse_atom()

		while True:
			t = self.peek()
			if t is None:
				break

			# Postfix operators
			if t.code in self._POSTFIX:
				if self._POSTFIX_BP <= min_bp:
					break
				self.advance()
				lhs = t.unary_op(lhs)
				continue

			# Store — right side is an lvalue, not a general expression
			if t.code == b'\x04':
				if self._BP[b'\x04'][0] <= min_bp:
					break
				self.advance()
				return lhs, self.parse_store_target()

			# Explicit binary operator
			bp = self._BP.get(t.code)
			if bp is not None:
				left_bp, right_bp = bp
				if left_bp <= min_bp:
					break
				self.advance()
				lhs = t.binary_op(lhs, self.parse_expr(right_bp))
				continue

			# Implicit multiplication
			if t.can_start_atom():
				left_bp, right_bp = self._IMPL_MUL_BP
				if left_bp <= min_bp:
					break
				lhs = lhs * self.parse_expr(right_bp)
				continue

			break

		return lhs

	# ── Store target parser ────────────────────────────────────────────────────

	def parse_store_target(self):
		t = self.advance()

		if t.code == b'\x72':
			return AnsTarget()

		if t.is_real_var():
			return RealVarTarget(t.text)

		if t.is_list_var():
			if self.eat_if(b'\x10'):
				idx = self.parse_expr();
				self.eat_if(b'\x11')
				return ListElementTarget(t.code, idx)
			return ListVarTarget(t.code)

		if t.is_matrix_var():
			if self.eat_if(b'\x10'):
				row = self.parse_expr();
				self.expect(b'\x2b')
				col = self.parse_expr();
				self.eat_if(b'\x11')
				return MatrixElementTarget(t.code, row, col)
			return MatrixVarTarget(t.code)

		if t.is_string_var():
			return StringVarTarget(t.code)

		if t.code == b'\xeb':  # user list
			name = self._read_name(5)
			if self.eat_if(b'\x10'):
				idx = self.parse_expr();  self.eat_if(b'\x11')
				return ListElementTarget(name, idx)
			return UserListTarget(name)

		raise ParseError(f"Invalid store target: {t.text!r}")

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		t = self.peek()
		if t is None:
			return None

		# Command tokens dispatch via Token.execute (implemented in operations.py)
		if t.code in EXEC_CALLS:
			self.advance()  # consume the command token
			return t.execute(self)

		# Expression statement (possibly ending with →)
		result = self.parse_expr()
		if isinstance(result, tuple):
			value, target = result
			return StoreResult(value, target)
		return ExprResult(result)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_line(tokens: list[Token], env: dict):
	"""Parse and evaluate a single program line. Returns a StatementResult."""
	return Parser(tokens, env).parse_statement()
