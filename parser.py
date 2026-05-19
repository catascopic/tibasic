from dataclasses import dataclass

from tokens import (
	Token, EOF_TOKEN,
	STORE, L_BRACKET, R_BRACKET, L_BRACE, R_BRACE, L_PAREN, R_PAREN,
	QUOTE, COMMA, DOT, COLON, NEWLINE, PRGM, ANS, NEG, LIST_PREFIX,
	RAND, DIM, SCI_E
)
from environment import Environment
from purefunctions import TiList, TiMatrix


class ParseError(Exception):
	pass


@dataclass
class Thunk:
	tokens: list[Token]
	env: Environment

	def eval(self):
		return Parser(self.tokens, self.env).parse_expr()


class ArgParser:
	"""Stateful helper for parsing comma-separated function arguments."""

	def __init__(self, parser: 'Parser'):
		self._p = parser
		self._first = True

	def _sep(self):
		if self._first:
			self._first = False
		else:
			self._p.expect(COMMA)

	def expr(self):
		self._sep()
		return self._p.parse_expr()

	def thunk(self):
		self._sep()
		return self._p.capture()

	def real_var(self) -> Token:
		self._sep()
		tok = self._p.advance()
		if not tok.is_real_var():
			raise ParseError(f"Expected a real variable, got {tok.text!r}")
		return tok

	def list_var(self):
		self._sep()
		return self._p.parse_list_var_key()

	def matrix_var(self):
		self._sep()
		return self._p.parse_matrix_var_key()

	def opt_expr(self, default):
		if self._p.eat_if(COMMA):
			return self._p.parse_expr()
		return default

	def finish(self):
		self._p.eat_if(R_PAREN)


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
			raise ParseError("Expected a label name")
		label = t.text
		if self.peek().is_name_char():
			label += self.advance().text

		return label

	def _read_name(self) -> str:
		"""Read alphanumeric tokens as an identifier (prgm, user list, etc.)."""
		t = self.advance()
		if not t.is_real_var():
			raise ParseError("Expected a label name")
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
			val = self.env.lists[self._read_name()]
			return self.parse_list_atom(val)

		# Function call
		if t.func is not None:
			return t.func(self)

		# Variables and nullary constants (π, e, rand, Ans, etc.)
		if t.resolve is not None:
			val = t.resolve(self.env)
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
			return val[int(idx) - 1]
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

		if t is LIST_PREFIX:
			self.parse_store_list(self._read_name(), value)

		elif t is DIM:
			self.parse_store_dim(value)

		elif t.is_list_var():
			self.parse_store_list(t, value)

		elif t.is_matrix_var():
			if self.eat_if(L_PAREN):
				self.env.matrices[t][self.parse_row_col()] = value
			else:
				t.store(self.env, value)

		elif t.store is not None:
			t.store(self.env, value)

		else:
			raise ParseError(f"Invalid store target: {t}")
	
	def parse_store_list(self, name, value):
		if self.eat_if(L_PAREN):
			idx = self.parse_expr()
			self.eat_if(R_PAREN)
			self.env.lists[name][idx] = value
		else:
			self.env.lists[name] = value
	
	def parse_store_dim(self, value):
		t = self.advance()
		if t.is_list_var():
			self.env.lists[t].set_dim(value)
		elif t is LIST_PREFIX:
			self.env.lists[self._read_name()].set_dim(value)
		elif t.is_matrix_var():
			self.env.matrices[t].set_dim(value)
		else:
			raise ParseError(f"Invalid store-to-dim target: {t}")

	# ── Variable key parsers (return the dict key for env.lists / env.matrices) ──

	def parse_list_var_key(self):
		t = self.advance()
		if t is LIST_PREFIX:
			return self._read_name()
		if t.is_list_var():
			return t
		raise ParseError(f"Expected a list variable, got {t.text!r}")

	def parse_matrix_var_key(self):
		t = self.advance()
		if t.is_matrix_var():
			return t
		raise ParseError(f"Expected a matrix variable, got {t.text!r}")

	# ── Custom-parsed functions ────────────────────────────────────────────────

	def call_seq(self) -> TiList:
		a = ArgParser(self)
		thunk = a.thunk()
		var = a.real_var()
		start = a.expr()
		end = a.expr()
		step = a.opt_expr(1.0)
		a.finish()
		saved = self.env.reals.get(var)
		result = []
		n = start
		while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
			self.env.reals[var] = n
			result.append(thunk.eval())
			n += step
		if saved is not None:
			self.env.reals[var] = saved
		else:
			self.env.reals.pop(var, None)
		return TiList(result)

	def call_sigma(self) -> float:
		a = ArgParser(self)
		thunk = a.thunk()
		var = a.real_var()
		start = int(a.expr())
		end = int(a.expr())
		a.finish()
		saved = self.env.reals.get(var)
		total = 0.0
		for i in range(start, end + 1):
			self.env.reals[var] = float(i)
			total += thunk.eval()
		if saved is not None:
			self.env.reals[var] = saved
		else:
			self.env.reals.pop(var, None)
		return total

	def call_nderiv(self) -> float:
		a = ArgParser(self)
		thunk = a.thunk()
		var = a.real_var()
		val = a.expr()
		h = a.opt_expr(1e-5)
		a.finish()
		saved = self.env.reals.get(var)
		self.env.reals[var] = val + h
		fwd = thunk.eval()
		self.env.reals[var] = val - h
		bwd = thunk.eval()
		if saved is not None:
			self.env.reals[var] = saved
		else:
			self.env.reals.pop(var, None)
		return (fwd - bwd) / (2 * h)

	def call_fnint(self) -> float:
		a = ArgParser(self)
		thunk = a.thunk()
		var = a.real_var()
		lo = a.expr()
		hi = a.expr()
		a.finish()
		saved = self.env.reals.get(var)
		n = 1000
		h = (hi - lo) / n
		def f(x):
			self.env.reals[var] = x
			return thunk.eval()
		total = f(lo) + f(hi)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(lo + i * h)
		if saved is not None:
			self.env.reals[var] = saved
		else:
			self.env.reals.pop(var, None)
		return total * h / 3

	def call_matr_to_list(self) -> None:
		a = ArgParser(self)
		mat = a.expr()
		if not isinstance(mat, TiMatrix):
			raise ParseError("Matr►list: first argument must be a matrix")
		keys = [a.list_var()]
		while self.peek() is COMMA:
			keys.append(a.list_var())
		a.finish()
		for col, key in enumerate(keys):
			self.env.lists[key] = TiList([mat.inner[r][col] for r in range(mat.rows)])

	def call_list_to_matr(self) -> None:
		a = ArgParser(self)
		list_vals = []
		while True:
			list_vals.append(a.expr())
			if self.peek() is not COMMA:
				raise ParseError("List►matr: expected matrix variable as last argument")
			next_tok = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else EOF_TOKEN
			if next_tok.is_matrix_var():
				mat_key = a.matrix_var()
				break
		a.finish()
		cols = len(list_vals)
		rows = max(len(lst) for lst in list_vals)
		self.env.matrices[mat_key] = TiMatrix([
			[list_vals[c].inner[r] if r < len(list_vals[c]) else 0.0 for c in range(cols)]
			for r in range(rows)
		])

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
			t.cmd(self)
			
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
	from tokens import TOKENS, ADD, MUL

	digits = TOKENS[0x2E:0x37]

	env = Environment()
	print(parse_line([L_PAREN, digits[2], ADD, NEG, RAND, digits[3], R_PAREN, MUL, digits[2]], env))
	print(env.ans)
