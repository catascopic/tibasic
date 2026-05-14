from __future__ import annotations
import math, cmath, random
from dataclasses import dataclass, field
from typing import Any
from tokens import Token

Value = float | complex | str | list  # list = TI list or matrix row-of-rows

# ── Token code constants ───────────────────────────────────────────────────────
STORE    = b'\x04'; MAT_L   = b'\x06'; MAT_R  = b'\x07'
LIST_L   = b'\x08'; LIST_R  = b'\x09'; RAD    = b'\x0a'
DEG      = b'\x0b'; INV     = b'\x0c'; SQ     = b'\x0d'
TRANS    = b'\x0e'; CUBE    = b'\x0f'; LPAREN = b'\x10'
RPAREN   = b'\x11'; QUOT    = b'\x2a'; COMMA  = b'\x2b'
IMAG_I   = b'\x2c'; FACT    = b'\x2d'; DECIMAL= b'\x3a'
SCI_E    = b'\x3b'; OR_OP   = b'\x3c'; XOR_OP = b'\x3d'
SEP      = b'\x3e'; NEWLINE = b'\x3f'; AND_OP = b'\x40'
THETA    = b'\x5b'; EQ_OP   = b'\x6a'; LT_OP  = b'\x6b'
GT_OP    = b'\x6c'; LE_OP   = b'\x6d'; GE_OP  = b'\x6e'
NE_OP    = b'\x6f'; ADD     = b'\x70'; SUB    = b'\x71'
ANS      = b'\x72'; MUL     = b'\x82'; DIV    = b'\x83'
RAND     = b'\xab'; PI_TOK  = b'\xac'; GETKEY = b'\xad'
NEG      = b'\xb0'; LIST_PFX= b'\xeb'; NPR    = b'\x94'
NCR      = b'\x95'; POW     = b'\xf0'; XROOT  = b'\xf1'
CONST_E  = b'\xbb\x31'

IF      = b'\xce'; THEN    = b'\xcf'; ELSE    = b'\xd0'
WHILE   = b'\xd1'; REPEAT  = b'\xd2'; FOR     = b'\xd3'
END     = b'\xd4'; RETURN  = b'\xd5'; LBL     = b'\xd6'
GOTO    = b'\xd7'; PAUSE   = b'\xd8'; STOP_CMD= b'\xd9'
IS_GT   = b'\xda'; DS_LT   = b'\xdb'; INPUT   = b'\xdc'
PROMPT  = b'\xdd'; DISP    = b'\xde'; DISPGRAPH=b'\xdf'
OUTPUT  = b'\xe0'; CLRHOME = b'\xe1'; MENU    = b'\xe6'
SEQ_FUNC= b'\x23'; FNINT   = b'\x24'; NDERIV  = b'\x25'
SIGMA_F = b'\xef\x33'

# ── Binding powers (left_bp, right_bp) ────────────────────────────────────────
_BP: dict[bytes, tuple[int, int]] = {
	STORE:  (5,  5),   # right side parsed as lvalue, not via bp
	OR_OP:  (20, 21),
	XOR_OP: (20, 21),
	AND_OP: (30, 31),
	EQ_OP:  (40, 41),  LT_OP: (40, 41),  GT_OP: (40, 41),
	LE_OP:  (40, 41),  GE_OP: (40, 41),  NE_OP: (40, 41),
	ADD:    (50, 51),  SUB:   (50, 51),
	MUL:    (60, 61),  DIV:   (60, 61),
	NPR:    (60, 61),  NCR:   (60, 61),  XROOT: (60, 61),
	POW:    (70, 69),  # right-assoc: right_bp < left_bp so 2^3^4 = 2^(3^4)
}
IMPL_MUL_BP = (60, 61)  # implicit multiplication
POSTFIX_BP  = 80

# ── Statement result types ─────────────────────────────────────────────────────
@dataclass
class ExprResult:
	value: Value

@dataclass
class StoreResult:
	value: Value
	target: StoreTarget

@dataclass
class IfResult:
	cond: bool

@dataclass
class WhileResult:
	cond: bool

@dataclass
class RepeatResult:
	pass

@dataclass
class ForResult:
	var: str
	start: float
	end: float
	step: float

@dataclass
class EndResult:
	pass

@dataclass
class LblResult:
	label: str

@dataclass
class GotoResult:
	label: str

@dataclass
class ReturnResult:
	pass

@dataclass
class StopResult:
	pass

@dataclass
class PauseResult:
	value: Value | None

@dataclass
class DispResult:
	values: list[Value]

@dataclass
class InputResult:
	prompt: str | None
	target: StoreTarget

@dataclass
class PromptResult:
	targets: list[StoreTarget]

@dataclass
class OutputResult:
	row: float
	col: float
	value: Value

@dataclass
class MenuResult:
	title: Value
	options: list[tuple[Value, str]]  # (display_name, label)

@dataclass
class ClrHomeResult:
	pass

@dataclass
class DispGraphResult:
	pass

# ── Store target types ─────────────────────────────────────────────────────────
@dataclass
class RealVarTarget:
	name: str  # single letter or 'θ'

@dataclass
class ListVarTarget:
	code: bytes  # 0x5Dxx

@dataclass
class UserListTarget:
	name: str

@dataclass
class MatrixVarTarget:
	code: bytes  # 0x5Cxx

@dataclass
class StringVarTarget:
	code: bytes  # 0xAAxx

@dataclass
class ListElementTarget:
	list_ref: bytes | str  # code (built-in) or name (user-defined)
	index: Value

@dataclass
class MatrixElementTarget:
	code: bytes
	row: Value
	col: Value

@dataclass
class AnsTarget:
	pass

StoreTarget = (RealVarTarget | ListVarTarget | UserListTarget | MatrixVarTarget
	| StringVarTarget | ListElementTarget | MatrixElementTarget | AnsTarget)

# ── Built-in math function dispatch ───────────────────────────────────────────
def _factorial(n: float) -> float:
	if n < 0 or n != int(n):
		raise ValueError("Domain")
	return float(math.factorial(int(n)))

def _ncr(n: float, r: float) -> float:
	return float(math.comb(int(n), int(r)))

def _npr(n: float, r: float) -> float:
	return float(math.perm(int(n), int(r)))

def _ti_int(x: float) -> float:
	return math.floor(x) if x >= 0 else math.ceil(x)  # TI int() = floor for positive

def _fpart(x: float) -> float:
	return x - _ti_int(x)

# Maps token code → callable accepting a flat arg list
_BUILTIN: dict[bytes, Any] = {
	b'\x12': lambda a, b=0: round(a, int(b)),   # round(
	b'\x19': lambda *a: max(a) if len(a) > 1 else max(a[0]),  # max(
	b'\x1a': lambda *a: min(a) if len(a) > 1 else min(a[0]),  # min(
	b'\x1f': lambda a, b=None: (float(sorted(a)[len(a)//2]) if b is None  # median(
		else None),  # TODO: weighted median
	b'\x21': lambda a, b=None: (sum(a)/len(a) if b is None else  # mean(
		sum(x*w for x,w in zip(a,b))/sum(b)),
	b'\xb1': _ti_int,           # int(
	b'\xb2': abs,               # abs(
	b'\xb5': lambda a: (float(len(a)) if isinstance(a, list) else  # dim(
		[float(len(a)), float(len(a[0]))]),
	b'\xb6': lambda a: sum(a),  # sum(
	b'\xb7': lambda a: math.prod(a),  # prod(
	b'\xb8': lambda a: float(not a),  # not(
	b'\xb9': math.trunc,        # iPart(
	b'\xba': _fpart,            # fPart(
	b'\xbc': lambda a: cmath.sqrt(a) if a < 0 else math.sqrt(a),  # √(
	b'\xbd': lambda a: -(-a)**(1/3) if a < 0 else a**(1/3),       # ³√(
	b'\xbe': cmath.log,         # ln(
	b'\xbf': cmath.exp,         # e^(
	b'\xc0': cmath.log10,       # log(
	b'\xc1': lambda a: 10**a,   # 10^(
	b'\xc2': math.sin,          # sin(
	b'\xc3': math.asin,         # sin⁻¹(
	b'\xc4': math.cos,          # cos(
	b'\xc5': math.acos,         # cos⁻¹(
	b'\xc6': math.tan,          # tan(
	b'\xc7': math.atan,         # tan⁻¹(
	b'\xc8': math.sinh,         # sinh(
	b'\xc9': math.asinh,        # sinh⁻¹(
	b'\xca': math.cosh,         # cosh(
	b'\xcb': math.acosh,        # cosh⁻¹(
	b'\xcc': math.tanh,         # tanh(
	b'\xcd': math.atanh,        # tanh⁻¹(
	b'\xe2': lambda v, lst: lst.__setitem__(slice(None), [v]*len(lst)) or lst,  # Fill(
	b'\xbb\x08': math.lcm,      # lcm(
	b'\xbb\x09': math.gcd,      # gcd(
	b'\xbb\x0a': lambda a, b, n=1: [float(random.randint(int(a), int(b))) for _ in range(int(n))][0] if n == 1 else [float(random.randint(int(a), int(b))) for _ in range(int(n))],  # randInt(
	b'\xbb\x0c': lambda s, start, length: s[int(start)-1:int(start)-1+int(length)],  # sub(
	b'\xbb\x0f': lambda s, sub: float(s.find(sub) + 1) if sub in s else 0.0,  # inString(
	b'\xbb\x2b': lambda s: float(len(s)),  # length(
	b'\xbb\x29': lambda a: [sum(a[:i+1]) for i in range(len(a))],  # cumSum(
	b'\xbb\x2c': lambda a: [a[i+1]-a[i] for i in range(len(a)-1)],  # ΔList(
	b'\xbb\x25': lambda a: complex(a.real, -a.imag),  # conj(
	b'\xbb\x26': lambda a: a.real,   # real(
	b'\xbb\x27': lambda a: a.imag,   # imag(
	b'\xbb\x28': lambda a: cmath.phase(a),  # angle(
	b'\xef\x02': NotImplemented,    # checkTmr( — executor provides
	b'\xef\x05': NotImplemented,    # timeCnv(
	b'\xef\x06': NotImplemented,    # dayOfWk(
	b'\xef\x32': lambda a, b: float(int(a) % int(b)),  # remainder(
	b'\xef\x34': lambda a, b: math.log(a) / math.log(b),  # logBASE(
	b'\xef\x35': lambda a, b, n=None: random.sample(range(int(a), int(b)+1), int(b-a+1) if n is None else int(n)),  # randIntNoRep(
}

# ── Error ──────────────────────────────────────────────────────────────────────
class ParseError(Exception):
	pass

# ── Token predicates ───────────────────────────────────────────────────────────
def _is_real_var(t: Token) -> bool:
	return (len(t.code) == 1 and 0x41 <= t.code[0] <= 0x5a) or t.code == THETA

def _is_list_var(t: Token) -> bool:
	return len(t.code) == 2 and t.code[0] == 0x5d

def _is_matrix_var(t: Token) -> bool:
	return len(t.code) == 2 and t.code[0] == 0x5c

def _is_string_var(t: Token) -> bool:
	return len(t.code) == 2 and t.code[0] == 0xaa

def _is_stat_var(t: Token) -> bool:
	return len(t.code) == 2 and t.code[0] in (0x62, 0x63)

def _is_digit(t: Token) -> bool:
	return len(t.code) == 1 and 0x30 <= t.code[0] <= 0x39

def _can_start_atom(t: Token) -> bool:
	return (
		_is_digit(t) or t.code == DECIMAL or _is_real_var(t) or
		_is_list_var(t) or _is_matrix_var(t) or _is_string_var(t) or
		t.text.endswith('(') or
		t.code in (LPAREN, LIST_L, QUOT, NEG, PI_TOK, ANS, RAND,
				   GETKEY, IMAG_I, CONST_E, LIST_PFX)
	)

# ── Parser ─────────────────────────────────────────────────────────────────────
class Parser:
	def __init__(self, tokens: list[Token], env: dict):
		self.tokens = tokens
		self.pos    = 0
		self.env    = env

	# ── Primitives ─────────────────────────────────────────────────────────────

	def peek(self) -> Token | None:
		while self.pos < len(self.tokens) and self.tokens[self.pos].code == b'\x29':
			self.pos += 1  # skip space tokens
		return self.tokens[self.pos] if self.pos < len(self.tokens) else None

	def advance(self) -> Token:
		t = self.peek()
		if t is None:
			raise ParseError("Unexpected end of input")
		self.pos += 1
		return t

	def eat_if(self, code: bytes) -> bool:
		if self.peek() is not None and self.peek().code == code:
			self.advance()
			return True
		return False

	def expect(self, code: bytes) -> Token:
		t = self.peek()
		if t is None or t.code != code:
			raise ParseError(f"Expected {code!r}, got {t!r}")
		return self.advance()

	def at_end(self) -> bool:
		return self.peek() is None

	# ── Numeric literal assembler ───────────────────────────────────────────────

	def parse_num_literal(self) -> float:
		s = ""
		while self.peek() is not None and (_is_digit(self.peek()) or self.peek().code == DECIMAL):
			s += self.advance().text
		if self.peek() is not None and self.peek().code == SCI_E:
			self.advance()
			s += "e"
			if self.peek() is not None and self.peek().code == NEG:
				self.advance(); s += "-"
			while self.peek() is not None and _is_digit(self.peek()):
				s += self.advance().text
		try:
			return float(s)
		except ValueError:
			raise ParseError(f"Bad number: {s!r}")

	# ── String literal ─────────────────────────────────────────────────────────

	def parse_string_literal(self) -> str:
		# Opening " already consumed. Reads tokens until the next " or end of line.
		parts = []
		while self.peek() is not None and self.peek().code != QUOT:
			parts.append(self.advance().text)
		self.eat_if(QUOT)  # closing " is optional (implicit)
		return "".join(parts)

	# ── List literal {a, b, c} ─────────────────────────────────────────────────

	def parse_list_literal(self) -> list:
		items: list[Value] = []
		if not self.eat_if(LIST_R):
			items.append(self.parse_expr())
			while self.eat_if(COMMA):
				items.append(self.parse_expr())
			self.eat_if(LIST_R)
		return items

	# ── Matrix literal [[r1][r2]] ──────────────────────────────────────────────

	def parse_matrix_literal(self) -> list[list]:
		rows: list[list] = []
		while self.peek() is not None and self.peek().code == MAT_L:
			self.advance()
			row: list[Value] = []
			row.append(self.parse_expr())
			while self.eat_if(COMMA):
				row.append(self.parse_expr())
			self.eat_if(MAT_R)
			rows.append(row)
		self.eat_if(MAT_R)
		return rows

	# ── Argument list parser ───────────────────────────────────────────────────

	def parse_args(self) -> list[Value]:
		"""Parse comma-separated expressions until ) or end of line. Consumes )."""
		args: list[Value] = []
		if self.peek() is not None and self.peek().code != RPAREN:
			args.append(self.parse_expr())
			while self.eat_if(COMMA):
				args.append(self.parse_expr())
		self.eat_if(RPAREN)
		return args

	def grab_until_comma(self) -> list[Token]:
		"""Return (without evaluating) tokens up to next top-level comma or )."""
		depth = 0
		result = []
		start = self.pos
		while self.pos < len(self.tokens):
			t = self.tokens[self.pos]
			if t.code in (LPAREN,) or t.text.endswith('('):
				depth += 1
			elif t.code == RPAREN:
				if depth == 0:
					break
				depth -= 1
			elif t.code == COMMA and depth == 0:
				break
			result.append(t)
			self.pos += 1
		return result

	# ── Special functions that need token-list re-evaluation ───────────────────

	def _call_seq(self) -> list:
		expr_toks = self.grab_until_comma()
		self.expect(COMMA)
		var_tok = self.advance()
		if not _is_real_var(var_tok):
			raise ParseError("seq: second arg must be a variable")
		var = var_tok.text
		self.expect(COMMA)
		start = self.parse_expr()
		self.expect(COMMA)
		end   = self.parse_expr()
		step  = 1.0
		if self.eat_if(COMMA):
			step = self.parse_expr()
		self.eat_if(RPAREN)

		saved = self.env.get(var)
		result = []
		n = start
		while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
			self.env[var] = n
			result.append(eval_tokens(expr_toks, self.env))
			n += step
		if saved is None:
			self.env.pop(var, None)
		else:
			self.env[var] = saved
		return result

	def _call_sigma(self) -> float:
		expr_toks = self.grab_until_comma()
		self.expect(COMMA)
		var_tok = self.advance()
		if not _is_real_var(var_tok):
			raise ParseError("Σ: second arg must be a variable")
		var = var_tok.text
		self.expect(COMMA)
		start = int(self.parse_expr())
		self.expect(COMMA)
		end   = int(self.parse_expr())
		self.eat_if(RPAREN)

		saved = self.env.get(var)
		total = 0.0
		for i in range(start, end + 1):
			self.env[var] = float(i)
			total += eval_tokens(expr_toks, self.env)
		if saved is None:
			self.env.pop(var, None)
		else:
			self.env[var] = saved
		return total

	def _call_nderiv(self) -> float:
		expr_toks = self.grab_until_comma()
		self.expect(COMMA)
		var_tok = self.advance()
		var = var_tok.text
		self.expect(COMMA)
		val = self.parse_expr()
		h = 1e-5
		if self.eat_if(COMMA):
			h = self.parse_expr()
		self.eat_if(RPAREN)

		saved = self.env.get(var)
		self.env[var] = val + h
		fwd = eval_tokens(expr_toks, self.env)
		self.env[var] = val - h
		bwd = eval_tokens(expr_toks, self.env)
		if saved is None:
			self.env.pop(var, None)
		else:
			self.env[var] = saved
		return (fwd - bwd) / (2 * h)

	def _call_fnint(self) -> float:
		expr_toks = self.grab_until_comma()
		self.expect(COMMA)
		var_tok = self.advance()
		var = var_tok.text
		self.expect(COMMA)
		a = self.parse_expr()
		self.expect(COMMA)
		b = self.parse_expr()
		self.eat_if(RPAREN)

		saved = self.env.get(var)
		# Simpson's rule with 1000 intervals
		n = 1000
		h = (b - a) / n
		def f(x):
			self.env[var] = x
			return eval_tokens(expr_toks, self.env)
		total = f(a) + f(b)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(a + i * h)
		if saved is None:
			self.env.pop(var, None)
		else:
			self.env[var] = saved
		return total * h / 3

	# ── Atom parser ────────────────────────────────────────────────────────────

	def parse_atom(self) -> Value:
		t = self.peek()
		if t is None:
			raise ParseError("Expected expression")

		# Numeric literal
		if _is_digit(t) or t.code == DECIMAL:
			return self.parse_num_literal()

		self.advance()

		# String literal
		if t.code == QUOT:
			return self.parse_string_literal()

		# List literal
		if t.code == LIST_L:
			return self.parse_list_literal()

		# Matrix literal
		if t.code == MAT_L:
			return self.parse_matrix_literal()

		# Parenthesised expression
		if t.code == LPAREN:
			val = self.parse_expr()
			self.eat_if(RPAREN)
			return val

		# Unary negation
		if t.code == NEG:
			return -self.parse_expr(65)  # binds tighter than +/-, looser than ^

		# Constants
		if t.code == PI_TOK:    return math.pi
		if t.code == CONST_E:   return math.e
		if t.code == IMAG_I:    return 1j
		if t.code == ANS:
			val = self.env.get('Ans', 0.0)
			# Ans(n) → index if list, else implicit mul handled by caller
			if isinstance(val, list) and self.peek() is not None and self.peek().code == LPAREN:
				self.advance()
				idx = int(self.parse_expr())
				self.eat_if(RPAREN)
				return val[idx - 1]
			return val
		if t.code == RAND:
			if self.peek() is not None and self.peek().code == LPAREN:
				# rand( with seed arg
				self.advance()
				seed = self.parse_expr()
				self.eat_if(RPAREN)
				random.seed(seed)
				return random.random()
			return random.random()
		if t.code == GETKEY:    return self.env.get('_getkey', 0.0)

		# User-defined list prefix: ∟NAME
		if t.code == LIST_PFX:
			name = self._read_name(5)
			val = self.env.get(f'∟{name}', [])
			if self.peek() is not None and self.peek().code == LPAREN:
				self.advance()
				idx = int(self.parse_expr())
				self.eat_if(RPAREN)
				return val[idx - 1]
			return val

		# Real variable (A-Z, θ)
		if _is_real_var(t):
			val = self.env.get(t.text, 0.0)
			return val

		# Built-in list variable (L1–L6) — indexable
		if _is_list_var(t):
			val = self.env.get(t.text, [])
			if self.peek() is not None and self.peek().code == LPAREN:
				self.advance()
				idx = int(self.parse_expr())
				self.eat_if(RPAREN)
				return val[idx - 1]
			return val

		# Matrix variable — indexable
		if _is_matrix_var(t):
			val = self.env.get(t.text, [[]])
			if self.peek() is not None and self.peek().code == LPAREN:
				self.advance()
				row = int(self.parse_expr())
				self.expect(COMMA)
				col = int(self.parse_expr())
				self.eat_if(RPAREN)
				return val[row - 1][col - 1]
			return val

		# String variable
		if _is_string_var(t):
			return self.env.get(t.text, "")

		# Stat / window variables (read-only in expressions)
		if _is_stat_var(t):
			return self.env.get(t.text, 0.0)

		# prgm — sub-program call
		if t.code == b'\x5f':
			name = self._read_name(8)
			raise NotImplementedError(f"prgm{name}")

		# Functions that need parser access
		if t.code == SEQ_FUNC:   return self._call_seq()
		if t.code == SIGMA_F:    return self._call_sigma()
		if t.code == NDERIV:     return self._call_nderiv()
		if t.code == FNINT:      return self._call_fnint()

		# General built-in function
		if t.text.endswith('('):
			fn = _BUILTIN.get(t.code)
			if fn is NotImplemented:
				raise NotImplementedError(t.text)
			if fn is None:
				raise ParseError(f"Unknown function: {t.text!r}")
			args = self.parse_args()
			return fn(*args)

		raise ParseError(f"Unexpected token in expression: {t.text!r}")

	# ── Expression parser (Pratt) ──────────────────────────────────────────────

	def parse_expr(self, min_bp: int = 0) -> Value:
		lhs = self.parse_atom()

		while True:
			t = self.peek()
			if t is None:
				break

			# Postfix operators
			if t.code in (SQ, CUBE, INV, FACT, TRANS, RAD, DEG):
				if POSTFIX_BP <= min_bp:
					break
				self.advance()
				if t.code == SQ:    lhs = lhs ** 2
				elif t.code == CUBE: lhs = lhs ** 3
				elif t.code == INV:  lhs = 1 / lhs
				elif t.code == FACT: lhs = _factorial(lhs)
				elif t.code == TRANS:
					lhs = [[lhs[r][c] for r in range(len(lhs))]
						   for c in range(len(lhs[0]))]
				elif t.code == RAD:  lhs = math.radians(lhs)  # °→rad conversion? Actually ʳ means "treat as radians"
				elif t.code == DEG:  lhs = math.radians(lhs)  # degree suffix
				continue

			# Store — parse right side as lvalue, not an expression
			if t.code == STORE:
				if _BP[STORE][0] <= min_bp:
					break
				self.advance()
				target = self.parse_store_target()
				return lhs, target  # caller wraps in StoreResult

			# Explicit infix operators
			bp = _BP.get(t.code)
			if bp is not None:
				left_bp, right_bp = bp
				if left_bp <= min_bp:
					break
				self.advance()
				rhs = self.parse_expr(right_bp)
				lhs = _apply_binop(t.code, lhs, rhs)
				continue

			# Implicit multiplication: if next token can start an atom
			if _can_start_atom(t):
				left_bp, right_bp = IMPL_MUL_BP
				if left_bp <= min_bp:
					break
				rhs = self.parse_expr(right_bp)
				lhs = _mul(lhs, rhs)
				continue

			break

		return lhs

	# ── Store target parser ────────────────────────────────────────────────────

	def parse_store_target(self) -> StoreTarget:
		t = self.advance()

		if t.code == ANS:
			return AnsTarget()

		if _is_real_var(t):
			return RealVarTarget(t.text)

		if _is_list_var(t):
			if self.eat_if(LPAREN):
				idx = self.parse_expr()
				self.eat_if(RPAREN)
				return ListElementTarget(t.code, idx)
			return ListVarTarget(t.code)

		if _is_matrix_var(t):
			if self.eat_if(LPAREN):
				row = self.parse_expr()
				self.expect(COMMA)
				col = self.parse_expr()
				self.eat_if(RPAREN)
				return MatrixElementTarget(t.code, row, col)
			return MatrixVarTarget(t.code)

		if _is_string_var(t):
			return StringVarTarget(t.code)

		if t.code == LIST_PFX:
			name = self._read_name(5)
			if self.eat_if(LPAREN):
				idx = self.parse_expr()
				self.eat_if(RPAREN)
				return ListElementTarget(name, idx)
			return UserListTarget(name)

		raise ParseError(f"Invalid store target: {t.text!r}")

	# ── Label name reader ──────────────────────────────────────────────────────

	def _read_name(self, max_len: int) -> str:
		"""Read up to max_len alphanumeric tokens as a name string."""
		name = ""
		while len(name) < max_len and self.peek() is not None:
			t = self.peek()
			ch = t.text
			if len(ch) == 1 and (ch.isalnum() or ch == 'θ'):
				name += ch
				self.advance()
			else:
				break
		if not name:
			raise ParseError("Expected a name")
		return name

	def parse_label_name(self) -> str:
		return self._read_name(2)

	# ── Statement dispatcher ───────────────────────────────────────────────────

	def parse_statement(self):
		t = self.peek()
		if t is None:
			return None

		if t.code == IF:
			self.advance()
			cond = self.parse_expr()
			return IfResult(bool(cond))

		if t.code == WHILE:
			self.advance()
			cond = self.parse_expr()
			return WhileResult(bool(cond))

		if t.code == REPEAT:
			self.advance()
			return RepeatResult()

		if t.code == FOR:
			self.advance()
			var_tok = self.advance()
			if not _is_real_var(var_tok):
				raise ParseError("For: first arg must be a variable")
			self.expect(COMMA)
			start = self.parse_expr()
			self.expect(COMMA)
			end   = self.parse_expr()
			step  = 1.0
			if self.eat_if(COMMA):
				step = self.parse_expr()
			self.eat_if(RPAREN)
			return ForResult(var_tok.text, float(start), float(end), float(step))

		if t.code == END:      self.advance(); return EndResult()
		if t.code == RETURN:   self.advance(); return ReturnResult()
		if t.code == STOP_CMD: self.advance(); return StopResult()
		if t.code == CLRHOME:  self.advance(); return ClrHomeResult()
		if t.code == DISPGRAPH:self.advance(); return DispGraphResult()
		if t.code == THEN:     self.advance(); return None  # handled by executor
		if t.code == ELSE:     self.advance(); return None

		if t.code == LBL:
			self.advance()
			return LblResult(self.parse_label_name())

		if t.code == GOTO:
			self.advance()
			return GotoResult(self.parse_label_name())

		if t.code == PAUSE:
			self.advance()
			val = self.parse_expr() if not self.at_end() else None
			return PauseResult(val)

		if t.code == DISP:
			self.advance()
			values = []
			if not self.at_end():
				values.append(self.parse_expr())
				while self.eat_if(COMMA):
					values.append(self.parse_expr())
			return DispResult(values)

		if t.code == INPUT:
			self.advance()
			# Input "prompt",Var  OR  Input Var
			if self.peek() is not None and self.peek().code == QUOT:
				self.advance()
				prompt = self.parse_string_literal()
				self.expect(COMMA)
			else:
				prompt = None
			target = self.parse_store_target()
			return InputResult(prompt, target)

		if t.code == PROMPT:
			self.advance()
			targets = [self.parse_store_target()]
			while self.eat_if(COMMA):
				targets.append(self.parse_store_target())
			return PromptResult(targets)

		if t.code == OUTPUT:
			self.advance()
			row = self.parse_expr()
			self.expect(COMMA)
			col = self.parse_expr()
			self.expect(COMMA)
			val = self.parse_expr()
			self.eat_if(RPAREN)
			return OutputResult(row, col, val)

		if t.code == MENU:
			return self._parse_menu()

		if t.code == IS_GT:
			self.advance()
			var_tok = self.advance()
			self.expect(COMMA)
			limit = self.parse_expr()
			self.eat_if(RPAREN)
			name = var_tok.text
			self.env[name] = self.env.get(name, 0.0) + 1
			return IfResult(self.env[name] > limit)

		if t.code == DS_LT:
			self.advance()
			var_tok = self.advance()
			self.expect(COMMA)
			limit = self.parse_expr()
			self.eat_if(RPAREN)
			name = var_tok.text
			self.env[name] = self.env.get(name, 0.0) - 1
			return IfResult(self.env[name] < limit)

		# Expression statement (possibly with →)
		result = self.parse_expr()
		if isinstance(result, tuple):
			value, target = result
			return StoreResult(value, target)
		return ExprResult(result)

	def _parse_menu(self) -> MenuResult:
		self.advance()  # consume Menu(
		title = self.parse_expr()
		options = []
		while self.eat_if(COMMA):
			name = self.parse_expr()
			self.expect(COMMA)
			label = self.parse_label_name()
			options.append((name, label))
		self.eat_if(RPAREN)
		return MenuResult(title, options)


# ── Operator helpers ───────────────────────────────────────────────────────────

def _mul(a, b):
	if isinstance(a, list) and not isinstance(a[0], list):
		return [x * b for x in a]
	if isinstance(b, list) and not isinstance(b[0], list):
		return [a * x for x in b]
	return a * b

def _apply_binop(code: bytes, lhs, rhs) -> Value:
	if code == ADD:    return lhs + rhs
	if code == SUB:    return lhs - rhs
	if code == MUL:    return _mul(lhs, rhs)
	if code == DIV:    return lhs / rhs
	if code == POW:    return lhs ** rhs
	if code == XROOT:  return rhs ** (1 / lhs)
	if code == NPR:    return _npr(lhs, rhs)
	if code == NCR:    return _ncr(lhs, rhs)
	if code == EQ_OP:  return 1.0 if lhs == rhs else 0.0
	if code == NE_OP:  return 1.0 if lhs != rhs else 0.0
	if code == LT_OP:  return 1.0 if lhs <  rhs else 0.0
	if code == GT_OP:  return 1.0 if lhs >  rhs else 0.0
	if code == LE_OP:  return 1.0 if lhs <= rhs else 0.0
	if code == GE_OP:  return 1.0 if lhs >= rhs else 0.0
	if code == AND_OP: return 1.0 if lhs and rhs else 0.0
	if code == OR_OP:  return 1.0 if lhs or  rhs else 0.0
	if code == XOR_OP: return 1.0 if bool(lhs) != bool(rhs) else 0.0
	raise ParseError(f"Unknown operator: {code!r}")


# ── Public helpers ─────────────────────────────────────────────────────────────

def eval_tokens(tokens: list[Token], env: dict) -> Value:
	"""Evaluate a token list as an expression (used for seq, Σ, nDeriv, fnInt)."""
	return Parser(tokens, env).parse_expr()


def parse_line(tokens: list[Token], env: dict):
	"""Parse and evaluate a single program line. Returns a statement result."""
	return Parser(tokens, env).parse_statement()
