"""Implementations of TI-Basic operators, functions, and commands.

Each dict maps a token code (bytes) to a callable.  The parser routes
through Token.binary_op / .unary_op / .call / .execute, which look up
their implementation here.

Special functions (_call_seq etc.) receive the live Parser object so they
can drive token consumption themselves.  They call parser.eval_token_list()
to re-evaluate a captured token sublist — avoiding any circular imports.
"""
from __future__ import annotations
import math, cmath, random, datetime as _dt
from typing import Any, Callable

from tokens import COMMA, R_PAREN, QUOTE
from results import (
	IfResult, WhileResult, RepeatResult, ForResult, EndResult,
	LblResult, GotoResult, ReturnResult, StopResult, PauseResult,
	DispResult, InputResult, PromptResult, OutputResult, MenuResult,
	ClrHomeResult, DispGraphResult, AnsTarget,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _factorial(n: float) -> float:
	if n < 0 or n != int(n):
		raise ValueError("Argument to ! must be a non-negative integer")
	return float(math.factorial(int(n)))

def _ncr(n: float, r: float) -> float:
	return float(math.comb(int(n), int(r)))

def _npr(n: float, r: float) -> float:
	return float(math.perm(int(n), int(r)))

def _ti_int(x: float) -> float:
	"""TI int(): floor for positive, ceiling for negative (truncation toward zero)."""
	return math.floor(x) if x >= 0 else math.ceil(x)

def _fpart(x: float) -> float:
	return x - _ti_int(x)

def _list_mul(a, b):
	if isinstance(a, list) and not isinstance(a[0], list):
		return [x * b for x in a]
	if isinstance(b, list) and not isinstance(b[0], list):
		return [a * x for x in b]
	return a * b

# ── Binary operators ───────────────────────────────────────────────────────────

BINARY_OPS: dict[bytes, Callable[[Any, Any], Any]] = {
	b'\x70': lambda a, b: a + b,                                    # +
	b'\x71': lambda a, b: a - b,                                    # -
	b'\x82': _list_mul,                                             # *
	b'\x83': lambda a, b: a / b,                                    # /
	b'\x3b': lambda a, b: a * (10 ** b),                            # ᴇ (scientific notation)
	b'\xf0': lambda a, b: a ** b,                                   # ^
	b'\xf1': lambda a, b: b ** (1 / a),                             # ×√ (xroot)
	b'\x94': _npr,                                                  # nPr
	b'\x95': _ncr,                                                  # nCr
	b'\x6a': lambda a, b: 1.0 if a == b else 0.0,                  # =
	b'\x6b': lambda a, b: 1.0 if a <  b else 0.0,                  # <
	b'\x6c': lambda a, b: 1.0 if a >  b else 0.0,                  # >
	b'\x6d': lambda a, b: 1.0 if a <= b else 0.0,                  # ≤
	b'\x6e': lambda a, b: 1.0 if a >= b else 0.0,                  # ≥
	b'\x6f': lambda a, b: 1.0 if a != b else 0.0,                  # ≠
	b'\x40': lambda a, b: 1.0 if a and b else 0.0,                 # and
	b'\x3c': lambda a, b: 1.0 if a or  b else 0.0,                 # or
	b'\x3d': lambda a, b: 1.0 if bool(a) != bool(b) else 0.0,     # xor
}

# ── Nullary values (0-arg, no parentheses) ─────────────────────────────────────
# Each entry is (env) -> Value.

NULLARY_CALLS: dict[bytes, Callable] = {
	b'\xac':      lambda env: math.pi,                              # π
	b'\xbb\x31':  lambda env: math.e,                              # e
	b'\x2c':      lambda env: 1j,                                   # 𝑖
	b'\xab':      lambda env: random.random(),                      # rand
	b'\xad':      lambda env: env.get('_getkey', 0.0),             # getKey
	b'\xef\x09':  lambda env: [float(_dt.date.today().year),       # getDate
	              float(_dt.date.today().month), float(_dt.date.today().day)],
	b'\xef\x0a':  lambda env: [float(_dt.datetime.now().hour),     # getTime
	              float(_dt.datetime.now().minute), float(_dt.datetime.now().second)],
	b'\xef\x0b':  lambda env: float(int(_dt.datetime.now().timestamp())),  # startTmr
	b'\xef\x0c':  lambda env: env.get('_dtfmt', 1.0),              # getDtFmt
	b'\xef\x0d':  lambda env: env.get('_tmfmt', 12.0),             # getTmFmt
	b'\xef\x0e':  lambda env: 1.0,                                  # isClockOn
}

# ── Unary / postfix operators ──────────────────────────────────────────────────

def _matrix_transpose(m):
	return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]

UNARY_OPS: dict[bytes, Callable[[Any], Any]] = {
	b'\xb0': lambda x: -x,                  # − (prefix negation)
	b'\x0d': lambda x: x ** 2,              # ²
	b'\x0f': lambda x: x ** 3,              # ³
	b'\x0c': lambda x: 1 / x,              # ⁻¹
	b'\x2d': _factorial,                    # !
	b'\x0e': _matrix_transpose,             # ᵀ
	b'\x0b': math.radians,                  # ° (degree → internal radian)
	b'\x0a': lambda x: x,                   # ʳ (already radians; no-op)
}

# ── Standard functions (parser calls parse_args then applies fn) ───────────────

def _wrap(fn: Callable) -> Callable:
	"""Wrap a plain math function so it takes (parser,) and calls parse_args()."""
	def call(parser):
		args = parser.parse_args()
		return fn(*args)
	return call

def _randint_call(parser):
	args = parser.parse_args()
	a, b = int(args[0]), int(args[1])
	n = int(args[2]) if len(args) > 2 else 1
	results = [float(random.randint(a, b)) for _ in range(n)]
	return results[0] if n == 1 else results

# ── Special functions (drive their own arg parsing via the parser) ─────────────

def _call_seq(parser) -> list:
	thunk = parser.grab_until_comma()
	parser.expect(COMMA)
	var_tok   = parser.advance()
	if not var_tok.is_real_var():
		raise ValueError("seq: second arg must be a variable")
	var = var_tok.text
	parser.expect(COMMA)
	start = parser.parse_expr()
	parser.expect(COMMA)
	end   = parser.parse_expr()
	step  = 1.0
	if parser.eat_if(COMMA):
		step = parser.parse_expr()
	parser.eat_if(R_PAREN)

	saved = parser.env.get(var)
	result = []
	n = start
	while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
		parser.env[var] = n
		result.append(thunk.eval())
		n += step
	parser.env[var] = saved if saved is not None else parser.env.pop(var, None)
	return result

def _call_sigma(parser) -> float:
	thunk = parser.grab_until_comma()
	parser.expect(COMMA)
	var_tok   = parser.advance()
	if not var_tok.is_real_var():
		raise ValueError("Σ: second arg must be a variable")
	var = var_tok.text
	parser.expect(COMMA)
	start = int(parser.parse_expr())
	parser.expect(COMMA)
	end   = int(parser.parse_expr())
	parser.eat_if(R_PAREN)

	saved = parser.env.get(var)
	total = 0.0
	for i in range(start, end + 1):
		parser.env[var] = float(i)
		total += thunk.eval()
	parser.env[var] = saved if saved is not None else parser.env.pop(var, None)
	return total

def _call_nderiv(parser) -> float:
	thunk = parser.grab_until_comma()
	parser.expect(COMMA)
	var = parser.advance().text
	parser.expect(COMMA)
	val = parser.parse_expr()
	h   = 1e-5
	if parser.eat_if(COMMA):
		h = parser.parse_expr()
	parser.eat_if(R_PAREN)

	saved = parser.env.get(var)
	parser.env[var] = val + h;  fwd = thunk.eval()
	parser.env[var] = val - h;  bwd = thunk.eval()
	parser.env[var] = saved if saved is not None else parser.env.pop(var, None)
	return (fwd - bwd) / (2 * h)

def _call_fnint(parser) -> float:
	thunk = parser.grab_until_comma()
	parser.expect(COMMA)
	var = parser.advance().text
	parser.expect(COMMA)
	a = parser.parse_expr()
	parser.expect(COMMA)
	b = parser.parse_expr()
	parser.eat_if(R_PAREN)

	saved = parser.env.get(var)
	n = 1000
	h = (b - a) / n
	def f(x):
		parser.env[var] = x
		return thunk.eval()
	total = f(a) + f(b)
	for i in range(1, n):
		total += (4 if i % 2 else 2) * f(a + i * h)
	parser.env[var] = saved if saved is not None else parser.env.pop(var, None)
	return total * h / 3

# ── Function dispatch ──────────────────────────────────────────────────────────
# Every entry is (parser) -> Value.  Use _wrap() for plain math functions.

FUNCTION_CALLS: dict[bytes, Callable] = {
	# Special (own arg parsing)
	b'\x23':      _call_seq,                                                  # seq(
	b'\xef\x33':  _call_sigma,                                                # Σ(
	b'\x25':      _call_nderiv,                                               # nDeriv(
	b'\x24':      _call_fnint,                                                # fnInt(

	# Arithmetic / rounding
	b'\x12': _wrap(lambda a, b=0: round(a, int(b))),                          # round(
	b'\xb1': _wrap(_ti_int),                                                  # int(
	b'\xb2': _wrap(abs),                                                      # abs(
	b'\xb9': _wrap(math.trunc),                                               # iPart(
	b'\xba': _wrap(_fpart),                                                   # fPart(

	# Powers / roots
	b'\xbc': _wrap(lambda a: cmath.sqrt(a) if a < 0 else math.sqrt(a)),       # √(
	b'\xbd': _wrap(lambda a: -(-a)**(1/3) if a < 0 else a**(1/3)),            # ³√(
	b'\xbf': _wrap(cmath.exp),                                                # e^(
	b'\xc1': _wrap(lambda a: 10**a),                                           # 10^(

	# Logarithms
	b'\xbe': _wrap(cmath.log),                                                # ln(
	b'\xc0': _wrap(cmath.log10),                                              # log(
	b'\xef\x34': _wrap(lambda a, b: math.log(a) / math.log(b)),              # logBASE(

	# Trig
	b'\xc2': _wrap(math.sin),   b'\xc3': _wrap(math.asin),                   # sin( sin⁻¹(
	b'\xc4': _wrap(math.cos),   b'\xc5': _wrap(math.acos),                   # cos( cos⁻¹(
	b'\xc6': _wrap(math.tan),   b'\xc7': _wrap(math.atan),                   # tan( tan⁻¹(
	b'\xc8': _wrap(math.sinh),  b'\xc9': _wrap(math.asinh),                  # sinh( sinh⁻¹(
	b'\xca': _wrap(math.cosh),  b'\xcb': _wrap(math.acosh),                  # cosh( cosh⁻¹(
	b'\xcc': _wrap(math.tanh),  b'\xcd': _wrap(math.atanh),                  # tanh( tanh⁻¹(

	# List operations
	b'\xb5': _wrap(lambda a: float(len(a)) if isinstance(a, list)            # dim(
		else [float(len(a)), float(len(a[0]))]),
	b'\xb6': _wrap(lambda a: sum(a)),                                         # sum(
	b'\xb7': _wrap(math.prod),                                                # prod(
	b'\x19': _wrap(lambda *a: max(a) if len(a) > 1 else max(a[0])),           # max(
	b'\x1a': _wrap(lambda *a: min(a) if len(a) > 1 else min(a[0])),           # min(
	b'\x1f': _wrap(lambda a, b=None: float(sorted(a)[len(a)//2])),            # median(
	b'\x21': _wrap(lambda a, b=None: (                                        # mean(
		sum(a)/len(a) if b is None else sum(x*w for x,w in zip(a,b))/sum(b))),
	b'\xbb\x29': _wrap(lambda a: [sum(a[:i+1]) for i in range(len(a))]),      # cumSum(
	b'\xbb\x2c': _wrap(lambda a: [a[i+1]-a[i] for i in range(len(a)-1)]),    # ΔList(
	b'\xef\x35': _wrap(lambda a, b, n=None: random.sample(              # randIntNoRep(
		range(int(a), int(b)+1), int(b-a+1) if n is None else int(n))),

	# Boolean
	b'\xb8': _wrap(lambda a: float(not a)),                                   # not(

	# String
	b'\xbb\x0c': _wrap(lambda s, start, length:                               # sub(
		s[int(start)-1 : int(start)-1+int(length)]),
	b'\xbb\x0f': _wrap(lambda s, sub:                                         # inString(
		float(s.find(sub) + 1) if sub in s else 0.0),
	b'\xbb\x2b': _wrap(lambda s: float(len(s))),                              # length(
	b'\xbb\x2a': NotImplemented,                                              # expr( — executor provides

	# Complex
	b'\xbb\x25': _wrap(lambda a: complex(a.real, -a.imag)),                   # conj(
	b'\xbb\x26': _wrap(lambda a: a.real),                                     # real(
	b'\xbb\x27': _wrap(lambda a: a.imag),                                     # imag(
	b'\xbb\x28': _wrap(cmath.phase),                                          # angle(

	# Random
	b'\xbb\x0a': _randint_call,                                               # randInt(
	b'\xbb\x1f': _wrap(lambda mu, sigma: random.gauss(mu, sigma)),            # randNorm(

	# Number theory
	b'\xbb\x08': _wrap(math.lcm),                                             # lcm(
	b'\xbb\x09': _wrap(math.gcd),                                             # gcd(
	b'\xef\x32': _wrap(lambda a, b: float(int(a) % int(b))),                  # remainder(

	# Matrix
	b'\xb3': _wrap(lambda m: float(sum(                                       # det(
		m[i][i] for i in range(len(m))))),   # placeholder — real det is harder
	b'\xb4': _wrap(lambda n: [[1.0 if r==c else 0.0                           # identity(
		for c in range(int(n))] for r in range(int(n))]),

	# Clock / system (NotImplemented — executor provides runtime values)
	b'\xef\x02': NotImplemented,   # checkTmr(
	b'\xef\x05': NotImplemented,   # timeCnv(
	b'\xef\x06': NotImplemented,   # dayOfWk(
	b'\xef\x07': NotImplemented,   # getDtStr(
	b'\xef\x08': NotImplemented,   # getTmStr(
}

# ── Command dispatch ───────────────────────────────────────────────────────────
# Every entry is (parser) -> StatementResult (or None for no-op tokens).

def _exec_if(parser):
	cond = parser.parse_expr()
	return IfResult(bool(cond))

def _exec_while(parser):
	cond = parser.parse_expr()
	return WhileResult(bool(cond))

def _exec_repeat(parser):
	return RepeatResult()

def _exec_for(parser):
	var_tok = parser.advance()
	if not var_tok.is_real_var():
		raise ValueError("For: first arg must be a variable")
	parser.expect(COMMA)
	start = parser.parse_expr()
	parser.expect(COMMA)
	end   = parser.parse_expr()
	step  = 1.0
	if parser.eat_if(COMMA):
		step = parser.parse_expr()
	parser.eat_if(R_PAREN)
	return ForResult(var_tok.text, float(start), float(end), float(step))

def _exec_lbl(parser):
	return LblResult(parser.parse_label_name())

def _exec_goto(parser):
	return GotoResult(parser.parse_label_name())

def _exec_pause(parser):
	val = parser.parse_expr() if not parser.at_end() else None
	return PauseResult(val)

def _exec_disp(parser):
	values = []
	if not parser.at_end():
		values.append(parser.parse_expr())
		while parser.eat_if(COMMA):
			values.append(parser.parse_expr())
	return DispResult(values)

def _exec_input(parser):
	if parser.peek() is QUOTE:
		parser.advance()
		prompt = parser.parse_string_literal()
		parser.expect(COMMA)
	else:
		prompt = None
	target = parser.parse_store_target()
	return InputResult(prompt, target)

def _exec_prompt(parser):
	targets = [parser.parse_store_target()]
	while parser.eat_if(COMMA):
		targets.append(parser.parse_store_target())
	return PromptResult(targets)

def _exec_output(parser):
	row = parser.parse_expr()
	parser.expect(COMMA)
	col = parser.parse_expr()
	parser.expect(COMMA)
	val = parser.parse_expr()
	parser.eat_if(R_PAREN)
	return OutputResult(row, col, val)

def _exec_menu(parser):
	title = parser.parse_expr()
	options = []
	while parser.eat_if(COMMA):
		name  = parser.parse_expr()
		parser.expect(COMMA)
		label = parser.parse_label_name()
		options.append((name, label))
	parser.eat_if(R_PAREN)
	return MenuResult(title, options)

def _exec_is_gt(parser):
	var_tok = parser.advance()
	parser.expect(COMMA)
	limit = parser.parse_expr()
	parser.eat_if(R_PAREN)
	name = var_tok.text
	parser.env[name] = parser.env.get(name, 0.0) + 1
	return IfResult(parser.env[name] > limit)

def _exec_ds_lt(parser):
	var_tok = parser.advance()
	parser.expect(COMMA)
	limit = parser.parse_expr()
	parser.eat_if(R_PAREN)
	name = var_tok.text
	parser.env[name] = parser.env.get(name, 0.0) - 1
	return IfResult(parser.env[name] < limit)

EXEC_CALLS: dict[bytes, Callable] = {
	b'\xce': _exec_if,          # If
	b'\xd1': _exec_while,       # While
	b'\xd2': _exec_repeat,      # Repeat
	b'\xd3': _exec_for,         # For(
	b'\xd4': lambda p: EndResult(),    # End
	b'\xd5': lambda p: ReturnResult(), # Return
	b'\xd9': lambda p: StopResult(),   # Stop
	b'\xe1': lambda p: ClrHomeResult(),# ClrHome
	b'\xdf': lambda p: DispGraphResult(), # DispGraph
	b'\xd6': _exec_lbl,         # Lbl
	b'\xd7': _exec_goto,        # Goto
	b'\xd8': _exec_pause,       # Pause
	b'\xde': _exec_disp,        # Disp
	b'\xdc': _exec_input,       # Input
	b'\xdd': _exec_prompt,      # Prompt
	b'\xe0': _exec_output,      # Output(
	b'\xe6': _exec_menu,        # Menu(
	b'\xda': _exec_is_gt,       # IS>(
	b'\xdb': _exec_ds_lt,       # DS<(
	b'\xcf': lambda p: None,    # Then — executor handles
	b'\xd0': lambda p: None,    # Else — executor handles
}
