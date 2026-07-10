from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from enum import IntFlag, auto

from errors import TiSyntaxError, UndefinedError


# CHARSET: TI-83+ byte -> Unicode character (None = undefined slot).
#
# This is the token layer's source of truth for decoding a token's display bytes
# into a human-readable string (Token.text), used purely for debugging/printing.  The
# characters are convenient renderings, not a canonical mapping: several have no
# faithful Unicode equivalent (𝐅, 𝟑, ẍ, ṕ, ...) and a couple of font glyphs are
# duplicated.  Byte D6 decodes to '\n' (the newline token) rather than its ↵ glyph
# so program text round-trips with real line breaks.


CHARSET: list[str | None] = [
#	0		1		2		3		4		5		6		7		8		9		A		B		C		D		E		F
	' ',	'𝑛',	'𝑢',	'𝑣',	'𝑤',	'►',	'🡅',	'🡇',	'∫',	'×',	'▫',	'﹢',	'·',	'ₜ',		'³',	'𝟊',	# 0
	'√',	'¹',	'²',	'∠',	'°',	'ʳ',	'ᵀ',	'≤',	'≠',	'≥',	'¯',	'ᴇ',	'→',	'⑩',	'↑',	'↓',	# 1
	' ',	'!',	'"',	'#',	'⁴',		'%',	'&',	"'",	'(',	')',	'*',	'+',	',',	'-',	'.',	'/',	# 2
	'0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	'8',	'9',	':',	';',	'<',	'=',	'>',	'?',	# 3
	'@',	'A',	'B',	'C',	'D',	'E',	'F',	'G',	'H',	'I',	'J',	'K',	'L',	'M',	'N',	'O',	# 4
	'P',	'Q',	'R',	'S',	'T',	'U',	'V',	'W',	'X',	'Y',	'Z',	'θ',	'\\',	']',	'^',	'_',	# 5
	'`',	'a',	'b',	'c',	'd',	'e',	'f',	'g',	'h',	'i',	'j',	'k',	'l',	'm',	'n',	'o',	# 6
	'p',	'q',	'r',	's',	't',	'u',	'v',	'w',	'x',	'y',	'z',	'{',	'|',	'}',	'~',	'≛',	# 7
	'₀',		'₁',		'₂',		'₃',		'₄',		'₅',		'₆',		'₇',		'₈',		'₉',		'Á',	'À',	'Â',	'Ä',	'á',	'à',	# 8
	'â',	'ä',	'É',	'È',	'Ê',	'Ë',	'é',	'è',	'ê',	'ë',	'Í',	'Ì',	'Î',	'Ï',	'í',	'ì',	# 9
	'î',	'ï',	'Ó',	'Ò',	'Ô',	'Ö',	'ó',	'ò',	'ô',	'ö',	'Ú',	'Ù',	'Û',	'Ü',	'ú',	'ù',	# A
	'û',	'ü',	'Ç',	'ç',	'Ñ',	'ñ',	'´',	'ˋ',	'¨',	'¿',	'¡',	'α',	'β',	'γ',	'Δ',	'δ',	# B
	'ε',	'[',	'λ',	'μ',	'π',	'ρ',	'Σ',	'σ',	'τ',	'φ',	'Ω',	'x̄',	'ȳ',	'ˣ',	'…',	'◄',	# C
	None,	None,	None,	None,	None,	'³',	'\n',	'𝑖',		'p̂',	'χ',	'𝐅',	'𝑒',		'ʟ',	'𝐍',	None,	None,	# D
	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# E
	None,	None,	'$',	None,	'ß',	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# F
]

def decode(display: bytes) -> str:
	"""Render display bytes as a human-readable string.

	The token layer's *debug/convenience* decoder — behind Token.text, TiString's
	str(), and the screen models' str() rendering.  A frontend that actually paints
	the screen owns its own, richer charset (see terminal.py); this one only has to
	round-trip the ASCII a program types and read it back."""
	return ''.join(CHARSET[b] for b in display)


# ── Named token codes ─────────────────────────────────────────────────────────
# Codes for tokens referenced directly by the parser/program (the only tokens
# that need a name outside catalog).  catalog passes these to token(); the parser
# and program compare t.code against them.  Codes are ints — one-byte tokens are
# 0x00–0xFF, two-byte tokens are 0x0100–0xFFFF — and this module is the single
# source of truth for each value.

EOF_CODE = -1  # sentinel for the end-of-input token (matches no real code)

# structural / punctuation
STORE       = 0x04
L_BRACKET   = 0x06
R_BRACKET   = 0x07
L_BRACE     = 0x08
R_BRACE     = 0x09
RAD         = 0x0A
DEG         = 0x0B
L_PAREN     = 0x10
R_PAREN     = 0x11
QUOTE       = 0x2A
COMMA       = 0x2B
DOT         = 0x3A
SCI_E       = 0x3B
COLON       = 0x3E
NEWLINE     = 0x3F
ANS         = 0x72
APOS        = 0xAE
NEG         = 0xB0
LIST_PREFIX = 0xEB

# store targets that need special handling
RAND        = 0xAB
DIM         = 0xB5

# control flow
IF          = 0xCE
THEN        = 0xCF
ELSE        = 0xD0
WHILE       = 0xD1
REPEAT      = 0xD2
FOR         = 0xD3
END         = 0xD4
LBL         = 0xD6


class Flag(IntFlag):
	"""A token's role and classification, declared explicitly in the catalog rather than
	inferred from its code range.  Roles (FUNCTION/COMMAND/INFIX/POSTFIX) drive parsing;
	the variable kinds (NUMERIC…EQUATION) classify assignable targets and VARIABLE is
	their union — what any_var / DelVar accept.  ATOM_START marks structural tokens that
	open an expression atom but carry no other role ('(', '{', '"', …); INVOKABLE marks
	accessor-type tokens that take a trailing '(arg)' (lists, matrices, equations).
	"""
	ATOM_START = auto()
	INVOKABLE  = auto()

	FUNCTION = auto()
	COMMAND  = auto()
	INFIX    = auto()
	POSTFIX  = auto()
	DIGIT    = auto()

	NUMERIC  = auto()
	LIST     = auto()
	MATRIX   = auto()
	STRING   = auto()
	SEQUENCE = auto()
	EQUATION = auto()
	STAT_VAR = auto()

	WINDOW_VAR = auto()

	PIC      = auto()
	GDB      = auto()

	VARIABLE = NUMERIC | LIST | MATRIX | STRING | SEQUENCE | EQUATION


class Token:
	"""A token in a program.  The base class is for *inert* tokens — punctuation, literals,
	mode keywords; behavior-carrying tokens are subclasses (FunctionToken, CommandToken,
	OperatorToken, the Accessor variable tokens …) defined in catalog.py, where they can
	import the operator/command modules that tokenbase (a leaf module) cannot.

	The parser drives tokens through the polymorphic hooks below (parse_prefix /
	parse_infix / apply_postfix / …) rather than inspecting nullable callable fields, so
	each kind owns its own parse behavior next to where it's wired up.  `flags` records
	the token's role/classification (see Flag); `char` is the single Unicode character it
	stands for, when one faithfully exists (None otherwise) — used by frontends, not the
	token spec.
	"""

	def __init__(self, code: int, display: bytes, flags: Flag = Flag(0), char: str | None = None):
		self.code = code
		self.display = display
		self.flags = flags
		self.char = char

	@property
	def size(self) -> int:
		return 1 if self.code < 0x100 else 2

	def __eq__(self, other):
		"""Tokens are value objects: equal iff same code.  Codes are unique in the
		catalog and the calculator itself compares tokens by their byte value, so a
		freshly constructed StringToken(3) IS Str4 — the catalog builds one instance
		of each, but identity is a convenience, not an invariant."""
		if isinstance(other, Token):
			return self.code == other.code
		return NotImplemented

	def __hash__(self):
		return hash(self.code)

	@property
	def text(self) -> str:
		"""Human-readable rendering of the display bytes (debugging/printing only)."""
		return decode(self.display)

	def code_to_bytes(self) -> bytes:
		"""Encode this token's code as the 1 or 2 bytes stored in a .8xp program."""
		return self.code.to_bytes(1 + (self.code > 0xFF))

	def is_digit(self) -> bool:
		return bool(self.flags & Flag.DIGIT)

	def is_numeric_var(self) -> bool:
		return bool(self.flags & Flag.NUMERIC)

	def is_list_var(self) -> bool:
		return bool(self.flags & Flag.LIST)

	def is_list_start(self):
		return self.code == LIST_PREFIX or self.is_list_var()

	def is_matrix_var(self) -> bool:
		return bool(self.flags & Flag.MATRIX)

	def is_sequence_var(self) -> bool:
		return bool(self.flags & Flag.SEQUENCE)

	def is_equation_var(self) -> bool:
		return bool(self.flags & Flag.EQUATION)

	def is_string_var(self) -> bool:
		return bool(self.flags & Flag.STRING)

	def is_stat_var(self) -> bool:
		return bool(self.flags & Flag.STAT_VAR)

	def is_window_var(self) -> bool:
		return bool(self.flags & Flag.WINDOW_VAR)

	def is_name_char(self):
		return self.is_numeric_var() or self.is_digit()

	def can_start_atom(self) -> bool:
		return bool(self.flags & (Flag.ATOM_START | Flag.FUNCTION | Flag.DIGIT | Flag.VARIABLE))

	def is_postfix(self) -> bool:
		return bool(self.flags & Flag.POSTFIX)

	def opens_paren_group(self) -> bool:
		return bool(self.flags & Flag.FUNCTION) or self.code == L_PAREN

	# ── Parse hooks (overridden by the behavior-carrying subclasses in catalog) ──

	def parse_prefix(self, parser):
		"""nud — produce this token's value when it leads an atom.  An Accessor reads its
		value and FunctionToken invokes a call; structural/literal atoms (numbers, "…",
		{…}, …) are handled by the parser directly.  Anything else can't start an
		expression."""
		raise TiSyntaxError(f"Unexpected token in expression: {self}")

	def apply_postfix(self, value):
		"""Apply this postfix operator to `value`.  Only reached when is_postfix() is True."""
		raise TiSyntaxError(f"{self} is not a postfix operator")

	def infix_bp(self):
		"""Left binding power if this token is an infix binary operator, else None.
		The Pratt loop uses it to decide whether to bind; OperatorToken overrides."""
		return None

	def parse_infix(self, parser, lhs):
		"""led — combine `lhs` via this infix operator.  Only reached when infix_bp()
		returns a value (i.e. for OperatorToken)."""
		raise TiSyntaxError(f"{self} is not an infix operator")

	def __repr__(self):
		return f"0x{self.code:0{4 if self.code > 0xFF else 2}X}:{self.text!r}"


class Reference:
	"""An accessor-token bound to an environment — what commands receive when they take a
	variable rather than its value.  Exposes a uniform mutable-variable surface so
	consumers (For/fnInt/Input/DelVar/…) work the same for every symbol kind.  `name`
	forwards to the accessor's source spelling as display tokens (what Prompt echoes)."""

	__slots__ = ('env', 'accessor')

	def __init__(self, env, accessor):
		self.env = env
		self.accessor = accessor

	def resolve(self):
		return self.accessor.resolve(self.env)

	def store(self, value):
		self.accessor.store(self.env, value)

	def delete(self):
		self.accessor.delete(self.env)

	def get(self):
		return self.accessor.get(self.env)

	def set(self, value):
		self.accessor.set(self.env, value)

	@contextmanager
	def scoped(self):
		"""Save the raw value, run the block, restore it — for temporarily binding a
		variable while evaluating a sub-expression (fnInt, solve, Σ, …)."""
		saved = self.accessor.get(self.env)
		try:
			yield
		finally:
			self.accessor.set(self.env, saved)

	def __repr__(self):
		return f"Reference({self.accessor!r})"


class Accessor(ABC):
	"""Protocol for environment-bound read/write targets — variables, constants, window
	settings, user lists.  resolve/store/invoke take env as a parameter (accessors are
	stateless singletons or lightweight named objects).  `get`/`set` are the raw slot
	accessors; `resolve`/`store` add auto-init and type checks.

	Concrete subclasses are either VariableToken (a catalog token that is also its own
	accessor) or a plain Accessor (e.g. UserList) built synthetically by the parser.
	"""

	def get(self, env):
		"""Raw slot read — no auto-init (may return None).  Prefer resolve()."""
		raise NotImplementedError(f"{type(self).__name__} does not support get")

	def set(self, env, value):
		"""Raw slot write — no validation, copy, or side-effects.  Prefer store() for a
		calculator Store; use set() only for direct installs (e.g. loading a file)."""
		raise NotImplementedError(f"{type(self).__name__} does not support set")

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			raise UndefinedError(f"{self} is not defined")
		return value

	def store(self, env, value):
		raise TiSyntaxError(f"Cannot store to {self}")

	def is_invokable(self):
		return False

	def invoke(self, arg_parser):
		raise TiSyntaxError(f"Cannot invoke {self}")

	def delete(self, env):
		raise TiSyntaxError(f"Cannot delete {self}")

	def name_tokens(self) -> list:
		"""The variable's source spelling as display tokens — what Prompt echoes and
		what a file's human-readable name decodes from.  Every accessor has one (a
		token spells itself, a named accessor spells its letters)."""
		raise NotImplementedError(f"{type(self).__name__} has no name spelling")

	def reference(self, env) -> Reference:
		return Reference(env, self)


class FileVar(Accessor):
	"""An Accessor with a .8x* file identity — a name field in the variable file's
	VAT entry.  This is a genuine partial capability (window/stat variables, Ans,
	constants have no file format), so it is declared by subclassing rather than by
	a raising stub on Accessor: tifile's `write_accessor` takes a FileVar, and
	mis-filing a non-file variable fails at the type level.

	Subclassing Accessor is not a convenience — a file *is* the serialization of an
	env slot (a load installs its value through the accessor's raw `set`), so every
	FileVar is necessarily an Accessor.  `name_bytes` returns only the *meaningful*
	bytes of the name; padding to the VAT's fixed 8-byte field is the file format's
	business and happens in tifile's name codec (write_accessor), not here."""

	@abstractmethod
	def name_bytes(self) -> bytes:
		"""The identifying bytes of this variable's file name field, unpadded."""


class TokenFileVar(FileVar):
	"""FileVar for token-named variables (A–Z/θ, L1–L6, [A]–[J], Str/Y/Pic vars):
	the name field is simply the token's code bytes."""

	def name_bytes(self) -> bytes:
		return self.code_to_bytes()


class Deletable:
	"""Mixin for accessors whose DelVar clears the slot to its undefined state — None.
	List it before VariableToken in the bases so this delete overrides the base's raising
	one (e.g. `class StringToken(Deletable, VariableToken)`)."""

	def delete(self, env):
		self.set(env, None)


class VariableToken(Token, Accessor):
	"""A catalog token that is also its own accessor — a variable like A, L1, [A], Y1.

	Inherits Token for code/display/flags/parse hooks and Accessor for resolve/store/
	invoke.  Token and Accessor share no method names, so the base order carries no
	behavior; the overrides below do the wiring:

	  * parse_prefix — Token's default raises (inert tokens can't lead an expression);
	    a variable token always routes through read_accessor.
	  * can_start_atom — hard True rather than Token's flag mask, because some variable
	    tokens carry flags outside the VARIABLE union (window/stat/TVM vars, constants
	    like π) yet still lead expressions and implicitly multiply (2π).
	  * is_invokable — flag-based (Flag.INVOKABLE), replacing Accessor's False.
	"""

	def is_invokable(self) -> bool:
		return bool(self.flags & Flag.INVOKABLE)

	def parse_prefix(self, parser):
		return parser.read_accessor(self)

	def can_start_atom(self) -> bool:
		return True

	def name_tokens(self) -> list:
		return [self]   # a variable token spells itself


class _EofToken(Token):
	"""Sentinel returned by Parser.peek() at end of input."""
	
	@property
	def text(self):
		raise ValueError

	def code_to_bytes(self):
		raise ValueError
	
	def __repr__(self):
		return '<EOF>'


EOF_TOKEN = _EofToken(EOF_CODE, b'')
