from dataclasses import dataclass
from collections.abc import Callable
from enum import IntFlag, auto

from errors import TiSyntaxError


# _CHARSET: TI-83+ byte -> Unicode character (None = undefined slot).
#
# This is the private source of truth for decoding a token's display bytes into a
# human-readable string (Token.text), used purely for debugging/printing.  The
# characters are convenient renderings, not a canonical mapping: several have no
# faithful Unicode equivalent (𝐅, 𝟑, ẍ, ṕ, ...) and a couple of font glyphs are
# duplicated.  Byte D6 decodes to '\n' (the newline token) rather than its ↵ glyph
# so program text round-trips with real line breaks.


_CHARSET: list[str | None] = [
#	0		1		2		3		4		5		6		7		8		9		A		B		C		D		E		F
	' ',	'𝑛',	'𝑢',	'𝑣',	'𝑤',	'►',	'🡅',	'🡇',	'∫',	'×',	'▫',	'﹢',	'·',	'ₜ',		'𝟑',	'𝟊',	# 0
	'√',	'¹',	'²',	'∠',	'°',	'ʳ',	'ᵀ',	'≤',	'≠',	'≥',	'⁻',		'ᴇ',	'→',	'⑽',	'↑',	'↓',	# 1
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
	'ε',	'[',	'λ',	'μ',	'π',	'ρ',	'Σ',	'σ',	'τ',	'φ',	'Ω',	'ẍ',	'ȳ',	'ˣ',	'…',	'◄',	# C
	None,	None,	None,	None,	None,	'³',	'\n',	'𝑖',		'ṕ',	'χ',	'𝐅',	'𝑒',		'ᴸ',	'𝐍',	'⸩',		'🡆',	# D
	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# E
	None,	None,	'$',	None,	'ß',	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# F
]

def decode(display: bytes) -> str:
	"""Render display bytes as a human-readable string (undefined bytes as \\xNN)."""
	return ''.join(_CHARSET[b] for b in display)


# _ENCODE: Unicode character -> TI display byte, the inverse of _CHARSET, built
# once at import.  Used to turn formatted text (number strings from tiformat) into
# the display bytes the home screen stores.  Where two bytes share a glyph the
# later one wins; that's harmless because the only thing encoded is ASCII, which is
# unambiguous.  Byte 0x00 and the space 0x20 both decode to ' ', so encode(' ')
# resolves to the real space glyph (0x20), leaving 0x00 free as the blank-cell fill.
_ENCODE: dict[str, int] = {ch: b for b, ch in enumerate(_CHARSET) if ch is not None}


def encode(text: str) -> bytes:
	"""Encode a string into TI display bytes (the inverse of decode).

	For the ASCII text produced by number formatting; raises KeyError on a
	character with no charset glyph.
	"""
	return bytes(_ENCODE[c] for c in text)


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
	"""A token's variable classification, declared explicitly in the catalog instead of
	inferred from its code range.  Only assignable variable kinds get a flag; VARIABLE
	is their union — what any_var / DelVar accept.  Lexical categories (digits, name
	chars) stay as range checks, and settings (window/stat vars) aren't variables here.
	"""
	ASCII    = auto()
	EXPR_START = auto()
	
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


@dataclass(slots=True, frozen=True, eq=False)
class Token:
	"""A token in a program.  The plain dataclass is the base for *inert* tokens —
	punctuation, literals, and plain variables; the subclasses that carry behavior
	(FunctionToken, CommandToken, OperatorToken, …) live in catalog.py, where they can
	import the operator/command/accessor modules that titoken (a leaf module) cannot.

	The parser drives tokens through the polymorphic hooks below (parse_prefix /
	parse_infix / run_statement / …) rather than inspecting nullable callable fields,
	so each kind owns its own parse behavior next to where it's wired up.
	"""
	code: int
	display: bytes
	flags = Flag(0)

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
		return bool(self.flag & Flag.LIST)

	def is_list_start(self):
		return self.code == LIST_PREFIX or self.is_list_var()

	def is_matrix_var(self) -> bool:
		return bool(self.flag & Flag.MATRIX)

	def is_sequence_var(self) -> bool:
		return bool(self.flag & Flag.SEQUENCE)

	def is_equation_var(self) -> bool:
		return bool(self.flag & Flag.EQUATION)

	def is_string_var(self) -> bool:
		return bool(self.flag & Flag.STRING)

	def is_stat_var(self) -> bool:
		return bool(self.flag & Flag.STAT_VAR)

	def is_window_var(self) -> bool:
		return bool(self.flag & Flag.WINDOW_VAR)

	def is_name_char(self):
		return self.is_numeric_var() or self.is_digit()

	def can_start_atom(self) -> bool:
		# TODO: Check if flags contain one of EXPR_START | FUNCTION | DIGIT | VARIABLE
		pass

	# ── Parse hooks (overridden by the behavior-carrying subclasses in catalog) ──

	def parse_prefix(self, parser):
		"""nud — produce this token's value when it leads an atom.  VariableToken reads
		its accessor and FunctionToken invokes a call; structural/literal atoms (numbers,
		"…", {…}, …) are handled by the parser directly.  Anything else can't start an
		expression."""
		raise TiSyntaxError(f"Unexpected token in expression: {self}")

	def is_postfix(self) -> bool:
		# TODO: check if POSTFIX in flags
		pass

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

	def opens_paren_group(self) -> bool:
		# TODO: Check if flags contain FUNCTION or if self is L_PAREN
		pass

	def run_statement(self, parser):
		# TODO: Return this logic to Parser
	
		"""Execute this token as the head of a statement.  The default is the expression
		statement: evaluate, then handle a → store, a ►conversion, and Ans.  CommandToken
		overrides to run a command instead."""
		value = parser.parse_expr()
		if parser.eat_if(STORE):
			parser.parse_store(value)
		elif parser.peek().converter is not None:
			value = parser.advance().converter(value)
		parser.env.ans = value
		parser.end_statement()

	def __repr__(self):
		return f"0x{self.code:0{4 if self.code > 0xFF else 2}X}:{self.text!r}"


class _EofToken(Token):
	"""Sentinel returned by Parser.peek() at end of input."""
	
	@property
	def text(self):
		raise ValueError

	def code_to_bytes(self):
		raise ValueError
	
	def __repr__(self):
		return '<EOF>'


EOF_TOKEN = Token(-1, b'')
