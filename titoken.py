from dataclasses import dataclass
from collections.abc import Callable


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
	None,	'𝑛',	'𝑢',	'𝑣',	'𝑤',	'►',	'🡅',	'🡇',	'∫',	'×',	'▫',	'﹢',	'·',	'ₜ',		'𝟑',	'𝟊',	# 0
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


@dataclass(slots=True, frozen=True, eq=False)
class Token:
	code: int                 # token code as stored in a program (1 or 2 bytes packed into an int)
	display: bytes            # large-font byte sequence that renders this token
	bp: tuple[int, int] | None = None
	operator:  Callable | None = None  # (lhs, rhs) -> value
	postfix:   Callable | None = None  # (operand) -> value (prefix or postfix)
	function:  Callable | None = None  # (ArgParser) -> value for function tokens
	command:   Callable | None = None  # (ArgParser) -> None for command tokens
	nullary:   Callable | None = None  # (env) -> value for read-only computed constants
	converter: Callable | None = None  # (value) -> value for ►DMS, ►Dec, ►Frac and others
	variable:  Callable | None = None  # (env) -> Variable for variable tokens

	@property
	def text(self) -> str:
		"""Human-readable rendering of the display bytes (debugging/printing only)."""
		return decode(self.display)

	def code_to_bytes(self) -> bytes:
		"""Encode this token's code as the 1 or 2 bytes stored in a .8xp program."""
		return self.code.to_bytes(1 + (self.code > 0xFF))

	def is_digit(self) -> bool:
		return 0x30 <= self.code <= 0x39

	def is_numeric_var(self) -> bool:
		return 0x41 <= self.code < 0x5C or self.code == 0x6221

	def is_list_var(self) -> bool:
		return 0x5D00 <= self.code <= 0x5DFF

	def is_list_start(self):
		return self.code == 0xEB or 0x5D00 <= self.code <= 0x5DFF

	def is_matrix_var(self) -> bool:
		return 0x5C00 <= self.code <= 0x5CFF

	def is_equation_var(self) -> bool:
		return 0x5E00 <= self.code <= 0x5EFF

	def is_string_var(self) -> bool:
		return 0xAA00 <= self.code <= 0xAAFF

	def is_stat_var(self) -> bool:
		return 0x6200 <= self.code <= 0x62FF

	def is_window_var(self) -> bool:
		return 0x6300 <= self.code <= 0x63FF

	def is_name_char(self):
		return self.is_numeric_var() or self.is_digit()

	def __repr__(self):
		return f"0x{self.code:0{4 if self.code > 0xFF else 2}X}:{self.text!r}"


class _EofToken:
	"""Sentinel returned by Parser.peek() at end of input.

	All type predicates return False; all callable/variable fields are None.
	Duck-type compatible with Token for all predicate and attribute access patterns
	used in the parser.
	"""
	text      = '<END-OF-INPUT>'
	code      = EOF_CODE
	display   = b''
	bp        = None
	operator  = None
	postfix   = None
	function  = None
	command   = None
	nullary   = None
	converter = None
	variable  = None

	def is_digit(self) -> bool:        return False
	def is_numeric_var(self) -> bool:  return False
	def is_list_var(self) -> bool:     return False
	def is_list_start(self) -> bool:   return False
	def is_matrix_var(self) -> bool:   return False
	def is_equation_var(self) -> bool: return False
	def is_string_var(self) -> bool:   return False
	def is_stat_var(self) -> bool:     return False
	def is_window_var(self) -> bool:   return False
	def is_name_char(self) -> bool:    return False

	def __repr__(self) -> str:         return '<EOF>'


EOF_TOKEN = _EofToken()
