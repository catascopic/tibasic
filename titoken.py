from dataclasses import dataclass
from collections.abc import Callable


# _CHARSET: TI-83+ large-font byte -> Unicode character (None = undefined slot).
#
# This is the private source of truth for decoding a token's display bytes into a
# human-readable string (Token.text), used purely for debugging/printing.  The
# characters are convenient renderings, not a canonical mapping: several have no
# faithful Unicode equivalent (𝐅, 𝟑, ẍ, ṕ, ...) and a couple of font glyphs are
# duplicated.  Byte D6 decodes to '\n' (the newline token) rather than its ↵ glyph
# so program text round-trips with real line breaks.
_CHARSET: list[str | None] = [
	None, '𝑛', '𝑢', '𝑣', '𝑤', '►', '🡅', '🡇',		# 00
	'∫', '×', '▫', '﹢', '·', 'ₜ', '𝟑', '𝟊',			# 08
	'√', '¹', '²', '∠', '°', 'ʳ', 'ᵀ', '≤',			# 10
	'≠', '≥', '⁻', 'ᴇ', '→', '⑽', '↑', '↓',			# 18
	' ', '!', '"', '#', '⁴', '%', '&', "'",			# 20
	'(', ')', '*', '+', ',', '-', '.', '/',			# 28
	'0', '1', '2', '3', '4', '5', '6', '7', 		# 30
	'8', '9', ':', ';', '<', '=', '>', '?',			# 38
	'@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',			# 40
	'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',			# 48
	'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',			# 50
	'X', 'Y', 'Z', 'θ', '\\', ']', '^', '_',		# 58
	'`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',			# 60
	'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',			# 68
	'p', 'q', 'r', 's', 't', 'u', 'v', 'w',			# 70
	'x', 'y', 'z', '{', '|', '}', '~', '≛',			# 78
	'₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇',				# 80
	'₈', '₉', 'Á', 'À', 'Â', 'Ä', 'á', 'à',			# 88
	'â', 'ä', 'É', 'È', 'Ê', 'Ë', 'é', 'è',			# 90
	'ê', 'ë', 'Í', 'Ì', 'Î', 'Ï', 'í', 'ì',			# 98
	'î', 'ï', 'Ó', 'Ò', 'Ô', 'Ö', 'ó', 'ò',			# A0
	'ô', 'ö', 'Ú', 'Ù', 'Û', 'Ü', 'ú', 'ù',			# A8
	'û', 'ü', 'Ç', 'ç', 'Ñ', 'ñ', '´', 'ˋ',			# B0
	'¨', '¿', '¡', 'α', 'β', 'γ', 'Δ', 'δ',			# B8
	'ε', '[', 'λ', 'μ', 'π', 'ρ', 'Σ', 'σ',			# C0
	'τ', 'φ', 'Ω', 'ẍ', 'ȳ', 'ˣ', '…', '◄',			# C8
	None, None, None, None, None, '³', '\n', '𝑖',	# D0
	'ṕ', 'χ', '𝐅', '𝑒', 'ᴸ', '𝐍', '⸩', '🡆',			# D8
	None, None, None, None, None, None, None, None,	# E0
	None, None, None, None, None, None, None, None,	# E8
	None, None, '$', None, 'ß', None, None, None,	# F0
	None, None, None, None, None, None, None, None,	# F8
]


def decode(display: bytes) -> str:
	"""Render display bytes as a human-readable string (undefined bytes as \\xNN)."""
	return ''.join(_CHARSET[b] for b in display)


@dataclass(slots=True, frozen=True, eq=False)
class Token:
	code: bytes               # token code as stored in a program (1 or 2 bytes)
	display: bytes            # large-font byte sequence that renders this token
	bp: tuple[int, int] | None = None
	operator:  Callable | None = None  # (lhs, rhs) -> value
	postfix:   Callable | None = None  # (operand) -> value (prefix or postfix)
	function:  Callable | None = None  # (ArgParser) -> value for function tokens
	command:   Callable | None = None  # (ArgParser) -> None for command tokens
	nullary:   Callable | None = None  # (env) -> value for read-only computed constants
	converter: Callable | None = None  # (value) -> value for ►DMS, ►Dec, ►Frac and others
	variable:  Callable | None = None  # Variable flyweight for storable typed variables

	@property
	def text(self) -> str:
		"""Human-readable rendering of the display bytes (debugging/printing only)."""
		return decode(self.display)

	# ── Token type predicates ──────────────────────────────────────────────────

	def is_digit(self) -> bool:
		return 0x30 <= self.code[0] <= 0x39

	def is_numeric_var(self) -> bool:
		return 0x41 <= self.code[0] < 0x5C or self.code == b'\x62\x21'

	def is_list_var(self) -> bool:
		return self.code[0] == 0x5D

	def is_list_start(self):
		return self.code[0] in {0x5D, 0xEB}

	def is_matrix_var(self) -> bool:
		return self.code[0] == 0x5C

	def is_equation_var(self) -> bool:
		return self.code[0] == 0x5E

	def is_string_var(self) -> bool:
		return self.code[0] == 0xAA

	def is_stat_var(self) -> bool:
		return self.code[0] == 0x62

	def is_window_var(self) -> bool:
		return self.code[0] == 0x63

	def is_name_char(self):
		return self.is_numeric_var() or self.is_digit()

	def __repr__(self):
		return f'0x{int.from_bytes(self.code):0{2 * len(self.code)}X}:{self.text!r}'


class EofToken:
	"""Sentinel returned by Parser.peek() at end of input.

	All type predicates return False; all callable/variable fields are None.
	Duck-type compatible with Token for all predicate and attribute access patterns
	used in the parser.
	"""
	text      = '<END-OF-INPUT>'
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


EOF_TOKEN = EofToken()
