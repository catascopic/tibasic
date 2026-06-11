"""Compute each token's *display bytes*: the TI-83+ large-font byte sequence the
calculator uses to render that token on screen.

A token's `text` is a human-readable Unicode string (e.g. 'sin(', 'Shade𝐅(').
The calculator instead renders text as bytes indexing its large-font glyph table.
Each character of `text` maps to exactly one font byte, so the display bytes are
built by encoding `text` character-by-character through _CHAR_TO_BYTE below.

Most tokens are pure ASCII and map trivially (sin( -> b'sin('); the interesting
cases are special glyphs (Shade𝐅( -> b'Shade\\xDA(') where `text` uses the
Unicode character the font assigns to that byte.

_CHAR_TO_BYTE is the authoritative encoding table (it was generated from the font
charset but is now self-contained, so there is no runtime file dependency).  A few
font glyphs have no distinct Unicode codepoint or are drawn by duplicate bytes;
those canonical choices are baked directly into the table rather than patched
afterward:

  * '🠅' ("Square Up") is drawn by two identical bytes, 06 and F3; text encodes to 06.
  * '\\n' (newline) encodes to D6.

This module deliberately does NOT modify Token/catalog; it produces a separate
{token.code: display_bytes} mapping that a renderer can consult.
"""
import catalog


# Unicode character -> TI-83+ large-font byte.  Ordered by byte value.
_CHAR_TO_BYTE: dict[str, int] = {
	'𝑛': 0x01,  # MATHEMATICAL ITALIC SMALL N
	'𝑢': 0x02,  # MATHEMATICAL ITALIC SMALL U
	'𝑣': 0x03,  # MATHEMATICAL ITALIC SMALL V
	'𝑤': 0x04,  # MATHEMATICAL ITALIC SMALL W
	'►': 0x05,  # BLACK RIGHT-POINTING POINTER
	'🡅': 0x06,  # UPWARDS HEAVY ARROW
	'🡇': 0x07,  # DOWNWARDS HEAVY ARROW
	'∫': 0x08,  # INTEGRAL
	'×': 0x09,  # MULTIPLICATION SIGN
	'▫': 0x0A,  # WHITE SMALL SQUARE
	'﹢': 0x0B,  # SMALL PLUS SIGN
	'·': 0x0C,  # MIDDLE DOT
	'ₜ': 0x0D,  # LATIN SUBSCRIPT SMALL LETTER T
	'𝟑': 0x0E,  # MATHEMATICAL BOLD DIGIT THREE
	'𝟊': 0x0F,  # MATHEMATICAL BOLD CAPITAL DIGAMMA
	'√': 0x10,  # SQUARE ROOT
	'¹': 0x11,  # SUPERSCRIPT ONE
	'²': 0x12,  # SUPERSCRIPT TWO
	'∠': 0x13,  # ANGLE
	'°': 0x14,  # DEGREE SIGN
	'ʳ': 0x15,  # MODIFIER LETTER SMALL R
	'ᵀ': 0x16,  # MODIFIER LETTER CAPITAL T
	'≤': 0x17,  # LESS-THAN OR EQUAL TO
	'≠': 0x18,  # NOT EQUAL TO
	'≥': 0x19,  # GREATER-THAN OR EQUAL TO
	'⁻': 0x1A,  # SUPERSCRIPT MINUS
	'ᴇ': 0x1B,  # LATIN LETTER SMALL CAPITAL E
	'→': 0x1C,  # RIGHTWARDS ARROW
	'⑽': 0x1D,  # PARENTHESIZED NUMBER TEN
	'↑': 0x1E,  # UPWARDS ARROW
	'↓': 0x1F,  # DOWNWARDS ARROW
	' ': 0x20,  # SPACE
	'!': 0x21,  # EXCLAMATION MARK
	'"': 0x22,  # QUOTATION MARK
	'#': 0x23,  # NUMBER SIGN
	'⁴': 0x24,  # SUPERSCRIPT FOUR
	'%': 0x25,  # PERCENT SIGN
	'&': 0x26,  # AMPERSAND
	"'": 0x27,  # APOSTROPHE
	'(': 0x28,  # LEFT PARENTHESIS
	')': 0x29,  # RIGHT PARENTHESIS
	'*': 0x2A,  # ASTERISK
	'+': 0x2B,  # PLUS SIGN
	',': 0x2C,  # COMMA
	'-': 0x2D,  # HYPHEN-MINUS
	'.': 0x2E,  # FULL STOP
	'/': 0x2F,  # SOLIDUS
	'0': 0x30,  # DIGIT ZERO
	'1': 0x31,  # DIGIT ONE
	'2': 0x32,  # DIGIT TWO
	'3': 0x33,  # DIGIT THREE
	'4': 0x34,  # DIGIT FOUR
	'5': 0x35,  # DIGIT FIVE
	'6': 0x36,  # DIGIT SIX
	'7': 0x37,  # DIGIT SEVEN
	'8': 0x38,  # DIGIT EIGHT
	'9': 0x39,  # DIGIT NINE
	':': 0x3A,  # COLON
	';': 0x3B,  # SEMICOLON
	'<': 0x3C,  # LESS-THAN SIGN
	'=': 0x3D,  # EQUALS SIGN
	'>': 0x3E,  # GREATER-THAN SIGN
	'?': 0x3F,  # QUESTION MARK
	'@': 0x40,  # COMMERCIAL AT
	'A': 0x41,  # LATIN CAPITAL LETTER A
	'B': 0x42,  # LATIN CAPITAL LETTER B
	'C': 0x43,  # LATIN CAPITAL LETTER C
	'D': 0x44,  # LATIN CAPITAL LETTER D
	'E': 0x45,  # LATIN CAPITAL LETTER E
	'F': 0x46,  # LATIN CAPITAL LETTER F
	'G': 0x47,  # LATIN CAPITAL LETTER G
	'H': 0x48,  # LATIN CAPITAL LETTER H
	'I': 0x49,  # LATIN CAPITAL LETTER I
	'J': 0x4A,  # LATIN CAPITAL LETTER J
	'K': 0x4B,  # LATIN CAPITAL LETTER K
	'L': 0x4C,  # LATIN CAPITAL LETTER L
	'M': 0x4D,  # LATIN CAPITAL LETTER M
	'N': 0x4E,  # LATIN CAPITAL LETTER N
	'O': 0x4F,  # LATIN CAPITAL LETTER O
	'P': 0x50,  # LATIN CAPITAL LETTER P
	'Q': 0x51,  # LATIN CAPITAL LETTER Q
	'R': 0x52,  # LATIN CAPITAL LETTER R
	'S': 0x53,  # LATIN CAPITAL LETTER S
	'T': 0x54,  # LATIN CAPITAL LETTER T
	'U': 0x55,  # LATIN CAPITAL LETTER U
	'V': 0x56,  # LATIN CAPITAL LETTER V
	'W': 0x57,  # LATIN CAPITAL LETTER W
	'X': 0x58,  # LATIN CAPITAL LETTER X
	'Y': 0x59,  # LATIN CAPITAL LETTER Y
	'Z': 0x5A,  # LATIN CAPITAL LETTER Z
	'θ': 0x5B,  # GREEK SMALL LETTER THETA
	'\\': 0x5C,  # REVERSE SOLIDUS
	']': 0x5D,  # RIGHT SQUARE BRACKET
	'^': 0x5E,  # CIRCUMFLEX ACCENT
	'_': 0x5F,  # LOW LINE
	'`': 0x60,  # GRAVE ACCENT
	'a': 0x61,  # LATIN SMALL LETTER A
	'b': 0x62,  # LATIN SMALL LETTER B
	'c': 0x63,  # LATIN SMALL LETTER C
	'd': 0x64,  # LATIN SMALL LETTER D
	'e': 0x65,  # LATIN SMALL LETTER E
	'f': 0x66,  # LATIN SMALL LETTER F
	'g': 0x67,  # LATIN SMALL LETTER G
	'h': 0x68,  # LATIN SMALL LETTER H
	'i': 0x69,  # LATIN SMALL LETTER I
	'j': 0x6A,  # LATIN SMALL LETTER J
	'k': 0x6B,  # LATIN SMALL LETTER K
	'l': 0x6C,  # LATIN SMALL LETTER L
	'm': 0x6D,  # LATIN SMALL LETTER M
	'n': 0x6E,  # LATIN SMALL LETTER N
	'o': 0x6F,  # LATIN SMALL LETTER O
	'p': 0x70,  # LATIN SMALL LETTER P
	'q': 0x71,  # LATIN SMALL LETTER Q
	'r': 0x72,  # LATIN SMALL LETTER R
	's': 0x73,  # LATIN SMALL LETTER S
	't': 0x74,  # LATIN SMALL LETTER T
	'u': 0x75,  # LATIN SMALL LETTER U
	'v': 0x76,  # LATIN SMALL LETTER V
	'w': 0x77,  # LATIN SMALL LETTER W
	'x': 0x78,  # LATIN SMALL LETTER X
	'y': 0x79,  # LATIN SMALL LETTER Y
	'z': 0x7A,  # LATIN SMALL LETTER Z
	'{': 0x7B,  # LEFT CURLY BRACKET
	'|': 0x7C,  # VERTICAL LINE
	'}': 0x7D,  # RIGHT CURLY BRACKET
	'~': 0x7E,  # TILDE
	'≛': 0x7F,  # STAR EQUALS
	'₀': 0x80,  # SUBSCRIPT ZERO
	'₁': 0x81,  # SUBSCRIPT ONE
	'₂': 0x82,  # SUBSCRIPT TWO
	'₃': 0x83,  # SUBSCRIPT THREE
	'₄': 0x84,  # SUBSCRIPT FOUR
	'₅': 0x85,  # SUBSCRIPT FIVE
	'₆': 0x86,  # SUBSCRIPT SIX
	'₇': 0x87,  # SUBSCRIPT SEVEN
	'₈': 0x88,  # SUBSCRIPT EIGHT
	'₉': 0x89,  # SUBSCRIPT NINE
	'Á': 0x8A,  # LATIN CAPITAL LETTER A WITH ACUTE
	'À': 0x8B,  # LATIN CAPITAL LETTER A WITH GRAVE
	'Â': 0x8C,  # LATIN CAPITAL LETTER A WITH CIRCUMFLEX
	'Ä': 0x8D,  # LATIN CAPITAL LETTER A WITH DIAERESIS
	'á': 0x8E,  # LATIN SMALL LETTER A WITH ACUTE
	'à': 0x8F,  # LATIN SMALL LETTER A WITH GRAVE
	'â': 0x90,  # LATIN SMALL LETTER A WITH CIRCUMFLEX
	'ä': 0x91,  # LATIN SMALL LETTER A WITH DIAERESIS
	'É': 0x92,  # LATIN CAPITAL LETTER E WITH ACUTE
	'È': 0x93,  # LATIN CAPITAL LETTER E WITH GRAVE
	'Ê': 0x94,  # LATIN CAPITAL LETTER E WITH CIRCUMFLEX
	'Ë': 0x95,  # LATIN CAPITAL LETTER E WITH DIAERESIS
	'é': 0x96,  # LATIN SMALL LETTER E WITH ACUTE
	'è': 0x97,  # LATIN SMALL LETTER E WITH GRAVE
	'ê': 0x98,  # LATIN SMALL LETTER E WITH CIRCUMFLEX
	'ë': 0x99,  # LATIN SMALL LETTER E WITH DIAERESIS
	'Í': 0x9A,  # LATIN CAPITAL LETTER I WITH ACUTE
	'Ì': 0x9B,  # LATIN CAPITAL LETTER I WITH GRAVE
	'Î': 0x9C,  # LATIN CAPITAL LETTER I WITH CIRCUMFLEX
	'Ï': 0x9D,  # LATIN CAPITAL LETTER I WITH DIAERESIS
	'í': 0x9E,  # LATIN SMALL LETTER I WITH ACUTE
	'ì': 0x9F,  # LATIN SMALL LETTER I WITH GRAVE
	'î': 0xA0,  # LATIN SMALL LETTER I WITH CIRCUMFLEX
	'ï': 0xA1,  # LATIN SMALL LETTER I WITH DIAERESIS
	'Ó': 0xA2,  # LATIN CAPITAL LETTER O WITH ACUTE
	'Ò': 0xA3,  # LATIN CAPITAL LETTER O WITH GRAVE
	'Ô': 0xA4,  # LATIN CAPITAL LETTER O WITH CIRCUMFLEX
	'Ö': 0xA5,  # LATIN CAPITAL LETTER O WITH DIAERESIS
	'ó': 0xA6,  # LATIN SMALL LETTER O WITH ACUTE
	'ò': 0xA7,  # LATIN SMALL LETTER O WITH GRAVE
	'ô': 0xA8,  # LATIN SMALL LETTER O WITH CIRCUMFLEX
	'ö': 0xA9,  # LATIN SMALL LETTER O WITH DIAERESIS
	'Ú': 0xAA,  # LATIN CAPITAL LETTER U WITH ACUTE
	'Ù': 0xAB,  # LATIN CAPITAL LETTER U WITH GRAVE
	'Û': 0xAC,  # LATIN CAPITAL LETTER U WITH CIRCUMFLEX
	'Ü': 0xAD,  # LATIN CAPITAL LETTER U WITH DIAERESIS
	'ú': 0xAE,  # LATIN SMALL LETTER U WITH ACUTE
	'ù': 0xAF,  # LATIN SMALL LETTER U WITH GRAVE
	'û': 0xB0,  # LATIN SMALL LETTER U WITH CIRCUMFLEX
	'ü': 0xB1,  # LATIN SMALL LETTER U WITH DIAERESIS
	'Ç': 0xB2,  # LATIN CAPITAL LETTER C WITH CEDILLA
	'ç': 0xB3,  # LATIN SMALL LETTER C WITH CEDILLA
	'Ñ': 0xB4,  # LATIN CAPITAL LETTER N WITH TILDE
	'ñ': 0xB5,  # LATIN SMALL LETTER N WITH TILDE
	'´': 0xB6,  # ACUTE ACCENT
	'ˋ': 0xB7,  # MODIFIER LETTER GRAVE ACCENT
	'¨': 0xB8,  # DIAERESIS
	'¿': 0xB9,  # INVERTED QUESTION MARK
	'¡': 0xBA,  # INVERTED EXCLAMATION MARK
	'α': 0xBB,  # GREEK SMALL LETTER ALPHA
	'β': 0xBC,  # GREEK SMALL LETTER BETA
	'γ': 0xBD,  # GREEK SMALL LETTER GAMMA
	'Δ': 0xBE,  # GREEK CAPITAL LETTER DELTA
	'δ': 0xBF,  # GREEK SMALL LETTER DELTA
	'ε': 0xC0,  # GREEK SMALL LETTER EPSILON
	'[': 0xC1,  # LEFT SQUARE BRACKET
	'λ': 0xC2,  # GREEK SMALL LETTER LAMDA
	'μ': 0xC3,  # GREEK SMALL LETTER MU
	'π': 0xC4,  # GREEK SMALL LETTER PI
	'ρ': 0xC5,  # GREEK SMALL LETTER RHO
	'Σ': 0xC6,  # GREEK CAPITAL LETTER SIGMA
	'σ': 0xC7,  # GREEK SMALL LETTER SIGMA
	'τ': 0xC8,  # GREEK SMALL LETTER TAU
	'φ': 0xC9,  # GREEK SMALL LETTER PHI
	'Ω': 0xCA,  # GREEK CAPITAL LETTER OMEGA
	'ẍ': 0xCB,  # LATIN SMALL LETTER X WITH DIAERESIS
	'ȳ': 0xCC,  # LATIN SMALL LETTER Y WITH MACRON
	'ˣ': 0xCD,  # MODIFIER LETTER SMALL X
	'…': 0xCE,  # HORIZONTAL ELLIPSIS
	'◄': 0xCF,  # BLACK LEFT-POINTING POINTER
	'³': 0xD5,  # SUPERSCRIPT THREE
	'\n': 0xD6,  # newline (calculator newline byte; glyph ↵)
	'↵': 0xD6,  # DOWNWARDS ARROW WITH CORNER LEFTWARDS
	'𝑖': 0xD7,  # MATHEMATICAL ITALIC SMALL I
	'ṕ': 0xD8,  # LATIN SMALL LETTER P WITH ACUTE
	'χ': 0xD9,  # GREEK SMALL LETTER CHI
	'𝐅': 0xDA,  # MATHEMATICAL BOLD CAPITAL F
	'𝑒': 0xDB,  # MATHEMATICAL ITALIC SMALL E
	'ᴸ': 0xDC,  # MODIFIER LETTER CAPITAL L
	'𝐍': 0xDD,  # MATHEMATICAL BOLD CAPITAL N
	'⸩': 0xDE,  # RIGHT DOUBLE PARENTHESIS
	'🡆': 0xDF,  # RIGHTWARDS HEAVY ARROW
	'$': 0xF2,  # DOLLAR SIGN
	'ß': 0xF4,  # LATIN SMALL LETTER SHARP S
}


def display_bytes(token) -> bytes:
	"""Return the font-byte sequence that renders `token` on the calculator."""
	out = bytearray()
	for ch in token.text:
		try:
			out.append(_CHAR_TO_BYTE[ch])
		except KeyError as e:
			raise KeyError(
				f"No font byte for {ch!r} (U+{ord(ch):04X}) in token "
				f"0x{int.from_bytes(token.code):X} {token.text!r}"
			) from e
	return bytes(out)


def build_all() -> dict[bytes, bytes]:
	"""Map every catalog token's code to its display bytes."""
	return {t.code: display_bytes(t) for t in catalog.ALL_TOKENS}


if __name__ == '__main__':
	mapping = build_all()
	print(f"{len(mapping)} tokens mapped.\n")
	for t in catalog.ALL_TOKENS:
		code = int.from_bytes(t.code)
		print(f"  0x{code:<4X} {t.text!r:<14} -> {mapping[t.code]!r}")
