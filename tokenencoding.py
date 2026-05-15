

# ── TI-83+ Large Font Display Encoding ────────────────────────────────────────
# Maps Unicode text → bytes in the TI-83+ Large Font charset (0x00–0xF7).
# Multi-char keys must appear before any single-char prefix they share so
# the greedy encoder in _encode_display() matches them first.

_D: dict[str, bytes] = {
	# ── Multi-char sequences (combining chars / ligatures) ──────────────────────
	'⁻¹':	  b'\x11',	   # inverse/reciprocal as single glyph
	'x̄': b'\xcb',	   # x̄  (x + combining macron = x-mean)
	'ȳ': b'\xcc',	   # ȳ  (y + combining macron = y-mean)
	'p̂': b'\xd8',	   # p̂  (p + combining circumflex = p-hat)
	'₁₀': b'\x90',  # subscript 10
	# ── Special glyphs 0x01–0x1F ────────────────────────────────────────────────
	'►':	   b'\x05',	   # right-pointing triangle (convert arrow)
	'🡅':	  b'\x06',	   # scroll up
	'🡇':	  b'\x07',	   # scroll down
	'∫':	   b'\x08',	   # integral
	'×':	   b'\x09',	   # multiplication cross
	'√':	   b'\x10',	   # square root radical
	'²':	   b'\x12',	   # superscript 2
	'∠':	   b'\x13',	   # angle
	'∟':	   b'\x13',	   # right angle → same glyph as ∠
	'°':	   b'\x14',	   # degree
	'ʳ':	   b'\x15',	   # superscript r (radian)
	'ᵀ':	   b'\x16',	   # superscript T (transpose)
	'≤':	   b'\x17',	   # less than or equal
	'≠':	   b'\x18',	   # not equal
	'≥':	   b'\x19',	   # greater than or equal
	'⁻':	   b'\x1a',	   # superscript minus (negation); also prefix of ⁻¹ above
	'ᴇ':	   b'\x1b',	   # scientific-notation E
	'→':	   b'\x1c',	   # right arrow (store)
	'↑':	   b'\x1e',	   # up arrow
	'↓':	   b'\x1f',	   # down arrow
	# ── ASCII-position remaps ────────────────────────────────────────────────────
	'[':	   b'\xc1',	   # left bracket (0x5B is θ in display charset)
	'³':	   b'\x0e',	   # superscript 3 / cube-root mark
	'−':	   b'\x2d',	   # math minus (U+2212) → regular dash
	# ── Other special Unicode ────────────────────────────────────────────────────
	'θ':	   b'\x5b',	   # theta (at 0x5B, where ASCII has '[')
	'←':	   b'\xcf',	   # left arrow
	'◄':	   b'\xcf',	   # left-pointing triangle → left arrow glyph
	'↵':	   b'\xd6',	   # enter/return arrow
	'…':	   b'\xce',	   # ellipsis
	'ȳ':	   b'\xcc',	   # y-bar precomposed (U+0233)
	'𝑒':	   b'\x65',	   # math italic e → regular e
	'𝑖':	   b'\xd7',	   # math italic i → imaginary-i glyph
	# ── Subscript digits 0–9 ─────────────────────────────────────────────────────
	'₀': b'\x80', '₁': b'\x81', '₂': b'\x82', '₃': b'\x83', '₄': b'\x84',
	'₅': b'\x85', '₆': b'\x86', '₇': b'\x87', '₈': b'\x88', '₉': b'\x89',
	# ── Greek letters ─────────────────────────────────────────────────────────────
	'α': b'\xbb', 'β': b'\xbc', 'γ': b'\xbd', 'Δ': b'\xbe', 'δ': b'\xbf',
	'ε': b'\xc0', 'λ': b'\xc2', 'μ': b'\xc3', 'π': b'\xc4', 'ρ': b'\xc5',
	'Σ': b'\xc6', 'σ': b'\xc7', 'τ': b'\xc8', 'φ': b'\xc9', 'Ω': b'\xca',
	'χ': b'\xd9',
	# ── Accented Latin – uppercase ────────────────────────────────────────────────
	'Á': b'\x8a', 'À': b'\x8b', 'Â': b'\x8c', 'Ä': b'\x8d',
	'É': b'\x92', 'È': b'\x93', 'Ê': b'\x94', 'Ë': b'\x95',
	'Í': b'\x9a', 'Ì': b'\x9b', 'Î': b'\x9c', 'Ï': b'\x9d',
	'Ó': b'\xa2', 'Ò': b'\xa3', 'Ô': b'\xa4', 'Ö': b'\xa5',
	'Ú': b'\xaa', 'Ù': b'\xab', 'Û': b'\xac', 'Ü': b'\xad',
	'Ç': b'\xb2', 'Ñ': b'\xb4',
	# ── Accented Latin – lowercase ────────────────────────────────────────────────
	'á': b'\x8e', 'à': b'\x8f', 'â': b'\x90', 'ä': b'\x91',
	'é': b'\x96', 'è': b'\x97', 'ê': b'\x98', 'ë': b'\x99',
	'í': b'\x9e', 'ì': b'\x9f', 'î': b'\xa0', 'ï': b'\xa1',
	'ó': b'\xa6', 'ò': b'\xa7', 'ô': b'\xa8', 'ö': b'\xa9',
	'ú': b'\xae', 'ù': b'\xaf', 'û': b'\xb0', 'ü': b'\xb1',
	'ç': b'\xb3', 'ñ': b'\xb5',
	# ── Punctuation / accent marks ────────────────────────────────────────────────
	'´': b'\xb6', '¨': b'\xb8', '¿': b'\xb9', '¡': b'\xba', 'ß': b'\xf4',
}

_D_KEYS = sorted(_D, key=len, reverse=True)


def _encode_display(text: str) -> bytes:
	"""Encode token display text into TI-83+ Large Font bytes (greedy longest-match)."""
	result = bytearray()
	i = 0
	while i < len(text):
		matched = False
		for key in _D_KEYS:
			if text.startswith(key, i):
				result.extend(_D[key])
				i += len(key)
				matched = True
				break
		if not matched:
			ch = text[i]
			code = ord(ch)
			if 0x20 <= code <= 0x7e and ch != '[':
				result.append(code)
			i += 1
	return bytes(result)
