"""Generate largefont.py from glyphs.json."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors titoken._CHARSET exactly.
CHARSET: list[str | None] = [
	None, '𝑛', '𝑢', '𝑣', '𝑤', '►', '🡅', '🡇',		# 00
	'∫', '×', '▫', '﹢', '·', 'ₜ', '𝟑', '𝟊',			# 08
	'√', '¹', '²', '∠', '°', 'ʳ', 'ᵀ', '≤',			# 10
	'≠', '≥', '⁻', 'ᴇ', '→', '⑽', '↑', '↓',			# 18
	' ', '!', '"', '#', '⁴', '%', '&', "'",			# 20
	'(', ')', '*', '+', ',', '-', '.', '/',				# 28
	'0', '1', '2', '3', '4', '5', '6', '7',			# 30
	'8', '9', ':', ';', '<', '=', '>', '?',				# 38
	'@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',			# 40
	'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',			# 48
	'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',			# 50
	'X', 'Y', 'Z', 'θ', '\\', ']', '^', '_',			# 58
	'`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',			# 60
	'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',			# 68
	'p', 'q', 'r', 's', 't', 'u', 'v', 'w',			# 70
	'x', 'y', 'z', '{', '|', '}', '~', '≛',			# 78
	'₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇',			# 80
	'₈', '₉', 'Á', 'À', 'Â', 'Ä', 'á', 'à',			# 88
	'â', 'ä', 'É', 'È', 'Ê', 'Ë', 'é', 'è',			# 90
	'ê', 'ë', 'Í', 'Ì', 'Î', 'Ï', 'í', 'ì',			# 98
	'î', 'ï', 'Ó', 'Ò', 'Ô', 'Ö', 'ó', 'ò',			# A0
	'ô', 'ö', 'Ú', 'Ù', 'Û', 'Ü', 'ú', 'ù',			# A8
	'û', 'ü', 'Ç', 'ç', 'Ñ', 'ñ', '´', 'ˋ',			# B0
	'¨', '¿', '¡', 'α', 'β', 'γ', 'Δ', 'δ',			# B8
	'ε', '[', 'λ', 'μ', 'π', 'ρ', 'Σ', 'σ',			# C0
	'τ', 'φ', 'Ω', 'ẍ', 'ȳ', 'ˣ', '…', '◄',			# C8
	None, None, None, None, None, '³', '\n', '𝑖',		# D0
	'ṕ', 'χ', '𝐅', '𝑒', 'ᴸ', '𝐍', '⸩', '🡆',			# D8
	None, None, None, None, None, None, None, None,	# E0
	None, None, None, None, None, None, None, None,	# E8
	None, None, '$', None, 'ß', None, None, None,		# F0
	None, None, None, None, None, None, None, None,	# F8
]


def row_to_cols(s: str) -> bytes:
	"""Convert a 35-char row-major string (5 wide × 7 tall) to 5 column bytes.

	Each column byte uses 7 bits: bit 6 = top row, bit 0 = bottom row.
	"""
	cols = []
	for col in range(5):
		b = 0
		for row in range(7):
			if s[row * 5 + col] == '1':
				b |= (1 << (6 - row))
		cols.append(b)
	return bytes(cols)


def char_label(i: int) -> str:
	ch = CHARSET[i] if i < len(CHARSET) else None
	if ch is None:
		return '(undefined)'
	if ch == '\n':
		return r'\n'
	return repr(ch).strip("'")


def bytes_literal(data: bytes) -> str:
	return "b'" + ''.join(f'\\x{b:02x}' for b in data) + "'"


def main() -> str:
	with open(os.path.join(ROOT, 'glyphs.json')) as f:
		raw: list[str] = json.load(f)

	assert len(raw) == 256

	lines = [
		'# Large-font bitmap data for the TI-83+ display byte charset.',
		'# Each entry corresponds to one display byte (index = byte value).',
		'# All glyphs are 5 columns × 7 rows; each byte encodes one column,',
		'# where bit 6 is the top row and bit 0 is the bottom row.',
		'# None = no glyph defined for this display byte.',
		'_LARGEFONT: list[bytes | None] = [',
	]

	for i, s in enumerate(raw):
		label = char_label(i)
		if not s:
			lines.append(f'\tNone,        # {i:02X} {label}')
		else:
			assert len(s) == 35, f'unexpected length {len(s)} at index {i}'
			blit = bytes_literal(row_to_cols(s))
			lines.append(f'\t{blit},  # {i:02X} {label}')

	lines.append(']')
	return '\n'.join(lines) + '\n'


if __name__ == '__main__':
	out_path = os.path.join(ROOT, 'largefont.py')
	content = main()
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write(content)
	print(f'Written {out_path}', file=sys.stderr)
