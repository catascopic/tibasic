"""Generate smallfont.py from the GIF sprites in webref/."""
from PIL import Image
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(ROOT, 'webref', 'TI-83 Plus Small Font - TI-Basic Developer_files')

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


def extract_glyph(path: str) -> bytes:
	img = Image.open(path).convert('RGB')
	w, h = img.size
	assert h == 12, f'unexpected height {h} in {path}'
	assert w % 2 == 0
	pixels = img.load()
	data = []
	for col in range(w // 2):
		col_byte = 0
		for row in range(6):
			r, g, b = pixels[col * 2, row * 2]
			if r < 128:
				col_byte |= (1 << (5 - row))
		data.append(col_byte)
	return bytes(data)


def char_label(i: int) -> str:
	ch = CHARSET[i] if i < len(CHARSET) else None
	if ch is None:
		return '(undefined)'
	if ch == '\n':
		return r'\n'
	return repr(ch).strip("'")


def bytes_literal(data: bytes) -> str:
	return "b'" + ''.join(f'\\x{b:02x}' for b in data) + "'"


def main():
	glyphs: dict[int, bytes] = {}
	for fname in os.listdir(FILES_DIR):
		m = re.match(r'^([0-9A-Fa-f]{2})h_', fname)
		if m and fname.endswith('.gif'):
			bv = int(m.group(1), 16)
			glyphs[bv] = extract_glyph(os.path.join(FILES_DIR, fname))

	print(f'Extracted {len(glyphs)} glyphs', file=sys.stderr)

	lines = [
		'# Small-font bitmap data for the TI-83+ display byte charset.',
		'# Each entry corresponds to one display byte (index = byte value).',
		'# Each bytes object encodes one column per byte; each byte uses 6 bits,',
		'# where bit 5 is the top row and bit 0 is the bottom row.',
		'# None = no glyph defined for this display byte.',
		'_SMALLFONT: list[bytes | None] = [',
	]

	for i in range(256):
		label = char_label(i)
		data = glyphs.get(i)
		if data is None:
			lines.append(f'\tNone,              # {i:02X} {label}')
		else:
			blit = bytes_literal(data)
			lines.append(f'\t{blit},  # {i:02X} {label}')

	lines.append(']')
	return '\n'.join(lines) + '\n'


if __name__ == '__main__':
	out_path = os.path.join(ROOT, 'smallfont.py')
	content = main()
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write(content)
	print(f'Written to {out_path}', file=sys.stderr)
