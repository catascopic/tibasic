import struct

from itertools import batched

from fonts import LARGE_FONT, GLYPH_HEIGHT, CELL_WIDTH, CELL_HEIGHT

ROWS = 64
COLS = 96
_PIXEL_DATA_OFFSET = 62


class Bitmap:
	"""The calculator's 96×64 monochrome LCD, as a dense pixel buffer.

	This is the shared surface every screen ultimately paints to: GraphScreen *is* a
	Bitmap driven as pixels, and the home/menu screens rasterize their character grid
	into one (through the font) to render or print_screen.  It owns everything that's
	purely about pixels — the pixel operations, the BMP encoding, and the half-block
	text rendering — with nothing graph-specific.

	The buffer is dense — one byte per pixel (0 = off, 1 = on) — to keep pixel access
	trivial.  Bit-packing is strictly a serialization concern (Pic / AppVar I/O) and
	lives elsewhere.

	Coordinates are (row, column) with the origin at the top-left, matching the
	argument order of TI's Pxl- commands: row is vertical (0–63), column is horizontal
	(0–95).  Point/graph commands translate their coordinates into this pixel space
	before drawing.

	`version` bumps on every pixel change — the cheap signal a frontend diffs against
	to decide whether to repaint (e.g. to animate a drawing as it's built up).
	"""

	def __init__(self):
		self.buffer = tuple(bytearray(COLS) for _ in range(ROWS))
		self.version = 0

	def get(self, row: int, col: int) -> bool:
		return bool(self.buffer[row][col])

	def set(self, row: int, col: int, on: bool = True) -> None:
		self.buffer[row][col] = on
		self.version += 1

	def set_off(self, row: int, col: int, on: bool = True) -> None:
		self.set(row, col, False)

	def toggle(self, row: int, col: int) -> None:
		self.buffer[row][col] ^= 1
		self.version += 1

	def clear(self) -> None:
		for row in self.buffer:
			row.__init__(COLS)  # hack?
		self.version += 1

	def blit_large(self, row: int, col: int, byte: int, invert: bool = False) -> int:
		"""Rasterize a large-font glyph as a full CELL_WIDTH×CELL_HEIGHT (6×8) block.

		Covers all 6×8 pixels in a single pass: glyph data in columns 0–4, rows 0–6;
		the gap column (5) and gap row (7) are set to 0 (or 1 when inverted).  With
		invert=True every bit is XOR'd with 1 — no second toggle pass needed.
		Returns the next column position (col + CELL_WIDTH).
		"""
		glyph = LARGE_FONT[byte]
		for dc in range(CELL_WIDTH):
			colbits = glyph[dc] if (glyph and dc < len(glyph)) else 0
			c = col + dc
			for dr in range(CELL_HEIGHT):
				bit = bool((colbits >> (GLYPH_HEIGHT - 1 - dr)) & 1) if dr < GLYPH_HEIGHT else False
				self.set(row + dr, c, bit ^ invert)
		return col + CELL_WIDTH

	def blit_cell(self, cell_row: int, cell_col: int, byte: int, invert: bool = False) -> None:
		"""Rasterize a large-font glyph at character grid position (cell_row, cell_col)."""
		self.blit_large(cell_row * CELL_HEIGHT, cell_col * CELL_WIDTH, byte, invert)

	def print_screen(self, path, pixel_size: int = 1) -> None:
		"""Save the buffer as a monochrome BMP.

		Each logical pixel becomes a `pixel_size` × `pixel_size` block.  On pixels
		are black; off pixels are white.  Written without any image library — the
		BMP format is simple enough to assemble directly.
		"""
		width  = COLS * pixel_size
		height = ROWS * pixel_size
		stride = ((width + 31) // 32) * 4   # row length in bytes, padded to 4-byte boundary

		with open(path, 'wb') as f:
			f.write(struct.pack(
				'<2sIHHIIiiHHIIiiII8B', b'BM',
				# file size, 2 reserved shorts
				_PIXEL_DATA_OFFSET + stride * height, 0, 0,
				# DIB header
				_PIXEL_DATA_OFFSET, 40, width, -height, 1, 1, 0, 0, 0, 0, 2, 0,
				# color table
				255, 255, 255, 0, 0, 0, 0, 0,
			))
			for r in range(ROWS):
				row = bytearray(stride)
				for c in range(COLS):
					if self.buffer[r][c]:
						for p in range(pixel_size):
							bit = c * pixel_size + p
							row[bit // 8] |= 0x80 >> (bit % 8)
				for _ in range(pixel_size):
					f.write(row)

	def rows(self) -> list[str]:
		"""The buffer as 32 rows of 96 half-block characters — two pixel rows packed
		into each text row (▀ top, ▄ bottom, █ both, space neither).  Border/framing
		is the caller's concern; this is the shared content a text frontend paints."""
		return [
			''.join(' ▀▄█'[~(px1 | (px2 << 1))] for px1, px2 in zip(row1, row2, strict=True))
			for row1, row2 in batched(self.buffer, 2)
		]

	def disp(self) -> None:
		border = '▒' * (COLS + 4)
		print(border)
		for row in self.rows():
			print('▒▒' + row + '▒▒')
		print(border)
