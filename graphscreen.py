import struct

from itertools import batched


ROWS = 64
COLS = 96
_PIXEL_DATA_OFFSET = 62


class GraphScreen:
	"""The calculator's graph screen: a 64 row × 96 column monochrome pixel buffer.

	This is the pixel surface the graph-mode and drawing commands (Pxl-/Pt-/Line/
	Circle/DrawF/Text/…) render onto.  The home screen reuses the same 64×96 LCD
	but addresses it as a 16×8 character grid, so it will be a separate type.

	The live buffer is dense — one byte per pixel (0 = off, 1 = on) — to keep
	pixel access trivial.  Bit-packing is strictly a serialization concern
	(Pic / AppVar I/O) and lives elsewhere.

	Coordinates are (row, column) with the origin at the top-left, matching the
	argument order of TI's Pxl- commands: row is vertical (0–63), column is
	horizontal (0–95).  Point/graph commands translate their coordinates into
	this pixel space before drawing.
	"""

	def __init__(self):
		self.buffer = tuple(bytearray(COLS) for _ in range(ROWS))
		self.valid = False
		# Bumped on every pixel change — the cheap signal a frontend diffs against to
		# decide whether to repaint (e.g. to animate a drawing as it's built up).
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
		self.valid = False
		self.version += 1

	def print_screen(self, path, pixel_size: int = 1) -> None:
		"""Save the graph buffer as a monochrome BMP.

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
