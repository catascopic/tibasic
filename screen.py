import sys

from itertools import batched


class Screen:
	"""The calculator's monochrome LCD: 64 rows × 96 columns.

	The live buffer is dense — one byte per pixel (0 = off, 1 = on) — to keep
	pixel access trivial.  Bit-packing is strictly a serialization concern
	(Pic / AppVar I/O) and lives elsewhere.

	Coordinates are (row, column) with the origin at the top-left, matching the
	argument order of TI's Pxl- commands: row is vertical (0–63), column is
	horizontal (0–95).  Point/graph commands translate their coordinates into
	this pixel space before drawing.
	"""

	ROWS = 64
	COLS = 96

	def __init__(self):
		self.buffer = tuple(bytearray(self.COLS) for _ in range(self.ROWS))

	def get(self, row: int, col: int) -> bool:
		return bool(self.buffer[row][col])

	def set(self, row: int, col: int, on: bool = True) -> None:
		self.buffer[row][col] = 1 if on else 0

	def set_off(self, row: int, col: int, on: bool = True) -> None:
		self.set(row, col, False)

	def toggle(self, row: int, col: int) -> None:
		self.buffer[row][col] ^= 1

	def clear(self) -> None:
		for row in self.buffer:
			row.__init__(self.COLS)  # hack?

	def show(self) -> str:
		border = '▒' * (self.COLS + 4)
		print(border)
		for row1, row2 in batched(self.buffer, 2):
			print('▒▒', end='')
			for px1, px2 in zip(row1, row2, strict=True):
				print(' ▀▄█'[~(px1 | (px2 << 1))], end='')
			print('▒▒')
		print(border)
