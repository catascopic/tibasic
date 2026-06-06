from __future__ import annotations


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
		self.buffer = bytearray(self.ROWS * self.COLS)

	def _index(self, row: int, col: int) -> int:
		return row * self.COLS + col

	def get(self, row: int, col: int) -> bool:
		return bool(self.buffer[self._index(row, col)])

	def set(self, row: int, col: int, on: bool = True) -> None:
		self.buffer[self._index(row, col)] = 1 if on else 0

	def toggle(self, row: int, col: int) -> None:
		self.buffer[self._index(row, col)] ^= 1

	def clear(self) -> None:
		self.buffer = bytearray(self.ROWS * self.COLS)
