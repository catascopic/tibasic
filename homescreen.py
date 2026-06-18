from titoken import decode

# Display byte for '…' (0xCE) — what Disp truncates a too-long line with, as its
# 16th column.  An empty cell holds 0x00, which decodes to a blank glyph.
_ELLIPSIS = 0xCE
_BLANK = 0x00


class HomeScreen:
	"""The TI home screen: a 16-column × 8-row grid of display bytes with a Disp cursor.

	Pure state, like Graph — the I/O commands mutate it and a Console renders it.
	Each cell holds one TI display byte (one glyph); an empty cell is 0x00.  Output(
	writes at an absolute (row, col), wrapping across rows; Disp appends on the
	cursor line and truncates with an ellipsis instead of wrapping; ClrHome wipes
	it.  Coordinates here are 0-indexed; the commands translate from TI's 1-indexed
	Output(row, col) and validate the range.

	Text arrives as display bytes (from tiformat / token.display), so what's stored
	is exactly what a canvas frontend draws through the font tables; render() decodes
	it back to characters for the terminal frontend.
	"""

	ROWS = 8
	COLS = 16

	def __init__(self):
		self.clear()

	def clear(self) -> None:
		self._cells = [bytearray([_BLANK] * self.COLS) for _ in range(self.ROWS)]
		self.cursor_row = 0

	def _scroll(self) -> None:
		"""Drop the top row and append a blank one at the bottom."""
		self._cells.pop(0)
		self._cells.append(bytearray([_BLANK] * self.COLS))

	def output(self, row: int, col: int, text: bytes) -> None:
		"""Write `text` from (row, col), wrapping to following rows and clipping at
		the bottom edge — matching the calculator, which silently drops overflow."""
		index = row * self.COLS + col
		for byte in text:
			if index >= self.ROWS * self.COLS:
				break
			self._cells[index // self.COLS][index % self.COLS] = byte
			index += 1

	def disp(self, text: bytes) -> None:
		"""Append `text` on the cursor line and drop to the next line, scrolling
		the grid up once the bottom is passed.  A line too long for the 16
		columns is truncated with an ellipsis as the 16th character, rather than
		wrapping — that's how the real calculator's Disp behaves; Output( and
		echo() are the ones that wrap.

		Disp always leaves at least one blank line visible at the bottom: if
		this call writes the last row, it scrolls again immediately afterward
		rather than deferring to whatever writes next.  echo() (Input/Prompt) is
		exempt — it's allowed to fill the bottom row and leave it filled.
		"""
		if self.cursor_row >= self.ROWS:
			self._scroll()
			self.cursor_row = self.ROWS - 1
		if len(text) > self.COLS:
			text = text[:self.COLS - 1] + bytes([_ELLIPSIS])
		row = self._cells[self.cursor_row]
		for col, byte in enumerate(text):
			row[col] = byte
		self.cursor_row += 1
		if self.cursor_row >= self.ROWS:
			self._scroll()
			self.cursor_row = self.ROWS - 1

	def echo(self, text: bytes) -> None:
		"""Write `text` starting at the cursor, wrapping to the next row instead
		of truncating, and leaving the cursor on a fresh line afterward.

		For mirroring a typed Input/Prompt response: on the real calculator
		you're typing this character by character at the cursor, and typing
		past column 16 wraps to the next row exactly like Output( does — it
		isn't a single precomputed value the way a Disp argument is, so it
		doesn't get Disp's truncate-with-ellipsis treatment.  Unlike disp(), it's
		allowed to fill the bottom row and leave it filled — only Disp guarantees
		a trailing blank line.
		"""
		col = 0
		for byte in text:
			if self.cursor_row >= self.ROWS:
				self._scroll()
				self.cursor_row = self.ROWS - 1
			self._cells[self.cursor_row][col] = byte
			col += 1
			if col >= self.COLS:
				col = 0
				self.cursor_row += 1
		if col != 0 or not text:
			self.cursor_row += 1

	def render(self) -> str:
		"""The grid as 8 lines of 16 characters (trailing blanks preserved)."""
		return '\n'.join(decode(bytes(row)) for row in self._cells)

	def __str__(self) -> str:
		return self.render()
