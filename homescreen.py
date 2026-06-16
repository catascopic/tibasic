class HomeScreen:
	"""The TI home screen: a 16-column × 8-row grid of characters with a Disp cursor.

	Pure state, like Graph — the I/O commands mutate it and a Console renders it.
	Output( writes at an absolute (row, col); Disp appends on the cursor line and
	scrolls; ClrHome wipes it.  Coordinates here are 0-indexed; the commands
	translate from TI's 1-indexed Output(row, col) and validate the range.
	"""

	ROWS = 8
	COLS = 16

	def __init__(self):
		self.clear()

	def clear(self) -> None:
		self._cells = [[' '] * self.COLS for _ in range(self.ROWS)]
		self.cursor_row = 0

	def output(self, row: int, col: int, text: str) -> None:
		"""Write `text` from (row, col), wrapping to following rows and clipping at
		the bottom edge — matching the calculator, which silently drops overflow."""
		index = row * self.COLS + col
		for ch in text:
			if index >= self.ROWS * self.COLS:
				break
			self._cells[index // self.COLS][index % self.COLS] = ch
			index += 1

	def disp(self, text: str) -> None:
		"""Append `text` on the cursor line (left-aligned, clipped to 16) and drop
		to the next line, scrolling the grid up once the bottom is passed."""
		if self.cursor_row >= self.ROWS:
			self._cells.pop(0)
			self._cells.append([' '] * self.COLS)
			self.cursor_row = self.ROWS - 1
		for col, ch in enumerate(text[:self.COLS]):
			self._cells[self.cursor_row][col] = ch
		self.cursor_row += 1

	def render(self) -> str:
		"""The grid as 8 lines of 16 characters (trailing blanks preserved)."""
		return '\n'.join(''.join(row) for row in self._cells)

	def __str__(self) -> str:
		return self.render()
