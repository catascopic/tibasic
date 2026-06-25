from titoken import decode
from tiformat import disp_lines, value_lines

# Display byte for '…' (0xCE) — what Disp truncates a too-long line with, as its
# 16th column.  An empty cell holds 0x00, which decodes to a blank glyph.
_ELLIPSIS = 0xCE
_BLANK = 0x00


class HomeScreen:
	"""The TI home screen: a 16-column × 8-row grid of display bytes with a Disp cursor.

	Pure calculator state, a peer of GraphScreen — the I/O commands mutate it and a
	Console renders it.  Each cell holds one TI display byte (one glyph); an empty
	cell is 0x00.  Output( writes at an absolute (row, col), wrapping across rows;
	Disp appends on the cursor line and truncates with an ellipsis instead of
	wrapping; ClrHome wipes it.  Coordinates here are 0-indexed; the commands
	translate from TI's 1-indexed Output(row, col) and validate the range.

	Text arrives as display bytes (from tiformat / token.display), so what's stored
	is exactly what a canvas frontend draws through the font tables; render() decodes
	it back to characters for the terminal frontend.

	Beyond the live grid, the screen keeps two append-only logs of what Disp has
	shown — `values` (the actual TiValues) and `transcript` (their rendered text,
	full-width and never truncated to the 16 columns) — so a frontend can present
	the whole history (a free-form scroll) without being confined to the 8×16
	window, and nothing scrolled off the top is ever lost.  These are host-only;
	TI-BASIC can't see them.  ClrHome clears the grid but not the logs.

	`version` bumps on every grid mutation, the cheap signal a frontend diffs
	against to decide whether a repaint is worth doing.
	"""

	ROWS = 8
	COLS = 16

	def __init__(self):
		self.values: list = []        # every value Disp'd, in order (host-only)
		self.transcript: list[str] = []  # rendered Disp lines, full-width (host-only)
		self.version = 0
		self.clear()

	def clear(self) -> None:
		"""Wipe the grid (ClrHome).  The Disp logs persist — they're the permanent
		record of everything shown, independent of the live window."""
		self._cells = [bytearray([_BLANK] * self.COLS) for _ in range(self.ROWS)]
		self.cursor_row = 0
		self.version += 1

	def _scroll(self) -> None:
		"""Drop the top row and append a blank one at the bottom."""
		self._cells.pop(0)
		self._cells.append(bytearray([_BLANK] * self.COLS))

	def disp(self, value) -> None:
		"""Disp a value: record it in the logs, then place it on the grid.

		The actual value goes to `values` and its full-width rendered lines to
		`transcript` (the permanent, untruncated history a free-form view shows);
		the grid gets the same value right-aligned and truncated to the 16-column
		window via write_line.  This is the high-level entry the Disp command calls;
		write_line/output are the byte-level grid primitives beneath it.
		"""
		self.values.append(value)
		self.transcript.extend(decode(line) for line in value_lines(value))
		for line in disp_lines(value, self.COLS):
			self.write_line(line)

	def output(self, row: int, col: int, text: bytes) -> None:
		"""Write `text` from (row, col), wrapping to following rows and clipping at
		the bottom edge — matching the calculator, which silently drops overflow."""
		index = row * self.COLS + col
		for byte in text:
			if index >= self.ROWS * self.COLS:
				break
			self._cells[index // self.COLS][index % self.COLS] = byte
			index += 1
		self.version += 1

	def write_line(self, text: bytes) -> None:
		"""Append one line of display bytes on the cursor line and drop to the next,
		scrolling the grid up once the bottom is passed.  A line too long for the 16
		columns is truncated with an ellipsis as the 16th character, rather than
		wrapping — that's how the real calculator's Disp behaves; Output( and
		echo() are the ones that wrap.

		Always leaves at least one blank line visible at the bottom: if this call
		writes the last row, it scrolls again immediately afterward rather than
		deferring to whatever writes next.  echo() (Input/Prompt) is exempt — it's
		allowed to fill the bottom row and leave it filled.
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
		self.version += 1

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
		self.version += 1

	def render(self) -> str:
		"""The grid as 8 lines of 16 characters (trailing blanks preserved)."""
		return '\n'.join(decode(bytes(row)) for row in self._cells)

	def print_screen(self, path, pixel_size: int = 1) -> None:
		"""Save the home screen as a monochrome BMP — the character-grid counterpart
		to GraphScreen.print_screen.

		Each cell's glyph is rasterized through the large font into a 6×8 pixel block
		(5×7 glyph plus the font's 1px right/bottom spacing), so the 16×8 grid fills
		exactly the same 96×64 LCD the graph uses.  The pixel buffer is handed to
		GraphScreen.print_screen, which owns the BMP encoding.
		"""
		from graphscreen import GraphScreen
		from fonts import LARGE_FONT
		GLYPH_H, CELL_W, CELL_H = 7, 6, 8
		surface = GraphScreen()
		for r in range(self.ROWS):
			for c in range(self.COLS):
				glyph = LARGE_FONT[self._cells[r][c]]
				if not glyph:
					continue
				base_r, base_c = r * CELL_H, c * CELL_W
				for dc, colbits in enumerate(glyph):
					for dr in range(GLYPH_H):
						if (colbits >> (GLYPH_H - 1 - dr)) & 1:
							surface.set(base_r + dr, base_c + dc)
		surface.print_screen(path, pixel_size)

	def __str__(self) -> str:
		return self.render()
