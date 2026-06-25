from titoken import decode
from tiformat import disp_lines

# Display byte for '…' (0xCE) — what Disp truncates a too-long line with, as its
# 16th column.  An empty cell holds 0x00, which decodes to a blank glyph.
_ELLIPSIS = 0xCE
_BLANK = 0x00


class HomeScreen:
	"""The TI home screen, modeled as one append-only scrollback of byte rows.

	Pure calculator state, a peer of GraphScreen — the I/O commands mutate it and a
	Console renders it.  `lines` holds every row ever written (each 16 display bytes,
	one glyph per cell, 0x00 = blank); the live 8-row window is just a view of it.
	Two indices pick out the window and the write position:

	  * `top` — the buffer index shown at the top of the window.  ClrHome sets it to
	    len(lines), so the whole current screen scrolls out of view at once with
	    nothing appended; the history (and the Disp `values` log) are kept.  A Disp
	    that fills the window scrolls by bumping `top`, not by storing blank rows.
	  * `cursor_row` — how many lines Disp has taken up since the last clear/scroll
	    (0..7).  *This*, not the buffer length, controls when the screen scrolls:
	    Output( can poke a row at the bottom without making Disp think the window is
	    full, so you can ClrHome, Output( to row 8, then Disp seven times before it
	    scrolls on the eighth.

	The window is `lines[top : top+8]`, padded blank past the end — so the blank line
	Disp keeps at the bottom is a rendering consequence, never a stored row.  Rows are
	materialized (as null bytes) only when something actually writes into them: Disp
	at the cursor, or Output( reaching down into the not-yet-written region.

	`values` is a host-only log of the actual TiValues Disp'd (not their text), so a
	free-form frontend can re-render the full history at full width; the 16-wide
	`lines` buffer is the faithful screen, `values` is the lossless record.  Both
	survive ClrHome.  `version` bumps on every mutation — the cheap signal a frontend
	diffs against to decide whether a repaint is worth doing.

	Text is stored as display bytes (from tiformat / token.display) — exactly what a
	canvas frontend draws through the font tables; render() decodes back to characters
	for the terminal frontend.
	"""

	ROWS = 8
	COLS = 16

	def __init__(self):
		self.values: list = []          # every value Disp'd, in order (host-only)
		self.lines: list[bytearray] = []  # the byte scrollback; window is its tail
		self.top = 0                    # buffer index at the top of the window
		self.cursor_row = 0             # Disp lines used since clear/scroll (0..7)
		self.version = 0

	def _blank(self) -> bytearray:
		return bytearray(self.COLS)

	def _row(self, index: int) -> bytearray:
		"""The buffer row at `index`, materializing blank rows up to it on demand."""
		while len(self.lines) <= index:
			self.lines.append(self._blank())
		return self.lines[index]

	def disp(self, value) -> None:
		"""Disp a value: log it, then place its line(s) on the grid.

		The actual value goes to `values` (the lossless host-only record); the grid
		gets it right-aligned and truncated to the 16-column window via write_line.
		This is the high-level entry the Disp command calls; write_line/output are the
		byte-level grid primitives beneath it.
		"""
		self.values.append(value)
		for line in disp_lines(value, self.COLS):
			self.write_line(line)

	def write_line(self, text: bytes) -> None:
		"""Write one line of display bytes at the Disp cursor and drop to the next,
		scrolling (top += 1) once the bottom is passed.  A line too long for the 16
		columns is truncated with an ellipsis as the 16th character, rather than
		wrapping — that's how the real calculator's Disp behaves; Output( and echo()
		are the ones that wrap.

		Always leaves at least one blank line visible at the bottom: if this call
		writes the last row, it scrolls again immediately afterward rather than
		deferring to whatever writes next.  echo() (Input/Prompt) is exempt — it's
		allowed to fill the bottom row and leave it filled.  The target row is cleared
		before writing, so a short Disp overwrites whatever Output( left there.
		"""
		if self.cursor_row >= self.ROWS:
			self.top += 1
			self.cursor_row = self.ROWS - 1
		if len(text) > self.COLS:
			text = text[:self.COLS - 1] + bytes([_ELLIPSIS])
		row = self._row(self.top + self.cursor_row)
		row[:] = self._blank()
		row[:len(text)] = text
		self.cursor_row += 1
		if self.cursor_row >= self.ROWS:
			self.top += 1
			self.cursor_row = self.ROWS - 1
		self.version += 1

	def output(self, row: int, col: int, text: bytes) -> None:
		"""Write `text` from window position (row, col), wrapping to following rows
		and clipping at the bottom edge — matching the calculator, which silently
		drops overflow.  Does not touch the Disp cursor: a positioned write doesn't
		count toward when Disp scrolls."""
		index = row * self.COLS + col
		for byte in text:
			if index >= self.ROWS * self.COLS:
				break
			r, c = divmod(index, self.COLS)
			self._row(self.top + r)[c] = byte
			index += 1
		self.version += 1

	def echo(self, text: bytes) -> None:
		"""Write `text` starting at the cursor, wrapping to the next row instead
		of truncating, and leaving the cursor on a fresh line afterward.

		For mirroring a typed Input/Prompt response: on the real calculator you're
		typing this character by character at the cursor, and typing past column 16
		wraps to the next row exactly like Output( does — it isn't a single
		precomputed value the way a Disp argument is, so it doesn't get Disp's
		truncate-with-ellipsis treatment.  Unlike write_line, it's allowed to fill the
		bottom row and leave it filled — only Disp guarantees a trailing blank line.
		"""
		col = 0
		for byte in text:
			if self.cursor_row >= self.ROWS:
				self.top += 1
				self.cursor_row = self.ROWS - 1
			self._row(self.top + self.cursor_row)[col] = byte
			col += 1
			if col >= self.COLS:
				col = 0
				self.cursor_row += 1
		if col != 0 or not text:
			self.cursor_row += 1
		self.version += 1

	def clear(self) -> None:
		"""ClrHome — scroll the current screen out of view by marking the window start
		past the end of the buffer.  Nothing is appended; the scrollback and the Disp
		`values` log are kept, the visible window simply goes blank and Disp restarts
		at the top."""
		self.top = len(self.lines)
		self.cursor_row = 0
		self.version += 1

	def blank_window(self) -> None:
		"""Wipe the 8 visible rows in place, without scrolling — for a frontend that
		repaints the window many times over (a Pause scroll view), where ClrHome's
		scroll-into-history would flood the buffer with a screen per frame."""
		for r in range(self.ROWS):
			self._row(self.top + r)[:] = self._blank()
		self.version += 1

	def _window_row(self, r: int) -> bytearray:
		"""The bytes shown at window row r (a fresh blank if past the buffer end —
		read-only, so the unwritten blank bottom line isn't materialized)."""
		index = self.top + r
		return self.lines[index] if index < len(self.lines) else self._blank()

	def render(self) -> str:
		"""The visible window as 8 lines of 16 characters (trailing blanks preserved)."""
		return '\n'.join(decode(bytes(self._window_row(r))) for r in range(self.ROWS))

	def print_screen(self, path, pixel_size: int = 1) -> None:
		"""Save the visible window as a monochrome BMP.

		Each cell's glyph is rasterized through the large font (fonts.blit_cell) into
		a 6×8 pixel block, so the 16×8 window fills exactly the same 96×64 LCD the graph
		uses — the home screen and the graph are two views of one Bitmap.
		"""
		from bitmap import Bitmap
		surface = Bitmap()
		for r in range(self.ROWS):
			row = self._window_row(r)
			for c in range(self.COLS):
				surface.blit_cell(r, c, row[c])
		surface.print_screen(path, pixel_size)

	def __str__(self) -> str:
		return self.render()
