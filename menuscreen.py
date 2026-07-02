"""The Menu( screen — the third screen type, beside the home and graph screens.

Unlike them it isn't persistent state a program writes to over time: it's a transient
modal that exists only for the duration of one Menu( call and, while up, covers
whatever screen was there (see env.menu / Screen.MENU).  But it still has a canonical
representation — an inverted title bar, numbered options, the highlighted selection —
and canonical navigation — up/down wrap around, a number key jumps, Enter commits.
Both are modeled here so every frontend renders and navigates the menu identically.

A frontend owns only the input and the blocking loop: it maps real keypresses onto
up()/down()/choose() and repaints styled_rows() until it commits.  How "inverted" is
shown (ANSI inverse video, flipped pixels, CSS) is the frontend's call — the model
just says *which* spans are inverted.
"""
from core import TiString
from tokenbase import decode
from bitmap import Bitmap

ROWS = 8     # a menu uses the home screen's 16×8 character grid
COLS = 16


def _display_bytes(ts: TiString) -> bytes:
	"""A TiString's display bytes — the concatenated glyph bytes of its tokens."""
	return b''.join(t.display for t in ts.tokens)


class MenuScreen:

	ROWS = ROWS
	COLS = COLS

	def __init__(self, title: TiString, options: list[TiString]):
		self.title = title
		self.options = options
		self.selected = 0

	def up(self) -> None:
		self.selected = (self.selected - 1) % len(self.options)

	def down(self) -> None:
		self.selected = (self.selected + 1) % len(self.options)

	def choose(self, index: int) -> None:
		"""Jump the highlight straight to `index` (a number-key selection)."""
		self.selected = index

	def styled_rows(self) -> list[list[tuple[bytes, bool]]]:
		"""The canonical layout: ROWS rows, each a list of (display-bytes, inverted)
		segments.

		The title bar is inverted; on the selected option only its "N:" prefix is.
		Options past the title fill one row each, truncated to the 16 columns, and
		the rest of the screen is padded with blank rows.  Segments are display bytes
		(the model's native form) — a frontend decodes them to characters (terminal)
		or blits them through the font (print_screen).
		"""
		title = _display_bytes(self.title)[:COLS]
		rows = [[(title, True), (b' ' * (COLS - len(title)), False)]]
		for i, option in enumerate(self.options):
			prefix = f'{i + 1}:'.encode('ascii')
			body = _display_bytes(option)[:COLS - len(prefix)].ljust(COLS - len(prefix))
			rows.append([(prefix, i == self.selected), (body, False)])
		rows += [[(b' ' * COLS, False)] for _ in range(ROWS - len(rows))]
		return rows

	def rows(self) -> list[str]:
		"""Each row's plain text (styling dropped, bytes decoded) — for simple
		frontends and tests."""
		return [decode(b''.join(text for text, _ in row)) for row in self.styled_rows()]

	def print_screen(self, path, pixel_size: int = 1) -> None:
		"""Save the menu as a monochrome BMP, like home/graph: each cell's display
		byte is rasterized through the large font into the shared 96×64 Bitmap, with
		the inverted title and selection drawn as flipped pixels (fonts.blit_cell)."""
		surface = Bitmap()
		for r, row in enumerate(self.styled_rows()):
			c = 0
			for text, inverted in row:
				for byte in text:
					surface.blit_cell(r, c, byte, invert=inverted)
					c += 1
		surface.print_screen(path, pixel_size)
