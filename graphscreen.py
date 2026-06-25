from bitmap import Bitmap, ROWS, COLS   # ROWS/COLS re-exported: callers import them from here


class GraphScreen(Bitmap):
	"""The calculator's graph screen — the 96×64 LCD bitmap driven as pixels.

	This is the surface the graph-mode and drawing commands (Pxl-/Pt-/Line/Circle/
	DrawF/Text/…) render onto.  It's just a Bitmap plus one piece of graph state:
	`valid`, whether the currently plotted functions are up to date (the graph is
	re-plotted lazily, only when displayed — see Environment.regraph).  The home and
	menu screens reuse the same Bitmap independently, addressing it as a character
	grid rather than subclassing it.
	"""

	def __init__(self):
		super().__init__()
		self.valid = False

	def clear(self) -> None:
		super().clear()
		self.valid = False
