"""A scrollable view of a paused list/matrix value.

When Pause is given a list or matrix that overflows the home screen, the user can
scroll it with the arrow keys.  A ScrollView holds the value's full content lines
(from tiformat.value_lines) plus a (row, col) scroll offset, and paints the
visible window into the home screen with edge indicators:

  * a left ellipsis (…) on a row whose content is scrolled off to the left, and a
    right ellipsis on a row with more content to the right;
  * up/down arrows at the right edge of the first and last content rows when the
    value is taller than the viewport — these override the right ellipsis.

The window is a plain character window over the formatted bytes (scrolling is
per-character horizontally and per-row vertically), and indicators overlay the
content cells at the edges rather than taking dedicated columns.  The bottom
screen row is always left blank, matching Pause/Disp.
"""
from tiformat import value_lines
from homescreen import HomeScreen

_ELLIPSIS = b'\xCE'  # …
_UP       = b'\x1E'  # ↑
_DOWN     = b'\x1F'  # ↓


class ScrollView:

	# The value fills every row but the last, which Pause/Disp keep blank.
	VIEW_ROWS = HomeScreen.ROWS - 1

	def __init__(self, value):
		self.lines = value_lines(value)
		self.row_off = 0
		self.col_off = 0

	@property
	def content_height(self) -> int:
		return len(self.lines)

	@property
	def content_width(self) -> int:
		return max(len(line) for line in self.lines)

	@property
	def scrollable(self) -> bool:
		"""Whether the value overflows the viewport in either direction."""
		return self.content_height > self.VIEW_ROWS or self.content_width > HomeScreen.COLS

	@property
	def can_up(self) -> bool:
		return self.row_off > 0

	@property
	def can_down(self) -> bool:
		return self.row_off + self.VIEW_ROWS < self.content_height

	@property
	def can_left(self) -> bool:
		return self.col_off > 0

	@property
	def can_right(self) -> bool:
		return self.col_off + HomeScreen.COLS < self.content_width

	def move(self, direction: str) -> None:
		"""Scroll one step in 'up'/'down'/'left'/'right', clamped at the edges.
		Any other value (or a blocked direction) is a no-op."""
		if direction == 'up' and self.can_up:
			self.row_off -= 1
		elif direction == 'down' and self.can_down:
			self.row_off += 1
		elif direction == 'left' and self.can_left:
			self.col_off -= 1
		elif direction == 'right' and self.can_right:
			self.col_off += 1

	def render_into(self, home: HomeScreen) -> None:
		"""Paint the current window (with edge indicators) onto `home`."""
		home.clear()
		visible = self.lines[self.row_off:self.row_off + self.VIEW_ROWS]
		right = HomeScreen.COLS - 1
		for r, line in enumerate(visible):
			home.output(r, 0, line[self.col_off:self.col_off + HomeScreen.COLS])
			if self.can_left:
				home.output(r, 0, _ELLIPSIS)
			if self.can_right:
				home.output(r, right, _ELLIPSIS)
		# Vertical arrows live at the right edge of the top/bottom content rows,
		# overriding any right ellipsis there.  Only a too-tall value shows them.
		if self.content_height > self.VIEW_ROWS:
			if self.can_up:
				home.output(0, right, _UP)
			if self.can_down:
				home.output(self.VIEW_ROWS - 1, right, _DOWN)
		home.cursor_row = len(visible)
