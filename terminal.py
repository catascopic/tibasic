"""The boundary between the interpreter and a frontend (terminal, HTML, tests).

The interpreter owns the HomeScreen *state*; a Console only renders it and supplies
user input.  Console methods return raw results — typed text, a key code, a chosen
index — and the I/O commands apply the TI semantics (parse, store, Goto), so every
frontend stays dumb.

The five methods are deliberately the only suspension points: when the HTML
frontend needs a suspendable eval loop (a synchronous `While 1:getKey→K` can't
block the browser), read_value / read_key / pause / choose are the hooks to yield at.
"""
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

from core import TiString
from titoken import Token
from homescreen import HomeScreen
from scrollview import ScrollView


class Console(ABC):

	@abstractmethod
	def update(self, home: HomeScreen) -> None:
		"""Re-render the home screen after a Disp / Output( / ClrHome."""

	@abstractmethod
	def read_tokens(self, prompt: TiString | None, home: HomeScreen) -> list[Token]:
		"""Blocking: display prompt (if any) and return the tokens the user entered."""

	@abstractmethod
	def read_key(self) -> int:
		"""Non-blocking: the TI key code currently pressed, or 0 for none (getKey)."""

	@abstractmethod
	def pause(self, home: HomeScreen, value=None) -> None:
		"""Blocking: render `home` and wait for the user to continue (Pause).

		`value` is the paused value (already rendered onto `home` Disp-style).  A
		frontend may offer to page it with the arrow keys when it's a list/matrix too
		big for the screen — `ScrollView.of(value)` decides that; frontends that can't
		take key input just render the initial window."""

	@abstractmethod
	def choose(self, title: str, options: list[str]) -> int:
		"""Blocking: present a menu, return the chosen 0-based index (Menu()."""

	def finish(self, home: HomeScreen) -> None:
		"""Called when program execution ends; default is a no-op."""


class ScriptedConsole(Console):
	"""Headless console for tests and prototyping.

	Input is drawn from pre-loaded queues; each rendered frame is captured in
	`frames` instead of printed, so a program's I/O can be driven and asserted
	deterministically.  This is the default console on a fresh Environment.
	"""

	def __init__(self, inputs=(), keys=(), choices=()):
		self.inputs = list(inputs)
		self.keys = list(keys)
		self.choices = list(choices)
		self.frames: list[str] = []

	def update(self, home: HomeScreen) -> None:
		self.frames.append(home.render())

	def read_tokens(self, prompt: TiString | None, home: HomeScreen) -> list[Token]:
		if not self.inputs:
			raise ValueError(f"ScriptedConsole: no input queued")
		return TiString.from_str(self.inputs.pop(0)).tokens

	def read_key(self) -> int:
		return self.keys.pop(0) if self.keys else 0

	def pause(self, home: HomeScreen, value=None) -> None:
		if view := ScrollView.of(value):
			view.render_into(home)
		self.frames.append(home.render())

	def choose(self, title: str, options: list[str]) -> int:
		if not self.choices:
			raise ValueError(f"ScriptedConsole: no choice queued for menu {title!r}")
		return self.choices.pop(0)


# The border carries one status indicator that stands in for the calculator's
# busy/pause icons.  Which spinner shows says what the program is doing; that it
# animates says the program is alive.  A third state — no indicator at all — means
# the program is either waiting for the user to type (Input/Prompt) or finished.
_RUNNING_SPINNER = '▙▛▜▟'	# the program is running
_PAUSE_SPINNER   = '▚▞'		# the program is blocked on the user (Pause / Menu)
_FRAME_SECONDS = 0.1        # spinner advances one frame this often (~10 fps)
_POLL_SECONDS = 0.01        # how often we check for a keypress within a frame
_BOUNDARY = '░'


def _now_frame() -> int:
	"""Index of the current animation frame, ticking once per _FRAME_SECONDS off
	the wall clock.  Every spinner glyph and the getKey-loop repaint throttle key
	off this, so the animation is a function of *when* we paint — never of how many
	times we happen to — and all of it stays in lockstep with elapsed time."""
	return int(time.monotonic() / _FRAME_SECONDS)


def _spinner_frame(spinner: str) -> str:
	"""The glyph `spinner` should show right now (see _now_frame)."""
	return spinner[_now_frame() % len(spinner)]

# Extended scan-code letter (after the '\x00'/'\xe0' prefix) → scroll direction,
# for paging a scrollable Pause value (see _poll_scroll_key).
_ARROW_DIRECTIONS = {'H': 'up', 'P': 'down', 'K': 'left', 'M': 'right'}

# Inverse video for the menu's title bar and the selected item's "N:" marker —
# matches the real calculator showing the menu title in white-on-black.
_INV_ON  = '\033[7m'
_INV_OFF = '\033[27m'

class TerminalConsole(Console):
	"""Interactive command-line console for quick prototyping."""

	# The terminal's hardware cursor is kept hidden the whole time a program runs
	# (it would otherwise blink in the bottom-left corner where _paint parks it),
	# and revealed only while an input() is actually collecting keystrokes and once
	# the program finishes.  Class-level default so a bare __new__ instance (used
	# in menu-rendering tests) still has a sane value.
	_cursor_hidden = False

	def __init__(self):
		if sys.platform == 'win32':
			_enable_windows_vt()
		# stdout's inherited encoding can be a legacy codepage (e.g. cp1252) that
		# can't represent the spinner glyphs or the TI charset's Greek/symbol
		# characters (π, Σ, ², …); force UTF-8 so writing them never crashes.
		try:
			sys.stdout.reconfigure(encoding='utf-8', errors='replace')
		except (AttributeError, ValueError):
			pass
		self._last_home: HomeScreen | None = None
		self._last_run_frame = -1

	def update(self, home: HomeScreen) -> None:
		self._render(home, _spinner_frame(_RUNNING_SPINNER))

	def _render(self, home: HomeScreen, indicator: str | None = None) -> None:
		# Remembered so _tick_running_indicator can redraw the real home screen
		# while idle-polling getKey.  The menu screen (_menu_rows/_paint) never
		# touches this — it isn't the home screen and shouldn't be confused for it.
		self._last_home = home
		self._paint(home.render().split('\n'), indicator)

	def _paint(self, rows: list, indicator: str | None = None) -> None:
		"""Repaint `rows` (ROWS strings of COLS visible characters, ANSI styling
		allowed) framed in the border, with `indicator` optionally overlaid one
		cell left of the top-right corner.  Shared by the home screen and the menu
		screen — neither carries any state for the other.

		Repaints in place: home the cursor, overwrite each row clearing to its
		end, then erase anything below.  This avoids \033[2J — which clears only
		the viewport (old frames linger in scrollback) and is a no-op where VT is
		off — so the screen never scrolls.  The trailing newline drops the cursor
		below the grid so an Input prompt appears there (wiped on the next repaint).

		`indicator` is the calculator's status icon (the running or pause spinner);
		None means show none — the program is taking input or has finished.  It
		lives in the border, not a content cell, so it can never collide with
		anything a program Output(s, and nothing needs restoring afterward: the
		next repaint without it just redraws a plain border.
		"""
		framed = [(_BOUNDARY * (HomeScreen.COLS + 1)) + (indicator or _BOUNDARY)]
		for row in rows:
			framed.append(_BOUNDARY + row + _BOUNDARY)
		framed.append(_BOUNDARY * (HomeScreen.COLS + 2))
		painted = '\033[H' + '\n'.join(f'{row}\033[K' for row in framed) + '\033[J\n'
		sys.stdout.write(painted)
		# A painted frame parks the cursor in the corner, so hide it by default;
		# the two states that want it back (Input via _input_cursor, the finished
		# program via finish) re-show it explicitly after painting.
		self._hide_cursor()
		sys.stdout.flush()

	def _hide_cursor(self) -> None:
		if not self._cursor_hidden:
			sys.stdout.write('\033[?25l')
			self._cursor_hidden = True

	def _show_cursor(self) -> None:
		if self._cursor_hidden:
			sys.stdout.write('\033[?25h')
			self._cursor_hidden = False

	@contextmanager
	def _input_cursor(self):
		"""Reveal the cursor for the duration of an input() call, then hide it again
		so the program runs on with no blinking cursor."""
		self._show_cursor()
		sys.stdout.flush()
		try:
			yield
		finally:
			self._hide_cursor()
			sys.stdout.flush()

	def finish(self, home: HomeScreen) -> None:
		self._render(home)    # final repaint with no indicator — the program is done
		self._show_cursor()   # hand the cursor back to the terminal
		sys.stdout.flush()

	def read_tokens(self, prompt: TiString | None, home: HomeScreen) -> list[Token]:
		self._render(home)  # repaint without indicator before blocking on input
		with self._input_cursor():
			text = input(str(prompt) if prompt is not None else '')
		return TiString.from_str(text).tokens

	def read_key(self) -> int:
		"""Best-effort non-blocking poll; returns 0 where the platform has no support.

		Arrows (and other special keys) arrive from msvcrt as two characters: a
		prefix ('\\x00' or '\\xe0') then a scan code, not a single getwch() result —
		so a plain key needs one read, an extended one needs two.
		"""
		try:
			import msvcrt
		except ImportError:
			return 0
		# Tick on every call, not just when no key is found: while a key is held,
		# the OS auto-repeats it, so kbhit() keeps saying True and this branch
		# would otherwise never run — freezing the indicator for the whole hold,
		# then jumping when released.  _tick_running_indicator self-paces off
		# elapsed time, so calling it unconditionally just makes the scroll
		# genuinely continuous regardless of whether keys are flowing.
		self._tick_running_indicator()
		if not msvcrt.kbhit():
			# Tiny sleep so a tight `Repeat getKey…End` poll loop doesn't peg a CPU
			# core at 100% — far below human reaction time, so it costs nothing
			# perceptible while idling, but it's worth knowing it's here.
			time.sleep(_POLL_SECONDS)
			return 0
		ch = msvcrt.getwch()
		if ch in ('\x00', '\xe0'):
			# Once the prefix has arrived, the scan-code byte is guaranteed to
			# follow as the other half of the same key event — read it directly
			# rather than re-checking kbhit() first.  That check is wrong: the
			# second byte isn't necessarily buffered the instant we look (a modern
			# terminal forwards input through a pseudoconsole, which can add a
			# sliver of latency between the two), so kbhit() can still say False
			# in that gap — and re-checking would silently drop the byte, losing
			# the whole keypress.  A direct read here blocks for at most an
			# instant, which is a non-issue since the user already pressed a key.
			return _TI_EXTENDED_KEY_CODES.get(msvcrt.getwch(), 0)
		# .upper(): getwch() returns whatever case was actually typed (lowercase
		# without Caps Lock), but TI's ALPHA keys are uppercase-only, so the table
		# below only needs one entry per letter.
		return _TI_KEY_CODES.get(ch.upper(), 0)

	def _tick_running_indicator(self) -> None:
		"""Keep the running indicator animating during a getKey poll loop.

		A tight `Repeat getKey…End` never calls update(), so without this the
		indicator would freeze even though the program is plainly alive.  read_key
		can fire thousands of times a second, so repaint only when the frame index
		actually advances — the same ~10fps as every other spinner.  No-op before
		anything has been rendered.
		"""
		if self._last_home is None:
			return
		frame = _now_frame()
		if frame == self._last_run_frame:
			return
		self._last_run_frame = frame
		self._render(self._last_home, _spinner_frame(_RUNNING_SPINNER))

	def pause(self, home: HomeScreen, value=None) -> None:
		"""Animate the spinner in real time until Enter or Space is pressed.

		Needs msvcrt for non-blocking key polling (Windows-only) AND a real
		interactive terminal: msvcrt.kbhit() polls the console's own input buffer,
		not redirected stdin, so it would never see piped/redirected input and spin
		forever.  In either unsupported case this falls back to one static frame
		plus a plain blocking input(), which correctly reads from a pipe.

		When `value` overflows the screen (ScrollView.of returns a view), the arrow
		keys page it — each frame re-renders the current window before painting — and
		Enter/Space ends the Pause.  Without msvcrt/a terminal there's no scrolling:
		the initial window is shown and a plain input() waits.
		"""
		view = ScrollView.of(value)
		try:
			import msvcrt
		except ImportError:
			msvcrt = None
		if msvcrt is None or not sys.stdin.isatty():
			if view is not None:
				view.render_into(home)
			self._render(home, _PAUSE_SPINNER[0])   # static: no key polling to animate it
			with self._input_cursor():
				input()
			return
		# _paint keeps the cursor hidden while the spinner animates; no need to
		# manage it here — it stays hidden until an Input or the program ends.
		while True:
			if view is not None:
				view.render_into(home)
			self._render(home, _spinner_frame(_PAUSE_SPINNER))
			if view is None:
				if self._wait_for_key(msvcrt, {'\r', ' '}):
					return
			else:
				key = self._poll_scroll_key(msvcrt)
				if key == 'exit':
					return
				if key:
					view.move(key)

	def _poll_scroll_key(self, msvcrt):
		"""Poll for up to _FRAME_SECONDS; return 'exit' (Enter/Space), a scroll
		direction ('up'/'down'/'left'/'right'), or None if nothing relevant arrived.
		"""
		deadline = time.monotonic() + _FRAME_SECONDS
		while time.monotonic() < deadline:
			if msvcrt.kbhit():
				ch = msvcrt.getwch()
				if ch in ('\x00', '\xe0'):
					return _ARROW_DIRECTIONS.get(msvcrt.getwch())   # see read_key: don't re-check kbhit
				if ch in ('\r', ' '):
					return 'exit'
			time.sleep(_POLL_SECONDS)
		return None

	def _wait_for_key(self, msvcrt, accept: set[str]) -> bool:
		"""Poll for up to _FRAME_SECONDS; return True if a key in `accept` arrived.

		Other keys are consumed (so they don't pile up) but don't end the wait.
		"""
		deadline = time.monotonic() + _FRAME_SECONDS
		while time.monotonic() < deadline:
			if msvcrt.kbhit() and msvcrt.getwch() in accept:
				return True
			time.sleep(_POLL_SECONDS)
		return False

	def choose(self, title: str, options: list[str]) -> int:
		"""Render an actual bordered menu screen (title bar inverted, the
		selected item's "N:" inverted) and animate the Pause spinner while
		waiting — the menu is a genuine block-until-the-user-acts state, the
		same as Pause, not a polling loop like getKey.

		Up/Down move the highlighted item; a number key 1-7 jumps straight to
		that item and confirms it immediately, matching the real calculator;
		Enter confirms whichever item is currently highlighted.

		Falls back to a plain numbered prompt without msvcrt or a real terminal,
		for the same reason pause() does (msvcrt can't see piped/redirected input).
		"""
		try:
			import msvcrt
		except ImportError:
			msvcrt = None
		if msvcrt is None or not sys.stdin.isatty():
			print(title)
			for i, option in enumerate(options, 1):
				print(f'{i}: {option}')
			with self._input_cursor():
				while True:
					try:
						choice = int(input('> '))
					except ValueError:
						continue
					if 1 <= choice <= len(options):
						return choice - 1
		selected = 0
		# _paint keeps the cursor hidden while the menu animates; like Pause, the
		# menu never collects keystrokes through input(), so it stays hidden.
		while True:
			self._paint(self._menu_rows(title, options, selected), _spinner_frame(_PAUSE_SPINNER))
			result = self._poll_menu_key(msvcrt, len(options))
			if result is None:
				continue
			if result == 'up':
				selected = (selected - 1) % len(options)
			elif result == 'down':
				selected = (selected + 1) % len(options)
			elif result == 'enter':
				return selected
			else:                # a number key: direct, immediate selection
				return result
			sys.stdout.flush()

	def _menu_rows(self, title: str, options: list[str], selected: int) -> list:
		"""Build the menu's 8 display rows: an inverted title bar, then one
		numbered option per row with the selected item's "N:" inverted.

		Padding/truncation is done on *visible* width — the inverted spans embed
		invisible ANSI codes, so they're added after sizing the plain text to
		HomeScreen.COLS, never counted as part of it.
		"""
		title_text = title[:HomeScreen.COLS]
		title_row = _INV_ON + title_text + _INV_OFF + ' ' * (HomeScreen.COLS - len(title_text))
		rows = [title_row]
		for i, option in enumerate(options):
			prefix = f'{i + 1}:'
			body = option[:HomeScreen.COLS - len(prefix)]
			body = body.ljust(HomeScreen.COLS - len(prefix))
			if i == selected:
				prefix = _INV_ON + prefix + _INV_OFF
			rows.append(prefix + body)
		rows += [' ' * HomeScreen.COLS] * (HomeScreen.ROWS - len(rows))
		return rows

	def _poll_menu_key(self, msvcrt, n_options: int):
		"""Poll for up to _FRAME_SECONDS; returns None (nothing relevant), 'up',
		'down', 'enter', or an int 0..n_options-1 (a number key, chosen directly).
		"""
		deadline = time.monotonic() + _FRAME_SECONDS
		while time.monotonic() < deadline:
			if msvcrt.kbhit():
				ch = msvcrt.getwch()
				if ch in ('\x00', '\xe0'):
					ch2 = msvcrt.getwch()        # see read_key: don't re-check kbhit
					if ch2 == 'H':
						return 'up'
					if ch2 == 'P':
						return 'down'
				elif ch == '\r':
					return 'enter'
				elif ch.isdigit() and ch != '0':
					index = int(ch) - 1
					if index < n_options:
						return index
			time.sleep(_POLL_SECONDS)
		return None


def _enable_windows_vt() -> None:
	"""Turn on ANSI/VT escape processing for the Windows console.

	Windows interprets escape sequences only when ENABLE_VIRTUAL_TERMINAL_PROCESSING
	is set on the output handle, and Python doesn't set it.  Without this, the clear
	and cursor codes are ignored and the home screen just scrolls.  No-op on failure
	(e.g. output isn't a real console).
	"""
	import ctypes
	try:
		kernel32 = ctypes.windll.kernel32
		handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
		mode = ctypes.c_uint32()
		if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
			kernel32.SetConsoleMode(handle, mode.value | 0x0004)
	except Exception:
		pass


# Physical key → TI getKey code for the interactive terminal (best-effort; the
# scripted and HTML consoles supply key codes directly).
#
# Arrow codes (24/25/26/34) and Enter (105) match the standard TI-83+/84+ getKey
# table.  Letters/digits below are placeholders (0 = unmapped) — fill in real
# codes as you go; C and 7 are pre-filled from values you already confirmed.
# Looked up upper-cased (read_key does ch.upper()), so one entry covers both
# Caps Lock states.
_TI_KEY_CODES: dict[str, int] = {
	'A': 41, 'B': 42, 'C': 43, 
	'D': 51, 'E': 52, 'F': 53, 'G': 54, 'H': 55, 
	'I': 61, 'J': 62, 'K': 63, 'L': 64, 'M': 65, 
	'N': 71, 'O': 72, 'P': 73, 'Q': 74, 'R': 75,
	'S': 81, 'T': 82, 'U': 83, 'V': 84, 'W': 85, 
	'X': 91, 'Y': 92, 'Z': 93, '=': 95,
	' ': 102, '.': 103, '`': 104, '\r': 105,
	'\x08': 45,  # Backspace -> CLEAR
	'\x1b': 44,  # End       -> VARS
	'\x09': 22,  # Tab       -> MODE

	'7': 72, '8': 73, '9': 74,
	'4': 82, '5': 83, '6': 84, 
	'1': 92, '2': 93, '3': 94,
	'0': 102,
}

# Arrows (and other special keys) arrive as a prefix byte ('\x00' or '\xe0') then
# one of these classic DOS/Windows console scan-code letters.
_TI_EXTENDED_KEY_CODES: dict[str, int] = {
	'H': 25,   # Up
	'P': 34,   # Down
	'K': 24,   # Left
	'M': 26,   # Right
	'S': 23,   # Delete  -> DEL
	'G': 33,   # Home    -> STAT
	'R': 32,   # Insert  -> X,T,θ,n
	'I': 21,   # Page Up -> 2ND
	'O': 31,   # Page Dn -> ALPHA
	';': 11,   # F1
	'<': 12,   # F2
	'=': 13,   # F3
	'>': 14,   # F4
	'?': 15,   # F5
}
