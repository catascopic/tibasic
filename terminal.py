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

from homescreen import HomeScreen


class Console(ABC):

	@abstractmethod
	def update(self, home: HomeScreen) -> None:
		"""Re-render the home screen after a Disp / Output( / ClrHome."""

	@abstractmethod
	def read_value(self, prompt: str) -> str:
		"""Blocking: show `prompt`, return the raw text the user typed (Input/Prompt)."""

	@abstractmethod
	def read_key(self) -> int:
		"""Non-blocking: the TI key code currently pressed, or 0 for none (getKey)."""

	@abstractmethod
	def pause(self, home: HomeScreen) -> None:
		"""Blocking: render `home` and wait for the user to continue (Pause)."""

	@abstractmethod
	def choose(self, title: str, options: list[str]) -> int:
		"""Blocking: present a menu, return the chosen 0-based index (Menu()."""


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

	def read_value(self, prompt: str) -> str:
		if not self.inputs:
			raise RuntimeError(f"ScriptedConsole: no input queued for prompt {prompt!r}")
		return self.inputs.pop(0)

	def read_key(self) -> int:
		return self.keys.pop(0) if self.keys else 0

	def pause(self, home: HomeScreen) -> None:
		self.frames.append(home.render())

	def choose(self, title: str, options: list[str]) -> int:
		if not self.choices:
			raise RuntimeError(f"ScriptedConsole: no choice queued for menu {title!r}")
		return self.choices.pop(0)


_PAUSE_SPINNER   = '▚▞'   # one frame per redraw while a Pause is waiting
_RUNNING_SPINNER = '▙▛▜▟'         # one frame per redraw while idle-polling (getKey)
_FRAME_SECONDS = 0.1                # spinner redraw interval (~10 fps)
_POLL_SECONDS = 0.01                # how often we check for a keypress within a frame
_BOUNDARY = '█'

class TerminalConsole(Console):
	"""Interactive command-line console for quick prototyping."""

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
		self._pause_spin = 0
		self._run_spin = 0
		self._last_home: HomeScreen | None = None
		self._last_run_render = 0.0

	def update(self, home: HomeScreen) -> None:
		self._render(home)

	def _render(self, home: HomeScreen, marker: str | None = None) -> None:
		# Repaint in place: home the cursor, overwrite each row clearing to its end,
		# then erase anything below.  This avoids \033[2J — which clears only the
		# viewport (old frames linger in scrollback) and is a no-op where VT is off —
		# so the screen never scrolls.  The trailing newline drops the cursor below
		# the grid so an Input prompt appears there (and is wiped on the next repaint).
		#
		# `marker`, if given, overlays one glyph into the top border, one cell left
		# of the corner — standing in for the real calculator's status-bar icons
		# (pause indicator, run indicator), which live above the character grid,
		# not in it.  Putting it in the border rather than a content cell means it
		# can never collide with anything a program actually Output(s, and there's
		# nothing to restore afterward: the next plain update() just redraws the
		# border without it.
		self._last_home = home
		framed = [(_BOUNDARY * (HomeScreen.COLS + 1)) + (marker or _BOUNDARY)]
		for row in home.render().split('\n'):
			framed.append(_BOUNDARY + row + _BOUNDARY)
		framed.append(_BOUNDARY * (HomeScreen.COLS + 2))
		frame = '\033[H' + '\n'.join(f'{row}\033[K' for row in framed) + '\033[J\n'
		sys.stdout.write(frame)
		sys.stdout.flush()

	def read_value(self, prompt: str) -> str:
		return input(prompt)

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
		if not msvcrt.kbhit():
			self._tick_running_indicator()
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
		"""Animate the border's running indicator while idle-polling for a key.

		Distinguishes "the program is alive and looping" (e.g. inside
		`Repeat getKey…End`) from Pause's "stopped, waiting for you" — a lighter
		glyph, and not redrawn on every single poll (read_key can be called
		thousands of times a second), only at the same ~10fps pace as Pause's
		animation.  No-op before anything has ever been rendered.
		"""
		if self._last_home is None:
			return
		now = time.monotonic()
		if now - self._last_run_render < _FRAME_SECONDS:
			return
		self._last_run_render = now
		self._render(self._last_home, marker=_RUNNING_SPINNER[self._run_spin % len(_RUNNING_SPINNER)])
		self._run_spin += 1

	def pause(self, home: HomeScreen) -> None:
		"""Animate the spinner in real time until Enter or Space is pressed.

		Needs msvcrt for non-blocking key polling (Windows-only) AND a real
		interactive terminal: msvcrt.kbhit() polls the console's own input buffer,
		not redirected stdin, so it would never see piped/redirected input and spin
		forever.  In either unsupported case this falls back to one static frame
		plus a plain blocking input(), which correctly reads from a pipe.
		"""
		try:
			import msvcrt
		except ImportError:
			msvcrt = None
		if msvcrt is None or not sys.stdin.isatty():
			self._render(home, marker=_PAUSE_SPINNER[0])
			input()
			return
		sys.stdout.write('\033[?25l')   # hide the text cursor while animating
		sys.stdout.flush()
		try:
			while True:
				self._render(home, marker=_PAUSE_SPINNER[self._pause_spin % len(_PAUSE_SPINNER)])
				self._pause_spin += 1
				if self._wait_for_key(msvcrt, {'\r', ' '}):
					return
		finally:
			sys.stdout.write('\033[?25h')   # restore the cursor
			sys.stdout.flush()

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
		print(title)
		for i, option in enumerate(options, 1):
			print(f'{i}: {option}')
		while True:
			try:
				choice = int(input('> '))
			except ValueError:
				continue
			if 1 <= choice <= len(options):
				return choice - 1


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
	'\r': 105,   # Enter

	'A': 41, 'B': 42, 'C': 43, 
	'D': 51, 'E': 52, 'F': 53, 'G': 54, 'H': 55, 
	'I': 61, 'J': 62, 'K': 63, 'L': 64, 'M': 65, 
	'N': 71, 'O': 72, 'P': 73, 'Q': 74, 'R': 75,
	'S': 81, 'T': 82, 'U': 83, 'V': 84, 'W': 85, 
	'X': 91, 'Y': 92, 'Z': 93,

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
}
