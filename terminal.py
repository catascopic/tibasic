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


class TerminalConsole(Console):
	"""Interactive command-line console for quick prototyping."""

	_SPINNER = '⠁⠂⠄⡀⢀⠠⠐⠈'   # one frame per redraw while a Pause is waiting
	_FRAME_SECONDS = 0.1        # spinner redraw interval (~10 fps)
	_POLL_SECONDS = 0.01        # how often we check for a keypress within a frame

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
		self._spin = 0

	def update(self, home: HomeScreen) -> None:
		self._render(home)

	def _render(self, home: HomeScreen, marker: str | None = None) -> None:
		# Repaint in place: home the cursor, overwrite each row clearing to its end,
		# then erase anything below.  This avoids \033[2J — which clears only the
		# viewport (old frames linger in scrollback) and is a no-op where VT is off —
		# so the screen never scrolls.  The trailing newline drops the cursor below
		# the grid so an Input prompt appears there (and is wiped on the next repaint).
		#
		# `marker`, if given, overlays one glyph in the top-right cell — standing in
		# for the real calculator's pause indicator, a small dotted icon that lives
		# in the status bar above the character grid (so real programs can't
		# Output( over it).  We have no separate status-bar row, so the top-right
		# cell is the closest stand-in; it's purely cosmetic and never touches
		# HomeScreen's actual content, so whatever a program wrote there reappears
		# untouched on the next plain update().
		rows = home.render().split('\n')
		if marker is not None:
			rows[0] = rows[0][:-1] + marker
		frame = '\033[H' + '\n'.join(f'{row}\033[K' for row in rows) + '\033[J\n'
		sys.stdout.write(frame)
		sys.stdout.flush()

	def read_value(self, prompt: str) -> str:
		return input(prompt)

	def read_key(self) -> int:
		# Best-effort non-blocking poll; returns 0 where the platform has no support.
		try:
			import msvcrt
		except ImportError:
			return 0
		return _TI_KEY_CODES.get(msvcrt.getwch(), 0) if msvcrt.kbhit() else 0

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
			self._render(home, marker=self._SPINNER[0])
			input()
			return
		sys.stdout.write('\033[?25l')   # hide the text cursor while animating
		sys.stdout.flush()
		try:
			while True:
				self._render(home, marker=self._SPINNER[self._spin % len(self._SPINNER)])
				self._spin += 1
				if self._wait_for_key(msvcrt, {'\r', ' '}):
					return
		finally:
			sys.stdout.write('\033[?25h')   # restore the cursor
			sys.stdout.flush()

	def _wait_for_key(self, msvcrt, accept: set[str]) -> bool:
		"""Poll for up to _FRAME_SECONDS; return True if a key in `accept` arrived.

		Other keys are consumed (so they don't pile up) but don't end the wait.
		"""
		deadline = time.monotonic() + self._FRAME_SECONDS
		while time.monotonic() < deadline:
			if msvcrt.kbhit() and msvcrt.getwch() in accept:
				return True
			time.sleep(self._POLL_SECONDS)
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
_TI_KEY_CODES: dict[str, int] = {}
