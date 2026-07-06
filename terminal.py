"""The boundary between the interpreter and a frontend (terminal, HTML, tests).

The interpreter owns the *state* — the HomeScreen and GraphScreen on `env` — and a
Console only renders it and supplies user input.  Output is observation, not
notification: a command mutates the model and calls `present()`, a payload-free
"a good moment to repaint"; the console pulls whatever it needs from `self.env`
(home grid, Disp values log, graph pixels) and paints the view it wants.  Input is
the irreducible part — the console must block and ask — so read_tokens / read_key /
pause / menu are real request-response methods and the only suspension points (when
the HTML frontend needs a suspendable eval loop, these are the hooks to yield at).

The console reaches the model through `self.env`, attached by Environment when the
console is assigned, so no method has to be handed the screen.
"""
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

from core import TiString
from tokenbase import Token
from tiformat import disp_lines, value_lines
from modes import Screen
from homescreen import HomeScreen
from graphscreen import COLS as _GRAPH_COLS
from scrollview import ScrollView


# _CHARSET: TI-83+ display byte → Unicode character for rendering the home-screen
# grid.  This is the canonical frontend mapping — the authoritative glyph for each
# byte value.  The simplified debug-only version in tokenbase.py is used for Token
# repr output and is not suitable for display.  None = undefined/unused slot.
_CHARSET: list[str | None] = [
#	0		1		2		3		4		5		6		7		8		9		A		B		C		D		E		F
	' ',	'𝑛',	'𝑢',	'𝑣',	'𝑤',	'►',	'🡅',	'🡇',	'∫',	'×',	'▫',	'﹢',	'·',	'ₜ',		'𝟃',	'𝟊',	# 0
	'√',	'¹',	'²',	'∠',	'°',	'ʳ',	'ᵀ',	'≤',	'≠',	'≥',	'¯',    'ᴇ',    '→',	'⑽',	'↑',	'↓',	# 1
	' ',	'!',	'"',	'#',	'⁴',		'%',	'&',	"'",	'(',	')',	'*',	'+',	',',	'-',	'.',	'/',	# 2
	'0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	'8',	'9',	':',	';',	'<',	'=',	'>',	'?',	# 3
	'@',	'A',	'B',	'C',	'D',	'E',	'F',	'G',	'H',	'I',	'J',	'K',	'L',	'M',	'N',	'O',	# 4
	'P',	'Q',	'R',	'S',	'T',	'U',	'V',	'W',	'X',	'Y',	'Z',	'θ',	'\\',	']',	'^',	'_',	# 5
	'`',	'a',	'b',	'c',	'd',	'e',	'f',	'g',	'h',	'i',	'j',	'k',	'l',	'm',	'n',	'o',	# 6
	'p',	'q',	'r',	's',	't',	'u',	'v',	'w',	'x',	'y',	'z',	'{',	'|',	'}',	'~',	'≛',	# 7
	'₀',		'₁',		'₂',		'₃',		'₄',		'₅',		'₆',		'₇',		'₈',		'₉',		'Á',	'À',	'Â',	'Ä',	'á',	'à',	# 8
	'â',	'ä',	'É',	'È',	'Ê',	'Ë',	'é',	'è',	'ê',	'ë',	'Í',	'Ì',	'Î',	'Ï',	'í',	'ì',	# 9
	'î',	'ï',	'Ó',	'Ò',	'Ô',	'Ö',	'ó',	'ò',	'ô',	'ö',	'Ú',	'Ù',	'Û',	'Ü',	'ú',	'ù',	# A
	'û',	'ü',	'Ç',	'ç',	'Ñ',	'ñ',	'´',	'ˋ',	'¨',	'¿',	'¡',	'α',	'β',	'γ',	'Δ',	'δ',	# B
	'ε',	'[',	'λ',	'μ',	'π',	'ρ',	'Σ',	'σ',	'τ',	'φ',	'Ω',	'ẍ',	'ȳ',	'ˣ',	'…',	'◄',	# C
	None,	None,	None,	None,	None,	'³',	'\n',	'𝑖',		'ṕ',	'χ',	'𝐅',	'𝑒',		'ʟ',	'𝐍',	'⸩',		'🡆',	# D
	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# E
	None,	None,	'$',	None,	'ß',	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	None,	# F
]

def decode(display: bytes) -> str:
	"""Render display bytes as a human-readable string (undefined bytes as \\xNN)."""
	return ''.join(_CHARSET[b] for b in display)


class Console(ABC):

	env = None   # the Environment, attached by its `console` setter

	def present(self) -> None:
		"""A repaint opportunity — output changed (Disp/Output(/ClrHome/a drawing
		command).  Payload-free: pull from self.env.  Default is a no-op."""

	def read_tokens(self, prompt: list[Token]) -> list[Token]:
		"""Blocking: display the prompt (if any), read the typed entry, and mirror it
		onto the home grid so the screen reflects the interaction.

		The raw read is the per-console `_input`; `_echo_entry` then writes the
		prompt+entry back onto the model the way the calculator shows it.  The mirror
		is needed only because input is taken out-of-band (the host's input line)
		rather than typed straight into the grid — it's a frontend concern, not the
		Input/Prompt command's, so it lives here."""
		tokens = self._input(prompt)
		self._echo_entry(prompt, tokens)
		return tokens

	@abstractmethod
	def _input(self, prompt: list[Token]) -> list[Token]:
		"""Blocking raw read: display prompt (if any) and return the entered tokens.

		Implementations tokenize the typed text through `_tokenize`, so a caller gets a
		well-formed token list and never a raw tokenizer error."""

	def _echo_entry(self, prompt: list[Token], tokens: list[Token]) -> None:
		"""Mirror the entered prompt+text onto the home grid and repaint, exactly as
		the calculator shows a typed entry.  The prompt's own bytes precede the typed
		bytes.  A frontend that doesn't paint the grid (FreeFormConsole, whose host
		terminal already echoes the typed line) overrides this to do nothing."""
		if not tokens:
			return
		prompt_bytes = b''.join(t.display for t in prompt)
		self.env.home.echo(prompt_bytes + b''.join(t.display for t in tokens))
		self.present()

	@staticmethod
	def _tokenize(text: str) -> list[Token]:
		"""Tokenize typed input.  A character the calculator's keypad can't produce (e.g.
		a pasted multi-char function name) is a host-level problem — it can't happen on
		real hardware and only arises because a frontend accepts free-form text — so it
		surfaces as a plain ValueError, not a TiSyntaxError.  TiSyntaxError is reserved for
		genuine parse failures of valid tokens; the tokenizer signals an un-typeable
		character with KeyError."""
		try:
			return TiString.from_str(text).tokens
		except KeyError as bad:
			raise ValueError(f"input contains an un-typeable character: {bad}")

	@abstractmethod
	def read_key(self) -> int:
		"""Non-blocking: the TI key code currently pressed, or 0 for none (getKey)."""

	@abstractmethod
	def pause(self, value=None) -> None:
		"""Blocking: show `value` (if any) and wait for the user to continue (Pause).

		A frontend may offer to page it with the arrow keys when it's a list/matrix
		too big for the screen — `ScrollView.of(value)` decides that; frontends that
		can't take key input just render the initial window."""

	@abstractmethod
	def menu(self) -> int:
		"""Blocking: present env.menu (the active MenuScreen) and return the chosen
		0-based index (Menu().  The model holds the title/options/highlight; a frontend
		maps input onto menu.up()/down()/choose() and renders menu.styled_rows()."""

	def finish(self) -> None:
		"""Called when program execution ends; default is a no-op."""

	def _render_pause_value(self, value) -> None:
		"""Place a Pause value onto the home grid Disp-style, but without logging it
		to the Disp `values` log (Pause shows a value, it isn't a Disp).  A list/matrix
		too big for the screen is left for the caller to page via ScrollView instead.
		Shared by every console's pause()."""
		if value is not None and ScrollView.of(value) is None:
			for line in disp_lines(value, self.env.home.COLS):
				self.env.home.write_line(line)


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

	def _capture(self) -> None:
		"""Snapshot the active screen — home grid or graph — into `frames`."""
		if self.env.screen is Screen.GRAPH:
			self.frames.append('\n'.join(self.env.graph.rows()))
		else:
			self.frames.append(str(self.env.home))

	def present(self) -> None:
		self._capture()

	def _input(self, prompt: list[Token]) -> list[Token]:
		if not self.inputs:
			raise ValueError("ScriptedConsole: no input queued")
		return self._tokenize(self.inputs.pop(0))

	def read_key(self) -> int:
		return self.keys.pop(0) if self.keys else 0

	def pause(self, value=None) -> None:
		self._render_pause_value(value)
		if view := ScrollView.of(value):
			view.render_into(self.env.home)
		self._capture()

	def menu(self) -> int:
		if not self.choices:
			raise ValueError(f"ScriptedConsole: no choice queued for menu {self.env.menu.title!r}")
		return self.choices.pop(0)


class FreeFormConsole(Console):
	"""A grid-less frontend: a plain stdout stream and a blocking input().

	The home screen still exists and is fully maintained in the model; this console
	just doesn't paint its 8×16 window.  Instead it streams the Disp *values* —
	re-rendered at full width via value_lines, so output isn't confined to 16
	columns (the byte grid is the faithful 16-wide screen; `values` is the lossless
	log).  `present()` prints whatever values have appeared since last time (tracked
	by `_printed`), so each Disp shows up once as it happens.  Output( writes to the
	(unpainted) grid like any other frontend; getKey isn't supported.
	"""

	def __init__(self):
		self._printed = 0

	def present(self) -> None:
		values = self.env.home.values
		for value in values[self._printed:]:
			print('\n'.join(decode(line) for line in value_lines(value)))
		self._printed = len(values)

	def _input(self, prompt: list[Token]) -> list[Token]:
		return self._tokenize(input(decode(b''.join(t.display for t in prompt))))

	def _echo_entry(self, prompt: list[Token], tokens: list[Token]) -> None:
		pass   # the host terminal echoes the typed line; FreeForm never paints the grid

	def read_key(self) -> int:
		return 0   # no key support in free-form mode

	def pause(self, value=None) -> None:
		self.present()   # flush any pending Disp output first
		if value is not None:
			print('\n'.join(decode(line) for line in value_lines(value)))
		print('[PAUSED]')
		input()

	def menu(self) -> int:
		menu = self.env.menu
		print(menu.title)
		for i, option in enumerate(menu.options, 1):
			print(f"{i}: {option}")
		while True:
			try:
				choice = int(input('> '))
			except ValueError:
				continue
			if 1 <= choice <= len(menu.options):
				return choice - 1

	def finish(self) -> None:
		self.present()


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

class FramedConsole(Console):
	"""Shared machinery for the interactive terminal consoles: VT/ANSI setup,
	in-place framed repaints, cursor management, status spinners, and msvcrt key
	polling for getKey/Pause/Menu.

	A subclass decides only what the active screen looks like — `_screen_rows()`
	returns the printable rows for the current env.screen (TerminalConsole: text;
	PixelConsole: rasterized LCD pixels) — and every repaint path shares one _paint.
	"""

	# The terminal's hardware cursor is kept hidden the whole time a program runs
	# (it would otherwise blink in the bottom-left corner where _paint parks it),
	# and revealed only while an input() is actually collecting keystrokes and once
	# the program finishes.  Class-level default so a bare __new__ instance (used
	# in menu-rendering tests) still has a sane value.
	_cursor_hidden = False
	_present_delay = 0.0

	def __init__(self, present_delay: float = 0.0):
		"""`present_delay` pauses this long (seconds) after each output repaint, to
		mimic the real calculator's slowness — set it to watch a drawing build up
		pixel by pixel or text scroll at a readable pace.  0 (the default) runs at
		full speed.  It only paces command-driven output (Disp/Output(/drawing), not
		the getKey indicator tick, which self-throttles."""
		if sys.platform == 'win32':
			_enable_windows_vt()
		# stdout's inherited encoding can be a legacy codepage (e.g. cp1252) that
		# can't represent the spinner glyphs or the TI charset's Greek/symbol
		# characters (π, Σ, ², …); force UTF-8 so writing them never crashes.
		try:
			sys.stdout.reconfigure(encoding='utf-8', errors='replace')
		except (AttributeError, ValueError):
			pass
		self._present_delay = present_delay
		self._last_run_frame = -1

	def present(self) -> None:
		# A repaint opportunity from a command — not a settle point: a drawing
		# command's per-pixel present() reaches _screen_rows with settled=False,
		# which TerminalConsole uses to skip graph repaints (see its docstring).
		self._paint_current(_spinner_frame(_RUNNING_SPINNER))
		if self._present_delay:
			time.sleep(self._present_delay)

	def _paint_current(self, indicator: str | None, *, settled: bool = False) -> None:
		"""Repaint the active screen — home grid, Menu( modal, or graph — with
		`indicator` in the border.  Every repaint path (present/finish/the getKey
		tick/the pause and menu loops) funnels through here.

		`settled` marks the moments the program comes to rest on a screen — a Pause,
		the program finishing — as opposed to a mid-flight repaint.  _screen_rows may
		use it to decide a surface isn't worth painting yet; returning None skips the
		repaint and leaves the previous frame up."""
		painted = self._screen_rows(settled)
		if painted is None:
			return
		rows, width = painted
		self._paint(rows, indicator, width=width)

	def _screen_rows(self, settled: bool) -> tuple[list, int] | None:
		"""The active screen (env.screen) as (rows, visible_width) for _paint, or
		None to leave the previous frame in place.  Subclasses own the choice of
		representation."""
		raise NotImplementedError

	def _paint(self, rows: list, indicator: str | None = None, width: int = HomeScreen.COLS) -> None:
		"""Repaint `rows` (each `width` visible characters, ANSI styling allowed)
		framed in the border, with `indicator` optionally overlaid one cell left of
		the top-right corner.  Shared by the home grid, the graph, and the menu —
		none carries any state for the others.

		Repaints in place: home the cursor, overwrite each row clearing to its
		end, then erase anything below.  This avoids \033[2J — which clears only
		the viewport (old frames linger in scrollback) and is a no-op where VT is
		off — so the screen never scrolls.  The trailing newline drops the cursor
		below the grid so an Input prompt appears there (wiped on the next repaint).

		`width` is given explicitly rather than measured from `rows` because the menu
		embeds invisible ANSI codes (so len() overcounts); callers pass the visible
		width (16 for home/menu, 96 for the graph).

		`indicator` is the calculator's status icon (the running or pause spinner);
		None means show none — the program is taking input or has finished.  It
		lives in the border, not a content cell, so it can never collide with
		anything a program Output(s, and nothing needs restoring afterward: the
		next repaint without it just redraws a plain border.
		"""
		framed = [(_BOUNDARY * (width + 1)) + (indicator or _BOUNDARY)]
		for row in rows:
			framed.append(_BOUNDARY + row + _BOUNDARY)
		framed.append(_BOUNDARY * (width + 2))
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

	def finish(self) -> None:
		# Final repaint with no indicator — the program is done.  This is a settle
		# point: if the program ended on the graph, now is when it's worth showing.
		self._paint_current(None, settled=True)
		self._show_cursor()         # hand the cursor back to the terminal
		sys.stdout.flush()

	def _input(self, prompt: list[Token]) -> list[Token]:
		# Input/Prompt always run on the home screen; repaint it without an indicator
		# (the cursor stands in for one) before blocking.
		self._paint_current(None)
		with self._input_cursor():
			text = input(decode(b''.join(t.display for t in prompt)))
		return self._tokenize(text)

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

		A tight `Repeat getKey…End` never calls present(), so without this the
		indicator would freeze even though the program is plainly alive.  read_key
		can fire thousands of times a second, so repaint only when the frame index
		actually advances — the same ~10fps as every other spinner.  No-op before a
		console is attached (bare instances in tests).
		"""
		if self.env is None:
			return
		frame = _now_frame()
		if frame == self._last_run_frame:
			return
		self._last_run_frame = frame
		self._paint_current(_spinner_frame(_RUNNING_SPINNER))

	def pause(self, value=None) -> None:
		"""Animate the spinner in real time until Enter or Space is pressed.

		Pause shows whatever screen is active: the home grid (with the value, if any),
		or the graph when a drawing program pauses on it — a settle point, so even
		TerminalConsole paints the graph here.  The value (if any) is rendered onto
		the home grid — Pause shows it like Disp but doesn't log it to the Disp
		`values` log, so it uses write_line, not home.disp.  A list/matrix too big for
		the screen is shown through a ScrollView the arrow keys page instead.

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
		home = self.env.home
		self._render_pause_value(value)
		view = ScrollView.of(value)
		try:
			import msvcrt
		except ImportError:
			msvcrt = None
		if msvcrt is None or not sys.stdin.isatty():
			if view is not None:
				view.render_into(home)
			self._paint_current(_PAUSE_SPINNER[0], settled=True)   # static: no key polling to animate it
			with self._input_cursor():
				input()
			return
		# _paint keeps the cursor hidden while the spinner animates; no need to
		# manage it here — it stays hidden until an Input or the program ends.
		while True:
			if view is not None:
				view.render_into(home)
			self._paint_current(_spinner_frame(_PAUSE_SPINNER), settled=True)
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

	def menu(self) -> int:
		"""Render the active menu (env.menu) as a bordered screen and animate the
		Pause spinner while waiting — a menu is a genuine block-until-the-user-acts
		state like Pause, not a polling loop like getKey.

		Up/Down move the highlighted item; a number key 1-7 jumps straight to that
		item and confirms it immediately, matching the real calculator; Enter confirms
		whichever item is currently highlighted.  The highlight lives on the model
		(menu.selected); this just maps keys onto it and repaints.

		Falls back to a plain numbered prompt without msvcrt or a real terminal,
		for the same reason pause() does (msvcrt can't see piped/redirected input).
		"""
		menu = self.env.menu
		try:
			import msvcrt
		except ImportError:
			msvcrt = None
		if msvcrt is None or not sys.stdin.isatty():
			print(menu.title)
			for i, option in enumerate(menu.options, 1):
				print(f'{i}: {option}')
			with self._input_cursor():
				while True:
					try:
						choice = int(input('> '))
					except ValueError:
						continue
					if 1 <= choice <= len(menu.options):
						return choice - 1
		# _paint keeps the cursor hidden while the menu animates; like Pause, the
		# menu never collects keystrokes through input(), so it stays hidden.
		while True:
			self._paint_current(_spinner_frame(_PAUSE_SPINNER))
			result = self._poll_menu_key(msvcrt, len(menu.options))
			if result is None:
				continue
			if result == 'up':
				menu.up()
			elif result == 'down':
				menu.down()
			elif result == 'enter':
				return menu.selected
			else:                # a number key: direct, immediate selection
				menu.choose(result)
				return menu.selected
			sys.stdout.flush()

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


class TerminalConsole(FramedConsole):
	"""Text-first interactive console for quick prototyping: the home grid and menus
	paint as 16-wide character rows.

	The graph (96×32 half-blocks) is the deliberate exception: it dwarfs the home
	grid, and drawing commands present() per pixel, so it is painted only at settle
	points — a Pause on the graph, the program finishing — and mid-flight repaints
	leave the previous frame up rather than flood the terminal.  A graph-screen game
	that repaints on getKey is out of scope here; PixelConsole is the tool for that.
	"""

	def _screen_rows(self, settled: bool) -> tuple[list, int] | None:
		if self.env.screen is Screen.GRAPH:
			if not settled:
				return None
			return self.env.graph.rows(), _GRAPH_COLS
		if self.env.screen is Screen.MENU:
			return self._menu_rows(self.env.menu), HomeScreen.COLS
		return str(self.env.home).split('\n'), HomeScreen.COLS

	def _menu_rows(self, menu) -> list:
		"""The menu's display rows as ANSI strings — the model's canonical layout
		(menu.styled_rows(), whose segments are display bytes) decoded to characters,
		with each inverted segment wrapped in inverse-video codes.  The plain text is
		sized to the 16 columns by the model; the ANSI codes are invisible, so _paint
		is given the visible width explicitly."""
		return [
			''.join((_INV_ON + decode(text) + _INV_OFF) if inverted else decode(text)
			        for text, inverted in row)
			for row in menu.styled_rows()
		]


class PixelConsole(FramedConsole):
	"""Everything through the LCD: home, menu, and graph all rasterize through the
	font tables into the shared 96×64 Bitmap and paint as 32 rows of half-block
	pixels — one visual pipeline, the way the real screen works.

	present() always repaints, so a drawing builds up live and a getKey graph loop
	animates (the indicator tick already self-throttles to ~10fps).  The trade is
	space and glyph fidelity: text is font pixels, not characters.  TerminalConsole
	is the lighter tool when the graph only matters at settle points."""

	def _screen_rows(self, settled: bool) -> tuple[list, int]:
		if self.env.screen is Screen.GRAPH:
			return self.env.graph.rows(), _GRAPH_COLS
		if self.env.screen is Screen.MENU:
			return self.env.menu.rasterize().rows(), _GRAPH_COLS
		return self.env.home.rasterize().rows(), _GRAPH_COLS


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
