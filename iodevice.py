"""I/O devices — the swappable boundary between the interpreter and a frontend.

A command never touches screen lines directly: it calls a semantic operation on
`env.io` (Disp a value, Output at a cell, Pause, read an Input) and the device
decides how to realize it.  This lets three frontends coexist:

  * HomeScreenIO — the faithful calculator screen: an 8×16 byte grid (HomeScreen)
    with scrolling, positioned Output(, and scrollable Pause, painted by a
    swappable Console backend (terminal today, a web canvas later).
  * FreeFormIO — a loose text frontend with no grid at all: Disp is print(), Input
    is input(), Pause is print()+input(); Output( is weakly supported (printed
    without positioning) and getKey isn't supported.

Value→display formatting is shared via tiformat; only placement and the final
paint differ between devices.  HomeScreenIO keeps the grid logic that used to live
inside the Disp/Output/Pause/Input commands; the commands now just hand it the
value.
"""
from abc import ABC, abstractmethod

from core import TiString
from errors import DomainError, TiSyntaxError
from titoken import Token, encode, decode
from tiformat import disp_lines, output_text, value_lines
from homescreen import HomeScreen


class IODevice(ABC):
	"""The semantic I/O operations a command can ask of the frontend."""

	@abstractmethod
	def disp(self, value) -> None:
		"""Disp one value (append it to the output)."""

	@abstractmethod
	def refresh(self) -> None:
		"""Re-present the current screen with no new content (bare Disp)."""

	@abstractmethod
	def output(self, row: int, col: int, value) -> None:
		"""Output( a value at a 1-indexed (row, col)."""

	@abstractmethod
	def clear_home(self) -> None:
		"""ClrHome — wipe the text screen."""

	@abstractmethod
	def pause(self, value=None) -> None:
		"""Pause — show `value` (if any), then block until the user continues."""

	@abstractmethod
	def read_tokens(self, prompt: TiString | None) -> list[Token]:
		"""Blocking: display prompt (if any), return the tokens the user entered.

		The caller is responsible for including any '?' in the prompt TiString.
		A device that paints over its own input echo is responsible for committing
		the accepted entry to its display here."""

	@abstractmethod
	def menu(self, title: str, options) -> int:
		"""Menu( — present options, return the chosen 0-based index."""

	@abstractmethod
	def get_key(self) -> int:
		"""getKey — the TI key code currently pressed, or 0 for none."""

	@abstractmethod
	def finish(self) -> None:
		"""Called when program execution ends."""


class HomeScreenIO(IODevice):
	"""Faithful calculator I/O: an 8×16 byte grid (HomeScreen) whose final paint and
	key input are delegated to a swappable Console backend.  This is the device the
	terminal and (eventually) the web canvas share — only the Console differs."""

	def __init__(self, console):
		self.home = HomeScreen()
		self.console = console

	def disp(self, value) -> None:
		for line in disp_lines(value, self.home.COLS):
			self.home.disp(line)
		self.console.update(self.home)

	def refresh(self) -> None:
		self.console.update(self.home)

	def output(self, row: int, col: int, value) -> None:
		if not (1 <= row <= self.home.ROWS and 1 <= col <= self.home.COLS):
			raise DomainError(f"Output(: position out of range: ({row}, {col})")
		self.home.output(row - 1, col - 1, output_text(value))
		self.console.update(self.home)

	def clear_home(self) -> None:
		self.home.clear()
		self.console.update(self.home)

	def pause(self, value=None) -> None:
		if value is not None:
			for line in disp_lines(value, self.home.COLS):
				self.home.disp(line)
		self.console.pause(self.home, value)

	def read_tokens(self, prompt: TiString | None) -> list[Token]:
		try:
			tokens = self.console.read_tokens(prompt, self.home)
		except KeyError as bad:
			raise TiSyntaxError(f"Input: unsupported character {bad}")
		if tokens:
			self._echo_input(prompt, tokens)
		return tokens

	def _echo_input(self, prompt: TiString | None, tokens: list[Token]) -> None:
		prompt_bytes = b''.join(t.display for t in prompt.tokens) if prompt is not None else b''
		input_bytes  = b''.join(t.display for t in tokens)
		self.home.echo(prompt_bytes + input_bytes)
		self.console.update(self.home)

	def menu(self, title: str, options) -> int:
		return self.console.choose(title, options)

	def get_key(self) -> int:
		return self.console.read_key()

	def finish(self) -> None:
		self.console.finish(self.home)


class FreeFormIO(IODevice):
	"""A loose frontend with no home-screen grid: Disp/Pause/Input go straight to
	print()/input().  Output( is weakly supported — the value is printed without its
	position — and getKey returns 0 (no key support)."""

	@staticmethod
	def _text(value) -> str:
		return '\n'.join(decode(line) for line in value_lines(value))

	def disp(self, value) -> None:
		print(self._text(value))

	def refresh(self) -> None:
		pass

	def output(self, row: int, col: int, value) -> None:
		print(self._text(value))   # position ignored — weakly supported

	def clear_home(self) -> None:
		pass

	def pause(self, value=None) -> None:
		if value is not None:
			print(self._text(value))
		print('[PAUSED]')
		input()

	def read_tokens(self, prompt: TiString | None) -> list[Token]:
		text = input(str(prompt) if prompt is not None else '')
		try:
			return TiString.from_str(text).tokens
		except KeyError as bad:
			raise TiSyntaxError(f"Input: unsupported character {bad} in {text!r}")

	def menu(self, title: str, options) -> int:
		print(title)
		for i, option in enumerate(options, 1):
			print(f"{i}: {option}")
		while True:
			try:
				choice = int(input('> '))
			except ValueError:
				continue
			if 1 <= choice <= len(options):
				return choice - 1

	def get_key(self) -> int:
		return 0   # no key support in free-form mode

	def finish(self) -> None:
		pass
