from abc import ABC
from dataclasses import dataclass


@dataclass
class MenuOption:
	text: str
	label: str


try:
	import msvcrt
	def get_char() -> str:
		ch = msvcrt.getwch()
		if ch == '\x03':
			raise KeyboardInterrupt
		return ch
		
except ImportError:
	import tty, termios, sys
	def get_char() -> str:
		fd = sys.stdin.fileno()
		old = termios.tcgetattr(fd)
		try:
			tty.setraw(fd)
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old)
		if ch == '\x03':
			raise KeyboardInterrupt
		return ch


class Terminal(ABC):
	
	def disp(self, values):
		for val in values:
			print(val)
	
	def input(self, prompt):
		pass
	
	def menu(self, title: str, options: list[MenuOption]):
		print(title)
		for i, opt in enumerate(options, start=1):
			print(f"{i}:{opt.text}")
		while True:
			ch = get_char()
			try:
				choice = int(ch) - 1
			except ValueError:
				pass
			else:
				if 0 < choice < len(options):
					return options[choice]
			print(f"Invalid choice: {ch}")


result = Terminal().menu('Engima Machine', [MenuOption('Keyboard', 'K'), MenuOption('Encode', 'E'), MenuOption('Settings', 'S'), MenuOption('Exit', 'X')])
print(result)