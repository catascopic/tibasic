from pathlib import Path

from environment import Environment
from terminal import TerminalConsole
from tifile import ProgramFile, ListFile
from catalog import get_token, CHAR_TABLE

def load(dir_):
	env = Environment()
	readers = {'.8xp': ProgramFile, '.8xl': ListFile}
	for file in Path(dir_).iterdir():
		reader = readers.get(file.suffix)
		if reader:
			reader.load(file).store_to(env)
	return env


env = load(r'enigma')
env.console = TerminalConsole()

def run(prgm_name):
	env.submit([get_token(0x5F), *(CHAR_TABLE[c] for c in prgm_name)])


run('ENIGMA2')
# run('QR')
# env.graph.disp()
# env.graph.print_screen('qr.bmp', 3)
# env.dump()
