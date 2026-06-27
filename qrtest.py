from pathlib import Path

from environment import Environment
from core import TiList
from terminal import TerminalConsole
from program import Program
from tifile import ProgramFile, ListFile
from catalog import get_token, TEXT_INPUT

def load(dir_):
	env = Environment()
	for file in Path(dir_).iterdir():
		if file.suffix == '.8xp':
			prgm = ProgramFile.load(file)
			env.programs[prgm.name] = Program(prgm.tokens, prgm.name)
		if file.suffix == '.8xl':
			lst = ListFile.load(file)
			ti_list = TiList(lst.values)
			if lst.name.isdigit():
				env.lists[int(lst.name)] = ti_list
			else:
				env.user_lists[lst.name] = ti_list
	return env


env = load(r'enigma')
env.console = TerminalConsole()

def run(prgm_name):
	env.submit([get_token(0x5F), *(TEXT_INPUT[c] for c in prgm_name)])


run('ENIGMA2')
# run('QR')
# env.graph.disp()
# env.graph.print_screen('qr.bmp', 3)
# env.dump()
