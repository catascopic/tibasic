from pathlib import Path

from environment import Environment
from terminal import TerminalConsole
from program import Program
from tifile import ProgramFile
from catalog import get_token, TEXT_INPUT

def load(dir_):
	env = Environment()
	for file in Path(dir_).iterdir():
		if file.suffix == '.8xp':
			prgm = ProgramFile.load(file)
			env.programs[prgm.name] = prgm.tokens
	return env


env = load(r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_10')
env.console = TerminalConsole()

def run(prgm_name):
	env.run([get_token(0x5F), *(TEXT_INPUT[c] for c in prgm_name)])


run('QRLIST')
run('ENIGMA2')
env.graph.disp()
