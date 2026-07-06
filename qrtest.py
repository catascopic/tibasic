import sys
from pathlib import Path

import tifile
from environment import Environment
from terminal import TerminalConsole
from catalog import get_token, CHAR_TABLE


env = Environment()
tifile.load_environment(env, r'enigma')
env.console = TerminalConsole()

def run(prgm_name):
	env.submit([get_token(0x5F), *(CHAR_TABLE[c] for c in prgm_name)])


# run('ENIGMA2')
run('QR')
env.graph.disp()
# env.graph.print_screen('qr.bmp', 3)
# env.dump()
