from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

code = toks("""
4@I
8@J
1@K
1@L
Repeat A=105
	If I≠K or J≠L
	Then
		Output( K,L,"_
		Output( I,J,"X
		I@K
		J@L
	End
	getKey @A
	If A=24
		J-1@J
	If A=25
		I-1@I
	If A=26
		J+1@J
	If A=34
		I+1@I
	max( 1, min( 8,I@I
	max( 1, min( 16,J@J
End
""")

env = Environment()
env.console = TerminalConsole()
Program(code).run(env)

from tifile import ProgramFile
ProgramFile('MOVE', code).write('MOVE.8xp')
