from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

PROGRAM = """
4@I
8@J
4@K
8@L
Output( I,J,"X
Repeat A=105
	getKey @A
	If A=24
		J-1@J
	If A=25
		I-1@I
	If A=26
		J+1@J
	If A=34
		I+1@I
	If I≠K or J≠L
	Then
		Output( K,L,"_
		Output( I,J,"X
		I@K
		J@L
	End
End
"""

env = Environment()
env.console = TerminalConsole()
Program(toks(PROGRAM), env).run()
