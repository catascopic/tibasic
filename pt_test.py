"""Interactive test: enter Pt-On arguments and see the result on screen."""

from environment import Environment
from test_tibasic import run

env = Environment()

print("Pt-On( x, y [, mark] ) tester  —  blank line to quit")
print(f"Window: x=[{env.window.xmin.value}, {env.window.xmax.value}]  "
      f"y=[{env.window.ymin.value}, {env.window.ymax.value}]")
print()

while True:
	try:
		line = input("Pt-On( ").strip()
	except EOFError:
		break
	if not line:
		break
	try:
		run(f"Pt-On( {line}", env)
	except Exception as e:
		print(f"  Error: {e}")
		continue
	env.screen.show()
	print()
