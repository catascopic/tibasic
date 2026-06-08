"""Interactive drawing tester.

Starts in Pt-On mode.  Type arguments for the current command, then press
Enter to draw and see the screen.  Type a command name to switch to it
(case- and punctuation-insensitive; 'chi' matches χ).  Blank line to quit.

Commands:
  Pt-On       Pt-Off      Pt-Change
  Pxl-On      Pxl-Off     Pxl-Change
  ClrDraw     Circle      Line
  Vertical    Horizontal
  DrawF       DrawInv
  ShadeNorm   Shade_t     Shadeχ² (match: shadechi)   ShadeF
"""
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from environment import Environment, DrawMode
from test_tibasic import run


# ── Command table ────────────────────────────────────────────────────────────────
# Each entry: (normalised_key, display_name, ti_prefix, takes_args)
#   takes_args=True  → prompt the user for arguments, prepend ti_prefix, run
#   takes_args=False → run ti_prefix immediately (no arguments), stay in cur mode

_COMMANDS = [
	('pton',       'Pt-On',      'Pt-On( ',       True),
	('ptoff',      'Pt-Off',     'Pt-Off( ',      True),
	('ptchange',   'Pt-Change',  'Pt-Change( ',   True),
	('pxlon',      'Pxl-On',     'Pxl-On( ',      True),
	('pxloff',     'Pxl-Off',    'Pxl-Off( ',     True),
	('pxlchange',  'Pxl-Change', 'Pxl-Change( ',  True),
	('clrdraw',    'ClrDraw',    'ClrDraw',        False),
	('circle',     'Circle',     'Circle( ',       True),
	('line',       'Line',       'Line( ',         True),
	('vertical',   'Vertical',   'Vertical ',      True),
	('horizontal', 'Horizontal', 'Horizontal ',    True),
	('drawf',      'DrawF',      'DrawF ',         True),
	('drawinv',    'DrawInv',    'DrawInv ',       True),
	('shadenorm',  'ShadeNorm',  'ShadeNorm( ',    True),
	('shadet',     'Shade_t',    'Shade_t( ',      True),
	('shadechi',   'Shadeχ²',    'Shadeχ²( ',      True),
	('shadef',     'ShadeF',     'ShadeF( ',       True),
]

_CMD_LOOKUP = {key: (display, prefix, has_args) for key, display, prefix, has_args in _COMMANDS}


def _normalize(s: str) -> str:
	"""Lowercase, map χ→'chi', strip all non-alphanumeric characters."""
	s = s.lower().replace('χ', 'chi')
	return re.sub(r'[^a-z0-9]', '', s)


def _window_info(env) -> str:
	w = env.window
	return (f"Window: x=[{w.xmin.value}, {w.xmax.value}]  "
	        f"y=[{w.ymin.value}, {w.ymax.value}]")


def _print_commands() -> None:
	names = [display for _, display, _, _ in _COMMANDS]
	# Print in rows of four, left-aligned in 14-char columns
	cols = 4
	rows = [names[i:i + cols] for i in range(0, len(names), cols)]
	print('Commands (type a name to switch):')
	for row in rows:
		print('  ' + ''.join(n.ljust(14) for n in row))
	print('  (case- and punctuation-insensitive; "chi" matches χ)')


# ── Main loop ────────────────────────────────────────────────────────────────────

env = Environment()
current_prefix  = 'Pt-On( '     # TI prefix prepended to user input
current_display = 'Pt-On'       # shown in the prompt
# env.draw_mode = DrawMode.DOT

print('Interactive drawing tester — blank line to quit.')
print(_window_info(env))
print()
_print_commands()
print()

while True:
	try:
		user_input = input(f'{current_prefix}').strip()
	except (EOFError, KeyboardInterrupt):
		print()
		break
	if not user_input:
		break

	norm = _normalize(user_input)
	if norm in _CMD_LOOKUP:
		display, prefix, has_args = _CMD_LOOKUP[norm]
		if not has_args:
			# Execute immediately (e.g. ClrDraw); stay in current mode
			try:
				run(prefix, env)
				print(f'  [{display}]')
			except Exception as e:
				print(f'  Error: {e}')
			env.screen.show()
			print()
		else:
			current_display = display
			current_prefix  = prefix
			print(f'  → {display}')
	else:
		try:
			run(current_prefix + user_input, env)
			env.screen.show()
			print()
		except Exception as parse_err:
			print(parse_err)
			print()
			_print_commands()
			print()
