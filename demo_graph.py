"""Render the graph screen for Y1=X and Y2=X² under a chosen window / format.

A small CLI around Environment.regraph + GraphScreen.disp: it sets the window
bounds and scales, toggles the grid and axes, plots the two functions, and prints
the resulting pixel buffer as text.

Examples:
    python demo_graph.py                         # default ±10 window, axes on, grid off
    python demo_graph.py --grid                  # turn the grid on
    python demo_graph.py --no-axes               # hide the axes
    python demo_graph.py --xmin -5 --xmax 5 --xscl 0.5 --yscl 2 --grid
"""
import argparse
import sys

from core import TiString
from environment import Environment


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--xmin', type=float, default=-10.0, help='left edge of the window (default: -10)')
	p.add_argument('--xmax', type=float, default=10.0,  help='right edge of the window (default: 10)')
	p.add_argument('--ymin', type=float, default=-10.0, help='bottom edge of the window (default: -10)')
	p.add_argument('--ymax', type=float, default=10.0,  help='top edge of the window (default: 10)')
	p.add_argument('--xscl', type=float, default=1.0,   help='x tick / grid spacing (default: 1)')
	p.add_argument('--yscl', type=float, default=1.0,   help='y tick / grid spacing (default: 1)')
	p.add_argument('--axes', action=argparse.BooleanOptionalAction, default=True,
	               help='draw the axes with tick marks (default: on; use --no-axes to hide)')
	p.add_argument('--grid', action=argparse.BooleanOptionalAction, default=False,
	               help='draw grid points at every (A*Xscl, B*Yscl) (default: off)')
	return p


def main(argv=None) -> int:
	args = build_parser().parse_args(argv)

	if args.xmax <= args.xmin:
		print('error: --xmax must be greater than --xmin', file=sys.stderr)
		return 2
	if args.ymax <= args.ymin:
		print('error: --ymax must be greater than --ymin', file=sys.stderr)
		return 2

	env = Environment()
	w = env.window
	w.xmin.value, w.xmax.value = args.xmin, args.xmax
	w.ymin.value, w.ymax.value = args.ymin, args.ymax
	w.xscl.value, w.yscl.value = args.xscl, args.yscl
	env.axes_on = args.axes
	env.grid_on = args.grid

	# Plot Y1=X and Y2=X²; storing each equation also selects it for graphing.
	env.function[0].store(TiString.from_str('X'))
	env.function[1].store(TiString.from_str('X^2'))

	env.display_graph()   # set the graph as active and re-plot (axes/grid/functions)

	print(f'X:[{args.xmin}, {args.xmax}] step {args.xscl}   '
	      f'Y:[{args.ymin}, {args.ymax}] step {args.yscl}   '
	      f'axes {"on" if args.axes else "off"}   grid {"on" if args.grid else "off"}')
	print('Y1=X   Y2=X²')
	env.graph.disp()
	return 0


if __name__ == '__main__':
	# The disp() output uses Unicode block characters; make sure they survive a
	# legacy console code page (e.g. Windows cp1252) instead of crashing.
	try:
		sys.stdout.reconfigure(encoding='utf-8')
	except (AttributeError, ValueError):
		pass
	raise SystemExit(main())
