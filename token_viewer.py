"""Page through every token, rendering each with Text( on the graph screen.

Large mode (default): 8 tokens per page, large font, one per 8-pixel line.
Small mode:           10 tokens per page, small font, one per 6-pixel line.

Each token is stashed in a string variable and drawn with a real Text(
statement, shown, then waits for Enter before clearing and drawing the next
page.  Blank line input quits early.

Routing the token through a string variable (rather than a "..." literal) lets
Text( display the three tokens it normally can't — " , →, and newline — which
self-terminate a string literal.

Usage:
    python token_viewer.py          # large font
    python token_viewer.py small    # small font
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from environment import Environment
from catalog import ALL_TOKENS, get_token, DIGITS, COMMA, NEG, STRINGS
from core import TiString

TEXT = get_token(0x93)
STR_VAR = STRINGS[0]   # Str1 — scratch register for the token being drawn


def _row_digits(row: int) -> list:
	"""The digit tokens spelling out a row number (e.g. 56 -> [DIGITS[5], DIGITS[6]])."""
	return [DIGITS[int(c)] for c in str(row)]


def _draw_token(env, tok, row: int, large: bool) -> None:
	"""Draw one token at (row, 0) via Text(, reading it from a string variable."""
	STR_VAR.variable(env).value = TiString([tok])
	stmt = [TEXT]
	if large:
		stmt += [NEG, DIGITS[1], COMMA]          # -1 selects the large font
	stmt += _row_digits(row) + [COMMA, DIGITS[0], COMMA, STR_VAR]
	env.submit(stmt)


def main() -> None:
	small = len(sys.argv) > 1 and sys.argv[1].lower().startswith('s')
	large = not small
	per_page     = 8 if large else 10
	line_height  = 8 if large else 6
	font_name    = 'large' if large else 'small'

	env = Environment()
	total = len(ALL_TOKENS)
	pages = (total + per_page - 1) // per_page

	print(f"{total} tokens, {font_name} font, {per_page} per page, {pages} pages.")
	print("Enter to advance, blank line / Ctrl-C to quit.\n")

	for page in range(pages):
		batch = ALL_TOKENS[page * per_page:(page + 1) * per_page]
		env.graph.clear()
		for i, tok in enumerate(batch):
			_draw_token(env, tok, i * line_height, large)

		env.graph.show()
		# Console-side legend so you can match glyphs to codes/text.
		for i, tok in enumerate(batch):
			print(f"  line {i}: 0x{tok.code:0{4 if tok.code > 0xFF else 2}X}  {tok.text!r}")
		print(f"\n── page {page + 1}/{pages} ──")

		try:
			if input().strip():
				break
		except (EOFError, KeyboardInterrupt):
			print()
			break


if __name__ == '__main__':
	main()
