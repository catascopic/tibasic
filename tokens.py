from dataclasses import dataclass
import itertools


# ── TI-83+ Large Font Display Encoding ────────────────────────────────────────
# Maps Unicode text → bytes in the TI-83+ Large Font charset (0x00–0xF7).
# Multi-char keys must appear before any single-char prefix they share so
# the greedy encoder in _encode_display() matches them first.

_D: dict[str, bytes] = {
	# ── Multi-char sequences (combining chars / ligatures) ──────────────────────
	'⁻¹':      b'\x11',       # inverse/reciprocal as single glyph
	'x̄': b'\xcb',       # x̄  (x + combining macron = x-mean)
	'ȳ': b'\xcc',       # ȳ  (y + combining macron = y-mean)
	'p̂': b'\xd8',       # p̂  (p + combining circumflex = p-hat)
	'₁₀': b'\x90',  # subscript 10
	# ── Special glyphs 0x01–0x1F ────────────────────────────────────────────────
	'►':       b'\x05',       # right-pointing triangle (convert arrow)
	'🡅':      b'\x06',       # scroll up
	'🡇':      b'\x07',       # scroll down
	'∫':       b'\x08',       # integral
	'×':       b'\x09',       # multiplication cross
	'√':       b'\x10',       # square root radical
	'²':       b'\x12',       # superscript 2
	'∠':       b'\x13',       # angle
	'∟':       b'\x13',       # right angle → same glyph as ∠
	'°':       b'\x14',       # degree
	'ʳ':       b'\x15',       # superscript r (radian)
	'ᵀ':       b'\x16',       # superscript T (transpose)
	'≤':       b'\x17',       # less than or equal
	'≠':       b'\x18',       # not equal
	'≥':       b'\x19',       # greater than or equal
	'⁻':       b'\x1a',       # superscript minus (negation); also prefix of ⁻¹ above
	'ᴇ':       b'\x1b',       # scientific-notation E
	'→':       b'\x1c',       # right arrow (store)
	'↑':       b'\x1e',       # up arrow
	'↓':       b'\x1f',       # down arrow
	# ── ASCII-position remaps ────────────────────────────────────────────────────
	'[':       b'\xc1',       # left bracket (0x5B is θ in display charset)
	'³':       b'\x0e',       # superscript 3 / cube-root mark
	'−':       b'\x2d',       # math minus (U+2212) → regular dash
	# ── Other special Unicode ────────────────────────────────────────────────────
	'θ':       b'\x5b',       # theta (at 0x5B, where ASCII has '[')
	'←':       b'\xcf',       # left arrow
	'◄':       b'\xcf',       # left-pointing triangle → left arrow glyph
	'↵':       b'\xd6',       # enter/return arrow
	'…':       b'\xce',       # ellipsis
	'ȳ':       b'\xcc',       # y-bar precomposed (U+0233)
	'𝑒':       b'\x65',       # math italic e → regular e
	'𝑖':       b'\xd7',       # math italic i → imaginary-i glyph
	# ── Subscript digits 0–9 ─────────────────────────────────────────────────────
	'₀': b'\x80', '₁': b'\x81', '₂': b'\x82', '₃': b'\x83', '₄': b'\x84',
	'₅': b'\x85', '₆': b'\x86', '₇': b'\x87', '₈': b'\x88', '₉': b'\x89',
	# ── Greek letters ─────────────────────────────────────────────────────────────
	'α': b'\xbb', 'β': b'\xbc', 'γ': b'\xbd', 'Δ': b'\xbe', 'δ': b'\xbf',
	'ε': b'\xc0', 'λ': b'\xc2', 'μ': b'\xc3', 'π': b'\xc4', 'ρ': b'\xc5',
	'Σ': b'\xc6', 'σ': b'\xc7', 'τ': b'\xc8', 'φ': b'\xc9', 'Ω': b'\xca',
	'χ': b'\xd9',
	# ── Accented Latin – uppercase ────────────────────────────────────────────────
	'Á': b'\x8a', 'À': b'\x8b', 'Â': b'\x8c', 'Ä': b'\x8d',
	'É': b'\x92', 'È': b'\x93', 'Ê': b'\x94', 'Ë': b'\x95',
	'Í': b'\x9a', 'Ì': b'\x9b', 'Î': b'\x9c', 'Ï': b'\x9d',
	'Ó': b'\xa2', 'Ò': b'\xa3', 'Ô': b'\xa4', 'Ö': b'\xa5',
	'Ú': b'\xaa', 'Ù': b'\xab', 'Û': b'\xac', 'Ü': b'\xad',
	'Ç': b'\xb2', 'Ñ': b'\xb4',
	# ── Accented Latin – lowercase ────────────────────────────────────────────────
	'á': b'\x8e', 'à': b'\x8f', 'â': b'\x90', 'ä': b'\x91',
	'é': b'\x96', 'è': b'\x97', 'ê': b'\x98', 'ë': b'\x99',
	'í': b'\x9e', 'ì': b'\x9f', 'î': b'\xa0', 'ï': b'\xa1',
	'ó': b'\xa6', 'ò': b'\xa7', 'ô': b'\xa8', 'ö': b'\xa9',
	'ú': b'\xae', 'ù': b'\xaf', 'û': b'\xb0', 'ü': b'\xb1',
	'ç': b'\xb3', 'ñ': b'\xb5',
	# ── Punctuation / accent marks ────────────────────────────────────────────────
	'´': b'\xb6', '¨': b'\xb8', '¿': b'\xb9', '¡': b'\xba', 'ß': b'\xf4',
}

_D_KEYS = sorted(_D, key=len, reverse=True)


def _encode_display(text: str) -> bytes:
	"""Encode token display text into TI-83+ Large Font bytes (greedy longest-match)."""
	result = bytearray()
	i = 0
	while i < len(text):
		matched = False
		for key in _D_KEYS:
			if text.startswith(key, i):
				result.extend(_D[key])
				i += len(key)
				matched = True
				break
		if not matched:
			ch = text[i]
			code = ord(ch)
			if 0x20 <= code <= 0x7e and ch != '[':
				result.append(code)
			i += 1
	return bytes(result)


@dataclass(eq=False)
class Token:
	code: bytes
	key: str
	text: str
	category: str
	desc: str
	alias: set[str]
	display: bytes

	# ── Token type predicates ──────────────────────────────────────────────────

	def is_real_var(self) -> bool:
		return (len(self.code) == 1 and 0x41 <= self.code[0] <= 0x5a) or self.code == b'\x5b'

	def is_list_var(self) -> bool:
		return len(self.code) == 2 and self.code[0] == 0x5d

	def is_matrix_var(self) -> bool:
		return len(self.code) == 2 and self.code[0] == 0x5c

	def is_string_var(self) -> bool:
		return len(self.code) == 2 and self.code[0] == 0xaa

	def is_stat_var(self) -> bool:
		return len(self.code) == 2 and self.code[0] in (0x62, 0x63)

	def is_digit(self) -> bool:
		return len(self.code) == 1 and 0x30 <= self.code[0] <= 0x39

	def is_function(self) -> bool:
		"""True for tokens that open a function call (text ends with '(')."""
		return self.text.endswith('(')

	def is_nullary(self) -> bool:
		"""True for 0-arg no-parenthesis value tokens (π, e, 𝑖, rand, getKey, etc.)."""
		from operations import NULLARY_CALLS
		return self.code in NULLARY_CALLS

	def get_value(self, env: dict):
		"""Evaluate this nullary token against the given environment."""
		from operations import NULLARY_CALLS
		return NULLARY_CALLS[self.code](env)

	def can_start_atom(self) -> bool:
		"""True if this token can appear as the start of an expression atom."""
		return (
			self.is_digit() or self.is_real_var() or self.is_list_var() or
			self.is_matrix_var() or self.is_string_var() or self.is_function() or
			self.is_nullary() or
			self is DOT or self is L_PAREN or self is L_BRACE or
			self is QUOTE or self is NEG or self is ANS or self is LIST_PREFIX
		)

	# ── Operation delegation (implementations live in operations.py) ───────────

	def binary_op(self, lhs, rhs):
		from operations import BINARY_OPS
		fn = BINARY_OPS.get(self.code)
		if fn is None:
			raise NotImplementedError(f"No binary_op for {self.text!r}")
		return fn(lhs, rhs)

	def unary_op(self, operand):
		from operations import UNARY_OPS
		fn = UNARY_OPS.get(self.code)
		if fn is None:
			raise NotImplementedError(f"No unary_op for {self.text!r}")
		return fn(operand)

	def call(self, parser):
		"""Evaluate a function call. Parser is positioned after the opening '('."""
		from operations import FUNCTION_CALLS
		fn = FUNCTION_CALLS.get(self.code)
		if fn is None:
			raise NotImplementedError(f"No call implementation for {self.text!r}")
		if fn is NotImplemented:
			raise NotImplementedError(f"{self.text!r} requires executor context")
		return fn(parser)

	def execute(self, parser):
		"""Execute a command statement. Returns a StatementResult or None."""
		from operations import EXEC_CALLS
		fn = EXEC_CALLS.get(self.code)
		if fn is None:
			return None  # not a command token
		return fn(parser)


EOF_TOKEN = Token(b'', None, '', 'eof', 'eof', frozenset(), b'')

_SEEN: set[bytes] = set()

def token(code: bytes, text: str, category: str, desc: str, alt: str | set[str] | None = None, key: str | None = None) -> Token:
	if code in _SEEN:
		raise ValueError(f"Duplicate token code: {code!r} ({text!r})")
	_SEEN.add(code)
	alias = {text.lower()}
	if isinstance(alt, str):
		alias.add(alt.lower())
	elif alt is not None:
		alias.update(a.lower() for a in alt)
	return Token(code, key, text, category, desc, alias, _encode_display(text))

# ── Named syntactic tokens (referenced by identity in the parser) ──────────────

# Structural / delimiter
STORE       = token(b'\x04', "→",    "", "Stores a value into a variable", key='`', alt=("->", 'store'))
L_BRACKET   = token(b'\x06', "[",    "", "Opens a matrix literal", key='[')
R_BRACKET   = token(b'\x07', "]",    "", "Closes a matrix literal", key=']')
L_BRACE     = token(b'\x08', "{",    "", "Opens a list literal", key='{')
R_BRACE     = token(b'\x09', "}",    "", "Closes a list literal", key='}')
L_PAREN     = token(b'\x10', "(",    "", "Opens a parenthetical expression", key='(')
R_PAREN     = token(b'\x11', ")",    "", "Closes a parenthetical expression", key=')')
QUOTE       = token(b'\x2a', '"',    "", "Delimits a string literal", key='"')
COMMA       = token(b'\x2b', ",",    "", "Separates arguments in a function call", key=',')
DOT         = token(b'\x3a', ".",    "num", "Decimal point in numeric literals", key='.')
COLON       = token(b'\x3e', ":",    "", "Command separator", key=':')
NEWLINE     = token(b'\x3f', "↵",    "", "Program line separator (newline)", alt="newline")
PRGM        = token(b'\x5f', "prgm", "", "Calls a named subprogram")
ANS         = token(b'\x72', "Ans",  "variable", "Contains the result of the last evaluated expression")
NEG         = token(b'\xb0', "−",    "prefix", "Negates a value (unary minus)", alt=('~', "neg"), key='~')
LIST_PREFIX = token(b'\xeb', "∟",    "", "Prefix token for user-defined lists (e.g., ∟NAME)", alt="list-prefix", key='#')

# Postfix operators
RAD         = token(b'\x0a', "ʳ",   "postfix", "Radian angle-unit suffix", alt="rad")
DEG         = token(b'\x0b', "°",   "postfix", "Degree angle-unit suffix", alt="deg")
INV         = token(b'\x0c', "⁻¹",  "postfix", "Computes the multiplicative inverse (reciprocal)", alt=("inv", '^-1'))
SQ          = token(b'\x0d', "²",   "postfix", "Squares a value or matrix", alt="^2")
TRANSPOSE   = token(b'\x0e', "ᵀ",   "postfix", "Returns the transpose of a matrix", alt=("T", 'transpose'))
CUBE        = token(b'\x0f', "³",   "postfix", "Cubes a value", alt="^3")
FACT        = token(b'\x2d', "!",   "postfix", "Computes the factorial of a non-negative integer", key='!')

# Binary operators
SCI_E       = token(b'\x3b', "ᴇ",   "operator", "Scientific notation exponent (×10^n)", alt="E")
OR          = token(b'\x3c', "or",  "operator", "Boolean OR operator")
XOR         = token(b'\x3d', "xor", "operator", "Boolean XOR operator")
AND         = token(b'\x40', "and", "operator", "Boolean AND operator")
EQ          = token(b'\x6a', "=",   "operator", "Tests equality between two values", key='=')
LT          = token(b'\x6b', "<",   "operator", "Tests whether the left operand is less than the right", key='<')
GT          = token(b'\x6c', ">",   "operator", "Tests whether the left operand is greater than the right", key='>')
LE          = token(b'\x6d', "≤",   "operator", "Less than or equal to", alt="<=")
GE          = token(b'\x6e', "≥",   "operator", "Greater than or equal to", alt=">=")
NE          = token(b'\x6f', "≠",   "operator", "Not equal to", alt="!=")
ADD         = token(b'\x70', "+",   "operator", "Adds two numbers, matrices, or lists", key='+')
SUB         = token(b'\x71', "-",   "operator", "Subtraction", key='-')
MUL         = token(b'\x82', "*",   "operator", "Multiplies two numbers, scalars by matrices, or lists", key='*')
DIV         = token(b'\x83', "/",   "operator", "Division", key='/')
NPR         = token(b'\x94', "nPr", "operator", "Computes the number of permutations of n things taken r at a time")
NCR         = token(b'\x95', "nCr", "operator", "Computes the number of combinations of n things taken r at a time")
POW         = token(b'\xf0', "^",   "operator", "Raises the left operand to the power of the right operand", key='^')
XROOT       = token(b'\xf1', "×√",  "operator", "Computes the x-th root of a value", alt="xroot")



TOKENS: list[Token] = [
	# One-byte tokens
	token(b'\x01', "►DMS",        "converter",     "Converts a decimal angle to degree-minute-second display", alt='to-DMS'),
	token(b'\x02', "►Dec",        "converter",     "Converts a fraction or expression to decimal form", alt='to-Dec'),
	token(b'\x03', "►Frac",       "converter",     "Converts a decimal to the simplest fraction form", alt='to-Frac'),
	STORE,
	token(b'\x05', "Boxplot",     "enum",     "Selects the standard box-and-whisker plot type for a stat plot"),
	L_BRACKET, R_BRACKET, L_BRACE, R_BRACE,
	RAD, DEG, INV, SQ, TRANSPOSE, CUBE,
	L_PAREN, R_PAREN,
	token(b'\x12', "round(",      "func",     "Rounds a number to a specified number of decimal places"),
	token(b'\x13', "pxl-Test(",   "func",       "Returns 1 if the specified screen pixel is on, 0 otherwise"),
	token(b'\x14', "augment(",    "func",   "Concatenates two matrices or two lists horizontally"),
	token(b'\x15', "rowSwap(",    "func",   "Swaps two rows of a matrix"),
	token(b'\x16', "row+(",       "func",   "Adds one matrix row to another and stores the result"),
	token(b'\x17', "*row(",       "func",   "Multiplies a matrix row by a scalar"),
	token(b'\x18', "*row+(",      "func",   "Multiplies a matrix row by a scalar and adds it to another row"),
	token(b'\x19', "max(",        "func",     "Returns the maximum of two values or of all elements in a list"),
	token(b'\x1a', "min(",        "func",     "Returns the minimum of two values or of all elements in a list"),
	token(b'\x1b', "R►Pr(",       "func",     "Converts rectangular coordinates to the r component of polar coordinates", alt='R-to-Pr'),
	token(b'\x1c', "R►Pθ(",       "func",     "Converts rectangular coordinates to the θ component of polar coordinates", alt='R-to-P-theta'),
	token(b'\x1d', "P►Rx(",       "func",     "Converts polar coordinates to the x component of rectangular coordinates", alt='R-to-Px'),
	token(b'\x1e', "P►Ry(",       "func",     "Converts polar coordinates to the y component of rectangular coordinates", alt='R-to-Py'),
	token(b'\x1f', "median(",     "func",     "Returns the median of a list of values"),
	token(b'\x20', "randM(",      "func",   "Generates a random matrix of given dimensions with integer entries"),
	token(b'\x21', "mean(",       "func",     "Returns the arithmetic mean of a list"),
	token(b'\x22', "solve(",      "func",     "Numerically solves an equation for a specified variable near a given guess"),
	token(b'\x23', "seq(",        "func",     "Generates a list by evaluating an expression over a range of values"),
	token(b'\x24', "fnInt(",      "func",     "Numerically approximates the definite integral of a function"),
	token(b'\x25', "nDeriv(",     "func",     "Numerically approximates the derivative of a function at a point"),
	token(b'\x27', "fMin(",       "func",     "Finds the x-value at the minimum of a function on an interval"),
	token(b'\x28', "fMax(",       "func",     "Finds the x-value at the maximum of a function on an interval"),
	token(b'\x29', " ",           "str", "Space character used in strings and output", key=' '),
	QUOTE, COMMA,
	token(b'\x2c', "𝑖",           "val",     "The imaginary unit, equal to √(−1)",                        alt="imaginary"),  # maybe-key
	FACT,
	token(b'\x2e', "CubicReg ",    "cmd",     "Fits a cubic regression model to data"),
	token(b'\x2f', "QuartReg ",    "cmd",     "Fits a quartic regression model to data"),
	*[token(bytes([0x30 + i]), chr(0x30 + i), "num", f"Digit {chr(0x30 + i)}", key=chr(0x30 + i)) for i in range(10)],
	DOT, SCI_E, OR, XOR, COLON, NEWLINE, AND,
	# Variables A–Z (0x41–0x5A)
	*[token(bytes([0x41 + i]), chr(0x41 + i), "var", f"Real variable {chr(0x41 + i)}", key=chr(0x41 + i)) for i in range(26)],
	token(b'\x5b', "θ",           "var", "Variable theta",                                            alt="theta"),  # maybe-key
	PRGM,
	token(b'\x64', "Radian",      "mode",  "Sets angle mode to radians"),
	token(b'\x65', "Degree",      "mode",  "Sets angle mode to degrees"),
	token(b'\x66', "Normal",      "mode",  "Sets display notation to normal (non-scientific)"),
	token(b'\x67', "Sci",         "mode",  "Sets display notation to scientific notation"),
	token(b'\x68', "Eng",         "mode",  "Sets display notation to engineering notation"),
	token(b'\x69', "Float",       "mode",  "Sets the display to floating-point (full) decimal mode"),
	EQ, LT, GT, LE, GE, NE,
	ADD, SUB,
	ANS,
	token(b'\x73', "Fix",         "control",  "Sets the display to a fixed number of decimal places"),
	token(b'\x74', "Horiz",       "control",  "Sets the screen to horizontal split mode (graph + home)"),
	token(b'\x75', "Full",        "control",  "Sets the screen to full (non-split) mode"),
	token(b'\x76', "Func",        "control",  "Sets graphing mode to function (Y=) mode"),
	token(b'\x77', "Param",       "control",  "Sets graphing mode to parametric mode"),
	token(b'\x78', "Polar",       "control",  "Sets graphing mode to polar mode"),
	token(b'\x79', "Seq",         "control",  "Sets graphing mode to sequence mode"),
	token(b'\x7a', "IndpntAuto",  "control",  "Sets table independent variable to auto mode"),
	token(b'\x7b', "IndpntAsk",   "control",  "Sets table independent variable to ask mode"),
	token(b'\x7c', "DependAuto",  "control",  "Sets table dependent variable to auto mode"),
	token(b'\x7d', "DependAsk",   "control",  "Sets table dependent variable to ask mode"),
	token(b'\x7f', "<squaremark>",   "enum",  "", alt='square-mark'),
	token(b'\x80', "<crossmark>",   "enum",  "", alt='cross-mark'),
	token(b'\x81', "<dotmark>",   "enum",  "", alt='dot-mark'),
	MUL, DIV,
	token(b'\x84', "Trace",       "mode",       "Activates trace mode on the graph screen"),
	token(b'\x85', "ClrDraw",     "mode",       "Clears all drawn objects from the graph screen"),
	token(b'\x86', "ZStandard",   "mode",       "Sets the graphing window to standard zoom (±10 on both axes)"),
	token(b'\x87', "ZTrig",       "mode",       "Sets the graphing window optimized for trigonometric functions"),
	token(b'\x88', "ZBox",        "mode",       "Zooms in on a rectangular region you draw on the graph"),
	token(b'\x89', "Zoom In",     "mode",       "Zooms in on the graph centered at the cursor position",      alt="ZoomIn"),
	token(b'\x8a', "Zoom Out",    "mode",       "Zooms out on the graph centered at the cursor position",     alt="ZoomOut"),
	token(b'\x8b', "ZSquare",     "mode",       "Adjusts the window so pixels are square (equal aspect ratio)"),
	token(b'\x8c', "ZInteger",    "mode",       "Sets the zoom so each pixel represents one integer unit"),
	token(b'\x8d', "ZPrevious",   "mode",       "Restores the previous graphing window settings"),
	token(b'\x8e', "ZDecimal",    "mode",       "Sets the window so each pixel is 0.1 unit wide"),
	token(b'\x8f', "ZoomStat",    "mode",       "Adjusts the graphing window to show all stat plot data"),
	token(b'\x90', "ZoomRcl",     "mode",       "Restores a previously stored zoom window"),
	token(b'\x91', "PrintScreen", "mode", "Prints the current screen to a connected printer (legacy)"),
	token(b'\x92', "ZoomSto",     "mode",       "Saves the current graphing window settings"),
	token(b'\x93', "Text(",       "cmdfunc",       "Draws text on the graph screen at specified pixel coordinates"),
	NPR, NCR,
	token(b'\x96', "FnOn ",        "cmd",  "Turns on one or more Y= functions for graphing"),
	token(b'\x97', "FnOff ",       "cmd",  "Turns off one or more Y= functions for graphing"),
	token(b'\x98', "StorePic ",    "cmd",       "Saves the current graph screen image to a Pic variable"),
	token(b'\x99', "RecallPic ",   "cmd",       "Draws a previously stored Pic variable on the graph screen"),
	token(b'\x9a', "StoreGDB ",    "cmd",       "Saves the current graph database to a GDB variable"),
	token(b'\x9b', "RecallGDB ",   "cmd",       "Restores a graph database from a GDB variable"),
	token(b'\x9c', "Line(",       "cmdfunc",       "Draws or erases a line between two points on the graph screen"),
	token(b'\x9d', "Vertical ",    "cmd",       "Draws a vertical line at a specified x-value on the graph screen"),
	token(b'\x9e', "Pt-On(",      "cmdfunc",       "Turns on a point on the graph screen at specified coordinates"),
	token(b'\x9f', "Pt-Off(",     "cmdfunc",       "Turns off a point on the graph screen at specified coordinates"),
	token(b'\xa0', "Pt-Change(",  "cmdfunc",       "Toggles a point on the graph screen at specified coordinates"),
	token(b'\xa1', "Pxl-On(",     "cmdfunc",       "Turns on a pixel at specified row and column coordinates"),
	token(b'\xa2', "Pxl-Off(",    "cmdfunc",       "Turns off a pixel at specified row and column coordinates"),
	token(b'\xa3', "Pxl-Change(", "cmdfunc",       "Toggles a pixel at specified row and column coordinates"),
	token(b'\xa4', "Shade(",      "cmdfunc",       "Shades the area between two functions on the graph screen"),
	token(b'\xa5', "Circle(",     "cmdfunc",       "Draws a circle on the graph screen with a given center and radius"),
	token(b'\xa6', "Horizontal ",  "cmd",       "Draws a horizontal line at a specified y-value on the graph screen"),
	token(b'\xa7', "Tangent(",    "cmdfunc",       "Draws the tangent line to a function at a specified x-value"),
	token(b'\xa8', "DrawInv",     "io",       "Draws the inverse of a function on the graph screen"),
	token(b'\xa9', "DrawF",       "io",       "Draws a function on the graph screen"),
	token(b'\xab', "rand",        "var",     "Generates a uniformly random real number between 0 and 1"),  # TODO: is var correct?
	token(b'\xac', "π",           "val",     "The mathematical constant pi (≈3.14159...)",                 alt="pi"),  # maybe-key
	token(b'\xad', "getKey",      "val",       "Returns the keycode of the last key pressed, or 0 if none"),
	token(b'\xae', "'",           "str",   "Apostrophe / single-quote character",                        alt="apostrophe", key="'"),  # DMS MODE?
	token(b'\xaf', "?",           "str",       "Displays a question-mark prompt to wait for user input (legacy)", key='?'),
	NEG,
	token(b'\xb1', "int(",        "func",     "Returns the greatest integer less than or equal to a number (floor)"),
	token(b'\xb2', "abs(",        "func",     "Returns the absolute value of a number"),
	token(b'\xb3', "det(",        "func",   "Returns the determinant of a square matrix"),
	token(b'\xb4', "identity(",   "func",   "Returns an n×n identity matrix"),
	token(b'\xb5', "dim(",        "func",     "Returns the length of a list or the dimensions of a matrix"),
	token(b'\xb6', "sum(",        "func",     "Returns the sum of all elements in a list"),
	token(b'\xb7', "prod(",       "func",     "Returns the product of all elements in a list"),
	token(b'\xb8', "not(",        "func", "Returns the boolean NOT of a value (0→1, nonzero→0)"),
	token(b'\xb9', "iPart(",      "func",     "Returns the integer part (truncation toward zero) of a number"),
	token(b'\xba', "fPart(",      "func",     "Returns the fractional part of a number"),
	token(b'\xbc', "√(",          "func",     "Returns the square root of a non-negative number",           alt=("sqrt(", 'squareroot')),
	token(b'\xbd', "³√(",         "func",     "Returns the cube root of a number",                         alt=("cbrt(", 'cuberoot')),
	token(b'\xbe', "ln(",         "func",     "Returns the natural logarithm of a positive number"),
	token(b'\xbf', "e^(",         "func",     "Returns e raised to the specified power"),
	token(b'\xc0', "log(",        "func",     "Returns the base-10 logarithm of a positive number"),
	token(b'\xc1', "10^(",        "func",     "Returns 10 raised to the specified power"),
	token(b'\xc2', "sin(",        "func",     "Returns the sine of an angle"),
	token(b'\xc3', "sin⁻¹(",      "func",     "Returns the arcsine (inverse sine) of a value",             alt="arcsin("),
	token(b'\xc4', "cos(",        "func",     "Returns the cosine of an angle"),
	token(b'\xc5', "cos⁻¹(",      "func",     "Returns the arccosine (inverse cosine) of a value",         alt="arccos("),
	token(b'\xc6', "tan(",        "func",     "Returns the tangent of an angle"),
	token(b'\xc7', "tan⁻¹(",      "func",     "Returns the arctangent (inverse tangent) of a value",       alt="arctan("),
	token(b'\xc8', "sinh(",       "func",     "Returns the hyperbolic sine of a value"),
	token(b'\xc9', "sinh⁻¹(",     "func",     "Returns the inverse hyperbolic sine of a value",             alt="arcsinh("),
	token(b'\xca', "cosh(",       "func",     "Returns the hyperbolic cosine of a value"),
	token(b'\xcb', "cosh⁻¹(",     "func",     "Returns the inverse hyperbolic cosine of a value",          alt="arccosh("),
	token(b'\xcc', "tanh(",       "func",     "Returns the hyperbolic tangent of a value"),
	token(b'\xcd', "tanh⁻¹(",     "func",     "Returns the inverse hyperbolic tangent of a value",         alt="arctanh("),
	token(b'\xce', "If ",          "cmd",  "Conditionally executes the next statement or Then/Else block"),
	token(b'\xcf', "Then",        "cmd",  "Begins the body of an If block when the condition is true"),
	token(b'\xd0', "Else",        "cmd",  "Begins the alternate body of an If-Then block when the condition is false"),
	token(b'\xd1', "While ",       "cmd",  "Repeats a block as long as a condition remains true"),
	token(b'\xd2', "Repeat ",      "cmd",  "Repeats a block until a condition becomes true (always runs at least once)"),
	token(b'\xd3', "For(",        "cmdfunc",  "Iterates a variable from a start to an end value by a step"),  # TODO: Should be command?
	token(b'\xd4', "End",         "cmd",  "Marks the end of an If-Then, While, Repeat, or For block"),
	token(b'\xd5', "Return",      "cmd",  "Exits the current program or subprogram and returns to the caller"),
	token(b'\xd6', "Lbl ",         "cmd",  "Defines a label that can be targeted by Goto"),
	token(b'\xd7', "Goto ",        "cmd",  "Jumps execution unconditionally to a specified label"),
	token(b'\xd8', "Pause ",       "cmd",  "Pauses program execution until the user presses ENTER"),
	token(b'\xd9', "Stop",        "cmd",  "Terminates program execution immediately"),
	token(b'\xda', "IS>(",        "cmdfunc",  "Increments a variable and skips the next statement if it exceeds a limit"),
	token(b'\xdb', "DS<(",        "cmdfunc",  "Decrements a variable and skips the next statement if it goes below a limit"),
	token(b'\xdc', "Input ",       "cmd",       "Prompts the user to enter a value or string"),
	token(b'\xdd', "Prompt ",      "cmd",       "Prompts the user to enter values for one or more variables"),
	token(b'\xde', "Disp ",        "cmd",       "Displays values or strings on the home screen"),
	token(b'\xdf', "DispGraph",   "cmd",       "Displays the current graph screen"),
	token(b'\xe0', "Output(",     "cmdfunc",       "Displays a value or string at a specific row and column on the home screen"),
	token(b'\xe1', "ClrHome",     "cmd",       "Clears the home screen"),
	token(b'\xe2', "Fill(",       "func",     "Fills all elements of a list or matrix with a specified value"),
	token(b'\xe3', "SortA(",      "func",     "Sorts a list in ascending order in-place"),
	token(b'\xe4', "SortD(",      "func",     "Sorts a list in descending order in-place"),
	token(b'\xe5', "DispTable",   "cmd",       "Displays the function table"),
	token(b'\xe6', "Menu(",       "cmdfunc",       "Displays a menu and branches to a label based on user selection"),
	token(b'\xe7', "Send(",       "cmdfunc",       "Sends a list to a connected CBL/CBR device"),
	token(b'\xe8', "Get(",        "cmdfunc",       "Retrieves a list from a connected CBL/CBR device"),
	token(b'\xe9', "PlotsOn",     "cmd",     "Turns on one or all stat plots"),
	token(b'\xea', "PlotsOff",    "cmd",     "Turns off one or all stat plots"),
	LIST_PREFIX,
	token(b'\xec', "Plot1(",      "cmdfunc",     "Configures stat Plot 1 with a type and data sources"),
	token(b'\xed', "Plot2(",      "cmdfunc",     "Configures stat Plot 2 with a type and data sources"),
	token(b'\xee', "Plot3(",      "cmdfunc",     "Configures stat Plot 3 with a type and data sources"),
	POW, XROOT,
	token(b'\xf2', "1-Var Stats ", "cmd",     "Computes one-variable statistics for a dataset"),
	token(b'\xf3', "2-Var Stats ", "cmd",     "Computes two-variable statistics for a paired dataset"),
	token(b'\xf4', "LinReg(a+bx) ","cmd",     "Fits a linear regression of the form a+bx to data"),
	token(b'\xf5', "ExpReg ",      "cmd",     "Fits an exponential regression model to data"),
	token(b'\xf6', "LnReg ",       "cmd",     "Fits a logarithmic regression model to data"),
	token(b'\xf7', "PwrReg ",      "cmd",     "Fits a power regression model to data"),
	token(b'\xf8', "Med-Med ",     "cmd",     "Fits a median-median line to data"),
	token(b'\xf9', "QuadReg ",     "cmd",     "Fits a quadratic regression model to data"),
	token(b'\xfa', "ClrList ",    "cmd",     "Clears the contents of one or more lists"),
	token(b'\xfb', "ClrTable",    "cmd",       "Clears the function table values"),
	token(b'\xfc', "Histogram",   "enum",     "Selects the histogram plot type for a stat plot"),
	token(b'\xfd', "xyLine",      "enum",     "Selects the xy-line (connected scatter) plot type for a stat plot"),
	token(b'\xfe', "Scatter",     "enum",     "Selects the scatter plot type for a stat plot"),
	token(b'\xff', "LinReg(ax+b) ","cmd",     "Fits a linear regression of the form ax+b to data"),

	# Two-byte: Matrix variables 0x5C xx
	*[token(bytes([0x5c, i]), f"[{chr(0x41 + i)}]", "var", f"Matrix variable {chr(0x41 + i)}", alt=f"mat{chr(0x41 + i)}") for i in range(10)],

	# Two-byte: List variables 0x5D xx
	*[token(bytes([0x5d, i]), f"L{chr(0x2081 + i)}", "var", f"Built-in list variable L{i}", alt=f"L{i + 1}") for i in range(0, 6)],

	# Two-byte: Y= equation variables 0x5E xx
	*[token(
		bytes([0x5e, 0x10 + i]), 
		f"Y{chr(0x2080 + (i + 1) % 10)}", 
		"var", 
		f"Function Y{(i + 1) % 10}", 
		alt=f"Y{(i + 1) % 10}"
	) for i in range(10)],

	*[token(
		bytes([0x5e, 0x20 + i]),
		f"{x}{chr(0x2080 + n)}ₜ", 
		"var", 
		f"Parametric function {x}{n}", 
		alt=f"{x}{n}t"
	) for i, (n, x) in enumerate(itertools.product(range(1, 7), 'XY'))],

	*[token(
		bytes([0x5e, 0x40 + i]), 
		f"r{chr(0x2081 + i)}", 
		"var", 
		f"Polar function r{i + 1}", 
		alt=f"r{(i + 1)}"
	) for i in range(6)],
	
	token(b'\x5e\x80', "u",   "var", "Sequence function u",                alt="sequence-u"),
	token(b'\x5e\x81', "v",   "var", "Sequence function v",                alt="sequence-v"),
	token(b'\x5e\x82', "w",   "var", "Sequence function w",                alt="sequence-w"),

	# Two-byte: Picture variables 0x60 xx
	*[token(bytes([0x60, i]), f"Pic{(i + 1) % 10}", "var", f"Picture variable Pic{(i + 1) % 10}") for i in range(10)],

	# Two-byte: GDB variables 0x61 xx
	*[token(bytes([0x61, i]), f"GDB{(i + 1) % 10}", "var", f"Graph database variable GDB{(i + 1) % 10}") for i in range(10)],

	# Two-byte: String variables 0xAA xx (Str1=0x00 … Str9=0x08, Str0=0x09)
	*[token(bytes([0xaa, i]), f"Str{(i + 1) % 10}", "var", f"String variable Str{(i + 1) % 10}") for i in range(10)],

	# Two-byte: Statistics variables 0x62 xx
	token(b'\x62\x01', "RegEq", "stat", "The regression equation stored by the most recent regression command"),
	token(b'\x62\x02', "n",     "stat", "Number of data points used in 1-Var Stats"),
	token(b'\x62\x03', "x̄",    "stat", "Sample mean of x-values from 1-Var Stats",              alt="x-mean"),
	token(b'\x62\x04', "Σx",   "stat", "Sum of x-values from 1-Var Stats",                      alt="sum-x"),
	token(b'\x62\x05', "Σx²",  "stat", "Sum of squared x-values from 1-Var Stats",              alt="sum-x^2"),
	token(b'\x62\x06', "Sx",   "stat", "Sample standard deviation of x from 1-Var Stats"),
	token(b'\x62\x07', "σx",   "stat", "Population standard deviation of x from 1-Var Stats",   alt="sigma-x"),
	token(b'\x62\x08', "minX", "stat", "Minimum x-value from 1-Var Stats"),
	token(b'\x62\x09', "maxX", "stat", "Maximum x-value from 1-Var Stats"),
	token(b'\x62\x0a', "minY", "stat", "Minimum y-value from 2-Var Stats"),
	token(b'\x62\x0b', "maxY", "stat", "Maximum y-value from 2-Var Stats"),
	token(b'\x62\x0c', "ȳ",    "stat", "Sample mean of y-values from 2-Var Stats",              alt="y-mean"),
	token(b'\x62\x0d', "Σy",   "stat", "Sum of y-values from 2-Var Stats",                      alt="sum-y"),
	token(b'\x62\x0e', "Σy²",  "stat", "Sum of squared y-values from 2-Var Stats",              alt="sum-y^2"),
	token(b'\x62\x0f', "Sy",   "stat", "Sample standard deviation of y from 2-Var Stats"),
	token(b'\x62\x10', "σy",   "stat", "Population standard deviation of y from 2-Var Stats",   alt="sigma-y"),
	token(b'\x62\x11', "Σxy",  "stat", "Sum of x*y products from 2-Var Stats",                  alt="sum-xy"),
	token(b'\x62\x12', "r",    "stat", "Linear correlation coefficient from regression"),
	token(b'\x62\x13', "Med",  "stat", "Median value from 1-Var Stats"),
	token(b'\x62\x14', "Q1",   "stat", "First quartile (Q1) from 1-Var Stats"),
	token(b'\x62\x15', "Q3",   "stat", "Third quartile (Q3) from 1-Var Stats"),
	token(b'\x62\x16', "a",    "stat", "Regression coefficient a from the most recent regression"),
	token(b'\x62\x17', "b",    "stat", "Regression coefficient b from the most recent regression"),
	token(b'\x62\x18', "c",    "stat", "Regression coefficient c from the most recent regression"),
	token(b'\x62\x19', "d",    "stat", "Regression coefficient d from the most recent regression"),
	token(b'\x62\x1a', "e",    "stat", "Regression coefficient e from the most recent regression"),
	token(b'\x62\x1b', "x₁",  "stat", "x-value 1 from sinusoidal or other regression",           alt="x1"),
	token(b'\x62\x1c', "x₂",  "stat", "x-value 2 from sinusoidal or other regression",           alt="x2"),
	token(b'\x62\x1d', "x₃",  "stat", "x-value 3 from sinusoidal or other regression",           alt="x3"),
	token(b'\x62\x1e', "y₁",  "stat", "y-value 1 from sinusoidal or other regression",           alt="y1"),
	token(b'\x62\x1f', "y₂",  "stat", "y-value 2 from sinusoidal or other regression",           alt="y2"),
	token(b'\x62\x20', "y₃",  "stat", "y-value 3 from sinusoidal or other regression",           alt="y3"),
	token(b'\x62\x21', "n",   "stat", "Sample size n from hypothesis test output"),
	token(b'\x62\x22', "p",   "stat", "p-value from hypothesis test output"),
	token(b'\x62\x23', "z",   "stat", "z-statistic from hypothesis test output"),
	token(b'\x62\x24', "t",   "stat", "t-statistic from hypothesis test output"),
	token(b'\x62\x25', "χ²",  "stat", "Chi-squared statistic from hypothesis test output",       alt="chi2"),
	token(b'\x62\x26', "F",   "stat", "F-statistic from ANOVA or regression test output"),
	token(b'\x62\x27', "df",  "stat", "Degrees of freedom from hypothesis test output"),
	token(b'\x62\x28', "p̂",  "stat", "Estimated proportion from 1-Prop test output",            alt="p-hat"),
	token(b'\x62\x29', "p̂₁", "stat", "Estimated proportion from sample 1 in 2-Prop test",       alt="p-hat1"),
	token(b'\x62\x2a', "p̂₂", "stat", "Estimated proportion from sample 2 in 2-Prop test",       alt="p-hat2"),
	token(b'\x62\x2b', "x̄₁", "stat", "Sample mean of x from sample 1 in 2-Samp test",          alt="x-mean1"),
	token(b'\x62\x2c', "Sx₁", "stat", "Sample standard deviation from sample 1 in 2-Samp test", alt="Sx1"),
	token(b'\x62\x2d', "n₁",  "stat", "Sample size of sample 1 in 2-Samp test",                 alt="n1"),
	token(b'\x62\x2e', "x̄₂", "stat", "Sample mean of x from sample 2 in 2-Samp test",          alt="x-mean2"),
	token(b'\x62\x2f', "Sx₂", "stat", "Sample standard deviation from sample 2 in 2-Samp test", alt="Sx2"),
	token(b'\x62\x30', "n₂",  "stat", "Sample size of sample 2 in 2-Samp test",                 alt="n2"),
	token(b'\x62\x31', "Sxp", "stat", "Pooled sample standard deviation from 2-Samp t-test"),
	token(b'\x62\x32', "lower","stat", "Lower bound of a confidence interval"),
	token(b'\x62\x33', "upper","stat", "Upper bound of a confidence interval"),
	token(b'\x62\x34', "s",    "stat", "Standard deviation from a regression or test output"),
	token(b'\x62\x35', "r²",   "stat", "Coefficient of determination from regression",           alt="r^2"),
	token(b'\x62\x36', "R²",   "stat", "Coefficient of determination (alternate form)",          alt="R^2"),
	token(b'\x62\x37', "Factor df", "stat", "Degrees of freedom for the factor in one-way ANOVA",    alt="FactorDF"),
	token(b'\x62\x38', "Factor SS", "stat", "Sum of squares for the factor in one-way ANOVA",        alt="FactorSS"),
	token(b'\x62\x39', "Factor MS", "stat", "Mean square for the factor in one-way ANOVA",           alt="FactorMS"),
	token(b'\x62\x3a', "Error df",  "stat", "Degrees of freedom for the error in one-way ANOVA",     alt="ErrorDF"),
	token(b'\x62\x3b', "Error SS",  "stat", "Sum of squares for the error in one-way ANOVA",         alt="ErrorSS"),
	token(b'\x62\x3c', "Error MS",  "stat", "Mean square for the error in one-way ANOVA",            alt="ErrorMS"),

	# Two-byte: Window / Finance variables 0x63 xx
	token(b'\x63\x02', "Xscl",     "var", "X-axis tick mark spacing for the graphing window"),
	token(b'\x63\x03', "Yscl",     "var", "Y-axis tick mark spacing for the graphing window"),
	token(b'\x63\x0a', "Xmin",     "var", "Minimum x-value of the graphing window"),
	token(b'\x63\x0b', "Xmax",     "var", "Maximum x-value of the graphing window"),
	token(b'\x63\x0c', "Ymin",     "var", "Minimum y-value of the graphing window"),
	token(b'\x63\x0d', "Ymax",     "var", "Maximum y-value of the graphing window"),
	token(b'\x63\x0e', "Tmin",     "var", "Minimum value of parameter T in parametric mode"),
	token(b'\x63\x0f', "Tmax",     "var", "Maximum value of parameter T in parametric mode"),
	token(b'\x63\x10', "θmin",     "var", "Minimum angle in polar mode",                    alt="theta-min"),
	token(b'\x63\x11', "θmax",     "var", "Maximum angle in polar mode",                    alt="theta-max"),
	token(b'\x63\x1a', "TblStart", "var", "Starting value for the function table"),
	token(b'\x63\x1b', "PlotStart","var", "Starting term number for sequence plotting"),
	token(b'\x63\x1d', "nMax",     "var", "Maximum term number for sequence graphing"),
	token(b'\x63\x1f', "nMin",     "var", "Minimum term number for sequence graphing"),
	token(b'\x63\x21', "ΔTbl",     "var", "Table step (increment) for the function table",  alt="dTbl"),
	token(b'\x63\x22', "Tstep",    "var", "Step size for parameter T in parametric mode"),
	token(b'\x63\x23', "θstep",    "var", "Step size for angle in polar mode",              alt="theta-step"),
	token(b'\x63\x26', "ΔX",       "var", "Width of one pixel in the current graphing window", alt="dX"),
	token(b'\x63\x27', "ΔY",       "var", "Height of one pixel in the current graphing window", alt="dY"),
	token(b'\x63\x28', "XFact",    "var", "Zoom factor for x used by Zoom In / Zoom Out"),
	token(b'\x63\x29', "YFact",    "var", "Zoom factor for y used by Zoom In / Zoom Out"),
	token(b'\x63\x2b', "N",        "var", "Number of payment periods in TVM calculations"),
	token(b'\x63\x2c', "I%",       "var", "Annual interest rate (%) for TVM calculations"),
	token(b'\x63\x2d', "PV",       "var", "Present value for TVM calculations"),
	token(b'\x63\x2e', "PMT",      "var", "Payment amount per period for TVM calculations"),
	token(b'\x63\x2f', "FV",       "var", "Future value for TVM calculations"),
	token(b'\x63\x30', "P/Y",      "var", "Payments per year for TVM calculations"),
	token(b'\x63\x31', "C/Y",      "var", "Compounding periods per year for TVM calculations"),
	token(b'\x63\x34', "PlotStep", "var", "Step between terms plotted in sequence graphing mode"),
	token(b'\x63\x36', "Xres",     "var", "Graph resolution (1=every pixel, 8=every 8th pixel)"),

	# Two-byte: Graph format tokens 0x7E xx
	token(b'\x7e\x00', "Sequential", "cmd", "Sets graphs to plot in sequential order"),
	token(b'\x7e\x01', "Simul",      "cmd", "Sets graphs to plot all functions simultaneously"),
	token(b'\x7e\x02', "PolarGC",    "cmd", "Sets graph coordinates display to polar (r, θ) format"),
	token(b'\x7e\x03', "RectGC",     "cmd", "Sets graph coordinates display to rectangular (x, y) format"),
	token(b'\x7e\x04', "CoordOn",    "cmd", "Turns on coordinate display while tracing"),
	token(b'\x7e\x05', "CoordOff",   "cmd", "Turns off coordinate display while tracing"),
	token(b'\x7e\x06', "Connected",  "cmd", "Sets graphing to connect plotted points with line segments"),
	token(b'\x7e\x07', "Dot",        "cmd", "Sets graphing to plot individual points without connecting them"),
	token(b'\x7e\x08', "AxesOn",     "cmd", "Turns on the x- and y-axes on the graph screen"),
	token(b'\x7e\x09', "AxesOff",    "cmd", "Turns off the x- and y-axes on the graph screen"),
	token(b'\x7e\x0a', "GridOn",     "cmd", "Turns on the grid of dots on the graph screen"),
	token(b'\x7e\x0b', "GridOff",    "cmd", "Turns off the grid on the graph screen"),
	token(b'\x7e\x0c', "LabelOn",    "cmd", "Turns on axis labels (X and Y) on the graph screen"),
	token(b'\x7e\x0d', "LabelOff",   "cmd", "Turns off axis labels on the graph screen"),
	token(b'\x7e\x0e', "Web",        "cmd", "Sets sequence graphing to web (cobweb) plot format"),
	token(b'\x7e\x0f', "Time",       "cmd", "Sets sequence graphing to time (term vs. value) plot format"),
	token(b'\x7e\x10', "uvAxes",     "cmd", "Sets sequence graphing to plot u vs. v"),
	token(b'\x7e\x11', "vwAxes",     "cmd", "Sets sequence graphing to plot v vs. w"),
	token(b'\x7e\x12', "uwAxes",     "cmd", "Sets sequence graphing to plot u vs. w"),

	# Two-byte: Miscellaneous tokens 0xBB xx
	token(b'\xbb\x00', "npv(",         "func",   "Calculates net present value of a series of cash flows"),
	token(b'\xbb\x01', "irr(",         "func",   "Calculates the internal rate of return of a series of cash flows"),
	token(b'\xbb\x02', "bal(",         "func",   "Returns the balance of a loan after a specified payment number"),
	token(b'\xbb\x03', "Σprn(",        "func",   "Returns the sum of principal paid between two payment numbers", alt="sum-prn"),
	token(b'\xbb\x04', "ΣInt(",        "func",   "Returns the sum of interest paid between two payment numbers", alt="sum-int"),
	token(b'\xbb\x05', "►Nom(",        "func",   "Converts an effective interest rate to a nominal rate",        alt="to-Nom"),
	token(b'\xbb\x06', "►Eff(",        "func",   "Converts a nominal interest rate to an effective rate",        alt="to-Eff"),
	token(b'\xbb\x07', "dbd(",         "func",   "Calculates the number of days between two dates"),
	token(b'\xbb\x08', "lcm(",         "func",   "Returns the least common multiple of two integers"),
	token(b'\xbb\x09', "gcd(",         "func",   "Returns the greatest common divisor of two integers"),
	token(b'\xbb\x0a', "randInt(",     "func",   "Generates a uniformly random integer between two bounds"),
	token(b'\xbb\x0b', "randBin(",     "func",   "Generates a random integer from a binomial distribution"),
	token(b'\xbb\x0c', "sub(",         "func", "Extracts a substring from a string"),
	token(b'\xbb\x0d', "stdDev(",      "func",   "Returns the sample standard deviation of a list"),
	token(b'\xbb\x0e', "variance(",    "func",   "Returns the sample variance of a list"),
	token(b'\xbb\x0f', "inString(",    "func", "Returns the position of a substring within a string"),
	token(b'\xbb\x10', "normalcdf(",   "func",   "Computes the normal distribution CDF between two bounds"),
	token(b'\xbb\x11', "invNorm(",     "func",   "Returns the inverse normal (z-score) for a given area"),
	token(b'\xbb\x12', "tcdf(",        "func",   "Computes the Student's t distribution CDF between two bounds"),
	token(b'\xbb\x13', "χ²cdf(",       "func",   "Computes the chi-squared distribution CDF between two bounds", alt="chi2cdf"),
	token(b'\xbb\x14', "Fcdf(",        "func",   "Computes the F distribution CDF between two bounds"),
	token(b'\xbb\x15', "binompdf(",    "func",   "Computes the binomial probability for a given number of successes"),
	token(b'\xbb\x16', "binomcdf(",    "func",   "Computes the cumulative binomial probability"),
	token(b'\xbb\x17', "poissonpdf(",  "func",   "Computes the Poisson probability for a given number of events"),
	token(b'\xbb\x18', "poissoncdf(",  "func",   "Computes the cumulative Poisson probability"),
	token(b'\xbb\x19', "geometpdf(",   "func",   "Computes the geometric probability for the first success on trial k"),
	token(b'\xbb\x1a', "geometcdf(",   "func",   "Computes the cumulative geometric probability up to trial k"),
	token(b'\xbb\x1b', "normalpdf(",   "func",   "Computes the normal probability density at a given value"),
	token(b'\xbb\x1c', "tpdf(",        "func",   "Computes the Student's t probability density at a given value"),
	token(b'\xbb\x1d', "χ²pdf(",       "func",   "Computes the chi-squared probability density at a given value", alt="chi2pdf"),
	token(b'\xbb\x1e', "Fpdf(",        "func",   "Computes the F distribution probability density at a given value"),
	token(b'\xbb\x1f', "randNorm(",    "func",   "Generates a random number from a normal distribution"),
	token(b'\xbb\x20', "tvm_Pmt",      "finfunc",   "Computes the payment amount for a TVM calculation"),
	token(b'\xbb\x21', "tvm_I%",       "finfunc",   "Computes the interest rate for a TVM calculation"),
	token(b'\xbb\x22', "tvm_PV",       "finfunc",   "Computes the present value for a TVM calculation"),
	token(b'\xbb\x23', "tvm_N",        "finfunc",   "Computes the number of periods for a TVM calculation"),
	token(b'\xbb\x24', "tvm_FV",       "finfunc",   "Computes the future value for a TVM calculation"),
	token(b'\xbb\x25', "conj(",        "func",   "Returns the complex conjugate of a complex number"),
	token(b'\xbb\x26', "real(",        "func",   "Returns the real part of a complex number"),
	token(b'\xbb\x27', "imag(",        "func",   "Returns the imaginary part of a complex number"),
	token(b'\xbb\x28', "angle(",       "func",   "Returns the polar angle (argument) of a complex number"),
	token(b'\xbb\x29', "cumSum(",      "func",   "Returns a list of cumulative sums from a list"),
	token(b'\xbb\x2a', "expr(",        "func", "Evaluates a string as a mathematical expression"),
	token(b'\xbb\x2b', "length(",      "func", "Returns the number of characters in a string"),
	token(b'\xbb\x2c', "ΔList(",       "func",   "Returns a list of first differences of a list",           alt="dList("),
	token(b'\xbb\x2d', "ref(",         "func", "Reduces a matrix to row-echelon form"),
	token(b'\xbb\x2e', "rref(",        "func", "Reduces a matrix to reduced row-echelon form"),
	token(b'\xbb\x2f', "►Rect",        "converter",   "Converts a complex number from polar to rectangular display", alt="to-Rect"),
	token(b'\xbb\x30', "►Polar",       "converter",   "Converts a complex number from rectangular to polar display", alt="to-Polar"),
	token(b'\xbb\x31', "𝑒",            "val",   "The mathematical constant e (≈2.71828...)"),
	token(b'\xbb\x32', "SinReg ",       "cmd",   "Fits a sinusoidal regression model to data"),
	token(b'\xbb\x33', "Logistic ",     "cmd",   "Fits a logistic regression model to data"),
	token(b'\xbb\x34', "LinRegTTest ",  "cmd",   "Performs a linear regression t-test"),
	token(b'\xbb\x35', "ShadeNorm(",   "cmdfunc",   "Shades the area under a normal curve between two bounds on the graph"),
	token(b'\xbb\x36', "Shade_t(",     "cmdfunc",   "Shades the area under a t distribution curve between two bounds"),
	token(b'\xbb\x37', "Shadeχ²(",     "cmdfunc",   "Shades the area under a chi-squared curve between two bounds", alt="shade-chi^2"),
	token(b'\xbb\x38', "ShadeF(",      "cmdfunc",   "Shades the area under an F distribution curve between two bounds"),
	token(b'\xbb\x39', "Matr►list(",   "cmdfunc", "Copies columns of a matrix into lists",                  alt="Matr-to-list"),
	token(b'\xbb\x3a', "List►matr(",   "cmdfunc",   "Fills columns of a matrix from lists",                   alt="List-to-matr"),
	token(b'\xbb\x3b', "Z-Test(",      "cmdfunc",   "Performs a one-sample z-test for a mean"),
	token(b'\xbb\x3c', "T-Test",       "cmdfunc",   "Performs a one-sample t-test for a mean"),
	token(b'\xbb\x3d', "2-SampZTest(", "cmdfunc",   "Performs a two-sample z-test comparing two means"),
	token(b'\xbb\x3e', "1-PropZTest(", "cmdfunc",   "Performs a one-proportion z-test"),
	token(b'\xbb\x3f', "2-PropZTest(", "cmdfunc",   "Performs a two-proportion z-test"),
	token(b'\xbb\x40', "χ²-Test(",     "cmdfunc",   "Performs a chi-squared test for association on a matrix", alt="chi^2-test"),
	token(b'\xbb\x41', "ZInterval ",    "cmd",   "Computes a one-sample z confidence interval for a mean"),
	token(b'\xbb\x42', "2-SampZInt(",  "cmdfunc",   "Computes a two-sample z confidence interval"),
	token(b'\xbb\x43', "1-PropZInt(",  "cmdfunc",   "Computes a one-proportion z confidence interval"),
	token(b'\xbb\x44', "2-PropZInt(",  "cmdfunc",   "Computes a two-proportion z confidence interval"),
	token(b'\xbb\x45', "GraphStyle(",  "cmdfunc","Sets the line style for a Y= function"),
	token(b'\xbb\x46', "2-SampTTest ",  "cmd",   "Performs a two-sample t-test comparing two means"),
	token(b'\xbb\x47', "2-SampFTest ",  "cmd",   "Performs an F-test comparing two population variances"),
	token(b'\xbb\x48', "TInterval ",    "cmd",   "Computes a one-sample t confidence interval for a mean"),
	token(b'\xbb\x49', "2-SampTInt ",   "cmd",   "Computes a two-sample t confidence interval"),
	token(b'\xbb\x4a', "SetUpEditor ",  "cmd",   "Sets up the stat list editor with specified lists"),
	token(b'\xbb\x4b', "Pmt_End",      "cmd",   "Sets TVM payments to occur at end of period"),
	token(b'\xbb\x4c', "Pmt_Bgn",      "cmd",   "Sets TVM payments to occur at beginning of period"),
	token(b'\xbb\x4d', "Real",         "cmd","Sets the calculator to real-number mode"),
	token(b'\xbb\x4e', "re^θi",        "cmd","Sets the calculator to polar complex-number mode",        alt=('re^theta-i', "polar_complex")),
	token(b'\xbb\x4f', "a+bi",         "cmd","Sets the calculator to rectangular complex-number mode",  alt="rect_complex"),
	token(b'\xbb\x50', "ExprOn",       "cmd","Turns on expression display during tracing"),
	token(b'\xbb\x51', "ExprOff",      "cmd","Turns off expression display during tracing"),
	token(b'\xbb\x52', "ClrAllLists",  "cmd",   "Clears all list variables in memory"),
	token(b'\xbb\x53', "GetCalc(",     "cmdfunc",     "Retrieves a variable from a linked calculator"),
	token(b'\xbb\x54', "DelVar ",       "","Deletes a variable from memory"),
	token(b'\xbb\x55', "Equ►String(",  "cmdfunc", "Converts a Y= equation variable to a string",            alt="Equ-to-Str"),
	token(b'\xbb\x56', "String►Equ(",  "cmdfunc", "Stores a string into a Y= equation variable",            alt="Str-to-Equ"),
	token(b'\xbb\x57', "Clear Entries","cmd",     "Clears the calculator's entry history",                   alt="ClearEntries"),
	token(b'\xbb\x58', "Select(",      "cmdfunc",   "Selects elements of two lists based on a stat plot selection"),
	token(b'\xbb\x59', "ANOVA(",       "cmdfunc",   "Performs a one-way analysis of variance on two or more lists"),
	token(b'\xbb\x5a', "ModBoxplot",   "enum",   "Selects the modified box-and-whisker plot type (shows outliers)"),
	token(b'\xbb\x5b', "NormProbPlot", "enum",   "Selects the normal probability plot type for a stat plot"),
	token(b'\xbb\x64', "G-T",          "cmd","Sets the screen to graph-table split mode"),
	token(b'\xbb\x65', "ZoomFit",      "cmd",     "Adjusts the y-window to fit the function given the current x-window"),
	token(b'\xbb\x66', "DiagnosticOn", "cmd",   "Turns on display of r and r² in regression output"),
	token(b'\xbb\x67', "DiagnosticOff","cmd",   "Turns off display of r and r² in regression output"),
	token(b'\xbb\x68', "Archive ",     "cmd","Moves a variable from RAM to the archive (Flash) memory"),
	token(b'\xbb\x69', "UnArchive ",   "cmd","Moves a variable from archive memory back to RAM"),
	token(b'\xbb\x6a', "Asm(",         "cmdfunc","Executes an assembly language program"),
	token(b'\xbb\x6b', "AsmComp(",     "cmdfunc","Compiles a tokenized assembly source program"),
	token(b'\xbb\x6c', "AsmPrgm",      "","Marker token indicating an assembly program"),
	token(b'\xbb\x6d', "<compiledasm>", "","?"),

	# Accented Latin characters (0xBB6E–0xBB99; 0xBB7E unused — uppercase I-acute absent)
	*[token(bytes([0xBB, b]), ch, "str", name, alt=name)
		for b, ch, name in [
			(0x6e, "Á", "A-acute"),    (0x6f, "À", "A-grave"),    (0x70, "Â", "A-circumflex"), (0x71, "Ä", "A-umlaut"),
			(0x72, "á", "a-acute"),    (0x73, "à", "a-grave"),    (0x74, "â", "a-circumflex"), (0x75, "ä", "a-umlaut"),
			(0x76, "É", "E-acute"),    (0x77, "È", "E-grave"),    (0x78, "Ê", "E-circumflex"), (0x79, "Ë", "E-umlaut"),
			(0x7a, "é", "e-acute"),    (0x7b, "è", "e-grave"),    (0x7c, "ê", "e-circumflex"), (0x7d, "ë", "e-umlaut"),
			(0x7f, "Ì", "I-grave"),    (0x80, "Î", "I-circumflex"),(0x81, "Ï", "I-umlaut"),
			(0x82, "í", "i-acute"),    (0x83, "ì", "i-grave"),    (0x84, "î", "i-circumflex"), (0x85, "ï", "i-umlaut"),
			(0x86, "Ó", "O-acute"),    (0x87, "Ò", "O-grave"),    (0x88, "Ô", "O-circumflex"), (0x89, "Ö", "O-umlaut"),
			(0x8a, "ó", "o-acute"),    (0x8b, "ò", "o-grave"),    (0x8c, "ô", "o-circumflex"), (0x8d, "ö", "o-umlaut"),
			(0x8e, "Ú", "U-acute"),    (0x8f, "Ù", "U-grave"),    (0x90, "Û", "U-circumflex"), (0x91, "Ü", "U-umlaut"),
			(0x92, "ú", "u-acute"),    (0x93, "ù", "u-grave"),    (0x94, "û", "u-circumflex"), (0x95, "ü", "u-umlaut"),
			(0x96, "Ç", "C-cedilla"),  (0x97, "ç", "c-cedilla"),
			(0x98, "Ñ", "N-tilde"),    (0x99, "ñ", "n-tilde"),
		]],
	
	token(b'\xbb\x9a', "´", "str", "Acute accent", alt="acute-accent"),
	token(b'\xbb\x9b', "`", "str", "Grave accent", alt="grave-accent"),
	token(b'\xbb\x9c', "¨", "str", "Diaeresis / umlaut accent", alt="umlaut-accent"),
	token(b'\xbb\x9d', "¿", "str", "Inverted question mark", alt="?-inverted"),
	token(b'\xbb\x9e', "¡", "str", "Inverted exclamation mark", alt="!-inverted"),
	token(b'\xbb\x9f', "α", "str", "alpha",  alt="alpha"),
	token(b'\xbb\xa0', "β", "str", "beta",   alt="beta"),
	token(b'\xbb\xa1', "γ", "str", "gamma",  alt="gamma"),
	token(b'\xbb\xa2', "Δ", "str", "Delta",  alt="Delta"),
	token(b'\xbb\xa3', "δ", "str", "delta",  alt="delta"),
	token(b'\xbb\xa4', "ε", "str", "epsilon", alt="epsilon"),
	token(b'\xbb\xa5', "λ", "str", "lambda", alt="lambda"),
	token(b'\xbb\xa6', "μ", "str", "mu",     alt="mu"),
	token(b'\xbb\xa7', "π", "str", "pi (non-mathematical)", alt="pi-non-math"),  # deprecated
	token(b'\xbb\xa8', "ρ", "str", "rho",    alt="rho"),
	token(b'\xbb\xa9', "Σ", "str", "Sigma",  alt="Sigma"),
	token(b'\xbb\xab', "φ", "str", "phi",    alt="phi"),
	token(b'\xbb\xac', "Ω", "str", "Omega",  alt="Omega"),
	token(b'\xbb\xad', "ψ", "str", "Greek psi", alt="psi"),
	token(b'\xbb\xae', "χ", "str", "chi",    alt="chi"),
	token(b'\xbb\xaf', "F", "str", "Italic F used in F-statistic display"),
	
	# Lowercase letters a–k (0xBBB0–0xBBBA; 0xBBBB is unused)
	*[token(bytes([0xBB, 0xB0 + i]), chr(0x61 + i), "str", f"Lowercase {chr(0x61 + i)}", key=chr(0x61 + i)) for i in range(11)],
	# Lowercase letters l–z (0xBBBC–0xBBCA)
	*[token(bytes([0xBB, 0xBC + i]), chr(0x6C + i), "str", f"Lowercase {chr(0x6C + i)}", key=chr(0x6C + i)) for i in range(15)],
	
	token(b'\xbb\xcb', "σ", "str", "sigma (statistics display)", alt="sigma"),
	token(b'\xbb\xcc', "τ", "str", "tau (display character)", alt="tau"),
	token(b'\xbb\xcd', "Í", "str", "I-acute (extended)", alt="I-acute"),
	token(b'\xbb\xce', "GarbageCollect", "cmd", "Defragments archive memory to recover space"),
	token(b'\xbb\xcf', "~", "str", "Tilde character", alt="tilde"),
	token(b'\xbb\xd1', "@", "str", "At sign",       alt="at-sign",   key='@'),
	token(b'\xbb\xd2', "#", "str", "Number/hash sign", alt="hash",   key='#'),
	token(b'\xbb\xd3', "$", "str", "Dollar sign",   alt="dollar",    key='$'),
	token(b'\xbb\xd4', "&", "str", "Ampersand",     alt="ampersand", key='&'),
	token(b'\xbb\xd5', "`", "str", "Grave/backtick",alt="backtick"),
	token(b'\xbb\xd6', ";", "str", "Semicolon",     alt="semicolon", key=';'),
	token(b'\xbb\xd7', "\\","str", "Backslash",     alt="backslash", key='\\'),
	token(b'\xbb\xd8', "|", "str", "Pipe/vertical bar", alt="pipe",  key='|'),
	token(b'\xbb\xd9', "_", "str", "Underscore",    alt="underscore",key='_'),
	token(b'\xbb\xda', "%", "str", "Percent sign",  alt="percent",   key='%'),
	token(b'\xbb\xdb', "…", "str", "Ellipsis",      alt="ellipsis"),
	token(b'\xbb\xdc', "∠", "str", "Angle symbol", alt="angle"),
	token(b'\xbb\xdd', "ß", "str", "German sharp S", alt="sharp-s"),
	token(b'\xbb\xde', "x", "str", "Superscript x", alt="superscript-x"),  # deprecated
	token(b'\xbb\xdf', "T", "str", "Subscript T", alt="subscript-t"),  # deprecated
	
	*[token(bytes([0xBB, 0xE0 + i]), chr(0x2080 + i), "string", "Subscript {i}", alt=f"subscript-{i}") for i in range(10)],
	token(b'\xbb\xea', "₁₀", "str", "Subscript 10", alt=f"subscript-10"),
	
	token(b'\xbb\xeb', "←", "str", "Left arrow",   alt="left-arrow"),
	token(b'\xbb\xec', "→", "str", "Right arrow", alt="right-arrow"),
	token(b'\xbb\xed', "↑", "str", "Up arrow",     alt="up-arrow"),
	token(b'\xbb\xee', "↓", "str", "Down arrow",   alt="down-arrow"),
	token(b'\xbb\xf0', "x", "str", "x"),  # deprecated
	token(b'\xbb\xf1', "∫", "str", "Integral symbol", alt="integral"),
	token(b'\xbb\xf2', "🡅", "str", "scroll up"),
	token(b'\xbb\xf3', "🡇", "str", "scroll down"),
	token(b'\xbb\xf4', "√", "str", "Square root symbol", alt="root"),  # deprecated
	token(b'\xbb\xf5', "<funcon>", "str", "Function On", alt='function-on'),  # deprecated

	# Two-byte: TI-84+ extended tokens 0xEF xx
	# 0xEF00–0xEF0F and 0xEF10–0xEF1E: original TI-84+ OS; 0xEF17–0xEF1E also on TI-84+ non-CE newer OS
	# 0xEF30–0xEF3D: TI-84 Plus (non-CE) newer OS only
	token(b'\xef\x00', "setDate(",      "cmdfunc",      "Sets the date on the clock of an OS-enabled calculator"),
	token(b'\xef\x01', "setTime(",      "cmdfunc",      "Sets the time on the clock of an OS-enabled calculator"),
	token(b'\xef\x02', "checkTmr(",     "func",      "Returns the elapsed time in seconds since startTmr was called"),
	token(b'\xef\x03', "setDtFmt(",     "cmdfunc",      "Sets the date display format (M/D/Y, D/M/Y, or Y/M/D)"),
	token(b'\xef\x04', "setTmFmt(",     "cmdfunc",      "Sets the time display format (12-hour or 24-hour)"),
	token(b'\xef\x05', "timeCnv(",      "func",    "Converts a number of seconds into a {days,hours,min,sec} list"),
	token(b'\xef\x06', "dayOfWk(",      "func",    "Returns the day of the week (1=Sun … 7=Sat) for a given date"),
	token(b'\xef\x07', "getDtStr(",     "func",      "Returns the current date as a string in the active format"),
	token(b'\xef\x08', "getTmStr(",     "func",      "Returns the current time as a string in the active format"),
	token(b'\xef\x09', "getDate",       "val",      "Returns the current date as a {year, month, day} list"),
	token(b'\xef\x0a', "getTime",       "val",      "Returns the current time as a {hour, minute, second} list"),
	token(b'\xef\x0b', "startTmr",      "val",      "Starts a timer and returns a reference value for checkTmr("),
	token(b'\xef\x0c', "getDtFmt",      "val",      "Returns the current date format setting as a number"),
	token(b'\xef\x0d', "getTmFmt",      "val",      "Returns the current time format setting as a number"),
	token(b'\xef\x0e', "isClockOn",     "val",      "Returns 1 if the clock is currently on, 0 if off"),
	token(b'\xef\x0f', "ClockOff",      "cmd",      "Turns off the clock on OS-enabled calculators"),
	token(b'\xef\x10', "ClockOn",       "cmd",      "Turns on the clock on OS-enabled calculators"),
	token(b'\xef\x11', "OpenLib(",      "cmdfunc",      "Opens an application library for use with ExecLib"),
	token(b'\xef\x12', "ExecLib",       "cmd",      "Executes a routine from a library opened with OpenLib("),
	token(b'\xef\x13', "invT(",         "func",    "Returns the inverse t-distribution value for a given area and degrees of freedom"),
	token(b'\xef\x14', "χ²GOF-Test(",   "func",    "Performs a chi-squared goodness-of-fit test",              alt="chi^2-GOF-Test"),
	token(b'\xef\x15', "LinRegTInt ",    "cmd",    "Computes a linear regression t confidence interval"),
	token(b'\xef\x16', "Manual-Fit ",    "cmd",    "Fits a line manually to a scatter plot by dragging"),
	token(b'\xef\x17', "ZQuadrant1",    "cmd", "Zoom preset: zooms to show only quadrant 1"),
	token(b'\xef\x18', "ZFrac1/2",      "cmd", "Zoom preset: sets window for fractions with denominator 2"),
	token(b'\xef\x19', "ZFrac1/3",      "cmd", "Zoom preset: sets window for fractions with denominator 3"),
	token(b'\xef\x1a', "ZFrac1/4",      "cmd", "Zoom preset: sets window for fractions with denominator 4"),
	token(b'\xef\x1b', "ZFrac1/5",      "cmd", "Zoom preset: sets window for fractions with denominator 5"),
	token(b'\xef\x1c', "ZFrac1/8",      "cmd", "Zoom preset: sets window for fractions with denominator 8"),
	token(b'\xef\x1d', "ZFrac1/10",     "cmd", "Zoom preset: sets window for fractions with denominator 10"),
	token(b'\xef\x1e', "<mathprintbox>",  "",      ""),
	token(b'\xef\x30', "►n/d◄►Un/d",   "converter","Converts between proper fraction and mixed number display"),
	token(b'\xef\x31', "►F◄►D",         "converter","Converts between fraction and decimal display"),
	token(b'\xef\x32', "remainder(",    "func",    "Returns the remainder of integer division"),
	token(b'\xef\x33', "Σ(",            "func",    "Summation: evaluates an expression over a range of integer values", alt='sigma('),
	token(b'\xef\x34', "logBASE(",      "func",    "Returns the logarithm of a value in a specified base"),
	token(b'\xef\x35', "randIntNoRep(", "func",    "Returns a list of non-repeating random integers in a range"),
	token(b'\xef\x36', "MATHPRINT",     "cmd", "Mode setting: enables MathPrint display mode"),
	token(b'\xef\x37', "CLASSIC",       "cmd", "Mode setting: enables Classic (linear) display mode"),
	token(b'\xef\x38', "n/d",           "cmd","Fraction template: enters a proper fraction"),
	token(b'\xef\x39', "Un/d",          "cmd","Fraction template: enters a mixed number"),
	token(b'\xef\x3a', "AUTO",          "cmd", "Mode setting: sets fraction display to AUTO"),
	token(b'\xef\x3b', "DEC",           "cmd", "Mode setting: sets fraction display to DEC (decimal)"),
	token(b'\xef\x3c', "FRAC",          "cmd", "Mode setting: sets fraction display to FRAC"),
	token(b'\xef\x3d', "FRAC-APPROX",   "cmd", "Mode setting: sets fraction display to FRAC-APPROX"),

]

if __name__ == '__main__':
	@dataclass
	class NullToken:
		code: bytes
	
	check = [None] * 0x100
	check_misc = [None] * 0xF6
	duplicate = set()
	
	for code in [
		b'\x00', b'\x26', b'\x5c', b'\x5d', b'\x5e', b'\x60', b'\x61', b'\x62', b'\x63', b'\x7e', b'\xaa', b'\xbb', b'\xef',
		b'\xbb\x5c', b'\xbb\x5d', b'\xbb\x5e', b'\xbb\x5f', b'\xbb\x60', b'\xbb\x61', b'\xbb\x62', b'\xbb\x63', b'\xbb\x7e', b'\xbb\xaa', b'\xbb\xbb', b'\xbb\xd0', b'\xbb\xef', 
	]:
		old_len = len(duplicate)
		duplicate.add(code)
		if old_len == len(duplicate):
			raise ValueError(f"Duplicate: {token}")
		if len(code) == 1:
			check[code[0]] = NullToken(code)
		elif code[0] == 0xBB:
			check_misc[code[1]] = NullToken(code)
	
	for token in TOKENS:
		old_len = len(duplicate)
		duplicate.add(token.code)
		if old_len == len(duplicate):
			raise ValueError(f"Duplicate: {token}")
		if len(token.code) == 1:
			check[token.code[0]] = token
		elif token.code[0] == 0xBB:
			check_misc[token.code[1]] = token
	
	for i, token in enumerate(check):
		if token is None:
			print('MISSING:', hex(i))
	
	for i, token in enumerate(check_misc):
		if token is None:
			print('MISSING:', hex(0xBB00 + i))

	for token in sorted(TOKENS, key=lambda t: t.code):
		print(token.code.hex(), token.display.decode('latin-1'))