from dataclasses import dataclass, field
from pyscript import document, when


# ── Token Data Class ───────────────────────────────────────────────────────────

@dataclass
class Token:
	text: str	# the actual TI-Basic token, e.g. "√("
	type: str	# category, e.g. "math"
	desc: str	# short tooltip / autocomplete description
	name: str = ""	# keyboard-friendly alias for non-typeable symbols only


def tok(text, type, desc, name=""):
	return Token(text=text, type=type, desc=desc, name=name)


# ── Token Database ─────────────────────────────────────────────────────────────

TOKENS: list[Token] = [
	# Control flow
	tok("If",     "control", "If conditional"),
	tok("Then",   "control", "Start If body"),
	tok("Else",   "control", "Else branch"),
	tok("End",    "control", "End block"),
	tok("For(",   "control", "For loop: For(var,start,end[,step])"),
	tok("While",  "control", "While condition is true"),
	tok("Repeat", "control", "Repeat until condition is true"),
	tok("Return", "control", "Return from sub-program"),
	tok("Stop",   "control", "Stop program"),
	tok("Goto",   "control", "Jump to label"),
	tok("Lbl",    "control", "Define a label"),
	tok("IS>(",   "control", "Increment variable; skip next if >"),
	tok("DS<(",   "control", "Decrement variable; skip next if <"),
	tok("Menu(",  "control", "Display a menu"),
	# I/O
	tok("Disp",      "io", "Display value on home screen"),
	tok("DispGraph", "io", "Show graph screen"),
	tok("DispTable", "io", "Show table screen"),
	tok("Input",     "io", "Get input from user"),
	tok("Prompt",    "io", "Prompt user for variable"),
	tok("Output(",   "io", "Display text at row, col"),
	tok("getKey",    "io", "Read last keypress code"),
	tok("ClrHome",   "io", "Clear the home screen"),
	tok("ClrDraw",   "io", "Clear the drawing screen"),
	tok("Pause",     "io", "Pause execution"),
	# Math
	tok("abs(",      "math", "Absolute value of x"),
	tok("round(",    "math", "Round x to N decimal places"),
	tok("iPart(",    "math", "Integer part of x (truncate)"),
	tok("fPart(",    "math", "Fractional part of x"),
	tok("int(",      "math", "Greatest integer ≤ x"),
	tok("min(",      "math", "Minimum of two values or a list"),
	tok("max(",      "math", "Maximum of two values or a list"),
	tok("lcm(",      "math", "Least common multiple"),
	tok("gcd(",      "math", "Greatest common divisor"),
	tok("log(",      "math", "Base-10 logarithm"),
	tok("ln(",       "math", "Natural logarithm (base e)"),
	tok("e^(",       "math", "e raised to a power"),
	tok("10^(",      "math", "10 raised to a power"),
	tok("√(",        "math", "Square root of x",              name="sqrt("),
	tok("³√(",       "math", "Cube root of x",                name="cbrt("),
	tok("rand",      "math", "Random number in [0, 1)"),
	tok("randInt(",  "math", "Random integer between A and B"),
	tok("randNorm(", "math", "Random value from normal distribution"),
	tok("π",         "math", "Pi (3.14159…)",                 name="pi"),
	# Trig
	tok("sin(",    "trig", "Sine of x"),
	tok("cos(",    "trig", "Cosine of x"),
	tok("tan(",    "trig", "Tangent of x"),
	tok("sin⁻¹(", "trig", "Inverse sine (arcsine)",          name="arcsin("),
	tok("cos⁻¹(", "trig", "Inverse cosine (arccosine)",      name="arccos("),
	tok("tan⁻¹(", "trig", "Inverse tangent (arctangent)",    name="arctan("),
	tok("sinh(",   "trig", "Hyperbolic sine"),
	tok("cosh(",   "trig", "Hyperbolic cosine"),
	tok("tanh(",   "trig", "Hyperbolic tangent"),
	# Operators
	tok("+",  "operator", "Addition"),
	tok("-",  "operator", "Subtraction"),
	tok("*",  "operator", "Multiplication"),
	tok("/",  "operator", "Division"),
	tok("^",  "operator", "Exponentiation"),
	tok("²",  "operator", "Square (raised to power 2)",      name="^2"),
	tok("=",  "operator", "Equality test"),
	tok("≠",  "operator", "Inequality test",                 name="!="),
	tok("<",  "operator", "Less than"),
	tok(">",  "operator", "Greater than"),
	tok("≤",  "operator", "Less than or equal to",           name="<="),
	tok("≥",  "operator", "Greater than or equal to",        name=">="),
	tok("and","operator", "Logical AND"),
	tok("or", "operator", "Logical OR"),
	tok("xor","operator", "Logical XOR"),
	tok("not(","operator","Logical NOT"),
	tok("→",  "operator", "Store value to variable",         name="->"),
	tok(")",  "operator", "Close parenthesis"),
	tok(",",  "operator", "Argument separator"),
	tok(":",  "operator", "Statement separator"),
	# Variables A–Z
	*[tok(chr(c), "variable", f"Variable {chr(c)}")
	  for c in range(ord("A"), ord("Z") + 1)],
	tok("θ",  "variable", "Variable theta",                  name="theta"),
	tok("Ans","variable", "Last computed answer"),
	# List variables L₁–L₆  (subscript digits not typeable)
	*[tok(f"L{chr(0x2080 + i)}", "list", f"List variable {i}", name=f"L{i}")
	  for i in range(1, 7)],
	# List commands
	tok("dim(",    "list", "List or matrix dimension"),
	tok("Fill(",   "list", "Fill list or matrix with value"),
	tok("seq(",    "list", "Generate a sequence"),
	tok("sum(",    "list", "Sum of list elements"),
	tok("prod(",   "list", "Product of list elements"),
	tok("cumSum(", "list", "Cumulative sum of list"),
	tok("SortA(",  "list", "Sort list in ascending order"),
	tok("SortD(",  "list", "Sort list in descending order"),
	# Matrix variables [A]–[J]
	*[tok(f"[{chr(ord('A') + i)}]", "matrix", f"Matrix variable {chr(ord('A') + i)}")
	  for i in range(10)],
	# Matrix commands
	tok("det(",      "matrix", "Matrix determinant"),
	tok("identity(", "matrix", "Identity matrix of size N"),
	tok("randM(",    "matrix", "Random matrix"),
	tok("augment(",  "matrix", "Augment two matrices or lists"),
	# String commands
	tok("length(",   "string", "Length of a string"),
	tok("sub(",      "string", "Extract substring"),
	tok("inString(", "string", "Find position of substring"),
	# String variables Str0–Str9
	*[tok(f"Str{i}", "string", f"String variable {i}")
	  for i in range(10)],
]


# ── App State ──────────────────────────────────────────────────────────────────

lines: list[list[Token]] = [[]]
current_line: int = 0
selected_idx: int = 0
current_matches: list[Token] = []
active_category: str = ""
free_mode: bool = False


# ── Token Filtering ────────────────────────────────────────────────────────────

def filter_tokens(query: str) -> list[Token]:
	q = query.lower()
	results = []
	for token in TOKENS:
		if active_category and token.type != active_category:
			continue
		if token.text.lower().startswith(q):
			results.append(token)
		elif token.name and token.name.lower().startswith(q):
			results.append(token)
	results.sort(key=lambda t: t.type != "variable")
	return results[:18]


def make_number_token(text: str) -> Token:
	return Token(text=text, type="number", desc="Number literal")


def make_string_token(text: str) -> Token:
	return Token(text=text, type="string-lit", desc="String literal")


def make_literal_token(ch: str) -> Token:
	return Token(text=ch, type="literal", desc=f"Character")


def compute_matches(query: str) -> list[Token]:
	matches = filter_tokens(query)
	if query:
		try:
			float(query)
			matches = [make_number_token(query)] + matches
		except ValueError:
			pass
		if query.startswith('"'):
			matches = [make_string_token(query)] + matches
	return matches


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_editor():
	editor = document.getElementById("editor")
	editor.innerHTML = ""
	for li, line in enumerate(lines):
		row = document.createElement("div")
		row.className = "line" + (" line-active" if li == current_line else "")
		row.dataset.lineIdx = str(li)

		num = document.createElement("span")
		num.className = "line-number"
		num.textContent = str(li + 1)
		row.appendChild(num)

		tokens_wrap = document.createElement("div")
		tokens_wrap.className = "tokens"
		for ti, token in enumerate(line):
			chip = document.createElement("span")
			chip.className = f"token token-{token.type}"
			chip.textContent = token.text
			chip.title = f"{token.desc} (click to delete)"
			chip.dataset.lineIdx = str(li)
			chip.dataset.tokIdx = str(ti)
			tokens_wrap.appendChild(chip)

		row.appendChild(tokens_wrap)
		editor.appendChild(row)

	panel = document.getElementById("editor-panel")
	panel.scrollTop = panel.scrollHeight


def render_autocomplete():
	dropdown = document.getElementById("autocomplete")
	dropdown.innerHTML = ""
	if not current_matches:
		dropdown.classList.add("hidden")
		return
	dropdown.classList.remove("hidden")
	for i, token in enumerate(current_matches):
		item = document.createElement("div")
		item.className = f"ac-item token-{token.type}" + (" ac-selected" if i == selected_idx else "")
		item.dataset.acIdx = str(i)

		badge = document.createElement("span")
		badge.className = "ac-badge"
		badge.textContent = token.text
		item.appendChild(badge)

		if token.name:
			label = document.createElement("span")
			label.className = "ac-name"
			label.textContent = token.name
			item.appendChild(label)

		dsc = document.createElement("span")
		dsc.className = "ac-desc"
		dsc.textContent = token.desc
		item.appendChild(dsc)

		dropdown.appendChild(item)


# ── State Mutations ────────────────────────────────────────────────────────────

def commit_token(token: Token):
	global selected_idx, current_matches
	lines[current_line].append(token)
	current_matches = []
	selected_idx = 0
	document.getElementById("token-input").value = ""
	render_editor()
	render_autocomplete()


def new_line():
	global current_line
	lines.append([])
	current_line = len(lines) - 1
	render_editor()


def delete_last_token():
	global current_line
	if lines[current_line]:
		lines[current_line].pop()
		render_editor()
	elif len(lines) > 1:
		lines.pop(current_line)
		current_line = max(0, current_line - 1)
		render_editor()


# ── Free-type Mode Helper ──────────────────────────────────────────────────────

def _free_type_key(event) -> None:
	"""Handle a keypress while in free-type mode (Alt not held)."""
	global current_matches, selected_idx

	key = event.key

	if key in ("ArrowDown", "ArrowUp"):
		event.preventDefault()
		if current_matches:
			selected_idx += 1 if key == "ArrowDown" else -1
			selected_idx = max(0, min(selected_idx, len(current_matches) - 1))
			render_autocomplete()
		return

	if key == "Escape":
		event.preventDefault()
		current_matches = []
		selected_idx = 0
		render_autocomplete()
		return

	if key == "Enter":
		event.preventDefault()
		if current_matches:
			commit_token(current_matches[selected_idx])
		else:
			new_line()
		return

	if key == "Backspace":
		event.preventDefault()
		inp = document.getElementById("token-input")
		if inp.value:
			new_val = inp.value[:-1]
			inp.value = new_val
			current_matches = compute_matches(new_val) if new_val else []
			selected_idx = 0
			render_autocomplete()
		else:
			delete_last_token()
		return

	if len(key) != 1:
		return

	event.preventDefault()

	if current_matches:
		inp = document.getElementById("token-input")
		new_val = inp.value + key
		inp.value = new_val
		current_matches = compute_matches(new_val)
		selected_idx = 0
		render_autocomplete()
		return

	exact = next((t for t in TOKENS if t.text == key or (t.name and t.name == key)), None)
	commit_token(exact if exact else make_literal_token(key))


# ── Event Handlers ─────────────────────────────────────────────────────────────

@when("input", "#token-input")
def on_input(event):
	global current_matches, selected_idx
	if free_mode:
		return  # input value is managed by on_keydown in free mode
	query = event.target.value
	current_matches = compute_matches(query)
	selected_idx = 0
	render_autocomplete()


@when("keydown", "#token-input")
def on_keydown(event):
	global selected_idx, current_matches

	key = event.key

	if key == "Tab":
		event.preventDefault()
		on_mode_toggle(None)
		return

	if free_mode:
		if event.altKey and len(key) == 1:
			# Alt+key: open autocomplete for that character
			event.preventDefault()
			inp = document.getElementById("token-input")
			inp.value = key
			current_matches = compute_matches(key)
			selected_idx = 0
			render_autocomplete()
		else:
			_free_type_key(event)
		return

	# Token mode
	if key == "Escape":
		current_matches = []
		selected_idx = 0
		render_autocomplete()

	elif key == "ArrowDown":
		event.preventDefault()
		if current_matches:
			selected_idx = min(selected_idx + 1, len(current_matches) - 1)
			render_autocomplete()

	elif key == "ArrowUp":
		event.preventDefault()
		if current_matches:
			selected_idx = max(selected_idx - 1, 0)
			render_autocomplete()

	elif key == "Enter":
		event.preventDefault()
		query = document.getElementById("token-input").value.strip()
		if current_matches:
			commit_token(current_matches[selected_idx])
		elif query:
			try:
				float(query)
				commit_token(make_number_token(query))
			except ValueError:
				pass
		else:
			new_line()

	elif key == "Backspace":
		inp = document.getElementById("token-input")
		if inp.value == "":
			delete_last_token()


@when("click", "#editor")
def on_editor_click(event):
	global current_line
	target = event.target

	if target.classList.contains("token"):
		li = int(target.dataset.lineIdx)
		ti = int(target.dataset.tokIdx)
		lines[li].pop(ti)
		if not lines[li] and len(lines) > 1:
			lines.pop(li)
			if current_line >= len(lines):
				current_line = len(lines) - 1
		render_editor()
		return

	el = target
	while el and el.id != "editor":
		if el.dataset and el.dataset.lineIdx is not None and el.dataset.lineIdx != "undefined":
			try:
				current_line = int(el.dataset.lineIdx)
				render_editor()
				document.getElementById("token-input").focus()
			except (ValueError, TypeError):
				pass
			return
		el = el.parentElement


@when("click", "#autocomplete")
def on_ac_click(event):
	global selected_idx
	el = event.target
	while el and not el.classList.contains("ac-item"):
		el = el.parentElement
	if el and el.dataset.acIdx is not None:
		idx = int(el.dataset.acIdx)
		selected_idx = idx
		commit_token(current_matches[idx])
		document.getElementById("token-input").focus()


@when("click", "#toolbar")
def on_toolbar_click(event):
	global active_category, current_matches, selected_idx
	btn = event.target
	if not btn.classList.contains("cat-btn"):
		return
	active_category = btn.dataset.cat
	for b in document.querySelectorAll(".cat-btn"):
		b.classList.remove("active")
	btn.classList.add("active")
	query = document.getElementById("token-input").value
	current_matches = compute_matches(query)
	selected_idx = 0
	render_autocomplete()
	document.getElementById("token-input").focus()


@when("click", "#mode-toggle")
def on_mode_toggle(event):
	global free_mode, current_matches, selected_idx
	free_mode = not free_mode
	current_matches = []
	selected_idx = 0

	btn = document.getElementById("mode-toggle")
	inp = document.getElementById("token-input")
	hints_token = document.getElementById("hints-token")
	hints_free = document.getElementById("hints-free")

	if free_mode:
		btn.textContent = "FREE"
		btn.classList.add("free-active")
		inp.placeholder = "type token text directly…"
		hints_token.classList.add("hidden")
		hints_free.classList.remove("hidden")
	else:
		btn.textContent = "TOKEN"
		btn.classList.remove("free-active")
		inp.placeholder = "type a token name…"
		hints_token.classList.remove("hidden")
		hints_free.classList.add("hidden")

	inp.value = ""
	render_autocomplete()
	inp.focus()


@when("click", "#btn-clear-line")
def on_clear_line(event):
	lines[current_line].clear()
	render_editor()


@when("click", "#btn-clear-all")
def on_clear_all(event):
	global lines, current_line, current_matches, selected_idx
	lines = [[]]
	current_line = 0
	current_matches = []
	selected_idx = 0
	document.getElementById("token-input").value = ""
	render_editor()
	render_autocomplete()


# ── Boot ──────────────────────────────────────────────────────────────────────

document.getElementById("loading-overlay").classList.add("hidden")
document.getElementById("app").classList.remove("hidden")
render_editor()
document.getElementById("token-input").focus()
