from pyscript import document, when

# ── Token Database ─────────────────────────────────────────────────────────────
# Each entry: (display_text, type, description)

TOKENS = [
    # Control flow
    ("If",       "control", "If conditional"),
    ("Then",     "control", "Start If body"),
    ("Else",     "control", "Else branch"),
    ("End",      "control", "End block"),
    ("For(",     "control", "For loop"),
    ("While",    "control", "While loop"),
    ("Repeat",   "control", "Repeat until"),
    ("Return",   "control", "Return from sub"),
    ("Stop",     "control", "Stop program"),
    ("Goto",     "control", "Go to label"),
    ("Lbl",      "control", "Define label"),
    ("IS>(",     "control", "Increment and skip if >"),
    ("DS<(",     "control", "Decrement and skip if <"),
    ("Menu(",    "control", "Display menu"),
    # I/O
    ("Disp",      "io", "Display value"),
    ("DispGraph", "io", "Display graph screen"),
    ("DispTable", "io", "Display table screen"),
    ("Input",     "io", "Get input from user"),
    ("Prompt",    "io", "Prompt for variable"),
    ("Output(",   "io", "Display at row, col"),
    ("getKey",    "io", "Read last keypress"),
    ("ClrHome",   "io", "Clear home screen"),
    ("ClrDraw",   "io", "Clear drawing screen"),
    ("Pause",     "io", "Pause execution"),
    # Math
    ("abs(",     "math", "Absolute value"),
    ("round(",   "math", "Round to N decimals"),
    ("iPart(",   "math", "Integer part"),
    ("fPart(",   "math", "Fractional part"),
    ("int(",     "math", "Greatest integer ≤ x"),
    ("min(",     "math", "Minimum of values"),
    ("max(",     "math", "Maximum of values"),
    ("lcm(",     "math", "Least common multiple"),
    ("gcd(",     "math", "Greatest common divisor"),
    ("log(",     "math", "Base-10 logarithm"),
    ("ln(",      "math", "Natural logarithm"),
    ("e^(",      "math", "e raised to power"),
    ("10^(",     "math", "10 raised to power"),
    ("√(",       "math", "Square root"),
    ("³√(",      "math", "Cube root"),
    ("rand",     "math", "Random number [0,1)"),
    ("randInt(", "math", "Random integer in range"),
    ("randNorm(","math", "Random normal distribution"),
    ("π",        "math", "Pi (3.14159…)"),
    # Trig
    ("sin(",    "trig", "Sine"),
    ("cos(",    "trig", "Cosine"),
    ("tan(",    "trig", "Tangent"),
    ("sin⁻¹(", "trig", "Arcsine"),
    ("cos⁻¹(", "trig", "Arccosine"),
    ("tan⁻¹(", "trig", "Arctangent"),
    ("sinh(",   "trig", "Hyperbolic sine"),
    ("cosh(",   "trig", "Hyperbolic cosine"),
    ("tanh(",   "trig", "Hyperbolic tangent"),
    # Operators
    ("+",  "operator", "Addition"),
    ("-",  "operator", "Subtraction"),
    ("*",  "operator", "Multiplication"),
    ("/",  "operator", "Division"),
    ("^",  "operator", "Exponentiation"),
    ("²",  "operator", "Squared"),
    ("=",  "operator", "Equals"),
    ("≠",  "operator", "Not equals"),
    ("<",  "operator", "Less than"),
    (">",  "operator", "Greater than"),
    ("≤",  "operator", "Less than or equal"),
    ("≥",  "operator", "Greater than or equal"),
    ("and","operator", "Logical AND"),
    ("or", "operator", "Logical OR"),
    ("xor","operator", "Logical XOR"),
    ("not(","operator","Logical NOT"),
    ("→",  "operator", "Store to variable"),
    (")",  "operator", "Close parenthesis"),
    (",",  "operator", "Argument separator"),
    (":",  "operator", "Statement separator"),
    # Variables A–Z + θ + Ans
    *[(chr(c), "variable", f"Variable {chr(c)}") for c in range(ord("A"), ord("Z") + 1)],
    ("θ",  "variable", "Variable theta"),
    ("Ans","variable", "Last answer"),
    # Lists
    ("L₁", "list", "List 1"),
    ("L₂", "list", "List 2"),
    ("L₃", "list", "List 3"),
    ("L₄", "list", "List 4"),
    ("L₅", "list", "List 5"),
    ("L₆", "list", "List 6"),
    # List commands
    ("dim(",    "list", "List / matrix dimension"),
    ("Fill(",   "list", "Fill with value"),
    ("seq(",    "list", "Generate sequence"),
    ("sum(",    "list", "Sum of list"),
    ("prod(",   "list", "Product of list"),
    ("cumSum(", "list", "Cumulative sum"),
    ("SortA(",  "list", "Sort ascending"),
    ("SortD(",  "list", "Sort descending"),
    # Matrices
    ("[A]",       "matrix", "Matrix A"),
    ("[B]",       "matrix", "Matrix B"),
    ("[C]",       "matrix", "Matrix C"),
    ("[D]",       "matrix", "Matrix D"),
    ("det(",      "matrix", "Determinant"),
    ("identity(", "matrix", "Identity matrix"),
    ("randM(",    "matrix", "Random matrix"),
    ("augment(",  "matrix", "Augment matrix / list"),
    # Strings (commands)
    ("length(",   "string", "String length"),
    ("sub(",      "string", "Substring"),
    ("inString(", "string", "Find in string"),
    # String variables
    ("Str0", "string", "String variable 0"),
    ("Str1", "string", "String variable 1"),
    ("Str2", "string", "String variable 2"),
    ("Str3", "string", "String variable 3"),
    ("Str4", "string", "String variable 4"),
    ("Str5", "string", "String variable 5"),
]

# ── App State ──────────────────────────────────────────────────────────────────

lines: list[list[tuple]] = [[]]   # list of lines; each line = list of (text, type, desc)
current_line: int = 0
selected_idx: int = 0
current_matches: list[tuple] = []
active_category: str = ""          # "" = all categories

# ── Token Filtering ────────────────────────────────────────────────────────────

def filter_tokens(query: str) -> list[tuple]:
    q = query.lower()
    results = []
    for tok in TOKENS:
        text, typ, desc = tok
        if active_category and typ != active_category:
            continue
        if text.lower().startswith(q):
            results.append(tok)
    return results[:18]


def compute_matches(query: str) -> list[tuple]:
    matches = filter_tokens(query)
    if query:
        # Number literal option
        try:
            float(query)
            matches = [(query, "number", "Number literal")] + matches
        except ValueError:
            pass
        # String literal option
        if query.startswith('"'):
            matches = [(query, "string-lit", "String literal")] + matches
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
        for ti, (text, typ, desc) in enumerate(line):
            chip = document.createElement("span")
            chip.className = f"token token-{typ}"
            chip.textContent = text
            chip.title = f"{desc} (click to delete)"
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
    for i, (text, typ, desc) in enumerate(current_matches):
        item = document.createElement("div")
        item.className = f"ac-item token-{typ}" + (" ac-selected" if i == selected_idx else "")
        item.dataset.acIdx = str(i)

        badge = document.createElement("span")
        badge.className = "ac-badge"
        badge.textContent = text

        dsc = document.createElement("span")
        dsc.className = "ac-desc"
        dsc.textContent = desc

        item.appendChild(badge)
        item.appendChild(dsc)
        dropdown.appendChild(item)

# ── State Mutations ────────────────────────────────────────────────────────────

def commit_token(tok: tuple):
    global selected_idx, current_matches
    lines[current_line].append(tok)
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

# ── Event Handlers ─────────────────────────────────────────────────────────────

@when("input", "#token-input")
def on_input(event):
    global current_matches, selected_idx
    query = event.target.value
    current_matches = compute_matches(query)
    selected_idx = 0
    render_autocomplete()


@when("keydown", "#token-input")
def on_keydown(event):
    global selected_idx, current_matches

    key = event.key

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
            # Bare number fallback
            try:
                float(query)
                commit_token((query, "number", "Number literal"))
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

    # Click anywhere on a line row to make it active
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
    target = event.target
    # Walk up to ac-item
    el = target
    while el and not el.classList.contains("ac-item"):
        el = el.parentElement
    if el and el.dataset.acIdx is not None:
        idx = int(el.dataset.acIdx)
        global selected_idx
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
    # Update active style
    for b in document.querySelectorAll(".cat-btn"):
        b.classList.remove("active")
    btn.classList.add("active")
    # Re-filter with current query
    query = document.getElementById("token-input").value
    current_matches = compute_matches(query)
    selected_idx = 0
    render_autocomplete()
    document.getElementById("token-input").focus()


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
