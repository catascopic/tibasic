from pyscript import document, when
from tokens import Token, TOKENS

KEY_MAP: dict[str, Token] = {t.key: t for t in TOKENS if t.key is not None}

# ── App State ──────────────────────────────────────────────────────────────────

lines: list[list[Token]] = [[]]
current_line: int = 0
selected_idx: int = 0
current_matches: list[Token] = []
free_mode: bool = False


# ── Token Filtering ────────────────────────────────────────────────────────────

def compute_matches(query: str) -> list[Token]:
	query_norm = query.lower()
	results = []
	for token in TOKENS:
		if any(a.startswith(query_norm) for a in token.alias):
			results.append(token)
	results.sort(key=lambda t: t.category != "variable")
	return results[:18]


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
			chip.className = f"token token-{token.category}"
			chip.textContent = token.display.decode('latin-1')
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
		item.className = f"ac-item token-{token.category}" + (" ac-selected" if i == selected_idx else "")
		item.dataset.acIdx = str(i)

		badge = document.createElement("span")
		badge.className = "ac-badge"
		badge.textContent = token.text
		item.appendChild(badge)

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

	if key in KEY_MAP:
		commit_token(KEY_MAP[key])


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
		if current_matches:
			commit_token(current_matches[selected_idx])
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
