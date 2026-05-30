from errors import TiSyntaxError


def _exec_if(parser):
	return IfResult(bool(parser.parse_expr()))

def _exec_while(parser):
	return WhileResult(bool(parser.parse_expr()))

def _exec_repeat(parser):
	return RepeatResult()

def _exec_for(parser):
	var_tok = parser.advance()
	if not var_tok.is_numeric_var():
		raise TiSyntaxError("For: first arg must be a variable")
	parser.expect(COMMA)
	start = parser.parse_expr()
	parser.expect(COMMA)
	end   = parser.parse_expr()
	step  = 1.0
	if parser.eat_if(COMMA):
		step = parser.parse_expr()
	parser.eat_if(R_PAREN)
	return ForResult(var_tok.text, float(start), float(end), float(step))

def _exec_lbl(parser):
	return LblResult(parser.parse_label_name())

def _exec_goto(parser):
	return GotoResult(parser.parse_label_name())

def _exec_pause(parser):
	val = parser.parse_expr() if not parser.at_end() else None
	return PauseResult(val)

def _exec_disp(parser):
	values = []
	if not parser.at_end():
		values.append(parser.parse_expr())
		while parser.eat_if(COMMA):
			values.append(parser.parse_expr())
	return DispResult(values)

def _exec_input(parser):
	if parser.peek() is QUOTE:
		parser.advance()
		prompt = parser.parse_string_literal()
		parser.expect(COMMA)
	else:
		prompt = None
	target = parser.parse_store_target()
	return InputResult(prompt, target)

def _exec_prompt(parser):
	targets = [parser.parse_store_target()]
	while parser.eat_if(COMMA):
		targets.append(parser.parse_store_target())
	return PromptResult(targets)

def _exec_output(parser):
	row = parser.parse_expr()
	parser.expect(COMMA)
	col = parser.parse_expr()
	parser.expect(COMMA)
	val = parser.parse_expr()
	parser.eat_if(R_PAREN)
	return OutputResult(row, col, val)

def _exec_menu(parser):
	title = parser.parse_expr()
	options = []
	while parser.eat_if(COMMA):
		name  = parser.parse_expr()
		parser.expect(COMMA)
		label = parser.parse_label_name()
		options.append((name, label))
	parser.eat_if(R_PAREN)
	return MenuResult(title, options)

def _exec_is_gt(parser):
	var_tok = parser.advance()
	parser.expect(COMMA)
	limit = parser.parse_expr()
	parser.eat_if(R_PAREN)
	name = var_tok.text
	parser.env[name] = parser.env.get(name, 0.0) + 1
	return IfResult(parser.env[name] > limit)

def _exec_ds_lt(parser):
	var_tok = parser.advance()
	parser.expect(COMMA)
	limit = parser.parse_expr()
	parser.eat_if(R_PAREN)
	name = var_tok.text
	parser.env[name] = parser.env.get(name, 0.0) - 1
	return IfResult(parser.env[name] < limit)
