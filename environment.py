class Environment:

	def call_seq(self, parser) -> list:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var_tok = parser.advance()
		if not var_tok.is_real_var():
			raise ValueError("seq: second arg must be a variable")
		var = var_tok.text
		parser.expect(COMMA)
		start = parser.parse_expr()
		parser.expect(COMMA)
		end = parser.parse_expr()
		step = 1.0
		if parser.eat_if(COMMA):
			step = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		result = []
		n = start
		while (step > 0 and n <= end + 1e-10) or (step < 0 and n >= end - 1e-10):
			self.reals[var] = n
			result.append(thunk.eval())
			n += step
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return result

	def call_sigma(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var_tok = parser.advance()
		if not var_tok.is_real_var():
			raise ValueError("Σ: second arg must be a variable")
		var = var_tok.text
		parser.expect(COMMA)
		start = int(parser.parse_expr())
		parser.expect(COMMA)
		end = int(parser.parse_expr())
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		total = 0.0
		for i in range(start, end + 1):
			self.reals[var] = float(i)
			total += thunk.eval()
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return total

	def call_nderiv(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var = parser.advance().text
		parser.expect(COMMA)
		val = parser.parse_expr()
		h = 1e-5
		if parser.eat_if(COMMA):
			h = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		self.reals[var] = val + h
		fwd = thunk.eval()
		self.reals[var] = val - h
		bwd = thunk.eval()
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return (fwd - bwd) / (2 * h)

	def call_fnint(self, parser) -> float:
		from tokens import COMMA, R_PAREN
		thunk = parser.capture()
		parser.expect(COMMA)
		var = parser.advance().text
		parser.expect(COMMA)
		a = parser.parse_expr()
		parser.expect(COMMA)
		b = parser.parse_expr()
		parser.eat_if(R_PAREN)

		saved = self.reals.get(var)
		n = 1000
		h = (b - a) / n
		def f(x):
			self.reals[var] = x
			return thunk.eval()
		total = f(a) + f(b)
		for i in range(1, n):
			total += (4 if i % 2 else 2) * f(a + i * h)
		if saved is not None:
			self.reals[var] = saved
		else:
			self.reals.pop(var, None)
		return total * h / 3
