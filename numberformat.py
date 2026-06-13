from decimal import Decimal, ROUND_HALF_UP

_SIG = 10  # the TI-83+ displays 10 significant figures in Normal mode


def ti83_format(x) -> str:
	"""Format a real number the way a TI-83+ shows it in Normal/Float mode.

	10 significant figures, ties rounded away from zero.  Scientific notation
	when |x| ≥ 1e10 or 0 < |x| < 1e-3.  Positive exponents carry no '+' sign,
	and a pure fraction drops its leading zero (.5, not 0.5).
	"""
	d = Decimal(str(x))
	if d == 0:
		return "0"

	# Round to 10 significant figures.  adjusted() is the power of ten of the
	# leading digit, so exp-(_SIG-1) is the place value of the last kept digit.
	exp = d.adjusted()
	d = d.quantize(Decimal(1).scaleb(exp - (_SIG - 1)), rounding=ROUND_HALF_UP)
	exp = d.adjusted()  # re-read: rounding may have bumped it (9.999… → 10)

	if exp >= 10 or exp < -3:
		return _plain(d.scaleb(-exp)) + f"e{exp}"  # shift mantissa into [1, 10)
	return _plain(d)


def _plain(d: Decimal) -> str:
	"""Fixed-point string with trailing zeros and a pure-fraction leading zero removed."""
	s = format(d, 'f')
	if '.' in s:
		s = s.rstrip('0').rstrip('.')
	if s.startswith('0.'):
		return s[1:]
	if s.startswith('-0.'):
		return '-' + s[2:]
	return s
