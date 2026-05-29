def to_dms(x):
	return DMS(require_real(x))

def to_dec(x):
	return require_real(x)

def to_frac(x):
	x = require_real(x)
	f = Fraction(x).limit_denominator(10000)
	return float(f) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
