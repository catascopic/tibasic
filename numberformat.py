from decimal import Decimal, ROUND_HALF_UP

# I'm trying to write a function that matches the TI-83+ calculator's number format. It should return a string. The rules are: 10 decimal places, round .5 up, scientific notation for numbers outside of (1e-10, 1e10) and inside of (-1e-3, 1e-3). No plus sign for positive scientific exponents

_EPISLON = Decimal('1e-10')
_POS_E_LIMIT = Decimal('1e10')
_NEG_E_LIMIT = Decimal('1e-3')

def ti83_format(x):
    d = Decimal(str(x)).quantize(_EPISLON, rounding=ROUND_HALF_UP)    
    if d == 0:
        return "0"
    
    abs_d = abs(d)
    if abs_d >= _POS_E_LIMIT or abs_d < _NEG_E_LIMIT:
        # Find exponent by converting to string in scientific notation
        exp = int(abs_d.log10().to_integral_value(rounding=ROUND_HALF_UP))
        # Adjust if log10 rounded up past a power boundary
        if Decimal(10) ** exp > abs_d:
            exp -= 1
        mantissa = (d / Decimal(10) ** exp).quantize(_EPISLON, rounding=ROUND_HALF_UP)
        mantissa_str = strip_zeros(mantissa)
        return f"{mantissa_str}e{exp}"
    else:
        return strip_zeros(d)

def strip_zeros(d):
    s = str(d)
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s