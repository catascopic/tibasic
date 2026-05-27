"""
dispatch.py — maps Token objects to their callable implementations.

Five tables:
  CALLABLES  : Token → (ArgParser) → value | None   (functions + commands)
  VARIABLES  : Token → Variable                      (storable + computed)
  OPERATORS  : Token → (lhs, rhs) → value            (binary operators)
  POSTFIXES  : Token → (x) → value                   (postfix operators)
  CONVERTERS : Token → (x) → value                   (►DMS, ►Dec, ►Frac)
"""

import math
import operator as op
import random

import purefunctions as pf
import forms
from environment import (
	Environment, Variable, ComputedVar,
	NumericVar, ListVar, MatrixVar, StringVar, StatVar, WindowVar,
)
from forms import ArgParser
from tokens import (
	TOKENS, TOKENS_BY_TEXT,
	ANS, RAND,
	OR, XOR, AND, EQ, LT, GT, LE, GE, NE, ADD, SUB, MUL, DIV,
	NPR, NCR, POW, XTH_ROOT,
	INV, SQ, TRANSPOSE, CUBE, FACT,
)


# ── Dispatch tables ────────────────────────────────────────────────────────────

CALLABLES:  dict = {}   # Token → (ArgParser) → value | None
VARIABLES:  dict = {}   # Token → Variable
OPERATORS:  dict = {}   # Token → (lhs, rhs) → value
POSTFIXES:  dict = {}   # Token → (x) → value
CONVERTERS: dict = {}   # Token → (x) → value


# ── Helpers ────────────────────────────────────────────────────────────────────

def _t(text: str):
	"""Look up a token by its display text."""
	return TOKENS_BY_TEXT[text]


def _pure(f):
	"""Wrap a pure (*args) → value function as an ArgParser callable."""
	def wrapper(a: ArgParser):
		return f(*a.parse_args())
	return wrapper


# ── Storable variables (auto-registered from token codes) ─────────────────────

for _tok in TOKENS:
	_b0 = _tok.code[0]
	if 0x41 <= _b0 <= 0x5b:                    # A–Z, θ
		VARIABLES[_tok] = NumericVar(_b0 - 0x41)
	elif _b0 == 0x5d:                           # L1–L6
		VARIABLES[_tok] = ListVar(_tok.code[1])
	elif _b0 == 0x5c:                           # [A]–[J]
		VARIABLES[_tok] = MatrixVar(_tok.code[1])
	elif _b0 == 0xaa:                           # Str0–9
		VARIABLES[_tok] = StringVar(_tok.code[1])
	elif _b0 == 0x62:                           # stat vars
		VARIABLES[_tok] = StatVar(_tok.code[1])
	elif _b0 == 0x63:                           # window vars
		VARIABLES[_tok] = WindowVar(_tok.code[1])


# ── Computed (nullary) variables ───────────────────────────────────────────────

VARIABLES[ANS]          = ComputedVar(lambda env: env.ans)
VARIABLES[RAND]         = ComputedVar(lambda env: random.random())
VARIABLES[_t('π')]      = ComputedVar(lambda env: math.pi)
VARIABLES[_t('𝑒')]      = ComputedVar(lambda env: math.e)
VARIABLES[_t('𝑖')]      = ComputedVar(lambda env: 1j)
VARIABLES[_t('getKey')] = ComputedVar(lambda env: env.key_code)
VARIABLES[_t('getDate')]  = ComputedVar(lambda env: env.get_date())
VARIABLES[_t('getTime')]  = ComputedVar(lambda env: env.get_time())
VARIABLES[_t('startTmr')] = ComputedVar(lambda env: env.start_tmr())
VARIABLES[_t('getDtFmt')] = ComputedVar(lambda env: env.get_dt_fmt())
VARIABLES[_t('getTmFmt')] = ComputedVar(lambda env: env.get_tm_fmt())
VARIABLES[_t('isClockOn')] = ComputedVar(lambda env: env.is_clock_on())


# ── Binary operators ───────────────────────────────────────────────────────────

OPERATORS[OR]       = pf.or_
OPERATORS[XOR]      = pf.xor
OPERATORS[AND]      = pf.and_
OPERATORS[EQ]       = op.eq
OPERATORS[LT]       = op.lt
OPERATORS[GT]       = op.gt
OPERATORS[LE]       = op.le
OPERATORS[GE]       = op.ge
OPERATORS[NE]       = op.ne
OPERATORS[ADD]      = op.add
OPERATORS[SUB]      = op.sub
OPERATORS[MUL]      = op.mul
OPERATORS[DIV]      = op.truediv
OPERATORS[NPR]      = pf.npr
OPERATORS[NCR]      = pf.ncr
OPERATORS[POW]      = op.pow
OPERATORS[XTH_ROOT] = pf.xth_root


# ── Postfix operators ──────────────────────────────────────────────────────────

POSTFIXES[INV]       = pf.inv
POSTFIXES[SQ]        = lambda x: x ** 2
POSTFIXES[TRANSPOSE] = pf.transpose
POSTFIXES[CUBE]      = lambda x: x ** 3
POSTFIXES[FACT]      = pf.factorial
POSTFIXES[_t('%')]   = lambda x: x / 100


# ── Converters ─────────────────────────────────────────────────────────────────

CONVERTERS[_t('►DMS')]  = pf.to_dms
CONVERTERS[_t('►Dec')]  = pf.to_dec
CONVERTERS[_t('►Frac')] = pf.to_frac


# ── Functions (ArgParser → value) ─────────────────────────────────────────────

CALLABLES[ANS]                  = forms.ans_index_or_mul
CALLABLES[RAND]                 = _pure(pf.rand_list)

CALLABLES[_t('round(')]         = _pure(pf.round)
CALLABLES[_t('augment(')]       = _pure(pf.augment)
CALLABLES[_t('rowSwap(')]       = _pure(pf.rowswap)
CALLABLES[_t('row+(')]          = _pure(pf.row_plus)
CALLABLES[_t('*row(')]          = _pure(pf.times_row)
CALLABLES[_t('*row+(')]         = _pure(pf.times_row_plus)
CALLABLES[_t('max(')]           = _pure(pf.max)
CALLABLES[_t('min(')]           = _pure(pf.min)
CALLABLES[_t('R►Pr(')]         = _pure(pf.r_pr)
CALLABLES[_t('R►Pθ(')]        = _pure(pf.r_ptheta)
CALLABLES[_t('P►Rx(')]         = _pure(pf.p_rx)
CALLABLES[_t('P►Ry(')]         = _pure(pf.p_ry)
CALLABLES[_t('median(')]        = _pure(pf.median)
CALLABLES[_t('randM(')]         = _pure(pf.rand_m)
CALLABLES[_t('mean(')]          = _pure(pf.mean)
CALLABLES[_t('seq(')]           = forms.seq
CALLABLES[_t('fnInt(')]         = forms.fn_int
CALLABLES[_t('nDeriv(')]        = forms.n_deriv
CALLABLES[_t('dim(')]           = _pure(pf.dim)
CALLABLES[_t('int(')]           = _pure(pf.int_)
CALLABLES[_t('abs(')]           = _pure(pf.abs)
CALLABLES[_t('det(')]           = _pure(pf.det)
CALLABLES[_t('identity(')]      = _pure(pf.identity)
CALLABLES[_t('sum(')]           = _pure(pf.sum)
CALLABLES[_t('prod(')]          = _pure(pf.prod)
CALLABLES[_t('not(')]           = _pure(pf.not_)
CALLABLES[_t('iPart(')]         = _pure(pf.i_part)
CALLABLES[_t('fPart(')]         = _pure(pf.f_part)
CALLABLES[_t('√(')]             = _pure(pf.sqrt)
CALLABLES[_t('³√(')]            = _pure(pf.cbrt)
CALLABLES[_t('ln(')]            = _pure(pf.ln)
CALLABLES[_t('𝑒^(')]            = _pure(pf.exp)
CALLABLES[_t('log(')]           = _pure(pf.log)
CALLABLES[_t('⑽^(')]            = _pure(pf.pow10)
CALLABLES[_t('sin(')]           = _pure(pf.sin)
CALLABLES[_t('sin¹(')]          = _pure(pf.asin)
CALLABLES[_t('cos(')]           = _pure(pf.cos)
CALLABLES[_t('cos¹(')]          = _pure(pf.acos)
CALLABLES[_t('tan(')]           = _pure(pf.tan)
CALLABLES[_t('tan¹(')]          = _pure(pf.atan)
CALLABLES[_t('sinh(')]          = _pure(pf.sinh)
CALLABLES[_t('sinh¹(')]         = _pure(pf.asinh)
CALLABLES[_t('cosh(')]          = _pure(pf.cosh)
CALLABLES[_t('cosh¹(')]         = _pure(pf.acosh)
CALLABLES[_t('tanh(')]          = _pure(pf.tanh)
CALLABLES[_t('tanh¹(')]         = _pure(pf.atanh)
CALLABLES[_t('dbd(')]           = _pure(pf.dbd)
CALLABLES[_t('lcm(')]           = _pure(pf.lcm)
CALLABLES[_t('gcd(')]           = _pure(pf.gcd)
CALLABLES[_t('randInt(')]       = _pure(pf.rand_int)
CALLABLES[_t('randBin(')]       = _pure(pf.rand_bin)
CALLABLES[_t('sub(')]           = _pure(pf.sub_string)
CALLABLES[_t('stdDev(')]        = _pure(pf.stddev)
CALLABLES[_t('variance(')]      = _pure(pf.variance)
CALLABLES[_t('inString(')]      = _pure(pf.in_string)
CALLABLES[_t('normalcdf(')]     = _pure(pf.normalcdf)
CALLABLES[_t('invNorm(')]       = _pure(pf.invnorm)
CALLABLES[_t('tcdf(')]          = _pure(pf.tcdf)
CALLABLES[_t('χ²cdf(')]         = _pure(pf.chi2cdf)
CALLABLES[_t('Fcdf(')]          = _pure(pf.fcdf)
CALLABLES[_t('binompdf(')]      = _pure(pf.binompdf)
CALLABLES[_t('binomcdf(')]      = _pure(pf.binomcdf)
CALLABLES[_t('poissonpdf(')]    = _pure(pf.poissonpdf)
CALLABLES[_t('poissoncdf(')]    = _pure(pf.poissoncdf)
CALLABLES[_t('geometpdf(')]     = _pure(pf.geometpdf)
CALLABLES[_t('geometcdf(')]     = _pure(pf.geometcdf)
CALLABLES[_t('normalpdf(')]     = _pure(pf.normalpdf)
CALLABLES[_t('tpdf(')]          = _pure(pf.tpdf)
CALLABLES[_t('χ²pdf(')]         = _pure(pf.chi2pdf)
CALLABLES[_t('Fpdf(')]          = _pure(pf.fpdf)
CALLABLES[_t('randNorm(')]      = _pure(pf.rand_norm)
CALLABLES[_t('conj(')]          = _pure(pf.conj)
CALLABLES[_t('real(')]          = _pure(pf.real)
CALLABLES[_t('imag(')]          = _pure(pf.imag)
CALLABLES[_t('angle(')]         = _pure(pf.angle)
CALLABLES[_t('cumSum(')]        = _pure(pf.cum_sum)
CALLABLES[_t('expr(')]          = forms.expr
CALLABLES[_t('length(')]        = _pure(pf.length)
CALLABLES[_t('ΔList(')]         = _pure(pf.delta_list)
CALLABLES[_t('ref(')]           = _pure(pf.ref)
CALLABLES[_t('rref(')]          = _pure(pf.rref)
CALLABLES[_t('checkTmr(')]      = forms.check_tmr
CALLABLES[_t('timeCnv(')]       = _pure(pf.timecnv)
CALLABLES[_t('dayOfWk(')]       = _pure(pf.dayofwk)
CALLABLES[_t('getDtStr(')]      = forms.get_dt_str
CALLABLES[_t('getTmStr(')]      = forms.get_tm_str
CALLABLES[_t('invT(')]          = _pure(pf.invt)
CALLABLES[_t('remainder(')]     = _pure(pf.remainder)
CALLABLES[_t('Σ(')]             = forms.sigma
CALLABLES[_t('logBASE(')]       = _pure(pf.log_base)
CALLABLES[_t('randIntNoRep(')]  = _pure(pf.rand_int_no_rep)


# ── Commands (ArgParser → None, side-effect only) ─────────────────────────────

CALLABLES[_t('Fill(')]          = forms.fill
CALLABLES[_t('SortA(')]         = forms.sort_a
CALLABLES[_t('SortD(')]         = forms.sort_d
CALLABLES[_t('Matr►list(')]    = forms.matr_to_list
CALLABLES[_t('List►matr(')]    = forms.list_to_matr
CALLABLES[_t('setDate(')]       = forms.set_date
CALLABLES[_t('setTime(')]       = forms.set_time
CALLABLES[_t('setDtFmt(')]      = forms.set_dt_fmt
CALLABLES[_t('setTmFmt(')]      = forms.set_tm_fmt
CALLABLES[_t('ClockOff')]       = lambda a: a.env.clock_off()
CALLABLES[_t('ClockOn')]        = lambda a: a.env.clock_on()
