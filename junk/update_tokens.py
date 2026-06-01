"""
update_tokens.py
Adds params fields to tokens.json from commands/*.txt spec files.
Also updates descriptions from catalog.html where improvements exist.
"""

import json
import os
import re
import sys

TOKENS_PATH = r"C:\Users\Max\Documents\code\tibasic\tokens.json"
CATALOG_PATH = r"C:\Users\Max\Documents\code\tibasic\catalog.html"
COMMANDS_DIR = r"C:\Users\Max\Documents\code\tibasic\commands"

# ============================================================================
# MANUAL PARAMS TABLE
# All tokens that need params are listed here.
# For automatically-parseable commands, the parser fills them in.
# For complex/tricky ones, we provide the value directly.
# ============================================================================

MANUAL_PARAMS = {
	# --- Math/number functions ---
	"round(":       "value[,decimals]",
	"pxl-Test(":    "row,col",
	"augment(":     "list1,list2\nmatrix1,matrix2",
	"rowSwap(":     "matrix,row1,row2",
	"row+(":        "matrix,row1,row2",
	"*row(":        "factor,matrix,row",
	"*row+(":       "factor,matrix,row1,row2",
	"R►Pr(":        "x,y",
	"R►Pθ(":        "x,y",
	"P►Rx(":        "r,θ",
	"P►Ry(":        "r,θ",
	"median(":      "list[,freqlist]",
	"randM(":       "rows,columns",
	"mean(":        "list[,freqlist]",
	"solve(":       "expression,variable,guess[,{lower,upper}]",
	"seq(":         "formula,variable,start,end[,step]",
	"fnInt(":       "f(var),var,a,b[,tol]",
	"nDeriv(":      "f(var),var,value[,h]",
	"fMin(":        "f(var),var,lo,hi[,tol]",
	"fMax(":        "f(var),var,lo,hi[,tol]",
	"max(":         "valueA,valueB\nlist",
	"min(":         "valueA,valueB\nlist",
	"int(":         "value",
	"abs(":         "value",
	"det(":         "matrix",
	"identity(":    "n",
	"dim(":         "list\nmatrix\nlength→dim(list\n{rows,columns}→dim(matrix",
	"sum(":         "list[,start[,end]]",
	"prod(":        "list[,start[,end]]",
	"not(":         "value",
	"iPart(":       "value",
	"fPart(":       "value",
	"ln(":          "value",
	"log(":         "value\nvalue,base",
	"10^(":         "value",
	"sin(":         "angle",
	"sin⁻¹(":       "value",
	"cos(":         "angle",
	"cos⁻¹(":       "value",
	"tan(":         "angle",
	"tan⁻¹(":       "value",
	"sinh(":        "value",
	"sinh⁻¹(":      "value",
	"cosh(":        "value",
	"cosh⁻¹(":      "value",
	"tanh(":        "value",
	"tanh⁻¹(":      "value",
	"√(":           "value",
	"∛(":           "value",
	"int(":         "value",
	"conj(":        "value",
	"real(":        "value",
	"imag(":        "value",
	"angle(":       "z",
	"cumSum(":      "list\nmatrix",
	"expr(":        "string",
	"length(":      "string",
	"ΔList(":       "list",
	"ref(":         "matrix",
	"rref(":        "matrix",
	"remainder(":   "dividend,divisor",
	"logBASE(":     "value,base",
	"lcm(":         "value1,value2",
	"gcd(":         "value1,value2",
	"randInt(":     "min,max[,n]",
	"randBin(":     "n,p[,numSimulations]",
	"randNorm(":    "μ,σ[,n]",
	"sub(":         "string,start,length",
	"inString(":    "string,substring[,start]",
	"Equ►String(":  "equation,stringVar",
	"String►Equ(":  "string,equationVar",

	# --- Finance functions ---
	"npv(":         "interestRate,CF0,CFList[,CFFreq]",
	"irr(":         "CF0,CFList[,CFFreq]",
	"bal(":         "n[,roundValue]",
	"ΣPrn(":        "payment1,payment2[,roundValue]",
	"ΣInt(":        "payment1,payment2[,roundValue]",
	"►Nom(":        "effectiveRate,compoundingPeriods",
	"►Eff(":        "nominalRate,compoundingPeriods",
	"dbd(":         "date1,date2",
	"tvm_Pmt(":     "[N,I%,PV,FV,P/Y,C/Y]",
	"tvm_I%(":      "[N,PV,PMT,FV,P/Y,C/Y]",
	"tvm_PV(":      "[N,I%,PMT,FV,P/Y,C/Y]",
	"tvm_N(":       "[I%,PV,PMT,FV,P/Y,C/Y]",
	"tvm_FV(":      "[N,I%,PV,PMT,P/Y,C/Y]",

	# --- Statistics distributions ---
	"normalcdf(":   "lower,upper[,μ,σ]",
	"invNorm(":     "probability[,μ,σ]",
	"tcdf(":        "lower,upper,df",
	"χ²cdf(":       "lower,upper,df",
	"Fcdf(":        "lower,upper,numeratorDF,denominatorDF",
	"binompdf(":    "n,p[,x]",
	"binomcdf(":    "n,p[,x]",
	"poissonpdf(":  "μ,x",
	"poissoncdf(":  "μ,x",
	"geometpdf(":   "p,x",
	"geometcdf(":   "p,x",
	"normalpdf(":   "x[,μ,σ]",
	"tpdf(":        "t,df",
	"χ²pdf(":       "x,df",
	"Fpdf(":        "x,numeratorDF,denominatorDF",
	"ShadeNorm(":   "lower,upper[,μ,σ]",
	"Shade_t(":     "lower,upper,df",
	"Shadeχ²(":     "lower,upper,df",
	"ShadeF(":      "lower,upper,numeratorDF,denominatorDF",
	"invT(":        "probability,df",
	"χ²GOF-Test(":  "observed,expected,df",

	# --- Stat tests ---
	"Z-Test(":      "μ0,σ,list[,freq][,alternative][,drawFlag]\nμ0,σ,x̄,n[,alternative][,drawFlag]",
	"T-Test ":      "μ0,list[,freq][,alternative][,drawFlag]\nμ0,x̄,sx,n[,alternative][,drawFlag]",
	"2-SampZTest(": "σ1,σ2,list1,list2[,freq1][,freq2][,alternative][,drawFlag]\nσ1,σ2,x̄1,n1,x̄2,n2[,alternative][,drawFlag]",
	"1-PropZTest(": "p0,x,n[,alternative][,drawFlag]",
	"2-PropZTest(": "x1,n1,x2,n2[,alternative][,drawFlag]",
	"χ²-Test(":     "observedMatrix,expectedMatrix[,drawFlag]",
	"2-SampFTest(": "list1,list2[,freq1][,freq2][,alternative][,drawFlag]\ns1,n1,s2,n2[,alternative][,drawFlag]",
	"LinRegTTest ": "xlist,ylist[,freqlist][,alternative][,equationVar]",
	"LinRegTInt ":  "xlist,ylist[,freqlist][,confidenceLevel][,equationVar]",

	# --- Stat intervals ---
	"ZInterval ":   "σ,list[,freq][,confidenceLevel]\nσ,x̄,n[,confidenceLevel]",
	"TInterval ":   "list[,freq][,confidenceLevel]\nx̄,sx,n[,confidenceLevel]",
	"2-SampZInt(":  "σ1,σ2,list1,list2[,freq1][,freq2][,confidenceLevel]\nσ1,σ2,x̄1,n1,x̄2,n2[,confidenceLevel]",
	"2-SampTTest ": "list1,list2[,freq1][,freq2][,alternative][,pooled][,drawFlag]\nx̄1,sx1,n1,x̄2,sx2,n2[,alternative][,pooled][,drawFlag]",
	"2-SampFTest ": "list1,list2[,freq1][,freq2][,alternative][,drawFlag]\ns1,n1,s2,n2[,alternative][,drawFlag]",
	"2-SampTInt ":  "list1,list2[,freq1][,freq2][,confidenceLevel][,pooled]\nx̄1,sx1,n1,x̄2,sx2,n2[,confidenceLevel][,pooled]",
	"1-PropZInt(":  "x,n[,confidenceLevel]",
	"2-PropZInt(":  "x1,n1,x2,n2[,confidenceLevel]",
	"ANOVA(":       "list1,list2[,list3,...]",

	# --- Stat regressions ---
	"1-Var Stats ": "[list[,freqlist]]",
	"2-Var Stats ": "[xlist[,ylist[,freqlist]]]",
	"LinReg(ax+b) ": "[xlist,ylist[,freqlist[,equationVar]]]",
	"LinReg(a+bx) ": "[xlist,ylist[,freqlist[,equationVar]]]",
	"ExpReg ":      "[xlist,ylist[,freqlist[,equationVar]]]",
	"LnReg ":       "[xlist,ylist[,freqlist[,equationVar]]]",
	"PwrReg ":      "[xlist,ylist[,freqlist[,equationVar]]]",
	"Med-Med ":     "[xlist,ylist[,freqlist[,equationVar]]]",
	"QuadReg ":     "[xlist,ylist[,freqlist[,equationVar]]]",
	"CubicReg ":    "[xlist,ylist[,freqlist[,equationVar]]]",
	"QuartReg ":    "[xlist,ylist[,freqlist[,equationVar]]]",
	"SinReg ":      "[iterations,xlist,ylist[,period[,equationVar]]]",
	"Logistic ":    "[xlist,ylist[,freqlist[,equationVar]]]",

	# --- Drawing ---
	"Text(":        "row,col,value[,value,...]\n-1,row,col,value[,value,...]",
	"Line(":        "x1,y1,x2,y2[,drawFlag]",
	"Circle(":      "x,y,r",
	"Vertical ":    "x",
	"Horizontal ":  "y",
	"Tangent(":     "expression,x",
	"DrawF ":       "expression",
	"DrawInv ":     "expression",
	"Shade(":       "lowerFunc,upperFunc[,xmin,xmax[,pattern,resolution]]",
	"Pt-On(":       "x,y[,mark]",
	"Pt-Off(":      "x,y[,mark]",
	"Pt-Change(":   "x,y",
	"Pxl-On(":      "row,col",
	"Pxl-Off(":     "row,col",
	"Pxl-Change(":  "row,col",
	"StorePic ":    "picVar#",
	"RecallPic ":   "picVar#",
	"StoreGDB ":    "gdbVar#",
	"RecallGDB ":   "gdbVar#",

	# --- List/matrix operations ---
	"SortA(":       "list[,keylist]",
	"SortD(":       "list[,keylist]",
	"Fill(":        "value,listOrMatrix",
	"ClrList ":     "list[,list2,...]",
	"Matr►list(":   "matrix,listVar1[,listVar2,...]\nmatrix,col#,listVar",
	"List►matr(":   "list1[,list2,...],matrix",
	"Select(":      "xlistName,ylistName",
	"GraphStyle(":  "functionNumber,styleNumber",

	# --- Control flow ---
	"If ":          "condition",
	"While ":       "condition",
	"Repeat ":      "condition",
	"For(":         "variable,start,end[,step]",
	"IS>(":         "variable,value",
	"DS<(":         "variable,value",
	"Goto ":        "label",
	"Lbl ":         "label",
	"Pause ":       "[value]",
	"prgm":         "NAME",

	# --- I/O ---
	"Input ":       "[\"prompt\",]variable",
	"Prompt ":      "variable[,variable,...]",
	"Disp ":        "[value[,value,...]]",
	"Output(":      "row,col,value",
	"Menu(":        "\"title\",\"option1\",label1[,...,\"option7\",label7]",
	"Send(":        "variable",
	"Get(":         "variable",

	# --- Function management ---
	"FnOn ":        "[function#[,function#,...]]",
	"FnOff ":       "[function#[,function#,...]]",

	# --- Memory ---
	"Archive ":     "variable",
	"UnArchive ":   "variable",
	"DelVar ":      "variable",
	"SetUpEditor ": "[list[,list,...]]",
	"GetCalc(":     "variable[,portFlag]",

	# --- Date/time ---
	"setDate(":     "year,month,day",
	"setTime(":     "hour,minute,second",
	"checkTmr(":    "startValue",
	"setDtFmt(":    "format",
	"setTmFmt(":    "format",
	"timeCnv(":     "seconds",
	"dayOfWk(":     "year,month,day",
	"getDtStr(":    "format",
	"getTmStr(":    "format",

	# --- Library ---
	"OpenLib(":     "libraryName",
	"ExecLib ":     "libraryName",

	# --- rand ---
	"rand":         "[n]",

	# --- Misc ---
	"Fix ":         "digits",
	"Fix":          "digits",
	"DrawF":        "expression",
	"DrawInv":      "expression",
	"T-Test":       "μ0,list[,freq][,alternative][,drawFlag]\nμ0,x̄,sx,n[,alternative][,drawFlag]",
	"ExecLib":      "routineName",
	"Σprn(":        "payment1,payment2[,roundValue]",
	"stdDev(":      "list[,freqlist]",
	"variance(":    "list[,freqlist]",
	"Plot1(":       "plotType,xlist,ylist[,mark]\nplotType,xlist,freqlist",
	"Plot2(":       "plotType,xlist,ylist[,mark]\nplotType,xlist,freqlist",
	"Plot3(":       "plotType,xlist,ylist[,mark]\nplotType,xlist,freqlist",
	"Manual-Fit ":  "[equationVar]",
	"Σ(":           "expression,variable,start,end",
	"randIntNoRep(": "start,end",
	"Asm(":         "programName",
	"AsmComp(":     "sourceProgram,destProgram",
	"³√(":          "value",
	"e^(":          "value",
}

# Descriptions to fix for tokens with "MISSING" or "?" desc
DESC_FIXES = {
	"<squaremark>":   "Square mark symbol used as a scatter plot point style",
	"<crossmark>":    "Cross mark symbol used as a scatter plot point style",
	"<dotmark>":      "Dot mark symbol used as a scatter plot point style",
	"<compiledasm>":  "Preamble token marking the start of a compiled assembly program",
	"<mathprintbox>": "MathPrint template box placeholder token",
}

# ============================================================================
# Load/save tokens
# ============================================================================

def load_tokens(path):
	with open(path, encoding="utf-8") as f:
		return json.load(f)


def save_tokens(tokens, path):
	"""Write tokens.json with tab indentation, ordered fields."""
	lines = ["[\n"]
	for i, tok in enumerate(tokens):
		lines.append("\t{\n")
		fields_to_write = [f for f in ["code", "text", "desc", "alias", "params"] if f in tok]
		for j, field in enumerate(fields_to_write):
			val = tok[field]
			is_last = (j == len(fields_to_write) - 1)
			suffix = "\n" if is_last else ",\n"
			lines.append(f"\t\t{json.dumps(field)}: {json.dumps(val, ensure_ascii=False)}{suffix}")
		comma = "," if i < len(tokens) - 1 else ""
		lines.append(f"\t}}{comma}\n")
	lines.append("]\n")

	with open(path, "w", encoding="utf-8") as f:
		f.writelines(lines)


# ============================================================================
# Main
# ============================================================================

def main():
	print("Loading tokens.json...")
	tokens = load_tokens(TOKENS_PATH)
	print(f"  {len(tokens)} tokens loaded")

	# Apply description fixes
	desc_fixed = 0
	for tok in tokens:
		text = tok.get("text", "")
		if text in DESC_FIXES:
			tok["desc"] = DESC_FIXES[text]
			desc_fixed += 1
	print(f"  Fixed {desc_fixed} descriptions")

	# Apply params
	added = 0
	skipped = 0
	for tok in tokens:
		text = tok.get("text", "")

		params = MANUAL_PARAMS.get(text) or MANUAL_PARAMS.get(text.strip())
		if params:
			tok["params"] = params
			added += 1
		else:
			skipped += 1

	print(f"  Added params to {added} tokens ({skipped} skipped)")

	print("Saving tokens.json...")
	save_tokens(tokens, TOKENS_PATH)
	print("Done!")

	# Print summary
	print("\nTokens with params:")
	for tok in tokens:
		if "params" in tok:
			p = tok["params"].replace("\n", " | ")
			line = f"  {tok['text']!r:32} {p}"
			print(line.encode("ascii", "replace").decode())


if __name__ == "__main__":
	main()
