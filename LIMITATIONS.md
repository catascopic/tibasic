# Emulator Limitations

Known behavioral differences between this emulator and actual TI-84+ SE calculator behavior.

---

## `Input` — prompt expression parsing

### How the calculator works

The calculator uses a first-token classifier to decide whether the argument list begins
with a prompt string or directly names a storage target.  The first token must be one of:

- A string literal (`"`)
- A string variable (`Str0`–`Str9`)
- `Ans`
- `sub(`

Any other first token (a numeric variable, a number, `(`, etc.) is treated as the storage
target with no prompt.

The bare single-variable form `Input Str1` (no comma) is also recognized: if a string
variable appears as the only argument, it is the storage target, not a prompt.  This is
consistent with the first-token rule because there is no comma to signal a second argument.

Once the classifier commits to a prompt, the calculator applies an additional restriction:
the clock functions `getDtStr(` and `getTmStr(` (added in OS 1.10 as 0xEF-table tokens)
are rejected anywhere inside the prompt expression.  The depth of the rejected token
determines the error type:

| Example | Result |
|---|---|
| `Input getDtStr(1),X` | First token not in starter set → ERR:SYNTAX (target parse fails) |
| `Input "A"+getDtStr(1),X` | Top-level 0xEF token inside prompt → ERR:SYNTAX |
| `Input sub(getDtStr(1),1,1),X` | 0xEF token nested inside `sub(` → ERR:DATA TYPE |

A parenthesized expression cannot begin a prompt either:
`Input ("A"),X` → ERR:SYNTAX (explicit check, separate from the 0xEF restriction).

### How we implemented it

We use the same first-token classifier (`_prompt_starter` in `prgmcmds.py`): a string
literal, string variable, `Ans`, or `sub(` starts a prompt; anything else is the target.
The lone-`StrN` case is handled by peeking at the token immediately after the string
variable — if it is a statement boundary or EOF, the variable is the target.

After committing to a prompt, we call `expr()` and evaluate normally with no further
restrictions.

### Differences

**1. Clock functions inside a prompt are allowed.**

`Input "A"+getDtStr(1),X` and `Input sub(getDtStr(1),1,1),X` succeed in our emulator
rather than raising an error.  Enforcing the 0xEF restriction requires a dedicated
pre-parse walk of the captured prompt tokens to inspect every sub-expression at varying
depths — a substantial amount of code for a pair of obscure OS 1.10 clock functions.
We deliberately omit this check.

**2. Parenthesized first token fails for a different reason.**

`Input ("A"),X` raises an error in both the calculator and here, but via different paths.
The calculator has an explicit check against `(` as a prompt opener.  We reach the same
outcome because `(` is not in the prompt-starter set, so it falls through to target
parsing, which fails because `(` is not a variable reference.  The user-visible error is
equivalent; only the mechanism differs.

**3. Everything else matches.**

String concatenation (`"A"+"B"`, `Str1+"B"`, `Str1+Str2`), `Ans`, and `sub(` prompts
all behave identically to the calculator.  Type errors at evaluation time (e.g.
`"A"+5,X` → ERR:DATA TYPE) are also preserved, because they arise naturally from
expression evaluation rather than any special prompt logic.
