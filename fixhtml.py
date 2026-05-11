import re
from html import unescape

with open('catalog0.html', encoding='utf-8') as f:
	garbled = f.read()

def fix_match(m):
    text = m.group(0)
    chars = unescape(text)
    try:
        return chars.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text  # not mojibake, leave it alone
		
fixed = re.sub(r'(?:&[a-zA-Z]+;)+', fix_match, garbled)

with open('catalog.html', 'w', encoding='utf-8') as f:
	f.write(fixed)