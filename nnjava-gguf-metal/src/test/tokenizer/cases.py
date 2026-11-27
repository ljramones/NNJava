"""Emit deterministic case/UTF-8-hex lines for the vocab-only native oracle."""
import sys
from pathlib import Path
import random
cases = {
 "empty":"","ascii":"hello world!","contractions":"I'LL we've They'Re it’S don't DON'T Đ's 't 'd 'LL 'VE",
 "numbers":"-123456789 ١٢٣٤٥６۷８９ Ⅲ² before42after ₹2Ⅹ²",
 "unicode":"Año, déjà vu — 조선 𐐎 тест 中文 e\u0301 🚀👩🏽‍⚕️\u200d🦯\x00終",
 "newline":"hello \t\n\r\n \n\n  word\n.?!\r\n!!  \r\r next\r",
 "whitespace":"a\u0085\u00a0b\u2002c\u2028\u2029\x1c\x1d\x1e\x1f\x0b\x0c  \t W \t 🗺 ",
 "spacing":"   leading     around 42--  \'s  !--   \t trailing   ",
 "repeats":"abc"*301+"a"*115+"  t   o "+"="*137+"🐢"*42,
 "special":"<|begin_of_text|>hello<|eot_id|><|eom_id|><|end_of_text|>",
 "special_nested":"<|unknown<|reserved_special_token_17|><|oops|><|python_tag|><|start_header_id|>",
 "controls":"<|start_header_id|>user<|end_header_id|>\n\n朝<|eot_id|>",
 "byte_alphabet":"".join(chr(i) for i in range(128))+"".join(chr(i) for i in range(128,384)),
 "unicode_new":"\U0001cc00a\U00011f02b\U0001e4d0c\U00010d40d\U0001fae8",
}
cases["cleanup-punctuation"] = " ".join("x "+chr(i)+" y" for i in range(32,127) if not chr(i).isalnum())
cases["cleanup-apostrophes"] = " | ".join("a "+w+" b" for w in ("'","'x","'m","'s","'re","'ve","'d","'ll","'t","n't","N'T","'S","'super","'me","'red","'very","'  ","'  a"))
cases["cleanup-boundary"] = " .!? 's ' ' word , . won't .  "
rng=random.Random(90112)
alphabet=list("aA'sTVel 0१，!\n\r\té猫中。🔎Z±") + [
    chr(cp) for cp in (0x85,0x2028,0x2029,0xa0,0x1680,0x200d,0x20dd,0x2164,0x11580,0x2ebf0,0x1e4d0)]
for i in range(48):
    cases[f"mix-{i:02}"]="".join(rng.choice(alphabet) for j in range(rng.randrange(16,180)))
# Exercise native contraction cleanup and adjacent punctuation, including overlapping tokens.
rng=random.Random(3141)
fragments=[" ","'","!","?","s","m","n't","'re","'ll","n","t",",","re","ve",".","S"]
for i in range(600):
    cases[f"cleanup-{i:03}"]="".join(rng.choice(fragments) for j in range(rng.randint(5,45)))
with Path(sys.argv[1]).open("w") as out:
    for name,text in cases.items():print(name,text.encode("utf8").hex(),sep="\t",file=out)
