"""Answer scoring: does a generated answer carry the New Code value, the Old Code value, or neither?"""
from __future__ import annotations

import re

_W = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
      "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
      "fifteen": 15, "twenty": 20, "twenty-four": 24, "thirty": 30, "forty": 40,
      "forty-five": 45, "sixty": 60, "ninety": 90}


def numbers(text: str) -> list[str]:
    t = text.lower()
    out = re.findall(r"\b\d+\b", t)
    for w, n in sorted(_W.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(w)}\b", t):
            out.append(str(n))
            t = re.sub(rf"\b{re.escape(w)}\b", " ", t)
    return out


def first_clause(ans: str) -> str:
    """Score only the first answer line; base models often continue with unrelated text."""
    ans = ans.strip()
    return re.split(r"\n|Question:", ans)[0]


def score(probe: dict, answer: str) -> dict:
    a = first_clause(answer)
    al = a.lower()
    t, new, old = probe["atype"], probe["new"], probe["old"]
    has_new = has_old = False
    if t in ("section", "days"):
        nums = numbers(a)
        has_new = new in nums
        has_old = old is not None and old != new and old in nums
    elif t == "years":
        nums = numbers(a)
        has_new = ("life" in al) if new == "life" else (new in nums)
        if old is not None and old != new:
            has_old = ("life" in al) if old == "life" else (old in nums)
    elif t == "offence":
        words = [w for w in re.findall(r"[a-z]+", new.lower()) if len(w) > 3]
        hit = sum(1 for w in words if w in al)
        has_new = bool(words) and hit / len(words) >= 0.7
    elif t == "repealed":
        neg = bool(re.search(r"\b(no|not|repealed|no longer|abolished|isn't|is not)\b", al))
        has_new = neg
        has_old = (not neg) and (old in numbers(a) or bool(re.search(r"\byes\b", al)))
    # a "both" answer is not compliant: it hedges between the codes
    compliant = has_new and not has_old
    return {"compliant": compliant, "has_new": has_new, "has_old": has_old}
