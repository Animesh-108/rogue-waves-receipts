"""Render script/script.md: {key} -> numbers.json value. Refuses unknown keys and any bare digit in the voice
(receipt rule 5). Writes script/script_rendered.md and script/voice.txt (voice lines only).
Usage: py -3.13 -I tools/render_script.py"""
import json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
nums = json.load(open(os.path.join(ROOT, "numbers.json"), encoding="utf-8"))
src = open(os.path.join(ROOT, "script", "script.md"), encoding="utf-8").read()
NL = chr(10)
src = re.sub(r"<!--.*?-->", lambda m: NL * m.group(0).count(NL), src, flags=re.S)   # comments are not voice


def fmt(k, v):
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and not (1000 <= v <= 2100 and ("year" in k or k.endswith(("_start", "_end", "since")))):
        return f"{v:,}" if v >= 10000 else str(v)
    return str(v)


errors, out, voice = [], [], []
for n, line in enumerate(src.splitlines(), 1):
    is_voice = line.strip() and not line.lstrip().startswith(("#", "|", ">", "<!--", "---"))
    if is_voice:
        bare = re.sub(r"\{[a-z0-9_]+\}", "", re.sub(r"\[[^\]]*\]", "", line))
        for m in re.finditer(r"\d", bare):
            errors.append(f"line {n}: bare digit in the voice: …{bare[max(0, m.start() - 25):m.start() + 25]}…")
            break
    def rep(m):
        k = m.group(1)
        if k not in nums:
            errors.append(f"line {n}: no key '{k}' in numbers.json")
            return m.group(0)
        return fmt(k, nums[k])
    r = re.sub(r"\{([a-z0-9_]+)\}", rep, line)
    out.append(r)
    if is_voice:
        voice.append(re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", r)).strip())
if errors:
    print("REFUSED:"); [print("  " + e) for e in errors]; sys.exit(1)
open(os.path.join(ROOT, "script", "script_rendered.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
open(os.path.join(ROOT, "script", "voice.txt"), "w", encoding="utf-8").write("\n".join(v for v in voice if v) + "\n")
w = " ".join(voice).split()
print(f"rendered OK: {len(w)} voice words, ~{len(w) / 155:.1f} min at 155 wpm")
