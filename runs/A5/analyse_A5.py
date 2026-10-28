"""A5, EXPLORATORY: monsters in big seas (Hs >= 5 m; also >= 8 m), by the rules written in A5 (PRODUCTION_LOG.md).

    py -3.13 rogue_waves/runs/A5/analyse_A5.py

Same rule and calls as L3 (+ A2, A3): AI = H / Hs > 2; the fair line weighted by the big-sea waves' share per peak-period
band; the wider of Wilson and a station bootstrap (1,000 resamples of the 152 stations, default_rng(0)); "below / on /
above" each line with the two-thirds rule over stations with >= 10^4 big-sea waves; < 100 monsters -> too few to call.
Writes A5_results.json. Labelled exploratory everywhere: decided after the L3 result.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.normpath(os.path.join(HERE, "..", ".."))
DICE = math.exp(-8)
st = {}
for p in sorted(glob.glob(os.path.join(HERE, "stations", "*.json"))):
    j = json.load(open(p))
    st.setdefault(j["file"], j)
assert len(st) == 152, f"{len(st)} stations"
missing = [f for f, j in st.items() if j["status"] != "ok"]
S = [j for j in st.values() if j["status"] == "ok"]
K = S[0]["k"]
i2 = K.index(2.0)
A2 = json.load(open(os.path.join(EP, "runs", "L1", "A2_results.json")))["fair_line"]
P = {int(tp): np.array(v["p"]) for tp, v in A2["by_tp"].items()}


def wilson(c, n, z=1.96):
    p = c / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


out = dict(missing=missing, label="EXPLORATORY (A5): decided after the L3 result")
for key in ("hs_ge_5", "hs_ge_8"):
    N = np.array([j[key]["n_valid"] for j in S], float)
    C = np.array([j[key]["over"][i2] for j in S], float)
    n, c = int(N.sum()), int(C.sum())
    if c < 100:
        out[key] = dict(waves=n, monsters=c, call="too few to call (< 100 monsters)")
        continue
    rate = c / n
    wl = wilson(c, n)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(S), len(S))
        boot.append(C[idx].sum() / max(N[idx].sum(), 1))
    bs = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    dec = wl if (wl[1] - wl[0]) >= (bs[1] - bs[0]) else bs
    band_n = np.zeros(6)
    for j in S:
        for b, e in enumerate(j[key]["by_tp_band"]):
            band_n[b] += e["n"]
    w = band_n / band_n.sum()
    fair2 = float(sum(w[b] * P[6 + 2 * b][i2] for b in range(6)))
    big = [(j[key]["over"][i2] / j[key]["n_valid"]) for j in S if j[key]["n_valid"] >= 10 ** 4]
    big = np.array(big)

    def call(line):
        frac = float((big < line).mean()) if len(big) else float("nan")
        if dec[1] < line and frac >= 2 / 3:
            return "below", frac
        if dec[0] > line:
            return "above", frac
        if dec[0] <= line <= dec[1]:
            return "on", frac
        return "under the line but fewer than two thirds of the stations below", frac

    cd, fd = call(DICE)
    cf, ff = call(fair2)
    tops = sorted([(j[key]["top3_ai"][0], j["file"]) for j in S if j[key].get("top3_ai")], reverse=True)[:3]
    out[key] = dict(waves=n, monsters=c, rate=rate, one_in=round(1 / rate), wilson=wl, bootstrap=bs,
                    deciding="Wilson" if dec is wl else "station bootstrap", stations_with_big_seas=int((N > 0).sum()),
                    stations_ge_1e4=int(len(big)), dice_one_in=round(1 / DICE), ratio_to_dice=rate / DICE,
                    call_dice=cd, frac_below_dice=fd, fair_line=fair2, fair_one_in=round(1 / fair2),
                    ratio_to_fair=rate / fair2, call_fair=cf, frac_below_fair=ff, tp_band_share=w.tolist(), largest=tops)
    print(f"{key}: {c:,} monsters in {n:,} waves = 1 in {round(1 / rate):,} | interval ({out[key]['deciding']}) "
          f"{dec[0]:.3e}..{dec[1]:.3e} | {int((N > 0).sum())} stations with such seas, {len(big)} with >= 10^4 waves")
    print(f"   dice line 1 in {round(1 / DICE):,}: ratio {rate / DICE:.3f}, {100 * fd:.0f} % of those stations below -> {cd}")
    print(f"   fair line 1 in {round(1 / fair2):,}: ratio {rate / fair2:.3f}, {100 * ff:.0f} % below -> {cf}")
    print(f"   largest H/Hs: {tops}")
print("missing stations:", missing)
json.dump(out, open(os.path.join(HERE, "A5_results.json"), "w"), indent=1, default=float)
