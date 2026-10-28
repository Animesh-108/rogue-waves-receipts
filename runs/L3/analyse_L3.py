"""Level 2 (data check) and Level 3 (the count) results, by the frozen rules + A2 + A3 (PRODUCTION_LOG.md).

    py -3.13 rogue_waves/runs/L3/analyse_L3.py

Reads runs/L3/stations/*.json (one per station; for 208p1, the first of its three identical reads) and the fair-line
runs (runs/L1/A2_results.json). Writes L3_results.json and prints the calls. Nothing here is tuned: every threshold
is the pre-registered one, and choices the frozen text left open are marked "(choice)" and reported both ways.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.normpath(os.path.join(HERE, "..", ".."))
DICE = math.exp(-8)                                                      # exp(-2 k^2) at k = 2


def wilson(c, n, z=1.96):
    p = c / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


st = {}
for p in sorted(glob.glob(os.path.join(HERE, "stations", "*.json"))):
    j = json.load(open(p))
    st.setdefault(j["file"], j)                                         # first file per station (208p1: 3 identical)
assert len(st) == 152 and all(j["status"] == "ok" for j in st.values()), "missing stations"
S = list(st.values())
K = S[0]["k"]
i2 = K.index(2.0)
N = np.array([j["n_valid"] for j in S], float)
C2 = np.array([j["over"][i2] for j in S], float)
N_all = int(sum(j["n_total"] for j in S))
N_valid = int(N.sum())
missing = int(sum(j["n_missing"] for j in S))
count2 = int(C2.sum())
rate2 = count2 / N_valid
wl = wilson(count2, N_valid)
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(S), len(S))
    boot.append(C2[idx].sum() / N[idx].sum())
bs = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
dec = wl if (wl[1] - wl[0]) >= (bs[1] - bs[0]) else bs                    # A3 (d): the wider interval decides
dec_name = "Wilson" if dec is wl else "station bootstrap"

# fair line (A2 (2), A3 (c)): six curves weighted by the share of waves in each peak-period band
A2 = json.load(open(os.path.join(EP, "runs", "L1", "A2_results.json")))["fair_line"]
assert A2["k"] == K
P = {int(tp): np.array(v["p"]) for tp, v in A2["by_tp"].items()}
band_n = np.zeros(6)
band_c = np.zeros((6, len(K)))
for j in S:
    for b, e in enumerate(j["by_tp_band"]):
        band_n[b] += e["n"]
        band_c[b] += e["over"]
w = band_n / band_n.sum()
fair = sum(w[b] * P[6 + 2 * b] for b in range(6))
fair2 = float(fair[i2])

big = [j for j in S if j["n_valid"] >= 10 ** 6]
r_big = np.array([j["over"][i2] / j["n_valid"] for j in big])
frac_big_below_dice = float((r_big < DICE).mean())
frac_big_below_fair = float((r_big < fair2).mean())                       # (choice) the pooled fair line, as written
own_fair = []                                                            # (choice, reported) each station's own band mix
for j in big:
    nb = np.array([e["n"] for e in j["by_tp_band"]], float)
    own_fair.append(sum(nb[b] / nb.sum() * P[6 + 2 * b][i2] for b in range(6)))
frac_big_below_own_fair = float((r_big < np.array(own_fair)).mean())
r_all = np.array([j["over"][i2] / j["n_valid"] for j in S])


def call(line, interval, frac):
    if interval[1] < line and frac >= 2 / 3:
        return "below"
    if interval[0] > line:
        return "above"
    if interval[0] <= line <= interval[1]:
        return "on"
    return "under the line but fewer than two thirds of the big stations below (not 'below')"


res = dict(
    stations=len(S), waves_total=N_all, waves_valid=N_valid, waves_missing=missing,
    missing_share=missing / N_all, gigabytes_read=round(sum(sum(c["bytes"] for c in j["read"].values()) for j in S) / 1e9, 2),
    first_record=min(j["attrs"]["time_coverage_start"] for j in S)[:10],
    last_record=max(j["attrs"]["time_coverage_end"] for j in S)[:10],
    L2=dict(waves_in_band=bool(1.26e9 <= N_all <= 1.54e9), waves=N_all,
            share_over_2=rate2, share_band=[3.33e-5, 1.0e-4], share_in_band=bool(3.33e-5 <= rate2 <= 1.0e-4)),
    L3=dict(count_over_2=count2, rate_over_2=rate2, one_in=round(1 / rate2), wilson=wl, bootstrap=bs,
            deciding_interval=dec_name, dice_line=DICE, dice_one_in=round(1 / DICE), ratio_to_dice=rate2 / DICE,
            big_stations=len(big), big_below_dice=frac_big_below_dice, call_dice=call(DICE, dec, frac_big_below_dice),
            fair_line=fair2, fair_one_in=round(1 / fair2), ratio_to_fair=rate2 / fair2, fair_weights=w.tolist(),
            big_below_fair=frac_big_below_fair, big_below_own_fair=frac_big_below_own_fair,
            call_fair=call(fair2, dec, frac_big_below_fair),
            over_2_2=int(sum(j["over_2_2"] for j in S)), rate_over_2_2=sum(j["over_2_2"] for j in S) / N_valid,
            dice_2_2=math.exp(-2 * 2.2 ** 2),
            crest_over_1_25=int(sum(j["crest_over_1_25"] for j in S)),
            rate_crest=sum(j["crest_over_1_25"] for j in S) / N_valid, dice_crest=math.exp(-8 * 1.25 ** 2),
            max_ai=max((j["top5_ai"][0], j["file"]) for j in S)),
    curve=dict(k=K, real=[float(sum(j["over"][i] for j in S) / N_valid) for i in range(len(K))],
               dice=[math.exp(-2 * k * k) for k in K], fair=fair.tolist()),
    by_band=[dict(tp=6 + 2 * b, waves=int(band_n[b]), real_2=float(band_c[b, i2] / band_n[b]) if band_n[b] else None,
                  fair_2=float(P[6 + 2 * b][i2])) for b in range(6)],
    bet=dict(range=[4e-5, 1e-4], in_range=bool(4e-5 <= rate2 <= 1e-4),
             all_stations_below_dice=float((r_all < DICE).mean()), big_stations_below_dice=frac_big_below_dice,
             eighty_pct_met_all=bool((r_all < DICE).mean() >= 0.8)),
)
json.dump(res, open(os.path.join(HERE, "L3_results.json"), "w"), indent=1, default=float)
L = res["L3"]
print(f"stations {len(S)}, waves {N_all:,} (valid {N_valid:,}, missing {missing:,} = {100 * missing / N_all:.3f} %), "
      f"{res['gigabytes_read']} GB, records {res['first_record']} .. {res['last_record']}")
print(f"L2: waves in ±10 % of 1.4 billion: {res['L2']['waves_in_band']}; share over 2 Hs {rate2:.3e} in [3.33e-5, 1e-4]: "
      f"{res['L2']['share_in_band']}")
print(f"L3: over 2 Hs {count2:,} of {N_valid:,} = {rate2:.3e} (1 in {L['one_in']:,})")
print(f"    Wilson {wl[0]:.3e}..{wl[1]:.3e}; station bootstrap {bs[0]:.3e}..{bs[1]:.3e}; deciding: {dec_name}")
print(f"    dice line {DICE:.3e} (1 in {L['dice_one_in']:,}): ratio {L['ratio_to_dice']:.3f}; big stations below "
      f"{100 * frac_big_below_dice:.1f} % of {len(big)} -> CALL: {L['call_dice']}")
print(f"    fair line {fair2:.3e} (1 in {L['fair_one_in']:,}): ratio {L['ratio_to_fair']:.3f}; big below (pooled fair) "
      f"{100 * frac_big_below_fair:.1f} %, (own mix) {100 * frac_big_below_own_fair:.1f} % -> CALL: {L['call_fair']}")
print(f"    over 2.2: {L['over_2_2']:,} = {L['rate_over_2_2']:.3e} (dice {L['dice_2_2']:.3e}); crest > 1.25 Hs: "
      f"{L['crest_over_1_25']:,} = {L['rate_crest']:.3e} (dice {L['dice_crest']:.3e}); max H/Hs {L['max_ai']}")
print("    by band:", [(b["tp"], b["waves"], f"{b['real_2']:.2e}" if b["real_2"] else None, f"{b['fair_2']:.2e}") for b in res["by_band"]])
print(f"bet: pooled in 4e-5..1e-4: {res['bet']['in_range']}; stations below dice: all {100 * res['bet']['all_stations_below_dice']:.1f} %, "
      f"big {100 * frac_big_below_dice:.1f} % (bet: >= 80 %)")
