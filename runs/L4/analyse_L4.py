"""Level 4, the duel: results by the frozen rules + A4 (PRODUCTION_LOG.md).

    py -3.13 rogue_waves/runs/L4/analyse_L4.py

Per sea and arm: monsters (H > 2 Hs) / waves, pooled over the 50 seeds; 95 % interval from a seed bootstrap (1,000
resamples of the seeds, numpy default_rng(0), A4 (e)); ratios B/A, C/A (and C/C0, diagnostic) with paired seed-bootstrap
intervals. Calls (frozen): "matters" if the ratio >= 2 with non-overlapping arm intervals; "small correction" if the
ratio < 1.5 with overlapping intervals; otherwise "unclear". Writes L4_results.json.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "runs.jsonl"))]
k2 = rows[0]["k"].index(2.0)
data = defaultdict(dict)                                                  # (sea, arm) -> seed -> (monsters, waves)
for r in rows:
    data[(r["sea"], r["arm"])][r["seed"]] = (r["over"][k2], r["n_waves"])
seas = sorted({r["sea"] for r in rows}, key=[r["sea"] for r in rows].index)
rng = np.random.default_rng(0)
seeds = np.arange(50)
B = rng.integers(0, 50, (1000, 50))                                       # the same resamples for every arm (paired)


def arm_stats(d):
    m = np.array([d[s][0] for s in seeds], float)
    w = np.array([d[s][1] for s in seeds], float)
    boot = m[B].sum(1) / w[B].sum(1)
    return m, w, dict(monsters=int(m.sum()), waves=int(w.sum()), rate=float(m.sum() / w.sum()),
                      ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


def ratio(num, den):
    (mn, wn), (md, wd) = num, den
    r = (mn.sum() / wn.sum()) / (md.sum() / wd.sum())
    boot = (mn[B].sum(1) / wn[B].sum(1)) / (md[B].sum(1) / wd[B].sum(1))
    return float(r), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def call(r, ci_num, ci_den):
    overlap = not (ci_num[0] > ci_den[1] or ci_num[1] < ci_den[0])
    if r >= 2 and not overlap:
        return "the special mechanism matters"
    if r < 1.5 and overlap:
        return "a small correction"
    return "unclear"


out = {}
for sea in seas:
    assert all(len(data[(sea, a)]) == 50 for a in ("A", "B", "C", "C0")), f"{sea}: not all 50 seeds"
    st = {a: arm_stats(data[(sea, a)]) for a in ("A", "B", "C", "C0")}
    res = {a: st[a][2] for a in st}
    for name, (n_, d_) in (("B/A", ("B", "A")), ("C/A", ("C", "A")), ("C/C0", ("C", "C0"))):
        r, ci = ratio(st[n_][:2], st[d_][:2])
        res[name] = dict(ratio=r, ci=ci, call=call(r, res[n_]["ci"], res[d_]["ci"]) if name != "C/C0" else "diagnostic")
    out[sea] = res
    print(sea)
    for a in ("A", "B", "C", "C0"):
        x = res[a]
        print(f"  {a:2s}: {x['monsters']:6,} / {x['waves']:,} = {x['rate']:.3e} (1 in {round(1 / x['rate']):,}), "
              f"95 % {x['ci'][0]:.3e}..{x['ci'][1]:.3e}")
    for name in ("B/A", "C/A", "C/C0"):
        x = res[name]
        print(f"  {name}: {x['ratio']:.3f} (95 % {x['ci'][0]:.3f}..{x['ci'][1]:.3f}) -> {x['call']}")
bet = all(out[s]["B/A"]["ratio"] < 1.5 for s in seas) and all(out[s]["C/A"]["ratio"] >= 2 for s in seas)
out["_bet_B_A_lt_1_5_and_C_A_ge_2_everywhere"] = bet
print("bet (B/A < 1.5 and C/A >= 2 in all three seas):", bet)
json.dump(out, open(os.path.join(HERE, "L4_results.json"), "w"), indent=1)
