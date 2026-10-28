"""Amendment A2 runs (written in PRODUCTION_LOG.md before this ran): the L1.1 re-test on fresh seeds and the fair
dice line (pure chance measured FOWD's way). Simulated seas only; no buoy data.

    py -3.13 rogue_waves/runs/L1/run_A2.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import wavesim as ws  # noqa: E402

N = 2 ** 24
KS = np.round(np.arange(1.0, 2.5001, 0.05), 2)
R = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), numpy=np.__version__, amendment="A2")

# ---------------------------------------------------------------- (1) L1.1 re-test, seeds 110-209
TP = 10.0
narrow = lambda f: ws.spectrum_gauss(f, 1 / TP, 0.02)                   # noqa: E731
per = []
for s in range(110, 210):
    eta, m0 = ws.synth(narrow, N, TP / 40, s)
    _, H, _ = ws.zero_up_waves(eta)
    per.append([len(H)] + [int((H > k * 4 * np.sqrt(m0)).sum()) for k in (1.0, 1.5, 2.0)])
per = np.array(per)
n = per[:, 0].sum()
chk = {}
for j, k in enumerate((1.0, 1.5, 2.0), start=1):
    c = per[:, j]
    exp_ = n * np.exp(-2 * k * k)
    half = 1.96 * np.sqrt(len(c)) * c.std(ddof=1)
    chk[f"k{k}"] = dict(count=int(c.sum()), expected=float(exp_), ratio=float(c.sum() / exp_),
                        interval=[float(c.sum() - half), float(c.sum() + half)], ok=bool(abs(c.sum() / exp_ - 1) <= 0.05))
R["L1_1_retest"] = dict(seeds="110-209", n_waves=int(n), checks=chk, pass_=all(v["ok"] for v in chk.values()))
print("L1.1 re-test:", {k: (round(v["ratio"], 4), v["ok"]) for k, v in chk.items()}, f"{n:,} waves", flush=True)

# ---------------------------------------------------------------- (2) the fair dice line
fair = {}
for tp in (6, 8, 10, 12, 14, 16):
    jon = lambda f, tp=tp: ws.spectrum_jonswap(f, 1 / tp, 3.3)            # noqa: E731
    cnt = np.zeros(len(KS), int)
    tot = 0
    for s in range(1000, 1050):
        eta, m0 = ws.synth(jon, N, 1 / 1.28, s)
        st, H, _ = ws.zero_up_waves(eta)
        hs = ws.window_hs(eta, 1 / 1.28, st)
        cnt += (H[:, None] > KS[None, :] * hs[:, None]).sum(0)
        tot += len(H)
    fair[str(tp)] = dict(n_waves=int(tot), counts=cnt.tolist(), p=(cnt / tot).tolist())
    i2 = int(np.argmin(np.abs(KS - 2.0)))
    print(f"fair line Tp {tp} s: {tot:,} waves, P(H > 2 Hs) = {cnt[i2] / tot:.3e} "
          f"(dice {np.exp(-8):.3e}, ratio {cnt[i2] / tot / np.exp(-8):.3f})", flush=True)
R["fair_line"] = dict(k=KS.tolist(), spectrum="JONSWAP 3.3", sampling_hz=1.28, hs="4 x std of the 30-min window",
                      seeds="1000-1049", by_tp=fair)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(R, open(os.path.join(HERE, "A2_results.json"), "w"), indent=1)
print("done ->", os.path.join(HERE, "A2_results.json"))
