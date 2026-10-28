"""Level 1 tool checks (pre-registration frozen 2026-10-17 01:36 + amendment A1). Simulated seas only; no buoy data.

    py -3.13 rogue_waves/runs/L1/run_L1.py
Writes L1_results.json (+ the console log) next to this file. Pass / fail exactly as frozen.
"""
import json
import os
import platform
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import nls        # noqa: E402
import wavesim as ws  # noqa: E402

TP, N, SEEDS = 10.0, 2 ** 24, range(10)
FP = 1 / TP
KS = np.round(np.arange(1.0, 2.5001, 0.05), 2)
R = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), numpy=np.__version__,
         prereg_sha256="747b300494f2e5c2af1aa90e1dec84ab17341208376d4ef1d92a23516a2a635e", amendments=["A1"])


def count_run(S_fn, dt, label, window_diag=False):
    tot = 0
    cnt_true = np.zeros(len(KS), int)
    cnt_win = np.zeros(len(KS), int)
    for s in SEEDS:
        eta, m0 = ws.synth(S_fn, N, dt, s)
        st, H, _ = ws.zero_up_waves(eta)
        hs_true = 4 * np.sqrt(m0)
        tot += len(H)
        cnt_true += (H[:, None] > KS[None, :] * hs_true).sum(0)
        if window_diag:
            hs_w = ws.window_hs(eta, dt, st)
            cnt_win += (H[:, None] > KS[None, :] * hs_w[:, None]).sum(0)
        print(f"  {label} seed {s}: {len(H):,} waves", flush=True)
    out = dict(label=label, dt=dt, n_waves=int(tot), k=KS.tolist(), count_true_hs=cnt_true.tolist(),
               p_true_hs=(cnt_true / tot).tolist())
    if window_diag:
        out.update(count_window_hs=cnt_win.tolist(), p_window_hs=(cnt_win / tot).tolist())
    return out


def at(o, key, k):
    return o[key][int(np.argmin(np.abs(KS - k)))]


# ---------------------------------------------------------------- L1.1 the dice line (narrow-band)
print("L1.1 narrow-band Gaussian, dt = Tp/40")
narrow = lambda f: ws.spectrum_gauss(f, FP, 0.02)                       # noqa: E731
o = count_run(narrow, TP / 40, "L1.1", window_diag=True)
n = o["n_waves"]
r10 = at(o, "p_true_hs", 1.0) / ws.rayleigh(1.0)
r15 = at(o, "p_true_hs", 1.5) / ws.rayleigh(1.5)
lam = n * np.exp(-8)
lo_, hi_ = stats.poisson.ppf(0.025, lam), stats.poisson.ppf(0.975, lam)
c20 = at(o, "count_true_hs", 2.0)
o["checks"] = dict(
    enough_waves=dict(n=n, need=3_000_000, ok=n >= 3_000_000),
    k1_0=dict(ratio=r10, tol=0.05, ok=abs(r10 - 1) <= 0.05),
    k1_5=dict(ratio=r15, tol=0.05, ok=abs(r15 - 1) <= 0.05),
    k2_0=dict(count=c20, expected=lam, interval=[lo_, hi_], ok=bool(lo_ <= c20 <= hi_)))
o["pass"] = all(v["ok"] for v in o["checks"].values())
R["L1_1"] = o
print("  ->", {k: (round(v.get("ratio", v.get("count", v.get("n", 0))), 4), v["ok"]) for k, v in o["checks"].items()})

print("L1.1 diagnostic: the same sea sampled at 1.28 Hz (CDIP's rate)")
d = count_run(narrow, 1 / 1.28, "L1.1-diag-1.28Hz")
R["L1_1_diag_128Hz"] = d

# ---------------------------------------------------------------- L1.2 broad sea (JONSWAP 3.3)
print("L1.2 JONSWAP gamma 3.3, dt = Tp/40")
jon = lambda f: ws.spectrum_jonswap(f, FP, 3.3)                         # noqa: E731
r = ws.crest_trough_r(jon)
o2 = count_run(jon, TP / 40, "L1.2", window_diag=True)
rt15 = at(o2, "p_true_hs", 1.5) / float(ws.tayfun_exp(1.5, r))
o2["r"] = r
o2["checks"] = dict(k1_5=dict(ratio=rt15, tol=0.10, ok=abs(rt15 - 1) <= 0.10))
o2["pass"] = o2["checks"]["k1_5"]["ok"]
R["L1_2"] = o2
print(f"  -> r = {r:.3f}, P/PT at 1.5 = {rt15:.4f}, ok {o2['pass']}")

# ---------------------------------------------------------------- L1.3 the special mechanism's solver
print("L1.3a Peregrine breather")
L, M = 200.0, 4096
x = (np.arange(M) - M / 2) * (L / M)
k = 2 * np.pi * np.fft.fftfreq(M, L / M)
psi = nls.peregrine(x, -5.0)
peaks = []
psi, rec = nls.step_1d(psi, k, 1e-3, 6000, record=(10, lambda p, t: (t - 5.0, float(np.abs(p).max()))))
tpk, apk = max(rec, key=lambda z: z[1])
per = dict(peak=apk, at_t=tpk, expected=3.0, tol=0.01, ok=abs(apk / 3 - 1) <= 0.01)
print("  ->", per)

print("L1.3b Benjamin-Feir growth at K = sqrt(2)")
K = np.sqrt(2)
L2, M2 = 8 * 2 * np.pi / K, 512
x2 = np.arange(M2) * (L2 / M2)
k2 = 2 * np.pi * np.fft.fftfreq(M2, L2 / M2)
psi = (1 + 1e-6 * np.cos(K * x2)).astype(complex)
mode = int(round(K / (2 * np.pi / L2)))
psi, rec = nls.step_1d(psi, k2, 1e-3, 8000, record=(50, lambda p, t: (t, float(np.abs(np.fft.fft(p)[mode]) / M2))))
tt = np.array([z[0] for z in rec])
aa = np.log(np.array([z[1] for z in rec]))
sel = (tt >= 2) & (tt <= 8)
sigma = float(np.polyfit(tt[sel], aa[sel], 1)[0])
bf = dict(growth_rate=sigma, expected=1.0, tol=0.05, ok=abs(sigma - 1) <= 0.05)
print("  ->", bf)

print("L1.3c 2D NLS energy conservation")
rng = np.random.default_rng(0)
M3 = 256
L3 = 64 * np.pi
kx = 2 * np.pi * np.fft.fftfreq(M3, L3 / M3)
psi = (1 + 0.1 * (rng.standard_normal((M3, M3)) + 1j * rng.standard_normal((M3, M3)))).astype(complex)
e0 = float((np.abs(psi) ** 2).sum())
psi = nls.step_2d(psi, kx, kx, 2e-3, 5000)
e1 = float((np.abs(psi) ** 2).sum())
en = dict(rel_change=abs(e1 / e0 - 1), tol=1e-3, ok=abs(e1 / e0 - 1) <= 1e-3)
print("  ->", en)
R["L1_3"] = dict(peregrine=per, benjamin_feir=bf, energy_2d=en, pass_=per["ok"] and bf["ok"] and en["ok"])

R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
R["all_pass"] = bool(R["L1_1"]["pass"] and R["L1_2"]["pass"] and R["L1_3"]["pass_"])
json.dump(R, open(os.path.join(HERE, "L1_results.json"), "w"), indent=1, default=float)
print("ALL PASS" if R["all_pass"] else "NOT ALL PASS", "->", os.path.join(HERE, "L1_results.json"))
