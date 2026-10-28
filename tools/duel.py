"""Level 4, the duel (pre-registration + A4, PRODUCTION_LOG.md): one real sea state run by two rules.

    py -3.13 rogue_waves/tools/duel.py check                      # A4 (d): the solver check, before any duel run
    py -3.13 rogue_waves/tools/duel.py run SEA SEED ARM [ARM ...]  # SEA = index in runs/L4/seas.json; ARM in A B C C0

Deep-water NLS, lab frame, periodic, in metres and seconds:
    i(A_t + cg A_x) - (w0 / 8k0^2) A_xx + (w0 / 4k0^2) A_yy - (w0 k0^2 / 2)|A|^2 A = 0,   eta = Re[A e^{i(k0 x - w0 t)}]
split-step Fourier (Strang), dt = Tp/40, grid lambda/8 (A4 (g): lambda/4 failed the Peregrine check). A = the same equation without the |A|^2 A term (exact in one
step); B = 2D NLS; C = 1D NLS (all energy in one direction); C0 = 1D linear (diagnostic). Virtual buoys one per
wavelength record the last 20 peak periods every Tp/20; zero-upcrossing heights; AI = H / the real sea state's Hs.
Every run appends one line to runs/L4/runs.jsonl (nothing overwritten).
"""
import json
import math
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

G = 9.81
EP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
L4 = os.path.join(EP, "runs", "L4")
KS = np.round(np.arange(1.0, 2.5001, 0.05), 2)
N_LAM_2D, N_LAM_1D, PER_LAM, T_SPIN, T_REC = 64, 4096, 8, 30, 20       # grid lambda/8 (A4 (g))


# ------------------------------------------------------------------ spectrum
def jonswap(f, fp, gamma):
    s = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * s ** 2 * fp ** 2))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = f ** -5.0 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    return np.nan_to_num(S)


def goda_qp(fp, gamma):
    f = np.linspace(0.02 * fp, 12 * fp, 200000)
    S = jonswap(f, fp, gamma)
    m0 = np.trapezoid(S, f)
    return 2 * np.trapezoid(f * S ** 2, f) / m0 ** 2


def gamma_from_qp(fp, qp):
    lo, hi = 1.0, 20.0
    if qp <= goda_qp(fp, lo):
        return lo
    if qp >= goda_qp(fp, hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if goda_qp(fp, mid) < qp else (lo, mid)
    return 0.5 * (lo + hi)


def sea_params(sea):
    tp = sea["tp"]
    fp = 1 / tp
    w0 = 2 * math.pi / tp
    k0 = w0 ** 2 / G
    qp = 1 / (math.sqrt(math.pi) * sea["peakedness_goda"])              # FOWD stores sigma_Q = 1/(sqrt(pi) Qp) (A4 (h))
    gamma = gamma_from_qp(fp, qp)
    sig = math.radians(sea["spread_deg_at_peak"])
    s = max(1.0, 2 / sig ** 2 - 1)
    return dict(tp=tp, fp=fp, w0=w0, k0=k0, lam=2 * math.pi / k0, cg=w0 / (2 * k0), gamma=gamma, s=s, hs=sea["hs"], qp=qp)


def grid(p, dim):
    n_lam = N_LAM_2D if dim == 2 else N_LAM_1D
    L = n_lam * p["lam"]
    n = n_lam * PER_LAM
    kx = 2 * np.pi * np.fft.fftfreq(n, L / n)
    if dim == 1:
        return L, n, kx, None
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    return L, n, KX, KY


def start_sea(p, dim, seed):
    """Random complex Gaussian amplitudes on the NLS band, scaled so 4 sqrt(m0) = Hs (A4 (c))."""
    L, n, KX, KY = grid(p, dim)
    k0 = p["k0"]
    dk = 2 * np.pi / L
    if dim == 2:
        kxa, kya = k0 + KX, KY
        k = np.hypot(kxa, kya)
        th = np.arctan2(kya, kxa)
        keep = (kxa > 0) & (np.hypot(KX, KY) < k0)
        D = np.cos(th / 2) ** (2 * p["s"])
        D /= np.trapezoid(np.cos(np.linspace(-np.pi, np.pi, 20001) / 2) ** (2 * p["s"]), np.linspace(-np.pi, np.pi, 20001))
        cell = dk * dk
    else:
        k = k0 + KX
        keep = (k > 0) & (np.abs(KX) < k0)
        D = 1.0
        cell = dk
    k = np.where(keep, k, k0)
    w = np.sqrt(G * k)
    f = w / (2 * np.pi)
    dfdk = (w / (2 * k)) / (2 * np.pi)
    F = jonswap(f, p["fp"], p["gamma"]) * dfdk * D / (k if dim == 2 else 1.0)
    F = np.where(keep, F, 0.0)
    ff = np.linspace(0.02 * p["fp"], 12 * p["fp"], 200000)
    full_m0 = np.trapezoid(jonswap(ff, p["fp"], p["gamma"]), ff)
    band_share = float(F.sum() * cell / full_m0)                           # energy inside the NLS band (reported)
    F *= (p["hs"] / 4) ** 2 / (F.sum() * cell)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(F.shape) + 1j * rng.standard_normal(F.shape)
    C = np.sqrt(F * cell) * z
    return C * C.size, band_share, (L, n, KX, KY)                         # numpy ifft convention


def omega_lin(p, KX, KY):
    w0, k0, cg = p["w0"], p["k0"], p["cg"]
    O = cg * KX - w0 / (8 * k0 ** 2) * KX ** 2
    if KY is not None:
        O = O + w0 / (4 * k0 ** 2) * KY ** 2
    return O


def count_waves(eta, hs):
    """eta: (time, buoys). Zero-upcrossing waves per buoy; counts of H / Hs > k."""
    s = np.signbit(eta)
    up = s[:-1] & ~s[1:]                                                  # crossing from below to above zero
    heights = []
    for b in range(eta.shape[1]):
        idx = np.nonzero(up[:, b])[0] + 1
        if len(idx) < 2:
            continue
        seg = np.split(eta[idx[0]:idx[-1], b], idx[1:-1] - idx[0])
        heights.append(np.array([x.max() - x.min() for x in seg]))
    H = np.concatenate(heights) / hs
    srt = np.sort(H)
    return dict(n_waves=int(len(H)), over=[int(x) for x in len(srt) - np.searchsorted(srt, KS, side="right")],
                max_ai=round(float(srt[-1]), 4))


def run(sea, seed, arm):
    p = sea_params(sea)
    dim = 2 if arm in ("A", "B") else 1
    C, band_share, (L, n, KX, KY) = start_sea(p, dim, seed)
    O = omega_lin(p, KX, KY)
    dt = p["tp"] / 40
    nl = arm in ("B", "C")
    q = p["w0"] * p["k0"] ** 2 / 2
    n_steps = (T_SPIN + T_REC) * 40
    rec_from = T_SPIN * 40
    x = np.arange(n) * (L / n)
    bi = np.arange(0, n, PER_LAM)                                         # one buoy per wavelength
    carrier_x = np.exp(1j * p["k0"] * x[bi])
    if dim == 2:                                                          # scipy.fft, 4 threads: same numbers as numpy
        ifft, fft = (lambda z: sfft.ifft2(z, workers=4)), (lambda z: sfft.fft2(z, workers=4))  # (max diff 0.0), 2.9x faster
    else:
        ifft, fft = (lambda z: sfft.ifft(z, workers=4)), (lambda z: sfft.fft(z, workers=4))
    half = np.exp(-1j * O * dt / 2)
    full = np.exp(-1j * O * dt)
    rec = []
    t0 = time.time()
    Chat = C
    m_start = float(np.sum(np.abs(Chat) ** 2))
    for step in range(n_steps + 1):
        if step >= rec_from and (step - rec_from) % 2 == 0:
            if nl:
                A = ifft(Chat)
            else:
                A = ifft(C * np.exp(-1j * O * step * dt))                   # exact linear evolution
            t = step * dt
            Ab = A[np.ix_(bi, bi)] if dim == 2 else A[bi]
            cx = carrier_x[None, :] if dim == 2 else carrier_x
            eta = np.real(Ab * cx * np.exp(-1j * p["w0"] * t))
            rec.append(eta.ravel().astype("f4"))
        if step == n_steps:
            break
        if nl:                                                             # Strang: half linear, nonlinear, half linear
            A = ifft(Chat * half)
            A *= np.exp(-1j * q * np.abs(A) ** 2 * dt)
            Chat = fft(A) * half
    m_end = float(np.sum(np.abs(Chat) ** 2)) if nl else m_start
    res = count_waves(np.array(rec), sea["hs"])
    out = dict(sea=sea["station"], seed=seed, arm=arm, dim=dim, k=KS.tolist(), **res,
               energy_drift=m_end / m_start - 1, band_share=round(band_share, 4), gamma=round(p["gamma"], 3),
               s=round(p["s"], 2), steepness_kp_hs_2=round(p["k0"] * p["hs"] / 2, 4), grid=n, length_m=round(L, 1),
               seconds=round(time.time() - t0, 1), numpy=np.__version__, when=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(L4, exist_ok=True)
    with open(os.path.join(L4, "runs.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


def check():
    """A4 (d): the dimensional 1D solver reproduces the Peregrine peak (3.00 +- 1 %) and keeps sum |A|^2 (0.1 %)."""
    tp = 12.0
    w0 = 2 * math.pi / tp
    k0 = w0 ** 2 / G
    lam = 2 * math.pi / k0
    ka = 0.15
    a0 = ka / k0
    q, pd = w0 * k0 ** 2 / 2, w0 / (8 * k0 ** 2)
    tau = 1 / (q * a0 ** 2)
    Lx = math.sqrt(2 * pd / q) / a0                                       # X = x / Lx
    out = {}
    for name, per_lam in (("fine lambda/16", 16), ("lambda/4 (A4 as first written)", 4), ("duel grid lambda/8", 8)):
        L = 400 * lam
        n = int(400 * per_lam)
        x = (np.arange(n) - n / 2) * (L / n)
        X, T0 = x / Lx, -2.0
        psi = (1 - 4 * (1 + 2j * T0) / (1 + 4 * X ** 2 + 4 * T0 ** 2)) * np.exp(1j * T0)
        A = a0 * np.conj(psi)                                              # A* solves the standard focusing NLS
        kx = 2 * np.pi * np.fft.fftfreq(n, L / n)
        O = (w0 / (2 * k0)) * kx - pd * kx ** 2
        dt = tp / 40
        n_steps = int(round(2 * tau / dt))
        half = np.exp(-1j * O * dt / 2)
        Ch = np.fft.fft(A)
        m0 = np.sum(np.abs(Ch) ** 2)
        peak = 0.0
        for _ in range(n_steps + 200):
            B = np.fft.ifft(Ch * half)
            B *= np.exp(-1j * q * np.abs(B) ** 2 * dt)
            Ch = np.fft.fft(B) * half
            peak = max(peak, float(np.abs(B).max() / a0))
        drift = float(np.sum(np.abs(Ch) ** 2) / m0 - 1)
        out[name] = dict(peak=round(peak, 5), ok_peak=abs(peak / 3 - 1) <= 0.01, energy_drift=drift,
                         ok_energy=abs(drift) <= 1e-3, tau_in_tp=round(tau / tp, 2))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "check":
        r = check()
        os.makedirs(L4, exist_ok=True)
        json.dump(dict(r, when=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(L4, "solver_check.json"), "w"), indent=1)
        print(json.dumps(r, indent=1))
    elif sys.argv[1] == "run":
        seas = json.load(open(os.path.join(L4, "seas.json")))
        sea, seed = seas[int(sys.argv[2])], int(sys.argv[3])
        for arm in sys.argv[4:]:
            r = run(sea, seed, arm)
            i2 = r["k"].index(2.0)
            print(f"{r['sea']} seed {seed} {arm}: {r['n_waves']:,} waves, over 2x {r['over'][i2]}, max {r['max_ai']}, "
                  f"energy drift {r['energy_drift']:.1e}, band {r['band_share']}, {r['seconds']} s", flush=True)
