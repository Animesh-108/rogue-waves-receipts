"""Wave tools for the rogue-wave runs: random linear seas at a point, zero-crossing wave heights, the dice line.

Used by Level 1 (tool checks) and Level 4 (the duel). Pre-registration: rogue_waves/PRODUCTION_LOG.md (frozen
2026-10-17, amendment A1).
"""
import numpy as np

TWO_PI = 2 * np.pi


def spectrum_gauss(f, fp, rel_width):
    """Narrow-band Gaussian spectrum (unnormalised)."""
    return np.exp(-0.5 * ((f - fp) / (rel_width * fp)) ** 2)


def spectrum_jonswap(f, fp, gamma=3.3):
    """JONSWAP shape in frequency (unnormalised)."""
    f = np.asarray(f, float)
    s = np.where(f <= fp, 0.07, 0.09)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = f ** -5.0 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** np.exp(-((f - fp) ** 2) / (2 * s ** 2 * fp ** 2))
    return np.nan_to_num(S)


def synth(S_fn, n, dt, seed, hs=1.0):
    """Gaussian sea at one point: eta_j = sum sqrt(S df) (X cos + Y sin), X, Y ~ N(0, 1).
    Returns (eta, m0_true): the spectrum is scaled so that 4 sqrt(m0_true) = hs exactly."""
    f = np.fft.rfftfreq(n, dt)
    df = f[1] - f[0]
    S = S_fn(f)
    S[0] = 0
    S *= (hs / 4) ** 2 / (S.sum() * df)
    rng = np.random.default_rng(seed)
    c = (n / 2) * np.sqrt(S * df) * (rng.standard_normal(len(f)) - 1j * rng.standard_normal(len(f)))
    return np.fft.irfft(c, n), float(S.sum() * df)


def zero_up_waves(eta):
    """Zero up-crossing waves: (start index, height H = max - min, crest = max) for each complete wave."""
    up = np.flatnonzero((eta[:-1] < 0) & (eta[1:] >= 0)) + 1
    if len(up) < 2:
        return np.array([], int), np.array([]), np.array([])
    seg = eta[up[0]:up[-1]]
    idx = up[:-1] - up[0]
    mx = np.maximum.reduceat(seg, idx)
    mn = np.minimum.reduceat(seg, idx)
    return up[:-1], mx - mn, mx


def window_hs(eta, dt, start_idx, window_s=1800.0):
    """Hs = 4 * std of the non-overlapping window that contains each wave's start (Level 3's 30-min definition)."""
    w = int(round(window_s / dt))
    nwin = len(eta) // w
    hs_w = 4 * eta[:nwin * w].reshape(nwin, w).std(axis=1)
    k = np.minimum(start_idx // w, nwin - 1)
    return hs_w[k]


def crest_trough_r(S_fn, f_max=2.0, n=200_000):
    """Tayfun crest-trough correlation r = sqrt(rc^2 + rs^2) / m0 at tau = Tbar / 2, Tbar = m0 / m1 (in f)."""
    f = np.linspace(1e-6, f_max, n)
    S = S_fn(f)
    m0 = np.trapezoid(S, f)
    m1 = np.trapezoid(f * S, f)
    tau = 0.5 * m0 / m1
    rc = np.trapezoid(S * np.cos(TWO_PI * f * tau), f)
    rs = np.trapezoid(S * np.sin(TWO_PI * f * tau), f)
    return float(np.hypot(rc, rs) / m0)


def rayleigh(k):
    return np.exp(-2 * np.asarray(k) ** 2)


def tayfun_exp(k, r):
    """Häfner et al. 2023, eq. 28: P(H/Hs > k) = exp(-4 k^2 / (1 + r))."""
    return np.exp(-4 * np.asarray(k) ** 2 / (1 + r))
