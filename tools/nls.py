"""Nonlinear Schrödinger solvers (split-step Fourier, Strang splitting) for the special mechanism.

1D:  i psi_t + 1/2 psi_xx + |psi|^2 psi = 0          (focusing; Peregrine breather, Benjamin-Feir instability)
2D:  i psi_t + 1/2 psi_xx - psi_yy + |psi|^2 psi = 0  (deep-water, hyperbolic)
Pre-registration: rogue_waves/PRODUCTION_LOG.md, amendment A1 (d).
"""
import numpy as np


def step_1d(psi, k, dt, nsteps, record=None):
    lin_half = np.exp(-0.5j * k ** 2 * dt / 2)
    out = []
    for i in range(nsteps):
        p = np.fft.ifft(np.fft.fft(psi) * lin_half)
        p *= np.exp(1j * np.abs(p) ** 2 * dt)
        psi = np.fft.ifft(np.fft.fft(p) * lin_half)
        if record is not None and (i + 1) % record[0] == 0:
            out.append(record[1](psi, (i + 1) * dt))
    return psi, out


def step_2d(psi, kx, ky, dt, nsteps):
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    lin_half = np.exp(1j * (-0.5 * KX ** 2 + KY ** 2) * dt / 2)
    for _ in range(nsteps):
        p = np.fft.ifft2(np.fft.fft2(psi) * lin_half)
        p *= np.exp(1j * np.abs(p) ** 2 * dt)
        psi = np.fft.ifft2(np.fft.fft2(p) * lin_half)
    return psi


def peregrine(x, t):
    return (1 - 4 * (1 + 2j * t) / (1 + 4 * x ** 2 + 4 * t ** 2)) * np.exp(1j * t)
