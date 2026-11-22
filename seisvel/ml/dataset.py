"""Training data for the neural velocity picker.

Each example = multi-channel velocity spectrum image (time x velocity) computed
with *exactly the same* conditioning + scan code used in production, plus the
true Vrms curve.  Training on physics-based synthetics with heavy domain
randomisation (geology, acquisition, noise, multiples, AVO, wavelet, Q) is how
label scarcity is avoided; a few field labels are then used for fine-tuning.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from .. import precondition
from ..gather import Gather, offset_regularize
from ..semblance import VelocitySpectrum, velocity_scan
from ..synth.generator import Acquisition, NoiseSpec, random_model, synthesize_gather

ML_VELS = np.linspace(1400.0, 5500.0, 128)
ML_DT = 0.032                      # time step of the network grid (s)
CHANNELS = ("semblance", "ab", "semblance_rownorm")


def spectra_for_ml(g: Gather, offset_bin=50.0, band=None, window=0.04):
    gp = precondition.scan_flow(g, band=band)
    if offset_bin:
        gp = offset_regularize(gp, offset_bin)
    sem = velocity_scan(gp, ML_VELS, "semblance", window=window)
    ab = velocity_scan(gp, ML_VELS, "ab", window=window)
    return sem, ab


def to_image(sem: VelocitySpectrum, ab: VelocitySpectrum | None = None, tmax=None):
    """Spectra -> network input (C, T, V) on the ML grid (max-pooled in time,
    re-sampled in velocity if needed).  T is padded to a multiple of 16."""
    def grid(sp):
        S = np.maximum(sp.values, 0)
        if not np.array_equal(sp.velocities, ML_VELS):
            S = np.array([np.interp(ML_VELS, sp.velocities, S[:, j])
                          for j in range(S.shape[1])]).T
        k = max(1, int(round(ML_DT / sp.dt)))
        S = maximum_filter1d(S, k, axis=1, mode="nearest")[:, ::k]
        return S.T  # (T, V)
    a = grid(sem)
    b = grid(ab) if ab is not None else a
    t = sem.times[0] + ML_DT * np.arange(a.shape[0])
    if tmax is not None:
        n = int(np.searchsorted(t, tmax))
        a, b, t = a[:n], b[:n], t[:n]
    rn = a / np.maximum(a.max(axis=1, keepdims=True), 1e-6)
    x = np.stack([a / max(a.max(), 1e-6), b / max(b.max(), 1e-6), rn]).astype(np.float32)
    T = x.shape[1]
    Tp = int(np.ceil(T / 16) * 16)
    if Tp > T:
        x = np.pad(x, ((0, 0), (0, Tp - T), (0, 0)))
    return x, t, T


def make_target(vrms_dense, t_dense, t_grid, T_pad, sigma_bins=1.5):
    """Soft one-hot (Gaussian) target per time row + per-row weights."""
    v = np.interp(t_grid, t_dense, vrms_dense)
    idx = np.interp(v, ML_VELS, np.arange(len(ML_VELS)))
    j = np.arange(len(ML_VELS))[None, :]
    y = np.exp(-0.5 * ((j - idx[:, None]) / sigma_bins) ** 2)
    y /= y.sum(1, keepdims=True)
    out = np.zeros((T_pad, len(ML_VELS)), np.float32)
    out[: len(t_grid)] = y
    return out


def random_example(rng, nt=1500, dt=0.004):
    marine = rng.random() < 0.65
    model = random_model(rng, marine=marine)
    acq = Acquisition(nt=nt, dt=dt, min_offset=rng.uniform(50, 300),
                      max_offset=rng.uniform(1500, 6000), ntr=int(rng.integers(24, 100)),
                      offset_jitter=rng.uniform(0, 30),
                      missing_fraction=rng.choice([0, 0, 0.1, 0.25]))
    noise = NoiseSpec(snr=float(np.exp(rng.uniform(np.log(0.6), np.log(10)))),
                      ground_roll=0.0 if marine else rng.uniform(0, 2),
                      swell=rng.uniform(0, 1) if marine else 0.0,
                      spikes=int(rng.integers(0, 4)),
                      dead_fraction=rng.choice([0, 0.05, 0.1]),
                      flipped_fraction=rng.choice([0, 0, 0.03]),
                      amp_jitter=rng.uniform(0, 0.4))
    g, tr = synthesize_gather(model, acq, rng, fpeak=rng.uniform(12, 40),
                              q=rng.uniform(50, 300),
                              multiples=rng.uniform(0.3, 1.5) if marine else 0.0,
                              noise=noise)
    if not marine and noise.ground_roll > 0:
        g = precondition.top_mute(g, velocity=1200.0, t_intercept=0.05)
    sem, ab = spectra_for_ml(g)
    x, tg, T = to_image(sem, ab)
    y = make_target(tr["vrms"], g.times, tg, x.shape[1])
    w = np.zeros(x.shape[1], np.float32)
    first = tr["t0_primaries"][0] if len(tr["t0_primaries"]) else 0.0
    last = tr["t0_primaries"][-1] if len(tr["t0_primaries"]) else tg[-1]
    w[:T] = np.where((tg >= first - 0.05) & (tg <= last + 0.3), 1.0, 0.15)
    return x, y, w


def build_dataset(n, seed=0, path=None, verbose=True):
    rng = np.random.default_rng(seed)
    X, Y, W = [], [], []
    for i in range(n):
        x, y, w = random_example(rng)
        X.append(x); Y.append(y); W.append(w)
        if verbose and (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{n}", flush=True)
    X, Y, W = np.stack(X), np.stack(Y), np.stack(W)
    if path:
        np.savez_compressed(path, X=X, Y=Y, W=W)
    return X, Y, W
