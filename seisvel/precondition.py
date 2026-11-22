"""Pre-conditioning applied before velocity analysis.

Velocity analysis is a *coherence* measurement, so it is legitimate (and
standard practice) to apply aggressive amplitude balancing that you would NOT
use for the final, amplitude-preserving stack.  Keep the two flows separate:

    scan flow  : gain -> bandpass -> AGC -> (offset regularise) -> semblance
    stack flow : gain -> bandpass -> NMO -> mute -> stack   (no AGC if AVO matters)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from .gather import Gather


def tpow_gain(g: Gather, power: float = 2.0, t_ref: float = 1.0) -> Gather:
    """Spherical-divergence style gain t**power (power≈2 for v(z) media)."""
    t = np.maximum(g.times, g.dt) / t_ref
    return g.copy(g.data * (t ** power).astype(np.float32)[None, :])


def agc(g: Gather, window: float = 0.5, eps: float = 1e-10) -> Gather:
    """Automatic gain control: divide by running RMS (window in seconds)."""
    n = max(3, int(round(window / g.dt)) | 1)
    rms = np.sqrt(uniform_filter1d(g.data.astype(np.float64) ** 2, n, axis=1,
                                   mode="nearest"))
    ref = np.median(rms[rms > 0]) if np.any(rms > 0) else 1.0
    out = g.data / (rms + eps * ref + 1e-30)
    return g.copy(out.astype(np.float32))


def bandpass(g: Gather, fmin: float, fmax: float, order: int = 4) -> Gather:
    """Zero-phase Butterworth band-pass (fmin/fmax in Hz)."""
    nyq = 0.5 / g.dt
    lo, hi = fmin / nyq, min(fmax / nyq, 0.99)
    if lo <= 0:
        sos = butter(order, hi, btype="low", output="sos")
    else:
        sos = butter(order, [lo, hi], btype="band", output="sos")
    return g.copy(sosfiltfilt(sos, g.data, axis=1).astype(np.float32))


def trace_balance(g: Gather, eps: float = 1e-12) -> Gather:
    """Scale every trace to unit RMS (kills bad-coupling amplitude jumps)."""
    rms = np.sqrt(np.mean(g.data.astype(np.float64) ** 2, axis=1, keepdims=True))
    return g.copy((g.data / (rms + eps)).astype(np.float32))


def despike(g: Gather, k: float = 8.0) -> Gather:
    """Clip samples above k * median-absolute-amplitude (per trace)."""
    d = g.data.copy()
    mad = np.median(np.abs(d), axis=1, keepdims=True) + 1e-20
    lim = k * mad / 0.6745
    np.clip(d, -lim, lim, out=d)
    return g.copy(d)


def top_mute(g: Gather, velocity: float, t_intercept: float = 0.0,
             taper: float = 0.02) -> Gather:
    """Linear top mute t < t_intercept + |x|/velocity (removes direct/refractions)."""
    tm = t_intercept + np.abs(g.offsets)[:, None] / velocity
    t = g.times[None, :]
    w = np.clip((t - tm) / max(taper, g.dt), 0.0, 1.0)
    return g.copy((g.data * w).astype(np.float32))


def scan_flow(g: Gather, *, gain_power: float | None = 2.0,
              band: tuple[float, float] | None = None, agc_window: float | None = 0.4,
              despike_k: float | None = 10.0, drop_dead: bool = True) -> Gather:
    """Default conditioning chain for velocity scanning on real data."""
    if drop_dead:
        g = g.drop_dead()
    if despike_k:
        g = despike(g, despike_k)
    if gain_power:
        g = tpow_gain(g, gain_power)
    if band:
        g = bandpass(g, *band)
    if agc_window:
        g = agc(g, agc_window)
    return g


def detect_water_bottom(g: Gather, n_near: int = 4, tmax: float = 3.0,
                        rel: float = 0.35, water_velocity: float = 1500.0):
    """Zero-offset two-way time of the water bottom from the near traces: first
    envelope sample above `rel` x the early-record maximum, corrected for offset
    with the water velocity.  Returns None if nothing convincing is found."""
    from scipy.signal import hilbert
    o = np.argsort(np.abs(g.offsets))[:n_near]
    t = g.times
    m = t < tmax
    picks = []
    for i in o:
        env = np.abs(hilbert(g.data[i]))[m]
        env[t[m] < 0.02] = 0
        if env.max() <= 0:
            continue
        j = int(np.argmax(env > rel * env.max()))
        tx = t[m][j]
        x = abs(g.offsets[i])
        t0 = np.sqrt(max(tx ** 2 - (x / water_velocity) ** 2, 0.0))
        picks.append(t0)
    return float(np.median(picks)) if picks else None
