"""Moveout equations and NMO correction.

Traveltime models
-----------------
hyperbolic  :  t^2 = t0^2 + x^2 / V^2                         (Dix / Taner-Koehler)
eta (VTI)   :  t^2 = t0^2 + x^2/V^2
                     - 2 eta x^4 / ( V^2 [ t0^2 V^2 + (1 + 2 eta) x^2 ] )
               (Alkhalifah & Tsvankin, 1995) - also an effective description of
               non-hyperbolic moveout caused by vertical heterogeneity in
               isotropic layered media on long spreads.

NMO stretch
-----------
The NMO mapping t -> t0 stretches a wavelet by the factor dt0/dt ( >= 1 ).
We compute dt/dt0 numerically along each trace so the mute is correct for
any velocity function and either moveout law.  Samples whose stretch exceeds
`stretch_max` (fraction; 0.5 == 50 %) are muted with a short taper, and
samples where the mapping folds over (dt/dt0 <= 0, i.e. crossing events with
time-varying velocity) are always muted.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .gather import Gather

# ---------------------------------------------------------------------------
# windowed-sinc interpolation table (8-point Kaiser, 1/128-sample resolution)
# ---------------------------------------------------------------------------
_NSINC = 8
_NTAB = 128


def _make_sinc_table():
    half = _NSINC // 2
    tab = np.zeros((_NTAB + 1, _NSINC), dtype=np.float64)
    beta = 6.0
    for k in range(_NTAB + 1):
        frac = k / _NTAB
        x = np.arange(-half + 1, half + 1) - frac   # sample offsets
        w = np.kaiser(2 * half + 1, beta)  # symmetric window of odd length
        # evaluate window continuously by interpolation
        xw = np.linspace(-half, half, 2 * half + 1)
        win = np.interp(x, xw, w)
        s = np.sinc(x) * win
        tab[k] = s / s.sum()                 # unit DC gain
    return tab


SINC_TABLE = _make_sinc_table()


@njit(cache=True, inline="always")
def _interp(trace, f, nt, mode, table):
    """Sample `trace` at fractional index f. mode 0=linear, 1=sinc8."""
    if f < 0.0 or f > nt - 1:
        return 0.0
    i = int(f)
    a = f - i
    if mode == 0:
        if i >= nt - 1:
            return trace[nt - 1]
        return (1.0 - a) * trace[i] + a * trace[i + 1]
    k = int(a * 128.0 + 0.5)
    acc = 0.0
    for j in range(8):
        idx = i - 3 + j
        if idx < 0:
            idx = 0
        elif idx > nt - 1:
            idx = nt - 1
        acc += table[k, j] * trace[idx]
    return acc


@njit(cache=True, inline="always")
def traveltime(t0, x, v, eta):
    x2 = x * x
    v2 = v * v
    t2 = t0 * t0 + x2 / v2
    if eta != 0.0:
        t2 -= 2.0 * eta * x2 * x2 / (v2 * (t0 * t0 * v2 + (1.0 + 2.0 * eta) * x2))
    if t2 < 0.0:
        return -1.0
    return np.sqrt(t2)


def traveltime_np(t0, x, v, eta=0.0):
    t0 = np.asarray(t0, float)
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    x2, v2 = x * x, v * v
    t2 = t0 ** 2 + x2 / v2
    if np.any(eta):
        t2 = t2 - 2 * eta * x2 ** 2 / (v2 * (t0 ** 2 * v2 + (1 + 2 * eta) * x2))
    return np.sqrt(np.maximum(t2, 0.0))


# ---------------------------------------------------------------------------
# NMO correction
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _nmo_kernel(data, offsets, tstart, dt, vel, eta, stretch_max, taper_n, mode,
                table, inverse):
    ntr, nt = data.shape
    out = np.zeros_like(data)
    for i in prange(ntr):
        x = offsets[i]
        tr = data[i]
        if not inverse:
            # forward: out(t0) = in(t(t0))
            tprev = -1.0
            for it in range(nt):
                t0 = tstart + it * dt
                t = traveltime(t0, x, vel[it], eta[it])
                if t < 0:
                    continue
                f = (t - tstart) / dt
                out[i, it] = _interp(tr, f, nt, mode, table)
            # stretch mute: walk from the bottom up; once stretch exceeds the
            # limit everything shallower is muted (standard practice, avoids
            # "islands" of live data above the mute)
            muted_from = -1
            for it in range(nt - 1, 0, -1):
                t0a = tstart + (it - 1) * dt
                t0b = tstart + it * dt
                ta = traveltime(t0a, x, vel[it - 1], eta[it - 1])
                tb = traveltime(t0b, x, vel[it], eta[it])
                dtdt0 = (tb - ta) / dt
                if dtdt0 <= 1e-6:
                    muted_from = it
                    break
                stretch = 1.0 / dtdt0 - 1.0
                if stretch > stretch_max:
                    muted_from = it
                    break
            if muted_from >= 0:
                for it in range(0, muted_from + 1):
                    out[i, it] = 0.0
                for k in range(taper_n):
                    it = muted_from + 1 + k
                    if it < nt:
                        out[i, it] *= (k + 1.0) / (taper_n + 1.0)
        else:
            # inverse NMO: out(t) = in(t0(t)), t0(t) found by monotone search
            it0 = 0
            for it in range(nt):
                t = tstart + it * dt
                while it0 < nt - 1:
                    tt = traveltime(tstart + (it0 + 1) * dt, x, vel[it0 + 1], eta[it0 + 1])
                    if tt >= t:
                        break
                    it0 += 1
                ta = traveltime(tstart + it0 * dt, x, vel[it0], eta[it0])
                if it0 < nt - 1:
                    tb = traveltime(tstart + (it0 + 1) * dt, x, vel[it0 + 1], eta[it0 + 1])
                else:
                    tb = ta + dt
                if t < ta:
                    continue
                if tb - ta <= 0:
                    continue
                f = it0 + (t - ta) / (tb - ta)
                out[i, it] = _interp(tr, f, nt, mode, table)
    return out


def _as_profile(values, nt, name):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 0:
        return np.full(nt, float(a))
    if a.shape != (nt,):
        raise ValueError(f"{name} must be scalar or length nt={nt}")
    return a


def nmo_correct(g: Gather, velocity, eta=0.0, stretch_max: float = 0.5,
                interp: str = "sinc", taper: float = 0.02) -> Gather:
    """Forward NMO correction with stretch mute.

    velocity : scalar or array (nt,) of stacking (NMO) velocity at each t0 sample
    eta      : scalar or array (nt,) anellipticity (0 -> hyperbolic)
    stretch_max : maximum allowed stretch (fraction). Use np.inf to disable.
    """
    vel = _as_profile(velocity, g.nt, "velocity")
    et = _as_profile(eta, g.nt, "eta")
    mode = 1 if interp == "sinc" else 0
    taper_n = max(0, int(round(taper / g.dt)))
    smax = float(stretch_max) if np.isfinite(stretch_max) else 1e30
    out = _nmo_kernel(g.data, g.offsets, g.t0, g.dt, vel, et, smax, taper_n, mode,
                      SINC_TABLE, False)
    return g.copy(out)


def inverse_nmo(g: Gather, velocity, eta=0.0, interp: str = "sinc") -> Gather:
    """Undo NMO (t0 -> t). Used for residual-velocity loops and for modelling."""
    vel = _as_profile(velocity, g.nt, "velocity")
    et = _as_profile(eta, g.nt, "eta")
    mode = 1 if interp == "sinc" else 0
    out = _nmo_kernel(g.data, g.offsets, g.t0, g.dt, vel, et, 1e30, 0, mode,
                      SINC_TABLE, True)
    return g.copy(out)


def stretch_mute_times(offsets, times, velocity, eta=0.0, stretch_max=0.5):
    """Return, per offset, the t0 above which data survive the stretch mute
    (for drawing the mute line on displays)."""
    vel = _as_profile(velocity, len(times), "velocity")
    et = _as_profile(eta, len(times), "eta")
    dt = times[1] - times[0]
    res = np.full(len(offsets), times[0])
    for i, x in enumerate(offsets):
        t = traveltime_np(times, x, vel, et)
        dtdt0 = np.gradient(t, dt)
        bad = (dtdt0 <= 1e-6) | (1.0 / np.maximum(dtdt0, 1e-6) - 1.0 > stretch_max)
        idx = np.flatnonzero(bad)
        if idx.size:
            res[i] = times[min(idx.max() + 1, len(times) - 1)]
    return res
