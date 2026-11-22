"""Velocity spectra (coherence along trial moveout trajectories).

Measures implemented
--------------------
"semblance"   Neidell & Taner (1971) / Taner & Koehler (1969):
                 S = sum_w (sum_i a_i)^2 / ( N sum_w sum_i a_i^2 )
"ab"          AB semblance (Sarkar et al. 2001, 2002; Fomel 2009).  Correlation
              with an amplitude *trend* A + B*phi_i instead of a constant, so
              AVO trends and class-II polarity reversals do not kill the peak.
              phi_i = sin^2(theta_i) ~ x^2 / (x^2 + V^2 t0^2) (straight ray).
              Price: ~2x lower resolution and ~2x higher noise floor (Fomel 2009).
"similarity"  Similarity-weighted semblance: each sample is weighted by the local
              similarity between the moved-out trace and the trial stack, in the
              spirit of Chen, Liu & Chen (2015).  Sharper peaks, better noise
              rejection.  (Weighting form is this package's implementation.)
"ab_similarity" product of AB and similarity weights - AVO-robust and sharp
              (cf. the weighted-AB idea of Ebrahimi et al. 2017).
"stack_power" un-normalised stacked energy sum_w (sum_i a_i)^2 - not bounded,
              useful as an extra ML input channel.

All measures share one compiled kernel that sweeps trial velocities in parallel
and handles: irregular / missing offsets (per-sample live count N(t)), stretch
mute inside the scan, and optional eta (non-hyperbolic) moveout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from numba import njit, prange
from scipy.ndimage import uniform_filter1d

from .gather import Gather
from .moveout import SINC_TABLE, _interp, traveltime

MEASURES = ("semblance", "ab", "similarity", "ab_similarity", "stack_power")


@dataclass
class VelocitySpectrum:
    values: np.ndarray            # (nv, nt)
    velocities: np.ndarray        # (nv,)
    times: np.ndarray             # (nt,)
    measure: str = "semblance"
    key: Hashable = None
    eta: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def dt(self):
        return float(self.times[1] - self.times[0])

    def at(self, t, v):
        """Bilinear lookup of the spectrum at (t, v) points."""
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.velocities, self.times), self.values,
                                    bounds_error=False, fill_value=0.0)
        return f(np.column_stack([np.atleast_1d(v), np.atleast_1d(t)]))

    def normalized(self, mode: str = "time") -> np.ndarray:
        """Spectrum normalised for display / ML. mode='time' -> each t row to max 1."""
        s = np.maximum(self.values, 0.0)
        if mode == "time":
            m = s.max(axis=0, keepdims=True)
            return s / np.maximum(m, 1e-12)
        return s / max(s.max(), 1e-12)


# ---------------------------------------------------------------------------
# compiled kernels
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _moments_kernel(data, offsets, tstart, dt, vels, eta, smax, mode, table):
    """Per (velocity, t0) sums over live traces:
       0:n 1:sum a 2:sum a^2 3:sum phi 4:sum phi^2 5:sum a*phi"""
    ntr, nt = data.shape
    nv = vels.shape[0]
    M = np.zeros((nv, nt, 6))
    for iv in prange(nv):
        v = vels[iv]
        for i in range(ntr):
            x = offsets[i]
            x2 = x * x
            tr = data[i]
            for it in range(nt):
                t0 = tstart + it * dt
                t = traveltime(t0, x, v, eta)
                if t < 0:
                    continue
                if t0 > 0:
                    if (t - t0) / t0 > smax:
                        continue
                elif x != 0.0:
                    continue
                f = (t - tstart) / dt
                if f < 0.0 or f > nt - 1:
                    continue
                a = _interp(tr, f, nt, mode, table)
                den = x2 + v * v * t0 * t0
                phi = x2 / den if den > 0 else 0.0
                M[iv, it, 0] += 1.0
                M[iv, it, 1] += a
                M[iv, it, 2] += a * a
                M[iv, it, 3] += phi
                M[iv, it, 4] += phi * phi
                M[iv, it, 5] += a * phi
    return M


@njit(cache=True)
def _wsum(x, nw):
    """centred moving sum along the last axis of a 1-D array (window nw, odd)."""
    n = x.shape[0]
    h = nw // 2
    c = np.zeros(n + 1)
    for i in range(n):
        c[i + 1] = c[i] + x[i]
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        out[i] = c[hi] - c[lo]
    return out


@njit(parallel=True, cache=True)
def _similarity_kernel(data, offsets, tstart, dt, vels, eta, smax, mode, table, nw,
                       use_ab, power):
    """Window-summed numerator/denominator (nv, nt) of similarity-weighted
    semblance, and WB = window-mean of the average similarity weight.  With
    use_ab the similarity reference is the AB trend model instead of the stack."""
    ntr, nt = data.shape
    nv = vels.shape[0]
    NUM = np.zeros((nv, nt))
    DEN = np.zeros((nv, nt))
    WB = np.zeros((nv, nt))
    for iv in prange(nv):
        v = vels[iv]
        P = np.zeros((ntr, nt))
        L = np.zeros((ntr, nt))
        PHI = np.zeros((ntr, nt))
        for i in range(ntr):
            x = offsets[i]
            x2 = x * x
            for it in range(nt):
                t0 = tstart + it * dt
                t = traveltime(t0, x, v, eta)
                if t < 0:
                    continue
                if t0 > 0:
                    if (t - t0) / t0 > smax:
                        continue
                elif x != 0.0:
                    continue
                f = (t - tstart) / dt
                if f < 0.0 or f > nt - 1:
                    continue
                P[i, it] = _interp(data[i], f, nt, mode, table)
                L[i, it] = 1.0
                den = x2 + v * v * t0 * t0
                PHI[i, it] = x2 / den if den > 0 else 0.0
        # reference trace: plain stack, or AB trend model evaluated per trace
        n = np.zeros(nt)
        s1 = np.zeros(nt)
        sp = np.zeros(nt)
        spp = np.zeros(nt)
        sap = np.zeros(nt)
        for i in range(ntr):
            for it in range(nt):
                if L[i, it] > 0:
                    n[it] += 1.0
                    s1[it] += P[i, it]
                    sp[it] += PHI[i, it]
                    spp[it] += PHI[i, it] * PHI[i, it]
                    sap[it] += P[i, it] * PHI[i, it]
        A = np.zeros(nt)
        B = np.zeros(nt)
        for it in range(nt):
            if n[it] < 1:
                continue
            det = n[it] * spp[it] - sp[it] * sp[it]
            if use_ab and n[it] > 2 and det > 1e-12 * n[it] * n[it]:
                B[it] = (n[it] * sap[it] - sp[it] * s1[it]) / det
                A[it] = (s1[it] - B[it] * sp[it]) / n[it]
            else:
                A[it] = s1[it] / n[it]
        # local similarity weights and weighted sums
        wsum_t = np.zeros(nt)       # sum_i w_i a_i
        wsq_t = np.zeros(nt)        # sum_i w_i a_i^2
        wmean = np.zeros(nt)        # sum_i w_i
        ref = np.zeros(nt)
        pr = np.zeros(nt)
        rr = np.zeros(nt)
        pp = np.zeros(nt)
        for i in range(ntr):
            for it in range(nt):
                ref[it] = (A[it] + B[it] * PHI[i, it]) * L[i, it]
                pr[it] = P[i, it] * ref[it]
                rr[it] = ref[it] * ref[it]
                pp[it] = P[i, it] * P[i, it]
            spr = _wsum(pr, nw)
            srr = _wsum(rr, nw)
            spp_ = _wsum(pp, nw)
            for it in range(nt):
                if L[i, it] == 0:
                    continue
                d = np.sqrt(srr[it] * spp_[it])
                c = spr[it] / d if d > 0 else 0.0
                if c < 0:
                    c = 0.0
                w = c ** power
                a = P[i, it]
                wsum_t[it] += w * a
                wsq_t[it] += w * a * a
                wmean[it] += w
        num_t = wsum_t * wsum_t
        den_t = n * wsq_t
        wb = np.zeros(nt)
        for it in range(nt):
            if n[it] > 0:
                wb[it] = wmean[it] / n[it]
        NUM[iv] = _wsum(num_t, nw)
        DEN[iv] = _wsum(den_t, nw)
        WB[iv] = _wsum(wb, nw) / nw
    return NUM, DEN, WB


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def velocity_scan(g: Gather, velocities, measure: str = "semblance",
                  window: float = 0.04, eta: float = 0.0, stretch_max: float = 1.0,
                  interp: str = "linear", similarity_power: float = 2.0,
                  ) -> VelocitySpectrum:
    """Compute a velocity spectrum for one (super)gather.

    velocities  : trial stacking velocities (m/s), any monotonic array
    window      : coherence window length in seconds (≈ 1-2 dominant periods)
    eta         : fixed anellipticity used for all trials (eta scans: loop over eta)
    stretch_max : samples with NMO stretch above this are excluded from the scan
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    vels = np.asarray(velocities, dtype=np.float64)
    nw = max(1, int(round(window / g.dt)) | 1)
    mode = 1 if interp == "sinc" else 0
    smax = float(stretch_max) if np.isfinite(stretch_max) else 1e30
    data = g.data.astype(np.float64)
    tiny = 1e-30

    if measure in ("similarity", "ab_similarity"):
        NUM, DEN, WB = _similarity_kernel(data, g.offsets, g.t0, g.dt, vels,
                                          float(eta), smax, mode, SINC_TABLE, nw,
                                          measure == "ab_similarity",
                                          similarity_power)
        if measure == "ab_similarity":
            M = _moments_kernel(data, g.offsets, g.t0, g.dt, vels, float(eta), smax,
                                mode, SINC_TABLE)
            vals = _ab_from_moments(M, nw) * np.clip(WB, 0.0, 1.0)
        else:
            vals = NUM / (DEN + tiny)
        return VelocitySpectrum(np.clip(vals, 0.0, 1.0), vels, g.times, measure, g.key,
                                eta)

    M = _moments_kernel(data, g.offsets, g.t0, g.dt, vels, float(eta), smax, mode,
                        SINC_TABLE)
    if measure == "semblance":
        num = uniform_filter1d(M[..., 1] ** 2, nw, axis=1, mode="constant")
        den = uniform_filter1d(M[..., 0] * M[..., 2], nw, axis=1, mode="constant")
        vals = num / (den + tiny)
    elif measure == "ab":
        vals = _ab_from_moments(M, nw)
    else:  # stack_power
        vals = uniform_filter1d(M[..., 1] ** 2, nw, axis=1, mode="constant")
        vals = vals / max(vals.max(), tiny)
    return VelocitySpectrum(np.clip(vals, 0.0, 1.0), vels, g.times, measure, g.key, eta)


def _ab_from_moments(M, nw):
    n, s1, s2, sp, spp, sap = (M[..., k] for k in range(6))
    det = n * spp - sp ** 2
    ok = (n > 2) & (det > 1e-12 * np.maximum(n, 1) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ep_ab = (spp * s1 ** 2 - 2 * sp * s1 * sap + n * sap ** 2) / det
        ep_c = s1 ** 2 / np.maximum(n, 1)
    ep = np.where(ok, ep_ab, ep_c)
    ep = np.nan_to_num(ep, nan=0.0, posinf=0.0)
    num = uniform_filter1d(ep, nw, axis=1, mode="constant")
    den = uniform_filter1d(s2, nw, axis=1, mode="constant")
    return num / (den + 1e-30)


def default_velocities(vmin: float = 1400.0, vmax: float = 5500.0, n: int = 160):
    return np.linspace(vmin, vmax, n)


def multi_measure_scan(g: Gather, velocities, measures=("semblance", "ab"), **kw):
    """Several spectra at once (input channels for the ML picker)."""
    return {m: velocity_scan(g, velocities, measure=m, **kw) for m in measures}


def eta_scan(g: Gather, vnmo_profile, etas, window=0.04, stretch_max=1.0,
             t_index=None):
    """Second-pass eta scan (Alkhalifah 1997 two-step): with V_nmo fixed from a
    short-spread hyperbolic pick, scan eta and return semblance(eta, t0)."""
    from .moveout import _as_profile
    vel = _as_profile(vnmo_profile, g.nt, "velocity")
    out = np.zeros((len(etas), g.nt))
    nw = max(1, int(round(window / g.dt)) | 1)
    data = g.data.astype(np.float64)
    for k, e in enumerate(etas):
        # moments along the given V(t0) path for this eta
        M = _path_moments(data, g.offsets, g.t0, g.dt, vel, float(e), stretch_max)
        num = uniform_filter1d(M[:, 1] ** 2, nw, mode="constant")
        den = uniform_filter1d(M[:, 0] * M[:, 2], nw, mode="constant")
        out[k] = num / (den + 1e-30)
    return np.clip(out, 0, 1)


@njit(cache=True)
def _path_moments(data, offsets, tstart, dt, vel, eta, smax):
    ntr, nt = data.shape
    M = np.zeros((nt, 3))
    for i in range(ntr):
        x = offsets[i]
        for it in range(nt):
            t0 = tstart + it * dt
            t = traveltime(t0, x, vel[it], eta)
            if t < 0:
                continue
            if t0 > 0 and (t - t0) / t0 > smax:
                continue
            f = (t - tstart) / dt
            if f < 0 or f > nt - 1:
                continue
            j = int(f)
            a_ = f - j
            a = data[i, j] if j >= nt - 1 else (1 - a_) * data[i, j] + a_ * data[i, j + 1]
            M[it, 0] += 1
            M[it, 1] += a
            M[it, 2] += a * a
    return M
