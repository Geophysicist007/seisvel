"""Quality-control measures for velocity picks.

None of these needs ground truth, so they work on real data:
  * flatness           : semblance of the NMO-corrected gather at zero moveout
  * residual_moveout   : parabolic RMO scan after NMO (should peak at 0)
  * stack_power        : energy of the stacked trace (higher = better focus)
  * dix_violations     : count of non-physical interval velocities
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .gather import Gather
from .moveout import nmo_correct
from .velocity import dix_dense


def flatness(g_nmo: Gather, window: float = 0.04) -> tuple[float, np.ndarray]:
    """Energy-weighted mean zero-moveout semblance of an NMO-corrected gather."""
    d = g_nmo.data.astype(np.float64)
    live = (d != 0).astype(float)
    n = live.sum(0)
    s1 = d.sum(0)
    s2 = (d ** 2).sum(0)
    nw = max(1, int(round(window / g_nmo.dt)) | 1)
    num = uniform_filter1d(s1 ** 2, nw, mode="constant")
    den = uniform_filter1d(n * s2, nw, mode="constant")
    sem = num / (den + 1e-30)
    w = uniform_filter1d(s2, nw, mode="constant")
    return float((sem * w).sum() / (w.sum() + 1e-30)), sem


def residual_moveout(g_nmo: Gather, max_shift: float = 0.03, n: int = 41,
                     window: float = 0.04):
    """Scan parabolic residual moveout  t = t0 + delta * (x / xmax)^2  on an
    NMO-corrected gather.  Returns (deltas, semblance[n, nt]).  A well-picked
    gather peaks at delta = 0 at every event; delta > 0 means velocity too high
    (under-corrected curves downward).
    """
    deltas = np.linspace(-max_shift, max_shift, n)
    x = np.abs(g_nmo.offsets)
    xm = max(x.max(), 1e-9)
    t = g_nmo.times
    d = g_nmo.data.astype(np.float64)
    out = np.zeros((n, g_nmo.nt))
    nw = max(1, int(round(window / g_nmo.dt)) | 1)
    for k, dl in enumerate(deltas):
        s1 = np.zeros(g_nmo.nt)
        s2 = np.zeros(g_nmo.nt)
        cnt = np.zeros(g_nmo.nt)
        for i in range(g_nmo.ntr):
            ts = t + dl * (x[i] / xm) ** 2
            a = np.interp(ts, t, d[i], left=0, right=0)
            live = a != 0
            s1 += a
            s2 += a * a
            cnt += live
        num = uniform_filter1d(s1 ** 2, nw, mode="constant")
        den = uniform_filter1d(cnt * s2, nw, mode="constant")
        out[k] = num / (den + 1e-30)
    return deltas, out


def gather_report(g: Gather, v_profile, eta_profile=0.0, stretch_max=0.5,
                  vint_bounds=(1400.0, 7000.0)) -> dict:
    gn = nmo_correct(g, v_profile, eta_profile, stretch_max=stretch_max)
    fl, _ = flatness(gn)
    stack = gn.data.sum(0) / np.maximum((gn.data != 0).sum(0), 1)
    vint = dix_dense(g.times, np.asarray(v_profile), smooth=9)
    bad = np.isnan(vint) | (vint < vint_bounds[0]) | (vint > vint_bounds[1])
    return dict(flatness=fl, stack_power=float(np.mean(stack ** 2)),
                dix_violation_fraction=float(bad[g.times > 0.1].mean()))


def velocity_error(t_axis, v_est, v_true, tmin=0.2, tmax=None, weights=None):
    """Error statistics against a known model (synthetic benchmarks only)."""
    t_axis = np.asarray(t_axis)
    m = t_axis >= tmin
    if tmax is not None:
        m &= t_axis <= tmax
    rel = (np.asarray(v_est)[m] - np.asarray(v_true)[m]) / np.asarray(v_true)[m]
    w = np.ones_like(rel) if weights is None else np.asarray(weights)[m]
    return dict(mape_pct=float(100 * np.sum(w * np.abs(rel)) / np.sum(w)),
                rmse_pct=float(100 * np.sqrt(np.sum(w * rel ** 2) / np.sum(w))),
                p95_pct=float(100 * np.percentile(np.abs(rel), 95)),
                bias_pct=float(100 * np.sum(w * rel) / np.sum(w)))


def residual_velocity_update(g: Gather, v_profile, n_iter: int = 2, max_shift: float = 0.06,
                             n: int = 61, smooth: float = 0.3, stretch_max: float = 0.5,
                             eta_profile=0.0, min_t: float = 0.0, step: float = 0.7,
                             max_change: float = 0.04):
    """Residual velocity analysis: measure parabolic residual moveout on the
    NMO-corrected gather and convert it to a velocity correction,
        1/V_new^2 = 1/V^2 + 2 t0 delta / x_ref^2,
    (delta = residual time at the reference offset, > 0 when V is too fast).
    delta(t0) is the energy-weighted peak of the RMO panel, smoothed over
    `smooth` seconds; repeated n_iter times.  Each update is damped (`step`)
    and limited to +-max_change per iteration - RVA is a *residual* method and
    the parabolic RMO model is only valid for small errors.
    Returns (v_new, delta_last)."""
    from scipy.ndimage import uniform_filter1d
    v = np.asarray(v_profile, float).copy()
    t = g.times
    xr = max(np.abs(g.offsets).max(), 1.0)
    ns = max(3, int(round(smooth / g.dt)) | 1)
    delta = np.zeros_like(t)
    for _ in range(n_iter):
        gn = nmo_correct(g, v, eta_profile, stretch_max=stretch_max)
        deltas, P = residual_moveout(gn, max_shift, n)
        ib = np.argmax(P, axis=0)
        pk = deltas[ib]
        # weight = coherence x local energy: semblance alone is amplitude-
        # normalised, so wavelet side-lobes and near-silent rows would vote with
        # full weight for spurious residual moveouts
        nw = max(3, int(round(0.04 / g.dt)) | 1)
        energy = uniform_filter1d((gn.data.astype(np.float64) ** 2).sum(0), nw, mode="nearest")
        # peak contrast: how much the best residual moveout stands out from the
        # median over all trial moveouts (noise rows have flat, low-contrast
        # panels and must not vote)
        contrast = np.clip(P.max(axis=0) - np.median(P, axis=0), 0.0, None)
        w = contrast ** 2 * energy
        w[t < min_t] = 0.0
        delta = uniform_filter1d(w * pk, ns, mode="nearest") / \
            (uniform_filter1d(w, ns, mode="nearest") + 1e-12)
        inv2 = 1.0 / v ** 2 + step * 2.0 * np.maximum(t, g.dt) * delta / xr ** 2
        vn = 1.0 / np.sqrt(np.maximum(inv2, 1.0 / 9000.0 ** 2))
        v = np.clip(vn, v * (1 - max_change), v * (1 + max_change))
    return v, delta
