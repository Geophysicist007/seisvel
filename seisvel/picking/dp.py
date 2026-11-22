"""Physics-constrained automatic velocity picking by dynamic programming.

The picker finds the single best trajectory v(t) through a velocity spectrum
(or through ANY score map - e.g. the probability map of the neural picker),
maximising accumulated score subject to:

  * hard Dix constraint: the interval velocity implied by every step,
        Vint^2 = (v_j^2 t_{k+1} - v_i^2 t_k) / (t_{k+1} - t_k),
    must lie inside [vint_min, vint_max]  -> no imaginary / absurd Vint
  * smoothness penalty on dv between steps
  * optional guide function with an ASYMMETRIC penalty: going slower than the
    guide costs more than going faster -> rejects multiples, which stack at
    lower velocity than primaries at the same time
  * optional hard corridor vmin(t) <= v <= vmax(t)

Classic idea (e.g. SEP work by Toldi 1985 and constrained-optimisation pickers,
Zhang 1991) implemented as an exact global optimum by Viterbi decoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

from ..semblance import VelocitySpectrum
from ..velocity import VelocityFunction


@dataclass
class DPParams:
    t_step: float = 0.02          # minimum decoding time step (s)
    rel_step: float = 0.0         # >0: node spacing grows as max(t_step, rel_step*t):
                                  # keeps the Dix constraint resolvable on a
                                  # discrete velocity axis at late times
    tmin: float | None = None     # first decoded time (s)
    tmax: float | None = None
    vint_min: object = 1400.0     # scalar, array on spectrum times, or callable(t)
    vint_max: object = 6500.0
    smooth: float = 0.5           # weight of (dv / dv_ref)^2 penalty
    dv_ref: float = 100.0         # m/s per step at which penalty == smooth
    power: float = 2.0            # score = S^power (sharpens peaks)
    guide_weight: float = 0.0     # weight of guide penalty (0 = off)
    guide_sigma: float = 0.10     # relative width of guide corridor
    guide_asym: float = 3.0       # extra factor for v < guide (multiple rejection)
    row_norm_mix: float = 0.0     # mix of globally-normalised score (0..1)
    dv_fine: float = 5.0          # internal velocity resolution of the decoder
                                  # (m/s); fine enough that the Dix constraint
                                  # is resolvable at 20 ms steps


@dataclass
class PickResult:
    times: np.ndarray             # dense time axis (spectrum times)
    velocity: np.ndarray          # dense picked trajectory
    function: VelocityFunction    # discrete picks (what a human would pick)
    path_score: np.ndarray        # score along path (dense)
    coarse_t: np.ndarray = None
    coarse_v: np.ndarray = None


@njit(cache=True)
def _viterbi(score, tt, vels, vmin2_k, vmax2_k, smooth, dv_ref, gpen, t_step):
    nk, nv = score.shape
    cost = np.full((nk, nv), -1e30)
    back = np.full((nk, nv), -1, np.int64)
    v2 = vels * vels
    dvg = 0.0                      # velocity-grid spacing -> quantisation tolerance
    for j in range(1, nv):
        if vels[j] - vels[j - 1] > dvg:
            dvg = vels[j] - vels[j - 1]
    for j in range(nv):
        cost[0, j] = score[0, j] - gpen[0, j]
    for k in range(1, nk):
        t0 = tt[k - 1]
        t1 = tt[k]
        d = t1 - t0
        vint_min2 = vmin2_k[k]
        vint_max2 = vmax2_k[k]
        sm_k = smooth * t_step / d        # slope-based smoothness (step-size free)
        for j in range(nv):
            best = -1e30
            bi = -1
            # allowed v_i^2 range from the Dix constraint
            if t0 > 1e-9:
                # exact Dix bounds (the refined velocity axis makes them
                # resolvable; a tiny tolerance absorbs float round-off)
                tol = 1e-6 * v2[j]
                lo2 = (v2[j] * t1 - vint_max2 * d) / t0 - tol
                hi2 = (v2[j] * t1 - vint_min2 * d) / t0 + tol
            else:
                lo2 = -1.0
                hi2 = 1e30
                vij = v2[j]
                if vij < vint_min2 or vij > vint_max2:
                    continue
            ilo = np.searchsorted(v2, lo2)
            ihi = np.searchsorted(v2, hi2, side="right")
            for i in range(ilo, ihi):
                c = cost[k - 1, i]
                if c <= -1e29:
                    continue
                dv = (vels[j] - vels[i]) / dv_ref
                c -= sm_k * dv * dv
                if c > best:
                    best = c
                    bi = i
            if bi >= 0:
                cost[k, j] = best + score[k, j] - gpen[k, j]
                back[k, j] = bi
    # backtrack
    path = np.zeros(nk, np.int64)
    jb = 0
    bc = -1e30
    for j in range(nv):
        if cost[nk - 1, j] > bc:
            bc = cost[nk - 1, j]
            jb = j
    path[nk - 1] = jb
    for k in range(nk - 1, 0, -1):
        jb = back[k, jb]
        if jb < 0:
            jb = 0
        path[k - 1] = jb
    return path, bc


def decoding_times(tmin, tmax, t_step, rel_step):
    tt = [tmin]
    while True:
        nxt = tt[-1] + max(t_step, rel_step * tt[-1])
        if nxt > tmax + 1e-9:
            break
        tt.append(nxt)
    if tmax - tt[-1] > 0.25 * t_step:
        tt.append(tmax)
    return np.asarray(tt)


def _coarse_score(spec: VelocitySpectrum, tt, power, row_norm_mix):
    """Max-pool the spectrum onto the (non-uniform) decoding nodes (keeps peaks)."""
    S = np.maximum(spec.values, 0.0)
    edges = np.concatenate([[tt[0] - 0.5 * (tt[1] - tt[0]) if len(tt) > 1 else tt[0]],
                            0.5 * (tt[1:] + tt[:-1]),
                            [tt[-1] + 0.5 * (tt[-1] - tt[-2]) if len(tt) > 1 else tt[0]]])
    ie = np.clip(np.round((edges - spec.times[0]) / spec.dt).astype(int), 0, S.shape[1] - 1)
    Sm = np.empty((S.shape[0], len(tt)))
    for k in range(len(tt)):
        a, b = ie[k], max(ie[k + 1], ie[k] + 1)
        Sm[:, k] = S[:, a:b].max(axis=1)
    C = Sm.T ** power                        # (nk, nv)
    # Robust normalisation: every time row is scaled so its best peak equals its
    # "eventness" e(t) in [0, 1] (row peak relative to the 90th percentile of row
    # peaks).  A single very strong shallow event (water bottom, top-of-record
    # junk) therefore cannot dwarf the weaker deep primaries, while rows with no
    # coherent energy contribute little and the constraints bridge them.
    rm = C.max(axis=1, keepdims=True)
    ref = max(np.percentile(rm, 90), 1e-12)
    e = np.clip(rm / ref, 0.0, 1.0)
    rows = C / np.maximum(rm, 1e-12) * e
    glob = C / max(C.max(), 1e-12)
    return (1 - row_norm_mix) * rows + row_norm_mix * glob


def dp_pick(spec: VelocitySpectrum, params: DPParams | None = None, guide=None,
            vmin_corridor=None, vmax_corridor=None, score=None,
            extract: dict | None = None) -> PickResult:
    """Pick the optimal constrained trajectory on a velocity spectrum.

    guide          : array on spec.times (or None) - prior velocity trend
    vmin/vmax_corridor : arrays on spec.times; outside is forbidden
    score          : optional externally supplied score map (nv, nt) replacing
                     the spectrum (e.g. neural-network probabilities)
    """
    p = params or DPParams()
    t = spec.times
    tmin = p.tmin if p.tmin is not None else max(t[0], 0.05)
    tmax = p.tmax if p.tmax is not None else t[-1]
    tt = decoding_times(tmin, tmax, p.t_step, p.rel_step)
    vels0 = spec.velocities.astype(np.float64)
    src = spec if score is None else VelocitySpectrum(score, vels0, t, "score", spec.key)
    S0 = _coarse_score(src, tt, p.power, p.row_norm_mix)
    # refine the velocity axis for decoding
    nfine = int(np.ceil((vels0[-1] - vels0[0]) / p.dv_fine)) + 1
    vels = np.linspace(vels0[0], vels0[-1], max(nfine, len(vels0)))
    S = np.array([np.interp(vels, vels0, row) for row in S0])
    nk, nv = S.shape
    gpen = np.zeros((nk, nv))
    if guide is not None and p.guide_weight > 0:
        gv = np.interp(tt, t, guide)[:, None]
        r = (vels[None, :] - gv) / (p.guide_sigma * gv)
        r = np.where(r < 0, r * p.guide_asym, r)
        gpen += p.guide_weight * r ** 2
    big = 1e6
    if vmin_corridor is not None:
        gpen += np.where(vels[None, :] < np.interp(tt, t, vmin_corridor)[:, None], big, 0)
    if vmax_corridor is not None:
        gpen += np.where(vels[None, :] > np.interp(tt, t, vmax_corridor)[:, None], big, 0)
    def prof(x):
        if callable(x):
            return np.asarray(x(tt), float)
        a = np.asarray(x, float)
        return np.full(nk, float(a)) if a.ndim == 0 else np.interp(tt, t, a)
    # each node represents a time interval: weight its score by its length
    dk = np.gradient(tt) if nk > 1 else np.ones(1)
    wk = (dk / p.t_step)[:, None]
    path, _ = _viterbi(S * wk, tt, vels, prof(p.vint_min) ** 2, prof(p.vint_max) ** 2,
                       p.smooth, p.dv_ref, gpen * wk, p.t_step)
    vc = vels[path]
    # dense trajectory: linear in V^2 t (constant Vint between nodes)
    u = np.interp(t, tt, vc ** 2 * tt)
    v_dense = np.sqrt(np.maximum(u, 0) / np.maximum(t, 1e-9))
    v_dense[t <= tt[0]] = vc[0]
    v_dense[t >= tt[-1]] = vc[-1]
    # events are where the *measured* coherence along the path peaks, whatever
    # score drove the decoder (an NN probability map is smooth in time)
    path_score = spec.at(t, v_dense)
    vf = extract_picks(t, v_dense, path_score, spec, **(extract or {}))
    return PickResult(t, v_dense, vf, path_score, tt, vc)


def extract_picks(t, v_dense, path_score, spec: VelocitySpectrum,
                  min_separation: float = 0.08, rel_threshold: float = 0.15,
                  add_ends: bool = True) -> VelocityFunction:
    """Convert a dense trajectory into discrete picks at coherent events
    (local maxima of the spectrum along the path), with a confidence value:
        confidence = (path value / best value at that time) * (path value)^0.5
    i.e. high when the path sits on the dominant, strong peak."""
    dt = t[1] - t[0]
    ps = np.nan_to_num(path_score)
    if ps.max() <= 0:
        vf = VelocityFunction(t[[0, -1]], v_dense[[0, -1]])
        vf.confidence = np.zeros(2)
        return vf
    pk, _ = find_peaks(ps, height=rel_threshold * ps.max(),
                       distance=max(1, int(min_separation / dt)))
    row_max = np.maximum(spec.values.max(axis=0), 1e-12)
    conf = (ps / row_max) * np.sqrt(np.clip(ps, 0, 1))
    tp, vp, cp = list(t[pk]), list(v_dense[pk]), list(conf[pk])
    if add_ends:
        if not tp or tp[0] > t[0] + min_separation:
            tp.insert(0, t[0]); vp.insert(0, v_dense[0]); cp.insert(0, 0.0)
        if tp[-1] < t[-1] - min_separation:
            tp.append(t[-1]); vp.append(v_dense[-1]); cp.append(0.0)
    return VelocityFunction(np.array(tp), np.array(vp), None, np.array(cp))


def naive_pick(spec: VelocitySpectrum, smooth_s: float = 0.0):
    """Baseline: max-semblance velocity at every time sample (what a 'no physics'
    auto-picker does).  Returned for benchmarking only."""
    v = spec.velocities[np.argmax(spec.values, axis=0)].astype(float)
    if smooth_s:
        from scipy.ndimage import median_filter
        v = median_filter(v, max(1, int(smooth_s / spec.dt)) | 1)
    return v


def marine_vint_floor(t_wb: float, water_velocity: float = 1500.0,
                      sediment_floor: float = 1600.0, margin: float = 0.05):
    """Time-varying Vint lower bound for marine data: water velocity is allowed
    down to the water bottom, sediments below must be faster than
    `sediment_floor`.  Blocks the classic 'follow the water-bottom multiple'
    failure (multiples imply ~water interval velocity)."""
    def f(t):
        t = np.asarray(t, float)
        return np.where(t < t_wb + margin, water_velocity * 0.97, sediment_floor)
    return f


def predicted_multiple_velocity(t, v_primary, t_wb, v_water=1500.0, order=1):
    """Stacking velocity of first/second-order water-layer peg-leg multiples
    predicted from a primary trajectory (1-D Dix-type kinematics):
        t_m = t_p + order * t_wb,
        V_m^2 = (V_p^2 t_p + order * v_w^2 t_wb) / t_m
    Returns V_m on the time axis t (NaN where undefined)."""
    t = np.asarray(t, float)
    tp = t - order * t_wb
    out = np.full_like(t, np.nan)
    ok = tp > t_wb * 0.999
    vp = np.interp(tp[ok], t, v_primary)
    out[ok] = np.sqrt((vp ** 2 * tp[ok] + order * v_water ** 2 * t_wb) / t[ok])
    return out


def multiple_aware_pick(spec: VelocitySpectrum, t_wb: float, v_water: float = 1500.0,
                        params: DPParams | None = None, n_iter: int = 3,
                        width: float = 0.03, strength: float = 0.9, orders=(1, 2),
                        guide=None, score=None, sediment_floor: float = 1600.0):
    """Marine picking with self-consistent multiple suppression.

    Iterates: pick -> predict where the water-layer peg-legs of the picked
    primaries must appear in the spectrum -> attenuate those (t, v) bands ->
    re-pick.  Shallow primaries (above 2*t_wb, multiple-free) are reliable, so
    each pass predicts the next multiples correctly and the trajectory is
    pulled onto the faster primary branch.
    """
    p = params or DPParams()
    if not callable(p.vint_min) and np.ndim(p.vint_min) == 0:
        from dataclasses import replace
        p = replace(p, vint_min=marine_vint_floor(t_wb, v_water, sediment_floor))
    base = np.maximum(spec.values if score is None else score, 0.0)
    cur = base.copy()
    r = dp_pick(spec, p, guide=guide, score=cur)
    V = spec.velocities[:, None]
    for _ in range(n_iter):
        mask = np.ones_like(base)
        for o in orders:
            vm = predicted_multiple_velocity(spec.times, r.velocity, t_wb, v_water, o)
            ok = np.isfinite(vm)
            band = np.exp(-0.5 * ((V - vm[None, :]) / (width * np.where(ok, vm, 1.0)))
                          ** 2)
            band[:, ~ok] = 0.0
            mask *= 1.0 - strength * band
        # pure water-bottom multiples: t = m * t_wb, stacking velocity = water
        T = spec.times[None, :]
        for m in range(2, 8):
            tm = m * t_wb
            if tm > spec.times[-1]:
                break
            mask *= 1.0 - strength * np.exp(-0.5 * ((T - tm) / 0.03) ** 2) * \
                np.exp(-0.5 * ((V - v_water) / (0.04 * v_water)) ** 2)
        cur = base * mask
        r = dp_pick(spec, p, guide=guide, score=cur)
    r.extra_mask = mask
    return r
