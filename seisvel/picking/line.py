"""Line / survey-wide automatic velocity analysis.

Workflow (mirrors how a processing geophysicist works, made automatic):
  1. velocity spectra at analysis locations on conditioned supergathers
  2. REGIONAL guide: decode the average of all spectra (primaries are laterally
     consistent and add up; noise and most multiple-energy smear out)
  3. pass 1: constrained DP at every location with the regional guide
  4. lateral consensus: median of pass-1 trajectories over neighbouring
     locations -> local guide (outlier-robust)
  5. pass 2: DP with the tighter local guide
  6. discrete picks + confidence -> VelocityField; low-confidence locations are
     flagged for human review (the QC queue)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import median_filter

from .. import precondition
from ..gather import Gather, offset_regularize
from ..semblance import VelocitySpectrum, velocity_scan
from ..velocity import VelocityField
from .dp import DPParams, PickResult, dp_pick, multiple_aware_pick


@dataclass
class ScanParams:
    velocities: np.ndarray = field(default_factory=lambda: np.linspace(1400, 5500, 165))
    measure: str = "semblance"
    window: float = 0.04
    stretch_max: float = 1.0
    supergather: int = 2            # half-width in CMPs (2 -> 5 CMPs combined)
    offset_bin: float | None = 50.0
    band: tuple | None = None
    agc_window: float | None = 0.4
    gain_power: float | None = 2.0


def compute_spectrum(dataset, key, sp: ScanParams,
                     condition: Callable[[Gather], Gather] | None = None):
    g = dataset.supergather(key, sp.supergather) if sp.supergather else dataset.gather(key)
    if condition is None:
        g = precondition.scan_flow(g, gain_power=sp.gain_power, band=sp.band,
                                   agc_window=sp.agc_window)
    else:
        g = condition(g)
    if sp.offset_bin:
        g = offset_regularize(g, sp.offset_bin)
    return velocity_scan(g, sp.velocities, measure=sp.measure, window=sp.window,
                         stretch_max=sp.stretch_max)


def regional_guide(spectra: Sequence[VelocitySpectrum], params: DPParams):
    S = np.mean([s.normalized("time") for s in spectra], axis=0)
    s0 = spectra[0]
    avg = VelocitySpectrum(S, s0.velocities, s0.times, "regional")
    p = replace(params, smooth=params.smooth * 4, guide_weight=0.0)
    return dp_pick(avg, p).velocity


@dataclass
class LineResult:
    keys: list
    spectra: dict
    picks: dict                  # key -> PickResult (pass 2)
    field: VelocityField
    guide: np.ndarray
    review: list                 # keys flagged for manual QC


def autopick_line(dataset, keys: Sequence, scan: ScanParams | None = None,
                  dp: DPParams | None = None, spectra: dict | None = None,
                  lateral_window: int = 5, review_threshold: float = 0.35,
                  score_fn: Callable[[VelocitySpectrum], np.ndarray] | None = None,
                  water_bottom=None, v_water: float = 1500.0,
                  verbose: bool = False) -> LineResult:
    """Fully automatic two-pass velocity picking along a line.

    score_fn : optional mapping spectrum -> score map (nv, nt) used instead of
               the raw spectrum in the DP (hook for the neural picker)
    water_bottom : None (land / no multiple handling), 'auto' (detect per
               location from near traces) or a float t0 in seconds -> marine
               multiple-aware decoding (water-velocity floor + peg-leg masking)
    """
    scan = scan or ScanParams()
    dp = dp or DPParams()
    keys = list(keys)
    if spectra is None:
        spectra = {}
        for i, k in enumerate(keys):
            spectra[k] = compute_spectrum(dataset, k, scan)
            if verbose and i % 10 == 0:
                print(f"spectrum {i + 1}/{len(keys)}")
    scores = {k: (score_fn(spectra[k]) if score_fn else None) for k in keys}

    def spec_for_guide(k):
        if scores[k] is None:
            return spectra[k]
        s = spectra[k]
        return VelocitySpectrum(scores[k], s.velocities, s.times, "score", k)

    twb = {}
    if water_bottom is not None:
        for k in keys:
            if water_bottom == "auto":
                twb[k] = precondition.detect_water_bottom(dataset.gather(k),
                                                          water_velocity=v_water)
            else:
                twb[k] = float(water_bottom)

    def pick(k, params, guide_curve):
        if twb.get(k):
            return multiple_aware_pick(spectra[k], twb[k], v_water, params,
                                       guide=guide_curve, score=scores[k])
        return dp_pick(spectra[k], params, guide=guide_curve, score=scores[k])

    if twb:
        from .dp import marine_vint_floor
        tw = float(np.median([v for v in twb.values() if v]))
        gp = replace(dp, vint_min=marine_vint_floor(tw, v_water))
    else:
        gp = dp
    guide = regional_guide([spec_for_guide(k) for k in keys], gp)
    p1 = replace(dp, guide_weight=dp.guide_weight or 1.0, guide_sigma=0.15)
    pass1 = {k: pick(k, p1, guide) for k in keys}
    V1 = np.array([pass1[k].velocity for k in keys])
    w = min(lateral_window, len(keys)) | 1
    Vc = median_filter(V1, size=(w, 1), mode="nearest") if len(keys) > 2 else V1
    p2 = replace(dp, guide_weight=dp.guide_weight or 1.0, guide_sigma=0.06)
    picks, review = {}, []
    vfield = VelocityField()
    for i, k in enumerate(keys):
        r = pick(k, p2, Vc[i])
        picks[k] = r
        vfield[k] = r.function
        c = r.function.confidence
        valid = c[c > 0] if c is not None else np.array([])
        if valid.size == 0 or float(np.mean(valid)) < review_threshold:
            review.append(k)
    return LineResult(keys, spectra, picks, vfield, guide, review)


def dense_field(result: LineResult, t_axis=None) -> VelocityField:
    """VelocityField built from the dense DP trajectories (sampled every 40 ms),
    an alternative to the sparse event picks."""
    from ..velocity import VelocityFunction
    f = VelocityField(mode="vrms")
    for k, r in result.picks.items():
        t = r.times if t_axis is None else t_axis
        tp = np.arange(t[0], t[-1] + 1e-9, 0.04)
        f[k] = VelocityFunction(tp, np.interp(tp, r.times, r.velocity))
    return f
