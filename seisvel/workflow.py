"""Multi-pass velocity workflow (the Yilmaz-style iterative loop, automated):

    pass 1  : velocity analysis on raw data (multiple-aware, lateral consensus)
    demult  : parabolic Radon demultiple using pass-1 primary velocities
    pass 2  : velocity analysis on demultipled data, guided by pass 1
    (stack) : NMO + stack of the demultipled data with pass-2 velocities
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .picking.dp import DPParams
from .picking.line import LineResult, ScanParams, autopick_line
from .radon import DemultipledDataset
from .velocity import VelocityField, VelocityFunction


@dataclass
class TwoPassResult:
    pass1: LineResult
    pass2: LineResult
    demultipled: DemultipledDataset


def scaled_field(f: VelocityField, factor: float) -> VelocityField:
    out = VelocityField(mode=f.mode)
    for k in f.keys:
        vf = f[k]
        out[k] = VelocityFunction(vf.times, vf.velocities * factor, vf.etas, vf.confidence)
    return out


def two_pass_velocity(dataset, keys, scan: ScanParams | None = None,
                      dp: DPParams | None = None, water_bottom="auto",
                      v_water: float = 1500.0, q_cut: float = 0.02,
                      radon_velocity_factor: float = 1.0, score_factory=None,
                      verbose=False, **radon_kw) -> TwoPassResult:
    """Automatic two-pass velocity analysis with Radon demultiple in between.

    radon_velocity_factor : scale pass-1 velocities before demultiple (values
        slightly < 1, e.g. 0.97, protect primaries if pass 1 may be a bit fast)
    score_factory : optional callable dataset -> score_fn (e.g.
        lambda d: ml.infer.neural_score_fn(picker, d, scan)) for AI picking in
        both passes; pass 2 sees the demultipled data.
    """
    scan = scan or ScanParams()
    dp = dp or DPParams()
    sf = (lambda d: score_factory(d)) if score_factory else (lambda d: None)
    r1 = autopick_line(dataset, keys, scan, dp, score_fn=sf(dataset),
                       water_bottom=water_bottom, v_water=v_water, verbose=verbose)
    vf = r1.field if radon_velocity_factor == 1.0 else \
        scaled_field(r1.field, radon_velocity_factor)
    dem = DemultipledDataset(dataset, vf, q_cut=q_cut, water_bottom=water_bottom,
                             v_water=v_water, **radon_kw)
    # pass 2: guide by pass 1 (tighter), keep the marine floor
    r2 = autopick_line(dem, keys, scan, dp,
                       score_fn=sf(dem),
                       water_bottom=water_bottom, v_water=v_water, verbose=verbose)
    return TwoPassResult(r1, r2, dem)


@dataclass
class FieldWorkflowResult:
    field: VelocityField              # final velocities
    starts: dict                      # name -> VelocityField of each first-pass picker
    radon_field: VelocityField
    spread_pct: dict                  # key -> disagreement of the converged starts (%)
    review: list                      # keys whose spread exceeds the threshold


def dix_project(t, v, velocities=None, sigma=0.02, vint_min=1450.0, vint_max=6500.0,
                smooth=0.2):
    """Closest (in a Gaussian-likelihood sense) velocity curve to `v` whose Dix
    interval velocities stay within bounds, found by the constrained DP."""
    from .picking.dp import DPParams, dp_pick
    from .semblance import VelocitySpectrum
    vg = np.linspace(min(1400.0, v.min() * 0.95), max(v.max() * 1.05, 2000.0), 200) \
        if velocities is None else np.asarray(velocities, float)
    S = np.exp(-0.5 * ((vg[:, None] - v[None, :]) / (sigma * v[None, :])) ** 2)
    spec = VelocitySpectrum(S, vg, t, "rva")
    p = DPParams(vint_min=vint_min, vint_max=vint_max, smooth=smooth, power=1.0,
                 tmin=max(t[0], 0.02))
    return dp_pick(spec, p).velocity


def geometric_mean_field(fields, keys, t):
    out = VelocityField()
    for k in keys:
        v = np.exp(np.mean([np.log(f.at(k, t)) for f in fields], axis=0))
        tp = np.arange(t[0], t[-1] + 1e-9, 0.04)
        out[k] = VelocityFunction(tp, np.interp(tp, t, v))
    return out


def field_velocity_workflow(dataset, keys, scan: ScanParams | None = None,
                            neural=None, water_bottom="auto", v_water=1500.0,
                            q_cut=0.02, rva_iter=12, rva_max_change=0.06, review_spread_pct=5.0,
                            lateral_window=3, refine: str = "dp", guide_sigma: float = 0.08,
                            verbose=False) -> FieldWorkflowResult:
    """Velocity workflow designed for real, multiple-contaminated marine data.

    1. Independent first passes: physics (multiple-aware constrained DP, line
       consensus) and, if a NeuralPicker is given, AI + DP.
    2. Radon velocity = geometric mean of the first passes.  On field data the
       physics pass tends to be trapped by multiples (slow) and the AI pass to
       overshoot (fast); the mean lies between multiples and primaries, which is
       exactly what parabolic Radon needs (primaries ~flat, multiples q > 0).
    3. Demultiple every analysis supergather.
    4. Re-pick on the demultipled gathers, started from EACH first pass:
       refine='dp'  (default) constrained DP on the demultipled semblance,
                    guided (sigma=guide_sigma) by each first pass in turn;
       refine='rva' residual velocity analysis (experimental: on synthetics it
                    drifts away from the true velocity with iterations, see
                    docs/RESEARCH.md 4c).
       Final = mean of the refined curves; their spread (weighted by
       coherence) is the uncertainty that fills the review queue.
    5. Light lateral median smoothing.
    """
    from scipy.ndimage import median_filter
    from . import precondition, qc
    from .gather import offset_regularize
    from .radon import demultiple_gather
    from .moveout import nmo_correct

    scan = scan or ScanParams()
    keys = list(keys)
    t = dataset.t0 + dataset.dt * np.arange(dataset.nt)
    starts = {}
    r = autopick_line(dataset, keys, scan, water_bottom=water_bottom, v_water=v_water,
                      verbose=verbose)
    starts["physics"] = r.field
    spectra = r.spectra
    if neural is not None:
        from .ml.infer import neural_score_fn
        ra = autopick_line(dataset, keys, scan, spectra=spectra, water_bottom=water_bottom,
                           v_water=v_water, score_fn=neural_score_fn(neural, dataset, scan))
        starts["ai"] = ra.field
    vrad = geometric_mean_field(list(starts.values()), keys, t)
    finals, spread = [], {}
    for i, k in enumerate(keys):
        g = dataset.supergather(k, scan.supergather)
        twb = precondition.detect_water_bottom(g, water_velocity=v_water) \
            if water_bottom == "auto" else (water_bottom or 0.0)
        gd = demultiple_gather(g, vrad.at(k, t), q_cut=q_cut,
                               protect_until=1.9 * twb if twb else None)
        gc = precondition.scan_flow(gd, gain_power=scan.gain_power, band=scan.band,
                                    agc_window=scan.agc_window)
        if scan.offset_bin:
            gc = offset_regularize(gc, scan.offset_bin)
        if refine == "rva":
            conv = [qc.residual_velocity_update(gc, f.at(k, t), n_iter=rva_iter,
                                                max_change=rva_max_change,
                                                min_t=(twb or 0.0) + 0.2)[0]
                    for f in starts.values()]
        else:
            from .picking.dp import dp_pick, marine_vint_floor
            from .semblance import velocity_scan
            spec = velocity_scan(gc, scan.velocities, scan.measure, window=scan.window,
                                 stretch_max=scan.stretch_max)
            p = DPParams(guide_weight=1.0, guide_sigma=guide_sigma,
                         vint_min=marine_vint_floor(twb, v_water) if twb else 1400.0)
            conv = [dp_pick(spec, p, guide=f.at(k, t)).velocity for f in starts.values()]
        conv = np.array(conv)
        vm = conv.mean(axis=0)
        finals.append(vm)
        # spread weighted by coherence after NMO: where there are no events RVA
        # is unconstrained, and disagreement there means nothing
        _, sem = qc.flatness(nmo_correct(gc, vm, stretch_max=0.5))
        w = np.where(t > ((twb or 0.0) + 0.3), sem, 0.0)
        spread[k] = float(100 * np.sum(w * np.ptp(conv, axis=0) / vm) / max(w.sum(), 1e-12)) \
            if len(conv) > 1 else float("nan")
        if verbose:
            print(f"RVA {i + 1}/{len(keys)}  CDP {k}  spread {spread[k]:.1f}%", flush=True)
    V = np.array(finals)
    if len(keys) >= 3 and lateral_window > 1:
        V = median_filter(V, size=(lateral_window, 1), mode="nearest")
    # Dix projection: RVA has no physical constraint, so decode the closest
    # trajectory that satisfies the interval-velocity bounds
    V = np.array([dix_project(t, v, scan.velocities) for v in V])
    out = VelocityField()
    tp = np.arange(t[0], t[-1] + 1e-9, 0.04)
    for k, v in zip(keys, V):
        out[k] = VelocityFunction(tp, np.interp(tp, t, v))
    review = [k for k, s in spread.items() if np.isfinite(s) and s > review_spread_pct]
    return FieldWorkflowResult(out, starts, vrad, spread, review)
