"""End-to-end benchmark on synthetic 2-D lines written to / read from SEG-Y.

Compares velocity-picking strategies against the known model:
  naive       max-coherence at every time sample (no physics)
  dp          constrained DP, each location alone
  line        two-pass line workflow (regional guide + lateral consensus)
  line+mult   line workflow + marine multiple-aware decoding (marine only)
  ai          neural probability map decoded by the same constrained DP
Outputs metrics JSON and QC figures.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seisvel import VelocityField, VelocityFunction, offset_regularize, precondition, qc
from seisvel import plot
from seisvel.io.segy import SegyDataset, write_prestack
from seisvel.picking.dp import DPParams, dp_pick, naive_pick
from seisvel.picking.line import ScanParams, autopick_line
from seisvel.semblance import velocity_scan
from seisvel.stack import stack_line
from seisvel.synth.generator import Acquisition, NoiseSpec, synthetic_line

OUT = sys.argv[1] if len(sys.argv) > 1 else "bench_out"
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)

LINES = {
    "marine": dict(seed=7001, marine=True, multiples=1.0,
                   acq=Acquisition(nt=1500, max_offset=3500, ntr=60),
                   noise=NoiseSpec(snr=2.5, swell=0.5, spikes=2, dead_fraction=0.03)),
    "land": dict(seed=7002, marine=False, multiples=0.0,
                 acq=Acquisition(nt=1500, max_offset=3000, ntr=48, offset_jitter=25,
                                 missing_fraction=0.1),
                 noise=NoiseSpec(snr=1.5, ground_roll=1.5, dead_fraction=0.05,
                                 amp_jitter=0.3, spikes=3)),
}
N_CMP, EVERY = 121, 10

neural = None
if MODEL and os.path.exists(MODEL):
    from seisvel.ml.infer import NeuralPicker
    neural = NeuralPicker.load(MODEL, mc_samples=0)

results = {}
for name, cfg in LINES.items():
    print(f"=== {name} line", flush=True)
    gs, truth, models, vstack = synthetic_line(N_CMP, seed=cfg["seed"],
                                               marine=cfg["marine"], acq=cfg["acq"],
                                               noise=cfg["noise"],
                                               multiples=cfg["multiples"],
                                               return_vstack=True)
    if name == "land":   # standard land pre-processing: top mute of ground roll cone
        gs = [precondition.top_mute(g, 1200.0, 0.05) for g in gs]
    path = os.path.join(OUT, f"{name}.sgy")
    write_prestack(path, gs)
    ds = SegyDataset(path)                            # exercise the real-data path
    first = ds.keys[0]
    keys = ds.analysis_keys(EVERY)
    t = ds.t0 + ds.dt * np.arange(ds.nt)
    sp_params = ScanParams(velocities=np.linspace(1400, 5500, 165), supergather=2,
                           offset_bin=50.0)
    t0 = time.time()
    spectra, abspec = {}, {}
    for k in keys:
        g = precondition.scan_flow(ds.supergather(k, 2))
        g = offset_regularize(g, 50.0)
        spectra[k] = velocity_scan(g, sp_params.velocities, "semblance")
        abspec[k] = velocity_scan(g, sp_params.velocities, "ab")
    t_scan = (time.time() - t0) / len(keys)

    def window(k):
        tp = models[k - first].t0_interfaces()
        return tp[0] + 0.05, min(tp[-1], t[-1] - 0.1)

    est = {}
    timing = {"scan_per_location_s": t_scan}
    est["naive"] = {k: naive_pick(spectra[k]) for k in keys}
    t0 = time.time()
    est["dp"] = {k: dp_pick(spectra[k], DPParams()).velocity for k in keys}
    timing["dp_per_location_s"] = (time.time() - t0) / len(keys)
    t0 = time.time()
    lr = autopick_line(ds, keys, sp_params, spectra=spectra)
    timing["line_total_s"] = time.time() - t0
    est["line"] = {k: lr.picks[k].velocity for k in keys}
    fields = {"line": lr.field}
    if cfg["marine"]:
        lm = autopick_line(ds, keys, sp_params, spectra=spectra, water_bottom="auto")
        est["line+mult"] = {k: lm.picks[k].velocity for k in keys}
        fields["line+mult"] = lm.field
    if cfg["marine"]:
        from seisvel.workflow import two_pass_velocity
        t0 = time.time()
        tp2 = two_pass_velocity(ds, keys, sp_params, water_bottom="auto")
        timing["two_pass_total_s"] = time.time() - t0
        est["2pass demult"] = {k: tp2.pass2.picks[k].velocity for k in keys}
        fields["2pass demult"] = tp2.pass2.field
    if neural is not None:
        def score_fn(spec):
            s = neural.score_map(spec, abspec[spec.key])
            s = s / max(s.max(), 1e-12)
            return 0.85 * s + 0.15 * spec.values ** 2
        t0 = time.time()
        la = autopick_line(ds, keys, sp_params, spectra=spectra, score_fn=score_fn,
                           water_bottom="auto" if cfg["marine"] else None)
        timing["ai_line_total_s"] = time.time() - t0
        est["ai"] = {k: la.picks[k].velocity for k in keys}
        fields["ai"] = la.field
        if cfg["marine"]:
            from seisvel.ml.infer import neural_score_fn
            t0 = time.time()
            tpa = two_pass_velocity(ds, keys, sp_params, water_bottom="auto",
                                    score_factory=lambda d: neural_score_fn(neural, d,
                                                                            sp_params))
            timing["ai_two_pass_total_s"] = time.time() - t0
            est["ai 2pass demult"] = {k: tpa.pass2.picks[k].velocity for k in keys}
            fields["ai 2pass demult"] = tpa.pass2.field
        # NN alone, decoded by plain argmax (to show what the constraints add)
        est["ai_argmax"] = {}
        for k in keys:
            P = neural.score_map(spectra[k], abspec[k])
            est["ai_argmax"][k] = spectra[k].velocities[np.argmax(P, axis=0)]

    # ---------------------------------------------------------------- metrics
    m = {}
    for meth, d in est.items():
        errs = [qc.velocity_error(t, d[k], truth[k - first], *window(k)) for k in keys]
        m[meth] = {s: float(np.mean([e[s] for e in errs])) for s in errs[0]}
        m[meth]["worst_location_mape_pct"] = float(max(e["mape_pct"] for e in errs))
        # error vs the best-fit-hyperbola stacking velocity (what semblance measures)
        es = [qc.velocity_error(t, d[k], vstack[k - first], *window(k)) for k in keys]
        m[meth]["mape_vs_vstack_pct"] = float(np.mean([e["mape_pct"] for e in es]))
    # stacks
    true_f = VelocityField()
    for k in keys:
        tp = np.arange(0.0, t[-1], 0.04)
        true_f[k] = VelocityFunction(tp, np.interp(tp, t, truth[k - first]))
    pre = lambda g: precondition.tpow_gain(g, 2.0)
    st_true, _, skeys = stack_line(ds, true_f, preprocess=pre)
    stacks = {"true velocity": st_true}
    for meth, f in fields.items():
        st, _, _ = stack_line(ds, f, preprocess=pre)
        stacks[meth] = st
        # compare only where reflectors exist (deep noise-only zone excluded)
        tps = [mm.t0_interfaces() for mm in models]
        i0 = int(np.median([x[0] for x in tps]) / ds.dt)
        i1 = int(min(np.median([x[-1] for x in tps]) + 0.2, t[-1]) / ds.dt)
        # mean correlation over 0.5 s windows: every depth interval counts the
        # same, so the strong water-bottom event cannot dominate the score
        cs, ps = [], []
        w = int(0.5 / ds.dt)
        for j0 in range(i0, i1 - w // 2, w):
            a = st[:, j0:j0 + w].ravel()
            b = st_true[:, j0:j0 + w].ravel()
            cs.append(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-30))
            ps.append(np.mean(a ** 2) / (np.mean(b ** 2) + 1e-30))
        m[meth]["stack_corr_vs_true"] = float(np.mean(cs))
        m[meth]["stack_corr_worst_window"] = float(np.min(cs))
        m[meth]["stack_power_ratio"] = float(np.mean(ps))
    m["_timing"] = timing
    m["_review_queue"] = {"line": [int(x) for x in lr.review]}
    results[name] = m
    print(json.dumps(m, indent=1), flush=True)

    # ---------------------------------------------------------------- figures
    k = keys[len(keys) // 2]
    fig, ax = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    plot.gather(ax[0], precondition.scan_flow(ds.gather(k)).sort_by_offset(),
                title=f"{name} CMP {k} (conditioned)")
    curves = [(truth[k - first], "w-", "true Vrms"), (est["naive"][k], "c:", "naive max"),
              (est["line"][k], "k--", "auto line (DP)")]
    if "line+mult" in est:
        curves.append((est["line+mult"][k], "y--", "auto + multiple-aware"))
    if "ai" in est:
        curves.append((est["ai"][k], "m-", "AI + DP"))
    plot.spectrum(ax[1], spectra[k], curves, title="semblance")
    if neural is not None:
        P = neural.score_map(spectra[k], abspec[k])
        ax[2].imshow(P.T, aspect="auto", cmap="magma",
                     extent=[spectra[k].velocities[0], spectra[k].velocities[-1], t[-1], t[0]])
        ax[2].plot(truth[k - first], t, "w-", lw=1)
        ax[2].set_title("neural probability map")
    else:
        plot.spectrum(ax[2], abspec[k], [(truth[k - first], "w-", "true")], title="AB")
    fig.savefig(os.path.join(OUT, f"{name}_location.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    n = len(stacks)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 7), sharey=True)
    for a_, (lab, st) in zip(np.atleast_1d(ax), stacks.items()):
        plot.section(a_, st, skeys, ds.dt, title=lab)
    fig.savefig(os.path.join(OUT, f"{name}_stacks.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)

    vm = [m_ for m_ in ("line", "2pass demult", "ai", "ai 2pass demult") if m_ in fields]
    fig, ax = plt.subplots(1, len(vm) + 1, figsize=(5 * (len(vm) + 1), 6), sharey=True)
    vmin, vmax = truth.min(), truth.max()
    plot.velocity_section(ax[0], truth, skeys, t, "true Vrms", vmin, vmax)
    for a_, meth in zip(ax[1:], vm):
        plot.velocity_section(a_, fields[meth].grid(skeys, t), skeys, t, meth, vmin, vmax)
    fig.savefig(os.path.join(OUT, f"{name}_velocity_sections.png"), dpi=80,
                bbox_inches="tight")
    plt.close(fig)
    ds.close()

json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print("done")
