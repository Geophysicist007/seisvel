"""Full-line field test: Mobil AVO Viking Graben line 12 (SEG open data).
1001 shots x 120 channels, 25 m groups, offsets 262-3237 m, 6 s at 4 ms,
CDP 1-2142 at 12.5 m, nominal fold 60.

    python examples/field_viking_line.py seismic.segy out_dir models/velpicker_synth_v1.pt

No well logs are used (not in the open-data bucket), so validation is
seismic-only: method agreement, residual moveout on demultipled gathers,
Dix plausibility and stack images.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seisvel import offset_regularize, plot, precondition, qc
from seisvel.io.segy import SegyDataset, write_stack
from seisvel.ml.infer import NeuralPicker
from seisvel.moveout import nmo_correct
from seisvel.picking.line import ScanParams, autopick_line, compute_spectrum
from seisvel.radon import demultiple_gather
from seisvel.stack import stack_line
from seisvel.velocity import dix_dense

path, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else None
EVERY = int(os.environ.get("EVERY", 50))
os.makedirs(out, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


ds = SegyDataset(path)
full = [k for k in ds.keys if ds.fold(k) >= 50]
keys = full[::EVERY]
t = ds.t0 + ds.dt * np.arange(ds.nt)
log(f"{len(ds.keys)} CDPs, full-fold {full[0]}-{full[-1]}, {len(keys)} analysis locations")

cond = dict(gain_power=2.0, band=(6.0, 60.0), agc_window=0.4)
sp = ScanParams(velocities=np.linspace(1400, 4500, 125), supergather=2, offset_bin=50.0,
                band=(6.0, 60.0))

from seisvel.workflow import field_velocity_workflow
npk = NeuralPicker.load(model) if model else None
wf = field_velocity_workflow(ds, keys, sp, neural=npk, verbose=True)
fields = {"physics pass 1": wf.starts["physics"]}
if "ai" in wf.starts:
    fields["AI pass 1"] = wf.starts["ai"]
fields["final (demultiple + guided DP)"] = wf.field
for n, f in fields.items():
    f.save_csv(os.path.join(out, n.split(" (")[0].replace(" ", "_") + ".csv"))
log("workflow done; review queue:", wf.review)

# ------------------------------------------------- held-out QC (between locations)
# RMO is measured at CDPs half-way between analysis locations, which the
# pickers never saw, on gathers demultipled with the Radon velocity.
names = list(fields)
hold = [k + EVERY // 2 for k in keys[:-1] if (k + EVERY // 2) in ds.index]
rows = []
for k in hold:
    g = ds.supergather(k, 2)
    twb = precondition.detect_water_bottom(g) or 0.0
    gd = demultiple_gather(g, wf.radon_field.at(k, t), q_cut=0.02, protect_until=1.9 * twb)
    gd = precondition.scan_flow(gd, **cond)
    m = t > max(twb + 0.3, 0.8)
    row = {"cdp": int(k)}
    for n in names:
        v = fields[n].at(k, t)
        deltas, P = qc.residual_moveout(nmo_correct(gd, v, stretch_max=0.5), 0.06, 41)
        pk = deltas[np.argmax(P[:, m], axis=0)]
        w = P[:, m].max(axis=0)
        row[n] = dict(rmo_abs_ms=float(1e3 * np.sum(w * np.abs(pk)) / np.sum(w)),
                      rmo_bias_ms=float(1e3 * np.sum(w * pk) / np.sum(w)),
                      flat_after_demult=float(qc.flatness(nmo_correct(gd, v, stretch_max=0.5))[0]))
    rows.append(row)
summary = {n: {s: float(np.mean([r[n][s] for r in rows]))
               for s in ("rmo_abs_ms", "rmo_bias_ms", "flat_after_demult")} for n in names}
for n in names:
    V = np.array([fields[n].at(k, t) for k in keys])
    summary[n].update({f"v_{x}s": float(np.mean(np.interp(x, t, V.mean(0)))) for x in (1, 2, 3, 4, 5)})
    vi = np.array([dix_dense(t, fields[n].at(k, t), smooth=25) for k in keys])[:, t > 0.8]
    summary[n]["dix_bad_frac"] = float(np.mean(np.isnan(vi) | (vi < 1450) | (vi > 6500)))
if "AI pass 1" in fields:
    A = np.array([fields["AI pass 1"].at(k, t) for k in keys])
    P_ = np.array([fields["physics pass 1"].at(k, t) for k in keys])
    mm = t > 0.8
    summary["pass1_disagreement_pct"] = float(100 * np.mean(np.abs(A - P_)[:, mm] / P_[:, mm]))
summary["final_multistart_spread_pct_mean"] = float(np.nanmean(list(wf.spread_pct.values())))
summary["final_multistart_spread_pct_max"] = float(np.nanmax(list(wf.spread_pct.values())))
summary["review_queue"] = [int(k) for k in wf.review]
json.dump(dict(summary=summary, heldout=rows,
               spread={int(k): v for k, v in wf.spread_pct.items()}),
          open(os.path.join(out, "qc.json"), "w"), indent=1)
log(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- stacks
pre = lambda g: precondition.top_mute(precondition.bandpass(precondition.tpow_gain(g, 2.0),
                                                            6, 60), 1480.0, 0.06)
stacks = {}
for n in names:
    st, fold, skeys = stack_line(ds, fields[n], preprocess=pre, stretch_max=0.5)
    stacks[n] = st
    write_stack(os.path.join(out, f"stack_{list(fields).index(n)}.sgy"), st, skeys, ds.dt)
    log("stacked", n)

win = [k for k in ds.keys if 900 <= k <= 1300]
from seisvel.radon import DemultipledDataset
dem = DemultipledDataset(ds, wf.radon_field, q_cut=0.02, cache=False, water_bottom="auto")
st_dm, _, wkeys = stack_line(dem, wf.field, keys=win, preprocess=pre, stretch_max=0.5)
i0, i1 = skeys.index(win[0]), skeys.index(win[-1]) + 1
log("demultipled window stack done")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
plot.section(ax[0], stacks[names[-1]][i0:i1], wkeys, ds.dt, title="final velocity, raw data")
plot.section(ax[1], st_dm, wkeys, ds.dt, title="final velocity, Radon-demultipled data")
fig.savefig(os.path.join(out, "viking_window_demultiple_stack.png"), dpi=75,
            bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, len(names), figsize=(7 * len(names), 9), sharey=True)
for a_, n in zip(np.atleast_1d(ax), names):
    plot.section(a_, stacks[n], skeys, ds.dt, title=n)
fig.savefig(os.path.join(out, "viking_line_stacks.png"), dpi=70, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, len(names), figsize=(6 * len(names), 6), sharey=True)
for a_, n in zip(np.atleast_1d(ax), names):
    im = plot.velocity_section(a_, fields[n].grid(skeys[::10], t), skeys[::10], t, n, 1450, 3500)
fig.colorbar(im, ax=ax, shrink=0.6, label="m/s")
fig.savefig(os.path.join(out, "viking_line_velocities.png"), dpi=70, bbox_inches="tight")
plt.close(fig)

k = keys[len(keys) // 2]
fig, ax = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
plot.gather(ax[0], precondition.scan_flow(ds.supergather(k, 2), **cond).sort_by_offset(),
            title=f"CDP {k} supergather")
cs = [(fields[n].at(k, t), s, n) for n, s in zip(names, ["y-", "m-", "w-"])]
plot.spectrum(ax[1], compute_spectrum(ds, k, sp), cs, title="semblance (raw)")
gd = demultiple_gather(ds.supergather(k, 2), wf.radon_field.at(k, t))
sp2 = compute_spectrum(type("D", (), {"supergather": lambda s, kk, h: gd,
                                      "gather": lambda s, kk: gd})(), k,
                       ScanParams(velocities=sp.velocities, supergather=0, offset_bin=50.0,
                                  band=(6.0, 60.0)))
plot.spectrum(ax[2], sp2, cs, title="semblance after Radon demultiple")
fig.savefig(os.path.join(out, "viking_line_location.png"), dpi=80, bbox_inches="tight")
log("done")
