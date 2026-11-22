"""First field-data test: Mobil AVO Viking Graben line 12, 4-shot excerpt
(`viking_small.segy`, shipped with segyio's tutorials; 120 channels, 25 m group,
offsets 262-3237 m, 2.4 s at 4 ms).

The excerpt has CDP fold <= 4, too low for CMP velocity analysis, so the four
shot gathers are combined into one offset gather. In shallow, nearly flat North
Sea geology this is a reasonable proxy for a CMP supergather. Midpoints are smeared
over about 1.6 km, so treat the result as a robustness test, not as final
velocities.

    python examples/field_viking_small.py examples/viking_small.segy out_dir [models/velpicker_synth_v1.pt]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seisvel import offset_regularize, precondition, qc, velocity_scan
from seisvel import plot
from seisvel.gather import combine
from seisvel.io.segy import HeaderMap, MemoryDataset, SegyDataset
from seisvel.moveout import nmo_correct
from seisvel.picking.dp import DPParams, dp_pick, multiple_aware_pick, naive_pick
from seisvel.radon import demultiple_gather, radon_demultiple

path, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)

ds = SegyDataset(path, HeaderMap(cdp=9))           # key = field record (shot)
g = combine([ds.gather(k) for k in ds.keys], key=0).sort_by_offset()
g.offsets = np.abs(g.offsets)
print("traces", g.ntr, "offsets", g.offsets.min(), g.offsets.max(), "tmax", g.times[-1])

twb = precondition.detect_water_bottom(g)
print("water bottom t0 (s):", twb)
vels = np.linspace(1400, 4000, 105)


def spectra(gg):
    c = offset_regularize(precondition.scan_flow(gg, band=(6, 60)), 25.0)
    return {m: velocity_scan(c, vels, m) for m in ("semblance", "ab", "similarity")}, c


S, gc = spectra(g)
res = {"water_bottom_s": twb}
picks = {}
picks["naive"] = naive_pick(S["semblance"])
picks["dp"] = dp_pick(S["semblance"], DPParams()).velocity
picks["dp multiple-aware"] = multiple_aware_pick(S["semblance"], twb).velocity
picks["dp multiple-aware (similarity)"] = multiple_aware_pick(S["similarity"], twb).velocity
if model:
    from seisvel.ml.infer import NeuralPicker
    npk = NeuralPicker.load(model)
    picks["ai + dp"] = npk.pick(S["semblance"], S["ab"],
                                DPParams(vint_min=lambda t: np.where(t < twb + 0.05,
                                                                     1455.0, 1600.0))).velocity

# second pass after Radon demultiple, primaries from the multiple-aware pick
v1 = picks["dp multiple-aware"]
gd = demultiple_gather(g, v1, q_cut=0.02)
S2, gc2 = spectra(gd)
picks["2-pass after demultiple"] = multiple_aware_pick(S2["semblance"], twb).velocity
if model:
    picks["ai 2-pass after demultiple"] = npk.pick(
        S2["semblance"], S2["ab"],
        DPParams(vint_min=lambda t: np.where(t < twb + 0.05, 1455.0, 1600.0))).velocity

# QC without ground truth: flatness and Dix plausibility of each answer
for name, v in picks.items():
    data = gd if "demultiple" in name else g
    r = qc.gather_report(precondition.agc(data, 0.4), v, stretch_max=0.6)
    res[name] = {k: round(float(x), 4) for k, x in r.items()}
    res[name]["v_at_1s"] = float(np.interp(1.0, g.times, v))
    res[name]["v_at_2s"] = float(np.interp(2.0, g.times, v))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(out, "viking_small_qc.json"), "w"), indent=1)

# figures
fig, ax = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
plot.gather(ax[0], gc, title="4-shot offset gather (conditioned)")
styles = ["c:", "k--", "y--", "g--", "m-", "w-", "r-"]
curves = [(v, styles[i % len(styles)], n) for i, (n, v) in enumerate(picks.items())]
plot.spectrum(ax[1], S["semblance"], curves[:5], title="semblance, raw")
plot.spectrum(ax[2], S2["semblance"], [c for c in curves if "demultiple" in c[2]],
              title="semblance after Radon demultiple")
best = picks.get("ai 2-pass after demultiple", picks["2-pass after demultiple"])
plot.gather(ax[3], nmo_correct(precondition.agc(gd, 0.4), best, stretch_max=0.6),
            title="NMO with final velocity (demultipled)")
for a in ax:
    a.axhline(twb, color="r", lw=0.6)
fig.savefig(os.path.join(out, "viking_small_velocity.png"), dpi=90, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
c = np.percentile(np.abs(precondition.agc(g, 0.4).data), 98)
for a, d, ttl in [(ax[0], g, "raw"), (ax[1], gd, "demultipled"),
                  (ax[2], g.copy(g.data - gd.data), "removed (multiple model)")]:
    a.imshow(precondition.agc(g, 0.4).data.T * 0 + precondition.tpow_gain(d, 2).data.T,
             aspect="auto", cmap="gray_r",
             vmin=-np.percentile(np.abs(precondition.tpow_gain(g, 2).data), 98),
             vmax=np.percentile(np.abs(precondition.tpow_gain(g, 2).data), 98),
             extent=[0, d.ntr, d.times[-1], 0])
    a.set_title(ttl)
fig.savefig(os.path.join(out, "viking_small_demultiple.png"), dpi=90, bbox_inches="tight")
print("figures written to", out)
