"""Interactive velocity picker (matplotlib) - the classic processing-system layout:

   [ conditioned gather ]  [ velocity spectrum + picks ]  [ NMO-corrected gather ]

Mouse (on the spectrum panel)
   left click   add a pick          right click   delete nearest pick
Keys
   n / p        next / previous analysis location
   a            auto-pick this location (constrained DP)
   i            AI pick (neural + DP), if a NeuralPicker was supplied
   m            cycle coherence measure (semblance / ab / similarity)
   + / -        stretch-mute limit up / down (10 % steps)
   u            undo last edit       c   clear picks
   s            save all picks to CSV (path given at start)
Live feedback: NMO panel updates on every edit; Dix interval-velocity warnings
and pick confidences are shown in the title bar; picks of the neighbouring
locations are drawn as dashed guides.
"""
from __future__ import annotations

import numpy as np

from .. import precondition
from ..gather import offset_regularize
from ..moveout import nmo_correct
from ..picking.dp import DPParams, dp_pick
from ..picking.line import ScanParams
from ..semblance import velocity_scan
from ..velocity import VelocityField, VelocityFunction

MEASURE_CYCLE = ("semblance", "ab", "similarity")


class InteractivePicker:
    def __init__(self, dataset, keys=None, vfield: VelocityField | None = None,
                 scan: ScanParams | None = None, dp: DPParams | None = None,
                 neural=None, save_path="picks.csv", stretch_max=0.5, show=True):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.ds = dataset
        self.keys = list(keys if keys is not None else dataset.analysis_keys(20))
        self.vf = vfield or VelocityField()
        self.scan = scan or ScanParams()
        self.dp = dp or DPParams()
        self.neural = neural
        self.save_path = save_path
        self.stretch = stretch_max
        self.i = 0
        self.undo_stack = []
        self._cache = {}
        self.fig, self.ax = plt.subplots(1, 3, figsize=(16, 9), sharey=True,
                                         gridspec_kw=dict(width_ratios=[1, 1.2, 1]))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()
        if show:
            plt.show()

    # ---------------------------------------------------------------- data
    @property
    def key(self):
        return self.keys[self.i]

    def _prepare(self):
        k = (self.key, self.scan.measure)
        if k not in self._cache:
            g = self.ds.supergather(self.key, self.scan.supergather)
            gc = precondition.scan_flow(g, band=self.scan.band,
                                        agc_window=self.scan.agc_window)
            gr = offset_regularize(gc, self.scan.offset_bin) if self.scan.offset_bin else gc
            spec = velocity_scan(gr, self.scan.velocities, self.scan.measure,
                                 window=self.scan.window)
            self._cache[k] = (gc.sort_by_offset(), spec)
        return self._cache[k]

    def picks(self):
        if self.key in self.vf:
            f = self.vf[self.key]
            return list(f.times), list(f.velocities)
        return [], []

    def set_picks(self, t, v, conf=None):
        self.undo_stack.append((self.key, self.picks()))
        if len(t):
            self.vf[self.key] = VelocityFunction(np.array(t), np.array(v), None, conf)
        elif self.key in self.vf:
            del self.vf.functions[self.key]

    # -------------------------------------------------------------- drawing
    def load(self):
        self.g, self.spec = self._prepare()
        self.redraw()

    def _profile(self):
        t = self.g.times
        if self.key in self.vf and len(self.vf[self.key]) > 0:
            return self.vf[self.key].interpolate(t)
        if len(self.vf):
            return self.vf.at(self.key, t)
        return None

    def redraw(self):
        ax = self.ax
        for a in ax:
            a.cla()
        g, sp = self.g, self.spec
        t = g.times
        clip = np.percentile(np.abs(g.data), 98) + 1e-12
        ext = [0, g.ntr, t[-1], t[0]]
        ax[0].imshow(g.data.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip,
                     extent=ext)
        ax[0].set_title(f"CMP {self.key}  ({self.i + 1}/{len(self.keys)})")
        ax[0].set_ylabel("t0 (s)")
        ax[1].imshow(sp.values.T, aspect="auto", cmap="jet", origin="upper",
                     extent=[sp.velocities[0], sp.velocities[-1], sp.times[-1], sp.times[0]])
        # neighbour guides
        for dk in (-1, 1):
            j = self.i + dk
            if 0 <= j < len(self.keys) and self.keys[j] in self.vf:
                f = self.vf[self.keys[j]]
                ax[1].plot(f.interpolate(t), t, "w--", lw=0.8, alpha=0.7)
        tp, vp = self.picks()
        prof = self._profile()
        title = f"{sp.measure}  stretch {self.stretch * 100:.0f}%"
        if prof is not None:
            ax[1].plot(prof, t, "w-", lw=1.5)
            gn = nmo_correct(g, prof, stretch_max=self.stretch)
            ax[2].imshow(gn.data.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip,
                         extent=ext)
            if len(tp) >= 2:
                warn = VelocityFunction(tp, vp).check()
                title += f"   Dix warnings: {len(warn)}"
        if tp:
            ax[1].plot(vp, tp, "o", mfc="white", mec="k", ms=7)
            f = self.vf.functions.get(self.key)
            if f is not None and f.confidence is not None:
                for tt, vv, c in zip(f.times, f.velocities, f.confidence):
                    ax[1].annotate(f"{c:.2f}", (vv, tt), color="w", fontsize=7,
                                   xytext=(5, -3), textcoords="offset points")
        ax[1].set_title(title)
        ax[1].set_xlabel("velocity (m/s)")
        ax[2].set_title("NMO corrected")
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------------- events
    def on_click(self, ev):
        if ev.inaxes is not self.ax[1] or ev.xdata is None:
            return
        tp, vp = self.picks()
        if ev.button == 1:
            tp.append(ev.ydata)
            vp.append(ev.xdata)
        elif ev.button == 3 and tp:
            sv = np.ptp(self.spec.velocities)
            st = np.ptp(self.spec.times)
            d = [((t - ev.ydata) / st) ** 2 + ((v - ev.xdata) / sv) ** 2
                 for t, v in zip(tp, vp)]
            k = int(np.argmin(d))
            tp.pop(k)
            vp.pop(k)
        self.set_picks(tp, vp)
        self.redraw()

    def on_key(self, ev):
        k = ev.key
        if k == "n" and self.i < len(self.keys) - 1:
            self.i += 1
            self.load()
        elif k == "p" and self.i > 0:
            self.i -= 1
            self.load()
        elif k == "a":
            guide = self._profile()
            r = dp_pick(self.spec, self.dp, guide=guide)
            f = r.function
            self.set_picks(list(f.times), list(f.velocities), f.confidence)
            self.redraw()
        elif k == "i" and self.neural is not None:
            ab = velocity_scan(offset_regularize(self.g, self.scan.offset_bin)
                               if self.scan.offset_bin else self.g,
                               self.scan.velocities, "ab", window=self.scan.window)
            r = self.neural.pick(self.spec, ab, self.dp)
            f = r.function
            self.set_picks(list(f.times), list(f.velocities), f.confidence)
            self.redraw()
        elif k == "m":
            j = (MEASURE_CYCLE.index(self.scan.measure) + 1) % len(MEASURE_CYCLE) \
                if self.scan.measure in MEASURE_CYCLE else 0
            self.scan.measure = MEASURE_CYCLE[j]
            self.load()
        elif k in ("+", "="):
            self.stretch = min(self.stretch + 0.1, 3.0)
            self.redraw()
        elif k == "-":
            self.stretch = max(self.stretch - 0.1, 0.1)
            self.redraw()
        elif k == "u" and self.undo_stack:
            key, (t, v) = self.undo_stack.pop()
            if len(t):
                self.vf[key] = VelocityFunction(t, v)
            elif key in self.vf:
                del self.vf.functions[key]
            self.redraw()
        elif k == "c":
            self.set_picks([], [])
            self.redraw()
        elif k == "s":
            self.vf.save_csv(self.save_path)
            self.fig.suptitle(f"saved {len(self.vf)} locations -> {self.save_path}")
            self.fig.canvas.draw_idle()
