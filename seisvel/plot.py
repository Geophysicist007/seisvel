"""Static QC figures (matplotlib)."""
from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def gather(ax, g, clip_pct=98, title=None):
    c = np.percentile(np.abs(g.data), clip_pct) + 1e-12
    ax.imshow(g.data.T, aspect="auto", cmap="gray_r", vmin=-c, vmax=c,
              extent=[0, g.ntr, g.times[-1], g.times[0]])
    ax.set_xlabel("trace")
    if title:
        ax.set_title(title)


def spectrum(ax, spec, curves=(), picks=None, title=None):
    ax.imshow(spec.values.T, aspect="auto", cmap="jet",
              extent=[spec.velocities[0], spec.velocities[-1], spec.times[-1],
                      spec.times[0]])
    for v, style, label in curves:
        ax.plot(v, spec.times, style, lw=1.6, label=label)
    if picks is not None:
        ax.plot(picks.velocities, picks.times, "o", mfc="w", mec="k", ms=6)
    if curves:
        ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("velocity (m/s)")
    ax.set_title(title or spec.measure)


def section(ax, data, keys, dt, t0=0.0, clip_pct=99, title=None, cmap="gray_r"):
    c = np.percentile(np.abs(data), clip_pct) + 1e-12
    ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-c, vmax=c,
              extent=[keys[0], keys[-1], t0 + dt * data.shape[1], t0])
    ax.set_xlabel("CMP")
    ax.set_ylabel("t0 (s)")
    if title:
        ax.set_title(title)


def velocity_section(ax, vgrid, keys, t, title=None, vmin=None, vmax=None):
    im = ax.imshow(vgrid.T, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax,
                   extent=[keys[0], keys[-1], t[-1], t[0]])
    ax.set_xlabel("CMP")
    if title:
        ax.set_title(title)
    return im
