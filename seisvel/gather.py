"""CMP gather container and gather-level utilities.

Convention used everywhere in seisvel:
    data      : float32 array, shape (ntraces, nsamples)   (trace-major, like SEG-Y)
    offsets   : float64 array, shape (ntraces,)            (metres, signed allowed)
    dt        : sample interval in seconds
    t0        : time of first sample in seconds (delay recording time)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

import numpy as np


@dataclass
class Gather:
    data: np.ndarray
    offsets: np.ndarray
    dt: float
    t0: float = 0.0
    key: Hashable = None                  # CMP number (2D) or (inline, xline) (3D)
    headers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        self.offsets = np.asarray(self.offsets, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("Gather.data must be 2-D (ntraces, nsamples)")
        if self.data.shape[0] != self.offsets.shape[0]:
            raise ValueError(
                f"{self.data.shape[0]} traces but {self.offsets.shape[0]} offsets")

    # ------------------------------------------------------------------ basics
    @property
    def ntr(self) -> int:
        return self.data.shape[0]

    @property
    def nt(self) -> int:
        return self.data.shape[1]

    @property
    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.nt)

    def copy(self, data: np.ndarray | None = None) -> "Gather":
        return Gather(self.data.copy() if data is None else data, self.offsets.copy(),
                      self.dt, self.t0, self.key,
                      {k: v.copy() for k, v in self.headers.items()})

    def subset(self, idx: np.ndarray) -> "Gather":
        return Gather(self.data[idx], self.offsets[idx], self.dt, self.t0, self.key,
                      {k: v[idx] for k, v in self.headers.items()})

    # ------------------------------------------------------------- cleaning
    def sort_by_offset(self, absolute: bool = True) -> "Gather":
        o = np.abs(self.offsets) if absolute else self.offsets
        return self.subset(np.argsort(o, kind="stable"))

    def live_mask(self, rel_threshold: float = 1e-6) -> np.ndarray:
        """True for traces that are not dead (all-zero / NaN / tiny energy)."""
        d = self.data
        finite = np.all(np.isfinite(d), axis=1)
        rms = np.sqrt(np.mean(np.where(np.isfinite(d), d, 0.0) ** 2, axis=1))
        ref = np.median(rms[rms > 0]) if np.any(rms > 0) else 0.0
        return finite & (rms > rel_threshold * max(ref, 1e-30))

    def drop_dead(self) -> "Gather":
        return self.subset(np.flatnonzero(self.live_mask()))


# ---------------------------------------------------------------------------
# Supergathers and offset regularisation (industry-standard robustness tools)
# ---------------------------------------------------------------------------

def combine(gathers: Iterable[Gather], key: Hashable = None) -> Gather:
    """Concatenate neighbouring CMP gathers into a *supergather*.

    Supergathers raise fold for velocity analysis on noisy / low-fold data.
    Valid when the velocity varies slowly over the combined CMP range.
    """
    gathers = list(gathers)
    if not gathers:
        raise ValueError("no gathers to combine")
    g0 = gathers[0]
    for g in gathers[1:]:
        if g.nt != g0.nt or abs(g.dt - g0.dt) > 1e-9 or abs(g.t0 - g0.t0) > 1e-9:
            raise ValueError("gathers have different time axes")
    data = np.concatenate([g.data for g in gathers], axis=0)
    off = np.concatenate([g.offsets for g in gathers])
    return Gather(data, off, g0.dt, g0.t0, g0.key if key is None else key)


def offset_regularize(g: Gather, bin_size: float, max_offset: float | None = None,
                      ) -> Gather:
    """Average traces into regular absolute-offset bins.

    Removes the bias of uneven offset sampling in semblance (many near traces,
    few far traces), cuts computation on high-fold supergathers, and averages
    down random noise.  Empty bins are dropped (not zero-filled), so they don't
    dilute the semblance normalisation.
    """
    off = np.abs(g.offsets)
    if max_offset is None:
        max_offset = off.max() if off.size else 0.0
    nb = int(np.floor(max_offset / bin_size)) + 1
    b = np.clip(np.floor(off / bin_size).astype(int), 0, nb - 1)
    out, centers = [], []
    live = g.live_mask()
    for k in range(nb):
        idx = np.flatnonzero((b == k) & live)
        if idx.size == 0:
            continue
        out.append(g.data[idx].mean(axis=0))
        centers.append(off[idx].mean())
    if not out:
        return g.copy()
    return Gather(np.vstack(out), np.asarray(centers), g.dt, g.t0, g.key)


def resample_time(g: Gather, new_dt: float) -> Gather:
    """Anti-aliased decimation in time (used to speed up velocity scans)."""
    from scipy.signal import resample_poly
    ratio = new_dt / g.dt
    if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
        raise ValueError("new_dt must be an integer multiple of dt")
    q = int(round(ratio))
    if q == 1:
        return g.copy()
    d = resample_poly(g.data, 1, q, axis=1).astype(np.float32)
    return Gather(d, g.offsets.copy(), new_dt, g.t0, g.key)


def info(g: Gather) -> dict[str, Any]:
    o = np.abs(g.offsets)
    return dict(key=g.key, ntr=g.ntr, nt=g.nt, dt_ms=g.dt * 1e3,
                tmax_s=float(g.times[-1]), min_offset=float(o.min()),
                max_offset=float(o.max()), dead=int((~g.live_mask()).sum()))
