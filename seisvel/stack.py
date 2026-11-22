"""NMO + stack of a full line / survey."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .gather import Gather
from .moveout import nmo_correct
from .velocity import VelocityField


def stack_gather(g: Gather, method: str = "mean", trim: float = 0.1,
                 min_fold: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Stack an NMO-corrected gather.  Fold is counted per sample after mute
    (zero samples = muted), so the shallow part is not diluted by muted traces.

    method: 'mean' | 'trimmed' (alpha-trimmed mean, robust to spikes/bad traces)
            | 'median'
    """
    d = g.data
    live = d != 0.0
    fold = live.sum(axis=0)
    if method == "mean":
        s = d.sum(axis=0) / np.maximum(fold, 1)
    else:
        masked = np.where(live, d, np.nan)
        if method == "median":
            with np.errstate(all="ignore"):
                s = np.nanmedian(masked, axis=0)
        elif method == "trimmed":
            srt = np.sort(masked, axis=0)            # NaNs go last
            s = np.zeros(d.shape[1])
            for it in range(d.shape[1]):
                n = fold[it]
                if n == 0:
                    continue
                k = int(np.floor(trim * n))
                s[it] = np.mean(srt[k:n - k, it]) if n - 2 * k > 0 else np.mean(srt[:n, it])
        else:
            raise ValueError(method)
        s = np.nan_to_num(s)
    s[fold < min_fold] = 0.0
    return s.astype(np.float32), fold


def stack_line(dataset, vfield: VelocityField, keys: Sequence | None = None,
               preprocess: Callable[[Gather], Gather] | None = None,
               stretch_max: float = 0.5, method: str = "mean", use_eta: bool = True,
               progress: bool = False):
    """NMO-correct and stack every CMP. Returns (stack[ncmp, nt], fold[ncmp, nt], keys)."""
    keys = list(dataset.keys if keys is None else keys)
    out = np.zeros((len(keys), dataset.nt), np.float32)
    fold = np.zeros((len(keys), dataset.nt), np.int32)
    t = dataset.t0 + dataset.dt * np.arange(dataset.nt)
    for i, k in enumerate(keys):
        g = dataset.gather(k)
        if preprocess is not None:
            g = preprocess(g)
        v, e = vfield.at(k, t, with_eta=True)
        gn = nmo_correct(g, v, e if use_eta else 0.0, stretch_max=stretch_max)
        out[i], fold[i] = stack_gather(gn, method)
        if progress and i % 50 == 0:
            print(f"stack {i}/{len(keys)}")
    return out, fold, keys
