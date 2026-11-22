"""Inference: neural probability map -> physics-constrained DP decoding."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..picking.dp import DPParams, PickResult, dp_pick
from ..semblance import VelocitySpectrum
from .dataset import ML_DT, ML_VELS, to_image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class NeuralPicker:
    def __init__(self, model, mc_samples: int = 0):
        self.model = model
        self.mc = mc_samples

    @classmethod
    def load(cls, path, mc_samples=0):
        from .train import load
        return cls(load(path), mc_samples)

    def probability(self, sem: VelocitySpectrum, ab: VelocitySpectrum | None = None):
        """Returns (P[T, V] on ML grid, t_grid, sigma_v[T] uncertainty in m/s)."""
        x, tg, T = to_image(sem, ab)
        xt = torch.from_numpy(x[None])
        self.model.eval()
        with torch.no_grad():
            P = torch.softmax(self.model(xt), -1)[0].numpy()[:T]
        sigma = np.zeros(T)
        if self.mc > 0:
            # MC-dropout: enable dropout only
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.Dropout2d):
                    mod.train()
            ev = []
            with torch.no_grad():
                for _ in range(self.mc):
                    p = torch.softmax(self.model(xt), -1)[0].numpy()[:T]
                    ev.append((p * ML_VELS).sum(-1))
            self.model.eval()
            sigma = np.std(ev, axis=0)
        # aleatoric spread of the distribution itself
        mean = (P * ML_VELS).sum(-1)
        spread = np.sqrt((P * (ML_VELS[None] - mean[:, None]) ** 2).sum(-1))
        return P, tg, np.sqrt(sigma ** 2 + spread ** 2)

    def score_map(self, sem: VelocitySpectrum, ab: VelocitySpectrum | None = None):
        """Probability resampled onto the spectrum's own (nv, nt) grid - usable as
        the `score` argument of dp_pick / score_fn of autopick_line."""
        P, tg, _ = self.probability(sem, ab)
        # velocity axis
        Pv = np.array([np.interp(sem.velocities, ML_VELS, row) for row in P])  # (T, nv)
        # time axis
        S = np.array([np.interp(sem.times, tg, Pv[:, j]) for j in range(Pv.shape[1])])
        return np.clip(S, 0, None)

    def pick(self, sem: VelocitySpectrum, ab: VelocitySpectrum | None = None,
             params: DPParams | None = None, blend: float = 0.15, guide=None):
        """Hybrid pick: DP on (1-blend)*P_nn + blend*semblance  (blend keeps the
        decoder anchored to the measured coherence)."""
        p = params or DPParams()
        p = replace(p, power=1.0, row_norm_mix=0.0)
        nn = self.score_map(sem, ab)
        nn = nn / max(nn.max(), 1e-12)
        s = np.maximum(sem.values, 0) ** 2
        s = s / max(s.max(), 1e-12)
        score = (1 - blend) * nn + blend * s
        return dp_pick(sem, p, guide=guide, score=score)


def neural_score_fn(neural: "NeuralPicker", dataset, scan=None, blend: float = 0.15):
    """Score function for picking.line.autopick_line: computes the AB spectrum on
    the same conditioned supergather and returns the blended NN score map."""
    from .. import precondition
    from ..gather import offset_regularize
    from ..picking.line import ScanParams
    from ..semblance import velocity_scan
    sp = scan or ScanParams()

    def fn(spec):
        g = precondition.scan_flow(dataset.supergather(spec.key, sp.supergather),
                                   gain_power=sp.gain_power, band=sp.band,
                                   agc_window=sp.agc_window)
        if sp.offset_bin:
            g = offset_regularize(g, sp.offset_bin)
        ab = velocity_scan(g, spec.velocities, "ab", window=sp.window)
        s = neural.score_map(spec, ab)
        s = s / max(s.max(), 1e-12)
        return (1 - blend) * s + blend * np.maximum(spec.values, 0) ** 2
    return fn
