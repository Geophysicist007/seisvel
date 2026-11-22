"""Parabolic Radon transform and Radon demultiple (Hampson 1986; high-resolution
version after Sacchi & Ulrych 1995).

After NMO with *primary* velocities, primaries are flat (q ~ 0) and multiples,
which travel slower, are under-corrected and curve downward with a roughly
parabolic residual moveout  t = tau + q (x / x_ref)^2,  q > 0.  The data are
modelled in the (tau, q) domain, the multiple region q > q_cut is kept as the
multiple model and subtracted from the input:  primaries = d - L m_mult.

Least squares per frequency:  m = (L^H L + mu W)^-1 L^H d
High resolution: W is updated from |m| (iteratively re-weighted least squares),
which focuses energy and separates primaries from multiples with small
moveout differences much better than plain damped LS.
"""
from __future__ import annotations

import numpy as np

from .gather import Gather
from .moveout import inverse_nmo, nmo_correct


def _operators(offsets, qs, x_ref, freqs):
    phi = (np.asarray(offsets, float) / x_ref) ** 2           # (nx,)
    # L[f, x, q] = exp(-i w q phi_x)
    return np.exp(-2j * np.pi * freqs[:, None, None] * qs[None, None, :] *
                  phi[None, :, None])


def radon_forward_adjoint(g: Gather, qs, x_ref=None, fmax=None, mu=0.05,
                          high_res=False, n_iter=2, eps=0.05):
    """Invert data for the (q, tau) model.  Returns model M (nq, nt) and a
    function to synthesise data from any masked model."""
    x_ref = x_ref or max(np.abs(g.offsets).max(), 1.0)
    nt = g.nt
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    D = np.fft.rfft(g.data.astype(np.float64), nfft, axis=1)     # (nx, nf)
    f = np.fft.rfftfreq(nfft, g.dt)
    fmax = fmax or 0.8 * f[-1]
    use = np.flatnonzero((f > 0) & (f <= fmax))
    qs = np.asarray(qs, float)
    nq = len(qs)
    M = np.zeros((nq, len(f)), complex)
    off = np.abs(g.offsets)
    for chunk in np.array_split(use, max(1, len(use) // 64)):
        L = _operators(off, qs, x_ref, f[chunk])                  # (c, nx, nq)
        Lh = np.conj(np.transpose(L, (0, 2, 1)))                   # (c, nq, nx)
        LhL = Lh @ L                                               # (c, nq, nq)
        Lhd = (Lh @ D[:, chunk].T[:, :, None])[..., 0]             # (c, nq)
        scale = np.real(np.trace(LhL, axis1=1, axis2=2)) / nq      # (c,)
        I = np.eye(nq)[None]
        m = np.linalg.solve(LhL + mu * scale[:, None, None] * I, Lhd[..., None])[..., 0]
        if high_res:
            for _ in range(n_iter):
                a = np.abs(m)
                # weights shared across the chunk (frequency-consistent focusing)
                w = np.mean(a, axis=0)
                w = w / (w.max() + 1e-30)
                Wd = 1.0 / (w ** 2 + eps)
                Wd = Wd / Wd.min()       # strongest coefficients keep damping mu
                m = np.linalg.solve(LhL + mu * scale[:, None, None] * Wd[None, :, None] * I,
                                    Lhd[..., None])[..., 0]
        M[:, chunk] = m.T

    def synth(mask_q):
        """Data predicted by the model restricted to the q-weights mask_q (nq,)."""
        Dm = np.zeros_like(D)
        for chunk in np.array_split(use, max(1, len(use) // 64)):
            L = _operators(off, qs, x_ref, f[chunk])
            Dm[:, chunk] = (L @ (M[:, chunk].T * mask_q[None, :])[..., None])[..., 0].T
        return np.fft.irfft(Dm, nfft, axis=1)[:, :nt]

    tau_model = np.fft.irfft(M, nfft, axis=1)[:, :nt]
    return tau_model, synth, x_ref


def radon_demultiple(g_nmo: Gather, q_cut: float = 0.02, q_min: float = -0.1,
                     q_max: float = 1.5, nq: int = 201, taper: float = 0.01,
                     **kw) -> tuple[Gather, Gather, np.ndarray]:
    """Demultiple an NMO-corrected gather.

    q values are residual moveout (s) at the reference (= maximum) offset.
    Energy with q > q_cut (+ taper) is treated as multiple and subtracted.
    Returns (primaries, multiple_model, radon_panel).
    """
    qs = np.linspace(q_min, q_max, nq)
    panel, synth, _ = radon_forward_adjoint(g_nmo, qs, **kw)
    mask = np.clip((qs - q_cut) / max(taper, 1e-9), 0.0, 1.0)
    mult = synth(mask).astype(np.float32)
    live = g_nmo.data != 0                       # keep the mute
    mult *= live
    prim = (g_nmo.data - mult).astype(np.float32)
    return g_nmo.copy(prim), g_nmo.copy(mult), panel


def demultiple_gather(g: Gather, velocity, eta=0.0, q_cut=0.02, stretch_max=1.0,
                      protect_until: float | None = None, **kw):
    """Raw (un-NMO'd) gather -> demultipled raw gather, using primary velocities:
    NMO (with a generous stretch mute), Radon demultiple, inverse NMO.  Samples
    inside the stretch-mute / crossover zone are returned untouched (the Radon
    model is unreliable there), so nothing is silently zeroed.

    protect_until : zero-offset time (s) above which nothing is removed.  For
        marine data use ~2 x water-bottom time: no surface multiple can arrive
        earlier, so the water bottom and shallow primaries are never touched."""
    gn = nmo_correct(g, velocity, eta, stretch_max=stretch_max, taper=0.0)
    prim, mult, _ = radon_demultiple(gn, q_cut=q_cut, **kw)
    if protect_until:
        ramp = np.clip((g.times - protect_until) / 0.05, 0.0, 1.0).astype(np.float32)
        mult = mult.copy(mult.data * ramp[None, :])
    mult_raw = inverse_nmo(mult, velocity, eta)
    cover = inverse_nmo(gn.copy((gn.data != 0).astype(np.float32)), velocity, eta,
                        interp="linear").data
    w = np.clip(cover, 0.0, 1.0)
    # soft edge (~60 ms ramp) at the mute boundary: avoids a hard staircase
    from scipy.ndimage import uniform_filter1d
    n = max(3, int(round(0.06 / g.dt)) | 1)
    w = w * np.clip(2 * uniform_filter1d(w, n, axis=1, mode="nearest") - 1, 0, 1)
    out = g.data - w * mult_raw.data
    return g.copy(out.astype(np.float32))


class DemultipledDataset:
    """Wraps a dataset; every gather is Radon-demultipled on the fly with the
    velocity field of a previous pass (cached)."""

    def __init__(self, dataset, vfield, q_cut=0.02, cache=True, water_bottom=None,
                 v_water=1500.0, **kw):
        """water_bottom: None, 'auto' (detect per gather) or t0 in s -> protects
        everything above 0.95 x 2 x t_wb from the subtraction."""
        self.ds, self.vf, self.q_cut, self.kw = dataset, vfield, q_cut, kw
        self.water_bottom, self.v_water = water_bottom, v_water
        self.keys, self.dt, self.nt, self.t0 = dataset.keys, dataset.dt, dataset.nt, dataset.t0
        self.is3d = getattr(dataset, "is3d", False)
        self._cache = {} if cache else None

    def gather(self, key):
        if self._cache is not None and key in self._cache:
            return self._cache[key].copy()
        g = self.ds.gather(key)
        t = g.times
        v, e = self.vf.at(key, t, with_eta=True)
        prot = None
        if self.water_bottom == "auto":
            from .precondition import detect_water_bottom
            twb = detect_water_bottom(g, water_velocity=self.v_water)
            prot = 1.9 * twb if twb else None
        elif self.water_bottom:
            prot = 1.9 * float(self.water_bottom)
        out = demultiple_gather(g, v, e, self.q_cut, protect_until=prot, **self.kw)
        if self._cache is not None:
            self._cache[key] = out
        return out.copy()

    def fold(self, key):
        return self.ds.fold(key)

    def neighbours(self, key, half_width):
        return self.ds.neighbours(key, half_width)

    def supergather(self, key, half_width=2):
        from .gather import combine
        return combine([self.gather(k) for k in self.neighbours(key, half_width)], key)

    def analysis_keys(self, every):
        return self.ds.analysis_keys(every)

    def __iter__(self):
        for k in self.keys:
            yield self.gather(k)
