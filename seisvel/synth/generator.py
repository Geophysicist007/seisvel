"""Physically-based synthetic CMP gathers for testing and for training the AI picker.

What is modelled (1.5-D: a local 1-D earth at each CMP, laterally varying):
  * exact ray-traced traveltimes through isotropic layers (Snell's law) ->
    genuine non-hyperbolic moveout on long spreads
  * angle-dependent reflectivity (Aki-Richards), so class I/II/III AVO and
    polarity reversals appear naturally (e.g. gas sands)
  * marine surface multiples: water-bottom multiples + peg-legs (the classic
    'low-velocity' trend that fools naive pickers)
  * spherical divergence and constant-Q absorption (frequency drops with time)
  * noise: band-limited random noise, ground roll / linear noise, swell bursts,
    spikes, dead and polarity-flipped traces, trace-amplitude jitter, irregular
    and missing offsets

Not modelled (be honest about the gap to field data): dip/diffractions (no DMO
effect), converted waves, statics, 3-D scattering, source-signature variation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gather import Gather
from ..velocity import rms_from_interval


# ---------------------------------------------------------------------------
@dataclass
class LayeredModel:
    h: np.ndarray        # thickness (m), last layer = half-space (h ignored)
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    water: bool = False  # first layer is water

    @property
    def n(self):
        return len(self.vp)

    def t0_interfaces(self):
        """Two-way vertical time to the base of each layer (interface k = base of k)."""
        return np.cumsum(2 * self.h[:-1] / self.vp[:-1])

    def vint_trace(self, t_axis):
        tb = self.t0_interfaces()
        idx = np.searchsorted(tb, t_axis, side="right")
        return self.vp[np.minimum(idx, self.n - 1)]

    def vrms(self, t_axis):
        return rms_from_interval(t_axis, self.vint_trace(t_axis))


def _ray_trace(legs, h, v, x, n_iter=60):
    """Solve X(p)=x for a ray with `legs[j]` one-way passes through layer j.
    Returns traveltime T(x) and ray parameter p(x)."""
    m = legs > 0
    L, H, V = legs[m].astype(float), h[m], v[m]
    vmax = V.max()
    x = np.abs(np.asarray(x, float))
    lo = np.zeros_like(x)
    hi = np.full_like(x, 1.0 - 1e-9)          # u = p * vmax

    def X(u):
        p = u[:, None] / vmax
        s = p * V[None, :]
        return (L * H * s / np.sqrt(1 - s ** 2)).sum(1)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        big = X(mid) > x
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    p = 0.5 * (lo + hi) / vmax
    s = p[:, None] * V[None, :]
    T = (L * H / (V * np.sqrt(1 - s ** 2))).sum(1)
    return T, p


def _zero_offset_time(legs, model):
    return float(np.sum(legs * model.h / model.vp))


def aki_richards(vp1, vs1, r1, vp2, vs2, r2, theta):
    """P-P reflection coefficient, Aki-Richards linearised (theta = incidence angle)."""
    a, b, r = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (r1 + r2)
    da, db, dr = vp2 - vp1, vs2 - vs1, r2 - r1
    s2 = np.sin(theta) ** 2
    k = (b / a) ** 2 if a > 0 else 0.0
    return (0.5 * (1 - 4 * k * s2) * dr / r + 0.5 / np.cos(theta) ** 2 * da / a
            - 4 * k * s2 * (db / b if b > 0 else 0.0))


def ricker_spectrum(f, fpeak):
    return (2 / np.sqrt(np.pi)) * f ** 2 / fpeak ** 3 * np.exp(-(f / fpeak) ** 2)


# ---------------------------------------------------------------------------
@dataclass
class Acquisition:
    nt: int = 1500
    dt: float = 0.004
    min_offset: float = 100.0
    max_offset: float = 3000.0
    ntr: int = 60
    offset_jitter: float = 0.0      # metres (random)
    missing_fraction: float = 0.0   # fraction of traces removed

    def offsets(self, rng):
        off = np.linspace(self.min_offset, self.max_offset, self.ntr)
        if self.offset_jitter:
            off = np.abs(off + rng.normal(0, self.offset_jitter, off.size))
        if self.missing_fraction:
            keep = rng.random(off.size) >= self.missing_fraction
            keep[0] = keep[-1] = True
            off = off[keep]
        return np.sort(off)


@dataclass
class NoiseSpec:
    snr: float = 4.0                 # signal-rms / noise-rms (after gain)
    ground_roll: float = 0.0         # amplitude relative to signal rms (land)
    swell: float = 0.0               # marine swell bursts amplitude
    spikes: int = 0
    dead_fraction: float = 0.0
    flipped_fraction: float = 0.0
    amp_jitter: float = 0.0          # log-normal trace scaling sigma


def synthesize_gather(model: LayeredModel, acq: Acquisition, rng, *, fpeak=25.0,
                      q=120.0, multiples=1.0, n_wb_mult=3, n_peg=8,
                      noise: NoiseSpec | None = None, key=0, spreading=True,
                      scale: float | None = None):
    """Return (Gather, truth dict)."""
    off = acq.offsets(rng)
    nt, dt = acq.nt, acq.dt
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    f = np.fft.rfftfreq(nfft, dt)
    w = 2 * np.pi * f
    W = ricker_spectrum(f, fpeak)
    D = np.zeros((off.size, f.size), complex)
    events = []

    def add(legs, amp_fn, kind):
        T, p = _ray_trace(legs, model.h, model.vp, off)
        A = amp_fn(p)
        if spreading:
            # Newman (1973): amplitude ~ 1 / (v_rms^2 t)  -> normalise at 1 s, 2 km/s
            t0 = T.min()
            vr = max(model.vrms(np.array([max(t0, dt)]))[0], 1000.0)
            A = A * (2000.0 ** 2) / (vr ** 2 * np.maximum(T, dt))
        att = np.exp(-np.pi * f[None, :] * T[:, None] / q)
        D[:] += (A[:, None] * att) * np.exp(-1j * w[None, :] * T[:, None])
        ev = dict(kind=kind, t0=float(T.min()), legs=legs.copy())
        if kind == "primary":
            # best-fit hyperbola (what semblance measures): LS fit of t^2 - t0^2
            # vs x^2 over offsets where the NMO stretch stays below 100 %
            t0 = _zero_offset_time(legs, model)
            ok = (T - t0) / max(t0, 1e-6) <= 1.0
            if ok.sum() >= 3:
                x2 = off[ok] ** 2
                ev["vstack"] = float(np.sqrt(np.sum(x2 * x2) /
                                             max(np.sum(x2 * (T[ok] ** 2 - t0 ** 2)), 1e-12)))
                ev["t0"] = t0
        events.append(ev)

    n = model.n
    for k in range(n - 1):
        legs = np.zeros(n, int)
        legs[: k + 1] = 2

        def amp(p, k=k):
            th = np.arcsin(np.clip(p * model.vp[k], 0, 0.999))
            return aki_richards(model.vp[k], model.vs[k], model.rho[k],
                                model.vp[k + 1], model.vs[k + 1], model.rho[k + 1], th)
        add(legs, amp, "primary")

    if model.water and multiples > 0:
        r_wb = aki_richards(model.vp[0], model.vs[0], model.rho[0], model.vp[1],
                            model.vs[1], model.rho[1], 0.0)
        for m in range(1, n_wb_mult + 1):
            legs = np.zeros(n, int)
            legs[0] = 2 * (m + 1)
            add(legs, lambda p, m=m: multiples * (-1) ** m * r_wb ** (m + 1) *
                np.ones_like(p), f"wb_multiple_{m}")
        for k in range(1, min(n - 1, 1 + n_peg)):
            legs = np.zeros(n, int)
            legs[: k + 1] = 2
            legs[0] += 2
            r_k = aki_richards(model.vp[k], model.vs[k], model.rho[k], model.vp[k + 1],
                               model.vs[k + 1], model.rho[k + 1], 0.0)
            add(legs, lambda p, r_k=r_k: multiples * -2 * r_wb * r_k * np.ones_like(p),
                f"peg_leg_{k}")

    data = np.fft.irfft(D * W[None, :], nfft, axis=1)[:, :nt]
    t_axis = dt * np.arange(nt)
    # normalise to unit rms (or by a caller-given scale, so that variants of the
    # same model - e.g. with / without multiples - stay comparable)
    sig_rms = scale if scale is not None else np.sqrt(np.mean(data ** 2)) + 1e-12
    data = data / sig_rms

    if noise is not None:
        data = _add_noise(data, off, t_axis, dt, rng, noise, fpeak)

    g = Gather(data.astype(np.float32), off, dt, 0.0, key)
    vstack = model.vrms(t_axis)
    pe = [(e["t0"], e["vstack"]) for e in events if "vstack" in e]
    if len(pe) >= 2:
        tt, vv = np.array(pe).T
        # between events: follow the Vrms trend shifted by the local stacking bias
        bias = vv / np.interp(tt, t_axis, vstack)
        vstack = vstack * np.interp(t_axis, tt, bias)
    truth = dict(vrms=model.vrms(t_axis), vstack=vstack, vint=model.vint_trace(t_axis),
                 scale=sig_rms,
                 t0_primaries=model.t0_interfaces(), events=events)
    return g, truth


def _bandlimited(rng, shape, dt, flo, fhi):
    nfft = shape[1]
    z = rng.normal(size=shape)
    Z = np.fft.rfft(z, axis=1)
    f = np.fft.rfftfreq(nfft, dt)
    Z[:, (f < flo) | (f > fhi)] = 0
    out = np.fft.irfft(Z, nfft, axis=1)
    return out / (out.std() + 1e-12)


def _add_noise(data, off, t, dt, rng, ns: NoiseSpec, fpeak):
    ntr, nt = data.shape
    # ambient noise is stationary in the RAW domain; its level is set so that
    # the signal/noise rms ratio measured after a t^2 gain (how a processor
    # sees the data) equals ns.snr.  Deep data are therefore noisier than shallow.
    gain = np.maximum(t, 0.25) ** 2
    noise = _bandlimited(rng, (ntr, nt), dt, 3.0, 2.5 * fpeak)
    s_sig = np.sqrt(np.mean((data * gain) ** 2))
    s_noi = np.sqrt(np.mean((noise * gain) ** 2))
    data = data + noise * s_sig / (max(ns.snr, 1e-3) * s_noi)
    if ns.ground_roll > 0:
        gr = np.zeros_like(data)
        for vgr, fgr in ((rng.uniform(250, 450), rng.uniform(5, 9)),
                         (rng.uniform(500, 900), rng.uniform(8, 12))):
            for i, x in enumerate(off):
                tt = t - abs(x) / vgr
                env = np.exp(-np.clip(tt, 0, None) / 0.4) * (tt > 0)
                gr[i] += env * np.sin(2 * np.pi * fgr * tt) * 1.0 / (1 + abs(x) / 1500)
        sgr = np.sqrt(np.mean(data[:, : nt // 3] ** 2)) + 1e-12
        data = data + ns.ground_roll * sgr * gr / (np.abs(gr).max() + 1e-12) * 5
    if ns.swell > 0:
        for _ in range(max(1, ntr // 10)):
            i = rng.integers(ntr)
            burst = _bandlimited(rng, (1, nt), dt, 1, 6)[0]
            data[i] += ns.swell * np.sqrt(np.mean(data[i] ** 2)) * burst * 3
    for _ in range(ns.spikes):
        data[rng.integers(ntr), rng.integers(nt)] += rng.normal() * 50 * data.std()
    if ns.amp_jitter:
        data *= np.exp(rng.normal(0, ns.amp_jitter, (ntr, 1)))
    if ns.flipped_fraction:
        data[rng.random(ntr) < ns.flipped_fraction] *= -1
    if ns.dead_fraction:
        data[rng.random(ntr) < ns.dead_fraction] = 0.0
    return data


# ---------------------------------------------------------------------------
# random geology
# ---------------------------------------------------------------------------
def random_model(rng, *, marine=None, n_layers=None, tmax=5.5, gas_sand_prob=0.5,
                 inversion_prob=0.15):
    """Random but geologically plausible layered model (compaction trend, Gardner
    density, mudrock Vs, occasional velocity inversions and gas sands)."""
    if marine is None:
        marine = rng.random() < 0.6
    if n_layers is None:
        n_layers = int(rng.integers(8, 26))
    h, vp = [], []
    if marine:
        h.append(rng.uniform(80, 1200))
        vp.append(rng.uniform(1480, 1530))
    v = rng.uniform(1650, 2300)
    grad = rng.uniform(0.35, 1.3)          # m/s per m (compaction trend)
    t = 2 * h[0] / vp[0] if marine else 0.0
    while len(vp) < n_layers and t < tmax:
        th = rng.uniform(80, 500)
        if rng.random() < inversion_prob and len(vp) > 1:
            vlay = v * rng.uniform(0.82, 0.95)
        else:
            vlay = v
        h.append(th)
        vp.append(vlay)
        t += 2 * th / vlay
        v = min(v + grad * th * rng.uniform(0.5, 1.5), 6200)
    vp.append(min(v, 6200))
    h.append(500.0)
    vp = np.array(vp)
    h = np.array(h)
    vs = np.maximum((vp - 1360) / 1.16, 300.0)
    rho = 0.31 * vp ** 0.25 * 1000
    if marine:
        vs[0], rho[0] = 0.0, 1030.0
    # gas sand: lower Vp, much lower rho, high Vs relative -> class II/III AVO
    if rng.random() < gas_sand_prob and len(vp) > 4:
        k = int(rng.integers(2 if marine else 1, len(vp) - 1))
        vp[k] *= rng.uniform(0.85, 0.97)
        rho[k] *= rng.uniform(0.85, 0.93)
        vs[k] = vp[k] / rng.uniform(1.6, 1.8)
    return LayeredModel(h, vp, vs, rho, marine)


def perturb_model(base: LayeredModel, s: float, rng_state: dict):
    """Smooth lateral variation along a line, s in [0,1] (position)."""
    ph = rng_state["phases"]
    amp = rng_state["amps"]
    fac = 1 + amp * np.sin(2 * np.pi * (rng_state["freq"] * s) + ph)
    h = base.h * fac
    vp = base.vp * (1 + 0.03 * np.sin(2 * np.pi * s + ph * 0.5))
    if base.water:
        vp[0] = base.vp[0]
    return LayeredModel(h, vp, base.vs * vp / base.vp, base.rho, base.water)


def synthetic_line(n_cmp=200, seed=0, marine=True, acq=None, noise=None, fpeak=25.0,
                   multiples=1.0, q=120.0, first_cmp=1000, model=None,
                   return_vstack=False):
    """A 2-D line of CMP gathers with lateral variation. Returns
    (gathers, truth_vrms[n_cmp, nt], models)."""
    rng = np.random.default_rng(seed)
    acq = acq or Acquisition()
    base = model or random_model(rng, marine=marine, gas_sand_prob=1.0)
    st = dict(phases=rng.uniform(0, 2 * np.pi, base.n), amps=rng.uniform(0.02, 0.12, base.n),
              freq=rng.uniform(0.5, 1.5, base.n))
    gathers, truth, models, vst = [], [], [], []
    for i in range(n_cmp):
        m = perturb_model(base, i / max(n_cmp - 1, 1), st)
        g, tr = synthesize_gather(m, acq, rng, fpeak=fpeak, q=q, multiples=multiples,
                                  noise=noise, key=first_cmp + i)
        gathers.append(g)
        truth.append(tr["vrms"])
        vst.append(tr["vstack"])
        models.append(m)
    if return_vstack:
        return gathers, np.array(truth), models, np.array(vst)
    return gathers, np.array(truth), models
