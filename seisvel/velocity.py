"""Velocity functions (picks at one CMP) and velocity fields (picks along a line
or over a 3-D survey), Dix conversion and physical-plausibility checks."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


@dataclass
class VelocityFunction:
    """Time-velocity pairs picked at one analysis location."""
    times: np.ndarray                     # seconds, increasing
    velocities: np.ndarray                # m/s (stacking / NMO / RMS)
    etas: np.ndarray | None = None        # optional anellipticity per pick
    confidence: np.ndarray | None = None  # optional 0..1 per pick (auto pickers)

    def __post_init__(self):
        t = np.asarray(self.times, float)
        v = np.asarray(self.velocities, float)
        o = np.argsort(t, kind="stable")
        self.times, self.velocities = t[o], v[o]
        if self.etas is not None:
            self.etas = np.asarray(self.etas, float)[o]
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, float)[o]

    def __len__(self):
        return len(self.times)

    # ------------------------------------------------------------ evaluation
    def interpolate(self, t_axis, mode: str = "vrms") -> np.ndarray:
        """Velocity on a time axis.  mode='vrms': linear in Vrms (industry default);
        mode='vint': linear in V^2 t (constant interval velocity between picks,
        which is physically consistent and never produces wiggles)."""
        t_axis = np.asarray(t_axis, float)
        if len(self) == 0:
            raise ValueError("empty velocity function")
        if len(self) == 1:
            return np.full_like(t_axis, self.velocities[0])
        if mode == "vrms":
            return np.interp(t_axis, self.times, self.velocities)
        if mode == "vint":
            tt = np.concatenate([[0.0], self.times]) if self.times[0] > 0 else self.times
            vv = np.concatenate([[self.velocities[0]], self.velocities]) \
                if self.times[0] > 0 else self.velocities
            u = vv ** 2 * tt
            ui = np.interp(t_axis, tt, u)
            # beyond last pick: keep last interval velocity
            last = t_axis > tt[-1]
            if np.any(last) and len(tt) > 1:
                vint2 = max((u[-1] - u[-2]) / max(tt[-1] - tt[-2], 1e-9), 1.0)
                ui[last] = u[-1] + vint2 * (t_axis[last] - tt[-1])
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.sqrt(np.maximum(ui, 0) / np.maximum(t_axis, 1e-9))
            v[t_axis <= tt[0]] = vv[0]
            return v
        raise ValueError("mode must be 'vrms' or 'vint'")

    def interpolate_eta(self, t_axis) -> np.ndarray:
        if self.etas is None:
            return np.zeros(len(np.atleast_1d(t_axis)))
        return np.interp(t_axis, self.times, self.etas)

    # ------------------------------------------------------------------ Dix
    def dix(self):
        """Interval velocities between consecutive picks (Dix 1955).
        Returns (t_top, t_base, vint) - vint is NaN where non-physical."""
        t, v = self.times, self.velocities
        if len(t) < 2:
            return np.array([]), np.array([]), np.array([])
        num = v[1:] ** 2 * t[1:] - v[:-1] ** 2 * t[:-1]
        dt = t[1:] - t[:-1]
        with np.errstate(invalid="ignore"):
            vint = np.sqrt(np.where(num > 0, num / dt, np.nan))
        return t[:-1], t[1:], vint

    def check(self, vint_min: float = 1400.0, vint_max: float = 7000.0,
              min_dt: float = 0.02) -> list[str]:
        """Human-readable warnings, the same checks a processor does by eye."""
        warn = []
        tt, tb, vi = self.dix()
        for a, b, x in zip(tt, tb, vi):
            if not np.isfinite(x):
                warn.append(f"{a:.3f}-{b:.3f}s: imaginary Dix velocity (too fast decrease)")
            elif x < vint_min:
                warn.append(f"{a:.3f}-{b:.3f}s: Vint {x:.0f} < {vint_min:.0f} m/s")
            elif x > vint_max:
                warn.append(f"{a:.3f}-{b:.3f}s: Vint {x:.0f} > {vint_max:.0f} m/s")
        d = np.diff(self.times)
        for k in np.flatnonzero(d < min_dt):
            warn.append(f"picks at {self.times[k]:.3f}s and {self.times[k+1]:.3f}s "
                        f"closer than {min_dt*1e3:.0f} ms")
        return warn

    def to_dict(self):
        d = dict(times=self.times.tolist(), velocities=self.velocities.tolist())
        if self.etas is not None:
            d["etas"] = self.etas.tolist()
        if self.confidence is not None:
            d["confidence"] = self.confidence.tolist()
        return d

    @classmethod
    def from_dense(cls, t_axis, v_axis, t_picks):
        """Sample a dense curve at given pick times."""
        return cls(np.asarray(t_picks), np.interp(t_picks, t_axis, v_axis))


def rms_from_interval(t_axis, vint):
    """Vrms(t0) from an interval-velocity trace sampled on t_axis (two-way time)."""
    t_axis = np.asarray(t_axis, float)
    vint = np.asarray(vint, float)
    dt = np.diff(t_axis, prepend=0.0)
    cum = np.cumsum(vint ** 2 * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.sqrt(cum / np.maximum(t_axis, 1e-12))
    v[t_axis <= 0] = vint[0]
    return v


def dix_dense(t_axis, vrms, smooth: int = 1):
    """Dense Dix conversion of a dense Vrms(t0) curve (NaN where imaginary)."""
    u = np.asarray(vrms, float) ** 2 * np.asarray(t_axis, float)
    if smooth > 1:
        from scipy.ndimage import uniform_filter1d
        u = uniform_filter1d(u, smooth, mode="nearest")
    du = np.gradient(u, t_axis)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.where(du > 0, du, np.nan))


# ---------------------------------------------------------------------------
@dataclass
class VelocityField:
    """Collection of VelocityFunctions keyed by CMP (int) or (inline, xline)."""
    functions: dict[Hashable, VelocityFunction] = field(default_factory=dict)
    mode: str = "vrms"

    def __setitem__(self, key, vf: VelocityFunction):
        self.functions[key] = vf

    def __getitem__(self, key):
        return self.functions[key]

    def __contains__(self, key):
        return key in self.functions

    def __len__(self):
        return len(self.functions)

    @property
    def keys(self):
        return sorted(self.functions.keys())

    def _is3d(self):
        k = next(iter(self.functions))
        return isinstance(k, tuple) and len(k) == 2

    # ---------------------------------------------------------- interpolation
    def at(self, key, t_axis, with_eta=False):
        """Velocity (and eta) profile at any CMP: picks at analysis locations are
        evaluated in time first, then interpolated linearly in space at
        constant t0 (the standard processing-system behaviour)."""
        if not self.functions:
            raise ValueError("empty velocity field")
        t_axis = np.asarray(t_axis, float)
        if key in self.functions:
            vf = self.functions[key]
            v = vf.interpolate(t_axis, self.mode)
            return (v, vf.interpolate_eta(t_axis)) if with_eta else v
        if self._is3d():
            return self._at3d(key, t_axis, with_eta)
        ks = np.asarray(self.keys, float)
        x = float(key)
        if x <= ks[0] or x >= ks[-1] or len(ks) == 1:
            kk = self.keys[0] if x <= ks[0] else self.keys[-1]
            vf = self.functions[kk]
            v = vf.interpolate(t_axis, self.mode)
            return (v, vf.interpolate_eta(t_axis)) if with_eta else v
        j = int(np.searchsorted(ks, x))
        k0, k1 = self.keys[j - 1], self.keys[j]
        w = (x - ks[j - 1]) / (ks[j] - ks[j - 1])
        f0, f1 = self.functions[k0], self.functions[k1]
        v = (1 - w) * f0.interpolate(t_axis, self.mode) + w * f1.interpolate(t_axis, self.mode)
        if with_eta:
            e = (1 - w) * f0.interpolate_eta(t_axis) + w * f1.interpolate_eta(t_axis)
            return v, e
        return v

    def _at3d(self, key, t_axis, with_eta):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = np.array(self.keys, float)
        V = np.array([self.functions[k].interpolate(t_axis, self.mode) for k in self.keys])
        q = np.array([key], float)
        if len(pts) >= 3:
            try:
                v = LinearNDInterpolator(pts, V)(q)[0]
                if np.all(np.isfinite(v)):
                    if with_eta:
                        E = np.array([self.functions[k].interpolate_eta(t_axis)
                                      for k in self.keys])
                        return v, LinearNDInterpolator(pts, E)(q)[0]
                    return v
            except Exception:  # degenerate (collinear) geometry
                pass
        v = NearestNDInterpolator(pts, V)(q)[0]
        if with_eta:
            E = np.array([self.functions[k].interpolate_eta(t_axis) for k in self.keys])
            return v, NearestNDInterpolator(pts, E)(q)[0]
        return v

    def grid(self, keys, t_axis):
        """Velocity section (ncmp, nt) for display / migration input."""
        return np.array([self.at(k, t_axis) for k in keys])

    # ------------------------------------------------------------------ I/O
    def save_csv(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            three = self._is3d() if self.functions else False
            hdr = (["inline", "xline"] if three else ["cmp"]) + \
                ["time_ms", "velocity", "eta", "confidence"]
            w.writerow(hdr)
            for k in self.keys:
                vf = self.functions[k]
                for i in range(len(vf)):
                    kk = list(k) if three else [k]
                    w.writerow(kk + [round(vf.times[i] * 1e3, 3),
                                     round(float(vf.velocities[i]), 2),
                                     "" if vf.etas is None else round(float(vf.etas[i]), 4),
                                     "" if vf.confidence is None else
                                     round(float(vf.confidence[i]), 3)])

    @classmethod
    def load_csv(cls, path, mode="vrms"):
        rows = {}
        with open(path) as fh:
            r = csv.DictReader(fh)
            for row in r:
                k = (int(row["inline"]), int(row["xline"])) if "inline" in row \
                    else int(float(row["cmp"]))
                rows.setdefault(k, []).append(row)
        vfld = cls(mode=mode)
        for k, rr in rows.items():
            t = [float(x["time_ms"]) / 1e3 for x in rr]
            v = [float(x["velocity"]) for x in rr]
            e = [float(x["eta"]) for x in rr] if rr[0].get("eta") not in (None, "") else None
            c = [float(x["confidence"]) for x in rr] \
                if rr[0].get("confidence") not in (None, "") else None
            vfld[k] = VelocityFunction(t, v, e, c)
        return vfld

    def save_json(self, path):
        three = self._is3d() if self.functions else False
        out = dict(mode=self.mode, functions=[
            dict(key=list(k) if three else k, **self.functions[k].to_dict())
            for k in self.keys])
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)

    @classmethod
    def load_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        f = cls(mode=d.get("mode", "vrms"))
        for item in d["functions"]:
            k = tuple(item["key"]) if isinstance(item["key"], list) else item["key"]
            f[k] = VelocityFunction(item["times"], item["velocities"], item.get("etas"),
                                    item.get("confidence"))
        return f
