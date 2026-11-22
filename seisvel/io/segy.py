"""SEG-Y access for real prestack data.

Real-data realities handled here:
  * traces in any order (shot-sorted field files) -> index built from headers
  * configurable header bytes (CDP 21, offset 37 by SEG-Y rev1 default; 3-D
    inline 189 / xline 193; many vendors use other bytes - override them)
  * coordinate scalar (byte 71) applied to source/receiver/CDP coordinates
  * offsets missing -> computed from source/receiver coordinates
  * memory-mapped reading, gathers read on demand (works on files >> RAM)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterator, Sequence

import numpy as np

from ..gather import Gather, combine

try:
    import segyio
    from segyio import TraceField as TF
except ImportError:  # pragma: no cover
    segyio = None


@dataclass
class HeaderMap:
    cdp: int = 21
    offset: int = 37
    inline: int | None = None        # e.g. 189 for 3-D
    xline: int | None = None         # e.g. 193 for 3-D
    sx: int = 73
    sy: int = 77
    gx: int = 81
    gy: int = 85
    scalco: int = 71
    cdp_x: int = 181
    cdp_y: int = 185


def _need_segyio():
    if segyio is None:
        raise ImportError("segyio is required for SEG-Y I/O:  pip install segyio")


class SegyDataset:
    """Random access to CMP gathers of a prestack SEG-Y file.

    >>> ds = SegyDataset("line.sgy")                      # 2-D, CDP byte 21
    >>> ds3 = SegyDataset("cube.sgy", HeaderMap(inline=189, xline=193))
    >>> g = ds.gather(ds.keys[100]);  sg = ds.supergather(ds.keys[100], 2)
    """

    def __init__(self, path: str, headers: HeaderMap | None = None,
                 endian: str = "big"):
        _need_segyio()
        self.path = path
        self.hm = headers or HeaderMap()
        self._f = segyio.open(path, "r", ignore_geometry=True, endian=endian)
        self._f.mmap()
        f = self._f
        self.dt = segyio.tools.dt(f) / 1e6
        self.nt = len(f.samples)
        self.t0 = float(f.samples[0]) / 1e3
        self.ntraces = f.tracecount
        attr = lambda b: np.asarray(f.attributes(b)[:], dtype=np.int64)
        if self.hm.inline and self.hm.xline:
            il, xl = attr(self.hm.inline), attr(self.hm.xline)
            keys = list(zip(il.tolist(), xl.tolist()))
            self.is3d = True
        else:
            keys = attr(self.hm.cdp).tolist()
            self.is3d = False
        off = attr(self.hm.offset).astype(float)
        if np.all(off == 0):
            sc = attr(self.hm.scalco).astype(float)
            scale = np.where(sc < 0, -1.0 / np.where(sc == 0, 1, sc),
                             np.where(sc == 0, 1.0, sc))
            sx, sy = attr(self.hm.sx) * scale, attr(self.hm.sy) * scale
            gx, gy = attr(self.hm.gx) * scale, attr(self.hm.gy) * scale
            off = np.hypot(gx - sx, gy - sy)
        self.offsets_all = off
        index: dict[Hashable, list[int]] = {}
        for i, k in enumerate(keys):
            index.setdefault(k, []).append(i)
        self.index = {k: np.asarray(v) for k, v in index.items()}
        self.keys = sorted(self.index)
        self._pos = {k: i for i, k in enumerate(self.keys)}

    # -------------------------------------------------------------- access
    def fold(self, key) -> int:
        return len(self.index[key])

    def gather(self, key) -> Gather:
        idx = self.index[key]
        data = np.stack([self._f.trace.raw[int(i)] for i in idx]).astype(np.float32)
        return Gather(data, self.offsets_all[idx], self.dt, self.t0, key,
                      {"trace_index": idx})

    def neighbours(self, key, half_width: int) -> list:
        if self.is3d:
            il, xl = key
            return [k for k in ((il + a, xl + b)
                                for a in range(-half_width, half_width + 1)
                                for b in range(-half_width, half_width + 1))
                    if k in self.index]
        p = self._pos[key]
        return self.keys[max(0, p - half_width): p + half_width + 1]

    def supergather(self, key, half_width: int = 2) -> Gather:
        return combine([self.gather(k) for k in self.neighbours(key, half_width)], key)

    def __iter__(self) -> Iterator[Gather]:
        for k in self.keys:
            yield self.gather(k)

    def analysis_keys(self, every: int) -> list:
        """Regular analysis locations (every n-th CMP, or n-by-n grid in 3-D)."""
        if not self.is3d:
            return self.keys[::every]
        ils = sorted({k[0] for k in self.keys})[::every]
        xls = sorted({k[1] for k in self.keys})[::every]
        return [(a, b) for a in ils for b in xls if (a, b) in self.index]

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class MemoryDataset:
    """Same interface as SegyDataset for gathers already in memory."""

    def __init__(self, gathers: Sequence[Gather]):
        self._g = {g.key: g for g in gathers}
        self.keys = sorted(self._g)
        self._pos = {k: i for i, k in enumerate(self.keys)}
        g0 = gathers[0]
        self.dt, self.nt, self.t0 = g0.dt, g0.nt, g0.t0
        self.is3d = isinstance(g0.key, tuple)

    def gather(self, key):
        return self._g[key].copy()

    def fold(self, key):
        return self._g[key].ntr

    def neighbours(self, key, half_width):
        p = self._pos[key]
        return self.keys[max(0, p - half_width): p + half_width + 1]

    def supergather(self, key, half_width=2):
        return combine([self._g[k] for k in self.neighbours(key, half_width)], key)

    def analysis_keys(self, every):
        return self.keys[::every]

    def __iter__(self):
        for k in self.keys:
            yield self.gather(k)


# ---------------------------------------------------------------------------
def write_prestack(path: str, gathers: Sequence[Gather], cmp_spacing: float = 12.5):
    """Write CMP gathers to a SEG-Y rev1 file with CDP/offset/coordinate headers."""
    _need_segyio()
    g0 = gathers[0]
    spec = segyio.spec()
    spec.format = 5
    spec.samples = (g0.t0 + g0.dt * np.arange(g0.nt)) * 1e3
    spec.tracecount = sum(g.ntr for g in gathers)
    with segyio.create(path, spec) as f:
        i = 0
        for g in gathers:
            cdpx = int(round(float(g.key) * cmp_spacing * 100))
            for k in range(g.ntr):
                off = float(g.offsets[k])
                f.header[i] = {
                    TF.TRACE_SEQUENCE_LINE: i + 1, TF.CDP: int(g.key),
                    TF.CDP_TRACE: k + 1, TF.offset: int(round(off)),
                    TF.SourceGroupScalar: -100,
                    TF.SourceX: int(cdpx - off * 50), TF.GroupX: int(cdpx + off * 50),
                    TF.CDP_X: cdpx, TF.CDP_Y: 0,
                    TF.TRACE_SAMPLE_COUNT: g.nt,
                    TF.TRACE_SAMPLE_INTERVAL: int(round(g.dt * 1e6)),
                    TF.DelayRecordingTime: int(round(g.t0 * 1e3)),
                }
                f.trace[i] = g.data[k].astype(np.float32)
                i += 1
        f.bin.update(hdt=int(round(g0.dt * 1e6)), hns=g0.nt, format=5)


def write_stack(path: str, stack: np.ndarray, keys: Sequence, dt: float, t0: float = 0.0,
                cmp_spacing: float = 12.5):
    """Write a stacked section (ncmp, nt) to SEG-Y."""
    _need_segyio()
    spec = segyio.spec()
    spec.format = 5
    spec.samples = (t0 + dt * np.arange(stack.shape[1])) * 1e3
    spec.tracecount = stack.shape[0]
    with segyio.create(path, spec) as f:
        for i, k in enumerate(keys):
            cdp = int(k) if not isinstance(k, tuple) else i + 1
            hdr = {TF.TRACE_SEQUENCE_LINE: i + 1, TF.CDP: cdp,
                   TF.TRACE_SAMPLE_COUNT: stack.shape[1],
                   TF.TRACE_SAMPLE_INTERVAL: int(round(dt * 1e6)),
                   TF.CDP_X: int(round(cdp * cmp_spacing))}
            if isinstance(k, tuple):
                hdr[TF.INLINE_3D], hdr[TF.CROSSLINE_3D] = int(k[0]), int(k[1])
            f.header[i] = hdr
            f.trace[i] = stack[i].astype(np.float32)
        f.bin.update(hdt=int(round(dt * 1e6)), hns=stack.shape[1], format=5)


def write_processed_like(src: "SegyDataset", dst_path: str, processed_gather_fn,
                         keys=None, progress=False):
    """Copy the source SEG-Y (all headers preserved) and replace trace samples
    gather by gather with processed_gather_fn(key) -> Gather (same trace order as
    src.gather(key)).  Traces of gathers not processed are left unchanged."""
    import shutil
    _need_segyio()
    shutil.copyfile(src.path, dst_path)
    keys = src.keys if keys is None else keys
    with segyio.open(dst_path, "r+", ignore_geometry=True) as f:
        for n, k in enumerate(keys):
            g = processed_gather_fn(k)
            idx = src.index[k]
            if g.ntr != len(idx):
                raise ValueError(f"gather {k}: {g.ntr} traces returned, {len(idx)} expected")
            for j, i in enumerate(idx):
                f.trace[int(i)] = g.data[j].astype(np.float32)
            if progress and n % 50 == 0:
                print(f"wrote {n}/{len(keys)}")
