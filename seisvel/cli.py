"""Command-line interface:  python -m seisvel <command> ...

  info      SEG-Y summary (fold, offsets, sampling)
  synth     write a synthetic 2-D prestack line (+ true velocities) to SEG-Y
  scan      velocity spectrum figure for one CMP
  autopick  automatic velocity picking along a line -> CSV (optionally AI-assisted)
  pick      interactive picking GUI
  stack     NMO + stack with a velocity CSV -> SEG-Y (+ PNG)
  demultiple  Radon demultiple with a primary-velocity CSV -> SEG-Y (headers kept)
  train     train the neural picker on synthetic data
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _dataset(a):
    from .io.segy import HeaderMap, SegyDataset
    hm = HeaderMap(cdp=a.cdp_byte, offset=a.offset_byte,
                   inline=a.inline_byte, xline=a.xline_byte)
    return SegyDataset(a.segy, hm)


def _scan_params(a):
    from .picking.line import ScanParams
    return ScanParams(velocities=np.linspace(a.vmin, a.vmax, a.nv), measure=a.measure,
                      supergather=a.supergather, window=a.window,
                      band=tuple(a.band) if a.band else None)


def cmd_info(a):
    from .gather import info
    ds = _dataset(a)
    folds = [ds.fold(k) for k in ds.keys]
    print(f"file      : {a.segy}\ntraces    : {ds.ntraces}\nsamples   : {ds.nt} x "
          f"{ds.dt * 1e3:.2f} ms (t0={ds.t0:.3f}s)\nCMPs      : {len(ds.keys)} "
          f"({ds.keys[0]} .. {ds.keys[-1]}) 3D={ds.is3d}\nfold      : min {min(folds)} "
          f"median {int(np.median(folds))} max {max(folds)}")
    print("middle CMP:", info(ds.gather(ds.keys[len(ds.keys) // 2])))


def cmd_synth(a):
    from .io.segy import write_prestack
    from .synth.generator import Acquisition, NoiseSpec, synthetic_line
    acq = Acquisition(nt=a.nt, dt=a.dt / 1e3, max_offset=a.max_offset, ntr=a.ntr)
    gs, truth, _ = synthetic_line(a.ncmp, seed=a.seed, marine=not a.land, acq=acq,
                                  noise=NoiseSpec(snr=a.snr), multiples=a.multiples)
    write_prestack(a.out, gs)
    np.save(a.out.rsplit(".", 1)[0] + "_true_vrms.npy", truth)
    print(f"wrote {a.out} ({len(gs)} CMPs) and true Vrms grid")


def cmd_scan(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plot
    from .picking.dp import dp_pick
    from .picking.line import compute_spectrum
    ds = _dataset(a)
    key = a.cmp if a.cmp is not None else ds.keys[len(ds.keys) // 2]
    sp = compute_spectrum(ds, key, _scan_params(a))
    r = dp_pick(sp)
    fig, ax = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    plot.gather(ax[0], ds.gather(key), title=f"CMP {key}")
    plot.spectrum(ax[1], sp, [(r.velocity, "w-", "auto DP")], r.function)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


def cmd_autopick(a):
    from .picking.line import autopick_line
    ds = _dataset(a)
    keys = ds.analysis_keys(a.every)
    sp = _scan_params(a)
    factory = None
    from .ml.infer import NeuralPicker, neural_score_fn
    if a.model:
        npk = NeuralPicker.load(a.model)
        factory = lambda d: neural_score_fn(npk, d, sp)
    wb = "auto" if a.marine else None
    if a.field_workflow:
        from .workflow import field_velocity_workflow
        npk_ = NeuralPicker.load(a.model) if a.model else None
        wf = field_velocity_workflow(ds, keys, sp, neural=npk_, water_bottom=wb or "auto",
                                     q_cut=a.q_cut, verbose=True)
        for n, f in wf.starts.items():
            f.save_csv(a.out.replace(".csv", f"_{n}_pass1.csv"))
        wf.field.save_csv(a.out)
        print(f"final velocities -> {a.out}; review queue: {wf.review}")
        return
    if a.two_pass:
        from .workflow import two_pass_velocity
        tp = two_pass_velocity(ds, keys, sp, water_bottom=wb, q_cut=a.q_cut,
                               score_factory=factory, verbose=True)
        tp.pass1.field.save_csv(a.out.replace(".csv", "_pass1.csv"))
        res = tp.pass2
    else:
        res = autopick_line(ds, keys, sp, score_fn=factory(ds) if factory else None,
                            water_bottom=wb, verbose=True)
    res.field.save_csv(a.out)
    print(f"picked {len(keys)} locations -> {a.out};  {len(res.review)} flagged for "
          f"review: {res.review[:20]}")


def cmd_pick(a):
    from .gui.picker import InteractivePicker
    from .velocity import VelocityField
    ds = _dataset(a)
    vf = VelocityField.load_csv(a.picks) if a.picks else None
    neural = None
    if a.model:
        from .ml.infer import NeuralPicker
        neural = NeuralPicker.load(a.model)
    InteractivePicker(ds, ds.analysis_keys(a.every), vf, _scan_params(a), neural=neural,
                      save_path=a.out)


def cmd_stack(a):
    from .io.segy import write_stack
    from . import precondition
    from .stack import stack_line
    from .velocity import VelocityField
    ds = _dataset(a)
    vf = VelocityField.load_csv(a.picks)
    pre = (lambda g: precondition.tpow_gain(g, 2.0)) if a.gain else None
    st, fold, keys = stack_line(ds, vf, preprocess=pre, stretch_max=a.stretch / 100,
                                method=a.method, progress=True)
    write_stack(a.out, st, keys, ds.dt, ds.t0)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from . import plot
        fig, ax = plt.subplots(figsize=(14, 8))
        plot.section(ax, st, keys, ds.dt, ds.t0, title="NMO stack")
        fig.savefig(a.png, dpi=110, bbox_inches="tight")
    print("wrote", a.out)


def cmd_demultiple(a):
    from .io.segy import write_processed_like
    from .radon import DemultipledDataset
    from .velocity import VelocityField
    ds = _dataset(a)
    dem = DemultipledDataset(ds, VelocityField.load_csv(a.picks), q_cut=a.q_cut,
                             cache=False)
    write_processed_like(ds, a.out, dem.gather, progress=True)
    print("wrote", a.out)


def cmd_train(a):
    from .ml.dataset import build_dataset
    from .ml.train import save, train
    X, Y, W = build_dataset(a.n, seed=a.seed)
    m, hist = train(X, Y, W, epochs=a.epochs)
    save(m, a.out, dict(history=hist))
    print("saved", a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="seisvel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def seg(sp):
        sp.add_argument("segy")
        sp.add_argument("--cdp-byte", type=int, default=21)
        sp.add_argument("--offset-byte", type=int, default=37)
        sp.add_argument("--inline-byte", type=int, default=None)
        sp.add_argument("--xline-byte", type=int, default=None)

    def scanopts(sp):
        sp.add_argument("--vmin", type=float, default=1400)
        sp.add_argument("--vmax", type=float, default=5500)
        sp.add_argument("--nv", type=int, default=165)
        sp.add_argument("--measure", default="semblance")
        sp.add_argument("--supergather", type=int, default=2)
        sp.add_argument("--window", type=float, default=0.04)
        sp.add_argument("--band", type=float, nargs=2, default=None)

    s = sub.add_parser("info"); seg(s); s.set_defaults(f=cmd_info)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True); s.add_argument("--ncmp", type=int, default=200)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--land", action="store_true")
    s.add_argument("--nt", type=int, default=1500); s.add_argument("--dt", type=float, default=4.0)
    s.add_argument("--ntr", type=int, default=60)
    s.add_argument("--max-offset", type=float, default=3000)
    s.add_argument("--snr", type=float, default=3.0)
    s.add_argument("--multiples", type=float, default=1.0)
    s.set_defaults(f=cmd_synth)
    s = sub.add_parser("scan"); seg(s); scanopts(s)
    s.add_argument("--cmp", type=int); s.add_argument("--out", default="scan.png")
    s.set_defaults(f=cmd_scan)
    s = sub.add_parser("autopick"); seg(s); scanopts(s)
    s.add_argument("--every", type=int, default=20); s.add_argument("--out", default="picks.csv")
    s.add_argument("--model", default=None)
    s.add_argument("--marine", action="store_true",
                   help="water-bottom detection + multiple-aware picking")
    s.add_argument("--two-pass", action="store_true",
                   help="pass 1 -> Radon demultiple -> pass 2")
    s.add_argument("--q-cut", type=float, default=0.02)
    s.add_argument("--field-workflow", action="store_true",
                   help="physics + AI pass 1 -> Radon -> multi-start RVA -> Dix (marine)")
    s.set_defaults(f=cmd_autopick)
    s = sub.add_parser("demultiple"); seg(s)
    s.add_argument("--picks", required=True); s.add_argument("--out", default="demult.sgy")
    s.add_argument("--q-cut", type=float, default=0.02); s.set_defaults(f=cmd_demultiple)
    s = sub.add_parser("pick"); seg(s); scanopts(s)
    s.add_argument("--every", type=int, default=20); s.add_argument("--picks")
    s.add_argument("--out", default="picks.csv"); s.add_argument("--model", default=None)
    s.set_defaults(f=cmd_pick)
    s = sub.add_parser("stack"); seg(s)
    s.add_argument("--picks", required=True); s.add_argument("--out", default="stack.sgy")
    s.add_argument("--stretch", type=float, default=50); s.add_argument("--method", default="mean")
    s.add_argument("--gain", action="store_true"); s.add_argument("--png", default=None)
    s.set_defaults(f=cmd_stack)
    s = sub.add_parser("train")
    s.add_argument("--n", type=int, default=3000); s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="velpicker.pt")
    s.set_defaults(f=cmd_train)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main(sys.argv[1:])
