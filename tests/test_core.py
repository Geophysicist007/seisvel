import numpy as np
import pytest

from seisvel import (Gather, VelocityField, VelocityFunction, inverse_nmo, nmo_correct,
                     offset_regularize, precondition, qc, velocity_scan)
from seisvel.moveout import traveltime_np
from seisvel.picking.dp import DPParams, dp_pick, naive_pick
from seisvel.synth.generator import (Acquisition, LayeredModel, NoiseSpec,
                                     synthesize_gather)

DT, NT = 0.004, 1000


def spikes_gather(offsets, events, v_of_t0, eta=0.0, polarity=None, width=0.008):
    t = np.arange(NT) * DT
    d = np.zeros((len(offsets), NT), np.float32)
    for k, t0 in enumerate(events):
        v = v_of_t0(t0)
        for i, x in enumerate(offsets):
            tt = traveltime_np(t0, x, v, eta)
            amp = 1.0 if polarity is None else polarity(k, i)
            d[i] += amp * np.exp(-((t - tt) / width) ** 2) * np.cos(2 * np.pi * 25 * (t - tt))
    return Gather(d, offsets, DT)


OFF = np.linspace(50, 2500, 40)
VLIN = lambda t0: 1800 + 600 * t0


def test_nmo_flattens_and_inverse_roundtrip():
    g = spikes_gather(OFF, [1.0, 2.0, 3.0], VLIN)
    v = VLIN(g.times)
    gn = nmo_correct(g, v, stretch_max=np.inf)
    for t0 in (1.0, 2.0, 3.0):
        it = int(t0 / DT)
        peaks = gn.times[np.argmax(np.abs(gn.data[:, it - 25:it + 25]), axis=1) + it - 25]
        assert np.all(np.abs(peaks - t0) <= 2 * DT)
    gi = inverse_nmo(gn, v)
    sl = slice(int(1.5 / DT), NT)
    err = np.linalg.norm(gi.data[:, sl] - g.data[:, sl]) / np.linalg.norm(g.data[:, sl])
    assert err < 0.05


def test_stretch_mute_removes_shallow_far():
    g = spikes_gather(OFF, [0.3, 2.0], VLIN)
    gn = nmo_correct(g, VLIN(g.times), stretch_max=0.3)
    assert np.all(gn.data[-1, : int(0.3 / DT)] == 0)       # far trace shallow muted
    assert np.any(gn.data[0, : int(0.4 / DT)] != 0)        # near trace kept


@pytest.mark.parametrize("measure", ["semblance", "ab", "similarity", "ab_similarity"])
def test_scan_peaks_at_true_velocity(measure):
    g = spikes_gather(OFF, [0.8, 1.6, 2.4, 3.2], VLIN)
    vels = np.linspace(1500, 4500, 121)
    sp = velocity_scan(g, vels, measure)
    for t0 in (0.8, 1.6, 2.4, 3.2):
        it = int(round(t0 / DT))
        vpk = vels[np.argmax(sp.values[:, it])]
        assert abs(vpk - VLIN(t0)) / VLIN(t0) < 0.02, (measure, t0, vpk)


def test_ab_semblance_survives_polarity_reversal():
    # class-II style: polarity flips at mid offsets -> conventional semblance fails
    xm = OFF.max()
    pol = lambda k, i: (1.0 - 2.2 * (OFF[i] / xm) ** 2)
    g = spikes_gather(OFF, [1.5], VLIN, polarity=pol)
    vels = np.linspace(1500, 4500, 121)
    it = int(1.5 / DT)
    vt = VLIN(1.5)
    sem = velocity_scan(g, vels, "semblance")
    ab = velocity_scan(g, vels, "ab")
    v_ab = vels[np.argmax(ab.values[:, it])]
    assert abs(v_ab - vt) / vt < 0.02
    # conventional semblance at the true velocity is much weaker than AB's
    j = int(np.argmin(np.abs(vels - vt)))
    assert ab.values[j, it] > 3 * sem.values[j, it]


def test_irregular_and_missing_offsets():
    rng = np.random.default_rng(0)
    off = np.sort(rng.choice(OFF, 25, replace=False) + rng.normal(0, 20, 25))
    g = spikes_gather(np.abs(off), [1.2, 2.5], VLIN)
    g.data[3] = 0                                 # dead trace
    sp = velocity_scan(g.drop_dead(), np.linspace(1500, 4500, 121))
    for t0 in (1.2, 2.5):
        vpk = sp.velocities[np.argmax(sp.values[:, int(t0 / DT)])]
        assert abs(vpk - VLIN(t0)) / VLIN(t0) < 0.03


def test_eta_moveout_flattened_only_with_eta():
    g = spikes_gather(np.linspace(50, 5000, 50), [1.0], lambda t: 2500.0, eta=0.12)
    v = np.full(NT, 2500.0)
    f_eta, _ = qc.flatness(nmo_correct(g, v, 0.12, stretch_max=np.inf))
    f_hyp, _ = qc.flatness(nmo_correct(g, v, 0.0, stretch_max=np.inf))
    assert f_eta > 0.9 and f_eta > f_hyp + 0.1


def test_dix_and_interpolation():
    vint = [2000.0, 3000.0, 4000.0]
    tb = np.array([1.0, 2.0, 3.0])
    vr = [np.sqrt(sum(vint[i] ** 2 for i in range(k + 1)) / (k + 1)) for k in range(3)]
    vf = VelocityFunction(tb, vr)
    _, _, vi = vf.dix()
    assert np.allclose(vi, vint[1:], rtol=1e-6)
    assert vf.check(vint_min=1500, vint_max=4500) == []
    bad = VelocityFunction([1.0, 1.1], [3000.0, 2000.0])
    assert bad.check()                        # imaginary Dix flagged
    v = vf.interpolate(np.array([1.5]), mode="vint")[0]
    assert abs(v ** 2 * 1.5 - (vr[0] ** 2 * 1.0 + 3000 ** 2 * 0.5)) < 1.0


def test_velocity_field_interp_and_io(tmp_path):
    f = VelocityField()
    f[100] = VelocityFunction([0.5, 2.0], [2000, 3000])
    f[200] = VelocityFunction([0.5, 2.0], [2200, 3400])
    v = f.at(150, np.array([0.5, 2.0]))
    assert np.allclose(v, [2100, 3200])
    f.save_csv(tmp_path / "p.csv")
    g = VelocityField.load_csv(tmp_path / "p.csv")
    assert np.allclose(g.at(150, np.array([2.0])), [3200])
    f.save_json(tmp_path / "p.json")
    h = VelocityField.load_json(tmp_path / "p.json")
    assert h.keys == [100, 200]
    f3 = VelocityField()
    for il, xl, vv in [(1, 1, 2000), (1, 3, 2200), (3, 1, 2400), (3, 3, 2600)]:
        f3[(il, xl)] = VelocityFunction([1.0], [vv])
    assert abs(f3.at((2, 2), np.array([1.0]))[0] - 2300) < 1e-6


def _model():
    h = np.array([300, 400, 350, 500, 450, 600, 500.0])
    vp = np.array([1500, 2000, 2300, 2700, 3100, 3500, 3900.0])
    vs = np.r_[0, (vp[1:] - 1360) / 1.16]
    rho = np.r_[1030, 310 * vp[1:] ** 0.25]
    return LayeredModel(h, vp, vs, rho, water=True)


def test_dp_respects_dix_and_beats_naive_with_multiples():
    rng = np.random.default_rng(1)
    g, tr = synthesize_gather(_model(), Acquisition(nt=1000), rng, multiples=1.0,
                              noise=NoiseSpec(snr=4.0))
    gp = offset_regularize(precondition.scan_flow(g), 50)
    # strong hard-water-bottom multiples, no demultiple: similarity-weighted
    # semblance + multiple-aware decoding is the robust combination
    sp = velocity_scan(gp, np.linspace(1400, 4500, 125), "similarity")
    from seisvel.picking.dp import marine_vint_floor, multiple_aware_pick
    twb = precondition.detect_water_bottom(g)
    assert abs(twb - tr["t0_primaries"][0]) < 0.02
    p = DPParams(vint_min=marine_vint_floor(twb), vint_max=5000)
    r = dp_pick(sp, p)
    ct, cv = r.coarse_t, r.coarse_v
    vint2 = np.diff(cv ** 2 * ct) / np.diff(ct)
    vint = np.sqrt(np.maximum(vint2, 0))
    assert np.all(vint >= 1455 - 1) and np.all(vint <= 5000 + 1)
    assert np.all(vint[ct[1:] > twb + 0.06] >= 1600 - 1)
    a, b = tr["t0_primaries"][0] + 0.05, tr["t0_primaries"][-1]
    e_dp = qc.velocity_error(sp.times, r.velocity, tr["vrms"], a, b)["mape_pct"]
    e_nv = qc.velocity_error(sp.times, naive_pick(sp), tr["vrms"], a, b)["mape_pct"]
    assert e_dp < 0.6 * e_nv
    e_ma = qc.velocity_error(sp.times, multiple_aware_pick(sp, twb).velocity,
                             tr["vrms"], a, b)["mape_pct"]
    assert e_ma < 5.0


def test_segy_roundtrip_unsorted(tmp_path):
    segyio = pytest.importorskip("segyio")
    from seisvel.io.segy import SegyDataset, write_prestack
    gs = [spikes_gather(OFF[:10], [1.0], VLIN) for _ in range(3)]
    for k, g in enumerate(gs):
        g.key = 10 + k
    path = str(tmp_path / "a.sgy")
    write_prestack(path, gs)
    # shuffle trace order to mimic shot-sorted field data
    with segyio.open(path, "r+", ignore_geometry=True) as f:
        tr = [f.trace[i].copy() for i in range(f.tracecount)]
        hd = [dict(f.header[i]) for i in range(f.tracecount)]
        perm = np.random.default_rng(0).permutation(f.tracecount)
        for j, i in enumerate(perm):
            f.header[j] = hd[i]
            f.trace[j] = tr[i]
    ds = SegyDataset(path)
    assert ds.keys == [10, 11, 12]
    g = ds.gather(11).sort_by_offset()
    assert np.allclose(g.offsets, np.round(OFF[:10]))
    assert np.allclose(g.data, gs[1].data, atol=1e-6)
    assert ds.supergather(11, 1).ntr == 30


def test_gui_headless(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from types import SimpleNamespace
    from seisvel.gui.picker import InteractivePicker
    from seisvel.io.segy import MemoryDataset
    from seisvel.picking.line import ScanParams
    rng = np.random.default_rng(2)
    gs = []
    for k in range(5):
        g, _ = synthesize_gather(_model(), Acquisition(nt=800, ntr=30), rng, multiples=0.5)
        g.key = k
        gs.append(g)
    ds = MemoryDataset(gs)
    p = InteractivePicker(ds, [0, 2, 4], scan=ScanParams(np.linspace(1400, 4500, 64),
                          supergather=1), save_path=str(tmp_path / "x.csv"), show=False)
    ax = p.ax[1]
    p.on_click(SimpleNamespace(inaxes=ax, xdata=2000.0, ydata=1.0, button=1))
    p.on_click(SimpleNamespace(inaxes=ax, xdata=2500.0, ydata=2.0, button=1))
    assert len(p.vf[0]) == 2
    p.on_click(SimpleNamespace(inaxes=ax, xdata=2500.0, ydata=2.0, button=3))
    assert len(p.vf[0]) == 1
    p.on_key(SimpleNamespace(key="a"))
    assert len(p.vf[0]) >= 2
    p.on_key(SimpleNamespace(key="n"))
    p.on_key(SimpleNamespace(key="s"))
    assert (tmp_path / "x.csv").exists()


def test_radon_demultiple_removes_multiples_and_fixes_picking():
    from seisvel.radon import demultiple_gather
    m = _model()
    gp, tp = synthesize_gather(m, Acquisition(nt=1000), np.random.default_rng(0),
                               multiples=0.0)
    gm, tr = synthesize_gather(m, Acquisition(nt=1000), np.random.default_rng(0),
                               multiples=1.0, scale=tp["scale"])
    v = tr["vrms"]
    out = demultiple_gather(gm, v)
    sl = slice(125, int(2.3 / DT))
    G = (gm.times ** 2)[None, sl]
    before = np.linalg.norm((gm.data - gp.data)[:, sl] * G)
    after = np.linalg.norm((out.data - gp.data)[:, sl] * G)
    assert after < 0.6 * before
    a, b = tr["t0_primaries"][0] + 0.05, tr["t0_primaries"][-1]
    errs = []
    for g in (gm, out):
        sp = velocity_scan(offset_regularize(precondition.scan_flow(g), 50),
                           np.linspace(1400, 4500, 125))
        errs.append(qc.velocity_error(sp.times, dp_pick(sp).velocity, v, a, b)["mape_pct"])
    assert errs[1] < 3.0 and errs[1] < 0.3 * errs[0]


def test_residual_velocity_update_converges():
    g = spikes_gather(OFF, [0.8, 1.4, 2.0, 2.6, 3.2], VLIN)
    vt = VLIN(g.times)
    ev = np.array([0.8, 1.4, 2.0, 2.6, 3.2])
    for f0 in (0.92, 0.96, 1.04, 1.08):
        v, _ = qc.residual_velocity_update(g, vt * f0, n_iter=4, smooth=0.2)
        err = np.abs(np.interp(ev, g.times, v / vt) - 1)
        assert err.max() < 0.01, (f0, err)
