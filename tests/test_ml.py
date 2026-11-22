import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seisvel.ml.dataset import ML_VELS, random_example, spectra_for_ml, to_image
from seisvel.ml.infer import NeuralPicker
from seisvel.ml.model import VelocityUNet
from seisvel.ml.train import fine_tune, load, save, train
from seisvel.synth.generator import Acquisition, random_model, synthesize_gather


def test_train_infer_finetune_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    ex = [random_example(rng, nt=600) for _ in range(6)]
    X = np.stack([e[0] for e in ex]); Y = np.stack([e[1] for e in ex])
    W = np.stack([e[2] for e in ex])
    m, hist = train(X, Y, W, epochs=1, batch=3, base=4, verbose=False)
    assert np.isfinite(hist[0]["train"])
    save(m, tmp_path / "m.pt"); m2 = load(tmp_path / "m.pt")
    g, tr = synthesize_gather(random_model(rng), Acquisition(nt=600), rng)
    sem, ab = spectra_for_ml(g)
    npk = NeuralPicker(m2, mc_samples=2)
    P, tg, sig = npk.probability(sem, ab)
    assert P.shape[1] == len(ML_VELS) and np.allclose(P.sum(1), 1, atol=1e-4)
    S = npk.score_map(sem, ab)
    assert S.shape == sem.values.shape
    r = npk.pick(sem, ab)
    assert np.all(np.isfinite(r.velocity))
    x, tgrid, T = to_image(sem, ab)
    m3 = fine_tune(m2, [x], [(g.times, tr["vrms"])], [tgrid], epochs=1)
    assert isinstance(m3, VelocityUNet)
