"""Train the neural velocity picker from pre-generated synthetic spectra.

    python examples/train_picker.py train_1.npz train_2.npz --out velpicker.pt

Datasets are produced with seisvel.ml.dataset.build_dataset (or `seisvel train`).
Images generated on a finer time grid are max-pooled onto ML_DT automatically.
"""
import argparse
import json

import numpy as np

from seisvel.ml.dataset import ML_DT
from seisvel.ml.train import save, train


def pool_to(X, Y, W, factor):
    if factor == 1:
        return X, Y, W
    T = X.shape[2] // factor * factor
    X = X[:, :, :T].reshape(X.shape[0], X.shape[1], T // factor, factor, -1).max(3)
    Y = Y[:, :T].reshape(Y.shape[0], T // factor, factor, -1).mean(2)
    Y = Y / np.maximum(Y.sum(-1, keepdims=True), 1e-12)
    W = W[:, :T].reshape(W.shape[0], T // factor, factor).max(2)
    pad = (-X.shape[2]) % 16
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (0, pad), (0, 0)))
        Y = np.pad(Y, ((0, 0), (0, pad), (0, 0)))
        W = np.pad(W, ((0, 0), (0, pad)))
    return X.astype(np.float32), Y.astype(np.float32), W.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--out", default="velpicker.pt")
    a = ap.parse_args()
    Xs, Ys, Ws = [], [], []
    for p in a.data:                       # pool file by file (memory)
        d = np.load(p)
        X0 = d["X"]
        # files made on a finer grid (e.g. 16 ms, ~384 rows) are pooled to ML_DT
        f = 2 if X0.shape[2] > 300 else 1
        x, y, w = pool_to(X0, d["Y"], d["W"], f)
        del d
        Xs.append(x); Ys.append(y); Ws.append(w)
    X, Y, W = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)
    del Xs, Ys, Ws
    print("training set", X.shape, flush=True)
    model, hist = train(X, Y, W, epochs=a.epochs, max_minutes=a.max_minutes)
    save(model, a.out, dict(history=hist, n_train=len(X)))
    json.dump(hist, open(a.out.rsplit(".", 1)[0] + "_history.json", "w"), indent=1)
    print("saved", a.out)
