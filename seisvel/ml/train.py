"""Training / fine-tuning of the neural velocity picker."""
from __future__ import annotations

import time

import numpy as np

from .model import VelocityUNet, _need_torch, soft_ce

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _augment(x, y, w, rng):
    """Cheap, physically-safe augmentations on spectrum images."""
    # random global contrast (gamma) on spectra channels
    g = rng.uniform(0.7, 1.4)
    x = x.copy()
    x[:, :2] = np.clip(x[:, :2], 0, None) ** g
    # random time crop/shift (keeps velocity axis untouched: velocity is absolute)
    if rng.random() < 0.5:
        s = int(rng.integers(0, 4)) * 16
        if s:
            x = np.concatenate([x[:, :, s:], np.zeros_like(x[:, :, :s])], 2)
            y = np.concatenate([y[:, s:], np.zeros_like(y[:, :s])], 1)
            w = np.concatenate([w[:, s:], np.zeros_like(w[:, :s])], 1)
    return x, y, w


def train(X, Y, W, *, epochs=12, batch=16, lr=2e-3, val_frac=0.1, seed=0, base=12,
          model=None, verbose=True, max_minutes=None):
    _need_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    nv = int(round(val_frac * n))
    if val_frac > 0 and nv == 0 and n > 1:
        nv = 1
    vi, ti = idx[:nv], idx[nv:]
    model = model or VelocityUNet(X.shape[1], base=base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(ti) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    hist = []
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        rng.shuffle(ti)
        tl = []
        for b in range(0, len(ti), batch):
            j = ti[b:b + batch]
            x, y, w = _augment(X[j], Y[j], W[j], rng)
            x, y, w = map(torch.from_numpy, (x, y, w))
            loss = soft_ce(model(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        vl, verr = evaluate(model, X[vi], Y[vi], W[vi]) if nv else (np.nan, np.nan)
        hist.append(dict(epoch=ep + 1, train=float(np.mean(tl)), val=vl, val_bin_err=verr))
        if nv and vl < best:                # keep the best-validation weights
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {ep + 1}: train {np.mean(tl):.3f}  val {vl:.3f}  "
                  f"val |dv| {verr:.2f} bins  ({time.time() - t0:.0f}s)", flush=True)
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def evaluate(model, X, Y, W, batch=32):
    model.eval()
    ls, errs = [], []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            x = torch.from_numpy(X[b:b + batch])
            y = torch.from_numpy(Y[b:b + batch])
            w = torch.from_numpy(W[b:b + batch])
            lg = model(x)
            ls.append(soft_ce(lg, y, w).item())
            p = torch.softmax(lg, -1)
            j = torch.arange(p.shape[-1]).float()
            e = ((p * j).sum(-1) - (y * j).sum(-1)).abs()
            m = w > 0.5
            errs.append(e[m].mean().item() if m.any() else 0.0)
    return float(np.mean(ls)), float(np.mean(errs))


def fine_tune(model, spectra_images, vrms_curves, t_grids, *, epochs=30, lr=3e-4):
    """Adapt a pre-trained picker to a new survey from a handful of human-picked
    analysis locations (transfer learning, cf. Park & Sacchi; Wang et al. 2021).

    spectra_images : list of (C, T, V) arrays from ml.dataset.to_image
    vrms_curves    : list of (t, v) tuples - the human picks, interpolated
    """
    from .dataset import make_target
    X = np.stack(spectra_images)
    Y, W = [], []
    for x, (tc, vc), tg in zip(spectra_images, vrms_curves, t_grids):
        Y.append(make_target(vc, tc, tg, x.shape[1]))
        w = np.zeros(x.shape[1], np.float32)
        w[: len(tg)] = ((tg >= tc[0]) & (tg <= tc[-1])).astype(np.float32)
        W.append(w)
    Y, W = np.stack(Y), np.stack(W)
    # freeze encoder, adapt decoder only -> safe with very few labels
    for name, prm in model.named_parameters():
        prm.requires_grad = not name.startswith(("e1", "e2", "e3", "e4"))
    m, _ = train(X, Y, W, epochs=epochs, batch=min(8, len(X)), lr=lr, val_frac=0.0,
                 model=model, verbose=False)
    for prm in model.parameters():
        prm.requires_grad = True
    return m


def save(model, path, meta=None):
    torch.save(dict(state=model.state_dict(), base=model.e1.net[0].out_channels,
                    in_ch=model.e1.net[0].in_channels, meta=meta or {}), path)


def load(path):
    _need_torch()
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = VelocityUNet(d["in_ch"], base=d["base"])
    m.load_state_dict(d["state"])
    m.eval()
    return m
