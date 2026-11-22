"""Neural velocity picker: a compact U-Net that turns a multi-channel velocity
spectrum (time x velocity) into a per-time probability distribution over
velocity.  The output is NOT used directly as the answer: it becomes the score
map of the physics-constrained DP decoder (picking/dp.py), so every AI pick
still honours Dix / smoothness / corridor constraints.  MC-dropout gives a
per-time uncertainty that drives the human-review queue.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _need_torch():
    if torch is None:
        raise ImportError("PyTorch is required for the neural picker: pip install torch")


if nn is not None:
    class _Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())

        def forward(self, x):
            return self.net(x)

    class VelocityUNet(nn.Module):
        """Input (B, C, T, V) -> logits (B, T, V).  T, V multiples of 16."""

        def __init__(self, in_ch=3, base=12, p_drop=0.2):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.e1, self.e2 = _Block(in_ch, c[0]), _Block(c[0], c[1])
            self.e3, self.e4 = _Block(c[1], c[2]), _Block(c[2], c[3])
            self.bott = _Block(c[3], c[3])
            self.drop = nn.Dropout2d(p_drop)
            self.u4, self.d4 = nn.ConvTranspose2d(c[3], c[3], 2, 2), _Block(2 * c[3], c[2])
            self.u3, self.d3 = nn.ConvTranspose2d(c[2], c[2], 2, 2), _Block(2 * c[2], c[1])
            self.u2, self.d2 = nn.ConvTranspose2d(c[1], c[1], 2, 2), _Block(2 * c[1], c[0])
            self.u1, self.d1 = nn.ConvTranspose2d(c[0], c[0], 2, 2), _Block(2 * c[0], c[0])
            # long vertical context: dilated convs along time on the output
            self.ctx = nn.Sequential(
                nn.Conv2d(c[0], c[0], (5, 1), padding=(4, 0), dilation=(2, 1)), nn.GELU(),
                nn.Conv2d(c[0], c[0], (5, 1), padding=(8, 0), dilation=(4, 1)), nn.GELU())
            self.head = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            e3 = self.e3(F.max_pool2d(e2, 2))
            e4 = self.e4(F.max_pool2d(e3, 2))
            b = self.drop(self.bott(F.max_pool2d(e4, 2)))
            d4 = self.d4(torch.cat([self.u4(b), e4], 1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            d1 = d1 + self.ctx(d1)
            return self.head(d1)[:, 0]          # (B, T, V)


def soft_ce(logits, target, row_w):
    """Row-weighted cross-entropy with soft (Gaussian) labels over velocity."""
    logp = F.log_softmax(logits, dim=-1)
    ce = -(target * logp).sum(-1)               # (B, T)
    return (ce * row_w).sum() / row_w.sum().clamp_min(1e-6)
