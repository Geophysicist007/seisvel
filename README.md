# seisvel — velocity analysis, AI-assisted picking, NMO and stack

`seisvel` is a pure-Python (numpy / numba / scipy) module for the NMO-stack stage of
seismic processing. It is built for real prestack data, not only textbook examples.

* **Real SEG-Y input.** Traces can be in any order, header bytes are configurable
  (2-D CDP or 3-D inline/xline), coordinate scalars are applied, missing offsets are
  rebuilt from source/receiver coordinates, and files are memory-mapped.
* **Robust scanning.** Supergathers, offset regularisation, dead-trace and spike
  handling, and a scan-only conditioning flow (gain, band-pass, AGC).
* **Velocity spectra.** Semblance, AB semblance (AVO / polarity-reversal robust),
  similarity-weighted, AB×similarity and stack power. Optional η (non-hyperbolic)
  moveout. Per-sample live-trace normalisation and stretch-aware scanning.
* **NMO.** 8-point windowed-sinc interpolation, a stretch mute that is exact for
  any V(t0) or η, crossover muting, and inverse NMO.
* **Picking.**
  * Interactive GUI with live NMO preview, Dix warnings, neighbour guides and undo.
  * Automatic **physics-constrained DP picker**: Dix interval-velocity bounds,
    smoothness, an asymmetric guide that rejects multiples, and a marine
    water-bottom floor with peg-leg multiple masking.
  * **Line workflow**: regional guide, then lateral consensus, then a review queue.
  * **AI picker**: a U-Net probability map decoded by the same constrained DP,
    with uncertainty from MC-dropout and fine-tuning from a few human picks.
* **Velocity fields.** Time and space interpolation (2-D linear, 3-D triangulated),
  Dix conversion, plausibility checks, and CSV/JSON I/O.
* **Demultiple.** Parabolic Radon (least squares, optional high-resolution IRLS)
  and an automatic two-pass flow: pass 1, then Radon, then pass 2.
* **Stack.** Fold-normalised per sample after the mute; mean, alpha-trimmed or median.
* **QC.** Flatness, residual-moveout scan, Dix violations, and benchmark metrics.

## Install
```bash
pip install -e .            # core
pip install -e .[ai]        # + PyTorch for the neural picker
```

## Command line
```bash
seisvel info line.sgy                                    # fold, offsets, sampling
seisvel synth --out test.sgy --ncmp 200 --snr 3          # synthetic line + true velocities
seisvel scan line.sgy --cmp 1200 --out scan.png          # one velocity spectrum + auto pick
seisvel autopick line.sgy --every 20 --out picks.csv     # automatic line picking
seisvel autopick line.sgy --model models/velpicker_synth_v1.pt --marine --out ai_picks.csv
seisvel autopick line.sgy --marine --two-pass --out picks.csv       # pass1 -> Radon -> pass2
seisvel demultiple line.sgy --picks picks_pass1.csv --out demult.sgy  # headers preserved
seisvel autopick line.sgy --field-workflow --model models/velpicker_synth_v1.pt --out final.csv
#   physics + AI pass 1 -> Radon (blended velocity, water-bottom protected) -> guided re-pick
seisvel pick line.sgy --picks picks.csv --out edited.csv            # GUI QC / editing
seisvel stack line.sgy --picks edited.csv --out stack.sgy --gain --png stack.png
# non-standard headers:  --cdp-byte 189 --offset-byte 37   (3-D: --inline-byte 189 --xline-byte 193)
```

## Python
```python
import numpy as np
from seisvel.io.segy import SegyDataset
from seisvel.picking.line import ScanParams, autopick_line
from seisvel.stack import stack_line
from seisvel import precondition

ds = SegyDataset("line.sgy")
res = autopick_line(ds, ds.analysis_keys(20), ScanParams(measure="semblance"),
                    water_bottom="auto")          # marine; None for land
res.field.save_csv("picks.csv")
print("locations to review:", res.review)
stack, fold, keys = stack_line(ds, res.field, preprocess=lambda g: precondition.tpow_gain(g))
```

## Layout
```
seisvel/
  gather.py        Gather container, supergathers, offset regularisation
  precondition.py  gain, AGC, band-pass, despike, top mute, water-bottom detection
  moveout.py       traveltime laws, NMO / inverse NMO, stretch mute
  semblance.py     velocity spectra (all measures), eta scan
  velocity.py      VelocityFunction / VelocityField, Dix, I/O
  stack.py         NMO + stack
  radon.py         parabolic Radon transform, demultiple, DemultipledDataset
  workflow.py      two-pass velocity analysis with demultiple
  qc.py            flatness, residual moveout, error metrics
  picking/dp.py    constrained Viterbi picker, multiple-aware marine picking
  picking/line.py  two-pass line workflow
  ml/              synthetic training data, U-Net, training, hybrid inference
  synth/           physics-based synthetic gathers and lines
  gui/picker.py    interactive picker
  io/segy.py       SEG-Y read/write
docs/RESEARCH.md   literature review, design decisions, AI opportunities, roadmap
examples/          benchmark and training scripts
tests/             pytest suite
```

## Status and results
* **Synthetics.** Mean velocity error on two unseen noisy lines:
  * naive max-semblance: 14.7 %
  * physics-constrained line workflow: 1.2–1.6 %
  * AI + constrained DP: ~0.6 %
* **Real data: full Mobil AVO Viking Graben line 12** (1 001 shots, seismic only;
  no well logs available).
  * The physics picker is trapped by deep multiples.
  * The AI escapes the multiples but runs fast.
  * The field workflow (`--field-workflow`) produces the flattest physically valid
    field on held-out CDPs.
  * The deep velocities remain ambiguous: 27 % spread, and every location is
    flagged for review.
  * Full numbers and the residual-velocity approach we rejected are in
    `docs/RESEARCH.md` §4c.
* **Pre-trained model.** `models/velpicker_synth_v1.pt` is synthetic-trained.
* **Next.** Well-tie validation (§5).
