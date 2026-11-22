# NMO / velocity analysis / stacking — research review and design notes

This note covers what the literature says about each step, the decisions taken in
`seisvel`, and where AI can genuinely help. Read it with the code: every claim
below maps to a module.

---

## 1. The process, step by step

```
prestack SEG-Y (any sort order)
  └─ index by CDP (or inline/xline)           io/segy.py
      └─ supergather (2k+1 neighbouring CMPs)  gather.combine
          └─ scan conditioning                precondition.scan_flow
             (dead traces, despike, t^2 gain, band-pass, AGC)
              └─ offset regularisation         gather.offset_regularize
                  └─ velocity spectrum         semblance.velocity_scan
                      └─ picking: manual GUI / constrained DP / AI+DP
                          └─ QC: Dix checks, flatness, residual moveout, review queue
                              └─ velocity field (time + space interpolation)
                                  └─ NMO (sinc, stretch mute) + stack   stack.py
```

### 1.1 Moveout law
* **Hyperbolic** `t² = t0² + x²/V²` (Dix 1955; Taner & Koehler 1969). Exact for one
  homogeneous layer. For layered media it is the 2-term Taylor approximation, so it
  loses accuracy once offset/depth goes past about 1.
* **Non-hyperbolic (η)**, from Alkhalifah & Tsvankin (1995). In VTI media, P-wave
  moveout and all time processing depend on only two parameters, V_nmo and η. The
  same equation also describes the non-hyperbolic moveout caused by vertical
  heterogeneity in isotropic media. Alkhalifah (1997) gives a two-step workflow:
  short-spread V_nmo first, then an η scan. → `moveout.traveltime`, `semblance.eta_scan`.
* Stacking velocity equals RMS velocity only in a flat-layered 1-D earth at small
  spread. Dip raises the apparent velocity (cos θ) until DMO or time migration is
  applied. The synthetic generator here is 1.5-D, so it does **not** test dip
  effects.

### 1.2 Coherence measures
| measure | idea | strengths | weaknesses | in seisvel |
|---|---|---|---|---|
| Semblance (Neidell & Taner 1971) | output energy / input energy in a window | standard, fast, bounded 0–1 | low resolution; fails on polarity reversals | `"semblance"` |
| AB semblance (Sarkar et al. 2001, 2002; Fomel 2009) | correlation with an amplitude trend A + Bφ instead of a constant | robust to AVO and class-II polarity reversal; semblance/AB ratio is an AVO indicator | ≈2× lower resolution, ≈2× higher noise floor (Fomel 2009) | `"ab"` |
| Weighted semblance (Luo & Hale 2012) | offset weights chosen to *minimise* semblance | higher resolution | extra cost | future work |
| Similarity-weighted (Chen, Liu & Chen 2015) | weight samples by local similarity to the stack | sharper peaks, better against noise | cost (one moved-out panel per trial velocity) | `"similarity"` |
| Weighted AB (Ebrahimi et al. 2017; PCA-weighted AB, 2022) | AB plus a resolution-enhancing weight | AVO-friendly *and* sharp | cost | `"ab_similarity"` |
| Eigenstructure / MUSIC (Biondi & Kostov 1989) | signal-subspace spectra | highest resolution | expensive, sensitive to parameters | future work |

Implementation notes:
* One compiled numba kernel sweeps velocities in parallel and returns the
  moments (n, Σa, Σa², Σφ, Σφ², Σaφ). Semblance and AB are both closed-form from
  these. AB uses the per-sample least-squares projection onto [1, φ], with
  φ = sin²θ ≈ x²/(x² + V²t0²).
* N is counted **per sample**. Muted samples, missing offsets and dead traces
  don't dilute the normalisation, which matters on real, irregular data.
* Samples with NMO stretch above a limit are excluded inside the scan.

### 1.3 NMO stretch
Stretch is the factor dt0/dt. Conventional practice mutes samples above a stretch
limit. Miller (1992) shows on shallow data that an improper mute can cut stacked
bandwidth by tens of Hz. Stretch-free alternatives exist: Perroud & Tygel 2004
(non-stretch NMO), Trickett 2003 (stretch-free stacking by inversion to zero
offset), local stretch zeroing, and shaping-regularised inversion stacking (Fomel
group). `seisvel` computes dt/dt0 **numerically along each trace**, so the mute is
correct for any V(t0) and either moveout law. Crossing trajectories (dt/dt0 ≤ 0)
are always muted, and the mute is applied from the bottom up so no live "islands"
are left. Inversion-based stretch-free stacking is listed as a future module.

### 1.4 Picking constraints that processors apply by eye
* Dix interval velocity must be real and geologically plausible.
* The trend is smooth and laterally consistent.
* Multiples have **lower** stacking velocity than primaries at the same time, so
  the pick should stay on the faster branch.
* Shallow far-offset data are unreliable (stretch).

These become hard and soft constraints in `picking/dp.py`.

---

## 2. Automatic picking: what exists

* **Constrained optimisation / DP.** Toldi (1985) inverts for an interval-velocity
  model that best explains the semblance. Zhang (1991) uses constrained linear and
  non-linear optimisation. Guide-function constraints keep picks near a legacy or
  earlier velocity to avoid multiples (a common industrial patent theme).
* **Clustering.** k-means, DBSCAN and unsupervised ensembles (Wang et al. 2022,
  UEL). These need careful parameter tuning.
* **Supervised deep learning.**
  * Park & Sacchi (2020): CNN as classification; transfer learning when the field
    data differ from training.
  * Wang, McMechan et al. (2021, *Geophysics* 86(2) U1–U13): U-Net
    regression vs classification. Regression was more accurate.
  * YOLO detection + LSTM (Zhang et al.); multi-information fusion segmentation
    (Wang et al. 2022).
  * Attention networks with horizon constraints (Qiu et al. 2022). Their field test
    is visual only, with no quantitative error metrics.
* **Common gaps** in the DL literature: very few labelled field data; evaluation is
  often visual; outputs rarely guaranteed physically valid (Dix); weak handling of
  multiples; little uncertainty quantification.

---

## 3. Where AI helps — and the design chosen here

Velocity estimation from surface data is non-unique, and a network cannot remove
that. What AI *can* do is three things:

1. **Replace the human's pattern recognition on the spectrum.** It learns what
   primary energy looks like versus multiples, noise smears, AVO-distorted or
   stretched peaks — and does so consistently and fast.
2. **Quantify confidence**, so humans only review the doubtful locations.
3. **Adapt to a survey from a handful of human picks** (transfer learning).

### Architecture: "AI proposes, physics disposes"
```
spectra (semblance, AB, row-normalised) ──► U-Net ──► P(v | t)   (probability map)
                                                        │
          Dix bounds, smoothness, guide, marine floor ──► Viterbi DP ──► v(t), picks
                                                        │
                        MC-dropout + distribution spread ──► σ_v(t) ──► review queue
```
* The network is **never the final answer**. Its probability map replaces the
  semblance as the DP score, so every AI pick still satisfies the Dix and
  smoothness constraints. This answers the main objection to DL pickers:
  physically impossible output.
* Training data come from the platform's own physics-based generator, with heavy
  domain randomisation: geology, acquisition, noise, multiples, AVO, wavelet and Q.
  The spectra are computed with **the same code** used in production, so there is
  no train/inference mismatch in preprocessing.
* `ml.train.fine_tune` freezes the encoder and adapts the decoder from a few
  human-picked locations of a new survey.

### Physics-based automation (no AI needed)
* **Regional guide + lateral consensus.** Primaries are laterally consistent.
  Averaging spectra along the line and taking a median of first-pass picks
  suppresses location-specific failures.
* **Marine multiple-aware decoding.** Water bottom is auto-detected. The water
  velocity is allowed only above it, with a sediment Vint floor below. Water-layer
  peg-leg trajectories, predicted from the current primary pick
  (V_m² = (V_p² t_p + v_w² t_wb)/t_m), are iteratively attenuated.

### Honest limits
* Quantitative results are on synthetics (1.5-D, no dip, no statics). The only
  field test so far is a 4-shot excerpt with no ground truth (§4b).
* Parabolic Radon demultiple is implemented (`radon.py`, `workflow.py`), with the
  caveats in §4.
* η estimation is implemented (scan) but not yet automated in the picker.

---

## 4. Benchmark results (synthetic, v0.2 — `examples/benchmark.py`)

These are two unseen 121-CMP lines, written to SEG-Y and read back through the
real-data path. Analysis runs every 10 CMPs on 5-CMP supergathers.
* **Marine line:** hard water bottom, multiples at full strength, SNR 2.5, swell
  noise, spikes, 3 % dead traces.
* **Land line:** ground roll, SNR 1.5, irregular and missing offsets, 5 % dead
  traces, trace-amplitude jitter.

Errors are reported against two references:
* **Vrms** (the model RMS velocity).
* **V_stack** (the best-fit hyperbola to the ray-traced moveout, which is what
  semblance actually measures). On these 3–3.5 km spreads, V_stack is 1–1.5 %
  above Vrms.

Stack correlation is the mean over 0.5 s windows vs the stack made with the true
velocities. The neural picker is v1: 6 200 synthetic spectra, best-validation
checkpoint.

| method | marine MAPE vs Vrms | vs V_stack | marine stack corr | land MAPE vs Vrms | vs V_stack | land stack corr |
|---|---|---|---|---|---|---|
| naive max-semblance | 14.7 % | 14.8 % | – | 14.7 % | 14.7 % | – |
| constrained DP, single location | 9.5 % | 9.8 % | – | 1.07 % | 1.11 % | – |
| line workflow (regional guide + lateral consensus) | 1.64 % | 1.47 % | 0.925 | 1.21 % | 1.21 % | 0.986 |
| line + marine multiple-aware | 1.59 % | 1.41 % | 0.927 | – | – | – |
| line, 2-pass with Radon demultiple | 1.75 % | 1.72 % | 0.901 | – | – | – |
| **AI + constrained DP (line)** | **0.61 %** | **0.73 %** | **0.964** | **0.59 %** | **0.55 %** | **0.991** |
| AI, 2-pass with Radon demultiple | 1.34 % | 1.22 % | 0.880 | – | – | – |

Reading the numbers honestly:
* Physics constraints (Dix bounds plus lateral consistency) give the largest gain.
* The AI picker halves the remaining error even when it is scored against V_stack,
  the reference that favours semblance-based picking.
* **Radon demultiple did not help on this synthetic line.** The line workflow
  already avoids the multiples, so demultiple can only cost something. With a
  q-cut of 20 ms at far offset, the ~1.5 % velocity uncertainty of pass 1 already
  produces primary residual moveout of that size, and some primary energy is
  removed. Larger q-cuts leave more multiple energy: pass-2 error was 4.0 % at
  50 ms and 6.3 % at 80 ms. The AI picker also gets worse on demultipled data
  because it never saw demultipled spectra in training (domain shift).
* Demultiple *does* help where multiples dominate one location. On a single
  hard-water-bottom gather, DP error drops from 16.3 % to 1.9 % after demultiple
  with correct velocities, against 1.7 % for primaries only (`tests`).
* Demultiple also clearly helps the field excerpt below.
* Defaults chosen from this: AI picking on raw data. Demultiple is used for the
  stack and for semblance picking where multiples dominate (`--two-pass`). Adding
  demultipled examples to the AI training set is on the roadmap.

Speed (2 CPU cores): about 0.3 s per spectrum, 0.05 s per DP pick, about 6 s for
AI picking of a 13-location line, and 2–4 min for the 2-pass flow (Radon on
every supergather trace).

Implementation lesson: a Dix constraint evaluated between 20 ms nodes on a 25 m/s
velocity grid is not resolvable at late times. The decoder works on a refined
5 m/s axis with exact, binary-searched bounds.

## 4b. Field test 1 — Viking Graben excerpt (4 shots)
This is the small sample shipped with segyio, run through
`examples/field_viking_small.py`. It confirmed that the real-data path works and
that the water-velocity multiple trap appears. No conclusions about velocities.

## 4c. Field test 2 — full Mobil AVO Viking Graben line 12 (`examples/field_viking_line.py`)
The data come from SEG open data: 1 001 shots × 120 channels, 6 s at 4 ms,
CDP 1–2142, fold 60. There were 39 analysis locations (every 50 CDPs, 5-CDP
supergathers). The well logs are *not* in the open-data bucket, so validation is
seismic-only. Figures are in `docs/field/`.

**What happened on real data**
1. **Physics pass 1** (multiple-aware DP, line consensus) is trapped below about
   2.5 s by a deep multiple train: a vertical semblance column at ~1 950 m/s down to
   6 s. The water-layer peg-leg masking does not cover multiples generated by
   deeper strong reflectors.
2. **AI pass 1** (trained on synthetics only) escapes the multiples but
   overshoots: ~2 870 m/s at 4 s. The two pass-1 answers differ by 36 %
   (> 0.8 s).
3. **Field workflow** (`workflow.field_velocity_workflow`):
   1. Radon demultiple with the *geometric mean* of the two pass-1 fields. That
      velocity lies between the multiples and the primaries, which is what
      parabolic Radon needs.
   2. Everything above 1.9 × the water-bottom time is protected, because no
      surface multiple can arrive earlier. Without this, the first version
      visibly weakened the water-bottom zone of the stack.
   3. Constrained DP re-pick on the demultipled semblance, run once guided by each
      pass-1 field (σ = 8 %).
   4. Final = mean of the two re-picks. Their coherence-weighted spread is the
      uncertainty and drives the review queue.

**What we tried and rejected: residual velocity analysis (RVA) as the
refinement.** Started from both pass-1 curves on the real line, RVA "converged"
to a 3.5 % spread, and held-out RMO bias was ≈ 0. It looked like a breakthrough.
The synthetic check with known truth showed it is not:
* RVA drifts away from the true velocity with iterations, even when started *at*
  the truth (≈ 15 % error after 12 iterations).
* The line workflow with RVA ended at 5.2 % (marine) / 2.9 % (land), against
  0.6–1.2 % for either picker alone.
* Agreement between starts therefore meant both drifted to the same biased
  attractor.
* The held-out RMO check used the same criterion RVA optimises, so it could not
  catch this.

RVA remains available as `refine="rva"`, marked experimental.

**Results of the DP-refined workflow**

| | synthetic marine MAPE vs Vrms | synthetic land MAPE vs Vrms | Viking held-out RMO bias | Viking held-out flatness (demultipled) |
|---|---|---|---|---|
| physics pass 1 | 1.20 % | 0.88 % | −54 ms (too slow) | 0.035 |
| AI pass 1 | **0.56 %** | **0.54 %** | +20 ms (too fast) | 0.176 |
| **final (demultiple + guided DP)** | 0.92 % | 0.81 % | −21 ms | **0.204** |

The synthetic numbers come from a separate run (13 locations, physics pass 1 via
`autopick_line`), so they differ slightly from §4.

Final Viking trend (line average): 1 680 m/s at 1 s, 1 950 at 2 s, 2 220 at 3 s,
2 420 at 4 s and 2 530 at 5 s. On the demultipled spectra it follows the primary
energy column; see `docs/field/viking_line_final_comparison.png`.

**Honest reading**
* On real data the final field is the flattest physically valid answer and sits
  on the primary energy exposed by demultiple. It removes the gross error of the
  physics picker (multiple trap). Held-out RMO suggests it is still slightly slow,
  while the AI is slightly fast.
* The two guided re-picks stay **27 % apart** on average at depth. Every location
  is flagged for review. The data alone do not settle the deep velocities between
  the AI trend and the DP trend. This is the non-uniqueness you predicted, made
  measurable.
* On synthetics the DP-refined workflow is safe (better than physics, a little
  worse than AI). The AI alone remains the most accurate there, but it has not
  been validated on field data.
* Decisive next check: the two wells on line 12 (Keys & Foster, SEG open-file
  report #4). Their logs were not found in the open-data bucket.
* The Dix projection and RVA material in the code is kept for research.
  `docs/field/viking_line_stacks.png` and `viking_line_velocities.png` were made
  with the earlier RVA final, so read them as the RVA experiment.

---

## 5. Recommended roadmap (product view)
1. **Well-tie validation** of the Viking Graben result (logs from the Keys & Foster
   report or another source), then the Poland 2D Vibroseis line (land). Fine-tune
   the AI on 20–40 human-checked locations per survey (`ml.train.fine_tune`). The
   field workflow's review queue says which locations to check first.
2. Generalised multiple prediction for the physics picker: interbed and peg-legs
   from deep reflectors, not only the water layer.
3. Demultiple v2: add demultipled examples to AI training, time-variant q-cut,
   and hyperbolic / high-resolution Radon on noisy data. SRME is the longer-term
   option.
3. Residual moveout picking and automated η (two-step).
4. Stretch-free (inversion) stacking.
5. Active learning loop: the review queue feeds human corrections into `fine_tune`.
6. 3-D: velocity cube interpolation exists. Add a picking GUI for inline/xline grids
   and GPU kernels for production-size data.

## References (main)
* Alkhalifah, T. & Tsvankin, I. (1995). Velocity analysis for transversely isotropic media. *Geophysics* 60.
* Alkhalifah, T. (1997). Velocity analysis using nonhyperbolic moveout in TI media. *Geophysics* 62.
* Biondi, B. & Kostov, C. (1989). High-resolution velocity spectra using eigenstructure methods. *Geophysics* 54.
* Chen, Y., Liu, T. & Chen, X. (2015). Velocity analysis using similarity-weighted semblance. *Geophysics* 80(4) A75–A82.
* Ebrahimi, S. et al. (2017). A high-resolution weighted AB semblance for dealing with AVO. *Geophysics* 82(2) V85.
* Fomel, S. (2009). Velocity analysis using AB semblance. *Geophysical Prospecting* 57, 311–321.
* Luo, S. & Hale, D. (2012). Velocity analysis using weighted semblance. *Geophysics* 77(2) U15–U22.
* Miller, R. D. (1992). Normal moveout stretch mute on shallow-reflection data. *Geophysics* 57(11).
* Neidell, N. & Taner, M. T. (1971). Semblance and other coherency measures for multichannel data. *Geophysics* 36.
* Park, M. J. & Sacchi, M. D. (2020). Automatic velocity analysis using CNNs and transfer learning. *Geophysics* 85.
* Perroud, H. & Tygel, M. (2004). Nonstretch NMO. *Geophysics* 69.
* Qiu, C. et al. (2022). Horizontal layer constrained attention neural network for semblance velocity picking. arXiv:2207.00380.
* Sarkar, D., Castagna, J. P. & Lamb, W. (2001). AVO and velocity analysis. *Geophysics* 66, 1284–1293.
* Taner, M. T. & Koehler, F. (1969). Velocity spectra — digital computer derivation and applications. *Geophysics* 34, 859–881.
* Trickett, S. (2003). Stretch-free stacking. CSEG convention.
* Wang, H. et al. (2022). Automatic stack velocity picking using an unsupervised ensemble learning method. arXiv:2205.08372.
* Wang, W., McMechan, G. A., Ma, J. & Xie, F. (2021). Automatic velocity picking from semblances with a new deep-learning regression strategy. *Geophysics* 86(2) U1–U13.
* Yilmaz, Ö. (2001). *Seismic Data Analysis*. SEG.
