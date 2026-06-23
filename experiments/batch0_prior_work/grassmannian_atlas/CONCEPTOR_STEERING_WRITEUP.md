# Weight-Space Conceptor Steering via DAS Subspace Algebra

## The Idea in One Sentence

Train DAS (Distributed Alignment Search) to find a rank-k causal subspace for each task in GPT-2's residual stream, then treat the resulting projection matrices as *conceptors* and use boolean algebra (AND, NOT, OR) to surgically isolate task-specific computation.

## Background: DAS and Conceptors

**DAS** (Geiger et al., 2024) finds a rank-k orthonormal basis Q in R^{d_model} such that swapping activations within span(Q) between clean and corrupted inputs changes the model's answer — i.e., the causal information for the task lives in this subspace. The output is Q (d_model x k), and the associated projection matrix P = QQ^T.

**Conceptors** (Jaeger, 2014) are soft projection matrices that support boolean algebra:
- **AND**: P_A AND P_B isolates the shared subspace
- **NOT**: I - P removes the subspace
- **AND NOT**: P_A(I - P_B) isolates what's in A but not B

The key insight: a DAS projection P = QQ^T *is already a conceptor*. No additional training needed. If you have DAS for IOI and DAS for SVA, you can immediately compute P_IOI AND NOT P_SVA to get "the directions that carry IOI but not SVA."

## Method

### Setup
- **Model**: GPT-2 (117M, pretrained)
- **Tasks**: IOI (Indirect Object Identification) and SVA (Subject-Verb Agreement)
- **DAS**: Rank-32 subspace per task at layer 11, trained on 200 counterfactual pairs from Hanna et al. (2024)
- **DAS quality**: IOI IIA = 0.94, SVA IIA = 0.995
- **Intervention point**: `blocks.11.hook_resid_post`

### Conceptor Construction

From each DAS checkpoint, extract the orthonormal basis Q (768 x 32) and form:
```
P_task = Q_task @ Q_task^T     (rank-32 projection, 768 x 768)
```

### Composed Conceptors

Given P_IOI and P_SVA:
```
P_IOI_only = SVD_threshold(P_IOI @ (I - P_SVA), sigma > 0.1)
P_SVA_only = SVD_threshold(P_SVA @ (I - P_IOI), sigma > 0.1)
```
Both composed conceptors retain rank 32 because the subspaces are nearly orthogonal.

### Steering Interventions

For each conceptor P and each example (clean input, corrupted input):

1. **Project-in**: Keep only the P-subspace from clean, replace everything else with corrupted:
   ```
   h_steered = P @ h_clean + (I - P) @ h_corrupted
   ```

2. **Project-out**: Remove the P-subspace from clean, fill with corrupted:
   ```
   h_steered = (I - P) @ h_clean + P @ h_corrupted
   ```

### Metric

Logit difference: logit(correct) - logit(incorrect) at the last token. Reported as % of clean baseline. 100% = clean performance, 0% = chance, negative = inverted.

---

## Results: Vanilla GPT-2, Layer 11, k=32

### Baselines

| | IOI | SVA |
|---|---|---|
| Clean | +3.88 CI [+3.64, +4.13] | +3.44 CI [+3.25, +3.63] |
| Corrupted (floor) | -3.83 CI [-4.05, -3.61] | -3.51 CI [-3.68, -3.33] |

### EXP 1: Project-In (Is the subspace sufficient?)

| Task | Logit diff | % baseline |
|---|---|---|
| IOI | +4.14 CI [+3.74, +4.54] | **106.7%** |
| SVA | +4.73 CI [+4.47, +5.01] | **137.4%** |

Both tasks recover *above* baseline. The >100% recovery means projecting into the causal subspace actually *removes noise/interference* from irrelevant directions — the subspace is not just sufficient, it's cleaner than the full residual stream.

### EXP 2: Project-Out (Is the subspace necessary?)

| Task | Logit diff | % baseline |
|---|---|---|
| IOI | -4.10 CI [-4.46, -3.72] | **-105.6%** |
| SVA | -4.82 CI [-5.08, -4.57] | **-140.2%** |

Both tasks completely destroyed. Performance goes *below* corrupted baseline (negative = model confidently gives the wrong answer). The rank-32 subspace contains ALL the causal information.

### EXP 3: Random Baseline (Is the DAS subspace special?)

| Operation | IOI | SVA |
|---|---|---|
| Random proj-in | -3.52 (-90.7%) | -3.30 (-95.8%) |
| Random proj-out | +3.56 (91.9%) | +3.22 (93.7%) |

A random rank-32 subspace does nothing: proj-in gives near-corrupted performance, proj-out barely hurts. The DAS subspace is genuinely special.

### EXP 4: Cross-Task Selectivity (Are the subspaces task-specific?)

| Operation | Logit diff | % baseline |
|---|---|---|
| SVA clean, project out P_IOI | +3.34 (97.0%) | Nearly unaffected |
| IOI clean, project out P_SVA | +3.69 (95.0%) | Nearly unaffected |

Removing the *wrong* task's subspace barely affects the *right* task. The two tasks use nearly orthogonal subspaces.

### EXP 5: Composed Conceptors (Can we surgically isolate one task?)

| Conceptor | IOI logit diff | SVA logit diff |
|---|---|---|
| P_IOI AND NOT P_SVA proj-in | **+4.14 (106.7%)** | **-3.41 (-99.1%)** |
| P_SVA AND NOT P_IOI proj-in | **-3.64 (-94.0%)** | **+4.73 (137.4%)** |

**This is the key result.** Each composed conceptor recovers its own task at or above baseline while completely destroying the other. P_IOI AND NOT P_SVA gives +106.7% IOI while taking SVA to -99.1%. Perfect surgical isolation via boolean algebra on DAS projections.

The composed results match the uncomposed results because the subspaces are already nearly orthogonal — AND NOT has nothing to remove.

---

## Grassmannian Geometry

### Canonical Angles

All 32 principal angles between span(Q_IOI) and span(Q_SVA):

| Statistic | Value |
|---|---|
| Geodesic distance | 7.962 |
| Mean angle | 1.402 rad (80.3 deg) |
| Min angle | 1.084 rad (62.1 deg) |
| Max angle | 1.561 rad (89.4 deg) |
| Angles > 1.2 rad | 29 / 32 |

**Predominantly orthogonal.** Even the most aligned direction (62 deg) has substantial separation. No shared directions below 1.08 rad.

### Shared Direction Logit Lens

The most aligned direction (theta_0 = 1.08 rad, cos_sim = 0.47) decoded through the unembedding matrix:

- **IOI side**: Christine, Mark, Thomas, Nicholas, Stephen — **proper names**
- **SVA side**: Francois, Roz, Charles, Cantor, Roose — **proper names** (different ones)

Both sides decode to names, but different names. This is a multiplexed "entity/name" channel — IOI uses it for name identity, SVA for subject-verb agreement.

### Geodesic Interpolation

11-point sweep along Gr(32, 768) from Q_IOI (t=0) to Q_SVA (t=1):

| t | IOI proj-in | SVA proj-in | IOI proj-out | SVA proj-out |
|---|---|---|---|---|
| 0.0 | +4.14 (107%) | -3.41 (-99%) | -4.10 (-106%) | +3.34 (97%) |
| 0.2 | +3.48 | -2.28 | -3.43 | +2.21 |
| 0.4 | +1.77 | -0.18 | -1.73 | +0.11 |
| **0.5** | **+0.69** | **+1.02** | **-0.65** | **-1.10** |
| 0.6 | -0.43 | +2.19 | +0.47 | -2.27 |
| 0.8 | -2.44 | +4.03 | +2.48 | -4.12 |
| 1.0 | -3.64 (-94%) | +4.73 (137%) | +3.69 (95%) | -4.82 (-140%) |

**Smooth monotonic crossover.** IOI performance decreases continuously while SVA increases, crossing near t=0.5. The Grassmannian geodesic smoothly rotates between one task's causal representation and the other's.

### Frechet Mean

| Distance | Value |
|---|---|
| d(mean, IOI) | 4.865 |
| d(mean, SVA) | 4.747 |
| d(IOI, SVA) | 7.962 |
| d(mean, random) | 7.928 +/- 0.024 |
| d(euclidean avg, frechet mean) | 5.111 |

The Frechet and Euclidean means differ by d=5.11 — the Euclidean shortcut through ambient R^768 gives a meaningfully different point than the intrinsic Grassmannian mean.

Frechet mean steering: captures 23% of SVA but only 0.7% of IOI. The SVA representation has a broader footprint.

### Sectional Curvature

Triangle (IOI, SVA, Random): angular excess = 0.328, consistent with the positive curvature of Gr(32, 768). Nearly equilateral triangle — both task subspaces are as far from random as from each other.

---

## SVA Task Ablation (Double Dissociation)

Fine-tune GPT-2 to break SVA (KL divergence: match corrupted output on clean SVA inputs).

**After 200 steps**: SVA accuracy = 0%, IOI accuracy = 36% (collateral damage).

**DAS on ablated model**: SVA DAS IIA = 0.99 (!) — the DAS probe *still finds a distinguishing subspace* even though the task is dead. The representation separates clean/corrupted but the model no longer acts on it.

**Subspace shift**: SVA subspace rotated by d=6.83 (86% of original IOI-SVA distance). The model rearranged its geometry, not just scaled down the signal.

---

## Comparison: Factorized GPT-2 (1024 factors, dense selectors)

| Experiment | Vanilla GPT-2 | Factorized GPT-2 |
|---|---|---|
| IOI project-in | **+106.7%** | -50.3% |
| SVA project-in | **+137.4%** | +0.5% |
| IOI project-out | -105.6% | +47.0% |
| SVA project-out | -140.2% | +5.3% |

The factorized model's conceptor steering is dramatically weaker. The shared factor bank creates a different activation geometry where DAS at the residual stream level doesn't capture the full causal information. The factorized model also has one partially shared direction (theta_0 = 0.42 rad = 24 deg) that vanilla lacks (theta_0 = 1.08 rad = 62 deg).

---

## Layer 9 Results (Vanilla GPT-2)

| Experiment | IOI | SVA |
|---|---|---|
| Project-in | +0.42 (10.7%) | +3.46 (100.6%) |
| Project-out | -0.39 (-10.0%) | -3.58 (-104.1%) |

Layer 9 captures SVA (full recovery) but misses IOI (10.7%). Consistent with IOI's circuit structure: name-mover heads operate at layers 9-10, so at layer 9 the IOI computation isn't concentrated yet.

---

## What This Shows

1. **DAS subspaces are causally sufficient and necessary** (vanilla GPT-2). Project-in recovers >100%, project-out destroys >100%. Random baseline is inert.

2. **IOI and SVA use orthogonal subspaces.** 29/32 canonical angles > 1.2 rad. Cross-task removal preserves 95-97%.

3. **Boolean algebra over causal subspaces works.** P_IOI AND NOT P_SVA achieves +106.7% IOI / -99.1% SVA. Perfect surgical task isolation from a single matrix operation on DAS outputs.

4. **Geodesic interpolation reveals smooth causal structure.** Continuous crossover along Gr(32, 768) from one task to the other. The manifold geometry reflects causal geometry.

5. **Ablation causes subspace rotation, not deletion.** Breaking SVA moves the DAS subspace by d=6.83 but DAS still achieves IIA=0.99 — the representation persists as structure without function.

6. **Factorized model degrades conceptor steering.** The shared factor bank creates a different geometry where residual-stream DAS doesn't work as well. Vanilla GPT-2 has cleaner task separation.

---

## Code

Implementation: [`conceptor_steering.py`](conceptor_steering.py)

Key functions:
- `load_das_conceptor(path, F)` — Build P = QQ^T from DAS checkpoint
- `canonical_angles(Q1, Q2)` — Principal angles via SVD
- `grassmann_geodesic(Q1, Q2, t)` — Point on Gr(k,d) geodesic
- `grassmann_frechet_mean(Qs)` — Riemannian mean via gradient descent
- `make_steering_hook(P, corr_cache, project_in)` — TransformerLens hook
- `decode_shared_directions(Q1, Q2, W_U, tokenizer)` — Logit-lens decode of canonical directions

Full results with CSVs and source traceability: [`TECHNICAL_REPORT_GRASSMANNIAN_EXPERIMENTS.md`](TECHNICAL_REPORT_GRASSMANNIAN_EXPERIMENTS.md)
