# Grokking Paper — Data Catalog

All experimental data backing the paper. Every number in the manuscript traces to a JSONL path here.

## Data Locations

- **Modal volume**: `fc-results` at `atlas/<experiment>/`
- **Local mirror**: `experiments/results/<experiment>/`
- **Scripts**: `experiments/batch6_atlas/geometry/`

---

## 1. Depth-2 DMF: Architecture Hierarchy (Exp 1-6)

**Paper section**: Section 3 (minimal grokking architecture)

| File | Records | What |
|------|---------|------|
| `depth2_dmf/exp1.jsonl` | 202 | Depth-1 vs depth-2 (no weight decay) |
| `depth2_dmf/exp2.jsonl` | 505 | Grokking time vs init scale epsilon |
| `depth2_dmf/exp3.jsonl` | 808 | Factorization rank as controllable grokking knob |
| `depth2_dmf/exp4.jsonl` | 501 | SV trajectory (greedy rank-1 augmentation) |
| `depth2_dmf/exp5.jsonl` | 303 | Cross-architecture: depth-2 DMF vs 1-layer transformer |
| `depth2_dmf/exp6.jsonl` | 404 | Nuclear norm: explicit vs implicit vs none |

**Script**: `depth2_dmf_grokking.py`
**Modal**: `atlas/depth2_dmf/{1-6}/`
**Key fields**: `epoch, train_loss, test_loss, train_acc, test_acc, svd{effective_rank, condition_number, nuclear_norm, spectrum}, fourier{sv_alignments}`
**Key result**: Linear DMF never groks. Nuclear norm drops 7340->3877 but test acc stays ~0%.

---

## 2. DMF Followups: Nonlinearity + AGOP (Exp 7-12)

**Paper section**: Section 3 (minimal architecture), Section 5 (AGOP)

| File | Records | What |
|------|---------|------|
| `depth2_dmf_followups/exp7.jsonl` | 303 | DMF + ReLU (groks 18x faster than transformer) |
| `depth2_dmf_followups/exp8.jsonl` | 303 | Linearized transformer (identity attn + ReLU MLP) |
| `depth2_dmf_followups/exp9.jsonl` | 202 | Softmax-only transformer (no MLP, never groks) |
| `depth2_dmf_followups/exp10.jsonl` | 202 | **AGOP tracking** (101 DMF + 101 transformer) |
| `depth2_dmf_followups/exp11.jsonl` | 606 | Tasks where DMF should grok (mult, parity, low-rank) |
| `depth2_dmf_followups/exp12.jsonl` | 1,002 | Fine-grained Fourier trajectory (100-epoch resolution, 501 ckpts) |

**Script**: `depth2_dmf_followups.py`
**Modal**: `atlas/depth2_dmf_followups/{7-12}/`

### AGOP detail (exp10)

202 records: 101 per architecture (`transformer_wd1`, `dmf_wd1`), every 500 epochs across 50k.

**AGOP fields**: `agop_fourier_alignment, agop_effective_rank, agop_svd{spectrum}, agop_circulant_metric`

**Key numbers (transformer)**:

| Epoch | Test Acc | AGOP FA | Weight FA | AGOP Erank |
|-------|----------|---------|-----------|------------|
| 5,000 | 0.4% | 0.108 | 0.080 | 10.0 |
| 25,000 | 6.3% | **0.712** | 0.081 | 9.8 |
| 30,000 | 19.1% | **0.894** | 0.075 | 8.7 |
| 35,000 | 95.8% | 0.662 (dip) | 0.270 | 5.4 |
| 40,000 | 100% | **0.974** | 0.864 | 4.7 |
| 45,000 | 100% | 0.888 | 0.977 | 4.4 |

**DMF AGOP**: Flat 0.06-0.08 throughout. No structure. Erank ~60 (unchanged).

**Double-peak**: 0.894 (ep30k) -> 0.662 (ep35k, transition dip) -> 0.974 (ep40k, crystallization).

### Fine-grained Fourier trajectory (exp12)

1,002 records at 100-epoch resolution. Shows Fourier alignment crosses 0.30 ~1000 epochs AFTER test accuracy crosses 90%.

**Key numbers**:

| Epoch | Test Acc | Fourier Align |
|-------|----------|---------------|
| 37,000 | 65.6% | 0.070 |
| 37,500 | 91.3% | 0.093 |
| 38,000 | 98.6% | 0.143 |
| 38,500 | 99.7% | 0.260 |
| 39,500 | 99.7% | 0.498 |
| 40,000 | 99.8% | 0.543 |

### Grokking speed hierarchy (exp7-9)

| Architecture | Grok Epoch | Fourier Align |
|-------------|-----------|--------------|
| DMF + ReLU | ~2,000 | 0.095 |
| MLP-only | ~2,500 | 0.99 |
| Linearized TF | ~14,000 | 0.000 |
| Full transformer | ~37,000 | 0.10->0.54 |
| DMF linear | Never | 0.078 |
| Softmax-only | Never | 0.103 |

---

## 3. DAS vs SVD at Grokking (Exp A-E)

**Paper section**: Section 6 (weight-space vs activation-space)

| File | Records | What |
|------|---------|------|
| `das_vs_svd_grokking/exp_A.jsonl` | 21 | DAS sweep: 3 sites x 7 d_sub values |
| `das_vs_svd_grokking/exp_B.jsonl` | 210 | SVD sweep: per-matrix (FF/QK/OV/combined) x sites x d_sub |
| `das_vs_svd_grokking/exp_C.jsonl` | 18 | Grassmannian distance: DAS vs SVD subspaces |
| `das_vs_svd_grokking/exp_D.jsonl` | 36 | Combined DAS+SVD at matched d_sub |
| `das_vs_svd_grokking/exp_E.jsonl` | 608 | Spectral profile trajectory (per-epoch rank/condition/spectrum) |

**Script**: `unified/das_vs_svd_grokking.py`
**Modal**: `atlas/das_vs_svd_grokking/{A-E}/unified`

**Key numbers**:

| Method | Site | Best d_sub | IIA | Optimization |
|--------|------|-----------|-----|-------------|
| SVD(FF) | attn_out | 32 | **0.999** | None |
| DAS | attn_out | 8 | 0.150 | 1,000 steps |
| DAS | mlp_out | 8 | **0.631** | 1,000 steps |
| SVD(FF) | mlp_out | 64 | 0.444 | None |

Grassmannian distance between DAS and SVD subspaces: 1.5-8.2 radians (orthogonal).

---

## 4. Spectral Compression (Rank Collapse)

**Paper section**: Section 4

Data comes from two sources:

### Exp E (spectral trajectory)
`das_vs_svd_grokking/exp_E.jsonl` — 608 records tracking per-matrix effective rank through training.

**Key numbers (FF product matrix)**:

| Epoch | FF Erank | Stage |
|-------|----------|-------|
| 34,000 | 48 | Pre-grok |
| 37,500 | 14 | Transition |
| 41,000 | 9 | Post-grok |

5.3x compression.

### Exp 5 (cross-architecture)
`depth2_dmf/exp5.jsonl` — 303 records comparing DMF and transformer rank trajectories.

---

## 5. Factorized Grokking Geometry (Bank-Selector)

**Paper section**: Section 7

| Location | Configs | Records | JSONL Files |
|----------|---------|---------|-------------|
| `factorized_grokking_geometry/` | 98+ | 1,724+ | 188 |

**Script**: `factorized_grokking_geometry.py`
**Modal**: `atlas/factorized_grokking_geometry/` (94 subdirectories)
**Summary CSV**: `factorized_grokking_geometry/summary.csv` (78 core rows)

### Config structure
Directory names: `operation|selector|objective|aux|n_factors|fit_steps[|anneal_start|anneal_end]`

### Operations (13 total)
addition, subtraction, multiplication, division, composite_addition, squaring, cubing, polynomial, max_ab, abs_diff, affine, cubic_sum, sum_of_squares

### Selector types
- **Dense**: lambda=0, 0% sparsity, MSE~1e-12
- **JumpReLU-tanh**: lambda=0/0.01/1.0, 70-99% exact zeros, MSE~1e-4 to 1e-2
- **L1**: lambda=0.5/5/15/30, 0% exact zeros, soft shrinkage, MSE~1e-3 to 0.5

### Anneal variants (addition, division, max_ab)
- Two-phase: `500|500` (train 50% unregularized, then full lambda)
- Soft ramp 1k: `1|1000` (linear ramp over 1000 steps)
- Soft ramp 4k: `1|4000` (linear ramp over 4000 steps)

### Key fields per record
`epoch, stage, train_loss, test_loss, total_mse, factorization{effective_rank, svd_spectrum, mean_cos_sim, condition_number}, selector_sparsity{sparsity, weight_l1, weight_l2, effective_rank}, das{iia_mean, iia_std, mean_grassmann_dist}, alignment{factor_das_overlap, factor_das_grassmann}`

### Saved checkpoints
`factorized_grokking_geometry/checkpoints/` contains per-config:
- `*_factorized.pt` — full FactorizedHookedTransformer state dict
- `*_grokking.pt` — base grokking model checkpoint
- `*_selector.npy` — raw selector weight matrix (1536 x 128)
- `*_factors.npy` — factor bank (128 x 128)
- `*_cfg.pt` — HookedTransformerConfig

### Key results

**Pareto frontier**: JRT dominates L1 by 10-100x MSE at matched erank.

**Grokking operations** (4/9): addition, composite_addition, division, multiplication.
**Non-grokking**: subtraction, squaring, cubing, polynomial, max_ab.

**DAS comparison**: IIA=1.0 at all grokking stages (k=8 at mlp_out). Factor bank erank tracks grokking (119->66) while DAS is blind.

---

## 6. Grokking MLP Geometry (DAS Subspace Stability)

**Paper section**: Section 3 (supplementary)

| Location | Records | What |
|----------|---------|------|
| `grokking_mlp_geometry/addition/grokking_mlp_geometry.jsonl` | 8 | DAS refits at 4 stages, Grassmannian scatter |

**Script**: `grokking_mlp_geometry.py`
**Modal**: `atlas/grokking_mlp_geometry/addition/`
**Key fields**: `operation, epoch, stage, train_loss, test_loss, did_grok, k, n_refits, geometry, mean_grassmann_dist, mean_iia`

---

## 7. Grokking Nonlinear Hunt (16+ Operations)

**Paper section**: Section 3 (operation taxonomy)

**Modal**: `atlas/grokking_nonlinear_hunt/` — 94 subdirectories (16 base ops + multi-seed)
**Local**: `experiments/results/atlas_seeds_v2/` (55 seed directories)
**Script**: `grokking_nonlinear_hunt.py`
**Key fields**: `operation, grokked, intrinsic_dimension, circular_r2_top10, discovered_key_freqs, equivariance, circle_geometry, trajectory`

**Operations**: multiplication, composite_addition, division, subtraction, squaring, cubing, max_ab, abs_diff, sum_of_squares, power, shifted_mult, min_ab, floor_div, bitwise_xor, polynomial, affine, cubic_sum, modular_distance, quartic_sum, quintic_sum, affine_scaled

---

## 8. Grokking DAS Emergence

**Paper section**: Section 6 (supplementary)

| Location | Records | What |
|----------|---------|------|
| `grokking_das_v1.jsonl` | 1 | DAS IIA trajectory during training (50 checkpoints inside) |

**Modal**: `atlas/grokking_das/modular_addition/`
**Script**: `grokking_das_emergence.py`

---

## 9. Grokking Torus Geometry

**Paper section**: Section 3 (supplementary)

| Location | Records | What |
|----------|---------|------|
| `grokking_torus_geometry.jsonl` | 2 | Torus/circle geometry of causal subspace |
| `grokking_torus_geometry_v3.jsonl` | 1 | Updated |
| `grokking_torus_v5_controls.jsonl` | 1 | With random controls |

**Modal**: `atlas/grokking_torus_geometry/modular_addition/`
**Script**: `grokking_torus_geometry.py`
**Key fields**: `intrinsic_dimension, circular_r2, key_freq_r2, equivariance, trajectory`

---

## 10. Optimal Transport Grokking

**Paper section**: Supplementary

**Modal**: `atlas/optimal_transport_grokking/` — 14 operations
**Script**: `geometry_wild/optimal_transport_grokking.py`
**Key fields**: `operation, grokked, w2_from_init, w2_to_circle, w2_velocity, w2_precedes_loss`

---

## 11. Grokking Nonlinear DSI

**Paper section**: Section 6 (supplementary)

| Location | Records | What |
|----------|---------|------|
| `grokking_nonlinear_dsi_v1.jsonl` | 1 | Nonlinear DSI ladder on grokked model |
| `grokking_nonlinear_dsi_v2.jsonl` | 1 | Updated |

**Script**: `grokking_nonlinear_dsi.py`
**Key fields**: `ladder[{featurizer, iia_genuine, iia_random_control}]`

---

## 12. Additional Modal Experiments

All have data on `atlas/` volume, 14 operations each:

| Experiment | Records | Key finding |
|-----------|---------|-------------|
| `fourier_mode_selection` | 89 | Which Fourier modes dominate per operation |
| `equivariance_emergence` | 9 | When equivariance emerges during training |
| `spectral_gap_convergence` | 105+ | Spectral gap theory connection |
| `spectral_gap_initialization` | varies | Init effects on spectral gap |
| `weight_decay_mode_selection` | 50 | Weight decay as Fourier mode selector |
| `circulant_attention_theorem` | varies | Circulant structure in attention |

---

## 13. Existing Figures

### depth2_dmf figures
`experiments/results/depth2_dmf/figures/` — DMF experiment plots

### followup figures
`experiments/results/followup_figures/`:
- `fig9_*` — grokking speed hierarchy
- `fig10_*` — DMF+ReLU trajectory
- `fig11_*` — architecture ablation
- `fig12_*` — Fourier transition (fine-grained)
- `fig13_*` — DAS vs SVD comparison
- `fig14_*` — spectral compression
- `fig15_*` — DAS-SVD alignment

### factorized grokking figures
`factorized_grokking_geometry/figures/`:
- `pareto_mse_vs_erank.png` — JRT vs L1 Pareto frontier
- `sparsity_vs_mse.png` — hard gating vs soft shrinkage
- `erank_trajectory_comparison.png` — erank through grokking
- `selector_comparison_table.png` — full config table
- `selector_heatmaps.png` — visual selector matrices
- `selector_weight_distributions.png` — weight histograms
- `selector_factor_usage.png` — per-factor channel count
- `dense_vs_jrt.png`, `erank_trajectories.png`, `mse_comparison.png`, `fourier_vs_erank.png`, `sparsity_trajectories.png`, `sparsity_delta_bar.png`

### plotting scripts
- `depth2_dmf_figures.py` — DMF experiments
- `followup_figures.py` — exp 7-12
- `plot_factorized_grokking.py` — base factorization plots
- `plot_selector_comparison.py` — selector comparison
- `plot_selector_heatmaps.py` — selector heatmaps

---

## Total Data Inventory

| Dataset | Records | Configs/Exps | Paper Section |
|---------|---------|-------------|---------------|
| depth2_dmf (exp 1-6) | 2,723 | 6 experiments | S3 (architecture) |
| depth2_dmf_followups (exp 7-12) | 2,618 | 6 experiments | S3, S5 (AGOP) |
| das_vs_svd_grokking (A-E) | 893 | 5 experiments | S6 (SVD vs DAS) |
| factorized_grokking_geometry | 1,724+ | 98+ configs | S7 (bank-selector) |
| grokking_nonlinear_hunt | ~1,000+ | 21 ops x 10 seeds | S3 (operations) |
| grokking_mlp_geometry | 8 | 1 operation | S3 (supplementary) |
| grokking_das_emergence | 1 (50 ckpts) | 1 | S6 (supplementary) |
| grokking_torus_geometry | 4 | 1 | S3 (supplementary) |
| optimal_transport | 14 | 14 operations | Supplementary |
| nonlinear_dsi | 2 | 1 | S6 (supplementary) |
| fourier_mode_selection | 89 | 14 operations | S3 (supplementary) |
| **TOTAL** | **~9,000+** | **~150+ configs** | |
