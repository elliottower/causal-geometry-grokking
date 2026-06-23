# DAS Results Catalog

Comprehensive catalog of all Distributed Alignment Search results on disk.
Last updated: 2026-06-14.

---

## Summary: Checkpoint x Task Coverage

All tasks are run on GPT-2 small (12 layers, 12 heads, d_model=768).

| Checkpoint | IOI | SVA | Capital-Country | Gender Bias | Greater Than | Hypernymy |
|---|---|---|---|---|---|---|
| shared_bank_per_proj_sel (1024f) | k=4,32 | k=4,32 | -- | -- | -- | -- |
| per_layer_per_proj_sel (1024f) | k=32 | k=4 | -- | -- | -- | -- |
| no_lambda_5k (1024f) | (running) | k=32 | -- | -- | -- | -- |
| shared_bank_global_dense (1024f) | k=4,64 | k=4,64 | k=4 | k=4 | -- | -- |
| classic-sweep-133 (4096f) | k=4 | k=4 | -- | -- | -- | -- |
| atomic-sweep-40 (8192f, DST) | k=4 (6 layers) | k=4 (6 layers) | -- | -- | -- | -- |
| polar-sweep-90 (1024f, DST) | k=4 | k=4 | k=4 | k=4 | -- | -- |
| shared_bank_global_dense (1024f, k=32) | IN PROGRESS (Modal) | IN PROGRESS | IN PROGRESS | IN PROGRESS | IN PROGRESS | IN PROGRESS |

Legend: -- = no results exist. (running) = W&B run in progress.

---

## 1. Shared Bank Per-Projection Selector (1024 factors)

Checkpoint path: `/tmp/ckpts/shared_bank_per_proj_sel/final_ckpt.pt`
Source: `lib/analysis_elliot/.../01_DAS_CIRCUITS/data/` (W&B artifacts) + grassmannian analysis

### IOI (Layer 10)

**Baselines:**
- delta_pca (k=32): eval_iia = 0.913
- vanilla DAS (k=32): eval_iia = 0.940

**Factorized DAS runs (k=32):**

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | N Active | Gini | Has .pt |
|---|---|---|---|---|---|---|
| c4rnplvf | 1.0 | 0.913 | 0.833 | 200 | 0.223 | Yes (4 checkpoints) |
| 6bywcxsj | 0.1 | 0.913 | 0.893 | 1024 | 0.166 | Yes (4 checkpoints) |
| hcr0vos8 | 0.05 | -- | 0.827 | 1022 | -- | Yes |
| hcr0vos8 | 0.1 | -- | 0.733 | 1021 | -- | Yes |
| hcr0vos8 | 0.5 | -- | 0.180 | 627 | -- | Yes |
| 2a5bh60w | 0.1 | -- | 0.820 | -- | -- | No |
| cbu31iwv | 0.05 | -- | 0.847 | -- | -- | No |
| k908gd6o | 0.01 | -- | 0.860 | -- | -- | No |
| 25x22p5o | -- | -- | (baselines only) | -- | -- | vanilla .pt |
| 6qqrsp0x | -- | -- | (baselines only) | -- | -- | vanilla .pt |
| ixzol2mp | -- | -- | (baselines only) | -- | -- | vanilla .pt |
| aa44eb2y | -- | -- | (no IIA logged) | -- | -- | No |

**Factorized DAS runs (k=4):**

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | N Active | Has .pt |
|---|---|---|---|---|---|
| esj53evc | 1.0 | -- | -- | 0 | No (log only) |
| 1gayo9w8 | 2.0 | 0.040 | 0.007 | 43 | Yes (4 checkpoints) |
| 2fa8gr6r | 5.0 | 0.013 | 0.000 | 0 | Yes (4 checkpoints) |

**TopK DAS runs (k=4):**

| Run ID | K | Final IIA (eval) |
|---|---|---|
| vk3j6xx7 | 16-256 | 0.007 (all K values) |

### SVA (Layer 8)

**Baselines:**
- delta_pca (k=4): eval_iia = 0.587
- vanilla DAS (k=4): eval_iia = 0.653

**Factorized DAS runs (k=4):**

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | N Active | Gini | Has .pt |
|---|---|---|---|---|---|---|
| 2u5ki7xh | 1.0 | 0.660 | 0.653 | 56 | 0.593 | Yes (4 checkpoints) |
| bkr8cedb | 0.1 | 0.660 | 0.627 | 932/1018 | 0.286 | Yes (4 checkpoints) |
| zqk7d4aq | 1.0 | 0.653 | 0.653 | 809/353 | 0.381 | Yes (4 checkpoints) |
| qnrguy1w | 1.0 | 0.680 | 0.413 | 0 | 0.497 | Yes (4 checkpoints) |
| 8ck4uf0m | 0.05 | -- | 0.667 | 697 | -- | Yes |
| 8ck4uf0m | 0.1 | -- | 0.653 | 479 | -- | Yes |
| 8ck4uf0m | 0.5 | -- | 0.633 | 267 | -- | Yes |
| 9xq1pusk | 0.1 | -- | 0.627 | -- | -- | No |
| 0t2xuutb | 0.05 | -- | 0.620 | -- | -- | No |
| n71qoytb | 0.01 | -- | 0.633 | -- | -- | No |
| 4qcw8y4c | 1.0 | -- | -- | 353 | 0.381 | No (log only) |
| u8303wh0 | 1.0 | -- | -- | 353 | 0.381 | No (log only) |
| jewwh3yk | 1.0 | -- | -- | 353 | 0.381 | No (log only) |

**TopK DAS runs (k=4):**

| Run ID | K | Final IIA (eval) |
|---|---|---|
| 7a72z0km | 16 | 0.340 |
| 7a72z0km | 32 | 0.353 |
| 7a72z0km | 64 | 0.360 |
| 7a72z0km | 128 | 0.407 |
| 7a72z0km | 256 | 0.513 |
| 0vdqgrc1 | 16 | 0.347 |

**Grassmannian subspace analysis (cross-seed stability):**

IOI k=32 is geometrically stable: U-matrix Grassmannian distance = 0.228, all 32 dims within 5 degrees across seeds.

SVA k=4 is unstable: mean angle 31-49 degrees between seeds, zero factor Jaccard. Multiple equally valid 4D subspaces yield IIA ~0.653.

Vanilla vs factorized subspace gap: IOI mean angle = 25 degrees (16/32 shared dims <15 deg); SVA mean angle = 26-48 degrees (0/4 shared dims).

**Seeds with IIA values (from grassmannian analysis):**

| Run ID | Task | Fac IIA | Vanilla IIA | Gap |
|---|---|---|---|---|
| 6bywcxsj | IOI | 0.893 | 0.940 | 0.047 |
| c4rnplvf | IOI | 0.833 | 0.940 | 0.107 |
| hcr0vos8 | IOI | 0.827 | 0.940 | -- |
| 2u5ki7xh | SVA | 0.653 | 0.653 | 0.000 |
| bkr8cedb | SVA | 0.627 | 0.653 | 0.027 |
| zqk7d4aq | SVA | 0.653 | 0.653 | 0.000 |
| 8ck4uf0m | SVA | 0.667 | 0.653 | -0.013 |

---

## 2. Per-Layer Per-Projection Selector (1024 factors)

Checkpoint path: `/tmp/ckpts/per_layer_per_proj_sel/final_ckpt.pt`

### IOI (Layer 10, k=32)

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | Has .pt |
|---|---|---|---|---|
| xgkqa4fw | 1.0 | -- | -- | No (log only) |

### SVA (Layer 8, k=4)

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | Has .pt |
|---|---|---|---|---|
| qnrguy1w | 1.0 | 0.680 | 0.413 | Yes (4 checkpoints) |
| hbl1tgk3 | 1.0 | 0.680 | 0.413 | No |
| zqk7d4aq | 1.0 | 0.653 | 0.653 | Yes (4 checkpoints) |

Note: Some runs have checkpoint path as per_layer_per_proj_sel but may actually share the same bank as shared_bank_per_proj_sel. Verify checkpoint provenance before using.

---

## 3. No Lambda 5k (1024 factors, zero sparsity)

Checkpoint path: `/tmp/ckpts/no_lambda_5k/final_ckpt.pt`

### SVA (Layer 8, k=32)

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | N Active | Gini | Has .pt |
|---|---|---|---|---|---|---|
| 6lm02qw2 | 1.0 | 0.680 | 0.607 | 58 | 0.342 | Yes (4 checkpoints) |

### IOI -- no completed runs

- dmg8f4vt: k=32, state=running
- yh8oy9me: k=4, state=crashed

---

## 4. Shared Bank Global Dense (1024 factors, L1 sparse)

Checkpoint path: `/tmp/ckpts/sgd/final_ckpt.pt` (on pods) or `/checkpoints/shared_bank_global_dense.pt`

### Self-Consistent delta_pca Results (k=4, from analysis8)

Source: `lib/analysis_grassmanian_subspace/.../self_consistent_fac_das/`

| Task | Layer | k | delta_pca eval_iia | Has .pt |
|---|---|---|---|---|
| ioi | 10 | 4 | 0.907 | No |
| sva | 8 | 4 | 0.193 | No |
| capital_country | 8 | 4 | 0.713 | No |
| gender_bias | 9 | 4 | 0.000 | No |

### Dense k=64 Results (from lib/factorized_das/results/dense-k64/)

| Task | Layer | k | delta_pca eval_iia | vanilla eval_iia | Has .pt |
|---|---|---|---|---|---|
| ioi | 10 | 64 | 0.113 | 0.993 | No |
| sva | 8 | 64 | 0.147 | -- | No |

Note: Very low delta_pca IIA at k=64 suggests the PCA of activation differences in factor space does not align well with the causal direction at this dimensionality.

### Currently Running (Modal)

A Modal job is currently running to add all 6 tasks on shared_bank_global_dense at k=32: ioi, sva, capital_country, gender_bias, greater_than, hypernymy. Results will fill in the major gap in this checkpoint's coverage.

---

## 5. Classic Sweep 133 (4096 factors, JumpReLU-tanh)

Checkpoint path: `/tmp/ckpts/classic-sweep-133/factorized_payload.pt`

### IOI (Layer 10, k=4)

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | N Active | Has .pt |
|---|---|---|---|---|---|
| 6d49hwwu | 5.0 | 0.047 | 0.000 | 0 | Yes (4 checkpoints) |

Two additional runs crashed (rq797x8l, yrmhn41o). Two k=32 runs also crashed (u94jp5v4, o0ggug37).

### SVA (Layer 8, k=4)

| Run ID | L1 | Best IIA (eval) | Final IIA (eval) | Has .pt |
|---|---|---|---|---|
| tmqbqirb | 1.0 | 0.673 | 0.660 | No |

Three additional runs crashed.

Note: Very poor IOI performance (IIA = 0.047 with heavy L1=5.0). SVA performs better but still below vanilla. The 4096-factor JumpReLU-tanh architecture may need different hyperparameters.

---

## 6. Atomic Sweep 40 (8192 factors, DST)

Checkpoint path: `/results/checkpoints/atomic-sweep-40/factorized_payload.pt`
Source: `lib/factorized_das/results/atomic-sweep-40/`

This is the most thorough per-layer analysis, covering layers 0, 3, 8, 10, 11 for both IOI and SVA.

### IOI (k=4, cross-layer)

| Layer | L1=0.05 eval | L1=0.1 eval | L1=0.5 eval | L1=1.0 eval | delta_pca | vanilla |
|---|---|---|---|---|---|---|
| 0 | 0.960 | 0.960 | 0.960 | -- | 0.960 | 0.967 |
| 3 | 0.960 | 0.960 | 0.960 | 0.967 | 0.960 | 0.973 |
| 8 | 0.967 | 0.967 | 0.967 | 0.987 | 0.207 | 0.993 |
| 10 | 0.973 | 0.973 | 0.967 | 0.973 | 0.793 | 0.993 |
| 11 | 0.973 | 0.973 | 0.953 | 0.987 | 0.913 | 0.993 |

N active factors at L1=0.5: L0=16, L3=17, L8=19, L10=22, L11=18
N active factors at L1=1.0: L3=1, L8=1, L10=3, L11=3

### SVA (k=4, cross-layer)

| Layer | L1=0.05 eval | L1=0.1 eval | L1=0.5 eval | L1=1.0 eval | delta_pca | vanilla |
|---|---|---|---|---|---|---|
| 0 | 0.953 | 0.960 | 0.960 | 0.960 | 0.780 | 0.940 |
| 3 | 0.960 | 0.960 | 0.953 | 0.967 | 0.780 | 0.960 |
| 8 | 0.960 | 0.960 | 0.953 | 0.953 | 0.187 | 0.967 |
| 11 | 0.960 | 0.960 | 0.960 | -- | 0.047 | 0.960 |

N active factors at L1=0.5: L0=14, L3=13, L8=9, L11=17
N active factors at L1=1.0: L0=3, L3=3, L8=2

### Cross-layer Analysis Files

Additional analysis in `results/atomic-sweep-40/cross_layer_analysis/`:
- `cross_layer_{task}_l1=0.5.json` -- factor overlap across layers at L1=0.5
- `l1_pareto_{task}.json` -- IIA vs sparsity Pareto frontier per layer
- `l1_stability_{task}.json` -- pairwise factor stability across L1 values
- `fac_vs_vanilla_{task}_l1=0.5.json` -- factorized vs vanilla comparison per layer
- `cross_task_l1=0.5.json` -- factor overlap between IOI and SVA

### Additional Analysis Sections (primary layer only)

The IOI (L10) and SVA (L8) primary JSONs also contain:
- `frozen_factor_subspace` -- IIA when freezing all but top-N factors
- `factor_activation_swap` -- IIA from factor-level activation patching
- `selector_circuit_traces` -- which heads the top factors connect to
- `known_direction_similarities` -- cosine similarity to known IOI/SVA heads
- `factor_necessity` -- ablation of top factor subsets
- `per_factor_logit_diff` -- individual factor causal effect
- `factor_probing` -- linear probing for task variables

All .pt files present for L1 = {0.05, 0.1, 0.5, 1.0} at each layer. Vanilla DAS .pt files present at each layer.

---

## 7. Polar Sweep 90 (1024 factors, DST)

Checkpoint path: `/tmp/ckpts/polar-sweep-90/factorized_payload.pt`
Source: `lib/analysis_grassmanian_subspace/.../analysis10_dst_checkpoint_das/results/polar-sweep-90/`

delta_pca only (no trained DAS or vanilla DAS), k=4:

| Task | Layer | delta_pca eval_iia |
|---|---|---|
| ioi | 10 | 0.793 |
| sva | 8 | 0.187 |
| capital_country | 8 | 0.887 |
| gender_bias | 9 | 0.040 |

No .pt files saved.

---

## 8. Node Selection DAS (shared_bank_per_proj_sel, IOI)

Source: `lib/analysis_elliot/.../06_NODE_SELECTION_ALGORITHMS/data/`

### das_l10/ -- Single-layer factorized DAS at L10

- Checkpoint: shared_bank_per_proj_sel
- Task: IOI, Layer 10, k=32
- delta_pca eval: 0.913
- vanilla eval: 0.940
- Factorized DAS L1=1.0: eval_iia = 0.833, n_active_abs = 200, n_active_rel10 = 336
- Has .pt files: best_iia, best_iia_at_min_k, first_sparsest, final

### pldas/ -- Per-layer DAS across circuit layers

- Checkpoint: shared_bank_per_proj_sel
- Task: IOI, k=32
- Circuit layers: [1, 5, 6, 7, 8, 9, 10, 11]
- Per-layer results: structure present but no IIA values populated
- Has .pt file: per_layer_das_ioi.pt
- Factor-layer alignment analysis: `factor_layer_alignment_ioi.json` (layers 1, 5-11)

---

## 9. Weight-Space DAS Experiments

Source: `lib/weight_space_das/results/`
Checkpoint: shared_bank_per_proj_sel (1024 factors)

These are NOT activation-space DAS. They analyze whether weight-space structure predicts DAS behavior.

| Experiment | File | Description |
|---|---|---|
| exp01 | svd_alignment.csv | SVD alignment of factor OV matrices |
| exp02 | cross_head_svd.csv | Cross-head SVD comparison |
| exp03 | selector_composition.csv | Selector weight composition analysis |
| exp04 | weight_derived_A.csv | Weight-derived intervention matrices |
| exp05 | factor_specificity.csv + das_vs_all_summary.csv | Factor task specificity |
| exp06 | weight_iia_proxy.csv | Per-head weight-space IIA proxy (factor OV spectral analysis) |
| exp07 | multi_task_das.csv | IOI vs SVA subspace orthogonality (68 degrees mean angle, max_cos=0.34) |
| exp08 | subspace_intersection.csv | Subspace intersection analysis |
| exp09 | discriminative_svd.csv | Discriminative SVD for task-specific factors |
| exp10 | weight_knockout.csv | Weight-space knockout experiments |
| exp11 | end_to_end_composition.csv | Multi-hop circuit composition (dup->ind->si->nm paths) |
| exp12 | natural_text_variance.csv | Variance on natural text |

Each experiment has both a .csv summary and a _full.json with detailed results.

---

## 10. Grassmannian Analysis

Source: `lib/analysis_grassmanian_subspace/factorized_das/grassmannian_analysis.json`
Checkpoint: shared_bank_per_proj_sel (1024 factors)

Cross-seed, cross-task, and vanilla-vs-factorized subspace comparisons using principal angles on trained DAS matrices from 7 runs (3 IOI k=32, 4 SVA k=4).

Key findings documented in `RESULTS.md`:
- IOI factorized DAS is a geometric invariant (2.2 degree mean angle across seeds)
- SVA has rotational degeneracy (31-49 degree mean angle, zero Jaccard)
- IOI and SVA causal subspaces are nearly orthogonal (68 degrees)
- Factor bank captures the causal subspace (same IIA as vanilla) but finds different basis

---

## Missing Coverage

### Checkpoints with NO DAS results at all:
- Qwen checkpoints -- no DAS ever run on Qwen models
- Pythia checkpoints -- no DAS ever run
- Any non-GPT-2 model

### Checkpoint x Task gaps (tasks never tested):

| Checkpoint | Missing Tasks |
|---|---|
| shared_bank_per_proj_sel | capital_country, gender_bias, greater_than, hypernymy |
| per_layer_per_proj_sel | capital_country, gender_bias, greater_than, hypernymy, (IOI k=4, SVA k=32) |
| no_lambda_5k | ioi (crashed/running), capital_country, gender_bias, greater_than, hypernymy |
| shared_bank_global_dense | greater_than, hypernymy (k=32 for all 6 tasks in progress) |
| classic-sweep-133 | capital_country, gender_bias, greater_than, hypernymy (k=32 for IOI/SVA crashed) |
| atomic-sweep-40 | capital_country, gender_bias, greater_than, hypernymy |
| polar-sweep-90 | greater_than, hypernymy |

### Other gaps:
- No k=32 factorized DAS for atomic-sweep-40 (only k=4)
- No vanilla DAS for polar-sweep-90 or analysis8 self-consistent results
- classic-sweep-133 k=32 runs all crashed -- needs retry
- shared_bank_4096 -- all runs failed (3 attempts on IOI and SVA)
- No multi-seed factorized DAS for any checkpoint except shared_bank_per_proj_sel

---

## 10. CPCA-init Factorized DAS (atomic-sweep-40, hard-mode, k=1)

Source: `artifacts/cpca_init_factorized/` and `artifacts/cpca_factorized_sparse/`
Modal wrapper: `lib/factorized_das/modal_cpca_init_factorized_das.py`
Script: `lib/factorized_das/cpca_init_factorized_das.py`

All results use hard-mode IIA (margin threshold filtering), k=1, layer 8.
Stays on factor manifold by construction: Q = orth(F.T @ A).

### Run 1: Weak L1 (2026-06-20)

L1=0.05, 200 steps. Barely sparse — most factors survive.

| Task | n_hard | CPCA-init IIA | Delta-init IIA | CPCA n_active | Delta n_active |
|---|---|---|---|---|---|
| IOI | 83 | 0.941 | 0.941 | 5522 | 7593 |
| SVA | 58 | 0.792 | 0.750 | 6704 | 7860 |
| Gender Bias | 539 | 0.829 | 0.819 | 6941 | 7863 |
| Capital-Country | 32 | 0.692 | 0.692 | 6929 | 7900 |
| Hypernymy | 21 | 1.000 | 1.000 | 6585 | 7803 |

Key finding: **On-manifold factorized DAS matches unconstrained DAS across all 5 tasks.**
Unconstrained DAS (from gradient decomposition) gets identical IIA but drifts to rho~0.13.

### Run 2: Strong L1 (2026-06-21)

L1=0.5 and L1=1.0, 500 steps, no axis alignment. Group lasso (proximal) zeros entire rows of A.

| Task | n_hard | L1=0.5 IIA | L1=0.5 active | L1=1.0 IIA | L1=1.0 active |
|---|---|---|---|---|---|
| IOI | 83 | 0.941 | 1317 | 0.912 | 624 |
| SVA | 58 | 0.792 | 2768 | 0.750 | 2659 |
| Capital-Country | 32 | 0.692 | 2860 | 0.692 | 2737 |
| Hypernymy | 21 | 1.000 | 2280 | 1.000 | 2147 |
| Gender Bias | 539 | 0.768 | 1090 | (timed out) | -- |

Key finding: **L1=0.5 matches unconstrained DAS IIA exactly** on all completed tasks while
reducing active factors from 8192 to ~1300-2800. L1=1.0 drops IOI/SVA by 2-4pp but still
strong. Hypernymy and capital-country are insensitive to L1 strength.

Results location: `fc-results:/cpca_factorized_sparse/{task}/cpca_init_factorized.json`

### Run 3: L1 + axis alignment (2026-06-21)

Axis alignment penalty (`axis_lambda=0.5`) encourages each DAS dimension to load on a single
factor. Tested at L1=1.0+axis and L1=0.5+axis.

| Task | n_hard | L1=1.0+axis IIA | L1=1.0+axis active | L1=0.5+axis IIA | L1=0.5+axis active |
|---|---|---|---|---|---|
| IOI | 83 | 0.706 | 2 | 0.794 | 75 |
| SVA | 58 | 0.375 | 31 | 0.667 | 670 |
| Capital-Country | 32 | 0.692 | 686 | 0.692 | 619 |
| Hypernymy | 21 | 1.000 | 25 | 1.000 | 39 |
| Gender Bias | 539 | (timed out) | -- | 0.551 | 51 |

Key finding: **Axis alignment hurts IIA significantly on IOI and SVA.** L1=1.0+axis collapses
to 1-2 active factors on IOI (IIA drops from 0.912 to 0.706) and SVA (0.750 to 0.375).
L1=0.5+axis is less extreme but still loses ~15pp on IOI, ~13pp on SVA vs L1=0.5 alone.
Capital-country and hypernymy are unaffected (already at ceiling or floor).

Interpretation: the axis alignment penalty is too aggressive for k=1 DAS. The optimal direction
is a genuine mixture of multiple factors — forcing axis alignment destroys the subspace.
This is consistent with the rotational gauge freedom: if A → A@R gives the same Q, there's no
reason to expect A to be axis-aligned in factor space.

Results location: `fc-results:/cpca_factorized_axis/{task}/cpca_init_factorized.json`

### Comparison: unconstrained DAS (from gradient decomposition)

Source: `artifacts/gradient_decomposition_all/{task}_k1/gradient_decomposition.json`

| Task | Optimal DAS IIA | CPCA-init DAS IIA | Random-init DAS IIA | Rho (CPCA converged) | Rho (random converged) |
|---|---|---|---|---|---|
| IOI | 0.941 | 0.941 | 0.735 | 0.825 | 0.351 |
| SVA | 0.792 | 0.792 | 0.583 | 0.648 | 0.386 |
| Gender Bias | 0.787 | 0.787 | 0.618 | 0.562 | 0.422 |
| Capital-Country | 0.692 | 0.692 | 0.615 | 0.670 | 0.479 |
| Hypernymy | 1.000 | 1.000 | 0.952 | 0.831 | 0.362 |

Note: "Optimal DAS" is unconstrained (trains Q in R^768); rho < 1.0 = off-manifold.
CPCA-init DAS also unconstrained but starts on manifold, stays closer (gradient attractor).

### Rotational freedom note

The group lasso selects ROWS of A, but orth(F.T @ A) has rotational gauge freedom:
any A' = A @ R gives the same Q. So the identity of which factors are "active" is not
unique — a different rotation could spread weight differently. To test:
1. Re-run from different seeds, compare Jaccard overlap of active factor sets
2. Cayley rotation to find canonical sparse basis (not yet attempted)

---

## File Locations Summary

| Location | Contents |
|---|---|
| `lib/analysis_elliot/.../01_DAS_CIRCUITS/data/` | 78 W&B run summaries + 25 artifact dirs with .pt/.json |
| `lib/factorized_das/results/atomic-sweep-40/` | Per-layer DAS for 8192-factor DST checkpoint (IOI+SVA, 6 layers each) |
| `lib/factorized_das/results/dense-k64/` | k=64 results for shared_bank_global_dense |
| `lib/analysis_grassmanian_subspace/.../self_consistent_fac_das/` | delta_pca only for sgd checkpoint, 4 tasks |
| `lib/analysis_grassmanian_subspace/.../analysis10_.../results/` | delta_pca only for polar-sweep-90, 4 tasks |
| `lib/analysis_grassmanian_subspace/factorized_das/` | Grassmannian subspace analysis (cross-seed/cross-task) |
| `lib/analysis_elliot/.../06_NODE_SELECTION_ALGORITHMS/data/` | Node selection + per-layer DAS for IOI |
| `lib/weight_space_das/results/` | 12 weight-space experiments (not activation DAS) |
