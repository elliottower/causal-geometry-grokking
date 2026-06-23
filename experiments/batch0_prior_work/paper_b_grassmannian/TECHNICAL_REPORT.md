# Grassmannian Geometry of Task Representations in GPT-2

**Full Technical Report — All 40 Experiments, All Numbers, Full Traceability**

All result JSONs, scripts, and data sources are in this directory. Every number cited below links to a specific file. To verify any claim: load the linked JSON and check the cited key path.

---

## Data Sources

| Source | Path | Description |
|--------|------|-------------|
| Dense k=32 DAS | `lib/factorized_das/results/dense-k32-grassmann/{task}/` | 5 tasks, L1=0.1/0.5/1.0 |
| Dense k=64 DAS | `lib/factorized_das/results/dense-k64/` | IOI + SVA only |
| Atomic k=4 DAS | `lib/factorized_das/results/atomic-sweep-40/` | IOI + SVA at L0/L3/L8/L11 |
| Checkpoint | `artifacts/wandb_checkpoints/atomic-sweep-40/factorized_payload.pt` | 8192 factors, DST selector |
| GPT-2 weights | HuggingFace `gpt2` (for LN params not in factorized ckpt) | Via `transformers` library |

DAS result files contain: `A` (factor-space, 8192 x k or 1024 x k), `U` (d_model-space, 768 x k), `factor_importance` (per-factor), `l1_lambda`, `checkpoint_type`, `step`. We QR-decompose `U` to get orthonormal `Q` for all Grassmannian computations.

---

## 1. Universal Shared Subspace

### B14: Sum-of-projectors eigendecomposition
**Script**: [`exp_b14_multitask_universal.py`](exp_b14_multitask_universal.py) | **Results**: [`results_b14_multitask_universal.json`](results_b14_multitask_universal.json)

Sum of projectors $\sum P_t$ for 5 tasks (dense k=32). Maximum possible eigenvalue = 5.0 (direction in all 5 subspaces).

| Dimension | Eigenvalue | Fraction of 5.0 | Per-task projection |
|-----------|-----------|------------------|---------------------|
| 0 | 4.985 | 99.7% | IOI=0.999, SVA=0.999, GT=0.997, GB=0.998, CC=0.999 |
| 1 | 4.835 | 96.7% | (punctuation direction, see B29) |
| 2 | 4.327 | 86.5% | |
| 3 | 2.915 | 58.3% | Sharp drop — 3D universal subspace |
| 4 | 2.641 | 52.8% | |

Direction 0 top decoded tokens: `asons, ens, men, an, the, ms, s, ks, hens, land` — English morphological suffixes.

### B15: Statistical significance against random
**Script**: [`exp_b15_random_baseline.py`](exp_b15_random_baseline.py) | **Results**: [`results_b15_random_baseline.json`](results_b15_random_baseline.json)

1,000 random k=32 subspaces of R^768:
- Random theta_0 mean: **67.45 +/- 0.88 degrees**
- Observed theta_0 range: **2.88 -- 6.98 degrees** (all 10 pairs)
- All p-values: **0.0** (z > 68 for every pair)
- Random eigenvalue(sum-of-projectors) mean at dim 0: **1.897** vs observed **4.985** (z > 134)
- Random subspaces **never** produce eigenvalues above ~2.5

### B29: Artifact cleanup
**Script**: [`exp_b29_clean_universal.py`](exp_b29_clean_universal.py) | **Results**: [`results_b29_clean_universal.json`](results_b29_clean_universal.json)

Classified top-10 eigenvectors as garbage (punctuation/CJK) vs meaningful (3+ alphabetic tokens of length 3+):
- **2 of 10** are garbage (dims 1 and 5)
- Garbage accounts for **4-5%** of each task's energy
- After projecting out garbage: pairwise angles change by **< 1 degree**

### B27: Subspace intersection detail
**Script**: [`exp_b27_subspace_intersection.py`](exp_b27_subspace_intersection.py) | **Results**: [`results_b27_subspace_intersection.json`](results_b27_subspace_intersection.json)

All 10 task pairs have sigma_0 > 0.993 (the closest direction in each pair's subspace is nearly identical). At sigma > 0.9: 2-3 shared dims per pair. At sigma > 0.5: 3-10 shared dims.

---

## 2. Pairwise Geometry (Dense k=32)

### B15 pairwise angles (also B1 for layer sweep)
**Results**: [`results_b15_random_baseline.json`](results_b15_random_baseline.json) → `real_pairs`

| Pair | theta_0 | Geodesic dist |
|------|---------|---------------|
| IOI -- SVA | 2.88° | 406.8 |
| IOI -- greater_than | 4.56° | 392.7 |
| IOI -- gender_bias | 3.69° | 404.1 |
| IOI -- capital_country | 2.92° | 406.7 |
| SVA -- greater_than | 4.13° | 393.8 |
| SVA -- gender_bias | 4.50° | 399.2 |
| SVA -- capital_country | 3.40° | 423.5 |
| greater_than -- gender_bias | 6.98° | 409.5 |
| greater_than -- capital_country | 5.44° | 394.4 |
| gender_bias -- capital_country | 4.36° | 420.0 |

All theta_0 < 7 degrees (strong shared direction), but geodesic distances are large (392-424) — tasks share their closest direction but diverge rapidly in higher dimensions.

### B33: Bootstrap confidence intervals
**Script**: [`exp_b33_bootstrap_ci.py`](exp_b33_bootstrap_ci.py) | **Results**: [`results_b33_bootstrap_ci.json`](results_b33_bootstrap_ci.json)

At noise_scale=0.01, 500 bootstrap resamples. Example CIs from `perturbation.noise_0.01`:

| Pair | theta_0 | 95% CI |
|------|---------|--------|
| IOI -- SVA | 3.03° | [2.94, 3.12] |
| gender_bias -- greater_than | 7.32° | [7.19, 7.46] |
| capital_country -- IOI | 3.12° | [3.04, 3.20] |

All CI widths < 0.3 degrees. Random rotation stability: < 0.001 degree (Grassmannian invariant).

---

## 3. Layer-Dependent Geometry

### B1: Canonical angles across layers (atomic k=4)
**Script**: [`exp_b1_layer_sweep.py`](exp_b1_layer_sweep.py) | **Results**: [`results_b1_layer_sweep.json`](results_b1_layer_sweep.json) → `cross_task`

IOI-SVA theta_0 at L1=0.1:

| Layer | theta_0 | mean_angle | Interpretation |
|-------|---------|------------|----------------|
| L0 | 52.5° | 77.9° | Moderate sharing at embedding |
| L3 | 34.6° | 72.5° | Strong sharing at early processing |
| default | 80.7° | 86.1° | Near-orthogonal at computation |
| L8 | 82.9° | 86.2° | Also near-orthogonal |
| L11 | 32.2° | 68.3° | Strongest sharing at output |

Pattern: encoding (shared) → computation (orthogonal) → decoding (shared).

### B1: Shared direction decode
**Script**: [`exp_b1_shared_direction.py`](exp_b1_shared_direction.py) | **Results**: [`results_b1_shared_direction.json`](results_b1_shared_direction.json)

L11 shared direction (the direction with theta_0 = 32.2°) decoded via W_U promotes: `fits, keeps, lets, remains, allows, goes, is, hits, gives, stays` — **3rd person singular present tense verbs**.

L3 shared direction: mixed content (less interpretable). L3 and L11 shared directions are at **57 degrees** to each other.

### B1: Cross-layer decode
**Script**: [`exp_b1_decode_shared_layers.py`](exp_b1_decode_shared_layers.py) | **Results**: [`results_b1_decode_shared_layers.json`](results_b1_decode_shared_layers.json)

### B16: Geodesic trajectories
**Script**: [`exp_b16_geodesic_trajectory.py`](exp_b16_geodesic_trajectory.py) | **Results**: [`results_b16_geodesic_trajectory.json`](results_b16_geodesic_trajectory.json)

- IOI curvature ratio: **3.83** (path is 3.8x longer than direct geodesic)
- SVA curvature ratio: **2.30** (smoother trajectory)
- IOI has a massive jump at L3 (deviation = 2.46)
- L8 = L10 = default for IOI (distance 0.001)

### B23: Principal direction flow
**Script**: [`exp_b23_principal_flow.py`](exp_b23_principal_flow.py) | **Results**: [`results_b23_principal_flow.json`](results_b23_principal_flow.json)

IOI's principal direction v1 rotates **84-88 degrees** between adjacent layers:
- default/L10: promotes names (Matthew, Martin, Ben) — name-mover direction
- L0: generic content
- SVA is smoother: 20-48 degree rotations

### B13: Full geodesic matrix
**Script**: [`exp_b13_geodesic_distance_matrix.py`](exp_b13_geodesic_distance_matrix.py) | **Results**: [`results_b13_geodesic_matrix.json`](results_b13_geodesic_matrix.json)

SVA more stable across layers (mean within-SVA distance = 2.08) vs IOI (2.80).

---

## 4. Spectral Structure

### B18: Spectral profiles
**Script**: [`exp_b18_spectral_profile.py`](exp_b18_spectral_profile.py) | **Results**: [`results_b18_spectral_profile.json`](results_b18_spectral_profile.json)

All tasks have participation ratio ~1.4, dominated by a single direction:

| Task | PR | Top-1 energy | Spectral gap | Eff rank (90%) | Eff rank (99%) |
|------|-----|-------------|-------------|----------------|----------------|
| IOI | 1.4 | 84% | 3.6x | 2 | 19 |
| SVA | 1.3 | 89% | 4.9x | 2 | 21 |
| greater_than | 1.4 | 85% | 4.0x | 3 | 27 |
| gender_bias | 1.4 | 86% | 3.8x | 2 | 22 |
| capital_country | 1.5 | 84% | 3.6x | 4 | 25 |

### B24: k-sensitivity
**Script**: [`exp_b24_k_sensitivity.py`](exp_b24_k_sensitivity.py) | **Results**: [`results_b24_k_sensitivity.json`](results_b24_k_sensitivity.json)

**Critical finding**: IOI-SVA canonical angle depends dramatically on k:

| k | theta_0 (IOI-SVA) | PR | Top-1 energy |
|---|-------------------|-----|-------------|
| 4 | 80.7° | 3.1-3.7 | 32-47% |
| 32 | 2.9° | 1.4 | 84% |
| 64 | 3.3° | 1.5-1.7 | 77-81% |

At k=4, 4 dimensions cannot accommodate shared structure, forcing apparent orthogonality. At k >= 32 the true geometry emerges. k=32 is well-nested in k=64 (68-85% containment). k=4 only partially nested in k=32 (34-52%).

### B17: L1 regularization stability
**Script**: [`exp_b17_l1_stability.py`](exp_b17_l1_stability.py) | **Results**: [`results_b17_l1_stability.json`](results_b17_l1_stability.json)

- Dense k=32: theta_0 < 1 degree between L1=0.1 and L1=0.5 for most tasks
- Atomic k=4: more fragile — up to 53 degree shift at L1=1.0 for SVA
- greater_than (dense k=32): essentially identical across L1

---

## 5. Task Manifold and Capacity

### B20: Frechet mean and collective dimension
**Script**: [`exp_b20_frechet_mean.py`](exp_b20_frechet_mean.py) | **Results**: [`results_b20_frechet_mean.json`](results_b20_frechet_mean.json)

5 tasks combined: **64 out of 768 dimensions** (8.3% of d_model) at eigenvalue threshold > 1.0. Ward clustering: {gender_bias, SVA} vs {capital_country, greater_than, IOI}.

### B39: Sublinear dimension scaling
**Script**: [`exp_b39_effective_dim_scaling.py`](exp_b39_effective_dim_scaling.py) | **Results**: [`results_b39_effective_dim_scaling.json`](results_b39_effective_dim_scaling.json)

From `scaling` array and `power_law`:

| n tasks | Mean dim | Linear pred | Redundancy | Mean PR |
|---------|----------|-------------|------------|---------|
| 1 | 32.0 | 32 | 0.0% | 32.0 |
| 2 | 57.4 | 64 | 10.3% | 54.4 |
| 3 | 76.8 | 96 | 20.0% | 70.8 |
| 4 | 93.2 | 128 | 27.2% | 83.5 |
| 5 | 109.0 | 160 | 31.9% | 93.5 |

**Power law**: `dim = 32.78 * n^0.759` (`power_law.alpha`, `power_law.beta`)

Extrapolations: 10 tasks → 188 dims (24.5%), 20 → 319 (41.5%), 50 → 639 (83.2%).

Greedy ordering (`greedy_order`): SVA → capital_country → gender_bias → IOI → greater_than. greater_than most redundant.

### B22: Von Neumann entropy
**Script**: [`exp_b22_von_neumann_entropy.py`](exp_b22_von_neumann_entropy.py) | **Results**: [`results_b22_von_neumann_entropy.json`](results_b22_von_neumann_entropy.json)

Quantum fidelity between pairs: ~0.005 (vs 0.001 random, so 4-5x above chance). Effective dimension of ensemble average: 116.5. Holevo chi: 1.29.

### B21: MDS embedding
**Script**: [`exp_b21_mds_embedding.py`](exp_b21_mds_embedding.py) | **Results**: [`results_b21_mds_embedding.json`](results_b21_mds_embedding.json)

Primary organizing axis: **checkpoint** (dense vs atomic), not task or layer. 2D embedding captures only 36% of variance (stress = 0.51).

---

## 6. Interference and Transfer

### B5: Geodesic distance predicts interference
**Script**: [`exp_b5_cross_task_interference.py`](exp_b5_cross_task_interference.py) | **Results**: [`results_b5_cross_task_interference.json`](results_b5_cross_task_interference.json)

From `pairs` array, cross_proj_ij and geodesic_distance:

| Pair | Geodesic | Cross-projection | Random baseline |
|------|----------|-----------------|-----------------|
| cap_ctry -- gender | 7.33 | 0.143 | 0.042 |
| cap_ctry -- greater | 6.88 | 0.201 | 0.042 |
| cap_ctry -- IOI | 7.10 | 0.178 | 0.042 |
| cap_ctry -- SVA | 7.39 | 0.139 | 0.042 |
| gender -- greater | 7.15 | 0.165 | 0.042 |
| gender -- IOI | 7.05 | 0.177 | 0.042 |
| gender -- SVA | 6.97 | 0.192 | 0.042 |
| greater -- IOI | 6.85 | 0.202 | 0.042 |
| greater -- SVA | 6.87 | 0.205 | 0.042 |
| IOI -- SVA | 7.10 | 0.175 | 0.042 |

Geodesic distance vs cross-projection energy: **Pearson r = -0.994** (p < 0.0001). But theta_0 alone does NOT predict interference: r = 0.12, p = 0.74. The full geodesic distance (integrating all 32 angles) is required.

### B38: Cross-task transfer
**Script**: [`exp_b38_cross_task_transfer.py`](exp_b38_cross_task_transfer.py) | **Results**: [`results_b38_cross_task_transfer.json`](results_b38_cross_task_transfer.json)

Cross-projection is perfectly symmetric (= Grassmannian kernel). 80-84% unique energy per task. Procrustes alignment: 27-37%. greater_than--IOI highest (0.367). Transitive chains through greater_than can improve transfer.

### B35: Kernel spectral analysis
**Script**: [`exp_b35_kernel_spectral.py`](exp_b35_kernel_spectral.py) | **Results**: [`results_b35_kernel_spectral.json`](results_b35_kernel_spectral.json)

Dense kernel (5x5): condition number 2.2. lambda_0 = 1.71 (34.3%). Kernel alignment with linguistic categories: **0.81**. Atomic kernel (10x10): task alignment (0.669) ≈ layer alignment (0.662). sva_default = sva_L8 (kernel = 1.000).

---

## 7. Factor Bank Amplification

### B11: Factor overlap
**Script**: [`exp_b11_factor_overlap.py`](exp_b11_factor_overlap.py) | **Results**: [`results_b11_factor_overlap.json`](results_b11_factor_overlap.json)

At computation layer (L8/default): **zero Jaccard overlap** in top-50 factors.

Factor-space angles vs d_model-space angles:

| Layer | Factor theta_0 | d_model theta_0 | Amplification |
|-------|---------------|-----------------|---------------|
| L0 | 71.1° | 52.5° | +18.5° |
| L3 | 55.1° | 34.6° | +20.5° |
| L8 | 88.7° | 82.9° | +5.8° |
| default | 88.1° | 80.7° | +7.4° |
| L11 | 66.1° | 32.2° | +33.9° |

### B32: Dual-space geometry
**Script**: [`exp_b32_dual_space_geometry.py`](exp_b32_dual_space_geometry.py) | **Results**: [`results_b32_dual_space_geometry.json`](results_b32_dual_space_geometry.json)

From `correlation`: Pearson r = **0.907**, Spearman rho = **0.960**. F^T @ A ≈ U to high precision. Projection fidelity theta_0: 3.8-11.6 degrees. Containment: 94-98%.

### B7: Conceptor algebra
**Script**: [`exp_b7_multitask_conceptor.py`](exp_b7_multitask_conceptor.py) | **Results**: [`results_b7_multitask_conceptor.json`](results_b7_multitask_conceptor.json)

Aperture alpha=10. Pairwise OR perfectly additive: rank(C_i OR C_j) = 64 = 32 + 32. Universal AND: rank 1. NOT(all others) preserves 55-65%. 20-22 of 32 dimensions task-specific.

### B12: Conceptor boolean operations
**Script**: [`exp_b12_conceptor_boolean.py`](exp_b12_conceptor_boolean.py) | **Results**: [`results_b12_conceptor_boolean.json`](results_b12_conceptor_boolean.json)

At computation layer: **99.6%** of IOI preserved after NOT(SVA). At L3/L11: ~87-89%.

---

## 8. Circuit Prediction

### B8: Head alignment
**Scripts**: [`exp_b8_head_alignment.py`](exp_b8_head_alignment.py), [`exp_b8_multilayer.py`](exp_b8_multilayer.py) | **Results**: [`results_b8_head_alignment_ioi.json`](results_b8_head_alignment_ioi.json), [`results_b8_head_alignment_sva.json`](results_b8_head_alignment_sva.json), [`results_b8_multilayer.json`](results_b8_multilayer.json)

- IOI: AUROC = 0.624 at L10/default (p = 0.054)
- SVA: AUROC = 0.761 at L0 (p = 0.065)
- Layer-mismatched DAS reverses predictions (AUROC < 0.5 for IOI at L0/L3)

### B10: QK/OV geometry
**Script**: [`exp_b10_projection_geometry.py`](exp_b10_projection_geometry.py) | **Results**: [`results_b10_projection_geometry.json`](results_b10_projection_geometry.json)

Q-K selectors more similar than Q-V: p = 1.8e-15. OV-combined best circuit prediction AUROC: 0.642 (p = 0.033).

### B36: Weight-space overlap (PARTIAL NULL)
**Script**: [`exp_b36_projection_geometry.py`](exp_b36_projection_geometry.py) | **Results**: [`results_b36_projection_geometry.json`](results_b36_projection_geometry.json)

IOI circuit membership AUROC from individual head overlaps: **0.537** (near chance). Most overlaps near random baseline (0.042). But layer-wise average is interpretable: IOI overlap grows monotonically L0→L11. Most task-specific: L0H1 (greater_than, 0.142), L10H9 (gender_bias, 0.146).

### B37: Layer persistence
**Script**: [`exp_b37_layer_persistence.py`](exp_b37_layer_persistence.py) | **Results**: [`results_b37_layer_persistence.json`](results_b37_layer_persistence.json)

From `profiles` array, L11 attention write-out for each task (divided by random baseline 0.042):

| Task | L11 attn write (x random) |
|------|--------------------------|
| IOI | 5.59x |
| SVA | 5.02x |
| greater_than | 4.90x |
| gender_bias | 5.19x |
| capital_country | 4.81x |

Read-in at random baseline (~1.0x) for all tasks at all layers. Universal production arc: L1 attn spike → L3 MLP → gradual build → L10-L11 peak.

---

## 9. Task Discrimination

### B30: One-vs-all discriminant
**Script**: [`exp_b30_task_discriminant.py`](exp_b30_task_discriminant.py) | **Results**: [`results_b30_task_discriminant.json`](results_b30_task_discriminant.json)

Each discriminative direction has self-projection 0.997 and ~0.01 onto others.

| Task | Eigenvalue | Top decoded tokens |
|------|-----------|-------------------|
| IOI | 0.98 | baugh, merce, Beet, Adidas, Gingrich |
| SVA | 0.98 | include, VIDEOS, varied, ACTIONS, involve |
| greater_than | 0.98 | viron, lov, then, def, trade |
| gender_bias | 0.98 | academia, DOE, Hills, exploits, eminent |
| capital_country | 0.98 | Coco, Darling, Kardashian, rek |

Centroid uniqueness ranking: IOI (0.645, most unique) → SVA (0.715) → capital_country (0.776) → gender_bias (0.787) → greater_than (0.858, least unique).

### B26: Grassmannian classification
**Script**: [`exp_b26_grassmannian_regression.py`](exp_b26_grassmannian_regression.py) | **Results**: [`results_b26_grassmannian_regression.json`](results_b26_grassmannian_regression.json)

Leave-one-out nearest-neighbor: **100% accuracy**. But linguistic categories don't predict kernel similarity (ANOVA p = 0.92).

---

## 10. Manifold Geometry

### B31: Sectional curvature
**Script**: [`exp_b31_grassmannian_curvature.py`](exp_b31_grassmannian_curvature.py) | **Results**: [`results_b31_grassmannian_curvature.json`](results_b31_grassmannian_curvature.json)

From `curvature` array: all curvatures cluster at **K ≈ 0.064** (range [0.063, 0.066]). Near-constant across all task points and tangent planes. Triangle excess ratios: ~1.9-2.1x.

Per-task mean curvature:
- capital_country: 0.0644
- gender_bias: 0.0643
- greater_than: 0.0650
- IOI: 0.0645
- SVA: 0.0649

### B34: Geodesic interpolation
**Script**: [`exp_b34_grassmannian_interpolation.py`](exp_b34_grassmannian_interpolation.py) | **Results**: [`results_b34_grassmannian_interpolation.json`](results_b34_grassmannian_interpolation.json)

Midpoint decodings are **not interpretable** — no natural average task exists. All third-task distances are closer to the Frechet mean than to pairwise midpoints. Tasks arranged symmetrically around mean.

### B28: Principal geodesic analysis
**Script**: [`exp_b28_grassmannian_pca.py`](exp_b28_grassmannian_pca.py) | **Results**: [`results_b28_grassmannian_pca.json`](results_b28_grassmannian_pca.json)

PC0 captures 92% of tangent-space variance. But PGA does **not** preserve geodesic distances: r = -0.45. Tangent-space linearization breaks down at these distances.

### B19: Cross-checkpoint geometry
**Script**: [`exp_b19_cross_checkpoint.py`](exp_b19_cross_checkpoint.py) | **Results**: [`results_b19_cross_checkpoint.json`](results_b19_cross_checkpoint.json)

Atomic k=4 → dense k=32 overlap:
- IOI: containment = 34% (vs 0.5% random), theta_0 = 65°
- SVA: containment = 52%, theta_0 = 42.4°
- L11 verb direction exists in both checkpoints (theta_0 = 31.4°, containment = 47%)

---

## 11. Weight-Space Geometry (New Experiments)

### B40: Unembedding geometry
**Script**: [`exp_b40_unembedding_geometry.py`](exp_b40_unembedding_geometry.py) | **Results**: [`results_b40_unembedding_geometry.json`](results_b40_unembedding_geometry.json)

Each task's DAS subspace projected through W_U:

| Task | Eff rank (90%) | Entropy (bits) | Top-100 concentration | Top-1000 concentration |
|------|---------------|----------------|----------------------|----------------------|
| IOI | 25 | 15.53 | 0.005 | 0.042 |
| SVA | 23 | 15.53 | 0.007 | 0.046 |
| greater_than | 25 | 15.53 | 0.005 | 0.044 |
| gender_bias | 24 | 15.53 | 0.005 | 0.041 |
| capital_country | 25 | 15.53 | 0.006 | 0.044 |

Max entropy = 15.62 bits. All tasks near-maximal entropy in vocab space — **DAS subspaces project diffusely across the vocabulary**, not concentrated on specific tokens.

Gender_bias top energy tokens: `hers, she, she, him, his, girlfriend, he, actresses` — clearly gendered vocabulary. Capital_country: `Jinn, Rohing, Jiang, Uzbek, Turk, Sabha` — geographic/ethnic tokens.

Cross-task overlap in vocab space: cosine similarity of energy profiles **0.89-0.94** (high). Jaccard of top-100 tokens: only **0.005-0.031** (very low). Tasks project through similar *regions* of vocab space but target different specific tokens.

### B41: Embedding alignment
**Script**: [`exp_b41_embedding_alignment.py`](exp_b41_embedding_alignment.py) | **Results**: [`results_b41_embedding_alignment.json`](results_b41_embedding_alignment.json)

From `token_embedding`:

| Task | Mean containment | x random | >2x random | >5x random | Max |
|------|-----------------|----------|-----------|-----------|-----|
| IOI | 0.103 | 2.46x | 80.8% | 0.2% | 0.450 |
| SVA | 0.118 | 2.84x | 94.1% | 0.3% | 0.484 |
| greater_than | 0.101 | 2.43x | 80.8% | 0.3% | 0.411 |
| gender_bias | 0.093 | 2.24x | 65.3% | 0.2% | 0.522 |
| capital_country | 0.095 | 2.28x | 69.5% | 0.2% | 0.388 |

Random baseline: 0.042 (= k/d_model = 32/768). DAS subspaces contain **2-3x random** amount of token embedding energy — moderate overlap. Top tokens: function words (" the", " a", " and") have highest containment (~0.45-0.52).

Position encoding (from `position_encoding`): mean containment 0.038-0.057 (**0.9-1.4x random**). Position encoding has essentially no overlap with DAS subspaces — these tasks don't use positional information in the DAS-captured dimensions.

Cross-task containment correlations: Pearson r = 0.60-0.76. Tasks capture overlapping token embedding structure.

### B42: MLP composition circuits
**Script**: [`exp_b42_composition_circuits.py`](exp_b42_composition_circuits.py) | **Results**: [`results_b42_composition_circuits.json`](results_b42_composition_circuits.json)

From `mlp_write_overlap` (fraction of W_out column space in DAS subspace):

- **All tasks peak at L3** (0.119-0.125, or 2.86-2.99x random)
- Write overlap then declines through middle layers and rises again at L11
- Read overlap (W_in) is near random baseline for all layers (1.0-1.25x)

Top MLP-MLP compositions through DAS (from `mlp_composition_overlap`):
- **L3→L5, L3→L6 are the top compositions for ALL tasks** (~0.07-0.08, or 40-46x random^2)
- Universal pattern: L3 MLP writes DAS-aligned information, mid-layer MLPs read it

Composition vs layer gap (from `cross_layer_profile`): gap-insensitive (gap1 ≈ gap2 ≈ gap3), consistent with residual stream carrying DAS info forward.

### B43: LayerNorm interaction
**Script**: [`exp_b43_layernorm_interaction.py`](exp_b43_layernorm_interaction.py) | **Results**: [`results_b43_layernorm_interaction.json`](results_b43_layernorm_interaction.json)

From `per_layer` and `summary`:

Most LN layers have gain ratio ~1.0 (DAS directions get the same LN scaling as random directions). Two systematic deviations:

1. **ln_final amplifies DAS 2.3-2.9x**: IOI 2.86x, SVA 2.90x, greater_than 2.82x, gender_bias 2.30x, capital_country 2.45x. The final LayerNorm preferentially amplifies DAS directions before unembedding.

2. **L11 ln2 suppresses DAS ~0.87-0.89x**: Consistent across all tasks. L11 pre-MLP LayerNorm slightly dampens DAS directions.

3. **L5-L7 ln2 amplifies DAS ~1.07-1.12x**: Mid-layer pre-MLP LayerNorm slightly boosts DAS directions, especially for IOI and SVA.

Max deviation from unity (from `summary`): IOI 1.86, SVA 1.90 — driven entirely by ln_final.

### B44: Attention QK bias and MLP principal directions
**Script**: [`exp_b44_attention_pattern_geometry.py`](exp_b44_attention_pattern_geometry.py) | **Results**: [`results_b44_attention_pattern_geometry.json`](results_b44_attention_pattern_geometry.json)

From `qk_bias_overlap`: **L3 heads dominate QK-bias alignment** for all tasks:

| Task | Top head | Q overlap | K overlap | Mean |
|------|----------|-----------|-----------|------|
| IOI | L3H7 | 0.412 | 0.642 | 0.527 |
| SVA | L3H8 | 0.462 | 0.614 | 0.538 |
| gender_bias | L3H8 | 0.451 | 0.628 | 0.539 |
| greater_than | L4H11 | 0.333 | 0.479 | 0.406 |
| capital_country | L3H8 | 0.338 | 0.458 | 0.398 |

K-side bias consistently higher than Q-side (K mean ~0.37 vs Q mean ~0.31). This means the QK bias circuit in L3 is oriented to *attend to* tokens in DAS-relevant positions.

From `mlp_svd_overlap`: MLP output principal directions (from `mlp_svd_overlap`):
- Out overlap peaks at L10-L11 for all tasks (0.14-0.25, or 3-6x random)
- In overlap near baseline (~0.05)
- SVA has highest out overlap: L10 at 0.252 (6x random)

From `summary`:

| Task | Mean QK bias overlap | Mean MLP in | Mean MLP out |
|------|---------------------|-------------|-------------|
| capital_country | 0.292 | 0.052 | 0.106 |
| gender_bias | 0.345 | 0.061 | 0.129 |
| greater_than | 0.275 | 0.044 | 0.134 |
| IOI | 0.341 | 0.051 | 0.120 |
| SVA | 0.341 | 0.062 | 0.153 |

---

## 12. Null Results

| Experiment | Claim tested | Result | Source |
|-----------|-------------|--------|--------|
| B9 | Selector sparsity predicts DAS alignment | All p > 0.18 | [`results_b9_sparsity_geometry.json`](results_b9_sparsity_geometry.json) |
| B36 | Individual head weights predict circuit membership | AUROC = 0.537 | [`results_b36_projection_geometry.json`](results_b36_projection_geometry.json) |
| B28 | PGA preserves geodesic distances | r = -0.45 | [`results_b28_grassmannian_pca.json`](results_b28_grassmannian_pca.json) |
| B34 | Geodesic midpoints are interpretable | All decode to garbage | [`results_b34_grassmannian_interpolation.json`](results_b34_grassmannian_interpolation.json) |
| B26 | Linguistic categories predict kernel | ANOVA p = 0.92 | [`results_b26_grassmannian_regression.json`](results_b26_grassmannian_regression.json) |
| B38 | Cross-projection is asymmetric | Perfectly symmetric (= kernel) | [`results_b38_cross_task_transfer.json`](results_b38_cross_task_transfer.json) |
| B41 pos | Position encoding overlaps DAS | 0.9-1.4x random | [`results_b41_embedding_alignment.json`](results_b41_embedding_alignment.json) |

---

## 13. Full Experiment Inventory

| ID | Script | Results | One-line finding |
|----|--------|---------|-----------------|
| B1 sweep | `exp_b1_layer_sweep.py` | `results_b1_layer_sweep.json` | L3=34.6°, L8=82.9°, L11=32.2° |
| B1 shared | `exp_b1_shared_direction.py` | `results_b1_shared_direction.json` | L11 shared = 3rd person verbs |
| B1 decode | `exp_b1_decode_shared_layers.py` | `results_b1_decode_shared_layers.json` | Cross-layer decode |
| B1/B8 atomic | `exp_b1_b8_atomic_sweep.py` | `results_b1_b8_atomic_sweep.json` | Atomic sweep angles |
| B5 | `exp_b5_cross_task_interference.py` | `results_b5_cross_task_interference.json` | r=-0.994 geodesic vs cross-proj |
| B7 | `exp_b7_multitask_conceptor.py` | `results_b7_multitask_conceptor.json` | OR additive; 20-22 task-specific |
| B8 IOI | `exp_b8_head_alignment.py` | `results_b8_head_alignment_ioi.json` | AUROC 0.624 |
| B8 SVA | `exp_b8_head_alignment.py` | `results_b8_head_alignment_sva.json` | AUROC 0.761 |
| B8 multi | `exp_b8_multilayer.py` | `results_b8_multilayer.json` | Layer-matched required |
| B9 | `exp_b9_sparsity_geometry.py` | `results_b9_sparsity_geometry.json` | NULL: p > 0.18 |
| B10 | `exp_b10_projection_geometry.py` | `results_b10_projection_geometry.json` | QK vs QV p=1.8e-15 |
| B11 | `exp_b11_factor_overlap.py` | `results_b11_factor_overlap.json` | +6 to +34° amplification |
| B12 | `exp_b12_conceptor_boolean.py` | `results_b12_conceptor_boolean.json` | 99.6% at computation layer |
| B13 | `exp_b13_geodesic_distance_matrix.py` | `results_b13_geodesic_matrix.json` | SVA more stable (2.08 vs 2.80) |
| B14 | `exp_b14_multitask_universal.py` | `results_b14_multitask_universal.json` | 3D shared, eigenvalue 4.985/5 |
| B15 | `exp_b15_random_baseline.py` | `results_b15_random_baseline.json` | z>68 for theta_0, z>134 eigenvalues |
| B16 | `exp_b16_geodesic_trajectory.py` | `results_b16_geodesic_trajectory.json` | IOI 3.83x curvature ratio |
| B17 | `exp_b17_l1_stability.py` | `results_b17_l1_stability.json` | k=32 stable (<1°); k=4 fragile |
| B18 | `exp_b18_spectral_profile.py` | `results_b18_spectral_profile.json` | PR~1.4; top-1 = 84%+ |
| B19 | `exp_b19_cross_checkpoint.py` | `results_b19_cross_checkpoint.json` | 34-52% containment |
| B20 | `exp_b20_frechet_mean.py` | `results_b20_frechet_mean.json` | 64/768 = 8.3% |
| B21 | `exp_b21_mds_embedding.py` | `results_b21_mds_embedding.json` | Checkpoint > task > layer |
| B22 | `exp_b22_von_neumann_entropy.py` | `results_b22_von_neumann_entropy.json` | Holevo chi = 1.29 |
| B23 | `exp_b23_principal_flow.py` | `results_b23_principal_flow.json` | IOI v1 rotates 84-88° |
| B24 | `exp_b24_k_sensitivity.py` | `results_b24_k_sensitivity.json` | 80.7° at k=4 → 2.9° at k=32 |
| B26 | `exp_b26_grassmannian_regression.py` | `results_b26_grassmannian_regression.json` | 100% LOO; categories p=0.92 |
| B27 | `exp_b27_subspace_intersection.py` | `results_b27_subspace_intersection.json` | All pairs sigma_0 > 0.993 |
| B28 | `exp_b28_grassmannian_pca.py` | `results_b28_grassmannian_pca.json` | PGA r=-0.45 (breaks down) |
| B29 | `exp_b29_clean_universal.py` | `results_b29_clean_universal.json` | 2/10 garbage; 4-5% energy |
| B30 | `exp_b30_task_discriminant.py` | `results_b30_task_discriminant.json` | IOI most unique (0.645) |
| B31 | `exp_b31_grassmannian_curvature.py` | `results_b31_grassmannian_curvature.json` | K ≈ 0.064 constant |
| B32 | `exp_b32_dual_space_geometry.py` | `results_b32_dual_space_geometry.json` | r=0.91; +34° at L11 |
| B33 | `exp_b33_bootstrap_ci.py` | `results_b33_bootstrap_ci.json` | CI width < 0.3° |
| B34 | `exp_b34_grassmannian_interpolation.py` | `results_b34_grassmannian_interpolation.json` | Midpoints uninterpretable |
| B35 | `exp_b35_kernel_spectral.py` | `results_b35_kernel_spectral.json` | Cond=2.2; linguistic alignment=0.81 |
| B36 | `exp_b36_projection_geometry.py` | `results_b36_projection_geometry.json` | NULL: AUROC=0.54 |
| B37 | `exp_b37_layer_persistence.py` | `results_b37_layer_persistence.json` | L11 writes 5x for all tasks |
| B38 | `exp_b38_cross_task_transfer.py` | `results_b38_cross_task_transfer.json` | 80-84% unique; symmetric |
| B39 | `exp_b39_effective_dim_scaling.py` | `results_b39_effective_dim_scaling.json` | dim = 32.8 * n^0.759 |
| B40 | `exp_b40_unembedding_geometry.py` | `results_b40_unembedding_geometry.json` | Diffuse vocab; gender tokens clear |
| B41 | `exp_b41_embedding_alignment.py` | `results_b41_embedding_alignment.json` | 2-3x random; position=null |
| B42 | `exp_b42_composition_circuits.py` | `results_b42_composition_circuits.json` | L3 MLP write peak; L3→L5 top |
| B43 | `exp_b43_layernorm_interaction.py` | `results_b43_layernorm_interaction.json` | ln_final amplifies DAS 2.3-2.9x |
| B44 | `exp_b44_attention_pattern_geometry.py` | `results_b44_attention_pattern_geometry.json` | L3 heads dominate QK bias alignment |
| B45 | `exp_b45_attn_composition_atomic.py` | `results_b45_attn_composition_atomic.json` | L11H8 OV 5.3x; L0H0→L1H2 top comp |
| B46 | `exp_b46_l3_cross_composition.py` | `results_b46_l3_cross_composition.json` | MLP and QK bias use different subspaces (74°) |

---

## 14. Atomic-Sweep-40 Circuit Geometry (B45-B46)

### B45: Attention OV/QK circuit overlap (reconstructed from selector + factor bank)
**Script**: [`exp_b45_attn_composition_atomic.py`](exp_b45_attn_composition_atomic.py) | **Results**: [`results_b45_attn_composition_atomic.json`](results_b45_attn_composition_atomic.json)

Reconstructed all 12x12x4 attention weight matrices from the DST selector (95.8% sparse) + factor bank. Random baseline for k=4: 0.0052.

**OV circuit overlap** (from `ov_overlap`):

| Task | Top head | Overlap | x random | Mean |
|------|----------|---------|----------|------|
| IOI | L11H8 | 0.0275 | 5.28x | 0.0058 (1.12x) |
| SVA | L0H9 | 0.0189 | 3.62x | 0.0063 (1.20x) |

IOI concentrates OV overlap in L11 (the output layer). SVA concentrates in L0 and L9. Mean OV overlap is only slightly above random — the OV circuit structure is concentrated in a few heads.

**QK circuit overlap** (from `qk_overlap`):

| Task | Top head | Overlap | x random | Mean |
|------|----------|---------|----------|------|
| IOI | L11H2 | 0.0170 | 3.26x | 0.0071 (1.37x) |
| SVA | L2H7 | 0.0137 | 2.63x | 0.0072 (1.38x) |

IOI QK overlap peaks at L11 (all top-5 heads are L11). SVA QK is more distributed across layers.

**Inter-layer OV-QK composition** (from `ov_composition`): the most DAS-aligned composition paths:

| Composition | Overlap | x random |
|------------|---------|----------|
| **IOI**: L0H0→L1H2 | 0.0212 | 4.1x |
| IOI: L0H10→L1H11 | 0.0183 | 3.5x |
| IOI: L0H10→L3H6 | 0.0153 | 2.9x |
| **SVA**: L9H7→L10H9 | 0.0342 | 6.6x |
| SVA: L0H9→L1H5 | 0.0301 | 5.8x |
| SVA: L0H9→L3H4 | 0.0232 | 4.4x |

SVA has a strong late-layer composition path (L9H7→L10H9 at 6.6x random). IOI compositions concentrate in the first few layers (L0→L1, L0→L3).

Layer-pair means: IOI peaks at L0→L1 and L0→L3. SVA peaks at L5→L7 and L4→L5.

### B46: L3 cross-composition — MLP vs attention use different subspaces
**Script**: [`exp_b46_l3_cross_composition.py`](exp_b46_l3_cross_composition.py) | **Results**: [`results_b46_l3_cross_composition.json`](results_b46_l3_cross_composition.json)

From `mlp_qk_alignment`: MLP output top-12 principal directions vs QK bias subspace: **theta_0 = 74.3°** — they operate in largely **different** subspaces. The L3 MLP write and L3 attention QK bias are NOT sharing the same DAS information.

From `joint_das_overlap`:

| DAS subspace | MLP W_out overlap (x random) | QK bias mean | Joint theta_0 |
|-------------|------------------------------|-------------|---------------|
| IOI (default) | 0.0068 (1.31x) | 0.126 | 82.6° |
| SVA (default) | 0.0448 (8.61x) | 0.136 | 72.6° |
| IOI (L3-specific) | 0.0690 (13.24x) | 0.058 | 68.5° |
| SVA (L3-specific) | 0.0756 (14.52x) | 0.146 | 67.8° |

When using L3-specific DAS subspaces, MLP overlap jumps to **13-15x random** — L3 MLP is indeed writing L3-DAS-aligned information. The QK bias overlap is highest for SVA L3-specific (0.146-0.179).

From `per_head_analysis`: L3 head selector sparsity is 95.8-96.0% across all heads. The W_Q and W_K effective weight overlaps with DAS are near random (0.005-0.007), much lower than the QK *bias* overlaps (0.12-0.18). This means the DAS alignment is primarily through the **bias reparameterization**, not the factorized weight matrices themselves.

---

## 15. Key Claims (with evidence pointers)

1. **3D universal subspace** at z>68 above random encoding morphological suffixes → B14, B15, B29
2. **Layer-dependent geometry**: shared at encoding/decoding, orthogonal at computation → B1, B16, B23
3. **Geodesic distance predicts interference** with r=-0.994 → B5
4. **Factor bank amplifies separation** up to +34° → B11, B32
5. **Tasks are spectrally dominated**: PR~1.4, top-1 captures 84%+ → B18
6. **Sublinear capacity scaling**: dim = 32.8 * n^0.759, ~50 tasks fill d_model → B39
7. **L11 attention universally produces DAS information** at 5x random → B37
8. **Task discriminants encode expected linguistics** → B30
9. **Constant curvature K ≈ 0.064** in the task constellation → B31
10. **k-sensitivity**: k=4 hides geometry that k=32 reveals → B24
11. **MLP L3 is the universal DAS write layer** at 2.9x random, with top compositions L3→L5/L6 → B42
12. **ln_final preferentially amplifies DAS directions 2.3-2.9x** → B43
13. **L3 attention QK biases are maximally DAS-aligned** across all tasks → B44
14. **OV circuit overlap concentrates in a few heads**: L11H8 for IOI (5.3x), L0H9 for SVA (3.6x); means near random → B45
15. **L3 MLP and QK bias operate in different subspaces** (74° apart); DAS alignment at L3 comes through bias reparameterization, not factorized weights → B46
16. **SVA has a strong late-layer composition** L9H7→L10H9 (6.6x random); IOI compositions are early-layer L0→L1/L3 → B45

