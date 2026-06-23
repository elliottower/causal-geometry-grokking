# Master DAS Results Catalog

All Distributed Alignment Search results on disk, exhaustively cataloged.
Last updated: 2026-06-20.

All results are on **GPT-2 small** (12 layers, 12 heads, d_model=768).
The primary checkpoint throughout is **atomic-sweep-40** (8192 factors, DST sparsity) unless otherwise noted.

---

## Quick Reference: Best IIA by Task and Method

### Atomic-sweep-40 checkpoint (8192 factors, DST)

| Task | Method | k | Layer | IIA (eval) | Dataset | File |
|------|--------|---|-------|------------|---------|------|
| IOI | vanilla DAS | 32 | 10 | 0.993 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/ioi.json` |
| IOI | vanilla DAS | 4 | 8 | 0.993 | full (150) | `lib/factorized_das/results/atomic-sweep-40/ioi_L8/` |
| IOI | constrained PCA (frac=0.01) | 4 | 8 | 0.988 | full (80) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_ioi.json` |
| IOI | factorized DAS L1=1.0 | 4 | 8 | 0.987 | full (150) | `lib/factorized_das/results/atomic-sweep-40/ioi_L8/` |
| IOI | CPCA-init DAS | 4 | 8 | 0.971 | hard (34) | `artifacts/cpca_hard_ioi/cpca_hard_ioi/cpca_init_multitask_hard.json` |
| IOI | CPCA-init DAS (100 steps) | 1 | 8 | 0.912 | hard (34) | `artifacts/constrained_pca_fixes/fixes.json` |
| IOI | factorized DAS L1=0.1 | 1 | 8 | 0.882 | hard (34) | `artifacts/factorized_das_hard_v4/ioi.json` |
| IOI | delta PCA | 4 | 10 | 0.888 | full (80) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_ioi.json` |
| IOI | Riem. fac DAS L=0.05 | 32 | 10 | 0.993 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/ioi.json` |
| SVA | vanilla DAS | 32 | 8 | 0.973 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/sva.json` |
| SVA | Riem. fac DAS L=0.5 (best) | 32 | 8 | 0.967 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/sva.json` |
| SVA | vanilla DAS | 4 | 8 | 0.967 | full (150) | `lib/factorized_das/results/atomic-sweep-40/sva_L8/` |
| SVA | constrained PCA (frac=0.01) | 4 | 5-10 | 1.000 | full (80) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_sva.json` |
| SVA | CPCA-init fac DAS | 1 | 8 | 0.886 | hard (307) | `artifacts/cpca_das_v3/sva.json` |
| SVA | factorized DAS L1=0.1 | 4 | 8 | 0.960 | full (150) | `lib/factorized_das/results/atomic-sweep-40/sva_L8/` |
| SVA | CPCA-init DAS | 4 | 8 | 0.833 | hard (24) | `artifacts/cpca_hard_sva/cpca_hard_sva/cpca_init_multitask_hard.json` |
| Gender Bias | vanilla DAS | 32 | 9 | 0.847 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/gender_bias.json` |
| Gender Bias | Riem. fac DAS L=0.05 (best) | 32 | 9 | 0.793 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/gender_bias.json` |
| Gender Bias | constrained PCA (frac=0.01) | 4 | 5 | 0.700 | full (80) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_gender_bias.json` |
| Gender Bias | CPCA-init fac DAS | 1 | 8 | 0.810 | hard (270) | `artifacts/cpca_das_v3/gender_bias.json` |
| Capital Country | vanilla DAS | 32 | 8 | 0.947 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/capital_country.json` |
| Capital Country | Riem. fac DAS L=0.05 (best) | 32 | 8 | 0.933 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/capital_country.json` |
| Capital Country | constrained PCA (frac=0.01) | 4 | 9 | 0.886 | full (70) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_capital_country.json` |
| Capital Country | delta PCA | 4 | 5 | 0.871 | full (70) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_capital_country.json` |
| Greater Than | vanilla DAS | 32 | 8 | 0.000 | full (150) | `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/greater_than.json` |
| Hypernymy | constrained PCA (any) | any | any | 1.000 | full (78) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_hypernymy.json` |
| Hypernymy | delta PCA | 4 | 5 | 1.000 | full (78) | `artifacts/constrained_pca_sweep/constrained_pca_sweep_hypernymy.json` |

### Other checkpoints summary

| Checkpoint | Best IOI | Best SVA | Notes |
|-----------|----------|----------|-------|
| shared_bank_per_proj_sel (1024f) | 0.940 (vanilla k=32) | 0.680 (fac k=4) | Most runs; Grassmannian analysis |
| shared_bank_global_dense (1024f) | 0.993 (vanilla k=64) | -- | k=64 delta_pca only 0.113 |
| per_layer_per_proj_sel (1024f) | -- | 0.680 (fac k=4) | Limited runs |
| no_lambda_5k (1024f) | -- | 0.680 (fac k=32) | Zero sparsity checkpoint |
| classic-sweep-133 (4096f) | 0.047 (fac k=4) | 0.673 (fac k=4) | Poor IOI perf with heavy L1 |
| polar-sweep-90 (1024f, DST) | 0.793 (delta_pca k=4) | 0.187 (delta_pca k=4) | delta_pca only |

---

## SECTION 1: Atomic-sweep-40 -- Constrained PCA Sweep (Full Dataset)

Source: `artifacts/constrained_pca_sweep/`
Timestamp: 2026-06-17
Method: PCA on factor activation deltas, constrained to top-frac active factors per layer
Mode: Full dataset (200 pairs, 80 eval for IOI/SVA/gender_bias/gendered_pronoun, 70 for capital_country, 78 for hypernymy)

### 1a. IOI

| Layer | delta_pca k=4 | delta_pca k=8 | delta_pca k=16 | CPCA frac=0.01 k=4 | CPCA frac=0.01 k=8 | CPCA frac=0.02 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------|----------------|---------------------|---------------------|---------------------|---------------------|
| 5 | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| 7 | 0.863 | 0.813 | 0.775 | 0.975 | 0.975 | 0.975 | 0.963 |
| 8 | 0.213 | 0.200 | 0.188 | 0.988 | 0.975 | 0.975 | 0.863 |
| 9 | 0.488 | 0.338 | 0.188 | 0.988 | 0.975 | 0.975 | 0.963 |
| 10 | 0.888 | 0.675 | 0.475 | 0.988 | 0.988 | 0.988 | 0.988 |

### 1b. SVA

| Layer | delta_pca k=4 | delta_pca k=8 | delta_pca k=16 | CPCA frac=0.01 k=4 | CPCA frac=0.01 k=8 | CPCA frac=0.02 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------|----------------|---------------------|---------------------|---------------------|---------------------|
| 5 | 0.875 | 0.888 | 0.888 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 0.600 | 0.575 | 0.525 | 1.000 | 1.000 | 1.000 | 0.988 |
| 8 | 0.175 | 0.138 | 0.113 | 1.000 | 1.000 | 1.000 | 0.925 |
| 9 | 0.150 | 0.113 | 0.113 | 1.000 | 1.000 | 1.000 | 0.938 |
| 10 | 0.038 | 0.038 | 0.038 | 1.000 | 1.000 | 0.988 | 0.875 |

### 1c. Gender Bias

| Layer | delta_pca k=4 | CPCA frac=0.01 k=4 | CPCA frac=0.01 k=8 | CPCA frac=0.02 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------------|---------------------|---------------------|---------------------|
| 5 | 0.638 | 0.700 | 0.713 | 0.713 | 0.713 |
| 7 | 0.613 | 0.700 | 0.713 | 0.688 | 0.688 |
| 8 | 0.513 | 0.688 | 0.713 | 0.650 | 0.638 |
| 9 | 0.075 | 0.625 | 0.625 | 0.575 | 0.550 |
| 10 | 0.000 | 0.613 | 0.613 | 0.525 | 0.250 |

### 1d. Capital Country

| Layer | delta_pca k=4 | CPCA frac=0.01 k=4 | CPCA frac=0.02 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------------|---------------------|---------------------|
| 5 | 0.871 | 0.871 | 0.871 | 0.871 |
| 7 | 0.871 | 0.871 | 0.871 | 0.871 |
| 8 | 0.843 | 0.871 | 0.871 | 0.871 |
| 9 | 0.743 | 0.886 | 0.886 | 0.843 |
| 10 | 0.757 | 0.857 | 0.843 | 0.814 |

### 1e. Hypernymy

| Layer | delta_pca k=4 | CPCA frac=0.01 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------------|---------------------|
| 5 | 1.000 | 1.000 | 1.000 |
| 7 | 0.987 | 1.000 | 1.000 |
| 8 | 0.872 | 1.000 | 1.000 |
| 9 | 0.731 | 1.000 | 1.000 |
| 10 | 0.628 | 1.000 | 1.000 |

### 1f. Gendered Pronoun

| Layer | delta_pca k=4 | CPCA frac=0.01 k=4 | CPCA frac=0.02 k=4 | CPCA frac=0.05 k=4 |
|-------|---------------|---------------------|---------------------|---------------------|
| 5 | 0.063 | 0.000 | 0.000 | 0.000 |
| 7 | 0.413 | 0.000 | 0.000 | 0.000 |
| 8 | 0.588 | 0.000 | 0.000 | 0.000 |
| 9 | 0.938 | 0.000 | 0.000 | 0.025 |
| 10 | 0.988 | 0.000 | 0.000 | 0.050 |

Note: Constrained PCA completely fails on gendered_pronoun. The task direction does not align with top-active factor subspace.

---

## SECTION 2: Atomic-sweep-40 -- k=32 Riemannian DAS (Full Dataset)

Source: `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/`
Method: Vanilla DAS + delta PCA + Riemannian DAS + Riemannian factorized DAS (500 steps, k=32)
Mode: Full dataset (200 train, 150 eval)
Top factors saved: 50 per result (not full list -- need new runs for complete factor lists).

### 2a. Summary (eval IIA)

| Task | Layer | delta_pca | vanilla DAS | Riemannian DAS | Riem. fac L=0.05 | Riem. fac L=0.1 | Riem. fac L=0.5 |
|------|-------|-----------|-------------|----------------|------------------|-----------------|-----------------|
| IOI | 10 | 0.247 | 0.993 | 0.640 | **0.993** | **0.993** | **0.993** |
| SVA | 8 | 0.153 | 0.973 | 0.753 | 0.907 (best 0.960) | 0.867 (best 0.953) | 0.920 (best 0.967) |
| Gender Bias | 9 | 0.120 | 0.847 | 0.407 | 0.587 (best 0.793) | 0.487 (best 0.793) | 0.340 (best 0.707) |
| Greater Than | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Capital Country | 8 | 0.773 | 0.947 | 0.820 | 0.927 (best 0.933) | 0.900 (best 0.913) | 0.847 (best 0.927) |

Notes:
- IOI Riemannian factorized DAS matches vanilla ceiling (0.993) at all lambda values
- Greater Than completely fails across ALL methods
- "best" values are peak eval IIA during training (before potential overfitting)

---

## SECTION 3: Atomic-sweep-40 -- k=4 Cross-Layer Factorized DAS (Full Dataset)

Source: `lib/factorized_das/results/atomic-sweep-40/`
Method: Factorized DAS with L1 regularization sweeps
Mode: Full dataset (~150 eval)

### 3a. IOI (k=4)

| Layer | L1=0.05 | L1=0.1 | L1=0.5 | L1=1.0 | delta_pca | vanilla DAS |
|-------|---------|--------|--------|--------|-----------|-------------|
| 0 | 0.960 | 0.960 | 0.960 | -- | 0.960 | 0.967 |
| 3 | 0.960 | 0.960 | 0.960 | 0.967 | 0.960 | 0.973 |
| 8 | 0.967 | 0.967 | 0.967 | 0.987 | 0.207 | 0.993 |
| 10 | 0.973 | 0.973 | 0.967 | 0.973 | 0.793 | 0.993 |
| 11 | 0.973 | 0.973 | 0.953 | 0.987 | 0.913 | 0.993 |

N active factors at L1=0.5: L0=16, L3=17, L8=19, L10=22, L11=18
N active factors at L1=1.0: L3=1, L8=1, L10=3, L11=3

### 3b. SVA (k=4)

| Layer | L1=0.05 | L1=0.1 | L1=0.5 | L1=1.0 | delta_pca | vanilla DAS |
|-------|---------|--------|--------|--------|-----------|-------------|
| 0 | 0.953 | 0.960 | 0.960 | 0.960 | 0.780 | 0.940 |
| 3 | 0.960 | 0.960 | 0.953 | 0.967 | 0.780 | 0.960 |
| 8 | 0.960 | 0.960 | 0.953 | 0.953 | 0.187 | 0.967 |
| 11 | 0.960 | 0.960 | 0.960 | -- | 0.047 | 0.960 |

N active factors at L1=0.5: L0=14, L3=13, L8=9, L11=17
N active factors at L1=1.0: L0=3, L3=3, L8=2

### 3c. IOI k-variants (Layer 10, factorized DAS)

Source: `lib/factorized_das/results/atomic-sweep-40-k{2,3,5}/`

| k | delta_pca eval | vanilla DAS eval | fac DAS L1=0.5 eval | fac DAS L1=1.0 eval | n_active (L1=0.5) | n_active (L1=1.0) |
|---|----------------|------------------|---------------------|---------------------|--------------------|--------------------|
| 2 | 0.900 | 0.987 | 0.967 | 0.967 | 5 | 1 |
| 3 | 0.853 | 0.993 | 0.967 | 0.967 | 8 | 3 |
| 5 | 0.727 | 0.993 | 0.967 | 0.973 | 24 | 3 |

Key finding: delta_pca degrades sharply as k increases (0.900 at k=2 to 0.727 at k=5). Factorized DAS is stable at 0.967-0.973 regardless of k. At L1=1.0 with k=2, a SINGLE active factor achieves 0.967 IIA.

### 3d. Grassmann factorized DAS (k=32, Layer 10)

Source: `lib/factorized_das/results/atomic-sweep-40-k32-grassmann/ioi/`
50 steps only, lambda=0.1

| Method | eval IIA | n_active |
|--------|----------|----------|
| delta_pca | 0.247 | -- |
| vanilla DAS | 0.993 | -- |
| factorized DAS L1=0.1 | 0.993 | 8192 (no sparsification at 50 steps) |

---

## SECTION 4: Atomic-sweep-40 -- CPCA-init DAS (Hard Mode)

### 4a. CPCA v3: Multi-method comparison on hard examples

Source: `artifacts/cpca_das_v3/`
Method: Multiple methods compared head-to-head
Mode: Hard examples only (margin < 1.0), k=1, Layer 8
Checkpoint: atomic-sweep-40
n_active_cpca: 81 factors (top 1%)

#### Gender Bias (n_total=986, n_hard=539, 2 folds)

| Method | IIA mean +/- std | Strict IIA mean |
|--------|-----------------|-----------------|
| constrained_pca | 0.463 +/- 0.046 | 0.018 |
| delta_pca | 0.389 +/- 0.046 | 0.024 |
| vanilla_das_100 | 0.639 +/- 0.046 | 0.453 |
| vanilla_das_200 | 0.634 +/- 0.051 | 0.445 |
| cpca_init_unconstrained_100 | 0.732 +/- 0.028 | 0.620 |
| factorized_das_delta_s100_l0.05 | 0.810 +/- 0.042 | 0.817 |
| factorized_das_delta_s200_l0.1 | 0.815 +/- 0.046 | 0.817 |
| cpca_init_fac_s100_l0.05 | 0.810 +/- 0.051 | 0.825 |
| cpca_init_fac_s200_l0.05 | 0.810 +/- 0.051 | 0.825 |
| cpca_init_fac_s200_l0.1 | 0.810 +/- 0.051 | 0.817 |

#### SVA (n_total=12400, n_hard=614, 2 folds)

| Method | IIA mean +/- std | Strict IIA mean |
|--------|-----------------|-----------------|
| constrained_pca | 0.520 +/- 0.033 | 0.000 |
| delta_pca | 0.533 +/- 0.029 | 0.069 |
| vanilla_das_100 | 0.687 +/- 0.037 | 0.397 |
| vanilla_das_200 | 0.683 +/- 0.033 | 0.379 |
| cpca_init_unconstrained_100 | 0.817 +/- 0.029 | 0.622 |
| factorized_das_delta_s100_l0.05 | 0.846 +/- 0.049 | 0.701 |
| factorized_das_delta_s200_l0.1 | 0.862 +/- 0.065 | 0.738 |
| cpca_init_fac_s100_l0.05 | 0.886 +/- 0.033 | 0.762 |
| cpca_init_fac_s200_l0.05 | 0.886 +/- 0.033 | 0.762 |
| cpca_init_fac_s200_l0.1 | 0.886 +/- 0.033 | 0.770 |

#### Capital Country (n_total=190, n_hard=32, 5 folds)

**k=1:**

| Method | IIA mean +/- std | Strict IIA mean |
|--------|-----------------|-----------------|
| constrained_pca | 0.419 +/- 0.237 | 0.000 |
| delta_pca | 0.386 +/- 0.234 | 0.000 |
| vanilla_das_100 | 0.414 +/- 0.211 | 0.033 |
| cpca_init_unconstrained_100 | 0.419 +/- 0.237 | 0.050 |
| factorized_das_delta_s100_l0.05 | 0.448 +/- 0.227 | 0.090 |
| cpca_init_fac_s100_l0.05 | 0.476 +/- 0.193 | 0.123 |
| cpca_init_fac_s200_l0.05 | 0.476 +/- 0.193 | 0.123 |

**k=4:**

| Method | IIA mean +/- std | Strict IIA mean |
|--------|-----------------|-----------------|
| constrained_pca | 0.419 +/- 0.237 | 0.000 |
| delta_pca | 0.319 +/- 0.262 | 0.000 |
| vanilla_das_100 | 0.410 +/- 0.228 | 0.123 |
| cpca_init_unconstrained_100 | 0.419 +/- 0.237 | 0.050 |
| factorized_das_delta_s100_l0.05 | 0.448 +/- 0.227 | 0.090 |
| cpca_init_fac_s100_l0.05 | 0.448 +/- 0.227 | 0.090 |

### 4b. CPCA Hard IOI

Source: `artifacts/cpca_hard_ioi/cpca_hard_ioi/cpca_init_multitask_hard.json`
Task: IOI | Layer: 8 | n_hard=83, n_eval=34 | n_active=81

| Method | k | IIA | Strict IIA |
|--------|---|-----|------------|
| constrained_pca | 1 | 0.529 | 0.000 |
| cpca_init_das | 1 | 0.912 | 0.700 |
| constrained_pca | 4 | 0.471 | 0.000 |
| cpca_init_das | 4 | 0.971 | 0.900 |

### 4c. CPCA Hard SVA

Source: `artifacts/cpca_hard_sva/cpca_hard_sva/cpca_init_multitask_hard.json`
Task: SVA | Layer: 8 | n_hard=58, n_eval=24 | n_active=81

| Method | k | IIA | Strict IIA |
|--------|---|-----|------------|
| constrained_pca | 1 | 0.375 | 0.000 |
| cpca_init_das | 1 | 0.750 | 0.600 |
| constrained_pca | 4 | 0.250 | 0.000 |
| cpca_init_das | 4 | 0.833 | 0.733 |

---

## SECTION 5: Atomic-sweep-40 -- Constrained PCA Hard Mode (IOI L8)

### 5a. Hard-mode IOI sweep

Source: `artifacts/constrained_pca_hard/constrained_pca_hard.json`
Layer: 8 | n_hard=83, n_eval=34, n_flippable=10

| Method | k=1 IIA | k=1 strict | k=4 IIA | k=4 strict |
|--------|---------|------------|---------|------------|
| delta_pca | 0.088 | 0.0 | 0.088 | 0.0 |
| constrained frac=0.01 (81f) | 0.529 | 0.0 | 0.471 | 0.0 |
| constrained frac=0.02 (163f) | 0.353 | 0.0 | 0.324 | 0.0 |
| constrained frac=0.05 (409f) | 0.176 | 0.0 | 0.176 | 0.1 |
| constrained frac=0.1 (819f) | 0.147 | 0.1 | 0.118 | 0.1 |
| constrained frac=0.2 (1638f) | 0.088 | 0.0 | 0.088 | 0.0 |
| constrained frac=0.5 (4096f) | 0.088 | 0.0 | 0.088 | 0.0 |
| vanilla DAS | 0.735 | 0.2 | 0.882 | 0.6 |

### 5b. Method comparison (fixes.json)

Source: `artifacts/constrained_pca_fixes/fixes.json`
Layer: 8, k=1 | n_active=81 | n_eval=34

| Method | IIA | Strict IIA |
|--------|-----|------------|
| constrained_pca | 0.529 | 0.0 |
| margin_weighted_pca | 0.500 | 0.0 |
| causal_weighted_pca | 0.529 | 0.0 |
| das_in_constrained_100 | 0.706 | 0.0 |
| das_in_constrained_300 | 0.765 | 0.2 |
| cpca_init_das_50 | 0.794 | 0.3 |
| **cpca_init_das_100** | **0.912** | **0.7** |
| vanilla_das_100 | 0.735 | 0.2 |
| factorized_das | 0.882 | 0.6 |

### 5c. Alternative methods (alternatives.json)

Source: `artifacts/constrained_pca_alternatives/alternatives.json`
Layer: 8, n_active=81 | n_eval=34

| Method | k=1 IIA | k=4 IIA |
|--------|---------|---------|
| constrained_pca | 0.529 | 0.471 |
| soft_selector_pca | 0.088 | 0.088 |
| per_projection_union_pca (311f) | 0.147 | 0.147 |
| per_proj_fl_q (81f) | 0.559 | 0.559 |
| per_proj_fl_k (81f) | 0.529 | 0.471 |
| per_proj_fl_v (81f) | 0.529 | 0.500 |
| per_proj_fl_o (81f) | 0.471 | 0.353 |
| pr_weighted_pca | 0.529 | 0.441 |
| tucker_r2/r4/r8 | 0.529 | 0.471 |
| cross_layer_pca (L5,7,8,9) | 0.588 | 0.471 |
| fht_activations_pca | 0.529 | 0.471 |
| fht_full_factor_pca | 0.088 | 0.088 |
| margin_weighted_pca | 0.500 | 0.529 |
| vanilla_das | 0.735 | 0.882 |

### 5d. Diagnosis

Source: `artifacts/diagnose_constrained_pca/diagnosis.json`
Key finding: CPCA and DAS directions are 82.5 degrees apart at k=1. DAS capture fraction in constrained subspace = 0.14. The DAS direction lives mostly outside the factor subspace.

---

## SECTION 6: Atomic-sweep-40 -- Factorized DAS Hard Mode

### 6a. Hard v4 (bug-fixed)

Source: `artifacts/factorized_das_hard_v4/`
Layer: 8, k=1

#### IOI (n_hard=83, n_eval=34)

| Method | IIA | Strict IIA |
|--------|-----|------------|
| vanilla_das | 0.735 | 0.200 |
| factorized_das L1=0.1 | 0.882 | 0.600 |
| factorized_das L1=0.1 best_ckpt | 0.765 | 0.300 |

#### SVA (n_hard=58, n_eval=24)

| Method | IIA | Strict IIA |
|--------|-----|------------|
| vanilla_das | 0.667 | 0.467 |
| factorized_das L1=0.1 | 0.583 | 0.333 |

### 6b. Hard v3 (earlier, partial)

Source: `artifacts/factorized_das_hard_v3/factorized_das_hard_sva.json`
SVA only: vanilla_das k=1 IIA=0.667 (strict 0.467)

### 6c. Hard v1 (buggy -- factorized variants errored)

Source: `artifacts/factorized_das_hard/factorized_das_hard/factorized_das_hard_ioi.json`
IOI: vanilla_das k=1 IIA=0.735 (strict 0.200), k=2 IIA=0.794 (strict 0.500)

---

## SECTION 7: Atomic-sweep-40 -- Hard Examples Three Methods (IOI, Multi-Layer)

Source: `artifacts/hard_examples_three_methods/`
Task: IOI | Hard mode (n_hard=83, n_eval=34) | 300 steps, L1=0.1

### Factorized DAS across layers

| Layer | Method | k=1 IIA | k=2 IIA | k=4 IIA |
|-------|--------|---------|---------|---------|
| 5 | factor_pca | 0.706 | 0.706 | 0.706 |
| 5 | factorized_das | 0.706 | 0.676 | 0.735 |
| 5 | random | 0.706 | -- | -- |
| 8 | factor_pca | 0.529 | 0.471 | 0.471 |
| 8 | factorized_das | 0.647 | 0.882 | 0.912 |
| 8 | factorized_das best | 0.765 | 0.882 | 0.912 |
| 8 | random | 0.706 | -- | -- |
| 9 | factor_pca | 0.588 | 0.618 | 0.588 |
| 9 | factorized_das | 0.676 | 0.765 | 0.824 |
| 9 | factorized_das best | 0.735 | 0.794 | 0.853 |
| 10 | factor_pca | 0.706 | 0.706 | 0.706 |
| 10 | factorized_das | 0.735 | 0.824 | 0.853 |
| 10 | factorized_das best | 0.824 | 0.853 | 0.765 |

### Vanilla DAS across layers

| Layer | k=1 IIA | k=2 IIA | k=4 IIA |
|-------|---------|---------|---------|
| 8 | 0.824 | 0.882 | 0.912 |
| 9 | 0.735 | 0.794 | 0.824 |
| 10 | 0.765 | 0.735 | 0.853 |

---

## SECTION 8: Atomic-sweep-40 -- CPCA DAS v5 (Hard Mode, Per-Fold)

Source: `artifacts/cpca_das_v5/`
Status: Only capital_country completed (3/5 folds). sva and gender_bias have direction .pt files but no IIA evaluation results.
Layer: 8, k=1

### Capital Country (3-fold average, n_hard=32, ~6-7 eval per fold)

| Method | Mean IIA | Mean Strict IIA |
|--------|----------|-----------------|
| constrained_pca | 0.254 | 0.000 |
| delta_pca | 0.254 | 0.000 |
| vanilla_das_100 | 0.302 | 0.056 |
| vanilla_das_200 | 0.302 | 0.056 |
| cpca_init_unconstrained_100 | 0.310 | 0.083 |
| delta_fac_s100_l0.05 | 0.357 | 0.150 |
| cpca_fac_s100_l0.05 | 0.405 | 0.206 |
| cpca_fac_s200_l0.05 | 0.405 | 0.206 |

Note: Very small per-fold eval sizes (6-7 examples) make these estimates extremely noisy.

---

## SECTION 9: Atomic-sweep-40 -- Unique DAS and Tucker Basis DAS

### 9a. Unique DAS

Source: `artifacts/unique_das/unique_das/unique_das_L8.json`
Task: IOI | Layer: 8 | Hard mode (n_eval=34) | 5 seeds, 50 steps

| Method | k=1 IIA mean +/- std |
|--------|---------------------|
| vanilla_das | 0.941 +/- 0.000 |
| min_norm_das | 0.853 +/- 0.019 |
| kl_das | 0.088 +/- 0.000 |

### 9b. Tucker Basis DAS

Source: `artifacts/tucker_basis_das/tucker_basis_das/tucker_basis_das_ioi.json`
Task: IOI | Layer: 8 | Hard mode (n_eval=34)
Result: All Tucker variants (r=4,8,16) = random baseline (0.706 IIA). Tucker decomposition does not help.

### 9c. PCA Mode DAS

Source: `artifacts/pca_mode_das/pca_mode_das_ioi.json`
Task: IOI | Layer: 10 | Full dataset (n_eval=50)
Result: All per-mode factor subspace swaps = 0.02 IIA (random). Complete failure. Baseline full_swap = 0.96.

---

## SECTION 10: Atomic-sweep-40 -- Per-Variable Vanilla DAS (IOI Subtasks)

Source: `artifacts/per_variable_das/per_variable_das_exploration.json`
Method: Vanilla DAS | k=4 | n_pairs=200 | 100 steps
Mode: Full dataset

| Layer | s2_io_flip | s1_io_flip | abc | random_names | full_flip |
|-------|------------|------------|-----|--------------|-----------|
| 5 | 0.988 | 0.988 | 0.988 | 0.988 | 1.000 |
| 7 | 1.000 | 0.988 | 0.988 | 0.988 | 0.988 |
| 9 | 0.988 | 1.000 | 0.975 | 0.975 | 0.988 |
| 10 | 0.988 | 0.988 | 0.988 | 0.988 | 1.000 |

---

## SECTION 11: Atomic-sweep-40 -- IOI Subtask Decomposition

### 11a. V1

Source: `artifacts/ioi_subtask/ioi_subtask_decomposition.json`
Method: Constrained PCA (zero-training cross-eval) | k=4 | frac=0.01 (81 factors)
Mode: Full dataset (80 examples, only ~1 flippable per subtask)
Result: All cross-eval IIA = 0.975-0.988. No discrimination between subtasks at this easy difficulty level.

### 11b. V2

Source: `artifacts/ioi_subtask_v2/ioi_subtask_decomposition_v2.json`
Extended with k=1,2,4, ablation analysis, random baselines.
Key finding: Injection IIA saturates at ~0.988 for all subtasks. Ablation analysis is more informative -- ablating the constrained PCA direction at L8 drops abc IIA to 0.575, s1_io_flip to 0.375, s2_io_flip to 0.188.

---

## SECTION 12: Atomic-sweep-40 -- DAS+EAP Combined Analysis

Source: `lib/factorized_das/results/atomic-sweep-40/das_eap/`

### 12a. DAS-EAP attribution scores

Files: `ioi_L10_l1={0.5,1.0}_das_eap.json`, `ioi_k{2,3,5}_L10_l1=1.0_das_eap.json`
These contain per-edge attribution scores for factorized DAS at different k values, used for circuit discovery.

### 12b. Faithfulness evaluation

Source: `lib/factorized_das/results/atomic-sweep-40/das_eap/faithfulness/faithfulness_L10_l1=1.0.json`
Faithfulness of the sub-edge circuit at L10 with L1=1.0.

### 12c. Alternative metrics

Source: `lib/factorized_das/results/atomic-sweep-40/das_eap/alternative_metrics/`
Multiple continuous IIA alternatives: KL, JS, prob_diff, normalized_logit, logit_diff.

### 12d. Cross-metric projection

Source: `lib/factorized_das/results/atomic-sweep-40/das_eap/cross_metric_projection/`
Versions v1, v2, v3 of cross-metric projection analysis.

---

## SECTION 13: Constrained PCA (Full Detail, IOI with Counterfactual Types)

Source: `artifacts/constrained_pca/constrained_pca_results.json` and `lib/factorized_das/results/atomic-sweep-40/constrained_pca/`
Task: IOI | k=4 | Full dataset | Layers: 5, 7, 9, 10
Includes all 5 counterfactual types and frac sweeps from 0.01 to 0.5.

Key pattern: Constrained PCA at frac=0.01 achieves 0.975-0.988 IIA across all layers and counterfactual types. Performance degrades monotonically as frac increases (more factors = more noise). At frac=0.5, L9 drops to 0.663-0.788 and L10 drops to 0.713-0.763.

---

## SECTION 14: Random Direction Baselines (ceval)

Source: `artifacts/ceval-c01-random-*/c01_das_iia_random.json` (18 files)
Method: Random DAS direction (100 random trials each), k=1,2,4
Mode: Full dataset

| Task | k=1 random IIA | k=4 random IIA | Best head |
|------|----------------|----------------|-----------|
| IOI | 0.540-0.551 | 0.577 | varies |
| SVA | 0.450 | 0.450 | [0,4] |
| Greater Than | 0.630 | 0.630 | varies |
| Gendered Pronoun | 0.390 | 0.388 | varies |
| Induction | 0.528 | 0.527 | varies |
| Copy Suppression | 0.536-0.540 | 0.522-0.540 | varies |
| Acronym | 0.322 | 0.331 | [10,10] |
| RTI | 0.270 | 0.270 | varies |

---

## SECTION 15: Other Checkpoints (Non-atomic-sweep-40)

### 15a. shared_bank_per_proj_sel (1024 factors)

Source: `lib/analysis_elliot/writeups/.../01_DAS_CIRCUITS/data/` (W&B artifacts)

#### IOI (Layer 10)

- delta_pca k=32: eval_iia = 0.913
- vanilla DAS k=32: eval_iia = 0.940
- Factorized DAS k=32 L1=0.1 (6bywcxsj): eval_iia = 0.893
- Factorized DAS k=32 L1=1.0 (c4rnplvf): best_iia = 0.913, final = 0.833

#### SVA (Layer 8)

- delta_pca k=4: eval_iia = 0.587
- vanilla DAS k=4: eval_iia = 0.653
- Factorized DAS k=4 L1=1.0 (2u5ki7xh): eval_iia = 0.653 (matches vanilla)
- Best fac DAS: 0.680 (qnrguy1w, but final drops to 0.413)

### 15b. shared_bank_global_dense (1024 factors)

Source: `lib/analysis_grassmanian_subspace/.../self_consistent_fac_das/` and `lib/factorized_das/results/dense-k{32,64}/`

| Task | Layer | k | delta_pca | vanilla DAS |
|------|-------|---|-----------|-------------|
| IOI | 10 | 4 | 0.907 | -- |
| IOI | 10 | 64 | 0.113 | 0.993 |
| SVA | 8 | 4 | 0.193 | -- |
| SVA | 8 | 64 | 0.147 | -- |
| Capital Country | 8 | 4 | 0.713 | -- |
| Gender Bias | 9 | 4 | 0.000 | -- |

### 15c. Dense k=32 Grassmann (shared_bank_global_dense, 1024 factors)

Source: `lib/factorized_das/results/dense-k32-grassmann/`
500 steps, k=32, full dataset (200 train, 150 eval)

| Task | Layer | delta_pca eval | vanilla DAS eval | fac L1=0.1 eval | fac L1=0.5 eval | fac L1=1.0 eval | n_active (L1=1.0) |
|------|-------|----------------|------------------|-----------------|-----------------|-----------------|---------------------|
| IOI | 10 | 0.160 | 0.993 | 0.993 | 0.993 | 0.993 | 33 |
| SVA | 8 | 0.127 | 0.973 | 0.960 | 0.960 | 0.967 | 32 |
| Gender Bias | 9 | 0.060 | 0.860 | 0.833 | 0.820 | -- | -- |
| Greater Than | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 99 |
| Capital Country | 8 | 0.800 | 0.953 | 0.953 | 0.953 | -- | -- |

### 15c-ii. Dense k=64 (shared_bank_global_dense, 1024 factors)

Source: `lib/factorized_das/results/dense-k64/`

| Task | Layer | delta_pca eval | vanilla DAS eval | fac L1=0.1 eval | fac L1=0.5 eval | fac L1=1.0 eval | n_active (L1=1.0) |
|------|-------|----------------|------------------|-----------------|-----------------|-----------------|---------------------|
| IOI | 10 | 0.113 | 0.993 | 0.993 | 0.993 | 0.993 | 29 |
| SVA | 8 | 0.147 | 0.953 | 0.960 | 0.960 | -- | -- |

### 15d. classic-sweep-133 (4096 factors, JumpReLU-tanh)

IOI k=4: 0.047 (very poor, L1=5.0 too high)
SVA k=4: 0.673 (tmqbqirb)

### 15e. polar-sweep-90 (1024 factors, DST)

delta_pca only (k=4): IOI L10=0.793, SVA L8=0.187, capital_country L8=0.887, gender_bias L9=0.040

---

## SECTION 16: Grassmannian Subspace Analysis

Source: `lib/analysis_grassmanian_subspace/factorized_das/grassmannian_analysis.json`
Checkpoint: shared_bank_per_proj_sel (1024 factors)

Cross-seed comparison of factorized DAS directions:
- IOI k=32: Geometrically stable (2.2 degree mean angle across seeds)
- SVA k=4: Rotationally degenerate (31-49 degree mean angle, zero Jaccard)
- IOI vs SVA: Nearly orthogonal (68 degree mean angle)

### Analysis8: Teacher vs Factorized IIA

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis8_teacher_vs_factorized_iia/`
Delta PCA IIA on shared_bank_global_dense at k=4:
- IOI L10: 0.907
- SVA L8: 0.193
- Capital Country L8: 0.713
- Gender Bias L9: 0.000

### Analysis9: k-sweep

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis9_k_sweep/`
IOI k-sweep results at specific k values (k=1, 2, 16).

### Analysis7: Subspace trimming

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis7_subspace_trimming/`
IOI subspace trimming results.

---

## SECTION 17: Linguistic IIA (Head-Level Interventions, NOT DAS)

Source: `artifacts/ling-iia-*/`
Method: Multi-head activation patching (NOT learned subspace DAS)
Models: gpt2, gpt2-medium, gpt2-large, gpt2-xl
Note: These are head-level IIA, not DAS direction IIA. Included for completeness.

| Task | Model | Best IIA (gender axis) | Ceiling IIA (all heads) |
|------|-------|------------------------|------------------------|
| reflexive | gpt2 | 0.913 (number axis) | 0.813 |
| reflexive | gpt2-medium | 0.888 (gender) | 0.800 |
| reflexive | gpt2-large | 1.000 (gender) | 0.800 |
| reflexive | gpt2-xl | 1.000 (gender) | 0.838 |
| ellipsis | gpt2 | 0.000 | 0.000 |
| ellipsis | gpt2-medium | 0.000 | 0.760 |
| ellipsis | gpt2-large | 0.000 | 0.847 |
| ellipsis | gpt2-xl | 0.000 | 0.824 |
| but_reversal | all models | 0.000 | 0.187-0.803 |

---

## SECTION 18: Geodesic Interpolation and Gradient Decomposition

Source: `artifacts/geodesic_interpolation*/`, `artifacts/gradient_decomposition*/`, `artifacts/geodesic_line_search_5bases/`
All on atomic-sweep-40, Layer 8, Hard mode, n_active_factors=81

### 18a. Geodesic Interpolation -- endpoint IIA

| Task | k | CPCA IIA | DAS optimal IIA | DAS strict | delta_pca IIA | Angle CPCA-to-DAS |
|------|---|----------|-----------------|------------|---------------|-------------------|
| IOI | 1 | 0.529 | 0.941 | 0.8 | 0.088 | -- |
| IOI | 4 | 0.471 | 0.971 | 0.9 | 0.088 | 150.0 deg |
| SVA | 1 | 0.375 | 0.792 | 0.667 | 0.333 | 76.5 deg |

### 18b. Gradient Decomposition -- CPCA-init trajectory

| Task | k | Steps to reach optimal | Optimal IIA | CPCA start IIA |
|------|---|------------------------|-------------|----------------|
| IOI | 1 | 140 steps | 0.941 | 0.529 (step 0) -> 0.706 (step 5) -> 0.912 (step 85) -> 0.941 (step 140) |
| IOI | 4 | ~150 steps | 0.971 | 0.471 -> 0.971 |
| SVA | 1 | ~150 steps | 0.792 | 0.375 -> 0.792 |

### 18c. Geodesic Line Search across 5 Basis Initializations

IOI L8, n_eval=34, 10 starts per basis:

| Basis | k=1 mean IIA | k=1 max | k=4 mean IIA | k=4 max |
|-------|-------------|---------|-------------|---------|
| factor_bank | 0.706 | 0.706 | 0.709 | 0.735 |
| sae_decoders | 0.706 | 0.706 | 0.711 | 0.735 |
| delta_pca | 0.727 | **0.941** | 0.733 | **0.971** |
| probing | 0.813 | 0.824 | 0.805 | 0.824 |
| random | 0.709 | 0.735 | 0.706 | 0.706 |

delta_pca un-perturbed init achieves best peak IIA; probing is most consistently above baseline.

---

## SECTION 19: Mutual Information Gap and Decomposition Comparison

### 19a. Mutual Info Gap

Source: `artifacts/mutual_info_gap/mutual_info_gap.json` and `artifacts/mutual_info_gap_v2/mutual_info_gap.json`
Checkpoint: atomic-sweep-40 | k=4 | frac=0.01 | Full dataset

| Task | Layer | mi_selected | constrained_pca | best_factor | delta_pca |
|------|-------|-------------|-----------------|-------------|-----------|
| IOI | 5 | 0.988 | 0.988 | 0.950 | 0.613 |
| IOI | 7 | 0.988 | 0.988 | 0.950 | 0.250 |
| IOI | 8 | 0.988 | 0.988 | 0.950 | 0.213 |
| IOI | 9 | 0.988 | 0.988 | 0.963 | 0.213 |
| SVA | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVA | 7 | 1.000 | 1.000 | 0.725 | 1.000 |
| SVA | 8 | 1.000 | 1.000 | 0.875 | 1.000 |
| SVA | 9 | 1.000 | 1.000 | 0.688 | 1.000 |
| Capital Country | all | 0.882 | 0.882 | 0.855 | 0.171 |
| Hypernymy | 5-7 | 1.000 | 1.000 | 0.975-0.988 | 0.938-0.963 |
| Hypernymy | 8-9 | 1.000 | 1.000 | 0.950-0.975 | 0.663-0.900 |
| Gendered Pronoun | all | 0.000 | 0.000 | 0.000 | 0.000 |

### 19b. Decomposition Comparison (PCA vs ICA vs Sparse PCA vs NMF)

Source: `artifacts/decomposition_comparison/decomposition_comparison.json`
Checkpoint: atomic-sweep-40 | k=4 | Full dataset

| Task | Layer | PCA | ICA | Sparse PCA | NMF | delta_pca |
|------|-------|-----|-----|------------|-----|-----------|
| IOI | 5 | 0.988 | 0.988 | 0.988 | 0.988 | 0.613 |
| IOI | 7-9 | 0.988 | 0.988 | 0.988 | 0.988 | 0.213-0.250 |
| SVA | all | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Capital Country | all | 0.882 | 0.882 | 0.882 | 0.882 | 0.171 |
| Hypernymy | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.963 |
| Gender Bias | 7 | 0.700 | 0.713 | 0.700 | 0.700 | 0.525 |

Note: PCA, ICA, Sparse PCA, and NMF produce nearly identical IIA on full dataset. delta_pca dramatically worse at deeper layers.

### 19c. Hard Composition (IOI + SVA)

Source: `artifacts/hard_composition/hard_composition.json`
Layer: 8 | n_active=81

| Method | IOI hard k=1 IIA | SVA hard k=1 IIA |
|--------|------------------|------------------|
| CPCA | 0.600 | 0.500 |
| vanilla DAS | 0.900 | 0.800 |
| CPCA-init DAS | 0.967 | 0.950 |
| CPCA-init DAS k=4 | 1.000 | 1.000 |

### 19d. Cross-Task Composition

Source: `artifacts/cross_task_composition/`
IOI+SVA and IOI+SVA+gender_bias composition operators at Layer 5.

### 19e. Constrained EAP Results (with IIA)

Source: `artifacts/master_results/combined_constrained_eap_results_constrained.json`
IOI, atomic-sweep-40, Layer 8

| Method | k | IIA | avg_faithfulness |
|--------|---|-----|------------------|
| constrained frac=0.01 | 2 | 0.975 | 0.543 |
| constrained frac=0.01 | 4 | 0.975 | 0.568 |
| constrained frac=0.01 | 8 | 0.975 | 0.614 |
| constrained frac=0.01 | 16 | 0.975 | 0.623 |
| constrained frac=0.02 | 2 | 0.988 | 0.572 |
| constrained frac=0.05 | 4 | 0.975 | 0.590 |
| constrained frac=0.1 | 16 | 0.900 | 0.593 |

Source: `artifacts/master_results/combined_constrained_eap_results_weighted.json`
Weighted attr PCA: all IIA = 0.988 across all k and frac values.

### 19f. Cross-Metric Projection

Source: `artifacts/master_results/cross_metric_projection_cross_projection_results.json`
IOI, atomic-sweep-40

| Method | k | IIA | avg_faithfulness |
|--------|---|-----|------------------|
| attr_pca_iia | 2-16 | 0.988 | 0.584-0.712 |
| augmented attr+constrained | 6-12 | 0.975 | 0.593-0.717 |

### 19g. PR-Enhanced PCA

Source: `artifacts/pr_enhanced_pca/pr_enhanced_pca/pr_enhanced_pca_L8.json`
IOI, Layer 8, k=1, Hard mode (n_eval=34)

| Method | IIA | Strict IIA |
|--------|-----|------------|
| standard_pca | 0.529 | 0.0 |
| pr_weighted_pca | 0.529 | 0.0 |
| pr_factor_filter | 0.735 | 0.1 |
| pr_gated_das_weighted | 0.794 | 0.3 |
| vanilla_das_300 | 0.824 | 0.4 |
| random_baseline | 0.706 | 0.0 |

---

## SECTION 20: Weight-Space DAS (NOT Activation-Space)

Source: `lib/weight_space_das/results/`
Checkpoint: shared_bank_per_proj_sel (1024 factors)
These analyze whether weight-space structure predicts DAS behavior. Not activation DAS.
12 experiments (exp01-exp12) covering SVD alignment, cross-head SVD, selector composition, weight-derived A matrices, factor specificity, weight IIA proxy, multi-task DAS orthogonality, subspace intersection, discriminative SVD, weight knockout, end-to-end composition, natural text variance.

---

## SECTION 21: Teacher Model (pretrained GPT-2) Vanilla DAS

### 21a. Analysis9: k-sweep on teacher GPT-2

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis9_k_sweep/`
Task: IOI, Layer 10, 300 steps, vanilla DAS on pretrained GPT-2

| k | Train IIA | Eval IIA |
|---|-----------|----------|
| 1 | 0.970 | 0.967 |
| 2 | 0.985 | 0.987 |
| 4 | 0.995 | 0.993 |
| 8 | 0.995 | 0.993 |
| 16 | 0.990 | 0.993 |
| 32 | 0.990 | 0.993 |

Key finding: Even k=1 gets 0.967 IIA on IOI teacher. k>=4 saturates at 0.993.

### 21b. Analysis8: Teacher vs Factorized IIA comparison

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis8_teacher_vs_factorized_iia/`
Checkpoint comparison: teacher (pretrained GPT-2) vs shared_bank_per_proj_sel (factorized)

| Task | Layer | k | Teacher IIA | Factorized IIA | Gap |
|------|-------|---|-------------|----------------|-----|
| IOI | 10 | 32 | 0.820 | 0.913 | -0.093 (factorized wins) |
| SVA | 8 | 4 | 0.920 | 0.693 | +0.227 |
| Gender Bias | 9 | 4 | 0.687 | 0.560 | +0.127 |
| Greater Than | 8 | 4 | 0.000 | 0.000 | 0.000 |
| Capital Country | 8 | 4 | 0.813 | 0.700 | +0.113 |

### 21c. Analysis4: Per-dim IIA on teacher

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis4_logit_lens_distributed/`
Vanilla DAS on teacher at full k:

| Task | Layer | k | Full IIA | Per-dim IIA range |
|------|-------|---|----------|-------------------|
| IOI | 10 | 32 | 0.820 | 0.847-0.967 |
| SVA | 8 | 4 | 0.920 | 0.920-0.960 |
| Gender Bias | 9 | 4 | 0.687 | 0.687-0.693 |
| Greater Than | 8 | 4 | 0.000 | 0.000 |
| Capital Country | 8 | 4 | 0.813 | 0.860-0.907 |

**Dimensionality paradox**: Full 32-dim DAS on IOI gives 0.820 IIA, but any single dimension alone gives 0.847-0.967. Excess dimensions hurt.

Leave-one-out ablation:
- IOI (k=32, L10): Full=0.820, LOO range 0.813-0.873, best removal = dim15 (+0.053)
- SVA (k=4, L8): Full=0.920, LOO range 0.920-0.960, best removal = dim0 (+0.040)
- Capital Country: Full=0.813, LOO range 0.820-0.867, best removal = dim0 (+0.053)
- Gender Bias: Full=0.687, all LOO = 0.687 (no single dim matters)

### 21d. Analysis7: IOI Subspace Trimming

Source: `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/analysis7_subspace_trimming/`
Greedy subspace selection for IOI DAS:
- 1 dim (dim 7): IIA = 0.967
- 2 dims (+dim 21): IIA = 0.980
- 5 dims: IIA = 0.987
- 7 dims (final): IIA = 0.987

Single-dim ranking: dims 7, 12, 17, 19 all achieve IIA = 0.967 alone.

---

## SECTION 22: Per-Layer Factorized DAS (shared_bank_per_proj_sel, IOI)

Source: `lib/analysis_elliot/writeups/.../06_NODE_SELECTION_ALGORITHMS/data/pldas/`
Checkpoint: shared_bank_per_proj_sel (1024 factors), IOI, k=32

| Layer | Train IIA | Eval IIA | n_active |
|-------|-----------|----------|----------|
| 1 | 0.005 | 0.010 | 741 |
| 5 | 0.120 | 0.110 | 990 |
| 6 | 0.095 | 0.100 | 991 |
| 7 | 0.065 | 0.070 | 851 |
| 8 | 0.070 | 0.090 | 1003 |
| 9 | 0.900 | 0.910 | 966 |
| 10 | 0.990 | 0.990 | 995 |
| 11 | 1.000 | 1.000 | 1020 |

---

## SECTION 23: DAS RTI Experiment

Source: `artifacts/das-rti-20260509T001001:v0/das_rti.json`
Method: Vanilla DAS, k sweep (1,2,4,8,16,32,64), layers 0-11
Task: RTI (recursive template induction)
Mode: Full dataset

Best results per layer:
- L10: k=16 IIA=1.0, k=8 IIA=0.96
- L11: k=8 IIA=1.0
- L8: k=64 IIA=0.79
- L6: k=32 IIA=0.04
- L0-L4: all 0.0

---

## SECTION 24: Paper Numbers (Verified Claims)

Source: `docs/track2/paper-track2-das-numbers.json`
Checkpoint: shared_bank_per_proj_sel (1024 factors)

Verified claims used in paper:
- Vanilla DAS ceiling: IOI L10 eval=0.940, SVA L8 eval=0.653 (k=4)
- Factorized DAS: IOI best=0.913, SVA eval=0.653 (matches vanilla exactly)
- Factor interchange top-k: IOI=0.010, SVA=0.480
- Random subspace baseline: IOI=0.010, SVA=0.350
- Sparse DAS (lambda=1): IIA=0.673, 38 active factors (3.7% of bank)
- SVA sparsest: IIA=0.620 with only 3 factors

---

## GAPS AND MISSING DATA

### Tasks never tested with DAS:
- Greater Than: Only k=32 vanilla attempted (IIA=0.0 -- may need different layer or k)
- Gendered Pronoun: Constrained PCA = 0.0 everywhere; no vanilla/factorized DAS attempted
- No cross-model DAS (Qwen, Pythia, Gemma, Llama)

### Incomplete runs:
- cpca_das_v5: Only capital_country has results (3/5 folds). sva and gender_bias have directions but no eval.
- dense-k32-grassmann: JSON files exist but IIA values not extracted in this catalog
- Analysis9 k-sweep: IIA values exist at k=1,2,16 but not fully extracted

### Data quality notes:
- Top factors are truncated to 50 in Riemannian results (user wants full lists)
- Capital Country has very small n_hard (32), making per-fold estimates noisy
- Greater Than may need layer 10 or 11 instead of layer 8
- Hard-mode constrained PCA direction is 82.5 degrees from DAS direction (orthogonal)

---

## FILE INDEX

All file paths are relative to the repo root.

| Directory | Contents |
|-----------|----------|
| `artifacts/constrained_pca_sweep/` | Constrained PCA sweep (6 tasks, full dataset) |
| `artifacts/constrained_pca/` | Initial constrained PCA (IOI, 5 CF types) |
| `artifacts/constrained_pca_hard/` | Constrained PCA on hard IOI |
| `artifacts/constrained_pca_fixes/` | Method comparison on hard IOI |
| `artifacts/constrained_pca_alternatives/` | Alternative methods on hard IOI |
| `artifacts/diagnose_constrained_pca/` | CPCA vs DAS diagnostic |
| `artifacts/cpca_das_v3/` | Multi-method comparison (gender_bias, sva, capital_country) |
| `artifacts/cpca_das_v5/` | Per-fold CPCA DAS (capital_country partial) |
| `artifacts/cpca_hard_ioi/` | CPCA-init DAS hard IOI |
| `artifacts/cpca_hard_sva/` | CPCA-init DAS hard SVA |
| `artifacts/factorized_das_hard_v4/` | Factorized DAS hard (bug-fixed) |
| `artifacts/factorized_das_hard_v3/` | Factorized DAS hard SVA (earlier) |
| `artifacts/factorized_das_hard/` | Factorized DAS hard (buggy) |
| `artifacts/hard_examples_three_methods/` | Multi-layer hard examples comparison |
| `artifacts/unique_das/` | Unique/min-norm DAS variants |
| `artifacts/tucker_basis_das/` | Tucker basis DAS (failed) |
| `artifacts/pca_mode_das/` | PCA mode DAS (failed) |
| `artifacts/per_variable_das/` | Per-variable vanilla DAS (IOI subtasks) |
| `artifacts/ioi_subtask/` | IOI subtask decomposition v1 |
| `artifacts/ioi_subtask_v2/` | IOI subtask decomposition v2 |
| `artifacts/geodesic_interpolation*/` | Geodesic interpolation analysis |
| `artifacts/gradient_decomposition*/` | Gradient decomposition analysis |
| `artifacts/geodesic_line_search_5bases/` | Geodesic line search |
| `artifacts/grassmannian_atlas/` | Grassmannian atlas |
| `artifacts/ceval-c01-random-*/` | Random direction baselines (18 runs) |
| `artifacts/ling-iia-*/` | Linguistic head-level IIA (4 models x 3 tasks) |
| `artifacts/das-rti-*/` | DAS RTI experiment |
| `artifacts/master_results/` | Combined constrained EAP results |
| `lib/factorized_das/results/atomic-sweep-40/` | Per-layer factorized DAS (IOI+SVA) |
| `lib/factorized_das/results/atomic-sweep-40-k32-riemannian/` | k=32 Riemannian DAS (5 tasks) |
| `lib/factorized_das/results/atomic-sweep-40-k{2,3,5}/` | k-variant DAS |
| `lib/factorized_das/results/atomic-sweep-40-k32-grassmann/` | k=32 Grassmann DAS |
| `lib/factorized_das/results/dense-k32-grassmann/` | Dense k=32 Grassmann DAS |
| `lib/factorized_das/results/dense-k64/` | Dense k=64 DAS |
| `lib/analysis_grassmanian_subspace/factorized_das/` | Grassmannian subspace analysis |
| `lib/analysis_grassmanian_subspace/factorized_das_eap/` | Tucker decomposition analysis |
| `lib/analysis_grassmanian_subspace/factorized_peakiness_rotation/` | Analysis 7,8,9 (trimming, teacher, k-sweep) |
| `lib/analysis_elliot/writeups/.../01_DAS_CIRCUITS/` | W&B run artifacts |
| `lib/weight_space_das/results/` | Weight-space experiments (not activation DAS) |
| `docs/track2/` | Paper numbers JSON |
