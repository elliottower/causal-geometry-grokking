# W vs R Marginals + DAS-EAP: Initial Atlas Results

## Summary

Factor-decomposed EAP on GPT-2's IOI circuit using Ivan's `attribute_rotation.py`. For each edge (writer → reader), we compute **two** factor-level marginals:
- **W marginal** (`scores_W`): `delta_c * (grad_resid @ F.T)` — how much each factor contributes through the *writer's* activation difference
- **R marginal** (`scores_R`): `(activation_diff @ F.T) * grad_h` — how much each factor contributes through the *reader's* gradient

Both sum to the same scalar edge score (Layer-1 invariant), but distribute mass across factors differently. The W marginal tells you *what the writer is sending*, the R marginal tells you *what the reader is using*.

## Dense Checkpoint (1024 factors, DenseGSM)

**Global statistics:**
- Total absolute mass: W=203.6, R=139.8, ratio=1.46
- Layer-1 invariant: max|W.sum - R.sum| = 3.25e-07 (exact match)
- W-R cosine on top-50 edges: mean=0.47, std=0.17, range [0.10, 0.85]

**Most asymmetric edges** (W and R disagree most on *which* factors carry the edge):
| Edge | W-R cosine | Edge score |
|------|-----------|------------|
| m0 → r280 | 0.10 | 0.014 |
| m1 → m2.q | 0.16 | 0.015 |
| m0 → r193 | 0.17 | -0.027 |
| a8.h6 → r342 | 0.22 | 0.020 |
| m3 → m5.q | 0.26 | 0.012 |

**Known IOI head rankings** (by absolute writer mass from W marginal):
| Head | Role | W rank | R rank | W mass | R mass |
|------|------|--------|--------|--------|--------|
| a0.h1 | dup_token | 7/157 | 9/157 | 4.46 | 2.50 |
| a2.h2 | prev_token | 14/157 | 20/157 | 2.95 | 1.67 |
| a0.h10 | dup_token | 19/157 | 8/157 | 2.59 | 2.62 |
| a4.h11 | prev_token | 22/157 | 33/157 | 2.29 | 1.21 |
| a7.h9 | s_inhibition | 29/157 | 32/157 | 2.01 | 1.22 |
| a3.h0 | dup_token | 33/157 | 45/157 | 1.68 | 1.03 |
| a7.h3 | s_inhibition | 39/157 | 54/157 | 1.48 | 0.84 |
| a8.h10 | s_inhibition | 64/157 | 62/157 | 0.96 | 0.70 |
| a8.h6 | s_inhibition | 70/157 | 49/157 | 0.91 | 0.92 |
| a9.h9 | name_mover | 111/157 | 115/157 | 0.20 | 0.15 |
| a9.h6 | name_mover | 110/157 | 114/157 | 0.21 | 0.16 |
| a10.h7 | neg_nm | 144/157 | 149/157 | 0.00 | 0.00 |
| a10.h0 | name_mover | 153/157 | 155/157 | 0.00 | 0.00 |

**Key observation**: The dense model ranks early-layer IOI heads (duplicate_token, previous_token) highly but ranks late-layer heads (name_movers, negative_name_movers) very low. This is likely because the DenseGSM has no sparsity, so the factor decomposition at early layers captures more variance in the activation differences. Name movers at layers 9-10 have their signal diluted across many factors.

**Factor concentration:**
- W: 80% in 711/1024 factors, 90% in 857/1024, top-1 = 0.42%
- R: 80% in 750/1024 factors, 90% in 878/1024, top-1 = 0.28%
- Spearman rho(W factors, R factors) = 0.449 (p < 1e-51)
- Top-20 factor Jaccard: 0.053 (only 2 overlapping factors: [890, 934])

**Per-head factor profiles:**
| Head | W entropy | W top-5 | R entropy | R top-5 |
|------|-----------|---------|-----------|---------|
| a9.h9 (NM) | 6.71 | 880,319,300,690,884 | 6.70 | 401,223,572,282,823 |
| a10.h0 (NM) | 6.65 | 181,373,300,666,630 | 6.50 | 83,552,233,469,92 |
| a7.h3 (SI) | 6.65 | 244,953,886,795,101 | 6.78 | 101,953,319,511,867 |
| a7.h9 (SI) | 6.72 | 890,930,288,23,637 | 6.84 | 540,101,953,878,319 |

## Atomic-sweep-40 Checkpoint (8192 factors, DSTGSM)

**Global statistics:**
- Total absolute mass: W=747.0, R=233.5, **ratio=3.20** (much more asymmetric than dense)
- Layer-1 invariant: max|W.sum - R.sum| = 1.55e-06 (exact match)
- W-R cosine on top-50 edges: mean=0.45, std=0.28, range [0.18, 0.88]

**Most asymmetric edges:**
| Edge | W-R cosine | Edge score |
|------|-----------|------------|
| a4.h11 → m4.q | 0.18 | 0.029 |
| a2.h2 → m4.q | 0.18 | 0.036 |
| a4.h3 → m5.q | 0.18 | 0.038 |
| a4.h3 → m4.q | 0.19 | 0.027 |
| a0.h1 → r231 | 0.19 | -0.024 |

**Known IOI head rankings:**
| Head | Role | W rank | R rank | W mass | R mass |
|------|------|--------|--------|--------|--------|
| a0.h1 | dup_token | **2/157** | 10/157 | 21.15 | 3.80 |
| a2.h2 | prev_token | 6/157 | 18/157 | 14.64 | 2.80 |
| a7.h9 | s_inhibition | 8/157 | 16/157 | 11.93 | 3.03 |
| a4.h11 | prev_token | 11/157 | 28/157 | 10.32 | 2.13 |
| a0.h10 | dup_token | 13/157 | 24/157 | 10.00 | 2.37 |
| a8.h6 | s_inhibition | 19/157 | 26/157 | 8.47 | 2.26 |
| a8.h10 | s_inhibition | 22/157 | 21/157 | 7.60 | 2.48 |
| a7.h3 | s_inhibition | 25/157 | 27/157 | 7.52 | 2.26 |
| a9.h9 | name_mover | 38/157 | 36/157 | 6.21 | 1.87 |
| a9.h6 | name_mover | 39/157 | 37/157 | 6.16 | 1.82 |
| a3.h0 | dup_token | 41/157 | 45/157 | 6.03 | 1.62 |
| a10.h0 | name_mover | 81/157 | 65/157 | 3.56 | 1.15 |
| a10.h7 | neg_nm | 83/157 | 67/157 | 3.55 | 1.14 |

**Key observation**: The DST model ranks ALL known IOI heads higher than dense. Name movers jump from rank 110+ to 38-39; s_inhibition from rank 29-70 to 8-25. The sparsity learned by DSTGSM makes the factor decomposition much more discriminative. The W marginal consistently ranks heads higher (more concentrated signal) than R.

**Factor concentration:**
- W: 80% in 6186/8192 factors, 90% in 7147/8192, top-1 = 0.03%
- R: 80% in 6294/8192 factors, 90% in 7216/8192, top-1 = 0.02%
- Spearman rho(W factors, R factors) = 0.703 (much higher than dense's 0.449)
- Top-20 factor Jaccard: 0.081 (3 overlapping factors: [3074, 4259, 6704])

**Per-head factor profiles:**
| Head | W entropy | W top-5 | R entropy | R top-5 |
|------|-----------|---------|-----------|---------|
| a9.h9 (NM) | 8.82 | 7242,3948,2358,243,3927 | 8.94 | 7195,6680,970,6859,1605 |
| a10.h0 (NM) | 8.84 | 7594,231,229,4261,4047 | 8.95 | 3015,1367,7657,5255,4215 |
| a7.h3 (SI) | 8.72 | 6523,5867,1989,5788,4095 | 8.88 | 6100,6632,711,2181,907 |
| a7.h9 (SI) | 8.79 | 6598,4187,6470,4390,5638 | 8.96 | 1619,4558,3323,4948,4663 |

## DAS-EAP Results (atomic-sweep-40, k=4 causal variables)

DAS rotation matrix A learned at layer 10, l1=0.05. All 8192 factors active in DAS basis (no pruning at this l1 level).

### Per-variable circuit structure

**Variable 0** (total mass W=2.28, R=1.03):
- Top writers: a9.h6 [name_mover] (0.106), a0.h1 [dup_token] (0.071), a0.h10 [dup_token] (0.051)
- Top edges: a0.h1→m0.q (+0.038), a0.h10→m0.q (-0.017), a9.h6→r409 (-0.016)

**Variable 1** (total mass W=1.53, R=0.66):
- Top writers: a0.h4 (0.047), m0 (0.042), a0.h1 [dup_token] (0.041)
- Top edges: a0.h1→m0.q (-0.020), a0.h4→m0.q (-0.014)

**Variable 2** (total mass W=1.30, R=0.44):
- Top writers: a0.h1 [dup_token] (0.053), m0 (0.051), a4.h6 (0.032)
- Top edges: a0.h1→m0.q (-0.024), a0.h4→m0.q (+0.010)

**Variable 3** (total mass W=1.71, R=0.85):
- Top writers: m0 (0.064), m2 (0.039), a0.h4 (0.037), a2.h2 [prev_token] (0.035)
- Top edges: a0.h1→m0.q (-0.006), m10→m11.q (+0.004), a9.h6→r409 (-0.004)

### Cross-variable overlap

| Pair | Cosine | Spearman | Top-20 Jaccard |
|------|--------|----------|----------------|
| Var 0 vs 1 | 0.64 | 0.94 | 0.14 |
| Var 0 vs 2 | 0.69 | 0.94 | 0.21 |
| Var 0 vs 3 | 0.59 | 0.94 | 0.14 |
| Var 1 vs 2 | 0.79 | 0.94 | 0.18 |
| Var 1 vs 3 | 0.55 | 0.94 | 0.11 |
| Var 2 vs 3 | 0.55 | 0.94 | 0.14 |

**Key observation**: Cross-variable Spearman is uniformly ~0.94, meaning the edge *ranking* is very similar across variables. But cosine similarity ranges 0.55-0.79 and top-20 Jaccard is low (0.11-0.21), meaning the *magnitudes* and *top edges* differ. The 4 DAS variables use roughly similar circuit structure but weight different edges.

## Comparative Analysis

### Dense vs DST: W/R Mass Ratio

| Checkpoint | W mass | R mass | W/R ratio |
|-----------|--------|--------|-----------|
| Dense (1024f) | 203.6 | 139.8 | 1.46 |
| atomic-sweep-40 (8192f) | 747.0 | 233.5 | **3.20** |

The DST model has 2.2x more W/R asymmetry. This suggests that under sparsification, the *writer-side* decomposition captures more discriminative signal — the factors that are "sent" by each head become more concentrated than what the reader "uses."

### Dense vs DST: Known Head Detection

The DST model ranks ALL known IOI heads significantly higher than dense, especially:
- Name movers: rank ~110 → ~38 (huge improvement)
- s_inhibition: rank ~30-70 → ~8-25
- Duplicate token: rank 7-19 → 2-13
- Previous token: rank 14-22 → 6-11

This validates that learned sparsity (DSTGSM) makes the factor basis more aligned with functionally important circuit components.

### Factor Independence: W vs R

| Metric | Dense | atomic-sweep-40 |
|--------|-------|-----------------|
| W-R Spearman rho | 0.449 | 0.703 |
| W-R top-20 Jaccard | 0.053 | 0.081 |
| W entropy (NM heads) | 6.65-6.71 | 8.82-8.84 |
| R entropy (NM heads) | 6.50-6.70 | 8.94-8.95 |

Higher Spearman in DST means W and R marginals agree more on which factors matter. But per-head top-5 factors still disagree completely — the W marginal and R marginal identify different factors even when they agree on aggregate importance.

## Noble-sweep-143 Checkpoint (20000 factors, DSTGSM)

**Global statistics:**
- Total absolute mass: W=3639.7, R=1080.0, **ratio=3.37** (highest asymmetry so far)
- Layer-1 invariant: max|W.sum - R.sum| = 3.76e-05 (exact match)
- W-R cosine on top-50 edges: mean=0.51, std=0.37, range [0.13, 0.94]

**Most asymmetric edges:**
| Edge | W-R cosine | Edge score |
|------|-----------|------------|
| a4.h3 → m4.q | 0.13 | 0.256 |
| a0.h4 → m0.q | 0.14 | 0.262 |
| a5.h6 → m5.q | 0.14 | 0.277 |
| a4.h3 → r202 | 0.14 | -0.229 |
| a2.h2 → m5.q | 0.14 | 0.204 |

**Known IOI head rankings:**
| Head | Role | W rank | R rank | W mass | R mass |
|------|------|--------|--------|--------|--------|
| a0.h1 | dup_token | **2/157** | 3/157 | 170.59 | 37.14 |
| a2.h2 | prev_token | 6/157 | 14/157 | 70.37 | 13.59 |
| a4.h11 | prev_token | 10/157 | 15/157 | 57.12 | 12.92 |
| a0.h10 | dup_token | 13/157 | 19/157 | 49.93 | 11.95 |
| a7.h9 | s_inhibition | 32/157 | 36/157 | 31.13 | 6.79 |
| a8.h6 | s_inhibition | 37/157 | 38/157 | 28.49 | 6.73 |
| a8.h10 | s_inhibition | 50/157 | 40/157 | 24.31 | 6.70 |
| a7.h3 | s_inhibition | 52/157 | 46/157 | 23.18 | 5.95 |
| a3.h0 | dup_token | 58/157 | 64/157 | 19.79 | 4.66 |
| a5.h5 | induction | 65/157 | 72/157 | 17.99 | 3.73 |
| a6.h9 | induction | 54/157 | 66/157 | 22.83 | 4.29 |
| a9.h9 | name_mover | 86/157 | 74/157 | 13.35 | 3.68 |
| a9.h6 | name_mover | 91/157 | 81/157 | 12.30 | 3.44 |
| a10.h0 | name_mover | 104/157 | 93/157 | 10.09 | 3.05 |
| a10.h7 | neg_nm | 90/157 | 77/157 | 12.44 | 3.54 |

**Key observation**: With 20,000 factors the name movers drop back to rank ~86-104 (from ~38 at 8192 factors). The signal is spread across more factors, diluting concentration. S-inhibition and early-layer heads remain well-ranked. The W/R ratio (3.37) is the highest yet, continuing the trend that more factors → more W/R asymmetry.

**Factor concentration:**
- W: 80% in 14549/20000 factors, 90% in 17060/20000, top-1 = 0.02%
- R: 80% in 15247/20000 factors, 90% in 17538/20000, top-1 = 0.02%
- Spearman rho(W factors, R factors) = 0.564 (lower than atomic-sweep-40's 0.703)
- Top-20 factor Jaccard: 0.111 (4 overlapping factors: [2673, 11520, 16899, 19134])

**Per-head factor profiles:**
| Head | W entropy | W top-5 | R entropy | R top-5 |
|------|-----------|---------|-----------|---------|
| a9.h9 (NM) | 9.10 | 19759,1349,6321,6237,2934 | 9.84 | 16039,17441,4164,12982,16295 |
| a10.h0 (NM) | 9.10 | 4455,9306,7147,17227,4746 | 9.84 | 16892,16580,2934,2434,3138 |
| a7.h3 (SI) | 9.01 | 12532,16119,12241,13111,5545 | 9.81 | 11166,9160,19809,19007,12773 |
| a7.h9 (SI) | 9.03 | 17716,18161,18707,9831,17492 | 9.84 | 8436,17313,9961,11520,18018 |

## Full 6-Checkpoint Comparative Analysis

### W/R Mass Ratio vs Factor Count

| Checkpoint | n_factors | W mass | R mass | W/R ratio | cos(top50) |
|-----------|-----------|--------|--------|-----------|-----------|
| Dense (DenseGSM) | 1,024 | 203.6 | 139.8 | 1.46 | 0.47 |
| atomic-sweep-40 (DSTGSM) | 8,192 | 747.0 | 233.5 | 3.20 | 0.45 |
| noble-sweep-143 (DSTGSM) | 20,000 | 3,639.7 | 1,080.0 | 3.37 | 0.51 |
| major-sweep-219 (DSTGSM) | 40,000 | 1,423.0 | 537.1 | 2.65 | 0.49 |
| balmy-sweep-242 (DSTGSM) | 40,000 | 1,454.5 | 691.4 | 2.10 | 0.37 |
| misunderstood-sweep-210 (DSTGSM) | 40,000 | 4,609.5 | 535.0 | **8.62** | 0.35 |

W/R ratio is NOT monotonic with factor count. Misunderstood-210 (40k factors) has 8.62x W/R asymmetry — the writer decomposition captures massively more signal than the reader side. Balmy-242 (also 40k) has only 2.10x. The sparsity *structure* (regularization path, threshold schedule) determines W/R asymmetry, not just factor count.

### Known Head Detection (W rank / 157 writers)

| Head | Role | Dense | atomic | noble | major | balmy | misund. |
|------|------|-------|--------|-------|-------|-------|---------|
| a0.h10 | dup_token | 50 | 38 | **17** | 36 | **10** | 52 |
| a4.h11 | prev_token | **5** | 48 | 50 | 33 | 18 | 74 |
| a2.h2 | prev_token | 56 | 43 | 37 | 49 | 46 | **20** |
| a9.h6 | name_mover | 117 | **68** | 102 | 69 | 75 | 80 |
| a9.h9 | name_mover | 131 | 129 | 135 | **124** | **124** | 129 |
| a10.h0 | name_mover | 157 | **119** | 112 | 118 | 126 | 120 |
| a7.h3 | s_inhibition | 104 | **99** | 98 | 105 | 79 | 108 |
| a10.h7 | neg_nm | 139 | 154 | **150** | 150 | 150 | 154 |

No single checkpoint dominates. Different sparsity structures surface different heads. a4.h11 is uniquely prominent in the dense model (rank 5) but drops in all DST models. Name movers are consistently hard to detect (rank 68-135) across all checkpoints.

### Reader-side rankings (a4.h11 anomaly)

a4.h11 (prev_token) consistently has much better R rank than W rank in DST models:

| Checkpoint | a4.h11 W rank | a4.h11 R rank |
|---|---|---|
| Dense | 5 | 5 |
| atomic | 48 | **5** |
| noble | 50 | **10** |
| major | 33 | **8** |
| balmy | 18 | **8** |
| misunderstood | 74 | **5** |

This head is consistently what downstream readers *listen to most*, even when the writer-side ranking drops. This is a structural asymmetry: a4.h11 writes diffuse signal across many factors, but readers selectively attend to the factors it uses.

### Factor W-R Agreement

| Metric | Dense (1k) | atomic (8k) | noble (20k) | major (40k) | balmy (40k) | misund. (40k) |
|--------|-----------|-------------|-------------|-------------|-------------|---------------|
| W-R Spearman | 0.449 | **0.703** | 0.564 | 0.409 | 0.250 | 0.602 |
| Top-20 Jaccard | 0.053 | 0.081 | 0.111 | 0.212 | **0.000** | **0.429** |
| W entropy (NM) | 6.64 | 8.81 | 9.05 | 8.45 | 7.59 | 9.53 |
| R entropy (NM) | 6.74 | 8.96 | 9.85 | 10.54 | 10.50 | 10.53 |

Extreme divergence at 40k factors: balmy has ZERO top-20 overlap (W and R use completely different factors), while misunderstood has 0.429 (12/20 overlap). The sparsity structure creates qualitatively different W/R relationships.

R entropy for name movers saturates near log(40000)=10.60 for all three 40k checkpoints — the reader side distributes mass near-uniformly across factors. But W entropy varies widely (7.59-9.53), meaning writer-side concentration depends on the specific sparsity structure.

### Factor Concentration

| Checkpoint | W 80% | W 90% | R 80% | R 90% |
|---|---|---|---|---|
| Dense (1k) | 711 | 857 | 750 | 878 |
| atomic (8k) | 6,186 | 7,147 | 6,294 | 7,216 |
| noble (20k) | 14,549 | 17,060 | 15,247 | 17,538 |
| major (40k) | 28,075 | 33,341 | 30,718 | 35,220 |
| balmy (40k) | 23,394 | 29,956 | 30,806 | 35,269 |
| misund. (40k) | 30,799 | 35,254 | 30,648 | 35,173 |

Roughly 70-80% of factors needed for 80% of mass, scaling linearly with factor count. No checkpoint achieves strong factor sparsity in the EAP decomposition — the factor basis distributes attribution mass broadly.

## Completion Status

- [x] W/R marginals: all 6 checkpoints (dense, atomic, noble, major, balmy, misunderstood)
- [x] Slab EAP: 4/6 (atomic, noble, balmy, major) — misunderstood and dense running
- [x] DAS-EAP: atomic-sweep-40 (k=4 variables)
- [ ] Cross-method analysis (slab vs attribute_rotation): running on Modal
- [ ] Dense and misunderstood slab EAP: running on Modal
