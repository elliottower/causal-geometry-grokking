# Factorized EAP Implementation Analysis

Date: 2026-06-14
Checkpoint: shared_bank_global_dense (1024 factors, dense selector, L1
lambda=30 on layers 10/11, GPT-2 small)
Comparison checkpoint: atomic-sweep-40 (8192 factors, DST selector,
attn only, GPT-2 small)

## Two Implementations Exist

### Old (ivan-paper-setup branch) — used for all atlas results

Two forward passes (corrupted, clean+backward), then one batched einsum
over all edges simultaneously:

```
delta_c: (B, P, n_fwd, n_factors) — writer factor coords via S_O/S_out
grad_h:  (B, P, n_bwd, n_factors) — reader factor grads via S_q/S_k/S_v/S_in

s_W = einsum(delta_c, F, grad_resid, "b p w i, i d, b p v d -> w v i")
s_R = einsum(activation_difference, F, grad_h, "b p w d, j d, b p v j -> w v j")
```

Output: scores_W and scores_R, each (n_fwd, n_bwd, n_factors).
No per-edge loop. LN treated as linear (same approximation as standard EAP).

### New (june-sprint-canonical branch) — Ivan's rewrite

Same two forward passes, but post-hoc per-edge loop over ~70K edges.
Per edge (u, v) computes a slab of shape (|I_u|, |J_v|) including
LayerNorm Jacobian corrections:

```
M_block = F_u_r @ F_v_r.T                          # (|I_u|, |J_v|)
tilde_C = C / sigma                                 # LN scale correction
r_u = hat_x @ F_u_r.T                              # LN centering
r_v = hat_x @ F_v_r.T
T1 = einsum('bpi,bpj->ij', A, tilde_C)
T2 = einsum('bpi,bpj->ij', A*r_u, tilde_C*r_v)    # LN centering correction
slab = M_block * T1 - T2/d_m                       # (|I_u|, |J_v|)
```

Stores writer_marginal (sum over reader dim) by default; full slab opt-in.

## Why the New One Is Slow

The active factor set |I_u|, |J_v| is determined by
`graph.attach_factorization(sparsity_threshold=0.0)` which calls
`_column_support(S)` — columns where ANY channel exceeds threshold.

With threshold=0.0 and L1/dense selectors (nothing is exactly zero),
|I_u| = |J_v| = N_factors for every node. Each slab is N_F x N_F.

For 8192 factors: 8192 x 8192 = 67M entries per slab x ~70K edges =
4.7 trillion operations in the assembly loop. Result: ~10 hours.

The code's own docstring notes this (graph.py line 282): "set higher
for soft-sparsity (L1) selectors where small entries should be treated
as zero." But the `attribute()` entry point doesn't expose
`sparsity_threshold`, so the default 0.0 is always used.

## LN Correction Is NOT Second-Order (Empirically Verified)

**UPDATE (2026-06-14)**: Prior analysis assumed the LN correction was
negligible. Empirical comparison shows it is NOT:

| Method | Spearman vs standard EAP |
|---|---|
| Old (no LN correction) | 0.560 |
| Slab (with LN correction) | 0.966 |

The old factorized implementation's batched einsum approach projects
writer activations and reader gradients through the factor bank
separately, ignoring: (a) the LN Jacobian (sigma scaling, centering
correction), and (b) the cross-factor Gram matrix F_u @ F_v.T.

The slab approach computes the correct bipartite attribution per edge,
including the LN correction and Gram matrix. The resulting scalar
scores nearly match standard EAP (rho=0.97).

The cost argument no longer applies: the batched slab is 8.5s vs the
old code's 1.9s — a 4.5x slowdown, not 100x. For CI bootstrapping
(10 runs) this is 85s vs 19s. Acceptable.

## Memory Scaling for the Old Implementation

The old code allocates dense (B, P, n_nodes, n_factors) buffers:

| n_factors | delta_c (B=32, P=128) | grad_h | Total |
|---|---|---|---|
| 1024 | 2.5 GB | 7.0 GB | 9.5 GB |
| 4096 | 9.8 GB | 27.8 GB | 37.6 GB |
| 8192 | 19.7 GB | 55.7 GB | 75.4 GB |

Fix: reduce batch size. At B=4, 8192 factors needs ~10 GB total
(fits 24 GB GPU). Or chunk the writer dimension in the einsum.
The new per-edge code avoids this by storing per-node caches
individually, but the serial loop is far slower than chunked einsums.

## Checkpoint Selector Sparsity

### L1 Dense (shared_bank_global_dense, 1024 factors)

Global selector shape: (110,592 channels, 1024 factors)
= 36,864 attn channels + 73,728 MLP channels

| Threshold | % nonzero | Avg factors/channel |
|---|---|---|
| 0 | 100.0% | 1024 |
| 0.001 | 83.1% | 851 |
| 0.01 | 72.6% | 744 |
| 0.05 | 38.5% | 394 |
| 0.10 | 17.9% | 183 |
| 0.20 | — | 46 |

Column support: all 1024/1024 factors are active at every threshold
(at least one channel uses each factor).

Per-layer breakdown (thresh=0.05):

| Layer | Attn factors/ch | MLP factors/ch |
|---|---|---|
| L0 | 372 | 342 |
| L1 | 781 | 561 |
| L2 | 652 | 464 |
| L3 | 303 | 331 |
| L4 | 200 | 592 |
| L5 | 643 | 481 |
| L6 | 349 | 338 |
| L7 | 291 | 635 |
| L8 | 665 | 528 |
| L9 | 412 | **0.1** |
| L10 | 308 | **0.5** |
| L11 | 654 | **0.5** |

L1 lambda=30 on layers 10/11 crushed late MLP selectors to near-zero
(max abs = 5.76 but mean abs = 0.00086 for L10 MLP). L9 MLP also
died (max abs = 4.08, mean abs = 0.00043) — likely collateral from
the late-layer sparsity pressure making L9 MLP unnecessary.

Attention selectors are barely affected by the L1 penalty — L10/L11
attn still has 308-654 factors per channel at thresh=0.05.

### DST (atomic-sweep-40, 8192 factors, attn only)

Global selector shape: (36,864 channels, 8192 factors)
Learnable per-channel threshold: mean = 0.10, range [0.008, 8.53]

After DST hard mask: 4.2% nonzero, avg 344 factors/channel
(tight distribution: p10=312, p90=368)

All 8192/8192 factors active in column support.

### Key Differences

| Property | L1 Dense (1024) | DST (8192) |
|---|---|---|
| Selector type | Dense (no mask) | Learnable threshold |
| Decompose | both (attn + MLP) | attn only |
| Core objective | (weight match implied) | circuit_match |
| Factors | 1024 | 8192 |
| Exact zeros | None | 95.8% |
| Active per channel | ~400 @ 0.05 | ~344 post-mask |
| Late MLP | Dead (L1 killed) | Not factorized |
| Share projections | Yes (one GSM) | Yes (one GSM) |

## Old s_W/s_R vs New Slab: Different Decompositions

The old code computes s_W and s_R via SEPARATE einsums that go through
different paths — they are NOT marginals of the same tensor:

```
s_W[u, v, i] = einsum(delta_c, F, grad_resid, "b p w i, i d, b p v d -> w v i")
  → writer in factor space, reader in d_model space

s_R[u, v, j] = einsum(activation_difference, F, grad_h, "b p w d, j d, b p v j -> w v j")
  → writer in d_model space, reader in factor space
```

Each marginal keeps ONE side in factor space and the other in d_model
space, losing the cross-factor structure. They sum to the same scalar
(Layer-1 invariant, verified to ~1e-6) but differ per-factor:

| Metric | Value |
|---|---|
| Max element-wise diff | 0.031 |
| Mean element-wise diff | 5e-6 |
| Top-100 edges W-R cosine | mean=0.46, range [0.16, 0.91] |

The new code computes the FULL bipartite slab per edge:
```
slab[i, j] = M_block[i,j] * T1[i,j] - T2[i,j]/d_m
```
where M_block = F_u @ F_v.T is the cross-factor Gram matrix. This is
the exact factor-by-factor attribution with both sides in factor space
simultaneously. The writer marginal from the slab (slab.sum(dim=1)) is
NOT the same as the old s_W, because the slab properly accounts for
cross-factor correlations <F_i, F_j>.

The W-R cosine of 0.46 in the old code is NOT because the old code
captures two different valid perspectives — it's because each marginal
is a different approximation that ignores the cross-factor Gram matrix
on the opposite side. If factors were orthogonal (<F_i, F_j> = delta_ij),
s_W would equal s_R. They're not orthogonal (factor contrastive loss
was not used on this checkpoint), so the cross terms matter and the two
approximations diverge.

This means Ivan's slab approach is theoretically more correct. The
problem is purely computational: each slab is (|I_u|, |J_v|) per
edge, and with dense selectors |I_u| = |J_v| = N_F.

### What the new code stores

Only the writer marginal: `writer_marginal[i] = sum_j |slab[i, j]|`.
The reader marginal (slab.sum(dim=0)) is not computed — adding it
would be one line (`reader_marginal = slab.abs().sum(dim=0)`).

The full slab per edge is opt-in via `compute_factor_slabs=True`.
At 1024 factors this is (157, 445) edges x (1024, 1024) = ~130 GB.
Not feasible for 8192 factors.

### What the atlas analyses used

All atlas results used both s_W and s_R from the old code:
- `factor_mass = |scores_W| + |scores_R|` (combined importance)
- `combined = |sw| + |sr|` (edge profiles for PCA, NMF, etc.)
- Reader selectivity: same-reader edge profiles using `combined`
- W-R asymmetry: geodesic between |sw[u,v]| and |sr[u,v]| per edge
  — 54.8° average (past halfway to orthogonal)

The W-R asymmetry finding may partially reflect the different
approximation errors of the two marginals rather than a true
reader/writer asymmetry. The correct way to test reader selectivity
would be: compute slab for each edge, take both marginals from the
SAME slab, and compare. If the asymmetry persists, it's real.

The saved files at `artifacts/scores/shared_bank_global_dense/factor/`
contain: `scores` (157, 445), `scores_W` (157, 445, 1024),
`scores_R` (157, 445, 1024).

## Batched Slab Optimization

The per-edge slab is slow because it loops over ~70K edges serially in
Python. But all edges into the same reader share C, sigma, hat_x, and
F_v. Grouping by reader (445 groups) and batching writers per group
replaces 70K serial iterations with 445 batched matmul calls.

**Key insight**: for the shared-bank case, M_block = F @ F.T - (f1 f1^T)/d_m
is the SAME (N_F, N_F) matrix for every factorized edge. Precompute once.

**Memory-efficient marginals** (never materialize the W x N_F x N_F slab):

Writer marginal per reader group (W writers, each with A of (B,P,N_F)):
```
M_tilde_C = tilde_C @ M_block.T          # (B, P, N_F) — precomputed once
tC_r_sum = (tilde_C * r).sum(dim=-1)     # (B, P)

wm[w,i] = einsum('wbpi,bpi->wi', A_stack, M_tilde_C)
         - einsum('wbpi,bp->wi', A_stack*r, tC_r_sum) / d_m
```

Reader marginal:
```
M_A = A_stack @ M_block                   # (W, B, P, N_F)
A_r_sum = (A_stack * r).sum(dim=-1)       # (W, B, P)

rm[w,j] = einsum('wbpj,bpj->wj', M_A, tilde_C)
         - einsum('wbp,bpj->wj', A_r_sum, tilde_C*r) / d_m
```

**Memory scaling**: Peak per reader group = ~3x A_stack size. For the
logits group (largest, W=157):

| n_factors | B | Peak mem | Fits 24 GB? |
|---|---|---|---|
| 1024 | 10 | ~384 MB | yes |
| 8192 | 4 | ~1.2 GB | yes |
| 8192 | 32 | ~9.6 GB | yes |

**Input/logits edges** (F=I, <1% of total): handled by per-edge
fallback within the same function. No performance impact.

**Comparison script**: `compare_eap_implementations.py` runs all four
implementations (old einsum, per-edge slab, batched slab, standard EAP)
and verifies they agree. Needs GPU.

## Empirical Results (20 IOI examples, A10G GPU)

### Timing

| Method | Time | Notes |
|---|---|---|
| Old (batched einsum) | 1.9s | Fastest, but incorrect (see below) |
| Standard EAP | 6.6s | Non-factorized ground truth |
| Batched slab (+LN) | 8.5s | Best factorized implementation |
| Per-edge slab (+LN) | 40.1s | Superseded by batched slab |

### Accuracy vs Standard EAP Ground Truth

| Method | Spearman | Top-20 Jaccard | Max abs diff |
|---|---|---|---|
| Old (einsum, no LN) | 0.560 | 0.429 | 0.182 |
| Per-edge slab (+LN) | 0.966 | 0.905 | 0.020 |
| Batched slab (+LN) | 0.966 | 0.905 | 0.020 |

**The old implementation is significantly WRONG.** Spearman 0.56 and
top-20 Jaccard 0.43 vs standard EAP. The LN correction is NOT
negligible — it accounts for the gap from 0.56 to 0.97.

### Batched Slab vs Per-Edge Slab

Numerically identical: max abs diff 3.73e-08, writer marginal cosine
1.000000 on top-100 edges. The batched slab is 4.7x faster at 1024
factors — speedup grows with N_F since the serial loop is O(n_edges)
while batched groups are O(n_reader_groups).

### W-R Asymmetry IS Real

Slab W-R cosine (both marginals from SAME correct decomposition):
mean=0.290, min=-0.087 on top-200 edges. The asymmetry is genuine,
not an artifact of the old code's separate approximations.

Old code's W-R cosine (separate einsums): mean=0.350.

## Implications

1. **Atlas results used the wrong factorized EAP.** The old
   implementation has Spearman 0.56 vs standard EAP — factor-level
   rankings are substantially different from the true attribution.
   Qualitative findings (three-stream structure, universal mixture)
   likely survive because they are based on relative rank structure
   and factor co-occurrence, but quantitative edge rankings and
   per-factor mass values are biased. Any analysis that depends on
   precise edge ordering (top-k, bottleneck identification) should
   be re-run with the batched slab implementation.

2. **Two viable implementations going forward**:
   - Batched slab: correct, fast (8.5s vs 40s per-edge), memory-
     efficient. Should be the default for all future work.
   - Per-edge slab: same math, only useful if computing full
     (I_u, J_v) slabs per edge (opt-in via compute_factor_slabs=True).

3. **For 8192+ factors**, the batched slab with B=4 fits 24 GB.

4. **If using the new code on L1/dense checkpoints**, set
   sparsity_threshold > 0 in attach_factorization(). At threshold=0.1,
   |I_u| drops from 1024 to ~183 per channel, giving ~30x speedup.
   The attribute() function needs to expose this parameter.

5. **The L1 checkpoint's MLP story is interesting**: L1 on layers 10/11
   killed MLP selectors but not attention. L9 MLP also collapsed.
   The model decided late MLPs don't need factor-space routing —
   consistent with the atlas finding that sign flips are MLP-driven
   (the computation works through the remaining non-zero entries or
   through the raw d_model path).
