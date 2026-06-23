# EAP TODO

## 1. Tell Ivan about sparsity_threshold

`attribute()` doesn't expose `sparsity_threshold` to
`graph.attach_factorization()`. With L1/dense selectors, threshold=0
means |I_u|=|J_v|=N_factors and the per-edge loop takes 10 hours.
Fix: expose the parameter, default to 0.01 for L1 selectors.

## 2. Add reader marginal to new code (one line)

The new code computes the full slab[i,j] per edge but only stores
writer_marginal = slab.abs().sum(dim=1). Adding the reader marginal
is one line: `reader_marginal = slab.abs().sum(dim=0)`.

The slab-derived marginals are more correct than the old s_W/s_R
because they properly account for the cross-factor Gram matrix
<F_i, F_j>. The old s_W and s_R are different approximations that
each keep one side in d_model space — their disagreement (cosine
0.46 on top edges) partially reflects approximation error, not just
true reader/writer asymmetry.

Having both slab marginals would let us re-test the reader selectivity
finding with the correct decomposition.

## 3. Benchmark old vs new factorized EAP — SCRIPT READY

Script: `compare_eap_implementations.py`. Runs four implementations:
1. Old (batched einsum, no LN)
2. Per-edge slab (LN Jacobian correction, serial ~70K edges)
3. **Batched slab** (same math as per-edge, reader-grouped batched
   einsums, memory-efficient — never materializes W x N_F x N_F)
4. Standard (non-factorized) EAP as scalar ground truth

Compares:
- Scalar Spearman + Jaccard top-20 between all four
- Writer/reader marginal cosine (old s_W/s_R vs slab marginals)
- W-R asymmetry from same slab vs separate einsums
- Speedup: batched slab should be ~100x faster than per-edge

**Needs GPU to run.** 20 IOI examples, 1024-factor checkpoint.
Expected result: all scalar scores near-identical (rho > 0.99 vs
standard EAP), confirming LN correction is negligible.

The batched slab uses memory-efficient marginal formulas:
  wm[w,i] = einsum(A, M_tilde_C) - einsum(A*r, tC_r_sum) / d_m
  rm[w,j] = einsum(M_A, tilde_C) - einsum(A_r_sum, tC_r) / d_m
Peak memory per reader group: ~3x A_stack, no N_F x N_F tensor.
Scales to 8192 factors at B=4 with ~1.2 GB peak.

## 4. Scale old implementation to 8192 factors

The old code's dense buffers (B, P, n_fwd, n_factors) blow up at 8192.
Need to either:
- Reduce batch size (B=4 fits 24 GB)
- Chunk the writer dimension in the einsum
- Or both

Test on atomic-sweep-40 (8192, DST, attn only). This would give us
factor EAP scores for the DST checkpoint, enabling direct atlas
comparison with the L1 checkpoint.

## 5. Run factor EAP on classic-sweep-133

4096 factors, JumpReLU-tanh, circuit_match, decompose=both.
No factor EAP scores exist on disk — only faithfulness evals.
Would enable atlas analysis on a third selector type.

## 6. Grokking

Run factorized EAP (and possibly DAS) on a grokking model.
Motivation: stratification-based approaches (like SAE-based circuit
discovery) struggle when superposition is absent (post-grokking the
representations are clean, no superposition to decompose). Factorized
EAP decomposes through the factor bank regardless of whether the
model uses superposition or not — should work on both.

Steps:
- Train a small modular arithmetic model to grokking
- Fit a factorization (small factor count, maybe 64-256)
- Run factorized EAP at pre-grok and post-grok checkpoints
- Compare factor-level circuit structure with known grokking circuits
  (Nanda et al. 2023 — trig + quadratic features)
- Test whether factorized EAP recovers the known circuit better than
  standard EAP / ACDC / activation patching
