# Stratum Theory Audit — Evidence Assessment

Date: 2026-06-13
Scope: All batch6_atlas experiments (365 runs, 248 success, 111 error, 6 timeout)

## 1. The Theory

Transformer weight matrices (OV = W_V @ W_O, QK = W_Q @ W_K^T, FF = W_in @ W_out) can be classified into discrete **strata** based on their singular value distribution:

| Stratum | Criterion | Interpretation |
|---------|-----------|----------------|
| M_1 | gap_ratio > 3.0 AND k_90 <= 2 | Single dominant direction |
| M_k | top-8 cumvar > 0.85 AND k_90 <= 8 | Low-rank subspace (k dimensions) |
| M_Sigma | erank < 0.3 * dim | Curved manifold, moderate rank |
| M_D | none of the above | Distributed, full-rank |

The central claim is that these strata are **meaningful**: they track what algorithm the model has learned, change during grokking, and connect to Fourier structure in modular arithmetic models.

---

## 2. Experimental Evidence Inventory

### 2.1 Grokked Toy Models (1-layer, d_model=128, p=113)

**Source**: fourier_stratum_connection, depth_sweep, factorized_grokking_geometry (105 success)

**Fourier-stratum connection (multiplication, 1 model)**:
- All 4 attention heads: OV = M_k (erank 2.1-6.2), QK = M_1
- FF circuit: M_k (erank 5.5)
- MLP neurons: 0/512 Fourier-selective (0%)
- Embedding: Fourier PR normalized = 3.93 (nearly flat spectrum)
- Note: model de-grokked during training (test_loss 0.007 -> 0.12). Results may reflect partially-grokked state.

**Depth sweep (multiplication, depth=8 only completed)**:
- 8-layer model after 20K epochs: did NOT grok (test_loss = 2.06)
- Nevertheless: OV = 24x M_1 + 8x M_k, QK = 29x M_1 + 3x M_k, avg OV erank = 1.31
- All heads have near-rank-1 structure despite not solving the task
- Implication: M_1 classification says "low rank" but not "correct algorithm"

**Factorized grokking geometry (78 configs across 9 operations)**:
- Grokked ops (multiplication, division, addition, composite_addition): JumpReLU selectors achieve erank 40-60 with MSE ~1e-4
- Non-grokked ops (squaring, cubing, polynomial, max_ab): higher effective rank, worse MSE
- JumpReLU-tanh dominates the MSE-vs-rank Pareto frontier over L1 (10-100x better at matched erank)

### 2.2 Pretrained GPT-2 Small (12-layer, d_model=768)

**Source**: gpt2_circuit_stratum (1 run, all 144 heads + 12 MLPs + known circuits)

**All 144 heads**:
- OV strata: 143 M_Sigma, 1 M_1 (L11H8, erank=1.18, gap=11.49 — NOT a known circuit head)
- QK strata: 142 M_Sigma, 2 M_k (L1H3 and L1H10 — NOT known circuit heads)
- Zero circuit heads have clean strata

**Circuit vs non-circuit heads** (40 circuit, 104 non-circuit):

| Metric | Circuit heads | Non-circuit heads | Direction |
|--------|--------------|-------------------|-----------|
| OV erank (mean) | 54.6 | 49.5 | Circuit heads MORE distributed |
| OV erank (std) | 9.5 | 12.8 | Non-circuit more variable |
| OV gap ratio (mean) | 1.24 | 1.39 | Non-circuit has MORE spectral gap |
| OV gap ratio (max) | 3.63 | 11.49 | Only M_1 head is non-circuit |
| QK erank (mean) | 53.0 | 46.9 | Circuit heads MORE distributed |

**Per-role OV effective rank**:
- IOI name movers (L9H6, L9H9, L10H0): erank = 60.1-60.4, gap = 1.02 (flat spectrum)
- IOI s_inhibition (L7H3, L7H9, L8H6, L8H10): erank = 57.0-59.9 (flat)
- Induction heads (L5H1, L5H5, L6H9, L7H2, L7H10): erank = 51.5-61.0 (flat)
- Greater-than attention: erank = 51.5-61.0 (flat)
- SVA top-12: erank = 19.6-60.1 (two outliers at 19.6 and 23.1 are closest to structured)

**Most structured circuit heads** (lowest OV erank):
- L0H8 (SVA): erank=19.6, gap=1.84, k90=32 — still M_Sigma, not M_k (k90 too high)
- L11H4 (SVA): erank=23.1, gap=2.06, k90=42 — M_Sigma
- L10H10 (IOI backup): erank=32.8, gap=3.63, k90=48 — M_Sigma (gap > 3 but k90 = 48)

**MLP strata (FF = W_in @ W_out)**:

| Layer | Stratum | Effective rank |
|-------|---------|---------------|
| L0 | M_D | 284.6 |
| L1 | M_Sigma | 12.3 |
| L2 | M_Sigma | 9.3 |
| L3 | M_Sigma | 25.0 |
| L4-L8 | M_Sigma | 100-229 |
| L9 | M_D | 250.3 |
| L10 | M_Sigma | 148.1 |
| L11 | M_Sigma | 33.2 |

Early layers (L1-L2) have notably lower MLP erank. This is interesting but doesn't connect to known circuits.

### 2.3 GPT-2 Fine-Tuning on Modular Arithmetic

**Source**: gpt2_finetune_grokking (4 conditions, 5K epochs each)

| Init | Regime | Test loss | Test acc | Grokked |
|------|--------|-----------|----------|---------|
| pretrained | grok (wd=1.0) | 6.56 | 0.7% | No |
| pretrained | memorize (wd=0) | 9.01 | 0.3% | No |
| random | grok (wd=1.0) | 5.17 | 29.2% | No |
| random | memorize (wd=0) | 5.13 | 26.4% | No |

No strata summary was saved (field is None). 5K epochs insufficient for any condition. Random init outperforms pretrained 40:1 on accuracy. Pretrained GPT-2 weights actively interfere with modular arithmetic learning.

### 2.4 Un-Training Grokked Models

**Source**: un_training_grokking (3 operations, regime=wrong_operation)

| Operation | Still grokked? | OV erank | SVD PR | Interpretation |
|-----------|---------------|----------|--------|----------------|
| multiplication | No (loss=20.4) | 9.19 | 4.35 | Structure partially degraded |
| composite_addition | Yes (loss=9e-6) | 5.46 | 4.71 | Structure preserved |
| cubing | No (loss=9.19) | 6.23 | 5.69 | Higher PR = more distributed |

Composite_addition retains its grokked structure even when trained on a different operation's labels. The un-training didn't erase the algorithm. Multiplication loses grokking and its erank increases. But the changes are modest (erank 5-9 range throughout), not the dramatic M_1 -> M_D transition the theory would predict.

### 2.5 Other Experiments

**Anti-grokking contrastive** (polynomial, low wd=0.01):
- Not grokked, test_loss=87.5, erank=21.6, PR=6.26
- Higher erank than grokked models, consistent with theory

**G-score stratum mapping** (floor_div, known never-grok):
- gscore=0.075, diagonal_fraction=8.4%
- Classified as expected non-grokking

**Method agreement** (squaring, 60K epochs):
- Did not grok, test_loss=7.24
- Nonlinear/linear ratio = 1.0 (no separation between kNN and linear probe)

**Stratum trajectory** (squaring):
- Stayed stable at M_k throughout 60K epochs (erank=7.38, k90=8)
- Grassmannian distance to final state: 0.0013

---

## 3. Claim-by-Claim Assessment

### Claim 1: "Strata classify the type of algorithm a model has learned"

**Verdict: PARTIALLY SUPPORTED in toy models, NOT SUPPORTED in pretrained models**

In grokked 1-layer models (p=113 modular arithmetic):
- All 4 OV heads are M_k, all 4 QK heads are M_1 (Fourier experiment)
- FF circuit is M_k
- This is consistent: the model learns a Fourier-based algorithm that operates in a low-dimensional subspace

In pretrained GPT-2:
- 143/144 heads are M_Sigma regardless of function
- Known circuit heads (name movers, induction heads, s-inhibition) are indistinguishable from non-circuit heads by stratum
- The classification is "M_Sigma" everywhere — it has no discriminative power

**The gap**: The strata thresholds (gap > 3 for M_1, cumvar > 0.85 for M_k) were calibrated on 128-dimensional grokking models. GPT-2's d_head = 64, d_model = 768. The classification criteria may need dimension-dependent normalization to be meaningful at larger scale.

### Claim 2: "Strata change during grokking (phase transition)"

**Verdict: PARTIALLY SUPPORTED**

- Un-training multiplication shows erank increase (5.46 -> 9.19) when grokking is lost
- Composite_addition retains low erank even after un-training (structure is robust)
- Stratum trajectory on squaring (never-grok) shows stable M_k throughout — no transition because no grokking
- Missing: no successful stratum trajectory on a grokking operation (would show M_D -> M_k transition)

**Concern**: The depth=8 model has 24 M_1 and 8 M_k heads but DOESN'T grok (test_loss=2.06). Low rank alone does not imply correct algorithm. The model can have clean spectral structure without solving the task.

### Claim 3: "Fourier modes correspond to strata"

**Verdict: PARTIALLY SUPPORTED with important caveats**

- Attention heads: OV is M_k, QK is M_1, consistent with Fourier frequency selectivity
- MLP: 0/512 neurons are Fourier-selective, MLP is NOT frequency-decomposed
- Embedding: nearly flat Fourier spectrum (PR_normalized = 3.93)
- Model de-grokked during this experiment (test_loss went back up to 0.12)

The Fourier structure lives in **attention**, not in MLPs or embeddings. The stratum classification correctly identifies attention circuits as low-rank, but the Fourier interpretation only applies to the QK pathway (M_1 = single frequency pair), not the OV pathway (M_k = broader subspace).

### Claim 4: "The equivariance fraction is the primary discriminator"

**Verdict: STRONGLY SUPPORTED (from geometry_wild batch)**

From 22 operations:
- 7 operations are "Grassmannian" (>99% equivariance): multiplication, division, subtraction, composite_addition, bitwise_xOR, cubic_sum, Dyck-1 depth
- 6 operations are "Near-Grassmannian" (94-99%): shifted_mult, sum_of_squares, max, min, modular_distance, GCD
- 3 "Partial" (49-58%): digit_addition, floor_div, abs_diff
- 2 "Memorization artifact" (27-46%): squaring, cubing
- 3 "No structure" (<4%): power, polynomial, affine

Equivariance fraction cleanly separates the three-class partition (always/stochastic/never grok). No other metric — including strata — provides this separation.

### Claim 5: "Subspace identity is gauge freedom; equivariance within is what matters"

**Verdict: STRONGLY SUPPORTED (from geometry_wild experiments 10-12)**

- 10 seeds of multiplication all grok (IIA 0.94-1.00) but use completely different subspaces (overlap ~0.008)
- Basin of attraction is enormous: perturbing DAS subspace by 1+ radians barely drops IIA
- Spectral graph partition: all operations' subspaces equidistant (d_Gr in [1.93, 2.19])
- Convergence on Grassmannian is non-monotonic

The model doesn't pick a canonical subspace. There are exponentially many equivalent solutions. The strata classification of a specific subspace is measuring something that the model treats as arbitrary.

---

## 4. What the GPT-2 M_Sigma Result Actually Means

### The direct interpretation

GPT-2's weight matrices use the full available rank. OV circuits at (d_head=64) have effective rank 50-60 — they use ~80-95% of the available dimensions. This is not "distributed" in the sense of "no structure" — it's "the structure uses most of the space."

### Why this shouldn't be surprising

1. **GPT-2 solves hundreds of tasks simultaneously.** A single head that participates in IOI, induction, and greater-than needs different principal components for each task. Its OV matrix is the superposition of many rank-1 or rank-k contributions. The result is high effective rank even if each individual task-circuit is low-rank.

2. **Superposition predicts M_Sigma.** The features-in-superposition hypothesis (Elhage et al. 2022) says models pack more features than dimensions. If each feature contributes a roughly equal-magnitude direction, the singular value spectrum is flat, and erank approaches dim. M_Sigma is the expected outcome of superposition.

3. **Grokking models don't superpose.** A 1-layer model trained on a single modular arithmetic task has no reason to superpose. It can dedicate all capacity to one algorithm. That's why grokked models have clean strata — they're solving exactly one problem in a 128-dimensional space, and the solution only needs 2-8 dimensions.

### Three possible responses

**Option A: "Strata are a grokking-specific phenomenon."** Accept that the framework describes toy models learning single algorithms, and that pretrained models live in M_Sigma due to superposition. The theory is correct but narrow.

**Option B: "Strata need task-conditioned measurement."** Instead of classifying a head's OV matrix globally, project it into a task-specific subspace first (e.g., the DAS subspace for IOI), then measure the stratum within that subspace. The hypothesis: a head that's M_Sigma globally might be M_1 when projected into the IOI-relevant directions.

**Option C: "Strata are the wrong unit of analysis."** The meaningful structure isn't the spectral profile of individual weight matrices — it's the equivariance of the learned function within whatever subspace is active. Equivariance fraction already captures this without needing SVD at all.

### Recommendation

Option B is testable with existing data. For each known circuit head, project OV into the task-specific DAS subspace (from the atlas's DAS results), compute the SVD of the projected matrix, and classify the stratum. If IOI name movers are M_1 in the IOI subspace, that would rehabilitate the theory at the cost of requiring task-specific measurement. This is a CPU computation on existing artifacts.

Option C is what the data currently supports. The strata classification adds mechanical detail to what equivariance fraction already tells you.

---

## 5. Open Questions (Ranked by Testability)

### Testable now (CPU, existing data)

1. **Task-conditioned strata**: Project each head's OV through the DAS rotation for IOI/SVA/GT/GP. Does the projected OV have clean strata?

2. **Selector-count vs strata**: In the 78 factorized grokking configs, does the number of active JumpReLU-gated factors per projection match the stratum's implied rank? (M_1 -> 1 factor, M_k -> k factors)

3. **Dimension-dependent thresholds**: Are the current thresholds (gap > 3, cumvar > 0.85, erank < 0.3*dim) appropriate for d_head=64? What if we normalize by dimension or use a random-matrix baseline (Marchenko-Pastur)?

### Testable with ~1h GPU

4. **Per-task stratum at grokking transition**: Run stratum_trajectory on multiplication (not squaring) to observe the actual M_D -> M_k transition during grokking.

5. **Depth=2 and depth=4 completion**: The depth sweep only has depth=8 results. Depth=2 and 4 should grok — do they develop different strata?

### Testable with ~10h GPU

6. **Multi-operation strata comparison**: Classify strata across all 22 operations from V4 table at their grokked checkpoints. Do the five equivariance tiers map onto distinct strata?

### Would require new framework

7. **Superposition-aware strata**: Decompose GPT-2 heads into task-specific contributions (via activation patching or steering), then classify each contribution's stratum separately. This would test whether M_Sigma heads are superpositions of M_1 components.

---

## 6. Summary Scorecard

| Claim | Toy models | Pretrained GPT-2 | Overall |
|-------|-----------|-------------------|---------|
| Strata classify algorithm type | Supported | Not supported | Narrow |
| Strata change at grokking | Partially supported | N/A | Incomplete |
| Fourier modes match strata | Supported (attention only) | N/A | Narrow |
| Equivariance is primary discriminator | Strongly supported | Not tested | Strong |
| Subspace identity is gauge freedom | Strongly supported | Not tested | Strong |
| Strata generalize to pretrained models | N/A | **Contradicted** | Failed |

**Bottom line**: The stratum framework is a correct description of grokking models that solve single algorithmic tasks. It does not generalize to pretrained models in its current form. The theory's strongest result — the three-class partition by equivariance fraction — doesn't need strata at all. The most promising rescue is task-conditioned strata measurement (Option B above), which is testable now on existing data.
