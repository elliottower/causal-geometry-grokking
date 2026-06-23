# Grassmannian Atlas Paper — Master Status Document

**Paper**: "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian"
**Latest draft**: `GRASSMANNIAN_PAPER_DRAFT_V8.tex`
**Repo**: transformer-factorization-circuits/experiments/batch6_atlas/

---

## Paper Versions

| Version | Lines | Key additions |
|---------|-------|---------------|
| V1-V6 | — | Iterative drafts (not archived individually) |
| V7 | 306 | Complete draft: 14-op atlas, 3-class partition, stochastic grokking, equivariance diagnostics. Heavy topology section (holonomy, Schubert calculus). Perplexity-generated. |
| **V8** | ~450 | **Current**. Adds: structured VAE for nonlinear causal variables, hard-example IIA, linear DAS = 0.0 on grokked modular addition (new result), CPCA-init DAS connection. Removes: topology bloat (holonomy, Schubert, tropical Grassmannian). Adds Siddharth + Doumas as authors. |

---

## Core Claims & Evidence Status

### Claim 1: Operations partition into Always / Stochastic / Never Grassmannian
**Status**: COMPLETE — Table 1 in V7/V8
**Evidence**: DAS k-sweep + equivariance for all 14 operations
**Strength**: Strong (clean partition, no edge cases)
**Data files**: `experiments/results/grokking_*.jsonl` (one per operation)

### Claim 2: Grokking governs the partition
**Status**: COMPLETE for 1-3 seeds, INCOMPLETE for 10-seed stability
**Evidence**: Stochastic class (Power, Comp. addition) shows same op / different seeds / opposite outcomes
**Gap**: Paper promises "10 seeds per stochastic operation" — not yet run
**Data files**: `grokking_power.jsonl`, `grokking_sum_of_squares.jsonl`, etc.

### Claim 3: Linear DAS returns IIA = 0.0 on grokked modular addition
**Status**: COMPLETE
**Evidence**: `grokking_das_v1.jsonl` — 50 checkpoints, k=16, IIA = 0.0 at every single one including fully grokked (test loss < 1e-7)
**Strength**: Very strong (not just low IIA — literally zero)

### Claim 4: Memorization produces high IIA (squaring/cubing ≥ 0.86 at k=2)
**Status**: COMPLETE
**Evidence**: Atlas table, k-sweep curves
**Strength**: Strong (cleanest IIA ≠ validity demonstration)

### Claim 5: Equivariance distinguishes genuine structure from memorization
**Status**: COMPLETE
**Evidence**: >95% for always-Grassmannian, <70% for never-Grassmannian, <1% for random controls
**Strength**: Strong

### Claim 6: Structured VAE recovers nonlinear causal variables
**Status**: NOT YET RUN — V8 has TODO placeholders
**Evidence needed**:
- VAE IIA > 0 on grokked modular addition (where linear DAS = 0.0)
- VAE equivariance > DAS equivariance for never-Grassmannian ops
- Linearized overlap comparison
**Script**: `06_21_2026_UPDATE/structured_vae_atlas.py` (TO WRITE)
**GPU estimate**: ~14 hours on A100

### Claim 7: Hard-example IIA sharpens the diagnostic
**Status**: NOT YET RUN
**Evidence needed**: 
- Hard IIA comparison (DAS vs VAE) on grokked operations
- Hard IIA on non-grokked operations (showing memorization still works)
**Connection to**: CPCA-init DAS methodology (outline_B_v6_split_das.tex, 91.2% vs 73.5%)

---

## Existing Experiment Results

### Results Files (in experiments/results/)
```
grokking_abs_diff.jsonl         — abs difference operation
grokking_bitwise_xor.jsonl      — XOR operation (always-Grassmannian)
grokking_cubing.jsonl           — cubing (never-Grassmannian)
grokking_das_v1.jsonl           — DAS emergence: 50 checkpoints, IIA=0.0 always
grokking_division.jsonl         — division (always-Grassmannian)
grokking_floor_div.jsonl        — floor division
grokking_gcd.jsonl              — GCD operation
grokking_max_ab.jsonl           — max(a,b) (always-Grassmannian)
grokking_min_ab.jsonl           — min(a,b)
grokking_multiplication.jsonl   — multiplication (always-Grassmannian)
grokking_nonlinear_dsi_v1.jsonl — nonlinear DSI first attempt
grokking_nonlinear_dsi_v2.jsonl — nonlinear DSI at k=16 (all IIA=1.0, k too high)
grokking_power.jsonl            — power (stochastic)
grokking_shifted_mult.jsonl     — shifted multiplication
grokking_squaring.jsonl         — squaring (never-Grassmannian)
grokking_sum_of_squares.jsonl   — sum of squares (always-Grassmannian)
grokking_torus_geometry.jsonl   — torus geometry analysis
grokking_torus_geometry_v3.jsonl
grokking_torus_v4_controls.jsonl
grokking_torus_v5_controls.jsonl
```

### Experiment Scripts (in batch6_atlas/)
```
Core atlas:
  grokking_das_emergence.py     — DAS IIA vs training epoch
  grokking_nonlinear_dsi.py     — nonlinear featurizer ladder (quad/MLP)
  grokking_torus_geometry.py    — torus visualization
  grokking_nonlinear_hunt.py    — hunting for nonlinear structure

Atlas analysis:
  cross_model_dsi.py            — DAS across GPT-2/Qwen/etc
  cross_task_overlap.py         — subspace overlap across tasks
  embedding_dsi.py              — embedding-level DAS
  causal_arithmetic.py          — causal intervention on arithmetic
  circuit_sufficiency.py        — circuit sufficiency tests
  construct_coherence.py        — construct coherence analysis
  composition_bottleneck.py     — compositional bottleneck
  decision_boundary.py          — decision boundary analysis

06_13_2026/ (previous dated update):
  analyze_edge_strata.py
  anti_grokking_contrastive.py
  attn_only_edge_analysis.py
  combined_weight_analysis.py
  das_svd_stratum_disagreement.py
  depth_sweep_grokking.py
  factor_composition_scores.py
  fourier_stratum_connection.py
  gpt2_circuit_stratum.py
  gpt2_finetune_grokking.py
  (+ more)
```

### Figures (in batch6_atlas/)
```
fig1_equivariance_bars.png          — main equivariance result
fig2c_iia_vs_equiv_grokked.png     — IIA vs equivariance scatter
fig8c_loss_vs_equiv_grokked.png    — loss vs equivariance
fig10-12 loss_vs_equiv variants
fig13_polynomial_degree_ladder.png
fig14_k_sweep.png
fig15_grokked_vs_not.png
fig15b_grokked_boxplot.png
fig16_equiv_vs_random.png
fig19_stochastic_grokking.png
(+ PDF versions of all)
```

---

## Connection to Other Papers

### Elliot's Factorized DAS Paper (outline_B_v6_split_das.tex)
**Title**: "Causal Variables Live on the Factor Manifold"
**Connection**: CPCA-init DAS (91.2% hard IIA), hard-example methodology, background/modulation decomposition
**This paper cites that**: CPCA-init methodology, hard-example IIA definition
**That paper cites this**: Grassmannian diagnostics, equivariance as validation

### Ivan's Factorization Paper (outline_B_v6_split_main.tex)
**Title**: TBD — factorized transformer architecture
**Connection**: Factor bank decomposition enables per-factor attribution
**This paper does NOT depend on**: factor bank (uses standard grokking models, not factorized)

### Neuro-Causal-Geometry Paper (neuro-causal-geometry/paper/main_v4.tex)
**Title**: "Do Different Metrics Tell the Same Story? Geometric Dissociation in Neural Population Codes"
**Connection**: Grassmannian distance metrics, exp57_structured_vae.py is the VAE architecture we adapt
**Shared method**: Structured VAE for disentangling causal from nuisance factors

---

## Validity Assessment

Using the 5-tier validity framework:

| Claim | Tier | Reason |
|-------|------|--------|
| 3-class partition | Mechanistically Supported | Multiple seeds, equivariance controls, k-sweep convergence |
| Grokking ↔ Grassmannian | Causally Suggestive | Stochastic class proves conditional, but only 2-3 seeds |
| IIA ≠ validity | Mechanistically Supported | Memorization artifacts clearly demonstrated |
| Equivariance diagnostic | Mechanistically Supported | Random controls, consistent across operations |
| Linear DAS = 0.0 | Validated | 50 checkpoints, k=16, no ambiguity |
| Structured VAE recovery | **NOT YET TESTED** | Experiments to run |
| 10-seed stability | **NOT YET RUN** | Promised in V7, still pending |

**Overall paper tier**: Currently Causally Suggestive (limited by seed count and missing VAE experiments). With VAE results + 10-seed stability → Mechanistically Supported.

---

## TODO Priority List

1. **Write structured_vae_atlas.py** — adapt exp57 from neuro-causal-geometry
2. **Run structured VAE on all 14 operations** — the headline new result
3. **Run nonlinear DSI at k=2** (not k=16) — quick sanity check
4. **Run hard-example DAS** — connects to CPCA-init methodology
5. **Run 10-seed stability for Power + Comp. addition** — long but important
6. **Generate figures**: VAE latent space visualizations for grokked vs not
7. **Fill in V8 TODO sections** with actual numbers

---

## Perplexity Queries (for research clarification)

See `06_21_2026_UPDATE/PERPLEXITY_QUERIES.md`
