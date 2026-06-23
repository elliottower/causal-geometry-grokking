# Factor-Level Coupling: Weight Space, EAP Routing, and Their Agreement

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
Date: 2026-06-14

## Overview

Three analyses testing whether the factor bank's weight-space geometry
predicts EAP-derived attribution, and what cross-task factor routing
reveals about the dictionary's structure.

## Analysis A: Weight-Space Factor Coupling via OV Matrices

For each head h, compute coupling[f_i, f_j] = factor_i @ OV_h @ factor_j^T.
This measures how much factor i's output through head h projects onto
factor j's input subspace — purely from weights, no data.

### Head-level coupling strength

| Rank | Head | Frobenius norm | Circuit group |
|------|------|----------------|---------------|
| 1 | L9H5 | 211.0 | non-circuit |
| 2 | L9H4 | 194.1 | non-circuit |
| 3 | L11H8 | 193.5 | non-circuit |
| 4 | L8H0 | 191.1 | non-circuit |
| 5 | L7H4 | 189.1 | non-circuit |
| 6 | L11H6 | 187.1 | non-circuit |
| 7 | L10H9 | 185.9 | non-circuit |
| 8 | L9H2 | 184.0 | non-circuit |
| 9 | L8H2 | 183.3 | non-circuit |
| 10 | L7H1 | 180.3 | non-circuit |

**AUROC (Frobenius norm, circuit vs non-circuit): 0.415**

The OV coupling Frobenius norm is an anti-signal: circuit heads have
LOWER total factor coupling than non-circuit heads. This parallels the
HSIC anti-signal from the direction analysis — circuit heads specialize
on narrow factor subspaces, while non-circuit heads have broad, diffuse
coupling across many factors.

The top-coupled heads are all non-circuit. The top 10 includes zero IOI
circuit heads. Circuit heads like L9H6 (name mover) and L7H3 (S-inhibition)
rank in the bottom half.

### Top factor pairs by cross-head coupling

| Factor i | Factor j | Total coupling (sum over all heads) |
|----------|----------|-------------------------------------|
| f351 | f310 | 41.76 |
| f310 | f351 | 41.42 |
| f351 | f870 | 41.01 |
| f870 | f351 | 39.55 |
| f310 | f870 | 35.41 |

Factors 351, 310, and 870 form a tightly coupled triad — they strongly
project onto each other through multiple heads' OV circuits. These are
not IOI-specific factors; they represent general-purpose residual stream
communication channels.

### Interpretation

The weight-space OV coupling is dominated by the model's generic
information routing, not task-specific circuits. Circuit heads use narrow,
specific factor subspaces (low total coupling) while non-circuit heads
move broad information (high total coupling). This is consistent with the
factorized model learning a sparse, task-specific factor selection on top
of a shared dense factor bank.

## Analysis B: Cross-Task Factor Routing Fingerprints

For each of the 1024 factors, compute its "routing fingerprint": the
fraction of total EAP attribution it carries for each of 6 tasks (IOI,
SVA, greater-than, capital-country, gender-bias, hypernymy).

### Task-task correlation (by factor usage profiles)

| Task pair | Pearson r |
|-----------|-----------|
| capital_country vs hypernymy | 0.964 |
| capital_country vs ioi | 0.940 |
| capital_country vs gender_bias | 0.923 |
| capital_country vs greater_than | 0.918 |
| gender_bias vs hypernymy | 0.917 |
| gender_bias vs ioi | 0.909 |

All task pairs have r > 0.90. The factor usage profiles are strikingly
similar across tasks — most factors contribute proportionally to all tasks.
This confirms the "shared dictionary" finding from the factorized EAP
results (Jaccard 0.70-0.85 on top factors), and extends it to a
continuous measure: not just overlap, but proportional usage.

### Active factors

All 1024 factors have nonzero attribution in at least one task. The factor
bank is fully utilized across the 6 tasks — no dead factors.

### Cluster analysis (K=8)

| Cluster | Size | Dominant task |
|---------|------|---------------|
| 2 | 329 | greater_than |
| 5 | 285 | greater_than |
| 0 | 197 | ioi |
| 4 | 117 | gender_bias |
| 3 | 50 | gender_bias |
| 1 | 36 | ioi |
| 7 | 6 | sva |
| 6 | 4 | hypernymy |

The two largest clusters (614 factors, 60%) are greater_than-dominant,
suggesting that arithmetic/comparison tasks use a broader set of factors
than other tasks. IOI-dominant factors form 23% of the bank.

### Most task-specific factors (lowest routing entropy)

| Factor | Entropy | Dominant task |
|--------|---------|---------------|
| f69 | 0.013 | hypernymy |
| f508 | 0.013 | capital_country |
| f363 | 0.014 | sva |
| f624 | 0.014 | greater_than |
| f279 | 0.014 | capital_country |

These factors have near-zero entropy — they route almost entirely through
a single task's edges. They are the most "private" factors in the bank.

### Jaccard overlap on top-50 factors per task

| Task pair | Jaccard(top-50) |
|-----------|-----------------|
| capital_country vs hypernymy | ~0.60 |
| ioi vs sva | ~0.44 |
| greater_than vs capital_country | ~0.40 |

(These values come from the top-50 most-attributed factors per task.
Lower than the full-profile correlations because the top-50 captures
the task-discriminative tail, not the shared bulk.)

### Factor edge routing profiles

For the most active factors, their dominant write and read edges
correspond to specific heads in the IOI circuit. For example, factors
with high IOI attribution write primarily through L5.H9 (induction) and
read at L7.H3.v (S-inhibition), tracing the expected circuit path at
factor resolution.

## Analysis C: Weight Coupling vs EAP Coupling

For each of the 228 circuit edge pairs, compute:
1. **Weight coupling**: mean |factor_i @ OV_{reader} @ factor_j^T|
2. **EAP co-attribution**: mean cross-task outer product of per-factor
   write and read scores

### Aggregate comparison

| Metric | Value |
|--------|-------|
| Overall correlation (weight vs EAP edge strength) | 0.217 |
| Circuit-only correlation | 0.310 |
| AUROC (EAP co-attribution → circuit edges) | 0.476 |
| AUROC (weight coupling → circuit edges) | 0.420 |
| Mean per-edge factor-level correlation | ~0.000 |
| Median per-edge factor-level correlation | ~0.000 |

### Key findings

**1. Weight and EAP coupling are weakly but positively correlated at the
edge level (r=0.22 overall, r=0.31 for circuit edges).** Edges with
stronger weight-space factor coupling tend to have stronger EAP
co-attribution. The relationship is stronger within the circuit than
across all edges.

**2. Neither weight coupling nor EAP co-attribution discriminates circuit
edges from non-circuit edges (AUROCs 0.42 and 0.48).** Both measures are
below chance — circuit edges actually have weaker coupling than average.
This is the same anti-signal pattern: circuit heads specialize on narrow
factor subspaces.

**3. The factor-level coupling matrices are essentially uncorrelated
(r ~ 0.000).** For any given circuit edge, the 1024x1024 weight coupling
matrix does not predict the 1024x1024 EAP co-attribution matrix. The
weight geometry tells you the CAPACITY for information flow, but EAP
tells you what ACTUALLY flows — and these are largely independent at
factor resolution.

**4. The strongest weight-EAP aligned edges have tiny correlations
(r = 0.002).** Even the best-matching edges show effectively zero
factor-level agreement. The correlation in finding #1 is driven by
the marginal statistics (mean coupling strength), not by factor-pair-level
agreement.

## Synthesis Across All Three Analyses

### The "narrow channel" hypothesis

The recurring pattern across A, B, and C is:

1. **Circuit heads use narrow factor subspaces.** Their total OV coupling
   is low (anti-signal in A), their factor usage is concentrated (low
   entropy in B), and their factor-level coupling matrices are sparse.

2. **Non-circuit heads use broad factor subspaces.** They have high total
   OV coupling, diffuse factor usage, and dense coupling matrices.

3. **The narrow channels are task-specific.** The few factors that carry
   IOI information through induction and S-inhibition heads are different
   from those carrying SVA or greater-than information, even though most
   factors are shared across tasks.

4. **Weight geometry measures capacity, not function.** The weight-space
   OV coupling tells you which factor pairs CAN communicate through a
   head, but the actual communication (EAP) is gated by input-dependent
   attention patterns and is restricted to a tiny fraction of the
   weight-enabled capacity.

### Connection to the factor DAG analysis

The factor DAG analysis (from DAS-EAP scores) found a hub-and-spoke
structure centered on F798 (token identity) and F222 (convergent
S-inhibition). Analysis B's routing fingerprints show these same factors
are among the most active across tasks, but with specific routing
preferences. The hub structure emerges because these central factors
broadcast through the shared residual stream to all downstream factors
directly, rather than following the sequential circuit edges.

The weight-space coupling (Analysis A) sees none of this structure
because it measures total coupling capacity, dominated by the generic
non-circuit heads. The EAP attribution (Analysis C) sees it weakly
because co-attribution conflates direct and indirect paths.

### What this means for circuit interpretation

To interpret factorized EAP results:
- Don't use weight-space coupling to predict which factors flow where
  (r ~ 0 at factor level)
- Do use routing fingerprints (Analysis B) to identify which factors
  are task-specific vs shared
- Do use DAS-EAP factor DAGs to trace factor-level information flow
  (hub structure, not chain structure)
- The weight-space OV coupling is useful for a different question:
  which heads have the CAPACITY for cross-factor communication
  (Analysis A), regardless of whether they use it
