# Cross-Task Analysis: Factor EAP Structure Across 6 Tasks

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
Tasks: IOI, SVA, greater_than, capital_country, gender_bias, hypernymy
Date: 2026-06-14

## Analysis 43: Cross-Task Edge Rankings

Edge importance rankings are **extremely correlated** across all 6
tasks (Spearman rho 0.969--0.980). The same edges are important
for every task.

### Spearman correlation matrix

|                | IOI   | SVA   | GT    | CC    | GB    | HY    |
|----------------|-------|-------|-------|-------|-------|-------|
| IOI            | 1.000 | 0.972 | 0.972 | 0.974 | 0.974 | 0.974 |
| SVA            |       | 1.000 | 0.970 | 0.976 | 0.979 | 0.976 |
| greater_than   |       |       | 1.000 | 0.971 | 0.969 | 0.972 |
| capital_country|       |       |       | 1.000 | 0.978 | 0.980 |
| gender_bias    |       |       |       |       | 1.000 | 0.978 |
| hypernymy      |       |       |       |       |       | 1.000 |

### But top-50 edges diverge

Despite near-identical rank correlations, **top-50 edge Jaccard
overlap is low** (0.031--0.163):

- IOI vs SVA: J=0.031 (3/50) --- nearly disjoint top edges
- IOI vs greater_than: J=0.075 (7/50)
- capital_country vs hypernymy: J=0.163 (14/50) --- highest overlap

The 4 non-IOI/SVA tasks cluster with higher mutual overlap
(J=0.149--0.163). IOI and SVA use more idiosyncratic top edges.

### Interpretation

The rank correlation is driven by the bulk distribution: most edges
have similar importance rankings across tasks because the factor bank
provides shared infrastructure. But the "interesting" tail (top 50)
diverges --- each task selects different edges for its critical path.
This is the edge-level analog of the reader-selectivity finding from
the IOI-only atlas.

## Analysis 44: Task-Specific vs Shared Factors

With a dense selector and 1024 factors, virtually **all factors are
shared** across tasks:

- Max entropy (uniform across 6 tasks): 2.585 bits
- Mean factor entropy: 2.575 (99.6% of max)
- Median: 2.578
- **0 factors** with H < 50% of max (no task-specific factors)
- **1022 of 1024** with H > 90% of max

### Most "task-biased" factors (still shared, just slightly biased)

| Factor | Entropy | Dominant task | Dominance |
|--------|---------|---------------|-----------|
| f378   | 2.026   | SVA           | 55.4%     |
| f201   | 2.228   | SVA           | 46.5%     |
| f319   | 2.363   | gender_bias   | 40.0%     |
| f101   | 2.421   | gender_bias   | 35.9%     |
| f511   | 2.438   | gender_bias   | 35.1%     |
| f305   | 2.535   | IOI           | 27.1%     |

SVA has the most biased factors (f378, f201). Gender_bias has the
most factors that lean its way. IOI's "most specific" factor (f305)
still allocates 73% to other tasks.

### Why no specialization?

The dense selector lets every projection access every factor. There
is no sparsity pressure forcing task-specific allocations. The factor
bank acts as a fully shared vocabulary; task specificity emerges from
how selectors *weight* the shared factors on each edge, not from
which factors exist.

## Analysis 45: Per-Task Factor Concentration

| Task           | Top-10 | Top-50 | Top-100 | Equiv factors | Gini  |
|----------------|--------|--------|---------|---------------|-------|
| IOI            | 2.8%   | 11.1%  | 19.2%   | 884           | 0.000 |
| SVA            | 3.0%   | 11.8%  | 19.7%   | 867           | 0.021 |
| greater_than   | 2.5%   | 10.2%  | 17.7%   | 914           | 0.002 |
| capital_country| 2.6%   | 10.6%  | 18.0%   | 913           | 0.001 |
| gender_bias    | 3.6%   | 13.9%  | 22.6%   | 796           | 0.001 |
| hypernymy      | 2.8%   | 11.7%  | 19.3%   | 880           | 0.010 |

All tasks are **extremely diffuse**: equivalent factors range from
796--914 (out of 1024). Gender_bias is slightly more concentrated
(top-10 captures 3.6%, equiv=796) than greater_than (top-10 = 2.5%,
equiv=914).

### Factor usage Spearman correlations

|                | IOI   | SVA   | GT    | CC    | GB    | HY    |
|----------------|-------|-------|-------|-------|-------|-------|
| IOI            | 1.000 | 0.808 | 0.770 | 0.870 | 0.838 | 0.838 |
| SVA            |       | 1.000 | 0.727 | 0.837 | 0.832 | 0.832 |
| greater_than   |       |       | 1.000 | 0.821 | 0.774 | 0.785 |
| capital_country|       |       |       | 1.000 | 0.893 | 0.917 |
| gender_bias    |       |       |       |       | 1.000 | 0.868 |
| hypernymy      |       |       |       |       |       | 1.000 |

Capital_country and hypernymy are the most correlated (0.917).
Greater_than and SVA are most different (0.727) --- two tasks that
use different kinds of knowledge (arithmetic vs agreement).

## Analysis 46: IOI vs SVA Factor Profiles on Shared Edges

Loaded the full (157, 445, 1024) scores for IOI and SVA to compare
factor composition on the top-100 edges.

### Global statistics

| Metric | Value |
|--------|-------|
| Global factor cosine (IOI vs SVA) | 0.755 |
| Global Spearman rho | 0.188 |
| Per-edge cosine mean | 0.581 |
| Per-edge cosine std | 0.093 |
| Per-edge cosine range | [0.202, 0.748] |

The Spearman rho (0.188) is much lower than the cosine (0.755)
because cosine is dominated by a few high-mass shared factors while
Spearman captures the full rank disagreement.

### Most divergent edges

| Edge | IOI-SVA cos |
|------|-------------|
| input → M0.in | 0.202 |
| L7.H9 → logits | 0.327 |
| L7.H3 → logits | 0.399 |
| L8.H6 → logits | 0.410 |
| L9.H7 → logits | 0.418 |

Early (input) and late (→logits) edges diverge most. The input edge
starts from the same embedding but the two tasks read entirely
different factor components. Logits edges diverge because the output
space is task-specific.

### Most similar edges

| Edge | IOI-SVA cos |
|------|-------------|
| M0 → L7.H7.Q | 0.748 |
| L3.H0 → L5.H3.Q | 0.717 |
| M0 → L3.H1.K | 0.716 |
| M2 → M4.in | 0.702 |
| M2 → L5.H1.V | 0.699 |

Mid-layer MLP-to-attention and cross-layer edges are most similar.
These are the "shared infrastructure" edges where both tasks read
similar general-purpose factors.

### Top differentiating factors

| Factor | IOI share | SVA share | Difference |
|--------|-----------|-----------|------------|
| f201   | 0.059%    | 2.194%    | -2.14%     |
| f5     | 0.147%    | 1.024%    | -0.88%     |
| f378   | 0.066%    | 0.369%    | -0.30%     |
| f305   | 0.326%    | 0.125%    | +0.20%     |
| f519   | 0.219%    | 0.074%    | +0.14%     |

f201 and f5 are heavily SVA-biased; f305, f519, f337 are IOI-biased.
The IOI/SVA mass ratio on shared edges is enormous (median 1476x) ---
IOI dominates because IOI's top edges carry far more absolute mass
than SVA's.

## Analysis 47: Cross-Task Factor Overlap

### Cosine similarity of factor usage profiles

|                | IOI    | SVA    | GT     | CC     | GB     | HY     |
|----------------|--------|--------|--------|--------|--------|--------|
| IOI            | 1.0000 | 0.9790 | 0.9835 | 0.9917 | 0.9761 | 0.9884 |
| SVA            |        | 1.0000 | 0.9749 | 0.9826 | 0.9696 | 0.9829 |
| greater_than   |        |        | 1.0000 | 0.9912 | 0.9649 | 0.9864 |
| capital_country|        |        |        | 1.0000 | 0.9760 | 0.9943 |
| gender_bias    |        |        |        |        | 1.0000 | 0.9797 |
| hypernymy      |        |        |        |        |        | 1.0000 |

All pairwise cosines >0.96. The factor bank is **massively shared**.
Capital_country--hypernymy (0.9943) is the most similar pair.
Gender_bias--greater_than (0.9649) is the most different.

### Top-factor Jaccard overlap

| K   | Range       | Mean  |
|-----|-------------|-------|
| 50  | 0.587--0.818| 0.695 |
| 100 | 0.575--0.739| 0.646 |
| 200 | 0.544--0.754| 0.641 |

Overlap decreases with K, as expected (harder to agree on more
factors). But even at K=200, most pairs share >60% of their top
factors.

### Universal vs task-unique factors

- **66 factors** in top-100 for ALL 6 tasks (universal core)
- **171 factors** in top-100 for ANY task (union)
- Task-unique factors (in top-100 for only that task):
  - greater_than: 15 (most unique)
  - IOI: 11
  - SVA: 11
  - gender_bias: 11
  - hypernymy: 8
  - capital_country: 4 (least unique --- shares most with others)

## Synthesis

### 15. The Factor Bank Is General-Purpose Infrastructure

The 1024-factor bank with dense selector creates a **fully shared
vocabulary** across tasks:

1. **Edge rankings are near-identical** (rho > 0.97 all pairs): the
   same edges are important for every task.
2. **Factor usage is near-uniform** (99.8% of factors have >90% of
   max entropy): no task-specific factors exist.
3. **Factor usage profiles are nearly identical** (cosine > 0.96):
   tasks use the same factors in similar proportions.
4. **66 factors form a universal core** (top-100 for all 6 tasks).

### 16. Task Specificity Emerges at the Edge Tail, Not the Factor Level

Despite massive factor sharing:

1. **Top-50 edge overlap is low** (Jaccard 0.031--0.163): the
   critical edges differ across tasks.
2. **Per-edge factor composition diverges** (IOI vs SVA cos = 0.58):
   how top edges weight shared factors is task-specific.
3. **Early and late edges diverge most**: input and logits edges have
   the most task-specific factor profiles.

The shared factor bank is like a universal language; tasks differ in
what sentences they construct, not in what words are available.

### 17. IOI and SVA Are the Most Distinctive Tasks

- Lowest mutual edge Jaccard (0.031)
- Lowest mutual factor Spearman (0.808)
- Per-edge factor cosine only 0.58 on shared top edges
- Strongest differentiating factors: f201 (SVA-specific), f305 (IOI)

The other 4 tasks (greater_than, capital_country, gender_bias,
hypernymy) form a tighter cluster, especially capital_country and
hypernymy (factor cosine 0.994, Spearman 0.917).
