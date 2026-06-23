# Unsupervised Edge Mode Discovery from DAS-EAP Subspaces

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Setup

32 DAS dimensions, each with per-edge attribution scores (W and R
marginals). 500 top edges as observations. No prior labels used —
factor role assignments discovered independently from weight-space
analysis (node classification, OV SVD directions) and from DAS
factor identification. This analysis treats the 32 dims as unlabeled
variables and asks what structure emerges from edge co-occurrence alone.

## Finding 1: DAS dims do NOT cluster by factor identity

Hierarchical clustering (Ward linkage, correlation distance) at k=6
recovers clusters that mix dims from different factors.

Adjusted Rand Index vs ground-truth factor groupings: **0.028** (near zero).
Cluster purity: **32%** (chance would be ~25% for this group size distribution).
Permutation test on intra-group correlation: **p = 0.31** (not significant).

The ground-truth factor groups (e.g., all 6 dup_token dims) have mean
intra-group correlation of 0.544, but random groupings of the same
size achieve 0.479 +/- 0.137. The factor groupings are only marginally
more coherent than chance in their edge co-occurrence patterns.

This means: **dims belonging to the same factor do NOT co-activate
on the same edges.** The factor decomposition describes the causal
subspace structure (what information each factor carries), but edges
use factors in mixed combinations, not in factor-pure channels.

## Finding 2: Edges carry mixtures, not pure factor signals

NMF decomposition into 3-5 modes shows every mode loads on multiple
factors. There are no "pure dup_token edges" or "pure S_inhib edges"
in the mode structure.

NMF k=3 modes:

| Mode | Top GT factors (by loading) | Interpretation |
|------|---------------------------|----------------|
| 1 | dup_token(1.02), S_inhib(0.65), positional(0.51) | Full-circuit mode |
| 2 | S_inhib(1.03), dup_token(0.71), conv_S_inhib(0.17) | S-inhibition-weighted |
| 3 | dup_token(0.88), positional(0.83), S_inhib(0.18) | Position-weighted |

The modes differ in their relative weighting of the three dominant
factors, not in which factors they use. Every mode includes dup_token.

## Finding 3: PC1 = total importance, PC2 = circuit stage contrast

PCA on the edge x dim matrix:

| PC | Variance | Positive loading | Negative loading |
|----|----------|-----------------|-----------------|
| PC1 | 62.3% | All factors (dup_token, S_inhib, positional) | Nothing |
| PC2 | 12.3% | S_inhib, conv_S_inhib | dup_token, auxiliary |
| PC3 | 6.1% | S_inhib, dup_token | positional, dup_token |
| PC4 | 5.0% | dup_token, S_inhib, late_comp | positional, auxiliary |
| PC5 | 4.1% | S_inhib, dup_token, positional | dup_token, positional |

PC1 is a scale component — edges that carry more signal carry more
of everything. The 62% variance on PC1 means most of the variation
across edges is "some edges are more important than others," not
"different edges carry different things."

PC2 is the first structural contrast: S_inhib vs dup_token. Edges
split into S_inhib-weighted and dup_token-weighted along this axis.
This 12% is the actual circuit-stage signal.

PC3 contrasts S_inhib/dup_token (positive) against positional
(negative). Edges split into "content-routing" vs "position-routing."

## Finding 4: Layer-by-layer subspace flow

Cluster composition of edges grouped by reader layer
(using the discovered k=6 clusters, C1 and C2 dominant):

| Reader layer | C1 | C2 | Dominant mode |
|-------------|-----|-----|--------------|
| L0-L3 | 40-50% | 33-46% | Mixed |
| L4-L7 | 51-59% | 29-31% | C1 dominant (mid-circuit) |
| L8 | 45% | 36% | Transition |
| L9 | 45% | 41% | Balanced (name movers reading) |
| logits | 36% | 53% | C2 dominant (output) |

C1 dominates mid-circuit edges (L4-L7), C2 dominates output edges
(logits). The crossover happens at L8-L9 where name movers begin
reading. This layer-dependent mode switching is consistent with the
circuit's functional stages even though the modes don't align with
individual factors.

## Finding 5: Head-level subspace selectivity

**Translator heads** (read one mode, write another):

| Head | Reads | Writes | Known role |
|------|-------|--------|-----------|
| L3.H6 | C2 (55%) | C1 (43%) | — |
| L4.H8 | C2 (49%) | C1 (45%) | — |
| L5.H9 | C1 (48%) | C2 (45%) | Induction |
| L4.H3 | C1 (52%) | C2 (41%) | — |
| L7.H3 | C1 (45%) | C2 (45%) | S-inhibition |
| L7.H9 | C2 (49%) | C1 (40%) | S-inhibition |

Translator heads read one edge mode and write another — they sit
at the boundary between circuit stages. L5.H9 (induction) reads
C1 and writes C2, consistent with transforming early-circuit signals
into late-circuit format. L7.H3 and L7.H9 (S-inhibition) both
translate between modes.

## Finding 6: C1 and C2 anti-correlate but co-exist

Cluster fraction correlation across edges:

| | C1 | C2 | C3 | C4 |
|---|-----|-----|-----|-----|
| C1 | 1.00 | **-0.87** | -0.34 | -0.02 |
| C2 | -0.87 | 1.00 | 0.07 | -0.35 |
| C3 | -0.34 | 0.07 | 1.00 | -0.08 |
| C4 | -0.02 | -0.35 | -0.08 | 1.00 |

C1 and C2 trade off strongly (r = -0.87): edges that carry more C1
carry less C2, and vice versa. This is NOT a scale artifact (it's
computed on fractions, not raw mass). It reflects genuine
specialization: circuit edges divide into two modes of operation
that exclude each other.

C3 anti-correlates weakly with C1, and C4 anti-correlates with C2.
These are minor modes that trade off with specific major modes.

## Finding 7: Causal mediation between modes

Significant mediation paths (Sobel p < 0.01):

| Path | % mediated |
|------|-----------|
| C1 -> C2 -> C5 | 83% |
| C1 -> C4 -> C5 | 75% |
| C2 -> C1 -> C4 | 70% |
| C2 -> C1 -> C3 | 60% |
| C2 -> C3 -> C1 | 33% |

C2 mediates 83% of C1's effect on C5 — the minor mode C5 is
almost entirely accessible through the major mode C2. The mediation
structure mirrors the circuit's information flow: early modes (C1)
influence late modes (C5) through the dominant mid-circuit mode (C2).

## Synthesis

The DAS subspaces within circuit edges organize into **usage modes**
that cut across factor boundaries. Individual factors (dup_token,
S_inhib, positional) were discovered independently from weight-space
analysis, but at the edge level these factors remix into new
combinations:

1. Factor roles (from weight space / DAS): "what information does
   this factor carry?" — token identity, position, inhibition
2. Edge modes (from this analysis): "how do edges bundle these
   roles?" — the circuit multiplexes multiple factor signals per
   edge, with mode composition shifting by layer depth

The weight-space factor classification (RF accuracy 42% with OV SVD
directions, node-level) and the edge-level mode discovery (this
analysis) are complementary: factors are the alphabet, edge modes
are the words. Neither reduces to the other.

The key claim: **circuit edges are not typed by single factors.
They carry factor mixtures whose composition shifts systematically
through the circuit's layer structure, with translator heads at
the mode boundaries.**
