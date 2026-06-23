# Graph-Theoretic Analyses: Factor Network, Transitions, and Conditional Structure

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 36: Factor Interaction Graph

The partial correlation graph (edges with |r_partial| > 0.3,
controlling for total edge mass) has only **2 positive edges and 9
negative edges**. The factor network is dominated by competition,
not co-occurrence.

### Two-cluster structure

**Cluster B (S_inhib + conv_S_inhib)**:
- Strong internal bond: r_partial = +0.465
- Both negatively connected to Cluster A factors

**Cluster A (dup_token + positional + late_comp)**:
- Weak internal bond: mean r_partial = +0.081
- positional-late_comp (+0.352) is the only strong positive link
- All three negatively connected to Cluster B

Between clusters: mean r_partial = -0.456.

**Isolated factors**: token_id and name_mover have no edges above
the 0.3 threshold — they don't participate in the factor competition
after controlling for scale.

### Interpretation

The circuit's factor structure is a two-team competition:
- Team 1 (Cluster A): content factors — what token is duplicated,
  where it appeared, late compositional features
- Team 2 (Cluster B): action factors — suppress the repeated name,
  converge on inhibition

When an edge carries more of Team 1, it carries less of Team 2, and
vice versa. This is not a scale artifact (it's after scale control)
— it reflects genuine functional specialization of the residual
stream into content-routing and action-routing modes.

## Analysis 37: Writer-Conditional Factor Profiles

Factor composition is remarkably invariant to writer type:

| Writer group | n | dup_token | S_inhib | positional |
|-------------|---|-----------|---------|------------|
| input | 13 | 26.6%* | 23.4% | 17.7%* |
| M0-M3 | 246 | 33.0% | 24.0% | 21.5% |
| M4-M7 | 131 | 32.8% | 24.3% | 22.0% |
| M8-M11 | 20 | 34.4% | 22.1% | 23.7% |
| early attn | 200 | 32.0% | 25.1% | 20.7% |
| mid attn | 324 | 32.8% | 21.7% | 23.1% |
| late attn | 66 | 34.2% | 22.3% | 23.0% |

(* = deviates >2% from population mean)

Only the input embedding has a notably different profile (lower
dup_token and positional). All other writer groups are within ±2%
of the population mean. This reinforces the "broadcast" model:
writers produce a generic mixture; the factor composition is set by
the DAS rotation, not by the writer.

## Analysis 38: L2-L3 Transition Deep Dive

The L2→L3 transition showed the largest single-step change in
factor composition (-12.7% S_inhib). Here's what drives it.

### L2 reader edges are S_inhib-heavy

| Edge | Mass | Top factor |
|------|------|-----------|
| M0 → M2.in | 0.433 | S_inhib 46.8% |
| L1.H11 → M2.in | 0.335 | S_inhib 50.9% |
| M1 → M2.in | 0.185 | S_inhib 36.4% |

### L3 reader edges are dup_token-heavy

| Edge | Mass | Top factor |
|------|------|-----------|
| L3.H0 → M3.in | 0.532 | dup_token 30.9% |
| M0 → L3.H0.Q | 0.203 | dup_token 53.6% |
| M0 → M3.in | 0.201 | dup_token 35.4% |

### Statistical confirmation

| Factor | L2 median | L3 median | Direction | p-value |
|--------|-----------|-----------|-----------|---------|
| S_inhib | 0.224 | 0.206 | L2 > L3 | 0.022 |
| dup_token | 0.314 | 0.353 | L3 > L2 | 0.001 |

### Duplicate token head L3.H0

L3.H0 (a known duplicate token detector) reads with strong dup_token
bias: 46.0% in Q, 31.3% in K, 34.2% in V. Its Q projection reads
only 11.5% S_inhib — the lowest S_inhib share of any L3 reader.

The L2→L3 transition happens because L3.H0's query selectively
amplifies dup_token from the residual stream while suppressing
S_inhib. This is reader selectivity in action: the same residual
stream carries both signals, but L3.H0's query projection is tuned
to extract dup_token.

## Analysis 39: Edge Anomaly Taxonomy

Classifying edges by their largest deviation from the population
mean (>5% threshold):

| Anomaly type | Count | % |
|-------------|-------|---|
| high_S_inhib | 114 | 22.8% |
| typical | 103 | 20.6% |
| high_positional | 88 | 17.6% |
| high_dup_token | 84 | 16.8% |
| high_auxiliary | 30 | 6.0% |
| low_S_inhib | 18 | 3.6% |
| low_dup_token | 15 | 3.0% |
| high_token_id | 14 | 2.8% |
| Other | 34 | 6.8% |

S_inhib anomalies are asymmetric: 114 edges have anomalously HIGH
S_inhib but only 18 have anomalously LOW S_inhib. This matches
the right-skewed distribution from Analysis 32 — most edges carry
moderate S_inhib, but a substantial minority (~23%) carry
anomalously high S_inhib.

Only 20.6% of edges are "typical" (within ±5% of the population
mean on all factors). Nearly 80% of edges deviate measurably from
the average profile in at least one factor.

## Analysis 40: Within-Writer Profile Consistency

How consistently does each writer produce the same factor profile
across its different reader edges?

| Metric | Value |
|--------|-------|
| Mean consistency (cosine) | 0.861 |
| Std | 0.015 |
| MLP writers | 0.859 |
| Attention writers | 0.861 |

Writers are highly consistent: an edge from M3 to L5.H3.Q has
nearly the same factor profile as an edge from M3 to L8.H10.Q
(cos ≈ 0.86).

Most consistent writer: L7.H0 (0.898). Least consistent: L0.H10
(0.832) — a duplicate token head that writes to diverse readers
with slightly different profiles.

The high consistency means factor composition is writer-determined
more than reader-determined — even though reader-marginal mass is
67.8x larger. The writer sets the mixture; the reader amplifies
specific components from it.

## Analysis 41: Top 20 Edge Profiles

Key patterns in the 20 strongest edges:

| Pattern | Count |
|---------|-------|
| MLP writer | 11/20 |
| Non-circuit reader | 14/20 |
| Logits reader | 2/20 |
| L9.H3 reader | 3/20 (#1, #2, #20) |
| L8.H6 writer | 5/20 |

The top 2 edges both feed L9.H3 (non-circuit). L8.H6 (S-inhibition)
is a prolific writer across the top 20. R/W ratios range from 80 to
1245 — massive reader dominance.

Notable: M8 → logits (edge #8) has the most distinctive profile
of any top edge: 42.9% dup_token + 34.0% S_inhib + only 7.8%
positional. This is almost a pure "suppress the repeated name"
signal — the closest any edge comes to a single-function wire.

## Analysis 42: Conditional Factor Interactions

When two of the three main factors are both above/below median,
what happens to the third?

### S_inhib is the most squeezable

| Condition | S_inhib share |
|-----------|--------------|
| dup_token high + positional high | 17.4% |
| dup_token low + positional low | **29.5%** |
| Population mean | 22.9% |

When both content factors (dup_token, positional) are high, S_inhib
drops to 17.4% — a 5.5% drop. When both are low, S_inhib rises to
29.5% — a 6.6% rise. S_inhib is the "residual" factor: it fills
whatever space the content factors don't occupy.

### dup_token is the most resilient

| Condition | dup_token share |
|-----------|----------------|
| S_inhib high + positional high | 29.1% |
| S_inhib low + positional low | 37.2% |
| Population mean | 32.8% |

dup_token varies only ±4% from the mean regardless of the other
factors. It's the "backbone" of the edge profile — always present,
barely compressed even when the other factors are at their highest.

### Interpretation

The compositional dynamics confirm the two-team structure:
- dup_token is the resilient backbone (Team 1 anchor)
- positional is the secondary content signal (Team 1 member)
- S_inhib is the action signal that fills remaining bandwidth (Team 2)

This isn't a symmetric three-way competition. dup_token has
structural priority; S_inhib is the flexible component that
expands when content factors contract and contracts when they expand.

## Updated Synthesis

42 analyses across 6 batches. The core picture:

1. **Two teams, not three streams**: The three-stream model is
   better described as a two-team competition between content
   factors (dup_token + positional + late_comp) and action factors
   (S_inhib + conv_S_inhib), with dup_token as the resilient anchor.

2. **Competition network**: The factor graph has 9 negative edges
   and only 2 positive edges after controlling for scale. The
   circuit's computational structure is implemented through factor
   competition, not factor cooperation.

3. **Reader selectivity at transition points**: The L2→L3 transition
   (the biggest single-step change) is driven by L3.H0's query
   selectively amplifying dup_token while suppressing S_inhib from
   the same residual stream.

4. **High writer consistency**: Writers produce the same mixture
   regardless of reader (cos = 0.86). The factor composition is
   writer-set but reader-amplified.
