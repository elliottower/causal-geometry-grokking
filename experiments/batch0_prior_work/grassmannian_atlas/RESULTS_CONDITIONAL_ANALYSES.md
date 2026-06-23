# Conditional Analyses: Structure Beyond Scale

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 19: Residualized Structure (PC1 Removed)

After projecting out the dominant scale component (PC1, 62.3%), the
residual spectrum redistributes across several structural axes:

| Component | Variance (of residual) | Positive loading | Negative loading |
|-----------|----------------------|-----------------|-----------------|
| rPC1 | 32.7% | S_inhib (+0.68) | dup_token (-0.20), positional (-0.20) |
| rPC2 | 16.1% | positional (+0.71) | S_inhib (-0.75) |
| rPC3 | 13.2% | — | — |
| rPC4 | 10.8% | — | — |

The two structural axes that emerge after scale removal:
1. **S_inhib vs everything else** (rPC1, 33%): edges that carry
   proportionally more S-inhibition vs those that carry more of the
   other two main factors
2. **Positional vs S_inhib** (rPC2, 16%): edges that carry
   proportionally more positional vs more S-inhibition

These match the factor interaction findings from Analysis 9
(positional vs S_inhib trade-off, r=-0.569). The trade-offs ARE
real structural axes — they're just hidden behind the overwhelming
scale component in the raw data.

### Clustering quality

Ward clustering on the residualized data (k=4) is poor: 491/500
edges land in one giant cluster (C4), with only 9 outlier edges
split across C1-C3. Cophenetic correlation = 0.658.

The edge population is essentially unimodal in residual space. The
"modes" from the unsupervised analysis are tendencies along
continuous axes, not discrete clusters.

## Analysis 20: Factor Profiles by Importance Quartile

| Quartile | n | dup_token | S_inhib | positional |
|----------|---|-----------|---------|------------|
| Top 25% | 8122 | 32.7% | 23.0% | 22.2% |
| 25-50% | 8122 | 30.4% | 23.5% | 21.1% |
| 50-75% | 8122 | 28.3% | 22.9% | 20.5% |
| Bottom 25% | 8123 | 26.9% | 25.3% | 20.8% |

The three-stream mixture is stable across importance levels. The
top quartile (which carries 97% of total mass) has only slightly
more dup_token and slightly less S_inhib than the bottom quartile.
Factor composition is not an artifact of edge importance — it's a
genuine property of the circuit's information routing.

## Analysis 21: The L9.H3 Surprise

### L9.H3 is the single most important head in DAS-EAP

| Head | Total DAS-EAP mass | IOI role |
|------|-------------------|----------|
| **L9.H3** | **12.71** | non-circuit |
| L8.H10 | 9.66 | S-inhibition |
| L8.H7 | 9.60 | non-circuit |
| L9.H2 | 8.62 | non-circuit |
| L5.H3 | 8.54 | non-circuit |

Only 2 of the top 15 heads by DAS-EAP read mass are IOI circuit
heads (L8.H10 at #2, L8.H6 at #14). The known name movers rank
far below:

| Head | Mass | Rank/144 | IOI role |
|------|------|----------|----------|
| L9.H9 | 0.30 | ~70 | name_mover |
| L9.H6 | 1.85 | ~30 | name_mover |
| L10.H0 | low | ~53 | name_mover |

**L9.H3 has 42x the DAS-EAP read mass of L9.H9 (the primary name
mover).** Circuit heads have median rank 53/144 — the bottom half.

### L9.H3's reading is Q-dominated

| Projection | Mass |
|------------|------|
| Q | 12.27 (96.5%) |
| K | 0.15 (1.2%) |
| V | 0.29 (2.3%) |

Almost all of L9.H3's DAS-EAP influence flows through its query
projection. Its top writers are M8 (3.34) and L8.H6 (2.52) — the
same nodes that feed the known name movers.

### L9.H3 profile vs name movers

| Head | dup_token | positional | S_inhib |
|------|-----------|------------|---------|
| L9.H3 | 34.2% | 23.4% | 18.2% |
| L9.H9 | 27.8% | 17.4% | 35.0% |
| L9.H6 | 39.2% | 16.3% | 13.5% |

L9.H3 has more positional information than either name mover, less
S_inhib than L9.H9, and a balanced three-stream profile.

### Interpretation

Two possibilities:
1. **The DAS rotation reveals a broader circuit** that includes
   L9.H3 (and many other non-circuit heads) as participants in the
   IOI task through the factorized model's shared factor bank
2. **The DAS-EAP scores measure something different from the
   standard IOI circuit** — they capture the residual stream's
   factor-subspace structure, which is dominated by generic
   information routing (non-circuit heads are generalists that
   contribute to everything), not task-specific circuits

The second interpretation is supported by Analysis 11 (circuit heads
have the same read selectivity as non-circuit heads) and by the
weight coupling analysis (circuit heads have LOWER OV coupling than
non-circuit heads). The DAS-EAP decomposition sees the model's
general information infrastructure, not just the IOI-specific circuit.

## Analysis 22: Layer-to-Layer Factor Transitions

The biggest single-step changes in factor composition:

| Transition | Factor | Change | Interpretation |
|------------|--------|--------|----------------|
| L2 -> L3 | S_inhib | -12.7% | Duplicate token heads activate |
| L9 -> logits | S_inhib | +10.8% | Output concentrates inhibition |
| L9 -> logits | positional | -10.0% | Output drops position signal |
| L3 -> L4 | dup_token | -6.7% | Post-dup-token adjustment |

The L2 -> L3 transition (cosine similarity drops to 0.865, lowest of
any adjacent-layer pair) marks the activation of the duplicate token
detection mechanism. S_inhib drops from 33.4% to 20.6% as the
dup_token signal takes over.

The L9 -> logits transition is the sharpest: positional drops 10%
and S_inhib rises 10.8%. The output discards position and amplifies
inhibition — consistent with the circuit's final function being to
suppress the repeated name.

Positional information rises monotonically from L0 (17.6%) to L9
(25.0%), then drops sharply at logits. It's accumulated across the
full circuit depth but consumed only at the output.

## Analysis 23: Logits Layer Deep Dive

Top writers to logits:

| Writer | Mass | dup_token | S_inhib | positional |
|--------|------|-----------|---------|------------|
| M8 | 1.14 | 42.9% | 34.0% | 7.8% |
| M9 | 0.98 | 36.8% | — | 25.8% |
| L8.H6 | 0.84 | 41.0% | 32.7% | 9.8% |
| L9.H3 | 0.36 | 29.6% | 39.2% | 16.1% |
| L9.H6 | 0.25 | 31.6% | 17.7% | 16.9% |

Logits edges carry +7.1% S_inhib and -7.5% positional vs the
population average. The output is where S-inhibition concentrates
and positional information is discarded.

M8 and L8.H6 dominate the logits output (M8: 42.9% dup_token +
34.0% S_inhib = nearly pure "suppress the repeated name" signal).

## Analysis 24: Writer-Reader Factor Asymmetry

Writers and readers of the same edge disagree substantially on
which factors the edge carries:

| Metric | Value |
|--------|-------|
| W-R cosine similarity | 0.567 +/- 0.143 |
| Min cosine | 0.171 |
| Max cosine | 0.924 |

Per-factor asymmetry (positive = reader reads MORE than writer writes):

| Factor | R-W asymmetry | p-value |
|--------|--------------|---------|
| dup_token | **+0.068** | <0.001 |
| positional | **+0.049** | <0.001 |
| S_inhib | **+0.048** | <0.001 |
| token_id | **-0.081** | <0.001 |
| conv_S_inhib | -0.033 | <0.001 |
| late_comp | -0.024 | <0.001 |

The three main factors (dup_token, positional, S_inhib) are all
**reader-amplified**: readers selectively consume more of these
factors than writers produce. token_id is **writer-biased**: it's
produced more than consumed.

This is a directional asymmetry in the residual stream's factor
routing: the embedding writes token_id broadly, but downstream
readers preferentially consume the three task-relevant factors and
let token_id pass through unread.

## Analysis 25: Factor Concentration (Herfindahl Index)

| Metric | Value |
|--------|-------|
| Mean HHI | 0.236 +/- 0.029 |
| Equivalent factors | 4.2 |

An HHI of 0.236 corresponds to 4.2 equally-weighted factors per
edge. The 9-factor decomposition has about 4 effective components
at the edge level.

HHI is uniform across layers (0.21-0.27), confirming the absence of
a depth-dependent specialization gradient. The most concentrated
edges are in early layers (L1.H11 -> L3.H0.Q: 54.7% dup_token) and
at the output (M8 -> logits: 42.9% dup_token).

## Analysis 26-27: Edge Length and Layer Span

Neither edge length (writer-reader layer gap) nor within-vs-cross-
layer classification affects factor composition:

- Edge length vs dup_token: r = -0.37, p = 0.29 (not significant)
- Within vs cross-layer cosine: 0.982
- Short vs long-range cosine: 0.993

The three-stream mixture is invariant to edge geometry. Short local
edges and long skip connections carry the same factor composition.
This is consistent with the residual stream being a position-
invariant communication bus where the factor content is set by global
circuit demands, not by local connectivity patterns.

## Synthesis

Three major findings emerge from the conditional analyses:

### 1. The "extended circuit" phenomenon

L9.H3 and other non-circuit heads dominate the DAS-EAP decomposition.
The factorized model's shared factor bank creates a situation where
all heads participate in the IOI task through the same factor
subspace — the "circuit boundary" between IOI and non-IOI heads is
blurred at the factor level.

This is NOT a bug in the analysis — it's a genuine property of the
factorized model. The shared factor bank means non-circuit heads
route the same factor information as circuit heads, just with
different attention patterns. The DAS rotation captures this shared
infrastructure.

### 2. Writer-reader asymmetry reveals directional routing

The residual stream is not a symmetric channel. Writers produce
broad, token_id-heavy signals; readers selectively consume the
three task-relevant factors (dup_token, positional, S_inhib). The
asymmetry quantifies the "narrow channel" hypothesis from the weight
coupling analysis: circuit function is implemented by reader
selectivity, not writer specificity.

### 3. Universal mixture with two structural axes

After removing scale (PC1), the two structural axes are S_inhib vs
everything else, and positional vs S_inhib. These axes are
continuous — the edge population is unimodal, not clustered. The
"edge modes" from the unsupervised analysis are positions along
these continuous axes, not discrete types.
