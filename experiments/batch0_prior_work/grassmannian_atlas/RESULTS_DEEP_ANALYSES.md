# Deep Analyses: Per-Layer, Per-Head, Spectral, and Interaction Structure

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 10: Per-Layer Subspace Composition

Aggregating DAS-dim attribution mass by reader layer reveals
remarkably stable composition across circuit depth:

| Reader layer | dup_token | positional | S_inhib |
|-------------|-----------|------------|---------|
| L0 | 29.2% | 17.8% | 24.7% |
| L3 | 34.1% | 21.7% | 21.9% |
| L5 | 31.9% | 21.4% | 25.6% |
| L8 | 32.0% | 23.8% | 21.3% |
| L9 | 34.1% | 23.7% | 19.8% |
| logits | 35.7% | 15.1% | 29.6% |

The three-stream composition holds across all layers, with narrow
variation: dup_token stays 27-36%, positional stays 15-24%,
S_inhib stays 17-30%.

The one notable transition is at the **logits layer**: positional
drops sharply (15.1%, from ~22-24% at L8-L9) while S_inhib rises
(29.6%, from ~20% at L9). The output edges carry more inhibition
signal and less positional signal — consistent with the circuit's
function: by the output, position has been consumed and the
remaining job is to suppress the repeated name.

Factor depth peaks: all factors peak at L8 in reader-marginal mass.
This means L8 is the circuit's "bottleneck" — the layer where the
most information is consumed, regardless of factor identity.

## Analysis 11: Head-Level Subspace Selectivity

### Key finding: circuit heads are NOT more selective

Read-profile entropy (Shannon entropy across 32 DAS dims):

| Group | Mean entropy | Std |
|-------|-------------|-----|
| Circuit heads | 4.563 | 0.066 |
| Non-circuit heads | 4.567 | 0.077 |
| Mann-Whitney p | 0.230 | — |

There is no significant difference in read selectivity between IOI
circuit heads and non-circuit heads at the DAS subspace level.
Both groups read broadly from the 32-dim space.

### Most selective readers (ALL non-circuit)

| Head | Entropy | Top factor | Role |
|------|---------|-----------|------|
| L2.H7 | 4.38 | dup_token 33.6% | non-circuit |
| L6.H3 | 4.39 | dup_token 40.7% | non-circuit |
| L5.H3 | 4.43 | S_inhib 28.4% | non-circuit |
| L7.H5 | 4.43 | dup_token 27.9% | non-circuit |
| L3.H6 | 4.44 | positional 28.1% | non-circuit |

The most selective readers are all non-circuit heads. L6.H3 has the
strongest dup_token dominance (40.7%) of any head in the model, yet
it is not part of the known IOI circuit.

### Most selective writers

All top-10 selective writers are L10-L11 heads (late non-circuit
heads). Their write profiles are strongly S_inhib + dup_token
bimodal, consistent with late layers consolidating the circuit's
two dominant signals for output.

### Interpretation

The DAS subspace is NOT a circuit-specific communication channel.
Both circuit and non-circuit heads use it equally. The circuit's
specialization must operate at a finer grain than the 32-dim DAS
decomposition — either through the specific directions within each
DAS dim, or through the attention pattern gating that determines
WHEN information flows.

## Analysis 12: Factor Flow by Circuit Depth

Tracking each factor's reader-marginal mass through layers:

| Factor | Onset | Peak | Pattern |
|--------|-------|------|---------|
| dup_token | L0 | L8 | 2% L0, 11% L5, 24% L8, 18% L9, 4% logits |
| S_inhib | L0 | L8 | 2% L0, 12% L5, 22% L8, 15% L9, 4% logits |
| positional | L0 | L8 | 2% L0, 11% L5, 26% L8, 18% L9, 2% logits |
| name_mover | L0 | L8 | 3% L0, 11% L5, 26% L8, 16% L9, 3% logits |
| token_id | L0 | L8 | 3% L0, 9% L5, 27% L8, 19% L9, 2% logits |

**All five factors have nearly identical depth profiles.** They all
onset at L0, ramp through L3-L7, peak at L8, and trail off through
L9-logits. This is the per-factor manifestation of PC1 = scale:
the variation across factors is dwarfed by the variation across
layers (where in the circuit the information is read).

The differences between factors are second-order effects within
this dominant shared trajectory.

## Analysis 13: Spectral Analysis

SVD of the 500-edge x 32-dim matrix (centered):

| SV | Variance | Cumulative |
|----|----------|------------|
| SV1 | 62.3% | 62.3% |
| SV2 | 12.3% | 74.7% |
| SV3 | 6.1% | 80.7% |
| SV4 | 5.0% | 85.7% |
| SV5 | 4.1% | 89.7% |
| SV6 | 1.9% | 91.6% |

Key metrics:
- **3 dims for 80% variance, 6 for 90%, 9 for 95%**
- Largest spectral gap: between SV1 and SV2 (gap ratio 0.555)
- Effective dimensionality: 1 (dominated by the scale component)
- 31 of 32 SVs above Marchenko-Pastur noise floor

The spectral structure confirms the PCA findings quantitatively.
The 32 DAS dims compress to ~3 effective dimensions for most
purposes, but nearly all dims carry some signal above noise.

## Analysis 14: Causal Path Tracing

### Surprise: L9.H3 dominates DAS-EAP reading

Top reader node for ALL three major factors:

| Factor | #1 reader | Mass |
|--------|-----------|------|
| dup_token | **L9.H3.Q** | 4.18 |
| positional | **L9.H3.Q** | 2.86 |
| S_inhib | **L9.H3.Q** | 2.21 |

L9.H3 is not part of the standard IOI circuit (Wang et al.), yet it
is the single strongest reader in the DAS-EAP decomposition across
all three major factors. L8.H10.Q and L8.H7.K are also top readers
and also non-circuit.

This suggests either: (a) the DAS rotation is capturing a broader
circuit than the manually-identified IOI heads, or (b) L9.H3
participates in the IOI task through the factorized model's
information routing even though it was not identified in the
original circuit analysis.

### All factors write through input

All three factors' top writer is the input embedding:
- dup_token: input = 0.329
- S_inhib: input = 0.249
- positional: input = 0.211

Confirms Analysis 1's R/W asymmetry: the initial embedding is the
sole meaningful write event; all subsequent structure is from
selective reading.

### Top edges per factor

| Factor | Top edge | Mass | Share |
|--------|----------|------|-------|
| dup_token | M8 -> L9.H3.Q | 1.27 | 38.2% |
| S_inhib | M3 -> L5.H3.Q | 0.66 | 33.6% |
| positional | M8 -> L9.H3.Q | 0.87 | 26.2% |

S_inhib's top edge (M3 -> L5.H3.Q) is distinct from the other two
factors' top edges, which both flow through M8 -> L9.H3.Q. The
S-inhibition pathway has its own characteristic routing that
separates earlier in the circuit (L5 vs L9).

## Analysis 15: MLP vs Attention Subspace Profiles

MLP-sourced and attention-sourced edges carry nearly identical DAS
subspace compositions (cosine similarity = 0.9945). The differences
are negligible:

- conv_S_inhib slightly higher in MLP edges (3.7% vs 2.9%, p<0.001)
- token_id slightly higher in attention edges (2.9% vs 2.3%, p=0.002)

The DAS subspace composition is writer-type-invariant. Whether
information comes from an MLP or an attention head, it carries the
same mixture of factor signals. This is consistent with the residual
stream being a shared communication bus where the subspace structure
is set by the DAS rotation, not by the writer.

## Analysis 16: Reader Projection Type Profiles

Comparing DAS profiles by what reads the information:

| Factor | Q | K | V | MLP | logits |
|--------|---|---|---|-----|--------|
| dup_token | 33.1% | 33.4% | 33.1% | 31.4% | 33.7% |
| positional | 23.1% | 22.6% | 23.3% | 20.8% | **15.6%** |
| S_inhib | 21.0% | 22.8% | 23.6% | 25.7% | **30.3%** |
| conv_S_inhib | 3.3% | 3.6% | 3.2% | 3.3% | **1.2%** |

Q, K, V, and MLP edges are nearly identical. The outlier is
**logits**: it has markedly lower positional (15.6%) and conv_S_inhib
(1.2%), and higher S_inhib (30.3%). The output layer consumes a
different subspace mixture than internal layers.

Pairwise cosine similarities confirm: Q/K/V/MLP cluster together
(cos > 0.94), while logits is distant (cos 0.68-0.84 with others).

## Analysis 17: Edge Specialization Gradient

Does edge specialization increase with circuit depth?

| Layer range | Mean entropy |
|------------|-------------|
| L0-L3 | 4.26 +/- 0.18 |
| L4-L7 | 4.25 +/- 0.12 |
| L8-L9 | 4.27 +/- 0.12 |
| logits | **3.91** +/- 0.31 |

Spearman correlation (layer vs entropy): r=0.045, p=0.894.

**No specialization gradient.** Edges do not become more specialized
with depth — the DAS profile entropy is flat from L0 through L9.
The sole exception is logits edges, which are significantly more
specialized (entropy 3.91 vs ~4.25).

This means the three-stream mixing is not a early-circuit phenomenon
that resolves into pure streams later. The mixing persists all the
way to L9 — only the final output projection specializes.

## Analysis 18: DAS Dim Interaction Network

### All dim-dim correlations are positive

The most negative dim-dim correlation is r = -0.012. There are NO
anti-correlated dimension pairs in the raw DAS-dim space.

Top positive correlations:

| Dim pair | r | Factor pair | Type |
|----------|---|-------------|------|
| d8-d11 | 0.904 | positional-late_comp | CROSS |
| d14-d15 | 0.858 | S_inhib-dup_token | CROSS |
| d11-d18 | 0.847 | late_comp-dup_token | CROSS |
| d8-d17 | 0.844 | positional-positional | SAME |
| d3-d11 | 0.842 | dup_token-late_comp | CROSS |

The strongest correlations are CROSS-factor, not within-factor.
d8 (positional) and d11 (late_comp) co-activate more strongly than
any same-factor pair.

### Within vs cross-factor correlation

| | Mean r | n |
|---|--------|---|
| Within-factor | 0.580 | 50 |
| Cross-factor | 0.468 | 446 |
| Mann-Whitney p | <0.0001 | — |

Within-factor correlations are significantly higher, but the
difference is modest (0.580 vs 0.468). Cross-factor correlations
are also high — the dims are not cleanly separated by factor.

### Where do the "trade-offs" come from?

Analysis 9 found strong negative correlations between factors
(positional vs S_inhib: r = -0.569). But Analysis 18 shows ALL
raw dim-dim correlations are positive. The trade-offs exist only in
**factor shares** (proportions), not in raw DAS-dim values.

This means: edges with more absolute S_inhib signal also have more
absolute positional signal (positive correlation in raw values),
but the PROPORTION of S_inhib is higher when the proportion of
positional is lower (negative correlation in shares). The anti-
correlation is a compositional artifact of the three-stream
structure, not a genuine mutual exclusion.

### Network density

80.6% of dim pairs have |r| > 0.3. The DAS dim interaction network
is extremely dense — almost all dims co-activate. This is the
per-dim manifestation of PC1 = scale: when an edge is important,
all dims tend to be elevated together.

## Synthesis

The deep analyses reveal a simpler picture than expected:

1. **One dominant axis**: Scale (PC1, 62% variance) dwarfs all
   structural variation. All factors, all layers, all projection
   types are dominated by "how important is this edge?" rather than
   "what does this edge carry?"

2. **Universal mixing**: The three-stream composition is remarkably
   stable across layers (Analysis 10), writer types (Analysis 15),
   projection types (Analysis 16), and circuit depth (Analysis 17).
   Only the logits layer departs from the universal mixture.

3. **Trade-offs are compositional**: The factor trade-offs (Analysis
   9) are proportional effects, not absolute exclusions. All dims
   co-activate positively (Analysis 18). The "modes" in the
   unsupervised analysis are modes within the small residual
   variation after the dominant scale component is removed.

4. **L9.H3 surprise**: A non-circuit head is the single strongest
   reader across all three major factors. The DAS-EAP decomposition
   sees a broader circuit than the standard IOI heads.

5. **The logits layer is special**: Lower positional share, higher
   S_inhib share, more specialized (lower entropy), distinct from
   all internal-layer edges. The output is the one place where the
   circuit narrows from the universal mixture.
