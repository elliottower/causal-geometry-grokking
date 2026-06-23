# Structural Deep Dive: Per-Head, Per-Projection, and Flow Path Analyses

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 48: Per-Head Reading Fingerprints

The top 20 heads by DAS-EAP reading mass are dominated by non-circuit
heads. Only 3 of 20 are IOI circuit heads (L8.H10, L8.H6, L9.H6).

### Top 5 heads

| Head   | Mass   | Circuit? | Top factor | 2nd | 3rd |
|--------|--------|----------|------------|-----|-----|
| L9.H3  | 12.622 | No       | dup_token 34.3% | positional 23.5% | S_inhib 18.2% |
| L8.H10 | 9.617  | Yes      | dup_token 35.1% | positional 24.3% | S_inhib 15.1% |
| L8.H7  | 9.477  | No       | dup_token 33.1% | positional 24.7% | S_inhib 20.0% |
| L9.H2  | 8.564  | No       | dup_token 32.7% | positional 28.4% | S_inhib 19.3% |
| L5.H3  | 8.449  | No       | S_inhib 30.6% | dup_token 29.5% | positional 21.3% |

L5.H3 is notable: it's the only top-5 head where S_inhib leads. This
non-circuit head is a major S_inhib consumer.

### Circuit heads are NOT more similar to each other

| Comparison | Mean cosine | Std |
|-----------|-------------|-----|
| Within-circuit | 0.805 | 0.094 |
| Within-non-circuit | 0.829 | 0.081 |
| Between | 0.817 | 0.088 |
| Mann-Whitney U, circuit vs between | p = 0.071 |

Circuit heads are slightly LESS internally similar than non-circuit
heads. The IOI circuit does not select heads with similar factor
profiles --- it selects heads along the same continuum as everyone
else.

## Analysis 49: Q/K/V Projection Factor Decomposition

### Aggregate factor shares are nearly identical across projections

| Factor | Q | K | V | Q-V diff |
|--------|---|---|---|----------|
| dup_token | 33.2% | 32.8% | 32.2% | +1.0% |
| positional | 23.1% | 22.4% | 22.5% | +0.6% |
| S_inhib | 21.4% | 23.0% | 23.9% | **-2.5%** |
| token_id | 2.6% | 2.8% | 2.7% | -0.0% |

The only notable asymmetry: **S_inhib is Q-depleted and V-enriched**
(Q=21.4% vs V=23.9%, delta=-2.5%). Q projections read slightly less
S_inhib than V projections. This makes functional sense: Q determines
"what to attend to" (primarily content-driven), while V determines
"what information to extract" (more action-relevant).

### Per-head Q-K-V divergence

| Pair | Mean cosine | Std |
|------|-------------|-----|
| Q-K | 0.813 | 0.115 |
| Q-V | 0.815 | 0.121 |
| K-V | 0.808 | 0.113 |

All three pairs are equally similar on average. No systematic
projection asymmetry.

Most divergent head: **L10.H4** (Q-K cos=0.307, Q-V cos=0.412).
This head reads fundamentally different factors through Q vs K/V.
L0 heads also show high divergence (early heads haven't converged).

## Analysis 50: MLP vs Attention Reader Profiles Per Layer

| Layer | MLP-Attn cos | MLP/Attn mass | Biggest diff |
|-------|-------------|---------------|--------------|
| L0 | 0.962 | 9.255 | dup_token +5.6% |
| L1 | 0.813 | 0.646 | S_inhib +9.0% |
| **L2** | **0.845** | **0.428** | **S_inhib +19.6%** |
| L3 | 0.863 | 0.269 | positional -7.7% |
| L5 | 0.893 | 0.205 | dup_token +6.9% |
| L9 | 0.879 | 0.035 | late_comp -2.6% |
| **L11** | **0.786** | **0.459** | **dup_token -7.6%** |

Two key findings:

1. **L2 MLP reads 19.6% more S_inhib than L2 attention**. This is
   the L2→L3 transition layer. The MLP is S_inhib-heavy while
   attention begins shifting to dup_token. The transition is not just
   between layers --- it happens within L2 between component types.

2. **L0 MLP dominates L0 attention by 9.3x** in reading mass. Early
   processing is MLP-driven. By L8-L9, attention dominates by 9-28x.

## Analysis 51: Layer-Pair Factor Transitions

### Consecutive layer transitions (ordered by magnitude)

| Transition | Cosine | Biggest change |
|-----------|--------|----------------|
| L11 → logits | **0.677** | S_inhib **+13.0%** |
| L10 → L11 | 0.871 | auxiliary -5.8% |
| L2 → L3 | 0.922 | S_inhib -6.4% |
| L3 → L4 | 0.954 | S_inhib +5.2% |
| L0 → L1 | 0.943 | positional +3.6% |
| L9 → L10 | 0.940 | dup_token -7.9% |

Three distinct transition regimes:

1. **L0-L2**: factor profiles are establishing (cos 0.94-0.95)
2. **L3-L9**: stable core processing (cos 0.96-0.98)
3. **L10-logits**: rapid recomposition (cos 0.68-0.87)

The L11→logits transition is the biggest single step, with S_inhib
surging +13.0% --- consistent with the logits exception found earlier.

### The middle layers form a plateau

L3-L6 cosine matrix shows pairwise cos > 0.98. The circuit's mid-
section maintains a near-constant factor mixture for four consecutive
layers. Structure changes happen at the edges (L0-L2, L10-logits).

## Analysis 52: Writer Factor Fingerprints

### Writer specialization by layer

| Layer | Attn writer top factor | MLP writer top factor | Note |
|-------|----------------------|---------------------|------|
| L0 | dup_token 24.2% | dup_token 22.8% | Both generic |
| L5 | dup_token 38.5% | dup_token 38.0% | High dup_token |
| L6 | dup_token 27.7% | **S_inhib 34.7%** | MLP writes S_inhib |
| L9 | **name_mover 29.0%** | token_id 45.2% | Functional specialization |
| L11 | **conv_S_inhib 53.4%** | dup_token 17.1% | Late S-convergence |

L9 attention writers are name_mover-dominant (29.0%) --- these are
the actual name mover heads. L11 attention writes heavily on
conv_S_inhib (53.4%) --- late convergent inhibition.

The only factor whose peak write is NOT L0 is **conv_S_inhib**
(peaks at L6). This convergent signal is built mid-circuit.

Global attn-writer vs mlp-writer cosine: **0.996**. Writers are
globally identical regardless of type.

## Analysis 53: Edge-Pair Similarity by Shared Nodes

| Relationship | n pairs | Cosine | Std |
|-------------|---------|--------|-----|
| Same-reader | 533 | **0.917** | 0.120 |
| Same-writer | 1,493 | 0.718 | 0.116 |
| No shared node | 17,874 | 0.688 | 0.120 |

### Reader selectivity dominates writer consistency

Same-reader pairs have cos=0.917 --- far higher than same-writer
(0.718) or random (0.688). Both effects are highly significant
(p < 10^-20 for same-writer, p < 10^-221 for same-reader).

**Same-writer > same-reader: p=1.0** (the test fails completely).
Same-reader similarity is overwhelmingly stronger than same-writer
similarity.

This resolves an apparent tension: writers are consistent (Analysis
40: within-writer cos=0.861), but readers are even more deterministic.
Two edges feeding the same Q projection have nearly identical factor
profiles regardless of which writer produced them. The reader's
selectivity is the dominant force shaping edge factor composition.

## Analysis 54: Factor Flow Paths

### All main factors peak write at L0, peak read at L8

| Factor | Peak write | Peak read | R/W ratio |
|--------|-----------|-----------|-----------|
| dup_token | L0 | L8 | 83.5 |
| positional | L0 | L8 | **87.1** |
| S_inhib | L0 | L8 | 85.5 |
| token_id | L0 | L8 | 16.5 |
| late_comp | L0 | L8 | 48.0 |
| conv_S_inhib | **L6** | L8 | 43.9 |

conv_S_inhib is the exception: it peaks at L6 (built by mid-circuit
MLPs and attention, not inherited from embeddings).

### Cumulative read profiles are identical

| Factor | 25% consumed | 50% consumed | 75% consumed |
|--------|-------------|-------------|-------------|
| dup_token | L5 | L7 | L8 |
| S_inhib | L5 | L7 | L8 |
| positional | L5 | L7 | L8 |

All three main factors have the **exact same consumption profile**.
The circuit doesn't read different factors at different times ---
it consumes everything together. Temporal differentiation is not a
feature of the IOI circuit's factor flow.

## Analysis 55: Head Clustering by Factor Profile

### No meaningful clusters exist

At 3, 5, or 8 clusters, all clusters have similar factor profiles
(inter-cluster cosine 0.74--0.95 for 5 clusters), span the full
layer range (L0-L11), and contain roughly proportional numbers of
circuit heads.

The 5-cluster solution:

| Cluster | n | Circuit | Layer range | Top factors |
|---------|---|---------|-------------|-------------|
| C1 | 15 | 0/15 | L1-L11 | dup_token 34.0%, S_inhib 26.6% |
| C2 | 46 | 6/46 | L0-L11 | dup_token 31.1%, positional 24.2% |
| C3 | 42 | 9/42 | L0-L11 | dup_token 31.0%, S_inhib 24.0% |
| C4 | 5 | 2/5 | L0-L10 | S_inhib 25.5%, dup_token 22.1% |
| C5 | 36 | 5/36 | L0-L11 | dup_token 30.2%, S_inhib 24.2% |

C4 (5 heads, 2 circuit) is the most distinct: S_inhib-dominant with
lowest inter-cluster cosine (0.74 to C1). But it's tiny and its
enrichment for circuit heads (40%) vs baseline (15%) is not robust
at n=5.

This strongly reinforces the unimodal edge population finding from
Analysis 27. Heads, like edges, form a continuum --- not discrete
functional types.

## Updated Synthesis (55 analyses across 8 batches)

### 18. Reader Selectivity > Writer Consistency

Same-reader edge pairs (cos=0.917) are far more similar than same-
writer pairs (cos=0.718). The reader's projection matrix is the
primary determinant of edge factor composition, not the writer's
output. Writers broadcast; readers define.

### 19. No Temporal Differentiation in Factor Consumption

All three main factors (dup_token, positional, S_inhib) have
identical cumulative read profiles: 25%@L5, 50%@L7, 75%@L8. The
circuit consumes everything at the same rate. There is no "dup_token
first, then S_inhib" temporal structure at the factor level.

### 20. Three Transition Regimes

The 12-layer circuit has three dynamical regimes:
- **Establishment (L0-L2)**: factor profiles shift, S_inhib peaks
- **Plateau (L3-L9)**: near-constant mixture (pairwise cos > 0.96)
- **Recomposition (L10-logits)**: rapid profile change, S_inhib +13%

### 21. conv_S_inhib Is the Only Emergent Factor

Every other factor peaks in writing at L0 (embedding). conv_S_inhib
peaks at L6 --- it is built mid-circuit, not inherited. This is the
convergent S-inhibition signal that the circuit constructs.

### 22. L2 MLP Is the S_inhib Hotspot

Within L2, MLP reads 19.6% more S_inhib than attention. The L2→L3
transition is not just between layers --- it has an intra-layer
component where MLP and attention handle different factor mixtures.
