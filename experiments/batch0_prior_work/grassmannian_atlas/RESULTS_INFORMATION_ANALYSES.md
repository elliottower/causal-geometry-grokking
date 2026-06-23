# Information-Theoretic and Signed Structure Analyses

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 56: Signed Factor Flow Patterns

Within each factor, paired DAS dims have opposite sign behavior
in writing vs reading:

| Factor | Dim pair | Writer sign ratio | Reader sign ratio |
|--------|----------|------------------|------------------|
| dup_token | d3 vs d26 | 2.05 vs 0.32 | 0.38 vs 1.61 |
| S_inhib | d14 vs d25 | 8.67 vs 0.47 | 0.84 vs 1.96 |

d3 is predominantly positive in writing but negative in reading.
d26 is the reverse. The DAS rotation finds paired dimensions where
one is positive-dominant and its partner is negative-dominant, and
writing/reading flip the dominance.

### Most positive vs negative edges

| Edge | Signed mass | Interpretation |
|------|-------------|---------------|
| input → M0.in | **+0.072** | Embedding sends positive signal to MLP0 |
| L9.H9 → logits | +0.040 | Name mover to output (positive) |
| L8.H6 → L9.H3.Q | +0.026 | S_inhib head feeds L9.H3 |
| M3 → M4.in | **-0.027** | MLP3→MLP4 is most negative edge |
| L8.H6 → logits | -0.012 | S_inhib head to logits (negative) |

L8.H6 sends positive signal to L9.H3 but negative signal to logits.
The same writer produces opposite-signed flow depending on reader.

## Analysis 57: Within-Factor Sign Correlations

### Same-factor dim pairs anti-correlate in sign

| Factor | Pair | r |
|--------|------|---|
| dup_token | d3-d26 | **-0.75** |
| dup_token | d21-d26 | -0.70 |
| S_inhib | d12-d24 | **-0.76** |
| positional | d8-d17 | -0.67 |
| positional | d4-d20 | -0.50 |

Within every multi-dim factor, at least one pair of dims strongly
anti-correlates in sign. The magnitude is shared (both large on
the same edges) but the sign flips --- encoding directional
information.

### Cross-factor sign correlations reveal hidden structure

| Pair | Strongest r | Dims |
|------|------------|------|
| positional-late_comp | **+0.72** | d3-d11 |
| dup_token-positional | -0.74 | d3-d8 |
| S_inhib-positional | -0.70 | d24-d30 |
| S_inhib-conv_S_inhib | **-0.79** | d12-d7 |
| dup_token-S_inhib | +0.60 | d18-d12 |

**S_inhib and conv_S_inhib anti-correlate** (r=-0.79). Despite the
name "convergent S_inhib", this factor opposes S_inhib in signed
space. They track different aspects of inhibition: S_inhib is the
core suppression signal, conv_S_inhib is a convergence/agreement
signal that works in the opposite direction.

## Analysis 58: Circuit Role Factor Profiles

### Functional roles do NOT align with namesake factors

| Circuit role | Expected top factor | Actual top factor | Namesake share |
|-------------|--------------------|--------------------|----------------|
| dup_token heads | dup_token | **dup_token (39.0%)** | 39.0% |
| S_inhib heads | S_inhib | **dup_token (32.7%)** | 19.3% |
| name_mover heads | name_mover | **dup_token (34.0%)** | 4.0% |

Only dup_token heads read their namesake factor most. S_inhib heads
read more dup_token than S_inhib. Name mover heads have only 4.0%
name_mover factor --- they primarily read dup_token and positional.

| Role | Mass | dup_token | S_inhib | positional |
|------|------|-----------|---------|------------|
| dup_token | 1.47 | **39.0%** | 16.0% | 23.4% |
| S_inhib | 16.75 | 32.7% | 19.3% | 23.7% |
| name_mover | 3.92 | **34.0%** | 19.5% | 19.8% |
| induction | 2.43 | 28.8% | **27.9%** | 21.0% |
| prev_token | 0.79 | 29.9% | **27.3%** | 16.1% |
| backup NM | 0.14 | 30.1% | 14.4% | 21.4% |

S_inhib heads have 11x more mass than dup_token heads (16.75 vs
1.47). The circuit roles (dup_token detection, S-inhibition, name
moving) are NOT implemented by selectively reading a single factor.
Every role reads the full mixture; the computation happens through
how factors combine, not through factor isolation.

## Analysis 59: Per-Dim Importance Ranking

### Top 8 dims capture 50% of mass

| Rank | Dim | Factor | Share | Cumulative |
|------|-----|--------|-------|------------|
| 1 | d26 | dup_token | 8.9% | 8.9% |
| 2 | d12 | S_inhib | 7.1% | 16.0% |
| 3 | d20 | positional | 6.5% | 22.5% |
| 4 | d14 | S_inhib | 6.5% | 29.0% |
| 5 | d15 | dup_token | 6.1% | 35.1% |
| 6 | d24 | S_inhib | 6.0% | 41.0% |
| 7 | d8 | positional | 5.1% | 46.1% |
| 8 | d21 | dup_token | 4.8% | 50.9% |

### Per-dim efficiency

| Factor | Total share | Dims | Share per dim |
|--------|------------|------|---------------|
| S_inhib | 23.0% | 4 | **5.8%** |
| dup_token | 32.6% | 6 | 5.4% |
| positional | 22.2% | 6 | 3.7% |
| auxiliary | 5.7% | 2 | 2.8% |
| token_id | 2.7% | 4 | **0.7%** |

S_inhib is the most efficient: each of its 4 dims carries 5.8% of
total mass. token_id is the least efficient at 0.7% per dim --- it
has 4 dims but uses them very lightly.

## Analysis 60: Per-Edge Factor Entropy

### Edges are spread across most dims

| Metric | Value |
|--------|-------|
| Max possible entropy | 5.000 bits |
| Mean entropy (active edges) | 4.297 (85.9% of max) |
| P10 | 4.102 (82.0%) |
| P90 | 4.475 (89.5%) |
| Top-50 edges mean entropy | 4.254 |

Edge factor profiles are high-entropy: information is distributed
across most of the 32 DAS dims. No edge concentrates its mass on
a few dims.

Entropy vs mass: r = -0.15 (p < 10^-82). Important edges are
slightly more focused, but the effect is small. Even the most
important edges use 73% of max entropy.

## Analysis 61: Within-Layer vs Cross-Layer Edge Profiles

Within-layer and cross-layer edges have **identical factor profiles**
(cosine = 0.990). Factor composition by edge distance:

| Distance | n | dup_token | S_inhib |
|----------|---|-----------|---------|
| 1 | 4,496 | 30.7% | 22.9% |
| 2 | 4,146 | 30.3% | 23.3% |
| 3 | 3,624 | 30.3% | 23.3% |
| 5 | 2,603 | 29.3% | 23.7% |
| 8 | 1,146 | 29.4% | 23.3% |

No distance effect whatsoever. Factor composition is invariant to
edge length, confirming Analysis 26.

## Analysis 62: Signed Factor Interaction Network

### Absolute vs signed correlations reveal opposite structure

The **absolute** factor-factor correlations are all strongly positive
(range 0.41--0.94). This is the scale effect: important edges carry
more of everything.

The **signed** correlations reveal a completely different structure:

| Pair | Absolute r | Signed r | Reversal? |
|------|-----------|---------|-----------|
| positional-late_comp | +0.930 | **-0.932** | YES |
| auxiliary-S_inhib | +0.784 | **-0.775** | YES |
| dup_token-late_comp2 | +0.573 | -0.605 | YES |
| dup_token-positional | +0.939 | -0.546 | YES |

The strongest effect: positional and late_comp have r=+0.93 in
magnitude but r=-0.93 in sign. They perfectly co-activate in
magnitude and perfectly oppose in sign. This means they carry
information in the same edges but encode opposite directions.

### Signed network clusters (different from absolute-value teams)

In signed space, the factor alliances reshape:
- {dup_token, late_comp, S_inhib, name_mover}: mutually positive
- {positional, token_id, auxiliary, late_comp2}: mutually positive

The signed teams are NOT the same as the absolute-value teams
(Analysis 36). In absolute values, S_inhib opposed dup_token.
In signed values, they are allies (r=+0.47 through late_comp).
The "two-team competition" is an artifact of absolute values ---
the real structure in signed space is a different partition.

## Analysis 63: Circuit vs Non-Circuit Edge Profiles

### Circuit and non-circuit edges are virtually identical

| Metric | Circuit | Non-circuit |
|--------|---------|-------------|
| n edges | 3,617 | 19,216 |
| Profile cosine | 0.993 | |
| Mean entropy | 4.309 ± 0.154 | 4.315 ± 0.150 |
| p (entropy diff) | 0.043 | |

| Factor | Circuit | Non-circuit | Difference |
|--------|---------|-------------|------------|
| dup_token | 29.2% | 30.1% | -0.9% |
| S_inhib | 23.9% | 23.2% | +0.6% |
| late_comp | 6.5% | 6.0% | +0.5% |
| name_mover | 4.6% | 4.1% | +0.4% |

Circuit edges carry slightly more S_inhib (+0.6%) and name_mover
(+0.4%) at the expense of dup_token (-0.9%). But the differences
are tiny (cosine 0.993). The factor decomposition does not
distinguish circuit from non-circuit edges.

## Updated Synthesis (63 analyses across 9 batches)

### 23. Sign Flips Encode Direction, Not Absence

DAS dims within each factor anti-correlate in sign (r ≈ -0.75)
while co-activating in magnitude. Sign encodes which direction an
edge points in factor space --- which of two counterfactual
conditions is active.

### 24. Circuit Roles ≠ Factor Names ≠ Edge Modes

Three levels of description that don't align:
- **Factor names** (from DAS): dup_token, S_inhib, etc.
- **Circuit roles** (from Wang et al.): dup_token heads, S_inhib heads
- **Edge modes** (from unsupervised clustering): continuous axes

S_inhib heads read more dup_token than S_inhib (32.7% vs 19.3%).
Name mover heads have only 4% name_mover factor. conv_S_inhib
anti-correlates with S_inhib in signed space (r=-0.79).

### 25. The Signed Network Reshuffles the Teams

In absolute values: content (dup_token+positional) vs action
(S_inhib+conv_S_inhib). In signed values: {dup_token, late_comp,
S_inhib} vs {positional, auxiliary, token_id}. The strongest signed
correlation is positional-late_comp (r=-0.932), a near-perfect
sign opposition invisible in absolute values.

### 26. Factor Profiles Are Universal Across ALL Contrasts

Now confirmed invariant to: layer, writer type, projection type
(Q/K/V), edge length, within vs cross-layer, circuit vs non-circuit
membership, edge importance, MLP vs attention, and edge distance.
The only exceptions remain: logits edges and L2→L3 transition.
