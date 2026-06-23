# Geodesic vs Cosine Audit: Grassmannian Subspace Distances

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Meta-Finding: When Does Geodesic ≠ Cosine?

For single-vector comparisons, geodesic distance = arccos(cosine) is
a monotonic transform. Cos-geo correlation across all analyses:
r = -0.9957. **The ranking never changes.** What changes is scale:
geodesic makes moderate similarity look more dramatic (cos=0.70
becomes 45°, halfway to orthogonal) and high similarity look tighter
(cos=0.95 becomes 18°, a small rotation).

The real payoff is **Grassmannian distance** (Analysis 78), which
measures subspace separation via principal angles. This reveals
structure invisible to single-vector cosine: two factors can have
cos=0.95 on their mean profiles but Grassmann=2.617 between their
subspaces.

## Analysis 72: Within-Writer Consistency

| Metric | Mean | Std |
|--------|------|-----|
| Cosine | 0.695 | 0.031 |
| Geodesic | 0.787 rad (45.1°) | 0.046 rad (2.6°) |

Writers are 45° apart on average across their outgoing edges. This
is more dramatic than cos=0.70 suggests --- halfway between identical
and orthogonal. The standard deviation is tight (±2.6°), meaning all
144 writers show similar levels of consistency.

## Analysis 73: Reader Selectivity Strengthens Under Geodesic

| Category | n | Cosine | Geodesic |
|----------|---|--------|----------|
| Same-reader | 541 | 0.916 | 18.9° |
| Same-writer | 1514 | 0.722 | 42.9° |
| No shared node | 17845 | 0.697 | 45.0° |

The reader-selectivity finding is **stronger** under geodesic.
Same-reader edges are 18.9° apart (nearly aligned). Same-writer edges
are 42.9° apart (nearly at the population baseline of 45.0°). The
gap is 24.0° — readers constrain edge profiles far more than writers.

Both contrasts are highly significant: same-reader vs unrelated
p = 2.8 × 10⁻²¹⁸, same-writer vs unrelated p = 3.7 × 10⁻¹⁵.

## Analysis 74: Layer Transitions

### Adjacent transitions

| Transition | Cosine | Geodesic |
|-----------|--------|----------|
| L0 → L1 | 0.949 | 18.5° |
| L1 → L2 | 0.955 | 17.3° |
| **L2 → L3** | **0.924** | **22.5°** |
| L3 → L4 | 0.955 | 17.4° |
| L4 → L5 | 0.964 | 15.4° |
| L5 → L6 | 0.983 | 10.5° |
| L6 → L7 | 0.983 | 10.5° |
| L7 → L8 | 0.979 | 11.8° |
| L8 → L9 | 0.966 | 15.0° |
| L9 → L10 | 0.941 | 19.8° |
| L10 → L11 | 0.874 | 29.1° |
| **L11 → logits** | **0.683** | **47.0°** |

Three regimes confirmed under geodesic:

1. **Establishment (L0-L2)**: 17-19° transitions. L2→L3 is the
   within-network maximum at 22.5°.
2. **Plateau (L3-L9)**: 10-17° transitions. L5-L7 is the tightest
   region at 10-11°.
3. **Recomposition (L10-logits)**: 19-47° transitions. L11→logits
   is 47° — nearly halfway to orthogonal. The logits layer genuinely
   lives in a different part of factor space.

### Long-range transitions

| Transition | Cosine | Geodesic |
|-----------|--------|----------|
| L0 → L6 | 0.863 | 30.4° |
| L0 → L11 | 0.805 | 36.4° |
| L3 → L9 | 0.961 | 16.0° |
| L6 → logits | 0.727 | 43.4° |

The plateau holds: L3→L9 spans 7 layers but only 16° of rotation.
The full network span (L0→L11) is only 36° — less than the single
L11→logits transition (47°). The logits layer is genuinely anomalous.

## Analysis 75: Circuit vs Non-Circuit

| Comparison | Cosine | Geodesic |
|-----------|--------|----------|
| Mean profile distance | 0.956 | 17.1° |
| Within-circuit pairwise | — | 48.1° |
| Within-non-circuit pairwise | — | 47.9° |
| Between pairwise | — | 48.4° |

Circuit and non-circuit edges have nearly identical mean factor
profiles (17° apart) and identical pairwise variability (~48°).
Under geodesic distance, there is **no evidence** that circuit edges
form a distinct cluster. The circuit signal is in the signs and
substructure, not in the factor mixture proportions.

## Analysis 76: Writer-Reader Asymmetry

| Metric | Mean | Std |
|--------|------|-----|
| Cosine | 0.567 | 0.150 |
| Geodesic | 0.956 rad (54.8°) | 0.188 rad (10.8°) |

Writer and reader profiles for the same edge are **54.8°** apart on
average — past orthogonal's halfway mark. Cosine=0.57 understates
how different writers and readers are. The transformation from what
a writer produces to what a reader consumes involves a >54° rotation
in factor space.

Cos-Geo correlation: r = -0.9957. The transform is monotonic, but
the geodesic representation makes the asymmetry more viscerally clear.

## Analysis 77: Head Clustering

| Category | Cos dist | Geodesic |
|----------|----------|----------|
| Within-circuit | 0.181 | 34.0° |
| Within-non-circuit | 0.165 | 32.3° |
| Between | 0.172 | 33.1° |

| Statistic | Value |
|-----------|-------|
| Range | 7.7° — 61.2° |
| P50 | 32.6° |
| P90 | 43.9° |

No discrete head clusters under geodesic distance either. All three
categories are within 2° of each other. The maximum pairwise distance
is 61° — well short of 90° (orthogonal), meaning no head pair has a
truly opposite factor profile.

## Analysis 78: Grassmannian Factor Subspace Distances

This is the **new finding** not available from cosine analysis.

Each factor defines a subspace in edge space (R¹⁰⁰⁰ over top-1000
edges) spanned by its DAS dimensions. Grassmannian distance measures
how far apart these subspaces are via principal angles.

### Distance matrix (radians)

|  | dup_tok | pos | S_inh | tok_id | late_c | name_m | conv_S | aux |
|--|---------|-----|-------|--------|--------|--------|--------|-----|
| **dup_token** | 0 | 2.62 | 1.73 | 2.28 | 1.29 | 1.47 | 1.01 | 1.06 |
| **positional** | 2.62 | 0 | 2.02 | 2.11 | 1.47 | 1.72 | 1.27 | 0.99 |
| **S_inhib** | 1.73 | 2.02 | 0 | 2.19 | 1.68 | 1.80 | 1.19 | 1.24 |
| **token_id** | 2.28 | 2.11 | 2.19 | 0 | 1.86 | 1.94 | 1.51 | 1.34 |
| **late_comp** | 1.29 | 1.47 | 1.68 | 1.86 | 0 | 1.48 | 1.49 | 1.23 |
| **name_mover** | 1.47 | 1.72 | 1.80 | 1.94 | 1.48 | 0 | 1.54 | 1.31 |
| **conv_S_inhib** | 1.01 | 1.27 | 1.19 | 1.51 | 1.49 | 1.54 | 0 | 1.50 |
| **auxiliary** | 1.06 | 0.99 | 1.24 | 1.34 | 1.23 | 1.31 | 1.50 | 0 |

### Normalized distances (fraction of max = min(k₁,k₂)·π/2)

| Pair | Grassmann | Max | Fraction |
|------|-----------|-----|----------|
| dup_token — positional | 2.62 | 9.42 | 28% |
| dup_token — S_inhib | 1.73 | 6.28 | 27% |
| dup_token — token_id | 2.28 | 6.28 | 36% |
| positional — S_inhib | 2.02 | 6.28 | 32% |
| S_inhib — token_id | 2.19 | 6.28 | 35% |

### Key findings

1. **Cosine hides subspace separation.** dup_token-positional has
   cos=0.95 (near-identical mean profiles) but Grassmann=2.62 rad
   (28% of maximum). The subspaces these factors span in edge space
   are substantially different even though their marginal profiles
   look the same.

2. **token_id is the most isolated factor.** Its mean Grassmannian
   distance to all other factors is 1.90 rad — the highest of any
   factor. This confirms L4.H6's role as a genuine specialist.

3. **conv_S_inhib and auxiliary are closest to their parent factors.**
   conv_S_inhib-dup_token = 1.01, auxiliary-positional = 0.99. These
   minor factors live in nearby subspaces, suggesting they are
   satellites of the main factors rather than independent signals.

4. **The three-stream model has subspace support.** The three main
   factors (dup_token, positional, S_inhib) are pairwise separated
   at 27-32% of maximum. Not orthogonal, but genuinely distinct
   subspaces. token_id is even more separated at 35-36%.

5. **Two-team structure partially holds.** Content factors (dup_token,
   positional) are close to their satellites (late_comp, auxiliary)
   but far from action factors (S_inhib, conv_S_inhib). However the
   separation is not as clean in Grassmannian space as in the signed
   correlation network.

## Analysis 79: Cross-Task Factor Usage

| Pair | Cosine | Geodesic |
|------|--------|----------|
| capital_country — hypernymy | 0.994 | 6.1° |
| ioi — capital_country | 0.992 | 7.4° |
| greater_than — capital_country | 0.991 | 7.6° |
| ioi — hypernymy | 0.988 | 8.7° |
| greater_than — hypernymy | 0.986 | 9.5° |
| ioi — greater_than | 0.984 | 10.4° |
| sva — hypernymy | 0.983 | 10.6° |
| sva — capital_country | 0.983 | 10.7° |
| ioi — sva | 0.979 | 11.8° |
| sva — greater_than | 0.975 | 12.9° |
| capital_country — gender_bias | 0.976 | 12.6° |
| ioi — gender_bias | 0.976 | 12.6° |
| gender_bias — hypernymy | 0.980 | 11.6° |
| sva — gender_bias | 0.970 | 14.2° |
| greater_than — gender_bias | 0.965 | 15.2° |

All task pairs are within 6-15° geodesically. The factor bank is
shared infrastructure (confirming result 15 from prior analyses).
Geodesic range [6°, 15°] is compact relative to the 90° maximum.

Ordering is preserved from cosine (r=0.99). Capital_country-hypernymy
is closest (6.1°); greater_than-gender_bias most distant (15.2°).
Previously reported IOI-SVA as "most distinctive" holds at 11.8° but
is not actually the maximum — gender_bias is the most distinctive
task geodesically.

## Synthesis: What Geodesic/Grassmannian Adds

### Confirmed (no qualitative change)

- Reader selectivity >> writer consistency (stronger: 18.9° vs 42.9°)
- Three transition regimes (establishment, plateau, recomposition)
- L2→L3 is the largest within-network transition
- No head clusters in any metric
- Circuit vs non-circuit edges are indistinguishable in factor profiles
- Cross-task factor usage is highly similar
- The logits layer is anomalous

### Strengthened

- **Writer-reader asymmetry**: 54.8° is past the halfway mark to
  orthogonal. The transformation IS the computation, more dramatically
  than cos=0.57 suggests.
- **Logits exception**: 47° from L11 — a single-step rotation larger
  than the cumulative drift from L0→L11 (36°).

### New (from Grassmannian)

- **Cosine masks subspace separation.** Factor pairs with cos≈0.95
  occupy substantially different subspaces (28-36% of max Grassmann
  distance). The three-stream model has genuine subspace support,
  not just marginal-profile support.
- **token_id is the most subspace-isolated factor** (mean Grassmann
  = 1.90 rad). It genuinely occupies a different region of edge space.
- **Minor factors are subspace satellites** of major factors
  (conv_S_inhib→dup_token: 1.01 rad, auxiliary→positional: 0.99 rad).

### Caveat

For single-vector comparisons (all analyses except 78), geodesic
distance is arccos(cosine) — a nonlinear rescaling, not new
information. Rankings are preserved (r > 0.99). The value is
interpretability: degrees are more intuitive than cosine values.
Grassmannian distance (multi-dimensional subspace comparison) is
the genuinely new metric.
