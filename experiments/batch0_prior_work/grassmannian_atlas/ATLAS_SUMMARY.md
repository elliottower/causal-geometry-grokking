# Causal Geometry Atlas: Master Summary

89 analyses across 12 batches on the 1024-factor dense-selector GPT-2
circuit, using DAS-EAP scores (32 DAS dims, 157x445 edge grid) for
IOI and raw factor EAP scores (1024 factors) across 6 tasks.

## Core Results

### 1. Three-Stream Multiplexing

Every circuit edge carries three concurrent signals: dup_token (34%),
positional (22%), S_inhib (22%). Discovered unsupervised; labels are
post-hoc. No "pure" edges exist --- every edge is a mixture.

### 2. Scale Dominates, Structure Is Secondary

PC1 = 62% variance = "how important is this edge?" All factors load
positively. The structural signal (PC2 = 12%) separates S_inhib
from dup_token/positional. Three dims capture 80% of variance, six
capture 90%.

### 3. Partial Correlations Unmask True Structure

After controlling for total edge importance, the S_inhib-positional
trade-off strengthens from r=-0.57 to r_partial=-0.72. Scale was
diluting the signal. Two factor clusters emerge:
- Cluster A: dup_token + positional + late_comp
- Cluster B: S_inhib + conv_S_inhib

### 4. Signed Structure Reveals Direction Encoding

In |absolute values|, all dim-dim correlations are positive. In
SIGNED values, same-factor dims anti-correlate (dup_token d3-d26:
r=-0.749, S_inhib d12-d24: r=-0.757). The strongest signed
anti-correlation is cross-factor: positional-late_comp d8-d11:
r=-0.911 (same pair has r=+0.904 in absolute values).

DAS dims encode information as sign contrasts between paired dims.

### 5. Extended Circuit (77% Non-Circuit)

IOI circuit heads carry only 16% of DAS-EAP mass. Non-circuit heads
carry 54%, MLPs 28%. L9.H3 (non-circuit) has 42x the mass of L9.H9
(name mover). 20 heads needed for 50% of mass (only 5 circuit). The
DAS subspace is shared infrastructure, not a private circuit channel.

### 6. Reader Selectivity, Not Writer Specificity

R/W ratio = 67.8x across all DAS dims. Writer-reader cosine
similarity = 0.57. The three main factors are reader-amplified
(+0.05-0.07 asymmetry), token_id is writer-biased (-0.08). Circuit
function is implemented by what readers choose to consume.

### 7. Universal Mixture (No Specialization Gradient)

Factor composition is invariant across:
- Layers (Analysis 10): stable from L0-L9, except logits
- Writer types (Analysis 15): MLP vs attention cos=0.995
- Projection types (Analysis 16): Q/K/V/MLP cos>0.94
- Edge length (Analysis 26): no length effect
- Importance quartiles (Analysis 20): stable across 4 orders of magnitude
- Within vs cross-layer (Analysis 27): cos=0.982

The sole exception: logits edges (-7.5% positional, +7.1% S_inhib).

### 8. Factor Roles ≠ Edge Modes

ARI = 0.028 between factor groupings and edge clusters (near chance).
Factor roles (from weight space) describe what information each
factor carries. Edge modes (from co-occurrence) describe how factors
combine on edges. Alphabet vs words.

### 9. Edge Population Is Unimodal

After removing PC1, clustering produces one giant cluster (491/500)
with 9 outliers. Cophenetic correlation = 0.658. The "modes" are
positions along continuous axes, not discrete clusters.

### 10. Information Bottleneck Is Distributed

25% of any factor's mass requires ~70 edges, 50% requires ~370
edges, 75% requires ~1600 edges. The same edges (M8→L9.H3.Q,
L8.H6→L9.H3.Q) dominate all three factors' bottlenecks.

## Key Transitions

| Transition | What changes | Magnitude |
|-----------|-------------|-----------|
| L2→L3 | S_inhib drops, dup_token rises | -12.7% S_inhib |
| L9→logits | Positional drops, S_inhib rises | -10% pos, +11% S_inhib |
| Embedding→all | Write is diffuse, reading is selective | 67.8x R/W ratio |

## Methodological Notes

- Bootstrap stability: PC1/PC2 self-correlation >0.98 (100 resamples)
- Edge count sensitivity: structure stable from n=100 to n=2000
- Identity vs DAS-EAP: Spearman rho=0.571, top-20 overlap 15%
- DAS rotation reveals MLP contributions invisible in raw basis

## File Inventory

| File | Contents |
|------|----------|
| RESULTS_UNSUPERVISED_EDGE_MODES.md | Analyses 1-7: unsupervised discovery |
| RESULTS_DAS_EDGE_DECOMPOSITION.md | Per-edge subspace profiles, SEM |
| RESULTS_FACTOR_COUPLING.md | Weight OV coupling, cross-task routing |
| RESULTS_FACTOR_DAG.md | Factor DAG from DAS-EAP scores |
| RESULTS_ROBUSTNESS_AND_DEEPENING.md | Analyses 1-9 (batch 2) |
| RESULTS_DEEP_ANALYSES.md | Analyses 10-18 |
| RESULTS_CONDITIONAL_ANALYSES.md | Analyses 19-27 |
| RESULTS_ADVANCED_ANALYSES.md | Analyses 28-35 |
| RESULTS_GRAPH_ANALYSES.md | Analyses 36-42 |
| INTUITION_CAUSAL_GEOMETRY.md | Non-technical walkthrough |
| causal_geometry_slides.tex | Beamer presentation (15 slides) |
| mechval_factor_coupling.py | Script: batch 1 |
| factor_dag_discovery.py | Script: factor DAG |
| atlas_deep_analyses.py | Script: batch 3 |
| atlas_conditional_analyses.py | Script: batch 4 |
| atlas_advanced_analyses.py | Script: batch 5 |
| atlas_graph_analyses.py | Script: batch 6 |
| RESULTS_CROSS_TASK.md | Analyses 43-47 |
| atlas_cross_task.py | Script: batch 7 (cross-task) |
| RESULTS_STRUCTURAL_ANALYSES.md | Analyses 48-55 |
| atlas_structural_analyses.py | Script: batch 8 (per-head, Q/K/V) |
| RESULTS_INFORMATION_ANALYSES.md | Analyses 56-63 |
| atlas_information_analyses.py | Script: batch 9 (signed, entropy) |
| RESULTS_EXCEPTION_ANALYSES.md | Analyses 64-71 |
| atlas_exception_analyses.py | Script: batch 10 (exceptions, paths) |
| RESULTS_GEODESIC_AUDIT.md | Analyses 72-79 |
| atlas_geodesic_audit.py | Script: batch 11 (geodesic/Grassmannian) |
| RESULTS_ALL_TASKS_RAW_FACTOR.md | Analyses 80-89 |
| atlas_all_tasks_raw_factor.py | Script: batch 12 (all 9 tasks, raw 1024-factor) |
| RESULTS_EAP_IMPLEMENTATION.md | Old vs new factorized EAP, selector sparsity |
| TODO_EAP.md | Next steps: benchmarks, scaling, grokking |

### 11. Two-Team Competition (Not Three-Stream Cooperation)

After controlling for scale, the factor network has 9 negative edges
and 2 positive edges. Two teams: content factors (dup_token +
positional + late_comp) vs action factors (S_inhib + conv_S_inhib).
Between-cluster mean r_partial = -0.456, within-B = +0.465,
within-A = +0.081.

### 12. dup_token Is the Resilient Anchor

Conditional analysis: dup_token varies only ±4% from its mean
regardless of other factors. S_inhib is the "squeezable" factor
(17.4% to 29.5% depending on content factors). S_inhib fills
whatever bandwidth content factors don't occupy.

### 13. L2→L3 Transition Is Reader-Driven

L2 edges: S_inhib-dominant (top edge 50.9%). L3 edges: dup_token-
dominant (top edge 53.6%). L3.H0's query reads 46.0% dup_token and
only 11.5% S_inhib from the same residual stream. Transition is
statistically significant (p=0.001 for dup_token, p=0.022 for
S_inhib).

### 14. Writers Are Highly Consistent

Within-writer cosine similarity = 0.861. Writers produce the same
factor mixture regardless of reader. MLP and attention writers
have identical consistency (0.859 vs 0.861). Factor composition is
writer-set but reader-amplified.

### 15. The Factor Bank Is General-Purpose Infrastructure

Cross-task analysis (6 tasks, raw 1024-factor EAP): edge rankings
are near-identical (Spearman rho > 0.97), factor usage profiles
have cosine > 0.96, 99.8% of factors have >90% of max entropy.
66 factors form a universal core (top-100 for all 6 tasks). The
dense selector creates a fully shared vocabulary.

### 16. Task Specificity at the Edge Tail

Despite massive sharing: top-50 edge Jaccard is only 0.031--0.163.
Per-edge factor cosine (IOI vs SVA) is 0.58. Early/late edges
diverge most. Specificity emerges in how tasks weight shared
factors, not in which factors exist.

### 17. IOI and SVA Are Most Distinctive

Lowest edge Jaccard (0.031), lowest factor Spearman (0.808),
per-edge cosine only 0.58. The other 4 tasks (greater_than,
capital_country, gender_bias, hypernymy) cluster tighter ---
capital_country-hypernymy cos=0.994.

### 18. Reader Selectivity > Writer Consistency

Same-reader edge pairs (cos=0.917) far more similar than same-writer
(cos=0.718). Reader projections are the primary driver of edge factor
composition. p < 10^-221.

### 19. No Temporal Differentiation in Factor Consumption

All three main factors have identical cumulative read profiles:
25%@L5, 50%@L7, 75%@L8. The circuit consumes everything at the
same rate.

### 20. Three Transition Regimes

Establishment (L0-L2), Plateau (L3-L9, cos > 0.96), Recomposition
(L10-logits). L11→logits is the biggest transition (cos=0.677,
S_inhib +13%).

### 21. conv_S_inhib Is the Only Emergent Factor

All other factors peak write at L0. conv_S_inhib peaks at L6 ---
built mid-circuit, not inherited from embeddings.

### 22. L2 MLP Is the S_inhib Hotspot

Within L2, MLP reads 19.6% more S_inhib than attention. The L2→L3
transition has an intra-layer component.

### 23. Sign Flips Encode Direction

DAS dims within each factor anti-correlate in sign (r ≈ -0.75)
while co-activating in magnitude. Sign encodes which counterfactual
condition is active.

### 24. Circuit Roles ≠ Factor Names

S_inhib heads read more dup_token (32.7%) than S_inhib (19.3%).
Name mover heads have only 4% name_mover factor. conv_S_inhib
anti-correlates with S_inhib (r=-0.79) in signed space.

### 25. Signed Network Reshuffles the Teams

Absolute: content vs action. Signed: {dup_token, late_comp, S_inhib}
vs {positional, auxiliary, token_id}. positional-late_comp goes from
r=+0.93 (absolute) to r=-0.93 (signed) --- perfect reversal.

### 26. Factor Profiles Are Universal

Now confirmed invariant to: layer, writer type, projection type
(Q/K/V), edge length, within vs cross-layer, circuit vs non-circuit,
importance, MLP vs attention, edge distance. Only exceptions:
logits edges and L2→L3 transition.

### 27. The Logits Exception Is S_inhib Inversion

Logits layer subtracts S_inhib (sign ratio -0.84). The model inverts
the suppression signal at the output to produce the non-suppressed
name. S_inhib +6.8%, positional -7.3% vs non-logits.

### 28. The IOI Computation Is a Sign Flip

S_inhib sign ratio: early/mid circuit -0.44 to -0.72, late circuit
+0.73. Non-circuit heads don't show this structure. The sign flip
IS the computation.

### 29. L9.H3 Is a Content-to-Action Translator

Receives dup_token/positional from M8 and L8.H6, outputs S_inhib
to logits. Feed→output cosine = 0.72.

### 30. M2.in Is the S_inhib Epicenter

M2.in reads 42.3% S_inhib --- highest of any reader. L2→L3
transition has MLP, attention, and per-head components.

### 31. L4.H6 Is the Token-ID Specialist

Only head with systematically distinctive profiles: token_id
35-51% vs 2.7% population average.

### 32. Geodesic Distance Confirms All Prior Findings

For single-vector comparisons, geodesic = arccos(cosine) preserves
all rankings (r > 0.99). Key restatements in degrees: reader
selectivity = 18.9° (vs 42.9° for writers), writer-reader asymmetry
= 54.8° (past halfway to orthogonal), logits transition = 47.0°
(larger than full L0→L11 drift of 36.4°), plateau L3→L9 = 16.0°.

### 33. Grassmannian Distance Reveals Hidden Subspace Separation

Cosine says dup_token-positional are near-identical (cos=0.95).
Grassmannian principal angle distance says their subspaces are 28%
of maximum separation (2.62 rad out of 9.42 max). The three main
factors occupy genuinely distinct subspaces in edge space, invisible
to marginal-profile cosine. token_id is the most isolated factor
(mean Grassmann = 1.90 rad). conv_S_inhib and auxiliary are subspace
satellites of dup_token and positional respectively (~1.0 rad).

### 34. Sign Flips Are Universal Across Tasks

Every task except MCQA shows significant sign flips at L9-L11.
F778 and F675 flip across IOI, SVA, gender_bias, hypernymy.
Sign flips are MLP-driven: consecutive MLPs push the same factor
in opposite directions (M9 positive → M10 negative for F675).

### 35. DAS Causal Importance ≠ Raw EAP Mass Importance

DAS-identified factors (F188/dup_token, F909/positional, F264/S_inhib)
rank 79th-817th in raw EAP mass but show the cleanest sign structure.
Grassmannian distance between top-20 raw factors and 7 DAS factors =
3.30 rad (30% of max). The computational subspace is different from
the mass subspace.

### 36. SVA Is the Structural Outlier

PC1 = 27.2% (vs 10-15% for most tasks), effective dim = 11 (vs ~27).
Only task where L10→L11 beats L11→logits as biggest transition.
F378 is SVA-specific (#1 factor, absent from other tasks' top-10).
SVA uses a more compressed, earlier-acting circuit.

## Open Questions

1. **Why L9.H3?** What is this non-circuit head doing that makes it
   the strongest DAS-EAP reader? Its attention pattern on IOI
   prompts would need activation-level analysis (GPU required).

2. **The signed structure**: What determines which sign a dim takes
   on a given edge? Is it the position of IO vs S in the prompt?

3. **Factor subspace geometry**: The 32 DAS dims compress to ~3
   effective dimensions. What are those 3 dimensions geometrically
   in the 768-dim residual stream?

4. **The logits exception**: Why is the output layer the only place
   where factor composition departs from the universal mixture?
   Architectural constraint (unembedding matrix) or functional
   necessity?

5. **Cross-task DAS rotation**: Would DAS-rotated (32-dim) scores
   for other tasks show the same three-stream structure? Would need
   per-task DAS training (currently only IOI has DAS rotation).
