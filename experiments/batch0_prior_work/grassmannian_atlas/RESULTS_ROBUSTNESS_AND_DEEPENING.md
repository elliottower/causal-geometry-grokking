# Robustness Checks and Deepening Analyses

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 1: Writer vs Reader Marginal Asymmetry

The DAS-EAP scores decompose into writer-marginal (scores_W) and
reader-marginal (scores_R) components. Across all 32 DAS dims summed
over the top 500 edges:

| Metric | Value |
|--------|-------|
| Total W mass | 6.21 |
| Total R mass | 421.0 |
| R/W ratio | **67.8x** |
| Top W node | input (1.26) |
| Top R node | L9.H9.Q (12.19) |

Almost all attribution is reader-side. The residual stream is a
broadcast medium — writers (MLPs, attention OV outputs) diffusely
populate it, readers (Q/K/V projections) selectively consume from it.

This is consistent with the "narrow channel" finding from the weight
coupling analysis: circuit heads restrict what they READ, not what
gets written.

## Analysis 2: Per-Dim Layer Trajectories

For each of the 32 DAS dims, the W-marginal peaks at the input
embedding layer, while the R-marginal peaks at layers 7-8
(S-inhibition / pre-name-mover region).

Pattern: information is written once at the embedding, carried
passively through the residual stream, and read selectively at
mid-to-late layers. There is no "relay" pattern where mid-circuit
writers amplify the signal — the embedding write is sufficient, and
all subsequent structure comes from selective reading.

## Analysis 3: Edge Anomalies

Measuring each edge's deviation from the population mean subspace
profile (Euclidean distance from centroid of the 32-dim DAS profile):

| Rank | Edge | Most anomalous dim | Interpretation |
|------|------|--------------------|----------------|
| Most anomalous | L9.H0 -> logits | S_inhib (36%) | Non-standard output path |
| Most typical | M6 -> L8.H4.K | — | Generic mid-circuit edge |

The most anomalous edges are output-layer edges (-> logits) that
carry unusual factor compositions, and early-layer edges where
positional information dominates atypically. Mid-circuit edges are
the most stereotyped.

## Analysis 4: Circuit Group Pair Profiles

Grouping edges by their (writer_group, reader_group) pair — where
groups are IOI circuit roles (induction, S-inhibition, name mover,
MLP, etc.) — reveals remarkably uniform DAS dim profiles across
all group pairs.

Exception: dup_token -> dup_token self-edges stand out at 43%
dup_token share (vs 34% population mean). Edges within a single
circuit stage carry more of that stage's dominant factor, as
expected.

The uniformity across other group pairs confirms that edges are
NOT specialized by circuit stage — the three-stream model applies
universally, not just to specific writer-reader combinations.

## Analysis 5: Bootstrap Stability of PCA

100 bootstrap resamples (sampling edges with replacement) of the
edge x DAS-dim matrix:

| Component | Self-correlation | Interpretation |
|-----------|-----------------|----------------|
| PC1 | 0.989 +/- 0.009 | Highly stable |
| PC2 | 0.984 +/- 0.011 | Highly stable |

Both principal components are robust — no single edge or small set
of edges drives the structure. The PC1 = scale, PC2 = S_inhib vs
dup_token contrast structure is a population-level property.

## Analysis 6: Edge Count Sensitivity

Varying the number of top edges used (n = 100, 200, 500, 1000, 2000):

| n_edges | PC1 alignment with n=500 | PC2 alignment with n=500 |
|---------|--------------------------|--------------------------|
| 100 | > 0.97 | > 0.97 |
| 200 | > 0.99 | > 0.98 |
| 1000 | > 0.99 | > 0.99 |
| 2000 | > 0.99 | > 0.99 |

The structure is not an artifact of the top-k cutoff. Even with only
100 edges, the PC structure aligns >0.97 with the 500-edge reference.

## Analysis 7: Identity EAP vs DAS-EAP Edge Rankings

Comparing edge importance rankings between identity EAP (768-dim
residual stream, no rotation) and DAS-EAP (32-dim rotated subspace):

| Metric | Value |
|--------|-------|
| Spearman rho | 0.571 |
| Top-20 overlap | 15% (3/20) |
| Top-50 overlap | ~30% |

The DAS rotation dramatically reshuffles which edges matter. Key
differences:

- **MLP edges dominate DAS-EAP**: M8, M3, M6 are top-ranked in
  DAS-EAP but mid-ranked in identity EAP. The DAS rotation reveals
  that MLPs carry critical task-relevant information that is invisible
  in the raw residual stream basis.
- **Attention edges dominate identity EAP**: L9.H9, L9.H6 (name
  movers) rank higher in identity EAP. In the raw basis, attention
  heads are the obvious circuit components; DAS reveals the MLP
  infrastructure that supports them.

This divergence validates the DAS rotation — it's not just a change
of basis, it's a change of perspective that reveals previously hidden
structure.

## Analysis 8: Signed Score Coherence Within Factor Groups

For each factor's DAS dims, measuring whether they point in the
same direction (coherent) or opposite directions (mixed) within
each edge:

**Result: Most factor groups show MIXED sign coherence.**

Dims belonging to the same factor often have opposite signs within
the same edge. This means factor dimensions are not simply "on" or
"off" per edge — they carry directional information. A dup_token
dim with positive score and one with negative score on the same edge
indicates the factor is being used in a more complex way than simple
presence/absence.

This finding reinforces the "edge modes cut across factors" result:
even within a single factor, the DAS dims are doing different things
at the edge level.

## Analysis 9: Factor Interaction Structure

Cross-factor correlation matrix across edges (each factor's total
attribution per edge):

**Strongest trade-offs (negative correlations):**

| Factor pair | r |
|-------------|---|
| positional vs S_inhib | **-0.569** |
| positional vs conv_S_inhib | -0.41 |
| dup_token vs auxiliary | -0.33 |

**Strongest co-occurrences (positive correlations):**

| Factor pair | r |
|-------------|---|
| positional + late_comp | **0.183** |
| S_inhib + conv_S_inhib | **0.166** |
| token_id + name_mover | **0.162** |

The positional vs S_inhib trade-off (r = -0.569) is the strongest
signal in the interaction matrix. Edges that carry positional
information carry less S-inhibition, and vice versa. This matches
the circuit's functional logic: positional encoding (early) and
S-inhibition (mid-late) operate at different circuit stages.

The S_inhib + conv_S_inhib co-occurrence (r = 0.166) confirms that
the convergent S-inhibition pathway co-travels with the main pathway,
consistent with the mediation analysis from the edge decomposition.

The token_id + name_mover co-occurrence (r = 0.162) captures the
direct path from token identity to name mover output, bypassing the
main dup_token -> S_inhib chain.

## Synthesis

These nine analyses collectively establish:

1. **The findings are robust** (Analyses 5-6): bootstrap stability
   >0.98, insensitive to edge count threshold.

2. **Reader selectivity drives circuit structure** (Analyses 1-2):
   writers broadcast, readers select. The 67.8x R/W asymmetry means
   circuit edges are defined by what downstream heads choose to read.

3. **DAS rotation reveals hidden MLP structure** (Analysis 7): the
   15% top-20 overlap between identity and DAS-EAP shows the
   rotation isn't cosmetic — it uncovers MLP contributions invisible
   in the raw basis.

4. **Factor dims are not monolithic** (Analysis 8): mixed sign
   coherence within factor groups means factors carry directional
   information, not just presence/absence signals.

5. **Factor interactions recapitulate circuit logic** (Analysis 9):
   the trade-off and co-occurrence structure matches expected IOI
   circuit stages (positional vs S_inhib opposition, S_inhib +
   conv_S_inhib co-travel).

6. **Edges are not stage-typed** (Analysis 4): uniform profiles
   across circuit group pairs confirm the three-stream model applies
   universally, not just within specific circuit modules.
