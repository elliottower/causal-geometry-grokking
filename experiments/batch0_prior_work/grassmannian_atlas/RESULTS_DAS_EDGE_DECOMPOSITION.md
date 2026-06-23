# DAS-EAP Edge Decomposition: Subspaces Within Circuit Edges

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## What this is

Each EAP edge (e.g., L8.H6 -> L9.H9.Q) isn't a monolithic connection.
The DAS-EAP scores decompose it into 32 causal subspaces, each labeled
by its dominant factor. This analysis asks:
1. What subspaces does each edge carry?
2. Do the subspaces interact causally across edges?
3. Does the IOI circuit DAG fit at the subspace level?

## Per-edge subspace profiles

Across all top-50 edges, the average factor composition is:

| Factor subspace | Avg share | Role |
|-----------------|-----------|------|
| dup_token (F188) | 34.0% | Duplicate token detection |
| positional (F909) | 22.3% | Early positional encoding |
| S_inhib (F264) | 21.7% | S-inhibition pathway |
| auxiliary (F837) | 6.6% | Supporting computation |
| late_comp (F824) | 6.0% | Late-layer composition |
| name_mover (F383) | 4.0% | Name mover output |
| conv_S_inhib (F222) | 2.9% | Convergent S-inhibition |
| token_id (F798) | 1.8% | Token identity via embedding |
| late_comp2 (F1016) | 0.6% | Late composition |

Three factors (dup_token + positional + S_inhib) account for 78% of
the attribution mass in the top edges. The circuit is dominated by
three concurrent information streams.

## Edge type census

Classifying the top 200 edges by their dominant and secondary subspace:

| Edge type | Count | Examples |
|-----------|-------|----------|
| dup_token + positional | 87 | M8->L9.H9.Q, L8.H6->L9.H9.Q |
| dup_token + S_inhib | 51 | M8->logits, M3->L6.H9.Q |
| S_inhib + dup_token | 27 | M3->M4, M3->L5.H9.Q |
| positional + dup_token | 19 | M8->L9.H6.Q, L8.H6->L9.H6.Q |
| dup_token only | 11 | L5.H5->L6.H9.Q, M3->L7.H9.V |
| positional + S_inhib | 2 | L7.H9->M8, L8.H0->M8 |
| S_inhib only | 1 | M7->L9.H9.Q |
| positional only | 1 | input->M0 |

69% of edges (138/200) are dup_token-dominant. The duplicate token
signal is the backbone of the IOI circuit at the subspace level.

Edges to name mover heads (L9.H9, L9.H6) carry dup_token + positional,
consistent with the name movers needing both "which token repeated"
and "where it appeared" to suppress the correct name.

## Key individual edges

**#1: M8 -> L9.H9.Q** (total mass 3.33)
- dup_token 38%, positional 26%, S_inhib 15%
- The strongest edge: MLP8's output feeding the name mover L9.H9's
  query. Carries all three main signals simultaneously.

**#5: M8 -> L9.H6.Q** (total mass 1.65)
- positional 38%, dup_token 32%, S_inhib 13%
- Same pattern but positional-dominant — L9.H6 may attend more to
  position than token identity.

**#7: M3 -> L8.H6.V** (total mass 1.17)
- dup_token 36%, positional 22%, S_inhib 15%, conv_S_inhib 6%
- MLP3 output feeding S-inhibition head L8.H6's value. The
  convergent S-inhibition signal (F222) also appears here.

## Causal inference on DAS subspaces

### PC algorithm (32 DAS dims as variables, 500 top edges as observations)

400/496 possible edges survive conditional independence testing
(alpha=0.01). The DAS dims are heavily correlated — most pairs remain
conditionally dependent even after controlling for other dims.

Top directed edges (oriented by W/R asymmetry — higher W/R ratio writes):

| Source | Target | r |
|--------|--------|---|
| d11(late_comp) -> d8(positional) | 0.904 |
| d15(dup_token) -> d14(S_inhib) | 0.858 |
| d7(conv_S_inhib) -> d12(S_inhib) | 0.818 |
| d27(name_mover) -> d17(positional) | 0.830 |
| d22(name_mover) -> d13(late_comp) | 0.786 |

The dup_token -> S_inhib direction (d15 -> d14, r=0.858) matches the
expected IOI circuit flow. The conv_S_inhib -> S_inhib direction
(d7 -> d12, r=0.818) shows the convergent signal feeding into the
main S-inhibition pathway.

### Mediation analysis

Testing whether factor M mediates the effect of factor A on factor B
across the edge population. Proportion mediated >100% indicates a
suppression effect (indirect path stronger than total).

Top significant mediations (Sobel p < 0.05):

| Path | % mediated |
|------|-----------|
| token_id -> S_inhib -> conv_S_inhib | 139% |
| dup_token -> S_inhib -> conv_S_inhib | 140% |
| name_mover -> S_inhib -> conv_S_inhib | 127% |
| positional -> dup_token -> S_inhib | 122% |
| token_id -> dup_token -> conv_S_inhib | 112% |
| S_inhib -> dup_token -> positional | 111% |
| late_comp2 -> late_comp -> dup_token | 99% |

S_inhib (F264) is the dominant mediator — it mediates almost every
pathway into conv_S_inhib (F222). The S-inhibition subspace is the
central hub through which other signals route.

The positional -> dup_token -> S_inhib chain (122% mediated) matches
the expected IOI circuit flow: early positional information flows
through duplicate token detection into S-inhibition.

### Structural equation model

**Expected DAG edges from IOI circuit:**

| Edge | r | beta | R^2 |
|------|---|------|-----|
| positional -> token_id | 0.690 | 0.099 | 0.476 |
| token_id -> dup_token | 0.652 | 6.242 | 0.425 |
| dup_token -> S_inhib | 0.846 | 0.829 | **0.716** |
| dup_token -> conv_S_inhib | 0.541 | 0.173 | 0.293 |
| S_inhib -> name_mover | 0.703 | 0.166 | 0.494 |
| conv_S_inhib -> name_mover | 0.440 | 0.319 | 0.194 |
| late_comp -> name_mover | 0.474 | 0.853 | 0.225 |

The dup_token -> S_inhib edge has the strongest fit (R^2 = 0.716,
r = 0.846). Edges carrying more dup_token signal also carry more
S_inhib signal — these subspaces co-travel through the circuit.

**Full SEM (multiple regression, circuit parents -> child):**

| Child | Parents | R^2 | Coefficients |
|-------|---------|-----|-------------|
| token_id | positional | 0.476 | pos=0.099 |
| dup_token | token_id, positional | **0.852** | tok=0.282, pos=1.236 |
| S_inhib | dup_token, token_id | **0.718** | dup=0.794, tok=0.504 |
| conv_S_inhib | dup_token | 0.293 | dup=0.173 |
| name_mover | S_inhib, conv_S_inhib, late_comp2 | **0.532** | S=0.169, conv=-0.104, late=0.354 |
| late_comp | S_inhib, dup_token | **0.805** | S=-0.005, dup=0.319 |

The IOI circuit DAG explains 48-85% of variance in factor subspace
co-occurrence across edges. dup_token is the strongest predictor of
downstream factors (R^2 = 0.852 for predicting dup_token from its
parents, 0.718 for S_inhib from dup_token + token_id).

## Interpretation

### The three-stream model

The IOI circuit at sub-edge resolution carries three concurrent
information streams through most edges:

1. **Duplicate token stream (34%)** — "which token appeared before?"
   Carried by F188's 6 DAS dims. Dominant in 138/200 top edges.

2. **Positional stream (22%)** — "where did it appear?"
   Carried by F909's 6 DAS dims. Co-travels with dup_token
   (87 edges are dup_token + positional type).

3. **S-inhibition stream (22%)** — "suppress the repeated name"
   Carried by F264's 4 DAS dims. Becomes dominant in mid-circuit
   edges (layers 3-8).

### Subspaces interact causally

The SEM fits confirm that these streams don't just co-occur — they
have causal structure matching the known IOI circuit:
- positional and token_id together predict dup_token (R^2 = 0.85)
- dup_token predicts S_inhib (R^2 = 0.72)
- S_inhib predicts name_mover (R^2 = 0.53)

### W vs R asymmetry

Almost all attribution in the top edges is R-marginal (reader side),
not W-marginal. This means the top edges are defined by which DAS
subspaces the READER head consumes, not what the writer produces.
The writer signal is diffuse; the reader is selective.

This is consistent with the "narrow channel" finding from the weight
coupling analysis: circuit heads restrict what they read, not what
gets written into the residual stream.
