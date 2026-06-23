# Advanced Analyses: Signed Structure, Conditional Independence, and Circuit Boundary

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 28: Signed DAS Dim Structure

### Sign consistency is at chance

Every DAS dim has ~50/50 positive/negative split across edges
(consistency 0.50-0.53 for all 32 dims). No dim has a preferred
sign direction. The signed scores are mean-zero across the edge
population.

### Within-factor sign coherence is low

| Factor | Mean sign coherence |
|--------|-------------------|
| auxiliary | 0.512 |
| conv_S_inhib | 0.480 |
| late_comp | 0.476 |
| name_mover | 0.343 |
| token_id | 0.292 |
| S_inhib | 0.245 |
| dup_token | 0.167 |
| positional | 0.117 |

(1.0 = all same sign, 0.0 = fully mixed)

The two largest factors (dup_token, positional) have the lowest
within-factor sign coherence — their dims point in opposite
directions within the same edge. The smaller factors (auxiliary,
conv_S_inhib) are more coherent.

### Signed correlations reveal hidden anti-structure

The absolute-value analysis (Analysis 18) found ALL correlations
positive. The **signed** analysis reveals a dramatically different
structure:

**Strongest positive signed correlations:**

| Dim pair | r | Factor pair |
|----------|---|-------------|
| d17-d27 | +0.787 | positional-name_mover |
| d3-d11 | +0.727 | dup_token-late_comp |
| d7-d24 | +0.707 | conv_S_inhib-S_inhib |
| d12-d30 | +0.679 | S_inhib-positional |
| d0-d7 | +0.659 | auxiliary-conv_S_inhib |

**Strongest negative signed correlations:**

| Dim pair | r | Factor pair |
|----------|---|-------------|
| d8-d11 | **-0.911** | positional-late_comp |
| d7-d12 | -0.791 | conv_S_inhib-S_inhib |
| d0-d12 | -0.758 | auxiliary-S_inhib |
| d12-d24 | **-0.757** | S_inhib-S_inhib (SAME) |
| d3-d26 | **-0.749** | dup_token-dup_token (SAME) |

**Same-factor dims anti-correlate in sign.** d3 and d26 (both
dup_token) have r = -0.749. d12 and d24 (both S_inhib) have
r = -0.757. When the absolute magnitudes co-activate (Analysis 18
showed positive correlations), the signs flip — the factor carries
information in BOTH directions simultaneously.

The strongest anti-correlation overall is CROSS-factor: d8
(positional) and d11 (late_comp) have r = -0.911. In absolute
values these had r = +0.904 (the strongest pair in Analysis 18).
Their magnitudes co-activate perfectly, but their signs are
perfectly anti-correlated. They carry a single dimension of
information encoded in the sign contrast.

### Interpretation

The DAS rotation produces dimensions that encode information as
sign contrasts, not as positive activations. Two dims that
co-activate in magnitude but anti-correlate in sign together encode
a single bit: which direction the information points. This is
consistent with the DAS rotation finding orthogonal directions that
separate counterfactual interventions — the sign encodes which
intervention is active.

## Analysis 29: Mutual Information

MI captures non-linear dependencies beyond correlation. Top MI
pairs track the top correlation pairs, but with excess MI indicating
non-linear structure:

| Dim pair | MI | r | Excess MI | Type |
|----------|-----|-----|-----------|------|
| d26-d30 | 0.697 | 0.740 | +0.327 | CROSS (dup_token-positional) |
| d3-d23 | 0.680 | 0.711 | +0.325 | SAME (dup_token-dup_token) |
| d19-d22 | 0.629 | 0.692 | +0.283 | SAME (name_mover-name_mover) |

Within-factor MI (0.341) > cross-factor MI (0.271), consistent with
the correlation findings but with additional non-linear structure
within factors.

## Analysis 30: Conditional Independence (Controlling for Scale)

### Scale was masking the true interaction structure

Partial correlations controlling for total edge mass reveal much
stronger factor interactions than the raw correlations:

| Factor pair | Raw r | Partial r (|scale) | Change |
|-------------|-------|-------------------|--------|
| S_inhib vs positional | -0.569 | **-0.724** | Amplified |
| S_inhib vs conv_S_inhib | +0.166 | **+0.465** | Amplified |
| conv_S_inhib vs late_comp | — | **-0.464** | New |
| conv_S_inhib vs positional | — | **-0.401** | New |
| S_inhib vs dup_token | — | **-0.395** | New |
| S_inhib vs late_comp | — | **-0.373** | New |
| auxiliary vs late_comp2 | — | **-0.390** | New |
| late_comp vs positional | +0.183 | **+0.352** | Amplified |

The S_inhib-positional trade-off strengthens from r=-0.569 to
r_partial=-0.724. The scale confound was DILUTING this relationship
— controlling for edge importance reveals it's one of the strongest
structural features of the circuit.

### Factor interaction network (after scale removal)

Two clusters emerge:

**Cluster A (anti-S_inhib):** dup_token, positional, late_comp
- All negatively partially correlated with S_inhib
- Positively partially correlated with each other (especially
  positional-late_comp: +0.352)

**Cluster B (pro-S_inhib):** S_inhib, conv_S_inhib
- Positively partially correlated (+0.465)
- Both negatively partially correlated with Cluster A factors

This is the two-mode structure from the unsupervised analysis,
now confirmed at the factor level with scale controlled.

## Analysis 31: Extended Circuit Boundary

### The standard IOI circuit is 16% of DAS-EAP mass

| Component | % of total DAS-EAP mass |
|-----------|------------------------|
| Non-circuit attention heads | 54.0% |
| MLPs | 27.8% |
| Circuit attention heads | 16.0% |
| Logits | 1.6% |
| Input embedding | 0.5% |

77% of attention head mass is in non-circuit heads. The DAS-EAP
decomposition sees the model's general information infrastructure,
not just the IOI-specific circuit.

### Cumulative mass capture

| Top N heads | % mass | Circuit heads in top N |
|-------------|--------|----------------------|
| 5 | 19.4% | 3 |
| 10 | 33.1% | 4 |
| 15 | 43.2% | 5 |
| 20 | 50.7% | 5 |
| 30 | 62.9% | 8 |
| 50 | 78.3% | 10 |
| 76 | 90.0% | 11 |

50% of head mass requires 20 heads but only 5 are IOI circuit
heads. 90% requires 76/144 heads — over half the model.

### Interpretation

The DAS-EAP scores measure the causal effect of the 32 DAS
subspaces on model behavior through each node. The broad mass
distribution means these subspaces are NOT restricted to the
IOI circuit — they capture general model computation that happens
to overlap with the IOI task. This is expected for a factorized
model with a shared factor bank: all heads compute with the same
factors, so all heads contribute some DAS-EAP mass.

## Analysis 32: Factor Share Distributions

| Factor | Distribution | Key stat |
|--------|-------------|----------|
| dup_token | Normal | Shapiro p=0.35, symmetric |
| positional | Normal | Shapiro p=0.41, symmetric |
| S_inhib | Right-skewed | Shapiro p=0.001, skew=0.44 |
| token_id | Highly skewed | skew=3.8, kurtosis=20.3 |
| name_mover | Right-skewed | skew=1.3 |
| conv_S_inhib | Right-skewed | skew=1.4 |

The two dominant factors (dup_token, positional) have normal share
distributions — most edges carry about the same proportion. S_inhib
is right-skewed: most edges carry moderate S_inhib, but some edges
carry anomalously high S_inhib (up to 52.1%). The minor factors
(token_id, name_mover, conv_S_inhib) are all right-skewed with
heavy tails: most edges carry almost none, but a few carry a lot.

## Analysis 33: Edge Distinctiveness

| Layer group | Mean cos_dist from population mean |
|------------|-----------------------------------|
| L0-L1 | 0.224 - 0.261 (most distinctive) |
| L5-L8 | 0.140 - 0.160 (most stereotyped) |
| L9 | 0.163 |
| logits | 0.259 (most distinctive) |

The circuit's periphery (early layers and output) has the most
distinctive edge profiles. The core (L5-L8) is the most
stereotyped — mid-circuit edges are the population mean.

Most distinctive individual edge: L9.H0 -> logits (cos_dist=0.395,
S_inhib dominant at 35.8%).

## Analysis 34: Factor Dominance Switching

74.2% of top-500 edges have dup_token as the dominant factor.
S_inhib dominates 17.4%, positional 8.4%.

| Reader layer | Dominant factor | Its share |
|-------------|----------------|-----------|
| L2 | S_inhib | 53% |
| L3-L8 | dup_token | 59-85% |
| L9 | dup_token | 75% |
| logits | dup_token (50%) + S_inhib (44%) | Split |

L2 is the only layer where S_inhib dominates — this is where
duplicate token detection begins and S_inhib information flows in.
The logits layer is the only place where dominance is contested
between dup_token and S_inhib.

## Analysis 35: Information Bottleneck

Factor mass is highly distributed — no single edge carries more than
2% of any factor's total:

| Factor | Edges for 25% | Edges for 50% | Edges for 75% |
|--------|--------------|--------------|--------------|
| dup_token | 69 | 368 | 1608 |
| S_inhib | 78 | 407 | 1743 |
| positional | 67 | 356 | 1594 |

All three factors flow through the same top bottleneck edges:
M8 -> L9.H3.Q and L8.H6 -> L9.H3.Q are in the top 5 for all
three factors. The bottleneck is not factor-specific — it's the
same high-bandwidth edges that carry everything.

## Grand Synthesis

Across all 35 analyses, the picture crystallizes:

### The DAS subspace is a shared broadcast channel

The 32 DAS dims encode the IOI task's causal structure, but they
propagate through the ENTIRE model, not just the IOI circuit.
77% of head mass and 28% of total mass from MLPs are in non-circuit
components. L9.H3 (non-circuit) carries 42x more mass than L9.H9
(name mover). The circuit boundary is blurred at the factor level.

### Two structural axes under a dominant scale component

- PC1 (62%): "How important is this edge?" — scale
- rPC1 (12% of total): S_inhib vs {dup_token + positional} —
  the primary structural contrast, strengthened to r=-0.724 after
  controlling for scale
- rPC2 (6% of total): positional vs S_inhib — the secondary
  structural contrast

### Signed structure reveals directional encoding

In absolute values, all dim-dim correlations are positive (edges
that carry more of one dim carry more of everything). In signed
values, same-factor dims ANTI-correlate (d3-d26 dup_token: r=-0.749,
d12-d24 S_inhib: r=-0.757) and cross-factor dims can have extreme
anti-correlation (d8-d11 positional-late_comp: r=-0.911).

The DAS dims encode information as sign contrasts between paired
dimensions. The absolute magnitude says "how much information is
flowing," the sign pattern says "which direction it points."

### The circuit's computational structure is in the 12%, not the 62%

The 62% scale component tells us that some edges matter more than
others. The 12% structural component tells us HOW the circuit
works: two modes (S_inhib-weighted vs positional/dup_token-weighted)
that trade off, with the transition happening at L2-L3 (where
duplicate token detection begins) and again at the logits layer
(where S-inhibition concentrates for output).
