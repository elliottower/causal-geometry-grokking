# Exception Zones and Causal Paths: Logits, L2→L3, and Signed Circuit Structure

Checkpoint: shared_bank_global_dense (1024 factors, dense selector, GPT-2 small)
DAS run: zgaibjs7 (k=32 DAS dims, 10 active factors, IIA=0.673)
Date: 2026-06-14

## Analysis 64: Logits Exception Decomposition

### The logits layer inverts S_inhib

| Factor | Logits | Non-logits | Difference |
|--------|--------|-----------|------------|
| S_inhib | 29.6% | 22.9% | **+6.8%** |
| dup_token | 35.8% | 32.5% | +3.3% |
| positional | 15.1% | 22.5% | **-7.3%** |
| conv_S_inhib | 1.3% | 3.4% | -2.1% |

The logits layer reads disproportionately more S_inhib and less
positional than all other layers.

### Signed logits profile: S_inhib is SUBTRACTED

| Factor | Signed | Unsigned | Sign ratio |
|--------|--------|----------|------------|
| S_inhib | -0.274 | 0.324 | **-0.844** |
| dup_token | +0.201 | 0.511 | +0.393 |
| positional | +0.063 | 0.125 | +0.503 |

S_inhib flows negatively into logits (sign ratio -0.84). The model
**subtracts** the S_inhib signal at the output. This makes functional
sense: S_inhib identifies which name to suppress. The logits need to
*output* the non-suppressed name, so the S_inhib contribution is
inverted at the output layer.

### Top logits feeders

| Writer | Mass | Profile |
|--------|------|---------|
| M8 | 1.141 | dup_token 42.9%, S_inhib 34.1% |
| M9 | 0.979 | dup_token 36.9%, positional 25.8% |
| L8.H6 | 0.841 | dup_token 41.0%, S_inhib 32.7% |
| L9.H3 | 0.361 | **S_inhib 39.2%**, dup_token 29.6% |

L9.H3 (the top non-circuit reader) sends S_inhib-dominated signal
to logits --- where it gets subtracted.

## Analysis 65: L2→L3 Transition Per-Head Breakdown

### M2.in is the S_inhib epicenter

| Reader | Mass | dup_token | S_inhib | positional |
|--------|------|-----------|---------|------------|
| **M2.in** | **2.093** | 27.1% | **42.3%** | 12.3% |
| M3.in | 2.013 | 35.5% | 24.3% | 15.7% |
| L3.H0* | 1.389 | **40.1%** | **15.0%** | 23.6% |
| L3.H1 | 1.222 | 36.4% | 21.1% | 22.2% |
| L2.H0 | 0.713 | 36.6% | 27.7% | 19.6% |

M2.in reads 42.3% S_inhib --- the highest S_inhib share of any
reader at any layer. The next MLP (M3.in) drops to 24.3%. The
MLP-level transition from L2 to L3 is even sharper than the
attention-level transition.

L3.H0 (circuit: duplicate token detector) drives the attention
shift: highest mass at L3, highest dup_token (40.1%), lowest
S_inhib (15.0%).

### The L2→L3 transition has three components

1. **MLP transition** (M2→M3): S_inhib drops from 42.3% to 24.3%
2. **Attention transition**: L3.H0 introduces dup_token dominance
3. **Head-level variation**: L3.H11 bucks the trend (S_inhib=30.5%)

## Analysis 66: L9.H3 Causal Paths

### L9.H3 is a translator head

L9.H3 receives dup_token/positional signal and outputs S_inhib
signal:

| Direction | Top feeder/consumer | Mass | Profile |
|-----------|-------------------|------|---------|
| Feed (→ L9.H3.Q) | M8 | 3.335 | dup_token 38.2%, positional 26.2% |
| Feed (→ L9.H3.Q) | L8.H6 | 2.521 | dup_token 35.8%, positional 26.4% |
| Output (L9.H3 →) | logits | 0.361 | **S_inhib 39.2%**, dup_token 29.6% |

Feed → output cosine = 0.72 --- measurable transformation. L9.H3
reads content signals (dup_token, positional) and writes an action
signal (S_inhib). It functions as a content-to-action translator.

## Analysis 67: Writer→Reader Factor Transformation

### Readers systematically amplify task-relevant factors

| Factor | Reader - Writer shift | p-value |
|--------|----------------------|---------|
| token_id | **-8.3%** | 8.6 x 10^-25 |
| dup_token | **+7.3%** | 5.0 x 10^-12 |
| positional | +5.5% | 1.6 x 10^-13 |
| S_inhib | +4.1% | 3.8 x 10^-6 |
| conv_S_inhib | -3.9% | 7.2 x 10^-16 |
| late_comp | -2.2% | 4.8 x 10^-5 |

Readers amplify the three main factors (dup_token +7.3%, positional
+5.5%, S_inhib +4.1%) and suppress minor factors (token_id -8.3%,
conv_S_inhib -3.9%). This is the mechanism of reader selectivity:
downstream projections actively filter toward task-relevant subspaces.

Mean W-R cosine = 0.567 ± 0.150. Writers and readers of the same
edge disagree substantially --- the transformation is the computation.

## Analysis 68: Factor Mass Percentile Profiles

Factor share distributions at each layer are remarkably stable:

| Factor | P50 range across layers | P25-P75 IQR |
|--------|------------------------|-------------|
| dup_token | 26.8%--31.4% | ~10% |
| S_inhib | 20.2%--24.4% | ~10% |
| positional | 18.8%--22.2% | ~10% |

The distributions overlap heavily across all layers. There is no
layer where a factor "turns on" or "turns off" --- the variation is
in the tail, not the bulk.

## Analysis 69: R/W Ratio Escalation

R/W ratios increase exponentially from early to late layers:

| Layer | dup_token | S_inhib | positional |
|-------|-----------|---------|------------|
| L0 | 14.9 | 20.1 | 10.6 |
| L2 | 75.9 | **96.8** | 46.7 |
| L5 | 52.2 | 112.2 | 105.1 |
| L8 | 799.1 | 607.8 | 882.5 |
| L9 | >1000 | >1000 | >1000 |

The R/W ratio jumps sharply at L2 (from 20-26x to 47-97x). By L8-L9,
writers contribute <0.1% of the total --- the signal is entirely
reader-dominated. The "broadcast and select" architecture becomes
more extreme with depth.

## Analysis 70: Edge Distinctiveness Map

### Logits edges are the most distinctive reader type

| Reader type | Mean distinctiveness | Std |
|------------|---------------------|-----|
| K | 0.162 | 0.064 |
| Q | 0.167 | 0.067 |
| V | 0.172 | 0.061 |
| MLP | 0.190 | 0.054 |
| **logits** | **0.249** | **0.078** |

Logits edges have 1.5x the distinctiveness of attention edges ---
they genuinely depart from the population mean profile.

### L4.H6: the token_id specialist

L4.H6 appears 4 times in the 10 most distinctive edges, always
with token_id dominant (35--51%). This is the only head with a
systematically unusual factor profile. It specializes in token
identity --- a minor factor globally but concentrated in this head.

## Analysis 71: Signed Factor Mass by Circuit Stage

### The circuit computation revealed in sign structure

| Stage | S_inhib sign ratio | dup_token sign ratio |
|-------|-------------------|---------------------|
| **Early (L0-L2)** | **-0.440** | -0.003 |
| **Mid (L3-L6)** | **-0.719** | -0.069 |
| **Late (L7-L9)** | **+0.733** | -0.286 |
| Output (L10-L11) | +0.274 | -0.058 |
| Non-circuit | +0.147 | +0.100 |

**S_inhib sign flips from negative to positive between mid and
late circuit stages.** Early/mid circuit heads write anti-S_inhib
(negative sign ratio -0.44 to -0.72). Late circuit heads write
pro-S_inhib (positive sign ratio +0.73).

This is the IOI computation in signed factor space:
1. Early heads identify which name is duplicated (dup_token neutral,
   S_inhib suppressed)
2. Mid heads build up the suppression signal (S_inhib strongly
   negative)
3. Late heads flip the S_inhib sign (from -0.72 to +0.73) ---
   converting "suppress this name" into "output the other name"

conv_S_inhib in the late stage has sign ratio = -1.000 (perfectly
negative), confirming it opposes S_inhib at the circuit's output.

### Non-circuit heads don't show this structure

Non-circuit heads have uniformly positive S_inhib and dup_token.
No sign flip. The signed computation is specific to the IOI circuit
--- the non-circuit infrastructure carries the signals but doesn't
perform the sign transformation.

## Updated Synthesis (71 analyses across 10 batches)

### 27. The Logits Exception Is S_inhib Inversion

The logits layer doesn't just have a different factor profile ---
it subtracts S_inhib (sign ratio -0.84). The model inverts the
suppression signal at the output to produce the non-suppressed name.

### 28. The IOI Computation Is a Sign Flip

The circuit's core computation is visible in the S_inhib sign ratio:
early/mid stages write negative S_inhib (-0.44 to -0.72), late
stages write positive S_inhib (+0.73). The sign flip IS the
computation. Non-circuit heads don't show this structure.

### 29. L9.H3 Is a Content-to-Action Translator

L9.H3 receives dup_token/positional from M8 and L8.H6, outputs
S_inhib to logits. Feed→output cosine = 0.72. It converts content
information into the action signal, despite not being in the
standard IOI circuit.

### 30. M2.in Is the S_inhib Epicenter

M2.in reads 42.3% S_inhib --- highest of any reader at any layer.
The L2→L3 transition has three components: MLP (M2→M3), attention
(L3.H0), and per-head variation.

### 31. L4.H6 Is the Token-ID Specialist

The only head with systematically distinctive factor profiles.
Token_id = 35--51% on its edges vs 2.7% population average.
