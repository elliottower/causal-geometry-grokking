# Paper Structure Assessment: Should These Be One Paper or Two?

**Three things you have:**
- `main.tex` — full journal draft, "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian"
- `grokking_geometry_v1.tex` — slides: "Grokking Creates Manifolds, Not Subspaces"
- `structured_pi_sae_v1.tex` — slides: "Learning Causal Variables, Not Causal Directions" (pi-SAE, validated on MIB benchmark)

The slides are more up to date than the draft in important ways — particularly the **structured pi-SAE** (which replaces the vanilla structured VAE in the draft) and the **MIB benchmark validation** (not in main.tex at all). Here's the honest assessment.

---

## What You Have vs. What the Draft Says

### The draft (`main.tex`) claims:
1. Atlas of 14 operations — three-class partition (Always / Stochastic / Never Grassmannian)
2. Linear DAS returns IIA = 0 on grokked modular addition (it's on S¹, not Gr(k,d))
3. Stochastic grokking: same op, different seeds → opposite causal geometries
4. Memorization artifacts: high IIA without structure (squaring, cubing)
5. Structured VAE recovers nonlinear causal variables where DAS fails
6. NL-DAS (unconstrained MLP featurizer) is vacuous — achieves perfect IIA via lookup table
7. Intervention faithfulness metrics: diversity ratio ρ, reconstruction fidelity, KL/JS divergence

### What the slides add that the draft doesn't have yet:
- **Structured pi-SAE** instead of plain structured VAE — this is a big upgrade
  - The 2×2 ablation: {structured prior, plain prior} × {VAE, SAE}
  - Plain SAE works on addition but degrades on multiplication and quartic sum
  - Structured pi-SAE is uniformly best across all tasks
- **IOI on GPT-2 (MIB benchmark)** — the draft only has synthetic tasks
  - IOI achieves IIA = 0.98 with structured pi-SAE vs. 0.30 with DAS
  - NL-DAS gets IIA = 1.00 on IOI — confirmed vacuous on real task
- **"IOI is not Grassmannian"** slide — major empirical claim not in the draft
- **Cross-task transfer** section
- **Subspaces to factors** — connecting the Grassmannian paper to the weight-space work

### What the draft has that the slides don't develop:
- Full methods section with formal NL-DAS+recon baseline
- Four-level hierarchy formalized (effectiveness → faithfulness → distributional fidelity → structural)
- Cross-distribution generalization on IOI (disjoint names/templates) — marked TODO
- Multi-seed stability experiments — marked TODO
- Full structured VAE results table — marked TODO

---

## Are These One Paper or Two?

**My honest assessment: one paper, but you need to decide which story you're telling.**

There are currently two stories competing:

### Story A — "Grokking and the Grassmannian"
> *When does a model's internal representation of a task become a linear causal variable?*

Core claim: grokking is the mechanism. The three-class partition of 14 operations is the evidence. The geometric diagnostics (equivariance, k-sweep, circle geometry) are the tools. This is the atlas story.

**Best audience:** TMLR, NeurIPS, ICLR theory track. Deep dive on a single phenomenon, clean experimental design.

### Story B — "Structured pi-SAE for Nonlinear Causal Discovery"
> *We have a method that works when DAS fails. Here is the method, here is the ablation, here is the validation on real tasks.*

Core claim: structured prior + sparse autoencoder jointly is necessary and sufficient for faithful nonlinear causal variable recovery. The NL-DAS vacuity result and the 2×2 ablation are the evidence. The MIB benchmark is the validation.

**Best audience:** TMLR methods paper, NeurIPS main track. Method + benchmark.

### The tension between them:
The current draft tries to tell both stories. The atlas (Story A) runs from §3.1–§4.5. The VAE/pi-SAE (Story B) runs §3.5–§4.6, §4.7. These are *related* but they have different protagonists (the operations vs. the method), different levels of completion (atlas is mostly done, VAE table is TODO), and different types of claims (existence theorem vs. practical algorithm).

**Separating them does strengthen both:**

| | Paper 1: The Atlas | Paper 2: The pi-SAE |
|---|---|---|
| **Core claim** | Grokking ↔ Grassmannian causal variables | Structured pi-SAE recovers nonlinear causal variables faithfully |
| **Main table** | Table 2 (operations × class) | Structured pi-SAE results table |
| **Key negative result** | Linear DAS = 0 on grokked mod-add | NL-DAS is vacuous (false positives) |
| **Key diagnostic** | Equivariance + k-sweep | Diversity ratio ρ + reconstruction MSE |
| **Model regime** | Single-layer transformers, synthetic | Single-layer (grokking) + GPT-2 (IOI) |
| **Venue target** | TMLR / NeurIPS findings | TMLR / NeurIPS main |
| **What's missing** | 10-seed multi-seed stability, mod-add VAE connection | Full ablation table, cross-distribution generalization |

**But they do strengthen each other:**
- The atlas establishes *why* you need a nonlinear method (linear DAS = 0 on mod-add is the motivation)
- The pi-SAE validates *that the nonlinear method does what it should* (IIA = 0 on non-grokked ops, IIA = 1 on grokked)
- "IOI is not Grassmannian" in Paper 2 refers back to Paper 1's diagnostic framework

So the cleanest structure is: **two papers, written to cite each other.** Paper 1 establishes the problem; Paper 2 solves it.

---

## What Needs to Happen Before Either Can Be Submitted

### Paper 1 (Atlas) is nearly submittable. Missing:
- [ ] **Multi-seed stability** for stochastic class (10 seeds for Power and Composite Addition) — this is the most important experiment, stochastic grokking is a key claim
- [ ] **Structured VAE / pi-SAE section** either needs full results OR a pointer to Paper 2 saying "we solve this problem in the companion paper"
- [ ] **Modular addition explanation** — why does DAS = 0? Need to show the S¹ structure explicitly (PCA of DAS projection showing circle, not plane)
- [ ] References (currently `\citep{?}` throughout)
- [ ] Figure generation (TikZ diagrams need `figures/generated/` populated)

### Paper 2 (pi-SAE) is further from submittable. Missing:
- [ ] **Full ablation table** with all four conditions across all operations (partially in slides, not in draft)
- [ ] **Diversity ratio ρ results for NL-DAS on IOI** — the table has `—` everywhere, this is the core evidence
- [ ] **Cross-distribution generalization** (disjoint names, disjoint templates) — marked TODO in draft
- [ ] **Hard-mode IIA** (CPCA-init style) — referenced but not reported
- [ ] **Theoretical connection to iVAE identifiability** (Khemakhem et al.) — mentioned in related work but not developed
- [ ] IOI circuit-level analysis: which heads does the pi-SAE recovery correspond to? Does the causal variable match the known IOI circuit?

---

## Specific Things the Slides Have That Need to Go in the Paper

From `structured_pi_sae_v1.tex`:

**1. The 2×2 ablation result is essential and missing from the draft:**
```
                    Plain prior     Structured prior
VAE                 IIA=0.02        IIA=0.00
SAE                 IIA=1.00*       IIA=1.00 (uniform)

*Plain SAE degrades: Multiplication 0.72, Quartic sum 0.32
Structured pi-SAE: all grokked = 1.00, all non-grokked ≈ 0
```
The key finding is that **neither component alone suffices**: plain SAE works on addition but breaks on harder operations; pi-VAE (no sparsity) doesn't work at all. Only the combination is robust.

**2. The reconstruction MSE diagnostic:**
```
              Grokked    Not grokked
Recon MSE    0.6–1.2     11–24
Classifier   1.00        0.06–0.63
```
High reconstruction MSE with low classifier accuracy = no structure to recover. This is the falsifiability criterion for whether the method *should* find something.

**3. IOI result: structured pi-SAE vs. DAS vs. NL-DAS at k=1 and k=2**
From the full results table in the slides:
- IOI, k=1: DAS=0.19, NL-DAS=1.00 (vacuous), Str. pi-SAE=**0.99**
- IOI, k=2: DAS=0.30, NL-DAS=1.00 (vacuous), Str. pi-SAE=**0.98**

This is the headline result for Paper 2. It's not in the draft.

**4. The "subspaces to factors" connection (sec 12b in slides):**
The slides mention connecting to weight-space / factorization work. This is where your other papers (the factorization circuits draft) connect in. If you're citing the Grassmannian paper in the weight paper, you probably want this section developed.

---

## Concrete Next Steps

### For TMLR submission (Paper 1, Atlas):

1. **Run 10-seed stability for Power and Composite Addition.** This is the bottleneck. Stochastic grokking is a novel claim; reviewers will want to know the base rate of grokking.

2. **Visualize the S¹ failure case.** Take the grokked modular addition model, project DAS activations to 2D, show the circle. Then show that DAS is trying to find a straight line in a circular space. This is one figure that makes the whole paper legible.

3. **Make the VAE results section a pointer instead of a TODO.** Write: "We introduce structured sparse autoencoders for nonlinear causal recovery in the companion paper [cite Paper 2]. Here we report two findings that motivate that work: (a) unconstrained NL-DAS is vacuous [Table X], (b) structured pi-SAE achieves IIA=1.0 on all grokked operations and IIA≈0 on all non-grokked operations [cite Paper 2, Table Y]."

4. **Fill in references.** The `\citep{?}` placeholders need actual keys. The main missing ones: Geiger et al. 2021 (causal abstraction), Geiger et al. 2024 (finding alignments / DAS), Nanda et al. 2023 (progress measures), Zhong et al. 2024 (clock and pizza), Makelov et al. 2024 (illusion paper), Power et al. 2022 (grokking), Liu et al. 2023 (grokking as compression), Thilak et al. 2022 (slingshot).

### For TMLR submission (Paper 2, pi-SAE):

1. **Fill the diversity ratio table.** This is the most urgent gap. The NL-DAS vacuity claim rests on ρ ≈ 0 for NL-DAS on IOI. Run this.

2. **Add the 2×2 ablation as a proper table in the draft.** It's in the slides but not the paper.

3. **Run cross-distribution generalization.** Disjoint names test is straightforward: you already have the IOI setup. Just swap name sets.

4. **Connect to the IOI circuit.** The IOI circuit is well-characterized (Wang et al. 2022, name mover heads). Does the pi-SAE's `z_causal` at k=2 align with the name mover head outputs? This would be the strongest possible validation — not just IIA but actual correspondence with known mechanistic structure.

---

## The Core Message for Both Papers

**Paper 1 (one sentence):** Grokking is the phase transition that converts unstructured memorization into a structured causal variable, and the Grassmannian geometry of that variable is sharp, predictable, and sometimes stochastically absent from the same architecture.

**Paper 2 (one sentence):** Structured sparse autoencoders with a task-conditional prior recover faithful causal variables where both linear DAS and unconstrained nonlinear methods fail, and the distinction matters because unconstrained nonlinearity achieves perfect IIA vacuously.

**The bridge between them:** The structured pi-SAE is the right nonlinear method *precisely because* the causal variables that emerge from grokking have a specific structure (circular/equivariant) that linear DAS cannot represent but that a sparse autoencoder with the right prior can.

---

## What TMLR Reviewers Will Focus On

For TMLR specifically (not a conference, open reviewing, reproducibility emphasized):

**Paper 1 likely concerns:**
- "Stochastic grokking" is a strong claim on N=2 seeds for the stochastic class. You need more seeds.
- Is the S¹ representation for mod-add actually new? Nanda et al. 2023 already documented the Fourier/circular structure. Your contribution is showing DAS specifically returns IIA=0 because of it, which IS novel, but you need to frame it right.
- Single-layer transformer only. Scope it clearly.

**Paper 2 likely concerns:**
- The diversity ratio ρ metric needs clear theoretical motivation — why is this the right falsifiability criterion?
- IOI: N=1 model (GPT-2 small). Is this a sufficiently diverse validation?
- The connection to iVAE identifiability theory should be made explicit in a proposition or theorem, not just cited.

---

*Generated from reading: `main.tex` (full draft), `grokking_geometry_v1.tex` (slides v1), `structured_pi_sae_v1.tex` (slides v1), plus context from prior conversation on factorization circuits, MECHVAL, and the weight-space paper.*
