# Paper Review: "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian"

**Review compiled June 23, 2026**  
Author: Elliot Tower | Venue target: TMLR

---

## Executive Summary

This is a well-motivated paper with a genuinely novel empirical contribution (the three-class partition of operations, the IIA=0 result on grokked addition, the 2×2 ablation) and a strong central thesis. The writing is largely clear and the Structured pi-SAE result on GPT-2 hypernymy is striking. However, the paper currently has **seven serious issues** that would likely draw major revisions at TMLR: (1) a title/framing mismatch that undersells the main result, (2) missing citations that reviewers will notice, (3) several pending TODOs that leave critical claims unsubstantiated, (4) scope limitations that are acknowledged but not addressed, (5) a proposition whose proof sketch is not tight, (6) a table with missing values that reads as incomplete, (7) framing of NL-DAS that conflates two distinct failure modes. Below is the full breakdown.

---

## Part 1: Framing and Title Issues

### 1.1 The title buries the lede

The title "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian" frames this as a diagnostic paper about DAS's limits. But the strongest empirical finding is the **Structured pi-SAE achieving IIA=0.97 on GPT-2 hypernymy where DAS gets 0.07** — a 14× improvement on a real language model task. That is a methods contribution, not just a diagnostic.

**Recommendation:** Consider a title like "Beyond the Grassmannian: Diagnosing and Fixing Linear Causal Abstraction with a Structured Sparse VAE" or "Linear Causal Abstraction Fails on Nonlinear Representations: Structured pi-SAE as the Fix." The diagnostic framing is important but should be secondary to the positive contribution.

### 1.2 The abstract lists four bold claims but the paper delivers them unevenly

The abstract's four numbered claims are:
1. Three-class partition (well-supported by Table 1)
2. IIA=0 on grokked addition (well-supported)
3. pi-SAE achieves 0.97 on hypernymy (supported, but Table 3 has pending values)
4. NL-DAS is vacuous (supported for IOI, but Table 4 has "---" entries everywhere)

Claims 3 and 4 need to either be fully supported or scoped down. A reviewer will immediately flag the pending entries.

### 1.3 The "atlas" framing is used but not fully developed

The introduction promises an "atlas" of 14 operations, but the paper never provides an actual atlas-style visualization (a 2D map, a tree, or even a sorted table by algebraic structure vs. outcome). The word "atlas" suggests a systematic map; the current Table 1 is a sorted list. Either build the atlas properly (e.g., plot operations in a 2D space of group-theoretic complexity vs. Grassmannian outcome) or drop the atlas framing.

---

## Part 2: Missing and Incorrectly Cited Works

### 2.1 Engels et al. (2024) — CRITICAL MISSING CITE

**"Not All Language Model Features Are Linear"** (Engels, Liao, Michaud, Gurnee, Tegmark, ICLR 2025, arXiv 2405.14860) is the most directly relevant concurrent paper and is not cited anywhere. Engels et al. independently discover circular representations for days of the week and months of the year in Mistral 7B and Llama 3 8B — the **same** circular structure you observe in grokked modular arithmetic. They also show these circles are causally used for modular arithmetic computation. This paper:

- Validates your central claim that circular features exist in real language models (not just grokking toys)
- Provides a precedent for calling these features "irreducibly multi-dimensional"
- Makes your IIA=0 on grokked addition much less surprising in hindsight — Engels et al. show that SAE-based decomposition CAN find these circles, suggesting the problem is the linear search not the representation

You must cite this. The recommended framing: "Concurrently, Engels et al. (2024) show that circular features for days/months are causally used in real LLMs; our work shows that the same circular structure causes linear DAS to fail and motivates the nonlinear extension."

### 2.2 Park et al. (2024) — cited as `marks2023geometry` but should be separately cited

The paper cites `marks2023geometry` and `park2024linear` in §5 (Related Work). However, **Park, Choe & Veitch (2024, ICML)** "The Linear Representation Hypothesis and the Geometry of Large Language Models" is directly relevant to your §2.2 framing: they give a formal counterfactual definition of linear representation and prove it connects to linear probing and steering. Your Grassmannian framing is the operational version of their theoretical one. Cite this in §2.2 Background, not just Related Work.

### 2.3 CausalGym (Arora, Jurafsky, Potts, ACL 2024) — missing

CausalGym benchmarks DAS against other causal interpretability methods on linguistic tasks in Pythia models. It finds DAS outperforms probing on most tasks but fails on others. This is directly relevant to your claim that DAS's failures are predictable and your hierarchy of evaluation metrics. The paper should be cited in Related Work under "Causal abstraction."

### 2.4 Pislar et al. (2025) — missing and directly relevant

**"Combining Causal Models for More Accurate Abstractions of Neural Networks"** (Pislar, Magliacane, Geiger, CLeaR 2025) proposes combining multiple causal models to handle partial faithfulness — exactly the situation you encounter in the stochastic class. They define a trade-off between hypothesis strength and faithfulness that formally captures what your "three-class partition" is doing informally. This would strengthen §5 Related Work.

### 2.5 Geiger et al. (2024) citation needs disambiguating

You cite `geiger2024causal` several times but the reference list likely has two 2024 Geiger papers: (1) the CLeaR 2024 DAS paper (geiger2024finding at PMLR 236) and (2) the JMLR 2024 paper "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability" (arXiv 2301.04709). These are different papers. The JMLR theoretical paper is important for §2.1's framing of "causal abstraction" and unification of MI methods; the CLeaR paper is what you're extending experimentally. Make sure these are separate bib entries.

### 2.6 Liu et al. (2023) "Grokking as Compression" — should be cited

Liu, Zhong & Tegmark (2023, arXiv 2310.05918) frame grokking as compression via Linear Mapping Number (LMN). This is the paper that gives compression a precise complexity measure, and it supports your claim that Grassmannian structure is the compressed solution. Your `liu2023grokking` citation is correct but the text says "Liu et al. frame grokking as compression" — check this is the right Liu et al. (there is also Liu et al. 2022 "Towards Understanding Grokking" which is a different paper by a different Liu).

### 2.7 Narayanaswamy et al. (2017) citation is correct but incomplete

You correctly cite Narayanaswamy et al. for the structured VAE framework. However, you should also cite **Siddharth et al. (2017) "Learning Disentangled Representations with Semi-Supervised Deep Generative Models"** (NeurIPS 2017), which is the NeurIPS version of the same work. Check which version is canonical — the arXiv is from the same group but often cited differently.

---

## Part 3: Structural and Writing Issues

### 3.1 The TODOs in the PDF are fatal for submission

There are at least **five explicit TODO comments** left in the compiled PDF:

1. `% TODO: Add CPCA-init DAS hard-example experiments for grokking operations`
2. `% TODO: Report multi-seed stability (10 seeds per stochastic operation)`
3. `% TODO: 10-seed stability results here`
4. `% TODO: Fill with hard-mode results showing NL-DAS+recon IIA values`
5. `% TODO: Fill with disjoint-name and disjoint-template results from hard-mode experiments`

These leave three entire subsections as stubs: §4.4 (stochastic grokking, no multi-seed data), §4.7 (NL-DAS+recon, no results), and §4.8 (cross-distribution, no results). The paper cannot be submitted with these. Either:
- Fill them in before submission (preferred — multi-seed stability and NL-DAS+recon are both runnable now)
- Remove the subsections and fold the claims into limitations
- Label them explicitly as "future work" in the section headings and remove the results framing

### 3.2 Table 4 (NL-DAS vacuous) has "---" everywhere

Table 4 (`tab:nldas_vacuous`) reports IIA for DAS, NL-DAS, NL-DAS+r, and VAE, plus diversity ratios. The columns for NL-DAS+r and the diversity ratios are filled with bold "---" and the caption explains these are "pending from hard-mode experiments." This means the central empirical claim of §4.9 (that NL-DAS is vacuous, diagnosed by diversity ratio) is **not demonstrated in any table**. The IOI diversity ratio numbers (ρ_NL ≈ 0 vs. ρ_pi-SAE ≈ 0.91) are mentioned in the text but don't appear in the table. Fill Table 4 before submission, even if you can only fill the IOI rows.

### 3.3 The iVAE Proposition is stated at "global optimum" but is not empirically verified

Proposition 1 states recovery holds "if the Structured pi-SAE is trained to a global optimum of L." This is a standard theoretical caveat but creates a problem: your ablation table shows pi-VAE (structured prior, no sparsity) achieves IIA=0 on grokked operations despite also being "trained to optimize L." This suggests the global optimum condition is doing real work, not just being formal. You should either:

- Add a sentence explaining why pi-VAE fails despite having a structured prior — the answer is in your text ("without sparsity, the encoder spreads the causal signal across dimensions in a rotation-dependent way") but this should be connected explicitly to why the global optimum in Proposition 1 doesn't apply to pi-VAE
- Or add a corollary: Proposition 1 applies only when the encoder capacity matches the causal dimension k, which pi-VAE with dense k-dimensional z violates because the rotation ambiguity within z is not resolved by the ELBO

This is important because otherwise a reviewer will point out that the Proposition predicts pi-VAE should work (it satisfies the three assumptions) but it doesn't.

### 3.4 The NL-DAS "lookup table" diagnosis conflates two failure modes

Section 4.9 attributes NL-DAS failure to a lookup table mode where "f maps every activation to a space where the first coordinate encodes the class label." This is correct but there are actually two failure modes being conflated:

**Failure mode A (lookup table):** The decoder produces the same output for all base inputs with the same source label, ignoring the base. Caught by the diversity ratio.

**Failure mode B (coordinate collapse):** The encoder maps all activations for the same class to the same latent code, so the decoder never needs to distinguish bases. Caught by reconstruction MSE.

These are the same behavior but at different layers. In your current framing, "diversity ratio ≈ 0" diagnoses both but the mechanism description mixes them. Adding a brief clarification (one paragraph or a figure) would make the section much cleaner.

### 3.5 Section 2.4 (Identifiability) comes before the method is introduced

The identifiability section (§2.4) proves properties of the Structured pi-SAE before the method is described in §3.5. A reader encountering Assumption 1 and Proposition 1 before knowing what the pi-SAE is will be confused. Options:
- Move §2.4 after §3.5, renaming it §3.6 "Identifiability of the Structured pi-SAE"
- Or add a one-sentence forward reference: "We prove this for the Structured pi-SAE, which we define in §3.5"

### 3.6 The modular addition IIA=0 result needs a cleaner explanation

The claim "linear DAS returns IIA=0.0 at every checkpoint from epoch 500 through 25,000, at k≤16" is striking, but the paper doesn't explain *why* this produces exactly zero rather than a small positive number. The explanation given ("a category error: searching Gr(k,d) when the variable lives on S^1") is correct but a reader might ask: "but can't a 2D linear subspace approximate the circle well enough to get some IIA?" 

The answer is yes it can approximate the circle, but DAS measures interchange effectiveness — swapping the *linear projection* of a circular representation does not swap the causal variable; it just changes which point on the circle is projected. A 1-sentence explanation of this would preempt the obvious question. Something like: "Projecting a circular representation onto any 2D linear subspace creates a map from S^1 to a 2D plane — the projection is non-injective when the circle is not contained in the plane, so swapping projections across two inputs does not swap the causal variable."

---

## Part 4: Experiment Design Issues and New Experiments to Add

### 4.1 The "stochastic" class needs 10+ seeds to be convincing (currently TODO)

The stochastic class has 2 operations × 2 seeds. This is too small to establish that the variation is genuinely stochastic (random seed dependent) rather than due to some other hyperparameter. The TODO for 10-seed stability is the right call — prioritize this.

**Proposed experiment:** For the two stochastic operations (Power, Composite Addition), run 10 seeds. Report: (a) grokking rate (fraction of seeds that grok), (b) correlation between grokking and equivariance, (c) whether IIA transitions sharply when equivariance does. If grokking rate is ~50% for these operations, this strongly supports the "stochastic Grassmannian structure" claim.

### 4.2 The grokking-Grassmannian link needs a training dynamics plot

The paper claims "the model produces a Grassmannian variable if and only if it generalizes" but only shows end-state results. A plot of IIA and equivariance over training epochs — showing them both jump at the grokking transition — would be far more compelling than a table. This is the natural "smoking gun" figure for the paper.

**Proposed experiment:** For multiplication (always Grassmannian) and squaring (never Grassmannian), plot IIA(k=2), equivariance, and test loss vs. training epoch. For multiplication, IIA and equivariance should jump at the same epoch as test loss. For squaring, IIA should remain flat (high) while equivariance stays low throughout.

### 4.3 The Structured pi-SAE E2E on stochastic/never-Grassmannian operations

The ablation table (Table 2) shows pi-SAE achieves IIA=1.0 on grokked operations and ≈0 on non-grokked ones. But what happens to a stochastic operation at a seed where grokking didn't occur? Does the pi-SAE also return near-zero? This would be the strongest test of whether the pi-SAE is detecting genuine causal structure (predicts near-zero) or just fitting to whatever signal is available (predicts nonzero on non-grokked seeds).

### 4.4 Hard IIA vs. standard IIA comparison

Section 3.3 introduces hard examples (base-source pairs where the correct output differs) but the results tables use standard IIA throughout. Adding a Hard IIA column to Tables 1 and 3 would (a) directly use the CPCA-init methodology you mention, and (b) stress-test the claim that memorized models achieve high IIA vacuously. Prediction: hard IIA will be much lower for squaring/cubing (which have no structure) but similar to standard IIA for multiplication/division (which have genuine structure).

### 4.5 Continuous distributional metrics for the always-Grassmannian class

Section 4.10 promises continuous metric results (KL, JS, normalized logit difference) for operations where IIA saturates. This is currently a TODO subsection. Even filling it in for just multiplication (DAS vs. VAE vs. NL-DAS) would be enough to establish the point.

### 4.6 Connect to the Engels et al. circle findings

Engels et al. (2024) show that Mistral 7B and Llama 3 8B causally use circular representations for days-of-week arithmetic. Your Structured pi-SAE should achieve high IIA on these tasks where DAS fails. **Proposed experiment:** Evaluate DAS and Structured pi-SAE on days-of-week modular arithmetic in Mistral 7B (the standard Engels et al. task). This would:
- Extend your grokking results to a pretrained language model
- Directly respond to the "single-layer synthetic only" limitation
- Provide a bridge between your grokking experiments and your GPT-2 language model experiments

---

## Part 5: Framing and Positioning Improvements

### 5.1 The hierarchy in §5.1 should be in the Introduction, not just Discussion

The four-level hierarchy (IIA → faithfulness → distributional fidelity → structural diagnostics) is the paper's main conceptual contribution after the pi-SAE results. It currently appears only in §5.1 Discussion. Moving a condensed version (2-3 sentences + a numbered list) to the Introduction paragraph after the contributions would make the paper's structure much clearer from the start.

### 5.2 The Proposition should be presented as a theorem, not a proposition

The iVAE recovery result is the theoretical core of the paper and fully cites prior work (Khemakhem et al. 2020). "Proposition" implies a minor claim; this is the central justification for why the pi-SAE works where NL-DAS doesn't. Renaming it Theorem 1 and giving the proof sketch a full proof (or pointing to Appendix A with a complete proof) would strengthen the paper considerably.

### 5.3 The NL-DAS failure should be framed as a general result about end-to-end interchange training

Currently the NL-DAS failure is presented as a specific finding about NL-DAS. But it generalizes: **any encoder-decoder trained end-to-end on interchange loss without a reconstruction constraint will degenerate to a lookup table.** This is a general negative result about a class of nonlinear causal abstraction methods. State it as such in §5.3 Implications, and cite Locatello et al. (2019) who proved the analogous impossibility theorem for unsupervised disentanglement. The connection: NL-DAS is to causal abstraction what unsupervised VAE is to disentanglement — both require inductive biases (reconstruction + supervised prior) to avoid degenerate solutions.

### 5.4 Scope: Single-layer vs. multi-layer transformers

The limitation (§5.4) acknowledges "single-layer transformers on synthetic tasks." This is the most significant weakness reviewers will raise. Consider adding one of the following to strengthen the paper:

**Option A (preferred):** Add a 2-layer transformer experiment. If the same three-class partition holds for 2-layer models (which is testable with existing code), the result is far more general. This could be a single paragraph in §4.

**Option B:** Add a theoretical argument for why single-layer results generalize. For grokking operations, the circular representation is in the residual stream *post-MLP*, which is also where DAS is applied. Multi-layer models have more complex residual stream dynamics, but the key claim (the causal variable is on S^1, not in Gr(k,d)) should still hold for operations with modular structure.

**Option C:** Explicitly connect the GPT-2 hypernymy result as partial evidence for multi-layer generalization. GPT-2 is a 12-layer model; the Structured pi-SAE at layer 8 is already working in a multi-layer setting.

---

## Part 6: Minor Writing Issues

| Location | Issue | Fix |
|----------|-------|-----|
| Abstract, contribution (1) | "Seven always produce Grassmannian variables" — Table 1 shows seven in Always class but text says "seven *always*" which could mean any seed | Clarify: "seven in all tested seeds" |
| §2.1, eq. (1) | The intervention equation uses $h' = h_b - QQ^\top h_b + QQ^\top h_s$ — this is correct but slightly non-standard. Geiger et al. write it as $h_b + QQ^\top(h_s - h_b)$, which is equivalent and easier to read | Consider rewriting as $h' = h_b + QQ^\top(h_s - h_b)$ |
| §3.1, Table 1 | "Composite addition* (unknown seed)" — the asterisk notation is confusing. What does "original seed" mean? Just say "seed unrecorded" | Fix caption |
| §3.1 | The stochastic class has 4 rows (Comp. addition × 2 seeds, Power × 2 seeds). The "Class" column says "Stochastic" for all four. But Composite addition at seed 42 *didn't* grok — should this be "Never" for that row? | Clarify the class assignment logic: is "Stochastic" a property of the operation or the (operation, seed) pair? |
| §4.7 | Claims "NL-DAS+recon degrades" but shows no numbers | Either add the numbers or say "we conjecture NL-DAS+recon degrades; this is confirmed in Fig. X [in preparation]" |
| §4.2 | "k-sweeps" should be "$k$-sweeps" consistently throughout | Fix LaTeX |
| §5.3 | "Prior work proposing neural network featurizers for causal abstraction (Geiger et al., 2024) should be re-evaluated with diversity ratio and reconstruction checks" | This is a strong claim about another paper. Soften or substantiate with data showing specifically that Geiger et al.'s method has the degeneracy problem |

---

## Part 7: Reference List Additions

The following citations should be added. They are either missing entirely or underutilized:

```
@article{engels2024notall,
  title={Not All Language Model Features Are Linear},
  author={Engels, Joshua and Liao, Isaac and Michaud, Eric J. and Gurnee, Wes and Tegmark, Max},
  journal={ICLR 2025},
  year={2024},
  note={arXiv:2405.14860}
}

@inproceedings{arora2024causalgym,
  title={CausalGym: Benchmarking causal interpretability methods on linguistic tasks},
  author={Arora, Aryaman and Jurafsky, Dan and Potts, Christopher},
  booktitle={ACL 2024},
  year={2024}
}

@inproceedings{park2024linear,
  title={The Linear Representation Hypothesis and the Geometry of Large Language Models},
  author={Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  booktitle={ICML 2024},
  year={2024}
}

@inproceedings{pislar2025combining,
  title={Combining Causal Models for More Accurate Abstractions of Neural Networks},
  author={P{\^i}slar, Theodora-Mara and Magliacane, Sara and Geiger, Atticus},
  booktitle={CLeaR 2025},
  year={2025}
}

@article{geiger2023causal_foundation,
  title={Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability},
  author={Geiger, Atticus and Ibeling, Duligur and Zur, Amir and Chaudhary, Maheep and others},
  journal={JMLR},
  year={2024},
  note={arXiv:2301.04709}
}

@inproceedings{mueller2025mib,
  title={MIB: A Mechanistic Interpretability Benchmark},
  author={Mueller, Aaron and Geiger, Atticus and others},
  booktitle={arXiv:2504.13151},
  year={2025}
}

@article{liu2023grokking_compression,
  title={Grokking as Compression: A Nonlinear Complexity Perspective},
  author={Liu, Ziming and Zhong, Ziqian and Tegmark, Max},
  year={2023},
  note={arXiv:2310.05918}
}
```

---

## Summary Checklist Before Submission

### Must-fix (fatal for submission)
- [ ] Fill in or remove all TODO subsections (§4.4, §4.7, §4.8)
- [ ] Fill Table 4 diversity ratio columns for at least IOI rows
- [ ] Add multi-seed stability results for stochastic class (even 5 seeds would help)
- [ ] Add Engels et al. (2024) citation — this is the most glaring omission
- [ ] Disambiguate the two Geiger et al. 2024 papers in the bib

### Should-fix (would draw reviewer criticism)
- [ ] Reconsider title to reflect the positive pi-SAE contribution
- [ ] Move four-level hierarchy to Introduction
- [ ] Add training dynamics figure (IIA + equivariance vs. epoch)
- [ ] Add CausalGym, Park et al. (2024), Pislar et al. (2025) citations
- [ ] Fix Table 1 stochastic class assignment ambiguity
- [ ] Clarify why Proposition 1 doesn't apply to pi-VAE

### Nice-to-have (strengthen the paper)
- [ ] Rename Proposition 1 to Theorem 1
- [ ] Add Hard IIA column to Tables 1 and 3
- [ ] Add 2-layer transformer or Engels et al. days-of-week experiment
- [ ] Section on continuous distributional metrics (KL/JS table for always-Grassmannian class)
- [ ] Add Liu et al. 2023 grokking-as-compression citation to §2.4

