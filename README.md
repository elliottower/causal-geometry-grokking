# When Does Linear Causal Abstraction Work?

Mapping the boundary between linear and nonlinear causal variables in neural networks. We construct an atlas of 14 modular arithmetic operations spanning the Grassmannian boundary, show that grokking is the mechanism that determines which side you land on, and demonstrate that unconstrained nonlinear methods (NL-DAS) achieve perfect IIA vacuously via lookup tables while a structured VAE recovers genuine nonlinear causal variables.

## Slides

**[Slides](slides/grassmannian_atlas_slides_v1.pdf)** ([LaTeX source](slides/grassmannian_atlas_slides_v1.tex)) --- comprehensive walkthrough covering: what DAS is and why it assumes linearity, the atlas of 14 operations, grokking as the mechanism, the NL-DAS vacuity problem, structured VAE as the fix, intervention faithfulness metrics, cross-task transfer validation on GPT-2 IOI, and practical implications.

To compile:
```bash
cd slides
pdflatex grassmannian_atlas_slides_v1.tex
```

## Paper

Draft paper in `paper/main.tex`. Working title: "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian."

## Key findings

1. **Three-class partition**: 14 operations split into Always Grassmannian (7), Stochastic (2), and Never Grassmannian (5) --- the boundary is sharp and governed by grokking
2. **Stochastic grokking**: same operation, same hyperparameters, opposite outcomes from random initialization alone --- Grassmannian variables appear if and only if the model generalizes
3. **Linear DAS returns zero IIA** on grokked modular addition at $k \leq 16$, confirming the causal variable is fundamentally nonlinear (lives on $S^1$, not a linear subspace)
4. **Memorization produces high IIA**: squaring and cubing achieve IIA $\geq 0.86$ at $k=2$ without grokking --- IIA alone cannot distinguish genuine causal variables from lookup tables
5. **NL-DAS is vacuous**: unconstrained nonlinear featurizers achieve perfect IIA by learning degenerate encoder-decoders (diversity ratio $\approx 0$)
6. **Structured VAE recovers nonlinear causal structure**: the ELBO prevents degeneracy; equivariance exceeds 95% for grokked operations
7. **Cross-task transfer validates causal variables**: VAE trained on one IOI template transfers to unseen templates (IIA 0.82--0.96), Jensen's DoubleIO/TripleIO (IIA 0.77--0.81), and across MIB subtask counterfactuals
8. **IOI is NOT Grassmannian**: sheaf consistency analysis shows local DAS subspaces differ significantly across activation clusters (geodesic distance ~2.2, cross-IIA ~0.21)

## Experiments

| Script | What it tests |
|--------|---------------|
| `experiments/grassmannian_geometry.py` | Core atlas: DAS k-sweeps, equivariance, circle geometry for 14 operations |
| `experiments/grokking_das_emergence.py` | DAS emergence during grokking training trajectory |
| `experiments/structured_vae_atlas.py` | Structured VAE across all 14 operations |
| `experiments/sparse_structured_vae.py` | Sparse VAE variants (L1, JumpReLU, TopK) |
| `experiments/k1_vae_vs_das.py` | Head-to-head DAS vs VAE at k=1 |
| `experiments/k1_hard_mode.py` | Hard-example IIA with continuous metrics |
| `experiments/multi_seed_stability.py` | 10-seed stability for stochastic operations |
| `experiments/cross_task_validation.py` | Cross-template transfer, persistent homology, sheaf consistency |
| `experiments/cyclic_and_jensen_validation.py` | Cyclic group equivariance, Jensen DoubleIO/TripleIO transfer |
| `experiments/ioi_subtask_transfer.py` | 8x8 transfer matrix across MIB IOI subtask counterfactuals |
| `experiments/ioi_subtask_transfer_baselines.py` | Baselines: random, per-subtask, joint, NL-DAS (7 parallel pods) |
| `experiments/generate_figures.py` | Generate all paper figures from cached results |

## Results

Pre-computed results are stored on the Modal `fc-results` volume under `grassmannian_atlas/`. Download with:

```bash
modal volume get fc-results results/grassmannian_atlas/ results/
```

## Setup

```bash
# Install dependencies
pip install torch transformer-lens transformers einops matplotlib tqdm datasets

# Run an experiment locally (CPU, will be slow)
python experiments/grassmannian_geometry.py

# Run on Modal GPU (recommended)
modal run --detach experiments/grassmannian_geometry.py
```

Most experiments were run on Modal A100 GPUs.

## License

MIT
