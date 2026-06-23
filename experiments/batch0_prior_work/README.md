# Batch 0: Prior Work from factorization-circuits

Experiments, writeups, and results from the factorization-circuits repo that are
relevant to the grokking/Grassmannian paper. Copied here for reference — these
ran on GPT-2 with factorized transformer decompositions (shared factor bank +
per-projection selectors).

## Key findings relevant to this paper

1. **Fourier structure comes AFTER grokking, not always present** — post-grokking
   representations are clean (no superposition), so SAE-based circuit discovery fails.
   Grassmannian subspace metrics needed instead. (TODO_EAP.md)

2. **Sign anti-correlations in DAS dimensions** — Fourier-like sign encoding detected
   (r ≈ -0.75 between dimension pairs), encoding computational direction. (ATLAS_SUMMARY.md)

3. **Anti-DAS confirms genuine causal variables** — orthogonal complement to DAS subspace
   carries anti-correlated information: IIA_Q >> IIA_perp (IOI 0.938 vs 0.162). (TECHNICAL_REPORT_V3.md)

4. **Canonical angles between task subspaces** — IOI and SVA subspaces are near-orthogonal
   (62.1° - 75.5°), confirming distinct causal variables. (TECHNICAL_REPORT_GRASSMANNIAN_EXPERIMENTS.md)

5. **Three-stream multiplexing discovered unsupervised** — dup_token (34%), positional (22%),
   S_inhib (22%) via factor clustering. (ATLAS_SUMMARY.md)

## Directory structure

```
batch0_prior_work/
├── README.md                          # This file
├── MASTER_DAS_RESULTS.md              # All DAS results across checkpoints and tasks
├── DAS_RESULTS_CATALOG.md             # Factorized DAS methods comparison
├── grassmannian_atlas/                # 89 analyses across 12 batches
│   ├── ATLAS_SUMMARY.md              # Main summary of all findings
│   ├── TECHNICAL_REPORT_GRASSMANNIAN_EXPERIMENTS.md  # Conceptor steering, canonical angles
│   ├── TODO_EAP.md                   # Grokking EAP plan (Fourier-after-grokking finding)
│   ├── CONCEPTOR_STEERING_WRITEUP.md # Steering with DAS subspaces
│   ├── INTUITION_CAUSAL_GEOMETRY.md  # Conceptual framing
│   ├── RESULTS_*.md                  # Per-analysis batch results (15 files)
│   ├── factor_dag_results.json       # Factor DAG discovery results
│   └── causal_discovery_results/     # Interventional data + scripts
├── paper_b_grassmannian/             # Extended Grassmannian experiments
│   ├── PLAN.md                       # Experiment plan
│   ├── TECHNICAL_REPORT.md           # V1 results
│   ├── TECHNICAL_REPORT_V2.md        # V2 results
│   └── TECHNICAL_REPORT_V3.md        # V3 results (58 experiments, anti-DAS, Riemannian DAS)
└── mechviews_audit/
    └── audit_grokking.jsonl          # Mechviews validity audit of OUR methods/claims applied to grokking
```

## Source

All files copied from `factorization-circuits/experiments/batch6_atlas/06_13_2026/`
and related directories. Original scripts remain in that repo.
