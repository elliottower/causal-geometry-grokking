# Wild Domain Experiments — Cross-Domain Mathematical Connections

## Motivation

These experiments connect the Grassmannian subspace analysis to deep results
from other mathematical domains: random matrix theory, algebraic combinatorics,
information theory, causal inference, statistical physics, topology, number
theory, quantum mechanics, and algebraic geometry. Each produces a quantitative
test of a cross-domain prediction about grokking.

## Scripts

### Phase 1: No new training required (use existing models + DAS)

| # | Script | Domain | What it measures |
|---|--------|--------|------------------|
| 1 | `rmt_gue_spacing.py` | Random Matrix Theory | GUE vs Poisson eigenvalue spacing of DAS covariance |
| 2 | `matroid_stratification.py` | Combinatorics | Matroid rank profile + Plücker Gini coefficient |
| 3 | `persistent_homology.py` | Topological Data Analysis | H0/H1 persistence of DAS projection point cloud |
| 4 | `gauss_sum_weil.py` | Number Theory | Circle radius vs Weil bound sqrt(p), Fourier mode ID |
| 5 | `motivic_consistency.py` | Algebraic Geometry | Cross-probe consistency (IIA vs equivariance vs k*) |
| 6 | `causal_interventional_distance.py` | Causal Inference | Causal Lipschitz constant d_causal / d_Gr |
| 7 | `mdl_complexity.py` | Information Theory | MDL optimal k vs algorithmic complexity proxy |

### Phase 2: Requires checkpoint trajectory

| # | Script | Domain | What it measures |
|---|--------|--------|------------------|
| 8 | `landau_critical_exponents.py` | Statistical Physics | Power-law exponent beta at grokking transition |
| 9 | `berry_phase.py` | Quantum Mechanics | Geometric phase accumulated by DAS along training |

### Phase 3: Spinoff experiments (checkpoint trajectory + extended analysis)

| # | Script | Domain | What it measures |
|---|--------|--------|------------------|
| 10 | `lyapunov_exponents.py` | Dynamical Systems | Maximal Lyapunov exponent of DAS trajectory — chaos vs convergence |
| 11 | `fisher_information_geometry.py` | Information Geometry | Fisher information matrix on Grassmannian — second Riemannian metric |
| 12 | `entanglement_entropy.py` | Quantum Information | Von Neumann entropy of causal/non-causal block decomposition |
| 13 | `rg_flow_k_sweep.py` | Renormalization Group | k-sweep as RG flow — beta function, fixed points, dimension classification |
| 14 | `symplectic_area.py` | Symplectic Geometry | Kahler form area enclosed by dual-seed training trajectories |
| 15 | `morse_landscape.py` | Morse Theory | IIA landscape topology — critical points, Hessian index, basin widths |

### Phase 4: Open questions from the minimal representation proof

Based on `das_minimal_representation_proof.tex` — the proof that DAS learns
the minimal faithful real representation of Z/pZ (k*=2 via Pontryagin duality
+ Weil bound). These experiments test the proof's 4 open gaps plus untested
predictions.

| # | Script | Tests | What it measures |
|---|--------|-------|------------------|
| 16 | `equivariance_emergence.py` | Gap 1 | Rotation matrix convergence during training — is equivariance an SGD attractor? |
| 17 | `grokking_onset_predictors.py` | Gap 2 | 6 signals tracked per checkpoint — which predicts grokking earliest? |
| 18 | `stochastic_representation_anatomy.py` | Gap 3 | 20 seeds per stochastic op — representation structure in grok vs non-grok outcomes |
| 19 | `fourier_superposition_mdl.py` | Gap 4 | Active Fourier modes + MDL tradeoff — why k* blows up for never-class |
| 20 | `product_group_dimension.py` | Thm 7.1 | Z/pZ x Z/pZ operations — tests novel k*=4 prediction |
| 21 | `fourier_mode_selection.py` | Prop 3.2 | Which of (p-1)/2 equivalent irreps does SGD select? |
| 22 | `rotation_matrix_convergence.py` | Core claim | Direct verification: action matrices are rotation by 2pi*k/p |
| 23 | `composite_moduli_representation.py` | Extension | Composite moduli (n=6,10,15,...) — representation theory of Z/nZ |

### Phase 5: Further cross-domain connections

| # | Script | Domain | What it measures |
|---|--------|--------|------------------|
| 24 | `spectral_graph_partition.py` | Spectral Graph Theory | Fiedler vector + Laplacian spectral gap on operation distance graph |
| 25 | `coding_theory_circle.py` | Coding Theory | PSK constellation quality, d_min, packing efficiency, SNR threshold |
| 26 | `optimal_transport_grokking.py` | Optimal Transport | W_2 distance to ideal circle during training — grokking as transport event |
| 27 | `tensor_rank_analysis.py` | Linear Algebra | Effective rank, stable rank, SV gap — rank should match k* |
| 28 | `spin_glass_order_parameter.py` | Statistical Mechanics | Edwards-Anderson q_EA from multi-seed overlaps — ferromagnet vs spin glass |
| 29 | `padic_grassmannian.py` | p-adic Geometry | p-adic distance vs Grassmannian distance correlation |
| 30 | `arithmetic_height.py` | Arithmetic Geometry | Weil height of Plucker coordinates as algebraic complexity |
| 31 | `modular_forms_connection.py` | Analytic Number Theory | Fourier coefficients vs Gauss sums and L(1,chi) values |

### Phase 6: Theorem-testing experiments (from open_theorems_conjectures.tex)

Tests the 6 provable theorems + 2 conjecture predictions from the companion proof document.

| # | Script | Tests | What it measures |
|---|--------|-------|------------------|
| 32 | `weil_radius_multi_prime.py` | E1, Weil bound | r_p / sqrt(p) = const across primes p ∈ {89,97,101,109,113,127} |
| 33 | `spectral_gap_convergence.py` | E2, Thm 2.6 | Spectral gap Δ_s at convergence predicts grokking (AUC > 0.9) |
| 34 | `spectral_gap_initialization.py` | E3, Thm 2.6 | Spectral gap at step 0 predicts grokking fate (AUC > 0.7) |
| 35 | `plucker_cv_trajectory.py` | E4, Conj 4.1 | Plücker CV jumps discontinuously at grokking transition |
| 36 | `weight_decay_mode_selection.py` | E5, Thm 2.1 | Weight decay λ ≥ 0.5 selects single Fourier mode in >95% of seeds |
| 37 | `gcd_polynomial_grokking.py` | E6, Thm 2.5 | a^n + b^n groks iff gcd(n, p-1) = 1 |
| 38 | `affine_coefficient_test.py` | E7, Thm 2.4 | αa + βb groks iff α ≡ ±β mod p |
| 39 | `winding_number_fourier.py` | E8, Thm 2.3 | Winding number of DAS centroids = dominant Fourier mode index |

## Data Flow

All scripts train grokking models from scratch and fit DAS, sharing utilities
from `grokking_nonlinear_hunt.py` (same pattern as geometry/).

Phase 1 scripts need only one trained model + DAS fit per operation.
Phase 2 scripts need checkpoint-saving during training.

## Key Predictions

- **RMT**: Grokked → GUE spacing (level repulsion). Memorized → Poisson.
- **Matroid**: Grokked → high Plücker Gini (sparse, structured). Memorized → low Gini (uniform).
- **Persistent homology**: Grokked → one persistent H1 loop (the Fourier circle). Memorized → no H1.
- **Gauss/Weil**: Circle radius / sqrt(p) = constant across all group-action ops. k*=2 because Weil bound.
- **Motivic**: High inter-probe correlation post-grokking (IIA, equiv, k* all agree).
- **Causal Lipschitz**: L_grokked << L_memorized by 10x+ (smooth vs discontinuous causal map).
- **MDL**: k_MDL correlates with logical depth of the operation, not string length.
- **Landau**: Same beta exponent within algebraic class = same universality class.
- **Berry phase**: Nonzero for stochastic ops (holonomy), near-zero for always-class.
- **Lyapunov**: Positive exponent during memorization (chaotic), crosses zero at grokking (stable attractor).
- **Fisher info**: High condition number for memorized (anisotropic). Low for grokked (isotropic, flat plateau).
- **Entanglement**: Low von Neumann entropy for grokked (clean causal/non-causal split). High for memorized.
- **RG flow**: Fixed point at k*=2 for group actions. Beta function zero-crossing identifies natural dimension.
- **Symplectic area**: Nonzero for stochastic ops (nontrivial holonomy). Near-zero for always-class.
- **Morse landscape**: Single deep basin for always-class. Multiple shallow basins for stochastic. No clear basin for never.
- **Equivariance emergence**: Rotation error drops BEFORE loss transition (equivariance is an attractor).
- **Grokking onset**: Spectral gap or Fourier magnitude fires earliest (before IIA or equivariance).
- **Stochastic anatomy**: Grokked seeds have same representation structure as always-class; non-grokked have high rotation error.
- **Fourier superposition**: Never-class has many active modes; k* ~ 2 * n_active_modes.
- **Product group**: k*=4 for Z/pZ x Z/pZ (2 dims per cyclic factor).
- **Mode selection**: All seeds pick same Fourier mode k (SGD has a selection mechanism beyond representation theory).
- **Rotation matrix**: Action matrices have stretch_error < 0.05 and angle matches 2*pi*k/p.
- **Composite moduli**: k* depends on CRT decomposition; even n allows 1D faithful reps.
- **Spectral graph**: Fiedler vector cleanly separates the three classes; large spectral gap.
- **Coding theory**: Grokked = near-perfect PSK code (code_quality ~ 1). Memorized = poor code.
- **Optimal transport**: W2_to_circle drops sharply at grokking. W2_velocity spikes.
- **Tensor rank**: Effective rank ~ 2 for grokked (matches k*). ~ d for memorized.
- **Spin glass**: Always-class = ferromagnet (q_EA ~ 1). Stochastic = spin glass. Never = paramagnet.
- **p-adic**: Grassmannian distance correlates with p-adic distance (polynomial degree).
- **Arithmetic height**: Low height = simple subspace (always-class). High = complex (never-class).
- **Modular forms**: Network Fourier coefficients proportional to Gauss sums g(chi_k).
- **Weil radius**: r_p / sqrt(p) = constant across primes (CV < 0.1).
- **Spectral gap (convergence)**: Δ_s at final epoch predicts grokking with AUC > 0.9.
- **Spectral gap (init)**: Δ_s at step 0 predicts grokking fate with AUC > 0.7 (shocking if true).
- **Plucker CV**: CV jumps discontinuously at grokking epoch (tropical degeneration).
- **Weight decay modes**: λ ≥ 0.5 → single Fourier mode in >95% of seeds.
- **gcd polynomial**: a^n + b^n groks iff gcd(n, p-1) = 1 (bijectivity criterion).
- **Affine coefficients**: αa + βb groks iff α ≡ ±β mod p.
- **Winding number**: Winding number of centroids = Fourier mode index in >99% of models.

## Extra Dependencies

```bash
pip install ripser persim    # Persistent homology (Exp 3)
pip install sympy            # Character theory (Exp 4)
```

## Compute Estimates

| Script | GPU time | Notes |
|--------|----------|-------|
| rmt_gue_spacing | ~1 hr | 14 ops × DAS fit + eigenvalue analysis |
| matroid_stratification | ~1 hr | 14 ops × rank sampling + Plücker |
| persistent_homology | ~1 hr | 14 ops × Rips complex |
| gauss_sum_weil | ~1 hr | 14 ops × circle fitting |
| motivic_consistency | ~30 min | Lightweight post-processing |
| causal_interventional_distance | ~2 hr | 14 ops × perturbation sweep |
| mdl_complexity | ~2 hr | 14 ops × k-sweep with classifier |
| landau_critical_exponents | ~2 hr/op | Dense checkpoints needed |
| berry_phase | ~2 hr/op | Full checkpoint trajectory |
| lyapunov_exponents | ~2 hr/op | Checkpoint trajectory + SVD per step |
| fisher_information_geometry | ~2 hr | 14 ops × tangent perturbation sweep |
| entanglement_entropy | ~1 hr | 14 ops × covariance decomposition |
| rg_flow_k_sweep | ~2 hr | 14 ops × k=2..32 DAS sweep |
| symplectic_area | ~2 hr/op | Dual-seed checkpoints + Kahler form |
| morse_landscape | ~2 hr | 14 ops × random Grassmannian sampling + Hessian |
| equivariance_emergence | ~2 hr/op | Checkpoint trajectory + per-ckpt DAS + rotation fit |
| grokking_onset_predictors | ~2 hr/op | Checkpoint trajectory + 6 signals per ckpt |
| stochastic_representation_anatomy | ~3 hr | 20 seeds x stochastic ops |
| fourier_superposition_mdl | ~2 hr | 14 ops x k=2..32 DAS + Fourier analysis |
| product_group_dimension | ~2 hr | Custom Z/pZ^2 data + k-sweep |
| fourier_mode_selection | ~2 hr | 10 seeds x group-action ops |
| rotation_matrix_convergence | ~1 hr | 14 ops x centroid rotation fit |
| composite_moduli_representation | ~3 hr | 8 moduli x 2 ops x k-sweep |
| spectral_graph_partition | ~1 hr | 14 ops × DAS + Laplacian eigenvectors |
| coding_theory_circle | ~1 hr | 14 ops × centroid analysis + SNR sweep |
| optimal_transport_grokking | ~2 hr/op | Checkpoint trajectory + W_2 computation |
| tensor_rank_analysis | ~1 hr | 14 ops × SVD + k-sweep |
| spin_glass_order_parameter | ~3 hr | 10 seeds × 14 ops |
| padic_grassmannian | ~1 hr | 14 ops × distance matrices |
| arithmetic_height | ~1 hr | 14 ops × Plucker coordinate computation |
| modular_forms_connection | ~1 hr | 3 ops × Gauss sum + L-function comparison |
| weil_radius_multi_prime | ~2 hr | 6 primes × train + DAS + circle fit |
| spectral_gap_convergence | ~4 hr | 2 ops × 20 seeds × train + covariance |
| spectral_gap_initialization | ~4 hr | 2 ops × 20 seeds × step-0 cov + full train |
| plucker_cv_trajectory | ~3 hr | 2 ops × 100 checkpoints × DAS + Plucker |
| weight_decay_mode_selection | ~5 hr | 5 λ × 10 seeds × train + Fourier analysis |
| gcd_polynomial_grokking | ~3 hr | 5 exponents × custom data + train + DAS |
| affine_coefficient_test | ~3 hr | 4 coefficient pairs × train + DAS k=2,4 |
| winding_number_fourier | ~4 hr | 6 ops × 10 seeds × train + DAS + winding |
| **Total** | **~81 hr** | **~$14.50 on RunPod A4000** |
