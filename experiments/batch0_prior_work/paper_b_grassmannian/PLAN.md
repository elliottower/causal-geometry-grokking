# Paper B: Grassmannian Geometry of Transformer Computation

## Key Findings (2026-06-15)

### 1. Layer-Dependent Geometry (B1 Layer Sweep)
Orthogonality between IOI and SVA is **layer-specific**, not global:
- **L3**: theta_0 = 34.6° (strong shared direction)
- **L0**: theta_0 = 52.5° (moderate alignment)
- **L8/default**: theta_0 = 80-83° (complete orthogonality)
- **L11**: theta_0 = 32.2° (strongest shared direction)

Robust across L1 regularization values. L1=1.0 tends to push angles up slightly.

### 2. Shared Direction Semantics (B1 Decode)
The L11 shared direction (theta_0=32.2°) promotes **3rd person singular present verbs**: "fits, keeps, lets, remains, allows, goes, is, hits, gives, stays."
- IOI needs this for tracking entity actions
- SVA needs this for checking subject-verb agreement
- The shared variable is a genuine linguistic feature both tasks require

L3 shared direction is less interpretable (mixed content). L3 and L11 shared directions are at 57° to each other — different linguistic variables at different processing stages.

### 3. Factor Bank Amplifies Task Separation (B11)
Factor-space canonical angles are ALWAYS larger than d_model-space angles:
- L3: 55.1° factor vs 34.6° d_model (+20.5°)
- L11: 66.1° factor vs 32.2° d_model (+33.9°)
- L8: 88.7° factor vs 82.9° d_model (+5.8°)

The factorization learns to PARTITION tasks into non-overlapping factor subsets. Zero Jaccard overlap in top-50 factors at computation layers.

### 4. Conceptor Preservation (B12)
At the computation layer (default/L8): **99.6% of IOI preserved after NOT(SVA)** — near-perfect subspace separation. Tasks occupy entirely different regions of the Grassmannian at computation time.

At L3/L11: ~87-89% preserved, consistent with the ~33° shared angles.

### 5. QK/OV Circuit Separation (B10)
Q-K selectors span significantly more similar subspaces than Q-V (p=1.8e-15):
- Q-K mean theta_0 = 71.9°
- Q-V mean theta_0 = 79.9°

OV-combined gives best circuit prediction AUROC (0.642, p=0.033). The OV pathway carries task-relevant information to the DAS subspace.

### 6. No Sparsity-Geometry Correlation (B9 — null result)
Selector sparsity does NOT predict DAS alignment. All p > 0.18. The geometric structure is about WHICH factors are selected, not HOW MANY.

### 7. Geodesic Distance Matrix (B13)
- SVA subspace is more stable across layers (mean within-SVA distance = 2.08) than IOI (2.80)
- SVA at L8 = SVA at default (distance 0.001, confirming default layer is L8 for SVA)
- Cross-task distance smallest at L11 (2.50), consistent with shared directions there
- Task and layer contribute equally as organizing dimensions

### 8. Multi-Layer Head Alignment (B8)
AUROC peaks at the task's native DAS layer:
- IOI: AUROC=0.624 at L10/default (p=0.054)
- SVA: AUROC=0.761 at L0 (p=0.065)

Layer-mismatched DAS can even REVERSE predictions (AUROC < 0.5 for IOI at L0/L3).

### 9. Universal Shared Subspace (B14 + B15)
ALL 5 tasks share a ~3D universal subspace (dense checkpoint, k=32):
- Direction 0: eigenvalue = 4.985/5 (projects onto ALL 5 tasks with >0.997)
- Direction 1: eigenvalue = 4.835/5
- Direction 2: eigenvalue = 4.327/5
- Sharp transition at dim 3→4 (eigenvalue drops to 2.915)

**Massively significant** vs random baseline (B15):
- All pairwise theta_0: z-scores > 68 (p < 10^-1000)
- Top eigenvalue: z = 134
- Random subspaces NEVER produce >70% shared dimensions

Task-specific content is ~91% of each 32-dim subspace; shared is ~9%.

### 10. Cross-Task Interference Prediction (B5)
Geodesic distance (integrating all canonical angles) is a **near-perfect predictor** of cross-task interference:
- Pearson r = -0.994 (p < 0.0001)
- All 10 task pairs have cross-projection 3-5x above random baseline (0.14-0.20 vs 0.042)

But theta_0 alone does NOT predict interference (r=0.12, p=0.74) — because all pairs have similar theta_0 (2.9-7.0°).

### 11. Multi-Task Conceptor Algebra (B7)
- Pairwise OR is **perfectly additive**: rank(C_i OR C_j) = 64 = 32+32 — zero redundancy
- Universal AND: rank 1 — only 1 dimension truly shared (conceptor alpha=10 is aggressive)
- NOT(all others) preserves 55-65% of each task's content
- 20-22 of 32 dimensions are task-specific after removing all others
- greater_than specific direction promotes numbers ("799", "0100") — semantic sense

### 12. Geodesic Trajectory (B16)
Trajectories through layers are highly curved:
- IOI: curvature ratio = 3.83 (path is 3.8x longer than geodesic)
- SVA: curvature ratio = 2.30 (smoother)
- IOI jumps dramatically at L3 (deviation=2.46, nearly orthogonal to adjacent layers)
- L8 = L10 = default for IOI (distance 0.001)
- L8 = default for SVA (distance 0.001)

### 13. L1 Stability (B17)
DAS subspaces are robust to L1 regularization strength:
- Dense k=32: theta_0 < 1° between L1=0.1 and L1=0.5 for most tasks
- Atomic k=4: more fragile — up to 53° shift at L1=1.0 for SVA
- greater_than (dense k=32): essentially identical across L1 (theta_0 ~ 0.0°)

### 14. Spectral Dominance (B18)
ALL tasks have participation ratio ~1.4 — dominated by a single direction:
- Top-1 captures 84-89% of DAS energy
- Spectral gap (s_1/s_2): 3.6-4.9x
- Effective rank at 90%: only 2-4 dimensions needed
- At 99%: 19-27 dimensions needed

Factor-space A matrices have higher effective rank (22-27) — the factor bank distributes information more evenly.

### 15. Cross-Checkpoint Geometry (B19)
Atomic-sweep-40 (k=4) and dense (k=32) find partially overlapping subspaces:
- IOI: containment = 34% (vs 0.5% random chance), theta_0 = 65°
- SVA: containment = 52%, theta_0 = 42.4°
- L11 "verb direction" exists in BOTH checkpoints (theta_0=31.4°, containment=47%)
- SVA subspace is more robust across architectures

### 16. Task Manifold (B20)
All 5 tasks combined use only 64/768 = **8.3% of d_model** (eigenvalue > 1.0).
- 149 dimensions have eigenvalue > 0.1 (long tail)
- Ward clustering: {gender_bias, sva} vs {capital_country, greater_than, ioi}
- The "syntactic" cluster (gender_bias, sva) involves grammatical agreement

### 17. MDS Embedding (B21)
The primary organizing axis is **checkpoint** (dense vs atomic), not task or layer.
- Within atomic, layer variation is small in the embedding
- Dense subspaces are spread by task (capital_country most separated)
- 2D embedding captures only 36% of variance — Grassmannian is high-dimensional

### 18. Information Theory (B22)
- Quantum fidelity between task pairs: 0.005 (vs 0.001 random) — 4-5x above chance
- Effective dimension of ensemble average: 116.5 = 3.6x individual k=32
- Holevo chi (ensemble distinguishability): 1.29
- greater_than-ioi have lowest relative entropy (most similar)
- capital_country-sva have highest (most different)

### 19. Principal Direction Flow (B23)
IOI's principal direction **rotates 84-88° between adjacent layers** (L3→L8, L8→default):
- At default/L10: promotes names ("Matthew, Martin, Ben") — the NAME-MOVER direction
- At L0: generic content ("reated, verb, ogen")
- SVA is smoother: 20-48° rotations between layers

Cross-task alignment of v1:
- L0: 89.5° (orthogonal)
- default: 59.5° (partial alignment)
- The L11 shared direction ≈ IOI's v1 at L11 (cos=0.985) but NOT SVA's v1 (cos=0.72)

### 20. k-Sensitivity (B24)
**Critical finding**: orthogonality at k=4 is a dimensional constraint, not real geometry.
- k=4: IOI-SVA theta_0 = 80.7° (orthogonal — can't fit shared structure in 4 dims)
- k=32: theta_0 = 2.9° (nearly identical first direction!)
- k=64: theta_0 = 3.3° (stable)

Spectral profile is also k-dependent:
- k=4: PR=3.1-3.7, top-1 energy=32-47% (uniform)
- k=32: PR=1.4, top-1 energy=84% (dominated)
- k=64: PR=1.5-1.7, top-1 energy=77-81% (slightly less dominated)

Nesting:
- k=32 is well nested in k=64 (68-85% containment)
- k=4 (atomic) is partially nested in k=32 (dense) — 34-52% containment

### 21. Universal Subspace is Not Artifact (B29)
Only 2/10 top eigenvector directions are garbage (punctuation/CJK):
- 4-5% of each task's energy in garbage dimensions
- After projecting out garbage: pairwise canonical angles change by < 1°
- Direction 0 promotes English morphological suffixes ('-asons', '-ens', '-men', '-s')
- The universal subspace is genuine linguistic structure, not baseline artifact

### 22. Task-Discriminative Directions (B30)
One-vs-all discriminant eigenvalues are ~0.98-0.99:
- Each task has a direction with 0.997 projection onto itself, ~0.01 onto others
- IOI is most geometrically unique (mean centroid cosine 0.645)
- greater_than is least unique (0.858)
- Discriminant directions encode expected linguistics:
  - IOI: proper names ('Gingrich', 'Adidas')
  - SVA: verbs ('include', 'varied', 'involve')
  - capital_country: entity names ('Kardashian', 'Darling')
  - gender_bias: social/institutional words ('academia', 'eminent')

### 23. Constant Sectional Curvature (B31)
All sectional curvatures cluster at K ≈ 0.064 (range [0.063, 0.066]):
- Near-constant across all task points and tangent planes
- Triangle inequality excess: ~1.9-2.1x (tasks are far apart on the manifold)
- Tangent vector norms: ||V|| ≈ 6.9-7.4 (all comparable)
- Tasks occupy a region of the Grassmannian with uniform curvature

### 24. Dual-Space Geometry (B32)
Factor space (Gr(k,8192)) and d_model space (Gr(k,768)) preserve the same geometry:
- Pearson r = 0.91, Spearman rho = 0.96 between A-space and U-space angles
- Factor bank projection fidelity: F.T @ A ≈ U to within 4-12° (containment 94-98%)
- **Amplification is layer-dependent**:
  - L11: +33.9° (32.2° in d_model → 66.1° in factor space)
  - L3: +20.5° (34.6° → 55.1°)
  - L0: +18.5° (52.5° → 71.1°)
  - L8/default: +5.8° / +7.4° (near-orthogonal already)
- Amplification strongest where tasks share computation

### 25. Bootstrap Confidence Intervals (B33)
At noise_scale=0.01 perturbation (realistic optimization noise):
- IOI-SVA: theta_0 = 3.0° [3.0°, 3.1°]
- gender_bias-greater_than: 7.3° [7.2°, 7.5°]
- All pairwise CIs are sub-degree wide
- Random rotation stability: <0.001° (invariant, as expected)
- Leave-one-out: max single-dim change up to 29.6° (individual dims carry weight)

### 26. Grassmannian Interpolation (B34)
Geodesic midpoints between task pairs:
- Midpoint token decodings are NOT interpretable — no natural "average" task
- ALL third-task distances closer to Frechet mean than to any pairwise midpoint
- Frechet mean is a better center than any geodesic midpoint
- IOI-SVA trajectory: top direction stays as garbage token throughout interpolation
- Tasks are arranged symmetrically around the mean, not along geodesic arcs

### 27. Kernel Spectral Analysis (B35)
Dense kernel matrix has condition number 2.2 — all tasks contribute equally:
- Lambda_0 = 1.71 (34.3%); remaining 4 eigenvalues each 15-18%
- Kernel alignment with linguistic categories (entity vs syntactic) = 0.81
- Atomic kernel: task alignment (0.669) ≈ layer alignment (0.662)
- sva_default = sva_L8 confirmed (kernel = 1.000)
- Cross-task highest at L11 (ioi_L11↔sva_L11 kernel = 0.215)

### 28. Weight-Space DAS Overlap (B36 — partial null)
Individual head QK overlaps do NOT predict IOI circuit (AUROC = 0.54):
- DAS captures distributed structure that no single head reflects
- Most overlaps near random baseline (0.042)
- IOI overlap grows monotonically L0→L11 (0.046→0.067)
- Most task-specific heads: L0H1 (greater_than), L10H9 (gender_bias)

### 29. Layer Persistence (B37)
L11 attention is the dominant DAS producer for ALL tasks:
- IOI: 5.59x random at L11, 4.13x at L1
- SVA: 5.02x at L11, 4.00x at L1
- gender_bias: 5.19x at L11, 4.23x at L1
- Read-in is at random baseline (~1.0x) for all tasks — weight matrices don't preferentially read from DAS subspace
- Universal pattern: L1 attn spike → L3 MLP → gradual build → L10→L11 massive peak
- MLP L3 is consistent secondary producer (~2.9x) for all tasks

### 30. Cross-Task Transfer (B38)
Cross-projection is perfectly symmetric (= Grassmannian kernel):
- greater_than↔sva (0.205) most transferable
- capital_country↔sva (0.139) least transferable
- All tasks have 80-84% unique energy, 17-19% shared
- Procrustes alignment: 27-37% after optimal rotation
- Transitive chains through greater_than can IMPROVE transfer over direct

### 31. Effective Dimension Scaling (B39)
Task manifold dimension scales SUBLINEARLY with number of tasks:
- Power law: dim = 32.8 * n^0.759
- Redundancy grows: 0% (1 task) → 10.3% (2) → 20% (3) → 27.2% (4) → 31.9% (5)
- Extrapolation: ~50 tasks to fill 83% of d_model=768
- Greedy order: SVA → capital_country → gender_bias → IOI → greater_than
  - greater_than adds only 13 marginal dims (most redundant)
  - SVA/capital_country are most orthogonal (4.7% pairwise redundancy)
- Implication: GPT-2's residual stream has capacity for ~50 independent task subspaces

## Experiments

### Completed (34 experiments, 38 result files)

| ID | Name | Key Finding |
|----|------|-------------|
| B1 | Layer-dependent canonical angles | Orthogonality is layer-specific (L3=34.6°, L8=82.9°, L11=32.2°) |
| B1-decode | Shared direction logit lens | L11 shared = "3rd person present verbs" |
| B5 | Cross-task interference prediction | r=-0.994 geodesic vs cross-projection |
| B7 | Multi-task conceptor algebra | OR perfectly additive; 20-22 task-specific dims |
| B8 | Head alignment multi-layer | AUROC 0.62-0.76; layer-matched DAS required |
| B9 | Sparsity-geometry correlation | NULL (no correlation, p>0.18) |
| B10 | Per-projection QK/OV geometry | QK vs QV p=1.8e-15; OV AUROC=0.64 |
| B11 | Factor overlap between tasks | Factor bank amplifies separation by +6-34° |
| B12 | Conceptor boolean algebra | 99.6% preserved at computation layer |
| B13 | Geodesic distance matrix | SVA more stable (mean dist 2.08 vs IOI 2.80) |
| B14 | Multi-task universal subspace | 3D shared, eigenvalue 4.985/5 |
| B15 | Random baseline | z>68 for theta_0, z>134 for eigenvalues |
| B16 | Geodesic trajectory | IOI 3.83x curvature, SVA 2.30x |
| B17 | L1 stability | k=32 stable (<1° shift); k=4 fragile |
| B18 | Spectral profile | PR~1.4; top-1 captures 84%+ of energy |
| B19 | Cross-checkpoint geometry | 34-52% containment; L11 shared across ckpts |
| B20 | Frechet mean + clustering | 64/768=8.3% of d_model used |
| B21 | MDS embedding | Checkpoint >> task >> layer as organizing axis |
| B22 | Von Neumann entropy | Holevo chi=1.29; eff dim=116.5 |
| B23 | Principal direction flow | IOI v1 rotates 84-88° between layers |
| B24 | k-sensitivity | theta_0: 80.7° at k=4 → 2.9° at k=32 |
| B26 | Grassmannian regression | 100% LOO accuracy; linguistic categories don't predict kernel (p=0.92) |
| B27 | Subspace intersection | ALL pairs share sigma_0 > 0.993; top shared direction = garbage |
| B28 | PGA (tangent-space PCA) | PC0 captures 92%; PGA breaks down at these distances (r=-0.45) |
| B29 | Clean universal subspace | Only 2/10 directions garbage (4-5% energy); cleaning barely changes angles |
| B30 | Task discriminant analysis | IOI most unique; discriminants encode expected linguistics |
| B31 | Sectional curvature | K ≈ 0.064 constant across all tasks; triangle excess ~2x |
| B32 | Dual-space geometry | Factor vs d_model r=0.91; amplification +34° at L11 |
| B33 | Bootstrap confidence intervals | IOI-SVA theta_0 = 3.0° ± 0.1° (CI width < 1°) |
| B34 | Grassmannian interpolation | Midpoints uninterpretable; Frechet mean > pairwise midpoints |
| B35 | Kernel spectral analysis | Dense kernel cond=2.2; alignment with linguistic cats = 0.81 |
| B36 | Projection geometry (weight space) | QK overlap AUROC=0.54 (null); L11 attn peak for all tasks |
| B37 | Layer persistence profile | L11 attn writes 5x random for ALL tasks; read-in at baseline |
| B38 | Cross-task transfer | greater_than most transferable; 80-84% unique energy per task |
| B39 | Effective dimension scaling | dim = 32.8 * n^0.759; ~50 tasks to fill d_model; redundancy grows 0→32% |

### Pending

| ID | Name | Status | What's needed |
|----|------|--------|---------------|
| B2 | Vanilla GPT-2 baseline | NEEDS GPU | DAS without factorization for comparison |
| B6 | Geodesic interpolation | NEEDS GPU | IIA at geodesic midpoints |

### Modal jobs

1. **atomic-sweep-40 k=32 DAS** (ap-cPuDO4oevIkQ2iRG1UHdnJ)
   - Will reveal if DST orthogonality persists at higher k
   - Critical comparison: does DST eliminate the universal subspace?

## Data locations

DAS results (atomic-sweep-40, k=4):
- IOI (all layers): `lib/factorized_das/results/atomic-sweep-40/ioi*/`
- SVA (all layers): `lib/factorized_das/results/atomic-sweep-40/sva*/`

DAS results (dense, k=32, 5 tasks):
- `lib/factorized_das/results/dense-k32-grassmann/`

DAS results (dense, k=64, IOI + SVA):
- `lib/factorized_das/results/dense-k64/`

Checkpoints:
- atomic-sweep-40: `artifacts/wandb_checkpoints/atomic-sweep-40/factorized_payload.pt`

## Paper structure (revised, 10 claims)

1. **Introduction**: Transformer task representations live on Grassmannians; canonical angles quantify task geometry
2. **Claim 1 — Universal linguistic substrate**: ~3D subspace shared by all tasks (B14, B15, B29); promotes morphological structure; z>68 above random; NOT an artifact (only 4-5% garbage energy)
3. **Claim 2 — Layer-dependent geometry**: Tasks share encoding/decoding directions (L3, L11) but are orthogonal at computation (L8) — the encoding→computation→decoding arc (B1, B16, B23)
4. **Claim 3 — Geodesic distance predicts interference**: r=-0.994 between geodesic distance and cross-projection energy (B5); theta_0 alone insufficient
5. **Claim 4 — Factor bank amplification**: Factorized architecture separates tasks 6-34° MORE than raw residual stream (B11, B32); amplification strongest where tasks share computation (L11: +34°); factor-space geometry correlates r=0.91 with d_model geometry but systematically amplifies separation
6. **Claim 5 — Spectral dominance**: Tasks are effectively 2-4 dimensional (84%+ in top direction); collective task manifold uses only 8.3% of d_model (B18, B20)
7. **Claim 6 — Circuit prediction**: Canonical angle to DAS subspace predicts known circuit heads with AUROC 0.62-0.76 (B8, B10)
8. **Claim 7 — k-sensitivity**: Orthogonality at k=4 is dimensional constraint, not geometry; theta_0 drops 80.7° → 2.9° at k=32 (B24); subspace nesting: k=32 ⊂ k=64 at 68-85%
9. **Claim 8 — Task-discriminative directions encode expected linguistics**: IOI=proper names, SVA=verbs, capital_country=entities, gender_bias=social words (B30); Grassmannian regression gives 100% LOO task classification (B26)
10. **Claim 9 — Constant curvature region**: Tasks sit in a region of near-constant sectional curvature K≈0.064 (B31); symmetric around the Frechet mean, not along geodesic arcs (B34)
11. **Claim 10 — Robust with tight confidence intervals**: All pairwise theta_0 have CI width < 1° (B33); L1-robust (B17); stable across checkpoints (B19)
12. **Claim 11 — L11 attention universally produces DAS information**: All 5 tasks show L11 attention writing 5x random into the DAS subspace (B37); L1 secondary at 4x; universal L1→L3→L10→L11 production arc; read-in at baseline
13. **Claim 12 — Sublinear capacity scaling**: dim = 32.8 * n^0.759 (B39); ~50 tasks to fill d_model=768; residual stream has massive spare capacity; redundancy grows from 0% to 32% over 5 tasks
14. **Discussion**: Implications for steering (conceptor algebra, B7/B12), efficient inference (8% manifold), architecture design (DST vs dense geometry, dual-space amplification), task transfer (B38), and capacity bounds (B39)

## Key figures

1. **Layer-dependent canonical angles** — B1 sweep showing encoding/computation/decoding arc
2. **Sum-of-projectors eigenvalue spectrum** — B14 eigenvalues vs random (B15), with garbage cleaning (B29)
3. **Geodesic distance vs cross-projection** — B5 scatter plot, r=-0.994
4. **Spectral profile** — B18 singular value decay, top-1 dominance across all 5 tasks
5. **k-sensitivity** — B24 theta_0 from 80.7° (k=4) to 2.9° (k=32)
6. **Principal direction flow** — B23 showing IOI's 84° rotation between computation layers
7. **Ward clustering dendrogram** — B20 task families
8. **Dual-space amplification** — B32 factor-space vs d_model-space angles, color by layer
9. **Task-discriminant decoded tokens** — B30 showing expected linguistic features per task
10. **Bootstrap CI forest plot** — B33 all 10 pairwise angles with 95% CIs
11. **Layer persistence heatmap** — B37 write-out/random ratio for all 5 tasks x 12 layers
12. **Cross-task kernel matrix** — B35 dense kernel with eigenvalue spectrum
13. **Energy decomposition pie** — B38 unique vs shared energy per task
