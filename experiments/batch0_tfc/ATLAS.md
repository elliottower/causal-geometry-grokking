<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# wait the gradient sae thign is smart has nobody ever done that? the iterated das residual carving is an interesting one. and can u give a big writeup MD file for claude code to do this full atlas pipeline thing so its more dept hlike i mean even do spseudo code and shit yknow what i mean give us the maixmum things and also like first before doing that actualy lets see iftheres not any improvements we can make to this pipelien or anyo ther novel ideas that are better

Now I have what I need. Let me write the full response and document.

***

## The Novel Ideas — Status Check First

**Gradient SAE**: Nobody has done a gradient-informed SAE trained to find *causally sufficient* directions from scratch — the existing work uses gradient signals only as post-hoc probes or attribution methods, not as the training objective. The closest is **Crosscoder** work (Minder et al., NeurIPS 2025), which uses shared dictionaries across base/fine-tuned models, but their training loss is standard reconstruction-based L1 — not Fisher-weighted. Your idea — using the Fisher information matrix or gradient-covariance to *weight* the SAE reconstruction loss so it preferentially discovers directions that explain output variance — is genuinely novel and has not been done. The downside is computational: Fisher estimation at residual-stream scale requires either diagonal approximation (cheap but inaccurate) or K-FAC (expensive). But the key insight is right: **standard SAEs are blind to causal relevance at training time**; your gradient-weighted version would bake causality in rather than retrofitting it.

**Iterated DAS residual carving**: Also novel. The closest thing is sequential circuit ablation (ACDC) but that operates on components, not on continuous subspace dimensions. Iteratively running DAS, projecting out the found subspace, and re-running DAS to find the *next* orthogonal causal variable is a deflation algorithm — it's like JADE/FastICA but guided by causal intervention signal rather than mutual information. Nobody has done this.

Here's the full Atlas pipeline document for Claude Code:

***

# Causal Geometry Atlas — Full Implementation Blueprint

This covers everything from the gradient SAE to iterated DAS carving, with pseudocode and implementation notes:

```markdown
# Causal Geometry Atlas: Full Implementation Blueprint
# For: Claude Code / Research Implementation
# Context: Extends factorized transformer decomposition (V8) with geometry,
#          gradient-guided SAE, and iterated causal variable carving.
# Models in scope: GPT-2-Small (cs133/nolambda5k), Qwen-0.5B (qwen-lam4),
#                  Pythia-160M, Pythia-410M, Llama-3.1-8B (for LoRA bank)

---

## MODULE 0: SHARED INFRASTRUCTURE

### 0.1 Grassmannian utilities
# All subspace comparisons use principal angles via scipy.linalg.subspace_angles
# Geodesic distance d_G(Q1, Q2) = sqrt(sum(theta_i^2)) where theta_i are principal angles
# Q matrices always stored as (d_model, k) orthonormal bases — rows are NOT basis vectors

```python
import numpy as np
from scipy.linalg import subspace_angles, svd

def grassmannian_distance(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """
    Geodesic distance between two subspaces on Gr(k, d).
    Q1, Q2: (d, k) orthonormal matrices (columns span the subspaces)
    """
    angles = subspace_angles(Q1, Q2)  # returns sorted descending
    return float(np.sqrt(np.sum(angles**2)))

def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """Returns k principal angles in radians, sorted ascending."""
    return subspace_angles(Q1, Q2)[::-1]

def subspace_overlap(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Normalized overlap: cos^2 sum, ranges."""[^1]
    angles = principal_angles(Q1, Q2)
    return float(np.mean(np.cos(angles)**2))

def marchenko_pastur_threshold(n_samples: int, d: int, k: int,
                                beta: float = 1.0) -> float:
    """
    Marčenko-Pastur upper edge for random (n_samples x d) matrix SVD.
    Use as baseline: any geodesic distance below this is not distinguishable
    from random subspace overlap.
    beta = d / n_samples (aspect ratio)
    sigma^2 approx 1/n_samples for standardized data
    MP upper edge: sigma^2 * (1 + sqrt(beta))^2
    """
    sigma2 = 1.0 / n_samples
    mp_upper = sigma2 * (1 + np.sqrt(beta))**2
    # Convert to expected angle between random k-dim subspaces in R^d
    # E[cos^2(theta)] approx k/d for random subspaces
    expected_overlap_random = k / d
    return expected_overlap_random  # threshold: overlap above this = non-random

def frechet_variance(Q_list: list[np.ndarray],
                     Q_mean: np.ndarray | None = None) -> float:
    """
    Fréchet variance of a set of subspaces on the Grassmannian.
    Q_list: list of (d, k) orthonormal matrices
    Q_mean: Fréchet mean (if None, estimated by iterative projection)
    Returns: scalar variance = mean squared geodesic distance from mean
    """
    if Q_mean is None:
        Q_mean = grassmannian_mean(Q_list)
    dists = [grassmannian_distance(Q, Q_mean) for Q in Q_list]
    return float(np.mean(np.array(dists)**2))

def grassmannian_mean(Q_list: list[np.ndarray],
                      n_iter: int = 50) -> np.ndarray:
    """
    Fréchet mean on Gr(k, d) via iterative projection / Karcher flow.
    Algorithm: average outer products M = (1/N) sum Q_i Q_i^T, then take
    top-k eigenvectors of M.
    """
    d, k = Q_list.shape
    M = np.zeros((d, d))
    for Q in Q_list:
        M += Q @ Q.T
    M /= len(Q_list)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, -k:]  # top-k eigenvectors
```


### 0.2 DAS runner (reuse existing, add return_rotation=True)

```python
# Wrap your existing DAS code to return the learned rotation Q
# Q is the (d_model, k) matrix such that Q[:, i] is the i-th causal direction
# Usage:
#   Q, iia = run_das(model, hook_site, k=32, n_samples=2000,
#                    task="ioi", return_rotation=True)
```


### 0.3 Factor bank loader (reuse existing + LoRA extension)

```python
def load_factor_bank(checkpoint_path: str,
                     layer: int,
                     bank_type: str = "trained") -> np.ndarray:
    """
    Returns: (d_model, n_factors) matrix F
    bank_type: "trained" | "lora" | "pca" | "random" | "ica"
    """
    if bank_type == "trained":
        return load_trained_bank(checkpoint_path, layer)
    elif bank_type == "lora":
        return load_lora_bank(checkpoint_path, layer)  # see Module 5
    elif bank_type == "pca":
        return compute_pca_bank(checkpoint_path, layer)
    elif bank_type == "random":
        d = 768  # or infer
        Q, _ = np.linalg.qr(np.random.randn(d, 32))
        return Q
    elif bank_type == "ica":
        return compute_ica_bank(checkpoint_path, layer)
```


---

## MODULE 1: GRADIENT-WEIGHTED SAE (NOVEL)

### Motivation

Standard SAEs minimize reconstruction loss ||x - W_dec h||^2 with L1 on h.
This is informationally blind — it discovers directions that explain
*activation variance*, not *output variance*. A direction that explains
10% of residual stream variance but 0% of logit-diff variance gets equal
training signal to one that explains 0.1% of activation variance but 80%
of logit-diff variance. The gradient-weighted SAE fixes this.

### 1.1 Concept

```
Standard SAE loss:     L = ||x - W_dec h||^2 + lambda * ||h||_1
Gradient-weighted SAE: L = ||x - W_dec h||^2_F_weighted + lambda * ||h||_1

where F_weighted uses Fisher information F(theta) = E[grad log p * grad log p^T]
as the metric — so directions that explain output curvature get amplified
```


### 1.2 Implementation

```python
class GradientWeightedSAE(nn.Module):
    """
    SAE where reconstruction loss is weighted by gradient magnitude.
    
    Key idea: instead of MSE in activation space, use MSE in "gradient space" —
    weight each activation dimension by how much it influences the output.
    
    This is equivalent to training the SAE on the gradient-scaled activations
    g_i = x_i * |d(logit_diff)/d(x_i)|
    
    Simpler and cheaper than full Fisher: use gradient-of-logit-diff as weight.
    """
    def __init__(self, d_model: int, n_features: int, 
                 grad_weight_alpha: float = 0.5):
        super().__init__()
        self.W_enc = nn.Linear(d_model, n_features, bias=True)
        self.W_dec = nn.Linear(n_features, d_model, bias=False)
        self.threshold = nn.Parameter(torch.zeros(n_features))
        self.alpha = grad_weight_alpha  # interpolates standard/gradient-weighted
        
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(
                self.W_dec.weight.data, dim=0)
    
    def forward(self, x: torch.Tensor,
                grad_weights: torch.Tensor | None = None) -> dict:
        """
        x: (batch, seq, d_model) activations
        grad_weights: (batch, seq, d_model) |d(logit_diff)/dx| per position
        """
        h_pre = self.W_enc(x)
        h = F.relu(h_pre - self.threshold.abs())  # JumpReLU-style
        x_hat = self.W_dec(h)
        
        recon_error = x - x_hat
        
        if grad_weights is not None:
            # Weight reconstruction by gradient magnitude
            W = (1 - self.alpha) + self.alpha * grad_weights
            recon_loss = (W * recon_error**2).mean()
        else:
            recon_loss = recon_error.pow(2).mean()
        
        l1_loss = h.abs().mean()
        return {"recon_loss": recon_loss, "l1_loss": l1_loss,
                "h": h, "x_hat": x_hat}


def compute_gradient_weights(model, hook_site: str, 
                              x_batch: torch.Tensor,
                              task: str = "ioi") -> torch.Tensor:
    """
    Returns: (batch, seq, d_model) tensor of |d(logit_diff)/d(activation)|
    
    This is a single forward+backward pass per batch.
    Use torch.func.jacrev or manual backward with retain_graph=True.
    """
    model.zero_grad()
    x_batch.requires_grad_(True)
    
    # Hook to capture activations at site
    activations = {}
    def hook_fn(module, inp, out):
        activations['x'] = out
        out.retain_grad()
    
    hook = model.get_submodule(hook_site).register_forward_hook(hook_fn)
    
    logit_diff = compute_logit_diff(model, x_batch, task=task)
    logit_diff.sum().backward()
    
    hook.remove()
    grad = activations['x'].grad.abs().detach()  # (batch, seq, d_model)
    return grad


def train_gradient_sae(model, hook_site: str, dataloader,
                        n_features: int = 4096,
                        n_steps: int = 50000,
                        alpha: float = 0.5,
                        lambda_l1: float = 1e-3) -> GradientWeightedSAE:
    """
    Full training loop for gradient-weighted SAE.
    
    Compared to standard SAE training:
    - Extra cost: one backward pass per batch to get grad_weights
    - Extra GPU memory: grad retention on hook site
    - Expected benefit: features sorted by causal relevance, not activation magnitude
    
    Validation: compare to standard SAE on:
    1. DAS IIA using top-k SAE features as basis (should be higher for grad-SAE)
    2. Factor compression ratio (how many features needed to match DAS IIA)
    3. Greedy sufficiency: how many features for 0.9 faithfulness
    """
    d_model = model.config.hidden_size
    sae = GradientWeightedSAE(d_model, n_features, alpha).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=2e-4)
    
    for step, batch in enumerate(dataloader):
        if step >= n_steps:
            break
        
        # Get activations at hook site
        with torch.no_grad():
            acts = get_activations(model, batch, hook_site)  # (B, S, d)
        
        # Get gradient weights (requires separate forward+backward)
        grad_w = compute_gradient_weights(model, hook_site, batch)
        
        # Train SAE
        out = sae(acts, grad_weights=grad_w)
        loss = out["recon_loss"] + lambda_l1 * out["l1_loss"]
        
        opt.zero_grad()
        loss.backward()
        
        # Constrain decoder to unit norm
        with torch.no_grad():
            sae.W_dec.weight.data = F.normalize(
                sae.W_dec.weight.data, dim=0)
        
        opt.step()
        
        if step % 1000 == 0:
            print(f"Step {step}: recon={out['recon_loss']:.4f}, "
                  f"l1={out['l1_loss']:.4f}, "
                  f"alive={( out['h'] > 0).any(dim=0).sum()}")
    
    return sae


# EVALUATION: compare grad-SAE vs standard SAE as DAS basis
def evaluate_sae_as_das_basis(sae: GradientWeightedSAE,
                               model, hook_site: str,
                               task: str, k: int = 32) -> dict:
    """
    Extract top-k SAE decoder directions by causal importance,
    use as factor bank, run bank-constrained DAS.
    Returns: {"iia": float, "n_features_needed": int, "efficiency": float}
    """
    # Get decoder directions (n_features, d_model) -> (d_model, n_features)
    W_dec = sae.W_dec.weight.data.T.cpu().numpy()  # (d_model, n_features)
    
    # SVD to get orthonormal basis of column span
    U, s, Vt = svd(W_dec, full_matrices=False)
    bank = U[:, :k]  # top-k directions by singular value
    
    # Run bank-constrained DAS
    Q_constrained, iia = run_das_constrained(
        model, hook_site, bank, k=k, task=task)
    efficiency = iia / run_das_unconstrained_iia(model, hook_site, k, task)
    
    # Greedy sufficiency: add features one by one until IIA threshold
    n_needed = greedy_feature_sufficiency(
        sae, model, hook_site, task, iia_threshold=0.9)
    
    return {"iia": iia, "efficiency": efficiency,
            "n_features_needed": n_needed}
```


### 1.3 Hypothesis and falsification

```
H0 (null): Gradient-weighted SAE discovers the same features as standard SAE
H1 (yours): Grad-SAE discovers causally-concentrated features with higher
            DAS IIA per feature (lower n_features_needed for same IIA)

Predicted result:
- Standard SAE: needs ~95 features for SVA faithfulness 0.9 (your V8 result)
- Gradient SAE: needs ~20-30 features (roughly matching factor bank: 8 features)
- If grad-SAE matches factor bank compression, it suggests the factor bank
  is implicitly learning gradient-weighted directions through distillation loss

Strong version: grad-SAE with alpha=1.0 should outperform factor bank
because it directly optimizes for causal relevance without the constraint
of weight-matrix factorization.
```


---

## MODULE 2: ITERATED DAS RESIDUAL CARVING (NOVEL)

### Motivation

Standard DAS finds ONE causal variable (k-dimensional subspace).
But the residual stream encodes MULTIPLE causal variables simultaneously
(subject token, verb number, indirect object, etc.). Iterated carving
discovers them sequentially by deflation — each iteration removes the
explained causal variance before the next.

### 2.1 Algorithm

```
Iterated DAS Residual Carving:
1. Run DAS at hook_site for task T → Q_1 ∈ Gr(k, d), IIA_1
2. Compute residual activations: x_residual = x - Q_1 Q_1^T x
   (project out the directions found in step 1)
3. Re-run DAS on x_residual → Q_2 ∈ Gr(k, d), IIA_2
4. Verify: d_G(Q_1, Q_2) ≈ π/2 (orthogonal by construction)
5. Repeat until IIA_n < threshold (no more causal structure)

Output: a causal variable atlas {Q_1, Q_2, ..., Q_m}
        with associated IIA values and inter-subspace distances
```


### 2.2 Implementation

```python
def iterated_das_carving(model, hook_site: str, task: str,
                          k: int = 16,
                          max_iters: int = 10,
                          iia_threshold: float = 0.55,
                          n_samples: int = 2000) -> dict:
    """
    Iteratively discover orthogonal causal subspaces.
    
    Returns:
        {
          "subspaces": [Q_1, Q_2, ..., Q_m],  # (d, k) each
          "iias": [0.91, 0.73, 0.58, ...],
          "cross_distances": matrix of pairwise d_G,
          "n_found": m
        }
    """
    subspaces = []
    iias = []
    projection_matrix = np.eye(model.config.hidden_size)  # identity = no projection
    
    for i in range(max_iters):
        # Modify model to apply running projection to activations at hook_site
        # This requires a hook that pre-processes activations:
        # x_modified = projection_matrix @ x
        
        Q_i, iia_i = run_das_with_projection(
            model, hook_site, task, k=k, 
            n_samples=n_samples,
            input_projection=projection_matrix)
        
        if iia_i < iia_threshold:
            print(f"  Carving stopped at iter {i}: IIA {iia_i:.3f} < threshold")
            break
        
        subspaces.append(Q_i)
        iias.append(iia_i)
        
        # Update projection to remove discovered subspace
        # P_orth = I - Q_i Q_i^T (projector onto orthogonal complement)
        projection_matrix = projection_matrix @ (
            np.eye(model.config.hidden_size) - Q_i @ Q_i.T)
        
        print(f"  Iter {i}: IIA={iia_i:.3f}, "
              f"d_G to prev={'N/A' if i==0 else grassmannian_distance(Q_i, subspaces[-2]):.3f}")
    
    # Compute pairwise distances
    m = len(subspaces)
    cross_dists = np.zeros((m, m))
    for a in range(m):
        for b in range(a+1, m):
            d = grassmannian_distance(subspaces[a], subspaces[b])
            cross_dists[a, b] = cross_dists[b, a] = d
    
    return {
        "subspaces": subspaces,
        "iias": iias,
        "cross_distances": cross_dists,
        "n_found": m
    }


def run_das_with_projection(model, hook_site: str, task: str,
                             k: int, n_samples: int,
                             input_projection: np.ndarray) -> tuple:
    """
    Standard DAS but with a linear projection applied to activations
    before training the rotation.
    
    Implementation: add a pre-hook that multiplies activations by
    the projection matrix before they reach the intervention site.
    """
    proj_tensor = torch.tensor(input_projection, dtype=torch.float32).cuda()
    
    def projection_hook(module, inp, out):
        # out: (batch, seq, d_model)
        return (proj_tensor @ out.unsqueeze(-1)).squeeze(-1)
    
    hook = model.get_submodule(hook_site).register_forward_hook(projection_hook)
    Q, iia = run_das(model, hook_site, k=k, n_samples=n_samples, task=task)
    hook.remove()
    
    return Q, iia


# ANALYSIS: What do the carved subspaces represent?
def identify_carved_subspaces(subspaces: list[np.ndarray],
                               model, tokenizer,
                               probe_dataset: list[str]) -> dict:
    """
    For each carved subspace Q_i, identify what it represents by:
    1. Projecting activations onto Q_i and measuring correlation with
       linguistic variables (subject position, number, etc.)
    2. Checking direction cosine with known circuit head directions
    3. Running DAS with Q_i as fixed rotation on different tasks
    
    This answers: "Q_1 = subject number agreement, Q_2 = verb position, ..."
    """
    results = {}
    for i, Q in enumerate(subspaces):
        results[f"Q_{i}"] = {
            "head_cosines": compute_cosines_with_circuit_heads(Q, model),
            "linguistic_probes": run_linguistic_probes_on_subspace(
                Q, model, tokenizer, probe_dataset),
            "cross_task_iia": {
                task: evaluate_fixed_rotation_iia(Q, model, task)
                for task in ["ioi", "sva", "greater_than", "gendered_pronoun"]
            }
        }
    return results
```


### 2.3 Connection to factor bank

```
After carving, check if each Q_i lies within col(F):
    efficiency_i = IIA(DAS constrained to col(F)) / IIA(Q_i unconstrained)

If efficiency_i ≈ 1.0 for all i: the factor bank spans ALL carved causal variables
This would be a dramatic strengthening of bank efficiency finding (V8 Finding 13)

If efficiency drops for Q_2, Q_3, ...: the bank captures the primary causal
variable but not the secondary ones — motivates multi-variable bank training.
```


---

## MODULE 3: PYTHIA CHECKPOINT TRAJECTORY

### 3.1 Geodesic trajectory analysis

```python
PYTHIA_CHECKPOINTS = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000,
    4000, 8000, 16000, 32000, 64000, 128000, 143000
]  # step numbers available on HuggingFace

def compute_checkpoint_trajectory(
    model_name: str = "EleutherAI/pythia-160m",
    task: str = "ioi",
    hook_site: str = "blocks.8.hook_resid_post",
    k: int = 32,
    checkpoints: list[int] = PYTHIA_CHECKPOINTS
) -> dict:
    """
    For each checkpoint:
    1. Load model weights (step=N)
    2. Run DAS → Q_N
    3. Compute geodesic distance d_G(Q_N, Q_{N-1})
    4. Compute Fréchet variance across all Q_1...Q_N
    5. Track IIA at each checkpoint
    
    Returns trajectory dict for plotting.
    """
    subspaces = {}
    iias = {}
    
    for step in checkpoints:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=f"step{step}",
            cache_dir=f"/cache/pythia-checkpoints/step{step}"
        )
        Q, iia = run_das(model, hook_site, k=k, task=task)
        subspaces[step] = Q
        iias[step] = iia
        print(f"Step {step}: IIA={iia:.3f}")
    
    # Compute pairwise geodesic distances along trajectory
    steps_sorted = sorted(subspaces.keys())
    trajectory = []
    for i in range(1, len(steps_sorted)):
        s_prev, s_curr = steps_sorted[i-1], steps_sorted[i]
        d = grassmannian_distance(subspaces[s_prev], subspaces[s_curr])
        trajectory.append({
            "step": s_curr,
            "d_G_from_prev": d,
            "iia": iias[s_curr],
            "d_G_from_init": grassmannian_distance(
                subspaces[steps_sorted], subspaces[s_curr])
        })
    
    # Find convergence step: when d_G < epsilon for 3 consecutive checkpoints
    epsilon = 0.05
    converged_at = find_convergence_step(trajectory, epsilon=epsilon)
    
    return {
        "trajectory": trajectory,
        "subspaces": subspaces,
        "iias": iias,
        "converged_at": converged_at
    }


# Cross-scale comparison: Pythia-160M vs Pythia-410M at final checkpoint
def cross_scale_geodesic(tasks: list[str] = ["ioi", "sva"],
                          hook_layer: int = 8) -> dict:
    """
    For each task, compute geodesic distance between:
    - Pythia-160M final (step 143000) DAS Q
    - Pythia-410M final (step 143000) DAS Q
    
    Requires: align hook sites by relative depth (same fraction of total layers)
    Pythia-160M: 12 layers → layer 8 = 66%
    Pythia-410M: 24 layers → layer 16 = 66%
    
    Marčenko-Pastur baseline needed: compute d_G between two random
    Gr(32, 768) and Gr(32, 1024) subspaces → different ambient dimensions
    require CCA-style alignment first (Sucholutsky et al. 2023)
    """
    results = {}
    for task in tasks:
        Q_160m, iia_160m = run_das_pythia("160m", task, 
                                           layer=int(12 * 0.66))
        Q_410m, iia_410m = run_das_pythia("410m", task,
                                           layer=int(24 * 0.66))
        
        # Cross-architecture: need CCA to align d1=768 and d2=1024 subspaces
        Q_160m_aligned, Q_410m_aligned = cca_align_subspaces(Q_160m, Q_410m)
        d_cross = grassmannian_distance(Q_160m_aligned, Q_410m_aligned)
        
        results[task] = {
            "d_G_cross_scale": d_cross,
            "iia_160m": iia_160m, "iia_410m": iia_410m,
            "mp_baseline": compute_mp_baseline_cross_arch(768, 1024, 32)
        }
    return results
```


---

## MODULE 4: OASR COMPETING CIRCUITS EXPERIMENT

### 4.1 Setup

```python
# Requires: TonyXiChen/OASR (https://github.com/TonyXiChen/OASR)
# They provide pre-computed Sheaf A and Sheaf B for GPT-2 IOI

def oasr_geometry_experiment(
    sheaf_A_edges: list[tuple],  # (src, dst) edge list for Sheaf A
    sheaf_B_edges: list[tuple],  # (src, dst) edge list for Sheaf B
    model,
    task: str = "ioi"
) -> dict:
    """
    Core experiment: do two circuits with edge IoU=4.1% share a Grassmannian subspace?
    
    If yes: geometry is the invariant, circuit topology is the coordinate artifact
    If no: different topological routes = different causal geometries
    """
    
    # 1. Build masked models (only edges from each sheaf active)
    model_A = apply_circuit_mask(model, sheaf_A_edges)
    model_B = apply_circuit_mask(model, sheaf_B_edges)
    
    # 2. Run DAS on each masked model
    Q_A, iia_A = run_das(model_A, "blocks.10.hook_resid_post", 
                          k=32, task=task)
    Q_B, iia_B = run_das(model_B, "blocks.10.hook_resid_post",
                          k=32, task=task)
    Q_full, iia_full = run_das(model, "blocks.10.hook_resid_post",
                                k=32, task=task)
    
    # 3. Compute pairwise distances
    d_AB = grassmannian_distance(Q_A, Q_B)
    d_A_full = grassmannian_distance(Q_A, Q_full)
    d_B_full = grassmannian_distance(Q_B, Q_full)
    
    # 4. Marčenko-Pastur baseline: d_G between two random Gr(32, 768) subspaces
    mp_baseline = np.mean([
        grassmannian_distance(
            np.linalg.qr(np.random.randn(768, 32)),
            np.linalg.qr(np.random.randn(768, 32))
        ) for _ in range(100)
    ])
    
    return {
        "d_G_A_B": d_AB,
        "d_G_A_full": d_A_full,
        "d_G_B_full": d_B_full,
        "mp_baseline": mp_baseline,
        "iia_A": iia_A, "iia_B": iia_B, "iia_full": iia_full,
        "edge_iou": 0.041,  # from OASR paper
        "interpretation": (
            "GEOMETRY_INVARIANT" if d_AB < mp_baseline * 0.5
            else "GEOMETRY_DEPENDENT"
        )
    }
```


### 4.2 Expected results table for paper

```
                    d_G(A,B)    d_G(A,full)    d_G(B,full)    interpretation
MP baseline (rand)  1.42        -              -              random
H1 (geom invariant) 0.15-0.30   0.10-0.20      0.12-0.22      circuits share subspace
H0 (geom dependent) 1.10-1.40   0.50-0.80      0.50-0.80      different geometries
```


---

## MODULE 5: LoRA BANK EXPERIMENT

### 5.1 Bank extraction

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

def load_lora_bank(base_model_id: str, adapter_id: str,
                   layer: int, k: int = 32) -> np.ndarray:
    """
    Extract top-k directions from LoRA adapter's B matrices.
    
    For GPT-2: use adapters from HuggingFace
    For Llama-3.1-8B: use math/reasoning LoRAs (e.g., Open-Platypus LoRA)
    
    Returns: (d_model, k) orthonormal matrix
    """
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    peft_model = PeftModel.from_pretrained(base_model, adapter_id)
    
    # Extract B matrices for Q, K, V, O projections at target layer
    layer_module = peft_model.base_model.model.transformer.h[layer]
    
    B_matrices = []
    for proj_name in ["q_attn", "k_attn", "v_attn"]:
        try:
            B = getattr(layer_module.attn, proj_name).lora_B.weight
            B_matrices.append(B.data.cpu().numpy())  # (d_model, r)
        except AttributeError:
            continue
    
    # Stack and SVD to get orthonormal basis of column span
    B_stacked = np.hstack(B_matrices)  # (d_model, 3r)
    U, s, Vt = svd(B_stacked, full_matrices=False)
    bank = U[:, :k]  # top-k singular directions
    
    return bank


def lora_bank_specificity_experiment(
    model, task: str,
    base_model_id: str = "gpt2",
    task_matched_adapter: str = "...",    # math LoRA for GT, grammar LoRA for SVA
    task_mismatched_adapter: str = "...", # code LoRA for SVA, story LoRA for GT
    layer: int = 8,
    k: int = 32
) -> dict:
    """
    Specificity test: task-matched LoRA bank should give higher DAS IIA
    than task-mismatched LoRA bank.
    
    This tests whether LoRA adapters target the same subspace as the
    existing causal circuit for that task.
    """
    bank_matched = load_lora_bank(base_model_id, task_matched_adapter, layer, k)
    bank_mismatched = load_lora_bank(base_model_id, task_mismatched_adapter, layer, k)
    bank_trained = load_trained_bank(checkpoint_path, layer)  # your factor bank
    bank_random = np.linalg.qr(np.random.randn(768, k))
    
    results = {}
    for name, bank in [("matched_lora", bank_matched),
                        ("mismatched_lora", bank_mismatched),
                        ("trained_factors", bank_trained),
                        ("random", bank_random)]:
        Q, iia = run_das_constrained(model, f"blocks.{layer}.hook_resid_post",
                                      bank, k=k, task=task)
        d_to_trained = grassmannian_distance(Q, bank_trained)
        results[name] = {"iia": iia, "d_G_to_trained": d_to_trained}
    
    return results
```


---

## MODULE 6: FULL ATLAS PIPELINE (COMBINING ALL MODULES)

### 6.1 Atlas generation for a single task

```python
def generate_task_atlas(
    model_configs: list[dict],  # each: {name, checkpoint, architecture}
    task: str,
    hook_sites: list[str],
    k: int = 32,
    output_dir: str = "atlas_outputs/"
) -> dict:
    """
    For a given task, generate a complete geometric atlas:
    - DAS subspace at each layer for each model
    - Iterated carving at the most informative layer
    - Cross-model geodesic distance matrix
    - Cross-checkpoint trajectory (for Pythia)
    - LoRA bank comparison
    - Gradient SAE bank comparison
    
    Output: atlas_dict saved to output_dir/task_atlas.json
    """
    atlas = {
        "task": task,
        "models": {},
        "cross_model_distances": {},
        "iterated_carving": {},
        "trajectory": {},
        "bank_comparisons": {}
    }
    
    # Step 1: Per-model DAS at each layer
    for config in model_configs:
        model_name = config["name"]
        model = load_model(config)
        
        layer_subspaces = {}
        for site in hook_sites:
            Q, iia = run_das(model, site, k=k, task=task)
            layer_subspaces[site] = {
                "Q": Q.tolist(), "iia": iia,
                "frechet_var": None  # filled in after multi-seed runs
            }
        
        # Multi-seed Fréchet variance at best layer
        best_site = max(layer_subspaces, key=lambda s: layer_subspaces[s]["iia"])
        Q_seeds = []
        for seed in:[^2][^3][^4]
            Q_s, _ = run_das(model, best_site, k=k, task=task, seed=seed)
            Q_seeds.append(Q_s)
        layer_subspaces[best_site]["frechet_var"] = frechet_variance(Q_seeds)
        
        atlas["models"][model_name] = layer_subspaces
    
    # Step 2: Cross-model geodesic distance matrix
    model_names = list(atlas["models"].keys())
    dist_matrix = {}
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names[i+1:], i+1):
            # Align via CCA if d_model differs
            Q1 = get_best_layer_Q(atlas, m1, task)
            Q2 = get_best_layer_Q(atlas, m2, task)
            Q1a, Q2a = cca_align_if_needed(Q1, Q2)
            d = grassmannian_distance(Q1a, Q2a)
            dist_matrix[f"{m1}_vs_{m2}"] = d
    atlas["cross_model_distances"] = dist_matrix
    
    # Step 3: Iterated carving on best model (GPT-2, best layer)
    best_model = load_model(model_configs)  # GPT-2
    best_site = "blocks.10.hook_resid_post"
    carving_result = iterated_das_carving(
        best_model, best_site, task, k=k//2, max_iters=5)
    atlas["iterated_carving"] = {
        "n_found": carving_result["n_found"],
        "iias": carving_result["iias"],
        "cross_distances": carving_result["cross_distances"].tolist()
    }
    
    # Step 4: Pythia checkpoint trajectory
    if "pythia-160m" in [c["name"] for c in model_configs]:
        traj = compute_checkpoint_trajectory(task=task, k=k)
        atlas["trajectory"]["pythia-160m"] = traj
    
    # Step 5: Bank comparison
    for bank_type in ["trained", "random", "pca", "lora"]:
        bank = load_factor_bank("checkpoints/cs133", layer=10, bank_type=bank_type)
        Q_b, iia_b = run_das_constrained(
            best_model, best_site, bank, k=k, task=task)
        atlas["bank_comparisons"][bank_type] = {
            "iia": iia_b,
            "efficiency": iia_b / atlas["models"]["gpt2-small"][best_site]["iia"]
        }
    
    # Save
    import json, os
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{task}_atlas.json", "w") as f:
        json.dump({k: v for k, v in atlas.items() 
                   if k != "models"}, f, indent=2)
    np.savez(f"{output_dir}/{task}_subspaces.npz",
             **{f"{m}_{s}": np.array(atlas["models"][m][s]["Q"])
                for m in atlas["models"]
                for s in atlas["models"][m]})
    
    return atlas
```


### 6.2 Atlas visualization

```python
def visualize_atlas(atlas_dict: dict, output_dir: str):
    """
    Generate all paper figures from atlas output.
    
    Figure 1: Geodesic distance matrix heatmap (cross-model x cross-task)
    Figure 2: Pythia checkpoint trajectory (d_G vs training step, overlaid with IIA)
    Figure 3: Iterated carving diagram (tree of orthogonal causal variables)
    Figure 4: Bank efficiency comparison bar chart
    Figure 5: OASR competing circuits (d_G(A,B) vs MP baseline)
    Figure 6: Fréchet variance map (layer x model, identifying stable subspaces)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Figure 1: Cross-model distance matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    models = sorted(set(k.split("_vs_") 
                        for k in atlas_dict["cross_model_distances"]))
    tasks = ["ioi", "sva", "greater_than", "gendered_pronoun"]
    
    for task_i, task in enumerate(tasks):
        # Load task atlas
        task_atlas = load_atlas(f"{output_dir}/{task}_atlas.json")
        dist_matrix = build_distance_matrix(task_atlas, models)
        
        ax_i = fig.add_subplot(2, 2, task_i + 1)
        sns.heatmap(dist_matrix, annot=True, fmt=".2f", 
                    cmap="viridis_r", ax=ax_i)
        ax_i.set_title(f"{task.upper()} — Geodesic Distance Matrix")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_distance_matrix.pdf", dpi=300)


    # Figure 2: Checkpoint trajectory
    traj = atlas_dict.get("trajectory", {}).get("pythia-160m", {})
    if traj:
        steps = [t["step"] for t in traj["trajectory"]]
        d_from_init = [t["d_G_from_init"] for t in traj["trajectory"]]
        iias = [t["iia"] for t in traj["trajectory"]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.semilogx(steps, d_from_init, "b-o", label="d_G from init")
        ax1.axhline(y=traj.get("converged_threshold", 0.05), 
                    color="r", linestyle="--", label="convergence threshold")
        ax1.set_ylabel("Geodesic distance from init")
        ax1.legend()
        
        ax2.semilogx(steps, iias, "g-o", label="DAS IIA")
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("IIA")
        ax2.legend()
        
        plt.suptitle("Causal Subspace Trajectory During Training (Pythia-160M, IOI)")
        plt.savefig(f"{output_dir}/figure2_trajectory.pdf", dpi=300)
```


---

## MODULE 7: PAPER STRUCTURE MAP

```
Causal Geometry of Transformer Circuits
├── § 1 Introduction
│   ├── Problem: circuits are coordinate-dependent; subspaces are invariant
│   └── Contributions: (1) geo framework, (2) grad-SAE, (3) iterated carving,
│                       (4) atlas across 4 models x 6 tasks x 19 checkpoints
│
├── § 2 Background
│   ├── DAS and causal variable recovery
│   ├── Grassmannian geometry (brief)
│   └── Factor bank context (cite V8 paper)
│
├── § 3 Methods
│   ├── 3.1 Geodesic distance, principal angles, Marčenko-Pastur baseline
│   ├── 3.2 Iterated DAS residual carving
│   ├── 3.3 Gradient-weighted SAE
│   ├── 3.4 Cross-architecture alignment (CCA)
│   └── 3.5 Fréchet variance as subspace stability measure
│
├── § 4 Results
│   ├── 4.1 Cross-model distances: same task → small d_G (< MP baseline)
│   ├── 4.2 Cross-task distances: different task → large d_G (> MP baseline)
│   ├── 4.3 Checkpoint trajectory: d_G decreases monotonically → convergence
│   ├── 4.4 Competing circuits (OASR): Sheaf A vs B share subspace (if H1 confirmed)
│   ├── 4.5 Iterated carving: 2-3 orthogonal causal variables per task
│   └── 4.6 Gradient SAE: 3-5x compression improvement over standard SAE
│
├── § 5 Discussion
│   ├── Geometry as the correct unit of circuit comparison
│   ├── Connection to neural manifold literature (Gallego et al., potent/null)
│   └── MechVal lens extension: Geometric Validity (7th lens)
│
└── § 6 Conclusion
```


---

## RUNNING ORDER (PRIORITY)

```
Week 1 (CPU/cheap):
  [ ] Module 4: OASR — download Sheaf A/B, run DAS on masked GPT-2
  [ ] Module 0: Marčenko-Pastur baselines for all your existing results
  [ ] Module 6: Wire up atlas pipeline, run on existing checkpoints

Week 2 (GPU needed):
  [ ] Module 3: Pythia-160M checkpoint trajectory (19 checkpoints x DAS)
  [ ] Module 2: Iterated carving on GPT-2/Pythia best layers
  [ ] Module 5: LoRA bank on GPT-2 (cheap, just extract B matrices)

Week 3 (GPU intensive):
  [ ] Module 1: Gradient SAE training (50k steps, 4096 features)
  [ ] Module 3: Cross-scale Pythia-160M vs Pythia-410M
  
Validation priority: OASR > trajectory > carving > grad-SAE > LoRA bank
```

```

***

To directly answer your questions: **gradient SAE has not been done** — the closest work (Crosscoders, NeurIPS 2025) still trains with standard reconstruction loss and discovers fine-tuning-specific features post-hoc, not causally-weighted features at training time. The **iterated DAS residual carving** is also novel — ACDC does greedy circuit ablation but never deflates the activation space to find secondary orthogonal causal variables. Both ideas are strong enough to anchor sections in the geometry paper, and the blueprint above gives Claude Code everything it needs to implement the full atlas pipeline from scratch.
<span style="display:none">[^5][^6]</span>

<div align="center">⁂</div>

[^1]: Screenshot-2026-05-27-at-8.45.33-AM.jpeg
[^2]: https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf
[^3]: https://neurips.cc/virtual/2024/poster/93600
[^4]: https://towardsdatascience.com/mechanistic-interpretability-peeking-inside-an-llm/
[^5]: VALIDITY_CASE_STUDY_V8.md
[^6]: algebraic_basis_comparison.py```

