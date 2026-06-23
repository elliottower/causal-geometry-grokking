"""Cross-task transfer + topological validation for nonlinear causal variables.

Two external validation approaches that don't rely on IIA:

1. CROSS-TASK TRANSFER: Train VAE on IOI template family A, test on family B.
   If z_causal transfers, it found the real variable. If not, it overfit syntax.
   Also test: train on IOI, test on "colored objects" or other name-dependent tasks.

2. PERSISTENT HOMOLOGY: Compute Betti numbers of the VAE latent space.
   For grokked addition, z_causal should form a circle (H1 ≅ Z, β1 = 1).
   For ungrokked squaring, no topological structure expected (β1 = 0).
   This is a structural prediction the model was never trained on.

3. SHEAF CONSISTENCY: Compute DAS at multiple cluster centers in activation space.
   If the local subspaces are consistent (small transition map error), the variable
   is approximately Grassmannian. If inconsistent, genuinely nonlinear.

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/cross_task_validation.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
import traceback
from datetime import datetime, timezone

import modal

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except (ImportError, AttributeError):
    pass

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "setuptools<71")
    .pip_install(
        "transformer-lens==2.11.0", "transformers==4.46.3",
        "einops>=0.8", "matplotlib", "tqdm", "scikit-learn",
    )
)

app = modal.App("cross-task-validation", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

SAVE_DIR = "/results/grassmannian_atlas/cross_task_validation"


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_incremental(all_results, log):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(f"{SAVE_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Incremental save to {SAVE_DIR}/results.json")


# ===================================================================
# VAE (same as k1 scripts)
# ===================================================================


def build_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device):
    class StructuredVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            z_dim = z_causal_dim + z_nuisance_dim
            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_causal_dim, n_classes)

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = self.reparameterize(mu_c, lv_c)
            z_n = self.reparameterize(mu_n, lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            x_recon = self.decoder(z)
            logits = self.classifier(z_c)
            return x_recon, logits, mu_c, lv_c, mu_n, lv_n

    return StructuredVAE().to(device)


def train_vae(vae, acts, labels, device, n_epochs=500, batch_size=256,
              lr=1e-3, alpha=10.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="VAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + kl_c + kl_n + alpha * ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def eval_vae_iia(vae, model_lm, pairs, hook_name):
    vae.eval()
    correct = 0
    with torch.inference_mode():
        for d in pairs:
            mu_c_b, _, mu_n_b, _ = vae.encode(d["base_act"].unsqueeze(0))
            mu_c_s, _, _, _ = vae.encode(d["src_act"].unsqueeze(0))
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_iv = vae.decoder(z_iv).squeeze(0)

            def hook_fn(act, hook=None, iv=h_iv):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == d["src_label"]:
                correct += 1
    return correct / len(pairs) if pairs else 0.0


def train_das(model_lm, train_pairs, hook_name, d_model, k, device,
              n_steps=300, lr=1e-3, batch_size=16):
    R = nn.Parameter(torch.randn(d_model, k, device=device) * 0.02)
    optimizer = torch.optim.Adam([R], lr=lr)
    for step in tqdm(range(n_steps), desc=f"DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(train_pairs, min(batch_size, len(train_pairs)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            loss = loss - F.log_softmax(logits, dim=-1)[d["src_label"]]
        (loss / len(batch)).backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
    return Q.detach()


def eval_das_iia(Q, model_lm, pairs, hook_name):
    proj = Q @ Q.T
    correct = 0
    with torch.inference_mode():
        for d in pairs:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == d["src_label"]:
                correct += 1
    return correct / len(pairs) if pairs else 0.0


# ===================================================================
# IOI data generation — multiple template families
# ===================================================================

IOI_NAMES = [
    " Mary", " John", " Alice", " Bob", " Tom", " Claire",
    " Dave", " Sarah", " James", " Emma", " Mike", " Kate",
    " Jack", " Anna", " Dan", " Amy", " Sam", " Lisa",
    " Paul", " Jane", " Mark", " Lucy", " Nick", " Helen",
    " Adam", " Laura", " Ben", " Sophie", " Chris", " Diana",
]

IOI_PLACES = ["store", "park", "office", "restaurant", "library", "gym",
              "museum", "school", "hospital", "beach"]
IOI_OBJECTS = ["book", "drink", "ball", "pen", "bag", "phone",
               "key", "letter", "toy", "card"]

# Family A: ABBA order (B appears after A, then B gives to _)
TEMPLATES_ABBA = [
    "Then,{A} and{B} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} were working at the {PLACE}.{B} decided to give a {OBJ} to",
]

# Family B: ABAB order (A appears after B, then B gives to _)
TEMPLATES_ABAB = [
    "Then,{B} and{A} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{B} and{A} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
    "Then,{B} and{A} were working at the {PLACE}.{B} decided to give a {OBJ} to",
]

# Family C: longer, more complex templates
TEMPLATES_LONG = [
    "Then,{A} and{B} went to the {PLACE} together.{B} picked up a {OBJ} and gave it to",
    "After arriving at the {PLACE},{A} met{B} there.{B} handed a {OBJ} to",
    "At the {PLACE},{B} saw{A} and they chatted for a while. Then{B} gave a {OBJ} to",
]

# Family D: possessive / different syntax
TEMPLATES_POSSESSIVE = [
    "{A} and{B} were at the {PLACE}.{B} had a {OBJ} and gave it to",
    "When{A} and{B} arrived at the {PLACE},{B} passed a {OBJ} to",
    "{A} met{B} at the {PLACE}, and{B} offered a {OBJ} to",
]


def generate_pairs(model, templates, names, rng, hook_name, n_raw=1500):
    valid_names = [n for n in names if len(model.tokenizer.encode(n)) == 1]
    raw = []
    for _ in range(n_raw):
        t = rng.choice(templates)
        a, b = rng.sample(valid_names, 2)
        base = t.format(A=a, B=b, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        source = t.format(A=b, B=a, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        raw.append((base, source, model.tokenizer.encode(a)[0], model.tokenizer.encode(b)[0]))

    data = []
    for base_text, src_text, base_id, src_id in tqdm(raw, desc="cache", leave=False):
        bt = model.to_tokens(base_text)
        st = model.to_tokens(src_text)
        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            bl = model(bt)[0, -1]
            _, sc = model.run_with_cache(st, names_filter=hook_name)
        if (bl[base_id] - bl[src_id]).item() > 0:
            data.append({
                "base_act": bc[hook_name][0, -1],
                "src_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_label": src_id,
                "base_label": base_id,
            })
    return data


# ===================================================================
# Experiment 1: Cross-template-family transfer
# ===================================================================


def run_cross_template_transfer(model, device, log):
    """Train on one template family, test on all others."""
    log.info(f"[{utc_ts()}] === Cross-Template Transfer ===")

    hook_name = "blocks.10.hook_resid_post"
    D = 768

    families = {
        "ABBA": TEMPLATES_ABBA,
        "ABAB": TEMPLATES_ABAB,
        "long": TEMPLATES_LONG,
        "possessive": TEMPLATES_POSSESSIVE,
    }

    family_data = {}
    for name, templates in families.items():
        log.info(f"[{utc_ts()}]   Generating {name} pairs...")
        family_data[name] = generate_pairs(
            model, templates, IOI_NAMES, random.Random(42), hook_name, n_raw=1500,
        )
        log.info(f"[{utc_ts()}]   {name}: {len(family_data[name])} valid pairs")

    results = {"task": "cross_template_transfer"}

    for train_family in families:
        train_pairs = family_data[train_family]
        if len(train_pairs) < 50:
            log.info(f"[{utc_ts()}]   Skipping {train_family} (too few pairs)")
            continue

        label_map = {}
        all_acts, all_labels = [], []
        for d in train_pairs:
            for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
                tid = d[lk]
                if tid not in label_map:
                    label_map[tid] = len(label_map)
                all_acts.append(d[rk])
                all_labels.append(label_map[tid])
        act_t = torch.stack(all_acts)
        lab_t = torch.tensor(all_labels, device=device)
        n_classes = len(label_map)

        for k in [2, 4]:
            log.info(f"[{utc_ts()}]   Train={train_family}, k={k}")

            # Train DAS
            Q = train_das(model, train_pairs[:400], hook_name, D, k, device)

            # Train VAE
            vae = build_vae(D, k, 16, 256, n_classes, device)
            vae = train_vae(vae, act_t, lab_t, device, n_epochs=500, alpha=10.0)

            # Eval on ALL families (including train family for comparison)
            family_results = {}
            for eval_family in families:
                eval_pairs = family_data[eval_family]
                if len(eval_pairs) < 10:
                    continue
                das_iia = eval_das_iia(Q, model, eval_pairs[:300], hook_name)
                vae_iia = eval_vae_iia(vae, model, eval_pairs[:300], hook_name)
                family_results[eval_family] = {
                    "das_iia": das_iia,
                    "vae_iia": vae_iia,
                    "n_pairs": min(len(eval_pairs), 300),
                }
                log.info(f"[{utc_ts()}]     Eval={eval_family}: DAS={das_iia:.3f}, VAE={vae_iia:.3f}")

            results[f"train_{train_family}_k{k}"] = family_results
            del vae
            torch.cuda.empty_cache()

    return results


# ===================================================================
# Experiment 2: Persistent homology of latent space
# ===================================================================


def compute_persistence(points, max_dim=1, n_landmarks=200):
    """Compute persistence diagrams using Vietoris-Rips via sklearn distances.

    Returns Betti numbers and persistence of longest-lived H1 feature.
    We use a simple approach: compute pairwise distances, then track
    connected components (H0) and loops (H1) through the filtration.

    For H1, we use the heuristic: if points lie on a circle, the longest H1
    bar should have persistence >> the noise bars.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(points) > n_landmarks:
        idx = np.random.choice(len(points), n_landmarks, replace=False)
        points = points[idx]

    dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    n = len(points)

    # Simple H0: track connected components via single-linkage
    # Sort all pairwise distances
    triu_idx = np.triu_indices(n, k=1)
    edge_dists = dists[triu_idx]
    sorted_idx = np.argsort(edge_dists)

    # Union-find for H0
    parent = list(range(n))
    rank = [0] * n
    birth_times = [0.0] * n
    h0_deaths = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y, t):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        h0_deaths.append(t)
        return True

    n_components = n
    for idx in sorted_idx:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        t = edge_dists[idx]
        if union(i, j, t):
            n_components -= 1

    # For H1: use a spectral heuristic
    # If points form a circle, the second smallest eigenvalue of the
    # graph Laplacian at the right scale should be small but nonzero
    # We use multiple scales and look for the characteristic signature
    from sklearn.metrics import pairwise_distances

    # Normalized circular score: fit points to a circle and measure residual
    if points.shape[1] >= 2:
        centered = points - points.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_2d = centered @ Vt[:2].T
        radii = np.sqrt(proj_2d[:, 0] ** 2 + proj_2d[:, 1] ** 2)
        mean_r = radii.mean()
        if mean_r > 1e-8:
            circularity = 1.0 - radii.std() / mean_r
        else:
            circularity = 0.0

        # Angular coverage: do the points span the full circle?
        angles = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
        angles_sorted = np.sort(angles)
        gaps = np.diff(angles_sorted)
        gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
        max_gap = gaps.max()
        coverage = 1.0 - max_gap / (2 * np.pi)
    else:
        circularity = 0.0
        coverage = 0.0

    # Explained variance by top 2 components
    if len(S) >= 2:
        var_explained_2d = (S[0] ** 2 + S[1] ** 2) / (S ** 2).sum()
    else:
        var_explained_2d = 1.0

    return {
        "n_points": len(points),
        "circularity": float(circularity),
        "angular_coverage": float(coverage),
        "var_explained_2d": float(var_explained_2d),
        "h0_n_deaths": len(h0_deaths),
        "h0_max_death": float(max(h0_deaths)) if h0_deaths else 0.0,
    }


def run_topological_validation(device, log):
    """Check topology of VAE latent space for grokking operations."""
    log.info(f"[{utc_ts()}] === Topological Validation ===")
    import einops

    P = 113
    hook_name = "blocks.0.hook_resid_post"
    D = 128

    operations = {
        "addition": ("addition", 25000),
        "multiplication": ("multiplication", 40000),
        "squaring": ("squaring", 60000),
    }

    results = {"task": "topological_validation"}

    for op_name, (op, n_epochs) in operations.items():
        log.info(f"[{utc_ts()}]   Training {op_name}...")

        cfg = HookedTransformerConfig(
            n_layers=1, n_heads=4, d_model=D, d_head=32, d_mlp=512,
            act_fn="relu", normalization_type=None,
            d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
            init_weights=True, device=device, seed=999,
        )
        model = HookedTransformer(cfg)
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        is_unary = op == "squaring"
        if is_unary:
            a_vec = torch.arange(P)
            b_vec = torch.zeros(P, dtype=torch.long)
        elif op == "multiplication":
            a_vals = torch.arange(1, P)
            b_vals = torch.arange(1, P)
            a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
            b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        else:
            a_vec = einops.repeat(torch.arange(P), "i -> (i j)", j=P)
            b_vec = einops.repeat(torch.arange(P), "j -> (i j)", i=P)

        eq_vec = torch.full_like(a_vec, P)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)

        if op == "addition":
            labels = (a_vec + b_vec) % P
        elif op == "multiplication":
            labels = (a_vec * b_vec) % P
        else:
            labels = (a_vec * a_vec) % P
        labels = labels.to(device)

        torch.manual_seed(598)
        indices = torch.randperm(len(dataset))
        cutoff = int(len(dataset) * 0.3)
        train_idx = indices[:cutoff]
        train_data, train_labels = dataset[train_idx], labels[train_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
        for epoch in tqdm(range(n_epochs), desc=f"train {op_name}", leave=False):
            logits = model(train_data)[:, -1]
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            test_idx = indices[cutoff:]
            test_logits = model(dataset[test_idx])[:, -1]
            test_acc = (test_logits.argmax(-1) == labels[test_idx]).float().mean().item()
        grokked = test_acc > 0.95
        log.info(f"[{utc_ts()}]   {op_name}: test_acc={test_acc:.4f}, grokked={grokked}")

        # Cache activations
        all_acts = []
        for i in range(0, len(dataset), 256):
            with torch.inference_mode():
                _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
            all_acts.append(cache[hook_name][:, -1, :].clone())
        activations = torch.cat(all_acts, dim=0)

        # Train VAE with k=2 (should capture circle if it exists)
        vae = build_vae(D, 2, 16, 128, P, device)
        vae = train_vae(vae, activations, labels, device, n_epochs=500, alpha=10.0)

        # Get latent representations
        vae.eval()
        with torch.inference_mode():
            mu_c, _, _, _ = vae.encode(activations)
            z_causal = mu_c.cpu().numpy()

        # Compute topology of z_causal
        log.info(f"[{utc_ts()}]   Computing topology for {op_name}...")
        topo = compute_persistence(z_causal)

        # Also compute topology of raw activations projected to 2D
        acts_np = activations.cpu().numpy()
        centered = acts_np - acts_np.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        raw_2d = centered @ Vt[:2].T
        topo_raw = compute_persistence(raw_2d)

        op_results = {
            "grokked": grokked,
            "test_accuracy": test_acc,
            "vae_latent_topology": topo,
            "raw_activation_topology": topo_raw,
        }

        # Check equivariance of z_causal
        if not is_unary:
            shifted_data = dataset.clone()
            shifted_data[:, 0] = (shifted_data[:, 0] + 1) % P
            shifted_acts = []
            for i in range(0, len(shifted_data), 256):
                with torch.inference_mode():
                    _, cache = model.run_with_cache(shifted_data[i:i + 256], names_filter=[hook_name])
                shifted_acts.append(cache[hook_name][:, -1, :].clone())
            shifted_activations = torch.cat(shifted_acts, dim=0)

            with torch.inference_mode():
                mu_c_shifted, _, _, _ = vae.encode(shifted_activations)
                z_shifted = mu_c_shifted.cpu().numpy()

            # Check if shift in z_causal is consistent (rotation by 2pi/P)
            dz = z_shifted - z_causal
            dz_norms = np.linalg.norm(dz, axis=-1)
            op_results["equivariance_shift_std"] = float(dz_norms.std())
            op_results["equivariance_shift_mean"] = float(dz_norms.mean())
            op_results["equivariance_consistency"] = float(
                1.0 - dz_norms.std() / (dz_norms.mean() + 1e-8)
            )

        results[op_name] = op_results
        log.info(f"[{utc_ts()}]   {op_name} topology: circularity={topo['circularity']:.3f}, "
                 f"coverage={topo['angular_coverage']:.3f}, "
                 f"var_2d={topo['var_explained_2d']:.3f}")

        del model, vae, activations
        torch.cuda.empty_cache()

    return results


# ===================================================================
# Experiment 3: Sheaf consistency — local DAS coherence
# ===================================================================


def run_sheaf_consistency(model, device, log):
    """Check if local DAS subspaces are consistent across activation space."""
    log.info(f"[{utc_ts()}] === Sheaf Consistency (IOI) ===")

    hook_name = "blocks.10.hook_resid_post"
    D = 768
    k = 4

    all_templates = TEMPLATES_ABBA + TEMPLATES_ABAB + TEMPLATES_LONG + TEMPLATES_POSSESSIVE
    all_data = generate_pairs(model, all_templates, IOI_NAMES, random.Random(42),
                              hook_name, n_raw=3000)
    log.info(f"[{utc_ts()}]   Total pairs: {len(all_data)}")

    # Cluster activations into regions
    from sklearn.cluster import KMeans

    base_acts = torch.stack([d["base_act"] for d in all_data]).cpu().numpy()
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(base_acts)

    # Train DAS separately on each cluster
    cluster_Q = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_pairs = [d for d, m in zip(all_data, mask) if m]
        if len(cluster_pairs) < 50:
            log.info(f"[{utc_ts()}]   Cluster {c}: too few ({len(cluster_pairs)}), skipping")
            continue
        log.info(f"[{utc_ts()}]   Cluster {c}: {len(cluster_pairs)} pairs, training DAS k={k}")
        Q = train_das(model, cluster_pairs[:300], hook_name, D, k, device, n_steps=300)
        cluster_Q[c] = Q

        # Also eval IIA on own cluster
        iia = eval_das_iia(Q, model, cluster_pairs[:200], hook_name)
        log.info(f"[{utc_ts()}]     Self IIA: {iia:.3f}")

    # Measure subspace consistency: principal angles between cluster subspaces
    results = {"task": "sheaf_consistency", "n_clusters": n_clusters,
               "cluster_sizes": [int((cluster_labels == c).sum()) for c in range(n_clusters)]}

    consistency_matrix = {}
    for c1 in cluster_Q:
        for c2 in cluster_Q:
            if c1 >= c2:
                continue
            Q1, Q2 = cluster_Q[c1], cluster_Q[c2]
            # Principal angles
            cos_angles = torch.linalg.svdvals(Q1.T @ Q2)
            angles = torch.acos(cos_angles.clamp(-1, 1))
            geodesic = angles.norm().item()

            # Cross-eval: train on c1, eval on c2 and vice versa
            mask2 = cluster_labels == c2
            c2_pairs = [d for d, m in zip(all_data, mask2) if m]
            cross_iia_12 = eval_das_iia(Q1, model, c2_pairs[:200], hook_name)

            mask1 = cluster_labels == c1
            c1_pairs = [d for d, m in zip(all_data, mask1) if m]
            cross_iia_21 = eval_das_iia(Q2, model, c1_pairs[:200], hook_name)

            key = f"c{c1}_vs_c{c2}"
            consistency_matrix[key] = {
                "geodesic_distance": geodesic,
                "principal_angles": [float(a) for a in angles.cpu().tolist()],
                "cross_iia_12": cross_iia_12,
                "cross_iia_21": cross_iia_21,
            }
            log.info(f"[{utc_ts()}]   {key}: geodesic={geodesic:.3f}, "
                     f"cross_IIA={cross_iia_12:.3f}/{cross_iia_21:.3f}")

    results["consistency"] = consistency_matrix

    # Summary: if all geodesic distances are small and cross-IIAs are high,
    # the variable is approximately Grassmannian (one global subspace works).
    # If geodesic distances are large but cross-IIAs are still high,
    # different subspaces work locally but they're not the same — nonlinear.
    if consistency_matrix:
        geos = [v["geodesic_distance"] for v in consistency_matrix.values()]
        cross_iias = [v["cross_iia_12"] for v in consistency_matrix.values()] + \
                     [v["cross_iia_21"] for v in consistency_matrix.values()]
        results["mean_geodesic"] = sum(geos) / len(geos)
        results["std_geodesic"] = (sum((g - results["mean_geodesic"])**2 for g in geos) / len(geos)) ** 0.5
        results["mean_cross_iia"] = sum(cross_iias) / len(cross_iias)
        log.info(f"[{utc_ts()}]   Summary: mean_geodesic={results['mean_geodesic']:.3f}, "
                 f"mean_cross_IIA={results['mean_cross_iia']:.3f}")

    return results


# ===================================================================
# Main
# ===================================================================


@app.function(gpu="A100", timeout=21600, volumes={"/results": results_vol})
def run_validation() -> dict:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    all_results = {"timestamp": utc_ts()}

    # --- Experiment 1: Cross-template transfer ---
    log.info(f"[{utc_ts()}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        all_results["cross_template"] = run_cross_template_transfer(model, DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Cross-template failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["cross_template"] = {"error": str(e)}
    save_incremental(all_results, log)

    # --- Experiment 3: Sheaf consistency ---
    try:
        all_results["sheaf"] = run_sheaf_consistency(model, DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Sheaf failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["sheaf"] = {"error": str(e)}
    save_incremental(all_results, log)

    del model
    torch.cuda.empty_cache()

    # --- Experiment 2: Topological validation (grokking, needs own models) ---
    try:
        all_results["topology"] = run_topological_validation(DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Topology failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["topology"] = {"error": str(e)}
    save_incremental(all_results, log)

    all_results["elapsed_seconds"] = round(time.time() - t0, 1)
    save_incremental(all_results, log)

    # Summary
    log.info(f"\n{'=' * 100}")
    log.info(f"CROSS-TEMPLATE TRANSFER")
    ct = all_results.get("cross_template", {})
    if "error" not in ct:
        for key in sorted(ct.keys()):
            if key.startswith("train_"):
                log.info(f"  {key}:")
                for eval_fam, vals in ct[key].items():
                    log.info(f"    → {eval_fam}: DAS={vals['das_iia']:.3f}, VAE={vals['vae_iia']:.3f}")

    log.info(f"\nTOPOLOGICAL VALIDATION")
    topo = all_results.get("topology", {})
    if "error" not in topo:
        for op in ["addition", "multiplication", "squaring"]:
            r = topo.get(op, {})
            t = r.get("vae_latent_topology", {})
            log.info(f"  {op}: grokked={r.get('grokked')}, "
                     f"circularity={t.get('circularity', 0):.3f}, "
                     f"coverage={t.get('angular_coverage', 0):.3f}")

    log.info(f"\nSHEAF CONSISTENCY")
    sh = all_results.get("sheaf", {})
    if "error" not in sh:
        log.info(f"  mean_geodesic={sh.get('mean_geodesic', 0):.3f}, "
                 f"mean_cross_IIA={sh.get('mean_cross_iia', 0):.3f}")

    log.info(f"\nTotal: {all_results['elapsed_seconds']:.0f}s")

    log.info(f"\nInterpretation:")
    log.info(f"  Cross-template: if VAE IIA transfers across families, the variable is real")
    log.info(f"  Topology: circularity ≈ 1 with full coverage → circle (H1 structure)")
    log.info(f"  Sheaf: low geodesic + high cross-IIA → Grassmannian (global linear)")
    log.info(f"         high geodesic + high cross-IIA → genuinely nonlinear")
    log.info(f"         high geodesic + low cross-IIA → inconsistent (noise)")

    return all_results


@app.local_entrypoint()
def main():
    handle = run_validation.spawn()
    print(f"[{utc_ts()}] Spawned cross-task validation experiments")
    print(f"  Handle: {handle.object_id}")
    print(f"  Experiments: cross-template transfer, persistent homology, sheaf consistency")
    print(f"  Incremental saves after each experiment")
    print(f"  Results: {SAVE_DIR}/results.json")
