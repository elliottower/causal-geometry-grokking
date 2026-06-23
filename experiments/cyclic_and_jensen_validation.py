"""Cyclic group topology + Jensen IOI variants + stratification validation.

Complements cross_task_validation.py with:

1. CYCLIC GROKKING: Train on mod-7 (days), mod-12 (months), mod-24 (hours).
   Small cyclic groups where we can directly visualize and verify that z_causal
   forms a circle with the right number of clusters. Equivariance check:
   shifting input by +1 should rotate z_causal by 2π/p.

2. JENSEN IOI VARIANTS (Nainani et al. 2024): Train VAE on base IOI, test on
   DoubleIO and TripleIO. These variants duplicate IO tokens, breaking the IOI
   algorithm, yet GPT-2 still solves them. If VAE's z_causal transfers from
   base IOI to DoubleIO/TripleIO, the variable is real.

3. STRATIFICATION: Cluster activations by prediction confidence, check if
   Grassmannian structure (DAS subspace) varies across strata. If subspaces
   differ by stratum, the representation is stratified.

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/cyclic_and_jensen_validation.py
"""
from __future__ import annotations

import json
import logging
import math
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

app = modal.App("cyclic-jensen-validation", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

SAVE_DIR = "/results/grassmannian_atlas/cyclic_jensen_validation"


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_incremental(all_results, log):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(f"{SAVE_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {SAVE_DIR}/results.json")


# ===================================================================
# VAE and DAS (shared utilities)
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
# Experiment 1: Cyclic group grokking (mod-7, mod-12, mod-24)
# ===================================================================


def train_grokking_model(P, operation, device, n_epochs, d_model=128):
    """Train a small transformer on modular arithmetic until grokking."""
    import einops

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=d_model, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
        init_weights=True, device=device, seed=42,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    a_vec = einops.repeat(torch.arange(P), "i -> (i j)", j=P)
    b_vec = einops.repeat(torch.arange(P), "j -> (i j)", i=P)
    eq_vec = torch.full_like(a_vec, P)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)

    if operation == "addition":
        labels = (a_vec + b_vec) % P
    elif operation == "subtraction":
        labels = (a_vec - b_vec) % P
    else:
        labels = (a_vec * b_vec) % P
    labels = labels.to(device)

    torch.manual_seed(598)
    indices = torch.randperm(len(dataset))
    cutoff = int(len(dataset) * 0.3)
    train_idx = indices[:cutoff]
    train_data, train_labels = dataset[train_idx], labels[train_idx]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)
    )

    for epoch in tqdm(range(n_epochs), desc=f"grok P={P}", leave=False):
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

    return model, dataset, labels, a_vec, b_vec, test_acc


def compute_topology_metrics(points):
    """Compute circularity, angular coverage, and explained variance."""
    centered = points - points.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_2d = centered @ Vt[:2].T

    radii = np.sqrt(proj_2d[:, 0] ** 2 + proj_2d[:, 1] ** 2)
    mean_r = radii.mean()
    circularity = 1.0 - radii.std() / (mean_r + 1e-8) if mean_r > 1e-8 else 0.0

    angles = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
    coverage = 1.0 - gaps.max() / (2 * np.pi)

    var_2d = (S[0] ** 2 + S[1] ** 2) / (S ** 2).sum() if len(S) >= 2 else 1.0

    return {
        "circularity": float(circularity),
        "angular_coverage": float(coverage),
        "var_explained_2d": float(var_2d),
        "proj_2d": proj_2d,
    }


def check_cluster_structure(z_causal, labels_np, P):
    """Check if z_causal has P distinct clusters arranged on a circle."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=P, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(z_causal)

    # Check if cluster assignments match modular labels
    # (up to permutation — find best mapping)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_np, cluster_labels)

    # Check if cluster centers form a circle
    centers = km.cluster_centers_
    center_topo = compute_topology_metrics(centers)

    return {
        "adjusted_rand_index": float(ari),
        "center_circularity": center_topo["circularity"],
        "center_coverage": center_topo["angular_coverage"],
    }


def check_equivariance(vae, model, dataset, labels, a_vec, P, hook_name, device):
    """Check if +1 shift in input produces consistent rotation in z_causal."""
    # Original activations
    all_acts = []
    for i in range(0, len(dataset), 256):
        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].clone())
    activations = torch.cat(all_acts, dim=0)

    # Shifted activations: a → a+1
    shifted = dataset.clone()
    shifted[:, 0] = (shifted[:, 0] + 1) % P
    shifted_acts = []
    for i in range(0, len(shifted), 256):
        with torch.inference_mode():
            _, cache = model.run_with_cache(shifted[i:i + 256], names_filter=[hook_name])
        shifted_acts.append(cache[hook_name][:, -1, :].clone())
    shifted_activations = torch.cat(shifted_acts, dim=0)

    vae.eval()
    with torch.inference_mode():
        mu_orig, _, _, _ = vae.encode(activations)
        mu_shifted, _, _, _ = vae.encode(shifted_activations)

    z_orig = mu_orig.cpu().numpy()
    z_shifted = mu_shifted.cpu().numpy()

    # The shift vector should be consistent across all examples
    dz = z_shifted - z_orig
    dz_norms = np.linalg.norm(dz, axis=-1)

    # Consistency: std of shift norms should be small relative to mean
    consistency = 1.0 - dz_norms.std() / (dz_norms.mean() + 1e-8)

    # Check if the shift is approximately a rotation by 2π/P
    # Project to 2D and check angle of rotation
    topo = compute_topology_metrics(z_orig)
    proj_orig = topo["proj_2d"]

    centered_shifted = z_shifted - z_shifted.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_orig - z_orig.mean(axis=0), full_matrices=False)
    proj_shifted = centered_shifted @ Vt[:2].T

    angles_orig = np.arctan2(proj_orig[:, 1], proj_orig[:, 0])
    angles_shifted = np.arctan2(proj_shifted[:, 1], proj_shifted[:, 0])
    angle_diffs = angles_shifted - angles_orig
    # Wrap to [-π, π]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

    expected_rotation = 2 * np.pi / P
    mean_rotation = np.abs(angle_diffs).mean()
    rotation_error = abs(mean_rotation - expected_rotation) / expected_rotation

    return {
        "shift_norm_mean": float(dz_norms.mean()),
        "shift_norm_std": float(dz_norms.std()),
        "shift_consistency": float(consistency),
        "mean_rotation_rad": float(mean_rotation),
        "expected_rotation_rad": float(expected_rotation),
        "rotation_error_fraction": float(rotation_error),
        "angle_diff_std": float(np.std(angle_diffs)),
    }, activations


def run_cyclic_experiments(device, log):
    """Train on small cyclic groups and check topology + equivariance."""
    log.info(f"[{utc_ts()}] === Cyclic Group Experiments ===")

    hook_name = "blocks.0.hook_resid_post"
    D = 128

    # Cyclic groups: "days" (mod 7), "months" (mod 12), "hours" (mod 24)
    # Also mod 5 (small prime) for comparison
    configs = [
        {"name": "mod5_addition", "P": 5, "op": "addition", "epochs": 30000, "label": "pentagon"},
        {"name": "mod7_days", "P": 7, "op": "addition", "epochs": 30000, "label": "days of week"},
        {"name": "mod12_months", "P": 12, "op": "addition", "epochs": 35000, "label": "months"},
        {"name": "mod24_hours", "P": 24, "op": "addition", "epochs": 40000, "label": "hours"},
        {"name": "mod7_subtraction", "P": 7, "op": "subtraction", "epochs": 30000, "label": "day offset (sub)"},
    ]

    results = {"task": "cyclic_groups"}

    for cfg in configs:
        log.info(f"[{utc_ts()}]   {cfg['name']} ({cfg['label']}): P={cfg['P']}, training...")
        t0 = time.time()

        model, dataset, labels, a_vec, b_vec, test_acc = train_grokking_model(
            cfg["P"], cfg["op"], device, cfg["epochs"], d_model=D
        )
        grokked = test_acc > 0.95
        log.info(f"[{utc_ts()}]   test_acc={test_acc:.4f}, grokked={grokked}, "
                 f"train_time={time.time() - t0:.0f}s")

        if not grokked:
            results[cfg["name"]] = {"grokked": False, "test_acc": test_acc}
            del model
            torch.cuda.empty_cache()
            continue

        # Cache all activations
        all_acts = []
        for i in range(0, len(dataset), 256):
            with torch.inference_mode():
                _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
            all_acts.append(cache[hook_name][:, -1, :].clone())
        activations = torch.cat(all_acts, dim=0)

        # Train VAE with k=2 (should capture circle)
        P = cfg["P"]
        vae = build_vae(D, 2, 16, 128, P, device)
        vae = train_vae(vae, activations, labels, device, n_epochs=600, alpha=10.0)

        # Get latent z_causal
        vae.eval()
        with torch.inference_mode():
            mu_c, _, _, _ = vae.encode(activations)
        z_causal = mu_c.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # 1. Topology of z_causal
        topo = compute_topology_metrics(z_causal)
        log.info(f"[{utc_ts()}]   Topology: circ={topo['circularity']:.3f}, "
                 f"cov={topo['angular_coverage']:.3f}, var2d={topo['var_explained_2d']:.3f}")

        # 2. Cluster structure (P clusters on a circle?)
        clusters = check_cluster_structure(z_causal, labels_np, P)
        log.info(f"[{utc_ts()}]   Clusters: ARI={clusters['adjusted_rand_index']:.3f}, "
                 f"center_circ={clusters['center_circularity']:.3f}")

        # 3. Equivariance check (+1 shift → 2π/P rotation)
        equivar, _ = check_equivariance(vae, model, dataset, labels, a_vec, P,
                                        hook_name, device)
        log.info(f"[{utc_ts()}]   Equivariance: consistency={equivar['shift_consistency']:.3f}, "
                 f"rotation_error={equivar['rotation_error_fraction']:.3f}")

        # 4. Topology of RAW activations (control — does structure exist before VAE?)
        raw_topo = compute_topology_metrics(activations.cpu().numpy())

        # 5. Per-label latent positions for visualization
        label_positions = {}
        for lab in range(P):
            mask = labels_np == lab
            if mask.sum() > 0:
                pts = z_causal[mask]
                label_positions[str(lab)] = {
                    "mean_x": float(pts[:, 0].mean()),
                    "mean_y": float(pts[:, 1].mean()),
                    "std": float(pts.std()),
                }

        results[cfg["name"]] = {
            "grokked": True,
            "test_accuracy": test_acc,
            "P": P,
            "label": cfg["label"],
            "topology": {k: v for k, v in topo.items() if k != "proj_2d"},
            "raw_topology": {k: v for k, v in raw_topo.items() if k != "proj_2d"},
            "clusters": clusters,
            "equivariance": equivar,
            "label_positions": label_positions,
        }

        del model, vae, activations
        torch.cuda.empty_cache()

    return results


# ===================================================================
# Experiment 2: Jensen's DoubleIO / TripleIO cross-task transfer
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

# Base IOI (standard)
TEMPLATES_BASE = [
    "When{A} and{B} went to the {PLACE},{B} gave a {OBJ} to",
    "Then,{A} and{B} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
]

# DoubleIO: IO appears twice (from Jensen et al.)
TEMPLATES_DOUBLE_IO = [
    "When{A} and{B} went to the {PLACE},{A} was happy.{B} gave a {OBJ} to",
    "Then,{A} and{B} went to the {PLACE}.{A} was happy.{B} gave a {OBJ} to",
    "Then,{A} and{B} had a lot of fun at the {PLACE}.{A} was excited.{B} gave a {OBJ} to",
]

# TripleIO: IO appears three times (from Jensen et al.)
TEMPLATES_TRIPLE_IO = [
    "When{A} and{B} went to the {PLACE},{A} was happy.{A} sat on a bench.{B} gave a {OBJ} to",
    "Then,{A} and{B} went to the {PLACE}.{A} was happy.{A} sat down.{B} gave a {OBJ} to",
]

# ABAB order (reverse of standard ABBA)
TEMPLATES_ABAB = [
    "When{B} and{A} went to the {PLACE},{B} gave a {OBJ} to",
    "Then,{B} and{A} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{B} and{A} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
]

# Possessive / different syntax
TEMPLATES_POSSESSIVE = [
    "{A} and{B} were at the {PLACE}.{B} had a {OBJ} and gave it to",
    "When{A} and{B} arrived at the {PLACE},{B} passed a {OBJ} to",
    "{A} met{B} at the {PLACE}, and{B} offered a {OBJ} to",
]


def generate_ioi_pairs(model, templates, names, rng, hook_name, n_raw=1200):
    """Generate IOI interchange intervention pairs."""
    valid_names = [n for n in names if len(model.tokenizer.encode(n)) == 1]

    raw = []
    for _ in range(n_raw):
        t = rng.choice(templates)
        a, b = rng.sample(valid_names, 2)
        place = rng.choice(IOI_PLACES)
        obj = rng.choice(IOI_OBJECTS)
        base = t.format(A=a, B=b, PLACE=place, OBJ=obj)
        # Counterfactual: swap IO (A) and S (B)
        source = t.format(A=b, B=a, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        raw.append((base, source,
                     model.tokenizer.encode(a)[0],
                     model.tokenizer.encode(b)[0]))

    data = []
    for base_text, src_text, io_id, s_id in tqdm(raw, desc="cache", leave=False):
        bt = model.to_tokens(base_text)
        st = model.to_tokens(src_text)
        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            bl = model(bt)[0, -1]
            _, sc = model.run_with_cache(st, names_filter=hook_name)

        if (bl[io_id] - bl[s_id]).item() > 0:
            data.append({
                "base_act": bc[hook_name][0, -1],
                "src_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_label": s_id,
                "base_label": io_id,
            })
    return data


def run_jensen_cross_task(model, device, log):
    """Train on base IOI, test on DoubleIO/TripleIO/ABAB/possessive."""
    log.info(f"[{utc_ts()}] === Jensen Cross-Task Transfer ===")

    hook_name = "blocks.10.hook_resid_post"
    D = 768

    families = {
        "base_IOI": TEMPLATES_BASE,
        "DoubleIO": TEMPLATES_DOUBLE_IO,
        "TripleIO": TEMPLATES_TRIPLE_IO,
        "ABAB": TEMPLATES_ABAB,
        "possessive": TEMPLATES_POSSESSIVE,
    }

    # Generate data for all families
    family_data = {}
    for name, templates in families.items():
        log.info(f"[{utc_ts()}]   Generating {name}...")
        family_data[name] = generate_ioi_pairs(
            model, templates, IOI_NAMES, random.Random(42), hook_name, n_raw=1200
        )
        log.info(f"[{utc_ts()}]   {name}: {len(family_data[name])} valid pairs")

    results = {"task": "jensen_cross_task"}

    # Train on base IOI only
    train_pairs = family_data["base_IOI"]
    if len(train_pairs) < 50:
        log.error(f"[{utc_ts()}]   Too few base IOI pairs: {len(train_pairs)}")
        return {"error": "too few base IOI pairs"}

    # Build label map from training data
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
        log.info(f"[{utc_ts()}]   Training on base_IOI, k={k}")

        # Train DAS on base IOI
        Q = train_das(model, train_pairs[:400], hook_name, D, k, device, n_steps=400)

        # Train VAE on base IOI
        vae = build_vae(D, k, 16, 256, n_classes, device)
        vae = train_vae(vae, act_t, lab_t, device, n_epochs=500, alpha=10.0)

        # Eval on ALL families
        k_results = {}
        for eval_name in families:
            eval_pairs = family_data[eval_name]
            if len(eval_pairs) < 10:
                continue
            n_eval = min(len(eval_pairs), 300)
            das_iia = eval_das_iia(Q, model, eval_pairs[:n_eval], hook_name)
            vae_iia = eval_vae_iia(vae, model, eval_pairs[:n_eval], hook_name)
            k_results[eval_name] = {
                "das_iia": das_iia,
                "vae_iia": vae_iia,
                "n_pairs": n_eval,
                "is_train_family": eval_name == "base_IOI",
            }
            marker = " (train)" if eval_name == "base_IOI" else ""
            log.info(f"[{utc_ts()}]     {eval_name}{marker}: DAS={das_iia:.3f}, VAE={vae_iia:.3f}")

        results[f"k{k}"] = k_results
        del vae
        torch.cuda.empty_cache()

    # Also compute the logit difference for each family (like Jensen's Table 1)
    log.info(f"[{utc_ts()}]   Computing logit differences per family...")
    for name, pairs in family_data.items():
        logit_diffs = []
        for d in pairs[:200]:
            with torch.inference_mode():
                logits = model(d["base_toks"])[0, -1]
            ld = (logits[d["base_label"]] - logits[d["src_label"]]).item()
            logit_diffs.append(ld)
        results[f"logit_diff_{name}"] = {
            "mean": float(np.mean(logit_diffs)),
            "std": float(np.std(logit_diffs)),
            "accuracy": float(np.mean([ld > 0 for ld in logit_diffs])),
        }
        log.info(f"[{utc_ts()}]   {name}: logit_diff={np.mean(logit_diffs):.3f}, "
                 f"acc={np.mean([ld > 0 for ld in logit_diffs]):.3f}")

    return results


# ===================================================================
# Experiment 3: Stratification — DAS subspace varies by confidence
# ===================================================================


def run_stratification(model, device, log):
    """Check if DAS subspace varies across confidence strata."""
    log.info(f"[{utc_ts()}] === Stratification Analysis (IOI) ===")

    hook_name = "blocks.10.hook_resid_post"
    D = 768
    k = 4

    # Generate a large pool of IOI pairs
    all_templates = TEMPLATES_BASE + TEMPLATES_DOUBLE_IO + TEMPLATES_ABAB
    all_pairs = generate_ioi_pairs(
        model, all_templates, IOI_NAMES, random.Random(42), hook_name, n_raw=3000
    )
    log.info(f"[{utc_ts()}]   Total pairs: {len(all_pairs)}")

    # Compute model confidence for each pair
    confidences = []
    for d in tqdm(all_pairs, desc="confidence", leave=False):
        with torch.inference_mode():
            logits = model(d["base_toks"])[0, -1]
            probs = F.softmax(logits, dim=-1)
            conf = probs[d["base_label"]].item()
        confidences.append(conf)
    confidences = np.array(confidences)

    # Stratify by confidence: low (<0.1), medium (0.1-0.5), high (>0.5)
    strata_defs = {
        "low": (0.0, 0.1),
        "medium": (0.1, 0.5),
        "high": (0.5, 1.0),
    }

    results = {"task": "stratification", "confidence_stats": {
        "mean": float(confidences.mean()),
        "std": float(confidences.std()),
        "min": float(confidences.min()),
        "max": float(confidences.max()),
    }}

    strata_Q = {}
    for sname, (lo, hi) in strata_defs.items():
        mask = (confidences >= lo) & (confidences < hi)
        stratum_pairs = [p for p, m in zip(all_pairs, mask) if m]
        n = len(stratum_pairs)
        log.info(f"[{utc_ts()}]   Stratum '{sname}' ({lo:.1f}-{hi:.1f}): {n} pairs")

        if n < 50:
            results[sname] = {"n_pairs": n, "skipped": True}
            continue

        # Train DAS on this stratum
        Q = train_das(model, stratum_pairs[:300], hook_name, D, k, device, n_steps=300)
        strata_Q[sname] = Q

        # Self-eval IIA
        self_iia = eval_das_iia(Q, model, stratum_pairs[:200], hook_name)
        results[sname] = {"n_pairs": n, "self_iia": self_iia}
        log.info(f"[{utc_ts()}]     Self IIA: {self_iia:.3f}")

    # Cross-stratum subspace comparison
    consistency = {}
    for s1 in strata_Q:
        for s2 in strata_Q:
            if s1 >= s2:
                continue
            Q1, Q2 = strata_Q[s1], strata_Q[s2]
            cos_angles = torch.linalg.svdvals(Q1.T @ Q2)
            angles = torch.acos(cos_angles.clamp(-1, 1))
            geodesic = angles.norm().item()

            # Cross-eval
            mask2 = (confidences >= strata_defs[s2][0]) & (confidences < strata_defs[s2][1])
            s2_pairs = [p for p, m in zip(all_pairs, mask2) if m]
            cross_12 = eval_das_iia(Q1, model, s2_pairs[:200], hook_name)

            mask1 = (confidences >= strata_defs[s1][0]) & (confidences < strata_defs[s1][1])
            s1_pairs = [p for p, m in zip(all_pairs, mask1) if m]
            cross_21 = eval_das_iia(Q2, model, s1_pairs[:200], hook_name)

            key = f"{s1}_vs_{s2}"
            consistency[key] = {
                "geodesic": geodesic,
                "principal_angles": [float(a) for a in angles.cpu().tolist()],
                "cross_iia_12": cross_12,
                "cross_iia_21": cross_21,
            }
            log.info(f"[{utc_ts()}]   {key}: geodesic={geodesic:.3f}, "
                     f"cross_IIA={cross_12:.3f}/{cross_21:.3f}")

    results["cross_stratum"] = consistency

    # Interpretation
    if consistency:
        geos = [v["geodesic"] for v in consistency.values()]
        cross = [v["cross_iia_12"] for v in consistency.values()] + \
                [v["cross_iia_21"] for v in consistency.values()]
        results["summary"] = {
            "mean_geodesic": sum(geos) / len(geos),
            "mean_cross_iia": sum(cross) / len(cross),
            "interpretation": (
                "uniform" if sum(geos) / len(geos) < 0.5 else
                "stratified" if sum(cross) / len(cross) < 0.3 else
                "mildly_stratified"
            ),
        }
        log.info(f"[{utc_ts()}]   Interpretation: {results['summary']['interpretation']}")

    return results


# ===================================================================
# Experiment 4: Save visualization data for cyclic groups
# ===================================================================


def save_visualization_data(all_results, log):
    """Save numpy arrays for publication-quality circle plots."""
    os.makedirs(f"{SAVE_DIR}/viz", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cyclic = all_results.get("cyclic", {})
    for name in cyclic:
        if name == "task":
            continue
        r = cyclic[name]
        if not r.get("grokked"):
            continue

        positions = r.get("label_positions", {})
        if not positions:
            continue

        P = r["P"]
        xs = [positions[str(i)]["mean_x"] for i in range(P)]
        ys = [positions[str(i)]["mean_y"] for i in range(P)]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, P, endpoint=False))
        for i in range(P):
            ax.scatter(xs[i], ys[i], c=[colors[i]], s=100, zorder=5)
            ax.annotate(str(i), (xs[i], ys[i]), fontsize=10, ha='center', va='bottom')

        # Draw unit circle for reference
        theta = np.linspace(0, 2 * np.pi, 100)
        r_scale = (max(max(xs) - min(xs), max(ys) - min(ys))) / 2
        cx, cy = np.mean(xs), np.mean(ys)
        ax.plot(cx + r_scale * np.cos(theta), cy + r_scale * np.sin(theta),
                'k--', alpha=0.3, linewidth=1)

        ax.set_title(f"{r['label']} (mod {P})\ncirc={r['topology']['circularity']:.3f}, "
                     f"ARI={r['clusters']['adjusted_rand_index']:.3f}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(f"{SAVE_DIR}/viz/{name}_circle.png", dpi=150)
        plt.close(fig)
        log.info(f"[{utc_ts()}]   Saved {name}_circle.png")

    results_vol.commit()


# ===================================================================
# Main
# ===================================================================


@app.function(gpu="A100", timeout=21600, volumes={"/results": results_vol})
def run_all() -> dict:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    all_results = {"timestamp": utc_ts()}

    # --- Experiment 1: Cyclic groups ---
    try:
        all_results["cyclic"] = run_cyclic_experiments(DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Cyclic failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["cyclic"] = {"error": str(e)}
    save_incremental(all_results, log)

    # Save visualizations
    try:
        save_visualization_data(all_results, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Viz failed: {e}")

    # --- Experiments 2-3: Jensen + Stratification (need GPT-2) ---
    log.info(f"[{utc_ts()}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        all_results["jensen"] = run_jensen_cross_task(model, DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Jensen failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["jensen"] = {"error": str(e)}
    save_incremental(all_results, log)

    try:
        all_results["stratification"] = run_stratification(model, DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] Stratification failed: {e}\n{traceback.format_exc()[-1500:]}")
        all_results["stratification"] = {"error": str(e)}
    save_incremental(all_results, log)

    del model
    torch.cuda.empty_cache()

    all_results["elapsed_seconds"] = round(time.time() - t0, 1)
    save_incremental(all_results, log)

    # Summary
    log.info(f"\n{'=' * 100}")
    log.info("SUMMARY")
    log.info(f"{'=' * 100}")

    cyclic = all_results.get("cyclic", {})
    if "error" not in cyclic:
        log.info("\nCYCLIC GROUPS:")
        for name in sorted(cyclic.keys()):
            if name == "task":
                continue
            r = cyclic[name]
            if r.get("grokked"):
                t = r["topology"]
                c = r["clusters"]
                e = r["equivariance"]
                log.info(f"  {name} (P={r['P']}): circ={t['circularity']:.3f}, "
                         f"ARI={c['adjusted_rand_index']:.3f}, "
                         f"equivar={e['shift_consistency']:.3f}, "
                         f"rot_err={e['rotation_error_fraction']:.3f}")
            else:
                log.info(f"  {name}: NOT GROKKED (acc={r['test_accuracy']:.3f})")

    jensen = all_results.get("jensen", {})
    if "error" not in jensen:
        log.info("\nJENSEN CROSS-TASK (train=base_IOI):")
        for kk in ["k2", "k4"]:
            kr = jensen.get(kk, {})
            if kr:
                log.info(f"  {kk}:")
                for fam, vals in kr.items():
                    marker = " *" if vals.get("is_train_family") else ""
                    log.info(f"    {fam}: DAS={vals['das_iia']:.3f}, VAE={vals['vae_iia']:.3f}{marker}")

    strat = all_results.get("stratification", {})
    if "error" not in strat and "summary" in strat:
        s = strat["summary"]
        log.info(f"\nSTRATIFICATION: {s['interpretation']} "
                 f"(geodesic={s['mean_geodesic']:.3f}, cross_IIA={s['mean_cross_iia']:.3f})")

    log.info(f"\nTotal: {all_results['elapsed_seconds']:.0f}s")

    return all_results


@app.local_entrypoint()
def main():
    handle = run_all.spawn()
    print(f"[{utc_ts()}] Spawned cyclic + Jensen + stratification experiments")
    print(f"  Handle: {handle.object_id}")
    print(f"  Experiments:")
    print(f"    1. Cyclic groups: mod-5/7/12/24 addition + mod-7 subtraction")
    print(f"       → topology, cluster structure, equivariance")
    print(f"    2. Jensen cross-task: train base IOI → eval DoubleIO/TripleIO/ABAB/possessive")
    print(f"    3. Stratification: DAS subspace by confidence stratum")
    print(f"  Incremental saves after each experiment")
    print(f"  Results: {SAVE_DIR}/results.json + viz/*.png")
