"""Cross-subtask transfer matrix for IOI using MIB benchmark counterfactuals.

The MIB IOI dataset has 8 counterfactual types:
  s2_io_flip, s1_io_flip, abc, random_names, s1_ioi_flip_s2_ioi_flip,
  random_names_s1_ioi_flip, random_names_s2_ioi_flip,
  random_names_s1_ioi_flip_s2_ioi_flip

Train DAS/VAE on each subtask → eval on ALL others.
If z_causal transfers across subtasks, the variable is real.
If it only works on the training subtask, it overfit the counterfactual structure.

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/ioi_subtask_transfer.py
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
    from transformer_lens import HookedTransformer
except (ImportError, AttributeError):
    pass

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "setuptools<71")
    .pip_install(
        "transformer-lens==2.11.0", "transformers==4.46.3",
        "einops>=0.8", "matplotlib", "tqdm", "datasets",
    )
)

app = modal.App("ioi-subtask-transfer", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

SAVE_DIR = "/results/grassmannian_atlas/ioi_subtask_transfer"

SUBTASK_NAMES = [
    "s2_io_flip_counterfactual",
    "s1_io_flip_counterfactual",
    "abc_counterfactual",
    "random_names_counterfactual",
    "s1_ioi_flip_s2_ioi_flip_counterfactual",
    "random_names_s1_ioi_flip_counterfactual",
    "random_names_s2_ioi_flip_counterfactual",
    "random_names_s1_ioi_flip_s2_ioi_flip_counterfactual",
]

SHORT_NAMES = {
    "s2_io_flip_counterfactual": "s2_flip",
    "s1_io_flip_counterfactual": "s1_flip",
    "abc_counterfactual": "abc",
    "random_names_counterfactual": "rand_names",
    "s1_ioi_flip_s2_ioi_flip_counterfactual": "full_flip",
    "random_names_s1_ioi_flip_counterfactual": "rand_s1",
    "random_names_s2_ioi_flip_counterfactual": "rand_s2",
    "random_names_s1_ioi_flip_s2_ioi_flip_counterfactual": "rand_full",
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_incremental(all_results, log):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(f"{SAVE_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {SAVE_DIR}/results.json")


# ===================================================================
# VAE
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


# ===================================================================
# DAS
# ===================================================================


def train_das(model, train_pairs, hook_name, d_model, k, device,
              n_steps=400, lr=1e-3, batch_size=16):
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

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            loss = loss - F.log_softmax(logits, dim=-1)[d["src_label"]]
        (loss / len(batch)).backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
    return Q.detach()


# ===================================================================
# Eval (IIA + continuous metrics)
# ===================================================================


def eval_das_full(Q, model, pairs, hook_name):
    proj = Q @ Q.T
    correct = 0
    kl_divs, js_divs, prob_diffs = [], [], []
    with torch.inference_mode():
        for d in pairs:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            clean_logits = model(d["base_toks"])[0, -1, :]

            if logits.argmax().item() == d["src_label"]:
                correct += 1

            p_iv = F.softmax(logits, dim=-1)
            p_cl = F.softmax(clean_logits, dim=-1)
            prob_diffs.append((p_iv[d["src_label"]] - p_cl[d["src_label"]]).item())

            lp_iv = F.log_softmax(logits, dim=-1)
            lp_cl = F.log_softmax(clean_logits, dim=-1)
            kl = (p_cl * (lp_cl - lp_iv)).sum().item()
            kl_divs.append(kl)
            m = 0.5 * (p_cl + p_iv)
            js = 0.5 * (p_cl * (lp_cl - m.log())).sum().item() + \
                 0.5 * (p_iv * (lp_iv - m.log())).sum().item()
            js_divs.append(js)

    n = len(pairs)
    return {
        "iia": correct / n if n else 0,
        "kl_mean": float(np.mean(kl_divs)) if kl_divs else 0,
        "js_mean": float(np.mean(js_divs)) if js_divs else 0,
        "prob_diff_mean": float(np.mean(prob_diffs)) if prob_diffs else 0,
        "n": n,
    }


def eval_vae_full(vae, model, pairs, hook_name):
    vae.eval()
    correct = 0
    kl_divs, js_divs, prob_diffs = [], [], []
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

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            clean_logits = model(d["base_toks"])[0, -1, :]

            if logits.argmax().item() == d["src_label"]:
                correct += 1

            p_iv = F.softmax(logits, dim=-1)
            p_cl = F.softmax(clean_logits, dim=-1)
            prob_diffs.append((p_iv[d["src_label"]] - p_cl[d["src_label"]]).item())

            lp_iv = F.log_softmax(logits, dim=-1)
            lp_cl = F.log_softmax(clean_logits, dim=-1)
            kl = (p_cl * (lp_cl - lp_iv)).sum().item()
            kl_divs.append(kl)
            m = 0.5 * (p_cl + p_iv)
            js = 0.5 * (p_cl * (lp_cl - m.log())).sum().item() + \
                 0.5 * (p_iv * (lp_iv - m.log())).sum().item()
            js_divs.append(js)

    n = len(pairs)
    return {
        "iia": correct / n if n else 0,
        "kl_mean": float(np.mean(kl_divs)) if kl_divs else 0,
        "js_mean": float(np.mean(js_divs)) if js_divs else 0,
        "prob_diff_mean": float(np.mean(prob_diffs)) if prob_diffs else 0,
        "n": n,
    }


# ===================================================================
# Load MIB IOI subtasks
# ===================================================================


def load_mib_ioi_pairs(model, subtask_name, hook_name, device, n_examples=500):
    """Load IOI pairs from mib-bench/ioi HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset("mib-bench/ioi", split="train")
    if n_examples < len(ds):
        ds = ds.select(range(n_examples))

    pairs = []
    for row in tqdm(ds, desc=f"load {SHORT_NAMES.get(subtask_name, subtask_name)}", leave=False):
        cf = row.get(subtask_name)
        if cf is None or cf.get("prompt") is None:
            continue

        base_prompt = row["prompt"]
        src_prompt = cf["prompt"]
        meta = row["metadata"]

        io_name = f" {meta['indirect_object']}"
        s_name = f" {meta['subject']}"

        io_ids = model.tokenizer.encode(io_name, add_special_tokens=False)
        s_ids = model.tokenizer.encode(s_name, add_special_tokens=False)
        if len(io_ids) != 1 or len(s_ids) != 1:
            continue
        io_id, s_id = io_ids[0], s_ids[0]

        bt = model.to_tokens(base_prompt)
        st = model.to_tokens(src_prompt)

        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            _, sc = model.run_with_cache(st, names_filter=hook_name)
            bl = model(bt)[0, -1]

        # For most subtasks, the "correct" answer for the base is IO,
        # and the intervention should flip it to S (or to the counterfactual IO).
        # We want to verify the model gets the base right first.
        if bl[io_id].item() > bl[s_id].item():
            pairs.append({
                "base_act": bc[hook_name][0, -1],
                "src_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_label": s_id,
                "base_label": io_id,
            })

    return pairs


# ===================================================================
# Grassmann distance between DAS subspaces
# ===================================================================


def grassmann_distance(Q1, Q2):
    cos_angles = torch.linalg.svdvals(Q1.T @ Q2)
    angles = torch.acos(cos_angles.clamp(-1, 1))
    return float(angles.norm()), [float(a) for a in angles.cpu().tolist()]


# ===================================================================
# Main experiment
# ===================================================================


@app.function(gpu="A100", timeout=21600, volumes={"/results": results_vol})
def run_transfer() -> dict:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    hook_name = "blocks.10.hook_resid_post"
    D = 768
    K_VALUES = [2, 4]

    log.info(f"[{utc_ts()}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_results = {"timestamp": utc_ts(), "subtasks": list(SHORT_NAMES.values())}

    # ---- Step 1: Load all subtask pairs ----
    log.info(f"[{utc_ts()}] Loading all 8 IOI subtask pairs...")
    subtask_pairs = {}
    for st in SUBTASK_NAMES:
        sn = SHORT_NAMES[st]
        try:
            pairs = load_mib_ioi_pairs(model, st, hook_name, DEVICE, n_examples=600)
            subtask_pairs[sn] = pairs
            log.info(f"[{utc_ts()}]   {sn}: {len(pairs)} valid pairs")
        except Exception as e:
            log.error(f"[{utc_ts()}]   {sn} failed: {e}")
            subtask_pairs[sn] = []

    all_results["pair_counts"] = {sn: len(p) for sn, p in subtask_pairs.items()}
    save_incremental(all_results, log)

    # ---- Step 2: Cross-subtask transfer matrix ----
    for k in K_VALUES:
        log.info(f"\n[{utc_ts()}] === k={k} Transfer Matrix ===")

        das_subspaces = {}
        das_matrix = {}
        vae_matrix = {}

        for train_st in subtask_pairs:
            train_pairs = subtask_pairs[train_st]
            if len(train_pairs) < 50:
                log.info(f"[{utc_ts()}]   Skip train={train_st} (only {len(train_pairs)} pairs)")
                continue

            log.info(f"[{utc_ts()}]   Training on {train_st} (k={k})...")

            # Split train/eval
            n_train = min(300, int(len(train_pairs) * 0.7))
            tr = train_pairs[:n_train]
            self_eval = train_pairs[n_train:]

            # Train DAS
            Q = train_das(model, tr, hook_name, D, k, DEVICE, n_steps=400)
            das_subspaces[train_st] = Q

            # Train VAE
            label_map = {}
            all_acts, all_labels = [], []
            for d in tr:
                for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
                    tid = d[lk]
                    if tid not in label_map:
                        label_map[tid] = len(label_map)
                    all_acts.append(d[rk])
                    all_labels.append(label_map[tid])
            act_t = torch.stack(all_acts)
            lab_t = torch.tensor(all_labels, device=DEVICE)
            n_classes = len(label_map)

            vae = build_vae(D, k, 16, 256, n_classes, DEVICE)
            vae = train_vae(vae, act_t, lab_t, DEVICE, n_epochs=500, alpha=10.0)

            # Eval on ALL subtasks
            das_row = {}
            vae_row = {}
            for eval_st in subtask_pairs:
                eval_pairs = subtask_pairs[eval_st]
                if len(eval_pairs) < 10:
                    continue

                # Use held-out portion if evaluating on own subtask
                if eval_st == train_st:
                    ep = self_eval[:200]
                else:
                    ep = eval_pairs[:200]

                das_res = eval_das_full(Q, model, ep, hook_name)
                vae_res = eval_vae_full(vae, model, ep, hook_name)

                das_row[eval_st] = das_res
                vae_row[eval_st] = vae_res

                marker = " *" if eval_st == train_st else ""
                log.info(f"[{utc_ts()}]     → {eval_st}{marker}: "
                         f"DAS={das_res['iia']:.3f} VAE={vae_res['iia']:.3f} "
                         f"KL={das_res['kl_mean']:.3f}/{vae_res['kl_mean']:.3f}")

            das_matrix[train_st] = das_row
            vae_matrix[train_st] = vae_row

            del vae
            torch.cuda.empty_cache()

        # Grassmann distances between all DAS subspaces
        grass_dists = {}
        st_list = sorted(das_subspaces.keys())
        for i, s1 in enumerate(st_list):
            for s2 in st_list[i + 1:]:
                dist, angles = grassmann_distance(das_subspaces[s1], das_subspaces[s2])
                grass_dists[f"{s1}_vs_{s2}"] = {"distance": dist, "angles": angles}
                log.info(f"[{utc_ts()}]   Grassmann {s1} vs {s2}: {dist:.3f}")

        all_results[f"k{k}"] = {
            "das_transfer_matrix": das_matrix,
            "vae_transfer_matrix": vae_matrix,
            "grassmann_distances": grass_dists,
        }
        save_incremental(all_results, log)

    # ---- Step 3: Summary ----
    all_results["elapsed_seconds"] = round(time.time() - t0, 1)
    save_incremental(all_results, log)

    log.info(f"\n{'=' * 120}")
    log.info("TRANSFER MATRIX SUMMARY")
    log.info(f"{'=' * 120}")

    for k in K_VALUES:
        kr = all_results.get(f"k{k}", {})
        das_m = kr.get("das_transfer_matrix", {})
        vae_m = kr.get("vae_transfer_matrix", {})
        if not das_m:
            continue

        log.info(f"\n--- k={k} DAS IIA ---")
        eval_sts = sorted(set().union(*[set(r.keys()) for r in das_m.values()]))
        header = f"{'train↓ eval→':>12}" + "".join(f"{s:>10}" for s in eval_sts)
        log.info(header)
        for train_st in sorted(das_m.keys()):
            row = das_m[train_st]
            vals = "".join(f"{row.get(e, {}).get('iia', 0):>10.3f}" for e in eval_sts)
            log.info(f"{train_st:>12}{vals}")

        log.info(f"\n--- k={k} VAE IIA ---")
        log.info(header)
        for train_st in sorted(vae_m.keys()):
            row = vae_m[train_st]
            vals = "".join(f"{row.get(e, {}).get('iia', 0):>10.3f}" for e in eval_sts)
            log.info(f"{train_st:>12}{vals}")

        # Transfer ratio: mean off-diagonal / mean diagonal
        das_diag, das_off = [], []
        for train_st in das_m:
            for eval_st in das_m[train_st]:
                iia = das_m[train_st][eval_st].get("iia", 0)
                if train_st == eval_st:
                    das_diag.append(iia)
                else:
                    das_off.append(iia)
        if das_diag and das_off:
            das_ratio = np.mean(das_off) / (np.mean(das_diag) + 1e-8)
            log.info(f"\nDAS transfer ratio (off-diag/diag): {das_ratio:.3f}")
            all_results[f"k{k}"]["das_transfer_ratio"] = float(das_ratio)

        vae_diag, vae_off = [], []
        for train_st in vae_m:
            for eval_st in vae_m[train_st]:
                iia = vae_m[train_st][eval_st].get("iia", 0)
                if train_st == eval_st:
                    vae_diag.append(iia)
                else:
                    vae_off.append(iia)
        if vae_diag and vae_off:
            vae_ratio = np.mean(vae_off) / (np.mean(vae_diag) + 1e-8)
            log.info(f"VAE transfer ratio (off-diag/diag): {vae_ratio:.3f}")
            all_results[f"k{k}"]["vae_transfer_ratio"] = float(vae_ratio)

    log.info(f"\nTotal: {all_results['elapsed_seconds']:.0f}s")
    log.info(f"\nInterpretation:")
    log.info(f"  Transfer ratio ≈ 1.0: variable transfers perfectly (same across subtasks)")
    log.info(f"  Transfer ratio << 1.0: overfit to training subtask's counterfactual structure")
    log.info(f"  High Grassmann distance + high transfer: genuinely different subspaces that")
    log.info(f"    both capture the right variable (nonlinear structure)")
    log.info(f"  High Grassmann distance + low transfer: different variables per subtask")

    save_incremental(all_results, log)
    return all_results


@app.local_entrypoint()
def main():
    handle = run_transfer.spawn()
    print(f"[{utc_ts()}] Spawned IOI subtask transfer matrix")
    print(f"  Handle: {handle.object_id}")
    print(f"  8 subtasks × 8 eval = 64 cells, for k=2 and k=4")
    print(f"  + Grassmann distances between all DAS subspaces")
    print(f"  + Continuous metrics (KL, JS, prob_diff)")
    print(f"  Incremental saves after each k value")
    print(f"  Results: {SAVE_DIR}/results.json")
