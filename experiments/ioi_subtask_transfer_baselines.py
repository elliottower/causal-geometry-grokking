"""IOI subtask transfer baselines — 7 methods on 7 parallel pods.

Each pod: loads GPT-2, caches all 8 subtask pairs, runs one method, saves.

Methods:
  1. random       — random orthogonal subspace (5 seeds)
  2. per_das      — per-subtask DAS transfer matrix
  3. joint_das    — DAS trained on all subtasks pooled
  4. per_vae      — per-subtask VAE transfer matrix
  5. joint_vae    — VAE trained on all subtasks pooled
  6. per_nldas    — per-subtask NL-DAS transfer matrix
  7. joint_nldas  — NL-DAS trained on all subtasks pooled

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/ioi_subtask_transfer_baselines.py
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

app = modal.App("ioi-subtask-baselines-parallel", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

SAVE_DIR = "/results/grassmannian_atlas/ioi_subtask_baselines"

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

K = 4
D = 768
HOOK = "blocks.10.hook_resid_post"


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_result(method_name, data, log):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = f"{SAVE_DIR}/{method_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved {path}")


# ===================================================================
# Shared: load model + data
# ===================================================================


def load_model_and_pairs(device, log):
    log.info(f"[{utc_ts()}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    log.info(f"[{utc_ts()}] Caching subtask pairs...")
    subtask_pairs = {}
    for st in SUBTASK_NAMES:
        sn = SHORT_NAMES[st]
        pairs = _load_pairs(model, st, HOOK, device)
        subtask_pairs[sn] = pairs
        log.info(f"[{utc_ts()}]   {sn}: {len(pairs)} pairs")
    return model, subtask_pairs


def _load_pairs(model, subtask_name, hook_name, device, n_examples=600):
    from datasets import load_dataset
    ds = load_dataset("mib-bench/ioi", split="train")
    if n_examples < len(ds):
        ds = ds.select(range(n_examples))

    pairs = []
    for row in tqdm(ds, desc=f"load {SHORT_NAMES.get(subtask_name, '')}", leave=False):
        cf = row.get(subtask_name)
        if cf is None or cf.get("prompt") is None:
            continue
        meta = row["metadata"]
        io_ids = model.tokenizer.encode(f" {meta['indirect_object']}", add_special_tokens=False)
        s_ids = model.tokenizer.encode(f" {meta['subject']}", add_special_tokens=False)
        if len(io_ids) != 1 or len(s_ids) != 1:
            continue
        io_id, s_id = io_ids[0], s_ids[0]
        bt = model.to_tokens(row["prompt"])
        st = model.to_tokens(cf["prompt"])
        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            _, sc = model.run_with_cache(st, names_filter=hook_name)
            bl = model(bt)[0, -1]
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
# Model builders / trainers
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


def train_nldas(model, train_pairs, hook_name, d_model, k, device,
                n_steps=400, lr=1e-3, batch_size=16):
    featurizer = nn.Sequential(
        nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, d_model),
    ).to(device)
    R = nn.Parameter(torch.randn(d_model, k, device=device) * 0.02)
    inv_featurizer = nn.Sequential(
        nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, d_model),
    ).to(device)
    params = list(featurizer.parameters()) + [R] + list(inv_featurizer.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in tqdm(range(n_steps), desc=f"NL-DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(train_pairs, min(batch_size, len(train_pairs)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            f_base = featurizer(d["base_act"])
            f_src = featurizer(d["src_act"])
            f_iv = f_base - proj @ f_base + proj @ f_src
            iv = inv_featurizer(f_iv)
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

    featurizer.eval()
    inv_featurizer.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
    return featurizer, Q.detach(), inv_featurizer


# ===================================================================
# Eval helpers
# ===================================================================


def eval_das_iia(Q, model, pairs, hook_name):
    proj = Q @ Q.T
    correct = 0
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
            if logits.argmax().item() == d["src_label"]:
                correct += 1
    return correct / len(pairs) if pairs else 0.0


def eval_vae_iia(vae, model, pairs, hook_name):
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
            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == d["src_label"]:
                correct += 1
    return correct / len(pairs) if pairs else 0.0


def eval_nldas_iia(featurizer, Q, inv_featurizer, model, pairs, hook_name):
    proj = Q @ Q.T
    correct = 0
    with torch.inference_mode():
        for d in pairs:
            f_base = featurizer(d["base_act"])
            f_src = featurizer(d["src_act"])
            f_iv = f_base - proj @ f_base + proj @ f_src
            iv = inv_featurizer(f_iv)
            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new
            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == d["src_label"]:
                correct += 1
    return correct / len(pairs) if pairs else 0.0


def _pool_train_pairs(subtask_pairs, n_per=200):
    all_pairs = []
    for sn, pairs in subtask_pairs.items():
        all_pairs.extend(pairs[:n_per])
    random.shuffle(all_pairs)
    return all_pairs


def _make_label_map_and_tensors(pairs, device):
    label_map = {}
    all_acts, all_labels = [], []
    for d in pairs:
        for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
            tid = d[lk]
            if tid not in label_map:
                label_map[tid] = len(label_map)
            all_acts.append(d[rk])
            all_labels.append(label_map[tid])
    return label_map, torch.stack(all_acts), torch.tensor(all_labels, device=device)


# ===================================================================
# 7 separate pod functions
# ===================================================================


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_random() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("random")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    results = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 10:
            continue
        iias = []
        for seed in range(5):
            torch.manual_seed(seed)
            R = torch.randn(D, K, device="cuda")
            Q, _ = torch.linalg.qr(R)
            iias.append(eval_das_iia(Q, model, pairs[:200], HOOK))
        results[sn] = {"mean": float(np.mean(iias)), "std": float(np.std(iias))}
        log.info(f"[{utc_ts()}] random → {sn}: {np.mean(iias):.3f} ± {np.std(iias):.3f}")

    out = {"method": "random", "k": K, "results": results,
           "elapsed": round(time.time() - t0, 1)}
    save_result("random", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_per_das() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("per_das")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    matrix = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 50:
            continue
        n_tr = min(300, int(len(pairs) * 0.7))
        Q = train_das(model, pairs[:n_tr], HOOK, D, K, "cuda", n_steps=400)
        row = {}
        for eval_sn, eval_pairs in subtask_pairs.items():
            if len(eval_pairs) < 10:
                continue
            ep = eval_pairs[n_tr:n_tr + 200] if eval_sn == sn else eval_pairs[:200]
            row[eval_sn] = eval_das_iia(Q, model, ep, HOOK)
        matrix[sn] = row
        log.info(f"[{utc_ts()}] DAS({sn}): self={row.get(sn, 0):.3f}, "
                 f"mean_other={np.mean([v for es, v in row.items() if es != sn]):.3f}")

    out = {"method": "per_das", "k": K, "matrix": matrix,
           "elapsed": round(time.time() - t0, 1)}
    save_result("per_das", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_joint_das() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("joint_das")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    pooled = _pool_train_pairs(subtask_pairs, n_per=200)
    log.info(f"[{utc_ts()}] Pooled {len(pooled)} pairs")
    Q = train_das(model, pooled[:800], HOOK, D, K, "cuda", n_steps=600)

    results = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 10:
            continue
        iia = eval_das_iia(Q, model, pairs[200:400], HOOK)
        results[sn] = iia
        log.info(f"[{utc_ts()}] joint_DAS → {sn}: {iia:.3f}")

    out = {"method": "joint_das", "k": K, "results": results,
           "elapsed": round(time.time() - t0, 1)}
    save_result("joint_das", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_per_vae() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("per_vae")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    matrix = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 50:
            continue
        n_tr = min(300, int(len(pairs) * 0.7))
        label_map, act_t, lab_t = _make_label_map_and_tensors(pairs[:n_tr], "cuda")
        vae = build_vae(D, K, 16, 256, len(label_map), "cuda")
        vae = train_vae(vae, act_t, lab_t, "cuda", n_epochs=500, alpha=10.0)

        row = {}
        for eval_sn, eval_pairs in subtask_pairs.items():
            if len(eval_pairs) < 10:
                continue
            ep = eval_pairs[n_tr:n_tr + 200] if eval_sn == sn else eval_pairs[:200]
            row[eval_sn] = eval_vae_iia(vae, model, ep, HOOK)
        matrix[sn] = row
        log.info(f"[{utc_ts()}] VAE({sn}): self={row.get(sn, 0):.3f}, "
                 f"mean_other={np.mean([v for es, v in row.items() if es != sn]):.3f}")
        del vae
        torch.cuda.empty_cache()

    out = {"method": "per_vae", "k": K, "matrix": matrix,
           "elapsed": round(time.time() - t0, 1)}
    save_result("per_vae", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_joint_vae() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("joint_vae")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    all_train = []
    for sn, pairs in subtask_pairs.items():
        all_train.extend(pairs[:200])
    label_map, act_t, lab_t = _make_label_map_and_tensors(all_train, "cuda")
    log.info(f"[{utc_ts()}] Pooled {len(all_train)} pairs, {len(label_map)} classes")

    vae = build_vae(D, K, 16, 256, len(label_map), "cuda")
    vae = train_vae(vae, act_t, lab_t, "cuda", n_epochs=500, alpha=10.0)

    results = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 10:
            continue
        iia = eval_vae_iia(vae, model, pairs[200:400], HOOK)
        results[sn] = iia
        log.info(f"[{utc_ts()}] joint_VAE → {sn}: {iia:.3f}")

    out = {"method": "joint_vae", "k": K, "results": results,
           "elapsed": round(time.time() - t0, 1)}
    save_result("joint_vae", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_per_nldas() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("per_nldas")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    matrix = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 50:
            continue
        n_tr = min(300, int(len(pairs) * 0.7))
        feat, Q, inv = train_nldas(model, pairs[:n_tr], HOOK, D, K, "cuda", n_steps=400)

        row = {}
        for eval_sn, eval_pairs in subtask_pairs.items():
            if len(eval_pairs) < 10:
                continue
            ep = eval_pairs[n_tr:n_tr + 200] if eval_sn == sn else eval_pairs[:200]
            row[eval_sn] = eval_nldas_iia(feat, Q, inv, model, ep, HOOK)
        matrix[sn] = row
        log.info(f"[{utc_ts()}] NL-DAS({sn}): self={row.get(sn, 0):.3f}, "
                 f"mean_other={np.mean([v for es, v in row.items() if es != sn]):.3f}")
        del feat, inv
        torch.cuda.empty_cache()

    out = {"method": "per_nldas", "k": K, "matrix": matrix,
           "elapsed": round(time.time() - t0, 1)}
    save_result("per_nldas", out, log)
    return out


@app.function(gpu="A100", timeout=10800, volumes={"/results": results_vol})
def run_joint_nldas() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("joint_nldas")
    model, subtask_pairs = load_model_and_pairs("cuda", log)
    t0 = time.time()

    pooled = _pool_train_pairs(subtask_pairs, n_per=200)
    log.info(f"[{utc_ts()}] Pooled {len(pooled)} pairs")
    feat, Q, inv = train_nldas(model, pooled[:800], HOOK, D, K, "cuda", n_steps=600)

    results = {}
    for sn, pairs in subtask_pairs.items():
        if len(pairs) < 10:
            continue
        iia = eval_nldas_iia(feat, Q, inv, model, pairs[200:400], HOOK)
        results[sn] = iia
        log.info(f"[{utc_ts()}] joint_NL-DAS → {sn}: {iia:.3f}")

    out = {"method": "joint_nldas", "k": K, "results": results,
           "elapsed": round(time.time() - t0, 1)}
    save_result("joint_nldas", out, log)
    return out


# ===================================================================
# Entrypoint: spawn all 7 in parallel
# ===================================================================


@app.local_entrypoint()
def main():
    handles = []
    for fn, name in [
        (run_random, "random"),
        (run_per_das, "per-subtask DAS"),
        (run_joint_das, "joint DAS"),
        (run_per_vae, "per-subtask VAE"),
        (run_joint_vae, "joint VAE"),
        (run_per_nldas, "per-subtask NL-DAS"),
        (run_joint_nldas, "joint NL-DAS"),
    ]:
        h = fn.spawn()
        handles.append((name, h))
        print(f"  [{utc_ts()}] Spawned {name}: {h.object_id}")

    print(f"\n[{utc_ts()}] All 7 pods spawned in parallel")
    print(f"  k={K}, all 8 IOI subtasks")
    print(f"  Each saves to {SAVE_DIR}/<method>.json")
    print(f"  Download all: modal volume get fc-results {SAVE_DIR.lstrip('/')}/ .")
