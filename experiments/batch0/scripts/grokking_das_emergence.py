"""Grokking DAS Emergence — When does the causal variable appear during grokking?

Trains a 1-layer transformer on modular addition (the canonical grokking setup
from Power et al. 2022 / Nanda et al. 2023), then at each training checkpoint:
1. Trains DAS to find a causal subspace for the modular addition task
2. Measures IIA — can we swap the causal variable between inputs?
3. Computes Nanda et al.'s Fourier progress measures for comparison

The key question: does the causal subspace emerge *gradually* during training,
or is there a *phase transition* where it suddenly becomes learnable?
If the DAS IIA curve shows a sharp jump at the same epoch where grokking occurs
(test loss drops), that's strong evidence that grokking = circuit formation
visible through causal abstraction.

Usage:
    python -u experiments/batch6_atlas/grokking_das_emergence.py \
        --device cuda --n-epochs 25000 --checkpoint-every 500

    # Skip training, load existing checkpoint file:
    python -u experiments/batch6_atlas/grokking_das_emergence.py \
        --device cuda --load-from experiments/results/grokking_checkpoints.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import einops
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598
KEY_FREQS = [17, 25, 32, 47]


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_grokking_data(device):
    """Build modular addition dataset: (a, b, =) -> (a+b) mod p."""
    a_vec = einops.repeat(torch.arange(P), "i -> (i j)", j=P)
    b_vec = einops.repeat(torch.arange(P), "j -> (i j)", i=P)
    eq_vec = einops.repeat(torch.tensor(P), " -> (i j)", i=P, j=P)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % P

    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(P * P)
    cutoff = int(P * P * FRAC_TRAIN)
    train_idx = indices[:cutoff]
    test_idx = indices[cutoff:]
    return dataset, labels, train_idx, test_idx


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def train_grokking_model(device, n_epochs=25000, checkpoint_every=500,
                          lr=1e-3, wd=1.0):
    """Train modular addition model with checkpoint saving."""
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
        init_weights=True, device=device, seed=999,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    dataset, labels, train_idx, test_idx = build_grokking_data(device)
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                   betas=(0.9, 0.98))

    train_losses, test_losses = [], []
    checkpoints, checkpoint_epochs = [], []

    for epoch in tqdm(range(n_epochs), desc="training grokking model"):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.inference_mode():
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if ((epoch + 1) % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            checkpoints.append(copy.deepcopy(model.state_dict()))
            logger.info(f"Epoch {epoch:6d}  train={train_loss.item():.6f}  "
                        f"test={test_loss.item():.6f}")

    return model, cfg, {
        "checkpoints": checkpoints,
        "checkpoint_epochs": checkpoint_epochs,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def build_das_pairs(dataset, labels, train_idx, n_pairs=200):
    """Build counterfactual pairs for DAS on modular addition.

    Each pair: two inputs (a1, b1, =) and (a2, b2, =) with different answers
    ((a1+b1) mod p ≠ (a2+b2) mod p).
    """
    train_data = dataset[train_idx]
    train_labels = labels[train_idx]
    pairs = []
    n = len(train_data)
    for i in range(0, n - 1, 2):
        if train_labels[i] != train_labels[i + 1]:
            pairs.append((i, i + 1))
        if len(pairs) >= n_pairs:
            break
    return pairs, train_data, train_labels


@torch.no_grad()
def cache_grokking_pairs(model, dataset, labels, train_idx, layer, device,
                          n_pairs=200):
    """Cache (base_tokens, base_acts, source_acts, source_target_id) for DAS."""
    pairs, train_data, train_labels = build_das_pairs(
        dataset, labels, train_idx, n_pairs,
    )
    hook_name = f"blocks.{layer}.hook_resid_post"
    cached = []
    for i, j in pairs:
        tokens_i = train_data[i].unsqueeze(0)
        tokens_j = train_data[j].unsqueeze(0)
        _, cache_i = model.run_with_cache(tokens_i, names_filter=[hook_name])
        _, cache_j = model.run_with_cache(tokens_j, names_filter=[hook_name])
        ba = cache_i[hook_name][0, -1, :]  # (d_model,) at last position
        sa = cache_j[hook_name][0, -1, :]
        si = train_labels[j].item()
        cached.append((tokens_i, ba, sa, si))
    return cached


def train_das_on_grokking(model, dataset, labels, train_idx, device,
                           k=8, n_steps=200, layer=0):
    """Train DAS to find causal subspace in the grokking model."""
    from factorization_circuits.pipeline.utils.factor_das_kernel import (
        VanillaQ, eval_iia, site_resid, make_hook,
    )

    site = site_resid(layer, model.cfg.d_model)
    cached = cache_grokking_pairs(model, dataset, labels, train_idx, layer,
                                   device, n_pairs=200)
    n_train = int(len(cached) * 0.75)
    cached_train = cached[:n_train]
    cached_eval = cached[n_train:]

    # Manual DAS training (simplified — avoid importing train_factor_das
    # which may assume different data format)
    param = VanillaQ(model.cfg.d_model, k, device=device)
    Q = param.get_Q()
    optimizer = torch.optim.Adam([param.A], lr=1e-3)

    for step in range(n_steps):
        optimizer.zero_grad()
        Q = param.get_Q()
        proj = Q @ Q.T

        loss = torch.tensor(0.0, device=device)
        batch_idx = torch.randint(0, len(cached_train), (min(16, len(cached_train)),))
        for idx in batch_idx:
            bt, ba, sa, si = cached_train[idx]
            iv = ba - ba @ proj + sa @ proj
            logits = model.run_with_hooks(
                bt, fwd_hooks=[(site.hook_name, make_hook(site, iv))]
            )[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            loss = loss - log_probs[si]
        loss = loss / len(batch_idx)
        loss.backward()
        optimizer.step()

    Q = param.get_Q().detach()
    iia = eval_iia(model, site, Q, cached_eval)
    return iia, Q


def fourier_progress_measures(model, dataset, labels, device):
    """Compute Nanda et al.'s Fourier cosine similarity progress measure."""
    logits = model(dataset)[:, -1, :]
    logits_cube = einops.rearrange(logits, "(a b) c -> a b c", a=P, b=P)

    cos_sims = {}
    for freq in KEY_FREQS:
        a = torch.arange(P)[:, None, None].to(device)
        b = torch.arange(P)[None, :, None].to(device)
        c = torch.arange(P)[None, None, :].to(device)
        predicted = torch.cos(freq * 2 * torch.pi / P * (a + b - c)).float()
        predicted = predicted / predicted.norm()
        coeff = (logits_cube * predicted).sum()
        cos_sim = coeff / logits_cube.norm()
        cos_sims[f"freq_{freq}"] = cos_sim.item()

    return cos_sims


def main():
    parser = argparse.ArgumentParser(description="Grokking DAS Emergence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=25000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--das-k", type=int, default=8)
    parser.add_argument("--das-steps", type=int, default=200)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    # Compatibility with modal_atlas.py
    parser.add_argument("--model", default="grokking")
    parser.add_argument("--task", default="modular_addition")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    das_k = args.k or args.das_k
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, labels, train_idx, test_idx = build_grokking_data(device)

    # Step 1: Train or load the grokking model
    if args.load_from and Path(args.load_from).exists():
        logger.info(f"[{utc_ts()}] Loading checkpoints from {args.load_from}")
        saved = torch.load(args.load_from, map_location=device)
        cfg = saved["config"]
        model = HookedTransformer(cfg)
        checkpoints = saved["checkpoints"]
        checkpoint_epochs = saved["checkpoint_epochs"]
        train_losses = saved["train_losses"]
        test_losses = saved["test_losses"]
    else:
        logger.info(f"[{utc_ts()}] Training grokking model ({args.n_epochs} epochs)...")
        model, cfg, training_data = train_grokking_model(
            device, n_epochs=args.n_epochs,
            checkpoint_every=args.checkpoint_every,
        )
        checkpoints = training_data["checkpoints"]
        checkpoint_epochs = training_data["checkpoint_epochs"]
        train_losses = training_data["train_losses"]
        test_losses = training_data["test_losses"]

        # Save checkpoints for reuse
        ckpt_file = output_dir / "grokking_checkpoints.pt"
        torch.save({
            "config": cfg,
            "checkpoints": checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_idx": train_idx.cpu(),
            "test_idx": test_idx.cpu(),
        }, ckpt_file)
        logger.info(f"  Saved {len(checkpoints)} checkpoints to {ckpt_file}")

    # Step 2: At each checkpoint, measure DAS IIA + Fourier progress
    logger.info(f"[{utc_ts()}] Evaluating {len(checkpoints)} checkpoints...")
    per_checkpoint = []

    for i, (epoch, sd) in enumerate(tqdm(
        zip(checkpoint_epochs, checkpoints), total=len(checkpoints),
        desc="evaluating checkpoints",
    )):
        model.load_state_dict(sd)
        model.eval()

        # Train loss / test loss at this checkpoint
        train_loss = train_losses[epoch] if epoch < len(train_losses) else None
        test_loss = test_losses[epoch] if epoch < len(test_losses) else None

        # DAS IIA
        try:
            iia, Q = train_das_on_grokking(
                model, dataset, labels, train_idx, device,
                k=das_k, n_steps=args.das_steps or 200,
            )
        except Exception as e:
            logger.warning(f"  Epoch {epoch}: DAS failed: {e}")
            iia = 0.0

        # Fourier progress measures
        with torch.no_grad():
            fourier = fourier_progress_measures(model, dataset, labels, device)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "das_iia": iia,
            "fourier_cosine_sims": fourier,
        }
        per_checkpoint.append(entry)
        logger.info(f"  Epoch {epoch:6d}  train={train_loss:.4f}  test={test_loss:.4f}  "
                    f"IIA={iia:.3f}  fourier_max={max(fourier.values()):.4f}")

    # Save results
    result = {
        "timestamp": utc_ts(),
        "task": "modular_addition",
        "p": P,
        "frac_train": FRAC_TRAIN,
        "n_epochs": args.n_epochs,
        "checkpoint_every": args.checkpoint_every,
        "das_k": das_k,
        "model_config": {
            "n_layers": 1, "n_heads": 4, "d_model": 128,
            "d_head": 32, "d_mlp": 512,
        },
        "n_checkpoints": len(checkpoints),
        "per_checkpoint": per_checkpoint,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"grokking_das_emergence_{ts}.jsonl"
    with open(out_file, "w") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info(f"[{utc_ts()}] Results written to {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Grokking DAS Emergence — modular addition (mod {P})")
    print(f"\n{'Epoch':>8s}  {'Train':>8s}  {'Test':>8s}  {'DAS IIA':>8s}  "
          f"{'Fourier':>8s}")
    print("-" * 50)
    for entry in per_checkpoint:
        fourier_max = max(entry["fourier_cosine_sims"].values())
        print(f"{entry['epoch']:8d}  {entry['train_loss']:8.4f}  "
              f"{entry['test_loss']:8.4f}  {entry['das_iia']:8.3f}  "
              f"{fourier_max:8.4f}")

    # Identify phase transition
    iia_values = [e["das_iia"] for e in per_checkpoint]
    max_iia = max(iia_values)
    max_iia_epoch = per_checkpoint[iia_values.index(max_iia)]["epoch"]

    # Find when IIA first exceeds 0.5
    iia_threshold_epoch = None
    for entry in per_checkpoint:
        if entry["das_iia"] > 0.5:
            iia_threshold_epoch = entry["epoch"]
            break

    # Find when test loss drops below 1.0
    grok_epoch = None
    for entry in per_checkpoint:
        if entry["test_loss"] is not None and entry["test_loss"] < 1.0:
            grok_epoch = entry["epoch"]
            break

    print(f"\nPhase transition analysis:")
    print(f"  Max DAS IIA: {max_iia:.3f} at epoch {max_iia_epoch}")
    if iia_threshold_epoch:
        print(f"  IIA > 0.5 first at epoch {iia_threshold_epoch}")
    if grok_epoch:
        print(f"  Test loss < 1.0 (grokking) at epoch {grok_epoch}")
    if iia_threshold_epoch and grok_epoch:
        gap = iia_threshold_epoch - grok_epoch
        if abs(gap) < 2000:
            print(f"  → DAS emergence coincides with grokking (gap={gap} epochs)")
        elif gap < 0:
            print(f"  → DAS emergence PRECEDES grokking by {-gap} epochs (causal variable forms before generalization)")
        else:
            print(f"  → DAS emergence FOLLOWS grokking by {gap} epochs")


if __name__ == "__main__":
    main()
