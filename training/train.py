import sys
import math
import time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.gpt import GPT
from data.dataset import load_dataset
from utils.config import GENERAL_CONFIG, ACTIVE_TIER
from utils.helpers import checkpoint_name
from utils.seed import set_seed

# hyperparameters and training config
TRAIN_CONFIG = {
    "batch_size":   8,
    "max_lr":       3e-4,
    "min_lr":       3e-5,       # max_lr / 10 — standard GPT-2 cosine schedule floor
    "warmup_steps": 2000,
    "max_steps":    100_000,
    "weight_decay": 0.1,
    "betas":        (0.9, 0.95),
    "eps":          1e-8,
    "grad_clip":    1.0,
    "val_every":    1000,       # run validation every N steps
    "ckpt_every":   5000,       # save a step checkpoint every N steps
    "ckpt_keep":    3,          # how many step checkpoints to keep on disk
    "use_amp":      True,       # fp16 mixed precision on GPU
    "overfit_test": False,      # True to run sanity overfit on 5 fixed batches
}


def get_lr(step: int) -> float:
    # learning increases linearly from 0  to max_lr during warmup, then uses cosine decay down to min_lr at max_steps.

    cfg = TRAIN_CONFIG
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    if step >= cfg["max_steps"]:
        return cfg["min_lr"]
    progress = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["max_lr"] - cfg["min_lr"]) * (1.0 + math.cos(math.pi * progress))


def make_param_groups(model: GPT) -> list[dict]:
    #biases, LayerNorm gamma/beta, and embedding weights are 1D — skip decay.
    #all weight matrices (Linear, Embedding lookup) are 2D+ — apply decay.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay,    "weight_decay": TRAIN_CONFIG["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def evaluate_by_position(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 500) -> list:
    model.eval()
    seq_len   = GENERAL_CONFIG["max_seq_len"]
    pos_losses = torch.zeros(seq_len, device=device)
    pos_counts = torch.zeros(seq_len, device=device)
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        B, T, V   = logits.shape
        per_token = F.cross_entropy(logits.view(B * T, V), y.view(B * T), reduction="none").view(B, T)
        pos_losses += per_token.sum(0)
        pos_counts += B
    model.train()
    return (pos_losses / pos_counts.clamp(min=1)).cpu().tolist()


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 200) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        if count >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=TRAIN_CONFIG["use_amp"], dtype=torch.float16):
            _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / count if count > 0 else float("inf")

#checkpointing
def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer,
                    step: int, val_loss: float,
                    ckpt_dir: Path, tag: str | None = None) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if tag:
        path = ckpt_dir / f"{tag}.pt"
    else:
        arch = "rope" if GENERAL_CONFIG.get("use_rope") else "base"
        path = ckpt_dir / checkpoint_name(ACTIVE_TIER, step, val_loss, arch=arch)
    torch.save({
        "step":         step,
        "val_loss":     val_loss,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "train_config": TRAIN_CONFIG,
        "gen_config":   GENERAL_CONFIG,
    }, path)
    return path

#be careful altering this, it deletes old checkpoints permanently and if removed will cause disk space to fill up with old checkpoints rapidly
def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    # Delete oldest step_*.pt files beyond the keep limit; never deletes best.pt or final.pt
    ckpts = sorted(ckpt_dir.glob("adria-*-step*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()


# ── Main training loop ─────────────────────────────────────────────────────────
def train() -> None:
    set_seed()
    device = torch.device(GENERAL_CONFIG["device"])

    #data
    train_ds = load_dataset("train")
    val_ds   = load_dataset("val")

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, drop_last=False,
    )

    #model
    model = GPT().to(device)

    if TRAIN_CONFIG["overfit_test"]:
        # Disable all dropout for the sanity overfit test
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        make_param_groups(model),
        lr=TRAIN_CONFIG["max_lr"],
        betas=TRAIN_CONFIG["betas"],
        eps=TRAIN_CONFIG["eps"],
    )

    # ── AMP scaler (no-op when use_amp=False) ─────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN_CONFIG["use_amp"])

    # ── Sanity overfit: 5 fixed batches cycled repeatedly ─────────────────────
    if TRAIN_CONFIG["overfit_test"]:
        # Zero weight decay — decay fights memorization and will mask real training bugs
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0

        # Flat LR for overfit — cosine decay fights memorization in real training (which we want)
        # but defeats the purpose of a memorization diagnostic
        overfit_lr = TRAIN_CONFIG["max_lr"]
        for group in optimizer.param_groups:
            group["lr"] = overfit_lr

        loader_iter     = iter(train_loader)
        overfit_batches = [next(loader_iter) for _ in range(5)]
        print("=== OVERFIT TEST: 5 fixed batches, dropout=0, wd=0, flat LR, target loss < 0.5 ===")
        model.train()
        for step in range(TRAIN_CONFIG["max_steps"]):
            x, y = overfit_batches[step % 5]
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0 or step < 10:
                print(f"  step {step:5d} | loss {loss.item():.4f} | ppl {math.exp(min(loss.item(), 20)):.2f} | gnorm {grad_norm.item():.3f}")
        print(f"=== overfit test done. final loss: {loss.item():.4f} ===")
        return

    # ── Normal training ────────────────────────────────────────────────────────
    ckpt_dir  = ROOT / "checkpoints"
    log_path  = ROOT / "figures" / "run_log.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "train_steps": [], "train_losses": [], "lrs": [], "gnorms": [], "tokens_seen": [],
        "val_steps": [], "val_losses": [],
        "layer_gnorm_steps": [], "layer_gnorms": [],
        "pos_losses": [],
    }
    best_loss    = float("inf")
    step         = 0
    tokens_seen  = 0

    model.train()
    pbar = tqdm(total=TRAIN_CONFIG["max_steps"], desc="Training", unit="step", dynamic_ncols=True)
    while step < TRAIN_CONFIG["max_steps"]:
        for x, y in train_loader:
            if step >= TRAIN_CONFIG["max_steps"]:
                break

            t0 = time.perf_counter()
            x, y = x.to(device), y.to(device)

            # Set LR before forward so warmup schedule is correct from step 0
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward + loss
            with torch.amp.autocast("cuda", enabled=TRAIN_CONFIG["use_amp"], dtype=torch.float16):
                _, loss = model(x, y)

            # Backward
            scaler.scale(loss).backward()

            # Unscale before clip so clip operates on true gradient magnitudes
            scaler.unscale_(optimizer)

            # Per-layer grad norms before clipping (sampled every 50 steps)
            if step % 50 == 0:
                layer_gnorms_now = []
                for block in model.blocks:
                    grads = [p.grad.detach() for p in block.parameters() if p.grad is not None]
                    block_gnorm = torch.stack([g.norm() for g in grads]).norm().item() if grads else 0.0
                    layer_gnorms_now.append(block_gnorm)
                log["layer_gnorm_steps"].append(step)
                log["layer_gnorms"].append(layer_gnorms_now)

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"]).item()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            tokens_seen += TRAIN_CONFIG["batch_size"] * GENERAL_CONFIG["max_seq_len"]
            log["train_steps"].append(step)
            log["train_losses"].append(loss.item())
            log["lrs"].append(lr)
            log["gnorms"].append(gnorm)
            log["tokens_seen"].append(tokens_seen)

            dt_ms = (time.perf_counter() - t0) * 1000
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{math.exp(min(loss.item(), 20)):.1f}", lr=f"{lr:.2e}", ms=f"{dt_ms:.0f}")

            # Validation — also saves log incrementally so a crash doesn't lose the curve
            if step > 0 and step % TRAIN_CONFIG["val_every"] == 0:
                val_loss = evaluate(model, val_loader, device)
                tqdm.write(f"  step {step:5d} | val loss {val_loss:.4f} | val ppl {math.exp(min(val_loss, 20)):.2f}")
                log["val_steps"].append(step)
                log["val_losses"].append(val_loss)
                torch.save(log, log_path)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, ckpt_dir, tag="best")
                    tqdm.write(f"  >> new best checkpoint saved ({val_loss:.4f})")

            # Periodic step checkpoint
            if step > 0 and step % TRAIN_CONFIG["ckpt_every"] == 0:
                save_checkpoint(model, optimizer, step, loss.item(), ckpt_dir)
                prune_checkpoints(ckpt_dir, keep=TRAIN_CONFIG["ckpt_keep"])

            step += 1

    pbar.close()

    # Final checkpoint and log save
    val_loss = evaluate(model, val_loader, device)
    log["val_steps"].append(step)
    log["val_losses"].append(val_loss)
    log["pos_losses"] = evaluate_by_position(model, val_loader, device)
    torch.save(log, log_path)
    save_checkpoint(model, optimizer, step, val_loss, ckpt_dir, tag="final")
    print(f"\nDone. Final val loss: {val_loss:.4f} | ppl: {math.exp(min(val_loss, 20)):.2f}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    train()
