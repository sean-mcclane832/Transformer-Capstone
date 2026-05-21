import sys
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.gpt import GPT
from data.dataset import load_dataset
from utils.config import GENERAL_CONFIG
from utils.seed import set_seed


# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size":   64,
    "max_lr":       3e-4,
    "min_lr":       3e-5,       # max_lr / 10
    "warmup_steps": 100,
    "max_steps":    5000,
    "weight_decay": 0.1,
    "betas":        (0.9, 0.95),
    "eps":          1e-8,
    "grad_clip":    1.0,
    "val_every":    200,        # run validation every N steps
    "ckpt_every":   500,        # save a step checkpoint every N steps
    "ckpt_keep":    3,          # how many step checkpoints to keep on disk
    "use_amp":      False,      # True once training on GPU
    "overfit_test": False,      # True to run sanity overfit on 5 fixed batches
}


# ── LR schedule: linear warmup → cosine decay ─────────────────────────────────
def get_lr(step: int) -> float:
    cfg = TRAIN_CONFIG
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    if step >= cfg["max_steps"]:
        return cfg["min_lr"]
    progress = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["max_lr"] - cfg["min_lr"]) * (1.0 + math.cos(math.pi * progress))


# ── Parameter groups: weight decay only on 2D+ tensors ────────────────────────
def make_param_groups(model: GPT) -> list[dict]:
    # Biases, LayerNorm gamma/beta, and embedding weights are 1D — skip decay.
    # All weight matrices (Linear, Embedding lookup) are 2D+ — apply decay.
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


# ── Validation pass ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=TRAIN_CONFIG["use_amp"], dtype=torch.float16):
            _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / count if count > 0 else float("inf")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer,
                    step: int, val_loss: float,
                    ckpt_dir: Path, tag: str | None = None) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = tag if tag else f"step_{step:06d}"
    path = ckpt_dir / f"{name}.pt"
    torch.save({
        "step":         step,
        "val_loss":     val_loss,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "train_config": TRAIN_CONFIG,
        "gen_config":   GENERAL_CONFIG,
    }, path)
    return path


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    # Delete oldest step_*.pt files beyond the keep limit; never deletes best.pt or final.pt
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()


# ── Main training loop ─────────────────────────────────────────────────────────
def train() -> None:
    set_seed()
    device = torch.device(GENERAL_CONFIG["device"])

    # ── Data ──────────────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────────
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
    scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_CONFIG["use_amp"])

    # ── Sanity overfit: 5 fixed batches cycled repeatedly ─────────────────────
    if TRAIN_CONFIG["overfit_test"]:
        loader_iter     = iter(train_loader)
        overfit_batches = [next(loader_iter) for _ in range(5)]
        print("=== OVERFIT TEST: 5 fixed batches, dropout=0, target loss < 0.5 ===")
        for step in range(TRAIN_CONFIG["max_steps"]):
            x, y = overfit_batches[step % 5]
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % 100 == 0 or step < 10:
                print(f"  step {step:5d} | loss {loss.item():.4f} | ppl {math.exp(min(loss.item(), 20)):.2f}")
        print(f"=== overfit test done. final loss: {loss.item():.4f} ===")
        return

    # ── Normal training ────────────────────────────────────────────────────────
    ckpt_dir  = ROOT / "checkpoints"
    best_loss = float("inf")
    step      = 0

    model.train()
    while step < TRAIN_CONFIG["max_steps"]:
        for x, y in train_loader:
            if step >= TRAIN_CONFIG["max_steps"]:
                break

            t0 = time.perf_counter()
            x, y = x.to(device), y.to(device)

            # Forward + loss
            with torch.cuda.amp.autocast(enabled=TRAIN_CONFIG["use_amp"], dtype=torch.float16):
                _, loss = model(x, y)

            # Backward
            scaler.scale(loss).backward()

            # Unscale before clip so clip operates on true gradient magnitudes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Apply LR schedule manually each step
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            dt_ms = (time.perf_counter() - t0) * 1000
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"ppl {math.exp(min(loss.item(), 20)):.2f} | "
                f"lr {lr:.2e} | {dt_ms:.1f}ms"
            )

            # Validation
            if step > 0 and step % TRAIN_CONFIG["val_every"] == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  → val loss {val_loss:.4f} | val ppl {math.exp(min(val_loss, 20)):.2f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, ckpt_dir, tag="best")
                    print(f"  → new best checkpoint saved ({val_loss:.4f})")

            # Periodic step checkpoint
            if step > 0 and step % TRAIN_CONFIG["ckpt_every"] == 0:
                save_checkpoint(model, optimizer, step, loss.item(), ckpt_dir)
                prune_checkpoints(ckpt_dir, keep=TRAIN_CONFIG["ckpt_keep"])

            step += 1

    # Final checkpoint after training completes
    val_loss = evaluate(model, val_loader, device)
    save_checkpoint(model, optimizer, step, val_loss, ckpt_dir, tag="final")
    print(f"\nDone. Final val loss: {val_loss:.4f} | ppl: {math.exp(min(val_loss, 20)):.2f}")


if __name__ == "__main__":
    train()
