# train_importance.py
# Purpose : Train TrOCR-Small for k proxy epochs on TRDG synthetic data,
#           record per-sample confidences / logits every epoch, compute ALL
#           importance scores (DUAL, DynUnc, EL2N, AUM, TDDS, Entropy,
#           Forgetting), log everything to Weights & Biases, and save
#           best / last checkpoints.
#
# Outputs :
#   <save_path>/npy/               – per-epoch _Conf.npy, _Logits.npy
#   <save_path>/scores/            – all score + mask .npy files
#   <save_path>/checkpoints/       – best_ckpt.pth, last_ckpt.pth
#   <save_path>/proxy_model/final/ – HF model + processor
#   <save_path>/log/               – plain-text training log
#   <save_path>/curves/            – loss_curve.png + confidence_curve.png
#
# Usage:
#   python train_importance.py \
#       --zip_path Data.zip --gt_train gt_train.txt --gt_val gt_val.txt \
#       --epochs 30 --window_size 10 --save_path ./importance_run \
#       --wandb_project trocr-importance
#
#   # Resume from checkpoint:
#   python train_importance.py ... --resume ./importance_run/checkpoints/last_ckpt.pth
#
#   # Disable W&B:
#   python train_importance.py ... --no_wandb

import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from numpy import linalg as LA
from tqdm import tqdm

# W&B – optional; script runs fine without it
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataloader import ZipOCRDataset, load_trocr_processor

########################################################################################################################
# Argument Parsing
########################################################################################################################

parser = argparse.ArgumentParser(
    description="TrOCR proxy training + importance scoring",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Data
parser.add_argument("--zip_path",          type=str,   default="Data.zip")
parser.add_argument("--gt_train",          type=str,   default="gt_train.txt")
parser.add_argument("--gt_val",            type=str,   default="gt_val.txt")
parser.add_argument("--max_target_length", type=int,   default=64)

# Model
parser.add_argument("--model_name", type=str, default="microsoft/trocr-small-stage1")

# Training
parser.add_argument("--epochs",       type=int,   default=30,   help="Proxy epochs k")
parser.add_argument("--train_batch",  type=int,   default=16)
parser.add_argument("--score_batch",  type=int,   default=32)
parser.add_argument("--lr",           type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--workers",      type=int,   default=0)
parser.add_argument("--grad_clip",    type=float, default=1.0,  help="Max grad norm (0 = off)")

# Sliding window for importance scores
parser.add_argument("--window_size",  type=int,   default=10,   help="Window J for DUAL / DynUnc / TDDS")

# Paths
parser.add_argument("--save_path",    type=str,   default="./importance_run")
parser.add_argument("--resume",       type=str,   default=None, help="Checkpoint path to resume from")

# W&B
parser.add_argument("--wandb_project",  type=str,  default="trocr-importance")
parser.add_argument("--wandb_run_name", type=str,  default=None,  help="Run display name (auto if None)")
parser.add_argument("--wandb_entity",   type=str,  default=None,  help="W&B team / entity (optional)")
parser.add_argument("--no_wandb",       action="store_true",       help="Disable W&B logging entirely")

# Misc
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--gpu",          type=str,   default="0")
parser.add_argument("--log_interval", type=int,   default=100,  help="Log step-loss to W&B every N steps")

args = parser.parse_args()

# ── Derived absolute paths ─────────────────────────────────────────────────────
args.zip_path  = os.path.abspath(args.zip_path)
args.gt_train  = os.path.abspath(args.gt_train)
args.gt_val    = os.path.abspath(args.gt_val)
args.save_path = os.path.abspath(args.save_path)

NPY_DIR   = os.path.join(args.save_path, "npy")
SCORE_DIR = os.path.join(args.save_path, "scores")
CKPT_DIR  = os.path.join(args.save_path, "checkpoints")
MODEL_DIR = os.path.join(args.save_path, "proxy_model", "final")
LOG_DIR   = os.path.join(args.save_path, "log")
CURVE_DIR = os.path.join(args.save_path, "curves")

for _d in (NPY_DIR, SCORE_DIR, CKPT_DIR, MODEL_DIR, LOG_DIR, CURVE_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
DEVICE   = f"cuda:{args.gpu}" if USE_CUDA else "cpu"
AMP_DEVICE_TYPE = "cuda" if USE_CUDA else "cpu"

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(args.seed)

########################################################################################################################
# Logging
########################################################################################################################

_log_file = open(os.path.join(LOG_DIR, f"seed_{args.seed}_log.txt"), "w")

def print_log(msg):
    s = str(msg)
    print(s)
    _log_file.write(s + "\n")
    _log_file.flush()

# ── W&B init ───────────────────────────────────────────────────────────────────
USE_WANDB = WANDB_AVAILABLE and not args.no_wandb

if USE_WANDB:
    run = wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = args.wandb_run_name,
        config  = vars(args),
        resume  = "allow",
    )
    print_log(f"W&B run  : {run.url}")
else:
    run = None
    if not WANDB_AVAILABLE:
        print_log("W&B not installed (pip install wandb). Running without logging.")

def wb_log(d, step=None):
    if USE_WANDB:
        wandb.log(d, step=step)

########################################################################################################################
# Model + Processor
########################################################################################################################

print_log("=" * 70)
print_log("  TrOCR Proxy Training + Importance Scoring")
print_log("=" * 70)
for _k, _v in vars(args).items():
    print_log(f"  {_k:<24}: {_v}")
print_log("=" * 70)

print_log("\nLoading processor...")
processor = load_trocr_processor(args.model_name)
print_log("Processor loaded.")

print_log("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
model.config.decoder_start_token_id = (
    processor.tokenizer.bos_token_id
    if processor.tokenizer.bos_token_id is not None
    else processor.tokenizer.cls_token_id
)
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size   = model.config.decoder.vocab_size

if model.config.decoder_start_token_id is None:
    raise ValueError("Could not determine decoder_start_token_id from tokenizer.")

model = model.to(DEVICE)
print_log(f"Model loaded → {DEVICE}")

if USE_WANDB:
    wandb.watch(model, log="gradients", log_freq=500)

########################################################################################################################
# Datasets & DataLoaders
########################################################################################################################

train_dataset = ZipOCRDataset(
    args.gt_train, args.zip_path, processor, max_target_length=args.max_target_length
)
val_dataset = ZipOCRDataset(
    args.gt_val, args.zip_path, processor, max_target_length=args.max_target_length
)

N = len(train_dataset)
print_log(f"Train : {N:,}  |  Val : {len(val_dataset):,}")

# shuffle=False is CRITICAL – keeps sample order stable for per-sample scoring
score_loader = DataLoader(
    train_dataset, batch_size=args.score_batch, shuffle=False,
    num_workers=args.workers, pin_memory=USE_CUDA,
)
train_loader = DataLoader(
    train_dataset, batch_size=args.train_batch, shuffle=True,
    num_workers=args.workers, pin_memory=USE_CUDA,
)
val_loader = DataLoader(
    val_dataset, batch_size=args.score_batch, shuffle=False,
    num_workers=args.workers, pin_memory=USE_CUDA,
)

########################################################################################################################
# Optimizer + Scaler
########################################################################################################################

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_CUDA)

########################################################################################################################
# Checkpoint helpers
########################################################################################################################

def save_checkpoint(tag, epoch, val_loss, best_val_loss):
    path = os.path.join(CKPT_DIR, f"{tag}.pth")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "val_loss":        val_loss,
        "best_val_loss":   best_val_loss,
        "args":            vars(args),
    }, path)
    print_log(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    start_ep   = ckpt["epoch"]
    best_vloss = ckpt.get("best_val_loss", float("inf"))
    print_log(f"  Resumed from {path}  (epoch {start_ep}, best_val={best_vloss:.4f})")
    return start_ep, best_vloss

# ── Resume ─────────────────────────────────────────────────────────────────────
start_epoch   = 0
best_val_loss = float("inf")

if args.resume:
    start_epoch, best_val_loss = load_checkpoint(args.resume)

########################################################################################################################
# Per-epoch measurement helpers
########################################################################################################################

def compute_val_loss(model, loader, device):
    model.eval()
    total, steps = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val loss", leave=False):
            pv      = batch["pixel_values"].to(device)
            lb      = batch["labels"].to(device)
            total  += model(pixel_values=pv, labels=lb).loss.item()
            steps  += 1
    model.train()
    return total / max(steps, 1)


def compute_confidences(model, loader, device):
    """
    q_t(x_i) = exp( mean_l  log P(y_l | y_<l, x_i) )
    Returns np.ndarray (N,) float32  ∈ [0, 1].
    """
    model.eval()
    confs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Scoring ", leave=False):
            pv  = batch["pixel_values"].to(device)
            lb  = batch["labels"].to(device)
            lg  = model(pixel_values=pv, labels=lb).logits        # (B, L, V)
            lp  = F.log_softmax(lg, dim=-1)

            gl  = lb.clone();  gl[gl == -100] = 0
            tlp = lp.gather(-1, gl.unsqueeze(-1)).squeeze(-1)     # (B, L)
            msk = lb.ne(-100)
            ln  = msk.sum(1).float().clamp(min=1)
            tlp = tlp * msk
            c   = (tlp.sum(1) / ln).exp().clamp(0.0, 1.0)
            confs.extend(c.cpu().numpy().astype(np.float32))
    model.train()
    return np.asarray(confs, dtype=np.float32)


def compute_logits_epoch(model, loader, device):
    """
    Mean-pool decoder logits over valid token positions → (N, V) float32.
    """
    model.eval()
    all_lg = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Logits  ", leave=False):
            pv   = batch["pixel_values"].to(device)
            lb   = batch["labels"].to(device)
            lg   = model(pixel_values=pv, labels=lb).logits       # (B, L, V)
            m    = lb.ne(-100).unsqueeze(-1).float()
            ln   = m.sum(1).clamp(min=1)
            pool = (lg * m).sum(1) / ln                            # (B, V)
            all_lg.extend(pool.cpu().numpy().astype(np.float32))
    model.train()
    return np.asarray(all_lg, dtype=np.float32)

########################################################################################################################
# Plotting helpers
########################################################################################################################

def _ax_style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_loss_curves(train_step_losses, train_epoch_losses, val_epoch_losses, save_dir):
    """
    Two-panel figure:
      Left  – step-level training loss with smoothed overlay
      Right – epoch-level train + val loss
    Also saves a standalone epoch-level PNG for quick viewing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # ── Left: step-level ──────────────────────────────────────────────────
    if train_step_losses:
        steps, vals = zip(*train_step_losses)
        axes[0].plot(steps, vals, color="#4C72B0", lw=0.6, alpha=0.5, label="Step loss")
        if len(vals) >= 30:
            k      = np.ones(30) / 30
            smooth = np.convolve(vals, k, mode="valid")
            axes[0].plot(steps[29:], smooth, color="#C44E52", lw=1.8, label="Smoothed (30-step MA)")
    _ax_style(axes[0], "Global step", "Cross-entropy loss", "Training loss (step level)")

    # ── Right: epoch-level ────────────────────────────────────────────────
    ep = list(range(1, len(train_epoch_losses) + 1))
    axes[1].plot(ep, train_epoch_losses, "o-", color="#4C72B0",
                 lw=1.8, ms=5, label="Train loss")
    if val_epoch_losses:
        vep = list(range(1, len(val_epoch_losses) + 1))
        axes[1].plot(vep, val_epoch_losses, "s--", color="#C44E52",
                     lw=1.8, ms=5, label="Val loss")
    _ax_style(axes[1], "Epoch", "Cross-entropy loss", "Epoch-level loss")

    plt.tight_layout()
    full_path = os.path.join(save_dir, "loss_curve_full.png")
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Standalone epoch-level
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(ep, train_epoch_losses, "o-", color="#4C72B0", lw=1.8, ms=5, label="Train")
    if val_epoch_losses:
        ax2.plot(vep, val_epoch_losses, "s--", color="#C44E52", lw=1.8, ms=5, label="Val")
    _ax_style(ax2, "Epoch", "Cross-entropy loss", "Proxy training — loss curve")
    plt.tight_layout()
    epoch_path = os.path.join(save_dir, "loss_curve_epoch.png")
    fig2.savefig(epoch_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return full_path, epoch_path


def plot_confidence_curve(mean_confs, std_confs, save_dir):
    epochs = list(range(1, len(mean_confs) + 1))
    means  = np.array(mean_confs)
    stds   = np.array(std_confs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, means, "o-", color="#55A868", lw=1.8, ms=5, label="Mean confidence")
    ax.fill_between(epochs, (means - stds).clip(0), (means + stds).clip(0, 1),
                    color="#55A868", alpha=0.18, label="±1 std")
    _ax_style(ax, "Epoch", "q_t (model confidence)", "Per-sample confidence trajectory")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(save_dir, "confidence_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_histogram(scores, name, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=80, color="#4C72B0", edgecolor="white", lw=0.3)
    ax.set_xlabel("Score");  ax.set_ylabel("Count")
    ax.set_title(f"{name} score distribution  (N={len(scores):,})", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name.lower()}_hist.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

########################################################################################################################
# Training Loop
########################################################################################################################

epoch_confidences  = []   # list[(N,)]   one per epoch
epoch_logits       = []   # list[(N,V)]  one per epoch
train_step_losses  = []   # list[(global_step, loss)]
train_epoch_losses = []   # list[float]  avg train loss per epoch
val_epoch_losses   = []   # list[float]  val loss per epoch
epoch_mean_confs   = []
epoch_std_confs    = []

global_step = 0

print_log(f"\n{'='*70}")
print_log(f"  Starting proxy training  (epochs {start_epoch+1} → {args.epochs})")
print_log(f"{'='*70}\n")

run_start = time.time()

for epoch in range(start_epoch, args.epochs):
    ep_start      = time.time()
    epoch_display = epoch + 1

    print_log(f"\n[Epoch {epoch_display:03d}/{args.epochs:03d}]")

    # ── Training pass ──────────────────────────────────────────────────────
    model.train()
    total_loss, steps = 0.0, 0

    for batch in tqdm(train_loader, desc=f"  Train epoch {epoch_display}"):
        pv = batch["pixel_values"].to(DEVICE, non_blocking=USE_CUDA)
        lb = batch["labels"].to(DEVICE, non_blocking=USE_CUDA)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_CUDA):
            loss = model(pixel_values=pv, labels=lb).loss

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        lv           = loss.item()
        total_loss  += lv
        steps       += 1
        global_step += 1
        train_step_losses.append((global_step, lv))

        # Step-level W&B logging
        if global_step % args.log_interval == 0:
            wb_log({"train/loss_step": lv}, step=global_step)

    avg_train = total_loss / max(steps, 1)
    train_epoch_losses.append(avg_train)

    # ── Validation loss ────────────────────────────────────────────────────
    val_loss = compute_val_loss(model, val_loader, DEVICE)
    val_epoch_losses.append(val_loss)
    elapsed  = time.time() - ep_start

    print_log(
        f"  Train loss : {avg_train:.4f}  |  "
        f"Val loss : {val_loss:.4f}  |  "
        f"Time : {elapsed:.1f}s"
    )

    # ── Epoch-level W&B metrics ────────────────────────────────────────────
    wb_log({
        "train/loss_epoch": avg_train,
        "val/loss_epoch":   val_loss,
        "epoch":            epoch_display,
        "lr":               optimizer.param_groups[0]["lr"],
    }, step=global_step)

    # ── Confidence scoring ─────────────────────────────────────────────────
    print_log(f"  Computing confidences ({N:,} samples)...")
    confs = compute_confidences(model, score_loader, DEVICE)

    if len(confs) != N:
        raise RuntimeError(f"Confidence count mismatch: {len(confs)} != {N}")

    epoch_confidences.append(confs)
    epoch_mean_confs.append(float(confs.mean()))
    epoch_std_confs.append(float(confs.std()))

    print_log(
        f"  Confidence  mean={confs.mean():.4f}  "
        f"std={confs.std():.4f}  "
        f"range=[{confs.min():.4f}, {confs.max():.4f}]"
    )
    wb_log({
        "confidence/mean": float(confs.mean()),
        "confidence/std":  float(confs.std()),
        "confidence/min":  float(confs.min()),
        "confidence/max":  float(confs.max()),
    }, step=global_step)

    # ── Logit collection ───────────────────────────────────────────────────
    print_log(f"  Collecting logits...")
    lgs = compute_logits_epoch(model, score_loader, DEVICE)

    if len(lgs) != N:
        raise RuntimeError(f"Logit count mismatch: {len(lgs)} != {N}")

    epoch_logits.append(lgs)

    # ── Save dynamics ──────────────────────────────────────────────────────
    np.save(os.path.join(NPY_DIR, f"{epoch}_Conf.npy"),   confs)
    np.save(os.path.join(NPY_DIR, f"{epoch}_Logits.npy"), lgs)

    # ── Checkpointing ──────────────────────────────────────────────────────
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        path = save_checkpoint("best_ckpt", epoch_display, val_loss, best_val_loss)
        print_log(f"  ★ New best val loss: {best_val_loss:.4f}")
        if USE_WANDB:
            wandb.save(path)

    # Last checkpoint (always overwrite)
    path = save_checkpoint("last_ckpt", epoch_display, val_loss, best_val_loss)
    if USE_WANDB:
        wandb.save(path)

    # ── Update curves every epoch ──────────────────────────────────────────
    full_p, epoch_p = plot_loss_curves(
        train_step_losses, train_epoch_losses, val_epoch_losses, CURVE_DIR
    )
    conf_p = plot_confidence_curve(epoch_mean_confs, epoch_std_confs, CURVE_DIR)

    if USE_WANDB:
        wandb.log({
            "charts/loss_curve_full":    wandb.Image(full_p),
            "charts/loss_curve_epoch":   wandb.Image(epoch_p),
            "charts/confidence_curve":   wandb.Image(conf_p),
        }, step=global_step)

print_log(f"\nTotal training time : {(time.time()-run_start)/60:.1f} min")

# ── Save proxy model ────────────────────────────────────────────────────────
model.save_pretrained(MODEL_DIR)
processor.save_pretrained(MODEL_DIR)
print_log(f"Proxy model saved → {MODEL_DIR}")

########################################################################################################################
# Importance Score Computation
########################################################################################################################

print_log(f"\n{'='*70}")
print_log("  Computing importance scores")
print_log(f"{'='*70}\n")

T = len(epoch_confidences)
J = args.window_size

if T == 0:
    raise RuntimeError(
        "No epoch dynamics were collected. Train for at least one epoch before computing importance scores."
    )

all_confs  = np.stack(epoch_confidences, axis=0)   # (T, N)
all_logits = np.stack(epoch_logits,      axis=0)   # (T, N, V)

conf_tensor   = torch.tensor(all_confs)
logits_tensor = torch.tensor(all_logits)
rearranged    = F.softmax(logits_tensor, dim=-1)   # (T, N, V)

# Pseudo-targets from mean logits (no single label in sequence model)
targets = torch.tensor(all_logits.mean(axis=0).argmax(axis=-1), dtype=torch.long)

# ── Shared helper: save arrays + histogram + log to W&B ────────────────────

def save_score(name, score, mask):
    np.save(os.path.join(SCORE_DIR, f"{name}_score.npy"), score)
    np.save(os.path.join(SCORE_DIR, f"{name}_mask.npy"),  mask)
    hist_path = plot_score_histogram(score, name.upper(), CURVE_DIR)
    print_log(f"  {name.upper():<16}  mean={score.mean():.6f}  "
              f"range=[{score.min():.6f}, {score.max():.6f}]")
    if USE_WANDB:
        wandb.log({
            f"scores/{name}_mean": float(score.mean()),
            f"scores/{name}_hist": wandb.Image(hist_path),
        })

# ── 1. DUAL ──────────────────────────────────────────────────────────────────
print_log("Computing DUAL...")

def compute_dual(conf_arr, T, J):
    eff_J  = min(J, T)
    M      = T - eff_J + 1
    scores = np.zeros(conf_arr.shape[1], dtype=np.float32)
    for k in range(M):
        w       = conf_arr[k:k + eff_J]
        scores += (w.std(axis=0) * (1.0 - w.mean(axis=0))).astype(np.float32)
    s = scores / M
    return s, s.argsort()

score, mask = compute_dual(all_confs, T, J)
save_score("dual", score, mask)

# ── 2. DynUnc ────────────────────────────────────────────────────────────────
print_log("Computing DynUnc...")

def compute_dynunc(conf_t, ws):
    wins = []
    for i in range(conf_t.size(0) - ws + 1):
        wins.append(conf_t[i:i + ws].std(dim=0) * 10)
    s = torch.stack(wins).mean(dim=0).numpy().astype(np.float32)
    return s, s.argsort()

if T >= J:
    score, mask = compute_dynunc(conf_tensor, J)
    save_score("dynunc", score, mask)
else:
    print_log(f"  DynUnc skipped (T={T} < J={J})")

# ── 3. EL2N ──────────────────────────────────────────────────────────────────
print_log("Computing EL2N...")

def compute_el2n(rea, tgts, ep_idx):
    snap = rea[ep_idx]
    oh   = F.one_hot(tgts, snap.shape[-1]).float()
    s    = torch.norm(oh - snap, p=2, dim=-1).numpy().astype(np.float32)
    return s, s.argsort()

mid_ep = T // 2
score, mask = compute_el2n(rearranged, targets, mid_ep)
save_score("el2n", score, mask)
print_log(f"  (snapshot at epoch {mid_ep + 1})")

# ── 4. AUM ───────────────────────────────────────────────────────────────────
print_log("Computing AUM...")

def compute_aum(rea, tgts):
    aum = torch.zeros(rea.shape[1])
    for t in range(rea.shape[0]):
        p   = rea[t].clone()
        tp  = p[range(p.size(0)), tgts]
        p[range(p.size(0)), tgts] = 0
        aum += tp - p.max(dim=1)[0]
    s = (-aum).numpy().astype(np.float32)   # negate: low AUM → harder
    return s, s.argsort()

score, mask = compute_aum(rearranged, targets)
save_score("aum", score, mask)

# ── 5. TDDS ──────────────────────────────────────────────────────────────────

def compute_tdds(T_l, J_l, rea):
    k, ma = 0, []
    while k < T_l - J_l + 1:
        kds = []
        window = rea[k:k + J_l]
        for j in range(J_l - 1):
            lr = torch.log(window[j + 1] + 1e-8) - torch.log(window[j] + 1e-8)
            kds.append(torch.abs(window[j + 1] * lr.nan_to_num(0.0)).sum(dim=-1))
        wa   = torch.stack(kds).mean(dim=0)
        norm = LA.norm(torch.stack([d - wa for d in kds]).cpu().numpy(), axis=0)
        ma.append(norm * 0.9 * (1 - 0.9) ** (T_l - J_l - k))
        k += 1
    s = np.squeeze(sum(np.array(ma), 0)).astype(np.float32)
    return s, s.argsort()

tdds_cfgs = []
if T >= 10: tdds_cfgs += [(min(T, 90), min(J, 10)), (min(T, 70), min(J, 10))]
if T >= 20: tdds_cfgs += [(min(T, 60), min(J, 20)), (min(T, 20), min(J, 10))]
if T >=  5: tdds_cfgs += [(min(T, 10), min(J,  5))]

for T_c, J_c in tdds_cfgs:
    if T_c < J_c or T_c > T:
        continue
    print_log(f"Computing TDDS (T={T_c}, J={J_c})...")
    score, mask = compute_tdds(T_c, J_c, rearranged[:T_c])
    save_score(f"tdds_{T_c}_{J_c}", score, mask)

# ── 6. Entropy ───────────────────────────────────────────────────────────────
print_log("Computing Entropy...")

def compute_entropy(rea):
    p = rea[-1]
    s = (-p * torch.log(p + 1e-10)).sum(dim=-1).numpy().astype(np.float32)
    return s, s.argsort()

score, mask = compute_entropy(rearranged)
save_score("entropy", score, mask)

# ── 7. Forgetting ────────────────────────────────────────────────────────────
print_log("Computing Forgetting...")

def compute_forgetting(rea, tgts):
    pred  = rea.argmax(dim=-1)                             # (T, N)
    tgt_x = tgts.unsqueeze(0).expand(T, -1)
    cor   = pred == tgt_x
    s     = (cor[:-1] > cor[1:]).sum(dim=0).numpy().astype(np.float32)
    return s, s.argsort()

score, mask = compute_forgetting(rearranged, targets)
save_score("forgetting", score, mask)

########################################################################################################################
# Summary + W&B finish
########################################################################################################################

print_log(f"\n{'='*70}")
print_log("  DUAL score distribution (thirds)")
print_log(f"{'='*70}")

dual_s       = np.load(os.path.join(SCORE_DIR, "dual_score.npy"))
p33, p66     = np.percentile(dual_s, 33), np.percentile(dual_s, 66)
low          = int((dual_s <  p33).sum())
mid_count    = int(((dual_s >= p33) & (dual_s < p66)).sum())
high         = int((dual_s >= p66).sum())

print_log(f"  Bottom 33%  (easy / prune) : {low:,}")
print_log(f"  Middle 33%                 : {mid_count:,}")
print_log(f"  Top    33%  (hard / keep)  : {high:,}")

if USE_WANDB:
    wandb.summary.update({
        "best_val_loss":        best_val_loss,
        "total_epochs_trained": T,
        "dual_easy_count":      low,
        "dual_hard_count":      high,
    })
    # Upload all score files as a W&B artifact
    artifact = wandb.Artifact("importance-scores", type="dataset")
    artifact.add_dir(SCORE_DIR)
    wandb.log_artifact(artifact)
    wandb.finish()

print_log(f"\nScore arrays   → {SCORE_DIR}/")
print_log(f"Epoch dynamics → {NPY_DIR}/")
print_log(f"Checkpoints    → {CKPT_DIR}/  (best_ckpt.pth, last_ckpt.pth)")
print_log(f"Proxy model    → {MODEL_DIR}/")
print_log(f"Curves         → {CURVE_DIR}/")
print_log("\nDone.")
_log_file.close()
