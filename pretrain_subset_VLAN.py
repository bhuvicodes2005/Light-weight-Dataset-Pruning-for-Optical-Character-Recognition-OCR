import os, sys, time, random, gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup, TrOCRProcessor
from torchmetrics.text import CharErrorRate

from data import load_data
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time


# ── Vocabulary ────────────────────────────────────────────────────────────────

VOCAB = (
    '-'
    + ''.join(chr(i) for i in range(ord('a'), ord('z') + 1))
    + ''.join(chr(i) for i in range(ord('A'), ord('Z') + 1))
    + ''.join(str(i) for i in range(10))
    + '|'           # EOS — MUST be the last entry
)
EOS_IDX     = len(VOCAB) - 1   # 63
CHAR2IDX    = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)       # 64


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(arch, num_classes):
    if arch == 'CRNN':
        from models.CRNN import CRNN
        return CRNN(img_channel=1, img_height=32, img_width=128, num_class=num_classes)
    elif arch == 'ViTSTR':
        from models.AI_ViTSTR import ViTSTR
        return ViTSTR(img_channel=1, img_height=32, img_width=128, num_class=num_classes)
    elif arch == 'PARSEQ':
        from models.PARSEQ import PARSeq
        return PARSeq(num_classes=num_classes, img_channel=1,
                      max_label_length=args.max_label_length)
    elif arch == 'VisionLAN':
        from models.VisionLAN import VisionLAN
        return VisionLAN(img_channel=1, img_height=32, img_width=128,
                         num_classes=num_classes,
                         max_label_length=args.max_label_length)
    elif arch == 'TrOCR':
        from transformers import VisionEncoderDecoderModel
        return VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    else:
        raise NotImplementedError(f"Model not supported: {arch}")


# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Train OCR models on a score-based subset of MJSynth',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dataset', type=str, default='MJSynth',
                    choices=['MJSynth', 'TRDG'])
parser.add_argument('--arch', type=str, default='CRNN',
                    choices=['CRNN', 'ViTSTR', 'PARSEQ', 'VisionLAN', 'TrOCR'])
parser.add_argument('--max_label_length', type=int, default=25)
parser.add_argument('--lf_epochs', type=int, default=5,
                    help='Epochs for Language-Free phase (VisionLAN only).')
parser.add_argument('--la_lr_scale', type=float, default=0.1,
                    help='LR multiplier at LF→LA boundary.')
parser.add_argument('--data_dir', type=str, default='./data/MJSynth')
parser.add_argument('--download', action='store_false', default=False)
parser.add_argument('--train_shuffle', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='')

# Optimisation
parser.add_argument('--epochs',        type=int,   default=12)   # <-- 12 epochs
parser.add_argument('--batch_size',    type=int,   default=128)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--momentum',      type=float, default=0.9)
parser.add_argument('--decay',         type=float, default=5e-4)

# Logging / checkpoints
parser.add_argument('--print_freq', type=int, default=12)
parser.add_argument('--save_path',  type=str, default='./save_subset')
parser.add_argument('--evaluate',   action='store_true', default=False)
parser.add_argument('--dynamics',   action='store_true', default=True)

# Hardware
parser.add_argument('--gpu',        type=str, default='0')
parser.add_argument('--workers',    type=int, default=4)
parser.add_argument('--manualSeed', type=int, default=42)

# Score-based subset selection
parser.add_argument('--mask-path', required=True, type=str,
                    help='Path to score/mask .npy file (ranked indices, boolean, or explicit index list).')
parser.add_argument('--subset_rate', default=0.3, type=float,
                    help='Fraction (<=1) or absolute count (>1) of samples to keep.')
parser.add_argument('--keep', type=str, default='highest',
                    choices=['lowest', 'highest'],
                    help='Which end of the ranked scores to keep.')

args = parser.parse_args()

gpu_id        = int(args.gpu)
args.use_cuda = torch.cuda.is_available() and gpu_id >= 0
args.device   = f'cuda:{gpu_id}' if args.use_cuda else 'cpu'

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = args.use_cuda


# ── Model-type helpers ────────────────────────────────────────────────────────

def is_ctc_model(arch):       return arch in ('CRNN', 'ViTSTR')
def is_seq2seq_model(arch):   return arch == 'TrOCR'
def is_parseq_model(arch):    return arch in ('PARSEQ', 'VisionLAN')
def is_visionlan_model(arch): return arch == 'VisionLAN'


# ── Label encoding ────────────────────────────────────────────────────────────

def encode_labels(texts, char2idx=CHAR2IDX):
    encoded, lengths = [], []
    for t in texts:
        ids = [char2idx[c] for c in t if c in char2idx]
        encoded.extend(ids)
        lengths.append(len(ids))
    return (torch.tensor(encoded, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))


def encode_seq_targets(texts, max_len, char2idx=CHAR2IDX):
    targets = torch.zeros((len(texts), max_len), dtype=torch.long)
    for row, text in enumerate(texts):
        ids = [char2idx[c] for c in text if c in char2idx][: max_len - 1]
        ids.append(EOS_IDX)
        targets[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return targets


def get_label_lengths(texts, char2idx=CHAR2IDX):
    return torch.tensor(
        [min(len([c for c in t if c in char2idx]), args.max_label_length)
         for t in texts],
        dtype=torch.long,
    )


def per_sample_seq_loss(logits, targets, ignore_index=0):
    bsz, seq_len, num_classes = logits.shape
    token_loss = F.cross_entropy(
        logits.reshape(-1, num_classes),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction='none',
    ).view(bsz, seq_len)
    valid_mask = targets.ne(ignore_index)
    return ((token_loss * valid_mask).sum(dim=1)
            / valid_mask.sum(dim=1).clamp_min(1))


def normalize_targets(texts, char2idx=CHAR2IDX):
    return [''.join(c for c in t if c in char2idx) for t in texts]


# ── Metrics ───────────────────────────────────────────────────────────────────

def make_cer_metric():
    return CharErrorRate()


def cer_percent(preds, targets):
    try:
        return CharErrorRate()(preds, targets).item() * 100.0
    except Exception as e:
        print(f"Warning: CharErrorRate failed ({e}), returning 0")
        return 0.0


def char_accuracy_from_cer(cer):
    return max(0.0, min(100.0, 100.0 - cer))


def word_accuracy(preds, targets):
    if not targets:
        return 0.0
    return sum(p.lower() == t.lower() for p, t in zip(preds, targets)) / len(targets) * 100.0


# ── Decode helpers ────────────────────────────────────────────────────────────

def ctc_decode(log_probs, vocab):
    indices = log_probs.argmax(2).permute(1, 0).cpu().numpy()
    results = []
    for row in indices:
        chars, prev = [], -1
        for idx in row:
            if idx != prev and idx != 0:
                chars.append(vocab[idx])
            prev = idx
        results.append(''.join(chars))
    return results


def greedy_decode(output, vocab, debug=False):
    eos_idx = len(vocab) - 1
    indices = output.argmax(2).cpu().numpy()
    results = []
    for row in indices:
        chars = []
        for idx in row:
            if idx == eos_idx:
                break
            if idx > 0:
                chars.append(vocab[idx])
        results.append(''.join(chars))
    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def recorder_to_state(recorder):
    return {
        'total_epoch':    recorder.total_epoch,
        'current_epoch':  recorder.current_epoch,
        'epoch_losses':   recorder.epoch_losses,
        'epoch_accuracy': recorder.epoch_accuracy,
    }


def load_recorder_from_state(recorder, state):
    if not isinstance(state, dict):
        return recorder
    keys = {'total_epoch', 'current_epoch', 'epoch_losses', 'epoch_accuracy'}
    if keys.issubset(state):
        recorder.total_epoch    = state['total_epoch']
        recorder.current_epoch  = state['current_epoch']
        recorder.epoch_losses   = state['epoch_losses']
        recorder.epoch_accuracy = state['epoch_accuracy']
    return recorder


def resize_recorder_total_epochs(recorder, total_epoch):
    """Resize recorder buffers when resumed run uses a different total epoch count."""
    if recorder.total_epoch == total_epoch:
        return recorder
    new_rec = RecorderMeter(total_epoch)
    copy_n  = min(recorder.total_epoch, total_epoch)
    new_rec.epoch_losses[:copy_n, :]   = recorder.epoch_losses[:copy_n, :]
    new_rec.epoch_accuracy[:copy_n, :] = recorder.epoch_accuracy[:copy_n, :]
    new_rec.current_epoch = min(recorder.current_epoch, total_epoch)
    return new_rec


def load_checkpoint_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        if 'Weights only load failed' not in str(e):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


# ── Logging / saving ──────────────────────────────────────────────────────────

def print_log(msg, log):
    print(msg)
    log.write(f'{msg}\n')
    log.flush()


def save_checkpoint(state, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, os.path.join(save_path, filename))


# ── Score-based subset selection ──────────────────────────────────────────────

def build_subset_indices(mask_path: str, subset_rate: float, keep: str) -> np.ndarray:
    """
    Supports three mask formats saved as .npy:
      1. Ranked mask  – 1-D array where values are a permutation of [0, N-1]
                        (e.g. dual_mask_T*.npy from importance_evaluation.py)
      2. Explicit indices – arbitrary list of integer sample indices to keep
      3. Boolean mask – True means include this sample
    """
    mask = np.load(mask_path).reshape(-1)

    if subset_rate <= 0:
        raise ValueError('subset_rate must be > 0')

    # ── Boolean mask ──────────────────────────────────────────────────────────
    if mask.dtype == np.bool_:
        keep_indices = np.where(mask)[0].astype(np.int64)
        if keep_indices.size == 0:
            raise ValueError('Boolean mask contains no True entries.')
        if subset_rate <= 1:
            subset_size = max(1, int(keep_indices.size * subset_rate))
        else:
            subset_size = min(int(subset_rate), keep_indices.size)
        chosen = keep_indices[:subset_size] if keep == 'lowest' else keep_indices[-subset_size:]
        return np.unique(chosen.astype(np.int64))

    # ── Integer mask (ranked or explicit) ─────────────────────────────────────
    mask_int    = np.asarray(mask, dtype=np.int64)
    unique_vals = np.unique(mask_int)
    is_ranked   = (
        mask_int.size > 0
        and unique_vals.size == mask_int.size
        and unique_vals.min() == 0
        and unique_vals.max() == mask_int.size - 1
    )

    source_len  = mask_int.size
    if subset_rate <= 1:
        subset_size = max(1, int(source_len * subset_rate))
    else:
        subset_size = min(int(subset_rate), source_len)

    chosen = mask_int[:subset_size] if keep == 'lowest' else mask_int[-subset_size:]

    if not is_ranked:
        chosen = np.sort(chosen)   # deterministic order for explicit index lists

    return np.unique(chosen.astype(np.int64))


def build_pruned_train_loader(args, train_loader_full):
    if not hasattr(train_loader_full, 'dataset'):
        raise TypeError('Subset training requires a map-style dataset with random access.')

    train_dataset = train_loader_full.dataset
    total_train   = len(train_dataset)

    subset_indices = build_subset_indices(args.mask_path, args.subset_rate, args.keep)
    subset_indices = subset_indices[(subset_indices >= 0) & (subset_indices < total_train)]

    if subset_indices.size == 0:
        raise ValueError('No valid subset indices remain after filtering against dataset length.')

    pruned_dataset = Subset(train_dataset, subset_indices.tolist())
    train_loader   = DataLoader(
        pruned_dataset,
        batch_size=args.batch_size,
        shuffle=args.train_shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    return train_loader, total_train, len(pruned_dataset)


# ── VisionLAN phase setter ────────────────────────────────────────────────────

def set_visionlan_phase(model, epoch, lf_epochs, log, optimizer=None):
    if not hasattr(model, 'set_lf_phase'):
        return
    if epoch < lf_epochs:
        model.set_lf_phase(True)
        print_log(f"  [VisionLAN] Phase: Language-Free (epoch {epoch+1}/{lf_epochs})", log)
    else:
        was_lf = model.lf_phase
        model.set_lf_phase(False)
        if was_lf and optimizer is not None:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.la_lr_scale
            new_lr = optimizer.param_groups[0]['lr']
            print_log(
                f"  [VisionLAN] Phase: Language-Aware — LR reduced to {new_lr:.2e} "
                f"(×{args.la_lr_scale})", log)
        else:
            print_log("  [VisionLAN] Phase: Language-Aware (masking enabled)", log)


# ── Train one epoch ───────────────────────────────────────────────────────────

def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log,
          trocr_processor=None):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    word_acc   = AverageMeter()
    char_acc   = AverageMeter()
    cer_metric = make_cer_metric()

    model.train()
    end = time.time()
    loss_epoch = index_epoch = y_epoch = None

    for t, batch in enumerate(train_loader):
        if t >= args.num_iter:
            break

        if len(batch) == 3:
            images, labels, sample_indices = batch
        else:
            images, labels = batch
            sample_indices = torch.arange(
                t * args.batch_size,
                t * args.batch_size + images.size(0),
            )

        data_time.update(time.time() - end)
        images         = images.to(args.device)
        sample_indices = torch.as_tensor(sample_indices, dtype=torch.long).reshape(-1)

        # ── Forward + Loss ────────────────────────────────────────────────────
        if is_ctc_model(args.arch):
            output = model(images).log_softmax(2)
            targets_flat, target_lengths = encode_labels(labels)
            targets_flat   = targets_flat.to(args.device)
            target_lengths = target_lengths.to(args.device)
            T, N, _ = output.shape
            input_lengths   = torch.full((N,), T, dtype=torch.long, device=args.device)
            per_sample_loss = F.ctc_loss(
                output, targets_flat, input_lengths, target_lengths,
                blank=0, reduction='none', zero_infinity=True,
            ) / target_lengths.float().to(args.device).clamp_min(1)
            loss = per_sample_loss.mean()

        elif is_seq2seq_model(args.arch):
            encoding = trocr_processor.tokenizer(
                list(labels), return_tensors='pt', padding=True).to(args.device)
            outputs = model(pixel_values=images, labels=encoding.input_ids)
            loss    = outputs.loss
            output  = outputs.logits
            per_sample_loss = loss.detach().expand(images.size(0))

        elif is_parseq_model(args.arch):
            if is_visionlan_model(args.arch):
                label_lengths = get_label_lengths(labels).to(args.device)
                output = model(images, max_length=args.max_label_length,
                               label_lengths=label_lengths)
            else:
                output = model(images, max_length=args.max_label_length)
            targets_seq     = encode_seq_targets(labels, args.max_label_length).to(args.device)
            per_sample_loss = per_sample_seq_loss(output, targets_seq, ignore_index=0)
            loss = per_sample_loss.mean()

        else:
            output = model(images)
            targets_flat, _ = encode_labels(labels)
            targets_flat = targets_flat.to(args.device)
            loss = criterion(output.view(-1, args.num_classes), targets_flat)
            per_sample_loss = loss.detach().expand(images.size(0))

        # ── Decode & metrics ──────────────────────────────────────────────────
        pred_texts   = (ctc_decode(output, VOCAB)
                        if is_ctc_model(args.arch)
                        else greedy_decode(output, VOCAB))
        target_texts = normalize_targets(labels)

        acc       = word_accuracy(pred_texts, target_texts)
        batch_cer = cer_percent(pred_texts, target_texts)
        cacc      = char_accuracy_from_cer(batch_cer)

        word_acc.update(acc, images.size(0))
        char_acc.update(cacc, images.size(0))
        losses.update(loss.item(), images.size(0))
        cer_metric.update(pred_texts, target_texts)

        # ── Dynamics bookkeeping ──────────────────────────────────────────────
        loss_batch  = per_sample_loss.detach().cpu().numpy().reshape(-1)
        index_batch = sample_indices.detach().cpu().numpy().reshape(-1)
        if t == 0:
            loss_epoch  = np.array(loss_batch)
            index_epoch = np.array(index_batch)
            y_epoch     = np.array(labels)
        else:
            loss_epoch  = np.concatenate([loss_epoch,  loss_batch])
            index_epoch = np.concatenate([index_epoch, index_batch])
            y_epoch     = np.concatenate([y_epoch, np.array(labels)])

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log(
                f'  Epoch: [{epoch+1:03d}][{t:03d}/{args.num_iter}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'WordAcc {word_acc.val:.3f} ({word_acc.avg:.3f})  '
                f'CharAcc {char_acc.val:.3f} ({char_acc.avg:.3f})  '
                f'CER(batch) {batch_cer:.3f}  ' + time_string(), log)

    epoch_cer = cer_metric.compute().item() * 100.0
    print_log(
        f'  **Train** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  '
        f'CER(epoch) {epoch_cer:.3f}  Loss {losses.avg:.4f}', log)

    return word_acc.avg, losses.avg, loss_epoch, index_epoch, y_epoch


# ── Validate ──────────────────────────────────────────────────────────────────

def validate(test_loader, args, model, criterion, log, trocr_processor=None):
    losses     = AverageMeter()
    word_acc   = AverageMeter()
    char_acc   = AverageMeter()
    cer_metric = make_cer_metric()
    debug_done = False

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images = images.to(args.device)

            if is_ctc_model(args.arch):
                output = model(images).log_softmax(2)
                targets_flat, target_lengths = encode_labels(labels)
                targets_flat   = targets_flat.to(args.device)
                target_lengths = target_lengths.to(args.device)
                T, N, _ = output.shape
                input_lengths = torch.full((N,), T, dtype=torch.long, device=args.device)
                loss = criterion(output, targets_flat, input_lengths, target_lengths)

            elif is_seq2seq_model(args.arch):
                encoding = trocr_processor.tokenizer(
                    list(labels), return_tensors='pt', padding=True).to(args.device)
                outputs = model(pixel_values=images, labels=encoding.input_ids)
                loss    = outputs.loss
                output  = outputs.logits

            elif is_parseq_model(args.arch):
                output      = model(images, max_length=args.max_label_length)
                targets_seq = encode_seq_targets(labels, args.max_label_length).to(args.device)
                loss = per_sample_seq_loss(output, targets_seq, ignore_index=0).mean()

            else:
                output = model(images)
                targets_flat, _ = encode_labels(labels)
                targets_flat = targets_flat.to(args.device)
                loss = criterion(output.view(-1, args.num_classes), targets_flat)

            pred_texts   = (ctc_decode(output, VOCAB)
                            if is_ctc_model(args.arch)
                            else greedy_decode(output, VOCAB, debug=not debug_done))
            target_texts = normalize_targets(labels)
            debug_done   = True

            acc       = word_accuracy(pred_texts, target_texts)
            batch_cer = cer_percent(pred_texts, target_texts)
            cacc      = char_accuracy_from_cer(batch_cer)

            word_acc.update(acc, images.size(0))
            char_acc.update(cacc, images.size(0))
            losses.update(loss.item(), images.size(0))
            cer_metric.update(pred_texts, target_texts)

    epoch_cer = cer_metric.compute().item() * 100.0
    print_log(
        f'  **Test** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  '
        f'CER(epoch) {epoch_cer:.3f}  Loss {losses.avg:.4f}', log)
    return word_acc.avg, losses.avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args.save_path = os.path.join(args.save_path, args.dataset, str(args.manualSeed))
    log_path = os.path.join(args.save_path, 'log')
    os.makedirs(log_path, exist_ok=True)
    log = open(os.path.join(log_path, f'seed_{args.manualSeed}_subset_log.txt'), 'w')

    print_log(f'save path : {args.save_path}', log)
    print_log({k: v for k, v in args._get_kwargs()}, log)
    print_log(f"Random Seed: {args.manualSeed}", log)
    print_log(f"Python: {sys.version.replace(chr(10), ' ')}", log)
    print_log(f"PyTorch: {torch.__version__}", log)
    print_log(f"cuDNN:  {torch.backends.cudnn.version()}", log)
    print_log(f"Dataset: {args.dataset}", log)
    print_log(f"Network: {args.arch}", log)
    print_log(f"Mask:    {args.mask_path}  |  rate={args.subset_rate}  |  keep={args.keep}", log)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader_full, test_loader = load_data(args)
    train_loader, full_train_n, pruned_train_n = build_pruned_train_loader(
        args, train_loader_full)

    print_log(f'Full train samples  : {full_train_n}', log)
    print_log(f'Pruned train samples: {pruned_train_n}', log)

    args.num_samples = pruned_train_n
    args.num_iter    = len(train_loader)
    args.num_classes = NUM_CLASSES

    # ── Model ─────────────────────────────────────────────────────────────────
    print_log(f"=> creating model '{args.arch}'", log)
    net = load_model(args.arch, args.num_classes)
    print_log(f"=> network:\n{net}", log)
    net = net.to(args.device)

    trocr_processor = None
    if is_seq2seq_model(args.arch):
        trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        print_log("=> TrOCRProcessor loaded.", log)

    # ── Loss ──────────────────────────────────────────────────────────────────
    if is_ctc_model(args.arch):
        criterion = torch.nn.CTCLoss(blank=0, reduction='mean',
                                     zero_infinity=True).to(args.device)
    elif is_parseq_model(args.arch) or is_seq2seq_model(args.arch):
        criterion = None
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(args.device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    if is_seq2seq_model(args.arch):
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate,
                                      weight_decay=args.decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=args.epochs * args.num_iter,
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
                                     weight_decay=args.decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * args.num_iter)

    recorder    = RecorderMeter(args.epochs)
    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        ckpt = load_checkpoint_compat(args.resume, map_location=args.device)
        net.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        if 'recorder_state' in ckpt:
            recorder = load_recorder_from_state(recorder, ckpt['recorder_state'])
        recorder    = resize_recorder_total_epochs(recorder, args.epochs)
        start_epoch = int(ckpt.get('epoch', 0))
        print_log(f"=> resumed from '{args.resume}' (next epoch: {start_epoch + 1})", log)

    if args.evaluate:
        validate(test_loader, args, net, criterion, log, trocr_processor)
        log.close()
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        if is_visionlan_model(args.arch):
            set_visionlan_phase(net, epoch, args.lf_epochs, log, optimizer=optimizer)

        try:
            current_lr = scheduler.get_last_lr()[0]
        except IndexError:
            current_lr = args.learning_rate

        need_h, need_m, need_s = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        print_log(
            f'\n==>>{time_string()} [Epoch={epoch+1:03d}/{args.epochs:03d}] '
            f'[Need: {need_h:02d}:{need_m:02d}:{need_s:02d}] '
            f'[lr={current_lr:.6f}] '
            f'[Best Acc={recorder.max_accuracy(False):.2f}]', log)

        epoch_start = time.time()

        train_acc, train_los, loss_epoch, index_epoch, y_epoch = train(
            train_loader, args, net, criterion, optimizer, scheduler,
            epoch, log, trocr_processor)

        val_acc, val_los = validate(
            test_loader, args, net, criterion, log, trocr_processor)

        if args.use_cuda:
            torch.cuda.empty_cache()
            gc.collect()

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        state = {
            'epoch':          epoch + 1,
            'arch':           args.arch,
            'state_dict':     net.state_dict(),
            'recorder_state': recorder_to_state(recorder),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
        }
        save_checkpoint(state, args.save_path, f'epoch_{epoch+1:03d}_subset_ckpt.pth.tar')
        if is_best:
            save_checkpoint(state, args.save_path, 'best_subset_ckpt.pth.tar')
        if epoch + 1 == args.epochs:
            save_checkpoint(state, args.save_path, 'last_subset_ckpt.pth.tar')

        epoch_time.update(time.time() - epoch_start)

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            recorder.plot_curve(
                os.path.join(args.save_path, f'{args.manualSeed}_subset_curve.png'))

        if args.dynamics and loss_epoch is not None:
            dyn_dir = os.path.join(args.save_path, 'npy')
            os.makedirs(dyn_dir, exist_ok=True)
            np.save(os.path.join(dyn_dir, f'{epoch}_Loss.npy'),  loss_epoch)
            np.save(os.path.join(dyn_dir, f'{epoch}_Index.npy'), index_epoch)
            print(f'Epoch {epoch} dynamics saved.')
            del loss_epoch, index_epoch, y_epoch

    log.close()


if __name__ == '__main__':
    main()