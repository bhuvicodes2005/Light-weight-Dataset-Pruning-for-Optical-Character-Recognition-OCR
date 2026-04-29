#!/usr/bin/env python3
import argparse
import importlib
import inspect
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BASE_VOCAB = (
    '-'
    + ''.join(chr(i) for i in range(ord('a'), ord('z') + 1))
    + ''.join(chr(i) for i in range(ord('A'), ord('Z') + 1))
    + ''.join(str(i) for i in range(10))
)
EOS_TOKEN = '|'
VOCAB = BASE_VOCAB + EOS_TOKEN
EOS_IDX = len(VOCAB) - 1
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}


def configure_vocab_for_arch(arch: str) -> None:
    global VOCAB, EOS_IDX, CHAR2IDX
    if arch == 'CRNN':
        VOCAB = BASE_VOCAB
        EOS_IDX = None
    else:
        VOCAB = BASE_VOCAB + EOS_TOKEN
        EOS_IDX = len(VOCAB) - 1
    CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}


def num_classes_for_arch(arch: str) -> int:
    # CRNN uses CTC blank + 62 characters (no EOS class).
    return len(BASE_VOCAB) if arch == 'CRNN' else len(BASE_VOCAB) + 1


def normalize_text(s: str) -> str:
    return ''.join(ch for ch in str(s).strip() if ch in VOCAB)


def collect_images(image_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    return sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in exts)


def load_checkpoint_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if 'Weights only load failed' not in str(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def available_arches(models_dir: Path):
    arches = []
    for p in sorted(models_dir.glob('*.py')):
        if p.stem.startswith('_') or p.stem == '__init__':
            continue
        arches.append(p.stem)
    return arches


def choose_arch_interactive(arches):
    print('Select model architecture:')
    for i, arch in enumerate(arches, start=1):
        print(f'  {i}. {arch}')
    while True:
        choice = input('Enter number: ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(arches):
            return arches[int(choice) - 1]
        print('Invalid choice, try again.')


def build_model(arch: str, max_label_length: int, num_classes: int):
    module = importlib.import_module(f'models.{arch}')

    candidate_class_names = [arch, arch.upper(), 'Model']
    model_cls = None
    for name in candidate_class_names:
        if hasattr(module, name):
            model_cls = getattr(module, name)
            break

    if model_cls is None:
        raise ValueError(f'No model class found for arch={arch} in models/{arch}.py')

    if arch == 'VitSTR':
        raise NotImplementedError(
            'VitSTR in this repo uses a legacy Model(opt) interface and is not '
            'compatible with this generic script yet. Use CRNN/SVTR/VisionLAN.'
        )

    ctor = inspect.signature(model_cls.__init__)
    kwargs = {}
    if 'img_channel' in ctor.parameters:
        kwargs['img_channel'] = 1
    if 'img_height' in ctor.parameters:
        kwargs['img_height'] = 32
    if 'img_width' in ctor.parameters:
        kwargs['img_width'] = 128
    if 'num_class' in ctor.parameters:
        kwargs['num_class'] = num_classes
    if 'num_classes' in ctor.parameters:
        kwargs['num_classes'] = num_classes
    if 'max_label_length' in ctor.parameters:
        kwargs['max_label_length'] = max_label_length

    return model_cls(**kwargs)


def forward_model(model, images, max_label_length, label_lengths=None):
    attempts = [
        dict(max_length=max_label_length, label_lengths=label_lengths),
        dict(max_length=max_label_length),
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return model(images, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f'Could not forward model with known signatures: {last_error}')


def is_ctc_output(logits, batch_size):
    return logits.dim() == 3 and logits.shape[1] == batch_size


def ctc_decode(logits):
    indices = logits.argmax(2).permute(1, 0).cpu().numpy()
    preds = []
    for row in indices:
        chars = []
        prev = -1
        for idx in row:
            if idx != prev and idx > 0:
                chars.append(VOCAB[idx])
            prev = idx
        preds.append(''.join(chars))
    return preds


def seq_decode(logits):
    indices = logits.argmax(2).cpu().numpy()
    preds = []
    for row in indices:
        chars = []
        for idx in row:
            if EOS_IDX is not None and idx == EOS_IDX:
                break
            if idx > 0:
                chars.append(VOCAB[idx])
        preds.append(''.join(chars))
    return preds


def encode_ctc_labels(texts):
    encoded = []
    lengths = []
    for t in texts:
        ids = [CHAR2IDX[c] for c in normalize_text(t) if c in CHAR2IDX]
        encoded.extend(ids)
        lengths.append(len(ids))
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def load_iiit5k_ground_truths(dataset_root: Path):
    labels = {}
    for mat_name, split_key in [('testdata.mat', 'testdata'), ('traindata.mat', 'traindata')]:
        mat_path = dataset_root / mat_name
        if not mat_path.exists():
            continue
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if split_key not in mat:
            continue
        for item in mat[split_key].ravel():
            img_name = str(getattr(item, 'ImgName', '')).replace('\\', '/')
            gt = normalize_text(getattr(item, 'GroundTruth', ''))
            if not img_name or not gt:
                continue
            labels[img_name] = gt
            labels[Path(img_name).name] = gt
    return labels


class IIIT5KDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, transform=None):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')

        labels = load_iiit5k_ground_truths(dataset_root)
        for img_path in collect_images(split_dir):
            rel_from_root = img_path.relative_to(dataset_root).as_posix()
            label = labels.get(rel_from_root, labels.get(img_path.name, ''))
            self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f'No images found in {split_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


def encode_seq_targets(texts, max_len):
    targets = torch.zeros((len(texts), max_len), dtype=torch.long)
    for row, text in enumerate(texts):
        reserve_eos = 1 if EOS_IDX is not None else 0
        ids = [CHAR2IDX[c] for c in normalize_text(text) if c in CHAR2IDX][: max_len - reserve_eos]
        if EOS_IDX is not None:
            ids.append(EOS_IDX)
        targets[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return targets


def get_label_lengths(texts, max_label_length):
    return torch.tensor(
        [min(len([c for c in normalize_text(t) if c in CHAR2IDX]), max_label_length) for t in texts],
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
    return (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def compute_metrics(preds, gts):
    pairs = [(normalize_text(p), normalize_text(t)) for p, t in zip(preds, gts) if normalize_text(t)]
    if not pairs:
        return {'word_acc': 0.0, 'cer': 0.0, 'wer': 0.0, 'char_acc': 0.0, 'count': 0}

    word_acc = 100.0 * sum(p.lower() == t.lower() for p, t in pairs) / len(pairs)
    total_ed = sum(edit_distance(p, t) for p, t in pairs)
    total_ch = sum(len(t) for _, t in pairs)
    cer = 100.0 * total_ed / max(total_ch, 1)
    wer = 100.0 * sum(p.lower() != t.lower() for p, t in pairs) / len(pairs)
    return {
        'word_acc': word_acc,
        'cer': cer,
        'wer': wer,
        'char_acc': 100.0 - cer,
        'count': len(pairs),
    }


def set_visionlan_phase(model, epoch, lf_epochs, optimizer=None, la_lr_scale=0.1):
    if not hasattr(model, 'set_lf_phase'):
        return
    if epoch < lf_epochs:
        model.set_lf_phase(True)
    else:
        was_lf = bool(getattr(model, 'lf_phase', False))
        model.set_lf_phase(False)
        if was_lf and optimizer is not None:
            for pg in optimizer.param_groups:
                pg['lr'] *= la_lr_scale


def run_epoch(model, loader, device, max_label_length, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_seen = 0
    all_preds, all_gts = [], []

    for images, labels, _ in loader:
        images = images.to(device)
        batch_size = images.size(0)
        total_seen += batch_size
        label_lengths = get_label_lengths(labels, max_label_length).to(device)
        logits = forward_model(model, images, max_label_length, label_lengths if is_train else None)

        if is_ctc_output(logits, batch_size):
            log_probs = logits.log_softmax(2)
            targets_flat, target_lengths = encode_ctc_labels(labels)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            T = log_probs.size(0)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
            loss = torch.nn.functional.ctc_loss(
                log_probs, targets_flat, input_lengths, target_lengths,
                blank=0, reduction='mean', zero_infinity=True,
            )
            preds = ctc_decode(logits)
        else:
            targets = encode_seq_targets(labels, max_label_length).to(device)
            per_sample = per_sample_seq_loss(logits, targets, ignore_index=0)
            loss = per_sample.mean()
            preds = seq_decode(logits)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * batch_size
        all_preds.extend(preds)
        all_gts.extend(labels)

    avg_loss = total_loss / max(total_seen, 1)
    metrics = compute_metrics(all_preds, all_gts)
    metrics['loss'] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune OCR model on IIIT5K train split.')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='test')
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='', help='Initial weights (.pth.tar or state_dict).')
    parser.add_argument('--resume', type=str, default='', help='Resume full training state from checkpoint.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_label_length', type=int, default=25)
    parser.add_argument('--lf_epochs', type=int, default=2)
    parser.add_argument('--la_lr_scale', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./save/IIIT5K_finetune')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if not args.dataset_root:
        # Prefer dataset co-located with this script, then fall back to parent project dir.
        candidate_roots = [
            (script_dir / 'IIIT5K').resolve(),
            (script_dir.parent / 'IIIT5K').resolve(),
        ]
        for candidate in candidate_roots:
            if (candidate / 'traindata.mat').exists() and (candidate / 'testdata.mat').exists():
                args.dataset_root = str(candidate)
                break
        if not args.dataset_root:
            args.dataset_root = str(candidate_roots[0])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    sys.path.insert(0, str(script_dir))
    models_dir = script_dir / 'models'
    arches = available_arches(models_dir)
    if not arches:
        raise RuntimeError(f'No model files found in {models_dir}')

    if not args.arch:
        args.arch = choose_arch_interactive(arches)
    elif args.arch not in arches:
        raise ValueError(f'arch={args.arch} not found. Available: {arches}')

    configure_vocab_for_arch(args.arch)
    model_num_classes = num_classes_for_arch(args.arch)
    model = build_model(args.arch, args.max_label_length, model_num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f'dataset_root not found: {dataset_root}')

    if not (dataset_root / args.train_split).exists():
        raise FileNotFoundError(
            f"Train split directory not found: {dataset_root / args.train_split}. "
            f"Available entries: {[p.name for p in sorted(dataset_root.iterdir())]}"
        )

    if not (dataset_root / args.val_split).exists():
        raise FileNotFoundError(
            f"Val split directory not found: {dataset_root / args.val_split}. "
            f"Available entries: {[p.name for p in sorted(dataset_root.iterdir())]}"
        )

    train_ds = IIIT5KDataset(dataset_root, args.train_split, transform=transform)
    val_ds = IIIT5KDataset(dataset_root, args.val_split, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
    )

    os.makedirs(args.save_path, exist_ok=True)
    log_path = Path(args.save_path) / 'finetune_iiit5k.log'

    start_epoch = 0
    best_word_acc = -1.0

    if args.resume:
        resume_path = Path(args.resume).resolve()
        ckpt = load_checkpoint_compat(resume_path, map_location=device)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        if isinstance(ckpt, dict):
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = int(ckpt.get('epoch', 0))
            best_word_acc = float(ckpt.get('best_word_acc', best_word_acc))
            if ckpt.get('arch') and ckpt['arch'] != args.arch:
                print(f"Warning: checkpoint arch={ckpt['arch']} but --arch={args.arch}")
        print(f'Resumed training from: {resume_path} (starting epoch {start_epoch + 1})')
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print(f'Loaded initial checkpoint: {ckpt_path}')
    else:
        raise ValueError('Provide either --checkpoint (start fine-tune) or --resume (continue training).')

    with open(log_path, 'a', encoding='utf-8') as log_f:
        log_f.write(f'\n[{datetime.now().isoformat(timespec="seconds")}] Fine-tune start\n')
        log_f.write(f'Args: {vars(args)}\n')

        best_state = None
        last_state = None

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            if args.arch == 'VisionLAN':
                set_visionlan_phase(model, epoch, args.lf_epochs, optimizer, args.la_lr_scale)

            train_metrics = run_epoch(model, train_loader, device, args.max_label_length, optimizer=optimizer)
            scheduler.step()
            val_metrics = run_epoch(model, val_loader, device, args.max_label_length, optimizer=None)

            lr_now = optimizer.param_groups[0]['lr']
            msg = (
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"Arch {args.arch} | "
                f"LR {lr_now:.3e} | "
                f"Train Loss {train_metrics['loss']:.4f} WAcc {train_metrics['word_acc']:.2f}% CER {train_metrics['cer']:.2f}% | "
                f"Val Loss {val_metrics['loss']:.4f} WAcc {val_metrics['word_acc']:.2f}% CER {val_metrics['cer']:.2f}% WER {val_metrics['wer']:.2f}% | "
                f"Time {time.time() - epoch_start:.1f}s"
            )
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()

            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_word_acc': best_word_acc,
                'args': vars(args),
            }
            last_state = dict(state)

            if val_metrics['word_acc'] >= best_word_acc:
                best_word_acc = val_metrics['word_acc']
                state['best_word_acc'] = best_word_acc
                best_state = dict(state)

        if last_state is not None:
            last_state['best_word_acc'] = best_word_acc
            torch.save(last_state, Path(args.save_path) / 'last_ckpt.pth.tar')
        if best_state is not None:
            best_state['best_word_acc'] = best_word_acc
            torch.save(best_state, Path(args.save_path) / 'best_ckpt.pth.tar')
        log_f.write(f'Best val WordAcc: {best_word_acc:.2f}%\n')

    print(f'Finished fine-tuning. Best val WordAcc: {best_word_acc:.2f}%')
    print(f'Checkpoints saved in: {Path(args.save_path).resolve()}')


if __name__ == '__main__':
    main()
