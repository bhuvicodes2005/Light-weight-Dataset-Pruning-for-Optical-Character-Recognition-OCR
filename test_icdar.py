#!/usr/bin/env python3
import argparse
import csv
import importlib
import inspect
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


BASE_CHARSET = (
    ''.join(chr(i) for i in range(ord('a'), ord('z') + 1))
    + ''.join(chr(i) for i in range(ord('A'), ord('Z') + 1))
    + ''.join(str(i) for i in range(10))
)
VOCAB_WITH_EOS = '-' + BASE_CHARSET + '|'
VOCAB_CTC = '-' + BASE_CHARSET

VOCAB = VOCAB_WITH_EOS
EOS_IDX = len(VOCAB) - 1


def set_vocab_for_arch(arch: str):
    global VOCAB, EOS_IDX
    if arch.strip().lower() == 'crnn':
        VOCAB = VOCAB_CTC
        EOS_IDX = None
    else:
        VOCAB = VOCAB_WITH_EOS
        EOS_IDX = len(VOCAB) - 1


def normalize_text(s: str) -> str:
    return ''.join(ch for ch in str(s).strip() if ch in VOCAB)


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
            'compatible with this generic script yet. Use CRNN/SVTR/PARSEQ/VisionLAN.'
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


def parse_icdar_gt(gt_path: Path):
    labels = {}
    with gt_path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            text = ','.join(row[1:]).strip() if len(row) > 1 else ''
            text = text.strip('"').strip()
            labels[name] = text
    return labels


def parse_icdar_coords(coords_path: Path):
    coords = {}
    with coords_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 9:
                continue
            name = parts[0]
            nums = []
            for p in parts[1:9]:
                try:
                    nums.append(int(float(p)))
                except ValueError:
                    nums.append(0)
            coords[name] = tuple(nums)
    return coords


class ICDARWordDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, transform=None):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')

        gt_path = split_dir / 'gt.txt'
        coords_path = split_dir / 'coords.txt'
        if not gt_path.exists() or not coords_path.exists():
            raise FileNotFoundError(f'Expected gt.txt and coords.txt in {split_dir}')

        labels = parse_icdar_gt(gt_path)
        coords = parse_icdar_coords(coords_path)

        for name, label in labels.items():
            img_path = split_dir / name
            if not img_path.exists():
                continue
            self.samples.append((img_path, label, coords.get(name, (0, 0, 0, 0, 0, 0, 0, 0))))

        if not self.samples:
            raise RuntimeError(f'No ICDAR samples found in {split_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, coord = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path), coord


def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR model on ICDAR split.')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--split', type=str, default='Test', choices=['Train', 'Test'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_label_length', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='icdar_predictions.csv')
    parser.add_argument('--log_file', type=str, default='icdar_metrics.log')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    if not args.dataset_root:
        args.dataset_root = str((script_dir / 'ICDAR').resolve())
    dataset_root = Path(args.dataset_root).resolve()

    models_dir = script_dir / 'models'
    arches = available_arches(models_dir)
    if not arches:
        raise RuntimeError(f'No model files found in {models_dir}')

    if not args.arch:
        args.arch = choose_arch_interactive(arches)
    elif args.arch not in arches:
        raise ValueError(f'arch={args.arch} not found. Available: {arches}')

    set_vocab_for_arch(args.arch)

    device = torch.device(args.device)
    model = build_model(args.arch, args.max_label_length, len(VOCAB)).to(device)
    model.eval()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (script_dir / ckpt_path).resolve()
    checkpoint = load_checkpoint_compat(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])

    dataset = ICDARWordDataset(dataset_root, args.split, transform=transform)
    print(f'ICDAR {args.split} words: {len(dataset)}')

    rows = []
    with torch.no_grad():
        for batch_idx in range(0, len(dataset), args.batch_size):
            inds = range(batch_idx, min(batch_idx + args.batch_size, len(dataset)))
            images = []
            batch_meta = []
            for i in inds:
                img, label, src, coord = dataset[i]
                images.append(img)
                batch_meta.append((label, src, coord))

            images = torch.stack(images).to(device)
            logits = forward_model(model, images, args.max_label_length, label_lengths=None)

            preds = ctc_decode(logits) if is_ctc_output(logits, images.size(0)) else seq_decode(logits)
            for (gt, src, coord), pred in zip(batch_meta, preds):
                coord_str = ','.join(map(str, coord))
                rows.append((src, coord_str, pred, gt))

            if (batch_idx // args.batch_size) % 10 == 0:
                print(f'  {min(batch_idx + args.batch_size, len(dataset))}/{len(dataset)}')

    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = script_dir / out_csv
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'coords', 'prediction', 'target'])
        writer.writerows(rows)

    labeled = [(normalize_text(p), normalize_text(t)) for _, _, p, t in rows if normalize_text(t)]

    word_acc = 100.0 * sum(p.lower() == t.lower() for p, t in labeled) / max(1, len(labeled))
    total_ed = sum(edit_distance(p, t) for p, t in labeled)
    total_ch = sum(len(t) for _, t in labeled)
    cer = 100.0 * total_ed / max(total_ch, 1)
    wer = 100.0 * sum(p.lower() != t.lower() for p, t in labeled) / max(1, len(labeled))

    log_file = Path(args.log_file)
    if not log_file.is_absolute():
        log_file = script_dir / log_file
    metric_lines = [
        f'[{datetime.now().isoformat(timespec="seconds")}] ICDAR Evaluation',
        f'arch            : {args.arch}',
        f'checkpoint      : {ckpt_path}',
        f'dataset         : {dataset_root}/{args.split}',
        f'total_words     : {len(rows)}',
        f'labeled_samples : {len(labeled)}',
        f'word_accuracy   : {word_acc:.2f}%',
        f'char_accuracy   : {100.0 - cer:.2f}%',
        f'cer             : {cer:.2f}%',
        f'wer             : {wer:.2f}%',
        f'predictions_csv : {out_csv}',
    ]

    print('\n===== Metrics =====')
    print(f'Arch            : {args.arch}')
    print(f'Labeled samples : {len(labeled)}')
    print(f'Word Accuracy   : {word_acc:.2f}%')
    print(f'Char Accuracy   : {100.0 - cer:.2f}%')
    print(f'CER             : {cer:.2f}%')
    print(f'WER             : {wer:.2f}%')

    with log_file.open('a', encoding='utf-8') as f:
        f.write('\n'.join(metric_lines) + '\n\n')

    print(f'\nPredictions -> {out_csv}')
    print(f'Metrics log -> {log_file}')


if __name__ == '__main__':
    main()
