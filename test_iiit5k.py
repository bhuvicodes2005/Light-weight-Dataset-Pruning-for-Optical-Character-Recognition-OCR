#!/usr/bin/env python3
import argparse
import csv
import importlib
import inspect
import os
import sys
import urllib.request
import tarfile
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat


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
    return ''.join(ch for ch in s.strip() if ch in VOCAB)


def greedy_decode(logits: torch.Tensor):
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


def forward_model(model, images, max_label_length):
    attempts = [
        dict(max_length=max_label_length, label_lengths=None),
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


def sequence_edit_distance(a, b) -> int:
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
    """Load a trusted checkpoint across PyTorch versions.

    PyTorch 2.6 defaults to weights_only=True, which breaks older checkpoints
    that store numpy objects or other non-tensor metadata. For this project we
    trust our own training checkpoints, so retry with weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if 'Weights only load failed' not in str(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def load_iiit5k_ground_truths(image_dir: Path):
    """Load official IIIT5K labels from the MATLAB split files.

    The dataset stores ground truth in testdata.mat / traindata.mat, with one
    entry per image containing ImgName and GroundTruth fields.
    """
    labels = {}
    search_roots = [image_dir, image_dir.parent, image_dir.parent.parent]
    mat_files = []
    for root in search_roots:
        if not root:
            continue
        for mat_name in ('testdata.mat', 'traindata.mat'):
            mat_path = root / mat_name
            if mat_path.exists() and mat_path not in mat_files:
                mat_files.append(mat_path)

    for mat_path in mat_files:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        split_key = 'testdata' if 'testdata' in mat else 'traindata'
        for item in mat[split_key].ravel():
            img_name = str(getattr(item, 'ImgName', '')).replace('\\', '/')
            gt = normalize_text(str(getattr(item, 'GroundTruth', '')))
            if not img_name or not gt:
                continue
            labels[img_name] = gt
            labels[Path(img_name).name] = gt

    return labels


def download_iiit5k(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz",
        "https://github.com/rbhatia2716/IIIT5K-Word_Localization/releases/download/v1.0/IIIT5K_words_v3.tar.gz",
        "https://raw.githubusercontent.com/rbhatia2716/IIIT5K-Word_Localization/master/IIIT5K_words_v3.tar.gz",
    ]
    tar_path = output_dir / "IIIT5K_words_v3.tar.gz"
    extract_dir = output_dir / "IIIT5K_words_v3"

    if extract_dir.exists() and len(list(extract_dir.glob('*/*.jpg'))) > 0:
        print(f'IIIT5K dataset already exists at: {extract_dir}')
        return extract_dir

    if not tar_path.exists():
        print('Downloading IIIT5K dataset (~45 MB)...')
        downloaded = False
        for url in urls:
            try:
                print(f'  Trying: {url}')
                urllib.request.urlretrieve(url, tar_path, reporthook=_download_hook)
                downloaded = True
                print(f'\nDownload complete: {tar_path}')
                break
            except Exception as e:
                print(f'  Failed: {e}')
                if tar_path.exists():
                    tar_path.unlink()
        if not downloaded:
            raise RuntimeError('Failed to download IIIT5K from all sources.')

    if not extract_dir.exists():
        print(f'Extracting to: {extract_dir}')
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        print('Extraction complete.')

    return extract_dir


def _download_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        print(f'\r  Download progress: {pct:.1f}%', end='', flush=True)


def collect_images(image_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    return sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in exts)


class HFDatasetWrapper(Dataset):
    """Wraps a HuggingFace dataset split. Expects 'image' and 'label'/'text'/'word' columns."""
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = sample.get('label', sample.get('text', sample.get('word', '')))
        return img, label, f'hf_sample_{idx}'


class IIIT5KDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        gt = {}
        for lf in [image_dir / 'labels.txt', image_dir / 'words.txt', image_dir / 'gt.txt']:
            if lf.exists():
                with lf.open('r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            gt[parts[0]] = ' '.join(parts[1:])
                break

        if not gt:
            gt = load_iiit5k_ground_truths(image_dir)

        for img_path in collect_images(image_dir):
            rel_name = img_path.name
            try:
                rel_name = img_path.relative_to(image_dir).as_posix()
            except ValueError:
                pass
            self.samples.append((img_path, gt.get(rel_name, gt.get(img_path.name, ''))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR model on IIIT5K dataset.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--hf_dataset', type=str, default='',
                        help='HuggingFace dataset repo, e.g. "nielsr/iiit5k-word"')
    parser.add_argument('--hf_split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_label_length', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='iit5k_predictions.csv')
    parser.add_argument('--log_file', type=str, default='iit5k_metrics.log')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    models_dir = script_dir / 'models'
    arches = available_arches(models_dir)
    if not arches:
        raise RuntimeError(f'No model files found in {models_dir}')

    if not args.arch:
        args.arch = choose_arch_interactive(arches)
    elif args.arch not in arches:
        raise ValueError(f'arch={args.arch} not found. Available: {arches}')

    set_vocab_for_arch(args.arch)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (script_dir / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    device = torch.device(args.device)

    model = build_model(args.arch, args.max_label_length, len(VOCAB)).to(device)
    model.eval()

    checkpoint = load_checkpoint_compat(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])

    # Dataset loading
    if args.download or args.hf_dataset:
        if args.hf_dataset:
            print(
                f'--hf_dataset was provided ({args.hf_dataset}), but this script '
                'now uses the direct IIIT5K download path to avoid HuggingFace '
                'network issues.'
            )
        image_dir = download_iiit5k(script_dir / 'IIIT5K_data')
        if (image_dir / 'images').exists():
            image_dir = image_dir / 'images'
        dataset = IIIT5KDataset(image_dir, transform=transform)
    elif args.image_dir:
        image_dir = Path(args.image_dir).resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f'image_dir not found: {image_dir}')
        dataset = IIIT5KDataset(image_dir, transform=transform)
    else:
        raise ValueError('Provide either --download or --image_dir')

    print(f'Dataset size: {len(dataset)} samples')

    # Single pass inference (test only, no training)
    rows = []
    with torch.no_grad():
        for batch_idx in range(0, len(dataset), args.batch_size):
            indices = range(batch_idx, min(batch_idx + args.batch_size, len(dataset)))
            imgs, batch_data = [], []
            for idx in indices:
                img, label, img_path = dataset[idx]
                imgs.append(img)
                batch_data.append((img_path, label))
            images = torch.stack(imgs).to(device)
            logits = forward_model(model, images, args.max_label_length)
            preds = ctc_decode(logits) if is_ctc_output(logits, images.size(0)) else greedy_decode(logits)
            for (img_path, gt_label), pred in zip(batch_data, preds):
                rows.append((img_path, Path(str(img_path)).name, pred, normalize_text(gt_label)))
            if (batch_idx // args.batch_size) % 10 == 0:
                print(f'  {min(batch_idx + args.batch_size, len(dataset))}/{len(dataset)}')

    # Save predictions CSV
    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = script_dir / out_csv
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['abs_path', 'basename', 'prediction', 'target'])
        writer.writerows(rows)

    # Compute and log metrics
    labeled = [(normalize_text(pred), gt) for _, _, pred, gt in rows if gt]
    log_file = Path(args.log_file)
    if not log_file.is_absolute():
        log_file = script_dir / log_file

    metric_lines = [
        f'[{datetime.now().isoformat(timespec="seconds")}] IIIT5K Evaluation',
        f'arch            : {args.arch}',
        f'checkpoint      : {ckpt_path}',
        f'dataset         : {image_dir}',
        f'total_images    : {len(rows)}',
        f'labeled_samples : {len(labeled)}',
        f'predictions_csv : {out_csv}',
    ]

    print('\n===== Metrics =====')
    if labeled:
        word_acc  = 100.0 * sum(p.lower() == t.lower() for p, t in labeled) / len(labeled)
        total_ed  = sum(edit_distance(p, t) for p, t in labeled)
        total_ch  = sum(len(t) for _, t in labeled)
        cer       = 100.0 * total_ed / max(total_ch, 1)
        wer       = 100.0 * sum(p.lower() != t.lower() for p, t in labeled) / len(labeled)

        print(f'Labeled samples : {len(labeled)}')
        print(f'Word Accuracy   : {word_acc:.2f}%')
        print(f'Char Accuracy   : {100.0 - cer:.2f}%')
        print(f'CER             : {cer:.2f}%')
        print(f'WER             : {wer:.2f}%')

        metric_lines += [
            f'word_accuracy   : {word_acc:.2f}%',
            f'char_accuracy   : {100.0 - cer:.2f}%',
            f'cer             : {cer:.2f}%',
            f'wer             : {wer:.2f}%',
        ]
    else:
        print('No labels found — only predictions written.')
        metric_lines.append('metrics: N/A (no ground truth labels)')

    with log_file.open('a', encoding='utf-8') as f:
        f.write('\n'.join(metric_lines) + '\n\n')

    print(f'\nPredictions → {out_csv}')
    print(f'Metrics log → {log_file}')


if __name__ == '__main__':
    main()