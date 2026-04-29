import os
import sys
import shutil
import time
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchmetrics.text import CharErrorRate

from data import load_data
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time


def load_model(arch, num_classes):
    if arch == 'CRNN':
        from models.CRNN import CRNN
        return CRNN(img_channel=1, img_height=32, img_width=128, num_class=num_classes)
    raise NotImplementedError(f"Model not supported: {arch}")


VOCAB = '-' + ''.join(
    [chr(i) for i in range(ord('a'), ord('z') + 1)] +
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    [str(i) for i in range(10)]
)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)


def encode_labels(texts, char2idx=CHAR2IDX):
    encoded, lengths = [], []
    for t in texts:
        ids = [char2idx[c] for c in t if c in char2idx]
        encoded.extend(ids)
        lengths.append(len(ids))
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def normalize_targets(texts, char2idx=CHAR2IDX):
    return [''.join([c for c in t if c in char2idx]) for t in texts]


def cer_percent(preds: list, targets: list) -> float:
    try:
        cer_metric = CharErrorRate()
        cer_score = cer_metric(preds, targets)
        return cer_score.item() * 100.0
    except Exception as e:
        print(f"Warning: CharErrorRate failed ({e}), returning 0")
        return 0.0


def _levenshtein_tokens(a_tokens, b_tokens) -> int:
    m = len(a_tokens)
    n = len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def wer_percent(preds: list, targets: list) -> float:
    total_edits = 0
    total_ref_words = 0
    for pred, target in zip(preds, targets):
        ref_words = str(target).split()
        hyp_words = str(pred).split()
        total_edits += _levenshtein_tokens(ref_words, hyp_words)
        total_ref_words += max(1, len(ref_words))
    if total_ref_words == 0:
        return 0.0
    return 100.0 * total_edits / total_ref_words


def ctc_decode(log_probs: torch.Tensor, vocab: str) -> list:
    indices = log_probs.argmax(2).permute(1, 0).cpu().numpy()
    results = []
    for row in indices:
        chars = []
        prev = -1
        for idx in row:
            if idx != prev and idx != 0:
                chars.append(vocab[idx])
            prev = idx
        results.append(''.join(chars))
    return results


def word_accuracy(preds: list, targets) -> float:
    correct = sum(p.lower() == t.lower() for p, t in zip(preds, targets))
    return correct / len(targets) * 100.0 if len(targets) > 0 else 0.0


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def recorder_to_state(recorder):
    return {
        'total_epoch': recorder.total_epoch,
        'current_epoch': recorder.current_epoch,
        'epoch_losses': recorder.epoch_losses,
        'epoch_accuracy': recorder.epoch_accuracy,
    }


def load_recorder_from_state(recorder, recorder_state):
    if not isinstance(recorder_state, dict):
        return recorder
    required = {'total_epoch', 'current_epoch', 'epoch_losses', 'epoch_accuracy'}
    if required.issubset(recorder_state.keys()):
        recorder.total_epoch = recorder_state['total_epoch']
        recorder.current_epoch = recorder_state['current_epoch']
        recorder.epoch_losses = recorder_state['epoch_losses']
        recorder.epoch_accuracy = recorder_state['epoch_accuracy']
    return recorder


def resize_recorder_total_epochs(recorder, total_epoch):
    """Resize recorder buffers when resumed run uses a different total epoch count."""
    if recorder.total_epoch == total_epoch:
        return recorder

    new_recorder = RecorderMeter(total_epoch)
    copy_epochs = min(recorder.total_epoch, total_epoch)

    new_recorder.epoch_losses[:copy_epochs, :] = recorder.epoch_losses[:copy_epochs, :]
    new_recorder.epoch_accuracy[:copy_epochs, :] = recorder.epoch_accuracy[:copy_epochs, :]
    new_recorder.current_epoch = min(recorder.current_epoch, total_epoch)

    return new_recorder


def load_checkpoint_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        if 'Weights only load failed' not in str(e):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def build_subset_indices(mask_path: str, subset_rate: float, keep: str) -> np.ndarray:
    mask = np.load(mask_path).reshape(-1)

    if subset_rate <= 0:
        raise ValueError('subset_rate must be > 0')

    if subset_rate <= 1:
        subset_size = max(1, int(len(mask) * subset_rate))
    else:
        subset_size = min(int(subset_rate), len(mask))

    if keep == 'lowest':
        chosen = mask[:subset_size]
    else:
        chosen = mask[-subset_size:]

    return np.unique(np.asarray(chosen, dtype=np.int64))


def build_pruned_train_loader(args, train_loader_full):
    if not hasattr(train_loader_full, 'dataset'):
        raise TypeError('Subset training requires map-style dataset with random access.')

    train_dataset = train_loader_full.dataset
    total_train = len(train_dataset)

    subset_indices = build_subset_indices(args.mask_path, args.subset_rate, args.keep)
    subset_indices = subset_indices[(subset_indices >= 0) & (subset_indices < total_train)]
    if subset_indices.size == 0:
        raise ValueError('No valid subset indices remain after filtering with dataset length.')

    pruned_dataset = Subset(train_dataset, subset_indices.tolist())
    train_loader = DataLoader(
        pruned_dataset,
        batch_size=args.batch_size,
        shuffle=args.train_shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    return train_loader, total_train, len(pruned_dataset)


def plot_metric_curves(history: dict, save_path: str):
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history['train_word_acc'], label='train-word-acc')
    axes[0, 0].plot(epochs, history['val_word_acc'], label='val-word-acc')
    axes[0, 0].set_title('Word Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['train_loss'], label='train-loss')
    axes[0, 1].plot(epochs, history['val_loss'], label='val-loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CTC Loss')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history['train_cer'], label='train-cer')
    axes[1, 0].plot(epochs, history['val_cer'], label='val-cer')
    axes[1, 0].set_title('CER')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CER (%)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history['train_wer'], label='train-wer')
    axes[1, 1].plot(epochs, history['val_wer'], label='val-wer')
    axes[1, 1].set_title('WER')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('WER (%)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


parser = argparse.ArgumentParser(
    description='Train OCR models on MJSynth subset selected by DUAL mask',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dataset', type=str, default='MJSynth', choices=['MJSynth'])
parser.add_argument('--arch', type=str, default='CRNN', choices=['CRNN'])
parser.add_argument('--data_dir', type=str, default='./data/MJSynth',
                    help='Path to local dataset cache (Arrow format).')
parser.add_argument('--download', action='store_false', default=False,
                    help='Disabled in subset script to avoid any network calls.')
parser.add_argument('--train_shuffle', action='store_true', default=True,
                    help='Shuffle training data for local map-style datasets.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to checkpoint to resume training from.')

# Optimization options
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (kept for CLI compatibility).')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# Checkpoints
parser.add_argument('--print_freq', default=12, type=int, metavar='N', help='print frequency')
parser.add_argument('--save_path', type=str, default='./save_subset', help='Folder to save logs')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')

# Acceleration
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

# Random seed
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

# DUAL subset selection
parser.add_argument('--subset_rate', default=0.3, type=float,
                    help='If <=1, treated as ratio. If >1, treated as absolute sample count.')
parser.add_argument('--mask-path', required=True, type=str,
                    help='Path to dual_mask_T*.npy generated by importance_evaluation.py')
parser.add_argument('--keep', type=str, default='lowest', choices=['lowest', 'highest'],
                    help='Choose which side of mask ranking to keep.')

args = parser.parse_args()
gpu_id = int(args.gpu)
args.use_cuda = torch.cuda.is_available() and gpu_id >= 0
args.device = f'cuda:{gpu_id}' if args.use_cuda else 'cpu'

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = args.use_cuda


def main():
    print(args.save_path)
    args.save_path = os.path.join(args.save_path, args.dataset, f'{args.manualSeed}')
    log_path = os.path.join(args.save_path, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    log = open(os.path.join(log_path, f'seed_{args.manualSeed}_subset_log.txt'), 'w')
    print_log('save path : {}'.format(args.save_path), log)

    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log('Random Seed: {}'.format(args.manualSeed), log)
    print_log('python version : {}'.format(sys.version.replace('\n', ' ')), log)
    print_log('torch  version : {}'.format(torch.__version__), log)
    print_log('cudnn  version : {}'.format(torch.backends.cudnn.version()), log)
    print_log('Dataset: {}'.format(args.dataset), log)
    print_log('Network: {}'.format(args.arch), log)
    print_log('Batchsize: {}'.format(args.batch_size), log)
    print_log('Learning Rate: {}'.format(args.learning_rate), log)
    print_log('Weight Decay: {}'.format(args.decay), log)
    print_log('Mask Path: {}'.format(args.mask_path), log)

    train_loader_full, test_loader = load_data(args)
    train_loader, full_train_samples, pruned_train_samples = build_pruned_train_loader(args, train_loader_full)

    args.num_samples = pruned_train_samples
    args.num_iter = len(train_loader)
    args.num_classes = NUM_CLASSES

    print_log('Full train samples: {}'.format(full_train_samples), log)
    print_log('Pruned train samples: {}'.format(pruned_train_samples), log)

    print_log("=> creating model '{}'".format(args.arch), log)
    net = load_model(args.arch, args.num_classes)
    print_log('=> network :\n {}'.format(net), log)
    net = net.to(args.device)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(args.device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * max(1, args.num_iter)),
    )

    recorder = RecorderMeter(args.epochs)
    start_epoch = 0

    history = {
        'train_word_acc': [],
        'val_word_acc': [],
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': [],
        'train_wer': [],
        'val_wer': [],
    }

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        checkpoint = load_checkpoint_compat(args.resume, map_location=args.device)
        net.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'recorder_state' in checkpoint:
            recorder = load_recorder_from_state(recorder, checkpoint['recorder_state'])
        elif 'recorder' in checkpoint:
            recorder = checkpoint['recorder']
        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            for k in history.keys():
                history[k] = list(checkpoint['history'].get(k, []))
        recorder = resize_recorder_total_epochs(recorder, args.epochs)
        start_epoch = int(checkpoint.get('epoch', 0))
        print_log(
            f"=> resumed from '{args.resume}' (next epoch: {start_epoch + 1}, zero-based index: {start_epoch})",
            log,
        )

    if args.evaluate:
        validate(test_loader, args, net, criterion, log)
        log.close()
        return

    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        try:
            current_learning_rate = scheduler.get_last_lr()[0]
        except IndexError:
            current_learning_rate = args.learning_rate

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.6f}]'.format(
                time_string(), epoch + 1, args.epochs, need_time, current_learning_rate)
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder.max_accuracy(False), 100 - recorder.max_accuracy(False)),
            log,
        )

        epoch_start = time.time()

        train_acc, train_loss, train_cer, train_wer = train(
            train_loader, args, net, criterion, optimizer, scheduler, epoch, log
        )
        val_acc, val_loss, val_cer, val_wer = validate(test_loader, args, net, criterion, log)

        history['train_word_acc'].append(float(train_acc))
        history['val_word_acc'].append(float(val_acc))
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_cer'].append(float(train_cer))
        history['val_cer'].append(float(val_cer))
        history['train_wer'].append(float(train_wer))
        history['val_wer'].append(float(val_wer))

        is_best = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'recorder_state': recorder_to_state(recorder),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history,
        }

        save_checkpoint(state, args.save_path, f'epoch_{epoch + 1:03d}_subset_ckpt.pth.tar')

        if is_best:
            save_checkpoint(state, args.save_path, 'best_subset_ckpt.pth.tar')

        if epoch + 1 == args.epochs:
            save_checkpoint(state, args.save_path, 'last_subset_ckpt.pth.tar')

        epoch_time.update(time.time() - epoch_start)

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            recorder.plot_curve(os.path.join(args.save_path, f'{args.manualSeed}_subset_curve.png'))

    np.savez(os.path.join(args.save_path, 'metrics_history.npz'), **history)
    plot_metric_curves(history, os.path.join(args.save_path, f'{args.manualSeed}_metrics_curve.png'))

    log.close()


def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    word_acc = AverageMeter()
    char_acc = AverageMeter()
    cer_meter = AverageMeter()
    wer_meter = AverageMeter()

    model.train()
    end = time.time()

    for t, batch in enumerate(train_loader):
        images, labels = batch[0], batch[1]

        data_time.update(time.time() - end)
        images = images.to(args.device)

        output = model(images).log_softmax(2)
        targets_flat, target_lengths = encode_labels(labels)
        targets_flat = targets_flat.to(args.device)
        target_lengths = target_lengths.to(args.device)

        T, N, _ = output.shape
        input_lengths = torch.full((N,), T, dtype=torch.long, device=args.device)

        per_sample_loss = torch.nn.functional.ctc_loss(
            output,
            targets_flat,
            input_lengths,
            target_lengths,
            blank=0,
            reduction='none',
            zero_infinity=True,
        ) / target_lengths.float().to(args.device).clamp_min(1)
        loss = per_sample_loss.mean()

        pred_texts = ctc_decode(output, VOCAB)
        target_texts = normalize_targets(labels)

        acc = word_accuracy(pred_texts, target_texts)
        cer = cer_percent(pred_texts, target_texts)
        wer = wer_percent(pred_texts, target_texts)
        cacc = 100.0 - cer

        word_acc.update(acc, images.size(0))
        char_acc.update(cacc, images.size(0))
        cer_meter.update(cer, images.size(0))
        wer_meter.update(wer, images.size(0))
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log(
                f'  Epoch: [{epoch + 1:03d}][{t:03d}/{args.num_iter}]   '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})   '
                f'WordAcc {word_acc.val:.3f} ({word_acc.avg:.3f})   '
                f'CharAcc {char_acc.val:.3f} ({char_acc.avg:.3f})   '
                f'CER {cer_meter.val:.3f} ({cer_meter.avg:.3f})   '
                f'WER {wer_meter.val:.3f} ({wer_meter.avg:.3f})   '
                + time_string(),
                log,
            )

    print_log(
        f'  **Train** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  CER {cer_meter.avg:.3f}  WER {wer_meter.avg:.3f}  Loss {losses.avg:.4f}',
        log,
    )

    return word_acc.avg, losses.avg, cer_meter.avg, wer_meter.avg


def validate(test_loader, args, model, criterion, log):
    losses = AverageMeter()
    word_acc = AverageMeter()
    char_acc = AverageMeter()
    cer_meter = AverageMeter()
    wer_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images = images.to(args.device)

            output = model(images).log_softmax(2)
            targets_flat, target_lengths = encode_labels(labels)
            targets_flat = targets_flat.to(args.device)
            target_lengths = target_lengths.to(args.device)
            T, N, _ = output.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=args.device)
            loss = criterion(output, targets_flat, input_lengths, target_lengths)

            pred_texts = ctc_decode(output, VOCAB)
            target_texts = normalize_targets(labels)
            acc = word_accuracy(pred_texts, target_texts)
            cer = cer_percent(pred_texts, target_texts)
            wer = wer_percent(pred_texts, target_texts)
            cacc = 100.0 - cer

            word_acc.update(acc, images.size(0))
            char_acc.update(cacc, images.size(0))
            cer_meter.update(cer, images.size(0))
            wer_meter.update(wer, images.size(0))
            losses.update(loss.item(), images.size(0))

    print_log(
        f'  **Test** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  CER {cer_meter.avg:.3f}  WER {wer_meter.avg:.3f}  Loss {losses.avg:.4f}',
        log,
    )

    return word_acc.avg, losses.avg, cer_meter.avg, wer_meter.avg


if __name__ == '__main__':
    main()
