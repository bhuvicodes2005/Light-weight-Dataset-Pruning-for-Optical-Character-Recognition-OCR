import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import numpy as np
from data import load_data
from transformers import get_linear_schedule_with_warmup, TrOCRProcessor
from torchmetrics.text import CharErrorRate


########## Loading models ##########################
def load_model(arch, num_classes):
    """Load OCR model based on arch name."""
    if arch == 'CRNN':
        from models.CRNN import CRNN
        return CRNN(img_channel=1, img_height=32, img_width=128, num_class=num_classes)
    elif arch == 'PARSeq':
        from models.PARSEQ import PARSeq
        return PARSeq(num_classes=num_classes, img_channel=1, img_height=32, img_width=128)
    else:
        raise NotImplementedError(f"Model not supported: {arch}")


########################################################################################################################
#  Training Baseline
########################################################################################################################

# Base VOCAB: 0='-' (blank/PAD), 1-26='a-z', 27-52='A-Z', 53-62='0-9'
# For PARSeq, we add EOS at index 63 (so num_classes becomes 64 effectively)
VOCAB_BASE = '-' + ''.join(
    [chr(i) for i in range(ord('a'), ord('z') + 1)] +    # a-z (1-26)
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +    # A-Z (27-52)
    [str(i) for i in range(10)]                           # 0-9 (53-62)
)

# Standard 63-class setup (for CTC/CRNN)
VOCAB = VOCAB_BASE
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)  # 63

# For PARSeq, we need an EOS token. We'll use index 63 as EOS.
# So PARSeq actually uses 64 classes internally (0-63)
PARSEQ_VOCAB = VOCAB_BASE + '|'  # Add EOS marker at index 63
PARSEQ_CHAR2IDX = {c: i for i, c in enumerate(PARSEQ_VOCAB)}
PARSEQ_NUM_CLASSES = len(PARSEQ_VOCAB)  # 64


def encode_labels(texts, char2idx=CHAR2IDX):
    """Convert list of strings to (targets_flat, target_lengths) for CTCLoss."""
    encoded, lengths = [], []
    for t in texts:
        ids = [char2idx[c] for c in t if c in char2idx]
        encoded.extend(ids)
        lengths.append(len(ids))
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def encode_parseq_targets(texts, max_len=25, char2idx=PARSEQ_CHAR2IDX):
    """
    Encode texts for PARSeq: [char1, char2, ..., EOS, PAD, PAD...]
    Returns tensor of shape (N, max_len) with EOS at end of each sequence.
    """
    targets = torch.zeros((len(texts), max_len), dtype=torch.long)
    eos_idx = len(PARSEQ_VOCAB) - 1  # 63
    
    for i, text in enumerate(texts):
        # Convert chars to indices (skip if not in vocab)
        ids = [char2idx[c] for c in text if c in char2idx][:max_len-1]  # Reserve room for EOS
        ids.append(eos_idx)  # Add EOS token
        # Fill in the targets
        targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        # Rest remains 0 (PAD)
    
    return targets


def normalize_targets(texts, char2idx=CHAR2IDX):
    return [''.join([c for c in t if c in char2idx]) for t in texts]


def cer_percent(preds: list, targets: list) -> float:
    """Calculate Character Error Rate (CER) in percent using torchmetrics."""
    try:
        cer_metric = CharErrorRate()
        cer_score = cer_metric(preds, targets)
        return cer_score.item() * 100.0
    except Exception as e:
        print(f"Warning: CharErrorRate failed ({e}), returning 0")
        return 0.0


parser = argparse.ArgumentParser(description='Trains OCR models on MJSynth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='MJSynth', choices=['MJSynth', 'TRDG'])
parser.add_argument('--arch', type=str, default='CRNN', choices=['CRNN', 'ASTER', 'SAR', 'ViTSTR', 'PARSeq', 'TrOCR'])
parser.add_argument('--data_dir', type=str, default='',
                    help='Path to local dataset cache (Arrow format). '
                         'If set, loads from disk instead of streaming. '
                         'Example: ./data/MJSynth')
parser.add_argument('--download', action='store_true', default=False,
                    help='Download dataset to --data_dir if not already cached.')
parser.add_argument('--train_shuffle', action='store_true', default=False,
                    help='Shuffle training data for local map-style datasets.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to checkpoint to resume training from.')

# Optimization options
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# Checkpoints and Dynamics
parser.add_argument('--print_freq', default=12, type=int, metavar='N', help='print frequency (default: 12)')
parser.add_argument('--save_path', type=str, default='./save', help='Folder to save logs')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',default= False, help='evaluate model on validation set')
parser.add_argument('--dynamics', default=True, action='store_true', help='save training dynamics')
# Acceleration
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

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

def is_ctc_model(arch):
    """CTC-based models need CTCLoss. Seq2Seq models handle loss internally."""
    return arch in ['CRNN']

def is_seq2seq_model(arch):
    """Seq2Seq models (TrOCR) compute loss internally."""
    return arch in ['TrOCR']

def is_parseq_model(arch):
    """PARSeq uses PLM and has custom loss computation."""
    return arch == 'PARSeq'


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
    if {'total_epoch', 'current_epoch', 'epoch_losses', 'epoch_accuracy'}.issubset(recorder_state.keys()):
        recorder.total_epoch = recorder_state['total_epoch']
        recorder.current_epoch = recorder_state['current_epoch']
        recorder.epoch_losses = recorder_state['epoch_losses']
        recorder.epoch_accuracy = recorder_state['epoch_accuracy']
    return recorder


def load_checkpoint_compat(path, map_location):
    """Load checkpoints across PyTorch versions (2.6+ changed weights_only default)."""
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        if 'Weights only load failed' not in str(e):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def main():
    # Init logger
    print(args.save_path)
    save_path_arch = os.path.join(args.save_path, args.dataset, f'{args.manualSeed}')
    log_path = os.path.join(save_path_arch, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'seed_{}_log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(save_path_arch), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Dataset: {}".format(args.dataset), log)
    print_log("Network: {}".format(args.arch), log)
    print_log("Batchsize: {}".format(args.batch_size), log)
    print_log("Learning Rate: {}".format(args.learning_rate), log)
    print_log("Momentum: {}".format(args.momentum), log)
    print_log("Weight Decay: {}".format(args.decay), log)

    # data loading 
    train_loader, test_loader = load_data(args)
    try:
        args.num_samples = len(train_loader.dataset)
    except (TypeError, AttributeError):
        args.num_samples = getattr(args, 'max_samples', 8_919_257)
    try:
        args.num_iter = len(train_loader)
    except TypeError:
        args.num_iter = max(1, args.num_samples // args.batch_size)

    # PARSeq uses NUM_CLASSES+1 (adds EOS token), others use standard NUM_CLASSES
    if is_parseq_model(args.arch):
        args.num_classes = PARSEQ_NUM_CLASSES  # 64
    else:
        args.num_classes = NUM_CLASSES  # 63
    
    # ── Model 
    print_log("=> creating model '{}'".format(args.arch), log)

    net = load_model(args.arch, NUM_CLASSES)  # Pass base 63, PARSeq adds EOS internally

    print_log("=> network :\n {}".format(net), log)
    net = net.to(args.device)

    trocr_processor = None
    if args.arch == 'TrOCR':
        trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        print_log("=> TrOCRProcessor loaded once.", log)

    # Loss
    if args.arch == 'CRNN':
        # CTC loss — blank token at index 0
        criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(args.device)
    elif args.arch == 'TrOCR':
        criterion = None   # HuggingFace computes CE loss internally when labels= is passed
    elif is_parseq_model(args.arch):
        criterion = None   # PARSeq computes PLM loss internally
    else:
        # ASTER, SAR, ViTSTR — standard cross entropy
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(args.device)

    # Optimizer & Scheduler
    if args.arch == 'TrOCR':
        # TrOCR is pretrained — use AdamW with linear warmup
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,      # recommend 5e-5
            weight_decay=args.decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=args.epochs * args.num_iter
        )

    else:
        # CRNN, PARSeq — Adam + CosineAnnealing
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,      # 1e-3 works well for PARSeq
            weight_decay=args.decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * args.num_iter
        )

    recorder = RecorderMeter(args.epochs)
    start_epoch = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
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
        start_epoch = int(checkpoint.get('epoch', 0))
        print_log(
            f"=> resumed from '{args.resume}' (next epoch: {start_epoch + 1}, zero-based index: {start_epoch})",
            log,
        )
    
    # evaluation
    if args.evaluate:
        time1 = time.time()
        validate(test_loader, args, net, criterion, log, trocr_processor)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Main loop
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):

        try:
            current_learning_rate = scheduler.get_last_lr()[0]
        except IndexError:
            current_learning_rate = args.learning_rate

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(
                time_string(), epoch + 1, args.epochs, need_time, current_learning_rate)
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder.max_accuracy(False), 100 - recorder.max_accuracy(False)), log)


        epoch_start = time.time()

        # train for one epoch
        train_acc, train_los, loss_epoch, index_epoch, y_epoch = train(
            train_loader, args, net, criterion, optimizer, scheduler, epoch, log, trocr_processor)

        # evaluate on validation set
        val_acc, val_los = validate(test_loader, args, net, criterion, log, trocr_processor)
        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'recorder_state': recorder_to_state(recorder),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        save_checkpoint(state, save_path_arch, f'epoch_{epoch + 1:03d}_ckpt.pth.tar')

        if is_best:
            save_checkpoint(state, save_path_arch, 'best_ckpt.pth.tar')

        if epoch + 1 == args.epochs:
            save_checkpoint(state, save_path_arch, 'last_ckpt.pth.tar')

        # measure elapsed time (now correctly wraps the full training epoch)
        epoch_time.update(time.time() - epoch_start)

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            recorder.plot_curve(os.path.join(save_path_arch, f'{args.manualSeed}_curve.png'))

        # save training dynamics

        if args.dynamics:
            dynamics_path = os.path.join(save_path_arch, 'npy')
            if not os.path.exists(dynamics_path):
                os.makedirs(dynamics_path)
            np.save(os.path.join(dynamics_path, f"{epoch}_Loss.npy"), loss_epoch)
            np.save(os.path.join(dynamics_path, f"{epoch}_Index.npy"), index_epoch)
            print('Epoch ' + str(epoch) + ' done!')
    log.close()


# train function

def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log, trocr_processor=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    word_acc = AverageMeter()
    char_acc = AverageMeter()
    cer_meter = AverageMeter()
    
    # switch to train mode
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
            sample_indices = torch.arange(t * args.batch_size, t * args.batch_size + images.size(0))

        data_time.update(time.time() - end)
        images = images.to(args.device)
        sample_indices = torch.as_tensor(sample_indices, dtype=torch.long).reshape(-1)

        # ── Forward + Loss
        if is_ctc_model(args.arch):
            # output: (T, N, C) — time-first for CTCLoss
            output = model(images).log_softmax(2)

            targets_flat, target_lengths = encode_labels(labels)
            targets_flat   = targets_flat.to(args.device)
            target_lengths = target_lengths.to(args.device)
            T, N, _ = output.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=args.device)
            # Compute per-sample losses in one pass; derive the mean loss from them.
            per_sample_loss = torch.nn.functional.ctc_loss(
                output, targets_flat, input_lengths, target_lengths,
                blank=0, reduction='none', zero_infinity=True,
            ) / target_lengths.float().to(args.device).clamp_min(1)
            loss = per_sample_loss.mean()

        elif is_seq2seq_model(args.arch):
            # TrOCR — pass pixel_values + labels, loss computed internally

            pixel_values = images.to(args.device)
            encoding = trocr_processor.tokenizer(
                list(labels), return_tensors='pt', padding=True).to(args.device)
            outputs = model(pixel_values=pixel_values, labels=encoding.input_ids)
            loss   = outputs.loss
            output = outputs.logits   # (N, seq_len, vocab)
            per_sample_loss = loss.detach().repeat(images.size(0))

        elif is_parseq_model(args.arch):
            # PARSeq: Prepare sequence targets with EOS
            targets_seq = encode_parseq_targets(labels, max_len=25).to(args.device)
            
            # Training: returns PLM loss averaged over K permutations
            loss = model(images, labels=targets_seq)
            
            # For metrics: do a forward pass to get logits (NAR mode)
            with torch.no_grad():
                output = model(images)  # Returns logits (N, S, C)
            
            # Per-sample loss for dynamics (approximate using CE)
            per_sample_loss = torch.nn.functional.cross_entropy(
                output.reshape(-1, args.num_classes),
                targets_seq.reshape(-1),
                ignore_index=0,
                reduction='none'
            ).view(images.size(0), -1).mean(dim=1)

        else:
            # Attention-based (ASTER, SAR, ViTSTR)
            targets_flat, target_lengths = encode_labels(labels)
            targets_flat = targets_flat.to(args.device)
            output = model(images)
            loss   = criterion(output.view(-1, args.num_classes), targets_flat)
            per_sample_loss = loss.detach().repeat(images.size(0))

        # ── Decode & Word Accuracy ────────────────────────────────────────────
        if is_parseq_model(args.arch):
            # Use PARSeq vocab for decoding (includes EOS at 63)
            pred_texts = greedy_decode_parseq(output, PARSEQ_VOCAB)
            target_texts = normalize_targets(labels, PARSEQ_CHAR2IDX)
        else:
            pred_texts = ctc_decode(output, VOCAB) if is_ctc_model(args.arch) else greedy_decode(output, VOCAB)
            target_texts = normalize_targets(labels, CHAR2IDX)
            
        acc = word_accuracy(pred_texts, target_texts)
        cer = cer_percent(pred_texts, target_texts)
        cacc = 100.0 - cer
        word_acc.update(acc, images.size(0))
        char_acc.update(cacc, images.size(0))
        cer_meter.update(cer, images.size(0))
        losses.update(loss.item(), images.size(0))

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
            y_epoch     = np.concatenate([y_epoch,     np.array(labels)])

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # helps stability
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log(
                f'  Epoch: [{epoch + 1:03d}][{t:03d}/{args.num_iter}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'WordAcc {word_acc.val:.3f} ({word_acc.avg:.3f})  '
                f'CharAcc {char_acc.val:.3f} ({char_acc.avg:.3f})  '
                f'CER {cer_meter.val:.3f} ({cer_meter.avg:.3f})  '
                + time_string(), log)

    print_log(
        f'  **Train** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  CER {cer_meter.avg:.3f}  Loss {losses.avg:.4f}',
        log
    )

    return word_acc.avg, losses.avg, loss_epoch, index_epoch, y_epoch

def validate(test_loader, args, model, criterion, log, trocr_processor=None):
    losses   = AverageMeter()
    word_acc = AverageMeter()
    char_acc = AverageMeter()
    cer_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # The streaming dataset yields (image, label, index) 3-tuples;
            # ignore the index in validation just like train does when len==3.
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

                pixel_values = images.to(args.device)
                encoding     = trocr_processor.tokenizer(
                    list(labels), return_tensors='pt', padding=True).to(args.device)
                outputs = model(pixel_values=pixel_values, labels=encoding.input_ids)
                loss    = outputs.loss
                output  = outputs.logits

            elif is_parseq_model(args.arch):
                # PARSeq validation
                targets_seq = encode_parseq_targets(labels, max_len=25).to(args.device)
                output = model(images)  # Logits (N, S, C)
                # Compute validation loss using standard CE (not PLM)
                loss = torch.nn.functional.cross_entropy(
                    output.reshape(-1, args.num_classes),
                    targets_seq.reshape(-1),
                    ignore_index=0
                )

            else:
                targets_flat, _ = encode_labels(labels)
                targets_flat    = targets_flat.to(args.device)
                output = model(images)
                loss   = criterion(output.view(-1, args.num_classes), targets_flat)

            if is_parseq_model(args.arch):
                pred_texts = greedy_decode_parseq(output, PARSEQ_VOCAB)
                target_texts = normalize_targets(labels, PARSEQ_CHAR2IDX)
            else:
                pred_texts = ctc_decode(output, VOCAB) if is_ctc_model(args.arch) else greedy_decode(output, VOCAB)
                target_texts = normalize_targets(labels, CHAR2IDX)
                
            acc = word_accuracy(pred_texts, target_texts)
            cer = cer_percent(pred_texts, target_texts)
            cacc = 100.0 - cer
            word_acc.update(acc, images.size(0))
            char_acc.update(cacc, images.size(0))
            cer_meter.update(cer, images.size(0))
            losses.update(loss.item(), images.size(0))

    print_log(
        f'  **Test** WordAcc {word_acc.avg:.3f}  CharAcc {char_acc.avg:.3f}  CER {cer_meter.avg:.3f}  Loss {losses.avg:.4f}',
        log
    )
    return word_acc.avg, losses.avg

# Decode helpers 

def ctc_decode(log_probs: torch.Tensor, vocab: str) -> list:
    """Greedy CTC decode. log_probs: (T, N, C)"""
    # argmax over classes at each timestep
    indices = log_probs.argmax(2).permute(1, 0).cpu().numpy()  # (N, T)
    results = []
    for row in indices:
        chars, prev = [], -1
        for idx in row:
            if idx != prev and idx != 0:   # skip blank (0) and repeats
                chars.append(vocab[idx])
            prev = idx
        results.append(''.join(chars))
    return results


def greedy_decode(output: torch.Tensor, vocab: str) -> list:
    """Greedy decode for attention/seq2seq outputs. output: (N, S, C)"""
    indices = output.argmax(2).cpu().numpy()  # (N, S)
    results = []
    for row in indices:
        chars = [vocab[i] for i in row if 0 < i < len(vocab)]
        results.append(''.join(chars))
    return results


def greedy_decode_parseq(output: torch.Tensor, vocab: str) -> list:
    """Greedy decode for PARSeq. Stops at EOS (index 63)."""
    indices = output.argmax(2).cpu().numpy()  # (N, S)
    eos_idx = len(vocab) - 1  # 63
    results = []
    for row in indices:
        chars = []
        for i in row:
            if i == eos_idx:
                break
            if i != 0:  # Skip PAD
                chars.append(vocab[i])
        results.append(''.join(chars))
    return results


def word_accuracy(preds: list, targets) -> float:
    """Word-level match accuracy (case-insensitive)."""
    correct = sum(p.lower() == t.lower() for p, t in zip(preds, targets))
    return correct / len(targets) * 100.0 if len(targets) > 0 else 0.0

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
