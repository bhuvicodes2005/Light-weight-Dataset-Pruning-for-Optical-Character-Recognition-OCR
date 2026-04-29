import os
import argparse
import re
import numpy as np
import torch
import torch.nn.functional as F


########################################################################################################################
#  Calculate Importance
########################################################################################################################

parser = argparse.ArgumentParser(description='Calculate sample-wise importance',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dynamics_path', type=str, required=True,
                    help='Folder to saved dynamics.')
parser.add_argument('--window_size', default=5, type=int,
                    help='Size of the sliding window. (for Dyn-Unc & DUAL & TDDS)')
parser.add_argument('--save_path', type=str, required=True,
                    help='Folder to save mask.')
parser.add_argument('--seed', default=42, type=int,
                    help='manual seed')
parser.add_argument('--dataset', default='MJSynth', type=str, choices=['MJSynth', 'TRDG'],
                    help='dataset')
parser.add_argument('--source', default='loss', type=str, choices=['auto', 'loss', 'output'],
                    help='Trajectory source: loss, output, or auto-select.')
parser.add_argument('--num_epochs', default=12, type=int,
                    help='Number of epochs to use for DUAL scoring.')

args = parser.parse_args()


def dual(preds, window_size=5, dim=0):
    """
     Window-based DUAL scoring from per-epoch losses.

     1) Convert loss to probability-like ease:
         p_{t,i} = exp(-loss_{t,i})
     2) For each sliding window, compute:
         p_mean = mean(p), uncertainty = sqrt(sum((p_{k+j} - p_mean)^2) / (J - 1))
     3) Window score = (1 - p_mean) * uncertainty
     4) Final score = mean(window scores)
    """
    if preds.dim() != 2:
        raise ValueError(f"Expected preds to be 2D (epochs, samples), got {tuple(preds.shape)}")

    if dim != 0:
        preds = preds.transpose(0, dim)

    p = torch.exp(-preds)
    num_epochs = p.size(0)

    if num_epochs < 2:
        raise ValueError("DUAL requires at least 2 epochs to estimate uncertainty.")

    w = max(2, min(int(window_size), num_epochs))
    num_windows = num_epochs - w + 1

    window_scores = []
    for start in range(num_windows):
        window = p[start:start + w]                    # (w, samples)
        p_mean = window.mean(dim=0)                    # (samples,)
        uncertainty = torch.sqrt(
            ((window - p_mean) ** 2).sum(dim=0) / (w - 1)
        )                                               # (samples,)
        window_scores.append((1 - p_mean) * uncertainty)  # (samples,)

    score = torch.stack(window_scores, dim=0).mean(dim=0)
    mask = np.argsort(score.numpy())
    return score, mask


def rearrange(args, values, indexes):
    """
    Reorganize per-batch sample values into per-epoch full-sample trajectories.
    Returns tensor of shape: (num_epochs, num_samples).
    """
    max_index  = max(int(np.max(idx)) for idx in indexes)
    num_samples = max_index + 1

    trajectories = []
    for epoch_values, epoch_indices in zip(values, indexes):
        epoch_values_t  = torch.tensor(epoch_values,  dtype=torch.float32).reshape(-1)
        epoch_indices_t = torch.tensor(epoch_indices, dtype=torch.long).reshape(-1)

        full_epoch = torch.zeros(num_samples, dtype=torch.float32)
        count_epoch = torch.zeros(num_samples, dtype=torch.float32)
        full_epoch = full_epoch.index_add(0, epoch_indices_t, epoch_values_t)
        count_epoch = count_epoch.index_add(0, epoch_indices_t, torch.ones_like(epoch_values_t))
        full_epoch = full_epoch / count_epoch.clamp_min(1.0)
        trajectories.append(full_epoch)

    rearranged = torch.stack(trajectories)

    os.makedirs(args.save_path, exist_ok=True)
    np.save(f"{args.save_path}/rearranged.npy", rearranged.numpy())
    return rearranged


def detect_epochs(dynamics_path: str):
    pattern   = re.compile(r'^(\d+)_(Output|Loss|Index)\.npy$')
    available = {'Output': set(), 'Loss': set(), 'Index': set()}

    for filename in os.listdir(dynamics_path):
        match = pattern.match(filename)
        if not match:
            continue
        epoch = int(match.group(1))
        kind  = match.group(2)
        available[kind].add(epoch)
    return available


def choose_source_and_epochs(args):
    available     = detect_epochs(args.dynamics_path)
    loss_epochs   = sorted(available['Loss']   & available['Index'])
    output_epochs = sorted(available['Output'] & available['Index'])

    if args.source == 'loss':
        if not loss_epochs:
            raise FileNotFoundError("No matching <epoch>_Loss.npy + <epoch>_Index.npy found.")
        return 'loss', loss_epochs

    if args.source == 'output':
        if not output_epochs:
            raise FileNotFoundError("No matching <epoch>_Output.npy + <epoch>_Index.npy found.")
        return 'output', output_epochs

    # auto: prefer loss first (better for CTC dynamics), then output fallback
    if loss_epochs:
        return 'loss', loss_epochs
    if output_epochs:
        return 'output', output_epochs
    raise FileNotFoundError("No valid dynamics files found. Need Index + (Output or Loss).")


def reduce_output_to_sample_scalar(epoch_output: np.ndarray, epoch_indices: np.ndarray) -> np.ndarray:
    """
    Convert model output of an epoch to one scalar per sample.
    """
    output_t    = torch.tensor(epoch_output, dtype=torch.float32)
    num_indices = int(np.asarray(epoch_indices).reshape(-1).shape[0])

    if output_t.dim() == 1:
        flat = output_t.reshape(-1)
        if flat.shape[0] != num_indices:
            raise ValueError("1D output length does not match number of indices.")
        return flat.numpy()

    if output_t.dim() == 2:
        if output_t.shape[0] != num_indices:
            raise ValueError("2D output does not align with index count on dim 0.")
        probs = F.softmax(output_t, dim=-1)
        conf  = probs.max(dim=-1).values
        return conf.numpy()

    if output_t.dim() == 3:
        if output_t.shape[0] == num_indices:
            logits = output_t
        elif output_t.shape[1] == num_indices:
            logits = output_t.permute(1, 0, 2)
        else:
            raise ValueError("3D output does not contain sample dimension matching indices.")

        probs          = F.softmax(logits, dim=-1)
        token_conf     = probs.max(dim=-1).values
        geom_mean_conf = torch.exp(torch.mean(torch.log(token_conf + 1e-12), dim=-1))
        return geom_mean_conf.numpy()

    raise ValueError(f"Unsupported output shape for reduction: {tuple(output_t.shape)}")


def load_trajectories(selected_source, selected_epochs, dynamics_path):
    indexes_local = []
    values_local  = []
    for epoch in selected_epochs:
        idx = np.load(os.path.join(dynamics_path, f'{epoch}_Index.npy'))
        indexes_local.append(idx)

        if selected_source == 'loss':
            val = np.load(os.path.join(dynamics_path, f'{epoch}_Loss.npy'))
            values_local.append(np.asarray(val).reshape(-1))
        else:
            out = np.load(os.path.join(dynamics_path, f'{epoch}_Output.npy'))
            values_local.append(reduce_output_to_sample_scalar(out, idx))
    return values_local, indexes_local


if __name__ == '__main__':
    dynamics_path = args.dynamics_path
    source, epochs = choose_source_and_epochs(args)

    print(f"Using source='{source}' with {len(epochs)} detected epochs.")
    try:
        values, indexes = load_trajectories(source, epochs, dynamics_path)
    except Exception as exc:
        if args.source == 'auto' and source == 'output':
            fallback_epochs = sorted(
                detect_epochs(args.dynamics_path)['Loss'] &
                detect_epochs(args.dynamics_path)['Index']
            )
            if not fallback_epochs:
                raise
            print(f"Output reduction failed ({exc}). Falling back to source='loss'.")
            source = 'loss'
            epochs = fallback_epochs
            values, indexes = load_trajectories(source, epochs, dynamics_path)
        else:
            raise

    print("Rearranging per-sample trajectories")
    target_probs = rearrange(args, values, indexes)

    T = min(args.num_epochs, target_probs.shape[0])
    if T < args.num_epochs:
        print(f"Warning: only {T} epochs available, running DUAL on T={T}")

    print(f"DUAL score processing (T={T}, window_size={args.window_size})")
    score, mask = dual(target_probs[:T], window_size=args.window_size)

    print(f"Total samples: {len(score)}")
    print(f"Score range: [{score.min():.6f}, {score.max():.6f}]")
    
    # Save everything
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, f'dual_score_T{T}.npy'), score.numpy())
    np.save(os.path.join(args.save_path, f'dual_mask_T{T}.npy'),  mask)
    
    print(f"\nSaved to {args.save_path}:")
    print(f"  - dual_score_T{T}.npy (raw DUAL scores)")
    print(f"  - dual_mask_T{T}.npy (ranked indices for subset selection)")
