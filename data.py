import io
import os
import itertools
from typing import Iterator, Optional, Tuple
import time
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms
from PIL import Image, ImageFile
from datasets import load_dataset as hf_load_dataset, Image as HFImage

ImageFile.LOAD_TRUNCATED_IMAGES = True

HF_DATASET_ID = "priyank-m/MJSynth_text_recognition"
DEFAULT_H = 32
DEFAULT_W = 128


# ── Public entry points ───────────────────────────────────────────────────────

def load_data(args):
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader


def load_dataset(args):
    if args.dataset == 'MJSynth':
        return load_MJSynth(args)
    elif args.dataset == 'TRDG':
        return load_TRDG(args)
    else:
        raise NotImplementedError(f"Dataset not supported: {args.dataset}")


def load_MJSynth(args, img_height: int = DEFAULT_H, img_width: int = DEFAULT_W):
    """
    Two modes controlled by args.data_dir:

    LOCAL MODE  (args.data_dir is set)
        Loads from a pre-downloaded HuggingFace dataset cache on disk
        created by save_to_disk().
        If args.download=True and the cache does not exist yet, it is
        downloaded first (one-time cost).  Supports shuffle=True and
        accurate __len__, which is better for training.

    STREAMING MODE  (args.data_dir is None / empty string)
        Streams directly from HuggingFace Hub — original behaviour.
        Falls back automatically when data_dir is not provided.
    """
    data_dir = getattr(args, 'data_dir', None) or ''
    download  = getattr(args, 'download', False)
    train_shuffle = getattr(args, 'train_shuffle', False)

    print('Loading MJSynth... ', end='', flush=True)
    time_start = time.time()

    transform = _build_transform(img_height, img_width)
    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,   # keep workers alive between epochs
    )

    if data_dir:
        # ── LOCAL MODE ────────────────────────────────────────────────────────
        train_cache = os.path.join(data_dir, 'train')
        test_cache  = os.path.join(data_dir, 'test')

        if download and not os.path.isdir(train_cache):
            _download_mjsynth(data_dir)

        if not os.path.isdir(train_cache):
            raise FileNotFoundError(
                f"Local MJSynth cache not found at '{data_dir}'. "
                "Expected subfolders named 'train' and 'test' created by "
                "HuggingFace save_to_disk(). Re-run with --download to fetch it automatically."
            )

        train_ds = MJSynthLocalDataset(train_cache, transform=transform)
        test_ds  = MJSynthLocalDataset(test_cache,  transform=transform)

        print(f"done (local, {len(train_ds):,} train / {len(test_ds):,} test) "
              f"in {time.time() - time_start:.2f}s.")

        return (
            DataLoader(train_ds, shuffle=train_shuffle,  **loader_kw),
            DataLoader(test_ds,  shuffle=False, **loader_kw),
        )

    else:
        # ── STREAMING MODE (original behaviour) ──────────────────────────────
        max_samples = getattr(args, 'max_samples', None)
        shared = dict(img_height=img_height, img_width=img_width,
                      max_samples=max_samples)

        train_ds = MJSynthStreamingDataset(split="train", **shared)
        test_ds  = MJSynthStreamingDataset(split="test",  **shared)

        print(f"done (streaming) in {time.time() - time_start:.2f}s.")

        return (
            DataLoader(train_ds, shuffle=False, **loader_kw),
            DataLoader(test_ds,  shuffle=False, **loader_kw),
        )


# ── Download helper ───────────────────────────────────────────────────────────

def _download_mjsynth(data_dir: str) -> None:
    """
    Download MJSynth from HuggingFace Hub and save Arrow files to data_dir.
    Runs once; subsequent runs load from disk instantly.
    """
    print(f"\nDownloading MJSynth to '{data_dir}' — this may take a while...")

    for split in ('train', 'test'):
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            print(f"  [{split}] already exists, skipping.")
            continue

        print(f"  [{split}] downloading...", flush=True)
        ds = hf_load_dataset(
            HF_DATASET_ID,
            split=split,
            streaming=False,           # full download
            trust_remote_code=True,
        )
        # Store images as raw bytes — much smaller Arrow files on disk.
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))
        if "img" in ds.features:
            ds = ds.cast_column("img", HFImage(decode=False))

        os.makedirs(split_dir, exist_ok=True)
        ds.save_to_disk(split_dir)
        print(f"  [{split}] saved ({len(ds):,} samples).")

    print("Download complete.\n")


# ── Local (map-style) Dataset ─────────────────────────────────────────────────

class MJSynthLocalDataset(Dataset):
    """
    Reads MJSynth from a local Arrow cache produced by _download_mjsynth().
    Supports __len__ and random-access __getitem__, enabling proper shuffling.
    """

    def __init__(self, cache_dir: str, transform=None,
                 img_height: int = DEFAULT_H, img_width: int = DEFAULT_W):
        from datasets import load_from_disk
        self.ds = load_from_disk(cache_dir)
        # Keep images as raw bytes; decode lazily in __getitem__
        if "image" in self.ds.features:
            self.ds = self.ds.cast_column("image", HFImage(decode=False))
        if "img" in self.ds.features:
            self.ds = self.ds.cast_column("img", HFImage(decode=False))
        self.transform = transform or _build_transform(img_height, img_width)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        sample = self.ds[idx]
        image  = _decode_image(sample)
        label  = _get_label(sample)
        return self.transform(image), label, idx


# ── Helper functions ──────────────────────────────────────────────────────────

def _build_transform(img_height: int, img_width: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])


def _decode_image(sample: dict) -> Image.Image:
    img = sample.get("image") or sample.get("img")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, dict) and img.get("path"):
        return Image.open(img["path"]).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    raise ValueError(f"Unrecognised image format. Keys: {list(sample.keys())}")


def _get_label(sample: dict) -> str:
    for key in ("label", "text", "word", "annotation"):
        if key in sample:
            return str(sample[key])
    raise KeyError(f"No label key found. Keys: {list(sample.keys())}")


# ── Streaming Dataset  ────────────────────────────

class MJSynthStreamingDataset(IterableDataset):
    def __init__(
        self,
        split: str = "train",
        transform=None,
        img_height: int = DEFAULT_H,
        img_width: int = DEFAULT_W,
        max_samples: Optional[int] = None,
    ):
        self.hf_ds = hf_load_dataset(
            HF_DATASET_ID, split=split, streaming=True, trust_remote_code=True,
        )
        if "image" in self.hf_ds.features:
            self.hf_ds = self.hf_ds.cast_column("image", HFImage(decode=False))
        if "img" in self.hf_ds.features:
            self.hf_ds = self.hf_ds.cast_column("img", HFImage(decode=False))
        self.transform = transform or _build_transform(img_height, img_width)
        self.max_samples = max_samples

    def __len__(self) -> int:
        if self.max_samples is None:
            raise TypeError(
                "Length is undefined for streaming MJSynth dataset unless "
                "max_samples is set.  Set args.max_samples or use --data_dir."
            )
        return int(self.max_samples)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str, int]]:
        worker_info   = get_worker_info()
        dataset_iter  = self.hf_ds
        local_limit   = self.max_samples
        worker_id     = 0
        worker_stride = 1

        if worker_info is not None:
            worker_id     = worker_info.id
            worker_stride = worker_info.num_workers
            dataset_iter  = itertools.islice(
                dataset_iter, worker_info.id, None, worker_info.num_workers)
            if local_limit is not None:
                base      = local_limit // worker_info.num_workers
                remainder = local_limit % worker_info.num_workers
                local_limit = base + (1 if worker_info.id < remainder else 0)

        count = 0
        for sample in dataset_iter:
            if local_limit is not None and count >= local_limit:
                break
            try:
                image = _decode_image(sample)
                label = _get_label(sample)
            except Exception:
                continue
            sample_index = worker_id + count * worker_stride
            yield self.transform(image), label, sample_index
            count += 1


# ── TRDG stub ─────────────────────────────────────────────────────────────────

def load_TRDG(args):
    raise NotImplementedError(
        "TRDG dataset loader is not yet implemented. "
        "Please implement load_TRDG() in data.py or use --dataset MJSynth."
    )