import os
from datasets import load_dataset, Image as HFImage

HF_DATASET_ID = "priyank-m/MJSynth_text_recognition"

def download_mjsynth(data_dir="./data/mjsynth_hf"):
    os.makedirs(data_dir, exist_ok=True)

    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            print(f"[{split}] already exists, skipping: {split_dir}")
            continue

        print(f"[{split}] downloading from {HF_DATASET_ID} ...")
        ds = load_dataset(
            HF_DATASET_ID,
            split=split,
            streaming=False,
            trust_remote_code=True,
        )

        # Save images as bytes for smaller/faster Arrow cache
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))
        if "img" in ds.features:
            ds = ds.cast_column("img", HFImage(decode=False))

        ds.save_to_disk(split_dir)
        print(f"[{split}] saved {len(ds):,} samples -> {split_dir}")

    print("Done.")

if __name__ == "__main__":
    download_mjsynth("./data/mjsynth_hf")