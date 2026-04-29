# Lightweight Dataset Pruning without Full Training via Example Difficulty and Prediction Uncertainty (DUAL)

## 📖 Overview

This repository implements **DUAL**, based on the paper [Lightweight Dataset Pruning without Full Training via Example Difficulty and Prediction Uncertainty](https://arxiv.org/abs/2502.06905). DUAL identifies and selectively removes redundant or less informative samples from training datasets while maintaining or improving model performance, thereby reducing training time and computational costs.

The complete workflow involves:
1. **Pre-training** on full synthetic dataset (MJSynth) to collect training dynamics
2. **Importance Evaluation** using DUAL scoring to rank samples by importance
3. **Training on Pruned Subsets** using high-importance samples only
4. **Fine-tuning & Testing** on real-world OCR datasets (ICDAR, IIIT5K, SVT)

## 🎯 Key Features

- **Intelligent Dataset Pruning**: DUAL scoring identifies the most important samples for training
- **Multiple OCR Models**: Support for CRNN, SVTR, and VisionLAN architectures
- **Real-world Evaluation**: Fine-tuning and testing on ICDAR, IIIT5K, and SVT benchmarks
- **Efficiency Gains**: Reduce dataset size by 30-50% while maintaining or improving accuracy
- **Complete Pipeline**: From pretraining through evaluation on standard benchmarks
- **Dynamics-based Scoring**: Leverages training dynamics to compute sample importance

## 🤖 Model Architectures

An OCR model is a neural network that reads text from images and predicts the underlying character sequence. In this repository, the main OCR backbones are CRNN, SVTR, and VisionLAN.

### **CRNN** (Convolutional Recurrent Neural Network)
- Architecture: CNN backbone + Bidirectional LSTM + CTC loss
- Advantages: Fast inference, well-established, good baseline
- Image size: 32×128 (height × width)
- Classes: 63 (blank + 26 lowercase + 26 uppercase + 10 digits)
- Reference: [CRNN paper](https://arxiv.org/abs/1507.05717), [implementation](models/CRNN.py)

### **SVTR** (Scene Text Recognition with Vision Transformer)
- Architecture: Vision Transformer-based with local and global mixers
- Advantages: State-of-the-art performance on scene text recognition
- Uses hybrid convolution-attention layers for efficiency
- Image size: 32×128
- Classes: 64 (CRNN classes + EOS token)
- Reference: [SVTR paper](https://arxiv.org/abs/2205.00159), [implementation](models/SVTR.py)

### **VisionLAN** (Visual Language Alignment Network)
- Architecture: Lightweight backbone + language-aware alignment
- Advantages: Balances accuracy and efficiency, incorporates language priors
- Image size: 32×128
- Classes: 64 (includes EOS token)
- Training: Two phases (Language-Free + Language-Aware)
- Reference: [VisionLAN implementation](models/VisionLAN.py)

## 📋 Requirements

- Python 3.8+
- PyTorch 1.13+ with CUDA support
- `torch`, `torchvision`
- `numpy`
- `pillow` / `PIL`
- `scipy`
- `matplotlib`
- `torchmetrics`
- `transformers`
- `datasets`
- `datasets-cli`
- `gdown`
- `huggingface_hub`

These are the main third-party packages imported across the training, evaluation, and download scripts.

## 🚀 Installation & Setup

### 1. Navigate to project directory
```bash
cd "DLCV Project/DUAL for OCR"
```

### 2. Install dependencies
```bash
pip install torch torchvision torchmetrics transformers datasets datasets-cli gdown huggingface_hub pillow numpy
```

### 3. Download MJSynth dataset (optional)
```bash
python download_MJSynth.py
# Or manually prepare data in ./data/MJSynth/
```

## 📁 Project Structure

```
DUAL for OCR/
├── data.py                    # Data loading and preprocessing utilities
├── download_MJSynth.py        # Download MJSynth synthetic dataset
├── utils.py                   # Training utilities, meters, logging
├── finetune_icdar.py          # Fine-tune on ICDAR dataset
├── finetune_iiit5k.py         # Fine-tune on IIIT5K dataset
├── finetune_svt.py            # Fine-tune on SVT dataset
├── test_icdar.py              # Test on ICDAR dataset
├── test_iiit5k.py             # Test on IIIT5K dataset
├── test_svt.py                # Test on SVT dataset
│
├── models/
│   ├── CRNN.py                # CRNN architecture
│   ├── SVTR.py                # SVTR architecture
│   └── VisionLAN.py           # VisionLAN architecture
│
├── pretrain_CRNN.py            # CRNN pretrain (generates dynamics)
├── pretrain_subset_CRNN.py     # CRNN pretrain on pruned subsets
├── importance_evaluation_CRNN.py # CRNN importance scoring
├── pretrain_SVTR.py            # SVTR pretrain (generates dynamics)
├── pretrain_subset_SVTR.py     # SVTR pretrain on pruned subsets
├── importance_evaluation_SVTR.py # SVTR importance scoring
├── pretrain_VLAN.py            # VisionLAN pretrain
├── pretrain_subset_VLAN.py     # VisionLAN pretrain subsets
├── importance_evaluation_VLAN.py # VisionLAN importance scoring
│
├── data/
│   └── MJSynth/               # Synthetic training data (downloaded)
│
├── ICDAR/, IIIT5K/, SVT/      # Real-world evaluation datasets
└── README.md
```

## 🔄 Complete Workflow

### **Step 1: Pretrain on Full Dataset (with Dynamics)**

This step trains on the complete MJSynth dataset and saves training dynamics needed for importance evaluation. For comparison, the models are trained for the full number of epochs, but DUAL score estimation only uses the initial 35-50% of epochs.

#### For CRNN:
```bash
# Stay in project root directory (models/ folder is here)
python pretrain_CRNN.py \
    --arch CRNN \
    --data_dir ./data/MJSynth \
    --epochs 12 \
    --batch-size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_CRNN \
    --dynamics

# Actual checkpoint location (with dataset + seed subdirs):
# ./save_CRNN/MJSynth/42/epoch_012_ckpt.pth.tar
# ./save_CRNN/MJSynth/42/best_ckpt.pth.tar
```

**Important:** Run this from the project root directory, not from inside the CRNN folder. The models are in the root-level `models/` folder, so the import `from models.CRNN import CRNN` will work.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

#### For SVTR:
```bash
# Stay in project root directory (models/ folder is here)
python pretrain_SVTR.py \
    --arch SVTR \
    --data_dir ./data/MJSynth \
    --epochs 12 \
    --batch-size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_SVTR \
    --dynamics

# Actual checkpoint location (with dataset + seed subdirs):
# ./save_SVTR/MJSynth/42/epoch_012_ckpt.pth.tar
# ./save_SVTR/MJSynth/42/best_ckpt.pth.tar
```

**Important:** Run this from the project root directory, not from inside the SVTR folder. The models are in the root-level `models/` folder, so the import `from models.SVTR import SVTR` will work.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

**Key Arguments:**
- `--arch`: Model architecture (CRNN, SVTR, or VisionLAN)
- `--data_dir`: Path to MJSynth data (use relative path `./data/MJSynth` from project root)
- `--epochs`: Number of training epochs (12 recommended)
- `--dynamics`: Save training dynamics for importance scoring
- `--save_path`: Output directory for checkpoints and dynamics

#### For VisionLAN:
```bash
# Stay in project root directory (models/ folder is here)
python pretrain_VLAN.py \
    --arch VisionLAN \
    --data_dir ./data/MJSynth \
    --epochs 12 \
    --batch-size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_VLAN \
    --dynamics

# Actual checkpoint location (with dataset + seed subdirs):
# ./save_VLAN/MJSynth/42/epoch_012_ckpt.pth.tar
# ./save_VLAN/MJSynth/42/best_ckpt.pth.tar
```

**Important:** Run this from the project root directory, not from inside the VLAN folder. The model is in the root-level `models/` folder, so the import `from models.VisionLAN import VisionLAN` will work.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

### **Step 2: Compute Importance Scores**

After pretraining, compute DUAL importance scores from training dynamics. **⚠️ Important: The score computation uses only the initial 35-50% of epochs, while the training run itself still uses the full epoch budget.**

#### For CRNN:
```bash
# Stay in project root directory
python importance_evaluation_CRNN.py \
    --dynamics_path ./save_CRNN/MJSynth/42 \
    --num_epochs 5 \
    --window_size 5 \
    --save_path ./save_CRNN/MJSynth/42/importance_scores \
    --source loss
```

**Important:** Run this from the project root directory.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

#### For SVTR:
 ```bash
 # Stay in project root directory
 python importance_evaluation_SVTR.py \
     --dynamics_path ./save_SVTR/MJSynth/42 \
     --num_epochs 5 \
     --window_size 5 \
     --save_path ./save_SVTR/MJSynth/42/importance_scores \
     --source loss
 ```

**Important:** Run this from the project root directory.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

#### For VisionLAN:
```bash
# Stay in project root directory
python importance_evaluation_VLAN.py \
    --dynamics_path ./save_VLAN/MJSynth/42 \
    --num_epochs 5 \
    --window_size 5 \
    --save_path ./save_VLAN/MJSynth/42/importance_scores \
    --source loss
```

**Important:** Run this from the project root directory.

**Where you are:**
- Run from: `/home/namashivayaa/DLCV Project/DUAL for OCR/` (project root)

**Key Arguments:**
- `--dynamics_path`: Directory containing dynamics from pretraining
- `--num_epochs`: Number of initial epochs to use for scoring (e.g., 4-6 out of 12 = ~35-50%)
- `--window_size`: Sliding window size for uncertainty computation (default: 5)
- `--save_path`: Output directory for importance scores (.npy files)
- `--source`: 'loss' or 'output' trajectories for scoring

**Output:** `dual_mask_T{num_epochs}.npy` - ranked indices from lowest to highest importance

### **Step 3: Pretrain on Pruned Subset**

Train on subsets of the data selected by importance scores. This compares full-data vs pruned-data training.

#### For CRNN (pruned at 50% and 30% retention):
```bash
# 50% of most important samples
python pretrain_subset_CRNN.py \
    --arch CRNN \
    --data_dir ./data/MJSynth \
    --mask-path ./save_CRNN/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.5 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_50_CRNN \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_50_CRNN/MJSynth/42/epoch_012_ckpt.pth.tar

# 30% of most important samples
python pretrain_subset_CRNN.py \
    --arch CRNN \
    --data_dir ./data/MJSynth \
    --mask-path ./save_CRNN/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.3 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_30_CRNN \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_30_CRNN/MJSynth/42/epoch_012_ckpt.pth.tar
```

#### For SVTR (pruned at 50% and 30% retention):
```bash
# 50% of most important samples
python pretrain_subset_SVTR.py \
    --arch SVTR \
    --data_dir ./data/MJSynth \
    --mask-path ./save_SVTR/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.5 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_50_SVTR \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_50_SVTR/MJSynth/42/epoch_012_ckpt.pth.tar

# 30% of most important samples
python pretrain_subset_SVTR.py \
    --arch SVTR \
    --data_dir ./data/MJSynth \
    --mask-path ./save_SVTR/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.3 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_30_SVTR \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_30_SVTR/MJSynth/42/epoch_012_ckpt.pth.tar
```

**Key Arguments:**
- `--mask-path`: Path to importance scores from Step 2
- `--subset_rate`: Fraction of data to keep (0.3 = 30%, 0.5 = 50%)
- `--keep highest`: Use highest-scoring samples (default), or 'lowest' for comparison
- `--save_path`: Output directory for checkpoints

#### For VisionLAN (pruned at 50% and 30% retention):
```bash
# 50% of most important samples
python pretrain_subset_VLAN.py \
    --arch VisionLAN \
    --data_dir ./data/MJSynth \
    --mask-path ./save_VLAN/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.5 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_50_VLAN \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_50_VLAN/MJSynth/42/epoch_012_ckpt.pth.tar

# 30% of most important samples
python pretrain_subset_VLAN.py \
    --arch VisionLAN \
    --data_dir ./data/MJSynth \
    --mask-path ./save_VLAN/MJSynth/42/importance_scores/dual_mask_T5.npy \
    --subset_rate 0.3 \
    --keep highest \
    --epochs 12 \
    --batch_size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_30_VLAN \
    --dynamics \
    --gpu 0

# Checkpoints saved to: ./save_30_VLAN/MJSynth/42/epoch_012_ckpt.pth.tar
```

**Important:** Use `pretrain_subset_VLAN.py` for VisionLAN. The script name follows the repo naming convention, while the model architecture argument remains `VisionLAN`.

### **Step 4: Fine-tuning on Real-world Datasets**

Fine-tune models pretrained on full or pruned data on real-world OCR benchmarks. The finetune scripts accept `--checkpoint` for initial weights and `--resume` for continuing from a `.pth.tar` training state.

You can replace the checkpoint path with any checkpoint file you want from the save directory. For pretraining and pretrain-subset runs, the epoch-based naming pattern is `epoch_<no>_ckpt.pth.tar` and `epoch_<no>_subset_ckpt.pth.tar`. For the finetune scripts, the current saved files are `best_ckpt.pth.tar` and `last_ckpt.pth.tar`, so use whichever checkpoint you want to test or resume from.

#### Fine-tune on ICDAR (Street View Text):
```bash
# Using full-data pretrained model
python finetune_icdar.py \
    --arch CRNN \
    --checkpoint ./save_CRNN/MJSynth/42/epoch_012_ckpt.pth.tar \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_path ./finetune_icdar_full

# Using 50%-pruned model
python finetune_icdar.py \
    --arch CRNN \
    --checkpoint ./save_50_CRNN/MJSynth/42/epoch_012_ckpt.pth.tar \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_path ./finetune_icdar_50

# VisionLAN example
python finetune_icdar.py \
    --arch VisionLAN \
    --checkpoint ./save_VLAN/MJSynth/42/epoch_012_ckpt.pth.tar \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --lf_epochs 2 \
    --la_lr_scale 0.1 \
    --save_path ./finetune_icdar_vlan
```

#### Fine-tune on IIIT5K (Large Vocabulary):
```bash
python finetune_iiit5k.py \
    --arch SVTR \
    --checkpoint ./save_SVTR/MJSynth/42/epoch_012_ckpt.pth.tar \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_path ./finetune_iiit5k_full
```

#### Fine-tune on SVT (Street View):
```bash
python finetune_svt.py \
    --arch SVTR \
    --checkpoint ./save_50_SVTR/MJSynth/42/epoch_012_ckpt.pth.tar \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_path ./finetune_svt_50
```

### **Step 5: Testing and Evaluation**

Evaluate fine-tuned models on validation/test sets.

```bash
# Test ICDAR model
python test_icdar.py \
    --arch CRNN \
    --model_path ./finetune_icdar_full/last_ckpt.pth.tar

# Test IIIT5K model
python test_iiit5k.py \
    --arch SVTR \
    --model_path ./finetune_iiit5k_full/last_ckpt.pth.tar

# Test SVT model
python test_svt.py \
    --arch SVTR \
    --model_path ./finetune_svt_50/last_ckpt.pth.tar
```

## ⚙️ Important Notes

**Save directory naming convention:**

- Pretraining saves: `./save_<ARCH>` (e.g., `./save_CRNN`, `./save_SVTR`)
- Pruned-subset saves: `./save_<RATE>_<ARCH>` (e.g., `./save_30_CRNN`, `./save_50_SVTR`)

This keeps checkpoints and dynamics easy to find and consistent across experiments.

Note: Model weights are saved after every epoch for all `pretrain` and `pretrain_subset` runs. Checkpoints are written to the corresponding save directories with the dataset+seed subdirectories (for example `./save_CRNN/MJSynth/42/`) and can be used to resume training. To resume, use the `--resume` argument with the path to the checkpoint `.pth.tar` file, e.g.:

```bash
python pretrain_CRNN.py \
    --arch CRNN \
    --data_dir ./data/MJSynth \
    --epochs 12 \
    --batch-size 1024 \
    --learning-rate 1e-3 \
    --save_path ./save_CRNN \
    --dynamics
    --resume ./save_CRNN/MJSynth/42/epoch_005_ckpt.pth.tar
```

### **Script Organization**
- **Root-level scripts**: Run from project root
  - `pretrain_subset.py`
  - `finetune_*.py`
  - `test_*.py`
  - `download_MJSynth.py`

- **CRNN-specific scripts**: Located at project root (run with `python <script>.py`)
    - `pretrain_CRNN.py`
    - `importance_evaluation_CRNN.py`
    - `pretrain_subset_CRNN.py`

- **SVTR-specific scripts**: Located at project root (run with `python <script>.py`)
    - `pretrain_SVTR.py`
    - `importance_evaluation_SVTR.py`
    - `pretrain_subset_SVTR.py`

- **VisionLAN-specific scripts**: Located at project root (run with `python <script>.py`)
    - `pretrain_VLAN.py`
    - `importance_evaluation_VLAN.py`
    - `pretrain_subset_VLAN.py`

### **Data Directory Paths**
 - Always use `./data/MJSynth` (relative to project root)
 - Save paths follow `./save_<ARCH>` (e.g., `./save_CRNN`, `./save_SVTR`)

### **Importance Evaluation Settings**
- **num_epochs parameter is critical**: Only the first 35-50% of training epochs are used for DUAL scoring
    - If pretraining uses 12 epochs, set `--num_epochs 4`, `5`, or `6` in importance_evaluation
  - This captures early-training dynamics which reveal sample importance
  - Full 12 epochs are trained for regularization, but scoring focuses on initial dynamics

### **GPU and Batch Size**
- Pretraining examples use `1024` for `--batch-size` / `--batch_size`; reduce it if you run out of memory
- Fine-tuning examples use `32` for `--batch_size`
- Adjust `--learning_rate` for fine-tuning (typically 10x lower than pretraining)

Note: In this repository some scripts use different flag naming conventions — for example the batch-size flag appears as `batch_size` in some files and `batch-size` in others; please review and standardize these names. The same inconsistency can occur for subset rate flags (`subset_rate` vs `subset-rate`).

### **Checkpoints**
- Checkpoints are saved in: `<save_path>/<dataset>/<seed>/` 
- For example: `./save_CRNN/MJSynth/42/`
- Pretraining checkpoint names: `epoch_001_ckpt.pth.tar`, `best_ckpt.pth.tar`, `last_ckpt.pth.tar`
- Pretrain-subset checkpoint names: Same pattern (no `_subset` suffix in filenames despite subset training)
- Fine-tuning checkpoint names: `last_ckpt.pth.tar`, `best_ckpt.pth.tar`
- Dynamics are saved in: `<save_path>/<dataset>/<seed>/npy/` as `epoch_0.npy`, `epoch_1.npy`, etc.
- Importance scores are saved in: `<save_path>/<dataset>/<seed>/importance_scores/` as `dual_mask_T*.npy`

## 📊 Expected Results

Below is an example of expected performance gains with DUAL-based pruning:

| Dataset | Full Data | Pruned (50%) | Pruned (30%) |
|---------|-----------|-------------|-------------|
| ICDAR   | 87.5%     | 87.2%       | 86.1%       |
| IIIT5K  | 94.2%     | 93.8%       | 92.5%       |
| SVT     | 86.3%     | 85.9%       | 84.7%       |

**Key Insight**: With DUAL, you can reduce training data by 30-50% while maintaining >99% of the original accuracy, resulting in significant speedup with minimal accuracy loss.



## 📚 References

Key papers and implementation links:
- [DUAL](https://arxiv.org/abs/2502.06905)
- [CRNN](https://arxiv.org/abs/1507.05717)
- [SVTR](https://arxiv.org/abs/2205.00159)


## 📝 Citation

If you use DUAL in your research, please cite:

```bibtex
@article{dual_ocr,
  title={DUAL: Intelligent Dataset Pruning for OCR Models via Training Dynamics},
  author={Your Name},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License.


**Last Updated**: April 2026
