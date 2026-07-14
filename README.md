# Defending Backdoor Attacked Images with Deep Hashing

> EfficientNetV2 + CSQ Deep Hashing + GMM Adaptive Threshold  
> 4 Datasets × 6 Attacks × Multiple Baselines

## Paper

- **Title**: Defending Backdoor Attacked Images with Deep Hashing
- **Authors**: Yunchun Zhang, Hao Liu, Feiyang Huang, Mingxiong Zhao
- **Affiliation**: National Pilot School of Software, Yunnan University

## Method

Three-stage pipeline:
1. **Backbone Training** — EfficientNetV2 classifier
2. **Hash Model Training** — CSQ (Central Similarity Quantization) with Hadamard centers + center loss + quantization loss
3. **GMM Detection** — Bimodal Gaussian fitting on Hamming distance distribution → intersection threshold

## Datasets & Attacks

| Dataset | Classes |
|---------|---------|
| MNIST | 10 |
| CIFAR-10 | 10 |
| GTSRB | 43 |
| ImageNet-100 | 100 |

| Attack | Type |
|--------|------|
| BadNets | Patch trigger |
| Blended | Blended trigger |
| SIG | Sinusoidal signal |
| WaNet | Warping-based |
| Refool | Reflection-based |
| Input-aware | Dynamic trigger |

## Project Structure

```
├── hash_*.py          # Hash training scripts (CSQ, DBDH, DPN, HashNet, etc.)
├── model_*.py         # Backbone networks (EfficientNetV2, ResNet, ConvNeXt, etc.)
├── *_detection.py     # Backdoor detection per dataset × attack (24 scripts)
├── to_*.py            # Attack image generation (blended, sig, wanet, refool, dynamic)
├── abl*.py            # ABL defense implementation
├── asr_*.py           # Attack Success Rate measurement
├── CW*.py FGSM*.py    # Adversarial training defense
├── tsne*.py           # t-SNE visualization
├── gmm.py             # GMM threshold detection
├── utils/             # Core utilities (data loading, mAP, config)
├── network.py         # Hash layer definitions
├── data/              # Dataset indices & labels (Excel + txt)
├── dataset/           # Image files
└── save/ log/         # Model checkpoints & training logs
```

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA RTX 50-series (Blackwell sm_120) |

### Dependencies

```bash
pip install -r requirements.txt
```

Full original environment: see `environment.yml`

## Quick Start

```bash
# Activate environment
conda activate deephash

# Train a hash model (CSQ, 16-bit, GTSRB)
python CSQ_3_16.py

# Generate attack images
python "to blended images.py"

# Run detection
python cifar10_blended_detection.py
```

## Notes

- All paths have been adapted for local Windows (`D:/deephash_original/`)
- Batch size set to 16 for 8GB VRAM (original scripts used 64 for server GPUs)
- Epochs set to 50 for quick iteration (original: 150)
- GPU device set to `cuda:0` (original scripts used multi-GPU indices)
