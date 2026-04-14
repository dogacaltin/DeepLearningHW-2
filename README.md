# CS515 HW2: Data Augmentation and Adversarial Samples

## Overview

This project evaluates the robustness of fine-tuned deep neural networks on CIFAR-10, covering:

1. **Corruption Robustness** — Evaluate on CIFAR-10-C benchmark (15 corruption types × 5 severities)
2. **AugMix Training** — Fine-tune with AugMix + JSD consistency regularization
3. **Adversarial Robustness** — PGD-20 attacks under L∞ (ε=4/255) and L2 (ε=0.25) norms
4. **Visualization** — Grad-CAM heatmaps and t-SNE embeddings for clean vs. adversarial samples
5. **Knowledge Distillation** — AugMix-trained teacher → student distillation
6. **Adversarial Transferability** — Generate adversarial samples on teacher, test on student

## Project Structure

```
cs515_hw2/
├── main.py            # CLI entry point (argparse)
├── train.py           # Training loop (standard + AugMix/JSD)
├── test.py            # Evaluation (corruption, adversarial, transferability)
├── parameters.py      # Dataclass configs (TrainParams, TestParams, etc.)
├── datasets.py        # CIFAR-10 loaders, AugMix, CIFAR-10-C
├── adversarial.py     # PGD attack (Linf / L2)
├── distillation.py    # Knowledge distillation training
├── visualizations.py  # Grad-CAM and t-SNE
├── models/
│   ├── __init__.py
│   └── resnet.py      # ResNet factory for CIFAR-10
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Download CIFAR-10-C from https://zenodo.org/record/2535967 and extract to `./data/CIFAR-10-C/`.

## Usage

### Run all experiments sequentially
```bash
python main.py --mode all --model_name resnet18 --teacher_model resnet50 --epochs 25
```

### Individual steps

```bash
# 1. Standard fine-tuning
python main.py --mode train --model_name resnet18

# 2. AugMix fine-tuning
python main.py --mode train --model_name resnet18 --use_augmix

# 3. CIFAR-10-C evaluation
python main.py --mode test_corruption --model_name resnet18 \
    --checkpoint ./checkpoints/resnet18_standard_best.pth

# 4. Adversarial evaluation + Grad-CAM + t-SNE
python main.py --mode test_adversarial --model_name resnet18 \
    --checkpoint ./checkpoints/resnet18_standard_best.pth

# 5. Knowledge distillation
python main.py --mode distill \
    --teacher_model resnet50 \
    --teacher_checkpoint ./checkpoints/resnet50_augmix_best.pth \
    --student_model resnet18

# 6. Adversarial transferability
python main.py --mode transferability \
    --teacher_model resnet50 \
    --teacher_checkpoint ./checkpoints/resnet50_augmix_best.pth \
    --student_model resnet18 \
    --student_checkpoint ./checkpoints/student_resnet18_from_resnet50_best.pth
```

## Key Implementation Details

- **AugMix**: 13 PIL-based augmentation ops, Dirichlet chain mixing, JSD consistency loss (weight=12.0)
- **PGD Attack**: Operates in [0,1] pixel space; model wrapped with `NormalizedModel` for proper normalization
- **Grad-CAM**: Hooks into `layer4[-1]` of ResNet; selects samples where adversarial perturbation flips the prediction
- **t-SNE**: Extracts penultimate-layer (pre-FC) features; plots clean vs. adversarial distributions
- **Distillation**: Soft-target KL divergence (T=4.0, α=0.7) combined with hard-label CE loss

## References

1. Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions", ICLR 2019
2. Hendrycks et al., "AugMix: A Simple Method to Improve Robustness and Uncertainty", ICLR 2020
3. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
