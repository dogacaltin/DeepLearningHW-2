"""
Training script for CIFAR-10 fine-tuning with optional AugMix.

Supports standard fine-tuning and AugMix training with Jensen-Shannon
Divergence (JSD) consistency regularization.

Usage:
    python train.py --model_name resnet18 --epochs 25
    python train.py --model_name resnet18 --epochs 25 --use_augmix
"""

import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import get_dataloaders
from models import get_model
from parameters import TrainParams


def jsd_loss(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
) -> torch.Tensor:
    """Compute Jensen-Shannon Divergence loss for AugMix.

    Encourages consistent predictions across clean and augmented views.

    Args:
        logits_clean: Logits from the clean image (B, C).
        logits_aug1: Logits from the first AugMix view (B, C).
        logits_aug2: Logits from the second AugMix view (B, C).

    Returns:
        Scalar JSD loss.
    """
    p_clean = F.softmax(logits_clean, dim=1).clamp(min=1e-8)
    p_aug1 = F.softmax(logits_aug1, dim=1).clamp(min=1e-8)
    p_aug2 = F.softmax(logits_aug2, dim=1).clamp(min=1e-8)

    # Mixture distribution
    m = (p_clean + p_aug1 + p_aug2) / 3.0

    jsd = (
        F.kl_div(m.log(), p_clean, reduction="batchmean")
        + F.kl_div(m.log(), p_aug1, reduction="batchmean")
        + F.kl_div(m.log(), p_aug2, reduction="batchmean")
    ) / 3.0

    return jsd


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_augmix: bool = False,
    jsd_weight: float = 12.0,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The network to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        criterion: Classification loss (CrossEntropy).
        device: Torch device.
        use_augmix: If True, expects AugMixDataset batches and adds JSD loss.
        jsd_weight: Weight for the JSD regularization term.

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if use_augmix:
            clean, aug1, aug2, labels = batch
            clean, aug1, aug2 = clean.to(device), aug1.to(device), aug2.to(device)
            labels = labels.to(device)

            logits_clean = model(clean)
            logits_aug1 = model(aug1)
            logits_aug2 = model(aug2)

            ce_loss = criterion(logits_clean, labels)
            jsd = jsd_loss(logits_clean, logits_aug1, logits_aug2)
            loss = ce_loss + jsd_weight * jsd

            preds = logits_clean.argmax(dim=1)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on the test set.

    Args:
        model: The network.
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(params: TrainParams) -> str:
    """Full training loop for CIFAR-10 fine-tuning.

    Args:
        params: Training configuration.

    Returns:
        Path to the best saved checkpoint.
    """
    torch.manual_seed(params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        use_augmix=params.use_augmix,
        augmix_severity=params.augmix_severity,
        augmix_width=params.augmix_width,
        augmix_depth=params.augmix_depth,
        augmix_alpha=params.augmix_alpha,
    )

    # Model
    model = get_model(params.model_name, params.pretrained, params.num_classes)
    model.to(device)

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay,
    )
    if params.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    os.makedirs(params.save_path, exist_ok=True)
    tag = "augmix" if params.use_augmix else "standard"
    best_acc = 0.0
    best_path = os.path.join(params.save_path, f"{params.model_name}_{tag}_best.pth")

    print(f"\n{'='*60}")
    print(f"Fine-tuning {params.model_name} ({'AugMix' if params.use_augmix else 'Standard'})")
    print(f"{'='*60}")

    for epoch in range(1, params.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, params.use_augmix
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch}/{params.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best! Saved to {best_path}")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    return best_path
