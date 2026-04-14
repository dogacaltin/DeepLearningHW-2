"""
Knowledge distillation training module.

Implements knowledge distillation from a teacher (e.g., AugMix-trained
ResNet-50) to a student (e.g., ResNet-18) using soft-target KL divergence
combined with hard-label cross-entropy loss.
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
from parameters import DistillationParams
from train import evaluate


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Compute the knowledge distillation loss.

    Combines soft-target KL divergence with hard-label cross-entropy.

    Args:
        student_logits: Student network output logits (B, C).
        teacher_logits: Teacher network output logits (B, C).
        labels: Ground-truth class indices (B,).
        temperature: Softmax temperature for soft targets.
        alpha: Weight for distillation loss (1 - alpha for CE loss).

    Returns:
        Scalar combined loss.
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


def train_distillation(params: DistillationParams) -> str:
    """Run knowledge distillation training.

    Args:
        params: Distillation configuration.

    Returns:
        Path to the best student checkpoint.
    """
    torch.manual_seed(params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distillation on: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(
        batch_size=params.batch_size,
        num_workers=params.num_workers,
    )

    # Teacher (frozen)
    teacher = get_model(
        params.teacher_model_name,
        pretrained=False,
        num_classes=10,
        checkpoint_path=params.teacher_checkpoint,
    )
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student
    student = get_model(params.student_model_name, pretrained=True, num_classes=10)
    student.to(device)

    optimizer = optim.SGD(
        student.parameters(),
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)

    os.makedirs(params.save_path, exist_ok=True)
    best_acc = 0.0
    best_path = os.path.join(
        params.save_path,
        f"student_{params.student_model_name}_from_{params.teacher_model_name}_best.pth",
    )

    print(f"\n{'='*60}")
    print(f"Knowledge Distillation: {params.teacher_model_name} -> {params.student_model_name}")
    print(f"Temperature={params.temperature}, Alpha={params.alpha}")
    print(f"{'='*60}")

    for epoch in range(1, params.epochs + 1):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(
                student_logits, teacher_logits, labels, params.temperature, params.alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(student, test_loader, device)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch}/{params.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), best_path)
            print(f"  -> New best! Saved to {best_path}")

    print(f"\nBest student test accuracy: {best_acc:.2f}%")
    return best_path
