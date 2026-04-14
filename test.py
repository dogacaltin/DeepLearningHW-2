"""
Evaluation script for corruption robustness, adversarial attacks, and
adversarial transferability.

Covers:
1. CIFAR-10-C corruption benchmark evaluation.
2. PGD adversarial robustness under L-inf and L2 norms.
3. Grad-CAM and t-SNE visualization of adversarial samples.
4. Adversarial transferability from teacher to student.

Usage:
    python test.py --mode corruption --checkpoint ./checkpoints/resnet18_standard_best.pth
    python test.py --mode adversarial --checkpoint ./checkpoints/resnet18_standard_best.pth
    python test.py --mode transferability --teacher_checkpoint ... --student_checkpoint ...
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from adversarial import evaluate_adversarial, pgd_attack
from datasets import (
    ALL_CORRUPTIONS,
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10c_loader,
    get_normalize_transform,
    get_test_transform,
)
from models import get_model
from parameters import AdversarialParams, TestParams, VisualizationParams
from visualizations import visualize_gradcam, visualize_tsne

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------- Corruption evaluation ----------


def evaluate_corruption_robustness(
    model: nn.Module,
    cifar10c_root: str,
    device: torch.device,
    params: TestParams,
) -> Dict[str, Dict[int, float]]:
    """Evaluate model accuracy on all CIFAR-10-C corruptions.

    Args:
        model: Trained classifier.
        cifar10c_root: Path to the CIFAR-10-C data directory.
        device: Torch device.
        params: Test configuration.

    Returns:
        Nested dict mapping corruption_name -> {severity -> accuracy%}.
    """
    model.eval().to(device)
    corruptions = params.corruption_types or ALL_CORRUPTIONS
    results: Dict[str, Dict[int, float]] = {}

    for corruption in corruptions:
        results[corruption] = {}
        for severity in params.severity_levels:
            try:
                loader = get_cifar10c_loader(
                    root=cifar10c_root,
                    corruption=corruption,
                    severity=severity,
                    batch_size=params.batch_size,
                    num_workers=params.num_workers,
                )
            except FileNotFoundError:
                print(f"  Skipping {corruption} severity {severity} (file not found)")
                continue

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = 100.0 * correct / total
            results[corruption][severity] = acc
            print(f"  {corruption:>25s} | severity {severity} | Acc: {acc:.2f}%")

    return results


# ---------- Adversarial evaluation ----------


def get_unnormalized_test_loader(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
) -> DataLoader:
    """Get a CIFAR-10 test loader WITHOUT normalization (pixel range [0,1]).

    This is needed for PGD attacks which operate in raw pixel space.

    Args:
        batch_size: Batch size.
        num_workers: Number of workers.
        data_root: Data directory.

    Returns:
        DataLoader with ToTensor-only transform.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


class NormalizedModel(nn.Module):
    """Wrapper that prepends normalization to a model.

    This allows PGD to operate on [0,1] pixel-space images while the
    underlying model still receives normalized inputs.

    Attributes:
        model: The base classifier.
        mean: Channel-wise mean as a (1, C, 1, 1) tensor.
        std: Channel-wise std as a (1, C, 1, 1) tensor.
    """

    def __init__(self, model: nn.Module, mean: Tuple, std: Tuple) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and forward-pass.

        Args:
            x: Input images in [0, 1] range.

        Returns:
            Classification logits.
        """
        return self.model((x - self.mean) / self.std)


def run_adversarial_evaluation(
    model: nn.Module,
    device: torch.device,
    test_params: TestParams,
    save_dir: str = "./visualizations",
) -> None:
    """Run full adversarial evaluation: PGD-Linf, PGD-L2, Grad-CAM, t-SNE.

    Args:
        model: Trained classifier.
        device: Torch device.
        test_params: Test configuration.
        save_dir: Directory for saving visualizations.
    """
    norm_model = NormalizedModel(model, CIFAR10_MEAN, CIFAR10_STD).to(device)
    norm_model.eval()

    raw_loader = get_unnormalized_test_loader(
        batch_size=test_params.batch_size, num_workers=test_params.num_workers
    )

    # --- PGD-Linf ---
    print("\n--- PGD20 L-inf (eps=4/255) ---")
    linf_params = AdversarialParams(
        norm="Linf", epsilon=4.0 / 255.0,
        step_size=1.0 / 255.0, num_steps=20,
    )
    clean_acc, adv_acc = evaluate_adversarial(
        norm_model, raw_loader,
        linf_params.epsilon, linf_params.step_size,
        linf_params.num_steps, linf_params.norm, device,
    )
    print(f"Clean Acc: {clean_acc:.2f}% | Adv Acc (Linf): {adv_acc:.2f}%")

    # --- PGD-L2 ---
    print("\n--- PGD20 L2 (eps=0.25) ---")
    l2_params = AdversarialParams(
        norm="L2", epsilon=0.25,
        step_size=0.05, num_steps=20,
    )
    _, adv_acc_l2 = evaluate_adversarial(
        norm_model, raw_loader,
        l2_params.epsilon, l2_params.step_size,
        l2_params.num_steps, l2_params.norm, device,
    )
    print(f"Adv Acc (L2): {adv_acc_l2:.2f}%")

    # --- Grad-CAM and t-SNE on a subset ---
    print("\nGenerating Grad-CAM and t-SNE visualizations...")
    vis_params = VisualizationParams(save_dir=save_dir)

    # Collect a batch for visualization
    all_clean, all_labels = [], []
    for imgs, labs in raw_loader:
        all_clean.append(imgs)
        all_labels.append(labs)
        if sum(x.size(0) for x in all_clean) >= vis_params.tsne_num_samples:
            break

    clean_imgs = torch.cat(all_clean, dim=0)[: vis_params.tsne_num_samples]
    labels_tensor = torch.cat(all_labels, dim=0)[: vis_params.tsne_num_samples]

    # Generate adversarial examples for visualization
    adv_imgs_list = []
    for i in range(0, len(clean_imgs), test_params.batch_size):
        batch = clean_imgs[i : i + test_params.batch_size].to(device)
        batch_labels = labels_tensor[i : i + test_params.batch_size].to(device)
        adv_batch = pgd_attack(
            norm_model, batch, batch_labels,
            linf_params.epsilon, linf_params.step_size,
            linf_params.num_steps, "Linf",
        )
        adv_imgs_list.append(adv_batch.cpu())
    adv_imgs = torch.cat(adv_imgs_list, dim=0)

    # Grad-CAM
    visualize_gradcam(
        model, clean_imgs, adv_imgs, labels_tensor,
        CIFAR10_CLASSES, save_dir, device,
        num_samples=vis_params.num_gradcam_samples,
    )

    # t-SNE (needs normalized images for feature extraction)
    normalize = get_normalize_transform()
    clean_norm = torch.stack([normalize(img) for img in clean_imgs])
    adv_norm = torch.stack([normalize(img) for img in adv_imgs])

    visualize_tsne(
        model, clean_norm, adv_norm, labels_tensor,
        CIFAR10_CLASSES, save_dir, device, vis_params,
    )


# ---------- Transferability ----------


def evaluate_transferability(
    teacher: nn.Module,
    student: nn.Module,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[float, float]:
    """Test adversarial transferability from teacher to student.

    Generates PGD-Linf adversarial examples targeting the teacher model,
    then evaluates both teacher and student accuracy on those samples.

    Args:
        teacher: Teacher model.
        student: Student model.
        device: Torch device.
        batch_size: Batch size.
        num_workers: Number of workers.

    Returns:
        Tuple of (teacher_adv_accuracy, student_adv_accuracy).
    """
    teacher_norm = NormalizedModel(teacher, CIFAR10_MEAN, CIFAR10_STD).to(device)
    student_norm = NormalizedModel(student, CIFAR10_MEAN, CIFAR10_STD).to(device)
    teacher_norm.eval()
    student_norm.eval()

    raw_loader = get_unnormalized_test_loader(batch_size, num_workers)

    eps = 4.0 / 255.0
    step_size = 1.0 / 255.0
    num_steps = 20

    teacher_correct = 0
    student_correct = 0
    total = 0

    print("\n--- Adversarial Transferability (Teacher -> Student) ---")
    print(f"PGD20 L-inf, eps=4/255, generated on teacher")

    for images, labels in raw_loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples on the teacher
        adv_images = pgd_attack(
            teacher_norm, images, labels, eps, step_size, num_steps, "Linf"
        )

        with torch.no_grad():
            teacher_preds = teacher_norm(adv_images).argmax(1)
            student_preds = student_norm(adv_images).argmax(1)

        teacher_correct += (teacher_preds == labels).sum().item()
        student_correct += (student_preds == labels).sum().item()
        total += labels.size(0)

    teacher_acc = 100.0 * teacher_correct / total
    student_acc = 100.0 * student_correct / total

    print(f"Teacher adv accuracy: {teacher_acc:.2f}%")
    print(f"Student adv accuracy (transfer): {student_acc:.2f}%")

    return teacher_acc, student_acc
