"""
Projected Gradient Descent (PGD) adversarial attack.

Implements PGD with both L-infinity and L2 norm constraints, following
Madry et al., "Towards Deep Learning Models Resistant to Adversarial
Attacks", ICLR 2018.
"""

from typing import Tuple

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    num_steps: int = 20,
    norm: str = "Linf",
    random_start: bool = True,
) -> torch.Tensor:
    """Generate adversarial examples using PGD.

    Args:
        model: Target classifier (must be in eval mode).
        images: Clean input images of shape (B, C, H, W).
        labels: Ground-truth labels of shape (B,).
        epsilon: Maximum perturbation budget.
        step_size: Step size (alpha) per PGD iteration.
        num_steps: Number of PGD iterations.
        norm: Lp norm constraint, either 'Linf' or 'L2'.
        random_start: If True, initialize perturbation uniformly at random.

    Returns:
        Adversarial images of the same shape as *images*, clamped to [0, 1].

    Raises:
        ValueError: If *norm* is not 'Linf' or 'L2'.
    """
    model.eval()
    images = images.clone().detach()
    adv_images = images.clone().detach()

    if random_start:
        if norm == "Linf":
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        elif norm == "L2":
            delta = torch.randn_like(adv_images)
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            r = torch.zeros(d_flat.size(0), 1, device=images.device).uniform_(0, 1)
            delta = delta * (r.view(-1, 1, 1, 1) * epsilon / n.view(-1, 1, 1, 1))
            adv_images = adv_images + delta
        else:
            raise ValueError(f"Unknown norm '{norm}'. Use 'Linf' or 'L2'.")
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

    criterion = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad = adv_images.grad.detach()

        with torch.no_grad():
            if norm == "Linf":
                adv_images = adv_images + step_size * grad.sign()
                # Project back to epsilon ball
                perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
                adv_images = torch.clamp(images + perturbation, 0.0, 1.0)
            elif norm == "L2":
                grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
                grad_norm = grad_norm.view(-1, 1, 1, 1).clamp(min=1e-8)
                adv_images = adv_images + step_size * grad / grad_norm
                # Project back to L2 epsilon ball
                delta = adv_images - images
                delta_flat = delta.view(delta.size(0), -1)
                delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
                factor = torch.min(
                    torch.ones_like(delta_norm), epsilon / delta_norm.clamp(min=1e-8)
                )
                delta = delta * factor.view(-1, 1, 1, 1)
                adv_images = torch.clamp(images + delta, 0.0, 1.0)

        adv_images = adv_images.detach()

    return adv_images


def evaluate_adversarial(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epsilon: float,
    step_size: float,
    num_steps: int = 20,
    norm: str = "Linf",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate model accuracy under PGD attack.

    Runs PGD on every batch in *data_loader* and reports both clean
    and adversarial accuracy.

    Args:
        model: Target model.
        data_loader: Test data loader (images should be in [0,1] range,
                      i.e. **without** normalization).
        epsilon: Perturbation budget.
        step_size: PGD step size.
        num_steps: PGD iterations.
        norm: 'Linf' or 'L2'.
        device: Torch device.

    Returns:
        Tuple of (clean_accuracy, adversarial_accuracy) as percentages.
    """
    model.eval()
    model.to(device)

    correct_clean = 0
    correct_adv = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_out = model(images)
            correct_clean += (clean_out.argmax(1) == labels).sum().item()

        # Adversarial accuracy
        adv_images = pgd_attack(
            model, images, labels, epsilon, step_size, num_steps, norm
        )
        with torch.no_grad():
            adv_out = model(adv_images)
            correct_adv += (adv_out.argmax(1) == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * correct_clean / total
    adv_acc = 100.0 * correct_adv / total
    return clean_acc, adv_acc
