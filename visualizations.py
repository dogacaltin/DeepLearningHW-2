"""
Visualization utilities for Grad-CAM and t-SNE analysis.

Provides functions to generate Grad-CAM heatmaps comparing clean vs.
adversarial samples and t-SNE embeddings showing the location of
adversarial points in feature space.
"""

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from parameters import VisualizationParams


# ---------- Grad-CAM ----------


class GradCAM:
    """Grad-CAM visualization for convolutional neural networks.

    Hooks into a target convolutional layer and computes class-discriminative
    localization maps.

    Attributes:
        model: The neural network.
        target_layer: The convolutional layer to visualize.
        gradients: Stored gradients from the backward hook.
        activations: Stored activations from the forward hook.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Register forward and backward hooks on *target_layer*.

        Args:
            model: The classifier network.
            target_layer: A ``nn.Conv2d`` (or similar) layer to inspect.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self, module: nn.Module, input: Tuple, output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self, module: nn.Module, grad_input: Tuple, grad_output: Tuple
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Class index to visualize; if None, uses the
                predicted class.

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional block of a ResNet for Grad-CAM.

    Args:
        model: A torchvision ResNet model.

    Returns:
        The final ``layer4`` sequential block.
    """
    return model.layer4[-1]


def visualize_gradcam(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_dir: str,
    device: torch.device,
    num_samples: int = 2,
) -> None:
    """Generate and save Grad-CAM comparisons (clean vs. adversarial).

    Selects samples where the adversarial perturbation causes a
    misclassification and overlays Grad-CAM heatmaps.

    Args:
        model: The classifier.
        clean_images: Clean images tensor (B, C, H, W), unnormalized [0,1].
        adv_images: Adversarial images tensor (B, C, H, W), unnormalized [0,1].
        labels: True labels (B,).
        class_names: List of class name strings.
        save_dir: Directory to save the figure.
        device: Torch device.
        num_samples: Number of misclassified samples to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(device)

    from datasets import get_normalize_transform

    normalize = get_normalize_transform()
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # Find misclassified adversarial samples
    with torch.no_grad():
        clean_norm = torch.stack([normalize(img) for img in clean_images]).to(device)
        adv_norm = torch.stack([normalize(img) for img in adv_images]).to(device)
        clean_preds = model(clean_norm).argmax(1)
        adv_preds = model(adv_norm).argmax(1)

    misclassified = (adv_preds != labels.to(device)) & (clean_preds == labels.to(device))
    mis_indices = misclassified.nonzero(as_tuple=True)[0].cpu().numpy()

    if len(mis_indices) == 0:
        print("No misclassified adversarial samples found. Skipping Grad-CAM.")
        return

    selected = mis_indices[: min(num_samples, len(mis_indices))]

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        clean_img = clean_images[idx]
        adv_img = adv_images[idx]
        true_label = labels[idx].item()

        # Grad-CAM for clean
        clean_input = normalize(clean_img).unsqueeze(0).to(device)
        cam_clean = grad_cam.generate(clean_input, target_class=true_label)

        # Grad-CAM for adversarial
        adv_input = normalize(adv_img).unsqueeze(0).to(device)
        adv_pred = model(adv_input).argmax(1).item()
        cam_adv = grad_cam.generate(adv_input, target_class=adv_pred)

        # Resize CAMs to image size
        from PIL import Image

        cam_clean_resized = np.array(
            Image.fromarray((cam_clean * 255).astype(np.uint8)).resize((32, 32))
        ) / 255.0
        cam_adv_resized = np.array(
            Image.fromarray((cam_adv * 255).astype(np.uint8)).resize((32, 32))
        ) / 255.0

        clean_np = clean_img.permute(1, 2, 0).cpu().numpy()
        adv_np = adv_img.permute(1, 2, 0).cpu().numpy()

        # Plot
        axes[row, 0].imshow(clean_np)
        axes[row, 0].set_title(f"Clean\nTrue: {class_names[true_label]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(clean_np)
        axes[row, 1].imshow(cam_clean_resized, cmap="jet", alpha=0.5)
        axes[row, 1].set_title(f"Clean Grad-CAM\nPred: {class_names[true_label]}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(adv_np)
        axes[row, 2].set_title(f"Adversarial\nTrue: {class_names[true_label]}")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(adv_np)
        axes[row, 3].imshow(cam_adv_resized, cmap="jet", alpha=0.5)
        axes[row, 3].set_title(f"Adv Grad-CAM\nPred: {class_names[adv_pred]}")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradcam_comparison.png"), dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {save_dir}/gradcam_comparison.png")


# ---------- t-SNE ----------


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Extract penultimate-layer features from a ResNet.

    Replaces the FC layer with an identity mapping and runs a forward pass
    to obtain the feature representations before classification.

    Args:
        model: A ResNet model.
        images: Images tensor of shape (B, C, H, W).
        device: Torch device.

    Returns:
        Feature array of shape (B, D).
    """
    model.eval().to(device)
    original_fc = model.fc
    model.fc = nn.Identity()

    with torch.no_grad():
        features = model(images.to(device)).cpu().numpy()

    model.fc = original_fc
    return features


def visualize_tsne(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_dir: str,
    device: torch.device,
    params: Optional[VisualizationParams] = None,
) -> None:
    """Create a t-SNE plot comparing clean and adversarial feature embeddings.

    Args:
        model: The classifier.
        clean_images: Clean images (B, C, H, W), normalized.
        adv_images: Adversarial images (B, C, H, W), normalized.
        labels: True labels (B,).
        class_names: Class name strings.
        save_dir: Output directory.
        device: Torch device.
        params: Visualization parameters (perplexity, iterations, etc.).
    """
    if params is None:
        params = VisualizationParams()

    os.makedirs(save_dir, exist_ok=True)

    n = min(params.tsne_num_samples, len(clean_images))
    clean_sub = clean_images[:n]
    adv_sub = adv_images[:n]
    labels_sub = labels[:n].numpy()

    clean_feats = extract_features(model, clean_sub, device)
    adv_feats = extract_features(model, adv_sub, device)

    all_feats = np.concatenate([clean_feats, adv_feats], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=params.tsne_perplexity,
        n_iter=params.tsne_n_iter,
        random_state=42,
    )
    embeddings = tsne.fit_transform(all_feats)

    clean_emb = embeddings[:n]
    adv_emb = embeddings[n:]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Clean samples colored by class
    scatter0 = axes[0].scatter(
        clean_emb[:, 0], clean_emb[:, 1], c=labels_sub, cmap="tab10",
        s=15, alpha=0.7
    )
    axes[0].set_title("t-SNE: Clean Samples")
    plt.colorbar(scatter0, ax=axes[0], ticks=range(10), label="Class")

    # Clean vs adversarial overlay
    axes[1].scatter(
        clean_emb[:, 0], clean_emb[:, 1], c="blue", s=15, alpha=0.4, label="Clean"
    )
    axes[1].scatter(
        adv_emb[:, 0], adv_emb[:, 1], c="red", s=15, alpha=0.4, label="Adversarial"
    )
    axes[1].set_title("t-SNE: Clean vs. Adversarial")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_visualization.png"), dpi=150)
    plt.close()
    print(f"t-SNE saved to {save_dir}/tsne_visualization.png")
