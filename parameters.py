"""
Parameter dataclasses for CS515 HW2: Data Augmentation and Adversarial Samples.

This module defines all configuration parameters used across training,
testing, adversarial attacks, knowledge distillation, and visualization.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainParams:
    """Parameters for model training and fine-tuning.

    Attributes:
        model_name: Architecture name ('resnet18', 'resnet50', etc.).
        pretrained: Whether to load ImageNet-pretrained weights.
        num_classes: Number of output classes (10 for CIFAR-10).
        epochs: Number of training epochs.
        batch_size: Mini-batch size for training.
        lr: Initial learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularization coefficient.
        scheduler: Learning rate scheduler ('cosine' or 'step').
        use_augmix: Whether to apply AugMix data augmentation.
        augmix_severity: AugMix severity (1-10).
        augmix_width: Number of parallel augmentation chains.
        augmix_depth: Depth of each augmentation chain (-1 for stochastic).
        augmix_alpha: Dirichlet distribution parameter for mixing weights.
        save_path: Directory to save trained model checkpoints.
        seed: Random seed for reproducibility.
        num_workers: Number of data-loading workers.
    """

    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 10
    epochs: int = 25
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    use_augmix: bool = False
    augmix_severity: int = 3
    augmix_width: int = 3
    augmix_depth: int = -1
    augmix_alpha: float = 1.0
    save_path: str = "./checkpoints"
    seed: int = 42
    num_workers: int = 4


@dataclass
class TestParams:
    """Parameters for model evaluation.

    Attributes:
        model_name: Architecture name to load.
        checkpoint_path: Path to the saved model checkpoint.
        batch_size: Mini-batch size for evaluation.
        num_classes: Number of output classes.
        num_workers: Number of data-loading workers.
        test_corruption: Whether to evaluate on CIFAR-10-C corruptions.
        corruption_types: List of corruption types to evaluate (None = all).
        severity_levels: Corruption severity levels to test.
    """

    model_name: str = "resnet18"
    checkpoint_path: str = "./checkpoints/best_model.pth"
    batch_size: int = 128
    num_classes: int = 10
    num_workers: int = 4
    test_corruption: bool = False
    corruption_types: Optional[list] = None
    severity_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass
class AdversarialParams:
    """Parameters for adversarial attack generation and evaluation.

    Attributes:
        attack: Attack method ('pgd').
        norm: Lp norm constraint ('Linf' or 'L2').
        epsilon: Perturbation budget.
        step_size: PGD step size (alpha).
        num_steps: Number of PGD iterations.
        random_start: Whether to use random initialization for PGD.
    """

    attack: str = "pgd"
    norm: str = "Linf"
    epsilon: float = 4.0 / 255.0
    step_size: float = 1.0 / 255.0
    num_steps: int = 20
    random_start: bool = True


@dataclass
class DistillationParams:
    """Parameters for knowledge distillation.

    Attributes:
        teacher_model_name: Architecture of the teacher model.
        teacher_checkpoint: Path to teacher model checkpoint.
        student_model_name: Architecture of the student model.
        temperature: Distillation temperature.
        alpha: Weight for distillation loss vs. hard-label loss.
        epochs: Number of distillation training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularization.
        save_path: Directory for saving student checkpoints.
        seed: Random seed.
        num_workers: Number of data-loading workers.
    """

    teacher_model_name: str = "resnet50"
    teacher_checkpoint: str = "./checkpoints/teacher_augmix_best.pth"
    student_model_name: str = "resnet18"
    temperature: float = 4.0
    alpha: float = 0.7
    epochs: int = 25
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    save_path: str = "./checkpoints"
    seed: int = 42
    num_workers: int = 4


@dataclass
class VisualizationParams:
    """Parameters for Grad-CAM and t-SNE visualization.

    Attributes:
        num_gradcam_samples: Number of samples for Grad-CAM visualization.
        tsne_perplexity: Perplexity for t-SNE embedding.
        tsne_n_iter: Number of t-SNE iterations.
        tsne_num_samples: Number of samples for t-SNE plot.
        save_dir: Directory to save visualization outputs.
    """

    num_gradcam_samples: int = 2
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    tsne_num_samples: int = 500
    save_dir: str = "./visualizations"
