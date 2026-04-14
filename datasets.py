"""
Dataset utilities for CIFAR-10, CIFAR-10-C, and AugMix augmentation.

Provides data loaders for clean CIFAR-10 training/testing, the
CIFAR-10-C corruption benchmark, and AugMix-augmented training.
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageEnhance

# ---------- Standard CIFAR-10 transforms ----------

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)


def get_normalize_transform() -> transforms.Normalize:
    """Return the CIFAR-10 channel-wise normalization transform."""
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def get_train_transform() -> transforms.Compose:
    """Standard CIFAR-10 training transform (crop + flip + normalize)."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_normalize_transform(),
        ]
    )


def get_test_transform() -> transforms.Compose:
    """Standard CIFAR-10 test transform (normalize only)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            get_normalize_transform(),
        ]
    )


# ---------- AugMix implementation ----------

# Individual augmentation operations (PIL-based)


def _autocontrast(pil_img: Image.Image, _level: int) -> Image.Image:
    return ImageOps.autocontrast(pil_img)


def _equalize(pil_img: Image.Image, _level: int) -> Image.Image:
    return ImageOps.equalize(pil_img)


def _posterize(pil_img: Image.Image, level: int) -> Image.Image:
    level = int((level / 10.0) * 4)
    return ImageOps.posterize(pil_img, 4 - level)


def _rotate(pil_img: Image.Image, level: int) -> Image.Image:
    degrees = (level / 10.0) * 30.0
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def _solarize(pil_img: Image.Image, level: int) -> Image.Image:
    threshold = 256 - int((level / 10.0) * 256)
    return ImageOps.solarize(pil_img, threshold)


def _shear_x(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        v = -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def _shear_y(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        v = -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def _translate_x(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * (pil_img.size[0] / 3.0)
    if np.random.uniform() > 0.5:
        v = -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def _translate_y(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * (pil_img.size[1] / 3.0)
    if np.random.uniform() > 0.5:
        v = -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def _color(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Color(pil_img).enhance(v)


def _contrast(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(v)


def _brightness(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(v)


def _sharpness(pil_img: Image.Image, level: int) -> Image.Image:
    v = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(v)


AUGMENTATIONS: List[Callable] = [
    _autocontrast,
    _equalize,
    _posterize,
    _rotate,
    _solarize,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
    _color,
    _contrast,
    _brightness,
    _sharpness,
]


def augmix(
    image: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """Apply AugMix augmentation to a PIL image.

    Generates *width* augmentation chains, mixes them with Dirichlet
    weights, and blends the result with the original using a Beta weight.

    Args:
        image: Input PIL image.
        severity: Maximum severity of each augmentation (1-10).
        width: Number of parallel augmentation chains.
        depth: Depth of each chain (-1 for stochastic depth 1-3).
        alpha: Dirichlet/Beta concentration parameter.

    Returns:
        Augmented PIL image.

    Reference:
        Hendrycks et al., "AugMix: A Simple Data Processing Method to
        Improve Robustness and Uncertainty", ICLR 2020.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(np.array(image), dtype=np.float32)
    for i in range(width):
        image_aug = image.copy()
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(chain_depth):
            op = np.random.choice(AUGMENTATIONS)
            level = np.random.randint(1, severity + 1)
            image_aug = op(image_aug, level)
        mix += ws[i] * np.array(image_aug, dtype=np.float32)

    mixed = (1 - m) * np.array(image, dtype=np.float32) + m * mix
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 255)))


class AugMixDataset(Dataset):
    """Wraps a CIFAR-10 dataset to apply AugMix and return a JSD triplet.

    For Jensen-Shannon Divergence consistency loss, each sample returns
    the clean image plus two independent AugMix versions.

    Attributes:
        dataset: Underlying CIFAR-10 dataset (without transforms applied).
        preprocess: Final tensor conversion + normalization.
        severity: AugMix severity.
        width: AugMix chain width.
        depth: AugMix chain depth.
        alpha: Dirichlet/Beta parameter.
    """

    def __init__(
        self,
        dataset: datasets.CIFAR10,
        preprocess: transforms.Compose,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.preprocess = preprocess
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        # Basic augmentation before AugMix
        self.basic_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (clean, augmix1, augmix2, label) for JSD training.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (clean_tensor, aug1_tensor, aug2_tensor, label).
        """
        image, label = self.dataset[idx]
        # image is a PIL Image (no transform on the base dataset)
        image_basic = self.basic_transform(image)

        aug1 = augmix(image_basic, self.severity, self.width, self.depth, self.alpha)
        aug2 = augmix(image_basic, self.severity, self.width, self.depth, self.alpha)

        return (
            self.preprocess(image_basic),
            self.preprocess(aug1),
            self.preprocess(aug2),
            label,
        )


# ---------- CIFAR-10-C loader ----------

ALL_CORRUPTIONS: List[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


class CIFAR10C(Dataset):
    """CIFAR-10-C corruption benchmark dataset.

    Loads numpy files from the CIFAR-10-C directory and presents them
    as a PyTorch Dataset.

    Attributes:
        data: Numpy array of shape (N, 32, 32, 3).
        targets: Numpy array of integer labels.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Load a specific corruption at a given severity.

        Args:
            root: Path to the CIFAR-10-C directory.
            corruption: Name of the corruption type.
            severity: Severity level (1-5).
            transform: Optional image transform.

        Raises:
            FileNotFoundError: If the corruption numpy file is missing.
        """
        assert 1 <= severity <= 5, "Severity must be in [1, 5]."
        data_path = os.path.join(root, f"{corruption}.npy")
        label_path = os.path.join(root, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Corruption file not found: {data_path}")

        all_data = np.load(data_path)
        all_labels = np.load(label_path)

        # Each severity level has 10,000 images
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = all_data[start:end]
        self.targets = all_labels[start:end]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.data[idx])
        label = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------- Data-loader factory ----------


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmix: bool = False,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders.

    Args:
        batch_size: Batch size for both loaders.
        num_workers: Number of data-loading workers.
        use_augmix: If True, wrap the training set with AugMixDataset.
        augmix_severity: AugMix severity parameter.
        augmix_width: AugMix width parameter.
        augmix_depth: AugMix depth parameter.
        augmix_alpha: AugMix alpha parameter.
        data_root: Root directory for CIFAR-10 data.

    Returns:
        (train_loader, test_loader) tuple.
    """
    test_transform = get_test_transform()

    if use_augmix:
        # AugMix: load raw PIL images (no transform on dataset)
        base_train = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=None
        )
        preprocess = transforms.Compose(
            [transforms.ToTensor(), get_normalize_transform()]
        )
        train_dataset = AugMixDataset(
            base_train, preprocess, augmix_severity, augmix_width, augmix_depth, augmix_alpha
        )
    else:
        train_transform = get_train_transform()
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar10c_loader(
    root: str = "./data/CIFAR-10-C",
    corruption: str = "gaussian_noise",
    severity: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for a single CIFAR-10-C corruption/severity.

    Args:
        root: Path to the CIFAR-10-C directory.
        corruption: Corruption type name.
        severity: Severity level (1-5).
        batch_size: Batch size.
        num_workers: Number of workers.

    Returns:
        A DataLoader for the specified corruption.
    """
    dataset = CIFAR10C(root, corruption, severity, transform=get_test_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
