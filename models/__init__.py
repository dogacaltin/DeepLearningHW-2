"""
Neural network model definitions for CS515 HW2.

Provides factory functions to create ResNet-based architectures
fine-tuned for CIFAR-10 classification.
"""

from models.resnet import get_model

__all__ = ["get_model"]
