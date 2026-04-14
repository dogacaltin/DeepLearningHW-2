"""
Main entry point for CS515 HW2: Data Augmentation and Adversarial Samples.

Orchestrates all experiments via command-line arguments:
    1. Standard fine-tuning on CIFAR-10.
    2. AugMix fine-tuning on CIFAR-10.
    3. Corruption robustness evaluation (CIFAR-10-C).
    4. Adversarial robustness evaluation (PGD attacks + Grad-CAM + t-SNE).
    5. Knowledge distillation with AugMix teacher.
    6. Adversarial transferability (teacher -> student).

Usage examples:
    # Step 1: Standard fine-tuning
    python main.py --mode train --model_name resnet18

    # Step 2: AugMix fine-tuning
    python main.py --mode train --model_name resnet18 --use_augmix

    # Step 3: Evaluate on CIFAR-10-C
    python main.py --mode test_corruption --model_name resnet18 \\
        --checkpoint ./checkpoints/resnet18_standard_best.pth

    # Step 4: Adversarial evaluation (PGD + Grad-CAM + t-SNE)
    python main.py --mode test_adversarial --model_name resnet18 \\
        --checkpoint ./checkpoints/resnet18_standard_best.pth

    # Step 5: Knowledge distillation (AugMix teacher -> student)
    python main.py --mode distill \\
        --teacher_model resnet50 \\
        --teacher_checkpoint ./checkpoints/resnet50_augmix_best.pth \\
        --student_model resnet18

    # Step 6: Adversarial transferability
    python main.py --mode transferability \\
        --teacher_model resnet50 \\
        --teacher_checkpoint ./checkpoints/resnet50_augmix_best.pth \\
        --student_model resnet18 \\
        --student_checkpoint ./checkpoints/student_resnet18_from_resnet50_best.pth

    # Run all experiments sequentially
    python main.py --mode all --model_name resnet18 --teacher_model resnet50
"""

import argparse
import os

import torch

from distillation import train_distillation
from models import get_model
from parameters import (
    DistillationParams,
    TestParams,
    TrainParams,
)
from test import (
    evaluate_corruption_robustness,
    evaluate_transferability,
    run_adversarial_evaluation,
)
from train import evaluate, train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW2: Data Augmentation and Adversarial Samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "test_corruption",
            "test_adversarial",
            "distill",
            "transferability",
            "all",
        ],
        help="Experiment mode to run.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture.",
    )
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained weights.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--use_augmix", action="store_true", help="Enable AugMix.")
    parser.add_argument("--augmix_severity", type=int, default=3,
                        help="AugMix severity (1-10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data-loader workers.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for evaluation.")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints.")
    parser.add_argument("--cifar10c_root", type=str, default="./data/CIFAR-10-C",
                        help="Path to CIFAR-10-C data.")
    parser.add_argument("--vis_dir", type=str, default="./visualizations",
                        help="Directory for visualization outputs.")

    # Distillation-specific
    parser.add_argument("--teacher_model", type=str, default="resnet50",
                        help="Teacher architecture for distillation.")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher checkpoint.")
    parser.add_argument("--student_model", type=str, default="resnet18",
                        help="Student architecture for distillation.")
    parser.add_argument("--student_checkpoint", type=str, default=None,
                        help="Path to student checkpoint (for transferability).")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature.")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Distillation loss weight.")

    return parser.parse_args()


def main() -> None:
    """Run the selected experiment(s) based on CLI arguments."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("train", "all"):
        # ---------- Standard fine-tuning ----------
        print("\n" + "=" * 70)
        print("STEP 1: Standard Fine-Tuning")
        print("=" * 70)
        standard_params = TrainParams(
            model_name=args.model_name,
            pretrained=args.pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_augmix=False if args.mode == "all" else args.use_augmix,
            seed=args.seed,
            num_workers=args.num_workers,
            save_path=args.save_path,
        )
        std_ckpt = train(standard_params)

        if args.mode == "all" or args.use_augmix:
            # ---------- AugMix fine-tuning ----------
            print("\n" + "=" * 70)
            print("STEP 2: AugMix Fine-Tuning")
            print("=" * 70)
            augmix_params = TrainParams(
                model_name=args.model_name,
                pretrained=args.pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_augmix=True,
                augmix_severity=args.augmix_severity,
                seed=args.seed,
                num_workers=args.num_workers,
                save_path=args.save_path,
            )
            aug_ckpt = train(augmix_params)

    if args.mode in ("test_corruption", "all"):
        # ---------- CIFAR-10-C evaluation ----------
        print("\n" + "=" * 70)
        print("STEP 3: Corruption Robustness (CIFAR-10-C)")
        print("=" * 70)
        test_params = TestParams(
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        checkpoints_to_test = []
        if args.mode == "all":
            checkpoints_to_test = [
                (f"{args.model_name}_standard_best.pth", "Standard"),
                (f"{args.model_name}_augmix_best.pth", "AugMix"),
            ]
        else:
            if args.checkpoint:
                checkpoints_to_test = [(args.checkpoint, "Model")]
            else:
                print("Error: --checkpoint required for test_corruption mode.")
                return

        for ckpt_name, label in checkpoints_to_test:
            ckpt_path = (
                os.path.join(args.save_path, ckpt_name)
                if args.mode == "all"
                else ckpt_name
            )
            print(f"\n--- {label} model: {ckpt_path} ---")
            model = get_model(args.model_name, pretrained=False, num_classes=10,
                              checkpoint_path=ckpt_path)
            evaluate_corruption_robustness(model, args.cifar10c_root, device, test_params)

    if args.mode in ("test_adversarial", "all"):
        # ---------- Adversarial evaluation ----------
        print("\n" + "=" * 70)
        print("STEP 4: Adversarial Robustness (PGD + Grad-CAM + t-SNE)")
        print("=" * 70)
        test_params = TestParams(
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        checkpoints_to_test = []
        if args.mode == "all":
            checkpoints_to_test = [
                (f"{args.model_name}_standard_best.pth", "Standard"),
                (f"{args.model_name}_augmix_best.pth", "AugMix"),
            ]
        else:
            if args.checkpoint:
                checkpoints_to_test = [(args.checkpoint, "Model")]
            else:
                print("Error: --checkpoint required for test_adversarial mode.")
                return

        for ckpt_name, label in checkpoints_to_test:
            ckpt_path = (
                os.path.join(args.save_path, ckpt_name)
                if args.mode == "all"
                else ckpt_name
            )
            print(f"\n--- {label} model: {ckpt_path} ---")
            model = get_model(args.model_name, pretrained=False, num_classes=10,
                              checkpoint_path=ckpt_path)
            vis_subdir = os.path.join(args.vis_dir, label.lower())
            run_adversarial_evaluation(model, device, test_params, vis_subdir)

    if args.mode in ("distill", "all"):
        # ---------- Knowledge distillation ----------
        print("\n" + "=" * 70)
        print("STEP 5: Knowledge Distillation (AugMix Teacher -> Student)")
        print("=" * 70)
        teacher_ckpt = args.teacher_checkpoint
        if args.mode == "all":
            teacher_ckpt = os.path.join(
                args.save_path, f"{args.teacher_model}_augmix_best.pth"
            )

        if teacher_ckpt is None:
            print("Error: --teacher_checkpoint required for distill mode.")
            return

        distill_params = DistillationParams(
            teacher_model_name=args.teacher_model,
            teacher_checkpoint=teacher_ckpt,
            student_model_name=args.student_model,
            temperature=args.temperature,
            alpha=args.alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            num_workers=args.num_workers,
            save_path=args.save_path,
        )
        student_ckpt = train_distillation(distill_params)

    if args.mode in ("transferability", "all"):
        # ---------- Adversarial transferability ----------
        print("\n" + "=" * 70)
        print("STEP 6: Adversarial Transferability (Teacher -> Student)")
        print("=" * 70)
        teacher_ckpt = args.teacher_checkpoint
        student_ckpt = args.student_checkpoint

        if args.mode == "all":
            teacher_ckpt = os.path.join(
                args.save_path, f"{args.teacher_model}_augmix_best.pth"
            )
            student_ckpt = os.path.join(
                args.save_path,
                f"student_{args.student_model}_from_{args.teacher_model}_best.pth",
            )

        if teacher_ckpt is None or student_ckpt is None:
            print("Error: both --teacher_checkpoint and --student_checkpoint required.")
            return

        teacher = get_model(args.teacher_model, pretrained=False, num_classes=10,
                            checkpoint_path=teacher_ckpt)
        student = get_model(args.student_model, pretrained=False, num_classes=10,
                            checkpoint_path=student_ckpt)

        evaluate_transferability(
            teacher, student, device, args.batch_size, args.num_workers
        )

    print("\n" + "=" * 70)
    print("All requested experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
