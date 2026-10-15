"""Convenience CLI script to generate Grad-CAM clinical explanations for TBSentinel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.explain import generate_validation_explanations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM clinical explanations for representative validation chest X-rays."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/tb_efficientnet_b3_best.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/processed/tb_val.csv",
        help="Path to validation split CSV.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/raw/TB/chest x ray/imgs",
        help="Root folder containing raw chest X-ray images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/explanations",
        help="Output folder where explanations will be saved.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of representative samples per category (TP, TN, FN). All FP are included.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default="auto",
        help="Target layer for Grad-CAM ('auto' or layer path like 'backbone.features.8').",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="Input image resolution.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Colormap for overlay ('jet', 'turbo', 'inferno', 'viridis').",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Heatmap blend opacity (0.0 to 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('auto', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_validation_explanations(
        checkpoint_path=args.checkpoint,
        val_csv=args.val_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        samples_per_category=args.samples_per_category,
        target_layer=args.target_layer,
        image_size=args.image_size,
        colormap=args.colormap,
        alpha=args.alpha,
        device=args.device,
    )


if __name__ == "__main__":
    main()
