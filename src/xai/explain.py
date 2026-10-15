"""Generate clinical Grad-CAM explanations for representative validation chest X-rays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TBXRayDataset, build_eval_transform
from src.utils.logger import get_logger
from src.xai.gradcam import TBGradCAMExplainer

LOGGER = get_logger("xai.explain")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve configured relative paths from project root, not Hydra output."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def create_clinical_overview_grid(
    selected_cases: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Create a unified clinical overview figure showcasing TP, TN, FP, and FN cases side-by-side."""
    categories = ["true_positives", "true_negatives", "false_positives", "false_negatives"]
    display_names = {
        "true_positives": "True Positive (Correct TB)",
        "true_negatives": "True Negative (Correct Non-TB)",
        "false_positives": "False Positive (Non-TB predicted as TB)",
        "false_negatives": "False Negative (TB predicted as Non-TB)",
    }

    # Pick 2 representative cases per category for a clean 4-row x 2-case figure
    num_rows = len(categories)
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(16, 4 * num_rows), constrained_layout=True)

    for row_idx, cat in enumerate(categories):
        cases = selected_cases.get(cat, [])
        for col_idx in range(num_cols):
            ax_orig = axes[row_idx, col_idx * 2]
            ax_cam = axes[row_idx, col_idx * 2 + 1]

            if col_idx < len(cases):
                case = cases[col_idx]
                expl = case["explanation"]
                img_name = Path(case["image_path"]).name
                conf = expl["confidence"]
                pred = expl["predicted_class_name"]

                ax_orig.imshow(expl["rgb_image"])
                ax_orig.set_title(f"Input: {img_name}", fontsize=9)
                ax_orig.axis("off")

                ax_cam.imshow(expl["overlay"])
                ax_cam.set_title(f"Overlay: {pred} ({conf:.1%})", fontsize=9, fontweight="bold")
                ax_cam.axis("off")
            else:
                ax_orig.axis("off")
                ax_cam.axis("off")

        # Category label on left
        axes[row_idx, 0].text(
            -0.15,
            0.5,
            display_names[cat],
            transform=axes[row_idx, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    fig.suptitle(
        "TBSentinel Explainable AI (Grad-CAM) - Clinical Validation Overview",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved clinical overview collage to: %s", output_path)


def generate_validation_explanations(
    checkpoint_path: str | Path,
    val_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    samples_per_category: int = 5,
    target_layer: str | None = "auto",
    image_size: int = 300,
    colormap: str = "jet",
    alpha: float = 0.5,
    device: str | torch.device | None = None,
    save_figure: bool = True,
    save_overlay_only: bool = True,
    create_summary_grid: bool = True,
) -> dict[str, Any]:
    """Evaluate validation split, categorize outcomes, and save Grad-CAM explanations."""
    checkpoint_file = resolve_path(checkpoint_path)
    val_csv_file = resolve_path(val_csv)
    image_root_dir = resolve_path(image_root)
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Initializing explainer with checkpoint: %s", checkpoint_file)
    explainer = TBGradCAMExplainer(
        checkpoint_path=checkpoint_file,
        target_layer=target_layer,
        device=device,
        image_size=image_size,
        colormap=colormap,
        alpha=alpha,
    )

    LOGGER.info("Loading validation dataset from: %s", val_csv_file)
    val_dataset = TBXRayDataset(
        annotations=val_csv_file,
        image_root=image_root_dir,
        transform=build_eval_transform(image_size=image_size),
        image_size=image_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    LOGGER.info("Running inference across validation set (%d images)...", len(val_dataset))
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation inference"):
            images = images.to(explainer.device)
            logits = explainer.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # Categorize indices
    tp_indices = np.where((preds_arr == 1) & (targets_arr == 1))[0].tolist()
    tn_indices = np.where((preds_arr == 0) & (targets_arr == 0))[0].tolist()
    fp_indices = np.where((preds_arr == 1) & (targets_arr == 0))[0].tolist()
    fn_indices = np.where((preds_arr == 0) & (targets_arr == 1))[0].tolist()

    LOGGER.info(
        "Validation outcome counts: TP=%d, TN=%d, FP=%d, FN=%d",
        len(tp_indices),
        len(tn_indices),
        len(fp_indices),
        len(fn_indices),
    )

    # Select representative samples
    # All FP cases (3 total) to provide complete clinician visibility on model errors
    selected_fp = fp_indices
    # Top N cases for FN, TP, TN
    selected_fn = fn_indices[:samples_per_category]
    selected_tp = tp_indices[:samples_per_category]
    selected_tn = tn_indices[:samples_per_category]

    category_mapping = {
        "true_positives": selected_tp,
        "true_negatives": selected_tn,
        "false_positives": selected_fp,
        "false_negatives": selected_fn,
    }

    summary_records: list[dict[str, Any]] = []
    cases_for_grid: dict[str, list[dict[str, Any]]] = {k: [] for k in category_mapping}

    for category_name, indices in category_mapping.items():
        category_dir = out_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Generating %d explanations for category: %s...",
            len(indices),
            category_name,
        )

        for rank, idx in enumerate(indices, start=1):
            image_tensor, target_tensor = val_dataset[idx]
            raw_image_path = val_dataset.image_paths[idx]
            img_stem = raw_image_path.stem
            ground_truth = int(target_tensor.item())

            # Generate explanation
            explanation = explainer.explain(image=image_tensor)

            # Save artifacts
            output_prefix = category_dir / f"{rank:02d}_{img_stem}"
            title_text = (
                f"{category_name.replace('_', ' ').title()} #{rank}: {img_stem}\n"
                f"Pred: {explanation['predicted_class_name'].upper()} "
                f"({explanation['confidence']:.1%}) | True: {explainer.class_names[ground_truth].upper()}"
            )
            saved_paths = explainer.save_explanation(
                explanation=explanation,
                output_path=output_prefix,
                ground_truth=ground_truth,
                title=title_text,
                save_figure=save_figure,
                save_overlay_only=save_overlay_only,
            )

            record = {
                "category": category_name,
                "rank": rank,
                "sample_index": int(idx),
                "image_path": str(raw_image_path),
                "filename": raw_image_path.name,
                "ground_truth": ground_truth,
                "ground_truth_name": explainer.class_names[ground_truth],
                "predicted_class_id": explanation["predicted_class_id"],
                "predicted_class_name": explanation["predicted_class_name"],
                "confidence": float(explanation["confidence"]),
                "probabilities": explanation["probabilities"],
                "overlay_path": str(saved_paths.get("overlay_path", "")),
                "figure_path": str(saved_paths.get("figure_path", "")),
            }
            summary_records.append(record)
            cases_for_grid[category_name].append({
                "image_path": str(raw_image_path),
                "explanation": explanation,
            })

    # Create overview grid
    if create_summary_grid:
        overview_path = out_dir / "xai_clinical_overview.png"
        create_clinical_overview_grid(cases_for_grid, overview_path)

    # Save summary metadata
    summary_json_path = out_dir / "explanation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_layer": explainer.target_layer.__class__.__name__,
                "total_validation_samples": len(val_dataset),
                "counts": {
                    "tp": len(tp_indices),
                    "tn": len(tn_indices),
                    "fp": len(fp_indices),
                    "fn": len(fn_indices),
                },
                "generated_counts": {k: len(v) for k, v in category_mapping.items()},
                "records": summary_records,
            },
            f,
            indent=2,
        )

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = out_dir / "explanation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    LOGGER.info(
        "Explanation generation finished! Saved %d cases to %s",
        len(summary_records),
        out_dir,
    )

    return {
        "output_dir": str(out_dir),
        "total_generated": len(summary_records),
        "category_counts": {k: len(v) for k, v in category_mapping.items()},
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for explanation generation."""
    xai_cfg = cfg.get("xai", {})
    checkpoint_path = xai_cfg.get(
        "checkpoint_path",
        f"{cfg.paths.checkpoints_dir}/tb_efficientnet_b3_best.pt",
    )
    val_csv = xai_cfg.get("val_csv", cfg.data.val_csv)
    image_root = xai_cfg.get("image_root", cfg.data.image_root)
    output_dir = xai_cfg.get(
        "output_dir",
        f"{cfg.paths.artifacts_dir}/explanations",
    )
    samples_per_category = int(xai_cfg.get("samples_per_category", 5))
    target_layer = str(xai_cfg.get("target_layer", "auto"))
    image_size = int(xai_cfg.get("image_size", cfg.model.image_size))
    colormap = str(xai_cfg.get("colormap", "jet"))
    alpha = float(xai_cfg.get("alpha", 0.5))
    device = cfg.get("device", "auto")
    save_figure = bool(xai_cfg.get("save_figure", True))
    save_overlay_only = bool(xai_cfg.get("save_overlay_only", True))
    create_summary_grid = bool(xai_cfg.get("create_summary_grid", True))

    LOGGER.info("Starting Hydra-configured XAI explanation pipeline...")
    results = generate_validation_explanations(
        checkpoint_path=checkpoint_path,
        val_csv=val_csv,
        image_root=image_root,
        output_dir=output_dir,
        samples_per_category=samples_per_category,
        target_layer=target_layer,
        image_size=image_size,
        colormap=colormap,
        alpha=alpha,
        device=device,
        save_figure=save_figure,
        save_overlay_only=save_overlay_only,
        create_summary_grid=create_summary_grid,
    )
    LOGGER.info("XAI pipeline finished: %s", results)


if __name__ == "__main__":
    main()
