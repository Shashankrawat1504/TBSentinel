"""Compare the centralized baseline against federated global checkpoints.

Reads the centralized baseline checkpoint metrics plus every
``artifacts/outputs/federated_*_results.json`` summary written by the Flower
simulation, and prints a validation comparison table in Markdown. All numbers
come from the existing artifacts; nothing is recomputed or retrained.

Example
-------
    python scripts/compare_federated_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CENTRALIZED_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "tb_efficientnet_b3_best.pt"
)
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "artifacts" / "outputs"

METRIC_COLUMNS = (
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("auc", "AUC-ROC"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
)


class FederatedRecord(TypedDict):
    """One federated strategy summary extracted from a results JSON file."""

    strategy: str
    best_round: object
    num_rounds: object
    local_epochs: object
    proximal_mu: object
    metrics: dict[str, float]
    summary_path: Path
    checkpoint: object


def as_float_map(raw: object) -> dict[str, float]:
    """Return the numeric entries of a mapping as floats, ignoring anything else."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[str(key)] = float(value)
    return result


def pretty_strategy(name: object) -> str:
    """Return a display label for a strategy identifier."""
    text = str(name)
    known = {"fedavg": "FedAvg", "fedprox": "FedProx"}
    return known.get(text.lower(), text.title())


def load_centralized_metrics(checkpoint_path: Path) -> dict[str, float]:
    """Return the validation metrics stored in the centralized baseline checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Centralized checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint has no metrics mapping: {checkpoint_path}")
    return as_float_map(metrics)


def load_federated_results(outputs_dir: Path) -> list[FederatedRecord]:
    """Return one record per federated strategy summary file, sorted by strategy name."""
    records: list[FederatedRecord] = []
    for summary_path in sorted(outputs_dir.glob("federated_*_results.json")):
        with summary_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        records.append(
            FederatedRecord(
                strategy=str(data.get("strategy", summary_path.stem)),
                best_round=data.get("best_round"),
                num_rounds=data.get("num_rounds"),
                local_epochs=data.get("local_epochs"),
                proximal_mu=data.get("proximal_mu"),
                metrics=as_float_map(data.get("best_metrics")),
                summary_path=summary_path,
                checkpoint=data.get("best_checkpoint"),
            )
        )
    return records


def format_markdown(
    centralized: dict[str, float],
    federated: list[FederatedRecord],
) -> str:
    """Render the comparison tables as Markdown."""
    header = "| Model | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |"
    divider = "| --- |" + " ---: |" * len(METRIC_COLUMNS)
    lines = [header, divider]

    def cells(metrics: dict[str, float]) -> str:
        values = []
        for key, _ in METRIC_COLUMNS:
            value = metrics.get(key)
            values.append(f"{value:.4f}" if value is not None else "n/a")
        return "| " + " | ".join(values) + " |"

    def delta_cells(metrics: dict[str, float]) -> str:
        values = []
        for key, _ in METRIC_COLUMNS:
            if key in metrics and key in centralized:
                values.append(f"{metrics[key] - centralized[key]:+.4f}")
            else:
                values.append("n/a")
        return "| " + " | ".join(values) + " |"

    lines.append(f"| Centralized baseline (best) {cells(centralized)}")
    for record in federated:
        label = (
            f"Federated {pretty_strategy(record['strategy'])} "
            f"(round {record['best_round']})"
        )
        lines.append(f"| {label} {cells(record['metrics'])}")

    lines.append("")
    lines.append("Delta versus centralized baseline (negative means lower than centralized):")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    for record in federated:
        label = f"Federated {pretty_strategy(record['strategy'])}"
        lines.append(f"| {label} {delta_cells(record['metrics'])}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Compare centralized and federated validation metrics."
    )
    parser.add_argument(
        "--centralized-checkpoint",
        type=Path,
        default=DEFAULT_CENTRALIZED_CHECKPOINT,
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to also write the Markdown comparison table.",
    )
    return parser.parse_args()


def main() -> None:
    """Print (and optionally save) the centralized versus federated comparison."""
    args = parse_args()
    centralized = load_centralized_metrics(args.centralized_checkpoint.expanduser().resolve())
    federated = load_federated_results(args.outputs_dir.expanduser().resolve())

    if not federated:
        raise FileNotFoundError(
            f"No federated_*_results.json found in {args.outputs_dir}. "
            "Run the federated simulation first."
        )

    table = format_markdown(centralized, federated)
    print(table)
    print("")
    for record in federated:
        print(f"- {record['strategy']}: {record['summary_path']}")
        print(f"  checkpoint: {record['checkpoint']}")

    if args.output_markdown is not None:
        output_path = args.output_markdown.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
        print("")
        print(f"Saved comparison table to {output_path}")


if __name__ == "__main__":
    main()
