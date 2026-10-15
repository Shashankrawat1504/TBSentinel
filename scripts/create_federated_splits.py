"""Partition the labeled TB training split among simulated federated clients."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("create_federated_splits")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "tb_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "federated_splits"
DEFAULT_CLIENTS = 3
DEFAULT_ALPHA = 0.5
DEFAULT_SEED = 42
REQUIRED_COLUMNS = {"image_path", "label", "label_name", "split"}


def partition_by_dirichlet(
    frame: pd.DataFrame,
    num_clients: int = DEFAULT_CLIENTS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    max_attempts: int = 100,
) -> list[pd.DataFrame]:
    """Partition rows by class using per-class Dirichlet proportions.

    The same image is assigned to exactly one client. Each class is shuffled
    independently and distributed according to a Dirichlet(alpha) draw, which
    creates heterogeneous client class distributions while covering all
    training records exactly once.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Training CSV contains no samples.")
    if frame["label"].isna().any():
        raise ValueError("Training CSV must contain labels for every row.")
    if not frame["split"].eq("train").all():
        raise ValueError("Only rows marked split='train' can be partitioned.")
    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError("Training labels must be binary integers 0 or 1.")
    if frame["image_path"].duplicated().any():
        raise ValueError("Training CSV contains duplicate image paths.")

    labels = numeric_labels.astype("int64").to_numpy()
    rng = np.random.default_rng(seed)
    alpha_vector = np.full(num_clients, alpha, dtype=np.float64)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(alpha_vector)
            counts = rng.multinomial(len(class_indices), proportions)
            start = 0
            for client_id, count in enumerate(counts):
                client_indices[client_id].extend(
                    class_indices[start : start + count].tolist()
                )
                start += int(count)

        if all(client_indices):
            return [
                frame.iloc[indices]
                .sample(frac=1.0, random_state=seed + client_id)
                .reset_index(drop=True)
                for client_id, indices in enumerate(client_indices)
            ]

    raise RuntimeError(
        f"Could not create {num_clients} non-empty clients after {max_attempts} "
        "Dirichlet draws. Increase the dataset size or max_attempts."
    )


def create_federated_splits(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_clients: int = DEFAULT_CLIENTS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> list[pd.DataFrame]:
    """Read the training CSV and write one deterministic CSV per client."""
    train_csv = train_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}. "
            "Run prepare_tb_image_splits.py first."
        )

    training = pd.read_csv(train_csv)
    clients = partition_by_dirichlet(
        training,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, client_frame in enumerate(clients):
        output_path = output_dir / f"client_{client_id}_train.csv"
        client_frame.to_csv(output_path, index=False)
        counts = client_frame["label_name"].value_counts().to_dict()
        LOGGER.info(
            "Client %d: %d samples, class counts=%s, saved to %s",
            client_id,
            len(client_frame),
            counts,
            output_path,
        )

    assigned_paths = [
        path for client in clients for path in client["image_path"].tolist()
    ]
    if len(assigned_paths) != len(training) or len(set(assigned_paths)) != len(
        training
    ):
        raise RuntimeError("Federated clients do not form an exact training partition.")
    return clients


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Create Dirichlet non-IID CSVs for simulated FL clients."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_CLIENTS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Generate federated clients and print a clear partition summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    clients = create_federated_splits(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(f"Dirichlet federated split summary (alpha={args.alpha}, seed={args.seed})")
    for client_id, frame in enumerate(clients):
        counts = frame["label_name"].value_counts().to_dict()
        print(f"  client_{client_id}: {len(frame):,} samples; classes={counts}")
    print(f"  total assigned: {sum(map(len, clients)):,}")


if __name__ == "__main__":
    main()
