# src/utils/seed.py

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Configure random seeds and backend behavior for reproducible experiments.

    Args:
        seed: Global random seed.
        deterministic: Enable deterministic PyTorch algorithms where supported.

    Note:
        For strict CUDA reproducibility, set CUBLAS_WORKSPACE_CONFIG
        before starting Python and before CUDA is initialized.
        Determinism is not guaranteed across different hardware or
        PyTorch/CUDA versions.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU and CUDA random number generators
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Backend behavior
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Enforce deterministic implementations where available.
    # Unsupported operations raise an error rather than silently
    # claiming full reproducibility.
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """
    Initialize Python and NumPy RNGs in a DataLoader worker.

    PyTorch assigns each worker a seed derived from the DataLoader's
    generator. The worker ID is supplied automatically by DataLoader.
    """
    del worker_id  # The worker-specific seed is already assigned by PyTorch.

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """
    Create a seeded generator for reproducible DataLoader shuffling.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_client_seed(global_seed: int, client_id: int) -> int:
    """
    Derive a stable, distinct seed for each federated client.
    """
    if global_seed < 0:
        raise ValueError("global_seed must be non-negative")

    if client_id < 0:
        raise ValueError("client_id must be non-negative")

    # Keep the result in the range accepted by common RNGs.
    return (global_seed + 1_000_003 * (client_id + 1)) % (2**32)


def make_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool | None = None,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a reproducible PyTorch DataLoader.

    For object-detection datasets, use a custom collate_fn when
    images have different numbers of bounding boxes.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = make_generator(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
