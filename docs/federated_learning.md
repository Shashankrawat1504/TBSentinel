# TBSentinel Federated Learning

## Setup

The FL experiment uses the image-only TBX11K EfficientNet-B3 classifier with
three simulated Flower clients. The clients train on fixed, seeded Dirichlet
partitions of the official 6,600-image training split (`alpha=0.5`, seed `42`).
Validation remains global and uses only the official 1,800-image validation
split. The unlabeled 3,302-image test split is not loaded or evaluated.
Genomic data is not used because no verified image/genomic pairs are available.

Client partitions (verified against the CSVs on disk; they partition the
training split exactly once with no validation overlap):

| Client | Images | Non-TB | TB |
| --- | ---: | ---: | ---: |
| 0 | 5,571 | 5,466 | 105 |
| 1 | 536 | 534 | 2 |
| 2 | 493 | 0 | 493 |

The client distribution is intentionally non-IID but extreme: client 2 has only
TB cases. This is a limitation of this particular Dirichlet draw, not evidence
that all clients are representative of separate clinical sites.

Clients are simulated in-process with Flower's virtual client engine. There is
no standalone Flower server process: aggregation runs through the FedAvg/FedProx
strategy object and `start_simulation`, which is why only client, strategy, and
simulation-runner modules exist under `src/federation/`.

## Completed runs

Two strategies were executed end to end with identical settings: five rounds,
one local epoch per round, batch size 16, learning rate `1e-4`, AdamW,
`weight_decay` `1e-4`, gradient clipping `1.0`, and CUDA mixed precision on an
RTX 3050 (4 GB). The best global checkpoint is selected by global validation
AUC. FedProx used `proximal_mu=0.01`; FedAvg disables the proximal penalty.

Per-round global validation history:

| Round | FedAvg Acc | FedAvg F1 | FedAvg AUC | FedProx Acc | FedProx F1 | FedProx AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 0.1617 | 0.1594 | 0.5443 | 0.1617 | 0.1594 | 0.5443 |
| 1 | 0.9594 | 0.8772 | 0.9641 | 0.9061 | 0.6091 | 0.8766 |
| 2 | 0.9678 | 0.9063 | 0.9832 | 0.9611 | 0.8839 | 0.9543 |
| 3 | 0.9783 | 0.9414 | 0.9801 | 0.9589 | 0.8752 | 0.9593 |
| 4 | 0.9722 | 0.9209 | 0.9807 | 0.9678 | 0.9063 | 0.9754 |
| 5 | 0.9811 | 0.9485 | 0.9936 | 0.9761 | 0.9331 | 0.9764 |

Both strategies selected round 5. Best-checkpoint metrics:

| Model | Accuracy | Macro-F1 | AUC-ROC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized baseline (best) | 0.9878 | 0.9679 | 0.9986 | 0.9050 | 0.9981 |
| Federated FedAvg (round 5) | 0.9811 | 0.9485 | 0.9936 | 0.8350 | 0.9994 |
| Federated FedProx (round 5) | 0.9761 | 0.9331 | 0.9764 | 0.7850 | 1.0000 |

Confusion matrices in `non_tb`, `tb` order:

- Centralized: `[[1597, 3], [19, 181]]`
- FedAvg: `[[1599, 1], [33, 167]]`
- FedProx: `[[1600, 0], [43, 157]]`

## Observations

- Both federated models trail the centralized baseline on accuracy, macro-F1,
  AUC, and sensitivity, and this is expected: the centralized model saw all
  6,600 training images each epoch, while each client sees a small, skewed
  local shard for a single local epoch per round.
- Sensitivity is the weak metric for every model. Only 200 of 1,800 validation
  images are TB-positive, and client 2 contributes TB-only data, so the
  aggregated model is pulled toward the majority class.
- Under this configuration FedAvg outperformed FedProx on every headline metric.
  The proximal term (`mu=0.01`) restricted local drift enough to slow early
  rounds, and five rounds were not sufficient for FedProx to catch up. This is a
  result of one configuration, not a general claim that FedProx is worse.
- These are validation numbers from a single seed on one host. Neither result is
  a test-set estimate, and the unlabeled test split was not used.

## Artifacts and reproduction

- Best models: `artifacts/checkpoints/tb_federated_fedavg_best.pt`,
  `artifacts/checkpoints/tb_federated_fedprox_best.pt`
- Last models: `artifacts/checkpoints/tb_federated_fedavg_last.pt`,
  `artifacts/checkpoints/tb_federated_fedprox_last.pt`
- Per-round validation history:
  `artifacts/outputs/federated_fedavg_results.json`,
  `artifacts/outputs/federated_fedprox_results.json`
- Comparison table: `artifacts/outputs/federated_comparison.md`
- Client split CSVs: `data/federated_splits/client_{0,1,2}_train.csv`
- Hydra entry point: `python -m src.training.train_federated`

Reproduction commands for the recorded runs:

```
python -m src.training.train_federated fl.num_rounds=5 fl.batch_size=16
python -m src.training.train_federated fl.strategy.name=fedprox fl.num_rounds=5 fl.batch_size=16
```

Verification and comparison helpers (no retraining, they read the saved
artifacts):

```
python scripts/evaluate_federated_checkpoint.py --checkpoint artifacts/checkpoints/tb_federated_fedavg_best.pt
python scripts/compare_federated_results.py
```

`evaluate_federated_checkpoint.py` recomputes metrics from a checkpoint on the
validation split. Both saved checkpoints were re-evaluated this way and matched
their stored metrics exactly (delta `0.0000` on every metric).

FedAvg explicitly disables the proximal penalty while FedProx passes
`proximal_mu` through the fit configuration to each client. These are simulated
clients using one dataset and one host; the experiment does not demonstrate
production cross-silo privacy or institutional generalization.
