import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig

from src.federation.run_fl_simulation import run_federated_simulation


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run federated simulation from src/training entrypoint."""
    run_federated_simulation(config)


if __name__ == "__main__":
    main()
