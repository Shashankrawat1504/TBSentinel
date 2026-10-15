<<<<<<< HEAD
# TBSentinel

**Federated Multimodal Clinical Decision Support System for Tuberculosis and Pneumonia Detection & Assessment using Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Federated%20Learning-Flower-green.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototype-brightgreen.svg)]()

---

## Abstract

TBSentinel is a privacy-preserving, federated multimodal clinical decision support system designed for the detection and severity assessment of Tuberculosis (TB) and Pneumonia from chest X-ray images combined with clinical tabular data.  

The system enables multiple healthcare institutions to collaboratively train a robust deep learning model **without sharing raw patient data**, addressing critical challenges of data privacy, data heterogeneity, and limited labeled medical datasets.  

It integrates:
- Multimodal fusion of imaging and clinical features
- Federated Learning (FedAvg / FedProx)
- Explainable AI techniques (Grad-CAM and SHAP) for clinical interpretability
- A ready-to-deploy inference service and clinician-friendly interface

TBSentinel aims to support early and accurate diagnosis in resource-constrained and privacy-sensitive healthcare environments while maintaining transparency and trustworthiness for clinical users.

---

## Key Features

- **Multi-Disease Detection** – Supports both binary (TB vs Non-TB) and multi-class (Normal / TB / Pneumonia) classification
- **Federated Learning** – Privacy-preserving collaborative training across institutions

- **Federated Learning** – Privacy-preserving collaborative training across institutions
- **Multimodal Architecture** – Combines Chest X-ray images + clinical/tabular data
- **Explainable AI** – Grad-CAM heatmaps + SHAP feature importance
- **Clinical Decision Support** – Diagnosis probabilities, confidence scores, and structured reports
- **Deployable Prototype** – Fully containerized with Docker Compose
- **Modular & Reproducible** – Configuration-driven design with experiment tracking

---

## System Architecture

```mermaid
flowchart TB
    subgraph Clients["Client Institutions (Simulated Hospitals)"]
        C1[Client 1<br/>Local Data + Model]
        C2[Client 2<br/>Local Data + Model]
        C3[Client 3<br/>Local Data + Model]
    end

    subgraph Server["Central Aggregation Server"]
        FL[Flower / FedAvg / FedProx<br/>Model Aggregation]
        GM[Global Model<br/>Versioning]
    end

    subgraph Inference["Inference & Serving Layer"]
        API[FastAPI Service<br/>/predict + /explain]
        XAI[Explainability Module<br/>Grad-CAM + SHAP]
    end

    subgraph Frontend["Clinician Interface"]
        UI[Streamlit / Gradio Dashboard]
    end

    C1 & C2 & C3 <-->|Model Updates Only| FL
    FL --> GM
    GM --> API
    API --> XAI
    XAI --> UI
    UI -->|Image + Clinical Data| API

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for training)
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TBSentinel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Clinical Decision Support Prototype

The TBSentinel prototype provides a ready-to-use clinical decision support interface with TB prediction and Grad-CAM explainability.

#### 1. Start the FastAPI Backend

```bash
python -m src.inference.main
```

The API will be available at `http://127.0.0.1:8000` with interactive docs at `http://127.0.0.1:8000/docs`.

#### 2. Start the Streamlit Frontend

In a new terminal:

```bash
streamlit run frontend/app.py
```

The dashboard (with its built-in login gate) will open in your browser at `http://localhost:8501`.

#### 3. Login

- **Username**: `admin`
- **Password**: `tbsentinel2026`

After successful login, you will be redirected to the main dashboard.

#### Running on this machine only (default)

The inference API has no authentication and the dashboard is protected by a
single shared password, so both bind to **loopback only** by default — they are
reachable from this machine and not from the local network:

- API: `.venv/Scripts/python.exe -m src.inference.main` → `127.0.0.1:8000`
- Dashboard: `.venv/Scripts/streamlit.exe run frontend/app.py` → `127.0.0.1:8501`

For a deliberate LAN demo (another device has to reach them), opt in per run:

```bash
# Windows PowerShell
$env:TBSENTINEL_API_HOST = "0.0.0.0"; python -m src.inference.main
streamlit run frontend/app.py --server.address 0.0.0.0
```

```bash
# macOS / Linux / Git Bash
TBSENTINEL_API_HOST=0.0.0.0 python -m src.inference.main
streamlit run frontend/app.py --server.address 0.0.0.0
```

The API logs a warning whenever it binds to a wildcard address, and
docs/INFERENCE_GUIDE.md walks through the same overrides.

#### Or: run the whole system with Docker (one command)

Docker is the fastest way to run the project without touching a virtual
environment — it needs only Docker Desktop (or Docker Engine + Compose v2), and
the model weights stay on your machine:

```bash
docker compose up --build
```

Then open `http://localhost:8501` and sign in. That single command builds both
images, starts the API, waits for its healthcheck and starts the dashboard
pointing at it. Full guide: [docs/DOCKER.md](docs/DOCKER.md).

#### 4. Use the Interface

- Upload a chest X-ray image (JPG/PNG)
- Click "Run assessment" to get the prediction
- View the three-class result — Normal / TB / Pneumonia — with a colour-coded badge
  (green / red / amber), the confidence score and the full probability distribution
- See the Grad-CAM explainability panel: **original, activation map and overlay side by side**,
  rendered in a perceptual `turbo` colormap with the map smoothed and peak-normalised
- Inspect **any class, not just the predicted one** — the panel's class selector re-renders the map
  (and the focus description below it) for Normal, TB or Pneumonia on the same radiograph
- Read the focus line: which region carried the activation (side and zone, in radiological
  convention), how much of the image was active and how strong the peak was
- Read the structured clinical explanation served from `configs/inference/inference.yaml`:
  a banded confidence assessment (High / Moderate / Low with the runner-up margin), the
  clinical impression, plausibility and safety notes, a ranked differential over the other
  classes, generated warnings for indeterminate screens, and the standing disclaimer
- Reopen any saved assessment in Patient History and get the same explanation back verbatim
- Save the assessment to the patient record store; review, filter and export it under Patient History

⚠️ **Medical Disclaimer**: This is a research prototype. Do not use for clinical diagnosis. Always consult qualified healthcare professionals.

For detailed inference documentation, see [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md).

### Authentication

The dashboard includes a simple authentication layer for this research prototype:

- **Default Credentials**:
  - Username: `admin`
  - Password: `tbsentinel2026`

- **Session Management**:
  - Sessions expire after 60 minutes of inactivity
  - Logout button available in the sidebar
  - Credentials come from `TBSENTINEL_USERNAME` / `TBSENTINEL_PASSWORD` when set, and otherwise fall back to the demo values above in `frontend/auth_config.py`

- **Security Note**: This is a lightweight authentication system suitable for a research prototype. For production deployment, implement proper authentication (OAuth, database-backed auth, or enterprise identity provider).

To change credentials, set the environment variables (preferred, keeps the
password out of the source tree) — the login page reminds you while the shipped
demo password is still active:

```bash
# Windows PowerShell
$env:TBSENTINEL_USERNAME = "your_username"; $env:TBSENTINEL_PASSWORD = "your_password"
streamlit run frontend/app.py
```

```bash
# macOS / Linux / Git Bash
export TBSENTINEL_USERNAME=your_username TBSENTINEL_PASSWORD='your_password'
streamlit run frontend/app.py
```

Alternatively edit `frontend/auth_config.py`, which is where the app reads the
fallback defaults.

---

## Project Structure

```
TBSentinel/
├── configs/                 # Hydra configuration files
│   ├── inference/           # Inference API configuration
│   ├── model/               # Model architecture config
│   ├── data/               # Data loading config
│   └── config.yaml         # Main config
├── src/
│   ├── data/               # Dataset and transforms
│   ├── models/             # Model architectures (EfficientNet-B3)
│   ├── training/           # Training scripts (centralized & federated)
│   ├── federation/         # Federated learning clients and strategies
│   ├── xai/                # Explainable AI (Grad-CAM, SHAP)
│   ├── inference/          # Inference service and FastAPI backend
│   └── utils/              # Logging, seeding, utilities
├── frontend/               # Streamlit clinical UI
│   └── app.py             # Main Streamlit application
├── artifacts/             # Trained models, logs, outputs
│   └── checkpoints/       # Saved model checkpoints
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

---

## Model Training

### Centralized Training

```bash
python -m src.training.train_centralized
```

### Federated Learning Simulation

```bash
python -m src.federation.run_fl_simulation
```

---

## Multi-Disease Support

TBSentinel now supports detection of both Tuberculosis and Pneumonia from chest X-rays.

### Supported Classifications

1. **Multi-Class Classification** (Default served model)
   - Classes: Normal (0), TB (1), Pneumonia (2)
   - Dataset: Combined TBX11K TB + RSNA Normal / Lung Opacity
   - Checkpoint: `artifacts/checkpoints/multiclass_efficientnet_b3_best.pt`
   - Configuration: `configs/inference/inference.yaml` (`model_type: multiclass`)

2. **Binary TB Classification** (earlier phase, still supported)
   - Classes: Non-TB (0), TB (1)
   - Dataset: TBX11K
   - Checkpoint: `artifacts/checkpoints/tb_efficientnet_b3_best.pt`

3. **Binary Pneumonia Classification**
   - Classes: Normal (0), Pneumonia (1)
   - Dataset: RSNA Pneumonia

4. **Federated TB models** (FedAvg / FedProx) used for the federated comparison
   - Checkpoints: `artifacts/checkpoints/tb_federated_{fedavg,fedprox}_best.pt`

### Preparing Multi-Class Dataset

```bash
# Step 1: Prepare Pneumonia data
python scripts/prepare_pneumonia_data.py

# Step 2: Create combined multi-class dataset
python scripts/create_multiclass_dataset.py
```

This creates the 12,029-image multi-class splits (8,419 train / 1,805 validation /
1,805 test) covering 4,499 Normal, 3,360 Pneumonia and 560 TB training images.
The three classes are deliberately unbalanced — TB is the minority class — so
per-class recall and macro-F1 are the metrics that matter, not accuracy.

### Selecting the Served Model

The served task is defined by two keys in `configs/inference/inference.yaml`, so
switching models never requires a code change and the number of classes is read
from the checkpoint itself:

```yaml
model_type: multiclass
checkpoint_path: ${paths.checkpoints_dir}/multiclass_efficientnet_b3_best.pt
class_names: [normal, tb, pneumonia]
```

Point `checkpoint_path` at a binary checkpoint and set `class_names: [tb, non_tb]`
to serve the binary model again. The clinical wording for every class lives in
the same file under `clinical:`, so the API and the dashboard cannot drift apart.

For detailed instructions, see [docs/MULTICLASS_INTEGRATION_GUIDE.md](docs/MULTICLASS_INTEGRATION_GUIDE.md).

### Verifying the Three-Class System

With the API running, these scripts check the live service end to end (real
radiographs from the held-out test split, not stubs) and exit non-zero on any
failure:

```bash
python scripts/verify_inference_3class.py     # /health, /predict contract, Grad-CAM, error handling
python scripts/verify_dashboard_3class.py     # New Assessment -> save -> Patient History -> Dashboard -> Metrics -> About
```

---

---

## Explainability

Generate Grad-CAM explanations for validation images:

```bash
python -m src.xai.explain
```

---

## Configuration

All training and inference settings are managed via Hydra configs in the `configs/` directory. Modify `configs/config.yaml` and sub-configs to adjust:

- Model architecture and hyperparameters
- Data paths and preprocessing
- Training settings (epochs, batch size, learning rate)
- Federated learning strategy (FedAvg/FedProx)
- Inference settings (model path, device, image size)

---

## Model Checkpoints

The following trained models are available in `artifacts/checkpoints/`:

- `multiclass_efficientnet_b3_best.pt` - **default served model**: Normal / TB / Pneumonia
  (95.51% accuracy, 0.9671 macro-F1, 0.9909 macro AUC on the held-out test split — see
  `artifacts/outputs/multiclass_test_metrics.json`)
- `tb_efficientnet_b3_best.pt` - Centralized binary TB EfficientNet-B3
- `tb_federated_fedavg_best.pt` - Federated Learning with FedAvg strategy
- `tb_federated_fedprox_best.pt` - Federated Learning with FedProx strategy

To switch between models, edit `configs/inference/inference.yaml` (and the matching
`class_names`):

```yaml
class_names: [normal, tb, pneumonia]
checkpoint_path: ${paths.checkpoints_dir}/multiclass_efficientnet_b3_best.pt  # default
# class_names: [tb, non_tb]
# checkpoint_path: ${paths.checkpoints_dir}/tb_efficientnet_b3_best.pt  # binary Centralized
# checkpoint_path: ${paths.checkpoints_dir}/tb_federated_fedavg_best.pt  # FedAvg
# checkpoint_path: ${paths.checkpoints_dir}/tb_federated_fedprox_best.pt  # FedProx
```

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_inference_api.py -v
pytest tests/test_inference_multiclass.py -v
pytest tests/test_gradcam.py -v
```

For the frontend storage/theme/metric helpers:

```bash
python frontend/_smoke_test.py
```

For live, non-stubbed end-to-end checks of the three-class system (API must be
running), see [Verifying the Three-Class System](#verifying-the-three-class-system).

---

## Docker Deployment

Two images, one command, no virtual environment to manage:

```bash
docker compose up --build      # API on 127.0.0.1:8000, dashboard on 127.0.0.1:8501
docker compose up --build -d   # detached
docker compose logs -f api     # follow the inference service
docker compose down            # stop; weights and patient records are kept
```

| Service | Image | Port (host) | What it runs |
|---|---|---|---|
| `api` | `tbsentinel-api` | `127.0.0.1:8000` | FastAPI + EfficientNet-B3 + Grad-CAM (`python -m src.inference.main`) |
| `dashboard` | `tbsentinel-dashboard` | `127.0.0.1:8501` | Streamlit clinical dashboard (`streamlit run frontend/app.py`) |

What the setup does deliberately:

- **Weights are mounted, not baked in** — `artifacts/checkpoints/` is bind-mounted read-only, so
  retraining never forces an image rebuild and a 129 MB checkpoint stays out of the layers.
- **Config is mounted read-only** — edit `configs/inference/inference.yaml` (thresholds, colormap,
  clinical copy) and `docker compose restart api` picks it up. No rebuild.
- **Patient records live on the host** — `data/patient_records/` is a volume, so the SQLite store
  survives `docker compose down` and stays inspectable at
  `data/patient_records/assessments.db`.
- **Loopback only** — both ports are published on `127.0.0.1`. The containers bind `0.0.0.0`
  internally because a published port cannot reach a loopback-only listener, but nothing outside
  this machine can connect.
- **No CUDA by default** — torch is installed from the CPU wheel index, pinned to `2.5.1` (the
  version these checkpoints were trained with). On a CUDA host:
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 docker compose up --build`.
- **Credentials via `.env`** — set `TBSENTINEL_USERNAME` / `TBSENTINEL_PASSWORD` in a `.env` file
  next to `docker-compose.yml` to replace the shipped demo login.

If `artifacts/checkpoints/multiclass_efficientnet_b3_best.pt` is missing the API container stops
with a `FileNotFoundError` naming the expected path; see
[docs/DOCKER.md](docs/DOCKER.md#troubleshooting) for that and the other common cases.

---

## Citation

If you use TBSentinel in your research, please cite:

```bibtex
@software{tbsentinel2024,
  title={TBSentinel: Federated Multimodal Clinical Decision Support for Tuberculosis Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TBSentinel}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- EfficientNet architecture by Google Research
- Flower framework for Federated Learning
- PyTorch Grad-CAM for explainability
- Medical imaging community for datasets and best practices

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact [your-email@example.com].
=======
# TBSentinel
>>>>>>> 3cf9965a09ac1ed16510ed4277cdae5a106c8d70
