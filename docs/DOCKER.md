# Running TBSentinel with Docker

Two containers, one command, no virtual environment to manage:

```bash
docker compose up --build
```

Then open **http://localhost:8501** and sign in (demo credentials `admin` /
`tbsentinel2026`; see [Credentials](#credentials) to replace them).

The first build downloads the CPU build of PyTorch (~200 MB) plus the serving
dependencies, so expect a few minutes. Later runs start in seconds.

---

## What is required

| Requirement | Notes |
|---|---|
| Docker Desktop, or Docker Engine with Compose v2 | `docker compose version` must work (v2 syntax) |
| The trained checkpoint | `artifacts/checkpoints/multiclass_efficientnet_b3_best.pt` must exist — it is mounted into the API container, never copied into an image |
| ~2 GB free disk | CPU image + weights + build cache |

Nothing else is needed: Python, CUDA and the virtual environment are not
involved when you run the Compose stack.

```bash
# Confirm the checkpoint is where the API expects it
ls artifacts/checkpoints/multiclass_efficientnet_b3_best.pt
```

---

## What gets built

| Service | Image | Host port | Process |
|---|---|---|---|
| `api` | `tbsentinel-api` | `127.0.0.1:8000` | `python -m src.inference.main` (FastAPI + EfficientNet-B3 + Grad-CAM) |
| `dashboard` | `tbsentinel-dashboard` | `127.0.0.1:8501` | `streamlit run frontend/app.py` |

* `docker/Dockerfile.api` — installs CPU torch/torchvision (pinned to `2.5.1` /
  `0.20.1`, the versions the checkpoints were trained with) from the PyTorch
  wheel index, then `docker/requirements-api.txt`, then runs
  `python -c "import src.inference.api"` as a **build-time dependency check**: a
  missing runtime package fails the build with a clear error instead of showing up
  as a 500 during a demo.
* `docker/Dockerfile.frontend` — installs `frontend/requirements.txt` only
  (Streamlit, requests, Pillow) and starts Streamlit with
  `--server.address=0.0.0.0`, overriding the loopback default in
  `.streamlit/config.toml` (CLI flags beat the config file). It also
  compiles every dashboard module at build time.
* Both containers run as an unprivileged user (`appuser`, UID 1000) and both
  declare a `HEALTHCHECK`; the dashboard waits for `condition: service_healthy`
  on the API, so the first page load already has a model behind it.

---

## Volumes and persistence

| Host path | Container path | Mode | Why |
|---|---|---|---|
| `artifacts/checkpoints/` | `/app/artifacts/checkpoints` | read-only | model weights stay ~129 MB out of the image and are reused after retraining |
| `configs/` | `/app/configs` | read-only | thresholds, colormap and clinical copy can be edited without a rebuild |
| `data/patient_records/` | `/app/data/patient_records` | read-write | the SQLite patient store survives `docker compose down` |

The record database stays a normal file on your machine:

```bash
sqlite3 data/patient_records/assessments.db "select record_id, patient_id, predicted_class, confidence from assessments order by id desc limit 5;"
```

Grad-CAM artifacts are stored inside that database as BLOBs, so nothing is lost
when the containers are removed.

---

## Common commands

```bash
docker compose up --build            # build and start everything (foreground)
docker compose up --build -d         # detached
docker compose logs -f api           # follow the inference log
docker compose ps                    # container + health status
docker compose restart api           # after editing configs/inference/inference.yaml
docker compose up --build api        # rebuild after changing src/
docker compose exec api python -c "import urllib.request,json;print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')))"
docker compose down                  # stop and remove containers (volumes and weights kept)
docker compose down --rmi local      # also drop the two built images
```

### Credentials

The dashboard reads `TBSENTINEL_USERNAME` / `TBSENTINEL_PASSWORD`. Put them in a
`.env` file next to `docker-compose.yml` so the password never sits in shell
history:

```dotenv
TBSENTINEL_USERNAME=dr.rao
TBSENTINEL_PASSWORD=<a strong password>
```

Then `docker compose up -d dashboard` (or `docker compose up --build`) — the
login page stops showing the "demo password" reminder once the shipped default is
no longer in use. See [AUTHENTICATION.md](AUTHENTICATION.md).

### GPU

CPU is the default because it keeps the image ~1 GB instead of ~5 GB. On a CUDA
host with the NVIDIA container runtime installed:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 docker compose up --build
```

The API reports the device it actually loaded on, visible at
`http://127.0.0.1:8000/health` (`"device": "cuda"` or `"cpu"`) and in the
dashboard sidebar. This project does not ship an NVIDIA `deploy:` reservation, so
add one under the `api` service if your Docker version needs it:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Ports

Both ports are published on `127.0.0.1` only, matching the project's
local-first stance: the API is unauthenticated and the dashboard is protected by
a single shared password. To move them, edit the `ports:` entries in
`docker-compose.yml` (`"127.0.0.1:8001:8000"`). To expose the dashboard to your
LAN for a deliberate demo, change the host part to `0.0.0.0:8501:8501` and
understand that anyone on the network will then reach the login page.

---

## Verifying the running stack

```bash
# 1. API health (expect "status": "healthy", model_loaded: true, class_names with 3 entries)
curl http://127.0.0.1:8000/health

# 2. A real prediction through the API
curl -F "file=@data/raw/Pneumonia/chest x ray/Test/0000a175-0e68-4ca4-b1af-167204a7e0bc.png" \
     http://127.0.0.1:8000/explain | head -c 300

# 3. Dashboard
#    Open http://localhost:8501, sign in, run an assessment, save it, then
#    reopen it under Patient History - the stored explanation must match.
```

The dashboard sidebar shows the served checkpoint, device, class list and the
"System ready" pill; if it says *Inference API offline*, the API container is not
healthy yet — check `docker compose logs api`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: Checkpoint not found: /app/artifacts/checkpoints/...` | the host checkpoint is missing or named differently; put `multiclass_efficientnet_b3_best.pt` in `artifacts/checkpoints/` or point `checkpoint_path` at the right file in `configs/inference/inference.yaml` |
| Dashboard shows *Inference API offline* | the API is still starting (first start loads the model and can take a minute) or `TBSENTINEL_API_URL` was changed away from `http://api:8000` |
| `Bind for 127.0.0.1:8000 failed: port is already allocated` | a local (non-Docker) API is still running: stop it, or change the host port in `docker-compose.yml` |
| `permission denied` writing `data/patient_records` (Linux hosts) | the container user is UID 1000; `sudo chown -R 1000:1000 data/patient_records` once, or run the dashboard service with `user: "0:0"` |
| Build fails at `import src.inference.api` | a serving dependency is missing from `docker/requirements-api.txt` — add it there (the check exists exactly to catch this) |
| Build stalls downloading torch | CPU wheels are ~200 MB; behind a proxy set `HTTP_PROXY`/`HTTPS_PROXY` for the Docker daemon, or build with `--build-arg TORCH_INDEX_URL=<your mirror>` |
| `ERROR: Docker Compose requires buildx` / `no configuration file provided` | you are on Compose v1 or in the wrong directory; run `docker compose version` from the repository root |
| Auth returns 503 in a browser-based deployment | unrelated to Docker; see the auth notes in [AUTHENTICATION.md](AUTHENTICATION.md) |

---

## What is intentionally *not* containerised

* **Training and federated simulation** (`src/training/`, `src/federation/`,
  `scripts/create_*`) — these are heavy, GPU-bound and dataset-driven. Run them in
  the project virtual environment as described in the README.
* **Datasets** — `data/raw/` is large and excluded from the build context by
  `.dockerignore`; only the patient-record directory is mounted.
* **Notebooks** — excluded for the same reason.
