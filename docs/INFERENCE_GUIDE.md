# TBSentinel Clinical Decision Support - Inference Guide

This guide explains how to run the TBSentinel Clinical Decision Support prototype, which includes a FastAPI backend for model serving and a Streamlit frontend for clinical visualization.

## Overview

The inference system consists of three main components:

1. **Inference Service** (`src/inference/service.py`) - Core prediction and Grad-CAM explanation logic
2. **FastAPI Backend** (`src/inference/api.py`) - REST API with endpoints for prediction and explanation
3. **Streamlit Frontend** (`frontend/app.py`) - Clinical UI for uploading images and viewing results

## Prerequisites

- Python 3.10+
- Trained model checkpoint in `artifacts/checkpoints/`
- All dependencies installed (see `requirements.txt`)

## Configuration

The inference system is configured via `configs/inference/inference.yaml`:

```yaml
# Model selection (the served task follows the checkpoint)
model_type: multiclass  # multiclass (normal/tb/pneumonia) or binary_tb

# Model paths (relative to project root)
checkpoint_path: ${paths.checkpoints_dir}/multiclass_efficientnet_b3_best.pt

# Runtime settings
device: auto  # auto, cuda, or cpu
image_size: 300
confidence_threshold: 0.5

# Class names, in model output order
class_names:
  - normal
  - tb
  - pneumonia

# Grad-CAM settings
gradcam:
  target_layer: auto
  colormap: jet
  alpha: 0.5

# API settings (loopback by default: the API has no authentication)
api:
  host: 127.0.0.1
  port: 8000
  reload: false
  log_level: info
```

### Switching Between Models

To serve a different checkpoint, edit `checkpoint_path` **and** `class_names` in
`configs/inference/inference.yaml`; the class count comes from the checkpoint:

```yaml
# Binary TB model
checkpoint_path: ${paths.checkpoints_dir}/tb_efficientnet_b3_best.pt
class_names: [tb, non_tb]

# For FedAvg model (binary TB)
checkpoint_path: ${paths.checkpoints_dir}/tb_federated_fedavg_best.pt

# For FedProx model (binary TB)
checkpoint_path: ${paths.checkpoints_dir}/tb_federated_fedprox_best.pt
```

## Running the FastAPI Backend

Both entry points read the `api:` block above, and both bind to **loopback** by
default. The API has no authentication, so anything that can reach the port can
call `/predict` and `/explain`; only widen the bind deliberately.

### Option 1: Direct Python Execution

```bash
python -m src.inference.main
```

This starts the server on `http://127.0.0.1:8000`, reachable from this machine
only, with interactive docs at `http://127.0.0.1:8000/docs`.

### Option 2: Using Uvicorn Directly

```bash
uvicorn src.inference.api:app --host 127.0.0.1 --port 8000
```

### Option 3: With Configuration (Development)

For development with auto-reload:

```bash
uvicorn src.inference.api:app --host 127.0.0.1 --port 8000 --reload --log-level debug
```

### Exposing the API on the Network (LAN demo only)

For a demo where another device must reach the backend, opt in per run instead
of editing the config:

```bash
TBSENTINEL_API_HOST=0.0.0.0 python -m src.inference.main
TBSENTINEL_API_PORT=8001 python -m src.inference.main
```

Anyone who can reach that port can run predictions without logging in. The
server logs a warning when it binds to a wildcard address.

### API Endpoints

Once the server is running, the following endpoints are available:

#### 1. Health Check

```bash
GET /health
```

Returns the API status, the served checkpoint, the task name and the class
list the model actually answers. The dashboard reads these fields, so it adapts
to a binary or a three-class checkpoint without a code change.

**Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "checkpoint_path": "/path/to/multiclass_efficientnet_b3_best.pt",
  "device": "cuda",
  "task": "multiclass_normal_tb_pneumonia",
  "class_names": ["normal", "tb", "pneumonia"],
  "num_classes": 3
}
```

#### 2. Prediction

```bash
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```

Returns the predicted class, its confidence, the full probability distribution
and the configured clinical interpretation for that class.

**Example Response:**
```json
{
  "predicted_class_id": 1,
  "predicted_class_name": "tb",
  "confidence": 0.89,
  "probabilities": {
    "normal": 0.06,
    "tb": 0.89,
    "pneumonia": 0.05
  },
  "logits": [-2.8, 2.1, -1.9],
  "interpretation": {
    "label": "Tuberculosis",
    "tone": "error",
    "headline": "Model screen: <b>tuberculosis</b> pattern with 89.0% confidence (TB 89.0%).",
    "notes": ["..."]
  }
}
```

A mislabelled or truncated upload (an `image/png` content type that is not a
PNG) returns **400**, not a 500, so a client can tell a bad request from a broken
model.

#### 3. Explanation (Grad-CAM)

```bash
POST /explain
Content-Type: multipart/form-data

file: <image_file>
```

Returns prediction with Grad-CAM visualizations as base64-encoded images.
`target_class_name` states which class the heatmap explains, so a client never
has to assume it from the predicted class.

**Example Response:**
```json
{
  "predicted_class_id": 1,
  "predicted_class_name": "tb",
  "confidence": 0.89,
  "probabilities": {
    "normal": 0.06,
    "tb": 0.89,
    "pneumonia": 0.05
  },
  "target_class_id": 1,
  "target_class_name": "tb",
  "heatmap_base64": "...",
  "overlay_base64": "...",
  "rgb_image_base64": "..."
}
```

### Testing the API

You can test the API using curl:

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST -F "file=@path/to/xray.png" http://localhost:8000/predict

# Explanation
curl -X POST -F "file=@path/to/xray.png" http://localhost:8000/explain
```

Or use the interactive API documentation at `http://localhost:8000/docs` (Swagger UI).

## Running the Streamlit Frontend

### Start the Frontend

```bash
streamlit run frontend/app.py
```

This launches the Streamlit application at `http://localhost:8501` and opens it
in your browser. Streamlit is pinned to loopback in `.streamlit/config.toml`; for
a LAN demo use `--server.address 0.0.0.0` on purpose.

### Frontend Features

The Streamlit frontend provides:

1. **Image Upload** - Drag and drop or browse to upload chest X-ray images
2. **Prediction Display** - Shows TB/Non-TB classification with confidence
3. **Grad-CAM Visualization** - Displays heatmap and overlay for explainability
4. **Clinical Summary** - Interprets the prediction in clinical context
5. **API Status** - Shows whether the backend API is healthy
6. **Medical Disclaimer** - Clear warning that this is a research prototype

### Frontend Configuration

The API URL can be changed in the sidebar settings if running the backend on a different host/port.

## Testing

### Run API Tests

```bash
pytest tests/test_inference_api.py -v
```

Note: Some tests may fail if the inference service is not properly initialized with a valid checkpoint.

## Troubleshooting

### Model Not Found Error

If you see "Checkpoint not found" error:

1. Ensure you have a trained checkpoint in `artifacts/checkpoints/`
2. Check the path in `configs/inference/inference.yaml`
3. Verify the checkpoint file exists at the specified path

### API Connection Refused

If the frontend cannot connect to the API:

1. Ensure the FastAPI server is running
2. Check the API URL in the Streamlit sidebar
3. Verify the port (default: 8000) is not blocked by firewall

### CUDA Out of Memory

If you encounter CUDA memory errors:

1. Set `device: cpu` in `configs/inference/inference.yaml`
2. Or reduce the batch size if processing multiple images

### Import Errors

If you encounter import errors:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Activate your virtual environment
3. Verify you're using Python 3.10+

## Architecture

### Inference Service

The `InferenceService` class in `src/inference/service.py` handles:

- Loading trained model checkpoints
- Preprocessing images with Albumentations transforms
- Running inference with EfficientNet-B3
- Generating Grad-CAM explanations using the existing XAI module
- Providing clinical summary text

### FastAPI Backend

The FastAPI application in `src/inference/api.py` provides:

- RESTful API endpoints
- File upload handling
- Pydantic models for request/response validation
- Error handling and logging
- Health check endpoint

### Streamlit Frontend

The Streamlit app in `frontend/app.py` provides:

- Clinical-style UI with custom CSS
- Image upload and preview
- API integration for prediction and explanation
- Base64 image decoding for visualization
- Medical disclaimer and warnings

## Future Extensions

The system is designed to be multimodal-ready. Future extensions could include:

- Tabular feature integration (age, symptoms, etc.)
- Batch processing for multiple images
- Export to DICOM format
- Integration with hospital PACS systems
- User authentication and access control
- Audit logging for clinical use

## Security Considerations

This is a research prototype. For production deployment, consider:

- Adding authentication/authorization
- Implementing rate limiting
- Securing API with HTTPS
- Adding input validation and sanitization
- Implementing audit logging
- Regular security updates

## Support

For issues or questions, refer to the main project README or open an issue in the repository.
