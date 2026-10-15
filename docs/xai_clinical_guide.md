# TBSentinel: Clinician Guide to Explainable AI (Grad-CAM)

## 1. Overview & Clinical Purpose

In computer-aided detection (CAD) of pulmonary tuberculosis (TB), clinician trust requires transparency. Deep neural networks must not act as black boxes; radiologists and pulmonologists need to verify that positive predictions stem from genuine radiological hallmarks of TB (such as apical cavitations, fibronodular infiltrates, pleural thickening, or miliary nodules) rather than spurious artifacts (such as radiographic markers, rib contours, or scanning equipment borders).

TBSentinel provides a modular, production-ready **Gradient-weighted Class Activation Mapping (Grad-CAM)** explainer tailored to its **EfficientNet-B3** centralized image classifier.

---

## 2. Model & Target Layer Selection

- **Checkpoint:** `artifacts/checkpoints/tb_efficientnet_b3_best.pt`
- **Validation Metrics:**
  - Accuracy: **0.9878**
  - Macro-F1: **0.9679**
  - AUC-ROC: **0.9986**
  - Sensitivity: **0.9050**
  - Specificity: **0.9981**
- **Target Convolutional Layer:** `model.backbone.features[-1]`
  - In torchvision's EfficientNet-B3 implementation, `features[-1]` (index 8) is the final high-level semantic representation layer: `Conv2dNormActivation(Conv2d(384, 1536, kernel_size=1), BatchNorm2d, SiLU)`.
  - Spatial resolution at this layer is `10x10` feature map grid with 1,536 channels.
  - Grad-CAM computes gradients of the target class score with respect to these feature maps to produce localized, high-resolution heatmaps.

---

## 3. Explanation Pipeline Artifacts

Explanations are organized under `artifacts/explanations/`:

```text
artifacts/explanations/
├── xai_clinical_overview.png           # Multi-panel clinical summary collage
├── explanation_summary.json            # Machine-readable metadata & case metrics
├── explanation_summary.csv             # Tabular summary of all generated cases
├── true_positives/                     # High-confidence correct TB detections (5 cases)
│   ├── 01_tb0004_figure.png
│   ├── 01_tb0004_overlay.png
│   └── ...
├── true_negatives/                     # Correct Non-TB classifications (5 cases)
│   ├── 01_h0004_figure.png
│   ├── 01_h0004_overlay.png
│   └── ...
├── false_positives/                    # Non-TB images flagged as TB (All 3 validation FP cases)
│   ├── 01_s0936_figure.png
│   ├── 01_s0936_overlay.png
│   ├── 02_s2505_figure.png
│   ├── 02_s2505_overlay.png
│   ├── 03_s2811_figure.png
│   └── 03_s2811_overlay.png
└── false_negatives/                    # TB cases missed by the classifier (5 cases)
    ├── 01_tb0003_figure.png
    ├── 01_tb0003_overlay.png
    └── ...
```

Each case includes:
1. **`*_figure.png`**: High-resolution 3-panel figure:
   - Panel 1: Original input chest X-ray
   - Panel 2: Grad-CAM heatmap with activation colorbar
   - Panel 3: Overlaid visualization (X-Ray + Heatmap) with clinical banner
2. **`*_overlay.png`**: Standalone RGB overlay (300x300, uint8 PNG) for direct UI/web embedding.

---

## 4. How to Generate Explanations

### Option A: Via Python Convenience Script
```bash
python scripts/generate_explanations.py --output-dir artifacts/explanations --samples-per-category 5
```

### Option B: Via Hydra Configuration
```bash
python -m src.xai.explain xai.samples_per_category=5 xai.colormap=jet
```

---

## 5. Production API Integration (FastAPI Ready)

`TBGradCAMExplainer` in `src/xai/gradcam.py` is fully decoupled from dataset loaders and supports raw PIL images, file paths, numpy arrays, or preprocessed tensors.

### Example: Calling Explainer in FastAPI Endpoint

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import base64
from src.xai.gradcam import TBGradCAMExplainer

app = FastAPI(title="TBSentinel Inference Service")

# Initialize once at server startup
explainer = TBGradCAMExplainer(
    checkpoint_path="artifacts/checkpoints/tb_efficientnet_b3_best.pt",
    device="auto"
)

@app.post("/predict_and_explain")
async def predict_and_explain(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Generate prediction & Grad-CAM explanation
    result = explainer.explain(image)
    
    # Encode overlay image as base64 PNG for frontend display
    overlay_pil = Image.fromarray(result["overlay"])
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return {
        "prediction": result["predicted_class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "overlay_base64": f"data:image/png;base64,{overlay_base64}",
    }
```
