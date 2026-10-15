"""Inference service for chest X-ray classification with Grad-CAM explainability.

The served task comes from the checkpoint: the multi-class model answers
Normal / TB / Pneumonia, while the binary TB checkpoints remain loadable. Every
method here is class-count agnostic so both keep working.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from src.data.dataset import build_eval_transform
from src.inference.clinical import build_interpretation
from src.models.image_model import TBImageClassifier
from src.utils.logger import get_logger
from src.xai.focus import DEFAULT_THRESHOLD, describe_focus
from src.xai.gradcam import TBGradCAMExplainer

LOGGER = get_logger("inference.service")

_MARKUP_PATTERN = re.compile(r"<[^>]+>")


def _strip_markup(text: str) -> str:
    """Drop HTML tags so configured headlines read well as plain text."""
    return _MARKUP_PATTERN.sub("", text).strip()


class InferenceService:
    """Inference service for chest X-ray classification with explainability.

    This service loads a trained model checkpoint and provides methods for:
    - Classification over the checkpoint's classes (binary TB or three-class)
    - Confidence scoring and the full class probability distribution
    - Grad-CAM heatmap generation for clinical interpretation
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        image_size: int = 300,
        confidence_threshold: float = 0.5,
        class_names: tuple[str, ...] = ("normal", "tb", "pneumonia"),
        interpretation_config: dict[str, Any] | None = None,
        gradcam_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the inference service with a trained model checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file).
            device: Computing device ('cuda', 'cpu', or torch.device). Auto-selected
                when None.
            image_size: Target square image dimension for preprocessing.
            confidence_threshold: Minimum confidence threshold for a positive call.
            class_names: Class names in model output order; must match the
                checkpoint head (two classes for the binary TB models, three for
                the Normal / TB / Pneumonia model).
            interpretation_config: Optional ``clinical`` block from the inference
                configuration used to word clinical summaries.
            gradcam_config: Optional ``gradcam`` block (colormap, alpha,
                eigen_smooth, blur/clip settings, focus threshold) so the served
                explanations follow the configured rendering.
        """
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self.interpretation_config = interpretation_config or {}
        self.gradcam_config = dict(gradcam_config or {})
        self.focus_threshold = float(
            self.gradcam_config.get("focus_threshold", DEFAULT_THRESHOLD)
        )

        # Device selection
        if device is None or str(device).lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Initialize Grad-CAM explainer
        self.explainer = TBGradCAMExplainer(
            model=self.model,
            target_layer=self.gradcam_config.get("target_layer") or None,
            device=self.device,
            class_names=list(class_names),
            image_size=image_size,
            colormap=self.gradcam_config.get("colormap", "turbo"),
            alpha=float(self.gradcam_config.get("alpha", 0.55)),
            eigen_smooth=bool(self.gradcam_config.get("eigen_smooth", False)),
            blur_kernel=int(self.gradcam_config.get("blur_kernel", 9)),
            clip_percentile=float(self.gradcam_config.get("clip_percentile", 100.0)),
        )

        # Preprocessing transform
        self.eval_transform = build_eval_transform(image_size=image_size)

        LOGGER.info(
            "InferenceService initialized: checkpoint=%s, device=%s, image_size=%d",
            self.checkpoint_path,
            self.device,
            image_size,
        )

    def _load_model(self) -> TBImageClassifier:
        """Load model from checkpoint."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        LOGGER.info("Loading model checkpoint from: %s", self.checkpoint_path)
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config = checkpoint.get("config", {})
        model_config = config.get("model", {}) if isinstance(config, dict) else {}

        num_classes = model_config.get("num_classes", 2)
        dropout = model_config.get("dropout", 0.3)

        model = TBImageClassifier(
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
            num_tabular_features=0,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)

        LOGGER.info("Model loaded successfully")
        return model

    def predict(
        self,
        image_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """Run inference on a preprocessed image tensor.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, 3, H, W).

        Returns:
            Dictionary containing:
                - predicted_class_id: int (index into ``class_names``)
                - predicted_class_name: str
                - confidence: float (probability of predicted class)
                - probabilities: dict[str, float] (all classes)
                - logits: list[float]
        """
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected tensor shape (1, 3, H, W), got {tuple(image_tensor.shape)}"
            )

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        predicted_class_id = int(torch.argmax(probs).item())
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = float(probs[predicted_class_id].item())

        probabilities_dict = {
            self.class_names[i]: float(probs[i].item()) for i in range(len(probs))
        }

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "logits": logits.squeeze(0).cpu().tolist(),
        }

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> dict[str, Any]:
        """Generate Grad-CAM explanation for a preprocessed image tensor.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to explain. ``None`` explains the predicted
                class; any configured class can be asked for, which is how the
                dashboard shows what each class is looking at.

        Returns:
            Dictionary containing prediction and Grad-CAM visualization data:
                - predicted_class_id: int
                - predicted_class_name: str
                - confidence: float
                - probabilities: dict[str, float]
                - target_class_id: int  (class the heatmap explains)
                - target_class_name: str
                - heatmap: np.ndarray (H, W) float32 in [0, 1] (refined)
                - heatmap_color: np.ndarray (H, W, 3) uint8 RGB heatmap
                - overlay: np.ndarray (H, W, 3) uint8 RGB in [0, 255]
                - rgb_image: np.ndarray (H, W, 3) float32 RGB in [0, 1]
                - focus: dict (region-level reading of the heatmap)
        """
        explanation = self.explainer.explain(
            image=image_tensor, target_category=target_class
        )
        # Region-level reading of the map: where the model looked, not just how
        # strongly. Attached here so every client (API, scripts, dashboard) gets
        # the same structured focus.
        explanation["focus"] = describe_focus(
            explanation["heatmap"], threshold=self.focus_threshold
        )
        return explanation

    def get_clinical_summary(
        self,
        prediction: dict[str, Any],
    ) -> str:
        """Generate a plain-text clinical summary for one prediction.

        Uses the configured clinical interpretation when the service was given
        one, otherwise falls back to a class-agnostic sentence built from the
        full probability map — so the wording stays correct for the binary and
        the three-class model alike.

        Args:
            prediction: Prediction dictionary from predict() method.

        Returns:
            Clinical summary text string.
        """
        class_name = str(prediction["predicted_class_name"])
        confidence = float(prediction["confidence"])
        probabilities = {
            str(name): float(value)
            for name, value in (prediction.get("probabilities") or {}).items()
        }

        interpretation = build_interpretation(
            class_name,
            confidence,
            probabilities,
            self.interpretation_config,
        )
        if interpretation is not None:
            lines = [_strip_markup(str(interpretation.get("headline") or ""))]
            lines.extend(
                f"- {_strip_markup(str(note))}"
                for note in interpretation.get("notes") or []
            )
            summary = "\n".join(line for line in lines if line)
            if summary:
                return summary

        distribution = ", ".join(
            f"{name.upper()}: {value:.1%}" for name, value in probabilities.items()
        )
        return (
            f"The model predicts {class_name.upper()} with "
            f"{confidence:.1%} confidence. Probability distribution: "
            f"{distribution}. The Grad-CAM heatmap highlights the regions that "
            "contributed most to this prediction. Clinical correlation and "
            "confirmatory testing are recommended."
        )
