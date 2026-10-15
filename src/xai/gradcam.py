"""Explainable AI (XAI) module using Grad-CAM for TB chest X-ray classification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn

from src.data.dataset import build_eval_transform
from src.models.image_model import TBImageClassifier
from src.utils.logger import get_logger

LOGGER = get_logger("TBGradCAMExplainer")

# ImageNet normalization parameters used in model preprocessing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLORMAP_MAP: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
    "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
    "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
}

# Canonical names in preference order; used to name a colormap that was passed as
# an OpenCV code so a rendered map and its legend can never disagree.
COLORMAP_NAMES: tuple[str, ...] = ("jet", "turbo", "inferno", "viridis", "plasma")

# Default rendering: turbo is perceptually near-uniform, so a hotspot keeps its
# visual weight across the whole scale (unlike jet, which has a bright cyan band
# that reads as 'hot' when it is not).
DEFAULT_COLORMAP = "turbo"


def resolve_colormap(colormap: str | int) -> tuple[int, str]:
    """Return the OpenCV colormap code and its canonical name.

    Args:
        colormap: Colormap name ('jet', 'turbo', …) or an OpenCV code.

    Returns:
        ``(cv2_code, name)`` — unknown names fall back to ``jet``, matching the
        behaviour of the renderers below.
    """
    if isinstance(colormap, str):
        name = colormap.strip().lower()
        if name in COLORMAP_MAP:
            return COLORMAP_MAP[name], name
        return cv2.COLORMAP_JET, "jet"

    code = int(colormap)
    for name in COLORMAP_NAMES:
        if COLORMAP_MAP.get(name) == code:
            return code, name
    return code, "jet"


def refine_heatmap(
    heatmap: np.ndarray,
    blur_kernel: int = 9,
    clip_percentile: float = 100.0,
) -> np.ndarray:
    """Normalise, smooth and optionally percentile-clip a raw Grad-CAM map.

    Raw maps from the last convolutional block are blocky and carry an arbitrary
    scale, so they are peak-normalised to a full 0–1 range (which is what makes an
    activation threshold like ``0.5`` mean the same thing in every map) and lightly
    blurred to remove the blockiness.

    ``clip_percentile`` additionally maps a *lower* percentile to full intensity.
    That is off by default (``>= 100``) because it is a non-linear stretch: measured
    on this checkpoint, a 90th–99th percentile stretch inflated the share of the
    frame above 0.5 from 6–31% to 11–37%, i.e. it makes maps look busier than the
    model's evidence. Turn it on only to saturate a map that is known to be flat.

    Args:
        heatmap: Raw map, any float scale.
        blur_kernel: Gaussian kernel size; ``<= 1`` disables smoothing.
        clip_percentile: Upper percentile mapped to 1.0 (``>= 100`` disables).

    Returns:
        float32 map in [0, 1] with the same shape as the input.
    """
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D activation map, got shape {array.shape}.")

    peak = float(np.nanmax(array)) if array.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    array = np.clip(array / peak, 0.0, 1.0).astype(np.float32)

    if 0.0 < float(clip_percentile) < 100.0:
        ceiling = float(np.percentile(array, float(clip_percentile)))
        if ceiling > 0.0:
            array = np.clip(array / ceiling, 0.0, 1.0).astype(np.float32)

    kernel = int(blur_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        array = cv2.GaussianBlur(array, (kernel, kernel), 0)

    peak = float(np.nanmax(array)) if array.size else 0.0
    if peak > 0.0:
        array = array / peak
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def colorize_heatmap(
    heatmap: np.ndarray,
    colormap: str | int = DEFAULT_COLORMAP,
) -> np.ndarray:
    """Apply a colormap to a normalised activation map.

    Args:
        heatmap: float map in [0, 1] with shape (H, W).
        colormap: Colormap name or OpenCV code.

    Returns:
        uint8 RGB image (H, W, 3) — the standalone coloured heatmap a clinician
        downloads, rather than an unreadable grayscale mask.
    """
    array = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D activation map, got shape {array.shape}.")
    code, _name = resolve_colormap(colormap)
    coloured = cv2.applyColorMap((array * 255.0).astype(np.uint8), code)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def denormalize_image(tensor: Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized float tensor (C, H, W) to float32 RGB (H, W, C) in [0, 1]."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got shape {tuple(tensor.shape)}.")
    image = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def to_rgb_float(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or numpy array to an RGB float32 array in [0, 1]."""
    if isinstance(image, Image.Image):
        image_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.clip(image_arr, 0.0, 1.0)

    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        image_arr = np.stack([image_arr] * 3, axis=-1)
    elif image_arr.ndim == 3 and image_arr.shape[2] == 1:
        image_arr = np.repeat(image_arr, 3, axis=-1)
    elif image_arr.ndim == 3 and image_arr.shape[2] == 4:
        image_arr = image_arr[:, :, :3]

    if image_arr.dtype == np.uint8 or image_arr.max() > 1.0:
        image_arr = image_arr.astype(np.float32) / 255.0
    else:
        image_arr = image_arr.astype(np.float32)
    return np.clip(image_arr, 0.0, 1.0)


def overlay_cam_on_image(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str | int = "jet",
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an RGB image.

    Args:
        rgb_image: Float32 image in range [0, 1] with shape (H, W, 3).
        heatmap: Float32 CAM mask in range [0, 1] with shape (H, W).
        alpha: Heatmap blend opacity in [0, 1].
        colormap: OpenCV colormap code or string name ('jet', 'turbo', etc.).

    Returns:
        Overlaid uint8 RGB image in range [0, 255] with shape (H, W, 3).
    """
    if rgb_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    cv2_colormap, _name = resolve_colormap(colormap)

    overlay = show_cam_on_image(
        img=np.clip(rgb_image, 0.0, 1.0).astype(np.float32),
        mask=np.clip(heatmap, 0.0, 1.0).astype(np.float32),
        use_rgb=True,
        colormap=cv2_colormap,
        image_weight=1.0 - alpha,
    )
    return overlay


class TBGradCAMExplainer:
    """Production-ready Grad-CAM explainer for TB EfficientNet-B3 classifier.

    Provides explainability for chest X-ray predictions by visualizing
    salient activation regions with Grad-CAM heatmaps and clinical overlays.
    Designed for standalone analysis, batch evaluation, and FastAPI serving.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        checkpoint_path: str | Path | None = None,
        target_layer: nn.Module | str | None = None,
        device: str | torch.device | None = None,
        class_names: Sequence[str] = ("non_tb", "tb"),
        image_size: int = 300,
        colormap: str | int = DEFAULT_COLORMAP,
        alpha: float = 0.55,
        eigen_smooth: bool = False,
        blur_kernel: int = 9,
        clip_percentile: float = 100.0,
    ) -> None:
        """Initialize the explainer with a model or checkpoint.

        Args:
            model: Optional pre-loaded PyTorch model.
            checkpoint_path: Path to checkpoint .pt file if model is not provided.
            target_layer: Specific target layer or layer name string (e.g., 'backbone.features.8').
                If None, automatically resolves the last convolutional block of EfficientNet-B3.
            device: Computing device ('cuda', 'cpu', or torch.device). Auto-selected if None.
            class_names: Sequence of category names matching class indices.
            image_size: Target square image dimension for evaluation transforms.
            colormap: OpenCV colormap identifier or name ('turbo', 'jet', …).
            alpha: Heatmap opacity weight for blending (0.0 to 1.0).
            eigen_smooth: Drop the first principal component of the activations
                before weighting them, which removes the class-agnostic
                'everything lights up' component that makes chest-radiograph
                maps look uniformly hot. Costs one extra light forward pass.
            blur_kernel: Gaussian kernel used to smooth the raw map (see
                :func:`refine_heatmap`).
            clip_percentile: Upper percentile mapped to full intensity.
        """
        self.class_names = list(class_names)
        self.image_size = image_size
        self.colormap = colormap
        self.alpha = float(alpha)
        self.eigen_smooth = bool(eigen_smooth)
        self.blur_kernel = int(blur_kernel)
        self.clip_percentile = float(clip_percentile)
        self.eval_transform = build_eval_transform(image_size=self.image_size)

        # Device selection
        if device is None or str(device).lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load or bind model
        if model is not None:
            self.model = model.to(self.device)
            self.checkpoint_metadata: dict[str, Any] = {}
        elif checkpoint_path is not None:
            self.model, self.checkpoint_metadata = self._load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=self.device,
                expected_num_classes=len(class_names),
            )
        else:
            raise ValueError("Either 'model' or 'checkpoint_path' must be provided.")

        self.model.eval()

        # Resolve target layer
        self.target_layer = self._resolve_target_layer(target_layer)
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

        LOGGER.info(
            "TBGradCAMExplainer initialized on %s with target layer: %s (classes: %s)",
            self.device,
            self.target_layer.__class__.__name__,
            self.class_names,
        )

    @staticmethod
    def _load_model_from_checkpoint(
        checkpoint_path: str | Path,
        device: torch.device,
        expected_num_classes: int | None = None,
    ) -> tuple[TBImageClassifier, dict[str, Any]]:
        """Instantiate TBImageClassifier and load weights from checkpoint atomically."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        LOGGER.info("Loading model checkpoint from: %s", path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config = checkpoint.get("config", {})
        model_config = config.get("model", {}) if isinstance(config, dict) else {}

        num_classes = model_config.get("num_classes", 2)
        dropout = model_config.get("dropout", 0.3)

        # Check compatibility if expected_num_classes is provided
        if expected_num_classes is not None and num_classes != expected_num_classes:
            raise ValueError(
                f"Checkpoint num_classes ({num_classes}) does not match expected "
                f"({expected_num_classes}). This checkpoint is incompatible with the "
                f"current model configuration."
            )

        model = TBImageClassifier(
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
            num_tabular_features=0,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        metadata = {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "class_names": checkpoint.get("class_names"),
            "checkpoint_path": str(path),
            "num_classes": num_classes,
        }
        return model, metadata

    def _resolve_target_layer(self, target_layer: nn.Module | str | None) -> nn.Module:
        """Find the target convolutional layer for Grad-CAM in EfficientNet-B3."""
        if isinstance(target_layer, nn.Module):
            return target_layer

        if isinstance(target_layer, str) and target_layer.lower() != "auto":
            return self._get_submodule_by_name(self.model, target_layer)

        # Automatic detection for TBImageClassifier with EfficientNet backbone
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            # features[-1] in torchvision EfficientNet-B3 is Conv2dNormActivation (layer index 8)
            return self.model.backbone.features[-1]

        # General fallback: find last Conv2d or container with Conv2d
        last_conv: nn.Module | None = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

        raise ValueError(
            "Could not automatically resolve target convolutional layer for Grad-CAM. "
            "Please specify 'target_layer' explicitly."
        )

    @staticmethod
    def _get_submodule_by_name(model: nn.Module, layer_name: str) -> nn.Module:
        """Resolve a submodule using dot notation and array indexing like 'backbone.features.8'."""
        curr: Any = model
        tokens = layer_name.replace("[", ".").replace("]", "").split(".")
        for token in tokens:
            if not token:
                continue
            if token.isdigit():
                curr = curr[int(token)]
            elif hasattr(curr, token):
                curr = getattr(curr, token)
            else:
                try:
                    curr = curr[int(token)]
                except (TypeError, ValueError, IndexError, KeyError) as err:
                    raise ValueError(
                        f"Failed to find attribute/item {token!r} in layer path {layer_name!r}."
                    ) from err
        if not isinstance(curr, nn.Module):
            raise TypeError(
                f"Resolved attribute for {layer_name!r} is of type {type(curr)}, expected nn.Module."
            )
        return curr

    def _prepare_input_tensor(
        self,
        image_input: Tensor | np.ndarray | Image.Image | str | Path,
    ) -> tuple[Tensor, np.ndarray]:
        """Convert input to preprocessed (1, 3, H, W) tensor and (H, W, 3) float32 RGB array."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            with Image.open(path) as pil_img:
                rgb_array = to_rgb_float(pil_img)
            uint8_img = (rgb_array * 255.0).astype(np.uint8)
            transformed = self.eval_transform(image=uint8_img)
            tensor = transformed["image"].unsqueeze(0).to(self.device)
            # Resize rgb_array to image_size for consistent overlay alignment
            if rgb_array.shape[0] != self.image_size or rgb_array.shape[1] != self.image_size:
                rgb_array = cv2.resize(
                    rgb_array,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            return tensor, rgb_array

        if isinstance(image_input, Image.Image):
            rgb_array = to_rgb_float(image_input)
            uint8_img = (rgb_array * 255.0).astype(np.uint8)
            transformed = self.eval_transform(image=uint8_img)
            tensor = transformed["image"].unsqueeze(0).to(self.device)
            if rgb_array.shape[0] != self.image_size or rgb_array.shape[1] != self.image_size:
                rgb_array = cv2.resize(
                    rgb_array,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            return tensor, rgb_array

        if isinstance(image_input, np.ndarray):
            rgb_array = to_rgb_float(image_input)
            uint8_img = (rgb_array * 255.0).astype(np.uint8)
            transformed = self.eval_transform(image=uint8_img)
            tensor = transformed["image"].unsqueeze(0).to(self.device)
            if rgb_array.shape[0] != self.image_size or rgb_array.shape[1] != self.image_size:
                rgb_array = cv2.resize(
                    rgb_array,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            return tensor, rgb_array

        if isinstance(image_input, Tensor):
            tensor = image_input.detach()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(
                    f"Expected tensor with shape (3, H, W) or (1, 3, H, W), got {tuple(tensor.shape)}."
                )
            tensor = tensor.to(self.device)
            # Reconstruct RGB representation for overlay via denormalization
            rgb_array = denormalize_image(tensor[0])
            return tensor, rgb_array

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def explain(
        self,
        image: Tensor | np.ndarray | Image.Image | str | Path,
        target_category: int | None = None,
        alpha: float | None = None,
        colormap: str | int | None = None,
    ) -> dict[str, Any]:
        """Generate a Grad-CAM explanation for a single image.

        Args:
            image: Single image as Tensor, numpy array, PIL Image, or file path.
            target_category: Class index to compute CAM for (e.g. 1 for TB).
                If None, targets the model's predicted class.
            alpha: Optional blend weight override for overlay.
            colormap: Optional colormap override for overlay.

        Returns:
            Dictionary containing:
                - 'predicted_class_id': int (0 for non_tb, 1 for tb)
                - 'predicted_class_name': str
                - 'confidence': float (probability of predicted class)
                - 'probabilities': dict[str, float]
                - 'logits': list[float]
                - 'target_class_id': int
                - 'target_class_name': str
                - 'heatmap': np.ndarray (H, W) float32 in [0, 1] (refined)
                - 'heatmap_color': np.ndarray (H, W, 3) uint8 RGB heatmap
                - 'overlay': np.ndarray (H, W, 3) uint8 RGB in [0, 255]
                - 'rgb_image': np.ndarray (H, W, 3) float32 RGB in [0, 1]
                - 'colormap': str, 'alpha': float (rendering actually used)
        """
        tensor, rgb_image = self._prepare_input_tensor(image)

        # Forward pass to obtain predictions
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        predicted_class_id = int(torch.argmax(probs).item())
        predicted_class_name = (
            self.class_names[predicted_class_id]
            if predicted_class_id < len(self.class_names)
            else f"class_{predicted_class_id}"
        )
        confidence = float(probs[predicted_class_id].item())

        # Select target category
        target_id = predicted_class_id if target_category is None else int(target_category)
        target_name = (
            self.class_names[target_id]
            if target_id < len(self.class_names)
            else f"class_{target_id}"
        )
        targets = [ClassifierOutputTarget(target_id)]

        # Run Grad-CAM
        grayscale_cam = self.cam(
            input_tensor=tensor,
            targets=targets,
            eigen_smooth=self.eigen_smooth,
        )
        heatmap = refine_heatmap(
            grayscale_cam[0],
            blur_kernel=self.blur_kernel,
            clip_percentile=self.clip_percentile,
        )

        # Generate overlay + standalone coloured map
        blend_alpha = self.alpha if alpha is None else float(alpha)
        used_colormap = self.colormap if colormap is None else colormap
        colormap_code, colormap_name = resolve_colormap(used_colormap)
        overlay = overlay_cam_on_image(
            rgb_image=rgb_image,
            heatmap=heatmap,
            alpha=blend_alpha,
            colormap=colormap_code,
        )
        heatmap_color = colorize_heatmap(heatmap, colormap_code)

        probabilities_dict = {
            self.class_names[i] if i < len(self.class_names) else f"class_{i}": float(
                probs[i].item()
            )
            for i in range(len(probs))
        }

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "logits": logits.squeeze(0).cpu().tolist(),
            "target_class_id": target_id,
            "target_class_name": target_name,
            "heatmap": heatmap,
            "heatmap_color": heatmap_color,
            "overlay": overlay,
            "rgb_image": rgb_image,
            "colormap": colormap_name,
            "alpha": blend_alpha,
        }

    def explain_batch(
        self,
        images: Tensor | Sequence[np.ndarray | Image.Image | str | Path],
        target_categories: Sequence[int | None] | int | None = None,
        alpha: float | None = None,
        colormap: str | int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate Grad-CAM explanations for a batch of images.

        Args:
            images: Tensor (B, 3, H, W) or sequence of image objects.
            target_categories: Single class index, list of class indices per sample, or None.
            alpha: Blend weight override.
            colormap: Colormap override.

        Returns:
            List of explanation dictionaries, one per input sample.
        """
        if isinstance(images, Tensor):
            tensor_batch = images.to(self.device)
            batch_size = tensor_batch.shape[0]
            rgb_images = [denormalize_image(tensor_batch[i]) for i in range(batch_size)]
        else:
            prepared = [self._prepare_input_tensor(img) for img in images]
            tensor_batch = torch.cat([t for t, _ in prepared], dim=0)
            rgb_images = [arr for _, arr in prepared]
            batch_size = len(rgb_images)

        if batch_size == 0:
            return []

        with torch.no_grad():
            logits_batch = self.model(tensor_batch)
            probs_batch = torch.softmax(logits_batch, dim=1)

        predicted_classes = torch.argmax(probs_batch, dim=1).cpu().tolist()

        # Build per-sample targets
        if target_categories is None:
            resolved_targets = predicted_classes
        elif isinstance(target_categories, int):
            resolved_targets = [target_categories] * batch_size
        else:
            resolved_targets = [
                pred if target is None else int(target)
                for pred, target in zip(predicted_classes, target_categories)
            ]

        targets = [ClassifierOutputTarget(target_id) for target_id in resolved_targets]
        grayscale_cams = self.cam(
            input_tensor=tensor_batch,
            targets=targets,
            eigen_smooth=self.eigen_smooth,
        )

        blend_alpha = self.alpha if alpha is None else float(alpha)
        used_colormap = self.colormap if colormap is None else colormap
        colormap_code, colormap_name = resolve_colormap(used_colormap)

        results: list[dict[str, Any]] = []
        for i in range(batch_size):
            pred_id = int(predicted_classes[i])
            pred_name = (
                self.class_names[pred_id]
                if pred_id < len(self.class_names)
                else f"class_{pred_id}"
            )
            prob_dist = probs_batch[i]
            conf = float(prob_dist[pred_id].item())
            target_id = int(resolved_targets[i])
            target_name = (
                self.class_names[target_id]
                if target_id < len(self.class_names)
                else f"class_{target_id}"
            )
            heatmap = refine_heatmap(
                grayscale_cams[i],
                blur_kernel=self.blur_kernel,
                clip_percentile=self.clip_percentile,
            )
            overlay = overlay_cam_on_image(
                rgb_image=rgb_images[i],
                heatmap=heatmap,
                alpha=blend_alpha,
                colormap=colormap_code,
            )
            heatmap_color = colorize_heatmap(heatmap, colormap_code)
            prob_dict = {
                self.class_names[j] if j < len(self.class_names) else f"class_{j}": float(
                    prob_dist[j].item()
                )
                for j in range(len(prob_dist))
            }
            results.append(
                {
                    "predicted_class_id": pred_id,
                    "predicted_class_name": pred_name,
                    "confidence": conf,
                    "probabilities": prob_dict,
                    "logits": logits_batch[i].cpu().tolist(),
                    "target_class_id": target_id,
                    "target_class_name": target_name,
                    "heatmap": heatmap,
                    "heatmap_color": heatmap_color,
                    "overlay": overlay,
                    "rgb_image": rgb_images[i],
                    "colormap": colormap_name,
                    "alpha": blend_alpha,
                }
            )

        return results

    def create_figure(
        self,
        explanation: dict[str, Any],
        ground_truth: int | str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (14.0, 4.5),
    ) -> plt.Figure:
        """Create a 3-panel clinician figure (Original, Heatmap, Overlay).

        Panel 2 is the coloured map rendered with the same colormap as the overlay,
        with a matching colour bar, and the focus summary is footnoted when the
        explanation carries one — so the figure answers *where* as well as *what*.
        """
        rgb_image = explanation["rgb_image"]
        heatmap = explanation["heatmap"]
        overlay = explanation["overlay"]
        pred_name = explanation["predicted_class_name"]
        confidence = explanation["confidence"]
        target_name = explanation["target_class_name"]

        # Keep panel 2 and 3 on the colormap the map was rendered with.
        colormap_name = str(explanation.get("colormap") or DEFAULT_COLORMAP).lower()
        if colormap_name not in COLORMAP_MAP:
            colormap_name = DEFAULT_COLORMAP
        heatmap_color = explanation.get("heatmap_color")

        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        # Panel 1: Original X-Ray
        axes[0].imshow(rgb_image)
        axes[0].set_title("Input Chest X-Ray", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Heatmap (coloured, same map as the overlay)
        if heatmap_color is not None:
            axes[1].imshow(heatmap_color)
        else:  # pragma: no cover - older explanations without a coloured map
            axes[1].imshow(heatmap, cmap=colormap_name, vmin=0.0, vmax=1.0)
        axes[1].set_title(
            f"Grad-CAM Heatmap ({target_name})", fontsize=11, fontweight="bold"
        )
        axes[1].axis("off")
        mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=colormap_name)
        cbar = fig.colorbar(mappable, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Relative activation", fontsize=9)

        # Panel 3: Overlaid Clinical View
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (X-Ray + Activation)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Clinical status computation
        status_text = ""
        if ground_truth is not None:
            gt_name = (
                self.class_names[ground_truth]
                if isinstance(ground_truth, int) and ground_truth < len(self.class_names)
                else str(ground_truth)
            )
            is_correct = pred_name.lower() == gt_name.lower()
            if is_correct:
                clinical_outcome = "True Positive (TP)" if pred_name.lower() == "tb" else "True Negative (TN)"
                outcome_color = "#15803d"  # green
            else:
                clinical_outcome = "False Positive (FP)" if pred_name.lower() == "tb" else "False Negative (FN)"
                outcome_color = "#b91c1c"  # red
            status_text = f" | Ground Truth: {gt_name.upper()} | Outcome: {clinical_outcome}"

        overall_title = title or (
            f"Prediction: {pred_name.upper()} ({confidence:.1%} confidence)"
            f"{status_text}"
        )
        fig.suptitle(overall_title, fontsize=12, fontweight="bold")

        # Footnote the computed focus (where) so the figure is self-contained.
        focus = explanation.get("focus") or {}
        focus_parts = [part for part in (focus.get("summary"),) if part]
        if focus.get("available") and focus.get("convention"):
            focus_parts.append(str(focus["convention"]))
        if focus_parts:
            fig.text(
                0.5,
                -0.015,
                " ".join(str(part) for part in focus_parts),
                ha="center",
                va="top",
                fontsize=8.5,
                color="#475569",
                wrap=True,
            )
        return fig

    def save_explanation(
        self,
        explanation: dict[str, Any],
        output_path: str | Path,
        ground_truth: int | str | None = None,
        title: str | None = None,
        save_figure: bool = True,
        save_overlay_only: bool = True,
    ) -> dict[str, Path]:
        """Save explanation artifacts (overlay image and/or full 3-panel figure) to disk.

        Args:
            explanation: Output dict from explain().
            output_path: Base file path or prefix (e.g., 'artifacts/explanations/sample_1.png').
            ground_truth: Optional true class label.
            title: Optional custom figure title.
            save_figure: Whether to render and save the 3-panel figure.
            save_overlay_only: Whether to save the standalone overlay PNG.

        Returns:
            Dictionary with saved paths: {'overlay_path': Path, 'figure_path': Path}.
        """
        base_path = Path(output_path).expanduser().resolve()
        base_path.parent.mkdir(parents=True, exist_ok=True)
        stem = base_path.stem
        parent = base_path.parent

        saved_files: dict[str, Path] = {}

        if save_overlay_only:
            overlay_file = parent / f"{stem}_overlay.png"
            Image.fromarray(explanation["overlay"]).save(overlay_file)
            saved_files["overlay_path"] = overlay_file

        if save_figure:
            figure_file = parent / f"{stem}_figure.png"
            fig = self.create_figure(
                explanation=explanation,
                ground_truth=ground_truth,
                title=title,
            )
            fig.savefig(figure_file, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_files["figure_path"] = figure_file

        return saved_files
