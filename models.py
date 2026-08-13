from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as transformers_logging

from config import DEVICE, MODEL_NAME


transformers_logging.set_verbosity_error()


_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None
_torch_device: Optional[torch.device] = None


class ModelLoadError(RuntimeError):
    """Raised when the CLIP model or processor cannot be loaded."""


class EmbeddingError(RuntimeError):
    """Raised when an image or text embedding cannot be generated."""


def _resolve_device() -> torch.device:
    """Return the configured torch device, validating availability."""
    requested_device = DEVICE.strip().lower()

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError(
            "CUDA was requested via IMAGE_SEARCH_DEVICE, but CUDA is not available. "
            "Use IMAGE_SEARCH_DEVICE=cpu or install a CUDA-enabled PyTorch build."
        )

    try:
        return torch.device(requested_device)
    except RuntimeError as exc:
        raise ModelLoadError(f"Invalid torch device configured: {DEVICE!r}") from exc


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a 1D float32 vector normalized to unit L2 length."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0.0:
        raise EmbeddingError("Model produced an invalid zero or non-finite embedding.")

    return vector / norm


def _move_inputs_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a Transformers input dictionary onto the selected torch device."""
    return {key: value.to(device) for key, value in inputs.items()}


def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor, _torch_device

    if _model is not None and _processor is not None:
        return _model, _processor

    _torch_device = _resolve_device()

    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME)
        model.to(_torch_device)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - convert library-specific errors into app-level error.
        raise ModelLoadError(
            f"Could not load CLIP model {MODEL_NAME!r}. "
            "Check that dependencies are installed and that the model has been "
            "downloaded at least once if running offline."
        ) from exc

    _model = model
    _processor = processor
    return _model, _processor


def embed_image(path: str) -> np.ndarray:
    image_path = Path(path).expanduser()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise EmbeddingError(f"Image path is not a file: {image_path}")

    model, processor = load_model()
    device = _torch_device or _resolve_device()

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
    except UnidentifiedImageError as exc:
        raise EmbeddingError(f"Unsupported or corrupt image file: {image_path}") from exc
    except OSError as exc:
        raise EmbeddingError(f"Could not read image file: {image_path}") from exc

    try:
        inputs = _move_inputs_to_device(inputs, device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vector = features[0].detach().cpu().numpy()
    except Exception as exc:  # noqa: BLE001 - keep embedding failures understandable to callers.
        raise EmbeddingError(f"Could not embed image: {image_path}") from exc

    return _normalize_vector(vector)


def embed_text(text: str) -> np.ndarray:
    query = text.strip()
    if not query:
        raise ValueError("Text query cannot be empty.")

    model, processor = load_model()
    device = _torch_device or _resolve_device()

    try:
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        inputs = _move_inputs_to_device(inputs, device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        vector = features[0].detach().cpu().numpy()
    except Exception as exc:  # noqa: BLE001 - keep embedding failures understandable to callers.
        raise EmbeddingError(f"Could not embed text query: {query!r}") from exc

    return _normalize_vector(vector)
