from __future__ import annotations

import os
from pathlib import Path


# Project paths -------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_METADATA_PATH = DATA_DIR / "index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"


# Image scanning ------------------------------------------------------------

# Configure one or more folders with a colon-separated environment variable:
#
#   IMAGE_SEARCH_FOLDERS="/path/to/images:/another/path" python app.py
#
# If not set, the app starts with no folders configured and the UI/indexer will
# show a helpful message.
def _parse_image_folders() -> list[Path]:
    raw_value = os.getenv("IMAGE_SEARCH_FOLDERS", "").strip()
    if not raw_value:
        return []

    return [Path(folder).expanduser().resolve() for folder in raw_value.split(os.pathsep) if folder.strip()]


IMAGE_FOLDERS = _parse_image_folders()

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


# Model settings ------------------------------------------------------------

# Small, widely used CLIP checkpoint from Hugging Face. The first run may need
# internet access to download it; after that, Transformers can load it from the
# local cache.
MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# CPU by default, as requested. Later we can allow "cuda" here if available.
DEVICE = os.getenv("IMAGE_SEARCH_DEVICE", "cpu")


# Search/UI settings --------------------------------------------------------

DEFAULT_TOP_K = int(os.getenv("IMAGE_SEARCH_DEFAULT_TOP_K", "20"))
MAX_TOP_K = int(os.getenv("IMAGE_SEARCH_MAX_TOP_K", "100"))

# Minimum cosine similarity score required for a result to be shown.
# CLIP scores vary by dataset/model, but 0.20-0.30 is a useful tuning range.
DEFAULT_MIN_SCORE = float(os.getenv("IMAGE_SEARCH_DEFAULT_MIN_SCORE", "0.22"))


# Flask settings ------------------------------------------------------------

FLASK_HOST = os.getenv("IMAGE_SEARCH_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("IMAGE_SEARCH_PORT", "5000"))
FLASK_DEBUG = os.getenv("IMAGE_SEARCH_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
