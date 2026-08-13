from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import DATA_DIR
from models import EmbeddingError, ModelLoadError, embed_text


INDEX_METADATA_FILENAME = "index.json"
EMBEDDINGS_FILENAME = "embeddings.npy"

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


_metadata: list[dict[str, Any]] = []
_embeddings: np.ndarray | None = None
_loaded_index_dir: Path | None = None


def load_index(index_dir: str) -> tuple[np.ndarray, list[dict[str, Any]]]:

    global _metadata, _embeddings, _loaded_index_dir

    resolved_index_dir = Path(index_dir).expanduser().resolve()
    metadata_path = resolved_index_dir / INDEX_METADATA_FILENAME
    embeddings_path = resolved_index_dir / EMBEDDINGS_FILENAME

    if not metadata_path.exists() or not embeddings_path.exists():
        logger.warning("Search index is missing in %s. Run indexing first.", resolved_index_dir)
        _metadata = []
        _embeddings = _empty_embeddings()
        _loaded_index_dir = resolved_index_dir
        return _embeddings, _metadata

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            loaded_metadata = json.load(file)
        loaded_embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load search index from %s: %s", resolved_index_dir, exc)
        _metadata = []
        _embeddings = _empty_embeddings()
        _loaded_index_dir = resolved_index_dir
        return _embeddings, _metadata

    if not isinstance(loaded_metadata, list):
        logger.warning("Search index metadata is invalid: expected a JSON list.")
        _metadata = []
        _embeddings = _empty_embeddings()
        _loaded_index_dir = resolved_index_dir
        return _embeddings, _metadata

    if loaded_embeddings.ndim != 2:
        logger.warning("Search embeddings are invalid: expected a 2D array, got shape %s.", loaded_embeddings.shape)
        _metadata = []
        _embeddings = _empty_embeddings()
        _loaded_index_dir = resolved_index_dir
        return _embeddings, _metadata

    if len(loaded_metadata) != loaded_embeddings.shape[0]:
        logger.warning(
            "Search index is inconsistent: %d metadata entries but %d embedding rows.",
            len(loaded_metadata),
            loaded_embeddings.shape[0],
        )
        _metadata = []
        _embeddings = _empty_embeddings()
        _loaded_index_dir = resolved_index_dir
        return _embeddings, _metadata

    _metadata = loaded_metadata
    _embeddings = loaded_embeddings
    _loaded_index_dir = resolved_index_dir

    if len(_metadata) == 0:
        logger.warning("Search index in %s is empty.", resolved_index_dir)
    else:
        logger.info("Loaded search index with %d image(s) from %s", len(_metadata), resolved_index_dir)

    return _embeddings, _metadata


def search_text(query: str, top_k: int = 20, min_score: float = 0.0) -> list[dict[str, Any]]:

    global _embeddings

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not np.isfinite(min_score):
        raise ValueError("min_score must be a finite number.")

    if _embeddings is None:
        load_index(str(DATA_DIR))

    if _embeddings is None or _embeddings.size == 0 or not _metadata:
        logger.warning("Search index is empty or missing. Run indexing before searching.")
        return []

    query_embedding = embed_text(query).astype(np.float32, copy=False)

    if query_embedding.ndim != 1:
        query_embedding = query_embedding.reshape(-1)

    if _embeddings.shape[1] != query_embedding.shape[0]:
        logger.warning(
            "Query embedding dimension %d does not match index dimension %d.",
            query_embedding.shape[0],
            _embeddings.shape[1],
        )
        return []

    scores = _embeddings @ query_embedding
    valid_indices = np.flatnonzero(scores >= min_score)
    result_count = min(top_k, len(valid_indices))

    if result_count == 0:
        logger.info("No results met the minimum similarity threshold %.3f.", min_score)
        return []

    valid_scores = scores[valid_indices]
    candidate_positions = np.argpartition(-valid_scores, result_count - 1)[:result_count]
    candidate_indices = valid_indices[candidate_positions]
    sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

    matches: list[dict[str, Any]] = []
    for index in sorted_indices:
        metadata = _metadata[int(index)]
        matches.append(
            {
                "id": metadata.get("id", int(index)),
                "path": metadata.get("path", ""),
                "filename": metadata.get("filename", Path(metadata.get("path", "")).name),
                "score": float(scores[index]),
                "metadata": metadata,
            }
        )

    logger.info(
        "Search for %r returned %d match(es) with min_score=%.3f",
        query,
        len(matches),
        min_score,
    )
    return matches


def _empty_embeddings() -> np.ndarray:
    """Return a consistent empty embedding matrix."""
    return np.empty((0, 0), dtype=np.float32)
