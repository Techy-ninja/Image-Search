from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import SUPPORTED_IMAGE_EXTENSIONS
from models import EmbeddingError, ModelLoadError, embed_image


INDEX_METADATA_FILENAME = "index.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
HASH_CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


IndexSummary = dict[str, Any]


def build_index(image_root: str, index_dir: str) -> IndexSummary:

    root = _validate_image_root(image_root)
    output_dir = Path(index_dir).expanduser().resolve()

    logger.info("Building fresh image index")
    logger.info("Image root: %s", root)
    logger.info("Index directory: %s", output_dir)

    image_paths = _scan_image_paths(root)
    logger.info("Found %d supported image(s)", len(image_paths))

    metadata: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    failed: list[dict[str, str]] = []

    for position, image_path in enumerate(image_paths, start=1):
        logger.info("[%d/%d] Embedding %s", position, len(image_paths), image_path)

        try:
            file_metadata = _make_file_metadata(image_path)
            embedding = embed_image(str(image_path))
        except (OSError, EmbeddingError, ModelLoadError) as exc:
            logger.warning("Failed to index %s: %s", image_path, exc)
            failed.append({"path": str(image_path), "error": str(exc)})
            if isinstance(exc, ModelLoadError):
                raise
            continue

        file_metadata["id"] = len(metadata)
        metadata.append(file_metadata)
        embeddings.append(embedding)

    embeddings_array = _stack_embeddings(embeddings)
    _save_index(output_dir, metadata, embeddings_array)

    summary: IndexSummary = {
        "scanned": len(image_paths),
        "added": len(metadata),
        "updated": 0,
        "skipped": 0,
        "removed": 0,
        "failed": len(failed),
        "failures": failed,
    }
    logger.info("Index build complete: %s", _format_summary(summary))
    return summary


def update_index(image_root: str, index_dir: str) -> IndexSummary:

    root = _validate_image_root(image_root)
    output_dir = Path(index_dir).expanduser().resolve()

    logger.info("Updating image index")
    logger.info("Image root: %s", root)
    logger.info("Index directory: %s", output_dir)

    existing_metadata, existing_embeddings = _load_index(output_dir)
    if existing_metadata is None or existing_embeddings is None:
        logger.info("No existing index found; building a fresh index")
        return build_index(str(root), str(output_dir))

    image_paths = _scan_image_paths(root)
    current_paths = {str(path) for path in image_paths}
    logger.info("Found %d supported image(s)", len(image_paths))

    existing_by_path = {entry["path"]: entry for entry in existing_metadata if "path" in entry}
    removed_count = sum(1 for entry in existing_metadata if entry.get("path") not in current_paths)

    new_metadata: list[dict[str, Any]] = []
    new_embeddings: list[np.ndarray] = []
    failed: list[dict[str, str]] = []
    added_count = 0
    updated_count = 0
    skipped_count = 0

    for position, image_path in enumerate(image_paths, start=1):
        image_path_str = str(image_path)
        logger.info("[%d/%d] Checking %s", position, len(image_paths), image_path)

        try:
            file_metadata = _make_file_metadata(image_path)
        except OSError as exc:
            logger.warning("Failed to read metadata for %s: %s", image_path, exc)
            failed.append({"path": image_path_str, "error": str(exc)})
            continue

        existing_entry = existing_by_path.get(image_path_str)
        existing_embedding = _embedding_for_entry(existing_entry, existing_embeddings)

        if (
            existing_entry is not None
            and existing_embedding is not None
            and existing_entry.get("sha256") == file_metadata["sha256"]
        ):
            logger.info("Skipping unchanged image: %s", image_path)
            file_metadata["id"] = len(new_metadata)
            new_metadata.append(file_metadata)
            new_embeddings.append(existing_embedding.astype(np.float32, copy=False))
            skipped_count += 1
            continue

        action = "Updating changed image" if existing_entry is not None else "Embedding new image"
        logger.info("%s: %s", action, image_path)

        try:
            embedding = embed_image(image_path_str)
        except (EmbeddingError, ModelLoadError) as exc:
            logger.warning("Failed to index %s: %s", image_path, exc)
            failed.append({"path": image_path_str, "error": str(exc)})
            if isinstance(exc, ModelLoadError):
                raise
            continue

        file_metadata["id"] = len(new_metadata)
        new_metadata.append(file_metadata)
        new_embeddings.append(embedding)

        if existing_entry is None:
            added_count += 1
        else:
            updated_count += 1

    embeddings_array = _stack_embeddings(new_embeddings)
    _save_index(output_dir, new_metadata, embeddings_array)

    summary: IndexSummary = {
        "scanned": len(image_paths),
        "added": added_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "removed": removed_count,
        "failed": len(failed),
        "failures": failed,
    }
    logger.info("Index update complete: %s", _format_summary(summary))
    return summary


def _validate_image_root(image_root: str) -> Path:
    """Validate and resolve the image root directory."""
    root = Path(image_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {root}")

    return root


def _scan_image_paths(root: Path) -> list[Path]:
    """Recursively collect supported image files under `root`."""
    image_paths: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(path.resolve())

    return sorted(image_paths, key=lambda path: str(path).lower())


def _make_file_metadata(image_path: Path) -> dict[str, Any]:
    """Create JSON-serializable metadata for an image file."""
    stat = image_path.stat()
    return {
        "id": -1,
        "path": str(image_path),
        "filename": image_path.name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": _hash_file(image_path),
    }


def _hash_file(path: Path) -> str:
    """Compute a SHA-256 hash of a file's contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _metadata_path(index_dir: Path) -> Path:
    """Return the metadata JSON path for an index directory."""
    return index_dir / INDEX_METADATA_FILENAME


def _embeddings_path(index_dir: Path) -> Path:
    """Return the embeddings NumPy path for an index directory."""
    return index_dir / EMBEDDINGS_FILENAME


def _load_index(index_dir: Path) -> tuple[list[dict[str, Any]] | None, np.ndarray | None]:
    """Load metadata and embeddings if both index files exist."""
    metadata_file = _metadata_path(index_dir)
    embeddings_file = _embeddings_path(index_dir)

    if not metadata_file.exists() or not embeddings_file.exists():
        return None, None

    try:
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        embeddings = np.load(embeddings_file).astype(np.float32, copy=False)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load existing index from %s: %s", index_dir, exc)
        return None, None

    if not isinstance(metadata, list):
        logger.warning("Existing metadata file is invalid; rebuilding index")
        return None, None

    if embeddings.ndim != 2:
        logger.warning("Existing embeddings file has invalid shape %s; rebuilding index", embeddings.shape)
        return None, None

    if len(metadata) != embeddings.shape[0]:
        logger.warning(
            "Existing index is inconsistent: %d metadata entries but %d embedding rows; rebuilding index",
            len(metadata),
            embeddings.shape[0],
        )
        return None, None

    logger.info("Loaded existing index with %d image(s)", len(metadata))
    return metadata, embeddings


def _save_index(index_dir: Path, metadata: list[dict[str, Any]], embeddings: np.ndarray) -> None:
    """Persist metadata and embeddings to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = _metadata_path(index_dir)
    embeddings_file = _embeddings_path(index_dir)

    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    np.save(embeddings_file, embeddings.astype(np.float32, copy=False))

    logger.info("Saved metadata: %s", metadata_file)
    logger.info("Saved embeddings: %s", embeddings_file)


def _stack_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Stack embedding vectors into a 2D float32 matrix."""
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack([np.asarray(embedding, dtype=np.float32).reshape(1, -1) for embedding in embeddings])


def _embedding_for_entry(entry: dict[str, Any] | None, embeddings: np.ndarray) -> np.ndarray | None:
    """Return the embedding row corresponding to a metadata entry, if valid."""
    if entry is None:
        return None

    try:
        index = int(entry["id"])
    except (KeyError, TypeError, ValueError):
        return None

    if index < 0 or index >= embeddings.shape[0]:
        return None

    return embeddings[index]


def _format_summary(summary: IndexSummary) -> str:
    """Format summary counts for human-readable logs."""
    return (
        f"scanned={summary['scanned']}, added={summary['added']}, "
        f"updated={summary['updated']}, skipped={summary['skipped']}, "
        f"removed={summary['removed']}, failed={summary['failed']}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or update a local semantic image index.")
    parser.add_argument("image_root", help="Folder to scan recursively for images.")
    parser.add_argument("index_dir", help="Directory where index.json and embeddings.npy are stored.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch instead of updating incrementally.",
    )
    args = parser.parse_args()

    if args.rebuild:
        build_index(args.image_root, args.index_dir)
    else:
        update_index(args.image_root, args.index_dir)
