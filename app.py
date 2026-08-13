from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from config import (
    DATA_DIR,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    IMAGE_FOLDERS,
    MAX_TOP_K,
)
from indexer import update_index
from models import EmbeddingError, ModelLoadError, load_model
from search import load_index, search_text


IMAGE_ROOT: Path | None = IMAGE_FOLDERS[0] if IMAGE_FOLDERS else None
INDEX_DIR: Path = DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "local-semantic-image-search-dev-key"


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the main search page."""
    return render_template(
        "index.html",
        query="",
        results=[],
        default_top_k=DEFAULT_TOP_K,
        max_top_k=MAX_TOP_K,
        min_score=DEFAULT_MIN_SCORE,
        image_root=str(IMAGE_ROOT) if IMAGE_ROOT else "",
        index_dir=str(INDEX_DIR),
        index_ready=_index_exists(),
    )


@app.route("/search", methods=["POST"])
def search() -> str:
    """Accept a text query, run semantic search, and render results."""
    query = request.form.get("query", "").strip()
    top_k = _parse_top_k(request.form.get("top_k"))
    min_score = _parse_min_score(request.form.get("min_score"))

    logger.info("Received search request: query=%r, top_k=%d, min_score=%.3f", query, top_k, min_score)

    if not query:
        flash("Please enter a search query.", "error")
        return render_template(
            "index.html",
            query=query,
            results=[],
            default_top_k=top_k,
            max_top_k=MAX_TOP_K,
            min_score=min_score,
            image_root=str(IMAGE_ROOT) if IMAGE_ROOT else "",
            index_dir=str(INDEX_DIR),
            index_ready=_index_exists(),
        )

    try:
        results = search_text(query, top_k=top_k, min_score=min_score)
    except ValueError as exc:
        logger.warning("Invalid search request: %s", exc)
        flash(str(exc), "error")
        results = []
    except ModelLoadError as exc:
        logger.exception("CLIP model could not be loaded during search")
        flash(str(exc), "error")
        results = []
    except EmbeddingError as exc:
        logger.exception("Could not embed search query")
        flash(str(exc), "error")
        results = []

    view_results = [_result_for_template(result) for result in results]

    if query and not view_results:
        flash(
            f"No matches met the minimum similarity threshold of {min_score:.3f}. "
            "Try a lower threshold, another query, or re-index your folder.",
            "info",
        )

    return render_template(
        "index.html",
        query=query,
        results=view_results,
        default_top_k=top_k,
        max_top_k=MAX_TOP_K,
        min_score=min_score,
        image_root=str(IMAGE_ROOT) if IMAGE_ROOT else "",
        index_dir=str(INDEX_DIR),
        index_ready=_index_exists(),
    )


@app.route("/reindex", methods=["POST"])
def reindex():
    """Rescan the configured image folder and update the on-disk index."""
    if IMAGE_ROOT is None:
        message = (
            "No image folder is configured. Set IMAGE_SEARCH_FOLDERS before starting the app, "
            "for example: IMAGE_SEARCH_FOLDERS='/home/me/Pictures' python app.py"
        )
        logger.warning(message)
        flash(message, "error")
        return redirect(url_for("index"))

    logger.info("Starting reindex for image root: %s", IMAGE_ROOT)

    try:
        summary = update_index(str(IMAGE_ROOT), str(INDEX_DIR))
        load_index(str(INDEX_DIR))
    except FileNotFoundError as exc:
        logger.warning("Image root does not exist: %s", exc)
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except NotADirectoryError as exc:
        logger.warning("Image root is not a directory: %s", exc)
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except ModelLoadError as exc:
        logger.exception("CLIP model could not be loaded during indexing")
        flash(str(exc), "error")
        return redirect(url_for("index"))

    message = _format_index_summary(summary)
    logger.info(message)
    flash(message, "success")
    return redirect(url_for("index"))


@app.route("/image/<int:image_id>", methods=["GET"])
def image(image_id: int):
    """Serve an indexed image by id without exposing arbitrary file access."""
    _, metadata = load_index(str(INDEX_DIR))

    if image_id < 0 or image_id >= len(metadata):
        logger.warning("Image id not found: %d", image_id)
        return "Image not found", 404

    image_path = Path(metadata[image_id].get("path", ""))
    if not image_path.exists() or not image_path.is_file():
        logger.warning("Indexed image file is missing: %s", image_path)
        return "Image file is missing", 404

    return send_file(image_path)


def startup() -> None:
    """Load the CLIP model and existing index when the app starts."""
    logger.info("Starting local semantic image search app")
    logger.info("Configured image root: %s", IMAGE_ROOT if IMAGE_ROOT else "not configured")
    logger.info("Index directory: %s", INDEX_DIR)

    try:
        logger.info("Loading CLIP model...")
        load_model()
        logger.info("CLIP model loaded successfully")
    except ModelLoadError as exc:
        logger.error("CLIP model failed to load on startup: %s", exc)

    load_index(str(INDEX_DIR))


def _parse_top_k(raw_value: str | None) -> int:
    """Parse and clamp a top-k form value."""
    try:
        top_k = int(raw_value or DEFAULT_TOP_K)
    except ValueError:
        top_k = DEFAULT_TOP_K

    return max(1, min(top_k, MAX_TOP_K))


def _parse_min_score(raw_value: str | None) -> float:
    """Parse and clamp the minimum similarity threshold form value."""
    try:
        min_score = float(raw_value if raw_value not in {None, ""} else DEFAULT_MIN_SCORE)
    except ValueError:
        min_score = DEFAULT_MIN_SCORE

    return max(-1.0, min(min_score, 1.0))


def _index_exists() -> bool:
    """Return True when both expected index files exist."""
    return (INDEX_DIR / "index.json").exists() and (INDEX_DIR / "embeddings.npy").exists()


def _result_for_template(result: dict[str, Any]) -> dict[str, Any]:
    """Add web-specific fields to a raw search result."""
    image_id = int(result.get("id", -1))
    path = result.get("path", "")

    return {
        "id": image_id,
        "path": path,
        "filename": result.get("filename") or Path(path).name,
        "score": result.get("score", 0.0),
        "image_url": url_for("image", image_id=image_id),
    }


def _format_index_summary(summary: dict[str, Any]) -> str:
    """Format an indexing summary for display in the UI."""
    return (
        "Index updated: "
        f"scanned={summary.get('scanned', 0)}, "
        f"added={summary.get('added', 0)}, "
        f"updated={summary.get('updated', 0)}, "
        f"skipped={summary.get('skipped', 0)}, "
        f"removed={summary.get('removed', 0)}, "
        f"failed={summary.get('failed', 0)}"
    )


startup()


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
