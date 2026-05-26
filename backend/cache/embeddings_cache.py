"""
Disk-backed cache for embedding vectors produced during image processing.

Each named "key" (e.g. "team_fused", "person_fused", "reid", "face") is stored
as a separate NumPy .npz file inside the cache directory so that a processing
run can be resumed without re-extracting embeddings for already-processed images.

Layout on disk
--------------
<cache_dir>/
    team_fused.npz    →  arrays: embeddings (N, D), paths (N,), det_indices (N,)
    person_fused.npz
    reid.npz
    face.npz
    team_clip.npz
    …
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """
    In-memory cache for named embedding stores with optional persistence.

    Usage::

        cache = EmbeddingsCache("/path/to/cache/dir")
        cache.load()                                  # restore previous run

        cache.add("team_fused", "img.jpg", 0, vector)
        matrix = cache.get_matrix("team_fused")       # np.ndarray (N, D) or None
        paths  = cache.get_paths("team_fused")        # List[str]
        dets   = cache.get_det_indices("team_fused")  # List[int]

        cache.save()                                  # flush to disk
    """

    def __init__(self, cache_dir: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # key → (embeddings_list, paths_list, det_indices_list)
        self._data: Dict[str, Tuple[List[np.ndarray], List[str], List[int]]] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all .npz files from the cache directory into memory."""
        for npz_path in sorted(self._dir.glob("*.npz")):
            key = npz_path.stem
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    embeddings = list(data["embeddings"])
                    paths = list(data["paths"])
                    det_indices = list(data["det_indices"].tolist())
                self._data[key] = (embeddings, paths, det_indices)
                logger.debug(
                    "Loaded embeddings cache '%s': %d entries", key, len(embeddings)
                )
            except Exception as exc:
                logger.warning("Failed to load embeddings cache '%s': %s", key, exc)

    def save(self) -> None:
        """Flush all in-memory embeddings to .npz files on disk."""
        for key, (embeddings, paths, det_indices) in self._data.items():
            if not embeddings:
                continue
            npz_path = self._dir / f"{key}.npz"
            try:
                np.savez_compressed(
                    npz_path,
                    embeddings=np.array(embeddings, dtype=np.float32),
                    paths=np.array(paths, dtype=object),
                    det_indices=np.array(det_indices, dtype=np.int32),
                )
                logger.debug(
                    "Saved embeddings cache '%s': %d entries", key, len(embeddings)
                )
            except Exception as exc:
                logger.warning("Failed to save embeddings cache '%s': %s", key, exc)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        key: str,
        image_path: str,
        det_idx: int,
        embedding: np.ndarray,
    ) -> None:
        """Append one embedding vector to the named store."""
        if key not in self._data:
            self._data[key] = ([], [], [])
        embeddings, paths, det_indices = self._data[key]
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        paths.append(image_path)
        det_indices.append(int(det_idx))

    def clear(self, key: Optional[str] = None) -> None:
        """Remove entries from the cache.  If *key* is None, clears everything."""
        if key is None:
            self._data.clear()
        else:
            self._data.pop(key, None)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_matrix(self, key: str) -> Optional[np.ndarray]:
        """Return a (N, D) float32 array of all embeddings for *key*, or None."""
        entry = self._data.get(key)
        if not entry or not entry[0]:
            return None
        return np.array(entry[0], dtype=np.float32)

    def get_paths(self, key: str) -> List[str]:
        """Return the list of image paths associated with *key*."""
        entry = self._data.get(key)
        return list(entry[1]) if entry else []

    def get_det_indices(self, key: str) -> List[int]:
        """Return the list of detection indices associated with *key*."""
        entry = self._data.get(key)
        return list(entry[2]) if entry else []

    def __len__(self) -> int:
        return sum(len(e[0]) for e in self._data.values())

    def __repr__(self) -> str:  # pragma: no cover
        keys = {k: len(v[0]) for k, v in self._data.items()}
        return f"EmbeddingsCache(dir={self._dir!r}, keys={keys})"
