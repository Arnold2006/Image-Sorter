"""
Duplicate detection: exact, perceptual, and near-duplicate via embeddings.
Handles burst-shot grouping using EXIF timestamps.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from backend.utils.config import AppConfig, get_config
from backend.utils.image_utils import file_hash, hamming_distance, perceptual_hash

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Multi-stage duplicate detection:

    Stage 1 – Exact hash: MD5 file hash comparison.
    Stage 2 – Perceptual hash: average-hash Hamming distance.
    Stage 3 – Embedding similarity: cosine distance on CLIP / Re-ID vectors.
    Stage 4 – Temporal burst: images taken within N seconds of each other.

    Results are stored as a mapping from each image path to its "keep"
    representative (or itself if it is the representative).
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self._md5_map: Dict[str, str] = {}            # md5 → first path seen
        self._phash_map: Dict[str, np.ndarray] = {}   # path → phash vector
        self._duplicates: Dict[str, str] = {}          # path → canonical path
        self._bursts: Dict[str, List[str]] = defaultdict(list)  # burst_key → paths

    # ---------- stage 1: exact ----------

    def check_exact(self, path: str) -> Optional[str]:
        """
        Return the canonical path this file is a duplicate of,
        or None if it is new.
        """
        if not self.cfg.duplicate.exact_hash:
            return None
        try:
            md5 = file_hash(path, "md5")
            if md5 in self._md5_map:
                dup_of = self._md5_map[md5]
                logger.debug("Exact duplicate: %s == %s", path, dup_of)
                return dup_of
            self._md5_map[md5] = path
        except Exception as exc:
            logger.warning("Hash failed for %s: %s", path, exc)
        return None

    # ---------- stage 2: perceptual ----------

    def register_phash(self, path: str, img_rgb: np.ndarray) -> None:
        """Register the perceptual hash for an image (call after loading)."""
        if self.cfg.duplicate.perceptual_hash:
            self._phash_map[path] = perceptual_hash(img_rgb)

    def check_perceptual(self, path: str) -> Optional[str]:
        """
        Compare *path*'s perceptual hash against all registered hashes.
        Returns the closest match path if within threshold, else None.
        """
        if not self.cfg.duplicate.perceptual_hash:
            return None
        if path not in self._phash_map:
            return None
        ph = self._phash_map[path]
        threshold = self.cfg.duplicate.perceptual_threshold
        best_path: Optional[str] = None
        best_dist = threshold + 1

        for other_path, other_ph in self._phash_map.items():
            if other_path == path:
                continue
            dist = hamming_distance(ph, other_ph)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best_path = other_path

        if best_path:
            logger.debug("Perceptual duplicate: %s ≈ %s (dist=%d)", path, best_path, best_dist)
        return best_path

    # ---------- stage 3: embedding similarity ----------

    def check_embedding_similarity(
        self,
        path: str,
        embedding: np.ndarray,
        all_paths: List[str],
        all_embeddings: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        Check if *embedding* is near-duplicate of any vector in *all_embeddings*.
        Returns the path of the closest match above threshold, or None.
        """
        if threshold is None:
            threshold = self.cfg.duplicate.near_duplicate_threshold

        if len(all_embeddings) == 0:
            return None

        # Cosine similarity (embeddings should already be L2-normalised)
        sims = all_embeddings @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold and all_paths[best_idx] != path:
            logger.debug(
                "Near-duplicate: %s ≈ %s (sim=%.3f)",
                path, all_paths[best_idx], best_sim,
            )
            return all_paths[best_idx]
        return None

    # ---------- stage 4: burst grouping ----------

    def register_for_burst(self, path: str, exif_datetime: Optional[str]) -> None:
        """
        Register an image for burst-shot detection.
        Images without timestamps are grouped by filename prefix.
        """
        if exif_datetime:
            try:
                import datetime
                dt = datetime.datetime.strptime(exif_datetime, "%Y:%m:%d %H:%M:%S")
                # Key = floor to burst_time_window_seconds bucket
                bucket = int(dt.timestamp() / self.cfg.duplicate.burst_time_window_seconds)
                self._bursts[str(bucket)].append(path)
                return
            except Exception:
                pass
        # Fallback: group by numeric stem prefix (e.g. IMG_1234, IMG_1235)
        stem = Path(path).stem
        # Strip trailing digits to get a common prefix
        prefix = stem.rstrip("0123456789")
        self._bursts[prefix].append(path)

    def get_burst_groups(self) -> Dict[str, List[str]]:
        """Return groups with >1 member (actual bursts)."""
        return {k: v for k, v in self._bursts.items() if len(v) > 1}

    def select_best_from_burst(
        self,
        burst_paths: List[str],
        quality_scores: Dict[str, float],
    ) -> str:
        """Return the highest-quality image from a burst group."""
        return max(burst_paths, key=lambda p: quality_scores.get(p, 0.0))

    # ---------- public summary ----------

    @property
    def duplicates(self) -> Dict[str, str]:
        return dict(self._duplicates)

    def mark_duplicate(self, path: str, canonical: str) -> None:
        self._duplicates[path] = canonical

    def is_duplicate(self, path: str) -> bool:
        return path in self._duplicates

    def get_canonical(self, path: str) -> str:
        return self._duplicates.get(path, path)
