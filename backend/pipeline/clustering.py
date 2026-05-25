"""
Clustering pipeline: hierarchical two-stage clustering
  Stage 1 – cluster detections by TEAM using colour + CLIP embeddings
  Stage 2 – cluster individuals WITHIN each team using fused person embeddings

Algorithms: HDBSCAN (primary), DBSCAN, agglomerative (fallbacks).
FAISS provides fast approximate kNN for large datasets.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize

from backend.data.schemas import PersonCluster, TeamCluster
from backend.utils.config import AppConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level clustering helpers
# ---------------------------------------------------------------------------

def _run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Run HDBSCAN. Returns label array (-1 = noise)."""
    try:
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        return clusterer.fit_predict(X)
    except ImportError:
        logger.warning("hdbscan not installed; falling back to DBSCAN.")
        return _run_dbscan(X, min_cluster_size, 0.35)


def _run_dbscan(
    X: np.ndarray,
    min_samples: int,
    eps: float,
) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return clusterer.fit_predict(X)


def _run_agglomerative(
    X: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return clusterer.fit_predict(X)


def _cluster(
    X: np.ndarray,
    method: str,
    min_cluster_size: int,
    min_samples: int,
    eps: float,
    n_clusters: Optional[int] = None,
) -> np.ndarray:
    """Dispatch to the selected clustering algorithm."""
    if len(X) < 2:
        return np.zeros(len(X), dtype=int)

    if method == "hdbscan":
        return _run_hdbscan(X, max(2, min_cluster_size), max(1, min_samples))
    elif method == "dbscan":
        return _run_dbscan(X, max(1, min_samples), eps)
    elif method == "agglomerative" and n_clusters:
        return _run_agglomerative(X, n_clusters)
    elif method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        n = n_clusters or max(2, int(np.sqrt(len(X) / 2)))
        km = MiniBatchKMeans(n_clusters=n, n_init=5, random_state=42)
        return km.fit_predict(X)
    else:
        return _run_hdbscan(X, max(2, min_cluster_size), max(1, min_samples))


def _compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute the mean (centroid) vector for each cluster label."""
    centroids: Dict[int, np.ndarray] = {}
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        centroid = X[mask].mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 1e-6:
            centroid /= n
        centroids[label] = centroid.astype(np.float32)
    return centroids


# ---------------------------------------------------------------------------
# TeamClusterer
# ---------------------------------------------------------------------------

class TeamClusterer:
    """
    Clusters a set of detection embeddings into teams.

    Input:  (N, D) matrix of team-feature vectors (colour hist + CLIP fused)
    Output: list of TeamCluster objects + label array

    Each unknown team is assigned an auto-generated ID (team_001, team_002…)
    Noise points (-1 labels) get assigned to the nearest cluster centroid.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()

    def cluster(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        det_indices: List[int],
        dominant_colors_list: Optional[List[List[Tuple]]] = None,
        detected_texts_list: Optional[List[List[str]]] = None,
    ) -> Tuple[np.ndarray, List[TeamCluster]]:
        """
        Parameters
        ----------
        embeddings : (N, D) float32 matrix, one row per detection
        image_paths : corresponding image paths
        det_indices : corresponding detection indices within each image
        dominant_colors_list : optional list of dominant-colour tuples per detection
        detected_texts_list  : optional list of OCR text strings per detection

        Returns
        -------
        labels      : (N,) int array; -1 = unassigned
        team_clusters : list of TeamCluster objects
        """
        cfg = self.cfg.clustering
        N = len(embeddings)
        if N == 0:
            return np.array([], dtype=int), []

        # L2-normalise for cosine-equivalent distance with Euclidean metrics
        X = normalize(embeddings, norm="l2")

        labels = _cluster(
            X,
            method=cfg.team_method,
            min_cluster_size=cfg.team_min_cluster_size,
            min_samples=cfg.team_min_samples,
            eps=cfg.team_eps,
        )

        # Assign noise points to nearest centroid
        centroids = _compute_centroids(X, labels)
        if centroids:
            noise_mask = labels == -1
            if noise_mask.any():
                labels = self._assign_noise(X, labels, centroids)

        # Build TeamCluster objects
        clusters: List[TeamCluster] = []
        unique_labels = sorted(set(labels) - {-1})

        for i, label in enumerate(unique_labels):
            team_id = f"team_{i + 1:03d}"
            mask = labels == label
            count = int(mask.sum())
            rep_images = [
                image_paths[j]
                for j in np.argsort(-X[mask].sum(axis=1))[:5]
                # pick top-5 images most aligned with centroid
            ]
            # Aggregate dominant colours
            agg_colors: List[Tuple] = []
            if dominant_colors_list:
                for j in np.where(mask)[0][:20]:
                    agg_colors.extend(dominant_colors_list[j])

            tc = TeamCluster(
                team_id=team_id,
                team_name=team_id,
                image_count=count,
                confidence=float(
                    np.mean(X[mask] @ centroids[label])
                    if label in centroids
                    else 0.0
                ),
                dominant_colors=list(set(agg_colors))[:5],
                representative_images=[image_paths[j] for j in np.where(mask)[0][:5]],
                embedding_centroid=centroids.get(label),
            )
            clusters.append(tc)
            # Re-label from raw integer to team_id position
            labels[labels == label] = i

        logger.info(
            "Team clustering: %d detections → %d teams (noise=%d)",
            N, len(clusters), int((labels == -1).sum()),
        )
        return labels, clusters

    @staticmethod
    def _assign_noise(
        X: np.ndarray,
        labels: np.ndarray,
        centroids: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Assign noise points to the nearest cluster centroid."""
        centroid_matrix = np.stack(list(centroids.values()))  # (K, D)
        centroid_labels = list(centroids.keys())
        noise_idx = np.where(labels == -1)[0]
        if len(noise_idx) == 0:
            return labels
        sims = X[noise_idx] @ centroid_matrix.T   # (noise_n, K)
        best = np.argmax(sims, axis=1)
        for i, ni in enumerate(noise_idx):
            labels[ni] = centroid_labels[best[i]]
        return labels


# ---------------------------------------------------------------------------
# PersonClusterer
# ---------------------------------------------------------------------------

class PersonClusterer:
    """
    Within a team, clusters individual athletes from their fused person embeddings.

    Uses the same algorithm family as TeamClusterer but with person-tuned
    hyperparameters and an optional active-learning weight update step.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()

    def cluster(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        det_indices: List[int],
        team_id: str,
    ) -> Tuple[np.ndarray, List[PersonCluster]]:
        """
        Cluster detections within a single team.

        Returns (labels, person_clusters) where labels[i] is the index into
        person_clusters for detection i, or -1 for unassigned.
        """
        cfg = self.cfg.clustering
        N = len(embeddings)
        if N == 0:
            return np.array([], dtype=int), []

        X = normalize(embeddings, norm="l2")

        labels = _cluster(
            X,
            method=cfg.person_method,
            min_cluster_size=cfg.person_min_cluster_size,
            min_samples=cfg.person_min_samples,
            eps=cfg.person_eps,
        )

        centroids = _compute_centroids(X, labels)
        if centroids:
            noise_mask = labels == -1
            if noise_mask.any():
                labels = TeamClusterer._assign_noise(X, labels, centroids)

        clusters: List[PersonCluster] = []
        unique_labels = sorted(set(labels) - {-1})

        for i, label in enumerate(unique_labels):
            person_id = f"{team_id}_gymnast_{i + 1:03d}"
            mask = labels == label
            pc = PersonCluster(
                person_id=person_id,
                person_name=person_id,
                team_id=team_id,
                image_count=int(mask.sum()),
                confidence=float(
                    np.mean(X[mask] @ centroids[label])
                    if label in centroids
                    else 0.0
                ),
                representative_images=[image_paths[j] for j in np.where(mask)[0][:5]],
                embedding_centroid=centroids.get(label),
            )
            clusters.append(pc)
            labels[labels == label] = i

        logger.info(
            "Person clustering [%s]: %d detections → %d individuals (noise=%d)",
            team_id, N, len(clusters), int((labels == -1).sum()),
        )
        return labels, clusters

    def update_from_correction(
        self,
        embeddings: np.ndarray,
        person_clusters: List[PersonCluster],
        corrected_idx: int,
        correct_cluster_idx: int,
    ) -> List[PersonCluster]:
        """
        Apply a single user correction:
        Move *corrected_idx* into *correct_cluster_idx* and recompute centroid.
        """
        if correct_cluster_idx >= len(person_clusters):
            return person_clusters
        pc = person_clusters[correct_cluster_idx]
        # Update centroid as running mean (online learning)
        if pc.embedding_centroid is not None:
            old_centroid = pc.embedding_centroid
            n = max(1, pc.image_count)
            new_centroid = (old_centroid * n + embeddings[corrected_idx]) / (n + 1)
            norm = np.linalg.norm(new_centroid)
            pc.embedding_centroid = (new_centroid / norm).astype(np.float32) if norm > 1e-6 else new_centroid
        pc.image_count += 1
        return person_clusters
