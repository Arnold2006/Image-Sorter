"""
Team identification model.

Multi-stage ensemble approach:
  A. Uniform / outfit colour analysis (dominant colours + HSV histogram)
  B. CLIP semantic outfit embeddings
  C. OCR detection of jersey numbers, team names, and logos
  D. Pose-aware feature extraction (torso region crop)
  E. Temporal / group similarity (images from the same time window share a team)

Results from all stages are combined using a weighted scoring scheme
defined in ClusteringConfig.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.data.schemas import BoundingBox, Detection, TeamPrediction
from backend.utils.config import AppConfig, get_config
from backend.utils.image_utils import (
    color_histogram,
    dominant_colors,
    histogram_similarity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """
    Thin wrapper around OpenAI CLIP for outfit / uniform semantic similarity.
    Supports both the `clip` (openai) package and `open_clip`.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._preprocess = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import clip  # openai/clip
            self._model, self._preprocess = clip.load(
                self.model_name, device=self.device
            )
            self._backend = "openai_clip"
            logger.info("CLIP (openai) loaded: %s on %s", self.model_name, self.device)
        except ImportError:
            try:
                import open_clip
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained="laion2b_s32b_b82k",
                    device=self.device,
                )
                self._backend = "open_clip"
                logger.info("CLIP (open_clip) loaded: %s on %s", self.model_name, self.device)
            except ImportError:
                raise RuntimeError(
                    "Neither 'clip' nor 'open_clip' is installed. "
                    "Run: pip install git+https://github.com/openai/CLIP.git"
                )

    def encode_images(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Encode a list of RGB numpy crops.
        Returns (N, D) float32 L2-normalised feature matrix.
        """
        import torch
        from PIL import Image as PILImage

        self._load()
        assert self._model is not None

        pil_images = [PILImage.fromarray(c.astype(np.uint8)) for c in crops]
        tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(self.device)

        with torch.no_grad():
            if self._backend == "openai_clip":
                import clip
                features = self._model.encode_image(tensors).float()
            else:
                features = self._model.encode_image(tensors).float()

        # L2 normalise
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        features = (features / norms).cpu().numpy()
        return features.astype(np.float32)


# ---------------------------------------------------------------------------
# OCR wrapper
# ---------------------------------------------------------------------------

class OCRDetector:
    """Detects team names, jersey numbers, and logos via EasyOCR."""

    def __init__(self, languages: List[str], device: str) -> None:
        self.languages = languages
        self.device = device
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr
            gpu = self.device.startswith("cuda") or self.device == "mps"
            self._reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
            logger.info("EasyOCR loaded (gpu=%s)", gpu)
        except ImportError:
            logger.warning("easyocr not installed; OCR stage disabled.")

    def read(self, img_rgb: np.ndarray) -> List[str]:
        """Return list of detected text strings (uppercased, stripped)."""
        if self._reader is None:
            self._load()
        if self._reader is None:
            return []
        try:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            results = self._reader.readtext(bgr, detail=0, paragraph=True)
            return [str(r).strip().upper() for r in results if r]
        except Exception as exc:
            logger.debug("OCR error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Torso segmentation helper
# ---------------------------------------------------------------------------

def _extract_torso_region(
    img_rgb: np.ndarray,
    det: Detection,
) -> Optional[np.ndarray]:
    """
    Extract the torso / uniform region of a detected athlete.
    Uses shoulder + hip keypoints when available; falls back to the central
    60% of the bounding box.
    """
    if det.crop is None:
        return None
    crop = det.crop
    h, w = crop.shape[:2]

    if det.pose and len(det.pose.keypoints) >= 13:
        kp = det.pose.keypoints
        # Indices in the crop-local coordinate system
        x_offset = det.bbox.x1
        y_offset = det.bbox.y1

        def _local(idx: int) -> Optional[Tuple[int, int]]:
            if kp[idx].confidence < 0.3:
                return None
            lx = int(kp[idx].x - x_offset)
            ly = int(kp[idx].y - y_offset)
            if 0 <= lx < w and 0 <= ly < h:
                return lx, ly
            return None

        ls = _local(5); rs = _local(6)
        lh = _local(11); rh = _local(12)

        if ls and rs and lh and rh:
            pts_x = [ls[0], rs[0], lh[0], rh[0]]
            pts_y = [ls[1], rs[1], lh[1], rh[1]]
            x1 = max(0, min(pts_x) - 10)
            y1 = max(0, min(pts_y) - 10)
            x2 = min(w, max(pts_x) + 10)
            y2 = min(h, max(pts_y) + 10)
            if x2 > x1 and y2 > y1:
                return crop[y1:y2, x1:x2]

    # Fallback: central 60% of height, full width
    y1 = int(h * 0.20)
    y2 = int(h * 0.80)
    return crop[y1:y2, :] if y2 > y1 else crop


# ---------------------------------------------------------------------------
# TeamIdentifier
# ---------------------------------------------------------------------------

class TeamIdentifier:
    """
    Identifies which team a detected athlete belongs to by combining:
    - uniform colour analysis
    - CLIP semantic embeddings
    - OCR text detection
    - pose-aware torso features
    - temporal grouping (provided externally as time window hint)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self._device = self.cfg.resolve_device()
        self._clip: Optional[CLIPEmbedder] = None
        self._ocr: Optional[OCRDetector] = None

    # ---------- lazy init ----------

    def _get_clip(self) -> CLIPEmbedder:
        if self._clip is None:
            self._clip = CLIPEmbedder(self.cfg.model.clip_model, self._device)
        return self._clip

    def _get_ocr(self) -> OCRDetector:
        if self._ocr is None:
            self._ocr = OCRDetector(self.cfg.model.ocr_languages, self._device)
        return self._ocr

    # ---------- feature extraction ----------

    def extract_color_features(
        self,
        img_rgb: np.ndarray,
        det: Detection,
    ) -> Dict:
        """
        Extract colour-based uniform features.
        Returns dict with 'histogram', 'dominant_colors', 'color_score' placeholder.
        """
        torso = _extract_torso_region(img_rgb, det)
        if torso is None or torso.size == 0:
            torso = det.crop if det.crop is not None else img_rgb

        hist = color_histogram(torso, bins=32)
        dom_colors = dominant_colors(torso, n_colors=5)
        return {
            "histogram": hist,
            "dominant_colors": dom_colors,
        }

    def extract_clip_features(
        self,
        crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute CLIP embeddings for a batch of crops.
        Returns (N, D) float32 array.
        """
        if not crops:
            return np.empty((0, 768), dtype=np.float32)
        try:
            embedder = self._get_clip()
            return embedder.encode_images(crops)
        except Exception as exc:
            logger.warning("CLIP encoding failed: %s", exc)
            dim = 768
            return np.zeros((len(crops), dim), dtype=np.float32)

    def extract_ocr_text(
        self,
        img_rgb: np.ndarray,
        det: Detection,
    ) -> List[str]:
        """Run OCR on the athlete crop. Returns list of detected strings."""
        if not self.cfg.model.ocr_enabled:
            return []
        crop = det.crop if det.crop is not None else img_rgb
        return self._get_ocr().read(crop)

    # ---------- scoring against known teams ----------

    def score_against_team(
        self,
        candidate_hist: np.ndarray,
        candidate_clip: np.ndarray,
        candidate_texts: List[str],
        team_hist_centroid: np.ndarray,
        team_clip_centroid: np.ndarray,
        team_texts: List[str],
    ) -> float:
        """
        Compute a [0,1] similarity score between a candidate detection and a
        known team prototype using the configured ensemble weights.
        """
        cfg = self.cfg.clustering

        # Colour score
        color_sim = histogram_similarity(candidate_hist, team_hist_centroid)

        # CLIP score (cosine similarity; both are L2-normalised)
        if candidate_clip.shape[-1] == team_clip_centroid.shape[-1]:
            clip_sim = float(np.dot(candidate_clip.flatten(),
                                    team_clip_centroid.flatten()))
            clip_sim = (clip_sim + 1.0) / 2.0  # map [-1,1] → [0,1]
        else:
            clip_sim = 0.0

        # OCR score: fraction of team texts found in candidate texts
        ocr_sim = 0.0
        if team_texts:
            matches = sum(1 for t in team_texts if any(t in c for c in candidate_texts))
            ocr_sim = matches / len(team_texts)

        score = (
            cfg.team_color_weight * color_sim
            + cfg.team_clip_weight * clip_sim
            + cfg.team_ocr_weight * ocr_sim
        )
        return float(np.clip(score, 0.0, 1.0))

    def build_team_prediction(
        self,
        team_id: str,
        team_name: str,
        color_score: float,
        clip_score: float,
        ocr_score: float,
        temporal_score: float,
        dom_colors: List[Tuple[int, int, int]],
        detected_texts: List[str],
    ) -> TeamPrediction:
        """Build a TeamPrediction from individual component scores."""
        cfg = self.cfg.clustering
        confidence = (
            cfg.team_color_weight * color_score
            + cfg.team_clip_weight * clip_score
            + cfg.team_ocr_weight * ocr_score
            + cfg.team_temporal_weight * temporal_score
        )
        return TeamPrediction(
            team_id=team_id,
            team_name=team_name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            color_score=color_score,
            clip_score=clip_score,
            ocr_score=ocr_score,
            temporal_score=temporal_score,
            dominant_colors=dom_colors,
            detected_text=detected_texts,
        )
