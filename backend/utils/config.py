"""
Centralised configuration for the gymnastics photo sorter.

All runtime settings are defined here as a dataclass and serialised to/from
a JSON config file stored inside the app's cache directory.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[2]

# User-writable data directories (relative to repo root by default)
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_MODEL_DIR = REPO_ROOT / "cache" / "models"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model selection and paths."""

    # YOLO person detection
    yolo_model: str = "yolov8x-pose.pt"          # also used for pose
    yolo_conf_threshold: float = 0.35
    yolo_iou_threshold: float = 0.45
    yolo_img_size: int = 1280                     # higher = better for far shots

    # InsightFace
    insightface_model: str = "buffalo_l"          # most accurate offline model
    face_conf_threshold: float = 0.4

    # CLIP
    clip_model: str = "ViT-L/14"                  # best quality local CLIP
    clip_batch_size: int = 32

    # OCR (EasyOCR)
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_enabled: bool = True

    # Person Re-ID (OSNet via torchreid or CLIP fallback)
    reid_model: str = "osnet_ain_x1_0"
    reid_batch_size: int = 64

    # Segmentation for uniform color (lightweight)
    segmentation_model: str = "sam_vit_b_01ec64.pth"  # SAM base
    segmentation_enabled: bool = True


@dataclass
class PipelineConfig:
    """Processing pipeline settings."""

    batch_size: int = 16                          # images per GPU batch
    num_workers: int = 4                          # DataLoader workers
    max_memory_mb: int = 4096                     # RAM cap before flushing cache
    thumbnail_size: Tuple = (256, 256)            # (w, h) thumbnails
    min_person_area_ratio: float = 0.005          # ignore very small detections
    crop_expansion: float = 0.15                  # expand bbox by 15%
    enable_multiprocessing: bool = True
    resume_enabled: bool = True                   # skip already-processed images
    save_crops: bool = False                      # save cropped person images


@dataclass
class ClusteringConfig:
    """Clustering algorithm settings."""

    # Team clustering
    team_method: str = "hdbscan"                  # hdbscan | dbscan | agglomerative
    team_min_cluster_size: int = 3
    team_min_samples: int = 2
    team_eps: float = 0.35                        # DBSCAN epsilon (cosine dist)

    # Individual clustering
    person_method: str = "hdbscan"
    person_min_cluster_size: int = 2
    person_min_samples: int = 2
    person_eps: float = 0.4

    # Feature fusion weights for team ID
    team_color_weight: float = 0.35
    team_clip_weight: float = 0.40
    team_ocr_weight: float = 0.15
    team_temporal_weight: float = 0.10

    # Feature fusion weights for person ID
    person_face_weight: float = 0.45
    person_reid_weight: float = 0.35
    person_pose_weight: float = 0.10
    person_hair_weight: float = 0.10

    # Confidence thresholds
    team_assignment_threshold: float = 0.50
    person_assignment_threshold: float = 0.45


@dataclass
class DuplicateConfig:
    """Duplicate detection settings."""
    enabled: bool = True
    exact_hash: bool = True                       # MD5/SHA hash comparison
    perceptual_hash: bool = True                  # pHash
    perceptual_threshold: int = 8                 # Hamming distance (0-64)
    near_duplicate_threshold: float = 0.95        # embedding cosine similarity
    burst_time_window_seconds: int = 2            # images within N seconds = burst


@dataclass
class UIConfig:
    """Gradio UI settings."""
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    max_upload_size_mb: int = 100
    thumbnail_cols: int = 5
    items_per_page: int = 50
    theme: str = "default"


@dataclass
class AppConfig:
    """Master configuration object."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Paths (stored as strings for JSON serialisation)
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    data_dir: str = str(DEFAULT_DATA_DIR)
    log_dir: str = str(DEFAULT_LOG_DIR)
    model_dir: str = str(DEFAULT_MODEL_DIR)

    # Device
    device: str = "auto"                          # auto | cuda | cpu | mps
    cuda_device_index: int = 0

    # Output
    file_mode: str = "copy"                       # copy | move | symlink
    output_folder_template: str = "{team}/{person}"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    # Misc
    app_version: str = "1.0.0"

    # ---------- helpers ----------

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [self.cache_dir, self.data_dir, self.log_dir, self.model_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Return the actual torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{self.cuda_device_index}"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        cfg.model = ModelConfig(**data.get("model", {}))
        cfg.pipeline = PipelineConfig(**data.get("pipeline", {}))
        cfg.clustering = ClusteringConfig(**data.get("clustering", {}))
        cfg.duplicate = DuplicateConfig(**data.get("duplicate", {}))
        cfg.ui = UIConfig(**data.get("ui", {}))
        for k, v in data.items():
            if k not in ("model", "pipeline", "clustering", "duplicate", "ui"):
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: Optional[str] = None) -> None:
        """Persist config to JSON."""
        target = Path(path or self.data_dir) / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.debug("Config saved to %s", target)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        """Load config from JSON, falling back to defaults."""
        if path is None:
            path = str(DEFAULT_DATA_DIR / "config.json")
        p = Path(path)
        if not p.exists():
            logger.info("No config file at %s; using defaults.", p)
            return cls()
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = cls.from_dict(data)
            logger.info("Config loaded from %s", p)
            return cfg
        except Exception as exc:
            logger.warning("Failed to load config (%s); using defaults.", exc)
            return cls()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the singleton AppConfig, loading from disk on first access."""
    global _config
    if _config is None:
        _config = AppConfig.load()
        _config.ensure_dirs()
    return _config


def set_config(cfg: AppConfig) -> None:
    """Replace the singleton config (used in tests and Gradio settings panel)."""
    global _config
    _config = cfg


# Fix missing Tuple import for dataclass field
from typing import Tuple  # noqa: E402 – placed after class to avoid circular
