"""
Data schemas and models for the gymnastics photo sorter.
Defines all data structures used throughout the application.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProcessingStatus(Enum):
    """Status of an image in the processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileMode(Enum):
    """How output files are created."""
    COPY = "copy"
    MOVE = "move"
    SYMLINK = "symlink"


class ClusteringMethod(Enum):
    """Clustering algorithm to use."""
    HDBSCAN = "hdbscan"
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"


class DeviceType(Enum):
    """Compute device."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> List[float]:
        return [self.x1, self.y1, self.width, self.height]

    def expand(self, ratio: float, max_w: int, max_h: int) -> "BoundingBox":
        """Expand the box by a ratio, clamped to image bounds."""
        dw = self.width * ratio / 2
        dh = self.height * ratio / 2
        return BoundingBox(
            x1=max(0.0, self.x1 - dw),
            y1=max(0.0, self.y1 - dh),
            x2=min(float(max_w), self.x2 + dw),
            y2=min(float(max_h), self.y2 + dh),
            confidence=self.confidence,
        )


@dataclass
class KeyPoint:
    """A single pose keypoint."""
    x: float
    y: float
    confidence: float
    name: str = ""


@dataclass
class PoseSkeleton:
    """Full pose skeleton with 17 COCO keypoints."""
    keypoints: List[KeyPoint]
    confidence: float = 0.0

    # COCO keypoint names for reference
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]


@dataclass
class Detection:
    """A single person detection in an image."""
    bbox: BoundingBox
    pose: Optional[PoseSkeleton] = None
    rotation_angle: float = 0.0          # degrees; 0 = upright
    is_upside_down: bool = False
    crop: Optional[np.ndarray] = None    # HxWxC uint8 crop
    crop_normalized: Optional[np.ndarray] = None  # Rotation-normalized crop


@dataclass
class TeamPrediction:
    """Team identity prediction for a detection."""
    team_id: str                    # e.g. "team_001" or user-assigned name
    team_name: str = ""
    confidence: float = 0.0
    color_score: float = 0.0
    clip_score: float = 0.0
    ocr_score: float = 0.0
    temporal_score: float = 0.0
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    detected_text: List[str] = field(default_factory=list)


@dataclass
class PersonPrediction:
    """Individual gymnast identity prediction."""
    person_id: str                  # e.g. "gymnast_001" or user-assigned name
    person_name: str = ""
    team_id: str = ""
    confidence: float = 0.0
    face_score: float = 0.0
    reid_score: float = 0.0
    pose_score: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class ImageResult:
    """Complete processing result for a single image."""
    image_path: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    detections: List[Detection] = field(default_factory=list)
    team_predictions: List[TeamPrediction] = field(default_factory=list)
    person_predictions: List[PersonPrediction] = field(default_factory=list)

    # Image-level metadata
    width: int = 0
    height: int = 0
    exif_datetime: Optional[str] = None
    exif_gps: Optional[Tuple[float, float]] = None
    file_hash: str = ""
    is_blurry: bool = False
    blur_score: float = 0.0
    quality_score: float = 0.0

    # Duplicate detection
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    is_burst: bool = False

    # Output
    output_team: str = ""
    output_person: str = ""
    output_path: str = ""

    # Error information
    error_message: str = ""
    processing_time: float = 0.0


# ---------------------------------------------------------------------------
# Cluster records
# ---------------------------------------------------------------------------

@dataclass
class TeamCluster:
    """Represents an identified team cluster."""
    team_id: str
    team_name: str
    image_count: int = 0
    confidence: float = 0.0
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    representative_images: List[str] = field(default_factory=list)
    embedding_centroid: Optional[np.ndarray] = None


@dataclass
class PersonCluster:
    """Represents an identified individual gymnast cluster."""
    person_id: str
    person_name: str
    team_id: str
    image_count: int = 0
    confidence: float = 0.0
    representative_images: List[str] = field(default_factory=list)
    embedding_centroid: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Job / pipeline state
# ---------------------------------------------------------------------------

@dataclass
class ProcessingJob:
    """Represents a processing job submitted by the user."""
    job_id: str
    input_folder: str
    output_folder: str
    file_mode: FileMode = FileMode.COPY
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionRecord:
    """A user correction for active learning."""
    image_path: str
    detection_index: int
    old_team_id: str
    new_team_id: str
    old_person_id: str
    new_person_id: str
    timestamp: str = ""
    applied: bool = False
