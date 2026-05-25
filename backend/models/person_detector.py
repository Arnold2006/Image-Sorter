"""
Person detection and pose estimation using YOLOv8.

Features:
- Detects people in any orientation (including upside-down, airborne)
- Estimates 17-keypoint COCO pose skeleton
- Handles partial bodies
- Orientation normalization via pose analysis
- Configurable confidence / IoU thresholds
- Automatic model download on first use
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.data.schemas import BoundingBox, Detection, KeyPoint, PoseSkeleton
from backend.utils.config import AppConfig, get_config

logger = logging.getLogger(__name__)

# YOLO pose model keypoint indices (COCO 17-point)
_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16


class PersonDetector:
    """
    YOLOv8-pose person detector.
    Lazily loads the model on first call to avoid startup overhead.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self._model = None
        self._device: Optional[str] = None

    # ---------- lazy loading ----------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        model_name = self.cfg.model.yolo_model
        model_dir = Path(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_name

        if not model_path.exists():
            logger.info(
                "YOLO model %s not found locally; downloading…", model_name
            )

        self._device = self.cfg.resolve_device()
        logger.info("Loading YOLO model %s on %s", model_name, self._device)
        self._model = YOLO(str(model_path) if model_path.exists() else model_name)
        # Move to device
        try:
            self._model.to(self._device)
        except Exception as e:
            logger.warning("Could not move YOLO to %s: %s", self._device, e)
            self._device = "cpu"
            self._model.to("cpu")
        logger.info("YOLO model loaded.")

    # ---------- public API ----------

    def detect(
        self,
        img_rgb: np.ndarray,
        augment: bool = False,
    ) -> List[Detection]:
        """
        Run person detection + pose estimation on an RGB image.
        Returns a list of Detection objects (one per person found).

        *augment* enables test-time augmentation (TTA) for better recall on
        partially visible or rotated athletes at the cost of ~3× latency.
        """
        self._load_model()
        assert self._model is not None

        # Convert RGB → BGR for ultralytics (it internally converts back)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = img_rgb.shape[:2]

        results = self._model(
            bgr,
            conf=self.cfg.model.yolo_conf_threshold,
            iou=self.cfg.model.yolo_iou_threshold,
            imgsz=self.cfg.model.yolo_img_size,
            augment=augment,
            verbose=False,
            classes=[0],  # person class only
        )

        detections: List[Detection] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes.xyxy.cpu().numpy()     # (N,4)
            confs = res.boxes.conf.cpu().numpy()      # (N,)
            kpts = (
                res.keypoints.xy.cpu().numpy()        # (N,17,2)
                if res.keypoints is not None
                else None
            )
            kpt_confs = (
                res.keypoints.conf.cpu().numpy()      # (N,17)
                if res.keypoints is not None and res.keypoints.conf is not None
                else None
            )

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(confs[i])

                # Filter out very small detections
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w_img * h_img)
                if area_ratio < self.cfg.pipeline.min_person_area_ratio:
                    continue

                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                )

                pose: Optional[PoseSkeleton] = None
                if kpts is not None:
                    kp_list = [
                        KeyPoint(
                            x=float(kpts[i, j, 0]),
                            y=float(kpts[i, j, 1]),
                            confidence=float(kpt_confs[i, j])
                            if kpt_confs is not None
                            else 0.5,
                            name=PoseSkeleton.KEYPOINT_NAMES[j]
                            if j < len(PoseSkeleton.KEYPOINT_NAMES)
                            else f"kp_{j}",
                        )
                        for j in range(kpts.shape[1])
                    ]
                    pose_conf = float(np.mean([k.confidence for k in kp_list]))
                    pose = PoseSkeleton(keypoints=kp_list, confidence=pose_conf)

                rotation_angle, is_upside_down = self._estimate_orientation(pose, bbox)

                # Expand bounding box for full-body capture
                expanded = bbox.expand(
                    self.cfg.pipeline.crop_expansion, w_img, h_img
                )
                crop = _crop_image(img_rgb, expanded)

                # Normalise rotation if needed
                crop_norm = _normalise_crop(crop, rotation_angle)

                det = Detection(
                    bbox=bbox,
                    pose=pose,
                    rotation_angle=rotation_angle,
                    is_upside_down=is_upside_down,
                    crop=crop,
                    crop_normalized=crop_norm,
                )
                detections.append(det)

        logger.debug("Detected %d persons in image", len(detections))
        return detections

    # ---------- orientation estimation ----------

    def _estimate_orientation(
        self,
        pose: Optional[PoseSkeleton],
        bbox: BoundingBox,
    ) -> Tuple[float, bool]:
        """
        Estimate the body rotation angle and whether the person is upside-down.
        Uses shoulder-hip vector as the primary axis.
        Falls back to bounding-box aspect ratio heuristics.

        Returns (rotation_degrees, is_upside_down).
        """
        if pose is None or len(pose.keypoints) < 13:
            return 0.0, False

        kp = pose.keypoints

        def _get_kp(idx: int) -> Optional[Tuple[float, float]]:
            if idx >= len(kp) or kp[idx].confidence < 0.3:
                return None
            return kp[idx].x, kp[idx].y

        ls = _get_kp(_L_SHOULDER)
        rs = _get_kp(_R_SHOULDER)
        lh = _get_kp(_L_HIP)
        rh = _get_kp(_R_HIP)

        # Mid-shoulder and mid-hip
        if ls and rs and lh and rh:
            mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

            dx = mid_hip[0] - mid_shoulder[0]
            dy = mid_hip[1] - mid_shoulder[1]

            # In an upright person, hip is BELOW shoulder → dy > 0 in image coords
            is_upside_down = dy < 0

            # Rotation = angle of the torso vector from vertical
            angle_from_vertical = math.degrees(math.atan2(dx, abs(dy)))
            return angle_from_vertical, is_upside_down

        # Fallback: check if head keypoints are below hip keypoints
        nose = _get_kp(_NOSE)
        head_y = nose[1] if nose else None
        hip_y = None
        if lh and rh:
            hip_y = (lh[1] + rh[1]) / 2
        elif lh:
            hip_y = lh[1]
        elif rh:
            hip_y = rh[1]

        if head_y is not None and hip_y is not None:
            is_upside_down = head_y > hip_y
            return 0.0, is_upside_down

        return 0.0, False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _crop_image(img: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop and return a sub-image from *img* using *bbox* pixel coordinates."""
    h, w = img.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img
    return crop


def _normalise_crop(crop: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the crop so the person appears upright.
    Handles 0, 90, 180, 270 rotations efficiently.
    """
    if abs(angle) < 5:
        return crop
    h, w = crop.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated = cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated
