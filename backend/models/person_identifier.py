"""
Individual gymnast identification.

Multi-modal approach combining:
  A. Face recognition  –  InsightFace (buffalo_l) with profile-face handling
  B. Body Re-ID        –  OSNet (torchreid) or CLIP fallback
  C. Pose skeleton     –  normalised limb-ratio feature vector
  D. Hair descriptor   –  colour + texture embedding from head crop
  E. Multi-image consensus  –  confidence improves with more evidence

All modalities produce a fixed-dimension embedding which is stored in FAISS /
SQLite for incremental clustering and active-learning re-labelling.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.data.schemas import Detection, PoseSkeleton, PersonPrediction
from backend.utils.config import AppConfig, get_config
from backend.utils.image_utils import color_histogram

logger = logging.getLogger(__name__)

# COCO keypoint index aliases
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_NOSE = 0


# ---------------------------------------------------------------------------
# InsightFace wrapper
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """InsightFace-based face detection and recognition."""

    def __init__(self, model_name: str, device: str, model_dir: str) -> None:
        self.model_name = model_name
        self.device = device
        self.model_dir = model_dir
        self._app = None

    def _load(self) -> None:
        if self._app is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
            ctx_id = 0 if self.device.startswith("cuda") else -1
            self._app = FaceAnalysis(
                name=self.model_name,
                root=self.model_dir,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace loaded: %s (ctx_id=%d)", self.model_name, ctx_id)
        except ImportError:
            logger.warning("insightface not installed; face recognition disabled.")

    def get_embedding(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a 512-d face embedding from the largest face in *img_rgb*.
        Returns None if no face is found or the model is unavailable.
        """
        if self._app is None:
            self._load()
        if self._app is None:
            return None
        try:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            faces = self._app.get(bgr)
            if not faces:
                return None
            # Pick the largest face by bounding box area
            best = max(faces, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ))
            emb = best.normed_embedding  # already L2-normalised
            return emb.astype(np.float32)
        except Exception as exc:
            logger.debug("Face recognition error: %s", exc)
            return None

    def get_embedding_with_confidence(
        self, img_rgb: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """Return (embedding, det_score). det_score in [0,1]."""
        if self._app is None:
            self._load()
        if self._app is None:
            return None, 0.0
        try:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            faces = self._app.get(bgr)
            if not faces:
                return None, 0.0
            best = max(faces, key=lambda f: float(f.det_score))
            score = float(best.det_score)
            if score < get_config().model.face_conf_threshold:
                return None, score
            return best.normed_embedding.astype(np.float32), score
        except Exception as exc:
            logger.debug("Face recognition error: %s", exc)
            return None, 0.0


# ---------------------------------------------------------------------------
# Person Re-ID wrapper (OSNet via torchreid, with CLIP fallback)
# ---------------------------------------------------------------------------

class PersonReID:
    """
    Person re-identification embeddings.
    Primary: OSNet (torchreid)
    Fallback: CLIP ViT image encoder (already available from team_identifier)
    """

    def __init__(self, model_name: str, device: str, model_dir: str) -> None:
        self.model_name = model_name
        self.device = device
        self.model_dir = model_dir
        self._model = None
        self._backend: str = ""
        self._transform = None

    def _load_torchreid(self) -> bool:
        try:
            import torchreid
            import torch
            self._model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1,       # feature-extraction mode
                pretrained=True,
            )
            self._model.eval()
            self._model.to(self.device)
            from torchvision import transforms
            self._transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self._backend = "torchreid"
            logger.info("Torchreid OSNet loaded (%s)", self.model_name)
            return True
        except ImportError:
            return False
        except Exception as exc:
            logger.warning("Torchreid load failed: %s", exc)
            return False

    def _load_clip_fallback(self) -> None:
        try:
            import clip
            import torch
            self._model, self._transform = clip.load("ViT-B/32", device=self.device)
            self._backend = "clip_fallback"
            logger.info("Re-ID using CLIP ViT-B/32 fallback")
        except ImportError:
            try:
                import open_clip
                self._model, _, self._transform = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
                )
                self._backend = "open_clip_fallback"
                logger.info("Re-ID using open_clip ViT-B-32 fallback")
            except ImportError:
                logger.error("No Re-ID backend available (torchreid or clip required).")

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._load_torchreid():
            self._load_clip_fallback()

    def encode(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Encode a list of RGB person crops.
        Returns (N, D) float32 L2-normalised matrix.
        """
        import torch
        from PIL import Image as PILImage

        self._load()
        if self._model is None:
            return np.zeros((len(crops), 512), dtype=np.float32)

        pil_imgs = [PILImage.fromarray(c.astype(np.uint8)) for c in crops]

        with torch.no_grad():
            if self._backend == "torchreid":
                tensors = torch.stack(
                    [self._transform(img) for img in pil_imgs]
                ).to(self.device)
                feats = self._model(tensors)  # (N, D)
            else:
                tensors = torch.stack(
                    [self._transform(img) for img in pil_imgs]
                ).to(self.device)
                feats = self._model.encode_image(tensors).float()

        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-6)
        return feats / norms


# ---------------------------------------------------------------------------
# Pose feature extractor
# ---------------------------------------------------------------------------

def pose_feature_vector(skeleton: PoseSkeleton) -> np.ndarray:
    """
    Build a rotation-invariant pose feature vector from 17 COCO keypoints.

    The vector encodes normalised limb lengths and body proportions relative
    to the torso height, making it insensitive to scale, translation, and
    partial rotation.  Length = 12 features.
    """
    kp = skeleton.keypoints

    def _pt(idx: int) -> Optional[np.ndarray]:
        if idx >= len(kp) or kp[idx].confidence < 0.3:
            return None
        return np.array([kp[idx].x, kp[idx].y])

    def _dist(a, b) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    ls, rs = _pt(_L_SHOULDER), _pt(_R_SHOULDER)
    lh, rh = _pt(_L_HIP), _pt(_R_HIP)
    lk, rk = _pt(_L_KNEE), _pt(_R_KNEE)
    la, ra = _pt(_L_ANKLE), _pt(_R_ANKLE)
    le, re = _pt(_L_ELBOW), _pt(_R_ELBOW)
    lw, rw = _pt(_L_WRIST), _pt(_R_WRIST)

    # Torso height as normalisation factor
    torso = _dist(
        _midpoint(ls, rs),
        _midpoint(lh, rh)
    )
    if torso is None or torso < 1e-3:
        return np.zeros(12, dtype=np.float32)

    def _norm(d: Optional[float]) -> float:
        return float(d / torso) if d is not None else 0.0

    features = np.array([
        _norm(_dist(ls, rs)),           # shoulder width
        _norm(_dist(lh, rh)),           # hip width
        _norm(_dist(ls, lh)),           # left torso side
        _norm(_dist(rs, rh)),           # right torso side
        _norm(_dist(lh, lk)),           # left thigh
        _norm(_dist(rh, rk)),           # right thigh
        _norm(_dist(lk, la)),           # left shin
        _norm(_dist(rk, ra)),           # right shin
        _norm(_dist(ls, le)),           # left upper arm
        _norm(_dist(rs, re)),           # right upper arm
        _norm(_dist(le, lw)),           # left forearm
        _norm(_dist(re, rw)),           # right forearm
    ], dtype=np.float32)
    return features


def _midpoint(a, b) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Hair / head appearance descriptor
# ---------------------------------------------------------------------------

def hair_descriptor(
    img_rgb: np.ndarray,
    det: Detection,
    head_fraction: float = 0.25,
) -> np.ndarray:
    """
    Compute a compact hair appearance descriptor from the top portion of the
    athlete crop (assumed to contain the head/hair region).

    Returns a 48-d float32 vector (HSV colour histogram of head region).
    """
    crop = det.crop_normalized if det.crop_normalized is not None else det.crop
    if crop is None or crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    h, w = crop.shape[:2]
    head_h = max(1, int(h * head_fraction))
    head_region = crop[:head_h, :]

    hist = color_histogram(head_region, bins=16)
    return hist[:48].astype(np.float32)


# ---------------------------------------------------------------------------
# PersonIdentifier  (main interface)
# ---------------------------------------------------------------------------

class PersonIdentifier:
    """
    Combines face, Re-ID, pose, and hair modalities to build
    a unified person identity embedding.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self._device = self.cfg.resolve_device()
        self._face = FaceRecognizer(
            self.cfg.model.insightface_model,
            self._device,
            self.cfg.model_dir,
        )
        self._reid = PersonReID(
            self.cfg.model.reid_model,
            self._device,
            self.cfg.model_dir,
        )

    # ---------- per-detection feature extraction ----------

    def extract_features(
        self,
        img_rgb: np.ndarray,
        det: Detection,
    ) -> dict:
        """
        Extract all identity features for a single detection.
        Returns a dict with keys: face, reid, pose, hair, fused.
        """
        crop = det.crop_normalized if det.crop_normalized is not None else det.crop
        if crop is None:
            return self._empty_features()

        # --- Face ---
        face_emb, face_conf = self._face.get_embedding_with_confidence(crop)

        # --- Re-ID ---
        reid_embs = self._reid.encode([crop])
        reid_emb = reid_embs[0] if len(reid_embs) > 0 else np.zeros(512, dtype=np.float32)

        # --- Pose ---
        pose_vec = np.zeros(12, dtype=np.float32)
        if det.pose is not None:
            pose_vec = pose_feature_vector(det.pose)

        # --- Hair ---
        hair_vec = hair_descriptor(img_rgb, det)

        # --- Fused ---
        fused = self._fuse(face_emb, reid_emb, pose_vec, hair_vec, face_conf)

        return {
            "face": face_emb,
            "face_conf": face_conf,
            "reid": reid_emb,
            "pose": pose_vec,
            "hair": hair_vec,
            "fused": fused,
        }

    def _fuse(
        self,
        face_emb: Optional[np.ndarray],
        reid_emb: np.ndarray,
        pose_vec: np.ndarray,
        hair_vec: np.ndarray,
        face_conf: float,
    ) -> np.ndarray:
        """
        Weighted concatenation of normalised modality embeddings.
        Dimensions: face(512) + reid(512) + pose(12) + hair(48) → padded to 512
        after PCA in the clustering stage.
        """
        cfg = self.cfg.clustering

        def _safe_norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return (v / n) if n > 1e-6 else v

        # Use face weight only if detection was confident
        face_w = cfg.person_face_weight * min(face_conf / 0.7, 1.0)
        face_part = _safe_norm(face_emb) * face_w if face_emb is not None else np.zeros(512, dtype=np.float32)

        reid_part = _safe_norm(reid_emb) * cfg.person_reid_weight
        pose_part = _safe_norm(pose_vec) * cfg.person_pose_weight
        hair_part = _safe_norm(hair_vec) * cfg.person_hair_weight

        # Pad shorter vectors to 512 for uniform shape
        def _pad(v: np.ndarray, target: int = 512) -> np.ndarray:
            if v.shape[0] >= target:
                return v[:target]
            return np.pad(v, (0, target - v.shape[0]))

        fused = _pad(face_part) + _pad(reid_part) + _pad(pose_part) + _pad(hair_part)
        n = np.linalg.norm(fused)
        return (fused / n).astype(np.float32) if n > 1e-6 else fused.astype(np.float32)

    def _empty_features(self) -> dict:
        return {
            "face": None,
            "face_conf": 0.0,
            "reid": np.zeros(512, dtype=np.float32),
            "pose": np.zeros(12, dtype=np.float32),
            "hair": np.zeros(48, dtype=np.float32),
            "fused": np.zeros(512, dtype=np.float32),
        }

    # ---------- batch encoding ----------

    def encode_batch(
        self,
        img_rgb: np.ndarray,
        detections: List[Detection],
    ) -> List[dict]:
        """Extract features for all detections in a single image."""
        return [self.extract_features(img_rgb, det) for det in detections]
