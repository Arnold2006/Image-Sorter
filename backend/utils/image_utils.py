"""
Image utility functions: loading, hashing, EXIF extraction, blur detection,
quality scoring, orientation detection, and thumbnail generation.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".gif",
}


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def collect_images(folder: str, recursive: bool = True) -> List[str]:
    """Return a sorted list of all image paths under *folder*."""
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    if recursive:
        paths = [str(f) for f in p.rglob("*") if is_image_file(str(f))]
    else:
        paths = [str(f) for f in p.iterdir() if f.is_file() and is_image_file(str(f))]
    return sorted(paths)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image_rgb(path: str, max_dim: int = 4096) -> Optional[np.ndarray]:
    """
    Load image as RGB numpy array.
    Handles EXIF orientation, converts HEIC via Pillow, caps very large images.
    Returns None on failure.
    """
    try:
        pil_img = Image.open(path)
        pil_img = _apply_exif_orientation(pil_img)
        # Downscale extremely large images to avoid OOM
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except UnidentifiedImageError:
        logger.warning("Cannot identify image: %s", path)
        return None
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def load_image_bgr(path: str, max_dim: int = 4096) -> Optional[np.ndarray]:
    """Load image as BGR numpy array (for OpenCV-based functions)."""
    rgb = load_image_rgb(path, max_dim)
    if rgb is None:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Rotate/flip image according to EXIF Orientation tag."""
    try:
        exif = img._getexif()  # type: ignore[attr-defined]
        if exif is None:
            return img
        orient_tag = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orient_tag is None or orient_tag not in exif:
            return img
        orientation = exif[orient_tag]
        ops = {
            2: (Image.FLIP_LEFT_RIGHT,),
            3: (Image.ROTATE_180,),
            4: (Image.FLIP_TOP_BOTTOM,),
            5: (Image.TRANSPOSE,),
            6: (Image.ROTATE_270,),
            7: (Image.TRANSVERSE,),
            8: (Image.ROTATE_90,),
        }
        for op in ops.get(orientation, []):
            img = img.transpose(op)
    except Exception:
        pass
    return img


# ---------------------------------------------------------------------------
# EXIF metadata
# ---------------------------------------------------------------------------

def extract_exif(path: str) -> dict:
    """Extract useful EXIF metadata. Returns empty dict on failure."""
    result: dict = {}
    try:
        pil_img = Image.open(path)
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return result
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        dt_tag = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        if dt_tag and dt_tag in exif_data:
            result["datetime"] = exif_data[dt_tag]
        # GPS
        gps_tag = tag_map.get("GPSInfo")
        if gps_tag and gps_tag in exif_data:
            gps = exif_data[gps_tag]
            lat = _parse_gps_coord(gps.get(2), gps.get(1))
            lon = _parse_gps_coord(gps.get(4), gps.get(3))
            if lat is not None and lon is not None:
                result["gps"] = (lat, lon)
    except Exception:
        pass
    return result


def _parse_gps_coord(coord, ref) -> Optional[float]:
    if coord is None or ref is None:
        return None
    try:
        d, m, s = [float(x) for x in coord]
        val = d + m / 60 + s / 3600
        if str(ref) in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def file_hash(path: str, algorithm: str = "md5") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(img_rgb: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    Compute a perceptual hash (average hash) of an image.
    Returns a binary array of length hash_size**2.
    """
    resized = cv2.resize(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
        (hash_size, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    mean = resized.mean()
    return (resized > mean).flatten()


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    return int(np.sum(hash1 != hash2))


# ---------------------------------------------------------------------------
# Blur / quality
# ---------------------------------------------------------------------------

def laplacian_variance(img_rgb: np.ndarray) -> float:
    """Measure image sharpness via Laplacian variance. Higher = sharper."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(img_rgb: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """Return (is_blurry, score). Score < threshold means blurry."""
    score = laplacian_variance(img_rgb)
    return score < threshold, score


def quality_score(img_rgb: np.ndarray) -> float:
    """
    Combined quality score [0, 1].
    Considers: sharpness, exposure, contrast, resolution.
    """
    h, w = img_rgb.shape[:2]
    resolution_score = min(1.0, (w * h) / (1920 * 1080))

    # Sharpness (normalised Laplacian variance)
    blur = laplacian_variance(img_rgb)
    sharpness_score = min(1.0, blur / 500.0)

    # Exposure (prefer mean luminance near 128)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    mean_lum = gray.mean()
    exposure_score = 1.0 - abs(mean_lum - 128.0) / 128.0

    # Contrast (std dev of luminance)
    contrast_score = min(1.0, gray.std() / 64.0)

    return float(
        0.35 * sharpness_score
        + 0.30 * resolution_score
        + 0.20 * exposure_score
        + 0.15 * contrast_score
    )


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by *angle* degrees (CCW) without cropping."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_bbox(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop a bounding box from an image, clamped to valid range."""
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Colour analysis
# ---------------------------------------------------------------------------

def dominant_colors(
    img_rgb: np.ndarray,
    n_colors: int = 5,
    mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int]]:
    """
    Extract *n_colors* dominant colours from *img_rgb* using K-means.
    *mask* should be a binary uint8 array (255 = include, 0 = exclude).
    Returns list of (R, G, B) tuples.
    """
    from sklearn.cluster import MiniBatchKMeans

    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    if mask is not None:
        flat_mask = mask.flatten()
        pixels = pixels[flat_mask > 0]
    if len(pixels) < n_colors:
        return [(int(img_rgb[..., 0].mean()), int(img_rgb[..., 1].mean()),
                 int(img_rgb[..., 2].mean()))]
    # Subsample for speed
    if len(pixels) > 10000:
        idx = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[idx]
    km = MiniBatchKMeans(n_clusters=n_colors, n_init=3, random_state=42)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_, minlength=n_colors)
    order = np.argsort(-counts)
    return [(int(centers[i, 0]), int(centers[i, 1]), int(centers[i, 2]))
            for i in order]


def color_histogram(
    img_rgb: np.ndarray,
    bins: int = 32,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a normalised HSV histogram feature vector."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [bins // 2], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [bins // 2], [0, 256])
    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya similarity [0,1] between two normalised histograms."""
    score = cv2.compareHist(h1.reshape(-1, 1), h2.reshape(-1, 1),
                            cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - score)


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

def make_thumbnail(
    img_rgb: np.ndarray,
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Generate a thumbnail maintaining aspect ratio inside *size*."""
    h, w = img_rgb.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_thumbnail(
    img_rgb: np.ndarray,
    output_path: str,
    size: Tuple[int, int] = (256, 256),
    quality: int = 85,
) -> None:
    """Save a JPEG thumbnail to *output_path*."""
    thumb = make_thumbnail(img_rgb, size)
    pil_thumb = Image.fromarray(thumb)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil_thumb.save(output_path, "JPEG", quality=quality, optimize=True)
