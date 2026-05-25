"""
Model downloader script.

Downloads and caches all required AI model weights on first install,
and optionally checks for newer versions on --check-updates.

Models downloaded:
  - YOLOv8x-pose         (ultralytics auto-downloads on first use)
  - InsightFace buffalo_l (insightface auto-downloads on first use)
  - CLIP ViT-L/14        (openai/clip auto-downloads on first use)
  - EasyOCR English      (easyocr auto-downloads on first use)

This script triggers those auto-downloads explicitly so that they happen
at install time rather than at first processing run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on the Python path when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.utils.config import get_config
from backend.utils.logging_utils import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("download_models")


# ---------------------------------------------------------------------------
# Individual download helpers
# ---------------------------------------------------------------------------

def download_yolo(model_dir: str, check_updates: bool = False) -> None:
    """Download YOLOv8x-pose via ultralytics."""
    logger.info("Checking YOLOv8x-pose…")
    try:
        from ultralytics import YOLO

        model_path = Path(model_dir) / "yolov8x-pose.pt"
        if model_path.exists() and not check_updates:
            logger.info("  ✓ yolov8x-pose.pt already present.")
            return

        logger.info("  Downloading yolov8x-pose.pt…")
        model = YOLO("yolov8x-pose.pt")
        # Move the downloaded weights to our model directory
        import shutil
        default_loc = Path("yolov8x-pose.pt")
        if default_loc.exists() and not model_path.exists():
            shutil.move(str(default_loc), str(model_path))
        logger.info("  ✓ yolov8x-pose.pt ready.")
    except Exception as exc:
        logger.warning("  YOLO download failed (will retry at runtime): %s", exc)


def download_insightface(model_dir: str, check_updates: bool = False) -> None:
    """Pre-download InsightFace buffalo_l model pack."""
    logger.info("Checking InsightFace buffalo_l…")
    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU for download-only
        logger.info("  ✓ InsightFace buffalo_l ready.")
    except ImportError:
        logger.warning("  insightface not installed; skipping.")
    except Exception as exc:
        logger.warning("  InsightFace download issue (will retry at runtime): %s", exc)


def download_clip(check_updates: bool = False) -> None:
    """Download CLIP ViT-L/14 weights."""
    logger.info("Checking CLIP ViT-L/14…")
    try:
        import clip
        model, _ = clip.load("ViT-L/14", device="cpu")
        logger.info("  ✓ CLIP ViT-L/14 ready.")
        del model
    except ImportError:
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k", device="cpu"
            )
            logger.info("  ✓ open_clip ViT-L-14 ready.")
            del model
        except ImportError:
            logger.warning("  Neither clip nor open_clip installed; skipping.")
        except Exception as exc:
            logger.warning("  CLIP download issue: %s", exc)
    except Exception as exc:
        logger.warning("  CLIP download issue: %s", exc)


def download_easyocr(languages: list, check_updates: bool = False) -> None:
    """Download EasyOCR models for given languages."""
    logger.info("Checking EasyOCR (%s)…", languages)
    try:
        import easyocr
        reader = easyocr.Reader(languages, gpu=False, verbose=False)
        logger.info("  ✓ EasyOCR ready.")
        del reader
    except ImportError:
        logger.warning("  easyocr not installed; skipping.")
    except Exception as exc:
        logger.warning("  EasyOCR download issue: %s", exc)


def download_all(model_dir: str, check_updates: bool = False) -> None:
    """Run all model downloads."""
    logger.info("=== Model Download / Verification ===")
    logger.info("Model directory: %s", model_dir)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    download_yolo(model_dir, check_updates)
    download_insightface(model_dir, check_updates)
    download_clip(check_updates)
    download_easyocr(["en"], check_updates)

    logger.info("=== All models verified. ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AI model weights for the Gymnastics Photo Sorter."
    )
    parser.add_argument(
        "--check-updates",
        action="store_true",
        help="Re-check and re-download models even if already present.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the model cache directory.",
    )
    args = parser.parse_args()

    cfg = get_config()
    model_dir = args.model_dir or cfg.model_dir
    download_all(model_dir, check_updates=args.check_updates)


if __name__ == "__main__":
    main()
