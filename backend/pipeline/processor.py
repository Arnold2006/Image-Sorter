"""
Main processing pipeline.

Orchestrates the full image-processing flow:
  1. Collect images from input folder
  2. Load + validate each image (EXIF, hash, blur, quality)
  3. Detect persons with YOLOv8-pose
  4. Extract multi-modal features (CLIP, Re-ID, face, pose, colour, OCR)
  5. Cache embeddings to FAISS / SQLite
  6. Run two-stage clustering (team → individual)
  7. Organise output files

Supports: batched GPU processing, resume mode, progress tracking,
multi-image consensus, active-learning correction propagation.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.cache.embeddings_cache import EmbeddingsCache
from backend.data.schemas import (
    Detection, FileMode, ImageResult, ProcessingJob, ProcessingStatus,
)
from backend.models.duplicate_detector import DuplicateDetector
from backend.models.person_detector import PersonDetector
from backend.models.person_identifier import PersonIdentifier
from backend.models.team_identifier import TeamIdentifier
from backend.pipeline.clustering import PersonClusterer, TeamClusterer
from backend.pipeline.job_queue import ProgressTracker
from backend.utils.config import AppConfig, get_config
from backend.utils.database import Database
from backend.utils.file_ops import organise_file
from backend.utils.image_utils import (
    collect_images, extract_exif, file_hash, is_blurry,
    load_image_rgb, quality_score, save_thumbnail,
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    End-to-end gymnastics photo sorting pipeline.

    Designed to be instantiated once and reused across multiple jobs.
    All heavy models are lazily loaded on first use.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()

        # ── models (lazy) ──
        self._detector: Optional[PersonDetector] = None
        self._team_id: Optional[TeamIdentifier] = None
        self._person_id: Optional[PersonIdentifier] = None
        self._dup_detector = DuplicateDetector(self.cfg)

        # ── storage ──
        db_path = str(Path(self.cfg.data_dir) / "sorter.db")
        self.db = Database(db_path)
        self.emb_cache = EmbeddingsCache(
            str(Path(self.cfg.cache_dir) / "embeddings")
        )

        # ── clustering ──
        self.team_clusterer = TeamClusterer(self.cfg)
        self.person_clusterer = PersonClusterer(self.cfg)

        # ── state ──
        self._cancel_requested = False

    # ---------- lazy model accessors ----------

    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            self._detector = PersonDetector(self.cfg)
        return self._detector

    @property
    def team_identifier(self) -> TeamIdentifier:
        if self._team_id is None:
            self._team_id = TeamIdentifier(self.cfg)
        return self._team_id

    @property
    def person_identifier(self) -> PersonIdentifier:
        if self._person_id is None:
            self._person_id = PersonIdentifier(self.cfg)
        return self._person_id

    # ---------- main entry point ----------

    def process_job(
        self,
        job: ProcessingJob,
        progress: ProgressTracker,
        on_image_done: Optional[Callable[[ImageResult], None]] = None,
    ) -> None:
        """
        Full pipeline for a ProcessingJob.
        *on_image_done* is called after each image completes (for live UI updates).
        """
        self._cancel_requested = False
        try:
            self._run_pipeline(job, progress, on_image_done)
        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            progress.fail(str(exc))

    def cancel(self) -> None:
        """Request cancellation of the current job."""
        self._cancel_requested = True

    # ---------- internal pipeline ----------

    def _run_pipeline(
        self,
        job: ProcessingJob,
        progress: ProgressTracker,
        on_image_done: Optional[Callable],
    ) -> None:
        # ── 0. Collect images ──────────────────────────────────────────────
        progress.log(f"Collecting images from: {job.input_folder}")
        all_paths = collect_images(job.input_folder)
        if not all_paths:
            progress.log("No images found in input folder.")
            progress.finish()
            return

        # ── 1. Resume: skip already processed ─────────────────────────────
        already_done: set = set()
        if self.cfg.pipeline.resume_enabled:
            already_done = self.db.get_processed_paths()
            if already_done:
                progress.log(f"Resuming: {len(already_done)} images already processed.")

        pending = [p for p in all_paths if p not in already_done]
        job.total_images = len(all_paths)
        progress.start(len(pending))
        progress.log(f"Processing {len(pending)} images ({len(already_done)} skipped).")

        # ── 2. Per-image processing ────────────────────────────────────────
        results: List[ImageResult] = []
        batch_size = self.cfg.pipeline.batch_size

        # Load embeddings cache from disk (for resume continuity)
        self.emb_cache.load()

        for i, image_path in enumerate(pending):
            if self._cancel_requested:
                progress.log("Job cancelled by user.")
                progress.cancel()
                break

            result = self._process_single_image(image_path, job, progress)
            results.append(result)

            if on_image_done:
                try:
                    on_image_done(result)
                except Exception:
                    pass

            # Flush embeddings cache periodically to disk
            if (i + 1) % 500 == 0:
                self.emb_cache.save()
                progress.log(f"Checkpoint: {i + 1} images processed, cache saved.")

        # ── 3. Clustering ──────────────────────────────────────────────────
        if not self._cancel_requested:
            progress.log("Running team clustering…")
            self._run_team_clustering(job, progress)

            progress.log("Running individual gymnast clustering…")
            self._run_person_clustering(job, progress)

        # ── 4. Organise output files ───────────────────────────────────────
        if not self._cancel_requested:
            progress.log("Organising output files…")
            self._organise_output(job, progress)

        # ── 5. Finalise ────────────────────────────────────────────────────
        self.emb_cache.save()
        progress.log("All done!")
        progress.finish()
        logger.info("Pipeline completed for job %s", job.job_id)

    # ---------- single image ----------

    def _process_single_image(
        self,
        image_path: str,
        job: ProcessingJob,
        progress: ProgressTracker,
    ) -> ImageResult:
        t_start = time.time()
        result = ImageResult(image_path=image_path, status=ProcessingStatus.PROCESSING)

        try:
            # Load image
            img_rgb = load_image_rgb(image_path)
            if img_rgb is None:
                result.status = ProcessingStatus.FAILED
                result.error_message = "Could not load image"
                progress.update(image_path, success=False)
                return result

            h, w = img_rgb.shape[:2]
            result.width, result.height = w, h

            # File hash (for duplicate detection)
            result.file_hash = file_hash(image_path)

            # EXIF metadata
            exif = extract_exif(image_path)
            result.exif_datetime = exif.get("datetime")

            # Blur / quality
            blurry, blur_val = is_blurry(img_rgb)
            result.is_blurry = blurry
            result.blur_score = blur_val
            result.quality_score = quality_score(img_rgb)

            # Duplicate checks
            exact_dup = self._dup_detector.check_exact(image_path)
            if exact_dup:
                result.is_duplicate = True
                result.duplicate_of = exact_dup
                result.status = ProcessingStatus.SKIPPED
                progress.update(image_path, success=True)
                self._save_result(result)
                return result

            self._dup_detector.register_phash(image_path, img_rgb)
            self._dup_detector.register_for_burst(image_path, result.exif_datetime)

            # Generate thumbnail
            thumb_dir = Path(self.cfg.cache_dir) / "thumbnails"
            thumb_path = thumb_dir / (Path(image_path).stem + "_thumb.jpg")
            if not thumb_path.exists():
                save_thumbnail(img_rgb, str(thumb_path))

            # Person detection
            detections = self.detector.detect(img_rgb)
            result.detections = detections

            if not detections:
                # No person detected; put image in an "undetected" bucket
                result.output_team = "undetected"
                result.output_person = "undetected"
                result.status = ProcessingStatus.COMPLETED
                self._save_result(result)
                progress.update(image_path, success=True)
                return result

            # Feature extraction per detection
            all_team_feats: List[dict] = []
            all_person_feats: List[dict] = []

            for det_idx, det in enumerate(detections):
                crop = det.crop_normalized if det.crop_normalized is not None else det.crop
                if crop is None:
                    continue

                # Team features
                team_color_feats = self.team_identifier.extract_color_features(img_rgb, det)
                ocr_texts = self.team_identifier.extract_ocr_text(img_rgb, det)
                all_team_feats.append({
                    "histogram": team_color_feats["histogram"],
                    "dominant_colors": team_color_feats["dominant_colors"],
                    "ocr_texts": ocr_texts,
                })

                # Person identity features
                person_feats = self.person_identifier.extract_features(img_rgb, det)
                all_person_feats.append(person_feats)

                # Store embeddings in cache
                if person_feats["fused"] is not None:
                    self.emb_cache.add(
                        "person_fused",
                        image_path,
                        det_idx,
                        person_feats["fused"],
                    )
                if person_feats["reid"] is not None:
                    self.emb_cache.add(
                        "reid",
                        image_path,
                        det_idx,
                        person_feats["reid"],
                    )
                if person_feats.get("face") is not None:
                    self.emb_cache.add(
                        "face",
                        image_path,
                        det_idx,
                        person_feats["face"],
                    )

            # Compute CLIP team embeddings for the crops
            crops = [
                det.crop_normalized if det.crop_normalized is not None else det.crop
                for det in detections
                if det.crop is not None
            ]
            if crops:
                clip_embs = self.team_identifier.extract_clip_features(crops)
                for det_idx, clip_emb in enumerate(clip_embs):
                    self.emb_cache.add("team_clip", image_path, det_idx, clip_emb)
                    # Fuse colour hist + CLIP for team clustering
                    if det_idx < len(all_team_feats):
                        hist = all_team_feats[det_idx]["histogram"]
                        # Concatenate hist (96-d) + clip (768-d) and normalise
                        fused_team = np.concatenate([hist, clip_emb]).astype(np.float32)
                        norm = np.linalg.norm(fused_team)
                        if norm > 1e-6:
                            fused_team /= norm
                        self.emb_cache.add("team_fused", image_path, det_idx, fused_team)

            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - t_start
            self._save_result(result)
            progress.update(image_path, success=True)

        except Exception as exc:
            result.status = ProcessingStatus.FAILED
            result.error_message = str(exc)
            result.processing_time = time.time() - t_start
            logger.warning("Failed to process %s: %s", image_path, exc)
            logger.debug(traceback.format_exc())
            self._save_result(result)
            progress.update(image_path, success=False)

        return result

    # ---------- clustering passes ----------

    def _run_team_clustering(
        self, job: ProcessingJob, progress: ProgressTracker
    ) -> None:
        emb_matrix = self.emb_cache.get_matrix("team_fused")
        paths = self.emb_cache.get_paths("team_fused")
        det_indices = self.emb_cache.get_det_indices("team_fused")

        if emb_matrix is None or len(paths) == 0:
            progress.log("No team embeddings available; skipping team clustering.")
            return

        labels, team_clusters = self.team_clusterer.cluster(
            emb_matrix, paths, det_indices,
        )

        # Persist team clusters
        for tc in team_clusters:
            self.db.upsert_team({
                "team_id": tc.team_id,
                "team_name": tc.team_name,
                "image_count": tc.image_count,
                "confidence": tc.confidence,
                "dominant_colors": json.dumps(tc.dominant_colors),
                "representative_imgs": json.dumps(tc.representative_images),
            })

        # Update detection records with team assignment
        for i, (path, det_idx) in enumerate(zip(paths, det_indices)):
            if i < len(labels) and labels[i] >= 0 and labels[i] < len(team_clusters):
                tc = team_clusters[labels[i]]
                self.db.upsert_image({
                    "image_path": path,
                    "status": "completed",
                    "output_team": tc.team_name,
                })

        progress.log(f"Team clustering: {len(team_clusters)} teams identified.")

    def _run_person_clustering(
        self, job: ProcessingJob, progress: ProgressTracker
    ) -> None:
        teams = self.db.get_all_teams()
        if not teams:
            progress.log("No teams; running global person clustering.")
            self._cluster_persons_for_team("global", progress)
            return

        for team in teams:
            self._cluster_persons_for_team(team["team_id"], progress)

    def _cluster_persons_for_team(
        self, team_id: str, progress: ProgressTracker
    ) -> None:
        # Gather person embeddings for this team's images
        all_paths = self.emb_cache.get_paths("person_fused")
        all_det = self.emb_cache.get_det_indices("person_fused")
        emb_matrix = self.emb_cache.get_matrix("person_fused")

        if emb_matrix is None:
            return

        if team_id != "global":
            # Filter to this team's images
            team_images = {
                row["image_path"]
                for row in self.db.get_all_images()
                if row.get("output_team") == team_id
            }
            mask = np.array([p in team_images for p in all_paths])
            if not mask.any():
                return
            sub_paths = [all_paths[i] for i in np.where(mask)[0]]
            sub_det = [all_det[i] for i in np.where(mask)[0]]
            sub_embs = emb_matrix[mask]
        else:
            sub_paths, sub_det, sub_embs = all_paths, all_det, emb_matrix

        labels, person_clusters = self.person_clusterer.cluster(
            sub_embs, sub_paths, sub_det, team_id
        )

        # Persist person clusters
        for pc in person_clusters:
            self.db.upsert_person({
                "person_id": pc.person_id,
                "person_name": pc.person_name,
                "team_id": pc.team_id,
                "image_count": pc.image_count,
                "confidence": pc.confidence,
                "representative_imgs": json.dumps(pc.representative_images),
            })

        # Tag images with person assignment
        for i, (path, _) in enumerate(zip(sub_paths, sub_det)):
            if i < len(labels) and labels[i] >= 0 and labels[i] < len(person_clusters):
                pc = person_clusters[labels[i]]
                self.db.upsert_image({
                    "image_path": path,
                    "status": "completed",
                    "output_person": pc.person_name,
                })

        progress.log(
            f"Person clustering [{team_id}]: {len(person_clusters)} individuals."
        )

    # ---------- output organisation ----------

    def _organise_output(
        self, job: ProcessingJob, progress: ProgressTracker
    ) -> None:
        images = self.db.get_images_by_status("completed")
        mode_map = {"copy": FileMode.COPY, "move": FileMode.MOVE, "symlink": FileMode.SYMLINK}
        fmode = mode_map.get(job.file_mode, FileMode.COPY)

        organised = 0
        for img in images:
            if self._cancel_requested:
                break
            path = img["image_path"]
            team = img.get("output_team") or "unknown_team"
            person = img.get("output_person") or "unknown_person"
            if img.get("is_duplicate"):
                team = "_duplicates"
                person = "all"
            try:
                out_path = organise_file(
                    path, job.output_folder, team, person, fmode
                )
                self.db.upsert_image({
                    "image_path": path,
                    "status": "completed",
                    "output_path": out_path,
                })
                organised += 1
            except Exception as exc:
                logger.warning("Could not organise %s: %s", path, exc)

        progress.log(f"Organised {organised} images into {job.output_folder}.")

    # ---------- persistence helpers ----------

    def _save_result(self, result: ImageResult) -> None:
        try:
            row = {
                "image_path": result.image_path,
                "status": result.status.value,
                "width": result.width,
                "height": result.height,
                "file_hash": result.file_hash,
                "is_blurry": int(result.is_blurry),
                "blur_score": result.blur_score,
                "quality_score": result.quality_score,
                "is_duplicate": int(result.is_duplicate),
                "duplicate_of": result.duplicate_of,
                "is_burst": int(result.is_burst),
                "exif_datetime": result.exif_datetime,
                "output_team": result.output_team,
                "output_person": result.output_person,
                "output_path": result.output_path,
                "error_message": result.error_message,
                "processing_time": result.processing_time,
            }
            self.db.upsert_image(row)
        except Exception as exc:
            logger.debug("DB save error for %s: %s", result.image_path, exc)

    # ---------- active learning ----------

    def apply_correction(
        self,
        image_path: str,
        det_index: int,
        new_team_name: str,
        new_person_name: str,
    ) -> None:
        """
        Apply a user correction: update the DB record and request
        centroid recalculation on next clustering pass.
        """
        img = self.db.get_image(image_path)
        old_team = img.get("output_team", "") if img else ""
        old_person = img.get("output_person", "") if img else ""

        self.db.insert_correction({
            "image_path": image_path,
            "det_index": det_index,
            "old_team_id": old_team,
            "new_team_id": new_team_name,
            "old_person_id": old_person,
            "new_person_id": new_person_name,
        })
        self.db.upsert_image({
            "image_path": image_path,
            "status": "completed",
            "output_team": new_team_name,
            "output_person": new_person_name,
        })
        logger.info(
            "Correction applied: %s → team=%s person=%s",
            Path(image_path).name, new_team_name, new_person_name,
        )
