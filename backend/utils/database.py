"""
SQLite database layer for the gymnastics photo sorter.

Tables:
  images        - one row per image, stores processing results
  detections    - person detections within images
  teams         - identified team clusters
  persons       - identified individual gymnast clusters
  corrections   - user corrections for active learning
  jobs          - processing job records
  embeddings    - cached embedding vectors (blob)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;   -- 64 MB page cache

CREATE TABLE IF NOT EXISTS images (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path          TEXT    NOT NULL UNIQUE,
    status              TEXT    NOT NULL DEFAULT 'pending',
    width               INTEGER,
    height              INTEGER,
    file_hash           TEXT,
    is_blurry           INTEGER DEFAULT 0,
    blur_score          REAL    DEFAULT 0,
    quality_score       REAL    DEFAULT 0,
    is_duplicate        INTEGER DEFAULT 0,
    duplicate_of        TEXT,
    is_burst            INTEGER DEFAULT 0,
    exif_datetime       TEXT,
    output_team         TEXT,
    output_person       TEXT,
    output_path         TEXT,
    error_message       TEXT,
    processing_time     REAL    DEFAULT 0,
    created_at          REAL    NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_images_status      ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_output_team ON images(output_team);
CREATE INDEX IF NOT EXISTS idx_images_hash        ON images(file_hash);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    det_index       INTEGER NOT NULL,
    bbox_x1         REAL,
    bbox_y1         REAL,
    bbox_x2         REAL,
    bbox_y2         REAL,
    bbox_conf       REAL,
    rotation_angle  REAL    DEFAULT 0,
    is_upside_down  INTEGER DEFAULT 0,
    team_id         TEXT,
    team_name       TEXT,
    team_confidence REAL    DEFAULT 0,
    person_id       TEXT,
    person_name     TEXT,
    person_confidence REAL  DEFAULT 0,
    detected_text   TEXT,   -- JSON array
    dominant_colors TEXT    -- JSON array of [r,g,b]
);
CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_detections_team  ON detections(team_id);
CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(person_id);

CREATE TABLE IF NOT EXISTS teams (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id             TEXT    NOT NULL UNIQUE,
    team_name           TEXT    NOT NULL,
    image_count         INTEGER DEFAULT 0,
    confidence          REAL    DEFAULT 0,
    dominant_colors     TEXT,   -- JSON
    representative_imgs TEXT,   -- JSON array of paths
    created_at          REAL    NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL    NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS persons (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id           TEXT    NOT NULL UNIQUE,
    person_name         TEXT    NOT NULL,
    team_id             TEXT,
    image_count         INTEGER DEFAULT 0,
    confidence          REAL    DEFAULT 0,
    representative_imgs TEXT,   -- JSON
    created_at          REAL    NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_persons_team ON persons(team_id);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path      TEXT    NOT NULL,
    det_index       INTEGER DEFAULT 0,
    old_team_id     TEXT,
    new_team_id     TEXT,
    old_person_id   TEXT,
    new_person_id   TEXT,
    applied         INTEGER DEFAULT 0,
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT    NOT NULL UNIQUE,
    input_folder    TEXT    NOT NULL,
    output_folder   TEXT    NOT NULL,
    file_mode       TEXT    DEFAULT 'copy',
    total_images    INTEGER DEFAULT 0,
    processed_images INTEGER DEFAULT 0,
    failed_images   INTEGER DEFAULT 0,
    status          TEXT    DEFAULT 'pending',
    config_snapshot TEXT,   -- JSON
    created_at      REAL    NOT NULL DEFAULT (unixepoch()),
    updated_at      REAL    NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path  TEXT    NOT NULL,
    det_index   INTEGER NOT NULL DEFAULT 0,
    emb_type    TEXT    NOT NULL,   -- 'clip', 'reid', 'face', 'pose'
    vector      BLOB    NOT NULL,   -- float32 bytes
    dim         INTEGER NOT NULL,
    created_at  REAL    NOT NULL DEFAULT (unixepoch()),
    UNIQUE(image_path, det_index, emb_type)
);
CREATE INDEX IF NOT EXISTS idx_emb_path ON embeddings(image_path);
CREATE INDEX IF NOT EXISTS idx_emb_type ON embeddings(emb_type);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Thread-safe SQLite wrapper using WAL mode for concurrent reads.
    Each method obtains a short-lived connection from a per-thread connection
    (SQLite connections are not thread-safe by default).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---------- Context manager ----------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection and commit/rollback automatically."""
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)
        logger.debug("Database schema initialised at %s", self.db_path)

    # ---------- Images ----------

    def upsert_image(self, result: Dict[str, Any]) -> int:
        """Insert or update an image record. Returns row id.

        Accepts partial dicts (e.g. only image_path + output_team).  Missing
        fields default to None for the INSERT, and the ON CONFLICT clause uses
        COALESCE so existing non-NULL values are never overwritten by NULL.
        """
        sql = """
            INSERT INTO images (image_path, status, width, height, file_hash,
                is_blurry, blur_score, quality_score, is_duplicate,
                duplicate_of, is_burst, exif_datetime, output_team,
                output_person, output_path, error_message, processing_time,
                updated_at)
            VALUES (:image_path, :status, :width, :height, :file_hash,
                :is_blurry, :blur_score, :quality_score, :is_duplicate,
                :duplicate_of, :is_burst, :exif_datetime, :output_team,
                :output_person, :output_path, :error_message, :processing_time,
                :updated_at)
            ON CONFLICT(image_path) DO UPDATE SET
                status=COALESCE(excluded.status, status),
                width=COALESCE(excluded.width, width),
                height=COALESCE(excluded.height, height),
                file_hash=COALESCE(excluded.file_hash, file_hash),
                is_blurry=COALESCE(excluded.is_blurry, is_blurry),
                blur_score=COALESCE(excluded.blur_score, blur_score),
                quality_score=COALESCE(excluded.quality_score, quality_score),
                is_duplicate=COALESCE(excluded.is_duplicate, is_duplicate),
                duplicate_of=COALESCE(excluded.duplicate_of, duplicate_of),
                is_burst=COALESCE(excluded.is_burst, is_burst),
                exif_datetime=COALESCE(excluded.exif_datetime, exif_datetime),
                output_team=COALESCE(excluded.output_team, output_team),
                output_person=COALESCE(excluded.output_person, output_person),
                output_path=COALESCE(excluded.output_path, output_path),
                error_message=COALESCE(excluded.error_message, error_message),
                processing_time=COALESCE(excluded.processing_time, processing_time),
                updated_at=excluded.updated_at
        """
        data = dict(result)
        # Ensure every named parameter in the SQL has a value (None is fine).
        for field in (
            "status", "width", "height", "file_hash", "is_blurry",
            "blur_score", "quality_score", "is_duplicate", "duplicate_of",
            "is_burst", "exif_datetime", "output_team", "output_person",
            "output_path", "error_message", "processing_time",
        ):
            data.setdefault(field, None)
        data.setdefault("updated_at", time.time())
        with self._conn() as conn:
            cur = conn.execute(sql, data)
            return cur.lastrowid or 0

    def get_image(self, image_path: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE image_path=?", (image_path,)
            ).fetchone()
        return dict(row) if row else None

    def get_images_by_status(self, status: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE status=? ORDER BY id", (status,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_images(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_processed_paths(self) -> set:
        """Return set of already-processed image paths (for resume)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT image_path FROM images WHERE status='completed'"
            ).fetchall()
        return {r["image_path"] for r in rows}

    def count_by_status(self) -> Dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM images GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ---------- Detections ----------

    def insert_detection(self, image_id: int, det: Dict[str, Any]) -> None:
        sql = """
            INSERT OR REPLACE INTO detections
            (image_id, det_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf,
             rotation_angle, is_upside_down, team_id, team_name, team_confidence,
             person_id, person_name, person_confidence, detected_text, dominant_colors)
            VALUES (:image_id,:det_index,:bbox_x1,:bbox_y1,:bbox_x2,:bbox_y2,
                    :bbox_conf,:rotation_angle,:is_upside_down,:team_id,:team_name,
                    :team_confidence,:person_id,:person_name,:person_confidence,
                    :detected_text,:dominant_colors)
        """
        with self._conn() as conn:
            conn.execute(sql, {"image_id": image_id, **det})

    def get_detections_for_image(self, image_id: int) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE image_id=? ORDER BY det_index",
                (image_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- Teams ----------

    def upsert_team(self, team: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO teams (team_id, team_name, image_count, confidence,
                dominant_colors, representative_imgs, updated_at)
            VALUES (:team_id,:team_name,:image_count,:confidence,
                    :dominant_colors,:representative_imgs,:updated_at)
            ON CONFLICT(team_id) DO UPDATE SET
                team_name=excluded.team_name,
                image_count=excluded.image_count,
                confidence=excluded.confidence,
                dominant_colors=excluded.dominant_colors,
                representative_imgs=excluded.representative_imgs,
                updated_at=excluded.updated_at
        """
        team.setdefault("updated_at", time.time())
        with self._conn() as conn:
            conn.execute(sql, team)

    def get_all_teams(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM teams ORDER BY team_name").fetchall()
        return [dict(r) for r in rows]

    def rename_team(self, team_id: str, new_name: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE teams SET team_name=?,updated_at=? WHERE team_id=?",
                (new_name, time.time(), team_id)
            )

    # ---------- Persons ----------

    def upsert_person(self, person: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO persons (person_id, person_name, team_id, image_count,
                confidence, representative_imgs, updated_at)
            VALUES (:person_id,:person_name,:team_id,:image_count,:confidence,
                    :representative_imgs,:updated_at)
            ON CONFLICT(person_id) DO UPDATE SET
                person_name=excluded.person_name,
                team_id=excluded.team_id,
                image_count=excluded.image_count,
                confidence=excluded.confidence,
                representative_imgs=excluded.representative_imgs,
                updated_at=excluded.updated_at
        """
        person.setdefault("updated_at", time.time())
        with self._conn() as conn:
            conn.execute(sql, person)

    def get_persons_for_team(self, team_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM persons WHERE team_id=? ORDER BY person_name",
                (team_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_persons(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM persons ORDER BY team_id, person_name").fetchall()
        return [dict(r) for r in rows]

    def rename_person(self, person_id: str, new_name: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE persons SET person_name=?,updated_at=? WHERE person_id=?",
                (new_name, time.time(), person_id)
            )

    def merge_persons(self, source_id: str, target_id: str) -> None:
        """Move all detections from source_id to target_id and delete source."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE detections SET person_id=?,person_name="
                "(SELECT person_name FROM persons WHERE person_id=?) "
                "WHERE person_id=?",
                (target_id, target_id, source_id)
            )
            conn.execute(
                "UPDATE images SET output_person="
                "(SELECT person_name FROM persons WHERE person_id=?) "
                "WHERE output_person="
                "(SELECT person_name FROM persons WHERE person_id=?)",
                (target_id, source_id)
            )
            conn.execute("DELETE FROM persons WHERE person_id=?", (source_id,))

    # ---------- Corrections ----------

    def insert_correction(self, corr: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO corrections
            (image_path, det_index, old_team_id, new_team_id,
             old_person_id, new_person_id)
            VALUES (:image_path,:det_index,:old_team_id,:new_team_id,
                    :old_person_id,:new_person_id)
        """
        with self._conn() as conn:
            conn.execute(sql, corr)

    def get_unapplied_corrections(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM corrections WHERE applied=0 ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_correction_applied(self, correction_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE corrections SET applied=1 WHERE id=?", (correction_id,)
            )

    # ---------- Jobs ----------

    def upsert_job(self, job: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO jobs (job_id, input_folder, output_folder, file_mode,
                total_images, processed_images, failed_images, status,
                config_snapshot, updated_at)
            VALUES (:job_id,:input_folder,:output_folder,:file_mode,
                    :total_images,:processed_images,:failed_images,:status,
                    :config_snapshot,:updated_at)
            ON CONFLICT(job_id) DO UPDATE SET
                total_images=excluded.total_images,
                processed_images=excluded.processed_images,
                failed_images=excluded.failed_images,
                status=excluded.status,
                updated_at=excluded.updated_at
        """
        job.setdefault("updated_at", time.time())
        with self._conn() as conn:
            conn.execute(sql, job)

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    # ---------- Embeddings ----------

    def store_embedding(
        self,
        image_path: str,
        det_index: int,
        emb_type: str,
        vector: np.ndarray,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embeddings
                   (image_path,det_index,emb_type,vector,dim)
                   VALUES (?,?,?,?,?)""",
                (image_path, det_index, emb_type,
                 vector.astype(np.float32).tobytes(), vector.shape[0])
            )

    def load_embedding(
        self,
        image_path: str,
        det_index: int,
        emb_type: str,
    ) -> Optional[np.ndarray]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vector,dim FROM embeddings "
                "WHERE image_path=? AND det_index=? AND emb_type=?",
                (image_path, det_index, emb_type)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim"])

    def load_all_embeddings(self, emb_type: str) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Load all embeddings of a given type.
        Returns (paths, det_indices, matrix) where matrix is (N, dim).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT image_path,det_index,vector,dim FROM embeddings "
                "WHERE emb_type=? ORDER BY id",
                (emb_type,)
            ).fetchall()
        if not rows:
            return [], [], np.empty((0,), dtype=np.float32)
        paths = [r["image_path"] for r in rows]
        indices = [r["det_index"] for r in rows]
        dim = rows[0]["dim"]
        matrix = np.stack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        ).reshape(-1, dim)
        return paths, indices, matrix

    # ---------- Export ----------

    def export_json(self, base_dir: str = "") -> str:
        """
        Export full dataset as JSON.

        The output file is always written inside *base_dir* (an application-
        controlled directory, typically config.data_dir) with an
        auto-generated timestamped filename.  No user input is used in
        constructing the output path, preventing path-injection risks.

        Returns the full path of the file written.
        """
        import os
        from datetime import datetime
        # base_dir must be a trusted, application-controlled directory
        if not base_dir:
            raise ValueError("base_dir must be provided to export_json.")
        # Resolve the trusted base directory
        safe_dir = os.path.realpath(os.path.abspath(base_dir))
        os.makedirs(safe_dir, exist_ok=True)
        # Build filename entirely from trusted data (timestamp, no user input)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"export_{timestamp}.json"
        full_path = os.path.join(safe_dir, safe_filename)
        data: Dict[str, Any] = {
            "teams": self.get_all_teams(),
            "persons": self.get_all_persons(),
            "images": self.get_all_images(),
        }
        with open(full_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Exported database to %s", full_path)
        return full_path

    def vacuum(self) -> None:
        """Reclaim disk space."""
        with self._conn() as conn:
            conn.execute("VACUUM")
