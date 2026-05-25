"""
File operations: copy, move, symlink with collision handling,
metadata preservation, and output folder tree creation.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from backend.data.schemas import FileMode

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Replace characters that are unsafe for Windows / macOS / Linux filenames."""
    UNSAFE = r'\/:*?"<>|'
    for ch in UNSAFE:
        name = name.replace(ch, "_")
    return name.strip(". ")[:128] or "unknown"


def build_output_path(
    output_root: str,
    team_name: str,
    person_name: str,
    original_filename: str,
) -> Path:
    """
    Build the full output path:
    <output_root>/<TEAM_NAME>/<PERSON_NAME>/<filename>
    """
    team_dir = safe_filename(team_name)
    person_dir = safe_filename(person_name)
    fname = Path(original_filename).name
    return Path(output_root) / team_dir / person_dir / fname


def resolve_collision(target: Path) -> Path:
    """
    If *target* already exists, append a numeric suffix until the path is free.
    e.g.  photo.jpg  →  photo_001.jpg  →  photo_002.jpg
    """
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def organise_file(
    source: str,
    output_root: str,
    team_name: str,
    person_name: str,
    file_mode: FileMode = FileMode.COPY,
) -> str:
    """
    Move/copy/symlink *source* into the organised output tree.
    Returns the actual destination path used.
    """
    target = build_output_path(output_root, team_name, person_name, source)
    target = resolve_collision(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if file_mode == FileMode.COPY:
            shutil.copy2(source, target)
        elif file_mode == FileMode.MOVE:
            shutil.move(source, target)
        elif file_mode == FileMode.SYMLINK:
            target.symlink_to(os.path.abspath(source))
        else:
            shutil.copy2(source, target)
        logger.debug("Organised %s  →  %s  [%s]", source, target, file_mode.value)
        return str(target)
    except Exception as exc:
        logger.error("Failed to organise %s: %s", source, exc)
        raise


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_files_in_folder(folder: str, extensions: Optional[set] = None) -> int:
    """Count files in *folder*, optionally filtered by extension set."""
    count = 0
    for _, _, files in os.walk(folder):
        for f in files:
            if extensions is None or Path(f).suffix.lower() in extensions:
                count += 1
    return count
