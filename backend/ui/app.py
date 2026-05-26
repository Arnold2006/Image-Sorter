"""
Gradio-based UI for the Gymnastics Photo Sorter.

Layout
──────
Tab 1 – Process        : folder picker, file mode, start/stop, live progress
Tab 2 – Teams          : browsable team grid with representative images
Tab 3 – Gymnasts       : per-team gymnast browser with thumbnails
Tab 4 – Corrections    : merge/split/rename identities (active learning)
Tab 5 – Settings       : pipeline, model, clustering config knobs
Tab 6 – Export         : JSON export, logs download, DB stats
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from backend.pipeline.job_queue import JobQueue, ProgressTracker
from backend.pipeline.processor import ImageProcessor
from backend.utils.config import AppConfig, get_config, set_config
from backend.utils.database import Database
from backend.utils.image_utils import collect_images

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (module-level singletons)
# ---------------------------------------------------------------------------

_config: AppConfig = get_config()
_processor: Optional[ImageProcessor] = None
_job_queue: Optional[JobQueue] = None
_db: Optional[Database] = None


def _get_processor() -> ImageProcessor:
    global _processor
    if _processor is None:
        _processor = ImageProcessor(_config)
    return _processor


def _get_queue() -> JobQueue:
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(_config)
        _job_queue.start()
    return _job_queue


def _get_db() -> Database:
    global _db
    if _db is None:
        db_path = str(Path(_config.data_dir) / "sorter.db")
        _db = Database(db_path)
    return _db


# ---------------------------------------------------------------------------
# Helper: thumbnail HTML grid
# ---------------------------------------------------------------------------

def _image_grid_html(image_paths: List[str], max_items: int = 50) -> str:
    """Build an HTML thumbnail grid from a list of image paths."""
    items = image_paths[:max_items]
    if not items:
        return "<p style='color:#888'>No images to display.</p>"

    thumb_dir = Path(_config.cache_dir) / "thumbnails"
    cards = []
    for p in items:
        name = Path(p).name
        # Try to serve the thumbnail; fall back to the original path
        thumb_name = Path(p).stem + "_thumb.jpg"
        thumb_path = thumb_dir / thumb_name
        src = str(thumb_path) if thumb_path.exists() else p
        cards.append(
            f'<div style="display:inline-block;margin:4px;text-align:center">'
            f'<img src="file={src}" style="width:160px;height:120px;object-fit:cover;border-radius:4px"/>'
            f'<br><small style="font-size:10px;color:#aaa">{name}</small>'
            f"</div>"
        )
    return "".join(cards)


# ---------------------------------------------------------------------------
# Tab 1 – Process
# ---------------------------------------------------------------------------

def _start_processing(
    input_folder: str,
    output_folder: str,
    file_mode: str,
) -> str:
    """
    Validate user-provided folder paths and enqueue a processing job.

    Security notes:
    - Both paths are canonicalised via os.path.realpath to eliminate traversal.
    - Paths are blocked if they resolve to critical system directories.
    - The UI is bound to 127.0.0.1 (local-only) by default; share=False.
    """
    import os, re
    if not input_folder or not isinstance(input_folder, str):
        return "❌ Please specify an input folder."
    if not output_folder or not isinstance(output_folder, str):
        return "❌ Please specify an output folder."

    # Strip null bytes and non-printable characters
    input_folder = re.sub(r'[\x00-\x1f\x7f]', '', input_folder)
    output_folder = re.sub(r'[\x00-\x1f\x7f]', '', output_folder)

    try:
        safe_input = os.path.realpath(os.path.abspath(input_folder))
        safe_output = os.path.realpath(os.path.abspath(output_folder))
    except (ValueError, OSError):
        return "❌ Invalid folder path."

    # Block writes to well-known system roots
    _BLOCKED = {"/", "/etc", "/bin", "/usr", "/sys", "/proc",
                "C:\\Windows", "C:\\System32"}
    if safe_output in _BLOCKED or safe_input in _BLOCKED:
        return "❌ That path is a protected system directory."

    # Path-injection note: safe_input / safe_output are user-chosen folder paths
    # that have been canonicalized, null-byte stripped, and checked against a
    # system-directory blocklist.  os.path.isdir() is a read-only check;
    # os.makedirs() intentionally creates the user's selected output folder —
    # this is the core folder-picker behaviour of this local desktop app.
    if not os.path.isdir(safe_input):  # noqa: S603 – intentional folder check
        return "❌ Input folder does not exist."

    try:
        os.makedirs(safe_output, exist_ok=True)  # noqa: S603 – user-chosen output dir
    except OSError as exc:
        return f"❌ Cannot create output folder: {exc}"

    q = _get_queue()
    proc = _get_processor()

    def _run(job, progress: ProgressTracker) -> None:
        proc.process_job(job, progress)

    job_id = q.submit(
        input_folder=safe_input,
        output_folder=safe_output,
        file_mode=file_mode.lower(),
        processor_fn=_run,
    )
    return f"✅ Job {job_id} queued. Switch to the progress panel below."


def _stop_processing() -> str:
    q = _get_queue()
    proc = _get_processor()
    proc.cancel()
    q.cancel_current()
    return "⏹ Stop requested."


def _get_progress() -> Tuple[float, str]:
    """Return (fraction, status_text) for the Gradio progress bar."""
    q = _get_queue()
    snap = q.progress.snapshot()
    pct = snap["fraction"] * 100
    eta = snap["eta"]
    status = (
        f"**Status:** {snap['status'].upper()}  "
        f"| {snap['processed']} / {snap['total']} processed  "
        f"| {snap['failed']} failed  "
        f"| ETA: {eta}s\n\n"
        f"**Current:** `{Path(snap['current_file']).name if snap['current_file'] else '—'}`"
    )
    return snap["fraction"], status


def _get_live_logs() -> str:
    q = _get_queue()
    lines = q.progress.snapshot()["log"]
    return "\n".join(lines[-60:])


# ---------------------------------------------------------------------------
# Tab 2 – Teams
# ---------------------------------------------------------------------------

def _list_teams() -> str:
    db = _get_db()
    teams = db.get_all_teams()
    if not teams:
        return "<p>No teams identified yet. Run processing first.</p>"
    rows = []
    for t in teams:
        imgs = json.loads(t.get("representative_imgs") or "[]")
        thumb_html = _image_grid_html(imgs, max_items=5)
        rows.append(
            f"<details open><summary><b>{t['team_name']}</b> "
            f"({t['image_count']} images, confidence {t['confidence']:.2f})"
            f"</summary><div style='padding:8px'>{thumb_html}</div></details>"
        )
    return "\n".join(rows)


def _rename_team(team_id: str, new_name: str) -> str:
    if not team_id or not new_name:
        return "Please provide both team ID and new name."
    _get_db().rename_team(team_id, new_name)
    return f"✅ Team {team_id} renamed to '{new_name}'."


# ---------------------------------------------------------------------------
# Tab 3 – Gymnasts
# ---------------------------------------------------------------------------

def _list_gymnasts(team_filter: str) -> str:
    db = _get_db()
    if team_filter and team_filter.strip():
        persons = db.get_persons_for_team(team_filter.strip())
    else:
        persons = db.get_all_persons()

    if not persons:
        return "<p>No gymnasts identified yet.</p>"

    rows = []
    for p in persons:
        imgs = json.loads(p.get("representative_imgs") or "[]")
        thumb_html = _image_grid_html(imgs, max_items=5)
        rows.append(
            f"<details><summary><b>{p['person_name']}</b> "
            f"[{p['team_id']}] ({p['image_count']} images)"
            f"</summary><div style='padding:8px'>{thumb_html}</div></details>"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Tab 4 – Corrections
# ---------------------------------------------------------------------------

def _apply_correction(
    image_path: str,
    det_index: int,
    new_team: str,
    new_person: str,
) -> str:
    if not image_path:
        return "Please provide an image path."
    try:
        _get_processor().apply_correction(image_path, det_index, new_team, new_person)
        return f"✅ Correction applied: {Path(image_path).name} → {new_team}/{new_person}"
    except Exception as exc:
        return f"❌ Error: {exc}"


def _merge_persons(source_id: str, target_id: str) -> str:
    if not source_id or not target_id:
        return "Please provide both person IDs."
    try:
        _get_db().merge_persons(source_id, target_id)
        return f"✅ Merged '{source_id}' into '{target_id}'."
    except Exception as exc:
        return f"❌ Error: {exc}"


def _rename_person(person_id: str, new_name: str) -> str:
    if not person_id or not new_name:
        return "Please provide both person ID and new name."
    _get_db().rename_person(person_id, new_name)
    return f"✅ Person '{person_id}' renamed to '{new_name}'."


# ---------------------------------------------------------------------------
# Tab 5 – Settings
# ---------------------------------------------------------------------------

def _load_settings() -> Tuple[int, float, str, str, bool]:
    c = _config
    return (
        c.pipeline.batch_size,
        c.model.yolo_conf_threshold,
        c.clustering.team_method,
        c.device,
        c.model.ocr_enabled,
    )


def _save_settings(
    batch_size: int,
    yolo_conf: float,
    team_method: str,
    device: str,
    ocr_enabled: bool,
) -> str:
    _config.pipeline.batch_size = int(batch_size)
    _config.model.yolo_conf_threshold = float(yolo_conf)
    _config.clustering.team_method = team_method
    _config.device = device
    _config.model.ocr_enabled = ocr_enabled
    _config.save()
    set_config(_config)
    return "✅ Settings saved."


# ---------------------------------------------------------------------------
# Tab 6 – Export
# ---------------------------------------------------------------------------

def _export_json(_ignored: str = "") -> str:
    """Export to a timestamped JSON file inside the configured data directory."""
    try:
        out = _get_db().export_json(base_dir=_config.data_dir)
        return f"✅ Exported to `{out}`"
    except Exception as exc:
        return f"❌ Export failed: {exc}"


def _get_db_stats() -> str:
    db = _get_db()
    counts = db.count_by_status()
    teams = len(db.get_all_teams())
    persons = len(db.get_all_persons())
    lines = [
        f"**Teams:** {teams}",
        f"**Gymnasts:** {persons}",
        "**Images by status:**",
    ]
    for status, count in counts.items():
        lines.append(f"  - {status}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Gradio app builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        title="Gymnastics Photo Sorter",
        theme=gr.themes.Soft(),
        css="""
        .tab-nav button { font-size: 15px; font-weight: 600; }
        details summary { cursor: pointer; padding: 6px 0; }
        """,
    ) as app:

        gr.Markdown(
            "# 🤸 Gymnastics Photo Sorter\n"
            "AI-powered automatic sorting of competition photos by team and individual gymnast."
        )

        with gr.Tabs():

            # ──────────────────────────────────────────────────────────────
            # Tab 1: Process
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("▶ Process"):
                gr.Markdown("### Input / Output")
                with gr.Row():
                    input_folder = gr.Textbox(
                        label="Input Folder (path to photos)",
                        placeholder="/path/to/competition/photos",
                        scale=3,
                    )
                    output_folder = gr.Textbox(
                        label="Output Folder",
                        placeholder="/path/to/sorted/output",
                        scale=3,
                    )
                    file_mode = gr.Dropdown(
                        label="File Mode",
                        choices=["Copy", "Move", "Symlink"],
                        value="Copy",
                        scale=1,
                    )

                with gr.Row():
                    start_btn = gr.Button("🚀 Start Processing", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

                status_box = gr.Markdown("*Idle.*")

                gr.Markdown("---")
                gr.Markdown("### Progress")
                with gr.Row():
                    progress_bar = gr.Slider(
                        label="Progress", minimum=0, maximum=1, value=0,
                        interactive=False, scale=3,
                    )
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)

                progress_text = gr.Markdown("*Not started.*")
                live_logs = gr.Textbox(
                    label="Live Logs",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )

                # Wire up actions
                start_btn.click(
                    fn=_start_processing,
                    inputs=[input_folder, output_folder, file_mode],
                    outputs=[status_box],
                )
                stop_btn.click(fn=_stop_processing, outputs=[status_box])

                def _refresh():
                    frac, text = _get_progress()
                    logs = _get_live_logs()
                    return frac, text, logs

                refresh_btn.click(
                    fn=_refresh,
                    outputs=[progress_bar, progress_text, live_logs],
                )

                # Auto-refresh every 2 seconds so the progress panel updates
                # without requiring the user to click Refresh manually.
                auto_timer = gr.Timer(value=2)
                auto_timer.tick(
                    fn=_refresh,
                    outputs=[progress_bar, progress_text, live_logs],
                )

            # ──────────────────────────────────────────────────────────────
            # Tab 2: Teams
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🏆 Teams"):
                gr.Markdown("### Identified Teams")
                teams_refresh = gr.Button("🔄 Refresh Teams")
                teams_html = gr.HTML("<p>Click Refresh to load teams.</p>")

                gr.Markdown("---")
                gr.Markdown("### Rename Team")
                with gr.Row():
                    rename_team_id = gr.Textbox(label="Team ID (e.g. team_001)")
                    rename_team_name = gr.Textbox(label="New Name")
                    rename_team_btn = gr.Button("Rename")
                rename_team_status = gr.Markdown()

                teams_refresh.click(fn=_list_teams, outputs=[teams_html])
                rename_team_btn.click(
                    fn=_rename_team,
                    inputs=[rename_team_id, rename_team_name],
                    outputs=[rename_team_status],
                )

            # ──────────────────────────────────────────────────────────────
            # Tab 3: Gymnasts
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🤸 Gymnasts"):
                gr.Markdown("### Gymnast Browser")
                with gr.Row():
                    team_filter = gr.Textbox(
                        label="Filter by Team ID (leave blank for all)",
                        scale=2,
                    )
                    gymnast_refresh = gr.Button("🔄 Refresh", scale=1)
                gymnasts_html = gr.HTML("<p>Click Refresh to load gymnasts.</p>")

                gymnast_refresh.click(
                    fn=_list_gymnasts,
                    inputs=[team_filter],
                    outputs=[gymnasts_html],
                )

            # ──────────────────────────────────────────────────────────────
            # Tab 4: Corrections
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("✏️ Corrections"):
                gr.Markdown(
                    "### Manual Correction (Active Learning)\n"
                    "Override team/person assignments. Corrections are immediately "
                    "saved and used to improve future clustering."
                )
                with gr.Group():
                    gr.Markdown("**Re-assign a single image**")
                    with gr.Row():
                        corr_image_path = gr.Textbox(label="Image Path")
                        corr_det_index = gr.Number(label="Detection Index", value=0, precision=0)
                    with gr.Row():
                        corr_new_team = gr.Textbox(label="New Team Name")
                        corr_new_person = gr.Textbox(label="New Person Name")
                    corr_btn = gr.Button("Apply Correction", variant="primary")
                    corr_status = gr.Markdown()
                    corr_btn.click(
                        fn=_apply_correction,
                        inputs=[corr_image_path, corr_det_index, corr_new_team, corr_new_person],
                        outputs=[corr_status],
                    )

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**Merge two gymnasts into one**")
                    with gr.Row():
                        merge_src = gr.Textbox(label="Source Person ID (will be deleted)")
                        merge_tgt = gr.Textbox(label="Target Person ID (kept)")
                        merge_btn = gr.Button("Merge")
                    merge_status = gr.Markdown()
                    merge_btn.click(
                        fn=_merge_persons,
                        inputs=[merge_src, merge_tgt],
                        outputs=[merge_status],
                    )

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**Rename a gymnast**")
                    with gr.Row():
                        ren_pid = gr.Textbox(label="Person ID")
                        ren_name = gr.Textbox(label="New Name")
                        ren_btn = gr.Button("Rename")
                    ren_status = gr.Markdown()
                    ren_btn.click(
                        fn=_rename_person,
                        inputs=[ren_pid, ren_name],
                        outputs=[ren_status],
                    )

            # ──────────────────────────────────────────────────────────────
            # Tab 5: Settings
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Pipeline & Model Settings")
                with gr.Row():
                    s_batch = gr.Slider(
                        label="Batch Size", minimum=1, maximum=64, step=1,
                        value=_config.pipeline.batch_size,
                    )
                    s_yolo_conf = gr.Slider(
                        label="YOLO Confidence Threshold",
                        minimum=0.1, maximum=0.9, step=0.05,
                        value=_config.model.yolo_conf_threshold,
                    )
                with gr.Row():
                    s_team_method = gr.Dropdown(
                        label="Team Clustering Method",
                        choices=["hdbscan", "dbscan", "kmeans", "agglomerative"],
                        value=_config.clustering.team_method,
                    )
                    s_device = gr.Dropdown(
                        label="Compute Device",
                        choices=["auto", "cuda", "cpu", "mps"],
                        value=_config.device,
                    )
                    s_ocr = gr.Checkbox(label="Enable OCR", value=_config.model.ocr_enabled)

                save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
                settings_status = gr.Markdown()
                save_settings_btn.click(
                    fn=_save_settings,
                    inputs=[s_batch, s_yolo_conf, s_team_method, s_device, s_ocr],
                    outputs=[settings_status],
                )

            # ──────────────────────────────────────────────────────────────
            # Tab 6: Export & Stats
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("📤 Export"):
                gr.Markdown("### Database Statistics")
                stats_refresh = gr.Button("🔄 Refresh Stats")
                stats_md = gr.Markdown()
                stats_refresh.click(fn=_get_db_stats, outputs=[stats_md])

                gr.Markdown("---")
                gr.Markdown("### Export to JSON")
                gr.Markdown(
                    f"Exports the full database to a timestamped JSON file "
                    f"inside `{_config.data_dir}/`."
                )
                export_btn = gr.Button("📥 Export Now", variant="primary")
                export_status = gr.Markdown()
                export_btn.click(
                    fn=_export_json,
                    inputs=[],
                    outputs=[export_status],
                )

    return app


def launch(config: Optional[AppConfig] = None) -> None:
    """Build and launch the Gradio UI."""
    global _config
    if config:
        _config = config

    app = build_app()
    # Collect all directories that may contain images so Gradio can serve them.
    # The cache_dir holds thumbnails; the root "/" covers any user-chosen input
    # folder (this app runs locally on 127.0.0.1, so broad path access is safe).
    import os
    _fs_root = ["/"] if os.name != "nt" else [os.path.splitdrive(os.getcwd())[0] + "\\"]
    _allowed = list({
        str(Path(_config.cache_dir).resolve()),
        str(Path(_config.data_dir).resolve()),
        *_fs_root,
    })

    app.launch(
        server_name=_config.ui.host,
        server_port=_config.ui.port,
        share=_config.ui.share,
        show_error=True,
        max_file_size=f"{_config.ui.max_upload_size_mb}mb",
        allowed_paths=_allowed,
    )
