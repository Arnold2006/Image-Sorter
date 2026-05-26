# 🤸 Gymnastics Photo Sorter

AI-powered desktop application that **automatically sorts thousands of gymnastics competition photos** into folders organised by **team** and then by **individual gymnast** — fully offline, no cloud APIs required, NVIDIA CUDA accelerated.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
   - [Pinokio (recommended)](#pinokio-recommended)
   - [Manual Installation](#manual-installation)
4. [GPU / CUDA Setup](#gpu--cuda-setup)
5. [Usage Guide](#usage-guide)
6. [AI Model Explanations](#ai-model-explanations)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Project Structure](#project-structure)
10. [License](#license)

---

## Features

| Category | Capability |
|---|---|
| **Detection** | YOLOv8x-pose person detection — works upside-down, airborne, partial |
| **Team ID** | Colour analysis + CLIP + OCR + temporal grouping ensemble |
| **Person ID** | InsightFace + OSNet Re-ID + pose skeleton + hair descriptor |
| **Clustering** | HDBSCAN / DBSCAN with FAISS nearest-neighbour search |
| **Rotation** | Automatic upside-down / flip / handstand normalisation |
| **Duplicates** | MD5 + perceptual hash + embedding similarity + burst detection |
| **UI** | Gradio web UI — team browser, gymnast browser, live logs |
| **Corrections** | Merge / split / rename with active-learning centroid updates |
| **Output** | Copy / Move / Symlink mode, SQLite DB, JSON export |
| **Scale** | 10 000+ photos, batched GPU processing, resume mode |

---

## Architecture

```
Image-Sorter/
├── main.py                        # CLI / launch entry point
├── requirements.txt
├── pinokio.js                     # Pinokio app descriptor (dynamic menu)
├── install.js                     # Pinokio install script
├── start.js                       # Pinokio start script
├── stop.js                        # Pinokio stop script
├── update.js                      # Pinokio update script  ← one-click update
├── package.json
│
├── backend/
│   ├── models/
│   │   ├── person_detector.py     # YOLOv8-pose wrapper
│   │   ├── team_identifier.py     # CLIP + colour + OCR team features
│   │   ├── person_identifier.py   # Face + ReID + pose + hair fusion
│   │   └── duplicate_detector.py  # Multi-stage duplicate detection
│   │
│   ├── pipeline/
│   │   ├── processor.py           # End-to-end orchestration
│   │   ├── clustering.py          # HDBSCAN team + person clustering
│   │   └── job_queue.py           # Async job queue + progress tracker
│   │
│   ├── ui/
│   │   └── app.py                 # Gradio Blocks UI
│   │
│   ├── utils/
│   │   ├── config.py              # AppConfig dataclass + singleton
│   │   ├── database.py            # SQLite WAL database layer
│   │   ├── file_ops.py            # Copy / move / symlink organiser
│   │   ├── image_utils.py         # Load, hash, blur, quality, colour
│   │   └── logging_utils.py       # Coloured rotating-file logging
│   │
│   ├── cache/
│   │   └── embeddings_cache.py    # FAISS indices + pickle persistence
│   │
│   └── data/
│       └── schemas.py             # Dataclass schemas
│
├── scripts/
│   └── download_models.py         # Pre-download all AI weights
│
├── cache/                         # Auto-created at runtime
│   ├── models/                    # Downloaded model weights
│   ├── thumbnails/
│   └── embeddings/
│
├── data/                          # Auto-created at runtime
│   ├── sorter.db                  # SQLite database
│   └── config.json                # Saved settings
│
└── logs/
    └── image_sorter.log
```

### Data-flow

```
Input Folder
    │
    ▼
[collect_images] ──► [duplicate_detector] ──► skip duplicates
    │
    ▼
[load_image_rgb + EXIF + quality]
    │
    ▼
[PersonDetector / YOLOv8-pose] ──► bounding boxes + 17-pt pose
    │
    ├── [TeamIdentifier]
    │       ├── Torso colour histogram (HSV 96-d)
    │       ├── CLIP ViT-L/14 embedding (768-d)
    │       └── EasyOCR jersey text
    │
    └── [PersonIdentifier]
            ├── InsightFace 512-d face embedding
            ├── OSNet Re-ID 512-d body embedding
            ├── Pose skeleton limb-ratio vector (12-d)
            └── Hair colour histogram (48-d)
    │
    ▼
[EmbeddingsCache / FAISS] ── persist to SQLite
    │
    ▼
[TeamClusterer (HDBSCAN)] ──► team_001 … team_N
    │
    ▼
[PersonClusterer (HDBSCAN per team)] ──► gymnast_001 … gymnast_M
    │
    ▼
[organise_file (copy/move/symlink)]
    │
    ▼
OUTPUT/
  TEAM_NAME/
    GYMNAST_NAME/
      image001.jpg …
```

---

## Installation

### Pinokio (recommended)

1. Install **Pinokio 7.x** from [pinokio.computer](https://pinokio.computer).
2. In Pinokio, click **Discover** → paste the URL of this repository.
3. Click **Install** — Pinokio runs `install.js` which:
   - Creates `.venv` (Python 3.11 virtual environment)
   - Detects CUDA and installs the correct PyTorch build
   - Installs all dependencies from `requirements.txt`
   - Downloads all AI model weights
4. Click **Start** to launch the web UI (opens automatically in your browser).
5. **To update:** click **Update** — Pinokio runs `update.js` which:
   - Stops the app if running
   - Pulls the latest code (`git pull --rebase --autostash`)
   - Upgrades pip dependencies
   - Re-detects CUDA and upgrades PyTorch if needed
   - Refreshes any updated model weights

### Manual Installation

**Requirements:** Python 3.11+, Git

```bash
git clone https://github.com/Arnold2006/Image-Sorter.git
cd Image-Sorter

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# CUDA 12.1 GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision torchaudio

pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

Open your browser at **http://127.0.0.1:7860**

---

## GPU / CUDA Setup

| Scenario | Action |
|---|---|
| NVIDIA GPU, CUDA 12.x | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| NVIDIA GPU, CUDA 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| Apple Silicon (M1/M2) | `pip install torch torchvision` (MPS enabled automatically) |
| CPU only | `pip install torch torchvision` |

Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

For InsightFace GPU: `pip uninstall onnxruntime && pip install onnxruntime-gpu`

For FAISS GPU: `pip uninstall faiss-cpu && pip install faiss-gpu`

---

## Usage Guide

1. **Process** tab → set input/output folders → choose file mode → Start
2. **Teams** tab → review and rename auto-detected teams
3. **Gymnasts** tab → browse gymnasts per team
4. **Corrections** tab → merge/split/rename identities (active learning)
5. **Export** tab → JSON export and DB stats

---

## AI Model Explanations

| Model | Role |
|---|---|
| **YOLOv8x-pose** | Person detection + 17-keypoint skeleton; orientation normalisation |
| **CLIP ViT-L/14** | Semantic outfit similarity embeddings for team clustering |
| **InsightFace buffalo_l** | 512-d ArcFace face embeddings for individual ID |
| **OSNet (torchreid)** | 512-d body Re-ID embeddings robust to pose/viewpoint changes |
| **HDBSCAN** | Auto-K density clustering; handles noise/outliers naturally |
| **FAISS** | Sub-millisecond nearest-neighbour search over millions of vectors |
| **EasyOCR** | Jersey number / team name text detection |

---

## Performance Tuning

| Setting | Recommendation |
|---|---|
| Batch Size | 16 default; increase to 32+ if VRAM > 8 GB |
| YOLO Confidence | 0.35 default; lower to 0.25 for distant athletes |
| Clustering Method | HDBSCAN (auto-K); switch to dbscan if too many clusters |
| VRAM < 6 GB | Switch CLIP to ViT-B/32 in settings |
| 10 000+ photos | Resume mode on (default); checkpoint every 500 images |

---

## Troubleshooting

**`ModuleNotFoundError`** → activate `.venv` then `pip install -r requirements.txt`

**CUDA not detected** → reinstall PyTorch with correct CUDA index URL

**InsightFace download fails** → manually place `buffalo_l.zip` in `cache/models/`

**No persons detected** → lower YOLO confidence to 0.25 in Settings

**All photos in one team** → lower `team_min_cluster_size` or switch to `dbscan`

**Port already in use** → `python main.py --port 7861`

---

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | CLI entry point |
| `pinokio.js` | **Pinokio dynamic menu descriptor** |
| `install.js` | Pinokio install script |
| `start.js` | Pinokio launch script |
| `stop.js` | Pinokio stop script |
| `update.js` | **Pinokio one-click update script** |
| `backend/pipeline/processor.py` | Full pipeline orchestrator |
| `backend/pipeline/clustering.py` | HDBSCAN team + person clustering |
| `backend/models/person_detector.py` | YOLOv8-pose wrapper |
| `backend/models/team_identifier.py` | CLIP + colour + OCR team features |
| `backend/models/person_identifier.py` | Face + Re-ID + pose + hair |
| `backend/ui/app.py` | Gradio UI |
| `backend/utils/database.py` | SQLite persistence layer |
| `scripts/download_models.py` | Pre-download all AI weights |

---

## License

MIT License
