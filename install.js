/**
 * install.js  –  Pinokio 7.x installation script for Gymnastics Photo Sorter
 *
 * Steps performed:
 *  1. Create a Python 3.11 virtual-environment (.venv)
 *  2. Upgrade pip / setuptools / wheel
 *  3. Detect CUDA and install the appropriate PyTorch build
 *  4. Install all Python dependencies from requirements.txt
 *  5. Download AI model weights (YOLOv8-pose, InsightFace buffalo_l, etc.)
 *  6. Create required data / cache directories
 */
module.exports = {
  run: [
    // ── 1. Announce ─────────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "Installing <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-gears",
      },
    },

    // ── 2. Create virtual environment ───────────────────────────────────────
    {
      method: "shell.run",
      params: {
        message: "python -m venv .venv",
      },
    },

    // ── 3. Upgrade pip toolchain ─────────────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install --upgrade pip setuptools wheel",
      },
    },

    // ── 4. Detect CUDA and install PyTorch ──────────────────────────────────
    //      We run a tiny Python probe; if nvidia-smi succeeds we install the
    //      CUDA 12.1 index, otherwise we fall back to CPU-only.
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        // Probe: try importing torch with CUDA; echo a tag for the next step
        message:
          "python -c \"import subprocess,sys; r=subprocess.run(['nvidia-smi'],capture_output=True); print('CUDA_OK' if r.returncode==0 else 'CPU_ONLY')\"",
        on: [
          {
            event: "/CUDA_OK/",
            done: true,
            // Install CUDA 12.1 PyTorch
            run: {
              method: "shell.run",
              params: {
                venv: ".venv",
                message:
                  "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
              },
            },
          },
          {
            event: "/CPU_ONLY/",
            done: true,
            run: {
              method: "shell.run",
              params: {
                venv: ".venv",
                message: "pip install torch torchvision torchaudio",
              },
            },
          },
        ],
      },
    },

    // ── 5. Install all other Python dependencies ─────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install -r requirements.txt",
      },
    },

    // ── 6. Download AI model weights ─────────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "python scripts/download_models.py",
      },
    },

    // ── 7. Create runtime directories ────────────────────────────────────────
    {
      method: "shell.run",
      params: {
        message:
          "python -c \"import pathlib; [pathlib.Path(d).mkdir(parents=True,exist_ok=True) for d in ['cache/models','cache/thumbnails','cache/embeddings','data','logs']]\"",
      },
    },

    // ── 8. Done ──────────────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "✅ <b>Gymnastics Photo Sorter</b> installed successfully!<br>Click <b>Start</b> to launch the UI.",
        icon: "fa-solid fa-circle-check",
      },
    },
  ],
};
