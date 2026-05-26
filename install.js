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
        message: "python -m pip install --upgrade pip setuptools wheel",
      },
    },

    // ── 4. Detect CUDA ───────────────────────────────────────────────────────
    //      Runs a tiny Python probe that prints CUDA_OK or CPU_ONLY.
    //      The matched token is captured via the `on` event handler and stored
    //      in `local.cuda` so subsequent `if` guards can branch correctly.
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message:
          "python -c \"import subprocess; r=subprocess.run(['nvidia-smi'],capture_output=True); print('CUDA_OK' if r.returncode==0 else 'CPU_ONLY')\"",
        on: [{ event: "/CUDA_OK|CPU_ONLY/", done: true }],
      },
    },
    {
      method: "local.set",
      params: { cuda: "{{input.event[0]}}" },
    },

    // ── 5a. Install PyTorch (CUDA 12.1) ──────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message:
          "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
      },
      if: "{{local.cuda === 'CUDA_OK'}}",
    },

    // ── 5b. Install PyTorch (CPU-only fallback) ───────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install torch torchvision torchaudio",
      },
      if: "{{local.cuda === 'CPU_ONLY'}}",
    },

    // ── 6. Install all other Python dependencies ─────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install -r requirements.txt",
      },
    },

    // ── 7. Download AI model weights ─────────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "python scripts/download_models.py",
      },
    },

    // ── 8. Create runtime directories ────────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message:
          "python -c \"import pathlib; [pathlib.Path(d).mkdir(parents=True,exist_ok=True) for d in ['cache/models','cache/thumbnails','cache/embeddings','data','logs']]\"",
      },
    },

    // ── 9. Done ──────────────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "✅ <b>Gymnastics Photo Sorter</b> installed successfully!<br>Click <b>Start</b> to launch the UI.",
        icon: "fa-solid fa-circle-check",
      },
    },
  ],
};
