/**
 * update.js  –  Pinokio 7.x one-click update script for Gymnastics Photo Sorter
 *
 * What this script does
 * ─────────────────────
 *  1.  Stop start.js if it is currently running (avoids file-lock issues on Windows).
 *  2.  Pull the latest code from the remote repository (git pull).
 *  3.  Upgrade pip / setuptools / wheel inside the existing virtual environment.
 *  4.  Re-install / upgrade all Python dependencies from requirements.txt.
 *  5.  Re-detect CUDA and upgrade PyTorch if the installed index has changed.
 *  6.  Re-run the model downloader to fetch any new or updated model weights.
 *  7.  Notify the user that the update is complete.
 *
 * No user interaction is required beyond clicking "Update" in the Pinokio menu.
 * The virtual environment (.venv) and all cached model weights are preserved
 * so that only the changed artefacts are re-downloaded.
 */
module.exports = {
  run: [
    // ── Step 0 : Stop the app if running ──────────────────────────────────────
    //
    //    script.stop terminates the running start.js process (and its shell)
    //    before we touch any files, preventing race conditions and "file in use"
    //    errors on Windows.  It is a no-op when start.js is not running.
    {
      method: "notify",
      params: {
        html: "⏹ Stopping any running instance before updating…",
        icon: "fa-solid fa-stop",
      },
    },
    {
      method: "script.stop",
      params: { uri: "start.js" },
    },

    // ── Step 1 : Announce update start ───────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "🔄 Updating <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-arrows-rotate",
      },
    },

    // ── Step 2 : Pull latest code from GitHub ─────────────────────────────────
    {
      method: "shell.run",
      params: {
        message: "git pull --rebase --autostash",
        on: [
          {
            event: "/error|conflict|fatal/i",
            done: true,
          },
        ],
      },
    },

    // ── Step 3 : Upgrade pip toolchain ───────────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install --upgrade pip setuptools wheel",
      },
    },

    // ── Step 4 : Upgrade Python dependencies ─────────────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install --upgrade -r requirements.txt",
      },
    },

    // ── Step 5 : Re-detect CUDA and upgrade PyTorch if needed ────────────────
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
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message:
          "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
      },
      if: "{{local.cuda === 'CUDA_OK'}}",
    },
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install --upgrade torch torchvision torchaudio",
      },
      if: "{{local.cuda === 'CPU_ONLY'}}",
    },

    // ── Step 6 : Download / refresh AI model weights ──────────────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "python scripts/download_models.py --check-updates",
      },
    },

    // ── Step 7 : Done ─────────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "✅ <b>Gymnastics Photo Sorter</b> is up to date!<br>Click <b>Start</b> to launch.",
        icon: "fa-solid fa-circle-check",
      },
    },
  ],
};
