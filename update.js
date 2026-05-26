/**
 * update.js  –  Pinokio 7.x one-click update script for Gymnastics Photo Sorter
 *
 * What this script does
 * ─────────────────────
 *  1.  Stop the app if it is currently running (avoids file-lock issues on Windows).
 *  2.  Pull the latest code from the remote repository (git pull).
 *  3.  Upgrade pip / setuptools / wheel inside the existing virtual environment.
 *  4.  Re-install / upgrade all Python dependencies from requirements.txt
 *      so any newly added packages are picked up automatically.
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
    //    Pinokio keeps a `local.running` flag while start.js is active.
    //    We stop the shell before touching files to prevent race conditions
    //    and "file in use" errors on Windows.
    //    Per-step `if` guards are used because Pinokio 7.x does not have a
    //    `condition` RPC method.
    {
      method: "local.get",
      params: { key: "running" },
    },
    {
      method: "notify",
      params: {
        html: "⏹ Stopping the running app before updating…",
        icon: "fa-solid fa-stop",
      },
      if: "{{local.running === true}}",
    },
    {
      method: "shell.stop",
      if: "{{local.running === true}}",
    },
    // Small pause so the process fully releases its handles
    {
      method: "wait",
      params: { seconds: 2 },
      if: "{{local.running === true}}",
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
    //
    //    `git pull --rebase --autostash` is used so that any uncommitted local
    //    tweaks (e.g. a custom config.json accidentally left in the working tree)
    //    are temporarily stashed, the remote changes applied, and then the local
    //    changes re-applied on top – minimising merge conflicts.
    {
      method: "shell.run",
      params: {
        message: "git pull --rebase --autostash",
        on: [
          {
            // Detect a non-zero exit (e.g. conflicts) and surface the error
            event: "/error|conflict|fatal/i",
            done: true,
            run: {
              method: "notify",
              params: {
                html: "⚠️ <b>git pull</b> encountered a problem.<br>Please resolve conflicts manually and re-run Update.",
                icon: "fa-solid fa-triangle-exclamation",
              },
            },
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
    //
    //    `--upgrade` ensures that newly pinned or loosened version constraints
    //    in requirements.txt are honoured.  Packages already at the correct
    //    version are skipped by pip automatically.
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "pip install --upgrade -r requirements.txt",
      },
    },

    // ── Step 5 : Re-detect CUDA and upgrade PyTorch if needed ────────────────
    //
    //    We probe for nvidia-smi so that users who have since installed or
    //    upgraded their NVIDIA drivers automatically receive the correct build.
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message:
          "python -c \"import subprocess; r=subprocess.run(['nvidia-smi'],capture_output=True); print('CUDA_OK' if r.returncode==0 else 'CPU_ONLY')\"",
        on: [
          {
            event: "/CUDA_OK/",
            done: true,
            run: {
              method: "shell.run",
              params: {
                venv: ".venv",
                message:
                  "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
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
                message:
                  "pip install --upgrade torch torchvision torchaudio",
              },
            },
          },
        ],
      },
    },

    // ── Step 6 : Download / refresh AI model weights ──────────────────────────
    //
    //    The model downloader script (scripts/download_models.py) is idempotent:
    //    it skips files that are already present and have the correct checksum,
    //    so only genuinely new or updated weights are re-downloaded.
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
