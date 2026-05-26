/**
 * start.js  –  Pinokio launch script for Gymnastics Photo Sorter
 *
 * Activates the virtual environment and starts the Gradio UI (main.py).
 * When the action "stop" is passed, the running process is terminated instead.
 *
 * Conditional logic uses per-step "if" guards because Pinokio does not have
 * a "condition" RPC method.
 */
module.exports = {
  run: [
    // ── Handle stop action ───────────────────────────────────────────────────
    {
      method: "shell.stop",
      if: "{{params.action === 'stop'}}",
    },
    {
      method: "local.set",
      params: { key: "running", value: false },
      if: "{{params.action === 'stop'}}",
    },

    // ── Start action steps (skipped when stopping) ───────────────────────────

    // ── Announce start ────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "🚀 Starting <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-images",
      },
      if: "{{params.action !== 'stop'}}",
    },

    // ── Mark as running so the menu shows "Stop" ──────────────────────────
    {
      method: "local.set",
      params: { key: "running", value: true },
      if: "{{params.action !== 'stop'}}",
    },

    // ── Launch the app ────────────────────────────────────────────────────
    {
      method: "shell.start",
      params: {
        venv: ".venv",
        message: "python main.py",
        // Open the browser tab once the Gradio server is ready
        on: [
          {
            event: "/Running on local URL:\\s+(http[^\\s]+)/",
            done: false,
            run: {
              method: "browser.open",
              params: {
                // Capture the URL from the regex match group
                url: "{{event.matches[0][1]}}",
              },
            },
          },
          // Keep the shell alive until manually stopped
          {
            event: "/.*/",
            done: false,
          },
        ],
      },
      if: "{{params.action !== 'stop'}}",
    },

    // ── Shell has exited – clear the running flag ─────────────────────────
    {
      method: "local.set",
      params: { key: "running", value: false },
      if: "{{params.action !== 'stop'}}",
    },
  ],
};
