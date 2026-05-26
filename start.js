/**
 * start.js  –  Pinokio 7.x launch script for Gymnastics Photo Sorter
 *
 * Activates the virtual environment, starts the Gradio UI (main.py), and
 * automatically opens the browser tab once the server is ready.
 *
 * The `daemon: true` flag keeps this script (and the underlying server
 * process) alive after the final step finishes.  Pinokio tracks the running
 * state via `info.running("start.js")` so the menu in pinokio.js can show
 * the correct buttons without any manual `local.set` calls.
 */
module.exports = {
  daemon: true,
  run: [
    // ── Announce start ──────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "🚀 Starting <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-images",
      },
    },

    // ── Launch the app and wait until Gradio reports its URL ────────────────
    {
      method: "shell.run",
      params: {
        venv: ".venv",
        message: "python main.py",
        on: [
          {
            // Capture the URL from Gradio's startup message
            event: "/Running on local URL:\\s+(http[^\\s]+)/",
            done: true,
          },
        ],
      },
    },

    // ── Open the browser tab ────────────────────────────────────────────────
    {
      method: "browser.open",
      params: {
        uri: "http://127.0.0.1:7860",
      },
    },
  ],
};
