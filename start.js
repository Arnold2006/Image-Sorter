/**
 * start.js  –  Pinokio 7.x launch script for Gymnastics Photo Sorter
 *
 * Activates the virtual environment and starts the Gradio UI (main.py).
 * When the action "stop" is passed, the running process is terminated instead.
 */
module.exports = {
  run: [
    // ── Handle stop action ───────────────────────────────────────────────────
    {
      method: "local.get",
      params: { key: "running" },
    },
    {
      method: "condition",
      params: {
        // If params.action === "stop" OR the app is already running, stop it
        expression: "{{params.action === 'stop'}}",
        run: {
          method: "shell.stop",
        },
      },
    },

    // ── Announce start ───────────────────────────────────────────────────────
    {
      method: "notify",
      params: {
        html: "🚀 Starting <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-images",
      },
    },

    // ── Launch the app ───────────────────────────────────────────────────────
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
    },
  ],
};
