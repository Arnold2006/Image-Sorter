/**
 * pinokio.js  –  Pinokio 7.x app descriptor for Gymnastics Photo Sorter
 *
 * Dynamic menu that reflects the current app state:
 *   • Not installed     → Install
 *   • Installing        → Installing… (spinner)
 *   • Installed, idle   → Start | Install | Update | Terminal
 *   • Running           → Stop | Terminal
 *   • Updating          → Updating… (spinner)
 */
module.exports = {
  version: "4.0",
  title: "Gymnastics Photo Sorter",
  description:
    "AI-powered tool that automatically sorts thousands of gymnastics competition photos into folders by team and individual gymnast. Uses YOLOv8, CLIP, InsightFace, ReID and FAISS – fully offline, CUDA-accelerated.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installing = info.running("install.js");
    const updating   = info.running("update.js");
    const running    = info.running("start.js");
    const installed  = info.exists(".venv");

    // ── App is currently being installed ────────────────────────────────
    if (installing) {
      return [{
        default: true,
        icon: "fa-solid fa-spinner fa-spin",
        text: "Installing…",
        href: "install.js",
      }];
    }

    // ── App is currently being updated ──────────────────────────────────
    if (updating) {
      return [{
        default: true,
        icon: "fa-solid fa-spinner fa-spin",
        text: "Updating…",
        href: "update.js",
      }];
    }

    // ── Not installed yet ────────────────────────────────────────────────
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Install",
        href: "install.js",
      }];
    }

    // ── App is running ───────────────────────────────────────────────────
    if (running) {
      return [
        {
          default: true,
          icon: "fa-solid fa-stop",
          text: "Stop",
          href: "stop.js",
        },
        {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          shell: { input: true, venv: ".venv" },
        },
      ];
    }

    // ── Installed, not running ───────────────────────────────────────────
    return [
      {
        default: true,
        icon: "fa-solid fa-play",
        text: "Start",
        href: "start.js",
      },
      {
        icon: "fa-solid fa-download",
        text: "Install",
        href: "install.js",
      },
      {
        icon: "fa-solid fa-arrows-rotate",
        text: "Update",
        href: "update.js",
      },
      {
        icon: "fa-solid fa-terminal",
        text: "Terminal",
        shell: { input: true, venv: ".venv" },
      },
    ];
  },
};
