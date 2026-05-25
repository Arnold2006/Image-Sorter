/**
 * pinokio.js  –  Pinokio 7.x app descriptor for Gymnastics Photo Sorter
 *
 * This file tells Pinokio the app's identity, icon, available actions,
 * and which script to run for each action.
 */
module.exports = {
  title: "Gymnastics Photo Sorter",
  description:
    "AI-powered tool that automatically sorts thousands of gymnastics competition photos into folders by team and individual gymnast. Uses YOLOv8, CLIP, InsightFace, ReID and FAISS – fully offline, CUDA-accelerated.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    // Determine whether the app is currently running
    const running = info?.local?.running ?? false;

    return [
      // ── Primary action ──────────────────────────────────────────────────
      running
        ? {
            icon: "fa-solid fa-stop",
            text: "Stop",
            href: "start.js",
            params: { action: "stop" },
          }
        : {
            icon: "fa-solid fa-play",
            text: "Start",
            href: "start.js",
          },

      // ── Install / Update ─────────────────────────────────────────────────
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
    ];
  },
};
