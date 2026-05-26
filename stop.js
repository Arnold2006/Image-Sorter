/**
 * stop.js  –  Pinokio 7.x stop script for Gymnastics Photo Sorter
 *
 * Terminates the running start.js process (and its underlying Python server).
 * Called from the "Stop" button in the Pinokio menu when start.js is running.
 */
module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "⏹ Stopping <b>Gymnastics Photo Sorter</b>…",
        icon: "fa-solid fa-stop",
      },
    },
    {
      method: "script.stop",
      params: { uri: "start.js" },
    },
  ],
};
