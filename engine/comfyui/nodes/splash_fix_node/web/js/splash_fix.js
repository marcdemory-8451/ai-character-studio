// splash_fix: auto-dismiss ComfyUI splash when WebSocket can't connect
// (e.g. cloudflared QUIC tunnel, Playwright browser, slow Drive load)
// Loaded via ComfyUI's custom node JS mechanism (WEB_DIRECTORY in __init__.py).
import { app } from "../../scripts/app.js";

const TIMEOUT_MS = 15000; // 15 seconds — generous for slow Drive model loads

app.registerExtension({
  name: "SplashFix",
  async setup() {
    setTimeout(() => {
      const splash = document.getElementById("splash-loader");
      if (splash) {
        console.warn("[SplashFix] Force-dismissing stuck splash after timeout");
        splash.style.transition = "opacity 0.6s ease";
        splash.style.opacity = "0";
        splash.style.pointerEvents = "none";
        setTimeout(() => splash.remove(), 650);
      }
    }, TIMEOUT_MS);
  },
});
