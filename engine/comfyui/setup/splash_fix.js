// ComfyUI extension: auto-dismiss stuck splash screen
// Place at: ComfyUI/web/extensions/splash_fix.js
// Fallback for when /api/userdata 404 prevents normal init completion.
(function () {
  const TIMEOUT_MS = 12000; // 12 seconds — generous enough for slow Drive loads
  setTimeout(() => {
    const splash = document.getElementById('splash-loader');
    if (splash) {
      console.log('[splash_fix] Force-dismissing stuck splash after timeout');
      splash.style.transition = 'opacity 0.5s';
      splash.style.opacity = '0';
      setTimeout(() => splash.remove(), 500);
    }
  }, TIMEOUT_MS);
})();
