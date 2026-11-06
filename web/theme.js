/* Runs before the stylesheets to restore the theme without a light-mode flash. */
(() => {
  "use strict";

  const storageKey = "rs-theme";
  const root = document.documentElement;
  const validTheme = value => value === "light" || value === "dark";
  let preference = null;
  let systemTheme = null;
  let toggle = null;

  try {
    const saved = localStorage.getItem(storageKey);
    if (validTheme(saved)) preference = saved;
  } catch { /* Storage can be unavailable in private or embedded browser contexts. */ }

  try { systemTheme = window.matchMedia("(prefers-color-scheme: dark)"); }
  catch { /* A light theme remains usable in older embedded browsers. */ }

  const applyTheme = () => {
    const theme = preference || (systemTheme?.matches ? "dark" : "light");
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    if (!toggle) return;
    const action = theme === "dark" ? "切换到亮色模式" : "切换到暗色模式";
    toggle.title = action;
    toggle.setAttribute("aria-label", action);
    toggle.setAttribute("aria-pressed", String(theme === "dark"));
    const label = toggle.querySelector(".theme-label");
    if (label) label.textContent = theme === "dark" ? "暗色" : "亮色";
  };

  applyTheme();

  const onSystemChange = () => { if (!preference) applyTheme(); };
  if (systemTheme?.addEventListener) systemTheme.addEventListener("change", onSystemChange);
  else if (systemTheme?.addListener) systemTheme.addListener(onSystemChange);

  // A preference change in another tab also updates an already-open workspace.
  window.addEventListener("storage", event => {
    if (event.key !== storageKey && event.key !== null) return;
    // sessionStorage also dispatches storage events in embedded same-origin frames.
    try { if (event.storageArea && event.storageArea !== localStorage) return; }
    catch { /* The event value still works if access to storage has been blocked. */ }
    preference = validTheme(event.newValue) ? event.newValue : null;
    applyTheme();
  });

  const bindToggle = () => {
    toggle = document.getElementById("theme-toggle");
    if (!toggle) return;
    toggle.addEventListener("click", () => {
      preference = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem(storageKey, preference); }
      catch { /* Retain the explicit choice for this window even if saving fails. */ }
      applyTheme();
    });
    applyTheme();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindToggle, { once: true });
  } else bindToggle();
})();
