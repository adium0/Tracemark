/* Integrity Theory demo: upload, call /api/analyze, render the explanation. Plain JS, no build step. */
(function () {
  "use strict";

  var MAX_BYTES = 20 * 1024 * 1024;
  var $ = function (id) { return document.getElementById(id); };

  var els = {
    card: $("upload-card"), dropzone: $("dropzone"), input: $("file-input"), dropTitle: $("drop-title"),
    selected: $("selected"), preview: $("preview"), previewFallback: $("preview-fallback"),
    fileMeta: $("file-meta"), fileName: $("file-name"), status: $("status"),
    error: $("error"), errorText: $("error-text"), results: $("results"), live: $("live")
  };

  var ICONS = {
    check: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    cross: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>',
    verdict: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/></svg>',
    evidence: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h16M7 16V9M12 16V5M17 16v-4"/></svg>',
    provenance: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 16h5"/></svg>',
    gradcam: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    note: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.5v.01"/></svg>',
    fakeBadge: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l2.2 6.3L20.5 12l-6.3 2.2L12 20.5l-2.2-6.3L3.5 12l6.3-2.7z"/></svg>',
    realBadge: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
  };
  var KIND_LABEL = { verdict: "Verdict", evidence: "Evidence", provenance: "Provenance", gradcam: "Grad-CAM", note: "Note" };

  /* ---------- helpers ---------- */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmtBytes(n) {
    if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
    if (n >= 1024) return Math.round(n / 1024) + " KB";
    return n + " bytes";
  }
  function f2(x) { return Number(x).toFixed(2); }
  function f4(x) { return Number(x).toFixed(4); }
  function signed(x, d) {
    d = d == null ? 3 : d;
    if (Math.abs(x) < 0.5 * Math.pow(10, -d)) return (0).toFixed(d);
    return (x > 0 ? "+" : "−") + Math.abs(x).toFixed(d);
  }
  function pct(conf) { return conf >= 0.9995 ? ">99.9%" : (conf * 100).toFixed(1) + "%"; }
  function png(b64) { return "data:image/png;base64," + b64; }
  function announce(msg) { els.live.textContent = ""; window.setTimeout(function () { els.live.textContent = msg; }, 50); }
  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ } }

  /* ---------- header controls ---------- */
  var root = document.documentElement;
  var calmBtn = $("calm-toggle");
  function syncCalm() { calmBtn.setAttribute("aria-checked", root.classList.contains("calm") ? "true" : "false"); }
  calmBtn.addEventListener("click", function () {
    var on = !root.classList.contains("calm");
    root.classList.toggle("calm", on);
    store("it-calm", on ? "on" : "off");
    syncCalm();
    announce(on ? "Calm mode on. Animations are off." : "Calm mode off.");
  });
  syncCalm();

  var themeBtn = $("theme-toggle"), themeLabel = $("theme-label");
  var THEMES = ["auto", "light", "dark"];
  var THEME_NAMES = { auto: "match my device", light: "light", dark: "dark" };
  var themeHint = $("theme-hint");
  function currentTheme() { var t = root.getAttribute("data-theme"); return t === "light" || t === "dark" ? t : "auto"; }
  function syncTheme() {
    var t = currentTheme();
    themeLabel.textContent = t === "auto" ? "Auto" : t === "light" ? "Light" : "Dark";
    themeHint.textContent = t === "auto" ? " (match my device)" : "";
  }
  themeBtn.addEventListener("click", function () {
    var next = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
    if (next === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", next);
    store("it-theme", next);
    syncTheme();
    announce("Theme: " + THEME_NAMES[next] + ".");
  });
  syncTheme();

  /* ---------- upload ---------- */
  var busy = false, previewUrl = null;

  function showError(msg) {
    els.errorText.textContent = msg;
    els.error.hidden = false;
  }
  function clearError() { els.error.hidden = true; els.errorText.textContent = ""; }
  function setLoading(on) {
    busy = on;
    els.status.hidden = !on;
    els.card.classList.toggle("is-loading", on);
    els.input.disabled = on;
    $("choose-another").disabled = on;
  }

  function showPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    els.preview.hidden = false;
    els.previewFallback.hidden = true;
    els.preview.alt = "Preview of " + file.name;
    els.preview.src = previewUrl;
    els.fileName.textContent = file.name;
    els.fileMeta.textContent = fmtBytes(file.size) + (file.type ? " · " + file.type : "");
    els.selected.hidden = false;
    els.card.classList.add("has-file");
    els.dropTitle.textContent = "Drop another image here";
  }
  els.preview.addEventListener("error", function () {
    if (!els.preview.getAttribute("src")) return;
    els.preview.hidden = true;
    els.previewFallback.hidden = false;
  });

  function handleFile(file) {
    if (!file || busy) return;
    clearError();
    els.results.hidden = true;   // never leave the previous image's results next to a new file
    if (file.size > MAX_BYTES || file.size === 0) {
      els.selected.hidden = true;
      els.card.classList.remove("has-file");
      els.dropTitle.textContent = "Drag an image here";
      var msg = file.size === 0 ? "The selected file is empty."
        : "This file is " + fmtBytes(file.size) + ". The limit is 20 MB. Try a smaller or more compressed copy of the image.";
      showError(msg);
      announce("Can't analyze this file. " + msg);
      return;
    }
    showPreview(file);
    analyze(file);
  }

  function analyze(file) {
    setLoading(true);
    announce("Analyzing " + file.name + ". This usually takes a second or two.");
    var form = new FormData();
    form.append("file", file, file.name);
    fetch("/api/analyze", { method: "POST", body: form })
      .then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) { return { ok: res.ok, status: res.status, data: data }; });
      })
      .then(function (r) {
        setLoading(false);
        if (!r.ok || !r.data || r.data.error) {
          var msg = r.data && r.data.error ? r.data.error.message : "The server returned an unexpected error (HTTP " + r.status + ").";
          showError(msg);
          announce("Can't analyze this file. " + msg);
          return;
        }
        render(r.data);
      })
      .catch(function () {
        setLoading(false);
        showError("Couldn't reach the local server. Is the window that started the demo (run.bat / run.sh) still open?");
        announce("Couldn't reach the local server.");
      });
  }

  els.input.addEventListener("change", function () {
    var file = els.input.files && els.input.files[0];
    handleFile(file);
    els.input.value = "";
  });
  $("choose-another").addEventListener("click", function () { els.input.click(); });
  $("analyze-another").addEventListener("click", function () {
    els.card.scrollIntoView({ block: "start" });
    els.input.focus();
    els.input.click();
  });

  var dragDepth = 0;
  ["dragenter", "dragover"].forEach(function (type) {
    els.card.addEventListener(type, function (e) {
      if (!e.dataTransfer || Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") === -1) return;
      e.preventDefault();
      if (type === "dragenter") dragDepth++;
      els.dropzone.classList.add("is-dragover");
      els.dropTitle.textContent = "Drop to analyze";
    });
  });
  els.card.addEventListener("dragleave", function () {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      els.dropzone.classList.remove("is-dragover");
      els.dropTitle.textContent = els.card.classList.contains("has-file") ? "Drop another image here" : "Drag an image here";
    }
  });
  els.card.addEventListener("drop", function (e) {
    e.preventDefault();
    dragDepth = 0;
    els.dropzone.classList.remove("is-dragover");
    els.dropTitle.textContent = els.card.classList.contains("has-file") ? "Drop another image here" : "Drag an image here";
    var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
  });
  // Stop the browser from opening a file dropped outside the card.
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) { e.preventDefault(); });
  // Paste an image from the clipboard.
  window.addEventListener("paste", function (e) {
    var items = (e.clipboardData && e.clipboardData.files) || [];
    if (items.length) handleFile(items[0]);
  });

  /* ---------- render ---------- */
  var camImages = {};

  function render(d) {
    var v = d.verdict;
    var fake = v.is_fake;

    // 02 Verdict
    var card = $("verdict-card");
    card.classList.toggle("is-fake", fake);
    card.classList.toggle("is-real", !fake);
    $("verdict-badge").innerHTML = fake ? ICONS.fakeBadge : ICONS.realBadge;
    $("verdict-word").textContent = fake ? "AI-generated" : "Real";
    $("verdict-sub").textContent = fake
      ? "The model's verdict: this image was likely made by an AI generator (class 1, Fake)."
      : "The model's verdict: this image is likely a real photo (class 0, Real).";
    $("verdict-conf").textContent = pct(v.confidence);
    $("prob-real").style.width = "calc(" + (v.p_real * 100).toFixed(3) + "% - 1px)";
    $("prob-fake").style.width = "calc(" + (v.p_fake * 100).toFixed(3) + "% - 1px)";
    $("prob-bar").setAttribute("aria-label", "P(real) " + f4(v.p_real) + ", P(fake) " + f4(v.p_fake) + ". The decision line is at 0.5.");
    $("p-real").textContent = f4(v.p_real);
    $("p-fake").textContent = f4(v.p_fake);
    $("timing").textContent = "Processed in " + Math.round(d.timing_ms.total) + " ms on the CPU";

    // 03 Summary
    $("summary").innerHTML = d.summary.map(function (s) {
      return '<li class="' + (s.kind === "verdict" ? "verdict-line" : "") + '"><span class="kind">' + (ICONS[s.kind] || ICONS.note) +
        '</span><div><span class="spec kind-label">' + esc(KIND_LABEL[s.kind] || s.kind) + "</span><p>" + esc(s.text) + "</p></div></li>";
    }).join("");

    // 04 Ablation
    renderAblation(d.ablation);

    // 05 Provenance
    renderProvenance(d.provenance);

    // 06 Grad-CAM
    var g = d.gradcam;
    camImages = { overlay: png(g.overlay_png), heatmap: png(g.heatmap_png), image: png(g.image_png) };
    setCamView("overlay");
    var camText = "Grad-CAM at " + g.layer + " (a " + g.grid[0] + " × " + g.grid[1] + " grid), computed by back-propagating the 'fake' logit (value " + f2(g.fake_logit) + ").";
    if (g.has_signal) {
      camText += " Strongest region: " + g.peak_region + " (" + Math.round(g.peak_region_share * 100) + "% of the total response); the top 10% of cells hold " +
        Math.round(g.top10_share * 100) + "% of it." + (fake ? "" : " The overall verdict is 'real', so these are the regions that pushed hardest toward 'fake'.");
    } else {
      camText += " No region raised the 'fake' score, so the map is empty.";
    }
    camText += " The overlay is smoothed; “Heatmap only” shows the true 14 × 14 cells.";
    $("cam-caption").textContent = camText;

    // 07 Frequency
    var fq = d.frequency;
    $("freq-full").src = png(fq.spectrum_png);
    $("freq-full").alt = "Log-magnitude frequency spectrum of the image, 256 by 256, with the central 32 by 32 square outlined.";
    $("freq-full-caption").textContent = "Full 256 × 256 log-magnitude spectrum, log(1 + |FFT|), of the grayscale image. The centre holds the lowest frequencies. " +
      "The outlined 32 × 32 square [112:144, 112:144] is the only part the model reads (values " + f2(fq.spectrum_min) + " to " + f2(fq.spectrum_max) + " across the whole spectrum).";
    $("freq-crop").src = png(fq.crop_png);
    $("freq-crop").alt = "The 32 by 32 crop of the spectrum that the frequency branch receives, enlarged.";
    $("freq-crop-caption").textContent = "The crop itself, enlarged: these 1,024 values are the frequency branch's input. Brightness is scaled to this crop's own range (" +
      f2(fq.crop_min) + " to " + f2(fq.crop_max) + ", mean " + f2(fq.crop_mean) + ").";

    // 08 ELA
    var e = d.ela;
    $("ela-img").src = png(e.png);
    $("ela-img").alt = "Error level analysis of the image: brighter areas changed more when the image was re-saved as a JPEG.";
    $("ela-caption").textContent = "The image was re-saved as JPEG at quality " + e.quality + " and compared with the original. Differences are amplified " +
      e.amplification.toFixed(1) + "× (largest difference " + e.max_diff + " of 255, average " + e.mean_diff.toFixed(2) + "). Regions that re-compress differently can stand out after editing. " +
      "This view is for comparison only: the model never sees it. It is most meaningful for JPEG files; images that were never JPEG-compressed (most PNGs) light up everywhere.";

    // 09 Details
    var fi = d.file, t = d.timing_ms;
    var rows = [
      ["File", esc(fi.name)],
      ["Detected format", esc(fi.format) + " <span class=\"spec\">(from the file contents)</span>"],
      ["Dimensions", fi.width + " × " + fi.height + " px (" + fi.megapixels.toFixed(1) + " MP), mode " + esc(fi.mode)],
      ["File size", fmtBytes(fi.bytes)],
      ["Model", esc(d.model.name) + " · " + esc(d.model.weights) + " · " + esc(d.model.device.toUpperCase())],
      ["Raw logits", "real " + f4(v.logits[0]) + ", fake " + f4(v.logits[1])],
      ["P(real) / P(fake)", Number(v.p_real).toPrecision(6) + " / " + Number(v.p_fake).toPrecision(6)],
      ["Processing time", Math.round(t.total) + " ms total · " + Math.round(t.inference) + " ms preprocessing + model · " + Math.round(t.explanations) + " ms explanations"]
    ];
    $("details").innerHTML = rows.map(function (r) { return "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>"; }).join("");

    // Show, animate in (unless calm), move focus for keyboard and screen reader users
    var sections = els.results.children;
    for (var i = 0; i < sections.length; i++) sections[i].style.setProperty("--i", i);
    els.results.classList.remove("reveal");
    els.results.hidden = false;
    void els.results.offsetWidth;
    els.results.classList.add("reveal");
    $("verdict-heading").setAttribute("tabindex", "-1");
    $("verdict-heading").focus({ preventScroll: true });
    $("verdict-card").scrollIntoView({ block: "start", behavior: root.classList.contains("calm") ? "auto" : "smooth" });
    announce("Analysis complete. Verdict: " + (fake ? "AI-generated" : "real") + ", " + pct(v.confidence).replace(">", "above ").replace("%", " percent") + " confidence.");
  }

  function niceDomain(maxAbs) {
    var steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1];
    for (var i = 0; i < steps.length; i++) if (maxAbs <= steps[i]) return steps[i];
    return 1;
  }

  function renderAblation(a) {
    var maxAbs = Math.max.apply(null, a.items.map(function (i) { return Math.abs(i.delta); }));
    var D = niceDomain(Math.max(maxAbs, 0.0001));
    $("tick-min").textContent = "−" + D;
    $("tick-max").textContent = "+" + D;
    $("ablation-rows").innerHTML = a.items.map(function (it) {
      var w = Number(Math.min(50, Math.abs(it.delta) / D * 50).toFixed(3));
      var dir = it.delta > 0.0005 ? "fake" : it.delta < -0.0005 ? "real" : "none";
      var text = dir === "none" ? "no effect" : "toward " + dir;
      var bar = "";
      if (dir === "fake") bar = '<span class="drow-bar to-fake" style="left:50%;width:' + w + '%"></span>';
      if (dir === "real") bar = '<span class="drow-bar to-real" style="right:50%;width:' + w + '%"></span>';
      var tipRoom = w <= 24;
      var pos;
      if (dir === "real") pos = tipRoom ? "right:calc(50% + " + w + "% + 10px);text-align:right" : "left:calc(50% + 10px)";
      else pos = tipRoom ? "left:calc(50% + " + w + "% + 10px)" : "right:calc(50% + 10px);text-align:right";
      var label = '<span class="drow-value" style="' + pos + '"><strong>' + signed(it.delta) + "</strong> " + text +
        "<small>P(fake) without: " + Number(it.p_fake_without).toFixed(3) + "</small></span>";
      return '<div class="drow"><div class="drow-label">' + esc(it.label) + '<span class="spec">' + esc(it.detail) + "</span></div>" +
        '<div class="drow-plot" role="img" aria-label="' + esc(it.label) + ": " + text + ", change " + signed(it.delta, 3) +
        ". P(fake) without this group: " + f4(it.p_fake_without) + '."><span class="drow-zero"></span>' + bar + label + "</div></div>";
    }).join("");
    $("ablation-table").innerHTML = a.items.map(function (it) {
      var dir = it.delta > 0.0005 ? "pushed toward fake" : it.delta < -0.0005 ? "pushed toward real" : "no effect";
      return "<tr><th scope=\"row\">" + esc(it.label) + "</th><td>" + it.dims + "</td><td>" + f4(a.p_fake_full) + "</td><td>" +
        f4(it.p_fake_without) + "</td><td>" + signed(it.delta, 4) + " (" + dir + ")</td></tr>";
    }).join("");
  }

  function flag(yes) {
    return '<span class="flag ' + (yes ? "yes" : "no") + '">' + (yes ? ICONS.check + "Found" : ICONS.cross + "Not found") + "</span>";
  }

  function renderProvenance(p) {
    var exifBody;
    if (p.exif) {
      var tags = p.exif_tags || {};
      var keys = Object.keys(tags);
      exifBody = "<p>" + p.exif_count + " EXIF tag" + (p.exif_count === 1 ? "" : "s") + " in the file.</p>" +
        (keys.length ? "<dl>" + keys.map(function (k) { return "<dt>" + esc(k) + "</dt><dd>" + esc(tags[k]) + "</dd>"; }).join("") + "</dl>" : "");
    } else {
      exifBody = "<p>No EXIF block. Screenshots, images saved from the web and most AI generators have none.</p>";
    }
    $("prov-exif").innerHTML = '<div class="prov-head"><span class="prov-name">EXIF metadata</span>' + flag(p.exif) + "</div>" +
      '<div class="prov-body">' + exifBody + '<p class="spec prov-model">Model input has_exif = ' + (p.exif ? "1.0" : "0.0") + "</p></div>";

    var c2paBody;
    if (p.c2pa) {
      c2paBody = "<p>The bytes " + p.c2pa_markers.map(function (m) { return "<strong>“" + esc(m) + "”</strong>"; }).join(" and ") +
        " appear in the file (first at byte " + Number(p.c2pa_offset).toLocaleString() + "). C2PA content credentials are attached by several generative AI tools, and by some cameras and editing apps.</p>";
    } else {
      c2paBody = "<p>Neither “c2pa” nor “JUMBF” appears anywhere in the file's bytes. Most images carry no content credentials.</p>";
    }
    $("prov-c2pa").innerHTML = '<div class="prov-head"><span class="prov-name">C2PA / JUMBF marker</span>' + flag(p.c2pa) + "</div>" +
      '<div class="prov-body">' + c2paBody + '<p class="spec prov-model">Model input has_c2pa = ' + (p.c2pa ? "1.0" : "0.0") + "</p></div>";
  }

  var CAM_ALT = {
    overlay: "Grad-CAM heatmap over the image. Brighter, yellower areas raised the 'fake' score more.",
    heatmap: "Grad-CAM heatmap alone, as a 14 by 14 grid. Brighter cells raised the 'fake' score more.",
    image: "The analyzed image, as the model reads it (without EXIF rotation)."
  };
  function setCamView(view) {
    var img = $("cam-img");
    img.src = camImages[view];
    img.alt = CAM_ALT[view];
    var buttons = document.querySelectorAll(".seg");
    for (var i = 0; i < buttons.length; i++) buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-view") === view ? "true" : "false");
  }
  Array.prototype.forEach.call(document.querySelectorAll(".seg"), function (b) {
    b.addEventListener("click", function () { setCamView(b.getAttribute("data-view")); });
  });
})();
