"use strict";

document.addEventListener("DOMContentLoaded", function() {

  var raw = sessionStorage.getItem("scanResult");
  if (!raw) {
    document.getElementById("no-data").hidden         = false;
    document.getElementById("results-content").hidden = true;
    return;
  }

  var data;
  try { data = JSON.parse(raw); }
  catch(e) {
    document.getElementById("no-data").hidden = false;
    return;
  }

  document.getElementById("results-content").hidden = false;
  renderResults(data);
  loadHeatmap(data.model_type || "SVM");
});

// ── Load heatmap from /get-heatmap ───────────────────────────────────────
function loadHeatmap(modelType) {
  var img     = document.getElementById("heatmap-img");
  var spinner = document.getElementById("heatmap-spinner");
  if (!img) return;

  if (spinner) spinner.style.display = "flex";
  img.style.display  = "none";
  img.style.opacity  = "0";

  // CRITICAL: attach handlers BEFORE setting src
  // otherwise a cached image fires onload before handler exists
  img.onload = function() {
    if (spinner) spinner.style.display = "none";
    img.style.display    = "block";
    img.style.transition = "opacity 0.5s ease";
    setTimeout(function() { img.style.opacity = "1"; }, 20);
  };

  img.onerror = function() {
    if (spinner) spinner.style.display = "none";
    var panel = document.getElementById("heatmap-panel-section");
    if (panel) {
      panel.innerHTML =
        '<div style="padding:32px;text-align:center;color:var(--text-3);font-size:14px;">' +
        '<p style="margin-bottom:12px;">⚠ Heatmap unavailable — session may have expired.</p>' +
        '<a href="/patient" style="color:var(--blue);font-weight:700;">← Run a new scan</a></div>';
    }
  };

  // Set src AFTER handlers — cache-bust so each scan is fresh
  img.src = "/get-heatmap?t=" + Date.now();

  // Safety fallback: if already complete (e.g. browser cache), fire manually
  if (img.complete && img.naturalWidth > 0) {
    img.onload();
  }
}

// ── Render diagnosis card ────────────────────────────────────────────────
function renderResults(data) {
  var label      = data.label      || "—";
  var detected   = data.detected   || false;
  var confidence = data.confidence || 0;
  var severity   = data.severity   || "normal";
  var organ      = data.organ      || "—";
  var modelType  = data.model_type || "SVM";
  var allProbs   = data.all_probs  || {};
  var patient    = data.patient    || {};

  var COLORS = { normal:"#16a34a", warning:"#d97706", danger:"#dc2626" };
  var ICONS  = { normal:"✓", warning:"⚠", danger:"⚠" };
  var BADGE  = {
    normal:  "verdict-badge--normal",
    warning: "verdict-badge--warning",
    danger:  "verdict-badge--detected"
  };
  var CLS_SEV = {
    "Normal":"normal", "Benign":"warning",
    "Malignant":"danger", "Pancreatic Tumor":"danger"
  };

  var sevColor = COLORS[severity]  || "#16a34a";
  var sevIcon  = ICONS[severity]   || "✓";
  var sevBadge = BADGE[severity]   || "verdict-badge--normal";

  // Verdict badge
  var badge = document.getElementById("verdict-badge");
  if (badge) {
    badge.textContent = sevIcon + "  " + label;
    badge.className   = "verdict-badge " + sevBadge;
  }

  var vl = document.getElementById("verdict-label");
  if (vl) vl.textContent = detected ? "Tumor Detected" : "No Tumor Detected";

  var vs = document.getElementById("verdict-sub");
  if (vs) vs.textContent = organ.charAt(0).toUpperCase() + organ.slice(1) + " · " + label;

  var vm = document.getElementById("verdict-model");
  if (vm) vm.textContent = "AI Model: " + modelType;

  // Confidence bar
  var cs = document.getElementById("conf-score");
  if (cs) cs.textContent = confidence + "%";
  var cf = document.getElementById("conf-fill");
  if (cf) {
    cf.style.background = sevColor;
    setTimeout(function() { cf.style.width = confidence + "%"; }, 60);
  }

  // Probability bars
  var pb = document.getElementById("probs-block");
  if (pb) {
    pb.innerHTML = "";
    Object.keys(allProbs).forEach(function(cls) {
      var pct      = allProbs[cls];
      var isActive = cls === label;
      var clsSev   = CLS_SEV[cls] || "normal";
      var barCol   = isActive ? (COLORS[clsSev] || "#1a56db") : "#cbd5e1";

      var row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML =
        '<div class="prob-row-label">' + cls + '</div>' +
        '<div class="prob-row-track"><div class="prob-row-fill" ' +
        'style="width:0%;background:' + barCol + '"></div></div>' +
        '<div class="prob-row-pct">' + pct + '%</div>';
      pb.appendChild(row);

      setTimeout(function() {
        var fill = row.querySelector(".prob-row-fill");
        if (fill) fill.style.width = pct + "%";
      }, 80);
    });
  }

  // Patient record
  var name   = patient.name      || "Unknown";
  var age    = patient.age       || "—";
  var gender = patient.gender    || "—";
  var sdate  = patient.scan_date || "—";

  var initials = (name !== "Unknown" && name.trim().length > 0)
    ? name.trim().split(" ").map(function(w){ return w[0]; }).slice(0,2).join("").toUpperCase()
    : "?";

  var ra = document.getElementById("res-avatar");
  var rn = document.getElementById("res-name");
  var rm = document.getElementById("res-meta");
  if (ra) ra.textContent = initials;
  if (rn) rn.textContent = name;
  if (rm) rm.textContent = "Age " + age + " · " + gender + " · " + sdate;

  var ig = document.getElementById("res-info-grid");
  if (ig) {
    ig.innerHTML = [
      ["Organ",    organ.charAt(0).toUpperCase() + organ.slice(1)],
      ["AI Model", modelType],
      ["Finding",  label],
      ["Scan Date", sdate],
    ].map(function(item) {
      return '<div class="info-item"><div class="info-key">' + item[0] +
             '</div><div class="info-val">' + item[1] + '</div></div>';
    }).join("");
  }

  // Interpretation note
  var INTERP = {
    "Normal":           { text: "No signs of malignancy detected. CT features appear consistent with normal tissue.", style: "background:#f0fdf4;color:#166534;border-color:#bbf7d0;" },
    "Benign":           { text: "A benign mass pattern detected. Likely non-cancerous, but clinical follow-up is recommended.", style: "background:#fffbeb;color:#92400e;border-color:#fde68a;" },
    "Malignant":        { text: "Features consistent with a malignant lesion detected. Urgent clinical review is strongly recommended.", style: "background:#fef2f2;color:#991b1b;border-color:#fecaca;" },
    "Pancreatic Tumor": { text: "Tumor-associated features detected in the pancreatic region. Immediate specialist consultation is strongly advised.", style: "background:#fef2f2;color:#991b1b;border-color:#fecaca;" },
  };
  var note   = document.getElementById("interpretation-note");
  var interp = INTERP[label];
  if (note && interp) {
    note.textContent   = interp.text;
    note.style.cssText = "margin-top:16px;padding:14px 16px;border-radius:8px;" +
      "font-size:13px;line-height:1.6;border:1px solid;display:block;" + interp.style;
  }

  // Dynamic heatmap description (SVM vs CNN)
  var isCNN  = modelType.toLowerCase().includes("cnn");
  var descEl = document.getElementById("heatmap-method-desc");
  if (descEl) {
    descEl.innerHTML = isCNN
      ? "<strong>Grad-CAM Explanation (CNN / ResNet-50):</strong> Panel 2 shows the " +
        "Class Activation Map — <strong>bright/warm = high neural attention</strong>. " +
        "Panel 3 is Guided Saliency (Sobel edges × CAM). Panel 4 overlays a " +
        "<strong>HOT colormap</strong> (white/orange/red = high attention regions)."
      : "<strong>SVM Feature Explanation (HOG + LBP):</strong> " +
        "Panel 2 shows HOG gradient orientations — the 8,100 features the SVM used. " +
        "Panel 3 is the LBP texture map (64-bin histogram). Panel 4 applies a " +
        "<strong>JET colormap</strong> overlay — warm colors mark high-gradient regions " +
        "that most influenced the SVM classification.";
  }

  // Heatmap panel label
  var panelLabel = document.getElementById("heatmap-panel-label");
  if (panelLabel) {
    panelLabel.textContent = isCNN
      ? "CNN Grad-CAM — 5-Panel Spatial Activation Analysis"
      : "SVM Feature Map — 5-Panel HOG+LBP Analysis";
  }

  // PDF download button
  var dlBtn = document.getElementById("download-btn");
  if (dlBtn) {
    dlBtn.addEventListener("click", function() {
      window.location.href = "/download-report";
    });
  }
}