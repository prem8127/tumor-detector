"use strict";
// TumorScan AI — patient.js
// Reads organ/model from data-val attributes (NOT data-value)
// Model values sent to backend: "svm" or "cnn" ONLY

document.addEventListener("DOMContentLoaded", function () {

  // ── State ─────────────────────────────────────────────────────
  var selectedOrgan = "lung";   // default
  var selectedModel = "svm";    // default
  var selectedFile  = null;

  // ── Element refs ──────────────────────────────────────────────
  var uploadZone    = document.getElementById("upload-zone");
  var fileInput     = document.getElementById("file-input");
  var runBtn        = document.getElementById("run-btn");
  var previewImg    = document.getElementById("preview-img");
  var previewEmpty  = document.getElementById("preview-empty");
  var loadingState  = document.getElementById("loading-state");
  var loadingSub    = document.getElementById("loading-sub");
  var errorPanel    = document.getElementById("error-panel");
  var errorText     = document.getElementById("error-text");

  // ── Set today's date ──────────────────────────────────────────
  var dateEl = document.getElementById("pt-date");
  if (dateEl) dateEl.valueAsDate = new Date();

  // ── Live patient card update ──────────────────────────────────
  function updateCard() {
    var name   = (document.getElementById("pt-name")?.value   || "").trim();
    var age    = document.getElementById("pt-age")?.value     || "—";
    var gender = document.getElementById("pt-gender")?.value  || "—";

    var av = document.getElementById("pt-avatar");
    var nd = document.getElementById("pt-name-display");
    var md = document.getElementById("pt-meta-display");

    if (av) av.textContent = name
      ? name.split(" ").map(function(w){ return w[0]; }).slice(0,2).join("").toUpperCase()
      : "?";
    if (nd) nd.textContent = name || "—";
    if (md) md.textContent = "Age " + age + " · " + gender;
  }
  ["pt-name","pt-age","pt-gender","pt-date"].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener("input", updateCard);
  });

  // ── Organ buttons ─────────────────────────────────────────────
  var organBtns = document.querySelectorAll("#organ-group button[data-val]");
  organBtns.forEach(function(btn) {
    btn.addEventListener("click", function() {
      organBtns.forEach(function(b) { b.classList.remove("active"); });
      btn.classList.add("active");
      selectedOrgan = btn.getAttribute("data-val");
      var badge = document.getElementById("badge-organ");
      if (badge) badge.textContent = selectedOrgan.charAt(0).toUpperCase() + selectedOrgan.slice(1);
    });
  });

  // ── Model buttons ─────────────────────────────────────────────
  var modelBtns = document.querySelectorAll("#model-group button[data-val]");
  modelBtns.forEach(function(btn) {
    btn.addEventListener("click", function() {
      modelBtns.forEach(function(b) { b.classList.remove("active"); });
      btn.classList.add("active");
      selectedModel = btn.getAttribute("data-val");
      var badge = document.getElementById("badge-model");
      if (badge) badge.textContent = selectedModel.toUpperCase();
    });
  });

  // ── Upload zone ───────────────────────────────────────────────
  if (uploadZone) {
    uploadZone.addEventListener("click", function() { fileInput && fileInput.click(); });
    uploadZone.addEventListener("keydown", function(e) {
      if (e.key === "Enter" || e.key === " ") { fileInput && fileInput.click(); }
    });
    uploadZone.addEventListener("dragover", function(e) {
      e.preventDefault();
      uploadZone.classList.add("dragover");
    });
    uploadZone.addEventListener("dragleave", function() {
      uploadZone.classList.remove("dragover");
    });
    uploadZone.addEventListener("drop", function(e) {
      e.preventDefault();
      uploadZone.classList.remove("dragover");
      var f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    });
  }
  if (fileInput) {
    fileInput.addEventListener("change", function() {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });
  }

  function handleFile(file) {
    if (!/\.(png|jpe?g|bmp|tiff?)$/i.test(file.name)) {
      showError("Unsupported format. Please upload PNG, JPG, BMP or TIFF.");
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showError("File too large. Maximum size is 16 MB.");
      return;
    }

    selectedFile = file;

    // Preview
    var reader = new FileReader();
    reader.onload = function(e) {
      if (previewImg)  { previewImg.src = e.target.result; previewImg.hidden = false; }
      if (previewEmpty) previewEmpty.hidden = true;
    };
    reader.readAsDataURL(file);

    // Update zone text
    var tz = uploadZone ? uploadZone.querySelector(".upload-text") : null;
    var hz = uploadZone ? uploadZone.querySelector(".upload-hint") : null;
    if (tz) tz.textContent = file.name;
    if (hz) hz.textContent = (file.size / 1024).toFixed(1) + " KB";
    if (uploadZone) uploadZone.classList.add("has-file");

    // Enable run button
    if (runBtn) runBtn.disabled = false;
    if (errorPanel) errorPanel.hidden = true;
  }

  // ── Run Analysis ──────────────────────────────────────────────
  if (runBtn) runBtn.addEventListener("click", runAnalysis);

  function runAnalysis() {
    if (!selectedFile) {
      showError("Please upload a CT scan image first.");
      return;
    }

    // Validate — must be exactly "svm" or "cnn"
    var organ = selectedOrgan;
    var model = selectedModel;
    if (!["lung","pancreas"].includes(organ)) organ = "lung";
    if (!["svm","cnn"].includes(model))       model = "svm";

    // Show loading
    if (loadingState) loadingState.hidden = false;
    if (errorPanel)   errorPanel.hidden   = true;
    if (runBtn)       runBtn.disabled     = true;

    var steps = [
      "Preprocessing image…",
      "Extracting features…",
      "Running " + model.toUpperCase() + " inference…",
      "Generating heatmap…",
      "Almost done…"
    ];
    var step = 0;
    var timer = setInterval(function() {
      step = (step + 1) % steps.length;
      if (loadingSub) loadingSub.textContent = steps[step];
    }, 900);

    // Build form data
    var fd = new FormData();
    fd.append("image",      selectedFile);
    fd.append("organ",      organ);
    fd.append("model_type", model);   // "svm" or "cnn" — backend validates these
    fd.append("name",       document.getElementById("pt-name")?.value   || "Unknown");
    fd.append("age",        document.getElementById("pt-age")?.value    || "");
    fd.append("gender",     document.getElementById("pt-gender")?.value || "");
    fd.append("scan_date",  document.getElementById("pt-date")?.value   || "");

    fetch("/predict", { method: "POST", body: fd })
      .then(function(res) {
        if (!res.ok) {
          return res.text().then(function(t) {
            throw new Error("Server " + res.status + ": " + t.slice(0,200));
          });
        }
        return res.json();
      })
      .then(function(data) {
        clearInterval(timer);
        if (loadingState) loadingState.hidden = true;
        if (runBtn)       runBtn.disabled     = false;

        if (data.error) { showError(data.error); return; }

        // Store ONLY small JSON — NO images in sessionStorage
        try {
          sessionStorage.setItem("scanResult", JSON.stringify(data));
        } catch(e) {
          sessionStorage.clear();
          sessionStorage.setItem("scanResult", JSON.stringify(data));
        }

        window.location.href = "/results";
      })
      .catch(function(err) {
        clearInterval(timer);
        if (loadingState) loadingState.hidden = true;
        if (runBtn)       runBtn.disabled     = false;
        showError(err.message || "Network error. Is Flask running on port 5000?");
      });
  }

  function showError(msg) {
    if (errorText)  errorText.textContent = msg;
    if (errorPanel) errorPanel.hidden     = false;
  }

}); // DOMContentLoaded