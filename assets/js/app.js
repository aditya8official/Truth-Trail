"use strict";

const $ = (id) => document.getElementById(id);
const API_BASE = (document.querySelector('meta[name="api-base-url"]')?.content || "http://127.0.0.1:8000").replace(/\/$/, "");
const HISTORY_KEY = "truth-trails.analysis-history.v1";
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MIME_BY_EXTENSION = {
  jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp",
  mp4: "video/mp4", mov: "video/quicktime", avi: "video/x-msvideo",
  webm: "video/webm", mkv: "video/x-matroska",
};
const $dropzone = $("dropzone");

let selectedFile = null;
let selectedKind = null;
let previewUrl = null;
let toastTimer = null;

function toast(message) {
  $("toast").textContent = message;
  $("toast").style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").style.display = "none"; }, 3500);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function logActivity(title, description) {
  const item = document.createElement("div");
  item.className = "feed-item";
  const heading = document.createElement("strong");
  const detail = document.createElement("p");
  heading.textContent = title;
  detail.textContent = description;
  item.append(heading, detail);
  $("feed").prepend(item);
}

function fileKind(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  const mime = (file.type || "").toLowerCase();
  if (mime.startsWith("image/") || ["jpg", "jpeg", "png", "webp"].includes(extension)) {
    return ["image/jpeg", "image/png", "image/webp"].includes(mime) || !mime
      ? "image"
      : null;
  }
  if (mime.startsWith("video/") || ["mp4", "mov", "avi", "webm", "mkv"].includes(extension)) {
    return Object.values(MIME_BY_EXTENSION).includes(mime) || !mime
      ? "video"
      : null;
  }
  return null;
}

function uploadMime(file) {
  const existing = (file.type || "").toLowerCase();
  if (Object.values(MIME_BY_EXTENSION).includes(existing)) return existing;
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  return MIME_BY_EXTENSION[extension] || "application/octet-stream";
}

function loadFile(file) {
  if (!file) return;
  const kind = fileKind(file);
  if (!kind) {
    toast("Choose a supported JPG, PNG, WEBP, MP4, MOV, AVI, WEBM, or MKV file.");
    $("fileInput").value = "";
    return;
  }
  const sizeLimit = kind === "image" ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;
  if (file.size > sizeLimit) {
    toast(kind === "image" ? "Image is larger than 10 MB." : "Video is larger than 100 MB.");
    $("fileInput").value = "";
    return;
  }

  releasePreview();
  selectedFile = file;
  selectedKind = kind;
  previewUrl = URL.createObjectURL(file);
  $("fileName").textContent = file.name;
  $("fileSize").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${kind}`;

  if (kind === "image") {
    $("imagePreview").src = previewUrl;
    $("imagePreview").classList.remove("hidden");
    $("videoPreview").classList.add("hidden");
    $("videoPreview").removeAttribute("src");
  } else {
    $("videoPreview").src = previewUrl;
    $("videoPreview").classList.remove("hidden");
    $("imagePreview").classList.add("hidden");
    $("imagePreview").removeAttribute("src");
    $("videoPreview").load();
  }

  $("uploadPrompt").classList.add("hidden");
  $("fileMeta").style.display = "flex";
  $("analyzeBtn").disabled = false;
  $("resultTag").textContent = "Ready";
  $("resultBox").innerHTML = '<div class="result-empty">Media ready. Select Analyze media to continue.</div>';
  $("printReport").disabled = true;
  logActivity("Media selected", file.name);
}

function releasePreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $("videoPreview").pause();
  $("videoPreview").removeAttribute("src");
  $("videoPreview").load();
  $("imagePreview").removeAttribute("src");
}

function clearSelected() {
  selectedFile = null;
  selectedKind = null;
  $("fileInput").value = "";
  releasePreview();
  $("fileMeta").style.display = "none";
  $("uploadPrompt").classList.remove("hidden");
  $("analyzeBtn").disabled = true;
  $("resultTag").textContent = "Waiting";
  $("resultBox").innerHTML = '<div class="result-empty">No media analyzed yet. Choose an image or video.</div>';
  $("printReport").disabled = true;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "Unavailable";
  return `${(Math.max(0, Math.min(1, number)) * 100).toFixed(1)}%`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown time" : date.toLocaleString();
}

function reportMarkup(data) {
  const fake = Number(data.fake_score);
  const real = Number(data.real_score);
  if (!Number.isFinite(fake) || !Number.isFinite(real)) {
    throw new Error("API response is missing valid model scores.");
  }
  const isFake = data.predicted_label === "fake";
  const type = data.analysis_type === "video" ? "Video" : "Image";
  const file = data.file || {};
  const source = data.source_record || {};
  const hash = file.sha256 || "Not provided";
  const modelLine = `${data.model || "MobileNetV2"}${data.model_version ? ` · ${data.model_version}` : ""}`;
  const scoreBar = (label, score, color = "var(--mint)") => `
    <div class="barlabel"><span>${label}</span><b>${formatPercent(score)}</b></div>
    <div class="bar"><span style="width:${Math.max(0, Math.min(1, Number(score))) * 100}%;background:${color}"></span></div>`;
  const frames = Array.isArray(data.frame_results) ? data.frame_results : [];
  const frameList = frames.length
    ? `<details class="frame-results"><summary>Frame details (${frames.length} analyzed frames)</summary><ul>${frames.map((frame) => {
      const faces = Array.isArray(frame.faces) ? frame.faces : [];
      const label = frame.predicted_label === "fake" ? "score at/above threshold" : "score below threshold";
      return `<li>${Number(frame.timestamp_seconds).toFixed(2)}s · ${faces.length} face(s) · fake score ${formatPercent(frame.fake_score)} · ${label}</li>`;
    }).join("")}</ul></details>`
    : "";

  return `
    <div class="prediction-hero ${isFake ? "fake" : "real"}">
      <div class="prediction-kicker">${type} model signal</div>
      <div class="prediction-label">${isFake ? "POTENTIALLY FAKE" : "LIKELY REAL"}</div>
      <div class="prediction-score">Predicted class score: <strong>${formatPercent(isFake ? fake : real)}</strong></div>
      <div class="prediction-note">Experimental result · not proof of authenticity</div>
    </div>
    <div class="result-status"><div class="symbol">${isFake ? "⚑" : "✓"}</div><div><strong>${isFake ? "Predicted fake" : "Predicted real"}</strong><small>${escapeHtml(modelLine)}</small></div></div>
    ${scoreBar("Fake-class model score", fake)}
    ${scoreBar("Real-class model score", real, "var(--blue)")}
    <p class="privacy">Threshold: ${Number(data.threshold ?? 0.5).toFixed(2)} · ${escapeHtml(data.score_semantics || "Uncalibrated model score; not a calibrated probability.")}</p>
    <p class="privacy">File: ${escapeHtml(file.filename || "Unknown")} · ${escapeHtml(file.content_type || "unknown type")} · ${Number(file.size_bytes || 0).toLocaleString()} bytes</p>
    <p class="privacy">Analyzed: ${escapeHtml(formatDate(data.created_at))} · SHA-256: <code>${escapeHtml(hash)}</code></p>
    <p class="privacy">Source reference: ${escapeHtml(source.reference || "Not supplied")} · ${escapeHtml(source.note || "Not verified")}</p>
    ${data.analysis_type === "video" ? `<p class="privacy">Video: ${Number(data.duration_seconds || 0).toFixed(2)}s · ${Number(data.frames_analyzed || 0)} frames analyzed · ${Number(data.faces_detected || 0)} faces · ${escapeHtml(data.aggregation_method || "Aggregation not specified")}</p>${frameList}` : ""}
    <p class="warning">${escapeHtml(data.warning || "Experimental output. This result may be incorrect and is not proof of authenticity.")}</p>`;
}

function renderReport(data) {
  try {
    $("resultBox").innerHTML = reportMarkup(data);
    $("resultTag").textContent = data.analysis_type === "video" ? "Video report" : "Image report";
    $("printReport").disabled = false;
  } catch (error) {
    $("resultBox").innerHTML = `<div class="result-status"><div class="symbol">!</div><div><strong>Could not display report</strong><small>${escapeHtml(error.message)}</small></div></div>`;
    $("resultTag").textContent = "Invalid response";
    $("printReport").disabled = true;
  }
}

function readHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveHistory(report) {
  try {
    const reports = [report, ...readHistory().filter((item) => item.analysis_id !== report.analysis_id)].slice(0, 20);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(reports));
    renderHistory();
  } catch {
    toast("Analysis completed, but browser history could not be saved.");
  }
}

function renderHistory() {
  const list = $("historyList");
  const reports = readHistory();
  list.replaceChildren();
  if (!reports.length) {
    const empty = document.createElement("p");
    empty.className = "empty-history";
    empty.textContent = "No saved analyses yet.";
    list.append(empty);
    return;
  }

  for (const report of reports) {
    const row = document.createElement("div");
    row.className = "history-row";
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.reportId = String(report.analysis_id || "");
    const title = document.createElement("strong");
    title.textContent = `${report.analysis_type === "video" ? "Video" : "Image"} · ${report.predicted_label || "unknown"} · ${report.file?.filename || "upload"}`;
    const date = document.createElement("small");
    date.textContent = formatDate(report.created_at);
    button.append(title, date);
    button.addEventListener("click", () => {
      renderReport(report);
      document.querySelector("#latestAnalysis").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    const score = document.createElement("small");
    score.textContent = `Fake score ${formatPercent(report.fake_score)}`;
    row.append(button, score);
    list.append(row);
  }
}

async function analyzeSelectedFile() {
  if (!selectedFile || !selectedKind) return;
  const button = $("analyzeBtn");
  button.disabled = true;
  button.textContent = "Analyzing…";
  $("resultTag").textContent = "Running";
  $("resultBox").innerHTML = '<div class="result-empty">Sending media to the local analysis service…</div>';
  $("printReport").disabled = true;

  const form = new FormData();
  const normalizedFile = new File([selectedFile], selectedFile.name, {
    type: uploadMime(selectedFile),
    lastModified: selectedFile.lastModified,
  });
  form.append("file", normalizedFile, normalizedFile.name);
  const reference = $("sourceReference").value.trim();
  if (reference) form.append("source_reference", reference);

  try {
    const endpoint = selectedKind === "video" ? "/analyze/video" : "/analyze/image";
    const response = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: form });
    const data = await response.json().catch(() => ({ detail: "Unreadable response from API." }));
    if (!response.ok) throw new Error(data.detail || `API request failed (${response.status}).`);
    if (data.predicted_label !== "real" && data.predicted_label !== "fake") {
      throw new Error("API response is missing a supported predicted label.");
    }
    renderReport(data);
    saveHistory(data);
    logActivity(`${selectedKind === "video" ? "Video" : "Image"} analyzed`, `${selectedFile.name} · ${data.predicted_label}`);
    toast("Analysis completed. Review the limitations in the report.");
  } catch (error) {
    $("resultBox").innerHTML = `<div class="result-status"><div class="symbol">!</div><div><strong>Analysis failed</strong><small>${escapeHtml(error.message || "Could not reach the local API.")}</small></div></div><p class="privacy">Check that the FastAPI service is running at ${escapeHtml(API_BASE)}. The selected media was not saved to history.</p>`;
    $("resultTag").textContent = "Error";
    logActivity("Analysis failed", error.message || "API request failed");
    toast("Could not analyze media. Check the local API and try again.");
  } finally {
    button.disabled = !selectedFile;
    button.textContent = "Analyze media ↗";
  }
}

async function checkApiHealth() {
  const status = $("apiStatus");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3500);
  try {
    const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (response.ok && data.model_loaded) {
      status.textContent = "● LOCAL API · MODEL READY";
      status.classList.remove("offline");
    } else {
      status.textContent = "● API READY · MODEL UNAVAILABLE";
      status.classList.add("offline");
    }
  } catch {
    status.textContent = "● LOCAL API · OFFLINE";
    status.classList.add("offline");
  } finally {
    clearTimeout(timeout);
  }
}

$("browseBtn").addEventListener("click", (event) => {
  event.stopPropagation();
  $("fileInput").click();
});
$dropzone.addEventListener("click", (event) => {
  if (!event.target.closest("button") && !selectedFile) $("fileInput").click();
});
$dropzone.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && !selectedFile) {
    event.preventDefault();
    $("fileInput").click();
  }
});
$("fileInput").addEventListener("change", (event) => loadFile(event.target.files[0]));
["dragenter", "dragover"].forEach((eventName) => $dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  $dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach((eventName) => $dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  $dropzone.classList.remove("drag");
}));
$dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
$("removeBtn").addEventListener("click", clearSelected);
$("analyzeBtn").addEventListener("click", analyzeSelectedFile);
$("printReport").addEventListener("click", () => window.print());
$("clearHistory").addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  toast("Saved report history cleared from this browser.");
});
$("clearLog").addEventListener("click", () => $("feed").replaceChildren());

document.querySelectorAll("[data-scroll]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  $(button.dataset.scroll).scrollIntoView({ behavior: "smooth", block: "start" });
}));

renderHistory();
checkApiHealth();
