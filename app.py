from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", "'cgi' is deprecated", DeprecationWarning)

import cgi
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import time
import uuid
import urllib.error
import urllib.request
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from readable_pdf import (
    DEFAULT_NMT_MODEL,
    DEFAULT_OLLAMA_MODEL,
    NMT_MODEL_OPTIONS,
    OLLAMA_MODEL_OPTIONS,
    OUTPUT_DIR,
    TMP_DIR,
    TranslationError,
    ensure_dirs,
    translate_pdf,
)
from readable_tex import translate_tex_project


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_gb_to_mb(name: str, default_gb: float) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default_gb * 1024)
    try:
        return max(0, int(float(raw) * 1024))
    except ValueError:
        return int(default_gb * 1024)


HOST = os.environ.get("HOST", os.environ.get("READABLE_HOST", "127.0.0.1"))
PORT = int(os.environ.get("PORT", "8765"))
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("READABLE_MAX_CONCURRENT_JOBS", "4")))
MAX_QUEUED_JOBS = max(0, int(os.environ.get("READABLE_MAX_QUEUED_JOBS", "2")))
JOB_TTL_SECONDS = max(60, int(os.environ.get("READABLE_JOB_TTL_SECONDS", "3600")))
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("READABLE_MAX_UPLOAD_MB", "300"))) * 1024 * 1024
RECOVER_OUTPUTS = os.environ.get("READABLE_RECOVER_OUTPUTS", "1").lower() not in {"0", "false", "no"}
DYNAMIC_JOB_LIMITS = env_flag("READABLE_DYNAMIC_JOB_LIMITS", True)
MEMORY_BUDGET_MB = env_gb_to_mb("READABLE_MEMORY_BUDGET_GB", 100)
DEFAULT_JOB_MEMORY_MB = env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_GB", 20)
MIN_FREE_MEMORY_MB = env_gb_to_mb("READABLE_MIN_FREE_MEMORY_GB", 8)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
NMT_HOST = os.environ.get("NMT_HOST", "http://127.0.0.1:8767").rstrip("/")
NMT_MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("READABLE_NMT_MAX_CONCURRENT_JOBS", "1")))
TEX_SINGLE_SUFFIXES = {".tex", ".zip", ".tar", ".tgz"}
TEX_DOUBLE_SUFFIXES = {(".tar", ".gz"), (".tar", ".xz"), (".tar", ".bz2")}

JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.RLock()
JOB_QUEUE: deque[str] = deque()
RUNNING_JOBS: set[str] = set()
GPU_CACHE: dict[str, object] = {"checked_at": 0.0, "items": []}


HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Readable Local</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #637381;
      --line: #d8dee9;
      --panel: #ffffff;
      --field: #f7f8fa;
      --accent: #0969da;
      --accent-dark: #0757b7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Yu Gothic", sans-serif;
      background: #f2f4f7;
      color: var(--ink);
    }
    main {
      width: min(960px, calc(100% - 32px));
      margin: 36px auto;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      text-align: right;
    }
    form {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      display: grid;
      gap: 18px;
      box-shadow: 0 12px 28px rgba(31, 41, 51, 0.06);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }
    label {
      display: grid;
      gap: 7px;
      font-size: 13px;
      font-weight: 600;
    }
    input, select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--field);
      color: var(--ink);
      padding: 9px 11px;
      font: inherit;
      letter-spacing: 0;
    }
    input[type="file"] {
      padding: 8px;
      background: #ffffff;
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 12px;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #ffffff;
      padding: 0 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: default;
      background: #8aa6c5;
    }
    .message {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: #ffffff;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .progress-panel {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #ffffff;
      display: grid;
      gap: 10px;
    }
    .progress-panel[hidden] { display: none; }
    .meter {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e6ebf1;
    }
    .bar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 180ms ease;
    }
    .progress-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .download {
      width: fit-content;
      border-radius: 6px;
      background: var(--accent);
      color: #ffffff;
      padding: 10px 14px;
      text-decoration: none;
      font-weight: 700;
      font-size: 14px;
    }
    .error-text { color: #b42318; }
    .viewer-panel {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }
    .viewer-panel[hidden] { display: none; }
    .viewer-head {
      min-height: 44px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .viewer-link {
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
      white-space: nowrap;
    }
    .viewer-frame {
      display: block;
      width: 100%;
      height: min(78vh, 760px);
      border: 0;
      background: #eef1f5;
    }
    @media (max-width: 760px) {
      main { width: min(100% - 20px, 960px); margin: 20px auto; }
      header { display: grid; align-items: start; gap: 8px; }
      h1 { font-size: 24px; }
      .status { white-space: normal; }
      .grid { grid-template-columns: 1fr; }
      form { padding: 16px; }
      .actions { justify-content: stretch; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Readable Local</h1>
      <div id="resource-status" class="status">英語PDF / TeX → 日本語PDF</div>
    </header>
    <form id="translate-form" action="/translate" method="post" enctype="multipart/form-data">
      <label>
        PDF / TeXソース
        <input name="pdf" type="file" accept="application/pdf,.pdf,.tex,.zip,.tar,.tar.gz,.tgz" required>
      </label>
      <div class="grid">
        <label>
          翻訳エンジン
          <select id="engine-select" name="engine">
            <option value="nmt" selected>高速NMT（NLLB 600M）</option>
            <option value="ollama">Ollama LLM</option>
            <option value="copy">確認用（翻訳なし）</option>
          </select>
        </label>
        <label>
          モデル
          <select id="model-select" name="model">
            __MODEL_OPTIONS__
          </select>
        </label>
        <label>
          レイアウト
          <select name="layout">
            <option value="fast" selected>高速本文訳（読み比べ）</option>
            <option value="overlay">元レイアウト寄せ（精密・遅い）</option>
            <option value="bilingual">読み比べ（標準）</option>
          </select>
        </label>
      </div>
      <div class="actions">
        <button id="submit-button" type="submit">PDFを作成</button>
      </div>
    </form>
    <section id="progress-panel" class="progress-panel" hidden>
      <div class="meter"><div id="progress-bar" class="bar"></div></div>
      <div class="progress-row">
        <span id="progress-text">待機中</span>
        <span id="progress-percent">0%</span>
      </div>
      <a id="download-link" class="download" href="#" hidden>PDFをダウンロード</a>
    </section>
    <section id="viewer-panel" class="viewer-panel" hidden>
      <div class="viewer-head">
        <strong>生成PDFプレビュー</strong>
        <a id="open-link" class="viewer-link" href="#" target="_blank" rel="noreferrer">別画面で開く</a>
      </div>
      <iframe id="pdf-viewer" class="viewer-frame" title="生成PDFプレビュー"></iframe>
    </section>
    __MESSAGE__
  </main>
  <script>
    const form = document.querySelector("#translate-form");
    const button = document.querySelector("#submit-button");
    const panel = document.querySelector("#progress-panel");
    const bar = document.querySelector("#progress-bar");
    const progressText = document.querySelector("#progress-text");
    const progressPercent = document.querySelector("#progress-percent");
    const downloadLink = document.querySelector("#download-link");
    const viewerPanel = document.querySelector("#viewer-panel");
    const pdfViewer = document.querySelector("#pdf-viewer");
    const openLink = document.querySelector("#open-link");
    const resourceStatus = document.querySelector("#resource-status");
    const engineSelect = document.querySelector("#engine-select");
    const modelSelect = document.querySelector("#model-select");
    const modelsByEngine = __MODEL_OPTIONS_JSON__;

    function updateModelOptions() {
      const models = modelsByEngine[engineSelect.value] || [];
      modelSelect.replaceChildren();
      if (!models.length) {
        const option = document.createElement("option");
        option.value = "copy";
        option.textContent = "モデルなし";
        modelSelect.append(option);
        modelSelect.disabled = true;
        return;
      }
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.value;
        option.textContent = model.label;
        option.selected = Boolean(model.default);
        modelSelect.append(option);
      }
      modelSelect.disabled = false;
    }

    function setProgress(percent, message, isError = false) {
      const value = Math.max(0, Math.min(100, Number(percent || 0)));
      panel.hidden = false;
      bar.style.width = `${value}%`;
      progressPercent.textContent = `${Math.round(value)}%`;
      progressText.textContent = message || "";
      progressText.classList.toggle("error-text", isError);
    }

    async function pollJob(jobId) {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      if (!response.ok) {
        setProgress(0, job.error || "ジョブが見つかりません。", true);
        button.disabled = false;
        button.textContent = "PDFを作成";
        downloadLink.hidden = true;
        viewerPanel.hidden = true;
        pdfViewer.removeAttribute("src");
        return;
      }
      setProgress(job.percent, job.message, job.status === "error");
      if (job.status === "complete") {
        button.disabled = false;
        button.textContent = "PDFを作成";
        downloadLink.href = job.download_url;
        downloadLink.hidden = false;
        const viewUrl = job.view_url || job.download_url;
        openLink.href = viewUrl;
        pdfViewer.src = viewUrl;
        viewerPanel.hidden = false;
        return;
      }
      if (job.status === "error") {
        button.disabled = false;
        button.textContent = "PDFを作成";
        downloadLink.hidden = true;
        viewerPanel.hidden = true;
        pdfViewer.removeAttribute("src");
        return;
      }
      window.setTimeout(() => pollJob(jobId), 900);
    }

    function gb(value) {
      return Number(value || 0).toFixed(1);
    }

    async function updateResourceStatus() {
      try {
        const response = await fetch("/api/status");
        const status = await response.json();
        if (!response.ok) return;
        const active = status.active_concurrent_jobs ?? status.max_concurrent_jobs;
        const running = status.running_jobs ?? 0;
        const memory = status.memory_control || {};
        const nmt = status.nmt || {};
        const nmtLabel = nmt.ready ? " / NMT GPU" : (nmt.status ? ` / NMT: ${nmt.status}` : "");
        if (memory.available) {
          resourceStatus.textContent = `実行 ${running}/${active} / GPU ${gb(memory.effective_used_gb)}/${gb(memory.budget_gb)}GB${nmtLabel}`;
        } else {
          resourceStatus.textContent = `実行 ${running}/${active}${nmtLabel}`;
        }
      } catch (error) {
        resourceStatus.textContent = "英語PDF / TeX → 日本語PDF";
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "処理中";
      downloadLink.hidden = true;
      viewerPanel.hidden = true;
      pdfViewer.removeAttribute("src");
      setProgress(1, "受付中");
      try {
        const response = await fetch("/api/jobs", { method: "POST", body: new FormData(form) });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "開始できませんでした。");
        }
        window.history.replaceState(null, "", `/translate?job=${data.job_id}`);
        pollJob(data.job_id);
      } catch (error) {
        setProgress(0, error.message || "開始できませんでした。", true);
        button.disabled = false;
        button.textContent = "PDFを作成";
      }
    });

    const params = new URLSearchParams(window.location.search);
    engineSelect.addEventListener("change", updateModelOptions);
    updateModelOptions();
    const existingJobId = params.get("job");
    if (existingJobId) {
      setProgress(1, "状態を確認しています");
      pollJob(existingJobId);
    }
    updateResourceStatus();
    window.setInterval(updateResourceStatus, 5000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ReadableLocal/0.1"

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def do_GET(self) -> None:
        cleanup_expired_jobs()
        path = urlparse(self.path).path
        if path == "/api/status" or path.startswith("/api/jobs/"):
            with JOBS_LOCK:
                dispatch_jobs_locked()
        if path == "/api/status":
            self.send_json(public_status())
            return
        if path == "/api/models":
            self.send_json(public_models())
            return
        if path.startswith("/api/jobs/"):
            self.handle_job_get(path)
            return
        if path not in ("/", "/translate"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_html()

    def do_POST(self) -> None:
        cleanup_expired_jobs()
        path = urlparse(self.path).path
        if path == "/api/jobs":
            self.handle_start_job()
            return
        if path != "/translate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.handle_translate()
        except (TranslationError, RuntimeError, ValueError) as exc:
            self.send_html(str(exc), status=HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self.send_html("処理中にエラーが発生しました。", status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_html(self, message: str = "", status: HTTPStatus = HTTPStatus.OK) -> None:
        message_html = ""
        if message:
            message_html = f'<div class="message">{html.escape(message)}</div>'
        model_options_by_engine = {
            "nmt": [
                {"value": value, "label": f"{label} ({value})", "default": value == DEFAULT_NMT_MODEL}
                for value, label in NMT_MODEL_OPTIONS
            ],
            "ollama": [
                {"value": value, "label": f"{label} ({value})", "default": value == DEFAULT_OLLAMA_MODEL}
                for value, label in OLLAMA_MODEL_OPTIONS
            ],
            "copy": [],
        }
        model_options = "\n".join(
            '<option value="{value}"{selected}>{label}</option>'.format(
                value=html.escape(value, quote=True),
                selected=" selected" if value == DEFAULT_NMT_MODEL else "",
                label=html.escape(f"{label} ({value})"),
            )
            for value, label in NMT_MODEL_OPTIONS
        )
        body = (
            HTML.replace("__MODEL_OPTIONS__", model_options)
            .replace("__MODEL_OPTIONS_JSON__", json.dumps(model_options_by_engine, ensure_ascii=False))
            .replace("__MESSAGE__", message_html)
            .encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_start_job(self) -> None:
        try:
            if not can_accept_job():
                self.send_json(
                    {"error": "翻訳ジョブが混み合っています。完了後にもう一度試してください。"},
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                )
                return
            upload_dir, input_path, safe_stem, engine, model, layout, source_type = self.read_translation_request()
            job_id = uuid.uuid4().hex[:12]
            output_path = OUTPUT_DIR / f"{safe_stem}_{job_id}_ja.pdf"
            with JOBS_LOCK:
                if not can_accept_job_locked():
                    shutil.rmtree(upload_dir, ignore_errors=True)
                    self.send_json(
                        {"error": "翻訳ジョブが混み合っています。完了後にもう一度試してください。"},
                        status=HTTPStatus.TOO_MANY_REQUESTS,
                    )
                    return
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "queued",
                    "percent": 1,
                    "message": "待機中",
                    "output_path": str(output_path),
                    "input_path": str(input_path),
                    "upload_dir": str(upload_dir),
                    "filename": output_path.name,
                    "engine": engine,
                    "model": model,
                    "layout": layout,
                    "source_type": source_type,
                    "created_at": time.time(),
                }
                JOB_QUEUE.append(job_id)
                dispatch_jobs_locked()
            self.send_json({"job_id": job_id})
        except (TranslationError, RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self.send_json({"error": "処理を開始できませんでした。"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_job_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        job_id = parts[2]
        if len(parts) == 4 and parts[3] == "download":
            self.handle_job_pdf(job_id, download=True)
            return
        if len(parts) == 4 and parts[3] == "view":
            self.handle_job_pdf(job_id, download=False)
            return
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
        if not job:
            recovered = recover_completed_job(job_id)
            if recovered:
                with JOBS_LOCK:
                    JOBS[job_id] = recovered
                job = dict(recovered)
            else:
                self.send_json({"error": "ジョブが見つかりません。"}, status=HTTPStatus.NOT_FOUND)
                return
        self.send_json(public_job(job))

    def handle_job_pdf(self, job_id: str, download: bool) -> None:
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
        if not job:
            recovered = recover_completed_job(job_id)
            if recovered:
                with JOBS_LOCK:
                    JOBS[job_id] = recovered
                job = dict(recovered)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        if job.get("status") != "complete":
            self.send_error(HTTPStatus.CONFLICT)
            return
        output_path = Path(str(job.get("output_path", "")))
        if not output_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_pdf(output_path, download=download)

    def handle_translate(self) -> None:
        upload_dir, input_path, safe_stem, engine, model, layout, source_type = self.read_translation_request()
        try:
            output_path = OUTPUT_DIR / f"{safe_stem}_ja.pdf"
            if source_type == "tex":
                translate_tex_project(input_path, output_path, model=model)
            else:
                translate_pdf(input_path, output_path, engine=engine, model=model, layout=layout)
            self.send_pdf(output_path)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    def read_translation_request(self) -> tuple[Path, Path, str, str, str, str, str]:
        ensure_dirs()
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise ValueError(f"ファイルが大きすぎます。上限は{max_mb}MBです。")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        file_item = form["source"] if "source" in form else (form["pdf"] if "pdf" in form else None)
        if file_item is None or not getattr(file_item, "filename", ""):
            raise ValueError("PDFまたはTeXソースを選んでください。")

        original_name = Path(file_item.filename).name
        source_type, input_suffix = detect_source_type(original_name)
        safe_stem = safe_upload_stem(original_name)

        engine = field_value(form, "engine", "nmt").lower()
        model = field_value(form, "model", DEFAULT_NMT_MODEL)
        layout = field_value(form, "layout", "fast")
        if engine not in {"nmt", "ollama", "copy"}:
            raise ValueError("翻訳エンジンが不正です。")

        if source_type == "tex":
            engine = "ollama"
            layout = "tex"
            if not model or model == DEFAULT_NMT_MODEL or model == "copy":
                model = DEFAULT_OLLAMA_MODEL
        elif engine == "nmt":
            model = DEFAULT_NMT_MODEL
            nmt = query_nmt_status()
            if not nmt.get("ready"):
                detail = str(nmt.get("status", "初回準備中です"))
                raise TranslationError(f"翻訳専用NMTモデルを準備中です: {detail}")
        elif engine == "copy":
            model = "copy"

        upload_dir = Path(tempfile.mkdtemp(prefix="upload-", dir=TMP_DIR))
        input_path = upload_dir / f"{safe_stem}{input_suffix}"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(file_item.file, handle)
        return upload_dir, input_path, safe_stem, engine, model, layout, source_type

    def send_pdf(self, path: Path, download: bool = True) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        disposition = "attachment" if download else "inline"
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def field_value(form: cgi.FieldStorage, name: str, default: str) -> str:
    if name not in form:
        return default
    value = form.getfirst(name, default)
    if value is None:
        return default
    return str(value).strip() or default


def detect_source_type(original_name: str) -> tuple[str, str]:
    path = Path(original_name)
    suffixes = tuple(suffix.lower() for suffix in path.suffixes)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf", ".pdf"
    if suffix in TEX_SINGLE_SUFFIXES:
        return "tex", suffix
    if len(suffixes) >= 2 and suffixes[-2:] in TEX_DOUBLE_SUFFIXES:
        return "tex", "".join(suffixes[-2:])
    raise ValueError("PDF、.tex、.zip、.tar、.tar.gz、.tgz のいずれかを選んでください。")


def safe_upload_stem(original_name: str) -> str:
    name = Path(original_name).name
    path = Path(name)
    suffixes = tuple(suffix.lower() for suffix in path.suffixes)
    stem = path.stem
    if len(suffixes) >= 2 and suffixes[-2:] in TEX_DOUBLE_SUFFIXES:
        stem = Path(stem).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe_stem or "paper"


def public_models() -> dict[str, object]:
    installed_models = query_ollama_model_names()
    has_installed_state = bool(installed_models)
    nmt = query_nmt_status()
    return {
        "default": {"engine": "nmt", "model": DEFAULT_NMT_MODEL},
        "ollama_host": OLLAMA_HOST,
        "nmt_host": NMT_HOST,
        "nmt": nmt,
        "engines": [
            {"value": "nmt", "label": "高速NMT（NLLB 600M）", "ready": bool(nmt.get("ready"))},
            {"value": "ollama", "label": "Ollama LLM", "ready": True},
            {"value": "copy", "label": "確認用（翻訳なし）", "ready": True},
        ],
        "nmt_models": [
            {"value": value, "label": label, "default": value == DEFAULT_NMT_MODEL, "installed": bool(nmt.get("ready"))}
            for value, label in NMT_MODEL_OPTIONS
        ],
        "models": [
            {
                "value": value,
                "label": label,
                "default": value == DEFAULT_OLLAMA_MODEL,
                "installed": (value in installed_models) if has_installed_state else None,
            }
            for value, label in OLLAMA_MODEL_OPTIONS
        ],
        "installed_models": sorted(installed_models),
    }


def query_nmt_status() -> dict[str, object]:
    try:
        with urllib.request.urlopen(f"{NMT_HOST}/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return {"ready": False, "status": "NMTコンテナの起動待ち"}
    if not isinstance(payload, dict):
        return {"ready": False, "status": "NMT状態を取得できません"}
    return payload


def query_ollama_model_names() -> set[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return set()
    names: set[str] = set()
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def update_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)


def can_accept_job() -> bool:
    with JOBS_LOCK:
        return can_accept_job_locked()


def can_accept_job_locked() -> bool:
    return len(RUNNING_JOBS) + len(JOB_QUEUE) < MAX_CONCURRENT_JOBS + MAX_QUEUED_JOBS


def dispatch_jobs_locked() -> None:
    blocked_message: str | None = None
    while JOB_QUEUE:
        active_limit = current_concurrent_job_limit_locked()
        if len(RUNNING_JOBS) >= active_limit:
            blocked_message = concurrency_wait_message(active_limit)
            break

        job_id = JOB_QUEUE[0]
        job = JOBS.get(job_id)
        if not job or job.get("status") != "queued":
            JOB_QUEUE.popleft()
            continue

        can_start, blocked_message = can_start_job_locked(job)
        if not can_start:
            break

        JOB_QUEUE.popleft()
        RUNNING_JOBS.add(job_id)
        job.update(
            status="running",
            percent=2,
            message="開始しています",
            started_at=time.time(),
            resource_reserved_mb=estimate_job_memory_mb(
                str(job.get("model", DEFAULT_OLLAMA_MODEL)),
                str(job.get("engine", "ollama")),
            ),
        )
        worker = threading.Thread(target=run_translation_job, args=(job_id,), daemon=True)
        worker.start()
    update_queue_positions_locked(blocked_message=blocked_message)


def update_queue_positions_locked(blocked_message: str | None = None) -> None:
    total = len(JOB_QUEUE)
    for index, job_id in enumerate(JOB_QUEUE, start=1):
        job = JOBS.get(job_id)
        if job and job.get("status") == "queued":
            message = blocked_message if index == 1 and blocked_message else f"待機中: {index}/{total}"
            job.update(message=message, percent=1)


def current_concurrent_job_limit() -> int:
    with JOBS_LOCK:
        return current_concurrent_job_limit_locked()


def current_concurrent_job_limit_locked() -> int:
    if not DYNAMIC_JOB_LIMITS:
        return MAX_CONCURRENT_JOBS
    control = memory_control_snapshot_locked()
    if not control.get("available"):
        return MAX_CONCURRENT_JOBS
    return min(MAX_CONCURRENT_JOBS, int(control.get("dynamic_concurrent_limit", MAX_CONCURRENT_JOBS)))


def can_start_job_locked(job: dict[str, object]) -> tuple[bool, str | None]:
    if len(RUNNING_JOBS) >= MAX_CONCURRENT_JOBS:
        return False, concurrency_wait_message(MAX_CONCURRENT_JOBS)
    engine = str(job.get("engine", "ollama"))
    if engine == "nmt":
        nmt_running = sum(1 for job_id in RUNNING_JOBS if JOBS.get(job_id, {}).get("engine") == "nmt")
        if nmt_running >= NMT_MAX_CONCURRENT_JOBS:
            return False, "NMT翻訳エンジン待機中"
    if not DYNAMIC_JOB_LIMITS:
        return True, None

    control = memory_control_snapshot_locked()
    if not control.get("available"):
        return True, None

    model = str(job.get("model", DEFAULT_OLLAMA_MODEL))
    estimate_mb = estimate_job_memory_mb(model, engine)
    budget_mb = int(control.get("budget_mb", 0))
    effective_used_mb = int(control.get("effective_used_mb", 0))
    projected_mb = effective_used_mb + estimate_mb
    if projected_mb <= budget_mb:
        return True, None

    return (
        False,
        (
            "GPUメモリ待機中: "
            f"使用見込み {mb_to_gb(projected_mb):.1f}GB / 予算 {mb_to_gb(budget_mb):.1f}GB"
        ),
    )


def concurrency_wait_message(active_limit: int) -> str:
    if active_limit <= 0:
        return "GPUメモリ待機中: 空きができたら開始します"
    return f"待機中: 実行枠 {len(RUNNING_JOBS)}/{active_limit}"


def memory_control_snapshot_locked() -> dict[str, object]:
    gpus = query_nvidia_smi_cached()
    total_mb = sum(int(gpu.get("memory_total_mb", 0)) for gpu in gpus)
    used_mb = sum(int(gpu.get("memory_used_mb", 0)) for gpu in gpus)
    free_mb = sum(int(gpu.get("memory_free_mb", 0)) for gpu in gpus)
    source = "nvidia-smi"

    if total_mb <= 0:
        system_memory = query_system_memory()
        if system_memory:
            total_mb = int(system_memory["memory_total_mb"])
            free_mb = int(system_memory["memory_available_mb"])
            used_mb = max(0, total_mb - free_mb)
            source = "system-unified-memory"

    if total_mb <= 0:
        return {
            "enabled": DYNAMIC_JOB_LIMITS,
            "available": False,
            "reason": "memory metrics not available",
            "hard_concurrent_limit": MAX_CONCURRENT_JOBS,
        }

    budget_mb = min(MEMORY_BUDGET_MB, max(0, total_mb - MIN_FREE_MEMORY_MB))
    reserved_mb = running_memory_reservation_locked()
    effective_used_mb = min(total_mb, used_mb + reserved_mb)
    free_in_budget_mb = max(0, budget_mb - effective_used_mb)
    additional_slots = free_in_budget_mb // max(1, DEFAULT_JOB_MEMORY_MB)
    dynamic_limit = min(MAX_CONCURRENT_JOBS, len(RUNNING_JOBS) + additional_slots)

    return {
        "enabled": DYNAMIC_JOB_LIMITS,
        "available": True,
        "source": source,
        "hard_concurrent_limit": MAX_CONCURRENT_JOBS,
        "dynamic_concurrent_limit": int(dynamic_limit),
        "budget_mb": int(budget_mb),
        "budget_gb": round(mb_to_gb(budget_mb), 2),
        "used_mb": int(used_mb),
        "used_gb": round(mb_to_gb(used_mb), 2),
        "free_mb": int(free_mb),
        "free_gb": round(mb_to_gb(free_mb), 2),
        "reserved_mb": int(reserved_mb),
        "reserved_gb": round(mb_to_gb(reserved_mb), 2),
        "effective_used_mb": int(effective_used_mb),
        "effective_used_gb": round(mb_to_gb(effective_used_mb), 2),
        "default_job_reserve_gb": round(mb_to_gb(DEFAULT_JOB_MEMORY_MB), 2),
        "min_free_gb": round(mb_to_gb(MIN_FREE_MEMORY_MB), 2),
    }


def running_memory_reservation_locked() -> int:
    total = 0
    for job_id in RUNNING_JOBS:
        job = JOBS.get(job_id)
        if not job:
            continue
        reserved = int(job.get("resource_reserved_mb", 0) or 0)
        if reserved <= 0:
            reserved = estimate_job_memory_mb(
                str(job.get("model", DEFAULT_OLLAMA_MODEL)),
                str(job.get("engine", "ollama")),
            )
        total += reserved
    return total


def estimate_job_memory_mb(model: str, engine: str = "ollama") -> int:
    if engine == "nmt":
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_NMT_GB", 6)
    normalized = (model or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", normalized)
    params_b = float(match.group(1)) if match else 0.0
    if params_b <= 0:
        return DEFAULT_JOB_MEMORY_MB
    if params_b <= 4:
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_3B_GB", 8)
    if params_b <= 10:
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_7_10B_GB", 18)
    if params_b <= 14:
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_14B_GB", 24)
    if params_b <= 22:
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_20B_GB", 36)
    if params_b <= 34:
        return env_gb_to_mb("READABLE_JOB_MEMORY_RESERVE_30B_GB", 48)
    return DEFAULT_JOB_MEMORY_MB


def mb_to_gb(value_mb: int) -> float:
    return value_mb / 1024


def public_job(job: dict[str, object]) -> dict[str, object]:
    job_id = str(job["id"])
    return {
        "id": job_id,
        "status": job.get("status", "queued"),
        "percent": job.get("percent", 0),
        "message": job.get("message", ""),
        "filename": job.get("filename", ""),
        "engine": job.get("engine", ""),
        "model": job.get("model", ""),
        "layout": job.get("layout", ""),
        "source_type": job.get("source_type", "pdf"),
        "queue_length": len(JOB_QUEUE),
        "running_jobs": len(RUNNING_JOBS),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_concurrent_jobs": current_concurrent_job_limit(),
        "download_url": f"/api/jobs/{job_id}/download",
        "view_url": f"/api/jobs/{job_id}/view",
    }


def recover_completed_job(job_id: str) -> dict[str, object] | None:
    if not RECOVER_OUTPUTS:
        return None
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        return None
    matches = sorted(
        OUTPUT_DIR.glob(f"*_{job_id}_ja.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    output_path = matches[0]
    return {
        "id": job_id,
        "status": "complete",
        "percent": 100,
        "message": "完了済みPDFを見つけました",
        "output_path": str(output_path),
        "filename": output_path.name,
        "engine": "",
        "model": "",
        "layout": "",
        "source_type": "pdf",
        "completed_at": time.time(),
    }


def cleanup_expired_jobs() -> None:
    now = time.time()
    output_paths: list[Path] = []
    upload_dirs: list[Path] = []
    with JOBS_LOCK:
        for job_id, job in list(JOBS.items()):
            status = str(job.get("status", ""))
            completed_at = float(job.get("completed_at", 0) or 0)
            if status not in {"complete", "error"} or not completed_at:
                continue
            if now - completed_at <= JOB_TTL_SECONDS:
                continue
            output_path = Path(str(job.get("output_path", "")))
            upload_dir = Path(str(job.get("upload_dir", "")))
            if output_path:
                output_paths.append(output_path)
            if upload_dir:
                upload_dirs.append(upload_dir)
            JOBS.pop(job_id, None)
    for path in output_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for path in upload_dirs:
        shutil.rmtree(path, ignore_errors=True)


def public_status() -> dict[str, object]:
    with JOBS_LOCK:
        active_concurrent_jobs = current_concurrent_job_limit_locked()
        memory_control = memory_control_snapshot_locked()
        jobs = [
            {
                "id": job_id,
                "status": job.get("status", ""),
                "filename": job.get("filename", ""),
                "message": job.get("message", ""),
                "source_type": job.get("source_type", "pdf"),
            }
            for job_id, job in JOBS.items()
        ]
        queue_length = len(JOB_QUEUE)
        running_jobs = len(RUNNING_JOBS)
    return {
        "host": HOST,
        "port": PORT,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_concurrent_jobs": active_concurrent_jobs,
        "max_queued_jobs": MAX_QUEUED_JOBS,
        "queue_length": queue_length,
        "running_jobs": running_jobs,
        "job_ttl_seconds": JOB_TTL_SECONDS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "memory_control": memory_control,
        "nmt": query_nmt_status(),
        "resources": resource_snapshot(),
        "jobs": jobs,
    }


def resource_snapshot() -> dict[str, object]:
    resources: dict[str, object] = {
        "cpu_count": os.cpu_count() or 0,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
    }
    try:
        usage = shutil.disk_usage(TMP_DIR)
        resources["tmp_free_gb"] = round(usage.free / (1024**3), 2)
    except OSError:
        pass
    system_memory = query_system_memory()
    if system_memory:
        resources["system_memory"] = system_memory
    gpu_info = query_nvidia_smi()
    if gpu_info:
        resources["nvidia_gpu"] = gpu_info
    return resources


def query_system_memory() -> dict[str, object]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {}
    values: dict[str, int] = {}
    try:
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            match = re.search(r"\d+", raw_value)
            if match:
                values[key] = int(match.group(0))
    except OSError:
        return {}

    total_kb = values.get("MemTotal", 0)
    available_kb = values.get("MemAvailable", values.get("MemFree", 0))
    if total_kb <= 0 or available_kb <= 0:
        return {}
    total_mb = total_kb // 1024
    available_mb = available_kb // 1024
    return {
        "memory_total_mb": total_mb,
        "memory_total_gb": round(mb_to_gb(total_mb), 2),
        "memory_available_mb": available_mb,
        "memory_available_gb": round(mb_to_gb(available_mb), 2),
        "memory_used_mb": max(0, total_mb - available_mb),
        "memory_used_gb": round(mb_to_gb(max(0, total_mb - available_mb)), 2),
    }


def query_nvidia_smi_cached() -> list[dict[str, object]]:
    now = time.time()
    checked_at = float(GPU_CACHE.get("checked_at", 0.0) or 0.0)
    if now - checked_at < 2:
        return list(GPU_CACHE.get("items", []))
    items = query_nvidia_smi()
    GPU_CACHE["checked_at"] = now
    GPU_CACHE["items"] = items
    return items


def query_nvidia_smi() -> list[dict[str, object]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, total, used, free, util = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": parse_int(total),
                "memory_used_mb": parse_int(used),
                "memory_free_mb": parse_int(free),
                "utilization_percent": parse_int(util),
            }
        )
    return gpus


def parse_int(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else 0


def run_translation_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))
    input_path = Path(str(job.get("input_path", "")))
    output_path = Path(str(job.get("output_path", "")))
    engine = str(job.get("engine", "ollama"))
    model = str(job.get("model", DEFAULT_OLLAMA_MODEL))
    layout = str(job.get("layout", "fast"))
    source_type = str(job.get("source_type", "pdf"))
    upload_dir = Path(str(job.get("upload_dir", "")))

    def progress(payload: dict[str, object]) -> None:
        update_job(job_id, status="running", **payload)

    try:
        if source_type == "tex":
            translate_tex_project(
                input_path=input_path,
                output_pdf=output_path,
                model=model,
                progress_callback=progress,
            )
        else:
            translate_pdf(
                input_pdf=input_path,
                output_pdf=output_path,
                engine=engine,
                model=model,
                layout=layout,
                progress_callback=progress,
            )
        update_job(job_id, status="complete", percent=100, message="完了しました", completed_at=time.time())
    except Exception as exc:
        traceback.print_exc()
        update_job(
            job_id,
            status="error",
            percent=100,
            message=str(exc) or "処理中にエラーが発生しました。",
            completed_at=time.time(),
        )
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with JOBS_LOCK:
            RUNNING_JOBS.discard(job_id)
            dispatch_jobs_locked()


def main() -> None:
    ensure_dirs()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
