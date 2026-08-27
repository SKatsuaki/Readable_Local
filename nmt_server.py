from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = os.environ.get("NMT_MODEL_ID", "facebook/nllb-200-distilled-600M")
PORT = int(os.environ.get("NMT_PORT", "8767"))
MAX_BATCH_TOKENS = max(256, int(os.environ.get("NMT_MAX_BATCH_TOKENS", "8192")))
MAX_INPUT_TOKENS = max(64, int(os.environ.get("NMT_MAX_INPUT_TOKENS", "384")))
MAX_NEW_TOKENS = max(64, int(os.environ.get("NMT_MAX_NEW_TOKENS", "512")))
NMT_NUM_BEAMS = max(1, int(os.environ.get("NMT_NUM_BEAMS", "1")))
NMT_REPETITION_PENALTY = max(1.0, float(os.environ.get("NMT_REPETITION_PENALTY", "1.12")))
NMT_NO_REPEAT_NGRAM_SIZE = max(0, int(os.environ.get("NMT_NO_REPEAT_NGRAM_SIZE", "4")))
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "jpn_Jpan"


class NMTService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device = ""
        self._attention_implementation = ""
        self._status = "初回準備を開始しています"
        self._error = ""
        threading.Thread(target=self._initialize, daemon=True).start()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self._model is not None and self._tokenizer is not None,
                "status": self._status,
                "error": self._error,
                "model": "nllb-200-distilled-600m",
                "model_id": MODEL_ID,
                "device": self._device,
                "attention_implementation": self._attention_implementation,
                "max_batch_tokens": MAX_BATCH_TOKENS,
                "num_beams": NMT_NUM_BEAMS,
                "repetition_penalty": NMT_REPETITION_PENALTY,
                "no_repeat_ngram_size": NMT_NO_REPEAT_NGRAM_SIZE,
            }

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._status = status
            self._error = error

    def _initialize(self) -> None:
        try:
            self._set_status("NMTモデルを取得しています")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SOURCE_LANGUAGE)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._set_status(f"NMTモデルを{device.upper()}へ読み込んでいます")
            model_kwargs: dict[str, object] = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, attn_implementation="sdpa", **model_kwargs)
            except (TypeError, ValueError):
                model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **model_kwargs)
            model.to(device)
            model.eval()
            with self._lock:
                self._tokenizer = tokenizer
                self._model = model
                self._device = device
                self._attention_implementation = str(getattr(model.config, "_attn_implementation", "eager"))
                self._status = "準備完了"
                self._error = ""
        except Exception as exc:
            traceback.print_exc()
            self._set_status("NMTモデルの準備に失敗しました", str(exc))

    def translate(self, texts: list[str]) -> list[str]:
        with self._lock:
            tokenizer = self._tokenizer
            model = self._model
            device = self._device
        if tokenizer is None or model is None:
            raise RuntimeError("NMTモデルを準備中です")

        pages = [self._split_page(text, tokenizer) for text in texts]
        flattened = [segment for page in pages for segment in page]
        if not flattened:
            return ["" for _ in texts]

        translated_segments: list[str] = []
        with self._inference_lock:
            for batch in self._translation_batches(flattened, tokenizer):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                generation_args: dict[str, object] = {
                    "forced_bos_token_id": tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "num_beams": NMT_NUM_BEAMS,
                    "do_sample": False,
                    "use_cache": True,
                    "repetition_penalty": NMT_REPETITION_PENALTY,
                }
                if NMT_NO_REPEAT_NGRAM_SIZE > 0:
                    generation_args["no_repeat_ngram_size"] = NMT_NO_REPEAT_NGRAM_SIZE
                if NMT_NUM_BEAMS > 1:
                    generation_args["early_stopping"] = True
                with torch.inference_mode():
                    generated = model.generate(**encoded, **generation_args)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for source, translated in zip(batch, decoded):
                    cleaned = self._clean_translation(translated.strip())
                    translated_segments.append(cleaned or source.strip())

        translations: list[str] = []
        cursor = 0
        for page_segments in pages:
            count = len(page_segments)
            translations.append("\n\n".join(part for part in translated_segments[cursor : cursor + count] if part).strip())
            cursor += count
        return translations

    def _translation_batches(self, texts: list[str], tokenizer: object) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            token_count = max(1, self._token_count(text, tokenizer))
            if current and current_tokens + token_count > MAX_BATCH_TOKENS:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += token_count
        if current:
            batches.append(current)
        return batches

    def _split_page(self, text: str, tokenizer: object) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if unit.strip()]
        segments: list[str] = []
        current = ""
        for unit in units:
            for piece in self._split_unit(unit, tokenizer):
                candidate = f"{current} {piece}".strip() if current else piece
                if current and self._token_count(candidate, tokenizer) > MAX_INPUT_TOKENS:
                    segments.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            segments.append(current)
        return segments

    def _split_unit(self, text: str, tokenizer: object) -> list[str]:
        if self._token_count(text, tokenizer) <= MAX_INPUT_TOKENS:
            return [text]
        words = text.split()
        pieces: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if current and self._token_count(candidate, tokenizer) > MAX_INPUT_TOKENS:
                pieces.append(current)
                current = word
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces or [text]

    @staticmethod
    def _token_count(text: str, tokenizer: object) -> int:
        return len(tokenizer.encode(text))

    @staticmethod
    def _clean_translation(text: str) -> str:
        if not text:
            return ""
        for size in range(1, 9):
            pattern = re.compile(rf"([A-Za-z0-9]{{{size}}})(?:\1){{5,}}")
            text = pattern.sub(
                lambda match: match.group(1) if re.search(r"[A-Za-z]", match.group(1)) else match.group(0),
                text,
            )
        text = re.sub(r"\b([A-Za-z][A-Za-z0-9_-]{0,11})(?:\s+\1){5,}\b", r"\1", text)
        return re.sub(r"[ \t]+", " ", text).strip()


SERVICE = NMTService()


class Handler(BaseHTTPRequestHandler):
    server_version = "ReadableNMT/0.1"

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_json(SERVICE.snapshot())

    def do_POST(self) -> None:
        if self.path != "/translate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        snapshot = SERVICE.snapshot()
        if not snapshot["ready"]:
            self.send_json(snapshot, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > 5 * 1024 * 1024:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            texts = payload.get("texts") if isinstance(payload, dict) else None
            if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
                raise ValueError("texts must be a non-empty string list")
            if len(texts) > 32 or sum(len(text) for text in texts) > 300000:
                raise ValueError("translation request is too large")
            translations = SERVICE.translate(texts)
            self.send_json({"translations": translations, "device": snapshot["device"]})
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc) or "translation failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"NMT service listening on 0.0.0.0:{PORT}")
    server.serve_forever()
