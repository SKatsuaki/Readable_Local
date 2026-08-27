from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WORK_DIR = Path(__file__).resolve().parent
TMP_DIR = Path(os.environ.get("READABLE_TMP_DIR", str(WORK_DIR / "tmp" / "pdfs"))).resolve()
OUTPUT_DIR = Path(os.environ.get("READABLE_OUTPUT_DIR", str(WORK_DIR / "output" / "pdf"))).resolve()

FONT_GOTHIC = "HeiseiKakuGo-W5"
FONT_MINCHO = "HeiseiMin-W3"
JAPANESE_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
    Path("/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
]

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass
class PageTranslation:
    page_number: int
    source_text: str
    translated_text: str
    image_path: Path


@dataclass
class TextRegion:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    translatable: bool = True
    translated_text: str = ""


@dataclass
class PageLayoutTranslation:
    page_number: int
    image_path: Path
    width: float
    height: float
    regions: list[TextRegion]


class TranslationError(RuntimeError):
    pass


class BaseTranslator:
    def translate(self, text: str) -> str:
        raise NotImplementedError

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]

    def translate_batch_structured(self, texts: list[str]) -> list[str]:
        return self.translate_batch(texts)


class CopyTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        return "[Copy only]\n" + text.strip()


DEFAULT_OLLAMA_MODEL = "qwen2.5:3b-instruct"
DEFAULT_NMT_MODEL = "nllb-200-distilled-600m"
MAX_LLM_PAGE_CHARS = max(2000, int(os.environ.get("READABLE_LLM_PAGE_CHARS", "6000")))
MAX_OVERLAY_PAGE_CHARS = max(4000, int(os.environ.get("READABLE_OVERLAY_PAGE_CHARS", "12000")))
MAX_OVERLAY_PAGE_ITEMS = max(8, int(os.environ.get("READABLE_OVERLAY_PAGE_ITEMS", "120")))
MAX_OLLAMA_OVERLAY_BATCH_CHARS = max(1200, int(os.environ.get("READABLE_OLLAMA_OVERLAY_BATCH_CHARS", "2200")))
MAX_OLLAMA_OVERLAY_BATCH_ITEMS = max(1, int(os.environ.get("READABLE_OLLAMA_OVERLAY_BATCH_ITEMS", "4")))
DEFAULT_OLLAMA_TIMEOUT_SECONDS = max(60, int(os.environ.get("READABLE_OLLAMA_TIMEOUT_SECONDS", "900")))
DEFAULT_NMT_TIMEOUT_SECONDS = max(60, int(os.environ.get("READABLE_NMT_TIMEOUT_SECONDS", "600")))
OLLAMA_MODEL_OPTIONS: list[tuple[str, str]] = [
    (DEFAULT_OLLAMA_MODEL, "軽量: qwen2.5 3B"),
    ("qwen2.5:3b", "軽量: qwen2.5 3B"),
    ("gemma3:4b", "軽量・多言語: Gemma 3 4B"),
    ("qwen3:4b", "軽量・新しめ: Qwen3 4B"),
    ("qwen2.5:7b-instruct", "中量・翻訳向け: qwen2.5 7B"),
    ("qwen2.5:7b", "中量: qwen2.5 7B"),
    ("qwen3:8b", "中量・新しめ: Qwen3 8B"),
    ("deepseek-r1:7b", "中量・推論重視: DeepSeek R1 7B"),
    ("llama3.1:8b", "中量・汎用: Llama 3.1 8B"),
    ("gemma2:9b", "中量・多言語: Gemma 2 9B"),
    ("gpt-oss:20b", "高品質・遅め: gpt-oss 20B"),
    ("qwen3:30b", "翻訳向け・遅め: Qwen3 30B"),
    ("gemma3:27b", "多言語・比較的安定: Gemma 3 27B"),
    ("deepseek-r1:32b", "推論重視・かなり遅め: DeepSeek R1 32B"),
]
NMT_MODEL_OPTIONS: list[tuple[str, str]] = [
    (DEFAULT_NMT_MODEL, "高速NMT: NLLB 200 Distilled 600M"),
]


class NMTTranslator(BaseTranslator):
    """Calls the in-project NMT service for English to Japanese."""

    def __init__(self, model: str = DEFAULT_NMT_MODEL, host: str | None = None, timeout: int | None = None) -> None:
        self.model = model
        self.host = (host or os.environ.get("NMT_HOST") or "http://127.0.0.1:8767").rstrip("/")
        self.timeout = timeout or DEFAULT_NMT_TIMEOUT_SECONDS

    def translate(self, text: str) -> str:
        results = self.translate_batch([text])
        return results[0] if results else ""

    def translate_batch(self, texts: list[str]) -> list[str]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        request = urllib.request.Request(
            f"{self.host}/translate",
            data=json.dumps({"model": self.model, "texts": cleaned}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except OSError:
                pass
            if exc.code == 503:
                raise TranslationError("翻訳専用NMTモデルを初回準備中です。少し待ってからもう一度実行してください。") from exc
            raise TranslationError(f"NMT翻訳エンジンのエラー: HTTP {exc.code} {detail[:160]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TranslationError("翻訳専用NMTエンジンに接続できません。コンテナの状態を確認してください。") from exc
        except json.JSONDecodeError as exc:
            raise TranslationError("NMT翻訳エンジンの応答を読み取れませんでした。") from exc

        translations = payload.get("translations") if isinstance(payload, dict) else None
        if not isinstance(translations, list) or len(translations) != len(cleaned):
            raise TranslationError("NMT翻訳エンジンから完全な翻訳結果が返りませんでした。")
        normalized: list[str] = []
        for source, translated in zip(cleaned, translations):
            candidate = translated.strip() if isinstance(translated, str) else ""
            normalized.append(normalize_japanese_translation(candidate or source))
        return normalized


class OllamaTranslator(BaseTranslator):
    def __init__(self, model: str, host: str | None = None, timeout: int | None = None) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = timeout or DEFAULT_OLLAMA_TIMEOUT_SECONDS

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        source_text = text.strip()
        user_prompt = (
            "次の<source>内の英文を自然な日本語に翻訳してください。"
            "翻訳文だけを返し、翻訳が終わったらそこで停止してください。\n\n"
            f"<source>\n{source_text}\n</source>"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": translation_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": context_limit_for_chars(len(source_text), self.model),
                "num_predict": prediction_limit_for(source_text, self.model),
            },
        }
        if is_reasoning_model(self.model):
            payload["think"] = False
        data = self.chat(payload)
        translated = ollama_message_content(data)
        if not translated and is_reasoning_model(self.model):
            payload["options"]["num_predict"] = retry_prediction_limit_for(source_text, self.model)
            data = self.chat(payload)
            translated = ollama_message_content(data)

        if not translated:
            raise TranslationError(f"Ollamaから翻訳結果が返りませんでした。{ollama_empty_response_hint(data)}")
        return normalize_japanese_translation(translated)

    def translate_batch(self, texts: list[str]) -> list[str]:
        return self._translate_batch(texts, fallback_to_single=True, structured=False)

    def translate_batch_structured(self, texts: list[str]) -> list[str]:
        return self._translate_batch(texts, fallback_to_single=False, structured=True)

    def _translate_batch(self, texts: list[str], *, fallback_to_single: bool, structured: bool) -> list[str]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        if len(cleaned) == 1 and fallback_to_single:
            return [self.translate(cleaned[0])]

        user_prompt = (
            "次のJSON配列にある各textを英語論文として自然な日本語に翻訳してください。"
            f"出力はJSONオブジェクトだけにしてください。形式は必ず "
            f'{{"translations": ["翻訳1", "翻訳2"]}} です。'
            f"translationsは入力と同じ{len(cleaned)}個・同じ順序にしてください。"
            "固有名詞、数式、引用番号、単位は必要な範囲で原文のまま残してください。"
            "説明、注釈、Markdown、前置きは出力しないでください。\n\n"
            f"{json.dumps([{'id': index, 'text': text} for index, text in enumerate(cleaned)], ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": translation_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": context_limit_for_chars(sum(len(text) for text in cleaned), self.model),
                "num_predict": (
                    structured_batch_prediction_limit_for(cleaned, self.model)
                    if structured
                    else batch_prediction_limit_for(cleaned, self.model)
                ),
            },
        }
        if is_reasoning_model(self.model):
            payload["think"] = False

        data = self.chat(payload)
        parsed = parse_translation_batch(ollama_message_content(data), expected_count=len(cleaned))
        if parsed is None and is_reasoning_model(self.model):
            payload["options"]["num_predict"] = (
                retry_structured_batch_prediction_limit_for(cleaned, self.model)
                if structured
                else retry_batch_prediction_limit_for(cleaned, self.model)
            )
            data = self.chat(payload)
            parsed = parse_translation_batch(ollama_message_content(data), expected_count=len(cleaned))
        if parsed is None:
            if not fallback_to_single:
                raise TranslationError("Ollamaのページ単位翻訳をJSONとして読み取れませんでした。")
            return [self.translate(text) for text in cleaned]
        return [normalize_japanese_translation(text) for text in parsed]

    def chat(self, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            if exc.code == 404:
                raise TranslationError(
                    f"Ollamaモデルが見つかりません: {self.model}。./scripts/pull_model.sh {self.model} を実行してください。"
                ) from exc
            message = f"Ollama APIエラー: HTTP {exc.code}"
            if detail:
                message += f" {detail[:240]}"
            raise TranslationError(message) from exc
        except urllib.error.URLError as exc:
            raise TranslationError(
                "Ollamaに接続できません。./scripts/start_ollama.sh を起動してからもう一度試してください。"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TranslationError("翻訳が時間内に終わりませんでした。自動で分割して再試行します。") from exc


def ollama_message_content(data: dict[str, object]) -> str:
    message = data.get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    return content.strip() if isinstance(content, str) else ""


def ollama_empty_response_hint(data: dict[str, object]) -> str:
    done_reason = data.get("done_reason")
    message = data.get("message", {})
    thinking = ""
    if isinstance(message, dict):
        raw_thinking = message.get("thinking", "")
        if isinstance(raw_thinking, str):
            thinking = raw_thinking
    details: list[str] = []
    if done_reason:
        details.append(f"終了理由: {done_reason}")
    if thinking:
        details.append(f"thinkingのみ返りました({len(thinking)}文字)。")
    return " ".join(details).strip()


def translation_system_prompt() -> str:
    return (
        "あなたは英語論文を日本語へ翻訳する専門翻訳者です。"
        "出力は自然な日本語だけにしてください。"
        "普通名詞や一般的な英語表現は必ず日本語に訳し、"
        "固有名詞、数式、引用番号、単位だけ原文のまま残してください。"
        "文中の数式変数はASCIIラテン文字か通常のギリシャ文字で保持し、"
        "数学用太字・斜体Unicodeは使わないでください。"
        "Figure 1は図1、Table 1は表1のように日本語表記へ変えてください。"
        "中国語の簡体字は使わず、日本語の漢字とかなを使ってください。"
        "論文と表記し、论文とは書かないでください。"
        "説明、要約、注釈、繰り返し文は追加しないでください。"
    )


def is_reasoning_model(model: str | None) -> bool:
    normalized = (model or "").lower()
    return normalized.startswith(("gpt-oss", "qwen3", "deepseek-r1"))


def prediction_limit_for(text: str, model: str | None = None) -> int:
    if is_reasoning_model(model):
        return min(4096, max(2048, int(len(text) * 2.2) + 1024))
    return min(6144, max(1024, int(len(text) * 0.55) + 800))


def retry_prediction_limit_for(text: str, model: str | None = None) -> int:
    if is_reasoning_model(model):
        return min(8192, max(4096, int(len(text) * 3.0) + 2048))
    return min(2400, max(900, int(len(text) * 1.8)))


def context_limit_for_chars(total_chars: int, model: str | None = None) -> int:
    if is_reasoning_model(model):
        return min(32768, max(8192, int(total_chars * 1.4) + 8192))
    return min(32768, max(8192, int(total_chars * 1.2) + 8192))


def batch_prediction_limit_for(texts: list[str], model: str | None = None) -> int:
    total_chars = sum(len(text) for text in texts)
    if is_reasoning_model(model):
        return min(16384, max(3072, int(total_chars * 1.8) + 1024))
    return min(12000, max(1800, int(total_chars * 1.25) + 1024))


def structured_batch_prediction_limit_for(texts: list[str], model: str | None = None) -> int:
    total_chars = sum(len(text) for text in texts)
    if is_reasoning_model(model):
        return min(8192, max(1536, int(total_chars * 1.3) + 768))
    return min(4096, max(512, int(total_chars * 0.9) + 512))


def retry_batch_prediction_limit_for(texts: list[str], model: str | None = None) -> int:
    total_chars = sum(len(text) for text in texts)
    if is_reasoning_model(model):
        return min(20000, max(6144, int(total_chars * 2.4) + 2048))
    return min(16000, max(3000, int(total_chars * 1.6) + 2048))


def retry_structured_batch_prediction_limit_for(texts: list[str], model: str | None = None) -> int:
    total_chars = sum(len(text) for text in texts)
    if is_reasoning_model(model):
        return min(12000, max(3072, int(total_chars * 1.7) + 1024))
    return min(6000, max(900, int(total_chars * 1.15) + 900))


def parse_translation_batch(content: str, expected_count: int) -> list[str] | None:
    value = parse_json_like(content)
    translations: object
    if isinstance(value, list):
        translations = value
    elif isinstance(value, dict):
        translations = (
            value.get("translations")
            or value.get("translated")
            or value.get("items")
            or value.get("results")
            or value.get("text")
            or value.get("output")
        )
        if isinstance(translations, str) and expected_count == 1:
            translations = [translations]
        if translations is None:
            numeric_items = [
                value[key]
                for key in sorted(value.keys(), key=lambda item: int(item) if str(item).isdigit() else 10**9)
                if str(key).isdigit()
            ]
            translations = numeric_items or None
    else:
        return None

    if not isinstance(translations, list) or len(translations) != expected_count:
        return None

    parsed: list[str] = []
    for item in translations:
        if isinstance(item, str):
            parsed.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("translation") or item.get("ja") or item.get("text")
            if not isinstance(text, str):
                return None
            parsed.append(text.strip())
        else:
            return None
    if any(not text for text in parsed):
        return None
    return parsed


def parse_json_like(content: str) -> object | None:
    cleaned = strip_markdown_fence(content.strip())
    for candidate in json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def json_candidates(text: str) -> list[str]:
    candidates = [text]
    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end > list_start:
        candidates.append(text[list_start : list_end + 1])
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])
    return candidates


def normalize_japanese_translation(text: str) -> str:
    text = normalize_math_symbols(text)
    replacements = {
        "论文": "論文",
        "图": "図",
        "公式": "式",
        "引用文献": "参考文献",
        "学術記事": "学術論文",
        "地域PDF": "ローカルPDF",
        "地元": "ローカル",
        "-figure": "図",
        "Figure": "図",
        "figure": "図",
        "Table": "表",
        "table": "表",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = collapse_repeated_runs(text)
    return dedupe_repeated_blocks(text).strip()


def normalize_math_symbols(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "−": "-",
        "–": "-",
        "—": "-",
        "∕": "/",
        "⁄": "/",
        "∗": "*",
        "⋅": "·",
        "∙": "·",
        "⟨": "<",
        "⟩": ">",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def collapse_repeated_runs(text: str) -> str:
    if not text:
        return text
    for size in range(1, 9):
        pattern = re.compile(rf"([A-Za-z0-9]{{{size}}})(?:\1){{5,}}")
        text = pattern.sub(
            lambda match: match.group(1) if re.search(r"[A-Za-z]", match.group(1)) else match.group(0),
            text,
        )
    text = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_-]{0,11})(?:\s+\1){5,}\b",
        r"\1",
        text,
    )
    return text


def dedupe_repeated_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if not blocks:
        return text.strip()
    output: list[str] = []
    seen_recent: list[str] = []
    for block in blocks:
        compact = re.sub(r"\s+", "", block)
        if compact and compact in seen_recent:
            continue
        output.append(block)
        seen_recent.append(compact)
        seen_recent = seen_recent[-6:]
    return "\n\n".join(output)


def translation_batch_limits(translator: BaseTranslator, mode: str = "quality") -> tuple[int, int]:
    if isinstance(translator, NMTTranslator):
        return (12, 200000) if mode in {"fast", "page"} else (8, 100000)
    if mode == "page" and isinstance(translator, OllamaTranslator):
        return 1, MAX_LLM_PAGE_CHARS
    if mode == "fast":
        if isinstance(translator, OllamaTranslator) and is_reasoning_model(translator.model):
            return 32, 10000
        if isinstance(translator, OllamaTranslator):
            return 48, 14000
        return 24, 8000
    if isinstance(translator, OllamaTranslator) and is_reasoning_model(translator.model):
        return 24, 6000
    if isinstance(translator, OllamaTranslator):
        return 24, 7000
    return 16, 4800


def iter_translation_batches(texts: list[str], translator: BaseTranslator, mode: str = "quality") -> list[list[str]]:
    max_items, max_chars = translation_batch_limits(translator, mode)
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        text_chars = len(text)
        if current and (len(current) >= max_items or current_chars + text_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_chars
    if current:
        batches.append(current)
    return batches


def build_translator(engine: str, model: str | None) -> BaseTranslator:
    normalized = engine.lower().strip()
    if normalized == "copy":
        return CopyTranslator()
    if normalized == "nmt":
        return NMTTranslator(model=model or DEFAULT_NMT_MODEL)
    if normalized == "ollama":
        return OllamaTranslator(model=model or DEFAULT_OLLAMA_MODEL)
    raise ValueError(f"Unknown translation engine: {engine}")


def register_fonts() -> None:
    font_path = japanese_font_path()
    for name in (FONT_GOTHIC, FONT_MINCHO):
        try:
            pdfmetrics.getFont(name)
        except KeyError:
            if font_path is not None:
                try:
                    pdfmetrics.registerFont(TTFont(name, str(font_path), subfontIndex=0))
                    continue
                except Exception:
                    pass
            pdfmetrics.registerFont(UnicodeCIDFont(name))


def japanese_font_path() -> Path | None:
    for path in JAPANESE_FONT_CANDIDATES:
        if path.exists():
            return path
    return None


def ensure_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    percent: int | None = None,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return
    payload: dict[str, object] = {"stage": stage, "message": message}
    if percent is not None:
        payload["percent"] = max(0, min(100, percent))
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    progress_callback(payload)


def command_path(name: str) -> str:
    bundled_bin = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "bin"
    path = os.environ.get("PATH", "")
    search_path = f"{bundled_bin}{os.pathsep}{path}"
    found = shutil.which(name, path=search_path)
    if not found:
        raise RuntimeError(f"{name} が見つかりません。Popplerをインストールしてください。")
    return found


def render_pdf_pages(input_pdf: Path, work_dir: Path, dpi: int = 120) -> list[Path]:
    prefix = work_dir / "page"
    pdftoppm = command_path("pdftoppm")
    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(input_pdf), str(prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths = [Path(p) for p in sorted(glob.glob(str(prefix) + "-*.png"))]
    if not paths:
        raise RuntimeError("PDFページの画像化に失敗しました。")
    return paths


def extract_text_pages(input_pdf: Path) -> list[str]:
    texts: list[str] = []
    with pdfplumber.open(str(input_pdf)) as pdf:
        for page in pdf.pages:
            texts.append(extract_page_text(page))
    return texts


def extract_layout_pages(input_pdf: Path, page_images: list[Path]) -> list[PageLayoutTranslation]:
    pages: list[PageLayoutTranslation] = []
    with pdfplumber.open(str(input_pdf)) as pdf:
        for index, page in enumerate(pdf.pages):
            image_path = page_images[index] if index < len(page_images) else page_images[-1]
            words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
            upright_words = [word for word in words if word.get("upright", True)]
            lines = words_to_lines(upright_words, split_gaps=True)
            regions = lines_to_regions(lines, float(page.width), float(page.height))
            page_text = " ".join(str(line.get("text", "")) for line in lines)
            if should_preserve_page(page_text):
                for region in regions:
                    region.translatable = False
            pages.append(
                PageLayoutTranslation(
                    page_number=index + 1,
                    image_path=image_path,
                    width=float(page.width),
                    height=float(page.height),
                    regions=regions,
                )
            )
    return pages


def should_preserve_page(text: str) -> bool:
    if not text.strip():
        return False
    if looks_japanese_page(text):
        return True
    if looks_garbled_text(text):
        return True
    return False


def looks_japanese_page(text: str) -> bool:
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = japanese + latin
    return total >= 80 and japanese / total >= 0.35


def looks_garbled_text(text: str) -> bool:
    if len(re.findall(r"\(cid:\d+\)", text)) >= 3 or text.count("(cid:") >= 3:
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 80:
        return False
    readable = len(re.findall(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff.,;:!?()[\]{}%/+=<>\-−–—_&@#'\"’“”°・、。]", compact))
    return readable / len(compact) < 0.62


def extract_page_text(page: object) -> str:
    words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
    if words:
        lines = words_to_lines(words)
        raw_lines = words_to_lines(words, split_gaps=False)
        if looks_two_column(lines, float(page.width)):
            column_text = extract_column_text_from_words(words, float(page.width), raw_lines)
            if column_text:
                return clean_extracted_text(column_text)
        column_text = extract_column_ordered_text(lines, float(page.width))
        if column_text:
            return clean_extracted_text(column_text)
    raw = page.extract_text(x_tolerance=1, y_tolerance=3, layout=True) or ""
    return clean_extracted_text(raw)


def words_to_lines(words: list[dict], split_gaps: bool = True) -> list[dict]:
    sorted_words = sorted(words, key=lambda item: (float(item["top"]), float(item["x0"])))
    lines: list[dict] = []
    tolerance = 3.0
    for word in sorted_words:
        top = float(word["top"])
        if not lines or abs(top - lines[-1]["top"]) > tolerance:
            lines.append({"top": top, "words": [word]})
        else:
            lines[-1]["words"].append(word)

    normalized: list[dict] = []
    for line in lines:
        line_words = sorted(line["words"], key=lambda item: float(item["x0"]))
        segments: list[list[dict]] = []
        for word in line_words:
            if not segments:
                segments.append([word])
                continue
            previous = segments[-1][-1]
            gap = float(word["x0"]) - float(previous["x1"])
            if split_gaps and gap > 10:
                segments.append([word])
            else:
                segments[-1].append(word)

        for segment in segments:
            text = " ".join(str(item["text"]) for item in segment).strip()
            if not text:
                continue
            normalized.append(
                {
                    "top": min(float(item["top"]) for item in segment),
                    "bottom": max(float(item.get("bottom", item["top"])) for item in segment),
                    "x0": min(float(item["x0"]) for item in segment),
                    "x1": max(float(item["x1"]) for item in segment),
                    "text": text,
                }
            )
    return normalized


def extract_column_ordered_text(lines: list[dict], page_width: float) -> str:
    if not lines:
        return ""

    mid = page_width / 2
    left = [line for line in lines if (float(line["x0"]) + float(line["x1"])) / 2 < mid]
    right = [line for line in lines if (float(line["x0"]) + float(line["x1"])) / 2 >= mid]
    total = len(lines)
    if not looks_two_column(lines, page_width):
        return "\n".join(str(line["text"]) for line in sorted(lines, key=lambda item: item["top"]))

    blocks = []
    for column in (left, right):
        ordered = sorted(column, key=lambda item: item["top"])
        if ordered:
            blocks.append("\n".join(str(line["text"]) for line in ordered))
    return "\n\n".join(blocks)


def looks_two_column(lines: list[dict], page_width: float) -> bool:
    if not lines:
        return False
    mid = page_width / 2
    left = [line for line in lines if (float(line["x0"]) + float(line["x1"])) / 2 < mid]
    right = [line for line in lines if (float(line["x0"]) + float(line["x1"])) / 2 >= mid]
    total = len(lines)
    return len(left) >= 4 and len(right) >= 2 and min(len(left), len(right)) / total > 0.15


def extract_column_text_from_words(words: list[dict], page_width: float, lines: list[dict]) -> str:
    mid = page_width / 2
    top = min(float(line["top"]) for line in lines) if lines else 0
    header_lines = [
        line
        for line in lines
        if float(line["top"]) <= top + 28
        and float(line["x0"]) < page_width * 0.25
        and float(line["x1"]) > page_width * 0.55
    ]
    header_tops = [float(line["top"]) for line in header_lines]
    body_words = [
        word
        for word in words
        if not any(abs(float(word["top"]) - header_top) <= 3 for header_top in header_tops)
    ]
    left_words = [word for word in body_words if float(word["x0"]) < mid]
    right_words = [word for word in body_words if float(word["x0"]) >= mid]
    blocks = []
    if header_lines:
        blocks.append("\n".join(str(line["text"]) for line in sorted(header_lines, key=lambda item: item["top"])))
    for column_words in (left_words, right_words):
        column_lines = words_to_lines(column_words, split_gaps=False)
        ordered = sorted(column_lines, key=lambda item: item["top"])
        if ordered:
            blocks.append("\n".join(str(line["text"]) for line in ordered))
    return "\n\n".join(blocks)


def lines_to_regions(lines: list[dict], page_width: float, page_height: float) -> list[TextRegion]:
    visible = [
        line
        for line in lines
        if str(line.get("text", "")).strip()
        and not is_margin_artifact(line, page_width, page_height)
    ]
    if not visible:
        return []

    mid = page_width / 2
    buckets: dict[str, list[dict]] = {"full": [], "left": [], "right": []}
    two_column = looks_two_column(visible, page_width)
    for line in visible:
        width = float(line["x1"]) - float(line["x0"])
        center = (float(line["x0"]) + float(line["x1"])) / 2
        if not two_column or width > page_width * 0.58:
            buckets["full"].append(line)
        elif center < mid:
            buckets["left"].append(line)
        else:
            buckets["right"].append(line)

    regions: list[TextRegion] = []
    for bucket_name in ("full", "left", "right"):
        regions.extend(group_lines_into_regions(buckets[bucket_name], page_width, page_height))
    return sorted(regions, key=lambda item: (item.top, item.x0))


def is_margin_artifact(line: dict, page_width: float, page_height: float) -> bool:
    text = str(line.get("text", "")).strip()
    x0 = float(line["x0"])
    x1 = float(line["x1"])
    top = float(line["top"])
    bottom = float(line.get("bottom", top))
    if x1 < 45 and top > page_height * 0.18:
        return True
    if x0 > page_width - 45 and top > page_height * 0.18:
        return True
    if x1 - x0 < 28 and not re.search(r"[A-Za-z]", text):
        return True
    if bottom < 24 or top > page_height - 24:
        return True
    return False


def group_lines_into_regions(lines: list[dict], page_width: float, page_height: float) -> list[TextRegion]:
    regions: list[TextRegion] = []
    current: list[dict] = []
    for line in sorted(lines, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not current:
            current = [line]
            continue
        previous = current[-1]
        gap = float(line["top"]) - float(previous.get("bottom", previous["top"]))
        x_shift = abs(float(line["x0"]) - float(previous["x0"]))
        same_flow = gap <= 4.8 and x_shift < max(28.0, page_width * 0.08)
        if same_flow:
            current.append(line)
        else:
            regions.append(region_from_lines(current, page_width, page_height))
            current = [line]
    if current:
        regions.append(region_from_lines(current, page_width, page_height))
    return [region for region in regions if region.text]


def region_from_lines(lines: list[dict], page_width: float, page_height: float) -> TextRegion:
    text = join_region_lines(lines)
    x0 = max(0.0, min(float(line["x0"]) for line in lines) - 1.5)
    top = max(0.0, min(float(line["top"]) for line in lines) - 1.2)
    x1 = min(page_width, max(float(line["x1"]) for line in lines) + 1.5)
    bottom = min(page_height, max(float(line.get("bottom", line["top"])) for line in lines) + 1.2)
    return TextRegion(
        text=text,
        x0=x0,
        top=top,
        x1=x1,
        bottom=bottom,
        translatable=is_translatable_region(text, x0, top, x1, bottom, page_width),
    )


def join_region_lines(lines: list[dict]) -> str:
    cleaned = clean_extracted_text("\n".join(str(line["text"]) for line in lines))
    return re.sub(r"\s*\n\s*", " ", cleaned).strip()


def is_translatable_region(text: str, x0: float, top: float, x1: float, bottom: float, page_width: float) -> bool:
    stripped = text.strip()
    if not stripped or not re.search(r"[A-Za-z]", stripped):
        return False
    if "(cid:" in stripped:
        return False
    if "@" in stripped or stripped.lower().startswith(("http://", "https://", "arxiv:")):
        return False
    if re.search(r"https?://", stripped) and len(re.findall(r"[A-Za-z][A-Za-z-]*", stripped)) <= 4:
        return False
    if "all rights reserved" in stripped.lower() or stripped.startswith("©"):
        return False
    if top < 180 and re.fullmatch(r"[A-Z][A-Za-z0-9,.\-* ]{1,60}", stripped) and len(stripped.split()) <= 2:
        return False
    if looks_author_or_affiliation(stripped):
        return False
    if looks_equationish(stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]*", stripped)
    if len(words) < 2 and len(stripped) < 3:
        return False
    if (x1 - x0) < page_width * 0.06 and len(stripped) > 24:
        return False
    return True


def looks_author_or_affiliation(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    lowered = compact.lower()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", compact)
    if len(words) < 3:
        return False
    capitalized = sum(1 for word in words if word[:1].isupper())
    connectors = {"and", "of", "for", "the", "in"}
    non_connectors = [word for word in words if word.lower() not in connectors]
    has_sentence_punctuation = bool(re.search(r"[.!?;:]", compact))
    affiliation_terms = ("university", "institute", "department", "laboratory", "authors", "affiliation")
    if any(token in lowered for token in affiliation_terms) and not has_sentence_punctuation and len(compact) < 180:
        return True
    return (
        len(non_connectors) >= 3
        and capitalized / max(1, len(words)) >= 0.72
        and not has_sentence_punctuation
        and len(compact) < 180
    )


def looks_equationish(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    math_chars = set("=<>±×÷∑∫√∞∂∇→←↔≤≥{}[]_^|")
    math_count = sum(1 for char in compact if char in math_chars or char.isdigit())
    words = re.findall(r"[A-Za-z][A-Za-z-]*", text)
    math_names = {
        "l",
        "p",
        "d",
        "z",
        "s",
        "t",
        "x",
        "y",
        "m",
        "qt",
        "det",
        "cls",
        "reg",
        "tra",
        "vae",
        "mse",
        "col",
        "bd",
        "gp",
        "ce",
        "mcls",
        "mreg",
        "sin",
        "cos",
        "tan",
        "log",
        "exp",
        "sqrt",
        "softmax",
        "argmax",
    }
    lowered = [word.lower().replace("-", "") for word in words]
    math_like_words = sum(1 for word in lowered if word in math_names or len(word) == 1)
    long_words = sum(1 for word in lowered if len(word) >= 6)
    if any(char in compact for char in math_chars) and len(words) <= 8:
        return True
    if math_like_words >= 2 and long_words <= 2:
        return True
    if math_like_words >= 1 and len(words) <= 4 and not re.search(r"[.!?]", text):
        return True
    return math_count / max(1, len(compact)) > 0.45 and len(words) <= 10


def clean_extracted_text(text: str) -> str:
    text = normalize_math_symbols(text)
    lines = [line.rstrip() for line in text.replace("\r", "\n").splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        compact = " ".join(line.split())
        if not compact:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        if cleaned and cleaned[-1].endswith("-") and compact[:1].islower():
            cleaned[-1] = cleaned[-1][:-1] + compact
        else:
            cleaned.append(compact)
        previous_blank = False
    return "\n".join(cleaned).strip()


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            pieces = textwrap.wrap(paragraph, width=max_chars, break_long_words=False, break_on_hyphens=False)
            chunks.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""
    if current:
        chunks.append(current)
    return chunks


def page_translation_chunks(text: str, translator: BaseTranslator, mode: str) -> list[str]:
    """Keep LLM calls at roughly one request per page in reading layouts."""
    if not text.strip():
        return []
    if isinstance(translator, NMTTranslator):
        # The NMT service performs its own token-aware sentence batching on the GPU.
        return [text.strip()]
    if isinstance(translator, OllamaTranslator):
        return chunk_text(text, max_chars=MAX_LLM_PAGE_CHARS)
    chunk_size = 2400 if mode == "fast" else 1200
    return chunk_text(text, max_chars=chunk_size)


def translate_page_text(text: str, translator: BaseTranslator) -> str:
    if not text.strip():
        return "このページから本文を抽出できませんでした。"
    translated_text, _ = translate_chunks(chunk_text(text), translator)
    return translated_text


def translate_batch_resilient(
    batch: list[str],
    translator: BaseTranslator,
    *,
    mode: str = "quality",
    structured: bool = False,
) -> list[str]:
    try:
        method = translator.translate_batch_structured if structured else translator.translate_batch
        results = method(batch)
        if len(results) != len(batch):
            raise TranslationError("翻訳結果の数が入力と一致しませんでした。")
        return results
    except TranslationError:
        if not isinstance(translator, OllamaTranslator):
            raise
        if len(batch) > 1:
            mid = max(1, len(batch) // 2)
            translated_parts: list[str] = []
            for part in (batch[:mid], batch[mid:]):
                try:
                    translated_parts.extend(
                        translate_batch_resilient(
                            part,
                            translator,
                            mode=mode,
                            structured=structured,
                        )
                    )
                except TranslationError:
                    translated_parts.extend(part)
            return translated_parts
        if not batch:
            return []

        source_text = batch[0]
        if structured:
            try:
                return [translator.translate(source_text)]
            except TranslationError:
                pass
        if len(source_text) <= 1200:
            return [source_text]
        split_size = max(900, min(MAX_LLM_PAGE_CHARS // 2, max(900, len(source_text) // 2)))
        pieces = chunk_text(source_text, max_chars=split_size)
        if len(pieces) <= 1:
            return [source_text]
        translated_pieces: list[str] = []
        for smaller_batch in iter_translation_batches(pieces, translator, "quality"):
            translated_pieces.extend(
                translate_batch_resilient(
                    smaller_batch,
                    translator,
                    mode="quality",
                    structured=False,
                )
            )
        return ["\n\n".join(part.strip() for part in translated_pieces if part.strip())]


def translate_chunks(
    chunks: list[str],
    translator: BaseTranslator,
    progress_callback: ProgressCallback | None = None,
    done: int = 0,
    total: int = 0,
    page_number: int = 0,
    mode: str = "quality",
) -> tuple[str, int]:
    translated_chunks = []
    for batch in iter_translation_batches(chunks, translator, mode):
        batch_results = translate_batch_resilient(batch, translator, mode=mode)
        for translated in batch_results:
            translated_chunks.append(translated)
            done += 1
            if total:
                percent = 8 + int(done / total * 84)
                emit_progress(
                    progress_callback,
                    stage="translating",
                    message=f"翻訳中: {done}/{total}",
                    percent=percent,
                    current=done,
                    total=total,
                )
            elif page_number:
                emit_progress(
                    progress_callback,
                    stage="translating",
                    message=f"{page_number}ページ目を翻訳中",
                )
        time.sleep(0.02)
    return "\n\n".join(part.strip() for part in translated_chunks if part.strip()), done


def translate_page_chunks_batched(
    page_chunks: list[list[str]],
    translator: BaseTranslator,
    progress_callback: ProgressCallback | None = None,
    mode: str = "quality",
) -> list[str]:
    flat: list[tuple[int, str]] = []
    for page_index, chunks in enumerate(page_chunks):
        for chunk in chunks:
            if chunk.strip():
                flat.append((page_index, chunk))

    translated_by_page: list[list[str]] = [[] for _ in page_chunks]
    if not flat:
        return ["このページから本文を抽出できませんでした。" for _ in page_chunks]

    texts = [text for _, text in flat]
    total = len(texts)
    done = 0
    cursor = 0
    for batch in iter_translation_batches(texts, translator, mode):
        batch_results = translate_batch_resilient(batch, translator, mode=mode)
        for offset, translated in enumerate(batch_results):
            page_index = flat[cursor + offset][0]
            translated_by_page[page_index].append(translated)
            done += 1
        cursor += len(batch)
        percent = 8 + int(done / total * 84)
        message = "高速本文翻訳中" if mode == "fast" else "本文翻訳中"
        emit_progress(
            progress_callback,
            stage="translating",
            message=f"{message}: {done}/{total}",
            percent=percent,
            current=done,
            total=total,
        )
        time.sleep(0.02)

    return [
        "\n\n".join(part.strip() for part in parts if part.strip())
        or "このページから本文を抽出できませんでした。"
        for parts in translated_by_page
    ]


def translate_layout_regions_pagewise(
    pages: list[PageLayoutTranslation],
    translator: BaseTranslator,
    progress_callback: ProgressCallback | None = None,
) -> None:
    page_groups: list[tuple[PageLayoutTranslation, list[list[TextRegion]]]] = []
    for page in pages:
        regions = [region for region in page.regions if region.translatable and region.text.strip()]
        if regions:
            page_groups.append((page, iter_overlay_region_batches(regions, translator)))
    total_batches = sum(len(batches) for _, batches in page_groups)
    if total_batches == 0:
        return

    done_batches = 0
    for page, batches in page_groups:
        for batch_index, batch_regions in enumerate(batches, start=1):
            sources = [region.text for region in batch_regions]
            try:
                translated = translate_batch_resilient(
                    sources,
                    translator,
                    mode="quality",
                    structured=True,
                )
            except TranslationError:
                translated = []
                for source in sources:
                    try:
                        translated.extend(
                            translate_batch_resilient(
                                [source],
                                translator,
                                mode="quality",
                                structured=False,
                            )
                        )
                    except TranslationError:
                        translated.append(source)
            for region, translated_text in zip(batch_regions, translated):
                region.translated_text = translated_text or region.text

            done_batches += 1
            percent = 8 + int(done_batches / total_batches * 84)
            emit_progress(
                progress_callback,
                stage="translating",
                message=(
                    f"{page.page_number}ページ目をレイアウト翻訳中: "
                    f"{batch_index}/{len(batches)}"
                ),
                percent=percent,
                current=done_batches,
                total=total_batches,
            )
            time.sleep(0.02)


def iter_overlay_region_batches(regions: list[TextRegion], translator: BaseTranslator) -> list[list[TextRegion]]:
    max_items = MAX_OVERLAY_PAGE_ITEMS
    max_chars = MAX_OVERLAY_PAGE_CHARS
    if isinstance(translator, OllamaTranslator):
        max_items = min(max_items, MAX_OLLAMA_OVERLAY_BATCH_ITEMS)
        max_chars = min(max_chars, MAX_OLLAMA_OVERLAY_BATCH_CHARS)
    batches: list[list[TextRegion]] = []
    current: list[TextRegion] = []
    current_chars = 0
    for region in regions:
        text_chars = len(region.text)
        if current and (
            len(current) >= max_items
            or current_chars + text_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(region)
        current_chars += text_chars
    if current:
        batches.append(current)
    return batches


def translate_layout_regions(
    pages: list[PageLayoutTranslation],
    translator: BaseTranslator,
    progress_callback: ProgressCallback | None = None,
    mode: str = "quality",
) -> None:
    translatable_regions = [
        region
        for page in pages
        for region in page.regions
        if region.translatable and region.text.strip()
    ]
    total = len(translatable_regions)
    done = 0
    if total == 0:
        return
    if isinstance(translator, OllamaTranslator) and mode == "page":
        translate_layout_regions_pagewise(pages, translator, progress_callback)
        return
    if isinstance(translator, CopyTranslator):
        for region in translatable_regions:
            region.translated_text = region.text
            done += 1
            percent = 8 + int(done / total * 84)
            emit_progress(
                progress_callback,
                stage="translating",
                message=f"レイアウト確認中: {done}/{total}",
                percent=percent,
                current=done,
                total=total,
            )
        return

    grouped_regions: dict[str, list[TextRegion]] = {}
    source_by_key: dict[str, str] = {}
    for region in translatable_regions:
        cache_key = re.sub(r"\s+", " ", region.text).strip()
        if cache_key not in grouped_regions:
            grouped_regions[cache_key] = []
            source_by_key[cache_key] = region.text
        grouped_regions[cache_key].append(region)

    keys = list(grouped_regions.keys())
    for batch_keys in iter_translation_batches(keys, translator, mode):
        batch_sources = [source_by_key[key] for key in batch_keys]
        batch_results = translate_batch_resilient(batch_sources, translator, mode=mode)
        for key, translated_text in zip(batch_keys, batch_results):
            for region in grouped_regions[key]:
                region.translated_text = translated_text
                done += 1
        percent = 8 + int(done / total * 84)
        emit_progress(
            progress_callback,
            stage="translating",
            message=f"レイアウト翻訳中: {done}/{total}",
            percent=percent,
            current=done,
            total=total,
        )
        time.sleep(0.02)


def fit_image(width: float, height: float, box_width: float, box_height: float) -> tuple[float, float]:
    ratio = min(box_width / width, box_height / height)
    return width * ratio, height * ratio


def wrap_line(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str | None]:
    wrapped: list[str | None] = []
    paragraphs = text.replace("\r", "\n").splitlines()
    for raw in paragraphs:
        paragraph = raw.strip()
        if not paragraph:
            if wrapped and wrapped[-1] is not None:
                wrapped.append(None)
            continue
        wrapped.extend(wrap_line(paragraph, font_name, font_size, max_width))
    return wrapped or ["本文がありません。"]


def draw_original_page(
    pdf: canvas.Canvas,
    image_path: Path,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
) -> None:
    image = ImageReader(str(image_path))
    img_w, img_h = image.getSize()
    draw_w, draw_h = fit_image(img_w, img_h, box_w, box_h)
    draw_x = box_x + (box_w - draw_w) / 2
    draw_y = box_y + (box_h - draw_h) / 2
    pdf.setFillColor(colors.white)
    pdf.rect(box_x, box_y, box_w, box_h, stroke=0, fill=1)
    pdf.drawImage(image, draw_x, draw_y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    pdf.setStrokeColor(colors.HexColor("#D8DEE9"))
    pdf.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)


def generate_bilingual_pdf(pages: list[PageTranslation], output_pdf: Path) -> None:
    register_fonts()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = landscape(A4)
    margin = 26
    gutter = 18
    header_h = 28
    footer_h = 18
    left_w = (page_w - margin * 2 - gutter) * 0.43
    right_w = page_w - margin * 2 - gutter - left_w
    body_top = page_h - margin - header_h
    body_bottom = margin + footer_h
    body_h = body_top - body_bottom
    left_x = margin
    right_x = margin + left_w + gutter

    font_size = 9.2
    line_h = 13.0
    title_font_size = 10.5

    pdf = canvas.Canvas(str(output_pdf), pagesize=landscape(A4))
    for item in pages:
        lines = wrap_text(item.translated_text, FONT_GOTHIC, font_size, right_w)
        idx = 0
        continuation = 1
        while idx < len(lines):
            pdf.setFillColor(colors.HexColor("#F7F8FA"))
            pdf.rect(0, 0, page_w, page_h, stroke=0, fill=1)

            pdf.setFillColor(colors.HexColor("#1F2933"))
            pdf.setFont(FONT_GOTHIC, title_font_size)
            title = f"Page {item.page_number}"
            if continuation > 1:
                title += f" continued {continuation}"
            pdf.drawString(margin, page_h - margin - 4, title)

            pdf.setFont(FONT_GOTHIC, 8)
            pdf.setFillColor(colors.HexColor("#637381"))
            pdf.drawRightString(page_w - margin, page_h - margin - 4, "Readable Local MVP")

            draw_original_page(pdf, item.image_path, left_x, body_bottom, left_w, body_h)

            pdf.setFillColor(colors.white)
            pdf.rect(right_x - 8, body_bottom, right_w + 16, body_h, stroke=0, fill=1)
            pdf.setStrokeColor(colors.HexColor("#D8DEE9"))
            pdf.rect(right_x - 8, body_bottom, right_w + 16, body_h, stroke=1, fill=0)

            y = body_top - 12
            pdf.setFont(FONT_GOTHIC, font_size)
            pdf.setFillColor(colors.HexColor("#172026"))
            while idx < len(lines) and y >= body_bottom + line_h:
                line = lines[idx]
                if line is None:
                    y -= line_h * 0.7
                else:
                    pdf.drawString(right_x, y, line)
                    y -= line_h
                idx += 1

            pdf.setFont(FONT_GOTHIC, 7.5)
            pdf.setFillColor(colors.HexColor("#637381"))
            pdf.drawString(margin, margin, "Original page image")
            pdf.drawRightString(page_w - margin, margin, "Japanese translation")
            pdf.showPage()
            continuation += 1

    pdf.save()


def generate_layout_pdf(pages: list[PageLayoutTranslation], output_pdf: Path) -> None:
    register_fonts()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_pdf))
    for page in pages:
        pdf.setPageSize((page.width, page.height))
        draw_page_background(pdf, page.image_path, page.width, page.height)
        for region in page.regions:
            if region.translatable and region.translated_text.strip():
                draw_translated_region(pdf, region, page.width, page.height)
        pdf.showPage()
    pdf.save()


def draw_page_background(pdf: canvas.Canvas, image_path: Path, page_w: float, page_h: float) -> None:
    image = ImageReader(str(image_path))
    pdf.setFillColor(colors.white)
    pdf.rect(0, 0, page_w, page_h, stroke=0, fill=1)
    pdf.drawImage(image, 0, 0, width=page_w, height=page_h, preserveAspectRatio=False, mask="auto")


def draw_translated_region(pdf: canvas.Canvas, region: TextRegion, page_w: float, page_h: float) -> None:
    pad_x = 1.6
    pad_y = 1.4
    x = max(0.0, region.x0 - pad_x)
    y = max(0.0, page_h - region.bottom - pad_y)
    w = min(page_w - x, region.x1 - region.x0 + pad_x * 2)
    h = min(page_h - y, region.bottom - region.top + pad_y * 2)
    if w <= 5 or h <= 4:
        return

    text = compact_overlay_text(region.translated_text)
    font_size, line_h, lines = fit_text_for_box(text, FONT_GOTHIC, w - 2.2, h - 1.4)
    if not lines:
        return

    pdf.setFillColor(colors.white)
    pdf.rect(x, y, w, h, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#172026"))
    pdf.setFont(FONT_GOTHIC, font_size)

    text_y = y + h - font_size - 0.8
    for line in lines:
        if line is None:
            text_y -= line_h * 0.55
            continue
        if text_y < y + 0.6:
            break
        pdf.drawString(x + 1.1, text_y, line)
        text_y -= line_h


def compact_overlay_text(text: str) -> str:
    text = normalize_math_symbols(text)
    return re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()


def fit_text_for_box(
    text: str,
    font_name: str,
    max_width: float,
    max_height: float,
) -> tuple[float, float, list[str | None]]:
    size = min(8.2, max(4.8, max_height * 0.46))
    while size >= 4.2:
        line_h = size * 1.22
        lines = wrap_text(text, font_name, size, max_width)
        required = sum(line_h * (0.6 if line is None else 1.0) for line in lines)
        if required <= max_height:
            return size, line_h, lines
        size -= 0.4

    size = 4.2
    line_h = size * 1.18
    max_lines = max(1, int(max_height // line_h))
    lines = [line for line in wrap_text(text, font_name, size, max_width) if line is not None]
    truncated = lines[:max_lines]
    if len(lines) > max_lines and truncated:
        truncated[-1] = trim_to_width(truncated[-1] + "...", font_name, size, max_width)
    return size, line_h, truncated


def trim_to_width(text: str, font_name: str, font_size: float, max_width: float) -> str:
    while text and pdfmetrics.stringWidth(text, font_name, font_size) > max_width:
        text = text[:-4] + "..." if len(text) > 4 else text[:-1]
    return text


def translate_pdf(
    input_pdf: Path,
    output_pdf: Path,
    engine: str = "copy",
    model: str | None = None,
    layout: str = "fast",
    keep_tmp: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    ensure_dirs()
    normalized_layout = layout.lower().strip()
    if normalized_layout not in {"fast", "bilingual", "overlay"}:
        raise ValueError(f"Unknown layout: {layout}")

    translation_mode = "fast" if normalized_layout == "fast" else "quality"
    translator = build_translator(engine, model)
    run_dir = Path(tempfile.mkdtemp(prefix="readable-", dir=TMP_DIR))
    try:
        emit_progress(
            progress_callback,
            stage="rendering",
            message="PDFページを画像化しています",
            percent=3,
        )
        render_dpi = 96 if translation_mode == "fast" else 120
        page_images = render_pdf_pages(input_pdf, run_dir, dpi=render_dpi)

        emit_progress(
            progress_callback,
            stage="extracting",
            message="本文と位置情報を読み取っています",
            percent=6,
        )
        if normalized_layout == "overlay":
            layout_pages = extract_layout_pages(input_pdf, page_images)
            overlay_mode = "page" if isinstance(translator, OllamaTranslator) else translation_mode
            translate_layout_regions(layout_pages, translator, progress_callback, mode=overlay_mode)
            emit_progress(
                progress_callback,
                stage="generating",
                message="元レイアウト寄せPDFを作成しています",
                percent=94,
            )
            generate_layout_pdf(layout_pages, output_pdf)
        else:
            page_texts = extract_text_pages(input_pdf)
            page_chunks = [page_translation_chunks(text, translator, translation_mode) for text in page_texts]
            request_mode = "page" if isinstance(translator, OllamaTranslator) else translation_mode
            translated_pages = translate_page_chunks_batched(
                page_chunks,
                translator,
                progress_callback,
                mode=request_mode,
            )
            pages: list[PageTranslation] = []
            total = max(len(page_images), len(page_texts))
            for index in range(total):
                image_path = page_images[index] if index < len(page_images) else page_images[-1]
                source_text = page_texts[index] if index < len(page_texts) else ""
                translated_text = (
                    translated_pages[index]
                    if index < len(translated_pages)
                    else "このページから本文を抽出できませんでした。"
                )
                pages.append(
                    PageTranslation(
                        page_number=index + 1,
                        source_text=source_text,
                        translated_text=translated_text,
                        image_path=image_path,
                    )
                )
            emit_progress(
                progress_callback,
                stage="generating",
                message="高速本文訳PDFを作成しています" if translation_mode == "fast" else "読み比べPDFを作成しています",
                percent=94,
            )
            generate_bilingual_pdf(pages, output_pdf)
        emit_progress(
            progress_callback,
            stage="done",
            message="完了しました",
            percent=100,
        )
        return output_pdf
    finally:
        if not keep_tmp:
            shutil.rmtree(run_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a PDF into a bilingual reading PDF.")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--engine", choices=["copy", "nmt", "ollama"], default="copy")
    parser.add_argument("--model", default=None)
    parser.add_argument("--layout", choices=["fast", "bilingual", "overlay"], default="fast")
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = translate_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        engine=args.engine,
        model=args.model,
        layout=args.layout,
        keep_tmp=args.keep_tmp,
    )
    print(output)


if __name__ == "__main__":
    main()
