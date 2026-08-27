from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from readable_pdf import (
    DEFAULT_OLLAMA_MODEL,
    OllamaTranslator,
    ProgressCallback,
    TranslationError,
    context_limit_for_chars,
    is_reasoning_model,
    normalize_japanese_translation,
    ollama_message_content,
    parse_translation_batch,
    retry_structured_batch_prediction_limit_for,
    structured_batch_prediction_limit_for,
)


TEXT_COMMANDS = {
    "title",
    "subtitle",
    "part",
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
    "caption",
    "item",
    "textbf",
    "textit",
    "textsc",
    "emph",
}

TABLE_ENVIRONMENTS = {
    "tabular",
    "tabular*",
    "array",
}

FLOAT_ENVIRONMENTS = {
    "figure",
    "figure*",
    "table",
    "table*",
}

TEXT_ENVIRONMENTS = {
    "abstract",
    "quote",
    "quotation",
}

SKIP_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "thebibliography",
    "lstlisting",
    "verbatim",
    "Verbatim",
    "tikzpicture",
    "algorithm",
    "algorithm*",
}

SKIP_ENV_NAMES_PATTERN = "|".join(re.escape(name) for name in sorted(SKIP_ENVIRONMENTS, key=len, reverse=True))
BLOCK_ENVIRONMENTS = SKIP_ENVIRONMENTS | TABLE_ENVIRONMENTS | FLOAT_ENVIRONMENTS
BLOCK_ENV_NAMES_PATTERN = "|".join(re.escape(name) for name in sorted(BLOCK_ENVIRONMENTS, key=len, reverse=True))
TEXT_ENV_NAMES_PATTERN = "|".join(re.escape(name) for name in sorted(TEXT_ENVIRONMENTS, key=len, reverse=True))

PROTECT_PATTERN = re.compile(
    r"(?s)"
    r"(\\begin\{("
    + SKIP_ENV_NAMES_PATTERN
    + r")\}.*?\\end\{\2\})"
    r"|(\$\$.*?\$\$)"
    r"|(\$.*?\$)"
    r"|(\\\[.*?\\\])"
    r"|(\\\(.*?\\\))"
    r"|(\\[a-zA-Z@]+\*?(?:\s*\[[^\[\]]*\])?(?:\s*\{[^{}]*\})*)"
    r"|(\\.)"
)

BLOCK_ENV_PATTERN = re.compile(
    r"(?s)\\begin\{(" + BLOCK_ENV_NAMES_PATTERN + r")\}.*?\\end\{\1\}"
)
TEXT_ENV_PATTERN = re.compile(
    r"(?s)(\\begin\{(" + TEXT_ENV_NAMES_PATTERN + r")\})(.*?)(\\end\{\2\})"
)
PLACEHOLDER_PATTERN = re.compile(
    r"_*\s*(?:READABLE|読みやすい|リーダブル)[\s_-]*(?:TEX|TeX|Tex|tex)[\s_-]*\d+\s*_*\s*",
    re.IGNORECASE,
)
COMMENT_MARKER_PATTERN = re.compile(r"\\readablelocalcomment\{(\d+)\}")


@dataclass
class PlannedSegment:
    raw: str
    protected: str
    placeholders: list[str]


class TexTranslationPlanner:
    def __init__(self) -> None:
        self.segments: list[PlannedSegment] = []

    def add(self, text: str) -> str:
        if not should_translate_text(text):
            return text
        protected, placeholders = protect_tex_fragments(text)
        if not should_translate_text(protected):
            return text
        marker = f"__READABLE_SEG_{len(self.segments)}__"
        self.segments.append(PlannedSegment(raw=text, protected=protected, placeholders=placeholders))
        return marker

    def apply(self, tex: str, translator: OllamaTranslator, progress_callback: ProgressCallback | None = None) -> str:
        if not self.segments:
            return tex
        translated = translate_tex_segments(
            [segment.protected for segment in self.segments],
            translator,
            progress_callback=progress_callback,
        )
        replacements: dict[str, str] = {}
        for index, segment in enumerate(self.segments):
            text = translated[index] if index < len(translated) else segment.protected
            restored = restore_tex_fragments(text, segment.placeholders)
            restored = PLACEHOLDER_PATTERN.sub("", restored)
            if translated_segment_is_unsafe(segment, restored):
                restored = segment.raw
            replacements[f"__READABLE_SEG_{index}__"] = restored or segment.raw
        for marker, translated_text in replacements.items():
            tex = tex.replace(marker, translated_text)
        return restore_unresolved_segment_markers(tex, self.segments)


def restore_unresolved_segment_markers(tex: str, segments: list[PlannedSegment]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(segments):
            return segments[index].raw
        return ""

    return re.sub(r"__READABLE_SEG_(\d+)__", replace, tex)


def translate_tex_project(
    input_path: Path,
    output_pdf: Path,
    *,
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    work_dir = input_path.parent / "tex-work"
    source_dir = work_dir / "source"
    build_dir = work_dir / "build"
    shutil.rmtree(work_dir, ignore_errors=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    emit_tex_progress(progress_callback, "extracting", "TeXソースを展開しています", 5)
    main_tex = prepare_tex_source(input_path, source_dir)

    emit_tex_progress(progress_callback, "translating", "TeX本文を翻訳しています", 8)
    translator = OllamaTranslator(model=model or DEFAULT_OLLAMA_MODEL)
    translate_tex_files(source_dir, main_tex, translator, progress_callback)

    emit_tex_progress(progress_callback, "generating", "TeXをコンパイルしています", 92)
    compiled_pdf = compile_tex(source_dir, build_dir, main_tex)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_pdf, output_pdf)
    return output_pdf


def prepare_tex_source(input_path: Path, source_dir: Path) -> Path:
    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    if input_path.suffix.lower() == ".tex":
        target = source_dir / input_path.name
        shutil.copy2(input_path, target)
        return target
    if input_path.suffix.lower() == ".zip":
        extract_zip(input_path, source_dir)
        return find_main_tex(source_dir)
    if suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"]) or input_path.suffix.lower() in {
        ".tar",
        ".tgz",
    }:
        extract_tar(input_path, source_dir)
        return find_main_tex(source_dir)
    raise ValueError("TeX入力は .tex、.zip、.tar、.tar.gz、.tgz に対応しています。")


def extract_zip(input_path: Path, source_dir: Path) -> None:
    with zipfile.ZipFile(input_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            target = safe_extract_path(source_dir, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def extract_tar(input_path: Path, source_dir: Path) -> None:
    with tarfile.open(input_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            target = safe_extract_path(source_dir, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def safe_extract_path(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError("TeXアーカイブ内に不正なパスがあります。")
    return target


def find_main_tex(source_dir: Path) -> Path:
    candidates = [path for path in source_dir.rglob("*.tex") if path.is_file()]
    if not candidates:
        raise ValueError("TeXファイルが見つかりませんでした。")

    def score(path: Path) -> tuple[int, int, int]:
        text = path.read_text(encoding="utf-8", errors="replace")
        has_documentclass = 0 if "\\documentclass" in text else 1
        has_begin = 0 if "\\begin{document}" in text else 1
        return has_documentclass, has_begin, len(path.parts)

    return sorted(candidates, key=score)[0]


def translate_tex_source(tex: str, translator: OllamaTranslator, progress_callback: ProgressCallback | None = None) -> str:
    tex, comments = protect_tex_comments(tex)
    planner = TexTranslationPlanner()
    tex = plan_text_command_arguments(tex, planner)
    if has_document_body(tex):
        tex = plan_document_body(tex, planner)
    else:
        tex = plan_body_segment(tex, planner)
    return restore_tex_comments(planner.apply(tex, translator, progress_callback), comments)


def protect_tex_comments(tex: str) -> tuple[str, list[str]]:
    comments: list[str] = []
    output: list[str] = []
    cursor = 0
    while cursor < len(tex):
        if tex[cursor] == "%" and not is_escaped(tex, cursor):
            line_end = tex.find("\n", cursor)
            if line_end < 0:
                comments.append(tex[cursor:])
                output.append(f"\\readablelocalcomment{{{len(comments) - 1}}}")
                break
            comments.append(tex[cursor:line_end])
            output.append(f"\\readablelocalcomment{{{len(comments) - 1}}}")
            cursor = line_end
            continue
        output.append(tex[cursor])
        cursor += 1
    return "".join(output), comments


def restore_tex_comments(tex: str, comments: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(comments):
            return comments[index]
        return ""

    return COMMENT_MARKER_PATTERN.sub(replace, tex)


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def has_document_body(tex: str) -> bool:
    begin = re.search(r"\\begin\{document\}", tex)
    end = re.search(r"\\end\{document\}", tex)
    return begin is not None and end is not None and begin.end() < end.start()


def translate_tex_files(
    source_dir: Path,
    main_tex: Path,
    translator: OllamaTranslator,
    progress_callback: ProgressCallback | None = None,
) -> None:
    tex_files = sorted(path for path in source_dir.rglob("*.tex") if path.is_file())
    total = len(tex_files)
    for index, path in enumerate(tex_files, start=1):
        relative_name = path.relative_to(source_dir)
        base_percent = 8 + int((index - 1) / max(1, total) * 80)
        end_percent = 8 + int(index / max(1, total) * 80)
        emit_tex_progress(
            progress_callback,
            "translating",
            f"TeXファイルを翻訳中: {index}/{total} {relative_name}",
            base_percent,
            index - 1,
            total,
        )
        raw_tex = path.read_text(encoding="utf-8", errors="replace")
        translated_tex = translate_tex_source(
            raw_tex,
            translator,
            tex_file_progress(progress_callback, base_percent, end_percent),
        )
        if path == main_tex:
            translated_tex = ensure_latex_japanese_support(translated_tex, latex_engine())
        path.write_text(translated_tex, encoding="utf-8")
        emit_tex_progress(
            progress_callback,
            "translating",
            f"TeXファイルを翻訳中: {index}/{total} {relative_name}",
            end_percent,
            index,
            total,
        )


def tex_file_progress(
    progress_callback: ProgressCallback | None,
    base_percent: int,
    end_percent: int,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None

    def wrapped(payload: dict[str, object]) -> None:
        local_percent = int(payload.get("percent", 8) or 8)
        local_ratio = max(0.0, min(1.0, (local_percent - 8) / 80))
        mapped = base_percent + int((end_percent - base_percent) * local_ratio)
        updated = dict(payload)
        updated["percent"] = mapped
        progress_callback(updated)

    return wrapped


def plan_text_command_arguments(tex: str, planner: TexTranslationPlanner) -> str:
    output: list[str] = []
    cursor = 0
    pattern = re.compile(r"\\([a-zA-Z@]+)\*?")
    while True:
        match = pattern.search(tex, cursor)
        if match is None:
            output.append(tex[cursor:])
            break
        command = match.group(1)
        if command not in TEXT_COMMANDS:
            output.append(tex[cursor : match.end()])
            cursor = match.end()
            continue

        brace_start = skip_spaces_and_options(tex, match.end())
        if brace_start >= len(tex) or tex[brace_start] != "{":
            output.append(tex[cursor : match.end()])
            cursor = match.end()
            continue
        brace_end = find_matching_brace(tex, brace_start)
        if brace_end is None:
            output.append(tex[cursor : match.end()])
            cursor = match.end()
            continue

        argument = tex[brace_start + 1 : brace_end]
        output.append(tex[cursor : brace_start + 1])
        output.append(planner.add(argument))
        output.append("}")
        cursor = brace_end + 1
    return "".join(output)


def skip_spaces_and_options(tex: str, index: int) -> int:
    cursor = index
    while cursor < len(tex) and tex[cursor].isspace():
        cursor += 1
    if cursor < len(tex) and tex[cursor] == "[":
        depth = 1
        cursor += 1
        while cursor < len(tex) and depth:
            if tex[cursor] == "[":
                depth += 1
            elif tex[cursor] == "]":
                depth -= 1
            cursor += 1
        while cursor < len(tex) and tex[cursor].isspace():
            cursor += 1
    return cursor


def find_matching_brace(tex: str, open_index: int) -> int | None:
    depth = 0
    cursor = open_index
    while cursor < len(tex):
        char = tex[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def plan_document_body(tex: str, planner: TexTranslationPlanner) -> str:
    begin = re.search(r"\\begin\{document\}", tex)
    end = re.search(r"\\end\{document\}", tex)
    if begin is None or end is None or begin.end() >= end.start():
        return tex
    prefix = tex[: begin.end()]
    body = tex[begin.end() : end.start()]
    suffix = tex[end.start() :]
    return prefix + plan_body_segment(body, planner) + suffix


def plan_body_segment(body: str, planner: TexTranslationPlanner) -> str:
    output: list[str] = []
    cursor = 0
    for match in BLOCK_ENV_PATTERN.finditer(body):
        output.append(plan_text_environments(body[cursor : match.start()], planner))
        env_name = match.group(1)
        if env_name in TABLE_ENVIRONMENTS:
            output.append(plan_table_environment(match.group(0), planner))
        elif env_name in FLOAT_ENVIRONMENTS:
            output.append(plan_wrapped_environment(match.group(0), planner))
        else:
            output.append(match.group(0))
        cursor = match.end()
    output.append(plan_text_environments(body[cursor:], planner))
    return "".join(output)


def plan_wrapped_environment(env_text: str, planner: TexTranslationPlanner) -> str:
    begin_match = re.match(r"(?s)\\begin\{([^}]+)\}", env_text)
    if begin_match is None:
        return env_text
    env_name = begin_match.group(1)
    end_token = f"\\end{{{env_name}}}"
    end_start = env_text.rfind(end_token)
    if end_start < 0:
        return env_text
    content_start = skip_spaces_and_options(env_text, begin_match.end())
    prefix = env_text[:content_start]
    content = env_text[content_start:end_start]
    suffix = env_text[end_start:]
    return prefix + plan_body_segment(content, planner) + suffix


def plan_text_environments(text: str, planner: TexTranslationPlanner) -> str:
    output: list[str] = []
    cursor = 0
    for match in TEXT_ENV_PATTERN.finditer(text):
        output.append(plan_plain_tex(text[cursor : match.start()], planner))
        output.append(match.group(1))
        output.append(plan_plain_tex(match.group(3), planner))
        output.append(match.group(4))
        cursor = match.end()
    output.append(plan_plain_tex(text[cursor:], planner))
    return "".join(output)


def plan_table_environment(env_text: str, planner: TexTranslationPlanner) -> str:
    begin_match = re.match(r"(?s)\\begin\{([^}]+)\}", env_text)
    if begin_match is None:
        return env_text
    env_name = begin_match.group(1)
    end_token = f"\\end{{{env_name}}}"
    end_start = env_text.rfind(end_token)
    if end_start < 0:
        return env_text

    content_start = skip_spaces_and_options(env_text, begin_match.end())
    mandatory_count = 0
    while content_start < end_start and env_text[content_start] == "{":
        brace_end = find_matching_brace(env_text, content_start)
        if brace_end is None:
            return env_text
        content_start = brace_end + 1
        mandatory_count += 1
        if env_name != "tabular*" and mandatory_count >= 1:
            break
        if env_name == "tabular*" and mandatory_count >= 2:
            break
        content_start = skip_spaces_and_options(env_text, content_start)

    prefix = env_text[:content_start]
    content = env_text[content_start:end_start]
    suffix = env_text[end_start:]
    return prefix + plan_table_content(content, planner) + suffix


def plan_table_content(content: str, planner: TexTranslationPlanner) -> str:
    separators = re.split(
        r"(&|\\\\(?:\[[^\]]*\])?|\\(?:toprule|midrule|bottomrule|hline)\b|\\cmidrule(?:\([^)]*\))?\{[^{}]*\})",
        content,
    )
    output: list[str] = []
    for index, part in enumerate(separators):
        if index % 2 == 1:
            output.append(part)
            continue
        output.append(plan_table_cell(part, planner))
    return "".join(output)


def plan_table_cell(cell: str, planner: TexTranslationPlanner) -> str:
    if not should_translate_text(cell):
        return cell
    leading_match = re.match(r"\s*", cell)
    trailing_match = re.search(r"\s*\Z", cell)
    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""
    start = len(leading)
    end = len(cell) - len(trailing)
    core = cell[start:end]
    if not core or core.startswith("\\"):
        return cell
    if "\\" in core or "$" in core:
        return cell
    return leading + planner.add(core) + trailing


def plan_plain_tex(text: str, planner: TexTranslationPlanner) -> str:
    parts = re.split(r"(\n\s*\n)", text)
    return "".join(plan_paragraph(part, planner) if index % 2 == 0 else part for index, part in enumerate(parts))


def plan_paragraph(paragraph: str, planner: TexTranslationPlanner) -> str:
    if not should_translate_text(paragraph):
        return paragraph
    stripped = paragraph.strip()
    if not stripped or stripped.startswith("%"):
        return paragraph
    if re.fullmatch(r"\\(?:begin|end)\{[^}]+\}", stripped):
        return paragraph
    if stripped.startswith("\\") and not stripped.startswith("\\item"):
        command_prefix, remaining = split_leading_command_lines(paragraph)
        if command_prefix and should_translate_text(remaining):
            return command_prefix + plan_paragraph(remaining, planner)
        return paragraph

    item_match = re.match(r"(?s)(\s*\\item(?:\s*\[[^\]]*\])?\s*)(.+)", paragraph)
    if item_match:
        return item_match.group(1) + planner.add(item_match.group(2))
    return planner.add(paragraph)


def split_leading_command_lines(paragraph: str) -> tuple[str, str]:
    lines = paragraph.splitlines(keepends=True)
    prefix: list[str] = []
    index = 0
    while index < len(lines) and not lines[index].strip():
        prefix.append(lines[index])
        index += 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            break
        if not stripped.startswith("\\") or stripped.startswith("\\item"):
            break
        prefix.append(lines[index])
        index += 1
    return "".join(prefix), "".join(lines[index:])


def should_translate_text(text: str) -> bool:
    if not text or not re.search(r"[A-Za-z]", text):
        return False
    letters = len(re.findall(r"[A-Za-z]", text))
    commands = len(re.findall(r"\\[a-zA-Z@]+", text))
    if letters < 3:
        return False
    return commands < max(6, letters // 8)


def protect_tex_fragments(text: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def replace(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"__READABLE_TEX_{len(placeholders) - 1}__"

    return PROTECT_PATTERN.sub(replace, text), placeholders


def restore_tex_fragments(text: str, placeholders: list[str]) -> str:
    restored = remove_latex_control_chars(text)
    for index, value in enumerate(placeholders):
        restored = restored.replace(f"__READABLE_TEX_{index}__", value)
        restored = placeholder_variant_pattern(index).sub(lambda _: value, restored)
    return restored


def placeholder_variant_pattern(index: int) -> re.Pattern[str]:
    return re.compile(
        r"_*\s*(?:(?:READABLE|読みやすい|リーダブル)[\s_-]*)?(?:TEX|TeX|Tex|tex)[\s_-]*"
        + str(index)
        + r"\s*_*\s*",
        re.IGNORECASE,
    )


def remove_latex_control_chars(text: str) -> str:
    allowed = {chr(10), chr(13), chr(9)}
    return "".join(char for char in text if char in allowed or ord(char) >= 32)

def translated_segment_is_unsafe(segment: PlannedSegment, translated: str) -> bool:
    if re.search(r"\\(?:bf|it|rm|sc|tt|sf)\b", segment.raw):
        return True
    if segment.placeholders and has_placeholder_artifact(translated):
        return True
    if segment.placeholders or re.search(r"[\\$&]", segment.raw):
        return False
    return bool(re.search(r"\\[a-zA-Z@]+|[$&]", translated))


def has_placeholder_artifact(text: str) -> bool:
    return bool(
        PLACEHOLDER_PATTERN.search(text)
        or re.search(r"__|\d+__|(?:TEX|TeX|Tex|tex)[\s_-]*\d+|READABLE|読みやすい|リーダブル|readablelocalcomment", text, re.IGNORECASE)
    )


def translate_tex_segments(
    texts: list[str],
    translator: OllamaTranslator,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    results: list[str] = []
    total = len(texts)
    done = 0
    for batch in iter_tex_batches(texts):
        translated = translate_tex_batch_resilient(batch, translator)
        results.extend(translated)
        done += len(batch)
        emit_tex_progress(
            progress_callback,
            "translating",
            f"TeX本文を翻訳中: {done}/{total}",
            8 + int(done / max(1, total) * 80),
            done,
            total,
        )
    return results


def iter_tex_batches(texts: list[str]) -> list[list[str]]:
    max_items = max(1, int(os.environ.get("READABLE_TEX_BATCH_ITEMS", "8")))
    max_chars = max(1200, int(os.environ.get("READABLE_TEX_BATCH_CHARS", "5000")))
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        if current and (len(current) >= max_items or current_chars + len(text) > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


def translate_tex_batch_resilient(texts: list[str], translator: OllamaTranslator) -> list[str]:
    try:
        translated = translate_tex_batch(texts, translator)
        if len(translated) == len(texts):
            return translated
    except TranslationError:
        pass
    if len(texts) <= 1:
        return [translate_tex_single(texts[0], translator)] if texts else []
    mid = max(1, len(texts) // 2)
    return translate_tex_batch_resilient(texts[:mid], translator) + translate_tex_batch_resilient(texts[mid:], translator)


def translate_tex_single(text: str, translator: OllamaTranslator) -> str:
    try:
        return translate_tex_batch([text], translator)[0]
    except Exception:
        return text


def translate_tex_batch(texts: list[str], translator: OllamaTranslator) -> list[str]:
    prompt = (
        "次のJSON配列のtextを、LaTeX論文の本文として自然な日本語に翻訳してください。"
        f"出力は必ずJSONオブジェクトだけにし、形式は"
        f'{{"translations": ["翻訳1", "翻訳2"]}}です。'
        f"translationsは入力と同じ{len(texts)}個・同じ順序にしてください。"
        "LaTeXプレースホルダ __READABLE_TEX_0__ のような文字列は、文字単位で絶対に変更しないでください。"
        "入力にないLaTeXコマンド、数式、表記号、環境を新しく作らないでください。"
        "数式、引用、参照、LaTeXコマンド、単位、固有名詞は必要な範囲で原文のまま残してください。"
        "説明、注釈、Markdown、前置きは出力しないでください。\n\n"
        f"{json.dumps([{'id': index, 'text': text} for index, text in enumerate(texts)], ensure_ascii=False)}"
    )
    payload: dict[str, object] = {
        "model": translator.model,
        "messages": [
            {"role": "system", "content": tex_translation_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_ctx": context_limit_for_chars(sum(len(text) for text in texts), translator.model),
            "num_predict": structured_batch_prediction_limit_for(texts, translator.model),
        },
    }
    if is_reasoning_model(translator.model):
        payload["think"] = False
    data = translator.chat(payload)
    parsed = parse_translation_batch(ollama_message_content(data), expected_count=len(texts))
    if parsed is None and is_reasoning_model(translator.model):
        payload["options"]["num_predict"] = retry_structured_batch_prediction_limit_for(texts, translator.model)
        data = translator.chat(payload)
        parsed = parse_translation_batch(ollama_message_content(data), expected_count=len(texts))
    if parsed is None:
        raise TranslationError("OllamaのTeX翻訳結果をJSONとして読み取れませんでした。")
    return [normalize_japanese_translation(text) for text in parsed]


def tex_translation_system_prompt() -> str:
    return (
        "あなたはLaTeX論文を日本語へ翻訳する専門翻訳者です。"
        "LaTeX構文、プレースホルダ、数式、引用、参照、ラベル、ファイル名は変更しません。"
        "入力に存在しないLaTeXコマンドや数式環境を追加しません。"
        "本文、見出し、キャプションなど自然言語だけを日本語にします。"
        "出力は指定されたJSONだけにしてください。"
    )


def latex_engine() -> str:
    engine = os.environ.get("READABLE_TEX_ENGINE", "lualatex").strip().lower()
    if engine not in {"lualatex", "xelatex"}:
        return "lualatex"
    return engine


def ensure_latex_japanese_support(tex: str, engine: str) -> str:
    if (
        "\\usepackage{luatexja}" in tex
        or "\\usepackage{zxjatype}" in tex
        or "\\usepackage{xeCJK}" in tex
        or "\\documentclass" not in tex
    ):
        return tex
    if engine == "xelatex":
        insert = r"""

% Readable Local: Japanese support for XeLaTeX
\usepackage{zxjatype}
\usepackage[ipa]{zxjafont}
"""
    else:
        insert = r"""

% Readable Local: Japanese support for LuaLaTeX
\usepackage{luatexja}
\usepackage[ipa]{luatexja-preset}
"""
    return re.sub(r"\\begin\{document\}", lambda _: insert + "\n\\begin{document}", tex, count=1)


def compile_tex(source_dir: Path, build_dir: Path, main_tex: Path) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmkが見つかりません。TeXコンパイル環境を確認してください。")
    relative_main = main_tex.relative_to(source_dir)
    engine = latex_engine()
    command = [
        latexmk,
        f"-{engine}",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={build_dir}",
        str(relative_main),
    ]
    timeout_seconds = max(60, int(os.environ.get("READABLE_TEX_COMPILE_TIMEOUT", "240")))
    try:
        result = subprocess.run(
            command,
            cwd=source_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        lines = "\n".join(str(output).splitlines()[-40:])
        raise RuntimeError(f"TeXのコンパイルが{timeout_seconds}秒でタイムアウトしました。\n{lines}") from exc
    output_name = relative_main.with_suffix(".pdf").name
    compiled_pdf = build_dir / output_name
    if result.returncode != 0 or not compiled_pdf.exists():
        lines = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"TeXのコンパイルに失敗しました。\n{lines}")
    return compiled_pdf


def emit_tex_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    percent: int,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return
    payload: dict[str, object] = {"stage": stage, "message": message, "percent": max(0, min(100, percent))}
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    progress_callback(payload)
