"""Markdown input handling: type detection, source archiving, markdown stripping.

Role per docs/architecture.md §2.2: "마크다운 제거 + 텍스트 추출 fallback
(한국어 전용)" — this module owns the pipeline's *input* side of F-01:

1. `detect_input_kind()` — decide whether the raw input is a `### N컷`
   storyboard (handled by `md2short.storyboard`) or plain markdown/text
   (handled by `md2short.script_engine`).
2. `write_source_md()` — archive the raw, unmodified input to
   `<job_dir>/inputs/source.md` (architecture.md §4.4). The job directory is
   always passed in; this module never hardcodes a `jobs/` root, so the
   pipeline (US-013) stays the only place that knows where jobs live.
3. `strip_markdown()` — the markdown-syntax stripper that backs
   `script_engine.fallback_extract()`.

Information-preservation (architecture.md §1.6) is the first principle of the
stripper: it removes *markup* only. It never shortens, reorders, summarizes or
drops a line of prose, so numbers like "7,230" / "0.94%" / "18%" survive
verbatim into the narration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from md2short.storyboard import is_storyboard_markdown

__all__ = [
    "InputKind",
    "detect_input_kind",
    "write_source_md",
    "strip_markdown",
    "SOURCE_MD_RELPATH",
]

#: Where the raw input markdown is archived inside a job directory.
SOURCE_MD_RELPATH = Path("inputs") / "source.md"

#: "storyboard" -> `### N컷` cut sheet, "markdown" -> plain markdown/text.
InputKind = Literal["storyboard", "markdown"]

# --- markdown stripping ----------------------------------------------------
# Each pattern removes *syntax* and keeps the surrounding text content.
_FENCED_CODE_RE = re.compile(r"^[ \t]*(?:```|~~~).*?$", re.M)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_ATX_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.M)
_SETEXT_UNDERLINE_RE = re.compile(r"^[ \t]*(?:={2,}|-{2,})[ \t]*$", re.M)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>+[ \t]?", re.M)
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)", re.M)
_THEMATIC_BREAK_RE = re.compile(r"^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*(?:-[ \t]*){3,}$", re.M)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
_INLINE_CODE_RE = re.compile(r"`+([^`]*)`+")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|~{2})(?=\S)(.+?)(?<=\S)\1", re.S)
_TABLE_DIVIDER_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.M)
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.M)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def detect_input_kind(markdown_text: str) -> InputKind:
    """Classify raw input as a `### N컷` storyboard or plain markdown.

    Detection is delegated to `storyboard.is_storyboard_markdown()` so the
    `### N컷` header grammar lives in exactly one module.
    """
    return "storyboard" if is_storyboard_markdown(markdown_text) else "markdown"


def write_source_md(job_dir: Path, markdown_text: str) -> Path:
    """Archive the raw input markdown to `<job_dir>/inputs/source.md`.

    Writes the content byte-for-byte as received (no stripping, no newline
    normalization beyond UTF-8 encoding), creating parent directories as
    needed, and returns the path written.
    """
    target = Path(job_dir) / SOURCE_MD_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown_text, encoding="utf-8")
    return target


def strip_markdown(markdown_text: str) -> str:
    """Return `markdown_text` with markdown syntax removed and prose preserved.

    Removes: fence delimiters, HTML comments, ATX/setext header markers,
    blockquote markers, list bullets/numbers, thematic breaks, table divider
    rows, link/image syntax (keeping the link *text*), inline-code backticks
    and emphasis markers (`*`, `_`, `~~`). Text inside a fenced code block is
    kept (only the fence lines go) because an input memo may legitimately put
    figures in a code block and §1.6 forbids dropping facts.

    Blank-line runs collapse to a single blank line and leading/trailing
    whitespace is trimmed; nothing else about the line order or wording
    changes.
    """
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _FENCED_CODE_RE.sub("", text)
    text = _THEMATIC_BREAK_RE.sub("", text)
    text = _TABLE_DIVIDER_RE.sub("", text)
    text = _SETEXT_UNDERLINE_RE.sub("", text)
    text = _ATX_HEADER_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _AUTOLINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    # Emphasis can nest (e.g. `***x***` or `**_x_**`); run to a fixed point.
    for _ in range(3):
        stripped = _EMPHASIS_RE.sub(r"\2", text)
        if stripped == text:
            break
        text = stripped
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()
