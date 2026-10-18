"""`### N컷` storyboard input parsing (docs/architecture.md §5.1, PRD §5.2).

A storyboard input declares the cuts explicitly instead of letting the script
engine write a script:

```markdown
### 1컷
- 자막: 지난 금요일 미국 증시, 강하게 상승 마감했습니다.
- 화면 연출: 상승 화살표와 함께 호랑이가 환호
- 이미지 프롬프트: a bull market trading floor, photorealistic
```

Field separation is the whole point of this module (architecture.md §5.1):

| field          | parsed key         | consumed by                                  |
| -------------- | ------------------ | -------------------------------------------- |
| `자막`         | `caption`          | TTS narration — **the only narrated field**  |
| `화면 연출`    | `visual_direction` | emoticon cue selection (US-008)              |
| `이미지 프롬프트` | `image_prompt`   | photo-mode image generation (US-009)         |

`visual_direction` / `image_prompt` are preserved on every parsed cut so the
later cue-planning stories can read them, but `join_captions()` never lets
them reach the narration text.

Accepted field-line shapes (all tolerated, because the storyboard is
hand-written): an optional list marker (`-`, `*`, `+`, `1.`), optional bold or
italic markers around the label, the label itself, then `:` or the fullwidth
`：`. A field's value continues onto following lines until the next field
label or the next cut header, so a multi-line 이미지 프롬프트 stays intact.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "CUT_HEADER_RE",
    "is_storyboard_markdown",
    "parse_storyboard",
    "join_captions",
    "CAPTION_JOINER",
]

#: Cut header, e.g. `### 1컷` / `## 2 컷` / `### 3컷 - 도입부`.
#: A heading of any level is accepted; the cut number is group 1.
CUT_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(\d+)[ \t]*컷[ \t]*(?:[-–—:·].*)?$", re.M)

#: Captions of consecutive cuts are joined with a newline, mirroring the
#: line-per-sentence style of a plain input memo (docs/sample.md). Newlines are
#: a sentence-ish boundary for the downstream TTS/Scribe stages, and keeping one
#: cut per line makes `logs/tts_input.txt` readable against the storyboard.
CAPTION_JOINER = "\n"

# label -> parsed key. Aliases cover the spacing variants a hand-written
# storyboard produces; the canonical labels are the first of each group.
_FIELD_LABELS: dict[str, str] = {
    "자막": "caption",
    "자 막": "caption",
    "내레이션": "caption",
    "화면 연출": "visual_direction",
    "화면연출": "visual_direction",
    "연출": "visual_direction",
    "이미지 프롬프트": "image_prompt",
    "이미지프롬프트": "image_prompt",
    "프롬프트": "image_prompt",
}

_FIELD_LINE_RE = re.compile(
    r"^[ \t]*"
    r"(?:[-*+][ \t]+|\d+[.)][ \t]+)?"  # optional list marker
    r"(?:\*{1,2}|_{1,2})?"  # optional emphasis open
    r"(?P<label>[^:：*_\n]{1,20}?)"
    r"(?:\*{1,2}|_{1,2})?"  # optional emphasis close
    r"[ \t]*[:：][ \t]*"
    r"(?P<value>.*)$"
)

_PARSED_KEYS = ("caption", "visual_direction", "image_prompt")


def is_storyboard_markdown(markdown_text: str) -> bool:
    """True if `markdown_text` contains at least one `### N컷` cut header."""
    return CUT_HEADER_RE.search(markdown_text or "") is not None


def _normalize_label(raw: str) -> str | None:
    """Map a raw field label to a parsed key, or None if it is not a field."""
    label = re.sub(r"[ \t]+", " ", raw.strip().strip("*_").strip())
    return _FIELD_LABELS.get(label)


def parse_storyboard(markdown_text: str) -> list[dict[str, Any]]:
    """Parse a `### N컷` storyboard into a list of cut dicts, in document order.

    Each cut dict has keys `cut_number` (int, as written in the header),
    `caption`, `visual_direction` and `image_prompt` (strings, empty when the
    cut omits that field). Text appearing under a cut header before any field
    label is treated as the cut's `caption`, so a minimal storyboard that
    writes only the narration line still works.

    Returns `[]` when no cut header is present.
    """
    text = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n")
    headers = list(CUT_HEADER_RE.finditer(text))
    if not headers:
        return []

    cuts: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        body_start = header.end()
        body_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        cut: dict[str, Any] = {"cut_number": int(header.group(1))}
        cut.update({key: "" for key in _PARSED_KEYS})
        parts: dict[str, list[str]] = {key: [] for key in _PARSED_KEYS}

        current = "caption"  # bare text right under the header narrates
        for line in text[body_start:body_end].split("\n"):
            if not line.strip():
                continue
            match = _FIELD_LINE_RE.match(line)
            key = _normalize_label(match.group("label")) if match else None
            if key is not None:
                current = key
                value = match.group("value").strip()  # type: ignore[union-attr]
                if value:
                    parts[key].append(value)
            else:
                parts[current].append(line.strip())

        for key in _PARSED_KEYS:
            cut[key] = "\n".join(parts[key]).strip()
        cuts.append(cut)

    return cuts


def join_captions(cuts: list[dict[str, Any]]) -> str:
    """Join every cut's `caption` (in order) into the TTS narration text.

    Only `caption` is read. `visual_direction` and `image_prompt` are ignored
    here by design (architecture.md §5.1: 자막만 내레이션에 사용) while
    remaining available on the cut dicts for US-008/US-009.
    """
    captions = [str(cut.get("caption", "")).strip() for cut in cuts]
    return CAPTION_JOINER.join(caption for caption in captions if caption)
