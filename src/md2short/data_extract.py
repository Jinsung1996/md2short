"""숫자 모션 그래픽 extraction for Remotion `NumberCard.tsx` (F-10).

See docs/architecture.md section 5.9. The narration script (not `words.json`)
is scanned with regexes for the four numeric mention categories:

| category   | shape                                                        | example        |
| ---------- | ------------------------------------------------------------ | -------------- |
| `percent`  | number + `%` / `퍼센트` / `프로`                             | `0.94%`        |
| `point`    | number + `p` / `포인트`                                      | `30포인트`     |
| `currency` | number + `조` / `억` / `만` + `원` / `달러`                  | `100억 달러`   |
| `index`    | `코스피`/`코스닥`/`S&P 500`/`나스닥`/`다우`/`니케이`/`항셍` + number | `S&P 500 종가 7,230` |

Timing: the script has no timestamps, so a match's **character position** is
mapped proportionally onto the real TTS audio duration
(`startMs = char_pos / len(script_text) * audio_duration_ms`). This is the one
place in the pipeline that does not read `words.json`, and it is an
approximation by design — a `NumberCard` is a decorative sticker, not a
caption, so a small drift is acceptable (docs/AGENTS.md section 3.2 applies to
caption/cue timing, which comes from `words.json`).

Priority note: the `index` branch refuses a number that is immediately
followed by a `%`/포인트/조억만 unit, so `나스닥은 0.94% 상승했습니다` is
extracted as a `percent` card (the unit is the meaningful figure) rather than
as an index value of `0.94`.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "NUMBER_PATTERN",
    "INDEX_NAMES",
    "NUMERIC_MENTION_RE",
    "DEFAULT_MAX_CARDS",
    "DEFAULT_COOLDOWN_MS",
    "DEFAULT_DURATION_MS",
    "plan_number_cards",
]

#: Default planner rules (architecture.md section 5.9: max 6, cooldown 2.5s,
#: duration 1.8s). `DEFAULT_DURATION_MS` matches `NumberCard.tsx`'s
#: `DEFAULT_DURATION_MS`.
DEFAULT_MAX_CARDS = 6
DEFAULT_COOLDOWN_MS = 2500
DEFAULT_DURATION_MS = 1800

#: `7,230`, `0.94`, `18` — thousands separators and one decimal part.
NUMBER_PATTERN = r"\d[\d,]*(?:\.\d+)?"

#: Index names that make a bare number meaningful, longest-first so that
#: `S&P 500` is not cut short by a shorter alternative.
INDEX_NAMES = (
    "S&P\\s?500",
    "코스피",
    "코스닥",
    "나스닥",
    "니케이",
    "항셍",
    "다우",
)

#: One pass over the script finds all four categories. Alternative order is
#: the priority order: the `index` branch is tried first because it starts
#: earlier in the sentence (at the index name), and it declines unit-suffixed
#: numbers so `percent`/`point`/`currency` keep those.
NUMERIC_MENTION_RE = re.compile(
    # index + number (number must not carry a %/포인트/조억만 unit)
    r"(?P<index>" + "|".join(INDEX_NAMES) + r")"
    r"[^\d\n]{0,10}?"
    r"(?P<index_num>" + NUMBER_PATTERN + r")"
    r"(?![\d.,])"
    r"(?!\s*(?:%|퍼센트|프로|포인트|p|조|억|만))"
    r"|"
    # number + 조|억|만 + 원|달러
    r"(?P<currency>" + NUMBER_PATTERN + r"\s*[조억만]\s*(?:원|달러))"
    r"|"
    # number + % | 퍼센트 | 프로
    r"(?P<percent>" + NUMBER_PATTERN + r"\s*(?:%|퍼센트|프로(?![가-힣])))"
    r"|"
    # number + p | 포인트
    r"(?P<point>" + NUMBER_PATTERN + r"\s*(?:포인트|p(?![A-Za-z가-힣])))"
)

_VALUE_GROUPS = ("currency", "percent", "point")


def _normalize(text: str) -> str:
    """Collapse internal whitespace so `100억  달러` renders as `100억 달러`."""
    return " ".join(text.split())


def plan_number_cards(
    script_text: str,
    audio_duration_ms: int,
    max_cards: int = DEFAULT_MAX_CARDS,
    cooldown_ms: int = DEFAULT_COOLDOWN_MS,
    duration_ms: int = DEFAULT_DURATION_MS,
) -> list[dict[str, Any]]:
    """Extract numeric mentions from `script_text` as `NumberCard` dicts.

    Each returned dict matches Remotion's `NumberCard` type
    (remotion/src/types.ts): `text` (the figure as it should be printed on the
    card), `startMs`, `durationMs`, and `label` (the index name for an `index`
    match, `None` otherwise).

    Rules:

    - `startMs = round(match.start() / len(script_text) * audio_duration_ms)`,
      clamped into `[0, audio_duration_ms]`.
    - A match whose `startMs` is less than `cooldown_ms` after the previously
      **kept** card is dropped, so two figures in one breath do not stack.
    - At most `max_cards` cards are returned. When more matches survive the
      cooldown, the **first `max_cards` in document order** are kept — the
      script is already written narrative-first (the headline figure comes
      first), and document order is trivially deterministic, unlike ranking by
      magnitude where `18%` and `18조 원` would need an arbitrary tie-break.
    - Every card gets `durationMs = duration_ms`.

    Returns `[]` for empty text, a non-positive duration, or a non-positive
    `max_cards`.
    """
    if not script_text or audio_duration_ms <= 0 or max_cards <= 0:
        return []

    total_chars = len(script_text)
    cards: list[dict[str, Any]] = []
    last_start_ms: int | None = None

    for match in NUMERIC_MENTION_RE.finditer(script_text):
        if match.group("index") is not None:
            text = _normalize(match.group("index_num"))
            label: str | None = _normalize(match.group("index"))
        else:
            group = next(name for name in _VALUE_GROUPS if match.group(name) is not None)
            text = _normalize(match.group(group))
            label = None

        start_ms = round(match.start() / total_chars * audio_duration_ms)
        start_ms = max(0, min(audio_duration_ms, start_ms))

        if last_start_ms is not None and start_ms - last_start_ms < cooldown_ms:
            continue

        cards.append(
            {
                "text": text,
                "startMs": start_ms,
                "durationMs": duration_ms,
                "label": label,
            }
        )
        last_start_ms = start_ms

        if len(cards) >= max_cards:
            break

    return cards
