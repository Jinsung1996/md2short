"""Korean opening phrase for narration + Remotion `openingText` (§5.10, §2.1).

`JobConfig.show_opening` drives two *different* consumers, so this module
produces two forms of the same phrase from the single template in
`schema.OPENING_TEMPLATES`:

- **spoken** (`"26년 5월 2일 데이브리핑"`): prepended to the TTS narration
  script. The template's line break becomes a space, because a literal `\\n`
  in TTS input is a layout artifact, not speech.
- **visual** (`"26년 5월 2일\\n데이브리핑"`): passed through to the Remotion
  `openingText` prop (architecture.md §4.3), which renders it as two lines.

The template is the only source of the wording — the spoken form is derived
from it rather than written out again, so the Remotion `OpeningBriefing`
overlay and the narration can never drift apart (§5.10 "계약 동기화").

`show_opening=False` leaves the narration untouched and reports no visual
text, matching §5.10 ("opening 문구를 대본에 붙이지 않고").
"""

from __future__ import annotations

from datetime import date
from typing import Any

from md2short.schema import OPENING_TEMPLATES

__all__ = ["build_opening_text", "build_opening"]


def build_opening_text(today: date | None = None, language: str = "ko") -> tuple[str, str]:
    """Return `(spoken, visual)` opening phrases for `today`.

    `today` defaults to `date.today()` and is injectable for tests. The year is
    rendered as two digits and month/day without zero padding, per the examples
    in architecture.md §4.3 / §5.10 (`26년 5월 2일\\n데이브리핑`).
    """
    day = today or date.today()
    visual = OPENING_TEMPLATES[language].format(
        year=f"{day.year % 100:02d}",
        month=day.month,
        day=day.day,
    )
    spoken = " ".join(visual.split("\n")).strip()
    return spoken, visual


def build_opening(
    narration: str,
    *,
    show_opening: bool,
    today: date | None = None,
    language: str = "ko",
) -> dict[str, Any]:
    """Prepend the spoken opening to `narration` when `show_opening` is True.

    Returns a dict with:

    - `narration`: the script to send to TTS (the original text when
      `show_opening` is False, otherwise `"<spoken>\\n<original>"`).
    - `spoken_text`: the spoken opening phrase, or `None`.
    - `visual_text`: the two-line Remotion `openingText` value, or `None`.
    - `show_opening`: the flag echoed back, so callers can serialize
      `inputs/script.json` without re-reading `JobConfig`.
    """
    if not show_opening:
        return {
            "narration": narration,
            "spoken_text": None,
            "visual_text": None,
            "show_opening": False,
        }

    spoken, visual = build_opening_text(today=today, language=language)
    prefixed = f"{spoken}\n{narration}" if narration.strip() else spoken
    return {
        "narration": prefixed,
        "spoken_text": spoken,
        "visual_text": visual,
        "show_opening": True,
    }
