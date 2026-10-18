"""TTS-only text preparation (`prepare_tts_text`).

docs/architecture.md section 5.2 [F-03]: "TTS에 보내는 텍스트는
`md2short.tts_text.prepare_tts_text()`가 생성한다. display script는 유지하고
TTS 전용 입력만 숫자·약어 발음, sentence/list punctuation pause를 보정한다."

`prepare_tts_text()` is a pure function: it takes the display/caption script
text and returns a *new* string with TTS-only pronunciation/pacing fixups
applied. It never mutates its input (Python strings are immutable, so this
is automatic, but the behavior is also covered by an explicit contract test
in tests/test_tts_text.py). The original display text used for captions
elsewhere in the pipeline is completely unaffected by calling this function.

This is explicitly a judgment call (the PRD does not specify an exact
algorithm). The specific, documented transformation rules implemented here,
applied in order:

1. **Percent-sign spacing** — insert exactly one space between a digit and
   an immediately following "%" (collapsing any existing whitespace down to
   one space too), e.g. "50%" -> "50 %". Without the space, ElevenLabs'
   Korean TTS can glue the number and "%" into one odd token instead of
   reading "오십 퍼센트" naturally.

2. **Acronym+number spacing** — insert exactly one space between a run of
   uppercase letters (optionally "&"-joined, e.g. "S&P") and an immediately
   following digit, when there is no space between them already, e.g.
   "S&P500" -> "S&P 500", "AMD10" -> "AMD 10". This keeps TTS from trying to
   sound out a glued alphanumeric token and instead reads the acronym and
   the number as two separate, naturally-paced words. Already-spaced input
   ("S&P 500") is left unchanged (idempotent).

3. **Ellipsis normalization** — collapse a run of 3 or more consecutive
   "." characters into a single Unicode ellipsis character "…", e.g.
   "그래서...그런데...." -> "그래서…그런데…". A run of literal periods can make
   some TTS engines stutter through each "."; a single ellipsis character
   reads as one soft pause instead.

4. **Repeated "!"/"?" collapsing** — collapse any run of 2 or more
   consecutive "!" and/or "?" characters (e.g. "!!!", "??", "?!") down to
   just the first character of the run (e.g. "!!!" -> "!", "?!" -> "?").
   Multiple consecutive exclamation/question marks otherwise read as a
   choppy multi-beat pause in TTS prosody.

5. **List-bullet softening** — at the start of any line, strip a leading
   list-marker bullet ("-", "·", "•", "※", or "*") plus the whitespace after
   it (e.g. a line "- 첫 번째 항목" becomes "첫 번째 항목"). TTS reads the raw
   bullet character as an odd, pause-heavy beat; the rest of the line's
   text is preserved untouched.
"""

from __future__ import annotations

import re

__all__ = ["prepare_tts_text"]

# Rule 1: digit immediately (or with stray whitespace) followed by "%".
_PERCENT_SPACING_RE = re.compile(r"(\d)\s*%")

# Rule 2: uppercase-letter (optionally "&"-joined) run immediately followed
# by a digit, with no space already between them.
_ACRONYM_NUMBER_RE = re.compile(r"([A-Z]+(?:&[A-Z]+)*)(\d)")

# Rule 3: 3+ consecutive literal periods.
_ELLIPSIS_RUN_RE = re.compile(r"\.{3,}")

# Rule 4: 2+ consecutive "!"/"?" characters, in any combination.
_REPEATED_BANG_QUESTION_RE = re.compile(r"[!?]{2,}")

# Rule 5: a leading list-bullet marker at the start of a line.
_LIST_BULLET_RE = re.compile(r"^[ \t]*[-·•※*][ \t]+", re.MULTILINE)


def prepare_tts_text(display_script: str) -> str:
    """Return a TTS-only variant of `display_script` (pure function).

    `display_script` is never modified; a new string is built and returned.
    See the module docstring for the exact, numbered transformation rules
    applied (percent spacing, acronym+number spacing, ellipsis
    normalization, repeated "!"/"?" collapsing, list-bullet softening).
    """
    text = display_script
    text = _PERCENT_SPACING_RE.sub(lambda m: f"{m.group(1)} %", text)
    text = _ACRONYM_NUMBER_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)
    text = _ELLIPSIS_RUN_RE.sub("…", text)
    text = _REPEATED_BANG_QUESTION_RE.sub(lambda m: m.group(0)[0], text)
    text = _LIST_BULLET_RE.sub("", text)
    return text
