"""Numeric/acronym display-text correction (docs/architecture.md section 1.5, 1.2).

ElevenLabs Scribe transcribes the TTS narration audio back into Korean text to
produce word-level timestamps (`words.json`). Two classes of ASR error are
corrected here, each by its own entry point:

1. `apply_numeric_display_overrides()` — when the narration speaks an English
   stock/index acronym such as "S&P500", Scribe sometimes renders the *sound*
   of the acronym as a Hangul phonetic transliteration (e.g. "에센피" for
   "S&P500") instead of the original written form. This detects those
   phonetic Hangul renderings and rewrites the *display text* back to the
   original acronym/number form, so captions show "S&P500" instead of "에센피".
2. `apply_script_number_corrections()` — Scribe can also simply mis-hear a
   *digit* ("7,230" transcribed as "7,030"). Captions render from
   `words.json`, while the NumberCard stickers render from the script text, so
   an ASR digit error puts two different values for the same figure on screen
   seconds apart — unacceptable for financial content. This cross-checks the
   transcribed numbers against the narration script that was actually sent to
   TTS and restores the script's digits.

Critical invariant (docs/AGENTS.md section 3.2 — `words.json` is the single
source of truth for timing): this module only ever rewrites the `text` field.
It never changes `start`/`end`, which must stay exactly as Scribe measured
them.

Scope notes:
- Real Korean words that happen to resemble transliterations (코스피, 나스닥,
  etc.) are *not* in `_ACRONYM_KO_VARIANTS` and are therefore left untouched.
- Multi-word token reassembly (e.g. Scribe splitting "S&P500" across two
  separate word entries, "S&P" + "500") is out of scope here — handling it
  correctly would require cross-word lookahead/merge logic and a timing
  policy for the merged span, which is a larger change. The narrower,
  reliably-scoped behavior implemented below is: if a single word's text
  *already* matches `_ACRONYM_TOKEN_RE` in full (e.g. Scribe already wrote
  "S&P500" or "AMD" correctly as one word), it is left alone and is never
  passed through the fuzzy Hangul matcher — this is what keeps the fuzzy
  matcher from ever mangling an already-correct acronym token.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

# Matches an acronym written as a single, already-correct token, in one of
# three shapes:
#   1. letters (optionally "&"-joined) immediately followed by digits,
#      e.g. "S&P500", "AMD10"
#   2. letters joined by "&" with no trailing digits, e.g. "S&P"
#   3. a bare multi-letter acronym/ticker with no digits, e.g. "AMD", "AI"
# Used with `.fullmatch()` against a single word's text to recognize tokens
# that Scribe already wrote correctly, so they are never re-written by the
# fuzzy Hangul matcher below.
_ACRONYM_TOKEN_RE = re.compile(
    r"[A-Z]+(?:&[A-Z]+)*\d+"
    r"|[A-Z]+(?:&[A-Z]+)+"
    r"|[A-Z]{2,}"
)

# Known Hangul phonetic transliterations of English financial acronyms, as
# Scribe may render them, mapped to the canonical display form that should
# appear in captions. Deliberately excludes real Korean words for real
# indices/terms (코스피, 나스닥, 다우, 니케이, 항셍, ...) — those are not
# phonetic transliterations of an English acronym and must pass through
# unchanged.
_ACRONYM_KO_VARIANTS: dict[str, str] = {
    "에센피": "S&P500",  # contracted/slurred "에스앤피" (S&P + 500)
    "에스앤피": "S&P",  # letter-by-letter "S", "&"("and"), "P"
    "에이엠디": "AMD",  # letter-by-letter "A", "M", "D"
    "에이아이": "AI",  # letter-by-letter "A", "I"
}

# Fuzzy-match tuning (docs/architecture.md section 7 "품질 기준"): a word's
# text must score >= this `difflib.SequenceMatcher` ratio against a known
# variant to be treated as a match.
ACRONYM_MATCH_THRESHOLD = 0.86

# Bound on how many leading words of the list are scanned for fuzzy acronym
# matches, to cap worst-case cost on long transcripts.
ACRONYM_MAX_SCAN_WORDS = 96


def _best_variant_match(text: str) -> str | None:
    """Return the canonical display form matching `text`, or `None`.

    Checks for an exact `_ACRONYM_KO_VARIANTS` key match first, then falls
    back to a bounded fuzzy match: the canonical form of whichever variant
    has the highest `difflib.SequenceMatcher` ratio against `text`, provided
    that ratio is >= `ACRONYM_MATCH_THRESHOLD`.
    """
    exact = _ACRONYM_KO_VARIANTS.get(text)
    if exact is not None:
        return exact

    best_ratio = 0.0
    best_canonical: str | None = None
    for variant, canonical in _ACRONYM_KO_VARIANTS.items():
        ratio = difflib.SequenceMatcher(None, text, variant).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_canonical = canonical

    if best_ratio >= ACRONYM_MATCH_THRESHOLD:
        return best_canonical
    return None


def apply_numeric_display_overrides(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new word list with Hangul-transliterated acronyms restored.

    Each entry is expected to be a plain dict with at least "text", "start",
    and "end" keys (structurally compatible with `md2short.schema.CaptionWord`
    or any other word-level timestamp record). For every entry in `words`:

    - `start`/`end` (and any other keys) are copied through unchanged.
    - If the entry's `text` already matches `_ACRONYM_TOKEN_RE` in full
      (e.g. already "S&P500" or "AMD"), it is left as-is.
    - Otherwise, within the first `ACRONYM_MAX_SCAN_WORDS` entries, if
      `text` exactly or fuzzy-matches (per `_best_variant_match`) a known
      `_ACRONYM_KO_VARIANTS` entry, `text` is replaced with the canonical
      display form.
    - Entries beyond the scan window, and entries with no match, keep their
      original `text`.
    """
    result: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        new_word = dict(word)
        text = word.get("text", "")

        if isinstance(text, str) and not _ACRONYM_TOKEN_RE.fullmatch(text):
            if index < ACRONYM_MAX_SCAN_WORDS:
                match = _best_variant_match(text)
                if match is not None:
                    new_word["text"] = match

        result.append(new_word)

    return result


# --- ASR digit mishearing correction --------------------------------------
# Matches one numeric token: a digit run that may carry inner group
# separators/decimal points and an optional trailing "%" ("7,230", "0.94%",
# "500", "2024"). The `(?:[\d,.]*\d)?` middle deliberately requires the run to
# *end* on a digit, so a sentence-final period or a Korean suffix stays outside
# the match ("7,030." -> "7,030", "500원" -> "500"); that trailing remainder is
# what `apply_script_number_corrections` preserves when it rewrites a word.
_NUMERIC_TOKEN_RE = re.compile(r"\d(?:[\d,.]*\d)?%?")

#: How many digit positions may differ before a reference/ASR pair is judged to
#: be two *different* numbers rather than one mis-heard number. Scribe's real
#: failure mode is a single substituted digit ("7,230" -> "7,030"), so the
#: tolerance stays tight.
NUMBER_MAX_DIGIT_DIFFS = 2


def _digits_only(token: str) -> str:
    """The token's digits with group separators, decimal points and "%" removed."""
    return "".join(char for char in token if char.isdigit())


def _is_misheard_number(reference: str, heard: str) -> bool:
    """True if `heard` is plausibly `reference` with a digit or two mis-heard.

    The pair must have the same digit count (a different length means Scribe
    heard a different figure, not a wrong digit), must actually differ (an
    exactly-correct ASR number is never rewritten), and must still share *most*
    of its digits: at most `NUMBER_MAX_DIGIT_DIFFS` positions differ **and** no
    more than half the digits differ. The second condition is what stops two
    unrelated short numbers from being force-matched — "18" vs "80" differs in
    both of its two positions and shares nothing, so it is rejected even though
    it is within the absolute tolerance.
    """
    reference_digits = _digits_only(reference)
    heard_digits = _digits_only(heard)
    if not reference_digits or len(reference_digits) != len(heard_digits):
        return False

    diffs = sum(1 for ref, got in zip(reference_digits, heard_digits) if ref != got)
    if diffs == 0:
        return False
    return diffs <= NUMBER_MAX_DIGIT_DIFFS and diffs * 2 <= len(reference_digits)


def apply_script_number_corrections(
    words: list[dict[str, Any]], reference_text: str
) -> list[dict[str, Any]]:
    """Return a new word list with mis-heard digits restored from the script.

    `reference_text` is the narration that was actually sent to TTS
    (`script_result["narration"]`, opening phrase included), so its numeric
    tokens appear in the same order as the spoken audio Scribe transcribed.
    `tts_text.prepare_tts_text()` only adjusts spacing *around* numbers, never
    the digit runs themselves, so the two orderings line up.

    Alignment is positional: the ordered numeric tokens of `reference_text` are
    paired index-by-index with the ordered words of `words` whose `text` starts
    with a numeric token. The whole pass **fails safe** — if the two sequences
    have different lengths, something other than a digit diverged (a dropped or
    merged word, a number read out in Hangul, ...) and positional pairing can no
    longer be trusted, so nothing at all is corrected.

    For each surviving pair, `_is_misheard_number()` decides whether the two are
    one number mis-heard or two genuinely different numbers; only the former is
    rewritten, and only its numeric prefix is replaced — with the reference's
    exact original formatting, keeping any trailing non-numeric remainder of the
    ASR word ("7,030." -> "7,230.", "500원" -> "500원").

    `start`/`end` (and every other key) are copied through untouched for every
    word, corrected or not (docs/AGENTS.md section 3.2).
    """
    result = [dict(word) for word in words]

    reference_numbers = _NUMERIC_TOKEN_RE.findall(reference_text or "")
    if not reference_numbers:
        return result

    # (index into `result`, the word's leading numeric token), in word order.
    heard_numbers: list[tuple[int, str]] = []
    for index, word in enumerate(result):
        text = word.get("text", "")
        if not isinstance(text, str):
            continue
        match = _NUMERIC_TOKEN_RE.match(text)
        if match is not None:
            heard_numbers.append((index, match.group(0)))

    if len(heard_numbers) != len(reference_numbers):
        return result

    for (index, heard), reference in zip(heard_numbers, reference_numbers):
        if not _is_misheard_number(reference, heard):
            continue
        suffix = result[index]["text"][len(heard) :]
        result[index]["text"] = reference + suffix

    return result
