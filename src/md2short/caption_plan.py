"""Caption segment planning shared with Remotion `Captions.tsx` (F-05).

See docs/architecture.md section 5.4 and the 품질 기준 row
"자막은 문장 boundary(구두점 또는 gap>=600ms) 기준 분할".

This module is the Python mirror of `Captions.tsx::buildChunksFromWords()`:
both must split the exact same `words.json` into the exact same boundaries,
because `captionSegments` in `props.json` *locks* the Remotion chunk
boundaries (`buildChunksFromSegments`) and the visual cue timeline
(`emoticon_map.plan_cues`, `photo_engine.plan_photo_cues`) to this split.
Any divergence shows up as captions and visuals cutting at different times.

Boundary rules, in the same order the TSX applies them:

1. the word's text ends with sentence-end punctuation (`.!?。！？…`, with
   trailing quotes/brackets tolerated),
2. the silence between this word's `end` and the next word's `start` is
   >= `GAP_BOUNDARY_MS` (600ms),
3. `chunk_size` words have accumulated — a *safety upper bound* for an
   abnormally long run-on sentence, not the primary rule. It always cuts on a
   word boundary (words are never split).

`start`/`end` of every word are passed through untouched (docs/AGENTS.md
section 3.2: `words.json` is the single source of truth for timing).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "SENTENCE_END_RE",
    "GAP_BOUNDARY_MS",
    "DEFAULT_CHUNK_SIZE",
    "CHUNK_SIZE_MIN",
    "CHUNK_SIZE_MAX",
    "TAIL_HOLD_MS",
    "clamp_chunk_size",
    "split_word_spans",
    "build_caption_line_segments",
]

#: Sentence-end punctuation that closes a segment. Mirrors `SENTENCE_END` in
#: `Captions.tsx` — trailing closing quotes/brackets after the punctuation are
#: tolerated (`끝났습니다."`).
SENTENCE_END_RE = re.compile(r"[.!?。！？…]+[\"'”’)\]]*$")

#: Inter-word silence that closes a segment (architecture.md section 5.4).
GAP_BOUNDARY_MS = 600

#: `--chunk-size` default and the 1~24 range enforced by `cli.py`.
DEFAULT_CHUNK_SIZE = 14
CHUNK_SIZE_MIN = 1
CHUNK_SIZE_MAX = 24

#: How long the last segment stays on screen past its last word. Mirrors the
#: `last.end + 400` tail in `Captions.tsx`.
TAIL_HOLD_MS = 400


def clamp_chunk_size(chunk_size: int) -> int:
    """Clamp `chunk_size` into the enforced 1~24 range (품질 기준 "chunk size 1~24")."""
    return min(CHUNK_SIZE_MAX, max(CHUNK_SIZE_MIN, round(chunk_size)))


def _sentence_ends_within(words: list[dict[str, Any]], start_index: int, max_extra: int) -> bool:
    """True if a *real* sentence/gap boundary lands within `max_extra` more words.

    Scans forward from `start_index` (not yet consumed) without mutating
    anything, so `split_word_spans` can decide whether to defer a
    `chunk_size`-forced cut: if the sentence is about to end anyway, cutting
    early just leaves a short, awkward tail span instead of preventing one.

    Running off the end of `words` before finding one is NOT a boundary --
    `split_word_spans` only ever asks this while a further word is already
    known to exist (its own `next_word is None` check always fires first and
    independently), so this would otherwise misreport "found a boundary"
    whenever the lookahead window merely reaches past a short remaining word
    list, deferring cuts far beyond what `chunk_size` is meant to allow (a
    punctuation-free run of N words would collapse into one giant span
    instead of cutting every `chunk_size` words, as its safety-cap purpose
    requires).
    """
    for offset in range(max_extra):
        index = start_index + offset
        if index >= len(words):
            return False
        word = words[index]
        if SENTENCE_END_RE.search(str(word.get("text", "")).strip()):
            return True
        next_word = words[index + 1] if index + 1 < len(words) else None
        if next_word is not None and next_word.get("start", 0) - word.get(
            "end", 0
        ) >= GAP_BOUNDARY_MS:
            return True
    return False


def split_word_spans(
    words: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[list[dict[str, Any]]]:
    """Split `words` into sentence-unit spans, in order, partitioning the input.

    Every word of `words` lands in exactly one span and the original order is
    preserved, so a caller can walk the spans while keeping a running index
    into `words` (that is how `emoticon_map.plan_cues` does its 2-word
    lookahead across a span boundary).

    `chunk_size` is a safety upper bound for an abnormally long run-on
    sentence, not the primary rule (module docstring). A plain word-count cut
    right at `chunk_size` can land a few words before an ordinary sentence
    would have ended anyway, leaving a short, visually jarring tail span with
    its own (likely unmatched) cue -- review.md #9, a real 16-word sentence
    cut into 14+2 with no punctuation at the 14-word mark. So once the cap is
    hit, a cut is deferred (up to `CHUNK_SIZE_MAX` words total) when the
    sentence is about to end anyway; a genuinely long run-on still gets cut,
    just never later than `CHUNK_SIZE_MAX`.

    `words` entries are the `words.json` shape: at least `text` (str) and
    `start`/`end` (int ms). The dicts are not copied or mutated.
    """
    cap = clamp_chunk_size(chunk_size)
    spans: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None

        ends_sentence = bool(SENTENCE_END_RE.search(str(word.get("text", "")).strip()))
        gap_boundary = (
            next_word is not None
            and next_word.get("start", 0) - word.get("end", 0) >= GAP_BOUNDARY_MS
        )

        if next_word is None or ends_sentence or gap_boundary:
            spans.append(current)
            current = []
            continue

        if len(current) >= cap:
            remaining_room = CHUNK_SIZE_MAX - len(current)
            if remaining_room > 0 and _sentence_ends_within(words, index + 1, remaining_room):
                continue
            spans.append(current)
            current = []

    if current:
        spans.append(current)
    return spans


def build_caption_line_segments(
    words: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Split a `words.json` word list into Remotion `CaptionSegment` dicts.

    Returns one dict per sentence-unit segment, in order, each with:

    - `startMs` — the segment's first word `start`
    - `endMs` — the *next* segment's first word `start`, so consecutive
      segments are gapless and cover the whole timeline (this is what lets
      `Captions.tsx::buildChunksFromSegments` and the visual cue layer share
      one boundary list). The final segment holds for `TAIL_HOLD_MS` past its
      last word.
    - `text` — the segment's word texts joined with a single space
    - `words` — the segment's word dicts, in order

    `startMs`/`endMs`/`words` carry Scribe's measured timings verbatim.
    Returns `[]` for an empty word list.
    """
    spans = split_word_spans(words, chunk_size)

    segments: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        next_span = spans[index + 1] if index + 1 < len(spans) else None
        end_ms = (
            next_span[0]["start"] if next_span is not None else span[-1]["end"] + TAIL_HOLD_MS
        )
        segments.append(
            {
                "startMs": span[0]["start"],
                "endMs": end_ms,
                "text": " ".join(str(word.get("text", "")) for word in span),
                "words": list(span),
            }
        )
    return segments
