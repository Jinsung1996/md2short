"""Unit tests for sentence-unit caption segmentation (docs/architecture.md
section 5.4). These boundaries must stay identical to Remotion
`Captions.tsx::buildChunksFromWords()`."""

from md2short.caption_plan import (
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    GAP_BOUNDARY_MS,
    TAIL_HOLD_MS,
    build_caption_line_segments,
    clamp_chunk_size,
    split_word_spans,
)


def _words(texts: list[str], step: int = 300, gap: int = 0) -> list[dict]:
    """Build a word list with `step`ms words separated by `gap`ms of silence."""
    words = []
    cursor = 0
    for text in texts:
        words.append({"text": text, "start": cursor, "end": cursor + step})
        cursor += step + gap
    return words


def test_punctuation_closes_a_segment() -> None:
    words = _words(["지난", "금요일", "미국", "증시,", "상승", "마감했습니다.", "5주", "연속", "상승입니다."])

    segments = build_caption_line_segments(words)

    assert [segment["text"] for segment in segments] == [
        "지난 금요일 미국 증시, 상승 마감했습니다.",
        "5주 연속 상승입니다.",
    ]


def test_trailing_quote_after_punctuation_still_closes_a_segment() -> None:
    words = _words(["핵심은", "연속", "상승이라는", "점입니다.”", "다음", "주가", "중요합니다."])

    segments = build_caption_line_segments(words)

    assert len(segments) == 2
    assert segments[0]["text"].endswith("점입니다.”")


def test_gap_at_or_above_boundary_closes_a_segment() -> None:
    # No punctuation anywhere: only the silence can create the boundary.
    words = [
        {"text": "나스닥은", "start": 0, "end": 400},
        {"text": "상승", "start": 400, "end": 800},
        {"text": "다우는", "start": 800 + GAP_BOUNDARY_MS, "end": 1600},
        {"text": "하락", "start": 1600, "end": 2000},
    ]

    segments = build_caption_line_segments(words)

    assert [segment["text"] for segment in segments] == ["나스닥은 상승", "다우는 하락"]


def test_gap_just_below_boundary_does_not_split() -> None:
    words = [
        {"text": "나스닥은", "start": 0, "end": 400},
        {"text": "상승", "start": 400 + GAP_BOUNDARY_MS - 1, "end": 1400},
    ]

    segments = build_caption_line_segments(words)

    assert len(segments) == 1


def test_chunk_size_cap_splits_a_run_on_sentence() -> None:
    # 30 words, no punctuation, no gaps -- only the safety cap can split it.
    words = _words([f"단어{index}" for index in range(30)])

    segments = build_caption_line_segments(words, chunk_size=7)

    assert [len(segment["words"]) for segment in segments] == [7, 7, 7, 7, 2]
    assert all(len(segment["words"]) <= 7 for segment in segments)


def test_chunk_size_cap_defers_a_cut_when_the_sentence_ends_soon_after() -> None:
    # Repro (review.md #9, job sf_20261012_fca08b): a real 16-word sentence
    # with no internal punctuation used to get force-cut at the chunk_size=14
    # mark, leaving an awkward 2-word tail span ("진입한 신호입니다.") with its
    # own (unmatched) emoticon cue flashing for under 2 seconds. The sentence
    # ends 2 words past the cap, well within CHUNK_SIZE_MAX (24), so the cut
    # must be deferred until the real sentence-end punctuation.
    words = _words(
        [
            "오라클이", "미", "국방부", "AI", "네트워크", "프로젝트에", "공식",
            "합류했으며", "정부의", "AI", "시장이", "본격", "매출", "사이클에",
            "진입한", "신호입니다.",
        ]
    )

    segments = build_caption_line_segments(words, chunk_size=14)

    assert len(segments) == 1
    assert len(segments[0]["words"]) == 16
    assert segments[0]["text"].endswith("신호입니다.")


def test_chunk_size_cap_still_fires_when_the_sentence_never_ends_within_chunk_size_max() -> None:
    # A genuinely long run-on (no punctuation anywhere) must still be capped
    # at CHUNK_SIZE_MAX overall -- deferral is bounded, not unlimited.
    words = _words([f"단어{index}" for index in range(40)], gap=0)

    segments = build_caption_line_segments(words, chunk_size=14)

    assert all(len(segment["words"]) <= CHUNK_SIZE_MAX for segment in segments)
    assert sum(len(segment["words"]) for segment in segments) == 40


def test_default_chunk_size_cap_is_applied_to_a_run_on_sentence() -> None:
    words = _words([f"단어{index}" for index in range(40)])

    segments = build_caption_line_segments(words)

    assert all(len(segment["words"]) <= 14 for segment in segments)
    assert len(segments) == 3  # 14 + 14 + 12


def test_chunk_size_is_clamped_into_the_enforced_range() -> None:
    assert clamp_chunk_size(0) == CHUNK_SIZE_MIN
    assert clamp_chunk_size(-5) == CHUNK_SIZE_MIN
    assert clamp_chunk_size(99) == CHUNK_SIZE_MAX

    words = _words([f"단어{index}" for index in range(60)])
    assert all(
        len(segment["words"]) <= CHUNK_SIZE_MAX
        for segment in build_caption_line_segments(words, chunk_size=99)
    )
    assert all(
        len(segment["words"]) == CHUNK_SIZE_MIN
        for segment in build_caption_line_segments(words, chunk_size=0)
    )


def test_segments_are_gapless_and_hold_the_tail() -> None:
    words = _words(["첫", "문장입니다.", "두", "번째", "문장입니다."])

    segments = build_caption_line_segments(words)

    for current, following in zip(segments, segments[1:]):
        assert current["endMs"] == following["startMs"]
    assert segments[0]["startMs"] == words[0]["start"]
    assert segments[-1]["endMs"] == words[-1]["end"] + TAIL_HOLD_MS


def test_spans_partition_the_input_in_order() -> None:
    words = _words(["하나.", "둘", "셋.", "넷", "다섯", "여섯."])

    spans = split_word_spans(words)

    flattened = [word for span in spans for word in span]
    assert flattened == words


def test_empty_word_list_returns_no_segments() -> None:
    assert build_caption_line_segments([]) == []
    assert split_word_spans([]) == []
