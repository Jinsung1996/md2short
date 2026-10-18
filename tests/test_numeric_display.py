"""Unit tests for Hangul-acronym display restoration (docs/AGENTS.md section 3.2:
`words.json` timing is the single source of truth and must never be altered
here; only `text` may change)."""

from md2short.numeric_display import (
    ACRONYM_MATCH_THRESHOLD,
    ACRONYM_MAX_SCAN_WORDS,
    NUMBER_MAX_DIGIT_DIFFS,
    apply_numeric_display_overrides,
    apply_script_number_corrections,
)


def _word(text: str, start: int, end: int) -> dict:
    return {"text": text, "start": start, "end": end}


def test_phonetic_variant_replaced_with_canonical_form() -> None:
    words = [_word("S&P", 0, 100), _word("에센피", 100, 500), _word("종가", 500, 800)]
    result = apply_numeric_display_overrides(words)
    assert result[1]["text"] == "S&P500"


def test_real_korean_index_words_are_not_altered() -> None:
    words = [_word("코스피", 0, 300), _word("나스닥", 300, 600), _word("다우", 600, 900)]
    result = apply_numeric_display_overrides(words)
    assert [w["text"] for w in result] == ["코스피", "나스닥", "다우"]


def test_timing_identical_before_and_after_for_every_word() -> None:
    words = [
        _word("지난", 0, 120),
        _word("에센피", 120, 500),
        _word("500", 500, 800),
        _word("코스피", 800, 1100),
        _word("에이엠디", 1100, 1500),
        _word("나스닥", 1500, 1800),
    ]
    original_timings = [(w["start"], w["end"]) for w in words]

    result = apply_numeric_display_overrides(words)

    assert [(w["start"], w["end"]) for w in result] == original_timings
    # Also confirm the input list itself was not mutated in place.
    assert [(w["start"], w["end"]) for w in words] == original_timings


def test_already_correct_acronym_token_is_left_alone() -> None:
    words = [_word("S&P500", 0, 400), _word("AMD", 400, 700), _word("AI", 700, 900)]
    result = apply_numeric_display_overrides(words)
    assert [w["text"] for w in result] == ["S&P500", "AMD", "AI"]


def test_threshold_and_scan_window_constants_match_spec() -> None:
    assert ACRONYM_MATCH_THRESHOLD == 0.86
    assert ACRONYM_MAX_SCAN_WORDS == 96


def test_near_miss_spelling_within_threshold_still_matches() -> None:
    # "에이엠디이" is "에이엠디" (AMD) with one extra trailing syllable —
    # SequenceMatcher ratio 0.8889, above ACRONYM_MATCH_THRESHOLD (0.86).
    words = [_word("에이엠디이", 0, 400)]
    result = apply_numeric_display_overrides(words)
    assert result[0]["text"] == "AMD"


def test_clearly_unrelated_text_below_threshold_does_not_match() -> None:
    # Ratio against every known variant is 0.0 — far below threshold.
    words = [_word("완전히다른단어", 0, 400)]
    result = apply_numeric_display_overrides(words)
    assert result[0]["text"] == "완전히다른단어"


def test_fuzzy_match_not_applied_beyond_scan_window() -> None:
    filler = [_word("그냥", i * 10, i * 10 + 5) for i in range(ACRONYM_MAX_SCAN_WORDS)]
    words = filler + [_word("에센피", 2000, 2300)]
    result = apply_numeric_display_overrides(words)
    # The variant word sits at index ACRONYM_MAX_SCAN_WORDS, outside the
    # [0, ACRONYM_MAX_SCAN_WORDS) scan window, so it must be left unchanged.
    assert result[ACRONYM_MAX_SCAN_WORDS]["text"] == "에센피"
    # Timing is still preserved for that word even though text is untouched.
    assert result[ACRONYM_MAX_SCAN_WORDS]["start"] == 2000
    assert result[ACRONYM_MAX_SCAN_WORDS]["end"] == 2300


# --- apply_script_number_corrections --------------------------------------
# Repro (job sf_20261012_e6cd2a): the script said "S&P 500 종가 7,230" but
# Scribe transcribed "7,030" into words.json, so the caption showed 7,030 while
# the NumberCard sticker for the same figure showed 7,230.


def test_misheard_digit_is_restored_from_the_script() -> None:
    reference = "S&P 500 종가 7,230 기록했습니다."
    words = [
        _word("S&P", 27000, 27400),
        _word("500", 27400, 27880),
        _word("종가", 27880, 28200),
        _word("7,030", 28200, 28580),
        _word("기록했습니다.", 28580, 29400),
    ]
    original_timings = [(w["start"], w["end"]) for w in words]

    result = apply_script_number_corrections(words, reference)

    assert result[3]["text"] == "7,230"
    # Every word's timing survives byte-for-byte (docs/AGENTS.md §3.2), and the
    # only text that changed is the mis-heard number.
    assert [(w["start"], w["end"]) for w in result] == original_timings
    assert [(w["start"], w["end"]) for w in words] == original_timings
    assert [w["text"] for w in result] == [
        "S&P",
        "500",
        "종가",
        "7,230",
        "기록했습니다.",
    ]


def test_correctly_recognized_numbers_are_left_exactly_as_transcribed() -> None:
    reference = "나스닥은 0.94% 상승했고 로블록스는 18% 급락했습니다."
    words = [
        _word("나스닥은", 0, 400),
        _word("0.94%", 400, 900),
        _word("상승했고", 900, 1300),
        _word("로블록스는", 1300, 1800),
        _word("18%", 1800, 2200),
        _word("급락했습니다.", 2200, 2800),
    ]
    original_texts = [w["text"] for w in words]

    result = apply_script_number_corrections(words, reference)

    assert [w["text"] for w in result] == original_texts


def test_numeric_sequence_length_mismatch_corrects_nothing() -> None:
    # The script carries three numbers but Scribe emitted only two numeric
    # words (as if "18%" had been dropped), so positional pairing can no longer
    # be trusted and the whole pass must fail safe.
    reference = "종가 7,230, 나스닥 0.94%, 로블록스 18% 급락했습니다."
    words = [
        _word("종가", 0, 300),
        _word("7,030", 300, 800),
        _word("나스닥", 800, 1200),
        _word("0.94%", 1200, 1700),
    ]
    original_texts = [w["text"] for w in words]

    result = apply_script_number_corrections(words, reference)

    assert [w["text"] for w in result] == original_texts


def test_two_unrelated_numbers_at_the_same_position_are_not_force_matched() -> None:
    # Same digit count, but every digit differs — these are two different
    # numbers, not one mis-heard number.
    words = [_word("9999", 0, 500)]

    result = apply_script_number_corrections(words, "기준선은 1111 입니다.")

    assert result[0]["text"] == "9999"


def test_short_numbers_sharing_no_digits_are_not_force_matched() -> None:
    # "18" vs "80" differs in both positions: within NUMBER_MAX_DIGIT_DIFFS in
    # absolute terms, but it shares none of its digits, so it is still rejected.
    assert NUMBER_MAX_DIGIT_DIFFS == 2
    words = [_word("80%", 0, 400)]

    result = apply_script_number_corrections(words, "로블록스는 18% 급락했습니다.")

    assert result[0]["text"] == "80%"


def test_trailing_suffix_of_the_asr_word_is_preserved() -> None:
    # Only the numeric prefix is ever replaced: a Korean unit suffix and a
    # sentence-final period both have to survive the rewrite.
    reference = "목표가는 7,230 이고 거래대금은 500 수준입니다."
    words = [
        _word("7,030.", 0, 500),
        _word("530원", 500, 1000),
    ]

    result = apply_script_number_corrections(words, reference)

    assert [w["text"] for w in result] == ["7,230.", "500원"]


def test_reference_without_numbers_is_a_no_op() -> None:
    words = [_word("7,030", 0, 500), _word("기록했습니다.", 500, 1200)]

    result = apply_script_number_corrections(words, "숫자가 전혀 없는 문장입니다.")

    assert [w["text"] for w in result] == ["7,030", "기록했습니다."]
