"""Unit tests for md2short.tts_text.prepare_tts_text.

Verifies each documented transformation rule individually, plus the pure-
function / no-mutation design contract.
"""

from __future__ import annotations

from md2short.tts_text import prepare_tts_text


def test_percent_sign_spacing_inserted():
    assert prepare_tts_text("오늘 수익률은 50%입니다") == "오늘 수익률은 50 %입니다"


def test_percent_sign_spacing_idempotent_when_already_spaced():
    assert prepare_tts_text("50 %입니다") == "50 %입니다"


def test_acronym_number_spacing_inserted():
    assert prepare_tts_text("S&P500 지수가 올랐다") == "S&P 500 지수가 올랐다"


def test_acronym_number_spacing_plain_acronym():
    assert prepare_tts_text("AMD10 주가") == "AMD 10 주가"


def test_acronym_number_spacing_left_unchanged_when_already_spaced():
    assert prepare_tts_text("S&P 500 지수") == "S&P 500 지수"


def test_ellipsis_run_normalized_to_single_char():
    assert prepare_tts_text("그래서...그런데....") == "그래서…그런데…"


def test_repeated_exclamation_collapsed():
    assert prepare_tts_text("정말 대단해요!!!") == "정말 대단해요!"


def test_repeated_question_collapsed():
    assert prepare_tts_text("진짜요??") == "진짜요?"


def test_mixed_bang_question_collapsed_to_first_char():
    assert prepare_tts_text("정말이야?!") == "정말이야?"


def test_list_bullet_stripped_at_line_start():
    display = "오늘의 요약\n- 첫 번째 항목\n- 두 번째 항목"
    assert prepare_tts_text(display) == "오늘의 요약\n첫 번째 항목\n두 번째 항목"


def test_list_bullet_variants_stripped():
    display = "· 하나\n• 둘\n※ 셋\n* 넷"
    assert prepare_tts_text(display) == "하나\n둘\n셋\n넷"


def test_combined_transformations_differ_from_input():
    display = "S&P500 지수가 50% 올랐대요!!\n- 정말요...?"
    result = prepare_tts_text(display)
    assert result != display
    assert "S&P 500" in result
    assert "50 %" in result
    assert "올랐대요!" in result
    assert "!!" not in result


def test_plain_text_without_triggers_is_unchanged():
    display = "오늘은 평범한 하루였습니다."
    assert prepare_tts_text(display) == display


def test_original_input_not_mutated_design_contract():
    """prepare_tts_text is a pure function: the caller's original string
    variable must be unchanged after the call (design contract, asserted
    explicitly even though Python strings are immutable)."""
    original = "S&P500 지수가 50%!!! 올랐습니다...\n- 항목"
    display_script = original

    result = prepare_tts_text(display_script)

    assert display_script == original
    assert display_script is original
    assert result != original
