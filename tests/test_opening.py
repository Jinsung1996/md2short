"""Unit tests for md2short.opening (architecture.md §5.10 / §2.1 / §4.3)."""

from __future__ import annotations

from datetime import date

from md2short.opening import build_opening, build_opening_text

TODAY = date(2026, 5, 2)
NARRATION = "지난 금요일 미국 증시, 강하게 상승 마감했습니다."


def test_build_opening_text_matches_architecture_example():
    spoken, visual = build_opening_text(today=TODAY)

    assert visual == "26년 5월 2일\n데이브리핑"
    assert spoken == "26년 5월 2일 데이브리핑"
    assert "\n" not in spoken


def test_show_opening_true_prepends_spoken_phrase():
    result = build_opening(NARRATION, show_opening=True, today=TODAY)

    assert result["show_opening"] is True
    assert result["spoken_text"] == "26년 5월 2일 데이브리핑"
    assert result["visual_text"] == "26년 5월 2일\n데이브리핑"
    assert result["narration"] == "26년 5월 2일 데이브리핑\n" + NARRATION
    assert result["narration"].startswith(result["spoken_text"])
    assert NARRATION in result["narration"]


def test_spoken_prefix_has_no_literal_newline_inside_the_phrase():
    result = build_opening(NARRATION, show_opening=True, today=TODAY)
    first_line = result["narration"].split("\n")[0]

    assert first_line == "26년 5월 2일 데이브리핑"


def test_show_opening_false_leaves_narration_unchanged():
    result = build_opening(NARRATION, show_opening=False, today=TODAY)

    assert result["narration"] == NARRATION
    assert result["spoken_text"] is None
    assert result["visual_text"] is None
    assert result["show_opening"] is False


def test_empty_narration_yields_opening_only():
    result = build_opening("", show_opening=True, today=TODAY)

    assert result["narration"] == "26년 5월 2일 데이브리핑"


def test_single_digit_day_is_not_zero_padded():
    spoken, visual = build_opening_text(today=date(2027, 11, 7))

    assert visual == "27년 11월 7일\n데이브리핑"
    assert spoken == "27년 11월 7일 데이브리핑"


def test_defaults_to_today_when_date_not_injected():
    spoken, _ = build_opening_text()
    today = date.today()

    assert spoken.startswith(f"{today.year % 100:02d}년 {today.month}월 {today.day}일")
