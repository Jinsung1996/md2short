"""Unit tests for NumberCard extraction (docs/architecture.md section 5.9).

The sample sentences are taken from the real input memo (docs/sample.md) so
the four regex categories are exercised against production-shaped Korean
financial narration.
"""

from md2short.data_extract import (
    DEFAULT_COOLDOWN_MS,
    DEFAULT_DURATION_MS,
    DEFAULT_MAX_CARDS,
    plan_number_cards,
)


def test_percent_category_is_extracted() -> None:
    cards = plan_number_cards("나스닥은 0.94% 상승했습니다.", 10_000)

    assert [card["text"] for card in cards] == ["0.94%"]
    assert cards[0]["label"] is None


def test_percent_word_forms_are_extracted() -> None:
    cards = plan_number_cards("애플은 3 퍼센트 올랐고 샌디스크는 8프로 급등했습니다.", 20_000, cooldown_ms=0)

    assert [card["text"] for card in cards] == ["3 퍼센트", "8프로"]


def test_point_category_is_extracted() -> None:
    cards = plan_number_cards("지수는 30포인트 올라 7,230p 로 마감했습니다.", 20_000, cooldown_ms=0)

    assert [card["text"] for card in cards] == ["30포인트", "7,230p"]


def test_currency_category_is_extracted() -> None:
    cards = plan_number_cards("국방부 계약 규모는 100억 달러, 국내 투자는 3조 원입니다.", 20_000, cooldown_ms=0)

    assert [card["text"] for card in cards] == ["100억 달러", "3조 원"]


def test_index_category_is_extracted_with_the_index_name_as_label() -> None:
    cards = plan_number_cards("S&P 500 종가 7,230 기록했습니다.", 10_000)

    assert cards[0]["text"] == "7,230"
    assert cards[0]["label"] == "S&P 500"


def test_index_branch_yields_to_a_unit_suffixed_number() -> None:
    # "나스닥은 0.94%" must be the percent card, not an index value of 0.94.
    cards = plan_number_cards("나스닥은 0.94% 상승했습니다.", 10_000)

    assert cards[0]["text"] == "0.94%"
    assert cards[0]["label"] is None


def test_all_four_categories_from_one_realistic_paragraph() -> None:
    script = (
        "S&P 500 종가 7,230 기록했습니다. "
        "나스닥은 0.94% 상승했습니다. "
        "국방부 계약은 100억 달러 규모입니다. "
        "코스피는 30포인트 올랐습니다."
    )

    cards = plan_number_cards(script, 60_000, max_cards=10, cooldown_ms=0)

    assert [card["text"] for card in cards] == ["7,230", "0.94%", "100억 달러", "30포인트"]


def test_start_ms_is_proportional_and_inside_the_audio_duration() -> None:
    script = "로블록스, 18% 급락했습니다. " * 4
    audio_duration_ms = 30_000

    cards = plan_number_cards(script, audio_duration_ms, cooldown_ms=0)

    assert len(cards) == 4
    for card in cards:
        assert 0 <= card["startMs"] <= audio_duration_ms
    assert [card["startMs"] for card in cards] == sorted(card["startMs"] for card in cards)
    # first mention sits in the first quarter, last mention in the last quarter
    assert cards[0]["startMs"] < audio_duration_ms / 4
    assert cards[-1]["startMs"] > audio_duration_ms * 3 / 4


def test_cooldown_skips_a_match_too_close_to_the_previous_card() -> None:
    # 5s of audio over a 41-char script puts the three mentions at ~488ms,
    # ~2073ms and ~3780ms. "3.2%" is within 2.5s of the kept "0.94%" and is
    # dropped; "8%" is 3.3s after the last *kept* card, so it comes back.
    script = "나스닥 0.94% 상승, 애플 3.2% 상승, 샌디스크 8% 급등했습니다."

    without_cooldown = plan_number_cards(script, 5_000, cooldown_ms=0)
    with_cooldown = plan_number_cards(script, 5_000, cooldown_ms=DEFAULT_COOLDOWN_MS)

    assert [card["text"] for card in without_cooldown] == ["0.94%", "3.2%", "8%"]
    assert [card["text"] for card in with_cooldown] == ["0.94%", "8%"]
    assert with_cooldown[1]["startMs"] - with_cooldown[0]["startMs"] >= DEFAULT_COOLDOWN_MS

    # Compressed further, every follow-up mention stays inside the cooldown.
    assert [card["text"] for card in plan_number_cards(script, 3_000)] == ["0.94%"]


def test_max_cards_cap_is_enforced_in_document_order() -> None:
    script = " ".join(f"{index}% 변동했습니다." for index in range(1, 21))

    cards = plan_number_cards(script, 120_000, cooldown_ms=0)

    assert len(cards) == DEFAULT_MAX_CARDS == 6
    assert [card["text"] for card in cards] == ["1%", "2%", "3%", "4%", "5%", "6%"]


def test_duration_is_applied_to_every_card() -> None:
    script = " ".join(f"{index}% 변동했습니다." for index in range(1, 6))

    defaults = plan_number_cards(script, 60_000, cooldown_ms=0)
    custom = plan_number_cards(script, 60_000, cooldown_ms=0, duration_ms=1200)

    assert all(card["durationMs"] == DEFAULT_DURATION_MS == 1800 for card in defaults)
    assert all(card["durationMs"] == 1200 for card in custom)


def test_text_with_no_numeric_mention_yields_no_cards() -> None:
    assert plan_number_cards("다우는 소폭 하락했습니다.", 10_000) == []


def test_degenerate_inputs_return_no_cards() -> None:
    assert plan_number_cards("", 10_000) == []
    assert plan_number_cards("나스닥은 0.94% 상승했습니다.", 0) == []
    assert plan_number_cards("나스닥은 0.94% 상승했습니다.", 10_000, max_cards=0) == []
