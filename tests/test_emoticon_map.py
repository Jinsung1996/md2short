"""Unit tests for 호랑이 cue planning (docs/architecture.md section 5.8).

The two properties that matter most here are **asset coverage** (every PNG in
`emoticon/emoticon/` must be reachable) and **determinism** (same input ->
byte-identical cue list, which is the whole reason emoticon mode exists).
"""

from pathlib import Path

from md2short.emoticon_map import (
    EMOTICON_KEYWORDS,
    FALLBACK_CATEGORIES,
    MAX_CUES_DEFAULT,
    NEUTRAL_DEFAULT,
    NEUTRAL_DEFAULT_POOL,
    SPARSE_COOLDOWN_MS,
    SPARSE_CUE_DURATION_MS,
    SPARSE_MIN_CUE_DURATION_MS,
    plan_cues,
    select_emoticon,
)

EMOTICON_DIR = Path(__file__).resolve().parents[1] / "emoticon" / "emoticon"


def _asset_file_names() -> set[str]:
    """Read the real asset directory -- never hardcode a count that can drift."""
    return {path.name for path in EMOTICON_DIR.glob("*.png")}


def _words(texts: list[str], step: int = 300, gap: int = 0) -> list[dict]:
    words = []
    cursor = 0
    for text in texts:
        words.append({"text": text, "start": cursor, "end": cursor + step})
        cursor += step + gap
    return words


def test_every_asset_png_is_reachable_through_a_keyword() -> None:
    assets = _asset_file_names()
    assert assets, f"no PNG assets found under {EMOTICON_DIR}"

    mapped = {file_name for names in EMOTICON_KEYWORDS.values() for file_name in names}

    assert assets - mapped == set(), "assets with no keyword"
    assert mapped - assets == set(), "keywords pointing at a missing file"


def test_neutral_default_is_a_real_asset() -> None:
    assert NEUTRAL_DEFAULT in _asset_file_names()


def test_neutral_default_pool_entries_are_real_assets_and_not_category_images() -> None:
    assets = _asset_file_names()
    category_images = {file_name for _category, _stems, file_name in FALLBACK_CATEGORIES}

    assert len(NEUTRAL_DEFAULT_POOL) > 1, "a pool of one is not a rotation"
    assert len(set(NEUTRAL_DEFAULT_POOL)) == len(NEUTRAL_DEFAULT_POOL), "no duplicates"
    for file_name in NEUTRAL_DEFAULT_POOL:
        assert file_name in assets
        # Membership-based detection in plan_cues (`file_name in NEUTRAL_DEFAULT_POOL`)
        # only works if no category fallback image is also a pool member.
        assert file_name not in category_images
    assert NEUTRAL_DEFAULT == NEUTRAL_DEFAULT_POOL[0]


def test_fill_gaps_makes_one_cue_per_span_covering_the_whole_timeline() -> None:
    words = _words(
        ["나스닥은", "상승했습니다.", "다우는", "하락했습니다.", "반도체는", "급등했습니다."]
    )

    cues = plan_cues(words)

    assert len(cues) == 3
    for cue in cues:
        assert cue["assetKind"] == "emoticon"
        assert cue["visualMode"] == "emoticon"
        assert cue["emoticonFileName"] in _asset_file_names()
        assert cue["durationMs"] > 0
    # gapless: each cue ends exactly where the next one starts
    for current, following in zip(cues, cues[1:]):
        assert current["startMs"] + current["durationMs"] == following["startMs"]
    assert cues[0]["startMs"] == words[0]["start"]
    assert cues[-1]["startMs"] + cues[-1]["durationMs"] > words[-1]["end"]


def test_fill_gaps_ignores_max_cues() -> None:
    words = _words([f"문장{index}입니다." for index in range(40)])

    cues = plan_cues(words, max_cues=5)

    assert len(cues) == 40


def test_longer_keyword_wins_over_a_shorter_overlapping_one() -> None:
    # "매도" -> 17_익절각_매도.png, "반도체" -> 26_반도체급등_로켓상승.png,
    # "반도체 매도" -> 41_반도체매도_주가하락.png. The longest must win.
    file_name, keyword = select_emoticon("반도체 매도 물량이 쏟아졌습니다.")

    assert keyword == "반도체 매도"
    assert file_name == "41_반도체매도_주가하락.png"

    assert select_emoticon("매도 물량이 쏟아졌습니다.")[1] == "매도"
    assert select_emoticon("반도체 업종입니다.")[1] == "반도체"


def test_keyword_matches_across_whitespace_variants() -> None:
    assert select_emoticon("오늘 실적 발표가 있습니다.")[1] == "실적 발표"
    assert select_emoticon("오늘 실적발표가 있습니다.")[1] == "실적 발표"


def test_lookahead_two_words_resolves_a_boundary_crossing_keyword() -> None:
    # Span 1 ("그래서 말이죠.") has no keyword and no category stem; the next
    # span's first word does. Lookahead must pull it in -- and since that
    # gives both spans the same image, the adjacent-same-image merge (#10)
    # collapses them into one cue. If lookahead had failed, span 1 would
    # have resolved to a *different* (neutral) image and no merge would
    # happen, so `len(cues) == 1` here is itself evidence lookahead worked,
    # not just a side effect of the merge.
    words = _words(["그래서", "말이죠.", "폭락했습니다."])

    cues = plan_cues(words)

    assert len(cues) == 1
    assert cues[0]["keyword"] == "폭락"
    assert cues[0]["emoticonFileName"] == "28_폭락_패닉하락.png"
    assert cues[0]["startMs"] == words[0]["start"]


def test_lookahead_does_not_reach_a_third_word() -> None:
    words = _words(["그래서", "말이죠.", "오늘", "장은", "폭락했습니다."])

    cues = plan_cues(words)

    assert cues[0]["keyword"] is None
    assert cues[0]["emoticonFileName"] == NEUTRAL_DEFAULT


def test_category_fallback_precedes_the_neutral_default() -> None:
    # "올랐" is a category stem (rise), not a keyword.
    file_name, keyword = select_emoticon("지수가 크게 올랐습니다.")

    assert keyword is None
    assert file_name == "09_가즈아_상승.png"


def test_generic_government_mention_does_not_select_the_rate_freeze_image() -> None:
    # Repro (job sf_20261012_e6cd2a): "정부" used to be a `policy` category stem,
    # so this sentence — which says nothing about interest rates — got the
    # FED rate-freeze tiger. A neutral image is the only defensible choice here.
    file_name, keyword = select_emoticon("이는 정부의 AI 시장이 본격 매출 사이클에 진입한 신호입니다")

    assert keyword is None
    assert file_name == NEUTRAL_DEFAULT


def test_generic_policy_words_alone_do_not_select_the_rate_freeze_image() -> None:
    assert select_emoticon("정부 정책 당국이 새 지원안을 내놨습니다")[0] == NEUTRAL_DEFAULT


def test_rate_decision_sentences_still_reach_the_rate_freeze_image() -> None:
    # Narrowing the stems must not make asset 45 unreachable: via a keyword...
    assert select_emoticon("기준금리를 그대로 유지했습니다")[0] == "45_금리동결_FED기준금리.png"
    # ...and via the narrowed `policy` category, for a rate sentence that
    # carries none of the keyword-dictionary terms.
    file_name, keyword = select_emoticon("시장은 이번 회의에서 동결을 예상합니다")
    assert keyword is None
    assert file_name == "45_금리동결_FED기준금리.png"


def test_no_match_at_all_falls_back_to_the_neutral_pool() -> None:
    words = _words(["그래서", "말이죠.", "오늘은", "여기까지입니다."])

    cues = plan_cues(words)

    file_names = [cue["emoticonFileName"] for cue in cues]
    assert [cue["keyword"] for cue in cues] == [None, None]
    assert all(name in NEUTRAL_DEFAULT_POOL for name in file_names)
    # review.md #7: two consecutive unmatched spans must not repeat the same
    # tiger -- that monotony is exactly what the rotation exists to fix.
    assert file_names[0] != file_names[1]
    assert file_names == [NEUTRAL_DEFAULT_POOL[0], NEUTRAL_DEFAULT_POOL[1]]


def test_neutral_pool_rotation_wraps_around_and_is_deterministic() -> None:
    # Five unmatched spans against a 4-entry pool: index 4 must wrap to 0.
    words = _words([f"그래서{index} 말이죠." for index in range(5)])

    first = plan_cues(words)
    second = plan_cues(words)

    file_names = [cue["emoticonFileName"] for cue in first]
    assert file_names == [
        NEUTRAL_DEFAULT_POOL[0],
        NEUTRAL_DEFAULT_POOL[1],
        NEUTRAL_DEFAULT_POOL[2],
        NEUTRAL_DEFAULT_POOL[3],
        NEUTRAL_DEFAULT_POOL[0],
    ]
    assert first == second, "rotation must not break determinism"


def test_category_fallback_does_not_advance_the_neutral_rotation() -> None:
    # "올랐습니다" hits the `rise` category stem (09_...), not the neutral
    # pool, for the first span; the second span is a genuine no-match. If a
    # category hit incorrectly advanced the rotation counter, the second span
    # would land on pool index 1 instead of 0.
    words = _words(["지수가", "올랐습니다.", "그래서", "말이죠."])

    cues = plan_cues(words)

    assert cues[0]["keyword"] is None
    assert cues[0]["emoticonFileName"] == "09_가즈아_상승.png"
    assert cues[1]["keyword"] is None
    assert cues[1]["emoticonFileName"] == NEUTRAL_DEFAULT_POOL[0]


def test_select_emoticon_neutral_rotation_index_wraps_and_defaults_to_zero() -> None:
    assert select_emoticon("그래서 말이죠.")[0] == NEUTRAL_DEFAULT_POOL[0] == NEUTRAL_DEFAULT
    assert select_emoticon("그래서 말이죠.", neutral_rotation_index=1)[0] == NEUTRAL_DEFAULT_POOL[1]
    assert select_emoticon(
        "그래서 말이죠.", neutral_rotation_index=len(NEUTRAL_DEFAULT_POOL)
    )[0] == NEUTRAL_DEFAULT_POOL[0]
    # A matched keyword must never be affected by the rotation index.
    assert select_emoticon("실적 발표가 있습니다.", neutral_rotation_index=3)[1] == "실적 발표"


def test_adjacent_cues_with_the_same_image_are_merged_into_one() -> None:
    # Repro (review.md #10, job sf_20261012_fca08b): "5주 연속 상승" matched
    # in two consecutive sentences (hook, then the start of body) -- a
    # script-quality question, not a wrong-image one (#2's dedup correctly
    # leaves this alone; the two sentences aren't near-duplicates). Showing
    # the one genuinely correct image across both spans as a single
    # continuous cue, instead of two back-to-back Ken-Burns resets of the
    # identical picture, keeps the content accurate while fixing the visual
    # stutter.
    words = _words(
        [
            "핵심은", "5주", "연속", "상승이라는", "점입니다.",
            "이번", "랠리는", "5주", "연속", "상승으로", "최장", "랠리입니다.",
            "4월", "기준으로는", "좋은", "성과입니다.",
        ]
    )

    cues = plan_cues(words)

    rally_cues = [cue for cue in cues if cue["emoticonFileName"] == "30_랠리_상승파티.png"]
    assert len(rally_cues) == 1, "the two same-image spans must collapse into one cue"
    assert rally_cues[0]["startMs"] == words[0]["start"]
    # Neither span's content was lost -- the next (different-image) span
    # still starts exactly where the merged cue ends (words[12] = "4월",
    # the first word of the next, un-merged segment).
    merged_end = rally_cues[0]["startMs"] + rally_cues[0]["durationMs"]
    assert merged_end == words[12]["start"]
    following = [cue for cue in cues if cue["startMs"] == merged_end]
    assert len(following) == 1
    assert following[0]["emoticonFileName"] != "30_랠리_상승파티.png"


def test_merge_does_not_bridge_two_different_images() -> None:
    # Guard against over-merging: adjacent cues with *different* images (the
    # overwhelmingly common case) must stay separate.
    words = _words(["지수가", "올랐습니다.", "다우는", "떨어졌습니다."])

    cues = plan_cues(words)

    assert len(cues) == 2
    assert cues[0]["emoticonFileName"] != cues[1]["emoticonFileName"]


def test_sparse_mode_is_not_merged_even_with_a_repeated_image() -> None:
    # Sparse-mode cues are not gapless by construction (cooldown/gap rules
    # can leave real silence between two same-image cues), so merging them
    # would silently stretch a cue across a gap that was never covered by
    # emoticon content. The merge must be scoped to fill_gaps only.
    # Two standalone "상승했습니다." sentences, far enough apart that they
    # are separate segments and clear sparse mode's cooldown/gap rules; each
    # matches the "상승" keyword directly, with nothing in between for the
    # 2-word lookahead to bleed across (unlike adjacent single-word spans).
    words = [
        {"text": "상승했습니다.", "start": 0, "end": 500},
        {"text": "상승했습니다.", "start": 6000, "end": 6500},
    ]

    cues = plan_cues(words, fill_gaps=False)

    assert len(cues) == 2
    assert cues[0]["emoticonFileName"] == cues[1]["emoticonFileName"] == "09_가즈아_상승.png"
    assert cues[1]["startMs"] > cues[0]["startMs"] + cues[0]["durationMs"]


def test_plan_cues_is_deterministic_across_runs() -> None:
    words = _words(
        [
            "지난", "금요일", "미국", "증시,", "강하게", "상승", "마감했습니다.",
            "나스닥은", "0.94%", "상승했습니다.",
            "로블록스,", "18%", "급락했습니다.",
            "오라클이", "미", "국방부", "AI", "네트워크", "프로젝트에", "합류했습니다.",
            "그래서", "말이죠.",
        ]
    )

    first = plan_cues(words)
    second = plan_cues(words)

    assert first == second
    assert [cue["emoticonFileName"] for cue in first] == [
        cue["emoticonFileName"] for cue in second
    ]


def test_sparse_mode_respects_max_cues_cap() -> None:
    # 60 one-word sentences, 3s apart -- every span clears the cooldown, so
    # only the hard cap can stop the planner.
    words = _words([f"문장{index}입니다." for index in range(60)], step=500, gap=2500)

    cues = plan_cues(words, fill_gaps=False)

    assert len(cues) == MAX_CUES_DEFAULT == 24


def test_sparse_mode_applies_cooldown_and_duration_rules() -> None:
    # Sentences 400ms apart: most are swallowed by the 2500ms cooldown.
    words = _words([f"문장{index}입니다." for index in range(20)], step=200, gap=200)

    cues = plan_cues(words, fill_gaps=False)

    assert 0 < len(cues) < 20
    for current, following in zip(cues, cues[1:]):
        assert following["startMs"] - current["startMs"] >= SPARSE_COOLDOWN_MS
    for cue in cues:
        assert SPARSE_MIN_CUE_DURATION_MS <= cue["durationMs"] <= SPARSE_CUE_DURATION_MS


def test_empty_word_list_produces_no_cues() -> None:
    assert plan_cues([]) == []
    assert plan_cues([], fill_gaps=False) == []
