"""호랑이 캐릭터 cue planning for `visual_mode=emoticon` (F-09).

See docs/architecture.md section 5.8. Cost is zero and the planner is
**deterministic**: the same `words.json` must always produce the exact same
cue list, so every selection step here is ordered explicitly (no set
iteration, no dict-order-dependent ties, no hashing).

Pipeline:

1. `caption_plan.build_caption_line_segments()` splits `words.json` into
   sentence-unit spans — the same boundaries Remotion `Captions.tsx` uses, so
   a visual never cuts in the middle of a caption line.
2. Each span's text is scanned against `EMOTICON_KEYWORDS` and the *most
   specific* match wins (longest keyword; ties broken by declaration order).
3. If the span itself has no match, the next `LOOKAHEAD_WORDS` words past the
   span boundary are appended and the scan is retried once — a keyword that
   straddles a sentence boundary still selects the image.
4. Still no match: `FALLBACK_CATEGORIES` (coarse 상승/하락/정책/... stems) is
   tried in declaration order, then `NEUTRAL_DEFAULT_POOL`, rotated by how many
   times the pool has already been used in this video (review.md #7: a single
   repeated neutral image was visually monotonous once the real bugs that used
   to overshadow it were fixed). The rotation only advances on a genuine
   no-match-at-all fallback -- a `FALLBACK_CATEGORIES` hit never touches it,
   even when a category's own image happens to also sit in the pool.

Keyword matching is substring-based against two variants of the span text:
the raw text and the same text with all whitespace removed. That is why the
dictionary can keep one spaced entry (`"실적 발표"`) and still match narration
that Scribe tokenized as `실적발표`.

Cue dicts match Remotion's `EmoticonCue` (remotion/src/types.ts) exactly:
`emoticonFileName` (bare PNG name resolved under `public/emoticon/`),
`startMs`, `durationMs`, `assetKind`, `visualMode`, `keyword`.
"""

from __future__ import annotations

from typing import Any

from md2short.caption_plan import DEFAULT_CHUNK_SIZE, build_caption_line_segments

__all__ = [
    "EMOTICON_KEYWORDS",
    "FALLBACK_CATEGORIES",
    "NEUTRAL_DEFAULT",
    "NEUTRAL_DEFAULT_POOL",
    "LOOKAHEAD_WORDS",
    "MAX_CUES_DEFAULT",
    "SPARSE_COOLDOWN_MS",
    "SPARSE_MIN_GAP_MS",
    "SPARSE_CUE_DURATION_MS",
    "SPARSE_MIN_CUE_DURATION_MS",
    "select_emoticon",
    "plan_cues",
]

#: Narration keyword -> candidate emoticon PNG file names, most preferred
#: first. Every file in `emoticon/emoticon/` (60 assets, architecture.md
#: section 5.8) is reachable through at least one keyword here; the entries are
#: declared in asset-number order so that a specificity tie is broken by the
#: lower asset number, deterministically.
EMOTICON_KEYWORDS: dict[str, list[str]] = {
    # 01 정독중_리포트
    "리포트": ["01_정독중_리포트.png"],
    "보고서": ["01_정독중_리포트.png"],
    "정독": ["01_정독중_리포트.png"],
    # 02 머리아파_혼란
    "혼란": ["02_머리아파_혼란.png"],
    "머리아파": ["02_머리아파_혼란.png"],
    "복잡": ["02_머리아파_혼란.png"],
    # 03 굿굿_칭찬
    "칭찬": ["03_굿굿_칭찬.png"],
    "호평": ["03_굿굿_칭찬.png"],
    "긍정적": ["03_굿굿_칭찬.png"],
    # 04 나도울고싶어_하락
    "하락": ["04_나도울고싶어_하락.png"],
    "눈물": ["04_나도울고싶어_하락.png"],
    "속상": ["04_나도울고싶어_하락.png"],
    # 05 분석중_차트
    "차트": ["05_분석중_차트.png"],
    "분석": ["05_분석중_차트.png"],
    "지표": ["05_분석중_차트.png"],
    # 06 움직임없음_관망
    "관망": ["06_움직임없음_관망.png"],
    "보합": ["06_움직임없음_관망.png"],
    "횡보": ["06_움직임없음_관망.png"],
    # 07 승인완료_도장
    "승인 완료": ["07_승인완료_도장.png"],
    "의결": ["07_승인완료_도장.png"],
    "통과": ["07_승인완료_도장.png"],
    # 08 전량매도_급락
    "전량 매도": ["08_전량매도_급락.png"],
    "투매": ["08_전량매도_급락.png"],
    # 09 가즈아_상승
    "가즈아": ["09_가즈아_상승.png"],
    "상승": ["09_가즈아_상승.png"],
    "오름": ["09_가즈아_상승.png"],
    # 10 살까말까_고민
    "고민": ["10_살까말까_고민.png"],
    "살까말까": ["10_살까말까_고민.png"],
    "망설": ["10_살까말까_고민.png"],
    # 11 물타기가즈아_하락매수
    "물타기": ["11_물타기가즈아_하락매수.png"],
    "추가 매수": ["11_물타기가즈아_하락매수.png"],
    "저가 매수": ["11_물타기가즈아_하락매수.png"],
    # 12 리포트작성_공부
    "리포트 작성": ["12_리포트작성_공부.png"],
    "공부": ["12_리포트작성_공부.png"],
    "학습": ["12_리포트작성_공부.png"],
    # 13 이게뭔말이야_당황
    "당황": ["13_이게뭔말이야_당황.png"],
    "황당": ["13_이게뭔말이야_당황.png"],
    "무슨 말": ["13_이게뭔말이야_당황.png"],
    # 14 상폐_장례식
    "상장 폐지": ["14_상폐_장례식.png"],
    "상폐": ["14_상폐_장례식.png"],
    "파산": ["14_상폐_장례식.png"],
    # 15 존버중_버티기
    "존버": ["15_존버중_버티기.png"],
    "버티": ["15_존버중_버티기.png"],
    "장기 보유": ["15_존버중_버티기.png"],
    # 16 손절_컷
    "손절": ["16_손절_컷.png"],
    "손실": ["16_손절_컷.png"],
    # 17 익절각_매도
    "익절": ["17_익절각_매도.png"],
    "차익 실현": ["17_익절각_매도.png"],
    "매도": ["17_익절각_매도.png"],
    # 18 테마주탑승_로켓
    "테마주": ["18_테마주탑승_로켓.png"],
    "급등": ["18_테마주탑승_로켓.png"],
    "로켓": ["18_테마주탑승_로켓.png"],
    # 19 FOMO_추격매수
    "FOMO": ["19_FOMO_추격매수.png"],
    "포모": ["19_FOMO_추격매수.png"],
    "추격 매수": ["19_FOMO_추격매수.png"],
    # 20 깜짝성장_실적호조
    "깜짝 성장": ["20_깜짝성장_실적호조.png"],
    "호실적": ["20_깜짝성장_실적호조.png"],
    "실적 호조": ["20_깜짝성장_실적호조.png"],
    # 21 소비심리추락_불안
    "소비 심리": ["21_소비심리추락_불안.png"],
    "불안": ["21_소비심리추락_불안.png"],
    "위축": ["21_소비심리추락_불안.png"],
    # 22 유가비상_원유급등
    "유가 비상": ["22_유가비상_원유급등.png"],
    "원유 급등": ["22_유가비상_원유급등.png"],
    # 23 총파업_초읽기
    "총파업": ["23_총파업_초읽기.png"],
    "파업": ["23_총파업_초읽기.png"],
    # 24 연준혼돈_금리불확실
    "연준": ["24_연준혼돈_금리불확실.png"],
    "금리 불확실": ["24_연준혼돈_금리불확실.png"],
    "불확실": ["24_연준혼돈_금리불확실.png"],
    # 25 애플깜짝성장_실적호조
    "애플": ["25_애플깜짝성장_실적호조.png"],
    "아이폰": ["25_애플깜짝성장_실적호조.png"],
    # 26 반도체급등_로켓상승
    "반도체 급등": ["26_반도체급등_로켓상승.png"],
    "반도체": ["26_반도체급등_로켓상승.png"],
    "칩": ["26_반도체급등_로켓상승.png"],
    # 27 AI계약_승인
    "AI 계약": ["27_AI계약_승인.png"],
    "계약": ["27_AI계약_승인.png"],
    "수주": ["27_AI계약_승인.png"],
    "합류": ["27_AI계약_승인.png"],
    # 28 폭락_패닉하락
    "폭락": ["28_폭락_패닉하락.png"],
    "패닉": ["28_폭락_패닉하락.png"],
    "급락": ["28_폭락_패닉하락.png", "08_전량매도_급락.png"],
    # 29 유가하락_오일약세
    "유가 하락": ["29_유가하락_오일약세.png"],
    "오일 약세": ["29_유가하락_오일약세.png"],
    # 30 랠리_상승파티
    "랠리": ["30_랠리_상승파티.png"],
    "상승 파티": ["30_랠리_상승파티.png"],
    "최장": ["30_랠리_상승파티.png"],
    # A generic "N주 연속 상승" belongs to the rally tiger, not to the
    # semiconductor-specific asset 49, which keeps "강세 마감"/"연속".
    "연속 상승": ["30_랠리_상승파티.png"],
    # 31 매파_금리압박
    "매파": ["31_매파_금리압박.png"],
    "금리 압박": ["31_매파_금리압박.png"],
    # 32 매파동결_국채금리급등
    "국채 금리": ["32_매파동결_국채금리급등.png"],
    "국채": ["32_매파동결_국채금리급등.png"],
    "매파 동결": ["32_매파동결_국채금리급등.png"],
    # 33 유가급등_이란봉쇄
    "이란": ["33_유가급등_이란봉쇄.png"],
    "유가 급등": ["33_유가급등_이란봉쇄.png"],
    # 34 내부분열_의견갈등
    "내부 분열": ["34_내부분열_의견갈등.png"],
    "의견 갈등": ["34_내부분열_의견갈등.png"],
    "이견": ["34_내부분열_의견갈등.png"],
    # 35 돈태운다_현금소각
    "현금 소각": ["35_돈태운다_현금소각.png"],
    "적자": ["35_돈태운다_현금소각.png"],
    "비용 부담": ["35_돈태운다_현금소각.png"],
    # 36 클라우드상승_성장
    "클라우드": ["36_클라우드상승_성장.png"],
    "성장": ["36_클라우드상승_성장.png"],
    # 37 검색강세_트렌드상승
    "검색": ["37_검색강세_트렌드상승.png"],
    "트렌드": ["37_검색강세_트렌드상승.png"],
    "광고": ["37_검색강세_트렌드상승.png"],
    # 38 혼조세_증시갈림길
    "혼조": ["38_혼조세_증시갈림길.png"],
    "갈림길": ["38_혼조세_증시갈림길.png"],
    "엇갈": ["38_혼조세_증시갈림길.png"],
    # 39 목표미달_AI수익성우려
    "목표 미달": ["39_목표미달_AI수익성우려.png"],
    "수익성 우려": ["39_목표미달_AI수익성우려.png"],
    "기대 이하": ["39_목표미달_AI수익성우려.png"],
    # 40 실적대기_발표긴장
    "실적 발표": ["40_실적대기_발표긴장.png"],
    "실적 대기": ["40_실적대기_발표긴장.png"],
    "발표 예정": ["40_실적대기_발표긴장.png"],
    # 41 반도체매도_주가하락
    "반도체 매도": ["41_반도체매도_주가하락.png"],
    "반도체 약세": ["41_반도체매도_주가하락.png"],
    "주가 하락": ["41_반도체매도_주가하락.png"],
    # 42 유가111달러_호르무즈위기
    "호르무즈 위기": ["42_유가111달러_호르무즈위기.png"],
    "유가 111": ["42_유가111달러_호르무즈위기.png"],
    # 43 UAE탈퇴_OPEC불확실
    "OPEC": ["43_UAE탈퇴_OPEC불확실.png"],
    "탈퇴": ["43_UAE탈퇴_OPEC불확실.png"],
    "감산": ["43_UAE탈퇴_OPEC불확실.png"],
    # 44 평화협상_중동긴장완화
    "평화 협상": ["44_평화협상_중동긴장완화.png"],
    "긴장 완화": ["44_평화협상_중동긴장완화.png"],
    "합의": ["44_평화협상_중동긴장완화.png"],
    # 45 금리동결_FED기준금리
    "금리 동결": ["45_금리동결_FED기준금리.png"],
    "기준금리": ["45_금리동결_FED기준금리.png"],
    "FED": ["45_금리동결_FED기준금리.png"],
    "금리": ["45_금리동결_FED기준금리.png"],
    # 46 사상최고치_지수신고가
    "사상 최고치": ["46_사상최고치_지수신고가.png"],
    "최고치": ["46_사상최고치_지수신고가.png"],
    "신고가": ["46_사상최고치_지수신고가.png"],
    # 47 협상교착_호르무즈긴장
    "협상 교착": ["47_협상교착_호르무즈긴장.png"],
    "교착": ["47_협상교착_호르무즈긴장.png"],
    "호르무즈": ["47_협상교착_호르무즈긴장.png"],
    # 48 특사취소_전화협상
    "특사": ["48_특사취소_전화협상.png"],
    "전화 협상": ["48_특사취소_전화협상.png"],
    "통화": ["48_특사취소_전화협상.png"],
    # 49 반도체18연속상승_강세마감
    "강세 마감": ["49_반도체18연속상승_강세마감.png"],
    "연속": ["49_반도체18연속상승_강세마감.png"],
    # 50 포테크놀로지폭락_47퍼센트하락
    "포테크놀로지": ["50_포테크놀로지폭락_47퍼센트하락.png"],
    "반토막": ["50_포테크놀로지폭락_47퍼센트하락.png"],
    "47%": ["50_포테크놀로지폭락_47퍼센트하락.png"],
    # 51 퀄컴오픈AI협력설_AI협업
    "퀄컴": ["51_퀄컴오픈AI협력설_AI협업.png"],
    "협력설": ["51_퀄컴오픈AI협력설_AI협업.png"],
    "AI 협업": ["51_퀄컴오픈AI협력설_AI협업.png"],
    # 52 글로벌금리결정대기_중앙은행회의
    "중앙은행": ["52_글로벌금리결정대기_중앙은행회의.png"],
    "금리 결정": ["52_글로벌금리결정대기_중앙은행회의.png"],
    "통화정책": ["52_글로벌금리결정대기_중앙은행회의.png"],
    # 53 60일만료_전쟁권한법시한
    "전쟁 권한": ["53_60일만료_전쟁권한법시한.png"],
    "시한": ["53_60일만료_전쟁권한법시한.png"],
    "만료": ["53_60일만료_전쟁권한법시한.png"],
    # 54 다크이글배치_극초음속미사일
    "극초음속": ["54_다크이글배치_극초음속미사일.png"],
    "미사일": ["54_다크이글배치_극초음속미사일.png"],
    "다크 이글": ["54_다크이글배치_극초음속미사일.png"],
    # 55 6500톤수령_군수물자지원
    "군수 물자": ["55_6500톤수령_군수물자지원.png"],
    "물자 지원": ["55_6500톤수령_군수물자지원.png"],
    "탄약": ["55_6500톤수령_군수물자지원.png"],
    # 56 아이언빔배치_UAE군사협력
    "아이언빔": ["56_아이언빔배치_UAE군사협력.png"],
    "군사 협력": ["56_아이언빔배치_UAE군사협력.png"],
    "방공": ["56_아이언빔배치_UAE군사협력.png"],
    # 57 여행금지_UAE긴급조치
    "여행 금지": ["57_여행금지_UAE긴급조치.png"],
    "긴급 조치": ["57_여행금지_UAE긴급조치.png"],
    "UAE": ["57_여행금지_UAE긴급조치.png"],
    # 58 봉쇄지속_유가급등
    "봉쇄 지속": ["58_봉쇄지속_유가급등.png"],
    "봉쇄": ["58_봉쇄지속_유가급등.png"],
    # 59 연쇄타격_물가환율압박
    "연쇄 타격": ["59_연쇄타격_물가환율압박.png"],
    "환율": ["59_연쇄타격_물가환율압박.png"],
    "물가": ["59_연쇄타격_물가환율압박.png"],
    # 60 카운트제외_휴전기간논리
    "휴전": ["60_카운트제외_휴전기간논리.png"],
    "카운트 제외": ["60_카운트제외_휴전기간논리.png"],
    "유예": ["60_카운트제외_휴전기간논리.png"],
}

#: Coarse category fallback, tried in declaration order when no keyword
#: matched. The stems here are deliberately *morphological fragments*
#: (`올랐`, `떨어`) that the keyword dictionary does not carry, so a sentence
#: that says "올랐습니다" instead of "상승" still gets a sensible image.
#:
#: A category's stems must stay as unspecific as its image. `policy` is the
#: exception that proves it: its image is the rate-freeze-specific
#: `45_금리동결_FED기준금리.png`, so its stems are rate-decision terms only.
#: They used to include bare `정책`/`당국`/`정부`, which put a FED rate-freeze
#: tiger over "이는 정부의 AI 시장이 본격 매출 사이클에 진입한 신호입니다" (job
#: sf_20261012_e6cd2a) — a sentence with no monetary-policy content at all. A
#: generic government/policy mention now falls through to `data`/`outlook` or
#: to `NEUTRAL_DEFAULT`, which is merely neutral instead of confidently wrong.
#: `금리`/`연준` are already `EMOTICON_KEYWORDS` entries and so are reached
#: before this list; they are kept here so the category stays correct on its
#: own terms, while `동결`/`FOMC` are the fragments that make it reachable.
FALLBACK_CATEGORIES: list[tuple[str, tuple[str, ...], str]] = [
    ("rise", ("올랐", "올라", "뛰었", "강세", "반등", "플러스"), "09_가즈아_상승.png"),
    ("fall", ("떨어", "빠졌", "약세", "내렸", "마이너스"), "04_나도울고싶어_하락.png"),
    ("policy", ("금리", "기준금리", "연준", "FOMC", "동결"), "45_금리동결_FED기준금리.png"),
    ("data", ("종가", "지수", "퍼센트", "포인트", "달러", "%"), "05_분석중_차트.png"),
    ("outlook", ("전망", "예상", "기대", "주목", "관심"), "01_정독중_리포트.png"),
]

#: Last resort when neither a keyword nor a category matched. Four genuinely
#: neutral/no-clear-signal images (watching, pondering, studying, mixed
#: signals), none of them a `FALLBACK_CATEGORIES` image, so membership in this
#: tuple unambiguously means "hit the final fallback", not "hit a category".
#: Rotated by `plan_cues` so a video with several unmatched connective
#: sentences doesn't show the exact same tiger four times in under a minute
#: (review.md #7, job sf_20261012_fb1911).
NEUTRAL_DEFAULT_POOL: tuple[str, ...] = (
    "06_움직임없음_관망.png",
    "10_살까말까_고민.png",
    "12_리포트작성_공부.png",
    "38_혼조세_증시갈림길.png",
)

#: Kept for backward compatibility (and as the index-0 rotation member) —
#: callers that only need "a" neutral image, not the rotating pool, still
#: have a single stable constant to import.
NEUTRAL_DEFAULT = NEUTRAL_DEFAULT_POOL[0]

#: Words past the span boundary that are appended before the second keyword
#: scan (architecture.md section 5.8 "boundary-crossing keyword lookahead 2 words").
LOOKAHEAD_WORDS = 2

# --- fill_gaps=False (sparse mode) rules, architecture.md section 5.8 -------
#: Hard cap on cue count in sparse mode.
MAX_CUES_DEFAULT = 24
#: Minimum distance between two consecutive cue *starts*.
SPARSE_COOLDOWN_MS = 2500
#: Minimum breathing gap between the previous cue's end and the next cue's start.
SPARSE_MIN_GAP_MS = 400
#: Nominal sparse cue lifetime, clamped down to the span it belongs to.
SPARSE_CUE_DURATION_MS = 1800
#: Floor for that clamp, so a very short span still produces a visible cue.
SPARSE_MIN_CUE_DURATION_MS = 900

#: Keyword scan order: longest keyword first ("most specific wins"), ties
#: broken by declaration order in `EMOTICON_KEYWORDS`. Precomputed once so the
#: ordering can never depend on runtime iteration order.
_KEYWORD_SCAN_ORDER: tuple[str, ...] = tuple(
    keyword
    for _, _, keyword in sorted(
        (
            (-len(keyword), index, keyword)
            for index, keyword in enumerate(EMOTICON_KEYWORDS)
        )
    )
)


def _match_keyword(text: str) -> tuple[str, str] | None:
    """Return `(keyword, file_name)` for the most specific match in `text`.

    Matching is substring-based against both `text` and its whitespace-free
    form (and the keyword's whitespace-free form), so `"실적 발표"` matches
    both `실적 발표` and `실적발표`. Scanning follows `_KEYWORD_SCAN_ORDER`,
    so the longest keyword wins and equal-length keywords are resolved by
    declaration order — never by dict/set iteration chance.
    """
    compact = "".join(text.split())
    for keyword in _KEYWORD_SCAN_ORDER:
        if keyword in text or "".join(keyword.split()) in compact:
            return keyword, EMOTICON_KEYWORDS[keyword][0]
    return None


def select_emoticon(
    text: str, lookahead_text: str = "", neutral_rotation_index: int = 0
) -> tuple[str, str | None]:
    """Pick the emoticon for one visual span: keyword -> category -> neutral.

    Returns `(file_name, keyword)`, where `keyword` is the matched keyword or
    `None` when a fallback was used. `lookahead_text` is the text of the next
    `LOOKAHEAD_WORDS` words after the span; it is only consulted when the span
    text itself has no keyword match. `neutral_rotation_index` only matters
    when execution reaches `NEUTRAL_DEFAULT_POOL` at the very end: it selects
    `NEUTRAL_DEFAULT_POOL[neutral_rotation_index % len(NEUTRAL_DEFAULT_POOL)]`.
    Defaults to 0 (`NEUTRAL_DEFAULT_POOL[0] == NEUTRAL_DEFAULT`) so a caller
    that doesn't care about rotation gets the original single-image behavior.
    """
    match = _match_keyword(text)
    if match is None and lookahead_text:
        match = _match_keyword(f"{text} {lookahead_text}")
    if match is not None:
        return match[1], match[0]

    compact = "".join(text.split())
    for _category, stems, file_name in FALLBACK_CATEGORIES:
        if any(stem in text or stem in compact for stem in stems):
            return file_name, None

    pool_index = neutral_rotation_index % len(NEUTRAL_DEFAULT_POOL)
    return NEUTRAL_DEFAULT_POOL[pool_index], None


def _cue(file_name: str, keyword: str | None, start_ms: int, duration_ms: int) -> dict[str, Any]:
    """Build one `EmoticonCue` dict (remotion/src/types.ts)."""
    return {
        "emoticonFileName": file_name,
        "startMs": int(start_ms),
        "durationMs": int(duration_ms),
        "assetKind": "emoticon",
        "visualMode": "emoticon",
        "keyword": keyword,
    }


def plan_cues(
    words: list[dict[str, Any]],
    fill_gaps: bool = True,
    max_cues: int = MAX_CUES_DEFAULT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Plan `EmoticonCue` dicts from a `words.json` word list.

    `fill_gaps=True` (default, architecture.md section 5.8): one cue per
    sentence-unit span from `caption_plan.build_caption_line_segments()`,
    each cue lasting exactly its span, so the visual timeline is gapless from
    the first word to the end of the last span. `max_cues` is **not** applied
    in this mode — a long video keeps generating cues to the end of
    `words.json`.

    `fill_gaps=False` (sparse): spans are walked in order and a cue is placed
    only when it clears `SPARSE_COOLDOWN_MS` since the previous cue's start and
    `SPARSE_MIN_GAP_MS` since the previous cue's end; its duration is the span
    length clamped into
    `[SPARSE_MIN_CUE_DURATION_MS, SPARSE_CUE_DURATION_MS]`; and at most
    `max_cues` cues are produced.

    Deterministic in both modes: same `words` in, byte-identical list out.
    """
    segments = build_caption_line_segments(words, chunk_size)
    if not segments:
        return []

    cues: list[dict[str, Any]] = []
    word_offset = 0
    last_start_ms: int | None = None
    last_end_ms: int | None = None
    neutral_fallback_count = 0

    for segment in segments:
        span_words = segment["words"]
        word_offset += len(span_words)
        lookahead_text = " ".join(
            str(word.get("text", ""))
            for word in words[word_offset : word_offset + LOOKAHEAD_WORDS]
        )
        file_name, keyword = select_emoticon(
            segment["text"], lookahead_text, neutral_fallback_count
        )
        # Only a genuine no-match-at-all fallback advances the rotation -- a
        # FALLBACK_CATEGORIES hit (also keyword=None) never lands in the pool.
        if keyword is None and file_name in NEUTRAL_DEFAULT_POOL:
            neutral_fallback_count += 1

        start_ms = segment["startMs"]
        span_duration_ms = max(1, segment["endMs"] - start_ms)

        if fill_gaps:
            cues.append(_cue(file_name, keyword, start_ms, span_duration_ms))
            continue

        if len(cues) >= max_cues:
            break
        if last_start_ms is not None and start_ms - last_start_ms < SPARSE_COOLDOWN_MS:
            continue
        if last_end_ms is not None and start_ms - last_end_ms < SPARSE_MIN_GAP_MS:
            continue

        duration_ms = max(
            SPARSE_MIN_CUE_DURATION_MS, min(span_duration_ms, SPARSE_CUE_DURATION_MS)
        )
        cues.append(_cue(file_name, keyword, start_ms, duration_ms))
        last_start_ms = start_ms
        last_end_ms = start_ms + duration_ms

    if fill_gaps:
        cues = _merge_adjacent_same_image_cues(cues)
    return cues


def _merge_adjacent_same_image_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge back-to-back `fill_gaps` cues that landed on the same image.

    A repeated keyword or phrase across two consecutive spans (review.md
    #10: "5주 연속 상승" matched in both the hook and the start of body) is
    a content/script-quality question, not an emoticon-selection one --
    picking a *different* image just for variety would make the picture
    less correct for one of the two spans, undoing #1/#4's whole point. The
    picture itself is right; two separate `VisualCue` mounts of it back to
    back is not -- each restarts its own Ken Burns cycle (`VisualLayer.tsx`),
    so the identical image visibly resets partway through what a viewer
    reads as one continuous shot. Merging them into a single, longer cue
    keeps the (correct) image and turns two abrupt resets into one smooth
    zoom across the full combined duration.

    Only `fill_gaps` cues are merged: that mode's spans are gapless by
    construction (`segment["endMs"]` is literally the next segment's
    `startMs`), so adjacency here never bridges a real silence. Sparse mode
    cues are not necessarily contiguous (cooldown/gap enforcement can leave
    a real gap even between same-image cues), so it is left untouched.
    """
    if not cues:
        return cues

    merged: list[dict[str, Any]] = [dict(cues[0])]
    for cue in cues[1:]:
        last = merged[-1]
        contiguous = last["startMs"] + last["durationMs"] == cue["startMs"]
        if contiguous and last["emoticonFileName"] == cue["emoticonFileName"]:
            last["durationMs"] += cue["durationMs"]
            continue
        merged.append(dict(cue))
    return merged
