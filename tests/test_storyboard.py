"""Unit tests for md2short.storyboard and md2short.script input handling.

Covers architecture.md §5.1 / PRD §5.2: `### N컷` detection, per-cut field
separation (자막 vs 화면 연출 vs 이미지 프롬프트) and narration joining.
"""

from __future__ import annotations

from pathlib import Path

from md2short.script import detect_input_kind, strip_markdown, write_source_md
from md2short.storyboard import is_storyboard_markdown, join_captions, parse_storyboard

STORYBOARD = """# 금요일 증시 브리핑

### 1컷
- 자막: 지난 금요일 미국 증시, 강하게 상승 마감했습니다.
- 화면 연출: 상승 화살표와 함께 호랑이가 환호하는 모습
- 이미지 프롬프트: photorealistic trading floor with green tickers

### 2컷
- 자막: S&P 500 종가 7,230 기록했습니다.
- 화면 연출: 숫자 카드가 아래에서 위로 등장
- 이미지 프롬프트: a large glowing number board

### 3컷
- 자막: 로블록스, 18% 급락했습니다.
- 화면 연출: 하락 화살표, 호랑이가 놀람
- 이미지 프롬프트: a red falling chart on a dark screen
"""

PLAIN_MARKDOWN = """## 금요일 증시

- 나스닥은 0.94% 상승했습니다.
- 애플, 호실적으로 3.2% 상승했습니다.
"""


def test_detects_storyboard_format():
    assert is_storyboard_markdown(STORYBOARD) is True
    assert detect_input_kind(STORYBOARD) == "storyboard"


def test_plain_markdown_is_not_storyboard():
    assert is_storyboard_markdown(PLAIN_MARKDOWN) is False
    assert detect_input_kind(PLAIN_MARKDOWN) == "markdown"
    assert detect_input_kind("") == "markdown"
    # "컷" used as a prose word, not as a cut header, must not trigger detection.
    assert is_storyboard_markdown("이 영상은 세 컷으로 구성됩니다.") is False


def test_parse_storyboard_separates_fields():
    cuts = parse_storyboard(STORYBOARD)

    assert [cut["cut_number"] for cut in cuts] == [1, 2, 3]
    assert cuts[0]["caption"] == "지난 금요일 미국 증시, 강하게 상승 마감했습니다."
    assert cuts[0]["visual_direction"] == "상승 화살표와 함께 호랑이가 환호하는 모습"
    assert cuts[0]["image_prompt"] == "photorealistic trading floor with green tickers"
    assert cuts[1]["caption"] == "S&P 500 종가 7,230 기록했습니다."
    assert cuts[2]["caption"] == "로블록스, 18% 급락했습니다."
    assert cuts[2]["visual_direction"] == "하락 화살표, 호랑이가 놀람"
    assert cuts[2]["image_prompt"] == "a red falling chart on a dark screen"


def test_parse_storyboard_tolerates_bold_labels_and_missing_fields():
    markdown = """### 1컷
**자막**: 첫 번째 컷입니다.
**화면 연출**: 호랑이 등장

### 2컷
자막: 두 번째 컷입니다.
"""
    cuts = parse_storyboard(markdown)

    assert cuts[0]["caption"] == "첫 번째 컷입니다."
    assert cuts[0]["visual_direction"] == "호랑이 등장"
    assert cuts[0]["image_prompt"] == ""
    assert cuts[1]["caption"] == "두 번째 컷입니다."
    assert cuts[1]["visual_direction"] == ""
    assert cuts[1]["image_prompt"] == ""


def test_parse_storyboard_keeps_multiline_image_prompt():
    markdown = """### 1컷
- 자막: 한 문장입니다.
- 이미지 프롬프트: a wide shot of a city skyline,
  golden hour lighting, photorealistic
"""
    cuts = parse_storyboard(markdown)

    assert cuts[0]["caption"] == "한 문장입니다."
    assert cuts[0]["image_prompt"] == (
        "a wide shot of a city skyline,\ngolden hour lighting, photorealistic"
    )


def test_parse_storyboard_returns_empty_for_plain_markdown():
    assert parse_storyboard(PLAIN_MARKDOWN) == []


def test_join_captions_preserves_cut_order():
    cuts = parse_storyboard(STORYBOARD)
    narration = join_captions(cuts)

    assert narration == (
        "지난 금요일 미국 증시, 강하게 상승 마감했습니다.\n"
        "S&P 500 종가 7,230 기록했습니다.\n"
        "로블록스, 18% 급락했습니다."
    )


def test_join_captions_excludes_visual_direction_and_image_prompt():
    cuts = parse_storyboard(STORYBOARD)
    narration = join_captions(cuts)

    for cut in cuts:
        assert cut["visual_direction"] not in narration
        assert cut["image_prompt"] not in narration
    assert "호랑이" not in narration
    assert "photorealistic" not in narration
    # ...but the fields are still available for US-008/US-009.
    assert all(cut["visual_direction"] and cut["image_prompt"] for cut in cuts)


def test_join_captions_skips_cuts_without_caption():
    cuts = [
        {"cut_number": 1, "caption": "첫 컷", "visual_direction": "x", "image_prompt": "y"},
        {"cut_number": 2, "caption": "", "visual_direction": "x", "image_prompt": "y"},
        {"cut_number": 3, "caption": "셋째 컷", "visual_direction": "", "image_prompt": ""},
    ]
    assert join_captions(cuts) == "첫 컷\n셋째 컷"


def test_write_source_md_archives_raw_input(tmp_path: Path):
    job_dir = tmp_path / "sf_20260502_abc123"
    written = write_source_md(job_dir, STORYBOARD)

    assert written == job_dir / "inputs" / "source.md"
    assert written.read_text(encoding="utf-8") == STORYBOARD


def test_strip_markdown_removes_syntax_and_keeps_facts():
    markdown = """# 증시 브리핑

## 요약

- **S&P 500** 종가 `7,230` 기록했습니다.
- 나스닥은 *0.94%* 상승했습니다.
- 자세한 내용은 [리포트](https://example.com/report)를 참고하세요.

> 핵심은 '5주 연속 상승'입니다.

---

로블록스, 18% 급락했습니다.
"""
    text = strip_markdown(markdown)

    assert "#" not in text
    assert "**" not in text
    assert "`" not in text
    assert "https://example.com/report" not in text
    assert ">" not in text
    assert "S&P 500 종가 7,230 기록했습니다." in text
    assert "나스닥은 0.94% 상승했습니다." in text
    assert "자세한 내용은 리포트를 참고하세요." in text
    assert "핵심은 '5주 연속 상승'입니다." in text
    assert "로블록스, 18% 급락했습니다." in text
    assert text.startswith("증시 브리핑")
