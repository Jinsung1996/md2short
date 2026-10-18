"""Unit tests for md2short.photo_engine (architecture.md §5.7, §1.7).

No real network calls: every OpenAI interaction (Responses prompt batching and
`gpt-image-2` image generation) goes through an `httpx.MockTransport` handler
injected as the `client` argument, and the handler call count is asserted where
"no second call" (prompt cache) is part of the contract.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from md2short.photo_engine import (
    OPENAI_IMAGES_URL,
    OPENAI_RESPONSES_URL,
    ModerationBlockedError,
    PhotoEngineError,
    _apply_keyword_focus,
    _prompt_cache_key,
    _sanitise_prompt,
    _SAFE_FALLBACK_PROMPT,
    extract_visual_subject,
    generate_image,
    generate_image_cached,
    generate_image_prompts,
    generate_image_with_retry_chain,
    plan_photo_cues,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-pixels"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")

WORDS = [
    {"text": "트럼프가", "start": 0, "end": 400},
    {"text": "합의를", "start": 400, "end": 800},
    {"text": "했습니다.", "start": 800, "end": 1200},
    {"text": "나스닥은", "start": 1600, "end": 2000},
    {"text": "급등했습니다.", "start": 2000, "end": 2600},
]


def _image_success_response() -> httpx.Response:
    return httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})


def _moderation_response() -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {
                "code": "moderation_blocked",
                "type": "image_generation_user_error",
                "message": "Your request was rejected by our safety system.",
            }
        },
    )


def _prompts_response(items: list[dict[str, object]]) -> httpx.Response:
    """A Responses API body carrying `items` as Structured Outputs JSON."""
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "model": "gpt-5-mini",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"prompts": items}, ensure_ascii=False),
                        }
                    ],
                }
            ],
        },
    )


# --- extract_visual_subject (§5.7 step 2) ----------------------------------


def test_extract_visual_subject_finds_person_and_keyword():
    hint = extract_visual_subject("트럼프가 합의를 했습니다.")
    assert hint["primary_visual_subject"] == "Donald Trump"
    assert "diplomatic agreement table" in hint["visual_keywords"]


def test_extract_visual_subject_falls_back_to_editorial_scene():
    hint = extract_visual_subject("오늘은 특별한 소식이 없었습니다.")
    assert hint["primary_visual_subject"] == "a financial news editorial scene"
    assert hint["visual_keywords"]


# --- _apply_keyword_focus (§5.7 step 3) ------------------------------------


def test_apply_keyword_focus_prefixes_when_subject_absent():
    focused = _apply_keyword_focus("Editorial photo of a podium", "Donald Trump")
    assert focused == "Primary visual subject: Donald Trump. Editorial photo of a podium"


def test_apply_keyword_focus_is_noop_when_subject_present_case_insensitive():
    prompt = "Editorial photo of donald trump at a podium"
    assert _apply_keyword_focus(prompt, "Donald Trump") == prompt


def test_apply_keyword_focus_is_noop_without_subject():
    assert _apply_keyword_focus("Editorial photo", "") == "Editorial photo"


# --- _sanitise_prompt (§5.7 retry stage 2) ---------------------------------


def test_sanitise_prompt_replaces_person_brand_and_conflict_triggers():
    sanitised = _sanitise_prompt(
        "Donald Trump beside a Tesla during the war, holding a missile"
    )
    assert "Trump" not in sanitised
    assert "Tesla" not in sanitised
    assert "war" not in sanitised.lower()
    assert "a public figure in a suit" in sanitised
    assert "a generic electric vehicle" in sanitised
    assert "a tense negotiation scene" in sanitised


def test_sanitise_prompt_leaves_neutral_prompt_unchanged():
    neutral = "Editorial photograph of a trading floor, soft lighting, no text"
    assert _sanitise_prompt(neutral) == neutral


# --- generate_image (§5.7 step 4) ------------------------------------------


def test_generate_image_decodes_base64_payload():
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == OPENAI_IMAGES_URL
        calls.append(json.loads(request.content))
        return _image_success_response()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        data = generate_image("Editorial photo", "sk-test", client=client)

    assert data == PNG_BYTES
    assert calls[0]["model"] == "gpt-image-2"
    assert calls[0]["size"] == "1024x1536"
    assert calls[0]["quality"] == "high"
    assert calls[0]["prompt"] == "Editorial photo"


def test_generate_image_raises_moderation_blocked():
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: _moderation_response())
    ) as client:
        with pytest.raises(ModerationBlockedError):
            generate_image("Blocked photo", "sk-test", client=client)


def test_generate_image_raises_photo_engine_error_on_other_http_error():
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": {"code": "invalid_size"}})
        )
    ) as client:
        with pytest.raises(PhotoEngineError):
            generate_image("Bad request", "sk-test", client=client)


def test_generate_image_requires_api_key():
    with pytest.raises(PhotoEngineError):
        generate_image("Editorial photo", "")


# --- generate_image_with_retry_chain (§5.7 retry chain) --------------------


def test_retry_chain_second_stage_uses_sanitised_prompt():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["prompt"]
        seen.append(prompt)
        if len(seen) == 1:
            return _moderation_response()
        return _image_success_response()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        data = generate_image_with_retry_chain(
            "Editorial photo of Donald Trump signing a deal",
            "Donald Trump",
            "sk-test",
            client=client,
        )

    assert data == PNG_BYTES
    assert len(seen) == 2
    assert "Donald Trump" in seen[0]
    assert seen[1] == _sanitise_prompt(seen[0])
    assert "Trump" not in seen[1]


def test_retry_chain_third_stage_uses_safe_fallback_prompt():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["prompt"]
        seen.append(prompt)
        if len(seen) < 3:
            return _moderation_response()
        return _image_success_response()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        data = generate_image_with_retry_chain(
            "Editorial photo of Donald Trump at war",
            "Donald Trump",
            "sk-test",
            client=client,
        )

    assert data == PNG_BYTES
    assert seen[2] == _SAFE_FALLBACK_PROMPT


def test_retry_chain_all_blocked_raises_photo_engine_error_chaining_cause():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["prompt"])
        return _moderation_response()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PhotoEngineError) as excinfo:
            generate_image_with_retry_chain(
                "Editorial photo of Donald Trump at war",
                "Donald Trump",
                "sk-test",
                client=client,
            )

    assert len(seen) == 3
    assert isinstance(excinfo.value.__cause__, ModerationBlockedError)
    assert "safety system" in str(excinfo.value.__cause__)


def test_retry_chain_does_not_retry_non_moderation_failures():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["prompt"])
        return httpx.Response(400, json={"error": {"code": "invalid_size"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PhotoEngineError):
            generate_image_with_retry_chain("Photo", "Subject", "sk-test", client=client)

    assert len(calls) == 1


# --- caching (§5.7 step 5) -------------------------------------------------


def test_prompt_cache_key_is_sha1_of_prompt():
    import hashlib

    assert _prompt_cache_key("abc") == hashlib.sha1(b"abc").hexdigest()


def test_generate_image_cached_reuses_file_without_second_api_call(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["prompt"])
        return _image_success_response()

    prompt = "Editorial photo of a trading floor"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = generate_image_cached(
            prompt, "sk-test", tmp_path, segment_index=0, client=client
        )
        second = generate_image_cached(
            prompt, "sk-test", tmp_path, segment_index=1, client=client
        )

    assert len(calls) == 1
    assert first == second
    assert first.name == f"seg_00_{_prompt_cache_key(prompt)}.png"
    assert first.read_bytes() == PNG_BYTES


# --- plan_photo_cues (§5.7 end to end) ------------------------------------


def _plan_handler(image_responses: list[httpx.Response] | None = None):
    """MockTransport handler: one Responses call, then image calls in order."""
    state: dict[str, object] = {"responses_calls": 0, "image_prompts": []}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == OPENAI_RESPONSES_URL:
            state["responses_calls"] = int(state["responses_calls"]) + 1
            body = json.loads(request.content)
            segment_count = body["input"][1]["content"][0]["text"].count('"segment_index"')
            return _prompts_response(
                [
                    {
                        "segment_index": index,
                        "prompt": f"Editorial photograph number {index}, no text, no logos",
                        "primary_visual_subject": f"subject {index}",
                        "visual_keywords": ["newsroom monitors"],
                    }
                    for index in range(segment_count)
                ]
            )
        assert url == OPENAI_IMAGES_URL
        prompts = state["image_prompts"]
        assert isinstance(prompts, list)
        prompts.append(json.loads(request.content)["prompt"])
        if image_responses:
            return image_responses[min(len(prompts) - 1, len(image_responses) - 1)]
        return _image_success_response()

    return handler, state


def test_plan_photo_cues_end_to_end(tmp_path):
    handler, state = _plan_handler()
    output_dir = tmp_path / "public" / "jobs" / "job1" / "photos"

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        cues = plan_photo_cues(WORDS, "sk-test", output_dir, chunk_size=14, client=client)

    assert len(cues) == 2
    assert state["responses_calls"] == 1  # exactly ONE batched prompt call
    assert len(state["image_prompts"]) == 2

    for index, cue in enumerate(cues):
        assert cue["assetKind"] == "photo"
        assert cue["visualMode"] == "photo"
        assert cue["primaryVisualSubject"] == f"subject {index}"
        assert cue["visualKeywords"] == ["newsroom monitors"]
        assert cue["emoticonFileName"].startswith("jobs/job1/photos/")
        assert "/" in cue["emoticonFileName"]
        assert cue["emoticonFileName"].endswith(".png")
        saved = output_dir / cue["emoticonFileName"].rsplit("/", 1)[-1]
        assert saved.is_file()
        assert saved.read_bytes() == PNG_BYTES

    # cue timeline comes from caption_plan segments, gapless and in order
    assert cues[0]["startMs"] == 0
    assert cues[0]["durationMs"] == 1600
    assert cues[1]["startMs"] == 1600


def test_plan_photo_cues_propagates_photo_engine_error(tmp_path):
    handler, _ = _plan_handler([_moderation_response()])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PhotoEngineError):
            plan_photo_cues(WORDS, "sk-test", tmp_path / "photos", client=client)


def test_plan_photo_cues_returns_empty_for_no_words(tmp_path):
    assert plan_photo_cues([], "sk-test", tmp_path / "photos") == []


# --- generate_image_prompts (§5.7 step 3) ---------------------------------


def test_generate_image_prompts_fills_segments_the_model_skipped():
    segments = [
        {"text": "트럼프가 합의를 했습니다.", "startMs": 0, "endMs": 1600, "words": []},
        {"text": "나스닥은 급등했습니다.", "startMs": 1600, "endMs": 2600, "words": []},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _prompts_response(
            [
                {
                    "segment_index": 0,
                    "prompt": "Editorial photograph of a podium",
                    "primary_visual_subject": "Donald Trump",
                    "visual_keywords": ["diplomatic agreement table"],
                }
            ]
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        items = generate_image_prompts(segments, "sk-test", client=client)

    assert len(items) == 2
    assert items[0]["prompt"] == "Editorial photograph of a podium"
    # the skipped segment keeps its local hint and a deterministic prompt
    assert items[1]["segment_index"] == 1
    assert items[1]["primary_visual_subject"] == "the Nasdaq market board"
    assert "Primary visual subject: the Nasdaq market board." in items[1]["prompt"]


def test_generate_image_prompts_raises_without_api_key():
    with pytest.raises(PhotoEngineError):
        generate_image_prompts([{"text": "무언가", "startMs": 0, "endMs": 1}], "")


def test_generate_image_prompts_raises_on_unparseable_output():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"output": [{"content": [{"type": "output_text", "text": "not json"}]}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PhotoEngineError):
            generate_image_prompts(
                [{"text": "무언가", "startMs": 0, "endMs": 1}], "sk-test", client=client
            )
