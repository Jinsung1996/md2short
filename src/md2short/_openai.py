"""Shared OpenAI Responses API helpers (`post_responses`, `extract_output_text`).

`script_engine` (narration script, architecture.md §5.1) and `photo_engine`
(batched image prompts, §5.7) drive the same Structured Outputs endpoint with
the same request envelope, so the endpoint URL, the POST/client lifecycle and
the response-text extraction live here once instead of once per caller. Each
caller still owns its own JSON schema, system prompt and timeout constant --
only the parts that are provably identical moved here.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from md2short._retry import post_with_retry

__all__ = ["OPENAI_RESPONSES_URL", "post_responses", "extract_output_text"]

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def post_responses(
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: float,
    client: Optional[httpx.Client] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """POST `payload` to the Responses API and return the decoded JSON body.

    Retries 429/5xx/timeout via `_retry.post_with_retry`. A `client` passed in
    (tests inject one backed by `httpx.MockTransport`) is left open for its
    owner to close; a client created here is closed before returning.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    owned_client = client is None
    active = client or httpx.Client(timeout=timeout)
    try:
        response = post_with_retry(
            active,
            OPENAI_RESPONSES_URL,
            headers=headers,
            json=payload,
            sleep=sleep,
        )
        return response.json()
    finally:
        if owned_client:
            active.close()


def extract_output_text(body: dict[str, Any]) -> str:
    """Pull the model's text output out of a Responses API body.

    Accepts the `output_text` convenience field (string or list of strings) and
    otherwise walks `output[].content[]` for `output_text`/`text` parts, so a
    field-name difference between API revisions does not break parsing.
    """
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    if isinstance(direct, list):
        joined = "".join(part for part in direct if isinstance(part, str))
        if joined.strip():
            return joined

    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (None, "output_text", "text"):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)
