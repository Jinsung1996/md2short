"""Shared httpx retry helper (`post_with_retry`).

Used by voice.py (ElevenLabs TTS), transcript.py (ElevenLabs Scribe STT) and
photo_engine.py (OpenAI image generation) per architecture.md §5.2 / quality
rules table: "TTS/STT/Image API retry 3회 + 0.5/1.0/2.0s backoff".

Retry-count interpretation (architecture.md wording is ambiguous)
-------------------------------------------------------------------
architecture.md says "최대 3회 재시도" (max 3 retries) and separately gives
exactly three backoff durations: 0.5s, 1.0s, 2.0s. This module reads "3회
재시도" literally as *3 retries after the initial attempt* -- i.e. up to 4
total attempts -- because that is the only reading under which all three
backoff durations are actually used as sleep times between consecutive
attempts:

    attempt 1 -> fail -> sleep 0.5s
    attempt 2 -> fail -> sleep 1.0s
    attempt 3 -> fail -> sleep 2.0s
    attempt 4 -> final attempt (raises on failure, no sleep after)

The alternative reading ("3회" = 3 attempts total) would only ever use two
of the three given backoff values (0.5s and 1.0s), leaving 2.0s dead code,
which does not fit a table that lists all three durations as meaningful.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

__all__ = ["post_with_retry"]


def _is_retryable_status(status_code: int) -> bool:
    """429 or any 5xx is retryable; other 4xx are not."""
    return status_code == 429 or 500 <= status_code < 600


def post_with_retry(
    client: httpx.Client,
    url: str,
    retries: int = 3,
    backoff: tuple[float, ...] = (0.5, 1.0, 2.0),
    sleep: Optional[Callable[[float], None]] = None,
    **httpx_kwargs,
) -> httpx.Response:
    """POST ``url`` via ``client``, retrying on 429 / 5xx / timeout / transport errors.

    Retries up to ``retries`` times after the initial attempt (so up to
    ``retries + 1`` total attempts), sleeping ``backoff[i]`` seconds after the
    (i+1)-th failed attempt (falling back to the last backoff value if
    ``retries`` exceeds ``len(backoff)``). Other 4xx responses (e.g. 404) are
    NOT retried -- ``httpx.HTTPStatusError`` is raised immediately via
    ``response.raise_for_status()``. After the final attempt, the last
    exception/error propagates to the caller rather than being swallowed.

    ``sleep`` defaults to ``None``, in which case ``time.sleep`` is resolved
    at call time (via module attribute lookup), so tests can patch it with
    ``unittest.mock.patch("md2short._retry.time.sleep")``. Callers may also
    pass an explicit ``sleep`` callable to inject a fake directly.
    """
    last_exc: BaseException
    total_attempts = retries + 1

    for attempt in range(total_attempts):
        try:
            response = client.post(url, **httpx_kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
        else:
            if _is_retryable_status(response.status_code):
                last_exc = httpx.HTTPStatusError(
                    f"Retryable status {response.status_code} received from {url}",
                    request=response.request,
                    response=response,
                )
            else:
                # Raises immediately for non-retryable 4xx; no-op for 2xx/3xx.
                response.raise_for_status()
                return response

        is_last_attempt = attempt == total_attempts - 1
        if is_last_attempt:
            raise last_exc

        if backoff:
            delay = backoff[attempt] if attempt < len(backoff) else backoff[-1]
            (sleep or time.sleep)(delay)

    raise last_exc  # pragma: no cover - loop above always returns or raises
