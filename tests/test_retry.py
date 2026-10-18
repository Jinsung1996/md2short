"""Unit tests for md2short._retry.post_with_retry.

Uses httpx.MockTransport exclusively -- no real network calls -- and patches
md2short._retry.time.sleep so backoff durations can be asserted without
actually sleeping.
"""

from __future__ import annotations

from unittest.mock import call, patch

import httpx
import pytest

from md2short._retry import post_with_retry

URL = "http://test/api"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_immediate_success_no_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        response = post_with_retry(client, URL)

    assert response.status_code == 200
    assert calls["n"] == 1
    mock_sleep.assert_not_called()


def test_429_twice_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        response = post_with_retry(client, URL)

    assert response.status_code == 200
    assert calls["n"] == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


def test_500_twice_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500)
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        response = post_with_retry(client, URL)

    assert response.status_code == 200
    assert calls["n"] == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


def test_exhausts_all_retries_propagates():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        with pytest.raises(httpx.HTTPStatusError):
            post_with_retry(client, URL)

    # retries=3 default -> 4 total attempts, 3 sleeps between them.
    assert calls["n"] == 4
    assert mock_sleep.call_args_list == [call(0.5), call(1.0), call(2.0)]


def test_404_does_not_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        with pytest.raises(httpx.HTTPStatusError):
            post_with_retry(client, URL)

    assert calls["n"] == 1
    mock_sleep.assert_not_called()


def test_timeout_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out")
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        response = post_with_retry(client, URL)

    assert response.status_code == 200
    assert calls["n"] == 2
    assert mock_sleep.call_args_list == [call(0.5)]


def test_transport_error_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client, patch("md2short._retry.time.sleep") as mock_sleep:
        response = post_with_retry(client, URL)

    assert response.status_code == 200
    assert calls["n"] == 2
    assert mock_sleep.call_args_list == [call(0.5)]
