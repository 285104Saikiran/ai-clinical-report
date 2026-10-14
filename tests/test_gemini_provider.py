"""Mocked, network-free tests for the Gemini SDK boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors

from clinical_report_assistant.config import GeminiConfig
from clinical_report_assistant.gemini_provider import (
    GeminiAuthenticationError,
    GeminiBlockedResponseError,
    GeminiEmptyResponseError,
    GeminiModelUnavailableError,
    GeminiProvider,
    GeminiQuotaError,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
)


def _config() -> GeminiConfig:
    return GeminiConfig(api_key="synthetic-test-key", model="explicit-test-model")


def _client_with_result(result: object) -> Mock:
    client = Mock()
    client.models.generate_content.return_value = result
    return client


def _api_error(code: int, status: str) -> errors.APIError:
    return errors.APIError(
        code,
        {
            "error": {
                "code": code,
                "status": status,
                "message": "raw provider detail must not escape",
            }
        },
    )


def test_successful_generation_uses_explicit_limits() -> None:
    response = SimpleNamespace(
        text="  Synthetic summary.  ",
        prompt_feedback=None,
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    client = _client_with_result(response)
    provider = GeminiProvider(_config(), max_output_tokens=123, client=client)

    result = provider.generate_text("Synthetic input only.")

    assert result == "Synthetic summary."
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "explicit-test-model"
    assert call.kwargs["contents"] == "Synthetic input only."
    assert call.kwargs["config"].max_output_tokens == 123


@pytest.mark.parametrize(
    ("code", "status", "expected_error"),
    [
        (401, "UNAUTHENTICATED", GeminiAuthenticationError),
        (404, "NOT_FOUND", GeminiModelUnavailableError),
        (429, "RESOURCE_EXHAUSTED", GeminiQuotaError),
        (503, "UNAVAILABLE", GeminiServiceUnavailableError),
        (504, "DEADLINE_EXCEEDED", GeminiTimeoutError),
    ],
)
def test_translates_sdk_api_errors(
    code: int,
    status: str,
    expected_error: type[Exception],
) -> None:
    client = Mock()
    client.models.generate_content.side_effect = _api_error(code, status)
    provider = GeminiProvider(_config(), client=client)

    with pytest.raises(expected_error) as error:
        provider.generate_text("Synthetic input only.")

    assert "raw provider detail" not in str(error.value)


def test_translates_transport_timeout() -> None:
    client = Mock()
    client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")
    provider = GeminiProvider(_config(), client=client)

    with pytest.raises(GeminiTimeoutError, match="timed out"):
        provider.generate_text("Synthetic input only.")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            text=None,
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
            candidates=[],
        ),
        SimpleNamespace(
            text=None,
            prompt_feedback=None,
            candidates=[SimpleNamespace(finish_reason="RECITATION")],
        ),
    ],
)
def test_rejects_blocked_responses(response: object) -> None:
    provider = GeminiProvider(_config(), client=_client_with_result(response))

    with pytest.raises(GeminiBlockedResponseError, match="blocked"):
        provider.generate_text("Synthetic input only.")


@pytest.mark.parametrize("text", [None, "", "   "])
def test_rejects_empty_responses(text: str | None) -> None:
    response = SimpleNamespace(
        text=text,
        prompt_feedback=None,
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    provider = GeminiProvider(_config(), client=_client_with_result(response))

    with pytest.raises(GeminiEmptyResponseError, match="no usable text"):
        provider.generate_text("Synthetic input only.")


def test_sdk_retry_policy_is_bounded_and_excludes_nonretryable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_client_factory(**kwargs: object) -> Mock:
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "clinical_report_assistant.gemini_provider.genai.Client",
        fake_client_factory,
    )

    GeminiProvider(_config(), timeout_seconds=12.5, retry_attempts=3)

    http_options = captured["http_options"]
    assert http_options.timeout == 12_500
    assert http_options.retry_options.attempts == 3
    statuses = http_options.retry_options.http_status_codes
    assert statuses == [408, 500, 502, 503, 504]
    assert all(code not in statuses for code in [401, 404, 429])
