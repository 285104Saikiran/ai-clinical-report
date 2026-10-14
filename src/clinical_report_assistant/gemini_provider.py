"""Small, testable boundary around the Google Gen AI SDK."""

from __future__ import annotations

from typing import Any, Never

import httpx
from google import genai
from google.genai import errors, types

from clinical_report_assistant.config import GeminiConfig

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = [408, 500, 502, 503, 504]
_BLOCKED_FINISH_REASONS = {
    "BLOCKLIST",
    "IMAGE_OTHER",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_SAFETY",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "SAFETY",
    "SPII",
}


class GeminiProviderError(RuntimeError):
    """Base class for safe, user-facing Gemini provider failures."""


class GeminiAuthenticationError(GeminiProviderError):
    """Raised when credentials are invalid or lack access."""


class GeminiModelUnavailableError(GeminiProviderError):
    """Raised when the explicitly configured model is unavailable."""


class GeminiQuotaError(GeminiProviderError):
    """Raised when the account or project has exhausted a quota."""


class GeminiTimeoutError(GeminiProviderError):
    """Raised when generation does not finish within the timeout."""


class GeminiServiceUnavailableError(GeminiProviderError):
    """Raised after bounded retries for a transient service failure."""


class GeminiBlockedResponseError(GeminiProviderError):
    """Raised when policy or safety controls block generation."""


class GeminiEmptyResponseError(GeminiProviderError):
    """Raised when generation succeeds without returning text."""


class GeminiRequestError(GeminiProviderError):
    """Raised for a non-retryable request or unexpected SDK failure."""


class GeminiProvider:
    """Generate text through one configured Gemini model."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        client: Any | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero.")
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be at least one.")

        self._model = config.model
        self._max_output_tokens = max_output_tokens
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=config.api_key.get_secret_value(),
                http_options=types.HttpOptions(
                    timeout=int(timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(
                        attempts=retry_attempts,
                        initial_delay=0.5,
                        max_delay=2.0,
                        exp_base=2.0,
                        jitter=0.25,
                        http_status_codes=_RETRYABLE_STATUS_CODES,
                    ),
                ),
            )
        )

    def generate_text(self, prompt: str) -> str:
        """Generate bounded text without logging the prompt or raw response."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    candidate_count=1,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - sanitize external SDK failures
            raise_sanitized_gemini_exception(exc)

        _raise_if_blocked(response)
        try:
            text = response.text
        except (AttributeError, ValueError) as exc:
            raise GeminiEmptyResponseError("Gemini returned no usable text.") from exc
        if not text or not text.strip():
            raise GeminiEmptyResponseError("Gemini returned no usable text.")
        return text.strip()


def _raise_for_api_error(error: errors.APIError) -> Never:
    code = error.code
    status = (error.status or "").upper()

    if code in {401, 403} or status in {"API_KEY_INVALID", "UNAUTHENTICATED"}:
        raise GeminiAuthenticationError(
            "Gemini authentication failed. Check GEMINI_API_KEY and its access."
        ) from error
    if code == 404 or status in {"MODEL_NOT_FOUND", "NOT_FOUND"}:
        raise GeminiModelUnavailableError(
            "The configured Gemini model is unavailable. Check GEMINI_MODEL."
        ) from error
    if code == 429 or status in {
        "QUOTA_EXCEEDED",
        "RATE_LIMIT_EXCEEDED",
        "RESOURCE_EXHAUSTED",
    }:
        raise GeminiQuotaError(
            "Gemini quota or rate limits were exceeded. Check the project quota."
        ) from error
    if code in {408, 504} or status == "DEADLINE_EXCEEDED":
        raise GeminiTimeoutError(
            "The Gemini request timed out. Try again later."
        ) from error
    if code in {500, 502, 503}:
        raise GeminiServiceUnavailableError(
            "Gemini is temporarily unavailable after bounded retries."
        ) from error
    raise GeminiRequestError(
        "Gemini rejected the request. Check the configured model and request."
    ) from error


def raise_sanitized_gemini_exception(
    error: Exception,
    *,
    fallback_message: str = (
        "Gemini generation failed before a usable response was returned."
    ),
) -> Never:
    """Map direct or wrapped SDK failures to safe, prompt-free errors."""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, GeminiProviderError):
            raise current
        if isinstance(current, (httpx.TimeoutException, TimeoutError)):
            raise GeminiTimeoutError(
                "The Gemini request timed out. Try again later."
            ) from error
        if isinstance(current, errors.APIError):
            _raise_for_api_error(current)
        current = current.__cause__ or current.__context__
    raise GeminiRequestError(fallback_message) from error


def _raise_if_blocked(response: Any) -> None:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    prompt_reason = _reason_name(getattr(prompt_feedback, "block_reason", None))
    if prompt_reason and not prompt_reason.endswith("UNSPECIFIED"):
        raise GeminiBlockedResponseError(
            "Gemini blocked the prompt or response under its safety policies."
        )

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = _reason_name(getattr(candidates[0], "finish_reason", None))
        if finish_reason in _BLOCKED_FINISH_REASONS:
            raise GeminiBlockedResponseError(
                "Gemini blocked the prompt or response under its safety policies."
            )


def _reason_name(reason: Any) -> str:
    if reason is None:
        return ""
    raw_value = getattr(reason, "value", reason)
    return str(raw_value).rsplit(".", maxsplit=1)[-1].upper()
