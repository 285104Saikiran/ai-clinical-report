"""Server-side configuration loading for Gemini access."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS = 8_192
MAX_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS = 4_096
MAX_SUMMARY_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_RECOMMENDATION_MAX_OUTPUT_TOKENS = 4_096
MAX_RECOMMENDATION_MAX_OUTPUT_TOKENS = 65_536


class GeminiConfigurationError(ValueError):
    """Raised when required Gemini configuration is absent or a placeholder."""


class GeminiConfig(BaseModel):
    """Validated Gemini credentials and explicit model selection."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    model: str
    report_analysis_max_output_tokens: int = Field(
        default=DEFAULT_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
        ge=1,
        le=MAX_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
    )
    summary_max_output_tokens: int = Field(
        default=DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
        ge=1,
        le=MAX_SUMMARY_MAX_OUTPUT_TOKENS,
    )
    recommendation_max_output_tokens: int = Field(
        default=DEFAULT_RECOMMENDATION_MAX_OUTPUT_TOKENS,
        ge=1,
        le=MAX_RECOMMENDATION_MAX_OUTPUT_TOKENS,
    )


def load_gemini_config(
    *,
    streamlit_secrets: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
) -> GeminiConfig:
    """Load Gemini settings without exposing secret values.

    Precedence is process environment, Streamlit secrets, then local .env.
    """

    environment_values = os.environ if environment is None else environment
    dotenv_values_map = dotenv_values(dotenv_path)
    configured: dict[str, str] = {}
    missing: list[str] = []

    for name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        value = _first_nonempty(
            environment_values.get(name),
            _mapping_value(streamlit_secrets, name),
            dotenv_values_map.get(name),
        )
        if value is None:
            missing.append(name)
        else:
            configured[name] = value

    if missing:
        names = ", ".join(missing)
        raise GeminiConfigurationError(
            f"Missing required Gemini configuration: {names}."
        )

    if _is_placeholder(configured["GEMINI_API_KEY"]):
        raise GeminiConfigurationError(
            "GEMINI_API_KEY still contains a placeholder value."
        )
    if _is_placeholder(configured["GEMINI_MODEL"]):
        raise GeminiConfigurationError(
            "GEMINI_MODEL still contains a placeholder value."
        )

    analysis_output_tokens = _optional_positive_integer(
        "REPORT_ANALYSIS_MAX_OUTPUT_TOKENS",
        _first_nonempty(
            environment_values.get("REPORT_ANALYSIS_MAX_OUTPUT_TOKENS"),
            _mapping_scalar_value(
                streamlit_secrets,
                "REPORT_ANALYSIS_MAX_OUTPUT_TOKENS",
            ),
            dotenv_values_map.get("REPORT_ANALYSIS_MAX_OUTPUT_TOKENS"),
        ),
        default=DEFAULT_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
        maximum=MAX_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
    )
    summary_output_tokens = _optional_positive_integer(
        "SUMMARY_MAX_OUTPUT_TOKENS",
        _first_nonempty(
            environment_values.get("SUMMARY_MAX_OUTPUT_TOKENS"),
            _mapping_scalar_value(
                streamlit_secrets,
                "SUMMARY_MAX_OUTPUT_TOKENS",
            ),
            dotenv_values_map.get("SUMMARY_MAX_OUTPUT_TOKENS"),
        ),
        default=DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
        maximum=MAX_SUMMARY_MAX_OUTPUT_TOKENS,
    )
    recommendation_output_tokens = _optional_positive_integer(
        "RECOMMENDATION_MAX_OUTPUT_TOKENS",
        _first_nonempty(
            environment_values.get("RECOMMENDATION_MAX_OUTPUT_TOKENS"),
            _mapping_scalar_value(
                streamlit_secrets,
                "RECOMMENDATION_MAX_OUTPUT_TOKENS",
            ),
            dotenv_values_map.get("RECOMMENDATION_MAX_OUTPUT_TOKENS"),
        ),
        default=DEFAULT_RECOMMENDATION_MAX_OUTPUT_TOKENS,
        maximum=MAX_RECOMMENDATION_MAX_OUTPUT_TOKENS,
    )

    return GeminiConfig(
        api_key=configured["GEMINI_API_KEY"],
        model=configured["GEMINI_MODEL"],
        report_analysis_max_output_tokens=analysis_output_tokens,
        summary_max_output_tokens=summary_output_tokens,
        recommendation_max_output_tokens=recommendation_output_tokens,
    )


def _mapping_value(
    values: Mapping[str, Any] | None,
    name: str,
) -> str | None:
    if values is None:
        return None
    value = values.get(name)
    return value if isinstance(value, str) else None


def _mapping_scalar_value(
    values: Mapping[str, Any] | None,
    name: str,
) -> str | None:
    if values is None:
        return None
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return str(value)


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


def _is_placeholder(value: str) -> bool:
    normalized = value.casefold()
    return normalized.startswith(("replace_with_", "your_", "<"))


def _optional_positive_integer(
    name: str,
    raw_value: str | None,
    *,
    default: int,
    maximum: int,
) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    if value < 1 or value > maximum:
        raise GeminiConfigurationError(
            f"{name} must be a whole number from 1 through {maximum}."
        )
    return value
