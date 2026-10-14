"""Offline tests for Gemini configuration precedence and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_report_assistant.config import (
    DEFAULT_RECOMMENDATION_MAX_OUTPUT_TOKENS,
    DEFAULT_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
    DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
    GeminiConfigurationError,
    load_gemini_config,
)


def test_requires_api_key_and_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(GeminiConfigurationError) as error:
        load_gemini_config(
            environment={},
            streamlit_secrets={},
            dotenv_path=tmp_path / "missing.env",
        )

    message = str(error.value)
    assert "GEMINI_API_KEY" in message
    assert "GEMINI_MODEL" in message


def test_rejects_placeholder_values(tmp_path: Path) -> None:
    with pytest.raises(GeminiConfigurationError, match="placeholder"):
        load_gemini_config(
            environment={
                "GEMINI_API_KEY": "replace_with_key",
                "GEMINI_MODEL": "explicit-model",
            },
            dotenv_path=tmp_path / "missing.env",
        )


def test_process_environment_has_highest_precedence(tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "GEMINI_API_KEY=dotenv-key\nGEMINI_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    config = load_gemini_config(
        environment={
            "GEMINI_API_KEY": "environment-key",
            "GEMINI_MODEL": "environment-model",
        },
        streamlit_secrets={
            "GEMINI_API_KEY": "streamlit-key",
            "GEMINI_MODEL": "streamlit-model",
        },
        dotenv_path=dotenv_file,
    )

    assert config.api_key.get_secret_value() == "environment-key"
    assert config.model == "environment-model"
    assert (
        config.report_analysis_max_output_tokens
        == DEFAULT_REPORT_ANALYSIS_MAX_OUTPUT_TOKENS
    )
    assert config.summary_max_output_tokens == DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS
    assert (
        config.recommendation_max_output_tokens
        == DEFAULT_RECOMMENDATION_MAX_OUTPUT_TOKENS
    )
    assert "environment-key" not in repr(config)


def test_streamlit_secrets_override_dotenv(tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "GEMINI_API_KEY=dotenv-key\nGEMINI_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    config = load_gemini_config(
        environment={},
        streamlit_secrets={
            "GEMINI_API_KEY": "streamlit-key",
            "GEMINI_MODEL": "streamlit-model",
        },
        dotenv_path=dotenv_file,
    )

    assert config.api_key.get_secret_value() == "streamlit-key"
    assert config.model == "streamlit-model"


def test_report_analysis_output_budget_is_separate_and_configurable(
    tmp_path: Path,
) -> None:
    config = load_gemini_config(
        environment={
            "GEMINI_API_KEY": "environment-key",
            "GEMINI_MODEL": "environment-model",
            "REPORT_ANALYSIS_MAX_OUTPUT_TOKENS": "12288",
        },
        streamlit_secrets={"REPORT_ANALYSIS_MAX_OUTPUT_TOKENS": 4096},
        dotenv_path=tmp_path / "missing.env",
    )

    assert config.report_analysis_max_output_tokens == 12_288


def test_summary_output_budget_is_separate_and_configurable(tmp_path: Path) -> None:
    config = load_gemini_config(
        environment={
            "GEMINI_API_KEY": "environment-key",
            "GEMINI_MODEL": "environment-model",
            "SUMMARY_MAX_OUTPUT_TOKENS": "6144",
        },
        streamlit_secrets={"SUMMARY_MAX_OUTPUT_TOKENS": 2048},
        dotenv_path=tmp_path / "missing.env",
    )

    assert config.summary_max_output_tokens == 6_144


def test_recommendation_output_budget_is_separate_and_configurable(
    tmp_path: Path,
) -> None:
    config = load_gemini_config(
        environment={
            "GEMINI_API_KEY": "environment-key",
            "GEMINI_MODEL": "environment-model",
            "RECOMMENDATION_MAX_OUTPUT_TOKENS": "5120",
        },
        streamlit_secrets={"RECOMMENDATION_MAX_OUTPUT_TOKENS": 2048},
        dotenv_path=tmp_path / "missing.env",
    )

    assert config.recommendation_max_output_tokens == 5_120


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "65537"])
def test_rejects_invalid_report_analysis_output_budget(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        GeminiConfigurationError,
        match="REPORT_ANALYSIS_MAX_OUTPUT_TOKENS",
    ):
        load_gemini_config(
            environment={
                "GEMINI_API_KEY": "environment-key",
                "GEMINI_MODEL": "environment-model",
                "REPORT_ANALYSIS_MAX_OUTPUT_TOKENS": value,
            },
            dotenv_path=tmp_path / "missing.env",
        )


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "65537"])
def test_rejects_invalid_summary_output_budget(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        GeminiConfigurationError,
        match="SUMMARY_MAX_OUTPUT_TOKENS",
    ):
        load_gemini_config(
            environment={
                "GEMINI_API_KEY": "environment-key",
                "GEMINI_MODEL": "environment-model",
                "SUMMARY_MAX_OUTPUT_TOKENS": value,
            },
            dotenv_path=tmp_path / "missing.env",
        )


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "65537"])
def test_rejects_invalid_recommendation_output_budget(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        GeminiConfigurationError,
        match="RECOMMENDATION_MAX_OUTPUT_TOKENS",
    ):
        load_gemini_config(
            environment={
                "GEMINI_API_KEY": "environment-key",
                "GEMINI_MODEL": "environment-model",
                "RECOMMENDATION_MAX_OUTPUT_TOKENS": value,
            },
            dotenv_path=tmp_path / "missing.env",
        )
