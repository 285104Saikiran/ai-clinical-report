"""Tests that the Gemini smoke command is explicit and single-attempt."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_report_assistant.config import GeminiConfig
from clinical_report_assistant.gemini_smoke import main


def test_without_live_flag_does_not_load_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called() -> None:
        raise AssertionError("configuration must not load without --live")

    monkeypatch.setattr(
        "clinical_report_assistant.gemini_smoke.load_gemini_config",
        fail_if_called,
    )

    assert main([]) == 0
    assert "No request sent" in capsys.readouterr().out


def test_live_flag_uses_one_attempt_and_builtin_synthetic_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    config = GeminiConfig(
        api_key="synthetic-test-key",
        model="explicit-test-model",
    )

    class FakeProvider:
        def __init__(
            self,
            received_config: GeminiConfig,
            *,
            retry_attempts: int,
        ) -> None:
            captured["config"] = received_config
            captured["retry_attempts"] = retry_attempts

        def generate_text(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "Synthetic smoke result."

    monkeypatch.setattr(
        "clinical_report_assistant.gemini_smoke.load_gemini_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "clinical_report_assistant.gemini_smoke.GeminiProvider",
        FakeProvider,
    )

    assert main(["--live"]) == 0
    assert captured["retry_attempts"] == 1
    assert "fictional patient" in captured["prompt"]
    assert "Synthetic smoke result." in capsys.readouterr().out
