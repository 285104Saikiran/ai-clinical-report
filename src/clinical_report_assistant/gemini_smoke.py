"""Explicit, synthetic-only live smoke test for the Gemini provider."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from clinical_report_assistant.config import (
    GeminiConfigurationError,
    load_gemini_config,
)
from clinical_report_assistant.gemini_provider import (
    GeminiProvider,
    GeminiProviderError,
)

_SYNTHETIC_PROMPT = """
Summarize this synthetic capstone report in one short, neutral sentence:
The fictional patient reports improved sleep after following a regular bedtime.
No real patient information is included.
""".strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one explicit live Gemini request with synthetic text."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Acknowledge that this command makes one quota-consuming API request.",
    )
    arguments = parser.parse_args(argv)

    if not arguments.live:
        print("No request sent. Re-run with --live to make one synthetic API call.")
        return 0

    try:
        config = load_gemini_config()
        provider = GeminiProvider(config, retry_attempts=1)
        generated_text = provider.generate_text(_SYNTHETIC_PROMPT)
    except (GeminiConfigurationError, GeminiProviderError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
