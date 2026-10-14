"""Regression test for the manual Report Analysis Agent fixture."""

import importlib.util
from pathlib import Path

from clinical_report_assistant.pdf_processing import process_pdf

_GENERATOR_PATH = Path(__file__).parents[1] / "scripts" / "generate_synthetic_report.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "synthetic_report_generator_under_test",
        _GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load the synthetic report generator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_report_has_selectable_page_referenced_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generator = _load_generator()
    output_path = tmp_path / "synthetic_lab_report.pdf"
    monkeypatch.setattr(generator, "OUTPUT_PATH", output_path)

    generator.main()

    report = process_pdf(output_path.read_bytes(), output_path.name)
    assert report.page_count == 2
    assert all(page.text.strip() for page in report.pages)
    assert "Glucose result: 110 mg/dL" in report.pages[0].text
    assert "70-99 mg/dL" in report.pages[0].text
    assert "Sodium result: 140 mmol/L" in report.pages[1].text
    assert "135-145 mmol/L" in report.pages[1].text
