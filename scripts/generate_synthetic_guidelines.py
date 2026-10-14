"""Generate substantial, non-authoritative PDFs for manual retrieval tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUTPUT_DIRECTORY = Path("generated/synthetic_guidelines")
_LABEL = "SYNTHETIC TRAINING GUIDELINE - NOT MEDICAL ADVICE"
_REVIEW_LABEL = "SYNTHETIC CAPSTONE REVIEW GUIDANCE - NOT MEDICAL ADVICE."
_REVIEW_FILENAME = "synthetic_capstone_review_guidance.pdf"
_EXAMPLES = {
    "synthetic_hydration_guideline.pdf": [
        (
            "Intake log exercise",
            [
                (
                    "This fictional capstone scenario uses the code River-7 for a "
                    "practice hydration log. The trainee records the synthetic "
                    "participant identifier, beverage label, simulated volume, and "
                    "entry time. River-7 is an invented workflow marker and is not "
                    "a clinical protocol."
                ),
                (
                    "For the indexing exercise, each entry also records whether the "
                    "amount was reported by the fictional participant or copied from "
                    "the training worksheet. Missing fields are marked NOT RECORDED; "
                    "they are never inferred. The purpose is to retrieve passages "
                    "about provenance, timestamps, and structured intake records."
                ),
                (
                    "A reviewer checks that the source label and time are present, "
                    "then signs the synthetic record with the phrase BLUE CUP CHECK. "
                    "This distinctive phrase exists solely to make retrieval results "
                    "easy to inspect during the capstone demonstration."
                ),
            ],
        ),
        (
            "Follow-up reconciliation exercise",
            [
                (
                    "During the fictional follow-up, River-7 entries are compared "
                    "with the practice worksheet. The reviewer lists discrepancies "
                    "without deciding which value is medically correct. A mismatch "
                    "is labeled TRAINING REVIEW NEEDED and remains visible in the "
                    "audit notes."
                ),
                (
                    "The reconciliation record includes the original synthetic "
                    "entry, the worksheet entry, the page reference, and a neutral "
                    "description of the difference. It does not diagnose, recommend "
                    "treatment, or replace review by a qualified professional."
                ),
                (
                    "The completion marker for this page is RIVER ARCHIVE COMPLETE. "
                    "A useful manual query is: What information is retained during "
                    "River-7 reconciliation? The retrieved result should cite this "
                    "document and page two."
                ),
            ],
        ),
    ],
    "synthetic_sleep_guideline.pdf": [
        (
            "Diary capture exercise",
            [
                (
                    "This fictional capstone scenario uses the code Moon-4 for a "
                    "practice sleep diary. The trainee records simulated lights-out "
                    "time, wake time, diary completion time, and the source of each "
                    "entry. Moon-4 is invented test vocabulary, not medical guidance."
                ),
                (
                    "If the training worksheet omits a time, the diary displays NOT "
                    "RECORDED rather than estimating a value. Notes distinguish a "
                    "participant-reported entry from a value supplied by the exercise "
                    "facilitator. This supports retrieval about missing data and "
                    "source attribution."
                ),
                (
                    "The page verification phrase is SILVER CLOCK CHECK. It is a "
                    "synthetic search marker that lets testers confirm the sleep "
                    "document is retrieved instead of the hydration document."
                ),
            ],
        ),
        (
            "Diary comparison exercise",
            [
                (
                    "At the fictional follow-up, Moon-4 diary rows are compared by "
                    "date. Differences are recorded as observations in the training "
                    "table and are not interpreted as symptoms or diagnoses. The "
                    "exercise preserves the original value and its page source."
                ),
                (
                    "The reviewer checks for duplicate dates, absent source labels, "
                    "and inconsistent fictional timestamps. Items needing another "
                    "look receive the label TRAINING REVIEW NEEDED. Nothing in this "
                    "example recommends a clinical action."
                ),
                (
                    "The completion marker for this page is MOON ARCHIVE COMPLETE. A "
                    "useful manual query is: How are missing Moon-4 diary times "
                    "represented? The retrieved passage should point to the relevant "
                    "document and page."
                ),
            ],
        ),
    ],
}

_REVIEW_PAGES = [
    (
        "Glucose fasting-status documentation review",
        [
            (
                "Glucose with unknown fasting status: ask the reviewer to verify and document fasting status before interpreting the supplied reference range. This fictional rule supports documentation review only."
            )
        ],
    ),
    (
        "Missing reference-range documentation review",
        [
            (
                "Missing reference range for a laboratory finding: ask the reviewer to obtain the original laboratory interval and record its source before comparing the result. Do not infer an interval."
            )
        ],
    ),
    (
        "Mismatched-unit documentation review",
        [
            (
                "Mismatched units for a laboratory finding: ask the reviewer to verify the reported result unit and the original laboratory interval unit before comparison. Do not convert or guess units."
            )
        ],
    ),
    (
        "Medication-uncertainty documentation review",
        [
            (
                "Medication uncertainty: ask the reviewer to reconcile the recorded medication name, dose field, and frequency field against the original record. This is a documentation check only."
            )
        ],
    ),
    (
        "Allergy-verification documentation review",
        [
            (
                "Incomplete allergy verification: ask the reviewer to verify and document the allergy history from the original record. Preserve whether verification remains incomplete."
            )
        ],
    ),
    (
        "Missing-follow-up documentation review",
        [
            (
                "Missing follow-up: ask the reviewer to verify whether a follow-up plan was recorded in the original record. Do not treat not recorded as evidence that no plan exists."
            )
        ],
    ),
]


def _draw_page(
    document: canvas.Canvas,
    *,
    title: str,
    paragraphs: list[str],
    page_number: int,
    label: str = _LABEL,
) -> None:
    document.setFont("Helvetica-Bold", 12)
    document.drawString(54, 748, label)
    document.setFont("Helvetica-Bold", 14)
    document.drawString(54, 718, title)

    text = document.beginText(54, 685)
    text.setFont("Helvetica", 10)
    text.setLeading(15)
    for paragraph in paragraphs:
        for line in wrap(paragraph, width=88):
            text.textLine(line)
        text.textLine("")
    text.textLine(f"Synthetic page {page_number}.")
    document.drawText(text)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, pages in _EXAMPLES.items():
        destination = OUTPUT_DIRECTORY / filename
        document = canvas.Canvas(str(destination), pagesize=letter)
        for page_number, (title, paragraphs) in enumerate(pages, start=1):
            _draw_page(
                document,
                title=title,
                paragraphs=paragraphs,
                page_number=page_number,
            )
            document.showPage()
        document.save()
        print(destination)

    destination = OUTPUT_DIRECTORY / _REVIEW_FILENAME
    document = canvas.Canvas(str(destination), pagesize=letter)
    for page_number, (title, paragraphs) in enumerate(_REVIEW_PAGES, start=1):
        _draw_page(
            document,
            title=title,
            paragraphs=paragraphs,
            page_number=page_number,
            label=_REVIEW_LABEL,
        )
        document.showPage()
    document.save()
    print(destination)


if __name__ == "__main__":
    main()
