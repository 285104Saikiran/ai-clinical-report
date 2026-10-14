'''Generate a clearly fictional report for the manual analysis walkthrough.'''

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUTPUT_PATH = Path('generated/synthetic_reports/synthetic_lab_report.pdf')
_PAGES = [
    [
        'SYNTHETIC CAPSTONE REPORT - NOT MEDICAL ADVICE',
        'Fictional record identifier: DEMO-R4-001',
        'Glucose result: 110 mg/dL.',
        'Glucose reference range: 70-99 mg/dL.',
        'This page exists only to test a deterministic above-range comparison.',
    ],
    [
        'SYNTHETIC CAPSTONE REPORT - NOT MEDICAL ADVICE',
        'Fictional record identifier: DEMO-R4-001',
        'Sodium result: 140 mmol/L.',
        'Sodium reference range: 135-145 mmol/L.',
        'This page exists only to test a deterministic within-range comparison.',
    ],
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(OUTPUT_PATH), pagesize=letter)
    for page_number, lines in enumerate(_PAGES, start=1):
        text = document.beginText(64, 730)
        text.setLeading(24)
        for line in lines:
            text.textLine(line)
        text.textLine(f'Synthetic page {page_number}.')
        document.drawText(text)
        document.showPage()
    document.save()
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
