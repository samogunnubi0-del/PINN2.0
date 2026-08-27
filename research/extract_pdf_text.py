"""Extract searchable page-delimited text from PDFs for literature review."""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_pdf_text.py INPUT.pdf OUTPUT.txt")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    reader = PdfReader(source)
    blocks = []
    for index, page in enumerate(reader.pages, start=1):
        blocks.append(f"\n\n===== PDF PAGE {index} =====\n\n")
        blocks.append(page.extract_text(extraction_mode="layout") or "")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(blocks), encoding="utf-8")
    print(f"{source.name}: {len(reader.pages)} pages -> {output}")


if __name__ == "__main__":
    main()
