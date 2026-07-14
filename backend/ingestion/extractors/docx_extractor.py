"""Extractor for Word (.docx) documents, using python-docx.

Only the modern `.docx` (Office Open XML) format is supported — legacy binary
`.doc` files require a different parser (e.g. antiword/textract) and are rejected
by the content-type registry before reaching this extractor.
"""

from __future__ import annotations

import io
import zipfile

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.base import ExtractedDocument, Extractor


class DocxExtractor(Extractor):
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        import docx
        from docx.opc.exceptions import PackageNotFoundError

        try:
            document = docx.Document(io.BytesIO(raw_bytes))
        except (PackageNotFoundError, zipfile.BadZipFile) as exc:
            raise ValidationAppError(f"Could not parse DOCX file: {exc}") from exc

        parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        core_props = document.core_properties
        return ExtractedDocument(
            text="\n".join(parts),
            title=core_props.title or None,
            author=core_props.author or None,
        )
