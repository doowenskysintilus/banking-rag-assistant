"""Extractor for Excel (.xlsx) spreadsheets, using openpyxl.

Only the modern `.xlsx` (Office Open XML) format is supported.
"""

from __future__ import annotations

import io
import zipfile

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.base import ExtractedDocument, Extractor


class ExcelExtractor(Extractor):
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        import openpyxl
        from openpyxl.utils.exceptions import InvalidFileException

        try:
            workbook = openpyxl.load_workbook(
                io.BytesIO(raw_bytes), data_only=True, read_only=True
            )
        except (InvalidFileException, KeyError, OSError, zipfile.BadZipFile) as exc:
            raise ValidationAppError(f"Could not parse Excel file: {exc}") from exc

        sheet_names: list[str] = []
        blocks: list[str] = []

        try:
            for sheet in workbook.worksheets:
                sheet_names.append(sheet.title)
                rows_text: list[str] = []
                for row in sheet.iter_rows(values_only=True):
                    cells = [
                        str(cell).strip()
                        for cell in row
                        if cell is not None and str(cell).strip()
                    ]
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    blocks.append(f"[Sheet: {sheet.title}]\n" + "\n".join(rows_text))
        finally:
            workbook.close()

        return ExtractedDocument(text="\n\n".join(blocks), sheet_names=sheet_names)
