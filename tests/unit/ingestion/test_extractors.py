import io

import pytest

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.docx_extractor import DocxExtractor
from backend.ingestion.extractors.excel_extractor import ExcelExtractor
from backend.ingestion.extractors.html_extractor import HTMLExtractor
from backend.ingestion.extractors.pdf_extractor import PDFExtractor
from backend.ingestion.extractors.registry import (
    DOCX_MIME,
    XLSX_MIME,
    get_extractor,
    resolve_content_type,
)
from backend.ingestion.extractors.txt_extractor import TextExtractor


class TestTextExtractor:
    def test_decodes_utf8_bytes(self):
        result = TextExtractor().extract("Hello, overdraft policy.".encode("utf-8"))
        assert result.text == "Hello, overdraft policy."

    def test_replaces_invalid_bytes_instead_of_raising(self):
        result = TextExtractor().extract(b"valid text \xff\xfe invalid")
        assert "valid text" in result.text


class TestHTMLExtractor:
    def test_extracts_visible_text_and_title(self):
        html = b"""
        <html><head><title>Overdraft Policy</title>
        <meta name="author" content="Compliance Team">
        <style>.x { color: red; }</style>
        </head>
        <body><h1>Overdraft Policy</h1><p>Fees apply after day one.</p>
        <script>alert('nope')</script>
        </body></html>
        """
        result = HTMLExtractor().extract(html)
        assert "Fees apply after day one." in result.text
        assert "alert" not in result.text
        assert "color: red" not in result.text
        assert result.title == "Overdraft Policy"
        assert result.author == "Compliance Team"

    def test_no_title_or_author_is_none(self):
        result = HTMLExtractor().extract(b"<html><body><p>Just text.</p></body></html>")
        assert result.title is None
        assert result.author is None


class TestDocxExtractor:
    def _build_docx_bytes(self) -> bytes:
        import docx

        document = docx.Document()
        document.add_paragraph("Overdraft protection terms.")
        document.add_paragraph("Fees are waived for premium accounts.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Tier"
        table.rows[0].cells[1].text = "Fee"
        document.core_properties.title = "Overdraft Policy"
        document.core_properties.author = "Compliance Team"

        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    def test_extracts_paragraphs_tables_and_core_properties(self):
        raw_bytes = self._build_docx_bytes()
        result = DocxExtractor().extract(raw_bytes)

        assert "Overdraft protection terms." in result.text
        assert "Fees are waived for premium accounts." in result.text
        assert "Tier | Fee" in result.text
        assert result.title == "Overdraft Policy"
        assert result.author == "Compliance Team"

    def test_raises_validation_error_on_corrupt_file(self):
        with pytest.raises(ValidationAppError):
            DocxExtractor().extract(b"not a real docx file")


class TestExcelExtractor:
    def _build_xlsx_bytes(self) -> bytes:
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Fees"
        sheet.append(["Tier", "Monthly Fee"])
        sheet.append(["Standard", 5])
        sheet.append(["Premium", 0])

        buffer = io.BytesIO()
        workbook.save(buffer)
        return buffer.getvalue()

    def test_extracts_rows_with_sheet_name(self):
        raw_bytes = self._build_xlsx_bytes()
        result = ExcelExtractor().extract(raw_bytes)

        assert result.sheet_names == ["Fees"]
        assert "[Sheet: Fees]" in result.text
        assert "Tier | Monthly Fee" in result.text
        assert "Standard | 5" in result.text

    def test_raises_validation_error_on_corrupt_file(self):
        with pytest.raises(ValidationAppError):
            ExcelExtractor().extract(b"not a real xlsx file")


class TestPDFExtractor:
    def test_aggregates_text_across_pages(self, monkeypatch):
        class FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class FakeMetadata:
            title = "Overdraft Policy"
            author = "Compliance Team"

        class FakeReader:
            def __init__(self, _stream) -> None:
                self.pages = [
                    FakePage("Page one content."),
                    FakePage("Page two content."),
                ]
                self.metadata = FakeMetadata()

        monkeypatch.setattr("pypdf.PdfReader", FakeReader)

        result = PDFExtractor().extract(b"%PDF-1.4 fake bytes")

        assert "Page one content." in result.text
        assert "Page two content." in result.text
        assert result.page_count == 2
        assert result.title == "Overdraft Policy"
        assert result.author == "Compliance Team"

    def test_raises_validation_error_on_corrupt_pdf(self, monkeypatch):
        from pypdf.errors import PdfReadError

        def _raise(_stream):
            raise PdfReadError("bad header")

        monkeypatch.setattr("pypdf.PdfReader", _raise)

        with pytest.raises(ValidationAppError):
            PDFExtractor().extract(b"not a pdf")


class TestRegistry:
    def test_trusts_supported_client_content_type(self):
        assert (
            resolve_content_type("application/pdf", "anything.bin") == "application/pdf"
        )

    def test_falls_back_to_extension_for_generic_content_type(self):
        assert (
            resolve_content_type("application/octet-stream", "policy.docx") == DOCX_MIME
        )
        assert (
            resolve_content_type("application/octet-stream", "sheet.xlsx") == XLSX_MIME
        )
        assert resolve_content_type(None, "notes.txt") == "text/plain"
        assert resolve_content_type(None, "page.html") == "text/html"

    def test_raises_for_unsupported_extension_and_content_type(self):
        with pytest.raises(ValidationAppError):
            resolve_content_type("application/zip", "archive.zip")

    def test_get_extractor_returns_matching_instance(self):
        assert isinstance(get_extractor("text/plain"), TextExtractor)
        assert isinstance(get_extractor(DOCX_MIME), DocxExtractor)

    def test_get_extractor_raises_for_unknown_type(self):
        with pytest.raises(ValidationAppError):
            get_extractor("application/unknown")
