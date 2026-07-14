import pytest

from backend.core.exceptions import ValidationAppError
from backend.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def pipeline() -> IngestionPipeline:
    return IngestionPipeline(min_text_length=10)


def test_process_txt_end_to_end(pipeline):
    raw = (
        "This is the overdraft policy for standard checking accounts. "
        "Fees apply after the first violation each month."
    ).encode("utf-8")

    processed = pipeline.process("Overdraft Policy.txt", "text/plain", raw)

    assert processed.document_key == "overdraft-policy.txt"
    assert processed.resolved_content_type == "text/plain"
    assert "overdraft policy" in processed.cleaned_text.lower()
    assert processed.metadata.language == "en"
    assert processed.metadata.word_count > 0
    assert processed.metadata.char_count == len(processed.cleaned_text)
    assert len(processed.metadata.content_hash) == 64
    assert isinstance(processed.metadata.simhash, int)


def test_process_is_deterministic_for_identical_input(pipeline):
    raw = b"Identical content used to verify hash and simhash determinism across runs."
    first = pipeline.process("doc.txt", "text/plain", raw)
    second = pipeline.process("doc.txt", "text/plain", raw)

    assert first.metadata.content_hash == second.metadata.content_hash
    assert first.metadata.simhash == second.metadata.simhash


def test_process_falls_back_to_extension_for_generic_content_type(pipeline):
    raw = b"<html><body><p>Policy content long enough to pass the minimum length check.</p></body></html>"
    processed = pipeline.process("page.html", "application/octet-stream", raw)
    assert processed.resolved_content_type == "text/html"
    assert "Policy content" in processed.cleaned_text


def test_process_raises_for_unsupported_format(pipeline):
    with pytest.raises(ValidationAppError):
        pipeline.process("archive.zip", "application/zip", b"some bytes")


def test_process_raises_when_extracted_text_too_short(pipeline):
    with pytest.raises(ValidationAppError):
        pipeline.process("empty.txt", "text/plain", b"hi")
