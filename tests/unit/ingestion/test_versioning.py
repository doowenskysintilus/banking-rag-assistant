from backend.ingestion.versioning import (
    DocumentRef,
    IngestionDecision,
    decide,
    normalize_document_key,
)


def test_normalize_document_key_lowercases_and_slugifies():
    assert normalize_document_key("Overdraft Policy.pdf") == "overdraft-policy.pdf"


def test_normalize_document_key_is_stable_across_separators_and_case():
    assert normalize_document_key("Overdraft Policy.pdf") == normalize_document_key(
        "overdraft   policy.pdf"
    )
    assert normalize_document_key("Overdraft_Policy.pdf") == normalize_document_key(
        "overdraft-policy.pdf"
    )


def test_normalize_document_key_strips_accents():
    assert (
        normalize_document_key("Politique Générale.docx") == "politique-generale.docx"
    )


def test_normalize_document_key_ignores_directory_components():
    assert (
        normalize_document_key("/uploads/2024/Overdraft Policy.pdf")
        == "overdraft-policy.pdf"
    )


def test_normalize_document_key_handles_empty_filename():
    assert normalize_document_key("") == "untitled"


def test_decide_new_when_no_candidates():
    result = decide(exact_duplicate_candidate=None, same_key_current_candidate=None)
    assert result.decision is IngestionDecision.NEW
    assert result.next_version == 1


def test_decide_exact_duplicate_takes_priority():
    duplicate = DocumentRef(id="doc-1", version=2)
    same_key = DocumentRef(id="doc-2", version=5)
    result = decide(
        exact_duplicate_candidate=duplicate, same_key_current_candidate=same_key
    )
    assert result.decision is IngestionDecision.EXACT_DUPLICATE
    assert result.duplicate_of == duplicate


def test_decide_new_version_when_same_key_but_different_content():
    previous = DocumentRef(id="doc-1", version=3)
    result = decide(exact_duplicate_candidate=None, same_key_current_candidate=previous)
    assert result.decision is IngestionDecision.NEW_VERSION
    assert result.supersedes == previous
    assert result.next_version == 4
