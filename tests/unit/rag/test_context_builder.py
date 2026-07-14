from backend.rag.context_builder import ContextBuilder, SourceChunk


def test_no_chunks_returns_the_no_context_fallback():
    result = ContextBuilder().build([], {})
    assert "No relevant internal documents were found" in result.context_text
    assert result.sources == []


def test_builds_numbered_source_markers_with_filenames():
    chunks = [
        SourceChunk(
            document_id="d1", chunk_id="c1", content="Overdraft fee text.", score=0.9
        ),
        SourceChunk(
            document_id="d2", chunk_id="c2", content="Wire transfer text.", score=0.8
        ),
    ]
    filenames = {"d1": "overdraft_policy.pdf", "d2": "wire_transfers.pdf"}

    result = ContextBuilder().build(chunks, filenames)

    assert "[Source 1: overdraft_policy.pdf]" in result.context_text
    assert "[Source 2: wire_transfers.pdf]" in result.context_text
    assert "Overdraft fee text." in result.context_text
    assert "Wire transfer text." in result.context_text


def test_includes_header_path_when_present():
    chunks = [
        SourceChunk(
            document_id="d1",
            chunk_id="c1",
            content="text",
            score=0.9,
            header_path="Article 5",
        )
    ]
    result = ContextBuilder().build(chunks, {"d1": "policy.pdf"})
    assert "[Source 1: policy.pdf — Article 5]" in result.context_text


def test_omits_header_path_marker_when_absent():
    chunks = [
        SourceChunk(
            document_id="d1", chunk_id="c1", content="text", score=0.9, header_path=""
        )
    ]
    result = ContextBuilder().build(chunks, {"d1": "policy.pdf"})
    assert "[Source 1: policy.pdf]" in result.context_text
    assert "—" not in result.context_text


def test_unknown_document_id_falls_back_to_a_placeholder_name():
    chunks = [
        SourceChunk(document_id="missing", chunk_id="c1", content="text", score=0.5)
    ]
    result = ContextBuilder().build(chunks, {})
    assert "Unknown document" in result.context_text


def test_source_indices_are_one_based_and_sequential():
    chunks = [
        SourceChunk(document_id="d", chunk_id=str(i), content=f"text {i}", score=0.5)
        for i in range(3)
    ]
    result = ContextBuilder().build(chunks, {})
    assert [s.index for s in result.sources] == [1, 2, 3]


def test_source_metadata_matches_input_chunk():
    chunk = SourceChunk(
        document_id="d1",
        chunk_id="c1",
        content="Full excerpt text here.",
        score=0.77,
        header_path="Clause 3",
    )
    result = ContextBuilder().build([chunk], {"d1": "doc.pdf"})

    source = result.sources[0]
    assert source.document_id == "d1"
    assert source.chunk_id == "c1"
    assert source.document_filename == "doc.pdf"
    assert source.header_path == "Clause 3"
    assert source.score == 0.77
    assert source.excerpt == "Full excerpt text here."


def test_excerpt_is_truncated_for_long_content():
    long_text = "x" * 500
    chunk = SourceChunk(document_id="d1", chunk_id="c1", content=long_text, score=0.5)
    result = ContextBuilder().build([chunk], {})
    assert len(result.sources[0].excerpt) == 280
