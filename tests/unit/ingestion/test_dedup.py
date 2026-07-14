from backend.ingestion.dedup import (
    compute_content_hash,
    compute_simhash,
    find_near_duplicate,
    hamming_distance,
    is_near_duplicate,
)


def test_content_hash_is_deterministic():
    text = "Overdraft policy version 1."
    assert compute_content_hash(text) == compute_content_hash(text)


def test_content_hash_differs_for_different_text():
    assert compute_content_hash("Policy A") != compute_content_hash("Policy B")


def test_content_hash_is_sha256_hex_digest():
    digest = compute_content_hash("anything")
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_simhash_identical_text_has_zero_distance():
    text = (
        "The overdraft limit is five hundred dollars per month for standard accounts."
    )
    a = compute_simhash(text)
    b = compute_simhash(text)
    assert hamming_distance(a, b) == 0
    assert is_near_duplicate(a, b)


def test_simhash_near_duplicate_small_edit_is_close():
    base = (
        "The overdraft limit is five hundred dollars per month for standard "
        "checking accounts opened after January of this year."
    )
    edited = (
        "The overdraft limit is five hundred dollars per month for standard "
        "checking accounts opened after February of this year."
    )
    a = compute_simhash(base)
    b = compute_simhash(edited)
    assert hamming_distance(a, b) <= 3


def test_simhash_unrelated_documents_are_far_apart():
    doc_a = "Interest rate policy for savings accounts and certificates of deposit."
    doc_b = "Employee onboarding checklist for new branch staff members."
    a = compute_simhash(doc_a)
    b = compute_simhash(doc_b)
    assert not is_near_duplicate(a, b)


def test_simhash_empty_text_is_zero():
    assert compute_simhash("") == 0


def test_find_near_duplicate_returns_closest_match_within_threshold():
    base = (
        "The overdraft limit is five hundred dollars per month for standard "
        "checking accounts opened after January of this year."
    )
    edited = (
        "The overdraft limit is five hundred dollars per month for standard "
        "checking accounts opened after February of this year."
    )
    target = compute_simhash(base)
    unrelated = compute_simhash("Vacation request process for part-time employees.")
    near = compute_simhash(edited)

    candidates = [("doc-unrelated", unrelated), ("doc-near", near)]
    result = find_near_duplicate(target, candidates, threshold=3)
    assert result == "doc-near"


def test_find_near_duplicate_returns_none_when_no_candidate_close_enough():
    target = compute_simhash(
        "A very specific and unique compliance clause about wire transfers."
    )
    candidates = [
        ("doc-a", compute_simhash("Completely unrelated content about parking policy."))
    ]
    assert find_near_duplicate(target, candidates, threshold=3) is None


def test_find_near_duplicate_with_no_candidates_returns_none():
    target = compute_simhash("Some text")
    assert find_near_duplicate(target, [], threshold=3) is None
