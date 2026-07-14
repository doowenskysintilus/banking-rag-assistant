from backend.retrieval.vector_store.filters import matches


def test_empty_filter_always_matches():
    assert matches({"owner_id": "u-1"}, None) is True
    assert matches({"owner_id": "u-1"}, {}) is True


def test_equality_shorthand():
    assert matches({"owner_id": "u-1"}, {"owner_id": "u-1"}) is True
    assert matches({"owner_id": "u-1"}, {"owner_id": "u-2"}) is False


def test_missing_field_does_not_match_equality():
    assert matches({}, {"owner_id": "u-1"}) is False


def test_multiple_keys_are_and_ed():
    metadata = {"owner_id": "u-1", "language": "en"}
    assert matches(metadata, {"owner_id": "u-1", "language": "en"}) is True
    assert matches(metadata, {"owner_id": "u-1", "language": "fr"}) is False


class TestOperators:
    def test_eq(self):
        assert matches({"score": 5}, {"score": {"$eq": 5}}) is True
        assert matches({"score": 5}, {"score": {"$eq": 6}}) is False

    def test_ne(self):
        assert matches({"owner_id": "u-1"}, {"owner_id": {"$ne": "u-2"}}) is True
        assert matches({"owner_id": "u-1"}, {"owner_id": {"$ne": "u-1"}}) is False

    def test_in(self):
        assert matches({"language": "fr"}, {"language": {"$in": ["en", "fr"]}}) is True
        assert matches({"language": "de"}, {"language": {"$in": ["en", "fr"]}}) is False

    def test_nin(self):
        assert matches({"language": "de"}, {"language": {"$nin": ["en", "fr"]}}) is True
        assert (
            matches({"language": "fr"}, {"language": {"$nin": ["en", "fr"]}}) is False
        )

    def test_range_operators(self):
        assert matches({"version": 3}, {"version": {"$gte": 2, "$lte": 5}}) is True
        assert matches({"version": 1}, {"version": {"$gte": 2, "$lte": 5}}) is False
        assert matches({"version": 6}, {"version": {"$gte": 2, "$lte": 5}}) is False

    def test_gt_lt_strict(self):
        assert matches({"version": 3}, {"version": {"$gt": 2}}) is True
        assert matches({"version": 2}, {"version": {"$gt": 2}}) is False
        assert matches({"version": 1}, {"version": {"$lt": 2}}) is True

    def test_comparison_on_missing_field_never_matches(self):
        assert matches({}, {"version": {"$gte": 1}}) is False


class TestLogicalOperators:
    def test_and(self):
        filter_ = {"$and": [{"owner_id": "u-1"}, {"language": "en"}]}
        assert matches({"owner_id": "u-1", "language": "en"}, filter_) is True
        assert matches({"owner_id": "u-1", "language": "fr"}, filter_) is False

    def test_or(self):
        filter_ = {"$or": [{"language": "en"}, {"language": "fr"}]}
        assert matches({"language": "fr"}, filter_) is True
        assert matches({"language": "de"}, filter_) is False

    def test_nested_and_or(self):
        filter_ = {
            "$and": [
                {"owner_id": "u-1"},
                {"$or": [{"language": "en"}, {"language": "fr"}]},
            ]
        }
        assert matches({"owner_id": "u-1", "language": "fr"}, filter_) is True
        assert matches({"owner_id": "u-1", "language": "de"}, filter_) is False
        assert matches({"owner_id": "u-2", "language": "fr"}, filter_) is False
