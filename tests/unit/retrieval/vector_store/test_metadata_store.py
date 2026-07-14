from backend.retrieval.vector_store.metadata_store import MetadataStore


def test_new_store_is_empty(tmp_path):
    store = MetadataStore(tmp_path / "metadata.json")
    assert len(store) == 0
    assert store.known_ids() == []
    assert store.metadata_of(1) == {}
    assert store.text_of(1) is None


def test_set_then_get(tmp_path):
    store = MetadataStore(tmp_path / "metadata.json")
    store.set(1, {"owner_id": "u-1"}, "some text")

    assert store.metadata_of(1) == {"owner_id": "u-1"}
    assert store.text_of(1) == "some text"
    assert store.get(1) == {"metadata": {"owner_id": "u-1"}, "text": "some text"}


def test_delete_removes_the_record(tmp_path):
    store = MetadataStore(tmp_path / "metadata.json")
    store.set(1, {"owner_id": "u-1"}, "text")
    store.delete(1)

    assert store.get(1) is None
    assert store.metadata_of(1) == {}
    assert 1 not in store.known_ids()


def test_delete_unknown_id_is_a_no_op(tmp_path):
    store = MetadataStore(tmp_path / "metadata.json")
    store.delete(999)  # must not raise


def test_save_and_reload_roundtrips(tmp_path):
    path = tmp_path / "metadata.json"
    store = MetadataStore(path)
    store.set(1, {"owner_id": "u-1", "language": "en"}, "hello world")
    store.set(2, {"owner_id": "u-2"}, None)
    store.save()

    assert path.exists()

    reloaded = MetadataStore(path)
    assert reloaded.metadata_of(1) == {"owner_id": "u-1", "language": "en"}
    assert reloaded.text_of(1) == "hello world"
    assert reloaded.metadata_of(2) == {"owner_id": "u-2"}
    assert reloaded.text_of(2) is None
    assert set(reloaded.known_ids()) == {1, 2}


def test_save_does_not_leave_a_temp_file_behind(tmp_path):
    path = tmp_path / "metadata.json"
    store = MetadataStore(path)
    store.set(1, {}, None)
    store.save()

    leftover_tmp_files = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp_files == []


def test_loading_a_missing_file_starts_empty(tmp_path):
    store = MetadataStore(tmp_path / "does_not_exist.json")
    assert len(store) == 0
