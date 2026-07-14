import base64

import pytest

from backend.security.encryption import (
    EncryptedString,
    EncryptionError,
    FieldEncryptor,
    _derive_key_from_secret,
    generate_key,
    get_field_encryptor,
    reset_field_encryptor_cache,
)


def _key() -> bytes:
    return base64.urlsafe_b64decode(generate_key())


class TestFieldEncryptor:
    def test_round_trips_plaintext(self):
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        token = encryptor.encrypt("sensitive banking detail")
        assert encryptor.decrypt(token) == "sensitive banking detail"

    def test_ciphertext_is_prefixed_with_the_key_version(self):
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        token = encryptor.encrypt("x")
        assert token.startswith("v1:")

    def test_two_encryptions_of_the_same_plaintext_differ(self):
        # Random nonce per call — ciphertext must not be deterministic.
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        assert encryptor.encrypt("same") != encryptor.encrypt("same")

    def test_tampered_ciphertext_fails_to_decrypt(self):
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        token = encryptor.encrypt("original")
        version, payload = token.split(":", 1)
        tampered_payload = ("A" if payload[0] != "A" else "B") + payload[1:]
        with pytest.raises(EncryptionError):
            encryptor.decrypt(f"{version}:{tampered_payload}")

    def test_unknown_key_version_raises(self):
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        with pytest.raises(EncryptionError):
            encryptor.decrypt("v99:whatever")

    def test_malformed_token_without_version_prefix_raises(self):
        encryptor = FieldEncryptor(keys={"v1": _key()}, current_version="v1")
        with pytest.raises(EncryptionError):
            encryptor.decrypt("not-a-valid-token")

    def test_rotation_old_key_still_decrypts_while_new_writes_use_current(self):
        old_key, new_key = _key(), _key()
        old_encryptor = FieldEncryptor(keys={"v1": old_key}, current_version="v1")
        old_token = old_encryptor.encrypt("archived value")

        rotated = FieldEncryptor(
            keys={"v1": old_key, "v2": new_key}, current_version="v2"
        )
        assert rotated.decrypt(old_token) == "archived value"
        assert rotated.encrypt("new value").startswith("v2:")

    def test_current_version_must_have_a_matching_key(self):
        with pytest.raises(ValueError):
            FieldEncryptor(keys={"v1": _key()}, current_version="v2")


class TestKeyDerivationFallback:
    def test_derives_a_32_byte_key(self):
        key = _derive_key_from_secret("some-secret-key-value")
        assert len(key) == 32

    def test_is_deterministic_for_the_same_secret(self):
        assert _derive_key_from_secret("same-secret") == _derive_key_from_secret(
            "same-secret"
        )

    def test_differs_for_different_secrets(self):
        assert _derive_key_from_secret("secret-a") != _derive_key_from_secret(
            "secret-b"
        )


class TestGetFieldEncryptorSingleton:
    def test_falls_back_to_deriving_from_secret_key_when_encryption_key_unset(
        self, monkeypatch
    ):
        from backend.core.config import Settings

        reset_field_encryptor_cache()
        fake_settings = Settings(secret_key="x" * 16, encryption_key="")
        monkeypatch.setattr(
            "backend.security.encryption.get_settings", lambda: fake_settings
        )

        encryptor = get_field_encryptor()
        token = encryptor.encrypt("hello")
        assert encryptor.decrypt(token) == "hello"
        reset_field_encryptor_cache()

    def test_cached_across_calls(self, monkeypatch):
        from backend.core.config import Settings

        reset_field_encryptor_cache()
        fake_settings = Settings(secret_key="y" * 16, encryption_key="")
        calls = {"n": 0}

        def _get_settings():
            calls["n"] += 1
            return fake_settings

        monkeypatch.setattr("backend.security.encryption.get_settings", _get_settings)
        get_field_encryptor()
        get_field_encryptor()
        assert calls["n"] == 1
        reset_field_encryptor_cache()


class TestEncryptedStringTypeDecorator:
    def test_process_bind_and_result_round_trip(self, monkeypatch):
        from backend.core.config import Settings

        reset_field_encryptor_cache()
        monkeypatch.setattr(
            "backend.security.encryption.get_settings",
            lambda: Settings(secret_key="z" * 16, encryption_key=""),
        )

        column_type = EncryptedString()
        bound = column_type.process_bind_param("plaintext value", dialect=None)
        assert bound != "plaintext value"
        assert (
            column_type.process_result_value(bound, dialect=None) == "plaintext value"
        )
        reset_field_encryptor_cache()

    def test_none_passes_through_unchanged(self):
        column_type = EncryptedString()
        assert column_type.process_bind_param(None, dialect=None) is None
        assert column_type.process_result_value(None, dialect=None) is None
