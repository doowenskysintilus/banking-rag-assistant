"""Field-level encryption at rest — AES-256-GCM (authenticated encryption:
tampering with ciphertext is detected on decrypt, not just confidentiality).

Versioned key format (`"<version>:<base64(nonce || ciphertext)>"`) so a key
can be rotated without a flag-day migration: writes always use
`Settings.encryption_key` (`"v1"`); if a second key is ever introduced, add it
to `_load_keys` and existing `"v1"`-tagged ciphertext keeps decrypting under
the retired key while new writes move to the new version. Only one key is
configured today — the format just avoids a breaking change later.

Apply `EncryptedString` to any SQLAlchemy column holding sensitive data that
must be unreadable to anyone with raw DB access (currently `AuditLog.detail` —
see that model for why encryption is applied on top of the PII redaction
already done before a value ever reaches this layer).
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

from backend.core.config import get_settings

_NONCE_BYTES = 12
_KEY_BYTES = 32  # AES-256
_CURRENT_VERSION = "v1"


class EncryptionError(Exception):
    """Raised when a ciphertext is malformed or fails authentication (either a
    corrupted/tampered value, or no matching key is configured for its
    version tag).
    """


def generate_key() -> str:
    """Generates a new urlsafe-base64-encoded 32-byte key — use this to produce
    a value for the `ENCRYPTION_KEY` environment variable. Run once per
    environment and keep it out of version control, exactly like `SECRET_KEY`.
    """
    return base64.urlsafe_b64encode(os.urandom(_KEY_BYTES)).decode("ascii")


def _decode_key(raw: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised with a clearer, actionable message
        raise ValueError(
            "ENCRYPTION_KEY is not valid urlsafe-base64 — see encryption.generate_key()."
        ) from exc
    if len(key) != _KEY_BYTES:
        raise ValueError(
            f"ENCRYPTION_KEY must decode to exactly {_KEY_BYTES} bytes (AES-256), got {len(key)}."
        )
    return key


def _derive_key_from_secret(secret: str) -> bytes:
    """Fallback used when `ENCRYPTION_KEY` isn't configured: derives a 32-byte
    key from `SECRET_KEY` via HKDF-SHA256, so field encryption works out of
    the box in dev/test without a second secret to manage. A real deployment
    should still set its own independent `ENCRYPTION_KEY` (see
    `generate_key()`) — otherwise rotating the JWT signing secret silently
    rotates the encryption key too and strands every previously encrypted
    value (it can no longer be decrypted).
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=None,
        info=b"banking-rag-assistant/field-encryption/v1",
    )
    return hkdf.derive(secret.encode("utf-8"))


class FieldEncryptor:
    def __init__(self, keys: dict[str, bytes], current_version: str) -> None:
        if current_version not in keys:
            raise ValueError(
                f"current_version {current_version!r} has no matching entry in `keys`"
            )
        self._keys = keys
        self._current_version = current_version

    def encrypt(self, plaintext: str) -> str:
        key = self._keys[self._current_version]
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
        payload = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return f"{self._current_version}:{payload}"

    def decrypt(self, token: str) -> str:
        version, sep, payload = token.partition(":")
        if not sep:
            raise EncryptionError("Malformed ciphertext: missing key-version prefix.")
        key = self._keys.get(version)
        if key is None:
            raise EncryptionError(
                f"No encryption key configured for version {version!r}."
            )

        try:
            raw = base64.urlsafe_b64decode(payload)
            nonce, ciphertext = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise EncryptionError(
                "Ciphertext failed authentication — it was corrupted or tampered with."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - malformed base64/nonce-length is also a decrypt failure
            raise EncryptionError("Ciphertext could not be decoded.") from exc
        return plaintext.decode("utf-8")


_encryptor_singleton: FieldEncryptor | None = None


def get_field_encryptor() -> FieldEncryptor:
    global _encryptor_singleton
    if _encryptor_singleton is None:
        settings = get_settings()
        key = (
            _decode_key(settings.encryption_key)
            if settings.encryption_key
            else _derive_key_from_secret(settings.secret_key)
        )
        _encryptor_singleton = FieldEncryptor(
            keys={_CURRENT_VERSION: key}, current_version=_CURRENT_VERSION
        )
    return _encryptor_singleton


def reset_field_encryptor_cache() -> None:
    """Test-only: forces `get_field_encryptor` to rebuild from current settings
    (e.g. after monkeypatching `ENCRYPTION_KEY` mid-test-session).
    """
    global _encryptor_singleton
    _encryptor_singleton = None


class EncryptedString(TypeDecorator):
    """Transparent encrypt-on-write / decrypt-on-read column type. Always backed
    by `Text` regardless of the logical field's length — ciphertext (nonce +
    tag + base64 overhead + version prefix) is meaningfully longer than the
    plaintext, so a bounded `String(n)` would risk silent truncation.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect) -> str | None:  # noqa: ANN001 - SQLAlchemy dialect param
        if value is None:
            return None
        return get_field_encryptor().encrypt(value)

    def process_result_value(self, value: str | None, dialect) -> str | None:  # noqa: ANN001
        if value is None:
            return None
        return get_field_encryptor().decrypt(value)
