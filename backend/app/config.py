import os


def load_jwt_secret_key() -> str:
    value = os.getenv("JWT_SECRET_KEY", "")
    if (
        not value.strip()
        or value.strip() == "replace_with_a_random_secret"
        or len(value.encode("utf-8")) < 32
    ):
        raise RuntimeError(
            "JWT_SECRET_KEY must be a non-placeholder secret of at least 32 bytes. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return value


# Validate once at import/startup; signing and verification share the same key.
JWT_SECRET_KEY = load_jwt_secret_key()
