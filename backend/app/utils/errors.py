"""Translate internal exceptions into client-safe API messages."""

from typing import Optional

# Only messages raised deliberately by request-validation paths may reach the
# client. Anything else (decode failures with absolute paths, DB errors, ...)
# collapses to a generic message and stays in the logs.
SAFE_ERROR_PREFIXES = (
    "File exceeds the",
    "File content does not match the",
    "Unsupported file type",
    "Unsupported image type",
    "Uploaded image could not be decoded",
    "Video prediction is not supported yet",
    "Prediction not found",
)
MAX_SAFE_LENGTH = 200


def public_error_message(exc: BaseException) -> Optional[str]:
    message = str(exc)
    if not message or len(message) > MAX_SAFE_LENGTH:
        return None
    if not message.startswith(SAFE_ERROR_PREFIXES):
        return None
    return message