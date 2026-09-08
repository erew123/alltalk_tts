"""Shared helpers for the per-model backends."""

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "pcm": "audio/L16",
}


def content_type_for(response_format: str) -> str:
    return CONTENT_TYPES.get((response_format or "wav").lower(), "application/octet-stream")


def require(payload: dict, *keys: str) -> None:
    missing = [k for k in keys if not payload.get(k)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
