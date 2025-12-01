"""Utility helpers for optional glove dependency."""


def ensure_glove():
    """Return Glove class or raise a helpful error when unavailable."""
    try:
        from glove import Glove  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - informative error path
        raise ImportError(
            "glove-python-binary is required when using the BiLSTM encoder. "
            "Install it manually (Python<3.9 or build from source) or switch "
            "to the default BERT encoder."
        ) from exc
    return Glove
