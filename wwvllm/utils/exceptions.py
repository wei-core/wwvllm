"""Errors raised when engine resources cannot satisfy a request."""


class KVBlockExhaustedError(RuntimeError):
    """No available KV blocks can make the current request progress."""


class KVCacheMemoryError(RuntimeError):
    """GPU memory cannot accommodate the requested KV cache."""
