"""Stable LLLM network errors."""

from __future__ import annotations


class NetworkError(RuntimeError):
    """Base LLLM network error."""


class ChannelExistsError(NetworkError):
    """Raised when creating a duplicate channel."""


class ChannelNotFoundError(NetworkError):
    """Raised when a channel does not exist."""


class EventNotFoundError(NetworkError):
    """Raised when an event does not exist."""


class InvalidPayloadError(NetworkError):
    """Raised when payloads, cursors, or request values are invalid."""


class ArtifactNotFoundError(NetworkError):
    """Raised when an artifact does not exist."""


class SnapshotNotFoundError(NetworkError):
    """Raised when a snapshot does not exist."""


class SubscriptionNotFoundError(NetworkError):
    """Raised when a subscription does not exist."""


class SubscriptionExistsError(NetworkError):
    """Raised when creating a conflicting duplicate subscription."""
