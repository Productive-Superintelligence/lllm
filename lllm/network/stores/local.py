"""SQLite/filesystem local network store."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..core import (
    Artifact,
    ArtifactNotFoundError,
    Channel,
    ChannelExistsError,
    ChannelNotFoundError,
    EventExistsError,
    EventNotFoundError,
    Event,
    InvalidPayloadError,
    Snapshot,
    SnapshotNotFoundError,
    Subscription,
    SubscriptionExistsError,
    SubscriptionNotFoundError,
)
from ...protocol._validation import copy_boundary_value

ModelT = TypeVar("ModelT", bound=BaseModel)


class LocalStore:
    """Boring local SQLite/filesystem backend."""

    def __init__(self, root: str | Path = ".lllm/network") -> None:
        self.root = Path(_path_value(root, "store root")).expanduser().resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.db_path = self.root / "network.sqlite"
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_channel(self, channel: Channel | dict[str, Any]) -> Channel:
        if isinstance(channel, Mapping) and "metadata" in channel:
            channel = {
                **channel,
                "metadata": _optional_mapping("channel.metadata", channel["metadata"]),
            }
        value = _model(Channel, channel, "channel")
        try:
            with self._connect() as db:
                db.execute(
                    """
                    insert into channels(name, schema_ref, form, description, metadata)
                    values (?, ?, ?, ?, ?)
                    """,
                    (
                        value.name,
                        value.schema,
                        value.form,
                        value.description,
                        _json(value.metadata),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ChannelExistsError(f"Channel already exists: {value.name}") from exc
        return value.model_copy(deep=True)

    def list_channels(self) -> tuple[Channel, ...]:
        with self._connect() as db:
            rows = db.execute(
                "select name, schema_ref, form, description, metadata from channels order by name"
            ).fetchall()
        return tuple(_channel(row) for row in rows)

    def get_channel(self, name: str) -> Channel:
        _require_segment("channel.name", name)
        with self._connect() as db:
            row = db.execute(
                "select name, schema_ref, form, description, metadata from channels where name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise ChannelNotFoundError(f"Channel not found: {name}")
        return _channel(row)

    def append_event(self, event: Event | dict[str, Any]) -> Event:
        if isinstance(event, Mapping) and "metadata" in event:
            event = {
                **event,
                "metadata": _optional_mapping("event.metadata", event["metadata"]),
            }
        value = _model(Event, event, "event")
        self.get_channel(value.channel)
        self._require_events(value.parent_ids)
        try:
            with self._connect() as db:
                db.execute(
                    """
                    insert into events(
                      id, channel, timestamp, source, kind, payload, schema_ref,
                      metadata, correlation_id, parent_ids
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        value.id,
                        value.channel,
                        value.timestamp,
                        value.source,
                        value.kind,
                        _json(value.payload),
                        value.schema,
                        _json(value.metadata),
                        value.correlation_id,
                        _json(list(value.parent_ids)),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if self._event_cursor(value.id):
                raise EventExistsError(f"Event already exists: {value.id}") from exc
            raise
        return value.model_copy(
            update={"cursor": self._event_cursor(value.id)},
            deep=True,
        )

    def query_events(
        self,
        channel: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
        kind: str | None = None,
    ) -> tuple[Event, ...]:
        after_cursor = _non_negative_int("after_cursor", after_cursor)
        limit = _positive_int("limit", limit)
        self.get_channel(channel)
        sql = """
            select rowid, id, channel, timestamp, source, kind, payload, schema_ref,
                   metadata, correlation_id, parent_ids
            from events
            where channel = ? and rowid > ?
        """
        args: list[Any] = [channel, after_cursor]
        if kind is not None:
            _require_token("event.kind", kind)
            sql += " and kind = ?"
            args.append(kind)
        sql += " order by rowid limit ?"
        args.append(limit)
        with self._connect() as db:
            rows = db.execute(sql, tuple(args)).fetchall()
        return tuple(_event(row) for row in rows)

    def get_event(self, event_id: str) -> Event:
        _require_segment("event.id", event_id)
        with self._connect() as db:
            row = db.execute(
                """
                select rowid, id, channel, timestamp, source, kind, payload, schema_ref,
                       metadata, correlation_id, parent_ids
                from events
                where id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            raise EventNotFoundError(f"Event not found: {event_id}")
        return _event(row)

    def create_subscription(
        self,
        channel: str,
        *,
        subscription_id: str | None = None,
        consumer: str | None = None,
        batch_size: int = 100,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Subscription:
        batch_size = _positive_int("batch_size", batch_size)
        self.get_channel(channel)
        subscription_filters = _optional_mapping("subscription.filters", filters)
        subscription_metadata = _optional_mapping("subscription.metadata", metadata)
        payload: dict[str, Any] = {
            "channel": channel,
            "consumer": consumer,
            "batch_size": batch_size,
            "filters": subscription_filters,
            "metadata": subscription_metadata,
        }
        if subscription_id is not None:
            payload["id"] = subscription_id
        sub = _model(Subscription, payload, "subscription")
        _subscription_kind(sub.filters)
        if subscription_id is not None:
            try:
                existing = self.get_subscription(sub.id)
            except SubscriptionNotFoundError:
                pass
            else:
                if existing.channel != channel:
                    raise SubscriptionExistsError(
                        "Subscription already exists for a different channel: "
                        f"{sub.id}"
                    )
                return existing
        with self._connect() as db:
            db.execute(
                """
                insert into subscriptions(id, channel, cursor, consumer, batch_size, filters, metadata)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sub.id,
                    sub.channel,
                    sub.cursor,
                    sub.consumer,
                    sub.batch_size,
                    _json(sub.filters),
                    _json(sub.metadata),
                ),
            )
        return sub.model_copy(deep=True)

    def pull_subscription(
        self,
        subscription_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[Event, ...]:
        if limit is not None:
            limit = _positive_int("limit", limit)
        sub = self.get_subscription(subscription_id)
        events = self.query_events(
            sub.channel,
            after_cursor=sub.cursor,
            limit=limit if limit is not None else sub.batch_size,
            kind=_subscription_kind(sub.filters),
        )
        if events:
            last_cursor = events[-1].cursor or self._event_cursor(events[-1].id)
            with self._connect() as db:
                db.execute(
                    "update subscriptions set cursor = ? where id = ?",
                    (last_cursor, subscription_id),
                )
        return events

    def get_subscription(self, subscription_id: str) -> Subscription:
        _require_segment("subscription.id", subscription_id)
        with self._connect() as db:
            row = db.execute(
                """
                select id, channel, cursor, consumer, batch_size, filters, metadata
                from subscriptions where id = ?
                """,
                (subscription_id,),
            ).fetchone()
        if row is None:
            raise SubscriptionNotFoundError(f"Subscription not found: {subscription_id}")
        return Subscription(
            id=row["id"],
            channel=row["channel"],
            cursor=row["cursor"],
            consumer=row["consumer"],
            batch_size=row["batch_size"],
            filters=_loads(row["filters"]),
            metadata=_loads(row["metadata"]),
        )

    def write_artifact(
        self,
        data: bytes | bytearray | memoryview,
        *,
        channel: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        event_ids: tuple[str, ...] = (),
    ) -> Artifact:
        payload = _artifact_bytes(data)
        artifact_metadata = _optional_mapping("artifact.metadata", metadata)
        artifact_event_ids = _segment_tuple("artifact.event_ids", event_ids)
        if channel is not None:
            self.get_channel(channel)
        self._require_events(artifact_event_ids)
        sha = hashlib.sha256(payload).hexdigest()
        artifact = Artifact(
            channel=channel,
            path=f"artifacts/{sha}",
            media_type=media_type,
            size=len(payload),
            sha256=sha,
            metadata=artifact_metadata,
            event_ids=artifact_event_ids,
        )
        path = self._artifact_payload_path(artifact.path)
        path.write_bytes(payload)
        with self._connect() as db:
            db.execute(
                """
                insert into artifacts(id, channel, path, media_type, size, sha256, metadata, event_ids)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.channel,
                    artifact.path,
                    artifact.media_type,
                    artifact.size,
                    artifact.sha256,
                    _json(artifact.metadata),
                    _json(list(artifact.event_ids)),
                ),
            )
        return artifact.model_copy(deep=True)

    def read_artifact(self, artifact_id: str) -> bytes:
        _require_segment("artifact.id", artifact_id)
        with self._connect() as db:
            row = db.execute(
                "select path from artifacts where id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}")
        path = self._artifact_payload_path(row["path"])
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact payload not found: {artifact_id}")
        return path.read_bytes()

    def get_artifact(self, artifact_id: str) -> Artifact:
        _require_segment("artifact.id", artifact_id)
        with self._connect() as db:
            row = db.execute(
                """
                select id, channel, path, media_type, size, sha256, metadata, event_ids
                from artifacts where id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}")
        return _artifact(row)

    def put_snapshot(
        self,
        snapshot: Snapshot | dict[str, Any] | str,
        value: Any = None,
        *,
        channel: str | None = None,
        timestamp: float | None = None,
        schema: str | None = None,
        source_event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Write a snapshot given as a model, a dict, or a name and value.

        ``put_snapshot(snapshot)`` takes a ``Snapshot`` or a dict of snapshot
        fields; ``put_snapshot(name, value, ...)`` stores ``value`` exactly as
        given, unless it is a ``Snapshot`` whose fields are used under that
        name. Field arguments that are not ``None`` override.
        """
        snapshot = _snapshot_fields(
            snapshot,
            value,
            channel=channel,
            timestamp=timestamp,
            schema=schema,
            source_event_id=source_event_id,
            metadata=metadata,
        )
        if isinstance(snapshot, Mapping) and "metadata" in snapshot:
            snapshot = {
                **snapshot,
                "metadata": _optional_mapping(
                    "snapshot.metadata",
                    snapshot["metadata"],
                ),
            }
        record = _model(Snapshot, snapshot, "snapshot")
        if record.channel is not None:
            self.get_channel(record.channel)
        if record.source_event_id is not None:
            self._require_events((record.source_event_id,))
        with self._connect() as db:
            db.execute(
                """
                insert into snapshots(name, channel, timestamp, value, schema_ref, source_event_id, metadata)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(name) do update set
                  channel = excluded.channel,
                  timestamp = excluded.timestamp,
                  value = excluded.value,
                  schema_ref = excluded.schema_ref,
                  source_event_id = excluded.source_event_id,
                  metadata = excluded.metadata
                """,
                (
                    record.name,
                    record.channel,
                    record.timestamp,
                    _json(record.value),
                    record.schema,
                    record.source_event_id,
                    _json(record.metadata),
                ),
            )
        return record.model_copy(deep=True)

    def get_snapshot(self, name: str) -> Snapshot:
        _require_segment("snapshot.name", name)
        with self._connect() as db:
            row = db.execute(
                """
                select name, channel, timestamp, value, schema_ref, source_event_id, metadata
                from snapshots where name = ?
                """,
                (name,),
            ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"Snapshot not found: {name}")
        return Snapshot(
            name=row["name"],
            channel=row["channel"],
            timestamp=row["timestamp"],
            value=_loads(row["value"]),
            schema=row["schema_ref"],
            source_event_id=row["source_event_id"],
            metadata=_loads(row["metadata"]),
        )

    def _event_cursor(self, event_id: str) -> int:
        with self._connect() as db:
            row = db.execute("select rowid from events where id = ?", (event_id,)).fetchone()
        if row is None:
            return 0
        return int(row["rowid"])

    def _require_events(self, event_ids: tuple[str, ...]) -> None:
        for event_id in event_ids:
            self.get_event(event_id)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def _artifact_payload_path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not _is_relative_to(target, self.artifacts_dir):
            raise ArtifactNotFoundError("Artifact payload path is outside artifact storage.")
        return target

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                create table if not exists channels (
                  name text primary key,
                  schema_ref text,
                  form text not null,
                  description text not null,
                  metadata text not null
                );
                create table if not exists events (
                  id text primary key,
                  channel text not null,
                  timestamp real not null,
                  source text,
                  kind text not null,
                  payload text,
                  schema_ref text,
                  metadata text not null,
                  correlation_id text,
                  parent_ids text not null
                );
                create table if not exists subscriptions (
                  id text primary key,
                  channel text not null,
                  cursor integer not null,
                  consumer text,
                  batch_size integer not null,
                  filters text not null,
                  metadata text not null
                );
                create table if not exists artifacts (
                  id text primary key,
                  channel text,
                  path text not null,
                  media_type text not null,
                  size integer not null,
                  sha256 text,
                  metadata text not null,
                  event_ids text not null
                );
                create table if not exists snapshots (
                  name text primary key,
                  channel text,
                  timestamp real not null,
                  value text,
                  schema_ref text,
                  source_event_id text,
                  metadata text not null
                );
                """
            )


def _channel(row: sqlite3.Row) -> Channel:
    return Channel(
        name=row["name"],
        schema=row["schema_ref"],
        form=row["form"],
        description=row["description"],
        metadata=_loads(row["metadata"]),
    )


def _event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        cursor=row["rowid"],
        channel=row["channel"],
        timestamp=row["timestamp"],
        source=row["source"],
        kind=row["kind"],
        payload=_loads(row["payload"]),
        schema=row["schema_ref"],
        metadata=_loads(row["metadata"]),
        correlation_id=row["correlation_id"],
        parent_ids=tuple(_loads(row["parent_ids"])),
    )


def _artifact(row: sqlite3.Row) -> Artifact:
    return Artifact(
        id=row["id"],
        channel=row["channel"],
        path=row["path"],
        media_type=row["media_type"],
        size=row["size"],
        sha256=row["sha256"],
        metadata=_loads(row["metadata"]),
        event_ids=tuple(_loads(row["event_ids"])),
    )


def _model(model_type: type[ModelT], value: ModelT | dict[str, Any], label: str) -> ModelT:
    if isinstance(value, model_type):
        return value.model_copy(deep=True)
    try:
        return model_type.model_validate(copy_boundary_value(value))
    except ValidationError as exc:
        raise InvalidPayloadError(f"Invalid {label}: {exc}") from exc


def _snapshot_fields(
    snapshot: Any,
    value: Any,
    *,
    channel: str | None,
    timestamp: float | None,
    schema: str | None,
    source_event_id: str | None,
    metadata: Any,
) -> Any:
    overrides = {
        key: item
        for key, item in (
            ("channel", channel),
            ("timestamp", timestamp),
            ("schema", schema),
            ("source_event_id", source_event_id),
            ("metadata", metadata),
        )
        if item is not None
    }
    if isinstance(snapshot, str):
        if isinstance(value, Snapshot):
            fields = value.model_dump(by_alias=True)
        else:
            fields = {"value": value}
        return {**fields, **overrides, "name": snapshot}
    if value is not None:
        overrides["value"] = value
    if not overrides:
        return snapshot
    if isinstance(snapshot, Snapshot):
        return {**snapshot.model_dump(by_alias=True), **overrides}
    if isinstance(snapshot, Mapping):
        return {**snapshot, **overrides}
    return snapshot


def _path_value(value: Any, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text


def _artifact_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise InvalidPayloadError("artifact data must be bytes.")


def _non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidPayloadError(f"{name} must be an integer.")
    if value < 0:
        raise InvalidPayloadError(f"{name} must be greater than or equal to 0.")
    return value


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidPayloadError(f"{name} must be an integer.")
    if value < 1:
        raise InvalidPayloadError(f"{name} must be greater than 0.")
    return value


def _require_segment(field_name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise InvalidPayloadError(f"{field_name} must be a non-empty path segment.")


def _require_token(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise InvalidPayloadError(f"{field_name} must be a non-empty token.")


def _segment_tuple(field_name: str, values: Any) -> tuple[str, ...]:
    if (
        values is None
        or isinstance(values, (str, bytes, bytearray))
        or isinstance(values, Mapping)
        or not isinstance(values, Iterable)
    ):
        raise InvalidPayloadError(f"{field_name} must be a list of path segments.")
    result: list[str] = []
    for value in values:
        _require_segment(field_name, value)
        result.append(value)
    return tuple(result)


def _optional_mapping(field_name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidPayloadError(f"{field_name} must be an object.")
    return copy_boundary_value(value)


def _is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
    except ValueError:
        return False
    return True


def _subscription_kind(filters: dict[str, Any]) -> str | None:
    unsupported = sorted(set(filters) - {"kind"})
    if unsupported:
        names = ", ".join(unsupported)
        raise InvalidPayloadError(
            f"subscription filters only support 'kind'; unsupported: {names}"
        )
    value = filters.get("kind")
    if value is None:
        return None
    _require_token("subscription filter 'kind'", value)
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(TypeAdapter(Any).dump_python(value, mode="json"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidPayloadError(f"Value is not JSON serializable: {exc}") from exc


def _loads(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)
