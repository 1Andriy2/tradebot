"""Persistence: Supabase (production) and an in-memory store (tests, dry runs)."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Protocol, cast

import pandas as pd


def _clean(value: Any) -> Any:
    """Make values JSON-safe for PostgREST (timestamps -> ISO, NaN/inf -> None)."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


class Store(Protocol):
    def insert(self, table: str, rows: list[dict[str, Any]]) -> None: ...
    def upsert(self, table: str, rows: list[dict[str, Any]], on_conflict: str) -> None: ...
    def update(self, table: str, values: dict[str, Any], filters: dict[str, Any]) -> None: ...
    def select(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def upload(self, bucket: str, path: str, data: bytes) -> None: ...
    def download(self, bucket: str, path: str) -> bytes: ...


def _roundtrip(row: dict[str, Any]) -> dict[str, Any]:
    """JSON round trip so in-memory rows look exactly like rows read back from Postgres."""
    return cast(dict[str, Any], json.loads(json.dumps(_clean(row))))


class MemoryStore:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.files: dict[str, bytes] = {}

    def upload(self, bucket: str, path: str, data: bytes) -> None:
        self.files[f"{bucket}/{path}"] = data

    def download(self, bucket: str, path: str) -> bytes:
        return self.files[f"{bucket}/{path}"]

    def insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        self.tables[table].extend(_roundtrip(r) for r in rows)

    def upsert(self, table: str, rows: list[dict[str, Any]], on_conflict: str) -> None:
        keys = on_conflict.split(",")
        for row in (_roundtrip(r) for r in rows):
            existing = self.tables[table]
            for i, old in enumerate(existing):
                if all(old.get(k) == row.get(k) for k in keys):
                    existing[i] = {**old, **row}
                    break
            else:
                existing.append(row)

    def update(self, table: str, values: dict[str, Any], filters: dict[str, Any]) -> None:
        rows = self.tables[table]
        for i, row in enumerate(rows):
            if all(row.get(k) == v for k, v in filters.items()):
                rows[i] = {**row, **_roundtrip(values)}

    def select(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        return [r for r in self.tables[table] if all(r.get(k) == v for k, v in filters.items())]


class SupabaseStore:
    """Writes with the service-role key. Never use this key in the dashboard."""

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client

        self.client = create_client(url, key)

    @classmethod
    def from_env(cls) -> SupabaseStore:
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        return cls(url, key)

    def insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.client.table(table).insert([_clean(r) for r in rows]).execute()

    def upsert(self, table: str, rows: list[dict[str, Any]], on_conflict: str) -> None:
        if rows:
            self.client.table(table).upsert([_clean(r) for r in rows], on_conflict=on_conflict).execute()

    def update(self, table: str, values: dict[str, Any], filters: dict[str, Any]) -> None:
        q = self.client.table(table).update(_clean(values))
        for k, v in filters.items():
            q = q.eq(k, v)
        q.execute()

    def select(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q = self.client.table(table).select("*")
        for k, v in (filters or {}).items():
            q = q.eq(k, v)
        return cast(list[dict[str, Any]], q.execute().data)

    def upload(self, bucket: str, path: str, data: bytes) -> None:
        storage = self.client.storage
        if not any(b.name == bucket for b in storage.list_buckets()):
            storage.create_bucket(bucket, options={"public": False})
        storage.from_(bucket).upload(path, data, {"content-type": "application/json", "upsert": "true"})

    def download(self, bucket: str, path: str) -> bytes:
        return bytes(self.client.storage.from_(bucket).download(path))
