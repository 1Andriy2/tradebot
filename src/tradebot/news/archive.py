"""Keeps the files the 15-minute jobs download small.

Supabase's free plan allows 5 GB of egress a month, and these files are fetched about 3000 times a
month, so a file that only grows would use that up within weeks. Rows older than ``HOT`` move to a
monthly archive about once a day. Only the nightly research reads the archives.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

import pandas as pd

from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "data"
HOT = pd.Timedelta("2D")  # as long as a run needs: 48h of story matching, 24h of levels
ROTATE_AFTER = pd.Timedelta("3D")  # so a rotation happens daily, not on every run
FIRST_MONTH = pd.Period("2026-11", "M")


def is_missing(exc: Exception) -> bool:
    """True for a file that does not exist yet, which is not the same as a failed download."""
    return isinstance(exc, KeyError) or str(getattr(exc, "status", "")) == "404"


def read_file(store: Store, name: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(io.BytesIO(store.download(BUCKET, name)))
    except Exception as exc:
        if is_missing(exc):
            return None
        raise


def write_file(store: Store, name: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    store.upload(BUCKET, name, buffer.getvalue())


def archive_name(stem: str, month: pd.Period) -> str:
    return f"{stem}_{month}.parquet"


def rotate(store: Store, stem: str, frame: pd.DataFrame, column: str, now: pd.Timestamp,
           worth_keeping: Callable[[pd.DataFrame], pd.DataFrame] = lambda rows: rows,
           together: str | None = None) -> pd.DataFrame:
    """Move old rows to monthly archives and return the rest.

    Rows sharing a ``together`` value move as one, when the newest of them is old: a story's age
    comes from its first report, which must not leave while a repeat of it stays.
    The archive is written before anything leaves ``frame``, so a failure loses nothing: the rows
    simply wait for the next run. Writing the same rows twice is harmless, they are de-duplicated.
    """
    if frame.empty or frame[column].min() >= now - ROTATE_AFTER:
        return frame
    newest = frame.groupby(together)[column].transform("max") if together else frame[column]
    old = newest < now - HOT
    try:
        kept = worth_keeping(frame[old])
        months = kept[column].dt.tz_localize(None).dt.to_period("M")
        for month, rows in kept.groupby(months):
            name = archive_name(stem, month)
            existing = read_file(store, name)
            merged = rows if existing is None or existing.empty else pd.concat([existing, rows])
            write_file(store, name, merged.drop_duplicates().reset_index(drop=True))
    except Exception as exc:
        log.warning("could not archive %s: %s", stem, type(exc).__name__)
        return frame
    return frame[~old].reset_index(drop=True)


def history(store: Store, stem: str, hot_name: str, now: pd.Timestamp) -> pd.DataFrame:
    """Archives and the hot file together, for research. Nothing here is written back."""
    months = pd.period_range(FIRST_MONTH, now.tz_localize(None).to_period("M"))
    names = [archive_name(stem, m) for m in months]
    parts = [read_file(store, name) for name in [*names, hot_name]]
    found = [p for p in parts if p is not None and not p.empty]
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame()
