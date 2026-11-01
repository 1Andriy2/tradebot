"""Recurring-pattern scan: hour-of-day and day-of-week effects in returns.

Each bucket gets a t-statistic and p-value; p-values are corrected for testing many buckets
(Benjamini-Hochberg q-values). A pattern counts as *stable* only if it has the same sign and
is individually significant (p < 0.05) in both halves of the history.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats


def bh_qvalues(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _bucket_stats(ret: pd.Series, key: pd.Index) -> pd.DataFrame:
    g = ret.groupby(key).agg(["mean", "std", "count"])
    g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"]))
    g["p"] = 2 * stats.norm.sf(g["t"].abs())
    return g


def scan_seasonality(df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
    """``df``: hourly candles. Returns one row per (kind, bucket)."""
    ret = df["close"].pct_change().dropna()
    idx = pd.DatetimeIndex(ret.index)
    half = len(ret) // 2
    rows: list[dict[str, Any]] = []
    for kind, key in (("hour", idx.hour), ("dow", idx.day_of_week)):
        full = _bucket_stats(ret, pd.Index(key))
        a = _bucket_stats(ret.iloc[:half], pd.Index(key[:half]))
        b = _bucket_stats(ret.iloc[half:], pd.Index(key[half:]))
        q = bh_qvalues(full["p"].to_numpy(float))
        for bucket, qv in zip(full.index.astype(int), q, strict=True):
            mean = float(full["mean"][bucket])
            stable = bool(
                bucket in a.index and bucket in b.index
                and np.sign(a["mean"][bucket]) == np.sign(b["mean"][bucket]) == np.sign(mean)
                and a["p"][bucket] < 0.05 and b["p"][bucket] < 0.05
            )
            rows.append({"symbol": symbol, "kind": kind, "bucket": int(bucket),
                         "mean_return": mean, "t_stat": float(full["t"][bucket]),
                         "p_value": float(full["p"][bucket]), "q_value": float(qv),
                         "n": int(full["count"][bucket]), "stable": stable})
    return rows
