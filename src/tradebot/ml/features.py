"""Feature engineering for the price-direction model.

All features at bar t use data up to and including bar t's close. Higher-timeframe (1h)
features are resampled from the trade timeframe and shifted so that only completed 1h
candles are visible — identical code runs in training and in live trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot import indicators as ind

LAGS = (1, 2, 4, 8, 16, 32, 96)


def _block(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close, vol = df["close"], df["volume"]
    logret = np.log(close).diff()
    f: dict[str, pd.Series] = {}
    for lag in LAGS:
        f[f"ret_{lag}"] = pd.Series(np.log(close / close.shift(lag)), index=df.index)
    for n in (16, 96):
        f[f"vol_{n}"] = logret.rolling(n, min_periods=n).std()
    f["rsi_14"] = ind.rsi(close, 14) / 100 - 0.5
    for fast, slow in ((12, 48), (48, 192)):
        f[f"ema_{fast}_{slow}"] = ind.ema(close, fast) / ind.ema(close, slow) - 1
    lower, mid, upper = ind.bollinger(close, 20, 2.0)
    f["bb_pos"] = (close - mid) / (upper - lower).replace(0, np.nan)
    f["atr_pct"] = ind.atr(df, 14) / close
    f["range_pct"] = (df["high"] - df["low"]) / close
    logv = np.log1p(vol)
    f["vol_z"] = (logv - logv.rolling(96, min_periods=96).mean()) / logv.rolling(96, min_periods=96).std()
    out = pd.DataFrame(f, index=df.index)
    return out.add_prefix(prefix)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def build_features(df: pd.DataFrame, bar: str = "15min") -> pd.DataFrame:
    """Feature matrix aligned to ``df.index`` (the trade timeframe)."""
    base = _block(df, "")
    h1 = resample_ohlcv(df, "1h")
    # A 1h candle labelled t (covering [t, t+1h)) is complete at t+1h minus one base bar.
    h1_feat = _block(h1, "h1_")
    h1_feat.index = pd.DatetimeIndex(h1_feat.index) + (pd.Timedelta("1h") - pd.Timedelta(bar))
    h1_feat = h1_feat.reindex(df.index, method="ffill")
    hour = df.index.hour + df.index.minute / 60  # type: ignore[attr-defined]
    cal = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": df.index.day_of_week / 6,  # type: ignore[attr-defined]
    }, index=df.index)
    return pd.concat([base, h1_feat, cal], axis=1).replace([np.inf, -np.inf], np.nan)


def make_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if the close ``horizon`` bars ahead is higher, 0 if lower, NaN at the end."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    return (fwd > 0).astype(float).where(fwd.notna())
