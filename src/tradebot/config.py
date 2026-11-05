"""Typed configuration loaded from YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
LOCAL_CONFIG = ROOT / "config" / "local.yaml"


class VenueConfig(BaseModel):
    exchange_id: str = "bybit"


class MarketsConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "1h"])
    trade_timeframe: str = "15m"
    history_start: str = "2020-01-01"


class CostsConfig(BaseModel):
    fee_rate: float = Field(0.001, ge=0)
    slippage_rate: float = Field(0.0005, ge=0)


class BacktestConfig(BaseModel):
    initial_equity: float = Field(10_000.0, gt=0)


class RiskParams(BaseModel):
    """Learnable risk parameters; clipped to hard caps by the RiskManager."""

    position_frac: float = Field(0.10, gt=0)
    leverage: float = Field(1.0, gt=0)
    stop_atr_mult: float | None = Field(3.0, gt=0)
    take_atr_mult: float | None = Field(6.0, gt=0)
    atr_period: int = Field(14, ge=2)


class NewsConfig(BaseModel):
    """Free LLM models and limits for the news risk filter (see tradebot.news)."""

    gemini_model: str = "gemini-3.1-flash-lite"
    groq_model: str = "openai/gpt-oss-120b"
    max_items_per_run: int = Field(40, gt=0)
    max_items_per_day: int = Field(500, gt=0)
    telegram_channels: list[str] = Field(default_factory=lambda: ["WatcherGuru", "bbcbreaking"])


class Settings(BaseModel):
    venue: VenueConfig = VenueConfig()
    markets: MarketsConfig = MarketsConfig()
    costs: CostsConfig = CostsConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskParams = RiskParams()
    news: NewsConfig = NewsConfig()
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT / p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(*paths: Path) -> Settings:
    """Load default.yaml, then local.yaml (if present), then any extra paths.

    ``EXCHANGE_ID`` in the environment overrides ``venue.exchange_id``."""
    data: dict[str, Any] = {}
    for path in (DEFAULT_CONFIG, LOCAL_CONFIG, *paths):
        if path.exists():
            data = _deep_merge(data, yaml.safe_load(path.read_text()) or {})
    if os.environ.get("EXCHANGE_ID"):
        data = _deep_merge(data, {"venue": {"exchange_id": os.environ["EXCHANGE_ID"]}})
    return Settings.model_validate(data)
