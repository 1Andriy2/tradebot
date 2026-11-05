# Architecture

```
                 ┌──────────── GitHub Actions (free, public repo) ────────────┐
 Bybit public ──▶│ trade.yml   every 15 min: tradebot step (one-shot)         │──▶ Bybit DEMO
 market data     │ retrain.yml nightly: LightGBM walk-forward + promotion     │    (api-demo.bybit.com,
                 │ rl-train.yml weekly: PPO (CPU, small budget)               │     allowlist-guarded)
                 └───────────────┬────────────────────────────────────────────┘
 Kaggle (free CPU) ─ rl_kaggle.ipynb ─┐  │ service-role key (server only)
                                      ▼  ▼
                         ┌────────── Supabase (free) ──────────┐
                         │ Postgres: agents, model_versions,   │
                         │ champion, trades, decisions, events,│
                         │ equity_snapshots, positions, ...    │
                         │ Storage: models/, data/ (candles)   │
                         └──────────────┬──────────────────────┘
                                        │ anon key + Auth, SELECT-only RLS, Realtime
                              ┌─────────▼─────────┐
                              │ Next.js dashboard │  Vercel Hobby
                              └───────────────────┘
```

Server mode (`docker-compose.yml`) runs the same code as `tradebot daemon` + a nightly trainer.

## Python package `src/tradebot`

| Module | Role |
|---|---|
| `config.py` | Typed settings from `config/default.yaml` (+ optional `config/local.yaml`). |
| `exchange/guard.py` | Hard-coded demo endpoint allowlist; every authenticated request is host-checked. |
| `exchange/venue.py` | `BybitDemoVenue` (ccxt, demo mode). There is no live venue. |
| `exchange/public.py` | Key-less market-data client. |
| `data/history.py`, `data/sync.py` | Paged OHLCV download, Parquet cache, Storage mirror. |
| `strategies/` | Strategy interface (target exposure in [-1, 1]) and 5 baselines. |
| `risk.py` | Hard caps + `RiskManager` (sizing, ATR stops, daily loss limit, kill switch). |
| `policy.py`, `simulate.py`, `paper.py` | Shared decision rules, one-bar simulation, paper account. |
| `backtest/engine.py` | Backtester built on `simulate.step_bar` (next-open fills, fees, slippage). |
| `metrics.py` | Return, Sharpe, Sortino, max drawdown, win rate, profit factor, trades, vs Buy & Hold. |
| `live/runner.py` | One step: shadow books for all agents + champion on the demo account. |
| `ml/` | Features, LightGBM model, walk-forward, Optuna tuning, nightly pipeline. |
| `rl/` | Gymnasium env, PPO training, `RLStrategy`. |
| `registry.py` | Model versions, champion/challenger promotion, model storage. |
| `store.py` | Supabase store (service role) and in-memory store for tests. |
| `news/` | News risk filter: `sources`, `filters`, `scoring`, `risk`, `pipeline`, `gate`. Only a shadow agent reads it. |

## Learning loop

1. **Pretraining analogue** — LightGBM learns 8h price direction from years of 15m/1h candles.
2. **RL analogue** — PPO trades in a simulator; reward = log growth after fees − drawdown penalty.
3. **Validation** — walk-forward only, purge gap, 0.1% fee + 0.05% slippage per side everywhere;
   last 9 months are a holdout no tuning has seen.
4. **Champion/challenger** — a version replaces the champion only if its holdout Sortino beats the
   champion's by 0.05 with ≥ 30 trades and < 20% drawdown.
5. **Nightly retraining** — `retrain.yml`; every version is stored with params and metrics.

## News risk filter

`news.yml` runs every 15 minutes and has no exchange keys, because it reads untrusted text.

1. **Collect** — 21 free sources: crypto RSS, Fed, SEC, CFTC, White House, BLS, Federal Register, 8-K
   filings, Truth Social (via trumpstruth.org), GDELT 15-minute files, Polymarket, public Telegram.
2. **Filter** — items over a day old, off the watch list or carrying obvious injection text never reach
   the model. A repeat of a known story joins its cluster and costs no call.
3. **Score** — Gemini, then Groq if it fails. The model has no tools, items are JSON data, and answers
   must pass a schema. Budget: 40 items a run, 500 a day, 3 new GDELT stories a run.
4. **Level** — per BTC, ETH and STOCKS: `trade`, `reduce`, `stand_aside`, or `unknown` when behind.
   Bad news cuts longs and good news cuts shorts; nothing raises a position. Standing aside needs at
   least two sources.

Results are `news_scored.parquet` and `news_risk.parquet` in the `data` bucket, outside the nightly
cache sync. Level changes are events of type `news_risk`.

5. **Test** — `news_<agent>` is a shadow twin of each model and of `buy_hold` and `ma_cross`. It holds the
   original's position times the level's multiplier, so it can never exceed it. Compare a twin with its
   original from the twin's first bar. The demo account never reads the levels.
6. **Read** — the dashboard's `/news` page shows levels and rated stories from `news_levels` and
   `news_items`.

## Database

See `supabase/migrations/20261101000000_init.sql`. The bot writes with the service-role key; the
dashboard reads as an authenticated user under SELECT-only RLS. `agent_state` (internal) has no
read policy. `prune_old_rows()` keeps the free tier small.
