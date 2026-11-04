"""Command line entry point: `tradebot <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import pandas as pd

from tradebot.backtest.engine import run_backtest
from tradebot.config import Settings, load_settings
from tradebot.data.history import load_cached, update_cache
from tradebot.exchange.public import make_public_client
from tradebot.live.runner import Trader, run_daemon
from tradebot.metrics import compute_metrics
from tradebot.store import MemoryStore, Store, SupabaseStore
from tradebot.strategies.baselines import baseline_strategies

log = logging.getLogger("tradebot")


def _store(args: argparse.Namespace) -> Store:
    return MemoryStore() if getattr(args, "memory_store", False) else SupabaseStore.from_env()


def _download(settings: Settings) -> None:
    from tradebot.data.funding import update_funding

    client = make_public_client(settings.venue.exchange_id)
    for symbol in settings.markets.symbols:
        for tf in settings.markets.timeframes:
            update_cache(client, settings.data_path, symbol, tf, settings.markets.history_start)
        try:
            update_funding(client, settings.data_path, symbol, settings.markets.history_start)
        except Exception:  # funding is optional; candles are what training needs
            log.exception("funding update failed for %s", symbol)


def _prepare_data(settings: Settings, args: argparse.Namespace, store: Store | None) -> None:
    """Optionally pull the cache from Supabase Storage, update it from the exchange, push it back."""
    from tradebot.data.sync import pull_cache, push_cache

    syms, tfs = settings.markets.symbols, settings.markets.timeframes
    if args.sync and store is not None:
        log.info("pulled %d cache files", pull_cache(store, settings.data_path, syms, tfs))
    if not args.no_download:
        try:
            _download(settings)
        except Exception:
            if not args.sync:
                raise
            log.exception("candle update failed; continuing with the synced cache")
    if args.sync and store is not None:
        log.info("pushed %d cache files", push_cache(store, settings.data_path, syms, tfs))


def _load_trade_frames(settings: Settings) -> dict[str, pd.DataFrame]:
    tf = settings.markets.trade_timeframe
    return {s: load_cached(settings.data_path, s, tf) for s in settings.markets.symbols}


def cmd_download(settings: Settings, args: argparse.Namespace) -> int:
    _prepare_data(settings, args, _store(args) if args.sync else None)
    return 0


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    symbols = [args.symbol] if args.symbol else settings.markets.symbols
    rows = []
    for symbol in symbols:
        df = load_cached(settings.data_path, symbol, args.timeframe)
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
        for strat in baseline_strategies():
            res = run_backtest(df, strat, settings.costs, settings.risk,
                               settings.backtest.initial_equity)
            m = compute_metrics(res.equity, res.trades, df["close"])
            rows.append({"symbol": symbol, "strategy": res.label, **m})
    table = pd.DataFrame(rows).sort_values(["symbol", "sortino"], ascending=[True, False])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    return 0


def _make_trader(settings: Settings, args: argparse.Namespace) -> Trader:
    from tradebot.exchange.venue import CcxtDemoVenue

    venue = None if args.shadow_only else CcxtDemoVenue.from_env()
    return Trader(settings, _store(args), make_public_client(settings.venue.exchange_id), venue)


def cmd_step(settings: Settings, args: argparse.Namespace) -> int:
    report = _make_trader(settings, args).step()
    print(report)
    return 1 if report.errors else 0


def cmd_daemon(settings: Settings, args: argparse.Namespace) -> int:
    trader = _make_trader(settings, args)
    tf_seconds = int(trader.public.parse_timeframe(settings.markets.trade_timeframe))
    run_daemon(trader, tf_seconds)
    return 0


def cmd_retrain(settings: Settings, args: argparse.Namespace) -> int:
    from tradebot.ml.pipeline import retrain

    store = _store(args)
    _prepare_data(settings, args, store)
    res = retrain(settings, store, _load_trade_frames(settings))
    print(f"version={res.version} promoted={res.promoted} "
          f"challenger_sortino={res.challenger['score']:.3f} "
          f"champion({res.champion_id})_sortino={res.champion['score']:.3f}")
    return 0


def cmd_rl_train(settings: Settings, args: argparse.Namespace) -> int:
    from tradebot.rl.train import RLConfig, train_and_register

    store = _store(args)
    _prepare_data(settings, args, store)
    version, promoted, summary = train_and_register(
        settings, store, _load_trade_frames(settings), RLConfig(timesteps=args.timesteps, device=args.device))
    print(f"rl version={version} promoted={promoted} {summary}")
    return 0


def cmd_maintenance(settings: Settings, args: argparse.Namespace) -> int:
    """Delete old rows to stay inside the Supabase free tier (see prune_old_rows in SQL)."""
    store = SupabaseStore.from_env()
    store.client.rpc("prune_old_rows").execute()
    log.info("pruned old rows")
    return 0


def cmd_research(settings: Settings, args: argparse.Namespace) -> int:
    from tradebot.data.funding import load_funding
    from tradebot.research.lab import ResearchConfig, run_research
    from tradebot.research.patterns import scan_seasonality

    store = _store(args)
    _prepare_data(settings, args, store)
    dfs = _load_trade_frames(settings)
    funding = {s: load_funding(settings.data_path, s) for s in settings.markets.symbols}
    cfg = ResearchConfig(workers=args.workers, initial_equity=settings.backtest.initial_equity)
    if args.families:
        cfg = dataclasses.replace(cfg, families=tuple(args.families))
    run_id, results = run_research(dfs, funding, settings.costs, settings.risk, cfg)
    now = pd.Timestamp.now(tz="UTC")
    store.insert("research_results", [{"run_id": run_id, "created_at": now, **r} for r in results])
    patterns = []
    for s in settings.markets.symbols:
        patterns += scan_seasonality(load_cached(settings.data_path, s, "1h"), s)
    store.insert("patterns", [{"run_id": run_id, "created_at": now, **p} for p in patterns])
    best = sorted(results, key=lambda r: r["metrics"]["dsr"], reverse=True)[:10]
    for r in best:
        m = r["metrics"]
        print(f"{r['family']:12s} {r['symbol']:15s} {r['timeframe']:4s} return={m['total_return']:+.3f} "
              f"sortino={m['sortino']:+.2f} dsr={m['dsr']:.2f} folds_won={m['fold_win_rate']:.2f} "
              f"trades={int(m['num_trades'])} fees={m['fees_paid']:.0f} funding={m['funding_paid']:.0f} "
              f"checks={int(m['checks_passed'])}/{int(m['checks_total'])}")
    stable = [p for p in patterns if p["stable"] and p["q_value"] < 0.1]
    robust = sum(r["metrics"]["robust"] == 1 for r in results)
    print(f"run {run_id}: {len(results)} results, {robust} robust, {len(stable)} stable patterns")
    return 0


def cmd_reset_kill_switch(settings: Settings, args: argparse.Namespace) -> int:
    """Manual (human) reset of a tripped kill switch for one agent/venue/symbol."""
    from tradebot.policy import PolicyState

    store = _store(args)
    key = {"agent_id": args.agent, "venue": args.venue, "symbol": args.symbol}
    rows = store.select("agent_state", key)
    if not rows:
        print(f"no state for {key}")
        return 1
    row = rows[0]
    state = PolicyState.from_dict(row["state"]["policy"])
    state.risk.killed = False
    state.risk.halted_day = None
    state.risk.peak_equity = args.equity
    state.risk.day_start_equity = args.equity
    row["state"]["policy"] = state.to_dict()
    store.upsert("agent_state", [{**key, "state": row["state"]}], on_conflict="agent_id,venue,symbol")
    store.insert("events", [{"ts": pd.Timestamp.now(tz="UTC"), "type": "info", "agent_id": args.agent,
                             "message": f"kill switch reset manually for {args.venue}/{args.symbol}; "
                                        f"new peak equity {args.equity}", "details": {}}])
    print(f"kill switch reset for {key}, peak equity = {args.equity}")
    return 0


def _data_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--memory-store", action="store_true", help="do not write to Supabase")
    p.add_argument("--no-download", action="store_true", help="use cached candles as they are")
    p.add_argument("--sync", action="store_true", help="pull/push the candle cache via Supabase Storage")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradebot")
    sub = p.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download/update historical candles")
    dl.add_argument("--sync", action="store_true", help="pull/push the candle cache via Supabase Storage")
    dl.set_defaults(no_download=False)
    bt = sub.add_parser("backtest", help="backtest baseline strategies on cached data")
    bt.add_argument("--symbol")
    bt.add_argument("--timeframe", default="1h")
    bt.add_argument("--start", help="ISO date to start from")
    for name, help_ in (("step", "run one trading step and exit (cron mode)"),
                        ("daemon", "run a trading step after every candle close")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--shadow-only", action="store_true", help="paper agents only, no demo orders")
        sp.add_argument("--memory-store", action="store_true", help="do not write to Supabase")
    rt = sub.add_parser("retrain", help="walk-forward retraining with champion/challenger promotion")
    _data_flags(rt)
    rl = sub.add_parser("rl-train", help="train the PPO agent and run champion/challenger")
    rl.add_argument("--timesteps", type=int, default=300_000)
    rl.add_argument("--device", default="auto", help="auto | cpu | cuda")
    _data_flags(rl)
    sub.add_parser("maintenance", help="prune old rows in Supabase (free-tier housekeeping)")
    rs = sub.add_parser("research", help="walk-forward scan of all strategy families and patterns")
    rs.add_argument("--workers", type=int, default=4)
    rs.add_argument("--families", nargs="*", help="subset of strategy families")
    _data_flags(rs)
    ks = sub.add_parser("reset-kill-switch", help="manually reset a tripped kill switch (human decision)")
    ks.add_argument("--agent", required=True)
    ks.add_argument("--venue", choices=["demo", "shadow"], required=True)
    ks.add_argument("--symbol", required=True)
    ks.add_argument("--equity", type=float, required=True, help="equity to use as the new peak")
    ks.add_argument("--memory-store", action="store_true", help=argparse.SUPPRESS)
    return p


COMMANDS = {"download": cmd_download, "backtest": cmd_backtest, "step": cmd_step, "daemon": cmd_daemon,
            "retrain": cmd_retrain, "rl-train": cmd_rl_train, "maintenance": cmd_maintenance,
            "reset-kill-switch": cmd_reset_kill_switch, "research": cmd_research}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](load_settings(), args)


if __name__ == "__main__":
    sys.exit(main())
