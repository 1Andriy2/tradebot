"use client";

import { useMemo } from "react";
import { Card, Empty, ErrorNote, PageTitle, Table } from "@/components/ui";
import { fmtNum, fmtPct, fmtTime, pnlClass } from "@/lib/format";
import { liveMetrics } from "@/lib/metrics";
import type { BacktestRow, EquityRow, TradeRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

export default function LeaderboardPage() {
  const equity = useRows<EquityRow>("equity_snapshots", {
    filters: { venue: "shadow" }, sinceDays: { column: "ts", days: 30 },
    order: { column: "ts", ascending: true }, limit: 30_000,
  });
  const trades = useRows<TradeRow>("trades", {
    filters: { venue: "shadow" }, sinceDays: { column: "exit_time", days: 30 }, limit: 10_000,
  });
  const backtests = useRows<BacktestRow>("backtest_results", { order: { column: "created_at" }, limit: 500 });

  const live = useMemo(() => {
    const ids = [...new Set(equity.rows.map((r) => r.agent_id))];
    return ids
      .map((id) => ({
        id,
        m: liveMetrics(equity.rows.filter((r) => r.agent_id === id), trades.rows.filter((t) => t.agent_id === id)),
      }))
      .sort((a, b) => b.m.sharpe - a.m.sharpe);
  }, [equity.rows, trades.rows]);

  // Latest backtest per (label, symbol, timeframe).
  const latestBt = useMemo(() => {
    const seen = new Map<string, BacktestRow>();
    for (const r of backtests.rows) {
      const k = `${r.label}|${r.symbol}|${r.timeframe}`;
      if (!seen.has(k)) seen.set(k, r);
    }
    return [...seen.values()].sort((a, b) => (b.metrics.sortino ?? 0) - (a.metrics.sortino ?? 0));
  }, [backtests.rows]);

  return (
    <>
      <PageTitle>Рейтинг стратегій і моделей</PageTitle>
      <ErrorNote error={equity.error ?? trades.error ?? backtests.error} />
      <Card title="Наживо (тіньові рахунки, 30 днів), сортування за Sharpe" className="mb-4">
        {live.length ? (
          <Table head={["#", "Агент", "Дохідність", "Sharpe", "Макс. просідання", "Win rate", "Profit factor", "Угод"]}>
            {live.map(({ id, m }, i) => (
              <tr key={id}>
                <td>{i + 1}</td>
                <td className="font-medium">{id}</td>
                <td className={pnlClass(m.totalReturn)}>{fmtPct(m.totalReturn)}</td>
                <td>{fmtNum(m.sharpe)}</td>
                <td>{fmtPct(m.maxDrawdown)}</td>
                <td>{fmtPct(m.winRate, 0)}</td>
                <td>{Number.isFinite(m.profitFactor) ? fmtNum(m.profitFactor) : "∞"}</td>
                <td>{m.trades}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty />}
      </Card>
      <Card title="Бектести (out-of-sample, з комісіями), сортування за Sortino">
        {latestBt.length ? (
          <Table head={["Стратегія", "Пара", "ТФ", "Період", "Дохідність", "vs Buy&Hold", "Sharpe", "Sortino", "Макс. просідання", "Win rate", "PF", "Угод"]}>
            {latestBt.map((r) => (
              <tr key={r.id}>
                <td className="font-medium">{r.label}</td>
                <td>{r.symbol.replace(":USDT", "")}</td>
                <td>{r.timeframe}</td>
                <td>{fmtTime(r.period_start).slice(0, 5)}–{fmtTime(r.period_end).slice(0, 5)}</td>
                <td className={pnlClass(r.metrics.total_return)}>{fmtPct(r.metrics.total_return)}</td>
                <td className={pnlClass(r.metrics.excess_vs_buy_hold)}>{fmtPct(r.metrics.excess_vs_buy_hold)}</td>
                <td>{fmtNum(r.metrics.sharpe)}</td>
                <td>{fmtNum(r.metrics.sortino)}</td>
                <td>{fmtPct(r.metrics.max_drawdown)}</td>
                <td>{fmtPct(r.metrics.win_rate, 0)}</td>
                <td>{r.metrics.profit_factor == null ? "∞" : fmtNum(r.metrics.profit_factor)}</td>
                <td>{r.metrics.num_trades}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Бектестів ще немає</Empty>}
      </Card>
    </>
  );
}
