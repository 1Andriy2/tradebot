import type { EquityRow, TradeRow } from "@/lib/types";

export interface LiveMetrics {
  totalReturn: number;
  maxDrawdown: number;
  sharpe: number;
  winRate: number;
  profitFactor: number;
  trades: number;
}

const PERIODS_PER_YEAR_15M = 365 * 96;

/** Same definitions as the Python `tradebot.metrics` module, on live snapshots. */
export function liveMetrics(equity: EquityRow[], trades: TradeRow[]): LiveMetrics {
  const eq = [...equity].sort((a, b) => a.ts.localeCompare(b.ts)).map((r) => r.equity);
  let peak = -Infinity;
  let maxDd = 0;
  const rets: number[] = [];
  eq.forEach((v, i) => {
    peak = Math.max(peak, v);
    maxDd = Math.max(maxDd, 1 - v / peak);
    if (i > 0 && eq[i - 1] > 0) rets.push(v / eq[i - 1] - 1);
  });
  const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length || 1));
  const wins = trades.filter((t) => t.pnl > 0);
  const grossWin = wins.reduce((a, t) => a + t.pnl, 0);
  const grossLoss = -trades.filter((t) => t.pnl < 0).reduce((a, t) => a + t.pnl, 0);
  return {
    totalReturn: eq.length > 1 ? eq[eq.length - 1] / eq[0] - 1 : 0,
    maxDrawdown: maxDd,
    sharpe: sd > 0 ? (mean / sd) * Math.sqrt(PERIODS_PER_YEAR_15M) : 0,
    winRate: trades.length ? wins.length / trades.length : 0,
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0,
    trades: trades.length,
  };
}
