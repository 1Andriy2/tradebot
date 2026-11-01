// Row types mirroring supabase/migrations/*.sql

export type Venue = "demo" | "shadow";

export interface AgentRow {
  id: string;
  kind: "baseline" | "ml" | "rl";
  label: string;
  params: Record<string, unknown>;
}

export interface EquityRow {
  ts: string;
  agent_id: string;
  venue: Venue;
  equity: number;
}

export interface PositionRow {
  agent_id: string;
  venue: Venue;
  symbol: string;
  units: number;
  avg_entry: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  updated_at: string;
}

export interface TradeRow {
  id: number;
  agent_id: string;
  venue: Venue;
  symbol: string;
  side: "long" | "short";
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  fees: number;
  return_pct: number | null;
  entry_reason: string;
  exit_reason: string;
}

export interface CandleRow {
  symbol: string;
  timeframe: string;
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface EventRow {
  id: number;
  ts: string;
  type: string;
  agent_id: string | null;
  message: string;
  details: Record<string, unknown>;
}

export interface DecisionRow {
  id: number;
  ts: string;
  agent_id: string;
  venue: Venue;
  symbol: string;
  signal: number;
  units_before: number;
  units_after: number;
  action: string;
  reason: string;
}

export interface ModelVersionRow {
  id: string;
  agent_id: string;
  version: number;
  status: "challenger" | "champion" | "retired" | "rejected";
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  train_start: string | null;
  train_end: string | null;
  created_at: string;
  promoted_at: string | null;
}

export interface BacktestRow {
  id: number;
  agent_id: string;
  label: string;
  symbol: string;
  timeframe: string;
  period_start: string;
  period_end: string;
  metrics: Record<string, number>;
  created_at: string;
}

export interface StatusRow {
  component: string;
  mode: string | null;
  last_step_at: string | null;
  last_ok_at: string | null;
  last_error: string | null;
  details: Record<string, unknown>;
}

export interface ChampionRow {
  agent_id: string;
  version_id: string | null;
  since: string;
}
