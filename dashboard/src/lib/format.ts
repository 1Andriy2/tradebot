const TZ = "Europe/Kyiv";

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Intl.DateTimeFormat("uk-UA", {
    timeZone: TZ,
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
}

export function fmtUsd(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${v.toLocaleString("uk-UA", { minimumFractionDigits: digits, maximumFractionDigits: digits })} USDT`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toLocaleString("uk-UA", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toLocaleString("uk-UA", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
}

/** Kyiv-local label for chart axes (timestamps stay in UTC seconds). */
export function kyivTick(unixSeconds: number): string {
  return new Intl.DateTimeFormat("uk-UA", {
    timeZone: TZ,
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(unixSeconds * 1000));
}

export const REASONS: Record<string, string> = {
  signal: "сигнал",
  stop_loss: "стоп-лосс",
  take_profit: "тейк-профіт",
  kill_switch: "kill switch",
  daily_loss_limit: "денний ліміт збитку",
  end_of_test: "кінець тесту",
  exchange_close: "закрито біржею",
  no_change: "без змін",
};

export const reason = (r: string) => REASONS[r] ?? r;

export const pnlClass = (v: number) =>
  v > 0 ? "text-emerald-600 dark:text-emerald-400" : v < 0 ? "text-rose-600 dark:text-rose-400" : "";
