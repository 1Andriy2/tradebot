// Labels and small helpers for the news page. Titles and links here come from outside the project:
// render them as plain text, and only ever link to http(s) addresses.

import type { NewsItemRow, NewsLevel } from "@/lib/types";

const TZ = "Europe/Kyiv";

export const SIGNIFICANT = 0.3; // below this the risk filter ignores a story
const STRONG = 0.7;

export const EVENT_TYPES: Record<string, string> = {
  regulation: "Регуляція",
  hack: "Злам",
  etf: "ETF",
  monetary: "Монетарна політика",
  appointment: "Призначення",
  sanctions: "Санкції",
  tariff: "Тарифи",
  election: "Вибори",
  geopolitics: "Геополітика",
  company: "Компанії",
  macro: "Макроекономіка",
  other: "Інше",
};

export const STANCES: Record<string, string> = { friendly: "прихильно", hostile: "вороже" };

export const CHANNELS: Record<string, string> = { direct: "Прямий вплив", indirect: "Непрямий вплив" };

export const ASSETS: Record<string, string> = { CRYPTO: "Крипторинок", STOCKS: "Акції" };

export const SCOPES: Record<string, string> = { BTC: "Bitcoin", ETH: "Ethereum", STOCKS: "Акції США" };

export const LEVELS: Record<NewsLevel, { label: string; hint: string; tone: string; dot: string }> = {
  trade: {
    label: "Торгувати",
    hint: "Новини не дають причин обмежувати позиції",
    tone: "text-emerald-700 dark:text-emerald-400",
    dot: "bg-emerald-500",
  },
  reduce: {
    label: "Зменшити",
    hint: "Новини тиснуть на ринок, частину позиції варто скоротити",
    tone: "text-amber-700 dark:text-amber-400",
    dot: "bg-amber-500",
  },
  stand_aside: {
    label: "Осторонь",
    hint: "Кілька джерел повідомляють про сильні події",
    tone: "text-rose-700 dark:text-rose-400",
    dot: "bg-rose-500",
  },
  unknown: {
    label: "Немає даних",
    hint: "Оцінка відстає, тому фільтр нічого не змінює",
    tone: "text-slate-600 dark:text-slate-400",
    dot: "bg-slate-400",
  },
};

const OUTLETS: Record<string, string> = {
  coindesk: "CoinDesk",
  cointelegraph: "Cointelegraph",
  decrypt: "Decrypt",
  bitcoinmagazine: "Bitcoin Magazine",
  theblock: "The Block",
  fed_press: "ФРС",
  fed_speeches: "ФРС, виступи",
  sec_press: "SEC",
  sec_8k: "SEC, звіти 8-K",
  cftc: "CFTC",
  white_house: "Білий дім",
  bls: "Бюро статистики праці",
  truth_trump: "Truth Social",
  fr_president: "Federal Register",
  fr_rules: "Federal Register",
  polymarket: "Polymarket",
};

export function outletName(source: string): string {
  if (source.startsWith("gdelt:")) return source.slice(6);
  if (source.startsWith("telegram:")) return `Telegram, ${source.slice(9)}`;
  return OUTLETS[source] ?? source;
}

export const assetList = (item: NewsItemRow): string[] => item.assets.split(",").filter(Boolean);

export function safeUrl(url: string): string | undefined {
  return /^https?:\/\//i.test(url) ? url : undefined;
}

export function fmtImpact(v: number): string {
  const text = Math.abs(v).toLocaleString("uk-UA", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${text}`;
}

export function strength(impact: number): string {
  const size = Math.abs(impact);
  if (size < SIGNIFICANT) return "слабкий";
  return size >= STRONG ? "сильний" : "помірний";
}

const dayKey = new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" });
const dayName = new Intl.DateTimeFormat("uk-UA", { timeZone: TZ, day: "numeric", month: "long" });
const clock = new Intl.DateTimeFormat("uk-UA", { timeZone: TZ, hour: "2-digit", minute: "2-digit" });

export const fmtClock = (iso: string): string => clock.format(new Date(iso));

/** When a story came out. A source's date is only trusted if it is not later than when we saw it. */
export function storyTime(item: NewsItemRow): string {
  return item.published < item.first_seen ? item.published : item.first_seen;
}

export function dayLabel(iso: string, now: number): string {
  const key = dayKey.format(new Date(iso));
  if (now && key === dayKey.format(new Date(now))) return "Сьогодні";
  if (now && key === dayKey.format(new Date(now - 86_400_000))) return "Вчора";
  return dayName.format(new Date(iso));
}

export function ago(iso: string, now: number): string {
  if (!now) return "";
  const minutes = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60_000));
  if (minutes < 1) return "щойно";
  if (minutes < 60) return `${minutes} хв тому`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `${hours} год тому` : `${Math.round(hours / 24)} дн тому`;
}

/** Stories in feed order, split into days. */
export function byDay(items: NewsItemRow[], now: number): { day: string; items: NewsItemRow[] }[] {
  const groups: { day: string; items: NewsItemRow[] }[] = [];
  for (const item of items) {
    const day = dayLabel(storyTime(item), now);
    if (groups.at(-1)?.day === day) groups.at(-1)!.items.push(item);
    else groups.push({ day, items: [item] });
  }
  return groups;
}
