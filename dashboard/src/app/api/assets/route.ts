// Looks up a ticker for the watchlist. The proxy already keeps this route behind the login. All
// sources are free and need no key. A source that fails only leaves its matches out.

import type { AssetCandidate } from "@/lib/types";

const DAY_MS = 86_400_000;
const USER_AGENT = "tradebot-research/1.0"; // the SEC answers 403 to an agent containing a URL
const TICKER = /^[A-Z0-9.\-]{1,12}$/;
const MAX_COINS = 25;
const MAJOR_COIN_RANK = 100; // above this a coin outranks a stock with the same ticker

interface Coin { id: string; symbol: string; name: string }
interface Company { cik_str: number; ticker: string; title: string }

// These lists are 1 to 2 MB and change rarely. A plain variable is enough: they are only read while
// subscribing, so a cold start that fetches them again costs nothing that matters.
const lists = new Map<string, { at: number; data: unknown }>();

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { "User-Agent": USER_AGENT }, signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(`${response.status} from ${new URL(url).host}`);
  return (await response.json()) as T;
}

async function daily<T>(url: string): Promise<T> {
  const hit = lists.get(url);
  if (hit && Date.now() - hit.at < DAY_MS) return hit.data as T;
  const data = await json<T>(url);
  lists.set(url, { at: Date.now(), data });
  return data;
}

async function coins(ticker: string): Promise<AssetCandidate[]> {
  // The full list shows every duplicate. It is heavy and CoinGecko limits shared addresses, so when
  // it is refused the light search still finds the main coin.
  const all = await daily<Coin[]>("https://api.coingecko.com/api/v3/coins/list").catch(async () =>
    (await json<{ coins: Coin[] }>(`https://api.coingecko.com/api/v3/search?query=${encodeURIComponent(ticker)}`)).coins);
  const same = all.filter((c) => c.symbol.toUpperCase() === ticker);
  if (!same.length) return [];
  const ids = same.slice(0, 250).map((c) => c.id).join(",");
  const markets = await json<{ id: string; current_price: number | null; market_cap: number | null;
    total_volume: number | null; market_cap_rank: number | null }[]>(
    `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&ids=${encodeURIComponent(ids)}`,
  ).catch(() => []);
  const byId = new Map(markets.map((m) => [m.id, m]));
  return same
    .map((c): AssetCandidate => {
      const m = byId.get(c.id);
      return {
        id: `crypto:${c.id}`, kind: "crypto", ticker, name: c.name, source_id: c.id,
        price: m?.current_price ?? null, market_cap: m?.market_cap ?? null, volume: m?.total_volume ?? null,
        rank: m?.market_cap_rank ?? null, exchange: null, currency: "USD",
      };
    })
    // Without market data, a short name is more likely the original than "Bridged X (Chain)".
    .sort((a, b) => (b.market_cap ?? -1) - (a.market_cap ?? -1) || a.name.length - b.name.length)
    .slice(0, MAX_COINS);
}

async function stocks(typed: string): Promise<AssetCandidate[]> {
  const ticker = typed.replace(".", "-"); // people write BRK.B, the SEC and Yahoo BRK-B
  const all = await daily<Record<string, Company>>("https://www.sec.gov/files/company_tickers.json");
  const company = Object.values(all).find((c) => c.ticker.toUpperCase() === ticker);
  if (!company) return [];
  const quote = await json<{ chart: { result: { meta: { regularMarketPrice?: number; currency?: string;
    fullExchangeName?: string; longName?: string } }[] } }>(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=1d&interval=1d`,
  ).then((r) => r.chart.result[0].meta).catch(() => null);
  return [{
    // Yahoo writes 'JPMorgan Chase & Co.' where the SEC has 'JPMORGAN CHASE & CO': closer to headlines.
    id: `stock:${ticker}`, kind: "stock", ticker, name: quote?.longName ?? company.title,
    source_id: String(company.cik_str),
    price: quote?.regularMarketPrice ?? null, market_cap: null, volume: null, rank: null,
    exchange: quote?.fullExchangeName ?? null, currency: quote?.currency ?? "USD",
  }];
}

export async function GET(request: Request) {
  const ticker = (new URL(request.url).searchParams.get("q") ?? "").trim().toUpperCase();
  if (!TICKER.test(ticker)) return Response.json({ error: "Введіть тікер, наприклад SOL або NVDA" }, { status: 400 });
  const found = await Promise.allSettled([coins(ticker), stocks(ticker)]);
  const [coinMatches, stockMatches] = found.map((r) => (r.status === "fulfilled" ? r.value : []));
  const isMajor = (c: AssetCandidate) => c.rank != null && c.rank <= MAJOR_COIN_RANK;
  const candidates = [...coinMatches.filter(isMajor), ...stockMatches, ...coinMatches.filter((c) => !isMajor(c))];
  const failed = found.filter((r) => r.status === "rejected").length;
  return Response.json({ candidates, partial: failed > 0 });
}
