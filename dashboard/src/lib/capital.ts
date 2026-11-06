// The demo balance is more than trading: an account can be topped up, swapped or counted differently.
// Those steps are shown as capital flows and kept out of the result, so one of them cannot read as a loss.

/**
 * A step this large is not trading. Exposure is capped at half the account (25% of it at 2x), so losing
 * 20% between two snapshots would take a 40% price move, and the kill switch acts at 20% anyway.
 */
export const FLOW_STEP = 0.2;

/**
 * Until this moment the balance counted every coin in the demo account (about 96 000 USDT), and from
 * it only the USDT that backs the trades (about 5 000). The two numbers cannot be compared, so the
 * account's history starts here. See commit fc1a6f6.
 */
export const COUNTED_IN_USDT_FROM = "2026-11-01T15:28:00Z";
export const OLD_COUNT_NOTE = "До 1 листопада баланс рахував усі монети демо-рахунку (близько 96 000 USDT), тепер лише USDT, якими бот торгує. Це зміна обліку, а не збиток, тому відлік іде від неї.";

export interface Point {
  ts: string;
  equity: number;
}

export interface Flow {
  ts: string;
  amount: number;
}

/** The part of the demo account's history that can be compared with today. */
export const countedInUsdt = <T extends Point>(points: T[]): T[] =>
  points.filter((p) => new Date(p.ts) >= new Date(COUNTED_IN_USDT_FROM));

export interface Capital {
  start: number;
  startedAt: string;
  current: number;
  flows: Flow[];
  added: number;
  /** Change in balance that came from trading: what is left after the flows. */
  trading: number;
  /** Trading return with flows taken out, each stretch compounded on the balance it started from. */
  tradingReturn: number;
}

const isFlow = (before: number, after: number) => before > 0 && Math.abs(after / before - 1) >= FLOW_STEP;

export function capital(points: Point[]): Capital | null {
  if (!points.length) return null;
  const flows: Flow[] = [];
  let growth = 1;
  points.forEach((p, i) => {
    if (i === 0) return;
    const before = points[i - 1].equity;
    if (isFlow(before, p.equity)) {
      flows.push({ ts: p.ts, amount: p.equity - before });
    } else if (before > 0) {
      growth *= p.equity / before;
    }
  });
  const start = points[0].equity;
  const current = points[points.length - 1].equity;
  const added = flows.reduce((sum, f) => sum + f.amount, 0);
  return {
    start, startedAt: points[0].ts, current, flows, added, trading: current - start - added, tradingReturn: growth - 1,
  };
}

/** Return in percent since the first point, with flows taken out. */
export function returnCurve(points: Point[]): { time: number; value: number }[] {
  let growth = 1;
  return points.map((p, i) => {
    const before = i ? points[i - 1].equity : p.equity;
    if (before > 0 && !isFlow(before, p.equity)) growth *= p.equity / before;
    return { time: Math.floor(new Date(p.ts).getTime() / 1000), value: (growth - 1) * 100 };
  });
}
