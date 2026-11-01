"use client";

import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  type SeriesMarker,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import { kyivTick } from "@/lib/format";
import type { CandleRow, TradeRow } from "@/lib/types";

const toSec = (iso: string) => Math.floor(new Date(iso).getTime() / 1000);

/** Candles with trade entry/exit markers snapped to existing candle times. */
export default function PriceChart({ candles, trades, height = 420 }: {
  candles: CandleRow[];
  trades: TradeRow[];
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const chart = createChart(ref.current, {
      autoSize: true,
      height,
      layout: { background: { color: "transparent" }, textColor: dark ? "#cbd5e1" : "#334155" },
      grid: {
        vertLines: { color: dark ? "#1e293b" : "#f1f5f9" },
        horzLines: { color: dark ? "#1e293b" : "#f1f5f9" },
      },
      timeScale: { timeVisible: true, tickMarkFormatter: (t: number) => kyivTick(t) },
      localization: { timeFormatter: (t: number) => kyivTick(t) },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#16a34a", downColor: "#dc2626", borderVisible: false,
      wickUpColor: "#16a34a", wickDownColor: "#dc2626",
    });
    const data = candles
      .map((c) => ({ time: toSec(c.ts), open: c.open, high: c.high, low: c.low, close: c.close }))
      .sort((a, b) => a.time - b.time);
    series.setData(data.map((d) => ({ ...d, time: d.time as UTCTimestamp })));

    // Marker times must match a candle: snap each event to the candle that contains it.
    const times = data.map((d) => d.time);
    const snap = (t: number) => {
      let lo = 0, hi = times.length - 1, best = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
      }
      return best >= 0 ? times[best] : null;
    };
    const markers: SeriesMarker<UTCTimestamp>[] = [];
    for (const t of trades) {
      const entry = snap(toSec(t.entry_time));
      const exit = snap(toSec(t.exit_time));
      const long = t.side === "long";
      if (entry != null) markers.push({
        time: entry as UTCTimestamp, position: long ? "belowBar" : "aboveBar",
        color: long ? "#16a34a" : "#dc2626", shape: long ? "arrowUp" : "arrowDown",
        text: long ? "Long" : "Short",
      });
      if (exit != null) markers.push({
        time: exit as UTCTimestamp, position: long ? "aboveBar" : "belowBar",
        color: t.pnl >= 0 ? "#2563eb" : "#64748b", shape: "circle",
        text: `Вихід ${t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(1)}`,
      });
    }
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    createSeriesMarkers(series, markers);
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [candles, trades, height]);

  return <div ref={ref} style={{ height }} className="w-full" />;
}
