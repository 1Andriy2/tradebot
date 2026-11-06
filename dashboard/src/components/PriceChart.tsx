"use client";

import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  LineSeries,
  LineStyle,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { fmtNum, kyivTick, reason } from "@/lib/format";
import type { CandleRow, PositionRow, TradeRow } from "@/lib/types";

const GREEN = "#16a34a";
const RED = "#dc2626";
const MAX_LINES = 150; // one chart series each: the newest ones are what a reader is looking at
const BAR_SECONDS = 900;
const FOCUS_PADDING_BARS = 24;

const toSec = (iso: string) => Math.floor(new Date(iso).getTime() / 1000);

/** The candle that contains a moment, since a marker's time must match a candle. */
function snapper(times: number[]) {
  return (t: number): number | null => {
    let lo = 0, hi = times.length - 1, best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return best >= 0 ? times[best] : null;
  };
}

/**
 * Candles with each trade drawn as one line from its entry to its exit, green for a profit and red for
 * a loss, so entries and exits of many trades cannot be mistaken for one another. Words stay out of
 * the chart: pointing at a candle lists what happened on it.
 */
export default function PriceChart({ candles, trades, position, days, focus, height = 460 }: {
  candles: CandleRow[];
  trades: TradeRow[];
  position?: PositionRow;
  days: number;
  focus: TradeRow | null;
  height?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const notesRef = useRef(new Map<number, string[]>());
  const [hover, setHover] = useState<{ time: number; notes: string[] } | null>(null);

  const bars = useMemo(() => candles
    .map((c) => ({ time: toSec(c.ts), open: c.open, high: c.high, low: c.low, close: c.close }))
    .sort((a, b) => a.time - b.time), [candles]);

  useEffect(() => {
    if (!box.current) return;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const chart = createChart(box.current, {
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
      upColor: dark ? "#475569" : "#94a3b8", downColor: dark ? "#1e293b" : "#334155", borderVisible: false,
      wickUpColor: dark ? "#475569" : "#94a3b8", wickDownColor: dark ? "#334155" : "#475569",
    });
    markersRef.current = createSeriesMarkers(series, []);
    chart.subscribeCrosshairMove((param) => {
      const time = param.time as number | undefined;
      const notes = time == null ? undefined : notesRef.current.get(time);
      setHover(time != null && notes ? { time, notes } : null);
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    seriesRef.current?.setData(bars.map((b) => ({ ...b, time: b.time as UTCTimestamp })));
  }, [bars]);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series || !bars.length) return;
    const snap = snapper(bars.map((b) => b.time));
    const markers: SeriesMarker<UTCTimestamp>[] = [];
    const notes = new Map<number, string[]>();
    const note = (time: number, text: string) => notes.set(time, [...(notes.get(time) ?? []), text]);
    const lines: ISeriesApi<"Line">[] = [];

    trades.slice(0, MAX_LINES).forEach((t) => {
      const entry = snap(toSec(t.entry_time));
      const exit = snap(toSec(t.exit_time));
      const long = t.side === "long";
      const color = t.pnl >= 0 ? GREEN : RED;
      const side = long ? "Long" : "Short";
      if (entry != null) {
        markers.push({
          time: entry as UTCTimestamp, position: long ? "belowBar" : "aboveBar", size: 1,
          color: long ? GREEN : RED, shape: long ? "arrowUp" : "arrowDown",
        });
        note(entry, `Вхід ${side} за ${fmtNum(t.entry_price)}`);
      }
      if (exit != null) {
        markers.push({ time: exit as UTCTimestamp, position: long ? "aboveBar" : "belowBar", size: 0.6, color, shape: "circle" });
        note(exit, `Вихід ${side} за ${fmtNum(t.exit_price)}: ${t.pnl >= 0 ? "+" : ""}${fmtNum(t.pnl)} USDT, ${reason(t.exit_reason)}`);
      }
      if (entry != null && exit != null && exit > entry) {
        const line = chart.addSeries(LineSeries, {
          color, lineWidth: 2, lineStyle: LineStyle.Solid, priceLineVisible: false, lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        line.setData([
          { time: entry as UTCTimestamp, value: t.entry_price },
          { time: exit as UTCTimestamp, value: t.exit_price },
        ]);
        lines.push(line);
      }
    });
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    markersRef.current?.setMarkers(markers);
    notesRef.current = notes;
    return () => {
      lines.forEach((l) => chart.removeSeries(l));
    };
  }, [trades, bars]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series || !position || Math.abs(position.units) < 1e-12) return;
    const levels: [number | null, string, string][] = [
      [position.avg_entry, "#0ea5e9", "Вхід"], [position.stop_loss, RED, "Стоп"], [position.take_profit, GREEN, "Тейк"],
    ];
    const made: IPriceLine[] = levels.filter(([price]) => price != null).map(([price, color, title]) =>
      series.createPriceLine({ price: price as number, color, title, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true }));
    return () => made.forEach((l) => series.removePriceLine(l));
  }, [position, bars]);

  useEffect(() => {
    const scale = chartRef.current?.timeScale();
    const last = bars.at(-1)?.time;
    if (!scale || last == null) return;
    if (focus) {
      const pad = FOCUS_PADDING_BARS * BAR_SECONDS;
      scale.setVisibleRange({
        from: (toSec(focus.entry_time) - pad) as UTCTimestamp, to: (toSec(focus.exit_time) + pad) as UTCTimestamp,
      });
    } else {
      scale.setVisibleRange({ from: (last - days * 86_400) as UTCTimestamp, to: (last + 4 * BAR_SECONDS) as UTCTimestamp });
    }
  }, [days, focus, bars]);

  return (
    <div className="relative">
      <div ref={box} style={{ height }} className="w-full" />
      {hover && (
        <div className="pointer-events-none absolute left-2 top-2 z-10 max-w-[85%] rounded-lg bg-white/95 p-2 text-xs shadow dark:bg-slate-800/95">
          <div className="text-slate-500 dark:text-slate-400">{kyivTick(hover.time)}</div>
          {hover.notes.map((text) => <div key={text}>{text}</div>)}
        </div>
      )}
    </div>
  );
}
