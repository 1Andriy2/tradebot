"use client";

import { createChart, LineSeries, LineStyle, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import { kyivTick } from "@/lib/format";

export interface Line {
  id: string;
  label: string;
  color: string;
  width?: 1 | 2 | 3;
  dashed?: boolean;
  points: { time: number; value: number }[];
}

const percent = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;

/** Return curves of several agents on one chart, in percent. Times are UTC seconds. */
export default function EquityChart({ lines, height = 340 }: { lines: Line[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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
      localization: { timeFormatter: (t: number) => kyivTick(t), priceFormatter: percent },
    });
    chartRef.current = chart;
    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const series = lines.map((l) => {
      const s = chart.addSeries(LineSeries, {
        color: l.color,
        lineWidth: l.width ?? 2,
        lineStyle: l.dashed ? LineStyle.Dashed : LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      const data = [...new Map(l.points.map((p) => [p.time, p])).values()]
        .sort((a, b) => a.time - b.time)
        .map((p) => ({ time: p.time as UTCTimestamp, value: p.value }));
      s.setData(data);
      return s;
    });
    chart.timeScale().fitContent();
    return () => {
      series.forEach((s) => chart.removeSeries(s));
    };
  }, [lines]);

  return <div ref={ref} style={{ height }} className="w-full" />;
}
