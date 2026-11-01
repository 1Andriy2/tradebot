"use client";

import { useMemo, useState } from "react";
import { Card, Empty, ErrorNote, PageTitle, Select, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, fmtTime, pnlClass } from "@/lib/format";
import type { PatternRow, ResearchRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const ALL = "__all__";
const DOW = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"];

type SortKey = "dsr" | "sortino" | "total_return" | "fold_win_rate" | "excess_vs_buy_hold";
const SORTS: { value: SortKey; label: string }[] = [
  { value: "dsr", label: "Deflated Sharpe (надійність)" },
  { value: "fold_win_rate", label: "Стабільність по фолдах" },
  { value: "sortino", label: "Sortino" },
  { value: "total_return", label: "Дохідність" },
  { value: "excess_vs_buy_hold", label: "Краще за Buy & Hold" },
];

function paramsText(p: Record<string, unknown> | undefined): string {
  if (!p || !Object.keys(p).length) return "—";
  return Object.entries(p).map(([k, v]) => `${k}=${v}`).join(", ");
}

/** Mini bar chart of per-fold returns: does the edge repeat over time? */
function FoldBars({ values }: { values: number[] }) {
  const max = Math.max(1e-9, ...values.map((v) => Math.abs(v)));
  return (
    <div className="flex h-6 items-center gap-px" title={values.map((v) => fmtPct(v, 1)).join("  ")}>
      {values.map((v, i) => (
        <span key={i} className={v >= 0 ? "bg-emerald-500" : "bg-rose-500"}
          style={{ width: 4, height: `${Math.max(8, (Math.abs(v) / max) * 100)}%` }} />
      ))}
    </div>
  );
}

export default function ResearchPage() {
  const results = useRows<ResearchRow>("research_results", { order: { column: "created_at" }, limit: 300, realtime: true });
  const patterns = useRows<PatternRow>("patterns", { order: { column: "created_at" }, limit: 300, realtime: true });
  const [tf, setTf] = useState(ALL);
  const [symbol, setSymbol] = useState(ALL);
  const [sort, setSort] = useState<SortKey>("dsr");

  const runId = results.rows[0]?.run_id;
  const latest = useMemo(() => results.rows.filter((r) => r.run_id === runId), [results.rows, runId]);
  const rows = useMemo(() => latest
    .filter((r) => (tf === ALL || r.timeframe === tf) && (symbol === ALL || r.symbol === symbol))
    .sort((a, b) => (b.metrics[sort] ?? -Infinity) - (a.metrics[sort] ?? -Infinity)), [latest, tf, symbol, sort]);

  const patRun = patterns.rows[0]?.run_id;
  const pats = patterns.rows
    .filter((p) => p.run_id === patRun && p.q_value < 0.2)
    .sort((a, b) => Number(b.stable) - Number(a.stable) || a.q_value - b.q_value);

  const profitable = latest.filter((r) => r.metrics.total_return > 0).length;
  const reliable = latest.filter((r) => r.metrics.dsr >= 0.95).length;
  const uniq = (f: (r: ResearchRow) => string) => [...new Set(latest.map(f))].sort();

  return (
    <>
      <PageTitle>Дослідження стратегій і патернів</PageTitle>
      <ErrorNote error={results.error ?? patterns.error} />
      <p className="mb-4 text-sm text-slate-500">
        Усе нижче — out-of-sample: параметри кожної стратегії обираються на попередніх 12 місяцях і
        перевіряються на наступних 3 (walk-forward). Комісії, проковзування та funding враховані.
        Deflated Sharpe ≥ 0.95 означає, що результат навряд чи випадковий з урахуванням кількості перевірених варіантів.
      </p>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Останній прогін" value={fmtTime(latest[0]?.created_at)} />
        <Stat label="Комбінацій перевірено" value={latest.length}
          sub={latest[0] ? `${latest[0].metrics.n_trials} варіантів параметрів` : undefined} />
        <Stat label="Прибуткових (OOS)" value={`${profitable} / ${latest.length}`} />
        <Stat label="Надійних (DSR ≥ 0.95)" value={reliable} />
      </div>

      <Card title="Стратегії (найкращі параметри з кожного фолду, склеєні в один OOS-тест)" className="mb-4">
        <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-3">
          <Select label="Таймфрейм" value={tf} onChange={setTf}
            options={[{ value: ALL, label: "Усі" }, ...uniq((r) => r.timeframe).map((v) => ({ value: v, label: v }))]} />
          <Select label="Пара" value={symbol} onChange={setSymbol}
            options={[{ value: ALL, label: "Усі" }, ...uniq((r) => r.symbol).map((v) => ({ value: v, label: v.replace(":USDT", "") }))]} />
          <Select label="Сортувати за" value={sort} onChange={(v) => setSort(v as SortKey)} options={SORTS} />
        </div>
        {rows.length ? (
          <Table head={["Стратегія", "Пара", "ТФ", "Дохідність", "vs B&H", "Sortino", "Макс. просід.", "Win rate", "PF", "Угод",
            "Фолди +", "Фолди", "DSR", "Комісії", "Funding", "Останні параметри"]}>
            {rows.map((r) => {
              const m = r.metrics;
              return (
                <tr key={r.id}>
                  <td className="font-medium">{r.family}</td>
                  <td>{r.symbol.replace(":USDT", "")}</td>
                  <td>{r.timeframe}</td>
                  <td className={pnlClass(m.total_return)}>{fmtPct(m.total_return)}</td>
                  <td className={pnlClass(m.excess_vs_buy_hold)}>{fmtPct(m.excess_vs_buy_hold)}</td>
                  <td>{fmtNum(m.sortino)}</td>
                  <td>{fmtPct(m.max_drawdown)}</td>
                  <td>{fmtPct(m.win_rate, 0)}</td>
                  <td>{m.profit_factor == null ? "∞" : fmtNum(m.profit_factor)}</td>
                  <td>{m.num_trades}</td>
                  <td>{fmtPct(m.fold_win_rate, 0)}</td>
                  <td><FoldBars values={r.fold_returns ?? []} /></td>
                  <td className={m.dsr >= 0.95 ? "font-semibold text-emerald-600" : ""}>{fmtNum(m.dsr)}</td>
                  <td>{fmtNum(m.fees_paid, 0)}</td>
                  <td>{fmtNum(m.funding_paid, 0)}</td>
                  <td className="max-w-xs truncate text-xs text-slate-500">{paramsText(r.params.latest)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Дослідження ще не запускалось</Empty>}
      </Card>

      <Card title="Повторювані патерни (година доби / день тижня, 1h-свічки)">
        <p className="mb-3 text-xs text-slate-500">
          q — p-value з поправкою на множинні перевірки. «Стабільний» — той самий знак і значущість в обох половинах історії.
        </p>
        {pats.length ? (
          <Table head={["Пара", "Тип", "Коли (UTC)", "Сер. дохідність", "t", "q", "Спостережень", "Стабільний"]}>
            {pats.map((p) => (
              <tr key={p.id}>
                <td>{p.symbol.replace(":USDT", "")}</td>
                <td>{p.kind === "hour" ? "година" : "день"}</td>
                <td>{p.kind === "hour" ? `${String(p.bucket).padStart(2, "0")}:00` : DOW[p.bucket]}</td>
                <td className={pnlClass(p.mean_return)}>{fmtPct(p.mean_return, 3)}</td>
                <td>{fmtNum(p.t_stat)}</td>
                <td>{fmtNum(p.q_value, 3)}</td>
                <td>{p.n}</td>
                <td>{p.stable ? "✅" : "—"}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Значущих патернів не знайдено</Empty>}
      </Card>
    </>
  );
}
