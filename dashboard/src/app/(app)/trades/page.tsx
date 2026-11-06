"use client";

import { useMemo, useState } from "react";
import { Badge, Card, Empty, ErrorNote, PageTitle, Select, SideBadge, Stat, Table } from "@/components/ui";
import { agentAbout, agentName, pairName } from "@/lib/agents";
import { fmtDuration, fmtNum, fmtPct, fmtSignedPct, fmtSignedUsd, fmtTime, pnlClass, reason } from "@/lib/format";
import type { TradeRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const ALL = "__all__";
const PAGE = 100;

const EXIT_TONES: Record<string, "green" | "red" | "amber" | "grey"> = {
  take_profit: "green", stop_loss: "red", kill_switch: "red", daily_loss_limit: "red", news_filter: "amber",
};

const HEAD = [
  "Угода", "Агент",
  { label: "Вхід", hint: "Коли і за якою ціною бот увійшов." },
  { label: "Вихід", hint: "Коли і за якою ціною бот вийшов." },
  "Тривалість",
  { label: "Результат", right: true, hint: "Прибуток чи збиток уже після комісій. Під ним те саме у відсотках і сума комісій." },
  { label: "Чому вийшов", hint: "Сигнал: стратегія передумала. Стоп-лосс чи тейк-профіт: ціна дійшла до заданого рівня." },
];

function TwoLines({ top, bottom }: { top: React.ReactNode; bottom: React.ReactNode }) {
  return (
    <>
      <div>{top}</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{bottom}</div>
    </>
  );
}

export default function TradesPage() {
  const { rows, error, loading } = useRows<TradeRow>("trades", {
    order: { column: "exit_time" }, sinceDays: { column: "exit_time", days: 90 }, limit: 5000, realtime: true,
  });
  const [agent, setAgent] = useState(ALL);
  const [venue, setVenue] = useState(ALL);
  const [entry, setEntry] = useState(ALL);
  const [exit, setExit] = useState(ALL);
  const [pnl, setPnl] = useState(ALL);
  const [limit, setLimit] = useState(PAGE);

  const uniq = (f: (t: TradeRow) => string) => [...new Set(rows.map(f))].sort();
  const opts = (values: string[], label = (v: string) => v) =>
    [{ value: ALL, label: "Усі" }, ...values.map((v) => ({ value: v, label: label(v) }))];
  const pick = (set: (v: string) => void) => (v: string) => { set(v); setLimit(PAGE); };

  const filtered = useMemo(() => rows.filter((t) =>
    (agent === ALL || t.agent_id === agent) &&
    (venue === ALL || t.venue === venue) &&
    (entry === ALL || t.entry_reason === entry) &&
    (exit === ALL || t.exit_reason === exit) &&
    (pnl === ALL || (pnl === "win" ? t.pnl > 0 : t.pnl <= 0)),
  ), [rows, agent, venue, entry, exit, pnl]);

  const total = filtered.reduce((a, t) => a + t.pnl, 0);
  const fees = filtered.reduce((a, t) => a + t.fees, 0);
  const wins = filtered.filter((t) => t.pnl > 0).length;

  return (
    <>
      <PageTitle intro="Усі закриті угоди за 90 днів. «Демо» означає угоду на демо-рахунку біржі, «тест» означає паперову угоду тестового агента.">
        Угоди
      </PageTitle>
      <ErrorNote error={error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Угод" value={filtered.length} />
        <Stat label="Сумарний результат" value={fmtSignedUsd(total)} tone={pnlClass(total)} />
        <Stat label="Прибуткових" value={filtered.length ? fmtPct(wins / filtered.length, 0) : "—"} sub={`${wins} з ${filtered.length}`} />
        <Stat label="Сплачено комісій" value={`${fmtNum(fees)} USDT`} />
      </div>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Select label="Агент" value={agent} onChange={pick(setAgent)} options={opts(uniq((t) => t.agent_id), agentName)} />
        <Select label="Рахунок" value={venue} onChange={pick(setVenue)}
          options={opts(["demo", "shadow"], (v) => (v === "demo" ? "демо" : "тест"))} />
        <Select label="Причина входу" value={entry} onChange={pick(setEntry)} options={opts(uniq((t) => t.entry_reason), reason)} />
        <Select label="Причина виходу" value={exit} onChange={pick(setExit)} options={opts(uniq((t) => t.exit_reason), reason)} />
        <Select label="Результат" value={pnl} onChange={pick(setPnl)}
          options={[{ value: ALL, label: "Усі" }, { value: "win", label: "Прибуткові" }, { value: "loss", label: "Збиткові" }]} />
      </div>
      <Card>
        {filtered.length ? (
          <>
            <Table head={HEAD}>
              {filtered.slice(0, limit).map((t) => (
                <tr key={t.id}>
                  <td>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{pairName(t.symbol).split("/")[0]}</span>
                      <SideBadge side={t.side} />
                    </div>
                  </td>
                  <td>
                    <TwoLines top={<span title={agentAbout(t.agent_id)}>{agentName(t.agent_id)}</span>}
                      bottom={t.venue === "demo" ? "демо" : "тест"} />
                  </td>
                  <td><TwoLines top={fmtTime(t.entry_time)} bottom={fmtNum(t.entry_price)} /></td>
                  <td><TwoLines top={fmtTime(t.exit_time)} bottom={fmtNum(t.exit_price)} /></td>
                  <td>{fmtDuration(t.entry_time, t.exit_time)}</td>
                  <td className="text-right">
                    <div className={`font-semibold ${pnlClass(t.pnl)}`}>{fmtSignedUsd(t.pnl)}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {fmtSignedPct(t.return_pct)}, комісії {fmtNum(t.fees)}
                    </div>
                  </td>
                  <td><Badge tone={EXIT_TONES[t.exit_reason] ?? "grey"}>{reason(t.exit_reason)}</Badge></td>
                </tr>
              ))}
            </Table>
            {filtered.length > limit && (
              <button type="button" onClick={() => setLimit((v) => v + PAGE)}
                className="mt-3 w-full rounded-lg bg-slate-100 py-2 text-sm hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700">
                Показати ще (залишилось {filtered.length - limit})
              </button>
            )}
          </>
        ) : <Empty>{loading ? "Завантаження…" : "Угод не знайдено"}</Empty>}
      </Card>
    </>
  );
}
