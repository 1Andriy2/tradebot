"use client";

import { useMemo, useState } from "react";
import { Card, Empty, ErrorNote, PageTitle, Select, Table } from "@/components/ui";
import { fmtNum, fmtPct, fmtTime, pnlClass, reason } from "@/lib/format";
import type { TradeRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const ALL = "__all__";

export default function TradesPage() {
  const { rows, error } = useRows<TradeRow>("trades", {
    order: { column: "exit_time" }, sinceDays: { column: "exit_time", days: 90 }, limit: 5000, realtime: true,
  });
  const [agent, setAgent] = useState(ALL);
  const [venue, setVenue] = useState(ALL);
  const [entry, setEntry] = useState(ALL);
  const [exit, setExit] = useState(ALL);
  const [pnl, setPnl] = useState(ALL);

  const uniq = (f: (t: TradeRow) => string) => [...new Set(rows.map(f))].sort();
  const opts = (values: string[], label = (v: string) => v) =>
    [{ value: ALL, label: "Усі" }, ...values.map((v) => ({ value: v, label: label(v) }))];

  const filtered = useMemo(() => rows.filter((t) =>
    (agent === ALL || t.agent_id === agent) &&
    (venue === ALL || t.venue === venue) &&
    (entry === ALL || t.entry_reason === entry) &&
    (exit === ALL || t.exit_reason === exit) &&
    (pnl === ALL || (pnl === "win" ? t.pnl > 0 : t.pnl <= 0)),
  ), [rows, agent, venue, entry, exit, pnl]);

  const total = filtered.reduce((a, t) => a + t.pnl, 0);

  return (
    <>
      <PageTitle>Угоди</PageTitle>
      <ErrorNote error={error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Select label="Агент" value={agent} onChange={setAgent} options={opts(uniq((t) => t.agent_id))} />
        <Select label="Рахунок" value={venue} onChange={setVenue}
          options={opts(["demo", "shadow"], (v) => (v === "demo" ? "демо" : "тіньовий"))} />
        <Select label="Причина входу" value={entry} onChange={setEntry} options={opts(uniq((t) => t.entry_reason), reason)} />
        <Select label="Причина виходу" value={exit} onChange={setExit} options={opts(uniq((t) => t.exit_reason), reason)} />
        <Select label="Результат" value={pnl} onChange={setPnl}
          options={[{ value: ALL, label: "Усі" }, { value: "win", label: "Прибуткові" }, { value: "loss", label: "Збиткові" }]} />
      </div>
      <Card title={`Знайдено: ${filtered.length}, сумарний PnL: ${fmtNum(total)} USDT`}>
        {filtered.length ? (
          <Table head={["Вихід", "Агент", "Рахунок", "Пара", "Напрям", "Вхід", "Ціна входу", "Ціна виходу", "PnL", "%", "Комісії", "Причина входу", "Причина виходу"]}>
            {filtered.slice(0, 500).map((t) => (
              <tr key={t.id}>
                <td>{fmtTime(t.exit_time)}</td>
                <td>{t.agent_id}</td>
                <td>{t.venue === "demo" ? "демо" : "тіньовий"}</td>
                <td>{t.symbol.replace(":USDT", "")}</td>
                <td>{t.side === "long" ? "Long" : "Short"}</td>
                <td>{fmtTime(t.entry_time)}</td>
                <td>{fmtNum(t.entry_price)}</td>
                <td>{fmtNum(t.exit_price)}</td>
                <td className={pnlClass(t.pnl)}>{fmtNum(t.pnl)}</td>
                <td className={pnlClass(t.pnl)}>{fmtPct(t.return_pct)}</td>
                <td>{fmtNum(t.fees)}</td>
                <td>{reason(t.entry_reason)}</td>
                <td>{reason(t.exit_reason)}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Угод не знайдено</Empty>}
      </Card>
    </>
  );
}
