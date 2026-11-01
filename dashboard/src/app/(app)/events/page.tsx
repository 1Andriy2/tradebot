"use client";

import { useState } from "react";
import { Card, Empty, ErrorNote, PageTitle, Select, Table } from "@/components/ui";
import { fmtNum, fmtTime, reason } from "@/lib/format";
import type { DecisionRow, EventRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const TYPES: Record<string, string> = {
  retrain: "🔁 перенавчання",
  champion_change: "🏆 зміна чемпіона",
  kill_switch: "🛑 kill switch",
  daily_loss_limit: "⛔ денний ліміт",
  error: "❗ помилка",
  info: "ℹ️ інфо",
};

const ACTIONS: Record<string, string> = {
  open: "відкрити", close: "закрити", flip: "розворот", increase: "збільшити", reduce: "зменшити", hold: "тримати",
};

export default function EventsPage() {
  const [type, setType] = useState("all");
  const events = useRows<EventRow>("events", {
    order: { column: "ts" }, limit: 300, realtime: true, ...(type === "all" ? {} : { filters: { type } }),
  });
  const decisions = useRows<DecisionRow>("decisions", {
    filters: { venue: "demo" }, order: { column: "ts" }, limit: 200, realtime: true,
  });

  return (
    <>
      <PageTitle>Журнал рішень і подій</PageTitle>
      <ErrorNote error={events.error ?? decisions.error} />
      <Card title="Події" className="mb-4">
        <div className="mb-3 max-w-xs">
          <Select label="Тип" value={type} onChange={setType}
            options={[{ value: "all", label: "Усі" }, ...Object.entries(TYPES).map(([value, label]) => ({ value, label }))]} />
        </div>
        {events.rows.length ? (
          <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
            {events.rows.map((e) => (
              <li key={e.id} className="py-2">
                <div className="flex flex-wrap gap-2">
                  <span className="text-slate-500">{fmtTime(e.ts)}</span>
                  <span className="font-medium">{TYPES[e.type] ?? e.type}</span>
                </div>
                <div className="break-words">{e.message}</div>
              </li>
            ))}
          </ul>
        ) : <Empty>Подій немає</Empty>}
      </Card>
      <Card title="Рішення чемпіона на демо-рахунку (з причинами)">
        {decisions.rows.length ? (
          <Table head={["Час свічки", "Агент", "Пара", "Сигнал", "Дія", "Причина", "Позиція до", "Позиція після"]}>
            {decisions.rows.map((d) => (
              <tr key={d.id}>
                <td>{fmtTime(d.ts)}</td>
                <td>{d.agent_id}</td>
                <td>{d.symbol.replace(":USDT", "")}</td>
                <td>{fmtNum(d.signal)}</td>
                <td>{ACTIONS[d.action] ?? d.action}</td>
                <td>{reason(d.reason)}</td>
                <td>{fmtNum(d.units_before, 4)}</td>
                <td>{fmtNum(d.units_after, 4)}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Рішень ще немає</Empty>}
      </Card>
    </>
  );
}
