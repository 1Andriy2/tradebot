"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { Card, ErrorNote, Empty, PageTitle, Stat, Table, colorFor } from "@/components/ui";
import { fmtNum, fmtTime, fmtUsd } from "@/lib/format";
import type { ChampionRow, EquityRow, PositionRow, StatusRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const EquityChart = dynamic(() => import("@/components/EquityChart"), { ssr: false });

export default function OverviewPage() {
  const equity = useRows<EquityRow>("equity_snapshots", {
    order: { column: "ts", ascending: true }, sinceDays: { column: "ts", days: 14 }, limit: 20_000, realtime: true,
  });
  const positions = useRows<PositionRow>("positions", { realtime: true });
  const champion = useRows<ChampionRow>("champion", { realtime: true });
  const status = useRows<StatusRow>("bot_status", { filters: { component: "trader" }, realtime: true });

  const championId = champion.rows[0]?.agent_id ?? "ma_cross";
  const demo = equity.rows.filter((r) => r.venue === "demo");
  const lastDemo = demo[demo.length - 1];
  const firstDemo = demo[0];

  const lines = useMemo(() => {
    const keys = [...new Set(equity.rows.map((r) => `${r.agent_id}|${r.venue}`))];
    const ids = [...new Set(equity.rows.map((r) => r.agent_id))];
    return keys.map((k) => {
      const [agent, venue] = k.split("|");
      return {
        id: k,
        label: venue === "demo" ? `${agent} (демо)` : agent,
        color: venue === "demo" ? "#0f172a" : colorFor(agent, ids),
        points: equity.rows
          .filter((r) => r.agent_id === agent && r.venue === venue)
          .map((r) => ({ time: Math.floor(new Date(r.ts).getTime() / 1000), value: r.equity })),
      };
    });
  }, [equity.rows]);

  const open = positions.rows.filter((p) => Math.abs(p.units) > 1e-12);

  return (
    <>
      <PageTitle>Огляд</PageTitle>
      <ErrorNote error={equity.error ?? positions.error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Баланс демо-рахунку" value={fmtUsd(lastDemo?.equity)}
          sub={firstDemo && lastDemo ? `за 14 днів: ${fmtNum((lastDemo.equity / firstDemo.equity - 1) * 100)}%` : undefined} />
        <Stat label="Чемпіон" value={championId} sub={champion.rows[0] ? `з ${fmtTime(champion.rows[0].since)}` : "за замовчуванням"} />
        <Stat label="Відкриті позиції (демо)" value={open.filter((p) => p.venue === "demo").length} />
        <Stat label="Останній крок бота" value={fmtTime(status.rows[0]?.last_step_at)}
          sub={status.rows[0]?.last_error ? "⚠️ є помилки" : "ок"} />
      </div>

      <Card title="Криві капіталу всіх агентів (14 днів)" className="mb-4">
        {lines.length ? <EquityChart lines={lines} /> : <Empty />}
        <div className="mt-2 flex flex-wrap gap-3 text-xs">
          {lines.map((l) => (
            <span key={l.id} className="flex items-center gap-1">
              <span className="inline-block h-2 w-4 rounded" style={{ background: l.color }} />
              {l.label}
            </span>
          ))}
        </div>
      </Card>

      <Card title="Поточні позиції">
        {open.length ? (
          <Table head={["Агент", "Рахунок", "Пара", "Напрям", "Обсяг", "Ціна входу", "Стоп", "Тейк"]}>
            {open.map((p) => (
              <tr key={`${p.agent_id}-${p.venue}-${p.symbol}`}>
                <td>{p.agent_id}</td>
                <td>{p.venue === "demo" ? "демо" : "тіньовий"}</td>
                <td>{p.symbol.replace(":USDT", "")}</td>
                <td>{p.units > 0 ? "🟢 Long" : "🔴 Short"}</td>
                <td>{fmtNum(Math.abs(p.units), 4)}</td>
                <td>{fmtNum(p.avg_entry)}</td>
                <td>{fmtNum(p.stop_loss)}</td>
                <td>{fmtNum(p.take_profit)}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Відкритих позицій немає</Empty>}
      </Card>
    </>
  );
}
