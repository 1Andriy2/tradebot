"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Card, Chips, Empty, ErrorNote, PageTitle, Select, SideBadge, Table } from "@/components/ui";
import { agentName, pairName } from "@/lib/agents";
import { fmtDuration, fmtSignedUsd, fmtTime, pnlClass, reason } from "@/lib/format";
import type { AgentRow, CandleRow, ChampionRow, PositionRow, TradeRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false });

const SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"];
const DEMO = "demo";
const LOADED_DAYS = 7;
const RANGES: [string, string][] = [["1", "1 день"], ["3", "3 дні"], ["7", "7 днів"]];
const RESULTS: [string, string][] = [["all", "Усі угоди"], ["win", "Прибуткові"], ["loss", "Збиткові"]];
const LISTED = 12;

function Legend() {
  const item = "flex items-center gap-1.5";
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
      <span className={item}><span className="h-0.5 w-5 bg-emerald-600" />угода з прибутком</span>
      <span className={item}><span className="h-0.5 w-5 bg-rose-600" />угода зі збитком</span>
      <span className={item}><span className="text-emerald-600">▲</span>вхід у long</span>
      <span className={item}><span className="text-rose-600">▼</span>вхід у short</span>
      <span className={item}><span>●</span>вихід</span>
      <span>Наведіть на свічку, щоб побачити деталі</span>
    </div>
  );
}

export default function ChartPage() {
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const [agent, setAgent] = useState(DEMO);
  const [days, setDays] = useState("3");
  const [result, setResult] = useState("all");
  const [focusId, setFocusId] = useState<number | null>(null);

  const agents = useRows<AgentRow>("agents", { order: { column: "id", ascending: true } });
  const champion = useRows<ChampionRow>("champion");
  const candles = useRows<CandleRow>("candles", {
    filters: { symbol, timeframe: "15m" }, order: { column: "ts", ascending: true },
    sinceDays: { column: "ts", days: LOADED_DAYS }, limit: 2000, realtime: true,
  });
  const trades = useRows<TradeRow>("trades", {
    filters: agent === DEMO ? { symbol, venue: "demo" } : { symbol, venue: "shadow", agent_id: agent },
    sinceDays: { column: "exit_time", days: LOADED_DAYS }, order: { column: "exit_time" }, limit: 500, realtime: true,
  });
  const positions = useRows<PositionRow>("positions", { filters: { symbol }, realtime: true });

  const options = useMemo(
    () => [{ value: DEMO, label: "Демо-рахунок" },
      ...agents.rows.map((a) => ({ value: a.id, label: `${agentName(a.id)} (тест)` }))],
    [agents.rows],
  );
  const shown = useMemo(() => trades.rows.filter((t) =>
    result === "all" || (result === "win" ? t.pnl > 0 : t.pnl <= 0)), [trades.rows, result]);
  const focus = shown.find((t) => t.id === focusId) ?? null;
  const owner = agent === DEMO ? champion.rows[0]?.agent_id : agent;
  const position = positions.rows.find((p) => p.agent_id === owner && p.venue === (agent === DEMO ? "demo" : "shadow"));

  const pick = (set: (v: string) => void) => (v: string) => { set(v); setFocusId(null); };

  return (
    <>
      <PageTitle>Графік ціни та угоди</PageTitle>
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <Select label="Пара" value={symbol} onChange={pick(setSymbol)}
          options={SYMBOLS.map((s) => ({ value: s, label: pairName(s) }))} />
        <Select label="Чиї угоди" value={agent} onChange={pick(setAgent)} options={options} />
        <Chips label="Період" value={days} onChange={pick(setDays)} options={RANGES} />
        <Chips label="Результат" value={result} onChange={pick(setResult)} options={RESULTS} />
      </div>
      <ErrorNote error={candles.error ?? trades.error} />
      <Card title="Свічки по 15 хвилин, час київський" className="mb-4">
        {candles.rows.length
          ? <PriceChart candles={candles.rows} trades={shown} position={position} days={Number(days)} focus={focus} />
          : <Empty />}
        <Legend />
      </Card>
      <Card title={`Угоди за ${LOADED_DAYS} днів: ${shown.length}`}
        hint="Натисніть на рядок, щоб наблизити цю угоду на графіку.">
        {shown.length ? (
          <Table head={["Вхід", "Напрям", "Тривалість", { label: "Результат", right: true }, "Чому вийшов"]}>
            {shown.slice(0, LISTED).map((t) => (
              <tr
                key={t.id}
                tabIndex={0}
                aria-selected={t.id === focusId}
                onClick={() => setFocusId(t.id === focusId ? null : t.id)}
                onKeyDown={(e) => { if (e.key === "Enter") setFocusId(t.id === focusId ? null : t.id); }}
                className={`cursor-pointer ${t.id === focusId ? "bg-sky-50 dark:bg-sky-950/40" : ""}`}
              >
                <td>{fmtTime(t.entry_time)}</td>
                <td><SideBadge side={t.side} /></td>
                <td>{fmtDuration(t.entry_time, t.exit_time)}</td>
                <td className={`text-right font-medium ${pnlClass(t.pnl)}`}>{fmtSignedUsd(t.pnl)}</td>
                <td>{reason(t.exit_reason)}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>За цей період угод немає</Empty>}
      </Card>
    </>
  );
}
