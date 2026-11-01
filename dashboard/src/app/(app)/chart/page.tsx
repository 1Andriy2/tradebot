"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Card, Empty, ErrorNote, PageTitle, Select } from "@/components/ui";
import type { AgentRow, CandleRow, TradeRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false });

const SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"];

export default function ChartPage() {
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const [agent, setAgent] = useState("demo");
  const agents = useRows<AgentRow>("agents", { order: { column: "id", ascending: true } });
  const candles = useRows<CandleRow>("candles", {
    filters: { symbol, timeframe: "15m" }, order: { column: "ts", ascending: true },
    sinceDays: { column: "ts", days: 7 }, limit: 2000, realtime: true,
  });
  const trades = useRows<TradeRow>("trades", {
    filters: agent === "demo" ? { symbol, venue: "demo" } : { symbol, venue: "shadow", agent_id: agent },
    sinceDays: { column: "exit_time", days: 7 }, order: { column: "exit_time" }, limit: 500, realtime: true,
  });
  const options = useMemo(
    () => [{ value: "demo", label: "Чемпіон (демо-рахунок)" },
      ...agents.rows.map((a) => ({ value: a.id, label: `${a.id} (тіньовий)` }))],
    [agents.rows],
  );

  return (
    <>
      <PageTitle>Графік ціни та угоди</PageTitle>
      <div className="mb-4 flex flex-wrap gap-3">
        <Select label="Пара" value={symbol} onChange={setSymbol}
          options={SYMBOLS.map((s) => ({ value: s, label: s.replace(":USDT", "") }))} />
        <Select label="Агент" value={agent} onChange={setAgent} options={options} />
      </div>
      <ErrorNote error={candles.error ?? trades.error} />
      <Card title="15 хв, останні 7 днів (час київський)">
        {candles.rows.length ? <PriceChart candles={candles.rows} trades={trades.rows} /> : <Empty />}
        <p className="mt-2 text-xs text-slate-500">▲ вхід у long, ▼ вхід у short, ● вихід (підпис — PnL у USDT)</p>
      </Card>
    </>
  );
}
