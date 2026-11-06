"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import type { Line } from "@/components/EquityChart";
import { Card, Empty, ErrorNote, Hint, Note, PageTitle, SideBadge, Stat, Table, colorFor } from "@/components/ui";
import { agentAbout, agentName, pairName } from "@/lib/agents";
import { capital, countedInUsdt, OLD_COUNT_NOTE, returnCurve } from "@/lib/capital";
import { fmtDay, fmtNum, fmtSignedPct, fmtSignedUsd, fmtTime, fmtUsd, pnlClass } from "@/lib/format";
import type { ChampionRow, EquityRow, PositionRow, StatusRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const EquityChart = dynamic(() => import("@/components/EquityChart"), { ssr: false });

const DAYS = 14;
const DEMO = "demo";
const DEMO_COLOR = "#f59e0b";
const BENCHMARK = "buy_hold";
const BENCHMARK_COLOR = "#94a3b8";

const POSITION_HEAD = ["Пара", "Напрям", { label: "Обсяг", right: true }, { label: "Ціна входу", right: true },
  { label: "Стоп", right: true }, { label: "Тейк", right: true }];

function PositionCells({ p }: { p: PositionRow }) {
  return (
    <>
      <td>{pairName(p.symbol)}</td>
      <td><SideBadge side={p.units > 0 ? "long" : "short"} /></td>
      <td className="text-right">{fmtNum(Math.abs(p.units), 4)}</td>
      <td className="text-right">{fmtNum(p.avg_entry)}</td>
      <td className="text-right">{fmtNum(p.stop_loss)}</td>
      <td className="text-right">{fmtNum(p.take_profit)}</td>
    </>
  );
}

function CapitalPart({ label, value, tone = "", hint }: { label: string; value: string; tone?: string; hint?: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{hint ? <Hint text={hint}>{label}</Hint> : label}</div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums ${tone}`}>{value}</div>
    </div>
  );
}

export default function OverviewPage() {
  const demoRows = useRows<EquityRow>("equity_hourly", {
    filters: { venue: "demo" }, order: { column: "ts", ascending: true }, limit: 20_000,
    realtime: true, listen: "equity_snapshots",
  });
  const shadowRows = useRows<EquityRow>("equity_hourly", {
    filters: { venue: "shadow" }, order: { column: "ts", ascending: true },
    sinceDays: { column: "ts", days: DAYS }, limit: 20_000,
  });
  const positions = useRows<PositionRow>("positions", { realtime: true });
  const champion = useRows<ChampionRow>("champion", { realtime: true });
  const status = useRows<StatusRow>("bot_status", { filters: { component: "trader" }, realtime: true });
  const [showAll, setShowAll] = useState(false);
  const [picked, setPicked] = useState<Record<string, boolean>>({});

  const championId = champion.rows[0]?.agent_id ?? "ma_cross";
  const demo = useMemo(() => countedInUsdt(demoRows.rows), [demoRows.rows]);
  const money = useMemo(() => capital(demo), [demo]);

  const lines = useMemo<(Line & { last: number })[]>(() => {
    const ids = [...new Set(shadowRows.rows.map((r) => r.agent_id))];
    const curves = ids.map((id) => {
      const points = returnCurve(shadowRows.rows.filter((r) => r.agent_id === id));
      return {
        id, label: agentName(id), points, last: points.at(-1)?.value ?? 0,
        color: id === BENCHMARK ? BENCHMARK_COLOR : colorFor(id, ids),
        dashed: id === BENCHMARK,
      };
    }).sort((a, b) => b.last - a.last);
    const account = returnCurve(demo);
    return account.length
      ? [{ id: DEMO, label: "Демо-рахунок", color: DEMO_COLOR, width: 3, points: account, last: account.at(-1)!.value }, ...curves]
      : curves;
  }, [shadowRows.rows, demo]);

  // Fifteen lines read as one: open with the ones that answer "are we doing well".
  const main = useMemo(() => {
    const agents = lines.filter((l) => l.id !== DEMO);
    return new Set([DEMO, BENCHMARK, championId, agents[0]?.id, agents.at(-1)?.id]);
  }, [lines, championId]);
  const visible = (id: string) => picked[id] ?? (showAll || main.has(id));
  const drawn = useMemo(
    () => lines.filter((l) => picked[l.id] ?? (showAll || main.has(l.id))),
    [lines, picked, showAll, main],
  );

  const open = positions.rows.filter((p) => Math.abs(p.units) > 1e-12);
  const demoOpen = open.filter((p) => p.venue === "demo" && p.agent_id === championId);
  const shadowOpen = open.filter((p) => p.venue === "shadow");
  const trader = status.rows[0];

  return (
    <>
      <PageTitle>Огляд</PageTitle>
      <ErrorNote error={demoRows.error ?? shadowRows.error ?? positions.error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Баланс демо-рахунку" value={fmtUsd(money?.current)}
          tone={money ? pnlClass(money.trading) : ""}
          sub={money ? `торгівля: ${fmtSignedPct(money.tradingReturn)} від старту` : undefined} />
        <Stat label="Хто торгує зараз" value={<Hint text={agentAbout(championId)}>{agentName(championId)}</Hint>}
          sub={champion.rows[0] ? `чемпіон з ${fmtDay(champion.rows[0].since)}` : "за замовчуванням"} />
        <Stat label="Відкриті позиції (демо)" value={demoOpen.length}
          sub={demoOpen.length ? undefined : "бот поза ринком"} />
        <Stat label="Останній крок бота" value={fmtTime(trader?.last_step_at)}
          sub={trader?.last_error ? "⚠️ була помилка, деталі в Статусі" : "без помилок"} />
      </div>

      {money && (
        <Card title="Капітал демо-рахунку" className="mb-4">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <CapitalPart label={`Старт, ${fmtDay(money.startedAt)}`} value={fmtUsd(money.start)} />
            <CapitalPart label="Поповнення і зняття" value={money.flows.length ? fmtSignedUsd(money.added) : "немає"}
              hint="Стрибки балансу понад 20% за крок. Торгівля так не рухає рахунок, тому вони не входять у результат." />
            <CapitalPart label="Результат торгівлі" value={fmtSignedUsd(money.trading)} tone={pnlClass(money.trading)} />
            <CapitalPart label="Зараз" value={fmtUsd(money.current)} />
          </div>
          {money.flows.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm">
              {money.flows.map((f) => (
                <li key={f.ts}>{fmtTime(f.ts)}: {f.amount > 0 ? "поповнення" : "зняття"} {fmtUsd(Math.abs(f.amount))}</li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{OLD_COUNT_NOTE}</p>
        </Card>
      )}

      <Card title={`Дохідність за ${DAYS} днів, у відсотках від старту кожного`} className="mb-4"
        hint="Демо-рахунок проти тестових агентів. Кожна лінія починається з нуля, коли агент почав торгувати. Натисніть на назву, щоб показати чи сховати лінію.">
        {drawn.length ? <EquityChart lines={drawn} /> : <Empty />}
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {lines.map((l) => (
            <button
              key={l.id}
              type="button"
              aria-pressed={visible(l.id)}
              onClick={() => setPicked((p) => ({ ...p, [l.id]: !visible(l.id) }))}
              className={`flex items-center gap-1.5 rounded-md border border-slate-200 px-2 py-1 text-xs dark:border-slate-700 ${visible(l.id) ? "" : "opacity-40"}`}
            >
              <span className="inline-block h-2 w-3 rounded-sm" style={{ background: l.color }} />
              {l.label}
              <span className={`tabular-nums ${pnlClass(l.last)}`}>{l.last > 0 ? "+" : ""}{fmtNum(l.last)}%</span>
            </button>
          ))}
          <button type="button" onClick={() => { setShowAll((v) => !v); setPicked({}); }}
            className="px-2 py-1 text-xs text-slate-500 underline hover:text-slate-900 dark:hover:text-white">
            {showAll ? "Лише основні" : "Показати всі"}
          </button>
        </div>
      </Card>

      <Card title="Позиції на демо-рахунку">
        {demoOpen.length ? (
          <Table head={POSITION_HEAD}>
            {demoOpen.map((p) => <tr key={p.symbol}><PositionCells p={p} /></tr>)}
          </Table>
        ) : <Empty>Відкритих позицій немає: {agentName(championId)} зараз поза ринком</Empty>}
        {shadowOpen.length > 0 && (
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-slate-500 hover:text-slate-900 dark:hover:text-white">
              Позиції тестових агентів: {shadowOpen.length} (на папері, не на біржі)
            </summary>
            <div className="mt-2">
              <Table head={["Агент", ...POSITION_HEAD]}>
                {shadowOpen.map((p) => (
                  <tr key={`${p.agent_id}-${p.symbol}`}><td>{agentName(p.agent_id)}</td><PositionCells p={p} /></tr>
                ))}
              </Table>
            </div>
          </details>
        )}
      </Card>
      {demoRows.loading || money ? null : <div className="mt-4"><Note>Даних про баланс ще немає</Note></div>}
    </>
  );
}
