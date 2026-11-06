"use client";

import { useMemo, useState } from "react";
import { Badge, Card, Chips, Empty, ErrorNote, PageTitle, SideBadge, Table } from "@/components/ui";
import { agentName, coinName, pairName } from "@/lib/agents";
import { COUNTED_IN_USDT_FROM } from "@/lib/capital";
import { groupErrors } from "@/lib/errors";
import { fmtDay, fmtNum, fmtPct, fmtTime, reason } from "@/lib/format";
import { LEVELS } from "@/lib/news";
import type { DecisionRow, EventRow, NewsLevel } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const MAIN_TYPES = ["retrain", "champion_change", "kill_switch", "daily_loss_limit", "info"];
const TABS: [string, string][] = [["trades", "Дії бота"], ["main", "Важливі події"], ["news", "Фільтр новин"], ["errors", "Помилки"]];
const COUNT_CHANGE_WINDOW_MS = 3_600_000;

const TITLES: Record<string, string> = {
  retrain: "🔁 Навчання", champion_change: "🏆 Новий чемпіон", kill_switch: "🛑 Аварійна зупинка",
  daily_loss_limit: "⛔ Денний ліміт збитку", info: "ℹ️ Службове",
};

/** What a decision did, in words. A close has no side of its own: it ends the one before it. */
function actionText(d: DecisionRow): string {
  const side = d.units_after > 0 ? "Long" : "Short";
  switch (d.action) {
    case "open": return `Відкрив ${side}`;
    case "close": return `Закрив ${d.units_before > 0 ? "Long" : "Short"}`;
    case "flip": return `Розвернув у ${side}`;
    case "increase": return `Збільшив ${side}`;
    case "reduce": return `Зменшив ${side}`;
    default: return d.action;
  }
}

function eventText(e: EventRow): string {
  const d = e.details as { rejection_reasons?: string[]; drawdown?: number; ledger?: number; exchange?: number };
  const version = /\(v=(\d+)\)/.exec(e.message)?.[1];
  const who = `${agentName(e.agent_id ?? "")}${version ? ` v${version}` : ""}`;
  switch (e.type) {
    case "retrain":
      return d.rejection_reasons?.length ? `${who} навчено, перевірку не пройшла. Причини на сторінці Навчання.` : `${who} навчено й призначено чемпіоном.`;
    case "champion_change":
      return `Торгувати на демо-рахунку тепер буде ${who}.`;
    case "kill_switch": {
      const moment = new Date(e.ts).getTime() - new Date(COUNTED_IN_USDT_FROM).getTime();
      if (moment >= 0 && moment < COUNT_CHANGE_WINDOW_MS) {
        return "Хибне спрацювання: баланс почав рахуватись лише в USDT і виглядав як обвал. Збитку не було, зупинку знято вручну.";
      }
      return `Просідання досягло ${fmtPct(d.drawdown ?? 0, 1)}, бот закрив позиції й зупинив торгівлю.`;
    }
    case "daily_loss_limit":
      return "Збиток за день досяг ліміту, нові позиції заборонені до кінця дня.";
    default:
      if (e.message.includes("resynced")) return `${coinName(e.message)}: бот узяв під нагляд позицію, яка вже була на біржі.`;
      if (e.message.includes("kill switch reset")) return "Аварійну зупинку знято вручну.";
      return e.message;
  }
}

function Now({ rows }: { rows: DecisionRow[] }) {
  const latest = [...new Map([...rows].reverse().map((d) => [d.symbol, d])).values()];
  return (
    <Card title="Що бот думає зараз" className="mb-4"
      hint="Останнє рішення поточного чемпіона на кожній парі. Сигнал +1 означає Long, −1 Short, 0 поза ринком.">
      {latest.length ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {latest.map((d) => (
            <div key={d.symbol} className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold">{pairName(d.symbol)}</span>
                {d.signal > 0 ? <SideBadge side="long" /> : d.signal < 0 ? <SideBadge side="short" /> : <Badge>поза ринком</Badge>}
              </div>
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                {agentName(d.agent_id)}, сигнал {fmtNum(d.signal)}, свічка {fmtTime(d.ts)}
              </div>
            </div>
          ))}
        </div>
      ) : <Empty />}
    </Card>
  );
}

export default function EventsPage() {
  const [tab, setTab] = useState("trades");
  const latest = useRows<DecisionRow>("decisions", { filters: { venue: "demo" }, order: { column: "ts" }, limit: 8, realtime: true });
  const actions = useRows<DecisionRow>("decisions", {
    filters: { venue: "demo" }, exclude: { action: ["hold"] }, order: { column: "ts" }, limit: 200, realtime: true,
  });
  const events = useRows<EventRow>("events", {
    exclude: { type: ["scenario", "event_study", "error", "news_risk"] }, order: { column: "ts" }, limit: 200, realtime: true,
  });
  const levels = useRows<EventRow>("events", { filters: { type: "news_risk" }, order: { column: "ts" }, limit: 100 });
  const errors = useRows<EventRow>("events", { filters: { type: "error" }, order: { column: "ts" }, limit: 300, realtime: true });
  const groups = useMemo(() => groupErrors(errors.rows), [errors.rows]);
  const main = events.rows.filter((e) => MAIN_TYPES.includes(e.type));

  return (
    <>
      <PageTitle intro="Запис того, що бот уже зробив і чому. Це не торгові поради: сигнали належать моделі, яка на перевірці поки збиткова.">
        Журнал
      </PageTitle>
      <ErrorNote error={latest.error ?? actions.error ?? events.error ?? errors.error} />
      <Now rows={latest.rows} />
      <div className="mb-3"><Chips label="Розділ журналу" value={tab} onChange={setTab} options={TABS} /></div>

      {tab === "trades" && (
        <Card title="Дії на демо-рахунку" hint="Лише моменти, коли бот щось змінив. Години без змін сховані.">
          {actions.rows.length ? (
            <Table head={["Час свічки", "Хто", "Пара", "Дія", "Чому", { label: "Обсяг після", right: true }]}>
              {actions.rows.map((d) => (
                <tr key={d.id}>
                  <td>{fmtTime(d.ts)}</td>
                  <td>{agentName(d.agent_id)}</td>
                  <td>{pairName(d.symbol)}</td>
                  <td className="font-medium">{actionText(d)}</td>
                  <td>{reason(d.reason)}</td>
                  <td className="text-right">{fmtNum(Math.abs(d.units_after), 4)}</td>
                </tr>
              ))}
            </Table>
          ) : <Empty>Дій ще не було</Empty>}
        </Card>
      )}

      {tab === "main" && (
        <Card title="Важливі події">
          {main.length ? (
            <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
              {main.map((e) => (
                <li key={e.id} className="py-2.5">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-medium">{TITLES[e.type] ?? e.type}</span>
                    <span className="text-xs text-slate-500">{fmtTime(e.ts)}</span>
                  </div>
                  <div className="break-words text-slate-700 dark:text-slate-300">{eventText(e)}</div>
                </li>
              ))}
            </ul>
          ) : <Empty>Подій немає</Empty>}
        </Card>
      )}

      {tab === "news" && (
        <Card title="Зміни рівня ризику від новин" hint="Це експеримент: рівень читають лише тестові агенти, на демо-рахунок він не впливає.">
          {levels.rows.length ? (
            <Table head={["Час", "Ринок", "Новий рівень"]}>
              {levels.rows.map((e) => {
                const [scope, level] = e.message.split(": ");
                const info = LEVELS[level as NewsLevel];
                return (
                  <tr key={e.id}>
                    <td>{fmtTime(e.ts)}</td>
                    <td>{scope}</td>
                    <td className={info?.tone ?? ""}>{info?.label ?? level}</td>
                  </tr>
                );
              })}
            </Table>
          ) : <Empty>Змін ще не було</Empty>}
        </Card>
      )}

      {tab === "errors" && (
        <Card title="Помилки" hint="Однакові помилки зібрані в один рядок. Зараз вони не заважають, якщо на сторінці Статус зелений індикатор.">
          {groups.length ? (
            <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
              {groups.map((g) => (
                <li key={g.sample} className="py-2.5">
                  <div className="font-medium">{g.title}{g.count > 1 && <span className="font-normal text-slate-500"> × {g.count}</span>}</div>
                  {g.explain && <div className="text-slate-600 dark:text-slate-400">{g.explain}</div>}
                  <div className="mt-0.5 text-xs text-slate-500">
                    {g.count > 1 ? `${fmtDay(g.first)}, ${fmtTime(g.first)} – ${fmtTime(g.last)}` : fmtTime(g.last)}
                  </div>
                  {!g.explain && <div className="mt-0.5 break-words text-xs text-slate-500">{g.sample}</div>}
                </li>
              ))}
            </ul>
          ) : <Empty>Помилок немає 🎉</Empty>}
        </Card>
      )}
    </>
  );
}
