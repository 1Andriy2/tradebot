"use client";

import { useMemo } from "react";
import { Badge, Card, Empty, ErrorNote, Hint, Note, PageTitle, Table } from "@/components/ui";
import { agentAbout, agentName, coinName } from "@/lib/agents";
import { fmtDay, fmtNum, fmtPct, fmtSignedPct, pnlClass } from "@/lib/format";
import type { AgentStatsRow, BacktestRow, ChampionRow, PositionRow } from "@/lib/types";
import { useNow } from "@/lib/useNow";
import { useRows } from "@/lib/useRows";

const MIN_TRADES = 30; // the same number the promotion rule asks for
const MIN_DAYS = 14;

/** Can anything be said yet? A result counts only when it is far from zero for its own noise: the
 *  spread of a sum of trades grows with the square root of their number. Trade results are not
 *  stored one by one, so their spread is estimated from the average size of a win or a loss. */
function verdict(s: { trades: number; gross_win: number; gross_loss: number; started_at: string }, now: number):
  { label: string; tone: "grey" | "green" | "red" } {
  const days = (now - new Date(s.started_at).getTime()) / 86_400_000;
  if (s.trades < MIN_TRADES) return { label: `мало угод: ${s.trades} з ${MIN_TRADES}`, tone: "grey" };
  if (days < MIN_DAYS) return { label: `мало часу: ${Math.floor(days)} з ${MIN_DAYS} дн`, tone: "grey" };
  const net = s.gross_win - s.gross_loss;
  const spread = 1.25 * ((s.gross_win + s.gross_loss) / s.trades) * Math.sqrt(s.trades);
  if (net > 2 * spread) return { label: "заробляє", tone: "green" };
  if (net < -2 * spread) return { label: "втрачає", tone: "red" };
  return { label: "не відрізняється від нуля", tone: "grey" };
}

const LIVE_HEAD = [
  "#", "Агент",
  { label: "Старт", hint: "Коли агент почав торгувати. Агенти додавались у різні дні, тому порівнюйте з поправкою на це." },
  { label: "З початку", right: true, hint: "Дохідність від старту агента. Рахунок не скидається: кожен почав із 10 000 USDT і веде його далі." },
  { label: "За добу", right: true, hint: "Зміна за останні 24 години. Ця колонка показує, що сталося сьогодні." },
  { label: "Макс. просідання", right: true, hint: "Найбільше падіння від піку рахунку. Менше краще." },
  { label: "Угод", right: true, hint: `Закриті угоди. Менше ${MIN_TRADES} угод ще нічого не доводять.` },
  { label: "Успішних", right: true, hint: "Частка прибуткових угод." },
  { label: "Profit factor", right: true, hint: "Сума прибутків, поділена на суму збитків. Вище 1 означає, що стратегія заробляє." },
  { label: "Висновок", hint: `Чи вже можна щось сказати. Потрібно щонайменше ${MIN_TRADES} угод і ${MIN_DAYS} днів, а результат має бути далеко від нуля з огляду на розкид угод. Інакше місце в таблиці вирішує випадок.` },
  "Зараз",
];

const TEST_HEAD = [
  "Стратегія",
  { label: "BTC", right: true, hint: "Дохідність на BTC за весь період тесту, з комісіями." },
  { label: "ETH", right: true, hint: "Дохідність на ETH за весь період тесту, з комісіями." },
  { label: "Проти утримання", right: true, hint: "Наскільки стратегія краща чи гірша за просте утримання монети. Середнє за BTC і ETH." },
  { label: "Sortino", right: true, hint: "Дохідність відносно падінь. Вище нуля добре, нижче нуля стратегія втрачала. Середнє за BTC і ETH." },
  { label: "Макс. просідання", right: true, hint: "Найбільше падіння від піку, гірше з двох монет." },
  { label: "Угод", right: true, hint: "Угоди за весь період на обох монетах разом." },
  { label: "Пораховано", hint: "Коли тест запускався востаннє. Моделі перевіряються, коли навчаються." },
];

interface Test {
  id: string;
  btc?: BacktestRow;
  eth?: BacktestRow;
}

const mean = (values: (number | undefined)[]) => {
  const known = values.filter((v): v is number => v != null);
  return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null;
};

export default function LeaderboardPage() {
  const now = useNow();
  const stats = useRows<AgentStatsRow>("agent_live_stats");
  const positions = useRows<PositionRow>("positions", { filters: { venue: "shadow" } });
  const champion = useRows<ChampionRow>("champion");
  const backtests = useRows<BacktestRow>("backtest_results", { order: { column: "created_at" }, limit: 500 });
  const championId = champion.rows[0]?.agent_id;

  const live = useMemo(() => stats.rows
    .map((s) => ({ ...s, total: s.equity / s.start_equity - 1, day: s.equity_day_ago ? s.equity / s.equity_day_ago - 1 : null }))
    .sort((a, b) => b.total - a.total), [stats.rows]);

  const holding = (id: string) => positions.rows
    .filter((p) => p.agent_id === id && Math.abs(p.units) > 1e-12)
    .map((p) => `${coinName(p.symbol)} ${p.units > 0 ? "Long" : "Short"}`);

  // The newest test of each strategy. Versions of a model are one strategy, so only the last one counts.
  const tests = useMemo(() => {
    const seen = new Map<string, Test>();
    for (const r of backtests.rows) {
      const test = seen.get(r.agent_id) ?? { id: r.agent_id };
      const coin = coinName(r.symbol) === "BTC" ? "btc" : "eth";
      test[coin] ??= r;
      seen.set(r.agent_id, test);
    }
    const score = (t: Test) => mean([t.btc?.metrics.sortino, t.eth?.metrics.sortino]) ?? -Infinity;
    return [...seen.values()].sort((a, b) => score(b) - score(a));
  }, [backtests.rows]);

  const newest = tests.flatMap((t) => [t.btc, t.eth]).find(Boolean);
  const leader = live[0];
  const days = leader && now ? Math.max(1, Math.round((now - new Date(leader.started_at).getTime()) / 86_400_000)) : 0;

  return (
    <>
      <PageTitle intro="Два різні змагання. Перше йде просто зараз на свіжих цінах. Друге показує, як стратегії пройшли б минулі місяці.">
        Рейтинг стратегій
      </PageTitle>
      <ErrorNote error={stats.error ?? backtests.error} />

      <Card title="Наживо: паперові рахунки" className="mb-4"
        hint="Кожен агент торгує своїми 10 000 USDT на реальних цінах, рахунки не скидаються. Вища дохідність вгорі.">
        {live.length ? (
          <>
            <Table head={LIVE_HEAD}>
              {live.map((s, i) => {
                const lean = s.trades < MIN_TRADES;
                return (
                  <tr key={s.agent_id}>
                    <td className="text-slate-500">{i + 1}</td>
                    <td>
                      <span className="font-medium"><Hint text={agentAbout(s.agent_id)}>{agentName(s.agent_id)}</Hint></span>
                      {s.agent_id === championId && <span className="ml-2"><Badge tone="amber">торгує на демо</Badge></span>}
                    </td>
                    <td>{fmtDay(s.started_at)}</td>
                    <td className={`text-right font-semibold ${pnlClass(s.total)}`}>{fmtSignedPct(s.total)}</td>
                    <td className={`text-right ${s.day == null ? "" : pnlClass(s.day)}`}>{fmtSignedPct(s.day)}</td>
                    <td className="text-right">{fmtPct(s.max_drawdown)}</td>
                    <td className={`text-right ${lean ? "text-slate-400" : ""}`}>{s.trades}</td>
                    <td className="text-right">{s.trades ? fmtPct(s.wins / s.trades, 0) : "—"}</td>
                    <td className="text-right">
                      {s.gross_loss > 0 ? fmtNum(s.gross_win / s.gross_loss) : s.gross_win > 0 ? "∞" : "—"}
                    </td>
                    <td>{now ? (() => { const v = verdict(s, now); return <Badge tone={v.tone}>{v.label}</Badge>; })() : "—"}</td>
                    <td className="text-xs text-slate-500 dark:text-slate-400">{holding(s.agent_id).join(", ") || "поза ринком"}</td>
                  </tr>
                );
              })}
            </Table>
            {days > 0 && days < 30 && (
              <div className="mt-3">
                <Note>Змагання триває лише {days} дн. За такий час місця в таблиці вирішує випадок, а не якість стратегії.</Note>
              </div>
            )}
          </>
        ) : <Empty />}
      </Card>

      <Card title="Історичний тест: останні 9 місяців"
        hint={<>Щоночі той самий тест запускається заново, а вікно зсувається на один день{newest ? ` (зараз ${fmtDay(newest.period_start)} – ${fmtDay(newest.period_end)})` : ""}. Тому цифри майже не змінюються день у день: це нормально, це не скидання. Позиції тут менші, ніж на біржі: 10% рахунку.</>}>
        {tests.length ? (
          <Table head={TEST_HEAD}>
            {tests.map((t) => {
              const excess = mean([t.btc?.metrics.excess_vs_buy_hold, t.eth?.metrics.excess_vs_buy_hold]);
              const sortino = mean([t.btc?.metrics.sortino, t.eth?.metrics.sortino]);
              const drawdown = Math.max(t.btc?.metrics.max_drawdown ?? 0, t.eth?.metrics.max_drawdown ?? 0);
              const made = (t.btc ?? t.eth)?.created_at;
              return (
                <tr key={t.id}>
                  <td className="font-medium"><Hint text={agentAbout(t.id)}>{agentName(t.id)}</Hint></td>
                  <td className={`text-right ${pnlClass(t.btc?.metrics.total_return ?? 0)}`}>{fmtSignedPct(t.btc?.metrics.total_return)}</td>
                  <td className={`text-right ${pnlClass(t.eth?.metrics.total_return ?? 0)}`}>{fmtSignedPct(t.eth?.metrics.total_return)}</td>
                  <td className={`text-right ${pnlClass(excess ?? 0)}`}>{fmtSignedPct(excess)}</td>
                  <td className="text-right">{fmtNum(sortino)}</td>
                  <td className="text-right">{fmtPct(drawdown)}</td>
                  <td className="text-right">{(t.btc?.metrics.num_trades ?? 0) + (t.eth?.metrics.num_trades ?? 0)}</td>
                  <td className="text-slate-500 dark:text-slate-400">{fmtDay(made)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Тестів ще немає</Empty>}
      </Card>
    </>
  );
}
