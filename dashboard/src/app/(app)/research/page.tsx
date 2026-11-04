"use client";

import { useMemo, useState } from "react";
import { Card, Empty, ErrorNote, PageTitle, Select, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, fmtTime, pnlClass } from "@/lib/format";
import type { EventRow, PatternRow, ResearchRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const ALL = "__all__";
const DOW = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"];

type SortKey = "checks_passed" | "dsr" | "sortino" | "total_return" | "fold_win_rate" | "excess_vs_buy_hold";
const SORTS: { value: SortKey; label: string }[] = [
  { value: "checks_passed", label: "Пройдені перевірки" },
  { value: "dsr", label: "Deflated Sharpe (надійність)" },
  { value: "fold_win_rate", label: "Стабільність по фолдах" },
  { value: "sortino", label: "Sortino" },
  { value: "total_return", label: "Дохідність" },
  { value: "excess_vs_buy_hold", label: "Краще за Buy & Hold" },
];

/** Missing statistics stay visible as a dash: they mean "could not be computed", not zero. */
function optNum(v: number | null | undefined, digits = 2): string {
  return v == null ? "—" : fmtNum(v, digits);
}

function optPct(v: number | null | undefined): string {
  return v == null ? "—" : fmtPct(v, 0);
}

interface ScenarioDetails {
  symbol: string;
  horizon: string;
  n: number;
  p_up: number;
  base_p_up: number;
  q10: number;
  q50: number;
  q90: number;
  typical_worst: number;
  typical_best: number;
}

interface StudyDetails {
  symbol: string;
  test: string;
  horizon_hours: number;
  n: number;
  mean_after?: number;
  base_mean?: number;
  q_value?: number;
}

const STUDY_LABELS: Record<string, string> = {
  long_liquidation_spike: "Сплеск ліквідацій лонгів",
  short_liquidation_spike: "Сплеск ліквідацій шортів",
  open_interest_jump: "Стрибок open interest",
  crowded_longs: "Перекіс у лонги",
  good_news: "Хороші новини",
  bad_news: "Погані новини",
};

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
  const scenarios = useRows<EventRow>("events", { filters: { type: "scenario" }, order: { column: "ts" }, limit: 12 });
  const studies = useRows<EventRow>("events", { filters: { type: "event_study" }, order: { column: "ts" }, limit: 40 });
  const [tf, setTf] = useState(ALL);
  const [symbol, setSymbol] = useState(ALL);
  const [sort, setSort] = useState<SortKey>("checks_passed");

  const runId = results.rows[0]?.run_id;
  const latest = useMemo(() => results.rows.filter((r) => r.run_id === runId), [results.rows, runId]);
  const rows = useMemo(() => latest
    .filter((r) => (tf === ALL || r.timeframe === tf) && (symbol === ALL || r.symbol === symbol))
    .sort((a, b) => (b.metrics[sort] ?? -Infinity) - (a.metrics[sort] ?? -Infinity)), [latest, tf, symbol, sort]);

  const patRun = patterns.rows[0]?.run_id;
  const pats = patterns.rows
    .filter((p) => p.run_id === patRun && p.q_value < 0.2)
    .sort((a, b) => Number(b.stable) - Number(a.stable) || a.q_value - b.q_value);

  const scenarioTime = scenarios.rows[0]?.ts;
  const outlook = scenarios.rows.filter((e) => e.ts === scenarioTime);

  const studyTime = studies.rows[0]?.ts;
  const studyRows = studies.rows.filter((e) => e.ts === studyTime);

  const profitable = latest.filter((r) => r.metrics.total_return > 0).length;
  const reliable = latest.filter((r) => r.metrics.dsr >= 0.95).length;
  const robust = latest.filter((r) => r.metrics.robust === 1).length;
  const uniq = (f: (r: ResearchRow) => string) => [...new Set(latest.map(f))].sort();

  return (
    <>
      <PageTitle>Дослідження стратегій і патернів</PageTitle>
      <ErrorNote error={results.error ?? patterns.error ?? scenarios.error ?? studies.error} />
      <p className="mb-4 text-sm text-slate-500">
        Усе нижче — out-of-sample: параметри кожної стратегії обираються на попередніх 12 місяцях і
        перевіряються на наступних 3 (walk-forward). Комісії, проковзування та funding враховані.
        Deflated Sharpe ≥ 0.95 означає, що результат навряд чи випадковий з урахуванням кількості перевірених варіантів.
      </p>
      <p className="mb-4 text-sm text-slate-500">
        «Стійка» стратегія прибуткова й пройшла всі доступні перевірки, яких має бути щонайменше чотири: DSR, значущість сигналу (q),
        PBO (ризик, що переможець просто вгаданий), шляхи CPCV, сусідні параметри, інша пара, інші таймфрейми, Monte Carlo.
        Прочерк означає, що показник порахувати не вдалося. Ці показники лише інформують і не змінюють, хто торгує.
      </p>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Останній прогін" value={fmtTime(latest[0]?.created_at)} />
        <Stat label="Комбінацій перевірено" value={latest.length}
          sub={latest[0] ? `${latest[0].metrics.n_trials} варіантів параметрів` : undefined} />
        <Stat label="Прибуткових (OOS)" value={`${profitable} / ${latest.length}`} />
        <Stat label="Надійних (DSR ≥ 0.95)" value={reliable} />
        <Stat label="Стійких (усі перевірки)" value={robust} />
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
            "Фолди +", "Фолди", "DSR", "Перевірки", "Сигнал q", "PBO", "PBO всіх", "CPCV +", "Сусідні парам.", "MC збиток", "MC просід. ≥20%",
            "Ulcer", "CVaR 5%", "Комісії", "Funding", "Останні параметри"]}>
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
                  <td className={m.robust === 1 ? "font-semibold text-emerald-600" : ""}>
                    {m.checks_total == null ? "—" : `${m.checks_passed} / ${m.checks_total}`}
                  </td>
                  <td>{optNum(m.signal_q, 3)}</td>
                  <td>{optNum(m.pbo)}</td>
                  <td>{optNum(m.pbo_universe)}</td>
                  <td>{optPct(m.cpcv_positive_paths)}</td>
                  <td>{optPct(m.param_stability)}</td>
                  <td>{optPct(m.mc_loss_prob)}</td>
                  <td>{optPct(m.mc_ruin_prob)}</td>
                  <td>{m.ulcer_index == null ? "—" : fmtPct(m.ulcer_index)}</td>
                  <td>{m.cvar_5 == null ? "—" : fmtPct(m.cvar_5)}</td>
                  <td>{fmtNum(m.fees_paid, 0)}</td>
                  <td>{fmtNum(m.funding_paid, 0)}</td>
                  <td className="max-w-xs truncate text-xs text-slate-500">{paramsText(r.params.latest)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Дослідження ще не запускалось</Empty>}
      </Card>

      <Card title="Сценарії: що було після схожих моментів в історії" className="mb-4">
        <p className="mb-3 text-xs text-slate-500">
          Це опис минулого, а не прогноз. Чи має він прогнозну силу, показує рядок стратегії analog у таблиці вище.
          Порівнюйте «Частка росту» з базовою: якщо вони майже однакові, поточний стан нічого не додає.
        </p>
        {outlook.length ? (
          <Table head={["Пара", "Горизонт", "Схожих моментів", "Частка росту", "Базова", "Погані 10%", "Медіана", "Добрі 10%", "Типове падіння", "Типовий ріст"]}>
            {outlook.map((e) => {
              const d = e.details as unknown as ScenarioDetails;
              return (
                <tr key={e.id}>
                  <td>{d.symbol.replace(":USDT", "")}</td>
                  <td>{d.horizon}</td>
                  <td>{d.n}</td>
                  <td>{fmtPct(d.p_up, 0)}</td>
                  <td>{fmtPct(d.base_p_up, 0)}</td>
                  <td className={pnlClass(d.q10)}>{fmtPct(d.q10, 1)}</td>
                  <td className={pnlClass(d.q50)}>{fmtPct(d.q50, 1)}</td>
                  <td className={pnlClass(d.q90)}>{fmtPct(d.q90, 1)}</td>
                  <td className={pnlClass(d.typical_worst)}>{fmtPct(d.typical_worst, 1)}</td>
                  <td className={pnlClass(d.typical_best)}>{fmtPct(d.typical_best, 1)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Сценарії зʼявляться після наступного нічного дослідження</Empty>}
      </Card>

      <Card title="Перевірка гіпотез: ліквідації, open interest, новини" className="mb-4">
        <p className="mb-3 text-xs text-slate-500">
          Чи рухається ціна інакше після таких подій, ніж зазвичай. Дані збираються від листопада 2026, тому спершу майже всюди буде
          «мало даних»: тест починається від 30 окремих подій. Значущим вважається q &lt; 0.05. Торгівлю ці рядки не змінюють.
        </p>
        {studyRows.length ? (
          <Table head={["Пара", "Подія", "Горизонт", "Подій", "Після події", "Зазвичай", "q"]}>
            {studyRows.map((e) => {
              const d = e.details as unknown as StudyDetails;
              return (
                <tr key={e.id}>
                  <td>{d.symbol.replace(":USDT", "")}</td>
                  <td>{STUDY_LABELS[d.test] ?? d.test}</td>
                  <td>{d.horizon_hours} год</td>
                  <td>{d.n}</td>
                  <td className={d.mean_after == null ? "" : pnlClass(d.mean_after)}>
                    {d.mean_after == null ? "мало даних" : fmtPct(d.mean_after, 2)}
                  </td>
                  <td>{d.base_mean == null ? "—" : fmtPct(d.base_mean, 2)}</td>
                  <td className={d.q_value != null && d.q_value < 0.05 ? "font-semibold text-emerald-600" : ""}>{optNum(d.q_value, 3)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Перші результати зʼявляться після наступного нічного дослідження</Empty>}
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
