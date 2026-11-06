"use client";

import { useMemo, useState } from "react";
import { Card, Empty, ErrorNote, Hint, PageTitle, Select, Stat, Table, type Column } from "@/components/ui";
import { agentAbout, agentName, pairName } from "@/lib/agents";
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

const STRATEGY_HEAD: Column[] = [
  "Стратегія", "Пара",
  { label: "ТФ", hint: "Таймфрейм: тривалість однієї свічки." },
  { label: "Дохідність", right: true, hint: "Результат на даних, яких стратегія не бачила під час підбору параметрів, з комісіями." },
  { label: "Проти утримання", right: true, hint: "Наскільки краще чи гірше за просте утримання монети." },
  { label: "Sortino", right: true, hint: "Дохідність відносно падінь. Вище нуля добре, вище 1 вже цікаво." },
  { label: "Просідання", right: true, hint: "Найбільше падіння рахунку від піку." },
  { label: "Угод", right: true, hint: "Менше 30 угод ще нічого не доводять." },
  { label: "Періоди", hint: "Кожен стовпчик це окремий період перевірки: зелений у плюсі, червоний у мінусі. Краще, коли переважно зелені." },
  { label: "DSR", right: true, hint: "Deflated Sharpe: імовірність, що результат не випадковий, з поправкою на кількість перепробуваних варіантів. Надійно від 0,95." },
  { label: "Перевірки", right: true, hint: "Скільки перевірок на надійність стратегія пройшла. «Стійка» проходить усі, і їх щонайменше чотири." },
];

const DETAIL_HEAD: Column[] = [
  { label: "Успішних", right: true, hint: "Частка прибуткових угод." },
  { label: "Profit factor", right: true, hint: "Сума прибутків, поділена на суму збитків. Вище 1 добре." },
  { label: "Періодів у плюсі", right: true, hint: "Частка періодів перевірки з прибутком." },
  { label: "Сигнал q", right: true, hint: "Імовірність, що сигнал випадковий, з поправкою на кількість перевірок. Значуще нижче 0,05." },
  { label: "PBO", right: true, hint: "Ризик, що переможець просто вгаданий на минулому. Менше 0,5 добре." },
  { label: "PBO всіх", right: true, hint: "Той самий ризик для всіх стратегій разом." },
  { label: "CPCV +", right: true, hint: "Частка альтернативних історій, де стратегія заробила. Вище краще." },
  { label: "Сусідні параметри", right: true, hint: "Частка схожих налаштувань, які теж працюють. Низька означає, що результат тримається на випадковому числі." },
  { label: "Ризик збитку", right: true, hint: "Monte Carlo: імовірність закінчити в мінусі, якщо перемішати порядок угод." },
  { label: "Ризик просідання", right: true, hint: "Monte Carlo: імовірність просідання 20% і більше." },
  { label: "Ulcer", right: true, hint: "Наскільки глибокими й довгими були просідання. Менше краще." },
  { label: "CVaR 5%", right: true, hint: "Середній збиток у 5% найгірших випадків." },
  { label: "Комісії", right: true, hint: "Сплачені комісії, USDT." },
  { label: "Funding", right: true, hint: "Плата за утримання ф'ючерса, USDT." },
  { label: "Параметри", hint: "Налаштування стратегії в останньому періоді." },
];

const SCENARIO_HEAD: Column[] = [
  "Пара", "Горизонт",
  { label: "Схожих моментів", right: true, hint: "Скільки разів у минулому ринок виглядав подібно." },
  { label: "Частка росту", right: true, hint: "Як часто після таких моментів ціна зростала." },
  { label: "Зазвичай", right: true, hint: "Як часто ціна зростає в будь-який момент. Якщо майже те саме, поточний стан нічого не додає." },
  { label: "Погані 10%", right: true, hint: "Межа 10% найгірших результатів." },
  { label: "Медіана", right: true, hint: "Типовий результат: половина випадків вища, половина нижча." },
  { label: "Добрі 10%", right: true, hint: "Межа 10% найкращих результатів." },
  { label: "Типове падіння", right: true, hint: "Наскільки ціна зазвичай просідала дорогою." },
  { label: "Типовий ріст", right: true, hint: "Наскільки ціна зазвичай піднімалась дорогою." },
];

const STUDY_HEAD: Column[] = [
  "Пара", "Подія",
  { label: "Горизонт", hint: "За скільки годин після події міряється зміна ціни." },
  { label: "Подій", right: true, hint: "Окремі випадки. Близькі в часі рахуються як один." },
  { label: "Після події", right: true, hint: "Середня зміна ціни після події." },
  { label: "Зазвичай", right: true, hint: "Середня зміна за такий самий час у будь-який момент." },
  { label: "q", right: true, hint: "Імовірність, що різниця випадкова, з поправкою на кількість перевірок. Значуще нижче 0,05." },
];

const PATTERN_HEAD: Column[] = [
  "Пара", "Тип",
  { label: "Коли (UTC)", hint: "Час за Гринвічем. Київ зараз на 2 години попереду." },
  { label: "Сер. дохідність", right: true, hint: "Середня зміна ціни в цю годину чи день." },
  { label: "t", right: true, hint: "Сила ефекту відносно шуму. Понад 2 за модулем вже помітно." },
  { label: "q", right: true, hint: "Імовірність випадковості з поправкою на кількість перевірок. Значуще нижче 0,05." },
  { label: "Спостережень", right: true, hint: "Скільки годин чи днів у вибірці." },
  { label: "Стабільний", hint: "Той самий знак і значущість в обох половинах історії." },
];

const MIN_EVENTS = 30;

const STUDY_LABELS: Record<string, string> = {
  long_liquidation_spike: "Сплеск ліквідацій лонгів",
  short_liquidation_spike: "Сплеск ліквідацій шортів",
  open_interest_jump: "Стрибок open interest",
  crowded_longs: "Перекіс у лонги",
  good_news: "Хороші новини",
  bad_news: "Погані новини",
  llm_bad_news: "Сильні погані новини (оцінка моделі)",
  llm_good_news: "Сильні хороші новини (оцінка моделі)",
  filter_cut_longs: "Фільтр новин обмежив лонги",
  filter_cut_shorts: "Фільтр новин обмежив шорти",
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
  const studies = useRows<EventRow>("events", { filters: { type: "event_study" }, order: { column: "ts" }, limit: 100 });
  const [tf, setTf] = useState(ALL);
  const [symbol, setSymbol] = useState(ALL);
  const [sort, setSort] = useState<SortKey>("checks_passed");
  const [details, setDetails] = useState(false);

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
  const testedStudies = studyRows.filter((e) => (e.details as unknown as StudyDetails).mean_after != null);
  const waiting = studyRows
    .map((e) => e.details as unknown as StudyDetails)
    .filter((d) => d.mean_after == null)
    .sort((a, b) => b.n - a.n);

  const profitable = latest.filter((r) => r.metrics.total_return > 0).length;
  const reliable = latest.filter((r) => r.metrics.dsr >= 0.95).length;
  const robust = latest.filter((r) => r.metrics.robust === 1).length;
  const uniq = (f: (r: ResearchRow) => string) => [...new Set(latest.map(f))].sort();

  return (
    <>
      <PageTitle intro="Щоночі бот перевіряє десятки стратегій на історії, якої вони не бачили під час підбору параметрів, з комісіями. Наведіть на назву колонки, щоб побачити пояснення. Ці дані лише інформують і не змінюють, хто торгує.">
        Дослідження
      </PageTitle>
      <ErrorNote error={results.error ?? patterns.error ?? scenarios.error ?? studies.error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Останній прогін" value={fmtTime(latest[0]?.created_at)} />
        <Stat label="Комбінацій перевірено" value={latest.length}
          sub={latest[0] ? `${latest[0].metrics.n_trials} варіантів параметрів` : undefined} />
        <Stat label="Прибуткових (OOS)" value={`${profitable} / ${latest.length}`} />
        <Stat label="Надійних (DSR ≥ 0.95)" value={reliable} />
        <Stat label="Стійких (усі перевірки)" value={robust} />
      </div>

      <Card title="Стратегії" className="mb-4"
        hint="Параметри обираються на 12 місяцях і перевіряються на наступних 3, і так крок за кроком по всій історії.">
        <div className="mb-3 grid grid-cols-2 items-end gap-3 md:grid-cols-4">
          <Select label="Таймфрейм" value={tf} onChange={setTf}
            options={[{ value: ALL, label: "Усі" }, ...uniq((r) => r.timeframe).map((v) => ({ value: v, label: v }))]} />
          <Select label="Пара" value={symbol} onChange={setSymbol}
            options={[{ value: ALL, label: "Усі" }, ...uniq((r) => r.symbol).map((v) => ({ value: v, label: pairName(v) }))]} />
          <Select label="Сортувати за" value={sort} onChange={(v) => setSort(v as SortKey)} options={SORTS} />
          <label className="flex items-center gap-2 pb-1.5 text-sm">
            <input type="checkbox" checked={details} onChange={(e) => setDetails(e.target.checked)} />
            Усі показники
          </label>
        </div>
        {rows.length ? (
          <Table head={details ? [...STRATEGY_HEAD, ...DETAIL_HEAD] : STRATEGY_HEAD}>
            {rows.map((r) => {
              const m = r.metrics;
              return (
                <tr key={r.id}>
                  <td className="font-medium"><Hint text={agentAbout(r.family) || r.family}>{agentName(r.family)}</Hint></td>
                  <td>{pairName(r.symbol)}</td>
                  <td>{r.timeframe}</td>
                  <td className={`text-right ${pnlClass(m.total_return)}`}>{fmtPct(m.total_return)}</td>
                  <td className={`text-right ${pnlClass(m.excess_vs_buy_hold)}`}>{fmtPct(m.excess_vs_buy_hold)}</td>
                  <td className="text-right">{fmtNum(m.sortino)}</td>
                  <td className="text-right">{fmtPct(m.max_drawdown)}</td>
                  <td className="text-right">{m.num_trades}</td>
                  <td><FoldBars values={r.fold_returns ?? []} /></td>
                  <td className={`text-right ${m.dsr >= 0.95 ? "font-semibold text-emerald-600" : ""}`}>{fmtNum(m.dsr)}</td>
                  <td className={`text-right ${m.robust === 1 ? "font-semibold text-emerald-600" : ""}`}>
                    {m.checks_total == null ? "—" : `${m.checks_passed} / ${m.checks_total}`}
                  </td>
                  {details && (
                    <>
                      <td className="text-right">{fmtPct(m.win_rate, 0)}</td>
                      <td className="text-right">{m.profit_factor == null ? "∞" : fmtNum(m.profit_factor)}</td>
                      <td className="text-right">{fmtPct(m.fold_win_rate, 0)}</td>
                      <td className="text-right">{optNum(m.signal_q, 3)}</td>
                      <td className="text-right">{optNum(m.pbo)}</td>
                      <td className="text-right">{optNum(m.pbo_universe)}</td>
                      <td className="text-right">{optPct(m.cpcv_positive_paths)}</td>
                      <td className="text-right">{optPct(m.param_stability)}</td>
                      <td className="text-right">{optPct(m.mc_loss_prob)}</td>
                      <td className="text-right">{optPct(m.mc_ruin_prob)}</td>
                      <td className="text-right">{m.ulcer_index == null ? "—" : fmtPct(m.ulcer_index)}</td>
                      <td className="text-right">{m.cvar_5 == null ? "—" : fmtPct(m.cvar_5)}</td>
                      <td className="text-right">{fmtNum(m.fees_paid, 0)}</td>
                      <td className="text-right">{fmtNum(m.funding_paid, 0)}</td>
                      <td className="max-w-xs truncate text-xs text-slate-500">{paramsText(r.params.latest)}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Дослідження ще не запускалось</Empty>}
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Прочерк означає, що показник порахувати не вдалося, а не нуль.</p>
      </Card>

      <Card title="Сценарії: що було після схожих моментів в історії" className="mb-4"
        hint="Це опис минулого, а не прогноз. Порівнюйте «Частка росту» із «Зазвичай»: якщо майже однакові, поточний стан нічого не додає.">
        {outlook.length ? (
          <Table head={SCENARIO_HEAD}>
            {outlook.map((e) => {
              const d = e.details as unknown as ScenarioDetails;
              return (
                <tr key={e.id}>
                  <td>{pairName(d.symbol)}</td>
                  <td>{d.horizon.replace("h", " год")}</td>
                  <td className="text-right">{d.n}</td>
                  <td className="text-right">{fmtPct(d.p_up, 0)}</td>
                  <td className="text-right">{fmtPct(d.base_p_up, 0)}</td>
                  <td className={`text-right ${pnlClass(d.q10)}`}>{fmtPct(d.q10, 1)}</td>
                  <td className={`text-right ${pnlClass(d.q50)}`}>{fmtPct(d.q50, 1)}</td>
                  <td className={`text-right ${pnlClass(d.q90)}`}>{fmtPct(d.q90, 1)}</td>
                  <td className={`text-right ${pnlClass(d.typical_worst)}`}>{fmtPct(d.typical_worst, 1)}</td>
                  <td className={`text-right ${pnlClass(d.typical_best)}`}>{fmtPct(d.typical_best, 1)}</td>
                </tr>
              );
            })}
          </Table>
        ) : <Empty>Сценарії зʼявляться після наступного нічного дослідження</Empty>}
      </Card>

      <Card title="Перевірка гіпотез: ліквідації, open interest, новини" className="mb-4"
        hint={`Чи рухається ціна інакше після таких подій, ніж зазвичай. Тест починається від ${MIN_EVENTS} окремих подій, значущим вважається q < 0,05. Торгівлю ці рядки не змінюють.`}>
        {testedStudies.length ? (
          <Table head={STUDY_HEAD}>
            {testedStudies.map((e) => {
              const d = e.details as unknown as StudyDetails;
              return (
                <tr key={e.id}>
                  <td>{pairName(d.symbol)}</td>
                  <td>{STUDY_LABELS[d.test] ?? d.test}</td>
                  <td>{d.horizon_hours} год</td>
                  <td className="text-right">{d.n}</td>
                  <td className={`text-right ${pnlClass(d.mean_after ?? 0)}`}>{fmtPct(d.mean_after, 2)}</td>
                  <td className="text-right">{d.base_mean == null ? "—" : fmtPct(d.base_mean, 2)}</td>
                  <td className={`text-right ${d.q_value != null && d.q_value < 0.05 ? "font-semibold text-emerald-600" : ""}`}>{optNum(d.q_value, 3)}</td>
                </tr>
              );
            })}
          </Table>
        ) : studyRows.length ? <Empty>Жодна гіпотеза ще не набрала {MIN_EVENTS} подій</Empty>
          : <Empty>Перші результати зʼявляться після наступного нічного дослідження</Empty>}
        {waiting.length > 0 && (
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-slate-500 hover:text-slate-900 dark:hover:text-white">
              Ще збирають дані: {waiting.length}
            </summary>
            <ul className="mt-2 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
              {waiting.map((d) => (
                <li key={`${d.symbol}-${d.test}-${d.horizon_hours}`} className="flex justify-between gap-3">
                  <span>{pairName(d.symbol)}, {STUDY_LABELS[d.test] ?? d.test}, {d.horizon_hours} год</span>
                  <span className="tabular-nums text-slate-500">{d.n} / {MIN_EVENTS}</span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </Card>

      <Card title="Повторювані патерни: година доби й день тижня"
        hint="Лише ті, що пройшли поправку на множинні перевірки.">
        {pats.length ? (
          <Table head={PATTERN_HEAD}>
            {pats.map((p) => (
              <tr key={p.id}>
                <td>{pairName(p.symbol)}</td>
                <td>{p.kind === "hour" ? "година" : "день"}</td>
                <td>{p.kind === "hour" ? `${String(p.bucket).padStart(2, "0")}:00` : DOW[p.bucket]}</td>
                <td className={`text-right ${pnlClass(p.mean_return)}`}>{fmtPct(p.mean_return, 3)}</td>
                <td className="text-right">{fmtNum(p.t_stat)}</td>
                <td className="text-right">{fmtNum(p.q_value, 3)}</td>
                <td className="text-right">{p.n}</td>
                <td>{p.stable ? "✅" : "—"}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Значущих патернів не знайдено</Empty>}
      </Card>
    </>
  );
}
