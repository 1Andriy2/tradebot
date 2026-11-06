"use client";

import { Badge, Card, Empty, ErrorNote, Hint, Note, PageTitle, Stat, Table } from "@/components/ui";
import { agentName } from "@/lib/agents";
import { fmtDay, fmtNum, fmtPct, fmtSignedPct, fmtTime, pnlClass } from "@/lib/format";
import { fmtIn, fmtWhen, job, nextRun } from "@/lib/schedule";
import type { AgentRow, ChampionRow, ModelVersionRow } from "@/lib/types";
import { useNow } from "@/lib/useNow";
import { useRows } from "@/lib/useRows";

// The promotion rules of src/tradebot/registry.py.
const MARGIN = 0.05;
const MAX_DRAWDOWN = 0.2;
const MIN_TRADES = 30;

const STATUS: Record<ModelVersionRow["status"], { label: string; tone: "amber" | "grey" | "red" | "blue" }> = {
  champion: { label: "🏆 торгує", tone: "amber" },
  challenger: { label: "претендент", tone: "blue" },
  retired: { label: "замінена", tone: "grey" },
  rejected: { label: "не пройшла", tone: "red" },
};

const HEAD = [
  "Модель", "Статус", "Навчена",
  { label: "Дані", hint: "Період цін, на якому модель вчилась." },
  { label: "Sortino", right: true, hint: "Дохідність відносно падінь на перевірці. Вище нуля добре, нижче нуля модель втрачала." },
  { label: "Дохідність", right: true, hint: "Результат на останніх місяцях, яких модель не бачила під час навчання." },
  { label: "Просідання", right: true, hint: `Найбільше падіння на перевірці. Має бути менше ${fmtPct(MAX_DRAWDOWN, 0)}.` },
  { label: "Угод", right: true, hint: `На перевірці. Потрібно щонайменше ${MIN_TRADES}.` },
  { label: "Чому таке рішення", hint: "Які правила не виконані. Модель замінює чемпіона, лише якщо виконує всі." },
];

/** What kept a version from the demo account, in the order the rules are checked. */
function verdict(v: ModelVersionRow): string {
  if (v.status === "champion") return "Виконала всі правила";
  if (v.status === "retired") return "Її замінила краща версія";
  const m = v.metrics;
  const reasons = [];
  if (!(m.score > m.champion_score + MARGIN)) reasons.push(`Sortino ${fmtNum(m.score)} не кращий за чемпіона (${fmtNum(m.champion_score)})`);
  if (!(m.num_trades >= MIN_TRADES)) reasons.push(`лише ${m.num_trades} угод`);
  if (!(m.max_drawdown < MAX_DRAWDOWN)) reasons.push(`просідання ${fmtPct(m.max_drawdown, 1)}`);
  return reasons.join(", ") || "—";
}

function Schedule({ name, when, next, last, children }: {
  name: string;
  when: string;
  next?: string;
  last?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
      <div className="font-semibold">{name}</div>
      <div className="mt-0.5 text-sm text-slate-700 dark:text-slate-300">{when}</div>
      {next && <div className="mt-2 text-sm"><span className="text-slate-500 dark:text-slate-400">Наступне:</span> {next}</div>}
      {last && <div className="text-sm"><span className="text-slate-500 dark:text-slate-400">Останнє:</span> {last}</div>}
      <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{children}</p>
    </div>
  );
}

export default function TrainingPage() {
  const now = useNow();
  const versions = useRows<ModelVersionRow>("model_versions", { order: { column: "created_at" }, limit: 500, realtime: true });
  const champion = useRows<ChampionRow>("champion", { realtime: true });
  const agents = useRows<AgentRow>("agents");
  const champ = versions.rows.find((v) => v.status === "champion");
  const lastOf = (id: string) => versions.rows.find((v) => v.agent_id === id);
  const lgbm = lastOf("lgbm");
  const rl = lastOf("rl");
  const planned = (id: string) => {
    if (!now) return undefined;
    const at = nextRun(job(id).cron, now);
    return `${fmtWhen(at, now)} (${fmtIn(at, now)})`;
  };
  const fixed = agents.rows.filter((a) => a.kind === "baseline" && !a.id.startsWith("news_")).length;
  const rejected = versions.rows.filter((v) => v.status === "rejected").length;

  return (
    <>
      <PageTitle intro="Навчаються лише дві моделі, кожна за своїм розкладом. Нова версія торгує на демо-рахунку, тільки якщо на перевірці краща за поточного чемпіона.">
        Навчання моделей
      </PageTitle>
      <ErrorNote error={versions.error} />

      <div className="mb-4 grid gap-3 md:grid-cols-3">
        <Schedule name="LightGBM" when="Щоночі, одна нова версія" next={planned("retrain")}
          last={lgbm ? `v${lgbm.version}, ${fmtTime(lgbm.created_at)}` : undefined}>
          Тому в таблиці щодня зʼявляється один рядок: це та сама модель, навчена заново на свіжих даних.
        </Schedule>
        <Schedule name="RL-агент" when="Щонеділі, одна нова версія" next={planned("rl")}
          last={rl ? `v${rl.version}, ${fmtTime(rl.created_at)}` : undefined}>
          Вчиться значно довше, тому раз на тиждень.
        </Schedule>
        <Schedule name={`Інші стратегії${fixed ? `: ${fixed}` : ""}`} when="Не навчаються">
          Це фіксовані правила без моделі. Щоночі для них лише перераховується історичний тест у Рейтингу.
        </Schedule>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Хто торгує на демо" value={agentName(champion.rows[0]?.agent_id ?? "ma_cross")}
          sub={champ ? `версія ${champ.version}, з ${fmtDay(champ.promoted_at ?? champ.created_at)}` : "базова стратегія"} />
        <Stat label={<Hint text="Дохідність відносно падінь на перевірці. Вище нуля добре.">Sortino чемпіона</Hint>}
          value={fmtNum(champ?.metrics.sortino)} tone={champ ? pnlClass(champ.metrics.sortino) : ""} />
        <Stat label="Усього версій" value={versions.rows.length} />
        <Stat label="Не пройшли перевірку" value={`${rejected} з ${versions.rows.length}`} />
      </div>

      {champ && champ.metrics.sortino < 0 && (
        <div className="mb-4">
          <Note tone="warn">
            Чемпіон обраний як найменш збитковий, а не як прибутковий: Sortino нижче нуля означає, що на перевірці модель втрачала гроші.
            Поки жодна версія не показала прибутку.
          </Note>
        </div>
      )}

      <Card title="Версії моделей, новіші вгорі"
        hint={`Щоб стати чемпіоном, версія має мати Sortino вищий за чемпіона щонайменше на ${MARGIN}, хоча б ${MIN_TRADES} угод і просідання менше ${fmtPct(MAX_DRAWDOWN, 0)}. Усі цифри з останніх місяців, яких модель не бачила під час навчання.`}>
        {versions.rows.length ? (
          <Table head={HEAD}>
            {versions.rows.map((v) => (
              <tr key={v.id} className={v.status === "champion" ? "bg-amber-50 dark:bg-amber-950/30" : ""}>
                <td className="font-medium">{agentName(v.agent_id)} v{v.version}</td>
                <td><Badge tone={STATUS[v.status].tone}>{STATUS[v.status].label}</Badge></td>
                <td>{fmtTime(v.created_at)}</td>
                <td className="text-xs text-slate-500 dark:text-slate-400">{fmtDay(v.train_start)} – {fmtDay(v.train_end)}</td>
                <td className="text-right">{fmtNum(v.metrics.sortino)}</td>
                <td className={`text-right ${pnlClass(v.metrics.total_return ?? 0)}`}>{fmtSignedPct(v.metrics.total_return)}</td>
                <td className="text-right">{fmtPct(v.metrics.max_drawdown)}</td>
                <td className="text-right">{v.metrics.num_trades ?? "—"}</td>
                <td className="whitespace-normal text-xs text-slate-600 dark:text-slate-400">{verdict(v)}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Моделі ще не навчались</Empty>}
      </Card>
    </>
  );
}
