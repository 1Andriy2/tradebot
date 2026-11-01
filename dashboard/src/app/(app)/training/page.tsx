"use client";

import { Card, Empty, ErrorNote, PageTitle, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, fmtTime } from "@/lib/format";
import type { ChampionRow, ModelVersionRow } from "@/lib/types";
import { useRows } from "@/lib/useRows";

const STATUS: Record<ModelVersionRow["status"], string> = {
  champion: "🏆 чемпіон",
  challenger: "претендент",
  retired: "у відставці",
  rejected: "відхилено",
};

export default function TrainingPage() {
  const versions = useRows<ModelVersionRow>("model_versions", { order: { column: "created_at" }, limit: 500, realtime: true });
  const champion = useRows<ChampionRow>("champion", { realtime: true });
  const champ = versions.rows.find((v) => v.status === "champion");

  return (
    <>
      <PageTitle>Прогрес навчання</PageTitle>
      <ErrorNote error={versions.error} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Поточний чемпіон" value={champion.rows[0]?.agent_id ?? "ma_cross"}
          sub={champ ? `версія ${champ.version}` : "базова стратегія"} />
        <Stat label="Sortino чемпіона (OOS)" value={fmtNum(champ?.metrics.sortino)} />
        <Stat label="Версій моделей" value={versions.rows.length} />
        <Stat label="Останнє навчання" value={fmtTime(versions.rows[0]?.created_at)} />
      </div>
      <Card title="Версії моделей (метрики на walk-forward out-of-sample)">
        {versions.rows.length ? (
          <Table head={["Модель", "Версія", "Статус", "Створено", "Дані навчання", "Sortino", "Sharpe", "Дохідність", "Макс. просідання", "Угод"]}>
            {versions.rows.map((v) => (
              <tr key={v.id} className={v.status === "champion" ? "bg-amber-50 dark:bg-amber-950/40" : ""}>
                <td className="font-medium">{v.agent_id}</td>
                <td>v{v.version}</td>
                <td>{STATUS[v.status]}</td>
                <td>{fmtTime(v.created_at)}</td>
                <td>{fmtTime(v.train_start).slice(0, 5)}–{fmtTime(v.train_end).slice(0, 5)}</td>
                <td>{fmtNum(v.metrics.sortino)}</td>
                <td>{fmtNum(v.metrics.sharpe)}</td>
                <td>{fmtPct(v.metrics.total_return)}</td>
                <td>{fmtPct(v.metrics.max_drawdown)}</td>
                <td>{v.metrics.num_trades ?? "—"}</td>
              </tr>
            ))}
          </Table>
        ) : <Empty>Моделі ще не навчались</Empty>}
      </Card>
    </>
  );
}
