"use client";

import { Card, Empty, ErrorNote, PageTitle, Stat, Table } from "@/components/ui";
import { groupErrors } from "@/lib/errors";
import { fmtTime } from "@/lib/format";
import { fmtIn, fmtWhen, JOBS, nextRun } from "@/lib/schedule";
import type { EventRow, StatusRow } from "@/lib/types";
import { useNow } from "@/lib/useNow";
import { useRows } from "@/lib/useRows";

// How long each part may stay quiet before it is late. GitHub can start a job a few minutes after cron.
const COMPONENTS: Record<string, { label: string; staleMinutes: number }> = {
  trader: { label: "Торговий бот", staleMinutes: 40 },
  trainer: { label: "Нічне навчання", staleMinutes: 26 * 60 },
};

const MODES: Record<string, string> = {
  demo: "торгує на демо-рахунку", "shadow-only": "лише тестові агенти", nightly: "щонічне", weekly: "щотижневе",
};

export default function StatusPage() {
  const now = useNow();
  const status = useRows<StatusRow>("bot_status", { realtime: true });
  const errors = useRows<EventRow>("events", { filters: { type: "error" }, order: { column: "ts" }, limit: 200, realtime: true });
  const groups = groupErrors(errors.rows);

  return (
    <>
      <PageTitle>Стан системи</PageTitle>
      <ErrorNote error={status.error} />
      <div className="mb-4 grid gap-3 md:grid-cols-2">
        {status.rows.length ? status.rows.map((s) => {
          const part = COMPONENTS[s.component] ?? { label: s.component, staleMinutes: 40 };
          const age = !now ? 0 : s.last_step_at ? (now - new Date(s.last_step_at).getTime()) / 60_000 : Infinity;
          const late = age >= part.staleMinutes;
          return (
            <Card key={s.component} title={part.label}>
              <div className="mb-3 text-lg font-semibold">{late ? "🟠 Давно не запускався" : s.last_error ? "🔴 Помилка" : "🟢 Працює"}</div>
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Останній запуск" value={fmtTime(s.last_step_at)} />
                <Stat label="Останній успішний" value={fmtTime(s.last_ok_at)} />
              </div>
              <p className="mt-3 text-sm text-slate-500">Режим: {MODES[s.mode ?? ""] ?? s.mode ?? "—"}</p>
              {s.last_error && <p className="mt-2 break-words text-sm text-rose-600">{s.last_error}</p>}
            </Card>
          );
        }) : <Card><Empty>Бот ще не запускався</Empty></Card>}
      </div>

      <Card title="Наступні запуски" className="mb-4" hint="Час київський. GitHub може запустити завдання на кілька хвилин пізніше.">
        <Table head={["Завдання", "Коли запускається", "Наступний запуск", "Залишилось"]}>
          {JOBS.map((j) => {
            const at = now ? nextRun(j.cron, now) : null;
            return (
              <tr key={j.id}>
                <td className="font-medium">{j.label}</td>
                <td className="text-slate-500 dark:text-slate-400">{j.about}</td>
                <td>{at ? fmtWhen(at, now) : "—"}</td>
                <td className="text-slate-500 dark:text-slate-400">{at ? fmtIn(at, now) : "—"}</td>
              </tr>
            );
          })}
        </Table>
      </Card>

      <Card title="Останні помилки" hint="Однакові помилки зібрані в один рядок.">
        {groups.length ? (
          <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
            {groups.map((g) => (
              <li key={g.sample} className="py-2.5">
                <div className="font-medium">{g.title}{g.count > 1 && <span className="font-normal text-slate-500"> × {g.count}</span>}</div>
                {g.explain && <div className="text-slate-600 dark:text-slate-400">{g.explain}</div>}
                <div className="mt-0.5 text-xs text-slate-500">
                  {g.count > 1 ? `${fmtTime(g.first)} – ${fmtTime(g.last)}` : fmtTime(g.last)}
                </div>
                {!g.explain && <div className="mt-0.5 break-words text-xs text-slate-500">{g.sample}</div>}
              </li>
            ))}
          </ul>
        ) : <Empty>Помилок немає 🎉</Empty>}
      </Card>
    </>
  );
}
