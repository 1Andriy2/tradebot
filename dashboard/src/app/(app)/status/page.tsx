"use client";

import { Card, Empty, ErrorNote, PageTitle, Stat } from "@/components/ui";
import { fmtTime } from "@/lib/format";
import type { EventRow, StatusRow } from "@/lib/types";
import { useNow } from "@/lib/useNow";
import { useRows } from "@/lib/useRows";

const STALE_MIN = 40; // cron runs every 15 min; GitHub may delay a few minutes

export default function StatusPage() {
  const now = useNow();
  const status = useRows<StatusRow>("bot_status", { realtime: true });
  const errors = useRows<EventRow>("events", { filters: { type: "error" }, order: { column: "ts" }, limit: 20, realtime: true });

  return (
    <>
      <PageTitle>Стан системи</PageTitle>
      <ErrorNote error={status.error} />
      <div className="mb-4 grid gap-3 md:grid-cols-2">
        {status.rows.length ? status.rows.map((s) => {
          const age = !now ? 0 : s.last_step_at ? (now - new Date(s.last_step_at).getTime()) / 60_000 : Infinity;
          const ok = age < STALE_MIN && !s.last_error;
          return (
            <Card key={s.component} title={s.component === "trader" ? "Торговий бот" : s.component === "trainer" ? "Навчання" : s.component}>
              <div className="mb-3 text-lg font-semibold">{ok ? "🟢 Працює" : age >= STALE_MIN ? "🟠 Давно не запускався" : "🔴 Помилка"}</div>
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Останній крок" value={fmtTime(s.last_step_at)} />
                <Stat label="Останній успішний" value={fmtTime(s.last_ok_at)} />
              </div>
              <p className="mt-3 text-sm text-slate-500">Режим: {s.mode ?? "—"}</p>
              {s.last_error && <p className="mt-2 break-words text-sm text-rose-600">{s.last_error}</p>}
            </Card>
          );
        }) : <Card><Empty>Бот ще не запускався</Empty></Card>}
      </div>
      <Card title="Останні помилки">
        {errors.rows.length ? (
          <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
            {errors.rows.map((e) => (
              <li key={e.id} className="py-2">
                <span className="text-slate-500">{fmtTime(e.ts)}</span> — <span className="break-words">{e.message}</span>
              </li>
            ))}
          </ul>
        ) : <Empty>Помилок немає 🎉</Empty>}
      </Card>
    </>
  );
}
