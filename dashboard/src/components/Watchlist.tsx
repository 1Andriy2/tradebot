"use client";

import { type FormEvent, useState } from "react";
import { fmtMoney } from "@/lib/news";
import { supabaseBrowser } from "@/lib/supabase/client";
import type { AssetCandidate, WatchRow } from "@/lib/types";

const KINDS = { crypto: "Монета", stock: "Акція" };

function Candidate({ item, onPick, busy }: { item: AssetCandidate; onPick: () => void; busy: boolean }) {
  return (
    <li>
      <button
        type="button"
        disabled={busy}
        onClick={onPick}
        className="flex w-full flex-wrap items-baseline justify-between gap-x-4 gap-y-1 rounded-lg border border-slate-200 p-3 text-left hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
      >
        <span className="min-w-0">
          <span className="font-semibold">{item.name}</span>
          <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">
            {KINDS[item.kind]}{item.rank ? `, #${item.rank}` : ""}{item.exchange ? `, ${item.exchange}` : ""}
          </span>
        </span>
        <span className="flex gap-4 text-sm tabular-nums text-slate-600 dark:text-slate-300">
          <span>{fmtMoney(item.price, item.currency)}</span>
          {item.kind === "crypto" && (
            <>
              <span title="Капіталізація">кап. {fmtMoney(item.market_cap, "USD", true)}</span>
              <span title="Обсяг за добу">обс. {fmtMoney(item.volume, "USD", true)}</span>
            </>
          )}
        </span>
      </button>
    </li>
  );
}

export function Watchlist({ rows, onSubscribed, onRemoved }: {
  rows: WatchRow[];
  onSubscribed: (ticker: string) => void;
  onRemoved: (ticker: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [choices, setChoices] = useState<AssetCandidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  async function subscribe(item: AssetCandidate) {
    setBusy(true);
    const { error } = await supabaseBrowser().from("watchlist").insert({
      id: item.id, kind: item.kind, ticker: item.ticker, name: item.name, source_id: item.source_id,
    });
    setBusy(false);
    if (error && error.code !== "23505") { // already followed: nothing to do
      setNote(`Не вдалося підписатися: ${error.message}`);
      return;
    }
    setChoices([]);
    setQuery("");
    setNote(`${item.name}: збір новин запущено, перші з'являться за хвилину-дві`);
    onSubscribed(item.ticker);
  }

  async function search(event: FormEvent) {
    event.preventDefault();
    setChoices([]);
    setNote(null);
    setBusy(true);
    try {
      const response = await fetch(`/api/assets?q=${encodeURIComponent(query.trim())}`);
      const body = (await response.json()) as { candidates?: AssetCandidate[]; partial?: boolean; error?: string };
      const found = body.candidates ?? [];
      if (!response.ok) setNote(body.error ?? "Пошук не вдався");
      else if (found.length === 1 && !body.partial) await subscribe(found[0]);
      else if (found.length) setChoices(found);
      else setNote(body.partial ? "Джерело тимчасово недоступне, спробуйте ще раз" : "Такий тікер не знайдено");
    } catch {
      setNote("Пошук не вдався, спробуйте ще раз");
    }
    setBusy(false);
  }

  async function remove(row: WatchRow) {
    const { error } = await supabaseBrowser().from("watchlist").delete().eq("id", row.id);
    if (error) setNote(`Не вдалося відписатися: ${error.message}`);
    else onRemoved(row.ticker);
  }

  return (
    <section className="mb-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <h2 className="shrink-0 text-sm font-semibold text-slate-500 dark:text-slate-400">Мої активи</h2>
        <ul className="flex flex-1 flex-wrap gap-1.5">
          {rows.map((row) => (
            <li key={row.id} className="flex items-center gap-1 rounded-full bg-slate-100 py-0.5 pl-3 pr-1 text-sm dark:bg-slate-800">
              <span title={row.name}>{row.ticker}</span>
              <button
                type="button"
                onClick={() => void remove(row)}
                aria-label={`Відписатися від ${row.ticker}`}
                title="Відписатися"
                className="flex h-5 w-5 items-center justify-center rounded-full text-slate-500 hover:bg-slate-300 hover:text-slate-900 dark:hover:bg-slate-600 dark:hover:text-white"
              >
                ×
              </button>
            </li>
          ))}
          {rows.length === 0 && <li className="text-sm text-slate-500 dark:text-slate-400">Поки нічого. Додайте тікер, щоб стежити за монетою чи акцією.</li>}
        </ul>
        <form onSubmit={(e) => void search(e)} className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Тікер: SOL, NVDA"
            aria-label="Тікер монети чи акції"
            maxLength={12}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm uppercase placeholder:normal-case sm:w-40 dark:border-slate-700 dark:bg-slate-900"
          />
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="shrink-0 rounded-lg bg-slate-900 px-3 py-1.5 text-sm text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
          >
            {busy ? "Шукаю…" : "Стежити"}
          </button>
        </form>
      </div>
      {note && <p className="mt-2 text-sm text-slate-600 dark:text-slate-400" aria-live="polite">{note}</p>}
      {choices.length > 0 && (
        <div className="mt-3">
          <p className="mb-2 text-sm text-slate-600 dark:text-slate-400">Під цим тікером кілька активів. Оберіть потрібний:</p>
          <ul className="flex flex-col gap-2">
            {choices.map((item) => (
              <Candidate key={item.id} item={item} busy={busy} onPick={() => void subscribe(item)} />
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
