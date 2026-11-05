"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Empty, ErrorNote, PageTitle } from "@/components/ui";
import {
  ago, ASSETS, assetList, byDay, CHANNELS, EVENT_TYPES, fmtClock, fmtImpact, fmtMult, LEVELS, outletName,
  safeUrl, SCOPES, SHORTS_ONLY_HINT, SIGNIFICANT, STANCES, storyTime, strength,
} from "@/lib/news";
import type { NewsItemRow, NewsLevelRow } from "@/lib/types";
import { useNow } from "@/lib/useNow";
import { useRows } from "@/lib/useRows";

const DAYS = 7;
const MAX_STORIES = DAYS * 500; // the news job scores at most 500 a day, so this never cuts the week
const LEVEL_MAX_AGE_MS = 3_600_000; // the same limit the test agents apply
const SCOPE_ORDER = ["BTC", "ETH", "STOCKS"];

const VIEWS: Record<string, { label: string; test: (i: NewsItemRow) => boolean }> = {
  significant: { label: "Важливі", test: (i) => Math.abs(i.impact) >= SIGNIFICANT },
  bad: { label: "Негативні", test: (i) => i.impact <= -SIGNIFICANT },
  good: { label: "Позитивні", test: (i) => i.impact >= SIGNIFICANT },
  all: { label: "Усі", test: () => true },
};

const ASSET_FILTERS: Record<string, string> = { all: "Усі активи", BTC: "BTC", ETH: "ETH", CRYPTO: "Крипторинок", STOCKS: "Акції" };
const CHANNEL_FILTERS: Record<string, string> = { all: "Будь-який вплив", direct: "Прямий", indirect: "Непрямий" };

function Chips({ value, onChange, options, label }: {
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1.5">
      {options.map(([key, text]) => (
        <button
          key={key}
          type="button"
          aria-pressed={value === key}
          onClick={() => onChange(key)}
          className={`rounded-full px-3 py-1 text-sm transition-colors ${
            value === key
              ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
              : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          }`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function Pressure({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
      <span className="w-16 shrink-0">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="w-8 text-right tabular-nums">{Math.round(value * 100)}%</span>
    </div>
  );
}

function LevelCard({ row, now }: { row: NewsLevelRow; now: number }) {
  const expired = now > 0 && now - new Date(row.updated_at).getTime() > LEVEL_MAX_AGE_MS;
  const level = expired ? LEVELS.unknown : LEVELS[row.level] ?? LEVELS.unknown;
  const shortsOnly = !expired && row.long_mult === 1 && row.short_mult < 1;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400">{SCOPES[row.scope] ?? row.scope}</h2>
        <span className="text-xs text-slate-400">{ago(row.updated_at, now)}</span>
      </div>
      <div className={`mt-1 flex items-center gap-2 text-xl font-semibold ${level.tone}`}>
        <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${level.dot}`} />
        {level.label}
      </div>
      <p className="mt-1 min-h-10 text-sm text-slate-600 dark:text-slate-400">{shortsOnly ? SHORTS_ONLY_HINT : level.hint}</p>
      <div className="mt-3 space-y-1.5">
        <Pressure label="Негатив" value={row.negative} color="bg-rose-500" />
        <Pressure label="Позитив" value={row.positive} color="bg-emerald-500" />
      </div>
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        {!expired && <>Лонги {fmtMult(row.long_mult)}, шорти {fmtMult(row.short_mult)} · </>}
        Подій за добу: {row.events}, джерел: {row.sources}
      </p>
    </section>
  );
}

function ImpactMeter({ impact }: { impact: number }) {
  const width = Math.min(Math.abs(impact), 1) * 50;
  const bad = impact < 0;
  const tone = impact === 0 ? "text-slate-500" : bad ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400";
  return (
    <div className="w-full sm:w-20" title={`Вплив від −1 до +1: ${fmtImpact(impact)}`}>
      <div className={`text-lg font-semibold tabular-nums ${tone}`}>{fmtImpact(impact)}</div>
      <div className="relative mt-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className={`absolute inset-y-0 rounded-full ${bad ? "bg-rose-500" : "bg-emerald-500"}`}
          style={bad ? { right: "50%", width: `${width}%` } : { left: "50%", width: `${width}%` }}
        />
        <div className="absolute -inset-y-0.5 left-1/2 w-px bg-slate-400 dark:bg-slate-500" />
      </div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{strength(impact)}</div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-md bg-slate-100 px-1.5 py-0.5 text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-300">
      {children}
    </span>
  );
}

function Story({ item }: { item: NewsItemRow }) {
  const title = item.title_uk || item.title;
  const href = safeUrl(item.url);
  const others = item.outlets.filter((o) => o !== item.source).map(outletName);
  const stance = STANCES[item.stance];
  return (
    <article className="flex flex-col gap-3 py-4 sm:flex-row sm:gap-5">
      <ImpactMeter impact={item.impact} />
      <div className="min-w-0 flex-1">
        <h3 className="text-base font-semibold leading-snug break-words">
          {href ? (
            <a href={href} target="_blank" rel="noopener noreferrer nofollow" className="hover:underline">{title}</a>
          ) : title}
        </h3>
        {item.title_uk && item.title_uk !== item.title && (
          <p className="mt-0.5 text-xs text-slate-500 break-words dark:text-slate-400">{item.title}</p>
        )}
        {item.reason && <p className="mt-1.5 text-sm text-slate-700 break-words dark:text-slate-300">{item.reason}</p>}
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Tag>{EVENT_TYPES[item.event_type] ?? item.event_type}</Tag>
          {CHANNELS[item.channel] && <Tag>{CHANNELS[item.channel]}</Tag>}
          {assetList(item).map((a) => <Tag key={a}>{ASSETS[a] ?? a}</Tag>)}
          {item.subject && <Tag>{item.subject}{stance ? `, ${stance}` : ""}</Tag>}
        </div>
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          {fmtClock(storyTime(item))} · {outletName(item.source)}
          {others.length > 0 && <span title={others.join(", ")}> і ще {others.length}</span>}
          {" · "}впевненість {Math.round(item.confidence * 100)}%
        </p>
      </div>
    </article>
  );
}

export default function NewsPage() {
  const now = useNow();
  const [view, setView] = useState("significant");
  const [asset, setAsset] = useState("all");
  const [channel, setChannel] = useState("all");
  const [topic, setTopic] = useState("all");
  const [query, setQuery] = useState("");

  const news = useRows<NewsItemRow>("news_items", {
    order: { column: "first_seen", tiebreak: "cluster" }, limit: MAX_STORIES, sinceDays: { column: "first_seen", days: DAYS },
  });
  const levels = useRows<NewsLevelRow>("news_levels", { realtime: true });

  // Stories are the heavy table: refetch them once per news run, and only for a tab someone is looking at.
  const { reload } = news;
  const updatedAt = levels.rows.map((r) => r.updated_at).sort().at(-1);
  const loadedFor = useRef<string | undefined>(undefined);
  useEffect(() => {
    const refresh = () => {
      if (!updatedAt || updatedAt === loadedFor.current) return;
      if (!loadedFor.current) loadedFor.current = updatedAt; // the first load already has this run
      else if (document.visibilityState === "visible") {
        loadedFor.current = updatedAt;
        void reload();
      }
    };
    refresh();
    document.addEventListener("visibilitychange", refresh);
    return () => document.removeEventListener("visibilitychange", refresh);
  }, [updatedAt, reload]);

  const topics = useMemo(() => [...new Set(news.rows.map((i) => i.event_type))].sort(), [news.rows]);
  const shown = useMemo(() => {
    const words = query.trim().toLowerCase();
    return news.rows.filter((i) => VIEWS[view].test(i)
      && (asset === "all" || assetList(i).includes(asset))
      && (channel === "all" || i.channel === channel)
      && (topic === "all" || i.event_type === topic)
      && (!words || `${i.title} ${i.title_uk} ${i.subject} ${i.reason}`.toLowerCase().includes(words)))
      .sort((a, b) => new Date(storyTime(b)).getTime() - new Date(storyTime(a)).getTime());
  }, [news.rows, view, asset, channel, topic, query]);

  const cards = SCOPE_ORDER.flatMap((scope) => levels.rows.filter((r) => r.scope === scope));

  return (
    <>
      <PageTitle>Новини та ризик</PageTitle>
      <ErrorNote error={news.error ?? levels.error} />

      {cards.length > 0 && (
        <>
          <div className="grid gap-3 md:grid-cols-3">
            {cards.map((row) => <LevelCard key={row.scope} row={row} now={now} />)}
          </div>
          <p className="mt-2 mb-5 text-xs text-slate-500 dark:text-slate-400">
            Це поки експеримент: рівень читають лише тестові агенти, на демо-рахунок він не впливає.
          </p>
        </>
      )}

      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <Chips label="Які новини показати" value={view} onChange={setView}
              options={Object.entries(VIEWS).map(([key, v]) => [key, v.label])} />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Пошук у новинах"
              aria-label="Пошук у новинах"
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm sm:w-56 dark:border-slate-700 dark:bg-slate-900"
            />
          </div>
          <Chips label="Актив" value={asset} onChange={setAsset} options={Object.entries(ASSET_FILTERS)} />
          <div className="flex flex-wrap items-center gap-3">
            <Chips label="Тип впливу" value={channel} onChange={setChannel} options={Object.entries(CHANNEL_FILTERS)} />
            <select
              aria-label="Тема"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-900"
            >
              <option value="all">Усі теми</option>
              {topics.map((t) => <option key={t} value={t}>{EVENT_TYPES[t] ?? t}</option>)}
            </select>
            <span className="ml-auto text-xs text-slate-500 dark:text-slate-400" aria-live="polite">
              Показано {shown.length} з {news.rows.length}
            </span>
          </div>
        </div>

        {news.loading ? (
          <Empty>Завантаження…</Empty>
        ) : shown.length ? (
          byDay(shown, now).map((group) => (
            <div key={group.day} className="mt-4">
              <h2 className="sticky top-12 z-[5] -mx-1 bg-white/95 px-1 py-1 text-sm font-semibold text-slate-500 backdrop-blur dark:bg-slate-900/95 dark:text-slate-400">
                {group.day}
              </h2>
              <div className="divide-y divide-slate-100 dark:divide-slate-800">
                {group.items.map((item) => <Story key={item.cluster} item={item} />)}
              </div>
            </div>
          ))
        ) : (
          <Empty>{news.rows.length ? "За цими фільтрами новин немає" : "Новин ще немає"}</Empty>
        )}
      </section>
    </>
  );
}
