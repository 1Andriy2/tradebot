"use client";

import { useState, type FocusEvent, type MouseEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function Card({ title, children, className = "", hint }: {
  title?: ReactNode;
  children: ReactNode;
  className?: string;
  hint?: ReactNode;
}) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}>
      {title && <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</h2>}
      {hint && <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{hint}</p>}
      <div className={title || hint ? "mt-3" : ""}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone = "" }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${tone}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}

export function PageTitle({ children, intro }: { children: ReactNode; intro?: ReactNode }) {
  return (
    <div className="mb-4">
      <h1 className="text-2xl font-bold">{children}</h1>
      {intro && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-600 dark:text-slate-400">{intro}</p>}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <p className="mb-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">Помилка: {error}</p>;
}

export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  const colors = tone === "warn"
    ? "bg-amber-50 text-amber-900 dark:bg-amber-950/50 dark:text-amber-200"
    : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300";
  return <div className={`rounded-lg p-3 text-sm leading-relaxed ${colors}`}>{children}</div>;
}

export function Empty({ children = "Даних ще немає" }: { children?: ReactNode }) {
  return <p className="py-6 text-center text-sm text-slate-500">{children}</p>;
}

const TIP_WIDTH = 256;
const TIP_MARGIN = 8;
const TIP_ROOM = 170; // below this much space under the label, the tip opens upwards

/**
 * Explains a label on hover or focus. It lives in document.body, since tables scroll and would clip it.
 * The label is focusable, so a tap on a phone opens it too.
 */
export function Hint({ text, children }: { text: ReactNode; children: ReactNode }) {
  const [at, setAt] = useState<{ left: number; top?: number; bottom?: number } | null>(null);
  const show = (e: MouseEvent<HTMLElement> | FocusEvent<HTMLElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const left = Math.max(TIP_MARGIN, Math.min(box.left + box.width / 2 - TIP_WIDTH / 2, window.innerWidth - TIP_WIDTH - TIP_MARGIN));
    setAt(window.innerHeight - box.bottom < TIP_ROOM
      ? { left, bottom: window.innerHeight - box.top + 6 }
      : { left, top: box.bottom + 6 });
  };
  const hide = () => setAt(null);
  return (
    <span
      tabIndex={0}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      className="cursor-help underline decoration-slate-400 decoration-dotted underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-slate-400"
    >
      {children}
      {at && createPortal(
        <span
          role="tooltip"
          style={{ ...at, width: TIP_WIDTH }}
          className="pointer-events-none fixed z-50 whitespace-normal rounded-lg bg-slate-900 p-2.5 text-left text-xs font-normal leading-relaxed text-white shadow-lg dark:bg-slate-100 dark:text-slate-900"
        >
          {text}
        </span>,
        document.body,
      )}
    </span>
  );
}

export type Column = string | { label: string; hint?: ReactNode; right?: boolean };

/** Right-aligned columns need `text-right` on their cells too. */
export function Table({ head, children }: { head: Column[]; children: ReactNode }) {
  return (
    <div className="-mx-4 overflow-x-auto px-4">
      <table className="w-full min-w-max border-separate border-spacing-0 text-sm">
        <thead>
          <tr>
            {head.map((column) => {
              const { label, hint, right } = typeof column === "string" ? { label: column, hint: undefined, right: false } : column;
              return (
                <th
                  key={label}
                  className={`whitespace-nowrap border-b border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-500 first:rounded-tl-lg last:rounded-tr-lg dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-400 ${right ? "text-right" : "text-left"}`}
                >
                  {hint ? <Hint text={hint}>{label}</Hint> : label}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="tabular-nums [&>tr:hover]:bg-slate-50 dark:[&>tr:hover]:bg-slate-800/40 [&_td]:whitespace-nowrap [&_td]:border-b [&_td]:border-slate-100 [&_td]:px-3 [&_td]:py-2.5 dark:[&_td]:border-slate-800">
          {children}
        </tbody>
      </table>
    </div>
  );
}

const BADGES = {
  green: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  red: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  amber: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  blue: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  grey: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
};

export function Badge({ tone = "grey", children }: { tone?: keyof typeof BADGES; children: ReactNode }) {
  return <span className={`inline-block whitespace-nowrap rounded-md px-1.5 py-0.5 text-xs font-medium ${BADGES[tone]}`}>{children}</span>;
}

export function SideBadge({ side }: { side: "long" | "short" }) {
  return <Badge tone={side === "long" ? "green" : "red"}>{side === "long" ? "Long" : "Short"}</Badge>;
}

export function Chips({ value, onChange, options, label }: {
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

export function Select({ label, value, onChange, options }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-slate-500">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

export const AGENT_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777"];

export function colorFor(id: string, ids: string[]): string {
  const i = ids.indexOf(id);
  return AGENT_COLORS[(i < 0 ? 0 : i) % AGENT_COLORS.length];
}
