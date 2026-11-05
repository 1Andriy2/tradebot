"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/client";

interface Options {
  order?: { column: string; ascending?: boolean };
  /** Max rows; fetched in pages of 1000 (the PostgREST default cap). */
  limit?: number;
  filters?: Record<string, string | number | boolean>;
  /** Only rows with `column >= days ago`. */
  sinceDays?: { column: string; days: number };
  /** Re-fetch when this table changes (Supabase Realtime). */
  realtime?: boolean;
}

const PAGE = 1000;
const REALTIME_DELAY_MS = 1500; // one write can change many rows: refetch once, after they settle

/** Fetch rows from a table, and refetch on realtime changes. */
export function useRows<T>(table: string, opts: Options = {}) {
  const [rows, setRows] = useState<T[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const key = JSON.stringify(opts);
  const lastLoad = useRef(0);

  const load = useCallback(async () => {
    const id = ++lastLoad.current;
    const o: Options = JSON.parse(key);
    const limit = o.limit ?? PAGE;
    const out: T[] = [];
    for (let from = 0; from < limit; from += PAGE) {
      let q = supabaseBrowser().from(table).select("*");
      for (const [col, val] of Object.entries(o.filters ?? {})) q = q.eq(col, val);
      if (o.sinceDays) {
        const since = new Date(Date.now() - o.sinceDays.days * 86_400_000).toISOString();
        q = q.gte(o.sinceDays.column, since);
      }
      if (o.order) q = q.order(o.order.column, { ascending: o.order.ascending ?? false });
      const { data, error } = await q.range(from, Math.min(from + PAGE, limit) - 1);
      if (id !== lastLoad.current) return; // a newer load is running, and its result wins
      if (error) {
        setError(error.message);
        setLoading(false);
        return;
      }
      out.push(...((data ?? []) as T[]));
      if (!data || data.length < PAGE) break;
    }
    setError(null);
    setRows(out);
    setLoading(false);
  }, [table, key]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    void load();
    if (!opts.realtime) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const channel = supabaseBrowser()
      .channel(`rt-${table}-${key}`)
      .on("postgres_changes", { event: "*", schema: "public", table }, () => {
        clearTimeout(timer);
        timer = setTimeout(() => void load(), REALTIME_DELAY_MS);
      })
      .subscribe();
    return () => {
      clearTimeout(timer);
      void supabaseBrowser().removeChannel(channel);
    };
  }, [load, table, key, opts.realtime]);

  return { rows, error, loading, reload: load };
}
