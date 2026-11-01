"use client";

import { useEffect, useState } from "react";

/** Current time in ms (0 until mounted), refreshed every `everyMs`. Keeps renders pure. */
export function useNow(everyMs = 30_000): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, everyMs);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [everyMs]);
  return now;
}
