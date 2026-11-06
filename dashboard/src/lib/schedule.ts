// When the jobs run. These are copies of the Supabase Cron schedules in supabase/migrations, which the
// dashboard cannot read: change them together. Times are UTC, and GitHub can start a job a few minutes late.

export interface Job {
  id: string;
  label: string;
  about: string;
  cron: string;
}

export const JOBS: Job[] = [
  { id: "trade", label: "Торговий крок", about: "Кожні 15 хвилин, після закриття свічки", cron: "2,17,32,47 * * * *" },
  { id: "news", label: "Збір новин", about: "Кожні 15 хвилин", cron: "11,26,41,56 * * * *" },
  { id: "retrain", label: "Навчання LightGBM", about: "Щоночі, разом із перерахунком історичних тестів", cron: "17 2 * * *" },
  { id: "research", label: "Нічне дослідження", about: "Щоночі: стратегії, патерни, перевірка гіпотез", cron: "7 4 * * *" },
  { id: "rl", label: "Навчання RL-агента", about: "Щонеділі вночі", cron: "43 3 * * 0" },
];

export const job = (id: string): Job => JOBS.find((j) => j.id === id)!;

const MINUTE_MS = 60_000;
const WEEK_MINUTES = 7 * 24 * 60;

/** The next moment after `from` matching a schedule of "minutes hour * * weekday". */
export function nextRun(cron: string, from: number): Date {
  const [minutes, hour, , , weekday] = cron.split(" ");
  const wanted = new Set(minutes.split(",").map(Number));
  const first = Math.floor(from / MINUTE_MS) * MINUTE_MS + MINUTE_MS;
  for (let i = 0; i <= WEEK_MINUTES; i += 1) {
    const at = new Date(first + i * MINUTE_MS);
    if (wanted.has(at.getUTCMinutes())
      && (hour === "*" || at.getUTCHours() === Number(hour))
      && (weekday === "*" || at.getUTCDay() === Number(weekday))) return at;
  }
  throw new Error(`no run within a week: ${cron}`);
}

const TZ = "Europe/Kyiv";
const dayKey = new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" });
const clock = new Intl.DateTimeFormat("uk-UA", { timeZone: TZ, hour: "2-digit", minute: "2-digit" });
const date = new Intl.DateTimeFormat("uk-UA", { timeZone: TZ, weekday: "short", day: "numeric", month: "long" });

/** "сьогодні о 04:17", "завтра о 04:17" or "нд, 8 листопада о 05:43", in Kyiv time. */
export function fmtWhen(at: Date, now: number): string {
  const day = dayKey.format(at);
  const label = day === dayKey.format(new Date(now)) ? "сьогодні"
    : day === dayKey.format(new Date(now + 86_400_000)) ? "завтра" : date.format(at);
  return `${label} о ${clock.format(at)}`;
}

export function fmtIn(at: Date, now: number): string {
  const minutes = Math.max(1, Math.round((at.getTime() - now) / MINUTE_MS));
  if (minutes < 60) return `через ${minutes} хв`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `через ${hours} год ${minutes % 60} хв`;
  return `через ${Math.floor(hours / 24)} дн ${hours % 24} год`;
}
