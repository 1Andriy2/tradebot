"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { supabaseBrowser } from "@/lib/supabase/client";

const LINKS = [
  { href: "/", label: "Огляд" },
  { href: "/news", label: "Новини" },
  { href: "/chart", label: "Графік" },
  { href: "/trades", label: "Угоди" },
  { href: "/leaderboard", label: "Рейтинг" },
  { href: "/research", label: "Дослідження" },
  { href: "/training", label: "Навчання" },
  { href: "/events", label: "Журнал" },
  { href: "/status", label: "Статус" },
];

export function Nav() {
  const path = usePathname();
  const router = useRouter();

  async function logout() {
    await supabaseBrowser().auth.signOut();
    router.replace("/login");
  }

  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-2">
        <span className="shrink-0 font-bold">🤖 Демо-бот</span>
        <nav className="flex flex-1 gap-1 overflow-x-auto">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-sm ${
                  active
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <button onClick={logout} className="shrink-0 text-sm text-slate-500 hover:text-slate-900 dark:hover:text-white">
          Вийти
        </button>
      </div>
    </header>
  );
}
