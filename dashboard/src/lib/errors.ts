// Raw errors repeat every 15 minutes and are written for a developer. People need what happened,
// whether it is over, and how many times. Text comes from outside the project: render it as plain text.

import type { EventRow } from "@/lib/types";

const KNOWN: { test: RegExp; title: string; explain: string }[] = [
  {
    test: /OnMaintenance|"code":"50001"/,
    title: "Біржа OKX на обслуговуванні",
    explain: "Біржа тимчасово не відповідала. Бот пропустив крок і повторив його наступного разу. Ваших дій не потрібно.",
  },
  {
    test: /InsufficientFunds|51008/,
    title: "Не вистачило коштів на ордер",
    explain: "Біржа відхилила ордер через брак вільних USDT на демо-рахунку.",
  },
  {
    test: /50124|does not have trading permission/,
    title: "API-ключ без права торгівлі",
    explain: "Біржа відхилила запит: ключ не мав доступу до цього ринку.",
  },
  {
    test: /RemoteProtocolError|ConnectError|ReadTimeout|ConnectTimeout|RequestTimeout|NetworkError/,
    title: "Обірвався зв'язок",
    explain: "Запит не дійшов. Бот повторив його на наступному кроці.",
  },
  {
    test: /AssertionError/,
    title: "Внутрішня перевірка бота",
    explain: "Запис угоди не збігся з очікуваним станом, тому крок цього агента було пропущено.",
  },
];

export interface ErrorGroup {
  title: string;
  explain: string;
  count: number;
  first: string;
  last: string;
  sample: string;
}

/** Errors with the same meaning as one line. Newest group first, given rows newest first. */
export function groupErrors(rows: EventRow[]): ErrorGroup[] {
  const groups = new Map<string, ErrorGroup>();
  for (const row of rows) {
    const known = KNOWN.find((k) => k.test.test(row.message));
    const title = known?.title ?? "Інша помилка";
    const key = known ? title : row.message;
    const group = groups.get(key);
    if (group) {
      group.count += 1;
      group.first = row.ts;
    } else {
      groups.set(key, {
        title, explain: known?.explain ?? "", count: 1, first: row.ts, last: row.ts, sample: row.message,
      });
    }
  }
  return [...groups.values()];
}
