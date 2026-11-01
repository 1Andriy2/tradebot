# Навчальний торговий AI-агент (лише демо-гроші)

Агент торгує **демо-балансом** на OKX Demo Trading (ф'ючерси BTC/USDT та ETH/USDT, long і short),
пробує різні стратегії, вчиться на історії та нових даних і з часом покращується.
Все працює в хмарі безкоштовно: ноутбук можна вимкнути.

> ⚠️ Реальні гроші не використовуються взагалі. У коді немає режиму реальної торгівлі,
> а «запобіжник» (allowlist) аварійно зупиняє програму, якщо адреса біржі не демо
> (для OKX додатково перевіряється демо-заголовок `x-simulated-trading: 1` у кожному запиті).

## Як це влаштовано (коротко)

| Частина | Де працює | Навіщо |
|---|---|---|
| Торговий крок кожні 15 хв | GitHub Actions (`trade.yml`) | бере свіжі свічки, приймає рішення, ставить ордери на демо-рахунку OKX |
| Нічне перенавчання | GitHub Actions (`retrain.yml`) | LightGBM + walk-forward + champion/challenger |
| RL-агент | GitHub Actions (щотижня, CPU) або Kaggle (GPU) | PPO вчиться торгувати в симуляторі |
| База даних і файли моделей | Supabase (безкоштовно) | угоди, рішення, метрики, версії моделей |
| Дашборд | Vercel Hobby (безкоштовно) | графіки й таблиці українською, вхід за паролем |

Детальніше: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Усі технічні рішення: [docs/DECISIONS.md](docs/DECISIONS.md).

### Як агент «вчиться»

1. **Аналог pretraining.** Модель LightGBM вивчає роки свічок і вчиться передбачати, куди піде ціна за 8 годин.
2. **Аналог RL.** Агент PPO торгує в симуляторі. Нагорода дорівнює прибутку мінус комісії та штраф за просідання.
3. **Чесна перевірка.** Тільки walk-forward (модель тестується на даних, яких не бачила). Комісія 0.1% і проковзування враховані скрізь.
4. **Champion/Challenger.** Нова версія замінює поточного «чемпіона» на демо-рахунку **лише** тоді, коли перевершує його за ризик-скоригованою метрикою (Sortino) на відкладених даних.
5. **Щоночі** все перенавчається на свіжих даних. Кожна версія зберігається з метриками, тож прогрес видно на дашборді.

### Ризик-менеджмент (жорсткі межі в коді)

- позиція не більше 25% балансу, плече не більше 2x;
- денний збиток −5%: торгівля зупиняється до наступного дня;
- просідання −20% від піку: **kill switch** зупиняє торгівлю;
- кожне рішення записується в журнал разом із причиною.

---

## Покрокове налаштування (для новачка)

Усі акаунти мають бути на **andrijkostelnij72@gmail.com**. Жоден крок не вимагає банківської картки.
Якщо десь з'явиться вимога ввести картку чи перейти на платний тариф, **зупиніться** й нічого не вводьте.

### Крок 1. OKX Demo Trading: ключі API

> Чому OKX: Bybit блокує сервери GitHub (HTTP 403), а демо Kraken закрите. OKX доступний з GitHub.
> Код для Bybit Demo лишився: він працює локально та в серверному режимі (`EXCHANGE_ID=bybit`).

1. Зареєструйтесь на [okx.com](https://www.okx.com) (email або телефон).
   Якщо для демо вимагатимуть документи або картку, **зупиніться** й напишіть.
2. **Trade → Demo trading** (перемикач демо-режиму).
3. В демо-режимі: аватар → **Demo Trading API** → **Create Demo Trading API key**:
   права **Read** + **Trade** (без Withdraw), придумайте **passphrase**.
   Також у демо-режимі: **Settings → Account mode → Spot and futures** (у режимі Simple ф'ючерси недоступні).
4. ❗ Ключ має бути створений саме в демо-режимі: живий OKX такі ключі відхиляє.
5. Запишіть у `.env`:
   `EXCHANGE_ID=okx`, `EXCHANGE_API_KEY=...`, `EXCHANGE_API_SECRET=...`, `EXCHANGE_API_PASSPHRASE=...`

### Крок 2. Supabase: база даних

1. На [supabase.com](https://supabase.com) створіть безкоштовний проєкт (регіон: Frankfurt).
2. Відкрийте **SQL Editor** → вставте вміст файлу `supabase/migrations/20261101000000_init.sql` → **Run**.
3. Перейдіть в **Authentication → Sign In / Providers** і вимкніть **Allow new users to sign up**.
4. **Authentication → Users → Add user**: створіть користувача для входу на дашборд (ваш email і пароль).
5. У файл `.env` запишіть:
   - `SUPABASE_URL` і `SUPABASE_SERVICE_ROLE_KEY` (**Project Settings → API Keys**, ключ *secret / service_role*);
   - `SUPABASE_PROJECT_REF`: це частина адреси `https://supabase.com/dashboard/project/<ref>`;
   - `SUPABASE_ACCESS_TOKEN` (**Account → Access Tokens**).

> Ключ `service_role` дає повний доступ до бази. Його можна вписувати тільки в `.env` і в GitHub Secrets,
> **ніколи** в дашборд або у Vercel.

### Крок 3. GitHub: репозиторій і автоматичні запуски

1. Створіть **публічний** порожній репозиторій. Для публічних репозиторіїв хвилини Actions безкоштовні й необмежені, а секрети лишаються прихованими.
2. Створіть fine-grained токен ([налаштування](https://github.com/settings/personal-access-tokens/new)):
   - доступ лише до цього репозиторію;
   - Repository permissions: **Contents**, **Actions**, **Secrets**, **Workflows** у режимі *Read and write*;
   - Account permissions: **Email addresses** у режимі *Read*.
3. Запишіть токен у `.env` як `GH_TOKEN=...`.
4. Далі Claude Code сам надішле код у репозиторій і додасть секрети:
   `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, `EXCHANGE_API_PASSPHRASE`.
   Вручну це робиться в **Settings → Secrets and variables → Actions**.
5. Вкладка **Actions** → **Venue reachability check** → **Run workflow**. Ця перевірка показує, які біржі доступні з GitHub.

### Крок 4. Vercel: дашборд

1. На [vercel.com](https://vercel.com) увійдіть через GitHub (тариф **Hobby**, безкоштовний).
2. **Add New → Project** → оберіть репозиторій → **Root Directory**: `dashboard`.
3. У **Environment Variables** додайте:
   - `NEXT_PUBLIC_SUPABASE_URL`: та сама адреса, що й `SUPABASE_URL`;
   - `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`: **publishable / anon** ключ (не service_role!).
4. **Deploy**. Відкрийте адресу сайту й увійдіть користувачем із кроку 2.

### Крок 5 (необов'язково). Kaggle: навчання RL на GPU

1. Зареєструйтесь на [kaggle.com](https://www.kaggle.com) і підтвердіть телефон (без цього немає GPU та інтернету).
2. **Create → New Notebook → File → Import Notebook** → завантажте `notebooks/rl_kaggle.ipynb`.
3. У налаштуваннях ноутбука: **Accelerator** = GPU, **Internet** = On.
4. **Add-ons → Secrets**: додайте `SUPABASE_URL` і `SUPABASE_SERVICE_ROLE_KEY`.
5. Впишіть адресу свого репозиторію в `REPO_URL` → **Run All**.

---

## Робота з кодом локально

Потрібні [uv](https://docs.astral.sh/uv/) і Node.js 22.

```bash
uv sync --all-extras              # Python 3.11 + залежності в .venv
uv run pytest                     # тести
uv run tradebot download          # історія свічок з 2020 року в data/
uv run tradebot backtest --timeframe 1h          # базові стратегії
uv run tradebot step --shadow-only --memory-store # один крок без біржі та без бази
./scripts/with-env.sh uv run tradebot step        # один крок на демо-рахунку із записом у Supabase
./scripts/with-env.sh uv run tradebot retrain --sync   # перенавчання LightGBM
./scripts/with-env.sh ./scripts/check-accounts.sh      # перевірка, що токени з потрібного акаунта
```

Дашборд:

```bash
cd dashboard && cp .env.example .env.local   # вписати URL і publishable key
npm install && npm run dev                   # http://localhost:3000
```

`scripts/with-env.sh` підвантажує `.env` тільки для однієї команди. Глобальні налаштування
(git, gh, логіни) не змінюються, тому інші проєкти на комп'ютері не постраждають.

## Режим 24/7 на сервері (необов'язково)

На будь-якій Linux-VM з Docker (arm64 або amd64):

```bash
cp .env.example .env   # заповнити
docker compose up -d --build
docker compose logs -f trader
```

## Команди `tradebot`

| Команда | Що робить |
|---|---|
| `download [--sync]` | завантажує або оновлює історію свічок |
| `backtest` | бектест базових стратегій |
| `step [--shadow-only] [--memory-store]` | один торговий крок (режим cron) |
| `daemon` | крок після кожної 15-хвилинної свічки (режим сервера) |
| `retrain [--sync]` | нічне перенавчання LightGBM з champion/challenger |
| `rl-train [--timesteps N] [--device cuda]` | навчання RL-агента з champion/challenger |
| `maintenance` | видаляє старі записи, щоб не вийти за межі безкоштовного тарифу Supabase |

## Чесно про результати

На історії після комісій **усі** базові стратегії й перші моделі збиткові. Це нормально: ринок
ефективний, а комісії великі. Тому система **не** промоутить моделі, які не кращі за чемпіона.
Мета проєкту навчальна: бачити, як агент вчиться, порівнювати підходи й не втрачати реальних грошей.
