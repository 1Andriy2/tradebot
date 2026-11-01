-- Initial schema for the trading bot.
-- The bot writes with the service-role key (bypasses RLS).
-- The dashboard reads with the anon key after Supabase Auth login: SELECT only, for authenticated users.

create table if not exists agents (
    id           text primary key,              -- e.g. 'ma_cross', 'lgbm_v3'
    kind         text not null check (kind in ('baseline', 'ml', 'rl')),
    label        text not null,
    params       jsonb not null default '{}',
    is_active    boolean not null default true,
    created_at   timestamptz not null default now()
);

create table if not exists model_versions (
    id            uuid primary key default gen_random_uuid(),
    agent_id      text not null references agents(id),
    version       int not null,
    status        text not null check (status in ('challenger', 'champion', 'retired', 'rejected')),
    params        jsonb not null default '{}',
    metrics       jsonb not null default '{}',
    train_start   timestamptz,
    train_end     timestamptz,
    storage_path  text,
    created_at    timestamptz not null default now(),
    promoted_at   timestamptz,
    unique (agent_id, version)
);
-- Exactly one champion at a time.
create unique index if not exists one_champion on model_versions ((status)) where status = 'champion';

create table if not exists champion (
    id          int primary key default 1 check (id = 1),
    agent_id    text not null references agents(id),
    version_id  uuid references model_versions(id),
    since       timestamptz not null default now()
);

create table if not exists equity_snapshots (
    id         bigint generated always as identity primary key,
    ts         timestamptz not null,
    agent_id   text not null references agents(id),
    venue      text not null check (venue in ('demo', 'shadow')),
    equity     double precision not null,
    unique (agent_id, venue, ts)
);
create index if not exists equity_snapshots_ts on equity_snapshots (ts);

create table if not exists positions (
    agent_id    text not null references agents(id),
    venue       text not null check (venue in ('demo', 'shadow')),
    symbol      text not null,
    units       double precision not null,
    avg_entry   double precision,
    stop_loss   double precision,
    take_profit double precision,
    updated_at  timestamptz not null default now(),
    primary key (agent_id, venue, symbol)
);

create table if not exists trades (
    id            bigint generated always as identity primary key,
    agent_id      text not null references agents(id),
    venue         text not null check (venue in ('demo', 'shadow')),
    symbol        text not null,
    side          text not null check (side in ('long', 'short')),
    entry_time    timestamptz not null,
    exit_time     timestamptz not null,
    entry_price   double precision not null,
    exit_price    double precision not null,
    pnl           double precision not null,
    fees          double precision not null,
    return_pct    double precision,
    entry_reason  text not null,
    exit_reason   text not null,
    unique (agent_id, venue, symbol, entry_time, exit_time)
);
create index if not exists trades_exit_time on trades (exit_time desc);

create table if not exists decisions (
    id            bigint generated always as identity primary key,
    ts            timestamptz not null,
    agent_id      text not null references agents(id),
    venue         text not null check (venue in ('demo', 'shadow')),
    symbol        text not null,
    signal        double precision not null,
    units_before  double precision not null,
    units_after   double precision not null,
    action        text not null,                 -- open|close|increase|reduce|flip|hold
    reason        text not null,
    details       jsonb not null default '{}'
);
create index if not exists decisions_ts on decisions (ts desc);

create table if not exists events (
    id        bigint generated always as identity primary key,
    ts        timestamptz not null default now(),
    type      text not null,                     -- retrain|champion_change|kill_switch|daily_loss_limit|error|info
    agent_id  text,
    message   text not null,
    details   jsonb not null default '{}'
);
create index if not exists events_ts on events (ts desc);

create table if not exists candles (
    symbol     text not null,
    timeframe  text not null,
    ts         timestamptz not null,
    open       double precision not null,
    high       double precision not null,
    low        double precision not null,
    close      double precision not null,
    volume     double precision not null,
    primary key (symbol, timeframe, ts)
);

create table if not exists backtest_results (
    id          bigint generated always as identity primary key,
    agent_id    text not null,
    label       text not null,
    symbol      text not null,
    timeframe   text not null,
    period_start timestamptz not null,
    period_end   timestamptz not null,
    metrics     jsonb not null,
    created_at  timestamptz not null default now()
);

-- Internal bot state (policy/risk/paper books). Not exposed to the dashboard.
create table if not exists agent_state (
    agent_id    text not null,
    venue       text not null,
    symbol      text not null,
    state       jsonb not null,
    last_bar    timestamptz,
    updated_at  timestamptz not null default now(),
    primary key (agent_id, venue, symbol)
);

create table if not exists bot_status (
    component     text primary key,              -- 'trader', 'trainer'
    mode          text,
    last_step_at  timestamptz,
    last_ok_at    timestamptz,
    last_error    text,
    details       jsonb not null default '{}'
);

-- Row level security: authenticated users may only read; anon gets nothing.
do $$
declare t text;
begin
    foreach t in array array['agents','model_versions','champion','equity_snapshots','positions',
                             'trades','decisions','events','candles','backtest_results','bot_status']
    loop
        execute format('alter table %I enable row level security', t);
        execute format('drop policy if exists read_authenticated on %I', t);
        execute format('create policy read_authenticated on %I for select to authenticated using (true)', t);
    end loop;
end $$;
alter table agent_state enable row level security;  -- no policies: service role only

-- Realtime for live dashboard updates.
alter publication supabase_realtime add table equity_snapshots, trades, events, bot_status, positions;

-- Keep the free tier small: drop old candles/decisions/snapshots.
create or replace function prune_old_rows() returns void language sql as $$
    delete from candles where ts < now() - interval '60 days';
    delete from decisions where ts < now() - interval '90 days' and action = 'hold';
$$;
