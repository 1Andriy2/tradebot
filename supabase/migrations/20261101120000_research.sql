-- Research results (walk-forward strategy scan), recurring patterns, funding in trades.

alter table trades add column if not exists funding double precision not null default 0;

create table if not exists research_results (
    id          bigint generated always as identity primary key,
    run_id      uuid not null,
    created_at  timestamptz not null default now(),
    family      text not null,
    symbol      text not null,
    timeframe   text not null,
    metrics     jsonb not null,
    params      jsonb not null default '{}',
    fold_returns jsonb not null default '[]'
);
create index if not exists research_results_run on research_results (run_id);
create index if not exists research_results_created on research_results (created_at desc);

create table if not exists patterns (
    id           bigint generated always as identity primary key,
    run_id       uuid not null,
    created_at   timestamptz not null default now(),
    symbol       text not null,
    kind         text not null,          -- hour | dow
    bucket       int not null,
    mean_return  double precision not null,
    t_stat       double precision not null,
    p_value      double precision not null,
    q_value      double precision not null,
    n            int not null,
    stable       boolean not null
);
create index if not exists patterns_run on patterns (run_id);

do $$
declare t text;
begin
    foreach t in array array['research_results', 'patterns']
    loop
        execute format('alter table %I enable row level security', t);
        execute format('drop policy if exists read_authenticated on %I', t);
        execute format('create policy read_authenticated on %I for select to authenticated using (true)', t);
    end loop;
end $$;

-- Keep only the last 30 research runs.
create or replace function prune_old_rows() returns void language sql as $$
    delete from candles where ts < now() - interval '60 days';
    delete from decisions where ts < now() - interval '90 days' and action = 'hold';
    delete from research_results where created_at < now() - interval '30 days';
    delete from patterns where created_at < now() - interval '30 days';
$$;
