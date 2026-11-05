-- News for the dashboard. Parquet in the data bucket stays the full record; these tables hold what
-- the page shows. Titles and links come from outside: the page must render them as plain text.

create table if not exists news_items (
    cluster     text primary key,          -- one row per story, however many outlets told it
    first_seen  timestamptz not null,
    published   timestamptz not null,
    source      text not null,
    kind        text not null,
    title       text not null,
    title_uk    text not null default '',
    url         text not null,
    subject     text not null default '',
    stance      text not null default 'neutral',
    impact      double precision not null,
    confidence  double precision not null,
    channel     text not null default 'none',
    event_type  text not null default 'other',
    assets      text not null default '',  -- comma separated
    reason      text not null default '',
    outlets     jsonb not null default '[]'
);
create index if not exists news_items_first_seen on news_items (first_seen desc);

create table if not exists news_levels (
    scope       text primary key,          -- BTC | ETH | STOCKS
    level       text not null,             -- trade | reduce | stand_aside | unknown
    long_mult   double precision not null,
    short_mult  double precision not null,
    negative    double precision not null,
    positive    double precision not null,
    events      int not null,
    sources     int not null,
    stale       boolean not null,
    updated_at  timestamptz not null
);

do $$
declare t text;
begin
    foreach t in array array['news_items', 'news_levels']
    loop
        execute format('alter table %I enable row level security', t);
        execute format('drop policy if exists read_authenticated on %I', t);
        execute format('create policy read_authenticated on %I for select to authenticated using (true)', t);
    end loop;
end $$;

alter publication supabase_realtime add table news_items, news_levels;

create or replace function prune_old_rows() returns void language sql as $$
    delete from candles where ts < now() - interval '60 days';
    delete from decisions where ts < now() - interval '90 days' and action = 'hold';
    delete from research_results where created_at < now() - interval '30 days';
    delete from patterns where created_at < now() - interval '30 days';
    delete from news_items where first_seen < now() - interval '30 days';
$$;
