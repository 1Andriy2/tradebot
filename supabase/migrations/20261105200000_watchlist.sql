-- Assets the user follows. The dashboard writes here, the first table it may write to: sign-ups are
-- disabled, so "authenticated" is the owner alone. Removing a row is the whole unsubscribe.

create table if not exists watchlist (
    id         text primary key,            -- 'crypto:solana' or 'stock:NVDA': tickers are not unique
    kind       text not null check (kind in ('crypto', 'stock')),
    ticker     text not null,
    name       text not null,               -- what news is searched by
    source_id  text not null default '',    -- CoinGecko id or SEC CIK
    added_at   timestamptz not null default now()
);

alter table watchlist enable row level security;
drop policy if exists read_authenticated on watchlist;
create policy read_authenticated on watchlist for select to authenticated using (true);
drop policy if exists add_authenticated on watchlist;
create policy add_authenticated on watchlist for insert to authenticated with check (true);
drop policy if exists remove_authenticated on watchlist;
create policy remove_authenticated on watchlist for delete to authenticated using (true);

alter publication supabase_realtime add table watchlist;

-- A new subscription starts collecting at once instead of at the next quarter hour. It must never
-- block the subscription itself, so a failed dispatch is only a notice.
create or replace function public.collect_news_now() returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    perform public.dispatch_workflow('news.yml');
    return null;
exception when others then
    raise notice 'news dispatch failed: %', sqlerrm;
    return null;
end;
$$;

revoke all on function public.collect_news_now() from public, anon, authenticated;

drop trigger if exists watchlist_collect_now on watchlist;
create trigger watchlist_collect_now after insert on watchlist
    for each statement execute function public.collect_news_now();
