-- History for new subscriptions, and two read-only views that keep the dashboard's pages cheap.

-- Set once a followed asset's month of history was fetched. It stays empty for a new subscription.
alter table watchlist add column if not exists backfilled_at timestamptz;

-- Stories of followed assets are kept longer: they are few, and old ones are the point of following.
create or replace function prune_old_rows() returns void language sql as $$
    delete from candles where ts < now() - interval '60 days';
    delete from decisions where ts < now() - interval '90 days' and action = 'hold';
    delete from research_results where created_at < now() - interval '30 days';
    delete from patterns where created_at < now() - interval '30 days';
    delete from news_items where first_seen < now() - interval '30 days' and kind <> 'watch';
    delete from news_items where first_seen < now() - interval '180 days';
$$;

-- The last snapshot of each hour. A chart needs no more, and it is a quarter of the rows to download.
create or replace view equity_hourly with (security_invoker = true) as
select distinct on (agent_id, venue, date_trunc('hour', ts)) ts, agent_id, venue, equity
from equity_snapshots
order by agent_id, venue, date_trunc('hour', ts), ts desc;

-- One row per test agent since it started, so the ranking never depends on how many rows a page reads.
create or replace view agent_live_stats with (security_invoker = true) as
with curve as (
    select agent_id, ts, equity, max(equity) over (partition by agent_id order by ts) as peak
    from equity_snapshots
    where venue = 'shadow'
), summary as (
    select agent_id,
           min(ts) as started_at,
           (array_agg(equity order by ts))[1] as start_equity,
           (array_agg(equity order by ts desc))[1] as equity,
           (array_agg(equity order by ts desc) filter (where ts <= now() - interval '24 hours'))[1] as equity_day_ago,
           max(1 - equity / nullif(peak, 0)) as max_drawdown
    from curve
    group by agent_id
), results as (
    select agent_id,
           count(*) as trades,
           count(*) filter (where pnl > 0) as wins,
           coalesce(sum(pnl) filter (where pnl > 0), 0) as gross_win,
           coalesce(-sum(pnl) filter (where pnl < 0), 0) as gross_loss
    from trades
    where venue = 'shadow'
    group by agent_id
)
select s.*, coalesce(r.trades, 0) as trades, coalesce(r.wins, 0) as wins,
       coalesce(r.gross_win, 0) as gross_win, coalesce(r.gross_loss, 0) as gross_loss
from summary s
left join results r using (agent_id);

revoke all on equity_hourly, agent_live_stats from anon;
grant select on equity_hourly, agent_live_stats to authenticated;
