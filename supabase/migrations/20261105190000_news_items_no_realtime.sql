-- The news page refetches stories once per run, on a change to news_levels, so nothing listens here.
alter publication supabase_realtime drop table news_items;
