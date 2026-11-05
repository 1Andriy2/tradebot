-- News step every 15 minutes, placed to finish before the trade step at minute 2/17/32/47.
select cron.schedule('news-step', '41,56,11,26 * * * *', $$select public.dispatch_workflow('news.yml')$$);
