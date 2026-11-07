-- The 5-minute paper lane: a long job, dispatched hourly so a crashed one is back within the hour.
select cron.schedule('scout-lane', '5 * * * *', $$select public.dispatch_workflow('scout.yml')$$);

-- RL training goes from weekly to nightly. Same name, so this replaces the old schedule.
select cron.schedule('weekly-rl', '43 3 * * *', $$select public.dispatch_workflow('rl-train.yml')$$);
