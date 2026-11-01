-- Supabase Cron triggers the GitHub Actions workflows (GitHub's own `schedule` never fired
-- for this repository). Each job calls the workflow_dispatch API with a fine-grained token
-- (Actions: read & write on this repository only) stored in Supabase Vault as
-- `github_dispatch_token`. The token is added by the owner in the Dashboard, never in SQL.

create extension if not exists pg_cron;
create extension if not exists pg_net with schema extensions;

create or replace function public.dispatch_workflow(workflow text)
returns bigint
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
    token text;
begin
    select decrypted_secret into token from vault.decrypted_secrets
    where name = 'github_dispatch_token' limit 1;
    if token is null then
        raise notice 'github_dispatch_token missing in Vault; % not dispatched', workflow;
        return null;
    end if;
    return net.http_post(
        url := 'https://api.github.com/repos/1Andriy2/tradebot/actions/workflows/' || workflow || '/dispatches',
        headers := jsonb_build_object(
            'Authorization', 'Bearer ' || token,
            'Accept', 'application/vnd.github+json',
            'X-GitHub-Api-Version', '2022-11-28',
            'User-Agent', 'tradebot-supabase-cron'
        ),
        body := jsonb_build_object('ref', 'main')
    );
end;
$$;

-- Only the cron runner (postgres) may call it.
revoke all on function public.dispatch_workflow(text) from public, anon, authenticated;

select cron.schedule('trade-step', '2,17,32,47 * * * *', $$select public.dispatch_workflow('trade.yml')$$);
select cron.schedule('nightly-retrain', '17 2 * * *', $$select public.dispatch_workflow('retrain.yml')$$);
select cron.schedule('nightly-research', '7 4 * * *', $$select public.dispatch_workflow('research.yml')$$);
select cron.schedule('weekly-rl', '43 3 * * 0', $$select public.dispatch_workflow('rl-train.yml')$$);

-- cron.job_run_details and net._http_response are not cleaned automatically.
select cron.schedule('cron-history-cleanup', '30 5 * * *', $$
    delete from cron.job_run_details where end_time < now() - interval '7 days';
$$);
