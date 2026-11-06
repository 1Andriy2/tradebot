-- Two demo trades that were never recorded. On 2026-11-04 `rl` inherited ma_cross's positions and the
-- step crashed while their take-profits filled on the exchange (fixed in 683ffa2). Entries, fees and
-- levels are from ma_cross's stored ledger. The exit bar is the first candle that reached the level, and
-- the demo balance rose at both moments. The exit fee is the log's usual 0.1%, an estimate like all of them.
insert into trades (agent_id, venue, symbol, side, entry_time, exit_time, entry_price, exit_price,
                    pnl, fees, return_pct, entry_reason, exit_reason)
select 'ma_cross', 'demo', symbol, 'long', entry_time, exit_time, entry_price, exit_price,
       cash_flow + units * exit_price * 0.999, entry_fee + units * exit_price * 0.001,
       (cash_flow + units * exit_price * 0.999) / (units * entry_price), 'signal', 'take_profit'
from (values
    ('ETH/USDT:USDT', timestamptz '2026-11-04 06:15+00', timestamptz '2026-11-04 14:45+00', 2295.0::float8,
     2363.65469458031::float8, 0.108::float8, -248.10786::float8, 0.24786::float8),
    ('BTC/USDT:USDT', timestamptz '2026-11-04 06:30+00', timestamptz '2026-11-04 15:00+00', 77863.8::float8,
     79759.0322981407::float8, 0.0032::float8, -249.41332416::float8, 0.24916416::float8)
) as lost (symbol, entry_time, exit_time, entry_price, exit_price, units, cash_flow, entry_fee)
on conflict (agent_id, venue, symbol, entry_time, exit_time) do nothing;
