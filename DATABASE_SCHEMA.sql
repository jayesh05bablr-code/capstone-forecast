-- Capstone forecast: Supabase / PostgreSQL schema (migration 001).
-- Run once in the Supabase dashboard: SQL editor > New query > paste > Run.
-- Safe to re-run: every object uses IF NOT EXISTS.
--
-- Write access: the app writes with the service_role key (kept in Streamlit Secrets / .env),
-- which bypasses Row Level Security. RLS is enabled on every table with a read-only policy, so
-- an anon key can only SELECT.

create table if not exists stocks (
    symbol        text primary key check (symbol = upper(symbol) and length(symbol) between 1 and 20),
    india_sector  text,
    adr           text,
    us_etf        text,
    us_sector     text,
    fx_exposure   numeric,
    updated_at    timestamptz not null default now()
);

create table if not exists daily_predictions (
    id              bigint generated always as identity primary key,
    symbol          text not null references stocks(symbol),
    as_of_close     date not null,                -- the NSE close the forecast starts from
    target_date     date,                         -- the session whose open is forecast
    model_name      text not null,
    close           numeric not null check (close > 0),
    pred_gap_pct    numeric not null,
    q10_gap_pct     numeric,
    q90_gap_pct     numeric,
    pred_open       numeric not null check (pred_open > 0),
    pred_open_low   numeric,
    pred_open_high  numeric,
    direction       text check (direction in ('up', 'down', 'flat')),
    created_at      timestamptz not null default now(),
    unique (symbol, as_of_close, model_name)
);

create table if not exists evaluation_runs (
    run_id       uuid primary key,                -- deterministic: same data + settings = same id
    kind         text not null check (kind in ('daily_gap', 'weekly_return')),
    test_start   date,
    test_end     date,
    n_test_rows  integer,
    details      jsonb,
    created_at   timestamptz not null default now()
);

create table if not exists model_metrics (
    run_id             uuid not null references evaluation_runs(run_id) on delete cascade,
    model              text not null,
    symbol             text not null default 'ALL',  -- 'ALL' = every stock pooled
    n                  integer not null check (n >= 0),
    mae_pct            numeric,
    rmse_pct           numeric,
    bias_pct           numeric,
    direction_acc_pct  numeric check (direction_acc_pct between 0 and 100),
    created_at         timestamptz not null default now(),
    primary key (run_id, model, symbol)
);

create table if not exists strategy_configs (
    config_hash  text primary key,                -- sha256 of the canonical JSON below
    config       jsonb not null,
    created_at   timestamptz not null default now()
);

create table if not exists weekly_performance (
    symbol          text not null references stocks(symbol),
    week_start      date not null,                -- Monday of the calendar week
    entry_date      date not null,
    exit_date       date not null check (exit_date >= entry_date),
    sessions        smallint not null check (sessions between 1 and 7),
    entry_open      numeric not null,
    exit_close      numeric not null,
    weekly_return   numeric,
    valid           boolean not null,
    exclude_reason  text,
    updated_at      timestamptz not null default now(),
    primary key (symbol, week_start)
);

create table if not exists backtest_runs (
    run_id                  uuid primary key,
    config_hash             text not null references strategy_configs(config_hash),
    strategy                text not null,
    signal_model            text not null,
    first_week              date,
    last_week               date,
    weeks                   integer,
    trades                  integer,
    winning_trades          integer,
    losing_trades           integer,
    win_rate_pct            numeric,
    gross_pnl               numeric,
    total_costs             numeric,
    net_pnl                 numeric,
    ending_capital          numeric,
    cumulative_return_pct   numeric,
    max_drawdown_pct        numeric,
    sharpe_annualised       numeric,
    created_at              timestamptz not null default now()
);

create table if not exists backtest_trades (
    run_id          uuid not null references backtest_runs(run_id) on delete cascade,
    symbol          text not null references stocks(symbol),
    week_start      date not null,
    entry_date      date not null,
    exit_date       date not null,
    pred_pct        numeric,
    qty             integer not null check (qty > 0),
    entry_price     numeric not null,
    exit_price      numeric not null,
    gross_pnl       numeric not null,
    explicit_cost   numeric not null,
    spread_cost     numeric not null,
    slippage_cost   numeric not null,
    total_cost      numeric not null,
    net_pnl         numeric not null,
    primary key (run_id, symbol, week_start)    -- one trade per stock per week: no duplicates
);

create table if not exists monte_carlo_runs (
    run_id             uuid primary key,
    backtest_run_id    uuid not null references backtest_runs(run_id) on delete cascade,
    simulations        integer not null check (simulations > 0),
    seed               integer not null,
    block_weeks        integer not null,
    horizon_weeks      integer not null,
    initial_capital    numeric not null,
    prob_loss_pct      numeric,
    percentiles        jsonb not null,           -- ending capital / return / drawdown at 5,25,50,75,95
    created_at         timestamptz not null default now()
);

create index if not exists daily_predictions_date_idx on daily_predictions (as_of_close desc);
create index if not exists weekly_performance_week_idx on weekly_performance (week_start);

-- Row Level Security: read-only for anon/authenticated; writes only via service_role.
do $$
declare t text;
begin
  foreach t in array array['stocks','daily_predictions','evaluation_runs','model_metrics','strategy_configs',
                           'weekly_performance','backtest_runs','backtest_trades','monte_carlo_runs'] loop
    execute format('alter table %I enable row level security', t);
    if not exists (select 1 from pg_policies where tablename = t and policyname = 'read_only') then
      execute format('create policy read_only on %I for select using (true)', t);
    end if;
  end loop;
end $$;
