# Deployment guide

Covers: running locally on Windows, environment variables, Streamlit Community Cloud, Supabase, rotating the
exposed Twelve Data key, removing a secret from git history, and suggested commit messages.

## 1. Run locally (Windows, VS Code, PowerShell)

Open the project folder in VS Code, then **Terminal > New Terminal** (PowerShell) and run these one at a time:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

- If `py` is not recognised, install Python 3.12 from python.org and tick **"Add python.exe to PATH"**, then reopen
  VS Code. (`pip` on its own is often not on PATH; `python -m pip` always works once Python is.)
- If `Activate.ps1` is blocked ("running scripts is disabled"), run once:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.
- `source .venv/bin/activate` and `&&` are Mac/Linux syntax and fail in PowerShell.
- The prompt shows `(.venv)` when the environment is active.

In Notepad, paste your keys after the `=` signs (no quotes, no spaces), save and close. Then:

```powershell
python -m capstone.check_keys      # shows only whether each key is set and works, never the key
python -m capstone.train           # walk-forward evaluation, weekly analysis, backtests, Monte Carlo (~30 s)
python -m capstone.predict         # today's forecast
python -m pytest -q tests          # unit tests
streamlit run app.py               # dashboard at http://localhost:8501
```

`run_all.bat` does the install and opens the dashboard in one double-click. The dashboard also fetches, trains
and predicts by itself when an update is due.

## 2. Environment variables

| Variable | Required | Used for |
| --- | --- | --- |
| `TWELVEDATA_API_KEY` | yes (for US/FX data) | US ETFs, ADRs, USD/INR history; sent in a header, never in a URL |
| `EXCHANGERATE_API_KEY` | recommended | latest USD/INR |
| `TEJ_API_KEY` | optional | NSE prices (Yahoo Finance daily is the fallback) |
| `ALPHAVANTAGE_API_KEY` | optional | fallback when no Twelve Data key is set |
| `SUPABASE_URL` | optional | `https://<project-ref>.supabase.co` |
| `SUPABASE_KEY` | optional | Supabase **service_role** key (server side only) |

Precedence: `.env` (read with python-dotenv) > environment variables > Streamlit Secrets. Keys are never printed:
`check_keys` shows only length, error messages and the dashboard's update log pass through `env.redact()`.
`.env`, `.env.*` (except `.env.example`) and `.streamlit/secrets.toml` are git-ignored. **Never zip and share the
project folder while `.env` is inside it.**

## 3. Streamlit Community Cloud

1. Push the code to GitHub (public repo `jayesh05bablr-code/capstone-forecast`). Check that no `.env` or
   `secrets.toml` is in the commit (`git status` must not list them).
2. On https://share.streamlit.io open the app **pri-capstone** > **Settings > General**: Python 3.12, main file
   `app.py`. (New app: **Create app > Deploy a public app from GitHub**, pick the repo, branch and `app.py`.)
3. **Settings > Secrets**: paste the contents of `.streamlit/secrets.toml.example` with your real values.
4. Save. The app restarts; the first start downloads prices, trains (about a minute with the weekly analysis) and
   predicts.
5. Verify before calling it deployed: open https://pri-capstone.streamlit.app in a private window, check that the
   seven tabs load (Morning forecast, Model accuracy, Weekly analysis, Backtesting, Monte Carlo, Business
   insights, More tools), that the sidebar shows "Last update ... OK", and that **Download data** builds the
   workbook.

`requirements.txt` pins `streamlit==1.50.0` and `pandas<3` (a newer Streamlit produced "Failed to fetch
dynamically imported module" on Community Cloud in Nov 2026). `.streamlit/config.toml` holds the server settings.
The keep-awake workflow (`.github/workflows/keep-awake.yml`) opens the app on a schedule so it does not sleep, and
`tests.yml` runs the unit tests on every push.

Free-plan limits: the app sleeps after about 12 hours without visitors; the disk is wiped on restart, so local CSV
logs are lost (use Supabase to keep the prediction log and results).

## 4. Supabase (optional persistence)

1. Create a free project at https://supabase.com (region: Mumbai, `ap-south-1`).
2. **SQL Editor > New query**: paste `migrations/001_init.sql` (same as `DATABASE_SCHEMA.sql`) and **Run**. It is
   safe to run again.
3. **Project Settings > API**: copy the Project URL and the `service_role` key into `.env` / Streamlit Secrets as
   `SUPABASE_URL` and `SUPABASE_KEY`. The service_role key can write; keep it only in server-side secrets.
4. Run `python -m capstone.train` and `python -m capstone.predict`; each ends with a line such as
   `Saved to Supabase: stocks 31, weekly_performance 3100, ...` or a clear error (with the key redacted).

Tables: `stocks` (stock mapping), `daily_predictions` (daily prediction log), `evaluation_runs` + `model_metrics`
(model evaluation), `strategy_configs`, `weekly_performance`, `backtest_runs` + `backtest_trades`,
`monte_carlo_runs`. Every write is an upsert on the natural key, and run ids are deterministic hashes of the
inputs, so re-running saves no duplicates. Without the two variables everything runs in local mode.

## 5. Rotate the exposed Twelve Data key (Pri must do this)

No Twelve Data key was found in the code or in any of the 14 commits of `capstone-forecast` (see
IMPLEMENTATION_AUDIT.md), but the key was shared in chat and sits in a local `.env`, so treat it as exposed:

1. Sign in at https://twelvedata.com > **Dashboard > API Keys**.
2. Create a new key (or **Regenerate**), then delete/revoke the old one.
3. Put the new key in your local `.env` (`TWELVEDATA_API_KEY=...`) and in Streamlit **Secrets**. Do not paste it in
   chat, email, screenshots or code.
4. `python -m capstone.check_keys` should say "Twelve Data works".
5. Do the same for any other key that was shared (exchangerate.dev, TejHQ).

## 6. If a secret is ever found in git history

Deleting the file in a new commit does not remove it from history. Rotate the key first (step 5), then:

```bash
pip install git-filter-repo
git clone --mirror https://github.com/jayesh05bablr-code/capstone-forecast.git
cd capstone-forecast.git
# remove a whole file from every commit
git filter-repo --invert-paths --path .env
# or replace a leaked value everywhere: put "OLDKEY==>REMOVED" in replace.txt
git filter-repo --replace-text ../replace.txt
git push --force --mirror      # only after review and approval: rewrites every commit id
```

Afterwards every collaborator must re-clone, and GitHub may keep cached views of old commits for a while (contact
GitHub support to purge them). For this repository the scan found nothing to remove, so no rewrite is planned.

## 7. Suggested commit messages for this upgrade

The upgrade branch already contains these commits (one per phase):

1. `Sync repository with the deployed project copy (live.py, download tab cache, keep-awake, Streamlit config)`
2. `Add implementation audit for reviewer upgrade`
3. `Read credentials with python-dotenv and redact secrets from logs`
4. `Compare naive, historical-average, Linear Regression and HistGradientBoosting on identical walk-forward rows`
5. `Add trading cost model and weekly Monday-open to Friday-close analysis`
6. `Add weekly forecast model, unfiltered vs cost-aware backtests and Monte Carlo simulation`
7. `Add optional Supabase persistence with SQL migration`
8. `Persist daily predictions with interval and model name; add analysis pipeline and fundamentals check`
9. `Redesign dashboard around the six reviewer tabs and extend the Excel export`
10. `Add capstone documentation, deployment guide and demo script`

If you upload through the GitHub website instead, use one commit named
`Upgrade capstone per reviewer feedback: baselines, weekly analysis, costs, backtests, Monte Carlo, Supabase`.
