# MF Intelligence — Weekly Automated Pipeline

A fully automated system that fetches mutual fund holdings data every week,
stores it in `data/`, and displays it in a rich analytics dashboard.

```
mf_project/
├── .github/
│   └── workflows/
│       └── weekly_pipeline.yml   ← GitHub Actions (runs every Sunday)
├── data/                         ← all output CSVs + JSON (git-tracked)
│   ├── AMC.csv                   ← your AMC list (add to repo)
│   ├── holdings_cache.json
│   ├── holdings_top.csv
│   ├── holdings_sector.csv
│   ├── holdings_asset_classes.csv
│   ├── holdings_mom_changes.csv
│   ├── conviction_picks.csv
│   ├── idcw_fy_output.csv
│   ├── final_schemes_master.csv
│   ├── funds_mapped.csv
│   └── run_metadata.json
├── scripts/
│   └── pipeline.py               ← headless data fetcher
├── dashboard.py                  ← Streamlit visualization app
├── requirements.txt
└── README.md
```

## Quick start

### 1 — Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd mf_project
pip install -r requirements.txt
```

### 2 — Add your AMC list
Copy `AMC.csv` into the `data/` folder (one AMC name per row, column header `AMC`).

### 3 — Run the pipeline manually (first time)
```bash
python scripts/pipeline.py --data-dir data --amc-file data/AMC.csv
```
This will take 60–90 minutes for the full AMFI universe on the first run.
Subsequent weekly runs are much faster because of the holdings cache.

### 4 — Launch the dashboard
```bash
streamlit run dashboard.py
```
Open http://localhost:8501 in your browser.

---

## GitHub Actions automation

### How it works
The workflow file `.github/workflows/weekly_pipeline.yml` runs every **Sunday at 01:00 UTC** (06:30 IST).

It:
1. Installs dependencies
2. Restores the holdings cache from GitHub's cache store
3. Runs `scripts/pipeline.py`
4. Commits updated `data/` CSVs back to the repo
5. Uploads all data files as a downloadable artifact

### One-time setup steps

#### A. Push your repo to GitHub
```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

#### B. Enable Actions write permission
Go to your repo → **Settings → Actions → General → Workflow permissions**
and select **"Read and write permissions"**, then save.

#### C. Add AMC.csv to the repo
```bash
cp /path/to/AMC.csv data/AMC.csv
git add data/AMC.csv
git commit -m "Add AMC list"
git push
```

#### D. Trigger the first run manually
Go to your repo → **Actions → MF Data Pipeline — Weekly Refresh** → **Run workflow**.

You can also set `force_refresh = true` in the manual trigger to re-fetch
all holdings regardless of cache.

### Checking run status
Go to **Actions** tab in your GitHub repo to see live logs for each step.

---

## Dashboard tabs

| Tab | What you see |
|---|---|
| 📡 Live Signals | Conviction buys, fresh exits, new entries this week |
| 📈 IDCW Rankings | Fund yield league table with sparklines |
| 🏗️ Holdings Map | Treemap + sunburst of all current holdings |
| 📅 MoM Flow | Stock-level buy/sell flow, waterfall chart, weekly trend |
| 🔬 Stock X-Ray | Ownership timeline for any individual stock |
| 🏦 Fund DNA | Holdings, sector, asset mix for any fund |
| 📊 Sector Pulse | Sector rotation heatmap across all funds |
| 🕰️ History | All weekly runs summarised + downloadable CSVs |

---

## Deploy dashboard publicly (optional)

### Streamlit Cloud (free)
1. Push this repo to GitHub
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `dashboard.py`
4. Click **Deploy** — dashboard is live in ~2 minutes

The dashboard reads the `data/` folder from your repo, so every Sunday after
the pipeline commits new data, the dashboard auto-refreshes.

### Environment variables (not required, but useful)
No secrets are needed — yfinance and AMFI are public APIs.

---

## Adjusting the schedule

To change the run day/time, edit the `cron` line in `weekly_pipeline.yml`:

```yaml
- cron: "0 1 * * 0"   # Sunday 01:00 UTC
```

Cron format: `minute hour day month weekday`  
Examples:
- Every Monday 02:00 UTC: `0 2 * * 1`  
- Every day at midnight: `0 0 * * *`  
- Every Saturday + Sunday: `0 1 * * 6,0`  

Use https://crontab.guru to build cron expressions.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `TypeError: Object of type method is not JSON serializable` | Already fixed in `pipeline.py` — update from latest version |
| Pipeline times out (120 min limit) | Reduce `--batch-size` or split into multiple workflows |
| No data in dashboard | Check that `data/` has CSV files; run pipeline manually first |
| Yahoo Finance rate limit errors | Increase `--sleep-min` / `--sleep-max` arguments |
| Holdings cache stale | Run with `--force-refresh` flag |
