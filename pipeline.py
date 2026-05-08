"""
scripts/pipeline.py
====================
Headless data pipeline designed to run as three independent GitHub Actions
jobs, each fitting comfortably within a 60-minute timeout.

Usage (full run):
    python scripts/pipeline.py --data-dir data --amc-file data/AMC.csv

Usage (individual steps — used by GitHub Actions):
    python scripts/pipeline.py --step map      --data-dir data
    python scripts/pipeline.py --step nav      --data-dir data --time-budget 50
    python scripts/pipeline.py --step holdings --data-dir data --time-budget 50

--time-budget N   Stop fetching after N minutes and save whatever was
                  completed. The next run picks up where this one left off.
--force-refresh   Ignore daily holdings cache and re-fetch everything.

Output files (all in --data-dir):
    funds_mapped.csv             ISIN → Yahoo symbol mapping
    final_schemes_master.csv     AMC / Plan / Option per scheme
    idcw_fy_output.csv           FY-wise NAV + IDCW yield metrics
    holdings_cache.json          raw yfinance data (keyed by symbol)
    holdings_top.csv             flattened top-holdings per fund
    holdings_sector.csv          sector weightings per fund
    holdings_asset_classes.csv   asset-class allocation per fund
    holdings_mom_changes.csv     month-on-month weight changes
    conviction_picks.csv         high-conviction multi-fund buy signals
    run_metadata.json            run timestamp + summary stats
"""

import argparse
import json
import math
import os
import re
import time
import random
from datetime import datetime, date
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MF data pipeline")
    p.add_argument("--step",         default="all",
                   choices=["all", "map", "nav", "holdings"],
                   help="Which step to run (default: all)")
    p.add_argument("--data-dir",     default="data")
    p.add_argument("--amc-file",     default="data/AMC.csv")
    p.add_argument("--start-date",   default="2018-03-15")
    p.add_argument("--batch-size",   type=int,   default=150)
    p.add_argument("--sleep-min",    type=float, default=8)
    p.add_argument("--sleep-max",    type=float, default=14)
    p.add_argument("--time-budget",  type=int,   default=999,
                   help="Stop fetching after this many minutes (save progress and exit cleanly)")
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────
# WALL-CLOCK BUDGET GUARD
# ──────────────────────────────────────────────────────────
class Budget:
    """Simple wall-clock timer. Call .ok() before each expensive request."""
    def __init__(self, minutes: int):
        self._deadline = time.monotonic() + minutes * 60

    def ok(self) -> bool:
        return time.monotonic() < self._deadline

    def remaining_min(self) -> float:
        return max(0, (self._deadline - time.monotonic()) / 60)


# ──────────────────────────────────────────────────────────
# JSON SERIALISATION SAFETY
# ──────────────────────────────────────────────────────────
def _to_json_safe(obj, _depth=0):
    """Convert any yfinance / pandas / numpy object to plain JSON-safe Python."""
    if _depth > 20:
        return None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x, _depth + 1) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() if not pd.isnull(obj) else None
        if isinstance(obj, pd.DataFrame):
            return None if obj.empty else _to_json_safe(
                obj.reset_index().to_dict(orient="records"), _depth + 1)
        if isinstance(obj, pd.Series):
            return None if obj.empty else _to_json_safe(obj.to_dict(), _depth + 1)
    except Exception:
        pass
    if hasattr(obj, "items") and callable(obj.items):
        return {str(k): _to_json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x, _depth + 1) for x in obj]
    if callable(obj):
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:
        return None


def safe_json_dump(obj, path):
    """json.dump with automatic fallback through _to_json_safe."""
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except (TypeError, ValueError):
        with open(path, "w") as f:
            json.dump(_to_json_safe(obj), f)


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return {}


# ──────────────────────────────────────────────────────────
# STEP 1 — AMFI fetch + ISIN → Yahoo mapping
# ──────────────────────────────────────────────────────────
AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


def fetch_amfi():
    print("Fetching AMFI data …")
    text = requests.get(AMFI_URL, timeout=30).text
    lines = [l for l in text.splitlines() if ";" in l]
    df = pd.read_csv(StringIO("\n".join(lines)), sep=";", engine="python")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df = df.dropna(subset=["Scheme Code"])
    df["Scheme Code"] = df["Scheme Code"].astype(int)
    df["ISIN"] = df["ISIN Div Payout/ ISIN Growth"].fillna(df["ISIN Div Reinvestment"])
    df = df.dropna(subset=["ISIN"])
    print(f"  {len(df)} schemes loaded from AMFI")
    return df


def isin_to_yahoo(isin: str) -> dict | None:
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        r = requests.get(url,
                         params={"q": isin, "quotesCount": 5, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"},
                         timeout=10)
        for q in r.json().get("quotes", []):
            if q.get("quoteType") in ("MUTUALFUND", "ETF", "EQUITY"):
                return {"yahoo_symbol": q["symbol"],
                        "yahoo_quote_type": q.get("quoteType"),
                        "yahoo_exchange": q.get("exchange")}
    except Exception:
        pass
    return None


def run_map(amfi: pd.DataFrame, existing_mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Map ISINs to Yahoo symbols, skipping any ISIN already in existing_mapped.
    Returns the full merged DataFrame (existing + newly mapped).
    """
    already = set()
    if not existing_mapped.empty and "ISIN" in existing_mapped.columns:
        already = set(existing_mapped["ISIN"].dropna())

    isins = [i for i in amfi["ISIN"].unique() if i not in already]
    print(f"  {len(already)} ISINs already mapped, {len(isins)} new to look up")

    results = []
    for i, isin in enumerate(isins, 1):
        m = isin_to_yahoo(isin)
        if m:
            m["ISIN"] = isin
            results.append(m)
        time.sleep(0.3)
        if i % 200 == 0:
            print(f"  … {i}/{len(isins)} ISINs processed")

    new_map = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["ISIN", "yahoo_symbol", "yahoo_quote_type", "yahoo_exchange"])

    # Merge new entries into existing
    combined_map = pd.concat(
        [existing_mapped, amfi.merge(new_map, on="ISIN", how="inner")],
        ignore_index=True
    ).drop_duplicates(subset=["ISIN"])

    # Return full amfi with yahoo columns merged in
    return amfi.merge(
        combined_map[["ISIN", "yahoo_symbol", "yahoo_quote_type", "yahoo_exchange"]],
        on="ISIN", how="left"
    )


# ──────────────────────────────────────────────────────────
# STEP 1b — Scheme attribute extraction
# ──────────────────────────────────────────────────────────
def _clean(x):
    x = str(x).lower()
    x = re.sub(r"mutual fund|asset management company", "", x)
    return re.sub(r"[^a-z0-9 ]", "", x).strip()


def extract_attributes(funds_df: pd.DataFrame, amc_path: str) -> pd.DataFrame:
    try:
        amc_list = pd.read_csv(amc_path)
    except FileNotFoundError:
        print(f"  Warning: {amc_path} not found, using built-in fallback list")
        amc_list = pd.DataFrame({"AMC": [
            "Aditya Birla Sun Life Mutual Fund", "Axis Mutual Fund",
            "Bajaj Finserv Mutual Fund", "Bandhan Mutual Fund",
            "Bank of India Mutual Fund", "Canara Robeco Mutual Fund",
            "DSP Mutual Fund", "Edelweiss Mutual Fund",
            "Franklin Templeton Mutual Fund", "HDFC Mutual Fund",
            "HSBC Mutual Fund", "ICICI Prudential Mutual Fund",
            "IDFC Mutual Fund", "Invesco Mutual Fund",
            "ITI Mutual Fund", "Kotak Mahindra Mutual Fund",
            "LIC Mutual Fund", "Mirae Asset Mutual Fund",
            "Motilal Oswal Mutual Fund", "Nippon India Mutual Fund",
            "PGIM India Mutual Fund", "PPFAS Mutual Fund",
            "Quant Mutual Fund", "SBI Mutual Fund",
            "Sundaram Mutual Fund", "Tata Mutual Fund",
            "Taurus Mutual Fund", "Union Mutual Fund",
            "UTI Mutual Fund", "WhiteOak Capital Mutual Fund",
            "360 ONE Mutual Fund", "NJ Mutual Fund",
        ]})

    amc_list["clean"] = amc_list["AMC"].apply(_clean)
    funds_df = funds_df.copy()
    funds_df["clean"] = funds_df["Scheme Name"].apply(_clean)

    keys = sorted(set(amc_list["clean"].tolist() +
                      amc_list["clean"].str.split().str[0].dropna().tolist()),
                  key=len, reverse=True)
    pat = re.compile(r"^(" + "|".join(map(re.escape, keys)) + r")\b")
    mapping = {}
    for _, row in amc_list.iterrows():
        mapping[row["clean"]] = row["AMC"]
        mapping[row["clean"].split()[0]] = row["AMC"]

    def match(s):
        m = pat.search(s)
        return mapping.get(m.group(1), "Unknown") if m else "Unknown"

    funds_df["AMC"] = funds_df["clean"].apply(match)

    funds_df["Plan"] = "Unknown"
    funds_df.loc[funds_df["clean"].str.contains(r"\bdirect\b", na=False), "Plan"] = "Direct"
    funds_df.loc[funds_df["clean"].str.contains(r"\bregular\b", na=False), "Plan"] = "Regular"

    funds_df["Option"] = "Unknown"
    funds_df.loc[funds_df["clean"].str.contains(r"\bgrowth\b", na=False), "Option"] = "Growth"
    funds_df.loc[funds_df["clean"].str.contains(
        r"\b(?:idcw|dividend|div\b|div\.|payout|reinvestment)\b", na=False
    ), "Option"] = "IDCW"

    return funds_df[["yahoo_symbol", "Scheme Name", "AMC", "Plan", "Option"]]


# ──────────────────────────────────────────────────────────
# STEP 2 — NAV / dividend download
# ──────────────────────────────────────────────────────────
def get_fy(d):
    return d.year if d.month >= 4 else d.year - 1


def download_nav(funds_df: pd.DataFrame, existing_fy: pd.DataFrame,
                 start_date: str, batch_size: int, sleep_range: tuple,
                 budget: Budget) -> pd.DataFrame:
    """
    Download NAV history for funds not already in existing_fy.
    Respects budget — stops early and returns whatever was completed.
    """
    funds = funds_df[["yahoo_symbol", "Scheme Name"]].drop_duplicates()
    syms = funds["yahoo_symbol"].dropna().tolist()
    name_map = dict(zip(funds["yahoo_symbol"], funds["Scheme Name"]))
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Skip symbols already in existing data (they were fetched in a prior run)
    already_done = set()
    if not existing_fy.empty and "yahoo_symbol" in existing_fy.columns:
        already_done = set(existing_fy["yahoo_symbol"].dropna())

    pending = [s for s in syms if s not in already_done]
    print(f"  {len(already_done)} symbols already in FY data, {len(pending)} pending")

    all_rows = []
    total_batches = -(-len(pending) // batch_size)   # ceiling division

    for i, start in enumerate(range(0, len(pending), batch_size), 1):
        if not budget.ok():
            print(f"  Time budget reached after {i-1}/{total_batches} batches — saving progress")
            break

        batch = pending[start:start + batch_size]
        print(f"  Batch {i}/{total_batches} ({len(batch)} symbols) — {budget.remaining_min():.0f} min left")

        try:
            raw = yf.download(batch, start=start_date, end=end_date,
                              group_by="ticker", actions=True,
                              auto_adjust=False, threads=False, progress=False)
            for sym in batch:
                if sym not in raw:
                    continue
                df = raw[sym].reset_index()
                df["yahoo_symbol"] = sym
                df["fund_name"] = name_map.get(sym, sym)
                df["FY"] = df["Date"].apply(get_fy)
                all_rows.append(df[["Date", "Close", "Dividends",
                                    "yahoo_symbol", "fund_name", "FY"]])
        except Exception as e:
            print(f"  Batch {i} error: {e}")
            time.sleep(60)
            continue

        if i < total_batches:
            time.sleep(random.uniform(*sleep_range))

    if not all_rows:
        print("  No new NAV data fetched this run")
        return existing_fy

    new_data = pd.concat(all_rows, ignore_index=True)
    new_data.rename(columns={"Close": "NAV", "Dividends": "Dividend"}, inplace=True)
    new_data["Dividend"] = new_data["Dividend"].fillna(0)

    fy_new = _calc_fy_metrics(new_data)

    # Merge with existing FY data
    combined = pd.concat([existing_fy, fy_new], ignore_index=True).drop_duplicates(
        subset=["yahoo_symbol", "fund_name", "FY"]
    )
    print(f"  FY metrics: {len(fy_new)} new rows, {len(combined)} total")
    return combined


def _calc_fy_metrics(data: pd.DataFrame) -> pd.DataFrame:
    nav_fy = (data.sort_values("Date")
              .groupby(["yahoo_symbol", "fund_name", "FY"])
              .agg(April_NAV=("NAV", "first"), March_NAV=("NAV", "last"))
              .reset_index())
    div_fy = (data.groupby(["yahoo_symbol", "fund_name", "FY"])["Dividend"]
              .sum().reset_index())
    fy = nav_fy.merge(div_fy, on=["yahoo_symbol", "fund_name", "FY"], how="left")
    fy["Avg_NAV"] = (fy["April_NAV"] + fy["March_NAV"]) / 2
    fy["IDCW_Yield_pct"] = (fy["Dividend"] / fy["Avg_NAV"] * 100).round(2)
    fy["FY"] = fy["FY"].astype(str) + "-" + (fy["FY"] + 1).astype(str)
    return fy


# ──────────────────────────────────────────────────────────
# STEP 3 — Holdings fetch
# ──────────────────────────────────────────────────────────
def _safe_get(fd, attr):
    try:
        v = getattr(fd, attr)
        if callable(v) and not hasattr(v, "__self__"):
            return None
        return _to_json_safe(v)
    except Exception as e:
        return {"_error": str(e)}


def fetch_one_holding(symbol: str) -> dict:
    result = {k: None for k in [
        "symbol", "fetched_at", "top_holdings", "equity_holdings",
        "bond_holdings", "sector_weightings", "asset_classes", "bond_ratings",
        "fund_operations", "fund_overview", "quote_type", "description", "error"
    ]}
    result["symbol"] = symbol
    result["fetched_at"] = datetime.now().strftime("%Y-%m-%d")
    try:
        fd = yf.Ticker(symbol).funds_data
        if fd is None:
            result["error"] = "No funds_data"
            return result
        for attr in ["top_holdings", "equity_holdings", "bond_holdings",
                     "sector_weightings", "asset_classes", "bond_ratings",
                     "fund_operations", "fund_overview"]:
            result[attr] = _safe_get(fd, attr)
        result["quote_type"] = _to_json_safe(getattr(fd, "quote_type", None))
        result["description"] = _to_json_safe(getattr(fd, "description", None))
    except Exception as e:
        result["error"] = str(e)
    return result


def fetch_holdings(symbols: list, cache: dict, budget: Budget,
                   force: bool = False) -> dict:
    """
    Fetch holdings for symbols not yet in cache (or stale).
    Saves to cache incrementally every 50 symbols.
    Stops when budget is exhausted.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    pending = [s for s in symbols
               if force or s not in cache or cache[s].get("fetched_at") != today]
    print(f"  {len(symbols) - len(pending)} symbols current in cache, {len(pending)} to fetch")

    for i, sym in enumerate(pending, 1):
        if not budget.ok():
            print(f"  Time budget reached after {i-1}/{len(pending)} holdings — saving progress")
            break
        cache[sym] = fetch_one_holding(sym)
        time.sleep(1.0)

        if i % 50 == 0:
            print(f"  {i}/{len(pending)} fetched — {budget.remaining_min():.0f} min left")

    return cache


def snapshot_month(cache: dict) -> dict:
    """Store this month's top_holdings snapshot in cache[sym]['snapshots'][YYYY-MM]."""
    month = datetime.now().strftime("%Y-%m")
    for data in cache.values():
        snaps = data.setdefault("snapshots", {})
        if data.get("top_holdings") and isinstance(data["top_holdings"], list):
            snaps[month] = data["top_holdings"]
        while len(snaps) > 13:
            del snaps[sorted(snaps)[0]]
    return cache


# ──────────────────────────────────────────────────────────
# CSV BUILDERS
# ──────────────────────────────────────────────────────────
RUN_DATE = datetime.now().strftime("%Y-%m-%d")


def build_top_holdings(cache: dict, name_map: dict) -> pd.DataFrame:
    rows = []
    for sym, d in cache.items():
        th = d.get("top_holdings")
        if not th or not isinstance(th, list):
            continue
        for h in th:
            rows.append({
                "run_date": RUN_DATE,
                "fund_symbol": sym,
                "fund_name": name_map.get(sym, sym),
                "holding_symbol": h.get("symbol", h.get("index", "")),
                "holding_name": h.get("holdingName", ""),
                "weight_pct": round((h.get("holdingPercent") or 0) * 100, 4),
            })
    return pd.DataFrame(rows)


def build_sector(cache: dict, name_map: dict) -> pd.DataFrame:
    rows = []
    for sym, d in cache.items():
        sw = d.get("sector_weightings")
        if not sw or not isinstance(sw, list):
            continue
        for entry in sw:
            for k, v in entry.items():
                if k != "index":
                    rows.append({
                        "run_date": RUN_DATE,
                        "fund_symbol": sym,
                        "fund_name": name_map.get(sym, sym),
                        "sector": k,
                        "weight_pct": round((v or 0) * 100, 4),
                    })
    return pd.DataFrame(rows)


def build_asset_classes(cache: dict, name_map: dict) -> pd.DataFrame:
    rows = []
    for sym, d in cache.items():
        ac = d.get("asset_classes")
        if not ac or not isinstance(ac, dict):
            continue
        row = {"run_date": RUN_DATE, "fund_symbol": sym,
               "fund_name": name_map.get(sym, sym)}
        for k, v in ac.items():
            row[k] = v.get("value", v) if isinstance(v, dict) else v
        rows.append(row)
    return pd.DataFrame(rows)


def build_mom(cache: dict, name_map: dict) -> pd.DataFrame:
    rows = []
    for sym, d in cache.items():
        snaps = d.get("snapshots", {})
        if len(snaps) < 2:
            continue
        months = sorted(snaps)
        pm, cm = months[-2], months[-1]

        def to_map(snap):
            return {h.get("symbol", h.get("index", "")): (h.get("holdingPercent") or 0)
                    for h in (snap or [])}

        prev, curr = to_map(snaps[pm]), to_map(snaps[cm])
        for stock in set(prev) | set(curr):
            pw, cw = prev.get(stock, 0), curr.get(stock, 0)
            if pw == 0 and cw == 0:
                continue
            chg = cw - pw
            status = ("NEW" if pw == 0 and cw > 0 else
                      "EXITED" if pw > 0 and cw == 0 else "CHANGED")
            rows.append({
                "run_date": RUN_DATE,
                "fund_symbol": sym,
                "fund_name": name_map.get(sym, sym),
                "holding_symbol": stock,
                "prev_month": pm, "curr_month": cm,
                "prev_weight_pct": round(pw * 100, 4),
                "curr_weight_pct": round(cw * 100, 4),
                "change_pct": round(chg * 100, 4),
                "status": status,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_conviction(mom_df: pd.DataFrame) -> pd.DataFrame:
    if mom_df.empty:
        return pd.DataFrame()
    buying = mom_df[mom_df["change_pct"] > 0].copy()
    if buying.empty:
        return pd.DataFrame()
    agg = (buying.groupby("holding_symbol")
           .agg(fund_count=("fund_symbol", "nunique"),
                total_weight_change=("change_pct", "sum"),
                avg_weight_change=("change_pct", "mean"),
                new_entries=("status", lambda x: (x == "NEW").sum()),
                funds_buying=("fund_name", lambda x: ", ".join(sorted(x.unique())[:5])))
           .reset_index())
    agg["conviction_score"] = (
        agg["fund_count"] * 3 +
        agg["total_weight_change"] * 2 +
        agg["new_entries"] * 5
    ).round(2)
    agg["run_date"] = RUN_DATE
    return agg.sort_values("conviction_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# APPEND-SAFE CSV WRITER
# ──────────────────────────────────────────────────────────
def append_csv(df: pd.DataFrame, path: str):
    """Append df rows to path, deduplicating. Creates file if absent."""
    if df is None or df.empty:
        return
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            combined = pd.concat([existing, df], ignore_index=True).drop_duplicates()
        except Exception:
            combined = df
    else:
        combined = df
    combined.to_csv(path, index=False)
    print(f"  Saved {path} ({len(combined):,} total rows)")


# ──────────────────────────────────────────────────────────
# STEP RUNNERS
# ──────────────────────────────────────────────────────────
def step_map(args, P):
    print("\n── STEP 1: MAP ISINs ──────────────────────────────────")
    amfi = fetch_amfi()

    existing = pd.read_csv(P("funds_mapped.csv")) if os.path.exists(P("funds_mapped.csv")) else pd.DataFrame()
    funds = run_map(amfi, existing)
    funds.to_csv(P("funds_mapped.csv"), index=False)
    print(f"  Saved funds_mapped.csv ({len(funds)} rows)")

    scheme = extract_attributes(funds, args.amc_file)
    scheme.to_csv(P("final_schemes_master.csv"), index=False)
    print(f"  Saved final_schemes_master.csv ({len(scheme)} rows)")


def step_nav(args, P, budget):
    print("\n── STEP 2: NAV DOWNLOAD ───────────────────────────────")
    if not os.path.exists(P("funds_mapped.csv")):
        print("  funds_mapped.csv not found — run --step map first")
        return

    funds = pd.read_csv(P("funds_mapped.csv"))
    existing_fy = pd.read_csv(P("idcw_fy_output.csv")) if os.path.exists(P("idcw_fy_output.csv")) else pd.DataFrame()

    fy = download_nav(funds, existing_fy, args.start_date, args.batch_size,
                      (args.sleep_min, args.sleep_max), budget)
    append_csv(fy, P("idcw_fy_output.csv"))


def step_holdings(args, P, budget):
    print("\n── STEP 3: HOLDINGS ───────────────────────────────────")
    if not os.path.exists(P("funds_mapped.csv")):
        print("  funds_mapped.csv not found — run --step map first")
        return

    funds = pd.read_csv(P("funds_mapped.csv"))
    scheme = (pd.read_csv(P("final_schemes_master.csv"))
              if os.path.exists(P("final_schemes_master.csv"))
              else extract_attributes(funds, args.amc_file))
    name_map = dict(zip(scheme["yahoo_symbol"], scheme["Scheme Name"]))
    syms = funds["yahoo_symbol"].dropna().unique().tolist()

    # Load + update cache
    cache = load_json(P("holdings_cache.json"))
    cache = fetch_holdings(syms, cache, budget, force=args.force_refresh)
    cache = snapshot_month(cache)
    safe_json_dump(cache, P("holdings_cache.json"))
    print(f"  holdings_cache.json saved ({len(cache)} entries)")

    # Build and append CSVs
    top_df   = build_top_holdings(cache, name_map)
    sec_df   = build_sector(cache, name_map)
    asset_df = build_asset_classes(cache, name_map)
    mom_df   = build_mom(cache, name_map)
    conv_df  = build_conviction(mom_df)

    append_csv(top_df,   P("holdings_top.csv"))
    append_csv(sec_df,   P("holdings_sector.csv"))
    append_csv(asset_df, P("holdings_asset_classes.csv"))
    append_csv(mom_df,   P("holdings_mom_changes.csv"))
    append_csv(conv_df,  P("conviction_picks.csv"))

    # Metadata
    meta = {
        "last_run": datetime.now().isoformat(),
        "step": "holdings",
        "funds_mapped": int(funds["yahoo_symbol"].notna().sum()),
        "holdings_cached": len(cache),
        "top_holdings_rows": len(top_df),
        "mom_rows": len(mom_df) if not mom_df.empty else 0,
        "conviction_picks": len(conv_df) if not conv_df.empty else 0,
    }
    safe_json_dump(meta, P("run_metadata.json"))
    print("\n  Summary:")
    for k, v in meta.items():
        print(f"    {k}: {v}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    P = lambda f: os.path.join(args.data_dir, f)
    budget = Budget(args.time_budget)

    print("=" * 60)
    print(f"MF PIPELINE  |  step={args.step}  |  budget={args.time_budget} min")
    print(f"started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.step in ("all", "map"):
        step_map(args, P)

    if args.step in ("all", "nav"):
        step_nav(args, P, budget)

    if args.step in ("all", "holdings"):
        step_holdings(args, P, budget)

    print("\n" + "=" * 60)
    print("PIPELINE DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
