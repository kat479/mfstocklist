"""
scripts/pipeline.py
====================
Headless weekly data pipeline — runs in GitHub Actions (no Streamlit).

Usage:
    python scripts/pipeline.py --data-dir data --amc-file data/AMC.csv
    python scripts/pipeline.py --data-dir data --force-refresh

Outputs (all written to --data-dir):
    holdings_cache.json          raw yfinance holdings cache (keyed by symbol)
    funds_mapped.csv             ISIN → Yahoo symbol mapping
    idcw_fy_output.csv           FY-wise NAV + dividend metrics
    final_schemes_master.csv     scheme attributes (AMC / Plan / Option)
    holdings_top.csv             flattened top-holdings per fund
    holdings_sector.csv          sector weightings per fund
    holdings_asset_classes.csv   asset-class allocation per fund
    holdings_mom_changes.csv     month-on-month weight changes per stock
    conviction_picks.csv         high-conviction buy signals this week
    run_metadata.json            timestamp + summary stats for the dashboard
"""

import argparse
import json
import math
import os
import re
import time
import random
from collections import defaultdict
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
    p = argparse.ArgumentParser(description="MF weekly data pipeline")
    p.add_argument("--data-dir",      default="data",      help="Output directory")
    p.add_argument("--amc-file",      default="data/AMC.csv", help="AMC list CSV")
    p.add_argument("--start-date",    default="2018-03-15", help="NAV history start")
    p.add_argument("--batch-size",    type=int, default=300, help="yfinance batch size")
    p.add_argument("--sleep-min",     type=float, default=15, help="Min sleep between batches (s)")
    p.add_argument("--sleep-max",     type=float, default=25, help="Max sleep between batches (s)")
    p.add_argument("--force-refresh", action="store_true",  help="Ignore holdings cache")
    return p.parse_args()


# ──────────────────────────────────────────────────────────
# JSON SERIALISATION SAFETY
# ──────────────────────────────────────────────────────────
def _to_json_safe(obj, _depth=0):
    """Recursively convert any yfinance/pandas/numpy object to JSON-safe Python."""
    if _depth > 20:
        return None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except (TypeError, ValueError):
        with open(path, "w") as f:
            json.dump(_to_json_safe(obj), f)


# ──────────────────────────────────────────────────────────
# STEP 1 — AMFI + Yahoo mapping
# ──────────────────────────────────────────────────────────
AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


def fetch_amfi_data():
    print("📥 Fetching AMFI data …")
    text = requests.get(AMFI_URL, timeout=30).text
    lines = [l for l in text.splitlines() if ";" in l]
    amfi = pd.read_csv(StringIO("\n".join(lines)), sep=";", engine="python")
    amfi.columns = amfi.columns.str.strip().str.replace("\ufeff", "")
    amfi = amfi.dropna(subset=["Scheme Code"])
    amfi["Scheme Code"] = amfi["Scheme Code"].astype(int)
    amfi["ISIN"] = amfi["ISIN Div Payout/ ISIN Growth"].fillna(amfi["ISIN Div Reinvestment"])
    amfi = amfi.dropna(subset=["ISIN"])
    print(f"   {len(amfi)} schemes loaded")
    return amfi


def isin_to_yahoo(isin):
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        r = requests.get(url, params={"q": isin, "quotesCount": 5, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        for q in r.json().get("quotes", []):
            if q.get("quoteType") in ("MUTUALFUND", "ETF", "EQUITY"):
                return {"yahoo_symbol": q["symbol"],
                        "yahoo_quote_type": q.get("quoteType"),
                        "yahoo_exchange": q.get("exchange")}
    except Exception:
        pass
    return None


def map_isins(amfi):
    print("🔍 Mapping ISINs to Yahoo Finance …")
    results = []
    for i, isin in enumerate(amfi["ISIN"].unique()):
        m = isin_to_yahoo(isin)
        if m:
            m["ISIN"] = isin
            results.append(m)
        time.sleep(0.3)
        if i % 200 == 0:
            print(f"   {i} ISINs processed …")
    df = pd.DataFrame(results)
    print(f"   {len(df)} Yahoo symbols found")
    return amfi.merge(df, on="ISIN", how="left")


# ──────────────────────────────────────────────────────────
# STEP 2 — Performance / NAV
# ──────────────────────────────────────────────────────────
def get_fy(d):
    return d.year if d.month >= 4 else d.year - 1


def download_performance(funds_df, start_date, batch_size, sleep_range):
    print(f"\n📊 Downloading NAV/dividend data …")
    funds = funds_df[["yahoo_symbol", "Scheme Name"]].drop_duplicates()
    syms = funds["yahoo_symbol"].dropna().tolist()
    name_map = dict(zip(funds["yahoo_symbol"], funds["Scheme Name"]))
    end_date = datetime.now().strftime("%Y-%m-%d")
    all_rows = []

    for i, start in enumerate(range(0, len(syms), batch_size), 1):
        batch = syms[start:start + batch_size]
        print(f"   Batch {i} ({len(batch)} symbols)")
        try:
            raw = yf.download(batch, start=start_date, end=end_date,
                              group_by="ticker", actions=True,
                              auto_adjust=False, threads=False, progress=False)
            for sym in batch:
                if sym not in raw:
                    continue
                df = raw[sym].reset_index()
                df["yahoo_symbol"] = sym
                df["fund_name"] = name_map.get(sym)
                df["FY"] = df["Date"].apply(get_fy)
                all_rows.append(df[["Date", "Close", "Dividends", "yahoo_symbol", "fund_name", "FY"]])
        except Exception as e:
            print(f"   Batch {i} error: {e}")
            time.sleep(60)

        if i < -(-len(syms) // batch_size):
            time.sleep(random.uniform(*sleep_range))

    data = pd.concat(all_rows, ignore_index=True)
    data.rename(columns={"Close": "NAV", "Dividends": "Dividend"}, inplace=True)
    data["Dividend"] = data["Dividend"].fillna(0)
    print(f"   {len(data)} rows downloaded")
    return data


def calc_fy_metrics(data):
    nav_fy = (data.sort_values("Date")
              .groupby(["yahoo_symbol", "fund_name", "FY"])
              .agg(April_NAV=("NAV", "first"), March_NAV=("NAV", "last"))
              .reset_index())
    div_fy = data.groupby(["yahoo_symbol", "fund_name", "FY"])["Dividend"].sum().reset_index()
    fy = nav_fy.merge(div_fy, on=["yahoo_symbol", "fund_name", "FY"], how="left")
    fy["Avg_NAV"] = (fy["April_NAV"] + fy["March_NAV"]) / 2
    fy["IDCW_Yield_pct"] = (fy["Dividend"] / fy["Avg_NAV"] * 100).round(2)
    fy["FY"] = fy["FY"].astype(str) + "-" + (fy["FY"] + 1).astype(str)
    return fy


# ──────────────────────────────────────────────────────────
# STEP 3 — Scheme attributes
# ──────────────────────────────────────────────────────────
def _clean(x):
    x = str(x).lower()
    x = re.sub(r"mutual fund|asset management company", "", x)
    return re.sub(r"[^a-z0-9 ]", "", x).strip()


def extract_attributes(funds_df, amc_path):
    print("🏷️  Extracting scheme attributes …")
    try:
        amc_list = pd.read_csv(amc_path)
    except FileNotFoundError:
        amc_list = pd.DataFrame({"AMC": [
            "Aditya Birla Sun Life Mutual Fund", "HDFC Mutual Fund",
            "ICICI Prudential Mutual Fund", "SBI Mutual Fund",
            "Axis Mutual Fund", "Kotak Mahindra Mutual Fund",
            "UTI Mutual Fund", "DSP Mutual Fund",
            "Nippon India Mutual Fund", "Franklin Templeton Mutual Fund",
        ]})

    amc_list["clean"] = amc_list["AMC"].apply(_clean)
    funds_df["clean"] = funds_df["Scheme Name"].apply(_clean)
    keys = sorted(set(amc_list["clean"].tolist() +
                      amc_list["clean"].str.split().str[0].dropna().tolist()),
                  key=len, reverse=True)
    pat = re.compile(r"^(" + "|".join(map(re.escape, keys)) + r")\b")
    mapping = {row["clean"]: row["AMC"] for _, row in amc_list.iterrows()}
    mapping.update({row["clean"].split()[0]: row["AMC"] for _, row in amc_list.iterrows()})

    funds_df["AMC"] = funds_df["clean"].apply(
        lambda s: mapping.get((m := pat.search(s)) and m.group(1), "Unknown")
    )
    funds_df["Plan"] = "Unknown"
    funds_df.loc[funds_df["clean"].str.contains(r"\bdirect\b", na=False), "Plan"] = "Direct"
    funds_df.loc[funds_df["clean"].str.contains(r"\bregular\b", na=False), "Plan"] = "Regular"
    funds_df["Option"] = "Unknown"
    funds_df.loc[funds_df["clean"].str.contains(r"\bgrowth\b", na=False), "Option"] = "Growth"
    funds_df.loc[funds_df["clean"].str.contains(
        r"\b(?:idcw|dividend|div\b|div\.|payout|reinvestment)\b", na=False), "Option"] = "IDCW"
    return funds_df[["yahoo_symbol", "Scheme Name", "AMC", "Plan", "Option"]]


# ──────────────────────────────────────────────────────────
# STEP 4 — Holdings fetch + cache
# ──────────────────────────────────────────────────────────
def _safe_get(fd, attr):
    try:
        v = getattr(fd, attr)
        if callable(v) and not hasattr(v, "__self__"):
            return None
        return _to_json_safe(v)
    except Exception as e:
        return {"_error": str(e)}


def fetch_one(symbol):
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


def fetch_all_holdings(symbols, cache_path, force=False, sleep_sec=1.0):
    print(f"\n📂 Fetching holdings for {len(symbols)} symbols …")
    cache = {}
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"   {len(cache)} cached entries loaded")
        except Exception:
            cache = {}

    today = datetime.now().strftime("%Y-%m-%d")
    to_fetch = [s for s in symbols
                if s not in cache or cache[s].get("fetched_at") != today or force]
    print(f"   {len(to_fetch)} symbols to fetch")

    for i, sym in enumerate(to_fetch, 1):
        cache[sym] = fetch_one(sym)
        time.sleep(sleep_sec)
        if i % 50 == 0:
            safe_json_dump(cache, cache_path)
            print(f"   {i}/{len(to_fetch)} done, cache saved")

    safe_json_dump(cache, cache_path)
    print(f"   Holdings cache complete ({len(cache)} entries)")
    return cache


def snapshot_month(cache):
    """Store this month's top_holdings in snapshots[YYYY-MM]."""
    month = datetime.now().strftime("%Y-%m")
    for data in cache.values():
        snaps = data.setdefault("snapshots", {})
        if data.get("top_holdings"):
            snaps[month] = data["top_holdings"]
        while len(snaps) > 13:
            del snaps[sorted(snaps)[0]]
    return cache


# ──────────────────────────────────────────────────────────
# STEP 5 — Flatten to CSVs
# ──────────────────────────────────────────────────────────
def build_top_holdings(cache, name_map):
    rows = []
    for sym, d in cache.items():
        th = d.get("top_holdings")
        if not th or not isinstance(th, list):
            continue
        for h in th:
            rows.append({
                "run_date": datetime.now().strftime("%Y-%m-%d"),
                "fund_symbol": sym,
                "fund_name": name_map.get(sym, sym),
                "holding_symbol": h.get("symbol", h.get("index", "")),
                "holding_name": h.get("holdingName", ""),
                "weight_pct": round((h.get("holdingPercent") or 0) * 100, 4),
            })
    return pd.DataFrame(rows)


def build_sector(cache, name_map):
    rows = []
    for sym, d in cache.items():
        sw = d.get("sector_weightings")
        if not sw or not isinstance(sw, list):
            continue
        for entry in sw:
            for k, v in entry.items():
                if k != "index":
                    rows.append({
                        "run_date": datetime.now().strftime("%Y-%m-%d"),
                        "fund_symbol": sym,
                        "fund_name": name_map.get(sym, sym),
                        "sector": k,
                        "weight_pct": round((v or 0) * 100, 4),
                    })
    return pd.DataFrame(rows)


def build_asset_classes(cache, name_map):
    rows = []
    for sym, d in cache.items():
        ac = d.get("asset_classes")
        if not ac or not isinstance(ac, dict):
            continue
        row = {"run_date": datetime.now().strftime("%Y-%m-%d"),
               "fund_symbol": sym, "fund_name": name_map.get(sym, sym)}
        for k, v in ac.items():
            row[k] = v.get("value", v) if isinstance(v, dict) else v
        rows.append(row)
    return pd.DataFrame(rows)


def build_mom_changes(cache, name_map):
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
            chg = cw - pw
            if pw == 0 and cw == 0:
                continue
            status = ("NEW" if pw == 0 and cw > 0 else
                      "EXITED" if pw > 0 and cw == 0 else "CHANGED")
            rows.append({
                "run_date": datetime.now().strftime("%Y-%m-%d"),
                "fund_symbol": sym, "fund_name": name_map.get(sym, sym),
                "holding_symbol": stock,
                "prev_month": pm, "curr_month": cm,
                "prev_weight_pct": round(pw * 100, 4),
                "curr_weight_pct": round(cw * 100, 4),
                "change_pct": round(chg * 100, 4),
                "status": status,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_conviction(mom_df):
    """Aggregate MoM changes to find high-conviction multi-fund buys."""
    if mom_df.empty:
        return pd.DataFrame()
    buying = mom_df[mom_df["change_pct"] > 0].copy()
    if buying.empty:
        return pd.DataFrame()
    agg = (buying.groupby("holding_symbol")
           .agg(
               fund_count=("fund_symbol", "nunique"),
               total_weight_change=("change_pct", "sum"),
               avg_weight_change=("change_pct", "mean"),
               new_entries=("status", lambda x: (x == "NEW").sum()),
               funds_buying=("fund_name", lambda x: ", ".join(sorted(x.unique())[:5])),
           )
           .reset_index())
    agg["conviction_score"] = (
        agg["fund_count"] * 3 +
        agg["total_weight_change"] * 2 +
        agg["new_entries"] * 5
    ).round(2)
    agg["run_date"] = datetime.now().strftime("%Y-%m-%d")
    return agg.sort_values("conviction_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# STEP 6 — Append-safe CSV writer (accumulates history)
# ──────────────────────────────────────────────────────────
def append_or_create(df, path):
    """
    Append df to an existing CSV, deduplicating on (run_date + all columns).
    Creates the file if it doesn't exist yet.
    """
    if df is None or df.empty:
        return
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates()
    else:
        combined = df
    combined.to_csv(path, index=False)
    print(f"   💾 {path} ({len(combined)} total rows)")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    run_start = datetime.now()
    print("\n" + "=" * 60)
    print(f"MF WEEKLY PIPELINE  —  {run_start.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Paths
    P = lambda f: os.path.join(args.data_dir, f)

    # Step 1
    amfi = fetch_amfi_data()
    funds = map_isins(amfi)
    funds.to_csv(P("funds_mapped.csv"), index=False)

    # Step 2
    perf = download_performance(funds, args.start_date, args.batch_size,
                                (args.sleep_min, args.sleep_max))
    fy = calc_fy_metrics(perf)
    append_or_create(fy, P("idcw_fy_output.csv"))

    # Step 3
    scheme = extract_attributes(funds, args.amc_file)
    scheme.to_csv(P("final_schemes_master.csv"), index=False)
    name_map = dict(zip(scheme["yahoo_symbol"], scheme["Scheme Name"]))

    # Step 4
    syms = funds["yahoo_symbol"].dropna().unique().tolist()
    cache = fetch_all_holdings(syms, P("holdings_cache.json"), force=args.force_refresh)
    cache = snapshot_month(cache)
    safe_json_dump(cache, P("holdings_cache.json"))

    # Step 5 — flatten + append
    top_df    = build_top_holdings(cache, name_map)
    sec_df    = build_sector(cache, name_map)
    asset_df  = build_asset_classes(cache, name_map)
    mom_df    = build_mom_changes(cache, name_map)
    conv_df   = build_conviction(mom_df)

    append_or_create(top_df,   P("holdings_top.csv"))
    append_or_create(sec_df,   P("holdings_sector.csv"))
    append_or_create(asset_df, P("holdings_asset_classes.csv"))
    append_or_create(mom_df,   P("holdings_mom_changes.csv"))
    append_or_create(conv_df,  P("conviction_picks.csv"))

    # Step 6 — metadata
    meta = {
        "last_run": run_start.isoformat(),
        "duration_seconds": round((datetime.now() - run_start).total_seconds()),
        "funds_mapped": int(funds["yahoo_symbol"].notna().sum()),
        "fy_rows": len(fy),
        "holdings_cached": len(cache),
        "top_holdings_rows": len(top_df),
        "mom_rows": len(mom_df) if not mom_df.empty else 0,
        "conviction_picks": len(conv_df) if not conv_df.empty else 0,
    }
    safe_json_dump(meta, P("run_metadata.json"))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
