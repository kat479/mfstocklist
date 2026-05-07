"""
dashboard.py  —  MF Intelligence Dashboard
============================================
Reads pre-computed CSVs from the data/ folder (written by scripts/pipeline.py).
Launch with:  streamlit run dashboard.py

Tabs:
  1. 📡 Live Signals     — latest conviction buys + MoM movers
  2. 📈 IDCW Rankings    — fund yield league table with sparklines
  3. 🏗️ Holdings Map     — treemap + sunburst of current holdings
  4. 📅 MoM Flow         — stock-level buy/sell flow over time
  5. 🔬 Stock X-Ray      — single-stock ownership time series
  6. 🏦 Fund DNA         — single-fund holdings, sector, asset mix
  7. 📊 Sector Pulse     — sector rotation heatmap across funds
  8. 🕰️ History          — every weekly run in one timeline
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
DATA_DIR = "data"
st.set_page_config(
    page_title="MF Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# THEME  —  dark Bloomberg terminal
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#080d18;color:#c5cfe0;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
[data-testid="stSidebar"]{background:#0b1121;border-right:1px solid #182033;}
[data-testid="stSidebar"] *{color:#7a8faa!important;}
[data-testid="stSidebar"] label{font-size:10px!important;text-transform:uppercase;letter-spacing:.08em;color:#3d5a80!important;}
[data-testid="stMetric"]{background:#0c1628;border:1px solid #182033;border-radius:8px;padding:1rem 1.25rem;}
[data-testid="stMetricLabel"]{color:#3d5a80!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.06em;}
[data-testid="stMetricValue"]{color:#e2ecff!important;font-family:'IBM Plex Mono',monospace!important;font-size:1.5rem!important;}
[data-testid="stMetricDelta"]{font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;}
[data-testid="stDataFrame"]{border:1px solid #182033;border-radius:6px;}
.stTabs [data-baseweb="tab-list"]{background:#0b1121;border-bottom:1px solid #182033;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;color:#3d5a80;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;padding:.55rem .9rem;}
.stTabs [aria-selected="true"]{border-bottom:2px solid #2563eb!important;color:#93c5fd!important;background:transparent!important;}
h1{color:#e2ecff!important;font-weight:600!important;font-size:1.3rem!important;}
h2{color:#93c5fd!important;font-weight:500!important;font-size:.9rem!important;text-transform:uppercase;letter-spacing:.06em;}
hr{border-color:#182033!important;}
.chip-green{background:#052e16;color:#4ade80;border:1px solid #166534;border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;white-space:nowrap;}
.chip-red  {background:#450a0a;color:#f87171;border:1px solid #7f1d1d;border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;white-space:nowrap;}
.chip-blue {background:#0c1a40;color:#93c5fd;border:1px solid #1e3a8a;border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;white-space:nowrap;}
.chip-amber{background:#1c0f00;color:#fbbf24;border:1px solid #78350f;border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;white-space:nowrap;}
.hdr{display:flex;align-items:center;gap:10px;border-bottom:1px solid #182033;padding-bottom:.6rem;margin-bottom:1rem;}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 7px #22c55e;display:inline-block;}
.ts{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#1e3a5f;margin-left:auto;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ──────────────────────────────────────────────────────────
COLORS = ["#3b82f6","#22c55e","#f59e0b","#ec4899","#8b5cf6",
          "#06b6d4","#f97316","#a3e635","#14b8a6","#e11d48"]

def dark(fig, h=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d18",
        font=dict(color="#7a8faa", family="Inter", size=11),
        title_font=dict(color="#e2ecff", size=13),
        margin=dict(l=40, r=20, t=40, b=40),
        colorway=COLORS,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7a8faa"),
        **({"height": h} if h else {}),
    )
    fig.update_xaxes(gridcolor="#182033", linecolor="#182033", tickcolor="#263859")
    fig.update_yaxes(gridcolor="#182033", linecolor="#182033", tickcolor="#263859")
    return fig


# ──────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load(fname, **kw):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


@st.cache_data(ttl=600, show_spinner=False)
def load_meta():
    path = os.path.join(DATA_DIR, "run_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def p(fname): return os.path.join(DATA_DIR, fname)
def exists(fname): return os.path.exists(p(fname))

top_df   = load("holdings_top.csv")
mom_df   = load("holdings_mom_changes.csv")
conv_df  = load("conviction_picks.csv")
fy_df    = load("idcw_fy_output.csv")
sec_df   = load("holdings_sector.csv")
asset_df = load("holdings_asset_classes.csv")
scheme_df= load("final_schemes_master.csv")
meta     = load_meta()

# Normalise date columns
for df in [top_df, mom_df, conv_df, fy_df, sec_df, asset_df]:
    if not df.empty and "run_date" in df.columns:
        df["run_date"] = pd.to_datetime(df["run_date"])

# ── Header ────────────────────────────────────────────────
last_run = meta.get("last_run", "–")
if last_run != "–":
    try:
        last_run = datetime.fromisoformat(last_run).strftime("%d %b %Y %H:%M")
    except Exception:
        pass

data_ok = not top_df.empty
status_dot = "🟢" if data_ok else "🔴"
st.markdown(f"""
<div class="hdr">
  <span class="dot"></span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#e2ecff;font-weight:600;letter-spacing:-.01em;">
    MF INTELLIGENCE
  </span>
  <span style="font-size:10px;color:#2563eb;text-transform:uppercase;letter-spacing:.08em;">
    weekly data · {status_dot} last run {last_run}
  </span>
  <span class="ts">{datetime.now().strftime('%d %b %Y  %H:%M')}</span>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.warning(
        "No data found in `data/` folder.  \n"
        "Run `python scripts/pipeline.py --data-dir data` first, "
        "or trigger the GitHub Actions workflow.",
        icon="⚠️"
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️  Filters")
    st.divider()

    all_dates = sorted(top_df["run_date"].dt.strftime("%Y-%m-%d").unique()) if not top_df.empty else []
    sel_date  = st.selectbox("Week (run date)", ["Latest"] + list(reversed(all_dates)))
    run_date_filter = (top_df["run_date"].max() if sel_date == "Latest"
                       else pd.to_datetime(sel_date))

    if not scheme_df.empty and "AMC" in scheme_df.columns:
        all_amcs = sorted(scheme_df["AMC"].dropna().unique())
        sel_amcs = st.multiselect("AMC", all_amcs, default=all_amcs)
    else:
        sel_amcs = []

    if not top_df.empty and "fund_name" in top_df.columns:
        all_funds = sorted(top_df["fund_name"].dropna().unique())
        sel_funds = st.multiselect("Fund (leave blank = all)", all_funds)
    else:
        sel_funds = []

    min_fund_count = st.slider("Min funds holding (for stock views)", 1, 20, 2)

    st.divider()
    st.caption(f"Data rows loaded:  \n"
               f"Top holdings: {len(top_df):,}  \n"
               f"MoM changes: {len(mom_df):,}  \n"
               f"Conviction: {len(conv_df):,}")


def latest_top(df):
    if df.empty:
        return df
    return df[df["run_date"] == df["run_date"].max()]


def apply_fund_filter(df, col="fund_name"):
    if sel_funds and col in df.columns:
        return df[df[col].isin(sel_funds)]
    return df


# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "📡 Live Signals",
    "📈 IDCW Rankings",
    "🏗️ Holdings Map",
    "📅 MoM Flow",
    "🔬 Stock X-Ray",
    "🏦 Fund DNA",
    "📊 Sector Pulse",
    "🕰️ History",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📡 This week's signals")

    # ── KPIs ──
    k = st.columns(6)
    latest_mom = mom_df[mom_df["run_date"] == mom_df["run_date"].max()] if not mom_df.empty else pd.DataFrame()
    latest_conv = conv_df[conv_df["run_date"] == conv_df["run_date"].max()] if not conv_df.empty else pd.DataFrame()

    k[0].metric("Funds tracked",      meta.get("holdings_cached", "–"))
    k[1].metric("Holdings rows",      f"{meta.get('top_holdings_rows',0):,}")
    k[2].metric("MoM changes",        f"{meta.get('mom_rows',0):,}")
    k[3].metric("Conviction picks",   f"{meta.get('conviction_picks',0):,}")
    k[4].metric("New entries (week)", int((latest_mom["status"] == "NEW").sum()) if not latest_mom.empty else 0)
    k[5].metric("Full exits (week)",  int((latest_mom["status"] == "EXITED").sum()) if not latest_mom.empty else 0)

    st.divider()
    col_conv, col_exit = st.columns(2)

    with col_conv:
        st.markdown("### 💎 High Conviction Buys")
        if latest_conv.empty:
            st.info("No conviction data yet — need ≥2 weekly runs.")
        else:
            top15 = latest_conv.head(15).copy()
            fig = go.Figure(go.Bar(
                x=top15["conviction_score"],
                y=top15["holding_symbol"],
                orientation="h",
                marker=dict(
                    color=top15["conviction_score"],
                    colorscale=[[0,"#0c1a40"],[0.5,"#1d4ed8"],[1,"#60a5fa"]],
                ),
                text=top15["conviction_score"].round(1).astype(str),
                textfont=dict(color="#93c5fd", size=10),
                textposition="outside",
                customdata=top15[["fund_count","total_weight_change","new_entries"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Score: %{x:.1f}<br>"
                    "Funds buying: %{customdata[0]}<br>"
                    "Total Δweight: %{customdata[1]:.3f}%<br>"
                    "New entries: %{customdata[2]}<extra></extra>"
                ),
            ))
            fig.update_layout(
                title="Conviction = funds×3 + Δweight×2 + new_entries×5",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Score",
            )
            st.plotly_chart(dark(fig, 420), use_container_width=True)

    with col_exit:
        st.markdown("### 🚪 Fresh Exits & Big Trims")
        if latest_mom.empty:
            st.info("No MoM data.")
        else:
            exits = latest_mom[latest_mom["status"].isin(["EXITED","CHANGED"])].copy()
            exits = exits[exits["change_pct"] < 0].groupby("holding_symbol").agg(
                fund_count=("fund_symbol","nunique"),
                total_change=("change_pct","sum"),
            ).reset_index().sort_values("total_change").head(15)

            fig2 = go.Figure(go.Bar(
                x=exits["total_change"],
                y=exits["holding_symbol"],
                orientation="h",
                marker_color="#ef4444",
                text=exits["total_change"].round(3).astype(str) + "%",
                textfont=dict(color="#fca5a5", size=10),
                textposition="outside",
            ))
            fig2.update_layout(
                title="Stocks with biggest aggregate selling",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Total Δweight % (all funds)",
            )
            st.plotly_chart(dark(fig2, 420), use_container_width=True)

    # ── New entries this week ──
    st.divider()
    st.markdown("### 🆕 Brand-new Positions (first time held)")
    if not latest_mom.empty:
        new_e = latest_mom[latest_mom["status"] == "NEW"].sort_values("curr_weight_pct", ascending=False)
        if new_e.empty:
            st.caption("No new entries this week.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(new_e.head(12).iterrows()):
                with cols[i % 4]:
                    st.markdown(
                        f"**{row.get('holding_symbol','?')}**  \n"
                        f"<span class='chip-green'>+{row['curr_weight_pct']:.3f}%</span>  \n"
                        f"<span style='font-size:10px;color:#3d5a80'>{row['fund_name'][:30]}</span>",
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════════════════
# TAB 2 — IDCW RANKINGS
# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📈 IDCW Fund Rankings")

    if fy_df.empty:
        st.info("No FY data yet.")
    else:
        idcw = fy_df.copy()
        if not scheme_df.empty:
            idcw = idcw.merge(scheme_df[["yahoo_symbol","AMC","Plan","Option"]],
                              on="yahoo_symbol", how="left")
            idcw = idcw[idcw["Option"] == "IDCW"]

        if idcw.empty:
            st.warning("No IDCW funds found after filtering.")
        else:
            # Latest-FY per fund
            latest_fy = idcw.groupby("yahoo_symbol")["FY"].max().reset_index(name="LatestFY")
            latest = idcw.merge(latest_fy, left_on=["yahoo_symbol","FY"],
                                right_on=["yahoo_symbol","LatestFY"], how="inner")

            overall = (idcw.groupby(["fund_name","yahoo_symbol","AMC","Plan"])
                       .agg(Avg_Yield=("IDCW_Yield_pct","mean"),
                            Years=("FY","nunique"),
                            Total_Div=("Dividend","sum"))
                       .reset_index())

            merged = latest.merge(overall, on=["fund_name","yahoo_symbol","AMC","Plan"], how="left")
            merged["Score"] = (0.6 * merged["IDCW_Yield_pct"] + 0.4 * merged["Avg_Yield"]).round(2)
            merged = merged.sort_values("Score", ascending=False).reset_index(drop=True)

            # Sparklines
            spark = (idcw.sort_values("FY")
                     .groupby("fund_name")["Dividend"]
                     .apply(lambda x: x.fillna(0).tolist())
                     .reset_index(name="Trend"))
            merged = merged.merge(spark, on="fund_name", how="left")

            f1, f2, f3 = st.columns(3)
            min_yrs  = f1.slider("Min years history", 1, 7, 3)
            amc_filt = f2.multiselect("AMC", sorted(merged["AMC"].dropna().unique()))
            plan_filt= f3.multiselect("Plan", ["Direct","Regular"], default=["Direct"])

            view = merged[merged["Years"] >= min_yrs].copy()
            if amc_filt:  view = view[view["AMC"].isin(amc_filt)]
            if plan_filt: view = view[view["Plan"].isin(plan_filt)]

            st.dataframe(
                view[["fund_name","AMC","Plan","LatestFY","IDCW_Yield_pct",
                       "Avg_Yield","Years","Score","Trend"]].head(50),
                hide_index=True, use_container_width=True,
                column_config={
                    "IDCW_Yield_pct": st.column_config.NumberColumn("Latest Yield %", format="%.2f"),
                    "Avg_Yield":      st.column_config.NumberColumn("Avg Yield %",    format="%.2f"),
                    "Score":          st.column_config.ProgressColumn("Score", min_value=0,
                                                                       max_value=float(view["Score"].max() or 1),
                                                                       format="%.2f"),
                    "Trend":          st.column_config.LineChartColumn("Div trend", width="medium"),
                }
            )

            # Scatter: yield vs consistency
            fig3 = px.scatter(
                view, x="Years", y="IDCW_Yield_pct", size="Total_Div",
                color="AMC", hover_name="fund_name",
                title="Yield vs Consistency (bubble = total dividends paid)",
                labels={"Years":"Years of IDCW history","IDCW_Yield_pct":"Latest IDCW Yield %"},
            )
            st.plotly_chart(dark(fig3, 400), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — HOLDINGS MAP
# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🏗️ Holdings Map (latest week)")

    lt = latest_top(top_df)
    lt = apply_fund_filter(lt)

    if lt.empty:
        st.info("No holdings data.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            # Treemap: fund → holding
            fig_tree = px.treemap(
                lt, path=["fund_name","holding_name"],
                values="weight_pct",
                color="weight_pct",
                color_continuous_scale=[[0,"#0c1a40"],[0.5,"#1d4ed8"],[1,"#60a5fa"]],
                title="Holdings treemap — area ∝ weight",
            )
            fig_tree.update_traces(textfont_size=10, textinfo="label+percent entry")
            st.plotly_chart(dark(fig_tree, 500), use_container_width=True)

        with c2:
            # Top-20 cross-fund stocks by total weight
            top20 = (lt.groupby("holding_symbol")
                     .agg(total_weight=("weight_pct","sum"),
                          fund_count=("fund_name","nunique"))
                     .reset_index()
                     .sort_values("total_weight", ascending=False)
                     .head(20))
            fig_bar = go.Figure(go.Bar(
                x=top20["total_weight"], y=top20["holding_symbol"],
                orientation="h",
                marker=dict(color=top20["fund_count"],
                            colorscale=[[0,"#0c1a40"],[1,"#3b82f6"]],
                            colorbar=dict(title="# Funds")),
                text=top20["fund_count"].astype(str) + " funds",
                textfont=dict(size=10), textposition="auto",
            ))
            fig_bar.update_layout(title="Top 20 stocks by total weight (all funds)",
                                   yaxis=dict(autorange="reversed"))
            st.plotly_chart(dark(fig_bar, 500), use_container_width=True)

        # Sunburst
        fig_sun = px.sunburst(
            lt.head(300), path=["fund_name","holding_name"],
            values="weight_pct", color="fund_name",
            title="Sunburst — fund → holding weight",
        )
        fig_sun.update_traces(textfont_size=9)
        st.plotly_chart(dark(fig_sun, 550), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — MOM FLOW
# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 📅 Month-on-Month Flow")

    if mom_df.empty:
        st.info("No MoM data yet — need ≥2 weekly runs.")
    else:
        latest_mom2 = apply_fund_filter(
            mom_df[mom_df["run_date"] == mom_df["run_date"].max()]
        )

        # ── Waterfall of top movers ──
        movers = (latest_mom2.groupby("holding_symbol")["change_pct"]
                  .sum().reset_index()
                  .sort_values("change_pct"))
        top_n = pd.concat([movers.head(15), movers.tail(15)]).drop_duplicates()

        fig_wf = go.Figure(go.Bar(
            x=top_n["holding_symbol"], y=top_n["change_pct"],
            marker_color=["#22c55e" if v > 0 else "#ef4444" for v in top_n["change_pct"]],
            text=top_n["change_pct"].apply(lambda v: f"{v:+.3f}%"),
            textfont=dict(size=9), textposition="outside",
        ))
        fig_wf.update_layout(title="Biggest aggregate weight changes (all funds, latest week)",
                              xaxis_tickangle=-45, yaxis_title="Δ weight %")
        st.plotly_chart(dark(fig_wf, 380), use_container_width=True)

        # ── Flow over time (all weeks) ──
        st.divider()
        st.markdown("### Flow accumulation over all weekly runs")

        weekly_flow = (mom_df.groupby(["run_date","status"])
                       .size().reset_index(name="count"))
        fig_flow = px.bar(weekly_flow, x="run_date", y="count", color="status",
                          barmode="group",
                          color_discrete_map={"NEW":"#22c55e","CHANGED":"#3b82f6","EXITED":"#ef4444"},
                          title="New / Changed / Exited positions per week")
        fig_flow.update_layout(xaxis_title="Run date", yaxis_title="# changes")
        st.plotly_chart(dark(fig_flow, 340), use_container_width=True)

        # ── Detail table ──
        st.divider()
        st.markdown("### Detail table")
        status_sel = st.multiselect("Status filter",
                                    ["NEW","CHANGED","EXITED"],
                                    default=["NEW","EXITED"])
        detail = latest_mom2[latest_mom2["status"].isin(status_sel)]
        st.dataframe(
            detail[["fund_name","holding_symbol","prev_month","curr_month",
                    "prev_weight_pct","curr_weight_pct","change_pct","status"]]
            .sort_values("change_pct"),
            hide_index=True, use_container_width=True,
            column_config={
                "change_pct": st.column_config.NumberColumn("Δ Weight %", format="%+.4f"),
            }
        )
        st.download_button("📥 Download MoM CSV",
                           detail.to_csv(index=False), "mom_changes.csv", "text/csv")


# ══════════════════════════════════════════════════════════
# TAB 5 — STOCK X-RAY
# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔬 Stock X-Ray — ownership through time")

    if top_df.empty:
        st.info("No holdings data.")
    else:
        all_stocks = sorted(top_df["holding_symbol"].dropna().unique())
        sel_stock  = st.selectbox("Select stock", all_stocks)

        stock_ts = (top_df[top_df["holding_symbol"] == sel_stock]
                    .groupby("run_date")
                    .agg(fund_count=("fund_name","nunique"),
                         total_weight=("weight_pct","sum"),
                         avg_weight=("weight_pct","mean"))
                    .reset_index())

        if stock_ts.empty:
            st.warning("No data for this stock.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            latest_row = stock_ts.iloc[-1]
            prev_row   = stock_ts.iloc[-2] if len(stock_ts) > 1 else latest_row
            m1.metric("Funds holding (latest)",    int(latest_row["fund_count"]),
                      delta=int(latest_row["fund_count"] - prev_row["fund_count"]))
            m2.metric("Total weight % (latest)",   f"{latest_row['total_weight']:.2f}%",
                      delta=f"{latest_row['total_weight']-prev_row['total_weight']:+.2f}%")
            m3.metric("Avg weight per fund",        f"{latest_row['avg_weight']:.2f}%")
            m4.metric("Weeks tracked",              len(stock_ts))

            # Dual-axis: fund count + total weight
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(go.Scatter(
                x=stock_ts["run_date"], y=stock_ts["fund_count"],
                name="# Funds", line=dict(color="#3b82f6", width=2.5),
                mode="lines+markers", marker=dict(size=7)
            ), secondary_y=False)
            fig_dual.add_trace(go.Bar(
                x=stock_ts["run_date"], y=stock_ts["total_weight"],
                name="Total weight %", marker_color="rgba(34,197,94,0.25)",
                marker_line_color="#22c55e", marker_line_width=1,
            ), secondary_y=True)
            fig_dual.update_layout(title=f"{sel_stock} — ownership timeline",
                                   legend=dict(bgcolor="rgba(0,0,0,0)"))
            fig_dual.update_yaxes(title_text="# Funds", secondary_y=False,
                                  gridcolor="#182033", linecolor="#182033")
            fig_dual.update_yaxes(title_text="Total weight %", secondary_y=True,
                                  gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
            dark(fig_dual, 380)
            st.plotly_chart(fig_dual, use_container_width=True)

            # Fund-level pivot over time
            pivot = (top_df[top_df["holding_symbol"] == sel_stock]
                     .pivot_table(index="fund_name", columns="run_date",
                                  values="weight_pct", aggfunc="sum")
                     .fillna(0).round(4))
            st.markdown("### Per-fund weight history")
            st.dataframe(pivot, use_container_width=True)

            # MoM changes for this stock
            if not mom_df.empty:
                stock_mom = mom_df[mom_df["holding_symbol"] == sel_stock].copy()
                if not stock_mom.empty:
                    st.markdown("### MoM changes for this stock")
                    st.dataframe(
                        stock_mom[["run_date","fund_name","prev_month","curr_month",
                                   "prev_weight_pct","curr_weight_pct","change_pct","status"]],
                        hide_index=True, use_container_width=True
                    )


# ══════════════════════════════════════════════════════════
# TAB 6 — FUND DNA
# ══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🏦 Fund DNA")

    if top_df.empty:
        st.info("No data.")
    else:
        all_fund_names = sorted(top_df["fund_name"].dropna().unique())
        sel_fund = st.selectbox("Select fund", all_fund_names)

        fund_latest = latest_top(top_df)
        fund_latest = fund_latest[fund_latest["fund_name"] == sel_fund]

        if fund_latest.empty:
            st.warning("No latest holdings for this fund.")
        else:
            fa, fb, fc = st.columns(3)
            fa.metric("Holdings count", len(fund_latest))
            fb.metric("Top holding",    fund_latest.sort_values("weight_pct", ascending=False)
                      .iloc[0]["holding_symbol"])
            fc.metric("Top weight",
                      f"{fund_latest['weight_pct'].max():.2f}%")

            c1, c2 = st.columns([1.2, 1])
            with c1:
                fig_pie = px.pie(
                    fund_latest.sort_values("weight_pct", ascending=False).head(20),
                    names="holding_name", values="weight_pct", hole=0.4,
                    title=f"Top 20 holdings — {sel_fund}",
                )
                fig_pie.update_traces(textfont_size=10)
                st.plotly_chart(dark(fig_pie, 400), use_container_width=True)

            with c2:
                # Weight bar
                top15f = fund_latest.sort_values("weight_pct", ascending=False).head(15)
                fig_fb = go.Figure(go.Bar(
                    x=top15f["weight_pct"], y=top15f["holding_symbol"],
                    orientation="h", marker_color="#3b82f6",
                    text=top15f["weight_pct"].apply(lambda v: f"{v:.2f}%"),
                    textfont=dict(size=10), textposition="outside",
                ))
                fig_fb.update_layout(title="Top 15 by weight",
                                     yaxis=dict(autorange="reversed"))
                st.plotly_chart(dark(fig_fb, 400), use_container_width=True)

            # Sector if available
            if not sec_df.empty and "fund_name" in sec_df.columns:
                fund_sec = latest_top(sec_df)
                fund_sec = fund_sec[fund_sec["fund_name"] == sel_fund] if "fund_name" in fund_sec.columns else pd.DataFrame()
                if not fund_sec.empty:
                    st.markdown("### Sector breakdown")
                    fig_sec = px.bar(
                        fund_sec.sort_values("weight_pct", ascending=False),
                        x="sector", y="weight_pct", color="sector",
                        title=f"Sector weights — {sel_fund}",
                    )
                    st.plotly_chart(dark(fig_sec, 320), use_container_width=True)

            # Asset classes
            if not asset_df.empty and "fund_name" in asset_df.columns:
                fund_asset = asset_df[asset_df["fund_name"] == sel_fund]
                if not fund_asset.empty:
                    num_cols = [c for c in fund_asset.columns
                                if c not in ("fund_symbol","fund_name","fetched_at","run_date")
                                and pd.api.types.is_numeric_dtype(fund_asset[c])]
                    if num_cols:
                        row = fund_asset.iloc[-1]
                        vals = [float(row.get(c, 0) or 0) for c in num_cols]
                        fig_ac = px.pie(names=num_cols, values=vals, hole=0.5,
                                        title="Asset class allocation")
                        st.plotly_chart(dark(fig_ac, 300), use_container_width=True)

            # Weight over time (history)
            fund_ts = (top_df[top_df["fund_name"] == sel_fund]
                       .groupby(["run_date","holding_symbol"])["weight_pct"]
                       .sum().reset_index())
            if len(fund_ts["run_date"].unique()) > 1:
                st.markdown("### Holdings weight over time (top 10 by latest weight)")
                top10_syms = (fund_latest.sort_values("weight_pct", ascending=False)
                              .head(10)["holding_symbol"].tolist())
                fund_ts10 = fund_ts[fund_ts["holding_symbol"].isin(top10_syms)]
                fig_ot = px.line(fund_ts10, x="run_date", y="weight_pct",
                                 color="holding_symbol", markers=True,
                                 title="Weight % over weekly runs")
                st.plotly_chart(dark(fig_ot, 380), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 7 — SECTOR PULSE
# ══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## 📊 Sector Pulse")

    if sec_df.empty:
        st.info("No sector data yet.")
    else:
        latest_sec = latest_top(sec_df)
        latest_sec = apply_fund_filter(latest_sec)

        if latest_sec.empty:
            st.warning("No sector data for current filters.")
        else:
            # Heatmap: fund × sector
            pivot_sec = latest_sec.pivot_table(
                index="fund_name", columns="sector",
                values="weight_pct", aggfunc="mean"
            ).fillna(0)

            fig_heat = go.Figure(go.Heatmap(
                z=pivot_sec.values,
                x=pivot_sec.columns.tolist(),
                y=pivot_sec.index.tolist(),
                colorscale=[[0,"#080d18"],[0.3,"#1e3a8a"],[0.7,"#2563eb"],[1,"#93c5fd"]],
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
            ))
            fig_heat.update_layout(
                title="Sector allocation heatmap — all funds (latest week)",
                xaxis_tickangle=-40,
                height=max(350, len(pivot_sec) * 22 + 80),
            )
            st.plotly_chart(dark(fig_heat), use_container_width=True)

            # Sector aggregation
            sec_agg = (latest_sec.groupby("sector")
                       .agg(avg_weight=("weight_pct","mean"),
                            total_weight=("weight_pct","sum"),
                            fund_count=("fund_name","nunique"))
                       .reset_index().sort_values("total_weight", ascending=False))

            c1, c2 = st.columns(2)
            with c1:
                fig_sec_bar = px.bar(
                    sec_agg, x="sector", y="avg_weight",
                    color="fund_count",
                    color_continuous_scale=[[0,"#0c1a40"],[1,"#3b82f6"]],
                    title="Avg sector weight across all funds",
                    labels={"avg_weight":"Avg weight %","fund_count":"# Funds"},
                )
                fig_sec_bar.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(dark(fig_sec_bar, 360), use_container_width=True)

            with c2:
                fig_sec_pie = px.pie(
                    sec_agg, names="sector", values="total_weight",
                    title="Sector share of total holdings weight",
                    hole=0.4,
                )
                st.plotly_chart(dark(fig_sec_pie, 360), use_container_width=True)

            # Sector rotation over time
            if len(sec_df["run_date"].unique()) > 1:
                st.divider()
                st.markdown("### Sector rotation over weekly runs")
                sec_time = (sec_df.groupby(["run_date","sector"])
                            ["weight_pct"].mean().reset_index())
                fig_rot = px.line(sec_time, x="run_date", y="weight_pct",
                                  color="sector", markers=True,
                                  title="Avg sector weight % per weekly run")
                st.plotly_chart(dark(fig_rot, 400), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 8 — HISTORY
# ══════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## 🕰️ Pipeline Run History")

    # ── Run stats table ──
    if not top_df.empty:
        run_stats = (top_df.groupby("run_date")
                     .agg(funds=("fund_name","nunique"),
                          holdings=("holding_symbol","count"),
                          avg_weight=("weight_pct","mean"))
                     .reset_index().sort_values("run_date", ascending=False))
        st.dataframe(run_stats, hide_index=True, use_container_width=True,
                     column_config={
                         "run_date": "Run date",
                         "funds": "Funds tracked",
                         "holdings": "Holdings rows",
                         "avg_weight": st.column_config.NumberColumn("Avg weight %", format="%.3f"),
                     })

        # Holdings count over time
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=run_stats["run_date"], y=run_stats["holdings"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="#3b82f6", width=2),
            mode="lines+markers", name="Holdings rows",
        ))
        fig_hist.update_layout(title="Holdings rows per weekly run",
                               xaxis_title="Run date", yaxis_title="Rows")
        st.plotly_chart(dark(fig_hist, 320), use_container_width=True)

    # ── Conviction picks history ──
    if not conv_df.empty:
        st.divider()
        st.markdown("### Conviction picks across all weeks")
        pivot_conv = conv_df.pivot_table(
            index="holding_symbol", columns="run_date",
            values="conviction_score", aggfunc="mean"
        ).fillna(0).round(1)
        # Keep top 30 by latest score
        if not pivot_conv.empty:
            latest_col = sorted(pivot_conv.columns)[-1]
            top30 = pivot_conv.nlargest(30, latest_col)
            fig_conv_heat = go.Figure(go.Heatmap(
                z=top30.values,
                x=[str(c)[:10] for c in top30.columns],
                y=top30.index.tolist(),
                colorscale=[[0,"#080d18"],[0.4,"#1e3a8a"],[1,"#60a5fa"]],
                hovertemplate="<b>%{y}</b><br>Week: %{x}<br>Score: %{z:.1f}<extra></extra>",
            ))
            fig_conv_heat.update_layout(
                title="Conviction score heatmap — top 30 stocks over time",
                height=max(400, len(top30) * 18 + 80),
            )
            st.plotly_chart(dark(fig_conv_heat), use_container_width=True)

    # ── Download all data ──
    st.divider()
    st.markdown("### 📥 Download data files")
    dl_cols = st.columns(5)
    for i, (label, df) in enumerate([
        ("Holdings", top_df), ("MoM Changes", mom_df),
        ("Conviction", conv_df), ("FY Metrics", fy_df), ("Sectors", sec_df),
    ]):
        with dl_cols[i]:
            st.download_button(
                f"⬇ {label}",
                df.to_csv(index=False),
                f"{label.lower().replace(' ','_')}.csv",
                "text/csv",
                use_container_width=True,
            )

# ── Footer ────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:9px;color:#182033;"
    f"font-family:IBM Plex Mono,monospace;'>"
    f"MF INTELLIGENCE · data/ · last pipeline run: {last_run}"
    f"</div>",
    unsafe_allow_html=True
)
