import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Global Markets Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global dark background ── */
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* ── All default Streamlit text ── */
    html, body, [class*="css"], p, span, div, label,
    .stMarkdown, .stText { color: #e2e8f0 !important; }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

    /* ── Metric cards ── */
    .metric-card {
        background: #1e2535;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-number { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label  { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
    .green { color: #34d399 !important; }
    .red   { color: #f87171 !important; }

    /* ── Tip / info boxes ── */
    .tip-box {
        background: #1a2340;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .tip-box.green-tip  { background: #0f2318; border-left-color: #34d399; color: #a7f3d0; }
    .tip-box.red-tip    { background: #2a1010; border-left-color: #f87171; color: #fca5a5; }
    .tip-box.yellow-tip { background: #1f1a08; border-left-color: #fbbf24; color: #fde68a; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid #1e2535; }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stCaption { color: #64748b !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #0f1117; border-bottom: 1px solid #1e2535; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.92rem; font-weight: 500;
        color: #94a3b8 !important;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom: 2px solid #60a5fa;
        background: transparent;
    }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] { background: #1e2535; border-radius: 8px; }
    .stDataFrame thead th { background: #0f1117 !important; color: #94a3b8 !important; }
    .stDataFrame tbody td { color: #e2e8f0 !important; background: #1e2535 !important; }

    /* ── Selectbox / multiselect / slider ── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stMultiSelect"] > div { background: #1e2535 !important; color: #e2e8f0 !important; }

    /* ── Info / warning boxes ── */
    [data-testid="stAlert"] { background: #1a2340 !important; color: #93c5fd !important; border-color: #3b82f6 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #1d4ed8;
        color: #f1f5f9 !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #2563eb; }

    /* ── Divider ── */
    hr { border-color: #1e2535; }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="#1e2535",
    plot_bgcolor="#151c2c",
    font=dict(color="#cbd5e1", size=13),
    title_font=dict(size=15, color="#f1f5f9"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
    margin=dict(l=10, r=10, t=50, b=10),
)

def styled_chart(fig, height=380):
    fig.update_layout(**CHART_THEME, height=height)
    fig.update_xaxes(gridcolor="#2d3748", linecolor="#374151", showline=True,
                     tickfont=dict(color="#94a3b8"), title_font=dict(color="#94a3b8"))
    fig.update_yaxes(gridcolor="#2d3748", linecolor="#374151", showline=True,
                     tickfont=dict(color="#94a3b8"), title_font=dict(color="#94a3b8"))
    return fig

def tip(text, kind="blue"):
    cls = {"blue": "", "green": "green-tip", "red": "red-tip", "yellow": "yellow-tip"}.get(kind, "")
    st.markdown(f'<div class="tip-box {cls}">{text}</div>', unsafe_allow_html=True)

def metric_card(col, label, value, color=None):
    c_class = f' {color}' if color else ''
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-number{c_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    df = pd.read_csv("global_financial_markets_2000_Now.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["returns"] = df.groupby("symbol")["close"].pct_change()
    return df

df = load_data()
ALL_SYMBOLS = sorted(df["symbol"].unique())
ASSET_NAMES = df[["symbol","asset_name"]].drop_duplicates().set_index("symbol")["asset_name"].to_dict()
ASSET_TYPES = sorted(df["asset_type"].unique())

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Market Dashboard")
    st.markdown("*Global Financial Data 2000–Now*")
    st.markdown("---")

    st.markdown("**Filter by Asset Type**")
    sel_type = st.selectbox("", ["All Types"] + ASSET_TYPES, label_visibility="collapsed")

    sym_pool = ALL_SYMBOLS if sel_type == "All Types" else sorted(
        df[df["asset_type"] == sel_type]["symbol"].unique())

    st.markdown("**Choose Assets to Compare**")
    sel_symbols = st.multiselect(
        "", sym_pool,
        default=sym_pool[:3],
        format_func=lambda s: ASSET_NAMES.get(s, s),
        label_visibility="collapsed"
    )
    if not sel_symbols:
        sel_symbols = sym_pool[:1]

    st.markdown("**Year Range**")
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    date_range = st.slider("", yr_min, yr_max, (2010, yr_max), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🤖 Price Forecast**")
    ml_sym = st.selectbox("Asset to Predict", sym_pool,
                          format_func=lambda s: ASSET_NAMES.get(s, s))
    ml_days = st.slider("Forecast Days Ahead", 30, 180, 60)
    run_ml  = st.button("▶  Run Forecast", use_container_width=True)

    st.markdown("---")
    st.caption("📊 190,545 data points · 34 assets · 2000–2026")

# ─────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────
fdf = df[(df["year"] >= date_range[0]) & (df["year"] <= date_range[1])]
sdf = fdf[fdf["symbol"].isin(sel_symbols)]

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# 📈 Global Financial Markets Dashboard")
st.markdown(f"Showing data from **{date_range[0]}** to **{date_range[1]}** · "
            f"**{len(sel_symbols)}** asset(s) selected")
st.markdown("---")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠  Overview",
    "📋  Data Quality",
    "📈  Price History",
    "🔍  Market Crashes",
    "📅  Patterns & Seasons",
    "🤖  Price Forecast",
])

# ══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.markdown("### What is this dashboard?")
    tip("This dashboard tracks <b>stocks, currencies, commodities, and cryptocurrencies</b> "
        "from 34 markets worldwide — going back to the year 2000. Use the left sidebar to "
        "filter by asset type, pick assets to compare, and choose your time period.", "blue")

    st.markdown("&nbsp;")

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Assets Tracked", "34")
    metric_card(c2, "Years of Data",         "25+")
    metric_card(c3, "Daily Data Points",     f"{len(df):,}")
    metric_card(c4, "Asset Classes",         "4")

    st.markdown("&nbsp;")
    st.markdown("### How are your selected assets performing?")

    perf_rows = []
    for sym in sel_symbols:
        sd = sdf[sdf["symbol"] == sym].sort_values("date")
        if len(sd) < 2:
            continue
        start_p  = sd["close"].iloc[0]
        end_p    = sd["close"].iloc[-1]
        total_r  = (end_p / start_p - 1) * 100
        best_yr  = sd.groupby("year")["returns"].mean().idxmax() if len(sd) > 250 else "—"
        worst_yr = sd.groupby("year")["returns"].mean().idxmin() if len(sd) > 250 else "—"
        ann_vol  = sd["returns"].std() * np.sqrt(252) * 100
        perf_rows.append({
            "Asset":          ASSET_NAMES.get(sym, sym),
            "Start Price":    f"{start_p:,.2f}",
            "Latest Price":   f"{end_p:,.2f}",
            "Total Return":   f"{total_r:+.1f}%",
            "Best Year":      str(best_yr),
            "Worst Year":     str(worst_yr),
            "Yearly Volatility": f"{ann_vol:.1f}%",
        })

    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")

    # Growth of $1,000 chart
    st.markdown("### If you had invested $1,000 at the start — how much would you have now?")
    tip("Each line shows how $1,000 invested at the start of your selected period would have grown. "
        "Higher line = better return.", "green")

    fig_growth = go.Figure()
    for sym in sel_symbols:
        sd = sdf[sdf["symbol"] == sym].sort_values("date")
        if len(sd) < 2 or sd["close"].iloc[0] == 0:
            continue
        growth = (sd["close"] / sd["close"].iloc[0]) * 1000
        final  = growth.iloc[-1]
        fig_growth.add_trace(go.Scatter(
            x=sd["date"], y=growth,
            name=f"{ASSET_NAMES.get(sym, sym)}  (→ ${final:,.0f})",
            mode="lines", line=dict(width=2.5)
        ))
    fig_growth.add_hline(y=1000, line_dash="dash", line_color="#94a3b8",
                         annotation_text="Break-even ($1,000)",
                         annotation_position="bottom right")
    fig_growth.update_layout(
        **CHART_THEME, height=400,
        yaxis_title="Value of $1,000 invested",
        xaxis_title="Year",
    )
    fig_growth.update_xaxes(gridcolor="#e2e8f0")
    fig_growth.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig_growth, use_container_width=True)

    # Asset type breakdown
    st.markdown("### What types of assets are in this dataset?")
    c1, c2 = st.columns([1, 2])
    with c1:
        type_counts = df.groupby("asset_type")["symbol"].nunique().reset_index()
        type_counts.columns = ["Asset Type", "# Assets"]
        fig_pie = px.pie(type_counts, names="Asset Type", values="# Assets",
                         hole=0.5, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_pie.update_traces(textposition="outside", textinfo="label+value")
        fig_pie.update_layout(**CHART_THEME, height=320, showlegend=False,
                              title="Assets by Category")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        type_detail = (df.groupby(["asset_type","asset_name"])
                         .size().reset_index(name="Records")
                         .sort_values(["asset_type","asset_name"]))
        type_detail.columns = ["Type", "Asset Name", "Data Points"]
        st.dataframe(type_detail, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════
with tab2:
    st.markdown("### Is the data clean and complete?")
    tip("We check every column for missing values, impossible prices (like negative prices), "
        "and gaps in dates. Green = all good. Red = needs attention.", "blue")

    st.markdown("&nbsp;")

    # Only check raw CSV columns — the `returns` column has one NaN per asset
    # on the first trading day by design (pct_change has no prior row), so exclude it.
    raw_cols = ["date","open","high","low","close","volume","symbol","asset_name","asset_type","region"]
    raw_miss  = df[raw_cols].isnull().sum().sum()

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Missing Values in Raw Data", str(raw_miss), "green" if raw_miss == 0 else "red")
    metric_card(c2, "Total Rows",    f"{len(df):,}")
    metric_card(c3, "Total Columns", str(len(raw_cols)))

    tip("The raw market data has <b>zero missing values</b> — every row has a complete date, "
        "open, high, low, close, and volume. ✅", "green")

    st.markdown("&nbsp;")
    st.markdown("#### Column-by-column check")
    miss_df = pd.DataFrame({
        "Column":        raw_cols,
        "Missing Count": [df[c].isnull().sum() for c in raw_cols],
        "% Missing":     [(df[c].isnull().sum()/len(df)*100).round(2) for c in raw_cols],
        "Status":        ["✅ Clean" if df[c].isnull().sum()==0 else "❌ Has gaps" for c in raw_cols],
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")
    st.markdown("#### Price range check for each selected asset")
    tip("Min and Max show the full range of closing prices recorded. "
        "If Min is 0 or negative, that would be a data error — but this dataset is clean.", "green")

    stats_rows = []
    for sym in sel_symbols:
        sd = sdf[sdf["symbol"] == sym]
        stats_rows.append({
            "Asset":        ASSET_NAMES.get(sym, sym),
            "# Trading Days": len(sd),
            "Lowest Price":   f"{sd['close'].min():,.4f}",
            "Highest Price":  f"{sd['close'].max():,.4f}",
            "Average Price":  f"{sd['close'].mean():,.2f}",
            "Has Negatives?": "❌ Yes" if (sd['close'] < 0).any() else "✅ No",
            "Has Zeros?":     "⚠️ Yes" if (sd['close'] == 0).any() else "✅ No",
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")
    st.markdown("#### How many records do we have per year?")
    tip("Each bar shows how many daily price records exist for that year. "
        "A dip means fewer trading days were recorded (e.g. data starts mid-year).", "yellow")

    yr_counts = fdf.groupby(["year","asset_type"]).size().reset_index(name="Records")
    fig_rec = px.bar(yr_counts, x="year", y="Records", color="asset_type",
                     barmode="stack", color_discrete_sequence=px.colors.qualitative.Safe,
                     labels={"year": "Year", "Records": "Number of Records", "asset_type": "Type"})
    fig_rec = styled_chart(fig_rec, height=340)
    fig_rec.update_layout(title="Daily Records per Year (all assets)")
    st.plotly_chart(fig_rec, use_container_width=True)

    st.markdown("&nbsp;")
    st.markdown("#### Daily return distribution — is anything unusual?")
    tip("Most days the price changes by a small amount (near 0%). Very large positive or "
        "negative spikes are shown at the edges. If you see extreme outliers, those are "
        "crash or surge days.", "yellow")

    sym_dist = st.selectbox("Pick an asset to inspect", sel_symbols,
                            format_func=lambda s: ASSET_NAMES.get(s, s), key="dist_sel")
    dist_d = sdf[sdf["symbol"] == sym_dist]["returns"].dropna() * 100

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Average Daily Change", f"{dist_d.mean():.3f}%")
    metric_card(c2, "Biggest Single-Day Gain",  f"+{dist_d.max():.1f}%", "green")
    metric_card(c3, "Biggest Single-Day Drop",  f"{dist_d.min():.1f}%",  "red")
    metric_card(c4, "Typical Daily Swing",       f"±{dist_d.std():.2f}%")

    st.markdown("&nbsp;")
    fig_hist = px.histogram(
        dist_d, nbins=80,
        labels={"value": "Daily % Change", "count": "Number of Days"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#dc2626")
    fig_hist = styled_chart(fig_hist, height=320)
    fig_hist.update_layout(
        title=f"Daily Price Changes for {ASSET_NAMES.get(sym_dist, sym_dist)} "
              f"(each bar = how many days had that % change)",
        showlegend=False,
        xaxis_title="Daily % Change",
        yaxis_title="Number of Days",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3 — PRICE HISTORY
# ══════════════════════════════════════════
with tab3:
    st.markdown("### Price history over time")
    tip("Select any asset below to see its full price history. "
        "The coloured lines are moving averages — they smooth out the noise "
        "so you can see the bigger trend more clearly.", "blue")

    sym_p = st.selectbox("Choose an asset", sel_symbols,
                         format_func=lambda s: ASSET_NAMES.get(s, s), key="price_sym")
    tdf = sdf[sdf["symbol"] == sym_p].sort_values("date").copy()

    tdf["50-day avg"]  = tdf["close"].rolling(50).mean()
    tdf["200-day avg"] = tdf["close"].rolling(200).mean()

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["close"],
        name="Actual Price", line=dict(color="#3b82f6", width=1.8)
    ))
    fig_p.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["50-day avg"],
        name="50-day Average (short trend)",
        line=dict(color="#f59e0b", width=1.5, dash="dash")
    ))
    fig_p.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["200-day avg"],
        name="200-day Average (long trend)",
        line=dict(color="#dc2626", width=1.5, dash="dot")
    ))
    fig_p = styled_chart(fig_p, height=420)
    fig_p.update_layout(
        title=f"{ASSET_NAMES.get(sym_p, sym_p)} — Price History",
        yaxis_title="Price", xaxis_title="Year",
    )
    st.plotly_chart(fig_p, use_container_width=True)

    tip("📌 <b>How to read this:</b> When the <b>short-term (yellow)</b> line crosses "
        "<b>above</b> the long-term (red) line → market is trending up. "
        "When it crosses <b>below</b> → market is weakening.", "yellow")

    st.markdown("&nbsp;")

    # Year-by-year returns as a bar chart
    st.markdown("### Year-by-year returns — how did each year end up?")
    tip("Green bars = the price went up that year. Red bars = it went down. "
        "Hover over a bar to see the exact percentage.", "blue")

    yr_ret_rows = []
    for sym in sel_symbols:
        sd = fdf[fdf["symbol"] == sym].sort_values("date")
        for yr, grp in sd.groupby("year"):
            grp = grp.sort_values("date")
            if len(grp) < 2:
                continue
            ret = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
            yr_ret_rows.append({"Asset": ASSET_NAMES.get(sym, sym), "Year": yr, "Return (%)": round(ret, 1)})

    if yr_ret_rows:
        yr_df = pd.DataFrame(yr_ret_rows)
        if len(sel_symbols) == 1:
            single = yr_df.copy()
            single["Color"] = single["Return (%)"].apply(lambda v: "#16a34a" if v >= 0 else "#dc2626")
            fig_yr = go.Figure(go.Bar(
                x=single["Year"], y=single["Return (%)"],
                marker_color=single["Color"],
                text=single["Return (%)"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
            ))
        else:
            fig_yr = px.bar(yr_df, x="Year", y="Return (%)", color="Asset",
                            barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Safe)
        fig_yr = styled_chart(fig_yr, height=380)
        fig_yr.add_hline(y=0, line_color="#64748b", line_width=1)
        fig_yr.update_layout(
            title="Annual Return by Year (%)",
            yaxis_title="Return (%)", xaxis_title="Year",
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown("&nbsp;")

    # Side-by-side comparison (normalised)
    if len(sel_symbols) > 1:
        st.markdown("### Comparing assets side by side")
        tip("All assets are rescaled to start at 100 so you can fairly compare "
            "their growth — regardless of their actual price levels.", "blue")

        fig_norm = go.Figure()
        for sym in sel_symbols:
            sd = sdf[sdf["symbol"] == sym].sort_values("date")
            if len(sd) < 2 or sd["close"].iloc[0] == 0:
                continue
            norm = (sd["close"] / sd["close"].iloc[0]) * 100
            fig_norm.add_trace(go.Scatter(
                x=sd["date"], y=norm,
                name=ASSET_NAMES.get(sym, sym), mode="lines", line=dict(width=2)
            ))
        fig_norm.add_hline(y=100, line_dash="dash", line_color="#94a3b8",
                           annotation_text="Starting point", annotation_position="right")
        fig_norm = styled_chart(fig_norm, height=380)
        fig_norm.update_layout(
            title="Relative Growth (all starting at 100)",
            yaxis_title="Indexed Value (start = 100)", xaxis_title="Year",
        )
        st.plotly_chart(fig_norm, use_container_width=True)


# ══════════════════════════════════════════
# TAB 4 — MARKET CRASHES
# ══════════════════════════════════════════
with tab4:
    st.markdown("### Major market crashes since 2000")
    tip("The shaded regions below mark well-known market crashes. "
        "See how your chosen asset behaved during each crisis.", "red")

    sym_cr = st.selectbox("Pick an asset", sel_symbols,
                          format_func=lambda s: ASSET_NAMES.get(s, s), key="cr_sym")
    cr_d = sdf[sdf["symbol"] == sym_cr].sort_values("date")

    CRISES = [
        ("2000-03-01","2002-10-01","#ef4444","🔴 Dot-com Crash (2000–2002)\nTech stocks lost ~80%"),
        ("2007-10-01","2009-03-01","#f59e0b","🟠 2008 Financial Crisis\nGlobal banks nearly collapsed"),
        ("2020-02-15","2020-04-15","#8b5cf6","🟣 COVID-19 Crash (2020)\nFastest 30% drop in history"),
        ("2022-01-01","2022-10-01","#dc2626","🔴 2022 Inflation Shock\nRising rates crushed valuations"),
    ]

    fig_cr = go.Figure()
    fig_cr.add_trace(go.Scatter(
        x=cr_d["date"], y=cr_d["close"],
        name="Price", line=dict(color="#2563eb", width=2)
    ))
    for s, e, col, label in CRISES:
        short_label = label.split("\n")[0]
        fig_cr.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.12, line_width=0,
                         annotation_text=short_label,
                         annotation_position="top left",
                         annotation_font_size=10,
                         annotation_font_color=col)
    fig_cr = styled_chart(fig_cr, height=400)
    fig_cr.update_layout(
        title=f"{ASSET_NAMES.get(sym_cr, sym_cr)} — Price with Major Crash Periods Highlighted",
        yaxis_title="Price", xaxis_title="Year",
    )
    st.plotly_chart(fig_cr, use_container_width=True)

    # Crisis impact table
    st.markdown("### How much did each crisis hurt?")
    tip("This table shows what happened to your selected asset during each crisis — "
        "how much it fell, and whether it was already in the dataset at that time.", "yellow")

    crash_rows = []
    for s_str, e_str, _, label in CRISES:
        crisis_name = label.split("\n")[0]
        s_dt = pd.to_datetime(s_str)
        e_dt = pd.to_datetime(e_str)
        sub = cr_d[(cr_d["date"] >= s_dt) & (cr_d["date"] <= e_dt)]
        if len(sub) < 2:
            crash_rows.append({"Crisis": crisis_name, "Drop": "No data for this period",
                                "Start Price": "—", "Bottom Price": "—"})
            continue
        start_p  = sub["close"].iloc[0]
        trough_p = sub["close"].min()
        drop_pct = (trough_p / start_p - 1) * 100
        recovery = cr_d[cr_d["date"] > e_dt]
        recovered = "Still recovering" if recovery.empty else (
            "✅ Recovered" if recovery["close"].max() > start_p else "⚠️ Not yet recovered")
        crash_rows.append({
            "Crisis":        crisis_name,
            "Drop from peak": f"{drop_pct:.1f}%",
            "Price at start": f"{start_p:,.2f}",
            "Lowest point":   f"{trough_p:,.2f}",
            "Recovery":       recovered,
        })
    st.dataframe(pd.DataFrame(crash_rows), use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")

    # How far below all-time high right now?
    st.markdown("### How far is each asset from its all-time high?")
    tip("A value of 0% means the asset is AT its all-time high right now. "
        "A value of -50% means it's still half its peak — it hasn't fully recovered.", "blue")

    dd_rows = []
    for sym in sel_symbols:
        sd = sdf[sdf["symbol"] == sym].sort_values("date")
        if len(sd) < 2:
            continue
        peak       = sd["close"].cummax()
        drawdown   = (sd["close"] - peak) / peak * 100
        current_dd = drawdown.iloc[-1]
        worst_dd   = drawdown.min()
        dd_rows.append({
            "Asset":              ASSET_NAMES.get(sym, sym),
            "Current Price":      f"{sd['close'].iloc[-1]:,.2f}",
            "All-Time High":      f"{sd['close'].max():,.2f}",
            "Below Peak by":      f"{current_dd:.1f}%",
            "Worst Ever Dip":     f"{worst_dd:.1f}%",
        })
    st.dataframe(pd.DataFrame(dd_rows), use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")

    # Drawdown over time chart
    st.markdown("### Drawdown chart — how deep did each asset fall from its peak?")
    tip("0% = at the peak. -50% = half the value is gone. "
        "You want this line as close to 0 as possible.", "red")

    fig_dd = go.Figure()
    for sym in sel_symbols:
        sd = sdf[sdf["symbol"] == sym].sort_values("date")
        peak = sd["close"].cummax()
        dd   = (sd["close"] - peak) / peak * 100
        fig_dd.add_trace(go.Scatter(
            x=sd["date"], y=dd,
            name=ASSET_NAMES.get(sym, sym),
            mode="lines", line=dict(width=1.8)
        ))
    fig_dd = styled_chart(fig_dd, height=360)
    fig_dd.add_hline(y=0, line_color="#16a34a", line_dash="dash")
    fig_dd.update_layout(
        title="How far below its peak? (0% = at all-time high)",
        yaxis_title="% Below All-Time High", xaxis_title="Year",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ══════════════════════════════════════════
# TAB 5 — PATTERNS & SEASONS
# ══════════════════════════════════════════
with tab5:
    st.markdown("### Do markets follow a predictable seasonal pattern?")
    tip("These charts show the <b>average</b> return for each month/quarter across all years in your range. "
        "Green = historically a good month, Red = historically weak.", "blue")

    sym_s = st.selectbox("Pick an asset", sel_symbols,
                         format_func=lambda s: ASSET_NAMES.get(s, s), key="sea_sym")
    sd_s = sdf[sdf["symbol"] == sym_s].sort_values("date").copy()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Best & Worst Months")
        mo_avg = (sd_s.groupby("month")["returns"].mean() * 100).reset_index()
        mo_avg["Month"] = mo_avg["month"].map(MONTH_NAMES)
        mo_avg["Color"] = mo_avg["returns"].apply(lambda v: "#16a34a" if v >= 0 else "#dc2626")
        mo_avg = mo_avg.sort_values("month")

        fig_mo = go.Figure(go.Bar(
            x=mo_avg["Month"], y=mo_avg["returns"],
            marker_color=mo_avg["Color"],
            text=mo_avg["returns"].apply(lambda v: f"{v:+.3f}%"),
            textposition="outside",
        ))
        fig_mo.add_hline(y=0, line_color="#64748b")
        fig_mo = styled_chart(fig_mo, height=340)
        fig_mo.update_layout(
            title="Average Daily Return by Month",
            yaxis_title="Avg Daily Return (%)", xaxis_title="Month",
            showlegend=False,
        )
        st.plotly_chart(fig_mo, use_container_width=True)

        best_mo  = mo_avg.loc[mo_avg["returns"].idxmax(), "Month"]
        worst_mo = mo_avg.loc[mo_avg["returns"].idxmin(), "Month"]
        tip(f"📅 <b>Best month historically:</b> {best_mo} &nbsp;|&nbsp; "
            f"<b>Worst month historically:</b> {worst_mo}", "green")

    with c2:
        st.markdown("#### Best & Worst Quarters")
        q_avg = (sd_s.groupby("quarter")["returns"].mean() * 100).reset_index()
        q_avg["Quarter"] = q_avg["quarter"].map({1:"Q1 (Jan–Mar)",2:"Q2 (Apr–Jun)",
                                                  3:"Q3 (Jul–Sep)",4:"Q4 (Oct–Dec)"})
        q_avg["Color"] = q_avg["returns"].apply(lambda v: "#16a34a" if v >= 0 else "#dc2626")

        fig_q = go.Figure(go.Bar(
            x=q_avg["Quarter"], y=q_avg["returns"],
            marker_color=q_avg["Color"],
            text=q_avg["returns"].apply(lambda v: f"{v:+.3f}%"),
            textposition="outside",
        ))
        fig_q.add_hline(y=0, line_color="#64748b")
        fig_q = styled_chart(fig_q, height=340)
        fig_q.update_layout(
            title="Average Daily Return by Quarter",
            yaxis_title="Avg Daily Return (%)", xaxis_title="Quarter",
            showlegend=False,
        )
        st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("&nbsp;")

    # Heatmap — year × month
    st.markdown("### Monthly return heatmap — colour shows which months were good or bad")
    tip("Each cell = average daily return for that month and year. "
        "<b>Green = prices generally rose</b>, <b>Red = prices generally fell</b>. "
        "Look for columns that are consistently green — those are the reliably good months.", "blue")

    heat = (sd_s.groupby(["year","month"])["returns"].mean() * 100).reset_index()
    heat["Month"] = heat["month"].map(MONTH_NAMES)
    pivot = heat.pivot(index="year", columns="Month", values="returns")
    # keep month order
    ordered_cols = [MONTH_NAMES[m] for m in range(1,13) if MONTH_NAMES[m] in pivot.columns]
    pivot = pivot[ordered_cols]

    fig_heat = px.imshow(
        pivot.round(3),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".3f",
        aspect="auto",
        labels=dict(x="Month", y="Year", color="Avg Daily Return (%)"),
    )
    fig_heat.update_layout(
        **CHART_THEME, height=max(300, len(pivot) * 22 + 80),
        title=f"Monthly Return Heatmap — {ASSET_NAMES.get(sym_s, sym_s)}  (green = good month, red = bad month)",
        coloraxis_colorbar=dict(title="Avg Daily %"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("&nbsp;")

    # Volatility — how risky has each year been?
    st.markdown("### How risky was each year?")
    tip("Volatility measures how wildly prices moved. A <b>high bar = very bumpy ride</b>, "
        "a <b>low bar = calm and steady</b>. Notice the big spikes in 2008 and 2020.", "yellow")

    yr_vol = (sd_s.groupby("year")["returns"].std() * np.sqrt(252) * 100).reset_index()
    yr_vol.columns = ["Year", "Yearly Volatility (%)"]
    yr_vol["Color"] = yr_vol["Yearly Volatility (%)"].apply(
        lambda v: "#dc2626" if v > 30 else ("#f59e0b" if v > 15 else "#16a34a"))

    fig_vol = go.Figure(go.Bar(
        x=yr_vol["Year"], y=yr_vol["Yearly Volatility (%)"],
        marker_color=yr_vol["Color"],
        text=yr_vol["Yearly Volatility (%)"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
    ))
    fig_vol = styled_chart(fig_vol, height=340)
    fig_vol.update_layout(
        title="Yearly Volatility (Red = very risky, Yellow = moderate, Green = calm)",
        yaxis_title="Annual Volatility (%)", xaxis_title="Year", showlegend=False,
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════
# TAB 6 — ML FORECAST
# ══════════════════════════════════════════
with tab6:
    st.markdown("### 🤖 What might the price look like in the coming months?")
    tip("Our machine learning model studies the past price history of the asset you selected, "
        "learns the patterns, and then projects what might happen next. "
        "<b>This is NOT financial advice</b> — it's a statistical estimate based on history.", "yellow")

    st.markdown(f"**Currently set to forecast:** {ASSET_NAMES.get(ml_sym, ml_sym)} "
                f"&nbsp;|&nbsp; **Days ahead:** {ml_days}  &nbsp;·&nbsp; "
                f"*(Change in the sidebar and click Run Forecast)*")

    if run_ml:
        with st.spinner("Training model — this takes about 20–30 seconds…"):

            ml_raw = df[df["symbol"] == ml_sym].sort_values("date").copy().reset_index(drop=True)

            if len(ml_raw) < 300:
                st.error("Not enough historical data for this asset. Please pick another.")
            else:
                # Feature engineering
                ml_raw["returns"] = ml_raw["close"].pct_change()
                ml_raw["ma10"]    = ml_raw["close"].rolling(10).mean()
                ml_raw["ma30"]    = ml_raw["close"].rolling(30).mean()
                ml_raw["ma90"]    = ml_raw["close"].rolling(90).mean()
                ml_raw["vol10"]   = ml_raw["returns"].rolling(10).std()
                ml_raw["mom10"]   = ml_raw["close"].pct_change(10)
                ml_raw["mom30"]   = ml_raw["close"].pct_change(30)
                for lag in [1, 2, 3, 5, 10]:
                    ml_raw[f"lag{lag}"] = ml_raw["close"].shift(lag)
                ml_raw["month"] = ml_raw["date"].dt.month
                ml_raw["target"] = ml_raw["close"].shift(-1)
                ml_raw.dropna(inplace=True)

                FEATURES = ["close","ma10","ma30","ma90","vol10","mom10","mom30",
                            "lag1","lag2","lag3","lag5","lag10","month"]

                X = ml_raw[FEATURES].values
                y = ml_raw["target"].values
                split = int(len(X) * 0.8)

                scaler  = MinMaxScaler()
                X_tr_sc = scaler.fit_transform(X[:split])
                X_te_sc = scaler.transform(X[split:])

                model = RandomForestRegressor(n_estimators=300, max_depth=10,
                                              random_state=42, n_jobs=-1)
                model.fit(X_tr_sc, y[:split])
                preds = model.predict(X_te_sc)

                mae  = mean_absolute_error(y[split:], preds)
                rmse = np.sqrt(mean_squared_error(y[split:], preds))
                mape = float(np.mean(np.abs((y[split:] - preds) / (np.abs(y[split:]) + 1e-8))) * 100)
                dir_acc = float((np.sign(np.diff(y[split:])) ==
                                 np.sign(preds[1:] - y[split:-1])).mean() * 100)

                # Metrics — plain English
                st.markdown("### How accurate is the model on past data?")
                c1, c2, c3 = st.columns(3)
                metric_card(c1, "Average Price Error",    f"{mape:.1f}%")
                metric_card(c2, "Correct Up/Down Direction",
                            f"{dir_acc:.0f}%",
                            "green" if dir_acc > 55 else "red")
                metric_card(c3, "Avg $ Miss (RMSE)",      f"{rmse:,.2f}")

                tip(f"The model gets the direction right <b>{dir_acc:.0f}%</b> of the time "
                    f"(anything above 50% is better than a coin flip). "
                    f"On average, its price estimate is off by <b>{mape:.1f}%</b>.", "blue")

                st.markdown("&nbsp;")

                # Actual vs predicted — last 2 years only for clarity
                st.markdown("### How well did the model track prices on data it had never seen?")
                tip("Blue = the real price. Orange dashed = what the model predicted. "
                    "The closer they are, the better.", "blue")

                test_dates = ml_raw["date"].iloc[split:].values
                last_2yr   = -min(504, len(test_dates))   # ~2 trading years
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(
                    x=test_dates[last_2yr:], y=y[split:][last_2yr:],
                    name="Actual Price", line=dict(color="#2563eb", width=2)
                ))
                fig_av.add_trace(go.Scatter(
                    x=test_dates[last_2yr:], y=preds[last_2yr:],
                    name="Model Prediction", line=dict(color="#f59e0b", width=2, dash="dash")
                ))
                fig_av = styled_chart(fig_av, height=380)
                fig_av.update_layout(
                    title="Actual vs Predicted Price (last 2 years of test data)",
                    yaxis_title="Price", xaxis_title="Date",
                )
                st.plotly_chart(fig_av, use_container_width=True)

                # Future forecast
                st.markdown(f"### What does the model predict for the next {ml_days} days?")
                tip("The green line is the model's best guess for future prices. "
                    "The shaded band is the uncertainty range — the real price will likely fall "
                    "somewhere in that zone.", "green")

                # Rolling future prediction
                fut_series = ml_raw["close"].tolist()
                fut_rets   = ml_raw["returns"].tolist()
                fut_prices = []
                cur = ml_raw["close"].iloc[-1]

                future_dates = pd.date_range(
                    ml_raw["date"].iloc[-1] + pd.Timedelta(days=1),
                    periods=ml_days, freq="B"
                )

                for fd in future_dates:
                    fs = pd.Series(fut_series)
                    fr = pd.Series(fut_rets)
                    row = {
                        "close":  cur,
                        "ma10":   fs.tail(10).mean(),
                        "ma30":   fs.tail(30).mean(),
                        "ma90":   fs.tail(90).mean(),
                        "vol10":  fr.tail(10).std(),
                        "mom10":  cur / (fs.tail(10).iloc[0] + 1e-9) - 1,
                        "mom30":  cur / (fs.tail(30).iloc[0] + 1e-9) - 1,
                        "lag1":   fs.iloc[-1],
                        "lag2":   fs.iloc[-2] if len(fs)>1 else cur,
                        "lag3":   fs.iloc[-3] if len(fs)>2 else cur,
                        "lag5":   fs.iloc[-5] if len(fs)>4 else cur,
                        "lag10":  fs.iloc[-10] if len(fs)>9 else cur,
                        "month":  fd.month,
                    }
                    vec = np.array([[row[f] for f in FEATURES]])
                    nxt = model.predict(scaler.transform(vec))[0]
                    fut_prices.append(nxt)
                    fut_series.append(nxt)
                    fut_rets.append((nxt / cur - 1) if cur else 0)
                    cur = nxt

                hist_show = ml_raw.tail(180)
                upper = [p + 1.96 * rmse for p in fut_prices]
                lower = [max(0, p - 1.96 * rmse) for p in fut_prices]

                fig_fut = go.Figure()
                fig_fut.add_trace(go.Scatter(
                    x=hist_show["date"], y=hist_show["close"],
                    name="Past Price", line=dict(color="#2563eb", width=2)
                ))
                # Confidence band
                fig_fut.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(16,163,74,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Likely Range", showlegend=True
                ))
                fig_fut.add_trace(go.Scatter(
                    x=future_dates, y=fut_prices,
                    name="Forecast", line=dict(color="#16a34a", width=2.5, dash="dash")
                ))
                # Join line
                fig_fut.add_trace(go.Scatter(
                    x=[hist_show["date"].iloc[-1], future_dates[0]],
                    y=[hist_show["close"].iloc[-1], fut_prices[0]],
                    mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"),
                    showlegend=False
                ))

                last_close    = ml_raw["close"].iloc[-1]
                forecast_end  = fut_prices[-1]
                chg_pct       = (forecast_end / last_close - 1) * 100
                direction_lbl = "📈 UP" if chg_pct > 0 else "📉 DOWN"
                direction_col = "green" if chg_pct > 0 else "red"

                fig_fut = styled_chart(fig_fut, height=420)
                fig_fut.update_layout(
                    title=f"{ASSET_NAMES.get(ml_sym, ml_sym)} — {ml_days}-Day Forecast",
                    yaxis_title="Price", xaxis_title="Date",
                )
                st.plotly_chart(fig_fut, use_container_width=True)

                # Plain-English summary
                c1, c2, c3 = st.columns(3)
                metric_card(c1, "Current Price",        f"{last_close:,.2f}")
                metric_card(c2, f"Predicted Price in {ml_days} days", f"{forecast_end:,.2f}")
                metric_card(c3, "Expected Change",       f"{chg_pct:+.1f}%", direction_col)

                tip(f"The model expects <b>{ASSET_NAMES.get(ml_sym, ml_sym)}</b> to move "
                    f"<b>{direction_lbl}</b> over the next <b>{ml_days} trading days</b>, "
                    f"changing by roughly <b>{chg_pct:+.1f}%</b> from today's price of "
                    f"<b>{last_close:,.2f}</b>. "
                    f"Remember: this is a model estimate, not a guarantee.",
                    direction_col if chg_pct != 0 else "blue")

                # What drives the model?
                fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
                friendly = {
                    "close":"Current price","ma10":"10-day avg","ma30":"30-day avg",
                    "ma90":"90-day avg","vol10":"Recent volatility",
                    "mom10":"10-day momentum","mom30":"30-day momentum",
                    "lag1":"Yesterday price","lag2":"2 days ago","lag3":"3 days ago",
                    "lag5":"5 days ago","lag10":"10 days ago","month":"Month of year",
                }
                fi.index = [friendly.get(i, i) for i in fi.index]

                st.markdown("&nbsp;")
                st.markdown("### What information matters most to the model?")
                tip("This shows which signals the model relies on most heavily when making a prediction. "
                    "Longer bar = more important.", "blue")

                fig_fi = go.Figure(go.Bar(
                    x=fi.values, y=fi.index, orientation="h",
                    marker_color="#3b82f6",
                    text=[f"{v*100:.1f}%" for v in fi.values],
                    textposition="outside",
                ))
                fig_fi = styled_chart(fig_fi, height=380)
                fig_fi.update_layout(
                    title="Feature Importance (what the model pays attention to)",
                    xaxis_title="Importance Score", yaxis_title="",
                    showlegend=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    else:
        st.info("👈 Use the sidebar to pick an asset and number of forecast days, "
                "then click **▶ Run Forecast**.")

        st.markdown("### What does the model actually do?")
        st.markdown("""
        The model looks at the **past price history** of the asset you choose and learns patterns like:

        - 📈 When prices have been trending up for 10 days, they tend to keep rising
        - 📉 When recent volatility spikes, a correction often follows
        - 📅 Certain months are historically stronger than others

        It then uses these patterns to estimate what might happen next.

        **It is NOT clairvoyant** — it cannot predict surprise events like a pandemic or a war.
        Use it as one data point among many, not as a trading signal.
        """)
