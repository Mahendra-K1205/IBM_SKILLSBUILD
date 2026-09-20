"""
Generate static chart PNGs for the project report.
Uses kaleido (Plotly's static export engine) or falls back to PIL placeholder images.
"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

os.makedirs("report_images", exist_ok=True)

# ── colour palette (matches dark app theme on white bg for print)
SAVE_THEME = dict(
    paper_bgcolor="white", plot_bgcolor="#f8fafc",
    font=dict(color="#1e293b", size=13),
    title_font=dict(size=16, color="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b")),
    margin=dict(l=60, r=30, t=60, b=60),
)
GRID = dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", showline=True,
            tickfont=dict(color="#64748b"))

def save(fig, name, h=420):
    fig.update_layout(**SAVE_THEME, height=h)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    path = f"report_images/{name}.png"
    fig.write_image(path, width=1000, height=h, scale=2)
    print(f"  saved {path}")
    return path

# ─────────────────────────────────────────
print("Loading data …")
df = pd.read_csv("global_financial_markets_2000_Now.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["returns"] = df.groupby("symbol")["close"].pct_change()
NAMES = df[["symbol","asset_name"]].drop_duplicates().set_index("symbol")["asset_name"].to_dict()

SYMS   = ["^GSPC","^IXIC","^BSESN","GC=F","BTC-USD"]
COLORS = ["#2563eb","#7c3aed","#059669","#d97706","#dc2626"]
fdf = df[df["year"] >= 2010]
sdf = fdf[fdf["symbol"].isin(SYMS)]
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─── 1  Growth of $1,000
print("1 – Growth of $1,000 …")
fig = go.Figure()
for sym, col in zip(SYMS, COLORS):
    sd = sdf[sdf["symbol"]==sym].sort_values("date")
    if len(sd)<2 or sd["close"].iloc[0]==0: continue
    g = sd["close"]/sd["close"].iloc[0]*1000
    fig.add_trace(go.Scatter(x=sd["date"], y=g,
                             name=f"{NAMES.get(sym,sym)}  (${g.iloc[-1]:,.0f})",
                             mode="lines", line=dict(color=col, width=2.2)))
fig.add_hline(y=1000, line_dash="dash", line_color="#94a3b8",
              annotation_text="Break-even", annotation_position="right")
fig.update_layout(title="Growth of $1,000 Invested (2010 → Present)",
                  yaxis_title="Portfolio Value ($)", xaxis_title="Year")
save(fig, "01_growth_1000", 440)

# ─── 2  Annual returns – S&P 500
print("2 – Annual returns …")
sp = fdf[fdf["symbol"]=="^GSPC"].sort_values("date")
rows = []
for yr, g in sp.groupby("year"):
    g = g.sort_values("date")
    if len(g)<2: continue
    r = (g["close"].iloc[-1]/g["close"].iloc[0]-1)*100
    rows.append({"Year":yr,"Return":round(r,1)})
ydf = pd.DataFrame(rows)
colors = ["#059669" if v>=0 else "#dc2626" for v in ydf["Return"]]
fig = go.Figure(go.Bar(x=ydf["Year"], y=ydf["Return"],
                       marker_color=colors,
                       text=[f"{v:+.1f}%" for v in ydf["Return"]],
                       textposition="outside"))
fig.add_hline(y=0, line_color="#64748b")
fig.update_layout(title="S&P 500 — Annual Return by Year (%)",
                  yaxis_title="Return (%)", xaxis_title="Year", showlegend=False)
save(fig, "02_annual_returns", 400)

# ─── 3  Asset type pie
print("3 – Asset type distribution …")
tc = df.groupby("asset_type")["symbol"].nunique().reset_index()
tc.columns = ["Asset Type","Count"]
fig = px.pie(tc, names="Asset Type", values="Count", hole=0.5,
             color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_traces(textposition="outside", textinfo="label+percent")
fig.update_layout(title="Asset Class Distribution (34 Assets)", showlegend=True)
save(fig, "03_asset_types", 360)

# ─── 4  Market crashes overlay
print("4 – Crash overlay …")
cr = df[df["symbol"]=="^GSPC"].sort_values("date")
fig = go.Figure()
fig.add_trace(go.Scatter(x=cr["date"], y=cr["close"],
                         name="S&P 500", line=dict(color="#2563eb",width=1.8)))
CRISES = [
    ("2000-03-01","2002-10-01","#ef4444","Dot-com Crash"),
    ("2007-10-01","2009-03-01","#f59e0b","2008 GFC"),
    ("2020-02-15","2020-04-15","#8b5cf6","COVID-19"),
    ("2022-01-01","2022-10-01","#dc2626","2022 Inflation"),
]
for s,e,col,lbl in CRISES:
    fig.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.15, line_width=0,
                  annotation_text=lbl, annotation_position="top left",
                  annotation_font_color=col, annotation_font_size=11)
fig.update_layout(title="S&P 500 — Price History with Major Crash Periods",
                  yaxis_title="Index Level", xaxis_title="Year")
save(fig, "04_crash_overlay", 400)

# ─── 5  Drawdown chart
print("5 – Drawdown …")
fig = go.Figure()
for sym, col in zip(["^GSPC","^IXIC","GC=F"], ["#2563eb","#7c3aed","#d97706"]):
    sd = sdf[sdf["symbol"]==sym].sort_values("date")
    pk = sd["close"].cummax()
    dd = (sd["close"]-pk)/pk*100
    fig.add_trace(go.Scatter(x=sd["date"], y=dd,
                             name=NAMES.get(sym,sym), mode="lines",
                             line=dict(color=col, width=1.8)))
fig.add_hline(y=0, line_color="#059669", line_dash="dash")
fig.update_layout(title="Drawdown from All-Time High (%)",
                  yaxis_title="% Below Peak", xaxis_title="Year")
save(fig, "05_drawdown", 380)

# ─── 6  Monthly seasonality heatmap
print("6 – Seasonality heatmap …")
sp_all = df[df["symbol"]=="^GSPC"].sort_values("date").copy()
sp_all["returns"] = sp_all["close"].pct_change()
heat = (sp_all.groupby(["year","month"])["returns"].mean()*100).reset_index()
heat["Month"] = heat["month"].map(MONTH_NAMES)
pivot = heat.pivot(index="year", columns="Month", values="returns")
ordered = [MONTH_NAMES[m] for m in range(1,13) if MONTH_NAMES[m] in pivot.columns]
pivot = pivot[ordered]
fig = px.imshow(pivot.round(3), color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0, text_auto=".2f", aspect="auto",
                labels=dict(x="Month", y="Year", color="Avg Daily Ret (%)"))
theme6 = {k:v for k,v in SAVE_THEME.items() if k not in ("paper_bgcolor","font")}
fig.update_layout(title="S&P 500 - Monthly Return Heatmap (Green = good month, Red = bad)",
                  paper_bgcolor="white", height=500, font=dict(color="#1e293b"), **theme6)
fig.update_xaxes(**GRID)
fig.update_yaxes(**GRID)
path = "report_images/06_heatmap.png"
fig.write_image(path, width=1100, height=520, scale=2)
print(f"  saved {path}")

# ─── 7  Volatility by year
print("7 – Annual volatility …")
vol_rows = []
for sym, col in zip(["^GSPC","^IXIC","GC=F"], ["#2563eb","#7c3aed","#d97706"]):
    sd = df[df["symbol"]==sym].sort_values("date").copy()
    sd["returns"] = sd["close"].pct_change()
    yv = (sd.groupby("year")["returns"].std()*np.sqrt(252)*100).reset_index()
    yv.columns=["Year","Vol"]
    yv["Asset"]=NAMES.get(sym,sym)
    vol_rows.append(yv)
vol_df = pd.concat(vol_rows)
fig = px.line(vol_df, x="Year", y="Vol", color="Asset",
              color_discrete_sequence=["#2563eb","#7c3aed","#d97706"],
              markers=True)
fig.update_layout(title="Annual Realised Volatility (%) — Key Assets",
                  yaxis_title="Annualised Volatility (%)", xaxis_title="Year")
save(fig, "07_volatility", 380)

# ─── 8  ML — actual vs predicted  +  forecast
print("8 – ML forecast …")
ml = df[df["symbol"]=="^GSPC"].sort_values("date").copy().reset_index(drop=True)
ml["returns"] = ml["close"].pct_change()
ml["ma10"]  = ml["close"].rolling(10).mean()
ml["ma30"]  = ml["close"].rolling(30).mean()
ml["ma90"]  = ml["close"].rolling(90).mean()
ml["vol10"] = ml["returns"].rolling(10).std()
ml["mom10"] = ml["close"].pct_change(10)
ml["mom30"] = ml["close"].pct_change(30)
for lag in [1,2,3,5,10]:
    ml[f"lag{lag}"] = ml["close"].shift(lag)
ml["month"]  = ml["date"].dt.month
ml["target"] = ml["close"].shift(-1)
ml.dropna(inplace=True)
FEATS = ["close","ma10","ma30","ma90","vol10","mom10","mom30",
         "lag1","lag2","lag3","lag5","lag10","month"]
X = ml[FEATS].values; y = ml["target"].values
split = int(len(X)*0.8)
sc = MinMaxScaler()
mdl = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
mdl.fit(sc.fit_transform(X[:split]), y[:split])
preds = mdl.predict(sc.transform(X[split:]))

# actual vs predicted (last 500 days)
test_dates = ml["date"].iloc[split:].values
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates[-500:], y=y[split:][-500:],
                         name="Actual", line=dict(color="#2563eb",width=2)))
fig.add_trace(go.Scatter(x=test_dates[-500:], y=preds[-500:],
                         name="Predicted", line=dict(color="#f59e0b",width=2,dash="dash")))
fig.update_layout(title="S&P 500 — Actual vs ML Predicted Price (Test Set, last 500 days)",
                  yaxis_title="Index Level", xaxis_title="Date")
save(fig, "08_actual_vs_pred", 400)

# rolling 90-day forecast
print("8b – Forecast …")
ml_days = 90
fut_series = ml["close"].tolist(); fut_rets = ml["returns"].tolist()
fut_prices = []; cur = ml["close"].iloc[-1]
future_dates = pd.date_range(ml["date"].iloc[-1]+pd.Timedelta(days=1), periods=ml_days, freq="B")
for fd in future_dates:
    fs = pd.Series(fut_series); fr = pd.Series(fut_rets)
    row = {"close":cur,"ma10":fs.tail(10).mean(),"ma30":fs.tail(30).mean(),
           "ma90":fs.tail(90).mean(),"vol10":fr.tail(10).std(),
           "mom10":cur/(fs.tail(10).iloc[0]+1e-9)-1,"mom30":cur/(fs.tail(30).iloc[0]+1e-9)-1,
           "lag1":fs.iloc[-1],"lag2":fs.iloc[-2],"lag3":fs.iloc[-3],
           "lag5":fs.iloc[-5],"lag10":fs.iloc[-10],"month":fd.month}
    nxt = mdl.predict(sc.transform(np.array([[row[f] for f in FEATS]])))[0]
    fut_prices.append(nxt); fut_series.append(nxt)
    fut_rets.append((nxt/cur-1) if cur else 0); cur = nxt

from sklearn.metrics import mean_squared_error
rmse = float(np.sqrt(mean_squared_error(y[split:], preds)))
upper = [p+1.96*rmse for p in fut_prices]
lower = [max(0,p-1.96*rmse) for p in fut_prices]
hist_show = ml.tail(180)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_show["date"], y=hist_show["close"],
                         name="Historical", line=dict(color="#2563eb",width=2)))
fig.add_trace(go.Scatter(
    x=list(future_dates)+list(future_dates[::-1]),
    y=upper+lower[::-1],
    fill="toself", fillcolor="rgba(5,150,105,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Band"))
fig.add_trace(go.Scatter(x=future_dates, y=fut_prices,
                         name="90-day Forecast",
                         line=dict(color="#059669",width=2.5,dash="dash")))
chg = (fut_prices[-1]/ml["close"].iloc[-1]-1)*100
fig.update_layout(title=f"S&P 500 — 90-Day ML Forecast  ({chg:+.1f}% expected change)",
                  yaxis_title="Index Level", xaxis_title="Date")
save(fig, "09_forecast", 420)

# ─── 9  Feature importances
print("9 – Feature importances …")
fi = pd.Series(mdl.feature_importances_, index=FEATS).sort_values()
friendly = {"close":"Current price","ma10":"10-day avg","ma30":"30-day avg",
            "ma90":"90-day avg","vol10":"Recent volatility",
            "mom10":"10-day momentum","mom30":"30-day momentum",
            "lag1":"Yesterday price","lag2":"2 days ago","lag3":"3 days ago",
            "lag5":"5 days ago","lag10":"10 days ago","month":"Month of year"}
fi.index = [friendly.get(i,i) for i in fi.index]
fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                       marker_color="#2563eb",
                       text=[f"{v*100:.1f}%" for v in fi.values],
                       textposition="outside"))
fig.update_layout(title="Feature Importance — What the ML Model Relies On",
                  xaxis_title="Importance Score", showlegend=False)
save(fig, "10_feature_importance", 380)

# ─── 10  Data quality bar
print("10 – Data quality …")
raw_cols = ["date","open","high","low","close","volume","symbol","asset_name","asset_type","region"]
miss_counts = [df[c].isnull().sum() for c in raw_cols]
fig = go.Figure(go.Bar(x=raw_cols, y=miss_counts,
                       marker_color=["#dc2626" if v>0 else "#059669" for v in miss_counts],
                       text=[str(v) for v in miss_counts], textposition="outside"))
fig.update_layout(title="Missing Values per Column (0 = perfectly clean)",
                  yaxis_title="Missing Count", xaxis_title="Column", showlegend=False)
save(fig, "11_data_quality", 360)

print("\nAll screenshots saved to report_images/")
