"""
Quick smoke-test of every chart's data pipeline.
Runs the same logic used in app.py but outside Streamlit.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("global_financial_markets_2000_Now.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["returns"] = df.groupby("symbol")["close"].pct_change()

sel_symbols = ["^GSPC", "^IXIC", "GC=F"]
date_range  = (2010, 2024)
fdf = df[(df["year"] >= date_range[0]) & (df["year"] <= date_range[1])]
sdf = fdf[fdf["symbol"].isin(sel_symbols)]
ASSET_NAMES = df[["symbol","asset_name"]].drop_duplicates().set_index("symbol")["asset_name"].to_dict()
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── TAB 1: Overview
print("TAB1: Overview...")
for sym in sel_symbols:
    sd = sdf[sdf["symbol"] == sym].sort_values("date")
    assert len(sd) > 0, f"no data for {sym}"
    growth = (sd["close"] / sd["close"].iloc[0]) * 1000
    assert not growth.isnull().all()
type_counts = df.groupby("asset_type")["symbol"].nunique().reset_index()
assert len(type_counts) == 4
print("  OK")

# ── TAB 2: Data Quality
print("TAB2: Data Quality...")
raw_cols = ["date","open","high","low","close","volume","symbol","asset_name","asset_type","region"]
raw_miss  = df[raw_cols].isnull().sum().sum()
assert raw_miss == 0, f"Unexpected missing in raw cols: {raw_miss}"
dist_d = sdf[sdf["symbol"] == "^GSPC"]["returns"].dropna() * 100
assert len(dist_d) > 100
assert dist_d.max() > 0
assert dist_d.min() < 0
yr_counts = fdf.groupby(["year","asset_type"]).size().reset_index(name="Records")
assert len(yr_counts) > 0
print("  OK")

# ── TAB 3: Price History
print("TAB3: Price History...")
tdf = sdf[sdf["symbol"] == "^GSPC"].sort_values("date").copy()
tdf["50-day avg"]  = tdf["close"].rolling(50).mean()
tdf["200-day avg"] = tdf["close"].rolling(200).mean()
assert tdf["50-day avg"].dropna().shape[0] > 0

yr_ret_rows = []
for sym in sel_symbols:
    sd = fdf[fdf["symbol"] == sym].sort_values("date")
    for yr, grp in sd.groupby("year"):
        grp = grp.sort_values("date")
        if len(grp) < 2:
            continue
        ret = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
        yr_ret_rows.append({"Asset": sym, "Year": yr, "Return (%)": round(ret,1)})
assert len(yr_ret_rows) > 0
print("  OK")

# ── TAB 4: Market Crashes
print("TAB4: Market Crashes...")
cr_d = sdf[sdf["symbol"] == "^GSPC"].sort_values("date")
CRISES = [
    ("2000-03-01","2002-10-01","#ef4444","Dot-com"),
    ("2007-10-01","2009-03-01","#f59e0b","GFC"),
    ("2020-02-15","2020-04-15","#8b5cf6","COVID"),
    ("2022-01-01","2022-10-01","#dc2626","Inflation"),
]
for s_str, e_str, _, label in CRISES:
    s_dt = pd.to_datetime(s_str)
    e_dt = pd.to_datetime(e_str)
    sub  = cr_d[(cr_d["date"] >= s_dt) & (cr_d["date"] <= e_dt)]
    # OK to be empty (year range may not cover it)

for sym in sel_symbols:
    sd = sdf[sdf["symbol"] == sym].sort_values("date")
    peak = sd["close"].cummax()
    dd   = (sd["close"] - peak) / peak * 100
    assert dd.max() <= 0.01
print("  OK")

# ── TAB 5: Patterns & Seasons
print("TAB5: Seasons...")
sym_s = "^GSPC"
sd_s = sdf[sdf["symbol"] == sym_s].sort_values("date").copy()
mo_avg = (sd_s.groupby("month")["returns"].mean() * 100).reset_index()
assert len(mo_avg) > 0
q_avg = (sd_s.groupby("quarter")["returns"].mean() * 100).reset_index()
assert len(q_avg) == 4
heat = (sd_s.groupby(["year","month"])["returns"].mean() * 100).reset_index()
heat["Month"] = heat["month"].map(MONTH_NAMES)
pivot = heat.pivot(index="year", columns="Month", values="returns")
ordered = [MONTH_NAMES[m] for m in range(1,13) if MONTH_NAMES[m] in pivot.columns]
pivot = pivot[ordered]
assert pivot.shape[0] > 0
yr_vol = (sd_s.groupby("year")["returns"].std() * np.sqrt(252) * 100).reset_index()
assert yr_vol.shape[0] > 0
print("  OK")

# ── TAB 6: ML Forecast
print("TAB6: ML Forecast (small run)...")
ml_sym = "^GSPC"
ml_raw = df[df["symbol"] == ml_sym].sort_values("date").copy().reset_index(drop=True)
ml_raw["returns"] = ml_raw["close"].pct_change()
ml_raw["ma10"]    = ml_raw["close"].rolling(10).mean()
ml_raw["ma30"]    = ml_raw["close"].rolling(30).mean()
ml_raw["ma90"]    = ml_raw["close"].rolling(90).mean()
ml_raw["vol10"]   = ml_raw["returns"].rolling(10).std()
ml_raw["mom10"]   = ml_raw["close"].pct_change(10)
ml_raw["mom30"]   = ml_raw["close"].pct_change(30)
for lag in [1,2,3,5,10]:
    ml_raw[f"lag{lag}"] = ml_raw["close"].shift(lag)
ml_raw["month"]  = ml_raw["date"].dt.month
ml_raw["target"] = ml_raw["close"].shift(-1)
ml_raw.dropna(inplace=True)

FEATURES = ["close","ma10","ma30","ma90","vol10","mom10","mom30",
            "lag1","lag2","lag3","lag5","lag10","month"]
X = ml_raw[FEATURES].values
y = ml_raw["target"].values
split = int(len(X)*0.8)
scaler = MinMaxScaler()
model  = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
model.fit(scaler.fit_transform(X[:split]), y[:split])
preds = model.predict(scaler.transform(X[split:]))
mae  = mean_absolute_error(y[split:], preds)
rmse = np.sqrt(mean_squared_error(y[split:], preds))
mape = float(np.mean(np.abs((y[split:] - preds)/(np.abs(y[split:]) + 1e-8)))*100)
dir_acc = float((np.sign(np.diff(y[split:])) == np.sign(preds[1:] - y[split:-1])).mean()*100)
assert 0 < mape < 100, f"MAPE out of range: {mape}"
assert 0 < dir_acc < 100, f"Dir acc out of range: {dir_acc}"
print(f"  MAE={mae:.1f}, MAPE={mape:.2f}%, DirAcc={dir_acc:.1f}%  OK")

# Rolling forecast
ml_days = 30
last_close = ml_raw["close"].iloc[-1]
last_date  = ml_raw["date"].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=ml_days, freq="B")
fut_series = ml_raw["close"].tolist()
fut_rets   = ml_raw["returns"].tolist()
fut_prices = []
cur = last_close
for fd in future_dates:
    fs  = pd.Series(fut_series)
    fr  = pd.Series(fut_rets)
    row = {
        "close": cur, "ma10": fs.tail(10).mean(), "ma30": fs.tail(30).mean(),
        "ma90": fs.tail(90).mean(), "vol10": fr.tail(10).std(),
        "mom10": cur/(fs.tail(10).iloc[0]+1e-9)-1,
        "mom30": cur/(fs.tail(30).iloc[0]+1e-9)-1,
        "lag1": fs.iloc[-1],
        "lag2": fs.iloc[-2] if len(fs)>1 else cur,
        "lag3": fs.iloc[-3] if len(fs)>2 else cur,
        "lag5": fs.iloc[-5] if len(fs)>4 else cur,
        "lag10":fs.iloc[-10] if len(fs)>9 else cur,
        "month": fd.month,
    }
    vec = np.array([[row[f] for f in FEATURES]])
    nxt = model.predict(scaler.transform(vec))[0]
    fut_prices.append(nxt)
    fut_series.append(nxt)
    fut_rets.append((nxt/cur - 1) if cur else 0)
    cur = nxt
assert len(fut_prices) == ml_days
chg = (fut_prices[-1]/last_close - 1)*100
print(f"  30-day forecast: {last_close:.1f} to {fut_prices[-1]:.1f} ({chg:+.1f}%)  OK")

print("\n✅ All pipeline tests passed.")
