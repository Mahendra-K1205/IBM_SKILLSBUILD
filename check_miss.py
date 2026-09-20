import pandas as pd, numpy as np
df = pd.read_csv("global_financial_markets_2000_Now.csv", parse_dates=["date"])
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["returns"] = df.groupby("symbol")["close"].pct_change()

miss = df.isnull().sum()
print("Missing per column:")
print(miss[miss > 0])
print("Total missing:", miss.sum())
print("Columns:", list(df.columns))
