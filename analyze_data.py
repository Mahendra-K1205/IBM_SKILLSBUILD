import csv
from collections import defaultdict

with open('global_financial_markets_2000_Now.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

missing = defaultdict(int)
total = len(rows)
for row in rows:
    for k, v in row.items():
        if v == '' or v is None:
            missing[k] += 1

print('Total rows:', total)
print('Missing values per column:', dict(missing))

asset_names = defaultdict(set)
for row in rows:
    asset_names[row['asset_type']].add(f"{row['symbol']} ({row['asset_name']})")

for t, names in asset_names.items():
    print(f'\n{t}: {sorted(names)}')
