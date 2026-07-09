import os
import re
import pandas as pd
from TROPoe_outputs import lookup

# Hardcode your folder path
folder = lookup.data_location + "radiosonde_processed_csv_data_TEAMx/"
assert os.path.isdir(folder), f"Data folder not found: {folder}"

records = []
for fname in os.listdir(folder):
    m = re.search(r"(\d{10})", fname)
    if m:
        dt = pd.to_datetime(m.group(1), format="%Y%m%d%H")
        records.append({
            "filename": fname,
            "datetime": dt,
            "year": dt.year,
            "doy": dt.dayofyear,
        })
    else:
        print(f"No date found in: {fname}")

df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)

# Unique (year, DOY) pairs — important if your data ever spans multiple years
unique_doys = sorted(df[["year", "doy"]].drop_duplicates().itertuples(index=False, name=None))

print(f"Found {len(df)} files, {len(unique_doys)} unique dates\n")

just_DOY_list = []

for year, doy in unique_doys:
    # print(f"{year}-{doy:03d}")
    just_DOY_list.append(f"{doy:03d}")

df.to_csv('./date_list.csv')

print('end')