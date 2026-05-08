import datetime
import os
from mwrpy.level1.write_lev1_nc import lev1_to_nc

print('imports complete')

datestring = '20250219'

# Paths
path_to_raw = r"I:/User/Documents/PycharmProjects/TROPoe_outputs/create_TROPoe_inputs/data/" + datestring + '/'
output_dir = r"I:/User/Documents/PycharmProjects/TROPoe_outputs/create_TROPoe_inputs/output/" + datestring + '/'
output_file = os.path.join(output_dir, "innsbruck_1C01_" + datestring + ".nc")

# Date of your test data (filename 26040804 = 2026-04-08)
dt = datetime.datetime.strptime(datestring, '%Y%m%d')

year = dt.year  # 2025
month = dt.month  # 2
day = dt.day  # 19

date = datetime.date(year, month, day)

# Make output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print('created out dir:', output_dir)

# Run MWRpy Level 1C processing
lev1_to_nc(
    data_type="1C01",  # Combined TB + MET + IRT — E-PROFILE format
    path_to_files=path_to_raw,
    site="innsbruck",
    output_file=output_file,
    date=date,
    instrument_type="hatpro",
)

print(f"Done — output at: {output_file}")

print('end')
