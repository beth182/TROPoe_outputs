import datetime
import os
from mwrpy.level1.write_lev1_nc import lev1_to_nc

print('imports complete')

# Paths
path_to_raw = r"I:\User\Documents\Research\Running_TROPoe\Converting_RAW_HATPRO_for_TROPoe\test_day_raw"
output_dir  = r"I:\User\Documents\Research\Running_TROPoe\Converting_RAW_HATPRO_for_TROPoe\output"
output_file = os.path.join(output_dir, "20260408_innsbruck_1C01.nc")

# Date of your test data (filename 26040804 = 2026-04-08)
date = datetime.date(2026, 4, 8)

# Make output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print('created out dir:', output_dir)

# Run MWRpy Level 1C processing
lev1_to_nc(
    data_type="1C01",          # Combined TB + MET + IRT — E-PROFILE format
    path_to_files=path_to_raw,
    site="innsbruck",
    output_file=output_file,
    date=date,
    instrument_type="hatpro",
)

print(f"Done — output at: {output_file}")


print('end')