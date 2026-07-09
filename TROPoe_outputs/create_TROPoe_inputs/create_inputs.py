import datetime
import os
from mwrpy.level1.write_lev1_nc import lev1_to_nc

from TROPoe_outputs import lookup

print('imports complete')

# Base paths (parent dirs containing one subfolder per date, e.g. 20250219/)
base_raw_dir = lookup.data_location + "/raw_HATPRO/TOC/"
base_output_dir = lookup.data_location + "/HATPRO_input_for_TROPoe/TOC/"

assert os.path.isdir(base_raw_dir), f"Data folder not found: {base_raw_dir}"
assert os.path.isdir(base_output_dir), f"Data folder not found: {base_output_dir}"

# Find all subdirectories of base_raw_dir that look like dates (YYYYMMDD)
datestrings = []
for name in sorted(os.listdir(base_raw_dir)):
    full_path = os.path.join(base_raw_dir, name)
    if not os.path.isdir(full_path):
        continue
    try:
        datetime.datetime.strptime(name, '%Y%m%d')
    except ValueError:
        continue  # not a date-named folder, skip it
    datestrings.append(name)

print(f"Found {len(datestrings)} date folders: {datestrings}")

# Process each date
for datestring in datestrings:
    print(f"\n--- Processing {datestring} ---")

    path_to_raw = os.path.join(base_raw_dir, datestring) + '/'
    output_dir = os.path.join(base_output_dir, datestring) + '/'
    output_file = os.path.join(output_dir, "innsbruck_1C01_" + datestring + ".nc")

    dt = datetime.datetime.strptime(datestring, '%Y%m%d')
    date = datetime.date(dt.year, dt.month, dt.day)

    os.makedirs(output_dir, exist_ok=True)
    print('created out dir:', output_dir)

    try:
        lev1_to_nc(
            data_type="1C01",
            path_to_files=path_to_raw,
            site="innsbruck",
            output_file=output_file,
            date=date,
            instrument_type="hatpro",
        )
        print(f"Done — output at: {output_file}")
    except Exception as e:
        print(f"FAILED for {datestring}: {e}")
        continue

print('\nend')