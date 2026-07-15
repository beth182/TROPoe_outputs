"""
Copy TROPoe output NetCDF files from local run directory to Nextcloud.

Source:      C:\\Users\\c7071147\\Documents\\TROPoe_run\\TOC\\<datestring>\\tropoe\\hatpro\\tropoe_innsbruck.c1.<datestring>.*.nc
Destination: C:\\Users\\c7071147\\Nextcloud\\data\\TEAMx_IOPs\\TROPoe_output\\TOC\\<datestring>\\

Expects one matching .nc file per date folder. Warns if zero or multiple matches are found.
"""

import shutil
from pathlib import Path
import os
from TROPoe_outputs import lookup

# ---- Config ----
SOURCE_ROOT = Path(r"C:\Users\c7071147\Documents\TROPoe_run\TOC")
DEST_ROOT = Path(lookup.data_location + "TROPoe_output/TOC/")

assert os.path.isdir(SOURCE_ROOT), f"Data folder not found: {SOURCE_ROOT}"
assert os.path.isdir(DEST_ROOT), f"Data folder not found: {DEST_ROOT}"

OVERWRITE = False  # set False to skip files that already exist at the destination


def copy_all():
    if not SOURCE_ROOT.exists():
        print(f"Source root does not exist: {SOURCE_ROOT}")
        return

    # Each subfolder of SOURCE_ROOT is assumed to be a datestring, e.g. 20250116
    date_dirs = sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())

    if not date_dirs:
        print(f"No date folders found under {SOURCE_ROOT}")
        return

    copied, skipped, missing = 0, 0, []

    for date_dir in date_dirs:
        datestring = date_dir.name
        hatpro_dir = date_dir / "tropoe" / "hatpro"

        if not hatpro_dir.exists():
            missing.append(datestring)
            continue

        matches = list(hatpro_dir.glob(f"tropoe_innsbruck.c1.{datestring}.*.nc"))

        if len(matches) == 0:
            print(f"[{datestring}] no matching .nc file found in {hatpro_dir}")
            missing.append(datestring)
            continue
        elif len(matches) > 1:
            print(f"[{datestring}] WARNING: multiple matches found, copying all: {[m.name for m in matches]}")

        dest_dir = DEST_ROOT / datestring
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_file in matches:
            dest_file = dest_dir / src_file.name

            if dest_file.exists() and not OVERWRITE:
                print(f"[{datestring}] already exists, skipping: {dest_file.name}")
                skipped += 1
                continue

            shutil.copy2(src_file, dest_file)
            print(f"[{datestring}] copied: {src_file.name} -> {dest_dir}")
            copied += 1

    print("\n--- Summary ---")
    print(f"Copied:  {copied}")
    print(f"Skipped: {skipped}")
    if missing:
        print(f"No file found for {len(missing)} date(s): {missing}")


if __name__ == "__main__":
    copy_all()

    print('end')