"""
prep_tropoe_runs.py

Automates the boring prep work for running TROPoe manually (via WSL, as admin):
  - creates the date folder + hatpro/ + tropoe/hatpro/ subfolders
  - copies run_tropoe_ops.sh into the date folder
  - copies vip.txt into tropoe/hatpro/ (edit VIP_TEMPLATE per-date logic below if needed)
  - copies the prepared obs file(s) from Nextcloud into hatpro/
  - generates execute_line.txt with the correct docker run command for that date

You still have to open your admin WSL shell and run:
    bash <path from execute_line.txt>
for each date yourself.

Run this with plain Windows Python (not WSL) - it's just filesystem ops.


TOC/{date}/
├── run_tropoe_ops.sh
├── execute_line.txt
├── hatpro/                  ← obs files copied here (mwr_path)
└── tropoe/hatpro/
    └── vip.txt              ← output also lands here (output_path)

"""

import shutil
from pathlib import Path
import os

from TROPoe_outputs import lookup

# ----------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------

# T0Do: update this automatically
DATES = ["20250219"]

# Where the per-date run folders will be created
BASE_WIN = Path(lookup.TROPoe_run_location + "\TOC")
assert os.path.isdir(BASE_WIN), f"Data folder not found: {BASE_WIN}"

# Where the prepared obs files live, one subfolder per date
NEXTCLOUD_OBS_BASE = Path(lookup.data_location + "\HATPRO_input_for_TROPoe\TOC")
assert os.path.isdir(NEXTCLOUD_OBS_BASE), f"Data folder not found: {NEXTCLOUD_OBS_BASE}"

# Master templates (edit these once, they get copied into every date folder)
VIP_TEMPLATE = Path(lookup.TROPoe_run_file_template_location + "vip.txt")
SH_TEMPLATE = Path(lookup.TROPoe_run_file_template_location + "run_tropoe_ops.sh")
assert os.path.isfile(VIP_TEMPLATE), f"File not found: {VIP_TEMPLATE}"
assert os.path.isfile(SH_TEMPLATE), f"File not found: {SH_TEMPLATE}"

# docker run args (same as your example execute_line.txt)
PRIOR_FILE = "/home/tropoe/vip/src/tropoe/prior.MIDLAT.nc"
SHOUR = 0
EHOUR = 24
VERBOSE = 2
TMP_PATH = "/tmp/tropoe_tmp"
IMAGE_ID = "davidturner53/tropoe:v0.20"

# ----------------------------------------------------------------------


def win_to_wsl(path: Path) -> str:
    """C:\\Users\\foo -> /mnt/c/Users/foo"""
    s = str(path).replace("\\", "/")
    drive, rest = s.split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


def edit_vip_for_date(vip_text: str, date: str) -> str:
    """
    Hook for any per-date vip.txt edits. Currently a no-op (nothing in the
    vip file is date-dependent - TROPoe gets the date from the command line).
    Add string replacements here if that changes, e.g.:
        vip_text = vip_text.replace("output_rootname = tropoe_innsbruck.c1",
                                     f"output_rootname = tropoe_toc_{date}.c1")
    """
    return vip_text


def prep_date(date: str) -> None:
    date_folder = BASE_WIN / date
    hatpro_dir = date_folder / "hatpro"
    tropoe_hatpro_dir = date_folder / "tropoe" / "hatpro"

    hatpro_dir.mkdir(parents=True, exist_ok=True)
    tropoe_hatpro_dir.mkdir(parents=True, exist_ok=True)

    # 1. run_tropoe_ops.sh
    shutil.copy(SH_TEMPLATE, date_folder / "run_tropoe_ops.sh")

    # 2. vip.txt (edited if needed) -> tropoe/hatpro/vip.txt
    vip_text = VIP_TEMPLATE.read_text()
    vip_text = edit_vip_for_date(vip_text, date)
    (tropoe_hatpro_dir / "vip.txt").write_text(vip_text)

    # 3. copy obs file(s) from Nextcloud into hatpro/
    src_obs_dir = NEXTCLOUD_OBS_BASE / date
    if src_obs_dir.exists():
        copied = 0
        for f in src_obs_dir.iterdir():
            if f.is_file():
                shutil.copy(f, hatpro_dir / f.name)
                copied += 1
        if copied == 0:
            print(f"  WARNING [{date}]: source folder exists but has no files: {src_obs_dir}")
    else:
        print(f"  WARNING [{date}]: source obs folder not found: {src_obs_dir}")

    # 4. execute_line.txt
    wsl_folder = win_to_wsl(date_folder)
    exec_line = (
        f"bash {wsl_folder}/run_tropoe_ops.sh   {date}   "
        f"/data/tropoe/hatpro/vip.txt   {PRIOR_FILE}   "
        f"{SHOUR} {EHOUR} {VERBOSE}   {wsl_folder}   {TMP_PATH}   {IMAGE_ID}"
    )
    (date_folder / "execute_line.txt").write_text(exec_line + "\n")

    print(f"Prepared {date}: {date_folder}")
    print('end')


if __name__ == "__main__":
    for d in DATES:
        print(d)
        prep_date(d)

    print("\nDone. For each date, open your admin WSL shell and run the command in execute_line.txt.")
    print('end')