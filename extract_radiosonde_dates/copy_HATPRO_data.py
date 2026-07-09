"""
copy_hatpro_by_date.py

Reads a CSV of radiosonde filenames with an associated `datetime` column,
extracts the unique dates, and for each date copies all files from the
raw HATPRO archive folder for that date into the TROPoe inputs folder,
renaming the destination folder to YYYYMMDD.

Source structure (per date):
    Z:\\rawdata\\hatpro\\Y2025\\M02\\D03\\...   (~720 files/day)

Destination structure (per date):
    I:\\User\\Documents\\PycharmProjects\\TROPoe_outputs\\create_TROPoe_inputs\\data\\TOC\\20250203\\...

Run interactively (e.g. in PyCharm) - no CLI args needed, just edit the
CONFIG block below and run.
"""

import shutil
from pathlib import Path
from datetime import date

import pandas as pd


# ----------------------------------------------------------------------
# CONFIG - edit these paths/settings before running
# ----------------------------------------------------------------------

CSV_PATH = Path(r"I:\User\Documents\PycharmProjects\TROPoe_outputs\extract_radiosonde_dates\date_list.csv")

SOURCE_BASE = Path(r"Z:\rawdata\hatpro")
DEST_BASE = Path(r"I:\User\Documents\PycharmProjects\TROPoe_outputs\create_TROPoe_inputs\data\TOC")

DATETIME_COLUMN = "datetime"

# If True, files that already exist in the destination (same name) are
# skipped rather than re-copied/overwritten. Handy for re-running the
# script after it was interrupted partway through.
SKIP_EXISTING = True

# If True, only print what *would* happen - no folders created, no files
# copied. Useful for a first sanity check.
DRY_RUN = False


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def get_unique_dates(csv_path: Path, datetime_column: str) -> list[date]:
    """Read the CSV and return a sorted list of unique dates (date objects).

    Tries a fixed day-first format first (matches e.g. "14/02/2025 08:00").
    Falls back to pandas' mixed-format inference (dayfirst) if that fails,
    so this keeps working if the datetime column's format changes.
    """
    df = pd.read_csv(csv_path)
    raw = df[datetime_column]
    try:
        dt = pd.to_datetime(raw, format="%d/%m/%Y %H:%M")
    except ValueError:
        dt = pd.to_datetime(raw, format="mixed", dayfirst=True)
    unique_dates = sorted(dt.dt.date.unique())
    return list(unique_dates)


def build_source_dir(day: date, source_base: Path) -> Path:
    """Z:\\rawdata\\hatpro\\Y2025\\M02\\D03"""
    return source_base / f"Y{day.year}" / f"M{day.month:02d}" / f"D{day.day:02d}"


def build_dest_dir(day: date, dest_base: Path) -> Path:
    """...\\data\\TOC\\20250203"""
    return dest_base / f"{day.year}{day.month:02d}{day.day:02d}"


def copy_day_files(day: date, source_base: Path, dest_base: Path,
                    skip_existing: bool = True, dry_run: bool = False) -> dict:
    """Copy all files (not subfolders) from the source day-folder into the
    destination day-folder. Returns a summary dict for this date."""

    src_dir = build_source_dir(day, source_base)
    dst_dir = build_dest_dir(day, dest_base)

    summary = {
        "date": day,
        "src_dir": src_dir,
        "dst_dir": dst_dir,
        "src_exists": src_dir.is_dir(),
        "n_files_found": 0,
        "n_copied": 0,
        "n_skipped": 0,
    }

    if not src_dir.is_dir():
        return summary

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in src_dir.iterdir() if f.is_file()]
    summary["n_files_found"] = len(files)

    for f in files:
        dst_file = dst_dir / f.name
        if skip_existing and dst_file.exists():
            summary["n_skipped"] += 1
            continue
        if not dry_run:
            shutil.copy2(f, dst_file)
        summary["n_copied"] += 1

    return summary


def print_summary(summary: dict) -> None:
    day = summary["date"]
    if not summary["src_exists"]:
        print(f"{day}  MISSING source dir: {summary['src_dir']}")
        return
    print(
        f"{day}  found {summary['n_files_found']:4d}  "
        f"copied {summary['n_copied']:4d}  skipped {summary['n_skipped']:4d}  "
        f"-> {summary['dst_dir']}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    dates = get_unique_dates(CSV_PATH, DATETIME_COLUMN)
    print(f"Found {len(dates)} unique dates in {CSV_PATH.name}\n")

    all_summaries = []
    for day in dates:
        summary = copy_day_files(
            day, SOURCE_BASE, DEST_BASE,
            skip_existing=SKIP_EXISTING, dry_run=DRY_RUN,
        )
        print_summary(summary)
        all_summaries.append(summary)

    missing = [s for s in all_summaries if not s["src_exists"]]
    total_copied = sum(s["n_copied"] for s in all_summaries)
    total_skipped = sum(s["n_skipped"] for s in all_summaries)

    print("\n--- Done ---")
    print(f"Dates processed : {len(all_summaries)}")
    print(f"Files copied    : {total_copied}")
    print(f"Files skipped   : {total_skipped}")
    if missing:
        print(f"Dates with missing source folder ({len(missing)}):")
        for s in missing:
            print(f"  {s['date']}  ({s['src_dir']})")


if __name__ == "__main__":
    main()

    print('end')