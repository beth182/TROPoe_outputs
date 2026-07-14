"""
io/sonde_io.py
--------------
Radiosonde file loading. This logic only appeared once so far (in
HATPRO_raso_visu.py) — extracted here unchanged so future scripts can reuse
it without copy-pasting.
"""

import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd


def load_radiosonde(path):
    """
    Load a single TEAMx wEOP radiosonde file (ascent or descent).

    Parses the plain-text header to extract the launch/start time, then
    reads the comma-separated data rows.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    meta : dict  {'launch_time': datetime (UTC), 'kind': 'ascent'/'descent'}
    data : DataFrame  named columns, geopotential_height in metres, sorted
           by height ascending
    """
    path = Path(path)
    kind = "ascent" if "ascent" in path.name else "descent"

    header_lines = []
    data_lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("    "):
                header_lines.append(line.strip())
            else:
                data_lines.append(line)

    launch_time = None
    time_key = "ascent start time" if kind == "ascent" else "descent start time"
    for h in header_lines:
        if time_key in h:
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", h)
            if m:
                launch_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                launch_time = launch_time.replace(tzinfo=timezone.utc)

    col_name_line = [h for h in header_lines if "elapsed_time" in h]
    col_names = [c.strip() for c in col_name_line[0].split(",")] if col_name_line else None

    data_str = "".join(data_lines)
    data = pd.read_csv(StringIO(data_str), header=None, names=col_names)
    data = data.sort_values("geopotential_height").reset_index(drop=True)

    meta = {"launch_time": launch_time, "kind": kind}
    return meta, data


def load_all_radiosondes(sonde_paths, kinds=("ascent",)):
    """
    Load a list of radiosonde files, keep only the requested kind(s), and
    sort by launch time.

    Descent profiles are excluded by default because their data begins
    above the HATPRO ceiling (~10 km) and contribute no overlap.

    Parameters
    ----------
    sonde_paths : list of str or Path
    kinds       : tuple of str  e.g. ("ascent",) or ("ascent", "descent")

    Returns
    -------
    list of (meta dict, DataFrame), sorted by launch time
    """
    sondes = [load_radiosonde(p) for p in sonde_paths]
    sondes = [(m, d) for m, d in sondes if m["kind"] in kinds]
    sondes.sort(key=lambda x: x[0]["launch_time"])
    return sondes