"""
compare_tropoe_hatpro.py
========================
Compare TROPoe retrieval output (.nc) against HATPRO operational
retrieval CSVs (temperature, humidity, met) for a single day.

Each dataset is plotted on its own native height grid -- no interpolation.

HATPRO CSVs
  - First line: comment starting with '#'
  - Second line: header row, semicolon-delimited
  - Temperature: 39 levels, values in Kelvin  (converted to degC for plotting)
  - Humidity:    39 levels, values in g/m3
  - Cadence:     10-minute

TROPoe NetCDF
  - temperature:  (time=48, height=55), units = degC
  - waterVapor:   (time=48, height=55), units = g/kg
  - height:       (55,), units = km
  - Cadence:      ~30-minute

Outputs
-------
  - comparison_mean_profiles.png  : time-mean T and q profiles, 2 panels
  - comparison_timeseries.png     : T and q time series at selected heights

NOTE (refactor): TROPoe loading and HATPRO CSV loading now come from the
shared `tropoe_shared` package. Two things changed as a result -- see the
comments near the top of main() for what and why.
"""

import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- shared module import -----------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TROPoe_outputs import lookup

from TROPoe_outputs.functions.tropoe_io import load_tropoe
from TROPoe_outputs.functions.hatpro_io import load_hatpro_profiles, load_hatpro_met, select_hatpro_window
from TROPoe_outputs.functions.constants import HATPRO_HEIGHTS_KM

# --- CONFIGURATION -------------------------------------------------------------

# ToDo: loop over all dates
datestring = '20250219'

# Heights (km) at which to plot time series -- snapped to nearest level in each dataset
TIMESERIES_HEIGHTS_KM = [0.1, 0.5, 1.0, 2.0, 4.0]

_DATA = lookup.data_location
assert os.path.isdir(_DATA), f"Data folder not found: {_DATA}"

outdir = os.path.join(lookup.plot_save_location, "HATPRO_TROPoe_comparison/" + datestring + '/')
os.makedirs(outdir, exist_ok=True)

# condition based on the month for sEOP or wEOP
dt = datetime.strptime(datestring, '%Y%m%d')
month = dt.month  # 2 (an int, not zero-padded)
if month < 3:
    assert 1 <= month <= 2
    EOP = 'wEOP'
else:
    assert 6 <= month <= 7
    EOP = 'sEOP'

OUT_PREFIX = os.path.join(outdir, datestring + "_comparison")

T_CSV      = os.path.join(_DATA + 'HATPRO_processed_Massaro/TOC/', EOP + "_temperature.csv")
Q_CSV      = os.path.join(_DATA + 'HATPRO_processed_Massaro/TOC/', EOP + "_humidity.csv")
MET_CSV    = os.path.join(_DATA + 'HATPRO_processed_Massaro/TOC/', EOP + "_met.csv")
assert os.path.isfile(T_CSV), f"File not found: {T_CSV}"
assert os.path.isfile(Q_CSV), f"File not found: {Q_CSV}"
assert os.path.isfile(MET_CSV), f"File not found: {MET_CSV}"

FILE_PATTERN = _DATA + 'TROPoe_output/' + datestring + '/' + 'tropoe_innsbruck.c1.' + datestring + '*'
matches = glob.glob(FILE_PATTERN)
assert len(matches) == 1
NC_FILE = matches[0]

# --- PLOT FUNCTIONS ------------------------------------------------------------

def plot_mean_profiles(trop_hgt, trop_T, trop_wv,
                       hat_hgt, hat_T_C, hat_q,
                       out_prefix):
    """Two-panel time-mean profile plot -- each dataset on its own height grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    fig.suptitle(
        f"TROPoe vs HATPRO -- time-mean profiles\n{datestring[:4]}-{datestring[4:6]}-{datestring[6:8]}  |  Innsbruck",
        fontsize=12)

    ax1.plot(np.nanmean(hat_T_C, axis=0), hat_hgt,  "C1-o", ms=3, label="HATPRO")
    ax1.plot(np.nanmean(trop_T,  axis=0), trop_hgt, "C0-s", ms=3, label="TROPoe")
    ax1.set_xlabel("Temperature (degC)")
    ax1.set_ylabel("Height (km)")
    ax1.set_title("Temperature")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(np.nanmean(hat_q,   axis=0), hat_hgt,  "C1-o", ms=3, label="HATPRO (g/m3)")
    ax2.plot(np.nanmean(trop_wv, axis=0), trop_hgt, "C0-s", ms=3, label="TROPoe (g/m3)")
    ax2.set_xlabel("Water vapour (g/m3)")
    ax2.set_title("Humidity")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_file = f"{out_prefix}_mean_profiles.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Mean profiles saved -> {out_file}")


def plot_timeseries(trop_times, trop_hgt, trop_T, trop_wv,
                    hat_times, hat_hgt, hat_T_C, hat_q,
                    target_heights_km, out_prefix):
    """Time series of T and q at selected height levels.

    For each target height the nearest level in each dataset is used
    independently -- so HATPRO and TROPoe may be at slightly different
    actual heights, noted in the legend.
    """
    n_levels = len(target_heights_km)
    fig, axes = plt.subplots(n_levels, 2,
                             figsize=(14, 3 * n_levels),
                             sharex=True)
    fig.suptitle(f"TROPoe vs HATPRO -- time series  |  {datestring[:4]}-{datestring[4:6]}-{datestring[6:8]}", fontsize=12)

    for row, target_h in enumerate(target_heights_km):
        k_hat  = np.argmin(np.abs(hat_hgt  - target_h))
        k_trop = np.argmin(np.abs(trop_hgt - target_h))
        hat_label  = f"HATPRO  ({hat_hgt[k_hat]*1000:.0f} m)"
        trop_label = f"TROPoe  ({trop_hgt[k_trop]*1000:.0f} m)"

        ax = axes[row, 0]
        ax.plot(hat_times,  hat_T_C[:, k_hat],  "C1-",  lw=1,  label=hat_label)
        ax.plot(trop_times, trop_T[:, k_trop],  "C0-o", ms=3,  label=trop_label)
        ax.set_ylabel(f"~{target_h*1000:.0f} m\nT (degC)", fontsize=8)
        if row == 0:
            ax.set_title("Temperature")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(hat_times,  hat_q[:, k_hat],    "C1-",  lw=1,  label=hat_label)
        ax.plot(trop_times, trop_wv[:, k_trop], "C0-o", ms=3,  label=trop_label)
        ax.set_ylabel("q (g/m3)", fontsize=8)
        if row == 0:
            ax.set_title("Humidity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.set_xlabel("UTC")
    fig.autofmt_xdate()

    plt.tight_layout()
    out_file = f"{out_prefix}_timeseries.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Time series saved -> {out_file}")


# --- MAIN ----------------------------------------------------------------------

def main():
    print("Loading TROPoe ...")
    tro = load_tropoe(NC_FILE)
    trop_times = tro['timestamps']
    trop_hgt = tro['height']
    trop_T = tro['temp']
    trop_wv = tro['abs_hum_from_mixing']

    print("Loading HATPRO ...")
    hat_temp_k, hat_hum = load_hatpro_profiles(T_CSV, Q_CSV)
    hat_met = load_hatpro_met(MET_CSV)  # loaded for completeness; not plotted below

    # NEW: T_CSV/Q_CSV cover the whole EOP season, not just this day -- narrow
    # to the target date and align temp/hum onto matching timestamps.
    windowed = select_hatpro_window(datestring, temp=hat_temp_k, hum=hat_hum)
    hat_temp_k, hat_hum = windowed['temp'], windowed['hum']

    hat_times = hat_temp_k.index
    hat_T_C = hat_temp_k.values - 273.15  # HATPRO CSV is in Kelvin
    hat_q = hat_hum.values  # already g/m3

    plot_mean_profiles(trop_hgt, trop_T, trop_wv,
                       HATPRO_HEIGHTS_KM, hat_T_C, hat_q,
                       OUT_PREFIX)

    plot_timeseries(trop_times, trop_hgt, trop_T, trop_wv,
                    hat_times,  HATPRO_HEIGHTS_KM, hat_T_C, hat_q,
                    TIMESERIES_HEIGHTS_KM, OUT_PREFIX)


if __name__ == "__main__":
    main()