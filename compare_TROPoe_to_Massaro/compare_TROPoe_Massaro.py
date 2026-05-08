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
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURATION -------------------------------------------------------------

datestring = '20250219'

# HATPRO retrieval height grid [km above station]
# from 26040823_CMP_TPC.NC altitude_layers (m -> km)
HATPRO_HEIGHTS_KM = np.array([
    0.000, 0.010, 0.030, 0.050, 0.075, 0.100, 0.125, 0.150,
    0.200, 0.250, 0.325, 0.400, 0.475, 0.550, 0.625, 0.700,
    0.800, 0.900, 1.000, 1.150, 1.300, 1.450, 1.600, 1.800,
    2.000, 2.200, 2.500, 2.800, 3.100, 3.500, 3.900, 4.400,
    5.000, 5.600, 6.200, 7.000, 8.000, 9.000, 10.000,
])

# Heights (km) at which to plot time series -- snapped to nearest level in each dataset
TIMESERIES_HEIGHTS_KM = [0.1, 0.5, 1.0, 2.0, 4.0]

# Data lives in test_day_data/ next to this script
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data/" + datestring + '/')

assert os.path.isdir(_DATA), f"Data folder not found: {_DATA}"

outdir = os.path.join(_HERE, "plots/" + datestring + '/')
# Make output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

NC_FILE    = os.path.join(_DATA, "tropoe_innsbruck.c1." + datestring + ".000015.nc")
T_CSV      = os.path.join(_DATA, "data_temperature.csv")
Q_CSV      = os.path.join(_DATA, "data_humidity.csv")
MET_CSV    = os.path.join(_DATA, "data_met.csv")
OUT_PREFIX = os.path.join(outdir, datestring + "_comparison")

# --- DATA LOADING --------------------------------------------------------------

def load_tropoe(nc_path):
    """Load TROPoe temperature, water vapour, and height from NetCDF."""
    ds      = nc.Dataset(nc_path)
    base_dt = pd.to_datetime(datestring, format='%Y%m%d')
    hours   = ds.variables["hour"][:]
    times   = pd.to_datetime([base_dt + pd.Timedelta(hours=float(h)) for h in hours])
    heights = ds.variables["height"][:].data          # km, (55,)
    T_C     = ds.variables["temperature"][:].data     # degC, (time, height)
    wv_gkg  = ds.variables["waterVapor"][:].data      # g/kg, (time, height)
    ds.close()
    return times, heights, T_C, wv_gkg


def load_hatpro(t_path, q_path, met_path):
    """Load HATPRO temperature and humidity CSVs."""
    def read_csv(path):
        df = pd.read_csv(path, comment="#", sep=";", parse_dates=["rawdate"])
        df = df.rename(columns={"rawdate": "time"}).set_index("time").sort_index()
        df.columns = [f"lev{i:02d}" for i in range(df.shape[1])]
        return df

    T_df = read_csv(t_path)   # K
    q_df = read_csv(q_path)   # g/m3
    met  = read_csv(met_path)

    T_C = T_df - 273.15       # convert to degC

    return T_df.index, T_C.values, q_df.values, met


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
    ax2.plot(np.nanmean(trop_wv, axis=0), trop_hgt, "C0-s", ms=3, label="TROPoe (g/kg)")
    ax2.set_xlabel("Water vapour")
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
        ax.set_ylabel("q", fontsize=8)
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
    trop_times, trop_hgt, trop_T, trop_wv = load_tropoe(NC_FILE)

    print("Loading HATPRO ...")
    hat_times, hat_T_C, hat_q, hat_met = load_hatpro(T_CSV, Q_CSV, MET_CSV)

    plot_mean_profiles(trop_hgt, trop_T, trop_wv,
                       HATPRO_HEIGHTS_KM, hat_T_C, hat_q,
                       OUT_PREFIX)

    plot_timeseries(trop_times, trop_hgt, trop_T, trop_wv,
                    hat_times,  HATPRO_HEIGHTS_KM, hat_T_C, hat_q,
                    TIMESERIES_HEIGHTS_KM, OUT_PREFIX)


if __name__ == "__main__":
    main()