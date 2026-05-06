"""
compare_tropoe_hatpro.py
========================
Compare TROPoe retrieval output (.nc) against HATPRO operational
retrieval CSVs (temperature, humidity, met) for a single day.

Assumptions / known dataset conventions
----------------------------------------
HATPRO CSVs
  - First line: comment starting with '#'
  - Second line: header row, semicolon-delimited
  - Temperature: 39 levels, values in Kelvin
  - Humidity:    39 levels, values in g/m³ (absolute humidity)
  - Met:         surface obs (hs, ps, rf, ts, dd, ff)
  - Cadence:     10-minute

TROPoe NetCDF
  - temperature:  (time=48, height=55), units = degC
  - waterVapor:   (time=48, height=55), units = g/kg
  - height:       (55,), units = km
  - hour:         hours since midnight on the file date
  - Cadence:      ~30-minute

Unit conversions applied before comparison
-------------------------------------------
  HATPRO T [K]      → [°C]     :  T_C = T_K − 273.15
  HATPRO q [g/m³]   → [g/kg]   :  q_gkg = q_gm3 / rho_dry [kg/m³]
        where rho_dry ≈ P / (Rd * T_K)   (ideal gas)
        Pressure for each HATPRO level is interpolated from TROPoe pressure.

Height matching
---------------
HATPRO has 39 fixed retrieval heights (not stored in the CSVs — you must
supply them, or they are inferred from the instrument configuration).
Edit HATPRO_HEIGHTS_KM below to match your system.  The script then
interpolates TROPoe profiles onto those heights for a fair comparison.

Outputs
-------
  - comparison_profiles.png   : time-mean profiles of T and q side by side
  - comparison_timeseries.png : time series of T and q at selected levels
  - comparison_stats.csv      : bias, RMSE, correlation per height level
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────

# HATPRO retrieval height grid [km above station].
# These are the standard HATPRO RPG retrieval levels — adjust if yours differ.
HATPRO_HEIGHTS_KM = np.array([
    0.000, 0.010, 0.030, 0.050, 0.075, 0.100, 0.125, 0.150,
    0.200, 0.250, 0.325, 0.400, 0.475, 0.550, 0.625, 0.700,
    0.800, 0.900, 1.000, 1.150, 1.300, 1.450, 1.600, 1.800,
    2.000, 2.200, 2.500, 2.800, 3.100, 3.500, 3.900, 4.400,
    5.000, 5.600, 6.200, 7.000, 8.000, 9.000, 10.000,
])  # 39 levels — from 26040823_CMP_TPC.NC altitude_layers (m → km)

# Heights (km) at which to plot time-series comparisons
TIMESERIES_HEIGHTS_KM = [0.1, 0.5, 1.0, 2.0, 4.0]

# Data lives in test_day_data/ next to this script
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "test_day_data")

NC_FILE    = os.path.join(_DATA, "tropoe_innsbruck.c1.20260408.000015.nc")
T_CSV      = os.path.join(_DATA, "data_temperature.csv")
Q_CSV      = os.path.join(_DATA, "data_humidity.csv")
MET_CSV    = os.path.join(_DATA, "data_met.csv")
OUT_PREFIX = os.path.join(_HERE, "comparison")


# ─── HELPERS ───────────────────────────────────────────────────────────────────

Rd = 287.05  # J kg⁻¹ K⁻¹


def read_hatpro_csv(path, comment="#", sep=";"):
    """Read a HATPRO semicolon-delimited CSV with a leading comment line."""
    df = pd.read_csv(path, comment=comment, sep=sep, parse_dates=["rawdate"])
    df = df.rename(columns={"rawdate": "time"})
    df = df.set_index("time").sort_index()
    return df


def load_hatpro(t_path, q_path, met_path):
    """Return (T_K, q_gm3, met) DataFrames on the HATPRO time/height grid."""
    T_df  = read_hatpro_csv(t_path)   # K
    q_df  = read_hatpro_csv(q_path)   # g/m³
    met   = read_hatpro_csv(met_path)  # surface obs

    # Rename v01..v39 → height index for clarity
    n_lev = T_df.shape[1]
    T_df.columns = [f"lev{i:02d}" for i in range(n_lev)]
    q_df.columns = [f"lev{i:02d}" for i in range(n_lev)]

    return T_df, q_df, met


def load_tropoe(nc_path):
    """Return arrays from TROPoe NetCDF: times, heights_km, T_C, wv_gkg, P_mb."""
    ds = nc.Dataset(nc_path)
    base_dt = pd.Timestamp("2026-04-08 00:00:00")
    hours    = ds.variables["hour"][:]
    times    = pd.to_datetime([base_dt + pd.Timedelta(hours=float(h)) for h in hours])
    heights  = ds.variables["height"][:]       # km
    T_C      = ds.variables["temperature"][:].data   # (time, height) °C
    wv_gkg   = ds.variables["waterVapor"][:].data    # (time, height) g/kg
    P_mb     = ds.variables["pressure"][:].data      # (time, height) mb
    qc       = ds.variables["qc_flag"][:].data       # (time,)
    ds.close()
    return times, heights, T_C, wv_gkg, P_mb, qc


def interp_profile(src_heights, src_values, dst_heights, fill=np.nan):
    """Interpolate a single profile from src_heights → dst_heights (linear)."""
    f = interp1d(src_heights, src_values,
                 bounds_error=False, fill_value=fill)
    return f(dst_heights)


def convert_hatpro_q_to_gkg(q_gm3_row, T_K_row, P_mb_row):
    """Convert absolute humidity [g/m³] → mixing ratio [g/kg] using ideal gas.

    rho_dry = P / (Rd * T)  [kg/m³]
    q_gkg   = q_gm3 / rho_dry
    """
    T_K  = np.array(T_K_row, dtype=float)
    P_Pa = np.array(P_mb_row, dtype=float) * 100.0   # mb → Pa
    q    = np.array(q_gm3_row, dtype=float)
    rho  = P_Pa / (Rd * T_K)   # kg/m³
    return q / rho              # g/kg


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main(nc_file=NC_FILE, t_csv=T_CSV, q_csv=Q_CSV, met_csv=MET_CSV,
         out_prefix="comparison"):

    print("Loading TROPoe …")
    trop_times, trop_hgt, trop_T, trop_wv, trop_P, trop_qc = load_tropoe(nc_file)

    print("Loading HATPRO CSVs …")
    hat_T_df, hat_q_df, hat_met = load_hatpro(t_csv, q_csv, met_csv)

    # ── Match timestamps: for each TROPoe time find nearest HATPRO ────────────
    hat_times = hat_T_df.index
    matched_idx = []
    for tt in trop_times:
        diff = np.abs(hat_times - tt)
        # ToDo: flag here if difference is more than 15 seconds
        matched_idx.append(diff.argmin())

    hat_T_matched = hat_T_df.iloc[matched_idx].values   # (48, 39) K
    hat_q_matched = hat_q_df.iloc[matched_idx].values   # (48, 39) g/m³
    match_times   = trop_times

    # ── Interpolate TROPoe pressure onto HATPRO heights ───────────────────────
    trop_P_on_hat = np.full((len(trop_times), len(HATPRO_HEIGHTS_KM)), np.nan)
    for i in range(len(trop_times)):
        trop_P_on_hat[i] = interp_profile(trop_hgt, trop_P[i], HATPRO_HEIGHTS_KM)

    # ── Unit conversions ──────────────────────────────────────────────────────
    # HATPRO T: K → °C
    hat_T_C = hat_T_matched - 273.15

    # HATPRO q: g/m³ → g/kg  (use interpolated TROPoe pressure + HATPRO T[K])
    hat_q_gkg = np.full_like(hat_q_matched, np.nan)
    for i in range(len(match_times)):
        hat_q_gkg[i] = convert_hatpro_q_to_gkg(
            hat_q_matched[i], hat_T_matched[i], trop_P_on_hat[i]
        )

    # ── Interpolate TROPoe profiles onto HATPRO heights ───────────────────────
    trop_T_on_hat  = np.full((len(trop_times), len(HATPRO_HEIGHTS_KM)), np.nan)
    trop_wv_on_hat = np.full_like(trop_T_on_hat, np.nan)
    for i in range(len(trop_times)):
        trop_T_on_hat[i]  = interp_profile(trop_hgt, trop_T[i],  HATPRO_HEIGHTS_KM)
        trop_wv_on_hat[i] = interp_profile(trop_hgt, trop_wv[i], HATPRO_HEIGHTS_KM)

    # ── Statistics per height level ───────────────────────────────────────────
    bias_T  = np.nanmean(trop_T_on_hat  - hat_T_C,   axis=0)
    rmse_T  = np.sqrt(np.nanmean((trop_T_on_hat  - hat_T_C)**2,   axis=0))
    bias_q  = np.nanmean(trop_wv_on_hat - hat_q_gkg, axis=0)
    rmse_q  = np.sqrt(np.nanmean((trop_wv_on_hat - hat_q_gkg)**2, axis=0))

    corr_T = np.array([
        np.corrcoef(trop_T_on_hat[:, k], hat_T_C[:, k])[0, 1]
        for k in range(len(HATPRO_HEIGHTS_KM))
    ])
    corr_q = np.array([
        np.corrcoef(trop_wv_on_hat[:, k], hat_q_gkg[:, k])[0, 1]
        for k in range(len(HATPRO_HEIGHTS_KM))
    ])

    stats_df = pd.DataFrame({
        "height_km":   HATPRO_HEIGHTS_KM,
        "bias_T_degC": bias_T,
        "rmse_T_degC": rmse_T,
        "corr_T":      corr_T,
        "bias_q_gkg":  bias_q,
        "rmse_q_gkg":  rmse_q,
        "corr_q":      corr_q,
    })
    stats_file = f"{out_prefix}_stats.csv"
    stats_df.to_csv(stats_file, index=False, float_format="%.4f")
    print(f"Stats saved → {stats_file}")

    # ── Plot 1: time-mean profiles ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 7), sharey=True)
    fig.suptitle("TROPoe vs HATPRO — 2026-04-08  |  Innsbruck", fontsize=13)

    ax = axes[0]
    ax.plot(np.nanmean(hat_T_C,       axis=0), HATPRO_HEIGHTS_KM, "C1-o", ms=3, label="HATPRO")
    ax.plot(np.nanmean(trop_T_on_hat, axis=0), HATPRO_HEIGHTS_KM, "C0-s", ms=3, label="TROPoe")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Height (km)")
    ax.set_title("Mean T profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(bias_T, HATPRO_HEIGHTS_KM, "k-o", ms=3)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Bias  TROPoe − HATPRO  (°C)")
    ax.set_title("T bias & RMSE")
    ax.plot(rmse_T, HATPRO_HEIGHTS_KM, "r--o", ms=3, label="RMSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(np.nanmean(hat_q_gkg,     axis=0), HATPRO_HEIGHTS_KM, "C1-o", ms=3, label="HATPRO")
    ax.plot(np.nanmean(trop_wv_on_hat,axis=0), HATPRO_HEIGHTS_KM, "C0-s", ms=3, label="TROPoe")
    ax.set_xlabel("Water vapour (g/kg)")
    ax.set_title("Mean q profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(bias_q, HATPRO_HEIGHTS_KM, "k-o", ms=3, label="Bias")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.plot(rmse_q, HATPRO_HEIGHTS_KM, "r--o", ms=3, label="RMSE")
    ax.set_xlabel("Bias  TROPoe − HATPRO  (g/kg)")
    ax.set_title("q bias & RMSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    prof_file = f"{out_prefix}_profiles.png"
    fig.savefig(prof_file, dpi=150)
    plt.close(fig)
    print(f"Profiles plot saved → {prof_file}")

    # ── Plot 2: time series at selected height levels ─────────────────────────
    fig, axes = plt.subplots(len(TIMESERIES_HEIGHTS_KM), 2,
                             figsize=(14, 3*len(TIMESERIES_HEIGHTS_KM)),
                             sharex=True)
    fig.suptitle("TROPoe vs HATPRO — time series by level  |  2026-04-08", fontsize=12)

    hat_times_matched = hat_times[matched_idx]

    for row, target_h in enumerate(TIMESERIES_HEIGHTS_KM):
        k = np.argmin(np.abs(HATPRO_HEIGHTS_KM - target_h))
        label_h = f"{HATPRO_HEIGHTS_KM[k]*1000:.0f} m"

        ax = axes[row, 0]
        ax.plot(hat_times_matched, hat_T_C[:, k],       "C1-o", ms=3, label="HATPRO")
        ax.plot(match_times,       trop_T_on_hat[:, k], "C0-s", ms=3, label="TROPoe")
        ax.set_ylabel(f"{label_h}\nT (°C)", fontsize=8)
        if row == 0:
            ax.set_title("Temperature")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(hat_times_matched, hat_q_gkg[:, k],      "C1-o", ms=3, label="HATPRO")
        ax.plot(match_times,       trop_wv_on_hat[:, k], "C0-s", ms=3, label="TROPoe")
        ax.set_ylabel("q (g/kg)", fontsize=8)
        if row == 0:
            ax.set_title("Water vapour")
        ax.grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.set_xlabel("UTC")
    fig.autofmt_xdate()

    plt.tight_layout()
    ts_file = f"{out_prefix}_timeseries.png"
    fig.savefig(ts_file, dpi=150)
    plt.close(fig)
    print(f"Time-series plot saved → {ts_file}")

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n── Summary statistics (column mean across heights) ──")
    print(f"  Mean |bias| T  : {np.nanmean(np.abs(bias_T)):.3f} °C")
    print(f"  Mean RMSE  T   : {np.nanmean(rmse_T):.3f} °C")
    print(f"  Mean |bias| q  : {np.nanmean(np.abs(bias_q)):.4f} g/kg")
    print(f"  Mean RMSE  q   : {np.nanmean(rmse_q):.4f} g/kg")


if __name__ == "__main__":
    main(nc_file=NC_FILE, t_csv=T_CSV, q_csv=Q_CSV,
         met_csv=MET_CSV, out_prefix=OUT_PREFIX)