"""
HATPRO vs Radiosonde Comparison
--------------------------------
TEAMx wEOP campaign, Kolsass, 2025-02-19

Data handling and visualisation are separated into functions so that a
statistics module can be added later without touching the pipeline.

Sonde columns (0-indexed):
  0  elapsed_time          second
  1  latitude              degree
  2  longitude             degree
  3  geopotential_height   meter
  4  air_pressure          pascal
  5  wind_from_direction   degree
  6  wind_speed            m/s
  7  u_wind                m/s
  8  v_wind                m/s
  9  air_temperature       K
 10  dew_point_temperature K
 11  air_potential_temperature K
 12  relative_humidity     percent
 13  humidity_mixing_ratio g/kg

HATPRO files (semicolon-separated, first row = comment, second row = header):
  data_temperature.csv  : temperature profiles [K], columns v01..v39
  data_humidity.csv     : absolute humidity profiles [g/m³], columns v01..v39
  data_met.csv          : surface met (hs, ps, rf, ts, dd, ff)

HATPRO height grid (km above ground, 39 levels):
  see HATPRO_HEIGHTS_KM below
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HATPRO_HEIGHTS_KM = np.array([
    0.000, 0.010, 0.030, 0.050, 0.075, 0.100, 0.125, 0.150,
    0.200, 0.250, 0.325, 0.400, 0.475, 0.550, 0.625, 0.700,
    0.800, 0.900, 1.000, 1.150, 1.300, 1.450, 1.600, 1.800,
    2.000, 2.200, 2.500, 2.800, 3.100, 3.500, 3.900, 4.400,
    5.000, 5.600, 6.200, 7.000, 8.000, 9.000, 10.000,
])  # km agl

# Maximum time difference (minutes) between sonde launch and HATPRO retrieval
MATCH_WINDOW_MIN = 10

# HATPRO instrument altitude above sea level (Kolsass i-Box, ~545 m asl)
HATPRO_SITE_ELEV_M = 545.0  # 612.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hatpro_profiles(temp_path, hum_path):
    """
    Load HATPRO temperature and humidity profile CSVs.

    Parameters
    ----------
    temp_path : str or Path
    hum_path  : str or Path

    Returns
    -------
    temp_df : DataFrame  index=datetime, columns=39 height levels [K]
    hum_df  : DataFrame  index=datetime, columns=39 height levels [g/m³]
    """
    def _read(path):
        df = pd.read_csv(path, sep=";", skiprows=1, index_col=0,
                         parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=False)
        df.columns = range(len(df.columns))   # 0-based integer column index
        return df

    return _read(temp_path), _read(hum_path)


def load_hatpro_met(met_path):
    """
    Load HATPRO surface met file.

    Returns
    -------
    DataFrame  index=datetime, columns=[hs, ps, rf, ts, dd, ff]
    """
    df = pd.read_csv(met_path, sep=";", skiprows=1, index_col=0,
                     parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False)
    return df


def load_radiosonde(path):
    """
    Load a single TEAMx wEOP radiosonde file (ascent or descent).

    Parses the plain-text header to extract the launch/start time,
    then reads the comma-separated data rows.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    meta : dict  with keys 'launch_time' (datetime), 'kind' ('ascent'/'descent')
    data : DataFrame  with named columns, geopotential_height in metres
    """
    path = Path(path)

    # Determine ascent vs descent from filename
    kind = "ascent" if "ascent" in path.name else "descent"

    # Read header lines (those starting with spaces)
    header_lines = []
    data_lines = []
    col_names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("    "):
                header_lines.append(line.strip())
            else:
                data_lines.append(line)

    # Extract launch time from header
    launch_time = None
    time_key = "ascent start time" if kind == "ascent" else "descent start time"
    for h in header_lines:
        if time_key in h:
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", h)
            if m:
                launch_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                launch_time = launch_time.replace(tzinfo=timezone.utc)

    # Column names are the second-to-last header line
    col_name_line = [h for h in header_lines if "elapsed_time" in h]
    if col_name_line:
        col_names = [c.strip() for c in col_name_line[0].split(",")]

    # Parse data
    from io import StringIO
    data_str = "".join(data_lines)
    data = pd.read_csv(StringIO(data_str), header=None, names=col_names)

    # Sort by height (ascending for ascent, but descent arrives top-down)
    data = data.sort_values("geopotential_height").reset_index(drop=True)

    meta = {"launch_time": launch_time, "kind": kind}
    return meta, data


def load_all_radiosondes(sonde_paths, kinds=("ascent",)):
    """
    Load a list of radiosonde files and return a list of (meta, data) tuples,
    sorted by launch time.

    Descent profiles are excluded by default because their data begins above
    the HATPRO ceiling (~10 km) and therefore contribute no overlap.

    Parameters
    ----------
    sonde_paths : list of str or Path
    kinds       : tuple of str  which profile types to keep, e.g. ("ascent",)
                  or ("ascent", "descent") to include both

    Returns
    -------
    list of (meta dict, DataFrame)
    """
    sondes = [load_radiosonde(p) for p in sonde_paths]
    sondes = [(m, d) for m, d in sondes if m["kind"] in kinds]
    sondes.sort(key=lambda x: x[0]["launch_time"])
    return sondes


# ---------------------------------------------------------------------------
# Vertical interpolation
# ---------------------------------------------------------------------------

def interpolate_sonde_to_hatpro_levels(sonde_data, hatpro_heights_km,
                                        site_elev_m=HATPRO_SITE_ELEV_M):
    """
    Linearly interpolate radiosonde profiles onto HATPRO height levels.

    The sonde reports geopotential_height in metres above sea level.
    HATPRO heights are km above ground level.  We convert HATPRO levels to
    metres asl using the site elevation before interpolating.

    Parameters
    ----------
    sonde_data       : DataFrame  (sorted by geopotential_height)
    hatpro_heights_km: 1-D array  [km agl]
    site_elev_m      : float      instrument elevation [m asl]

    Returns
    -------
    dict  keys = sonde column names, values = array interpolated to HATPRO levels
          NaN where HATPRO level is outside the sonde range
    """
    hatpro_heights_m_asl = hatpro_heights_km * 1000.0 + site_elev_m
    sonde_z = sonde_data["geopotential_height"].values

    interp_vars = [
        "air_temperature",
        "air_potential_temperature",
        "relative_humidity",
        "humidity_mixing_ratio",
        "air_pressure",
    ]

    result = {"height_m_asl": hatpro_heights_m_asl}
    for var in interp_vars:
        if var not in sonde_data.columns:
            continue
        f = interp1d(sonde_z, sonde_data[var].values,
                     bounds_error=False, fill_value=np.nan)
        result[var] = f(hatpro_heights_m_asl)

    return result


def hatpro_abs_hum_to_rh(abs_hum_g_m3, temp_k):
    """
    Convert HATPRO absolute humidity [g/m³] to relative humidity [%]
    using only HATPRO temperature.

    Steps:
      1. Actual vapour pressure:  e = ρ * Rv * T        [Pa]
      2. Saturation vapour pressure via Clausius-Clapeyron with
         temperature-dependent latent heat (Koutsoyiannis 2012)
      3. RH = (e / esat) * 100

    Parameters
    ----------
    abs_hum_g_m3 : array  HATPRO absolute humidity [g/m³]
    temp_k       : array  HATPRO temperature [K]

    Returns
    -------
    array  relative humidity [%]
    """
    Rv    = 461.5        # specific gas constant for water vapour [J kg⁻¹ K⁻¹]
    esat0 = 611.7        # reference saturation vapour pressure [Pa] at T0
    T0    = 273.16       # reference temperature [K]

    # Step 1 — actual vapour pressure [Pa]
    rho_kg_m3 = abs_hum_g_m3 / 1000.0
    e = rho_kg_m3 * Rv * temp_k

    # Step 2 — saturation vapour pressure [Pa] (Koutsoyiannis 2012)
    Lv = 3.139e6 - 2336.0 * temp_k
    esat = esat0 * np.exp(-Lv / Rv * (1.0 / temp_k - 1.0 / T0))

    # Step 3 — relative humidity [%]
    rh = (e / esat) * 100.0
    return rh


def sonde_rh_to_abs_humidity(rh_percent, temp_k):
    """
    Convert sonde relative humidity [%] and temperature [K] to absolute
    humidity [g/m³] following Scheiber (2025) / Koutsoyiannis (2012).

    Steps
    -----
    1. Temperature-dependent latent heat:
           Lv = 3.139e6 − 2336 * T                          [J/kg]
    2. Saturation vapour pressure (Clausius-Clapeyron):
           esat = esat0 * exp(−Lv/Rv * (1/T − 1/T0))       [Pa]
           esat0 = 611.7 Pa, T0 = 273.16 K
    3. Actual vapour pressure:
           e = (RH / 100) * esat                            [Pa]
    4. Absolute humidity from ideal gas law (e = ρ * Rv * T):
           ρ = e / (Rv * T)                                 [kg/m³] → [g/m³]

    Parameters
    ----------
    rh_percent : array  relative humidity [%]
    temp_k     : array  air temperature [K]

    Returns
    -------
    array  absolute humidity [g/m³]
    """
    Rv    = 461.5    # specific gas constant for water vapour [J kg⁻¹ K⁻¹]
    esat0 = 611.7    # reference saturation vapour pressure [Pa] at T0
    T0    = 273.16   # reference temperature [K]

    Lv   = 3.139e6 - 2336.0 * temp_k
    esat = esat0 * np.exp(-Lv / Rv * (1.0 / temp_k - 1.0 / T0))
    e    = (rh_percent / 100.0) * esat
    rho  = e / (Rv * temp_k)     # [kg/m³]
    return rho * 1000.0           # → [g/m³]


# ---------------------------------------------------------------------------
# Temporal matching
# ---------------------------------------------------------------------------

def match_hatpro_to_sonde(hatpro_temp, hatpro_hum, sonde_launch_time,
                           window_min=MATCH_WINDOW_MIN):
    """
    Find the HATPRO profile closest in time to a sonde launch.

    Follows Scheiber (2025): prefer the first timestep within `window_min`
    minutes *after* the launch; fall back to the closest within `window_min`
    minutes *before* if no post-launch retrieval is available.

    Parameters
    ----------
    hatpro_temp      : DataFrame  (datetime index, 39 columns)
    hatpro_hum       : DataFrame  (datetime index, 39 columns)
    sonde_launch_time: datetime (timezone-aware UTC)
    window_min       : int

    Returns
    -------
    matched_time : Timestamp or None
    temp_profile : 1-D array [K] or None
    hum_profile  : 1-D array [g/m³] or None
    """
    # Make index timezone-naive for comparison (HATPRO timestamps are naive)
    launch_naive = sonde_launch_time.replace(tzinfo=None)
    delta = hatpro_temp.index - launch_naive          # TimedeltaIndex
    delta_min = delta.total_seconds() / 60.0

    window = pd.Timedelta(minutes=window_min)

    # First: within window_min minutes AFTER launch
    after_mask = (delta >= pd.Timedelta(0)) & (delta <= window)
    if after_mask.any():
        idx = delta[after_mask].argmin()
        matched_time = hatpro_temp.index[after_mask][idx]
    else:
        # Fallback: within window_min minutes BEFORE launch
        before_mask = (delta < pd.Timedelta(0)) & (delta >= -window)
        if before_mask.any():
            idx = (-delta[before_mask]).argmin()
            matched_time = hatpro_temp.index[before_mask][idx]
        else:
            return None, None, None

    temp_profile = hatpro_temp.loc[matched_time].values.astype(float)
    hum_profile  = hatpro_hum.loc[matched_time].values.astype(float)
    return matched_time, temp_profile, hum_profile


def build_comparison_table(sondes, hatpro_temp, hatpro_hum,
                            hatpro_heights_km=HATPRO_HEIGHTS_KM,
                            site_elev_m=HATPRO_SITE_ELEV_M,
                            window_min=MATCH_WINDOW_MIN):
    """
    For every sonde, match a HATPRO retrieval and interpolate the sonde
    onto HATPRO height levels.

    Parameters
    ----------
    sondes       : list of (meta, data) as returned by load_all_radiosondes
    hatpro_temp  : DataFrame
    hatpro_hum   : DataFrame
    hatpro_heights_km : array
    site_elev_m  : float
    window_min   : int

    Returns
    -------
    list of dicts, one per matched sonde, each containing:
        'launch_time'      datetime
        'kind'             'ascent' or 'descent'
        'hatpro_time'      Timestamp of matched HATPRO retrieval
        'heights_km'       array [km agl]
        'hatpro_temp'      array [K]
        'hatpro_hum'       array [g/m³]
        'sonde_temp'       array [K]  interpolated to HATPRO levels
        'sonde_abs_hum'    array [g/m³] absolute humidity from mixing ratio
        'sonde_rh'         array [%]  (retained for future use)
        'sonde_mixr'       array [g/kg] (retained for future use)
        'sonde_data_raw'   full sonde DataFrame (for future use)
    """
    results = []
    for meta, sonde_data in sondes:
        matched_time, temp_prof, hum_prof = match_hatpro_to_sonde(
            hatpro_temp, hatpro_hum, meta["launch_time"], window_min)

        if matched_time is None:
            print(f"  [!] No HATPRO match for {meta['launch_time']} ({meta['kind']})")
            continue

        interp = interpolate_sonde_to_hatpro_levels(
            sonde_data, hatpro_heights_km, site_elev_m)

        # Convert sonde RH + sonde temperature → absolute humidity [g/m³]
        # following Scheiber (2025) / Koutsoyiannis (2012)
        sonde_abs_hum = sonde_rh_to_abs_humidity(
            interp.get("relative_humidity"),
            interp.get("air_temperature"),
        )

        results.append({
            "launch_time":    meta["launch_time"],
            "kind":           meta["kind"],
            "hatpro_time":    matched_time,
            "heights_km":     hatpro_heights_km,
            "hatpro_temp":    temp_prof,
            "hatpro_hum":     hum_prof,
            "sonde_temp":     interp.get("air_temperature"),
            "sonde_abs_hum":  sonde_abs_hum,
            "sonde_rh":       interp.get("relative_humidity"),
            "sonde_mixr":     interp.get("humidity_mixing_ratio"),
            "sonde_data_raw": sonde_data,
        })
        print(f"  Matched {meta['kind']:7s} @ {meta['launch_time']} "
              f"-> HATPRO @ {matched_time}")
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_profile_comparison(comparison_table, max_height_km=10.0,
                             save_path=None):
    """
    Plot HATPRO vs radiosonde temperature and humidity profiles for every
    matched sonde in a grid layout.

    One column per sonde launch, two rows (temperature | humidity).

    Parameters
    ----------
    comparison_table : list of dicts (from build_comparison_table)
    max_height_km    : float  upper limit of y-axis [km agl]
    save_path        : str or Path or None  — if given, saves the figure
    """
    n = len(comparison_table)
    if n == 0:
        print("No matched sondes to plot.")
        return

    fig, axes = plt.subplots(
        2, n,
        figsize=(3.5 * n, 10),
        sharey=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, entry in enumerate(comparison_table):
        heights = entry["heights_km"]
        mask = heights <= max_height_km

        launch_dt = entry["launch_time"]
        title = f"{launch_dt.strftime('%H:%M')} UTC"

        # --- Temperature ---
        ax_t = axes[0, col]
        ax_t.plot(entry["hatpro_temp"][mask], heights[mask],
                  color="tab:red", linewidth=1.8, label="HATPRO")
        ax_t.plot(entry["sonde_temp"][mask], heights[mask],
                  color="tab:blue", linewidth=1.2, linestyle="--",
                  marker="o", markersize=3, markevery=3, label="Sonde")
        ax_t.set_title(title, fontsize=9)
        if col == 0:
            ax_t.set_ylabel("Height agl [km]", fontsize=9)
        ax_t.set_xlabel("Temperature [K]", fontsize=8)
        ax_t.tick_params(labelsize=8)
        ax_t.grid(True, alpha=0.3)
        if col == 0:
            ax_t.legend(fontsize=7, loc="upper right")

        # --- Absolute humidity ---
        ax_h = axes[1, col]
        ax_h.plot(entry["hatpro_hum"][mask], heights[mask],
                  color="tab:green", linewidth=1.8, label="HATPRO")
        ax_h.plot(entry["sonde_abs_hum"][mask], heights[mask],
                  color="tab:blue", linewidth=1.2, linestyle="--",
                  marker="o", markersize=3, markevery=3, label="Sonde")
        if col == 0:
            ax_h.set_ylabel("Height agl [km]", fontsize=9)
        ax_h.set_xlabel("Abs. humidity [g/m³]", fontsize=8)
        ax_h.tick_params(labelsize=8)
        ax_h.grid(True, alpha=0.3)
        if col == 0:
            ax_h.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"HATPRO vs Radiosonde — Kolsass, {comparison_table[0]['launch_time'].strftime('%Y-%m-%d')}",
        fontsize=12, fontweight="bold"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main — edit paths here to run interactively
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datestring = '20250219'

    # Make output directory if it doesn't exist
    output_dir = './plots/' + datestring + '/'
    os.makedirs(output_dir, exist_ok=True)

    # --- Paths (edit these) ---
    DATA_DIR  = Path("../compare_TROPoe_to_Massaro/data/" + datestring)          # folder containing the HATPRO CSVs
    SONDE_DIR = Path("../radiosonde_tools/sonde_data_" + datestring)          # folder containing the radiosonde CSVs

    assert os.path.isdir(DATA_DIR), f"Data folder not found: {DATA_DIR}"
    assert os.path.isdir(SONDE_DIR), f"Data folder not found: {SONDE_DIR}"

    hatpro_temp_path = DATA_DIR / "data_temperature.csv"
    hatpro_hum_path  = DATA_DIR / "data_humidity.csv"
    hatpro_met_path  = DATA_DIR / "data_met.csv"

    sonde_paths = sorted(SONDE_DIR.glob("raso_teamx_wEOP_kolsass_*.csv"))

    # --- Load ---
    print("Loading HATPRO data...")
    hatpro_temp, hatpro_hum = load_hatpro_profiles(hatpro_temp_path, hatpro_hum_path)
    hatpro_met = load_hatpro_met(hatpro_met_path)

    print(f"Loading {len(sonde_paths)} radiosonde files...")
    sondes = load_all_radiosondes(sonde_paths)

    # --- Match and interpolate ---
    print("Matching and interpolating...")
    comparison = build_comparison_table(sondes, hatpro_temp, hatpro_hum)

    # --- Visualise ---
    print("Plotting...")
    plot_profile_comparison(comparison, max_height_km=10.0,
                            save_path=output_dir + datestring + "_hatpro_raso_comparison.png")

    print('end')