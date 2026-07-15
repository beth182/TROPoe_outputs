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

  NOTE: these are season-wide files (one per EOP, not per day) -- see
  select_hatpro_window() usage in __main__ below.

HATPRO height grid (km above ground, 39 levels):
  see tropoe_shared.constants.HATPRO_HEIGHTS_KM

NOTE (refactor): HATPRO/TROPoe loading, RH<->absolute-humidity conversion,
sonde-to-instrument time matching, and vertical interpolation now come from
the shared `tropoe_shared` package. load_radiosonde/load_all_radiosondes and
the orchestration/plotting below (build_comparison_table, plot_profile_comparison,
plot_difference_profiles) stay here since they're specific to this 3-way
comparison.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TROPoe_outputs import lookup

from TROPoe_outputs.functions.constants import HATPRO_HEIGHTS_KM, HATPRO_SITE_ELEV_M, MATCH_WINDOW_MIN
from TROPoe_outputs.functions.humidity import humidity_rh_temp_to_abs, humidity_abs_to_rh
from TROPoe_outputs.functions.matching import match_profiles_to_time
from TROPoe_outputs.functions.interpolation import interpolate_profile_to_heights
from TROPoe_outputs.functions.hatpro_io import load_hatpro_profiles, load_hatpro_met, select_hatpro_window
from TROPoe_outputs.functions.tropoe_io import load_tropoe as _load_tropoe_full
from TROPoe_outputs.functions.sonde_io import load_radiosonde, load_all_radiosondes

# ---------------------------------------------------------------------------
# TROPoe loading adapter
# ---------------------------------------------------------------------------
# The shared load_tropoe() returns a dict of arrays (matching plot_tropoe.py's
# style). This script's matching/plotting code was written around pandas
# DataFrames indexed by timestamp instead (so it can reuse match_profiles_to_time
# the same way for both HATPRO and TROPoe). This adapter bridges the two
# without duplicating any loading logic.

def load_tropoe_as_frames(path):
    """
    Load a TROPoe file via the shared loader, then reshape into the
    DataFrame form this script's matching/plotting expects.

    Absolute humidity here uses humidity_rh_temp_to_abs() (TROPoe's own rh +
    temperature) — the Scheiber (2025)/Koutsoyiannis (2012) method, no
    cross-instrument data used. This matches the original HATPRO_raso_visu.py
    behaviour, and is a *different* method from compare_tropoe_hatpro.py's
    mixing-ratio-based conversion (see tropoe_shared/humidity.py docstring).

    Returns
    -------
    tropoe_temp : DataFrame  index=datetime (naive), columns=55 height levels [K]
    tropoe_hum  : DataFrame  index=datetime (naive), columns=55 height levels [g/m³]
    tropoe_heights_km : 1-D array [km agl]
    """
    data = _load_tropoe_full(path)
    naive_index = pd.DatetimeIndex(data['timestamps']).tz_localize(None)
    tropoe_temp = pd.DataFrame(data['temp_k'], index=naive_index)
    tropoe_hum = pd.DataFrame(data['abs_hum_from_rh'], index=naive_index)
    return tropoe_temp, tropoe_hum, data['height']


# ---------------------------------------------------------------------------
# Radiosonde loading
# ---------------------------------------------------------------------------
# load_radiosonde / load_all_radiosondes come from tropoe_shared.readers.sonde_io
# unchanged (this was the first script to need them).


# ---------------------------------------------------------------------------
# Vertical interpolation adapter
# ---------------------------------------------------------------------------

def interpolate_sonde_to_hatpro_levels(sonde_data, hatpro_heights_km,
                                        site_elev_m=HATPRO_SITE_ELEV_M):
    """
    Linearly interpolate radiosonde profiles onto HATPRO height levels.

    Thin wrapper around the shared interpolate_profile_to_heights(): does
    the sonde-specific unit conversion (geopotential height is m asl; HATPRO
    levels are km agl) and picks the variable set, then delegates the actual
    interpolation.

    Parameters
    ----------
    sonde_data       : DataFrame  (sorted by geopotential_height)
    hatpro_heights_km: 1-D array  [km agl]
    site_elev_m      : float      instrument elevation [m asl]

    Returns
    -------
    dict  keys = sonde column names (+ 'height_m_asl'), values interpolated
          to HATPRO levels; NaN where HATPRO level is outside sonde range
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
    source_vars = {v: sonde_data[v].values for v in interp_vars if v in sonde_data.columns}

    result = interpolate_profile_to_heights(sonde_z, source_vars, hatpro_heights_m_asl)
    result["height_m_asl"] = hatpro_heights_m_asl
    return result


# ---------------------------------------------------------------------------
# Comparison table (orchestration — stays script-specific)
# ---------------------------------------------------------------------------

def build_comparison_table(sondes, hatpro_temp, hatpro_hum,
                           tropoe_temp=None, tropoe_hum=None,
                           tropoe_heights_km=None,
                           hatpro_heights_km=HATPRO_HEIGHTS_KM,
                           site_elev_m=HATPRO_SITE_ELEV_M,
                           window_min=MATCH_WINDOW_MIN):
    """
    For every sonde, match a HATPRO retrieval (and optionally a TROPoe
    retrieval) and interpolate the sonde onto HATPRO height levels.

    Parameters
    ----------
    sondes            : list of (meta, data) as returned by load_all_radiosondes
    hatpro_temp       : DataFrame
    hatpro_hum        : DataFrame
    tropoe_temp       : DataFrame or None
    tropoe_hum        : DataFrame or None
    tropoe_heights_km : 1-D array or None
    hatpro_heights_km : array
    site_elev_m       : float
    window_min        : int

    Returns
    -------
    list of dicts, one per matched sonde (see original docstring for full
    key list — unchanged from before the refactor).
    """
    results = []
    for meta, sonde_data in sondes:
        matched_time, hatpro_profiles = match_profiles_to_time(
            meta["launch_time"], window_min=window_min,
            temp=hatpro_temp, hum=hatpro_hum)

        if matched_time is None:
            print(f"  [!] No HATPRO match for {meta['launch_time']} ({meta['kind']})")
            continue

        temp_prof = hatpro_profiles['temp']
        hum_prof = hatpro_profiles['hum']

        interp = interpolate_sonde_to_hatpro_levels(
            sonde_data, hatpro_heights_km, site_elev_m)

        # Convert HATPRO absolute humidity + HATPRO temperature -> RH
        hatpro_rh = humidity_abs_to_rh(hum_prof, temp_prof)

        # Convert sonde RH + sonde temperature -> absolute humidity [g/m3]
        sonde_abs_hum = humidity_rh_temp_to_abs(
            interp.get("relative_humidity"),
            interp.get("air_temperature"),
        )

        # --- TROPoe match (optional) ---
        t_time, t_temp, t_hum = None, None, None
        if tropoe_temp is not None:
            t_time, tropoe_profiles = match_profiles_to_time(
                meta["launch_time"], window_min=window_min,
                temp=tropoe_temp, hum=tropoe_hum)
            if t_time is None:
                print(f"  [!] No TROPoe match for {meta['launch_time']}")
            else:
                t_temp, t_hum = tropoe_profiles['temp'], tropoe_profiles['hum']

        results.append({
            "launch_time": meta["launch_time"],
            "kind": meta["kind"],
            "hatpro_time": matched_time,
            "heights_km": hatpro_heights_km,
            "hatpro_temp": temp_prof,
            "hatpro_hum": hum_prof,
            "hatpro_rh": hatpro_rh,
            "tropoe_time": t_time,
            "tropoe_temp": t_temp,
            "tropoe_hum": t_hum,
            "tropoe_heights_km": tropoe_heights_km,
            "sonde_temp": interp.get("air_temperature"),
            "sonde_abs_hum": sonde_abs_hum,
            "sonde_rh": interp.get("relative_humidity"),
            "sonde_mixr": interp.get("humidity_mixing_ratio"),
            "sonde_data_raw": sonde_data,
        })
        print(f"  Matched {meta['kind']:7s} @ {meta['launch_time']} "
              f"-> HATPRO @ {matched_time}"
              + (f", TROPoe @ {t_time}" if t_time is not None else ", TROPoe: no match"))
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_profile_comparison(comparison_table, max_height_km=10.0,
                             save_path=None):
    """
    Plot HATPRO, TROPoe, and radiosonde temperature and humidity profiles
    for every matched sonde. One column per launch, two rows (T | q).

    Each dataset is plotted on its own native height grid — no interpolation
    applied. Small markers show individual data points on each grid.

    HATPRO : 39 levels, plotted on HATPRO grid
    TROPoe : 55 levels (49 below 10 km), plotted on TROPoe grid
    Sonde  : native high-resolution ascent profile

    Colours: HATPRO = red, TROPoe = green, Sonde = black

    Parameters
    ----------
    comparison_table : list of dicts (from build_comparison_table)
    max_height_km    : float
    save_path        : str or Path or None
    """
    n = len(comparison_table)
    if n == 0:
        print("No matched sondes to plot.")
        return

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 10),
                              sharey=True, constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    plot_kwargs = dict(linewidth=1.2, markersize=3)

    for col, entry in enumerate(comparison_table):

        # --- HATPRO on its native grid ---
        h_heights = entry["heights_km"]
        h_mask    = h_heights <= max_height_km

        # --- TROPoe on its native grid ---
        if entry["tropoe_temp"] is not None:
            t_heights = entry["tropoe_heights_km"]
            t_mask    = t_heights <= max_height_km

        # --- Sonde on its native high-resolution grid ---
        sonde_raw    = entry["sonde_data_raw"]
        sonde_z_m    = sonde_raw["geopotential_height"].values
        sonde_z_km   = (sonde_z_m - HATPRO_SITE_ELEV_M) / 1000.0
        sonde_mask   = (sonde_z_km >= 0) & (sonde_z_km <= max_height_km)

        sonde_temp   = sonde_raw["air_temperature"].values
        sonde_abshum = humidity_rh_temp_to_abs(
            sonde_raw["relative_humidity"].values,
            sonde_raw["air_temperature"].values,
        )

        title = entry["launch_time"].strftime("%H:%M") + " UTC"

        # --- Temperature ---
        ax_t = axes[0, col]
        ax_t.plot(entry["hatpro_temp"][h_mask], h_heights[h_mask],
                  color="tab:red", marker="o", label="HATPRO", **plot_kwargs)
        ax_t.plot(sonde_temp[sonde_mask], sonde_z_km[sonde_mask],
                  color="black", marker=".", markersize=2, linewidth=1.0,
                  label="Sonde")
        if entry["tropoe_temp"] is not None:
            ax_t.plot(entry["tropoe_temp"][t_mask], t_heights[t_mask],
                      color="tab:green", marker="^", markersize=3,
                      linewidth=1.2, label="TROPoe")
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
        ax_h.plot(entry["hatpro_hum"][h_mask], h_heights[h_mask],
                  color="tab:red", marker="o", label="HATPRO", **plot_kwargs)
        ax_h.plot(sonde_abshum[sonde_mask], sonde_z_km[sonde_mask],
                  color="black", marker=".", markersize=2, linewidth=1.0,
                  label="Sonde")
        if entry["tropoe_hum"] is not None:
            ax_h.plot(entry["tropoe_hum"][t_mask], t_heights[t_mask],
                      color="tab:green", marker="^", markersize=3,
                      linewidth=1.2, label="TROPoe")
        if col == 0:
            ax_h.set_ylabel("Height agl [km]", fontsize=9)
        ax_h.set_xlabel("Abs. humidity [g/m³]", fontsize=8)
        ax_h.tick_params(labelsize=8)
        ax_h.grid(True, alpha=0.3)
        if col == 0:
            ax_h.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        comparison_table[0]['launch_time'].strftime("%Y-%m-%d"),
        fontsize=12, fontweight="bold"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Visualisation — difference profiles
# ---------------------------------------------------------------------------

def plot_difference_profiles(comparison_table, max_height_km=10.0,
                              save_path=None):
    """
    Plot sonde-minus-instrument difference profiles for temperature and
    absolute humidity.

    Layout: two panels side by side (temperature | humidity).
    Individual launches are coloured by time of day (plasma colormap);
    bold black line is the mean across all launches.
    A dashed vertical grey line marks zero.

    HATPRO differences in solid lines, TROPoe in dashed lines.

    For HATPRO: sonde is already on the HATPRO grid from build_comparison_table.
    For TROPoe: sonde is re-interpolated onto the TROPoe native grid here.

    Parameters
    ----------
    comparison_table : list of dicts (from build_comparison_table)
    max_height_km    : float
    save_path        : str or Path or None
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, (ax_t, ax_h) = plt.subplots(
        1, 2, figsize=(10, 7), sharey=True, constrained_layout=True
    )

    cmap      = cm.plasma
    norm      = mcolors.Normalize(vmin=0, vmax=24)
    launch_hours = [e["launch_time"].hour + e["launch_time"].minute / 60.0
                    for e in comparison_table]

    hatpro_temp_diffs   = []
    hatpro_hum_diffs    = []
    tropoe_temp_diffs   = []
    tropoe_hum_diffs    = []
    tropoe_heights_plot = None

    for entry, hour in zip(comparison_table, launch_hours):
        colour = cmap(norm(hour))

        # --- HATPRO differences (sonde already on HATPRO grid) ---
        h_heights = entry["heights_km"]
        h_mask    = h_heights <= max_height_km

        diff_t = entry["sonde_temp"][h_mask]    - entry["hatpro_temp"][h_mask]
        diff_h = entry["sonde_abs_hum"][h_mask] - entry["hatpro_hum"][h_mask]

        label = entry["launch_time"].strftime("%H:%M") + " UTC"
        ax_t.plot(diff_t, h_heights[h_mask], color=colour, linewidth=1.2,
                  alpha=0.85, linestyle="-", label=label)
        ax_h.plot(diff_h, h_heights[h_mask], color=colour, linewidth=1.2,
                  alpha=0.85, linestyle="-")

        hatpro_temp_diffs.append(diff_t)
        hatpro_hum_diffs.append(diff_h)

        # --- TROPoe differences (re-interpolate sonde onto TROPoe grid) ---
        if entry["tropoe_temp"] is not None:
            t_heights_full  = entry["tropoe_heights_km"]
            t_mask          = t_heights_full <= max_height_km
            t_heights       = t_heights_full[t_mask]
            t_heights_m_asl = t_heights * 1000.0 + HATPRO_SITE_ELEV_M

            sonde_raw = entry["sonde_data_raw"]
            sonde_z   = sonde_raw["geopotential_height"].values

            f_temp = interp1d(sonde_z, sonde_raw["air_temperature"].values,
                              bounds_error=False, fill_value=np.nan)
            f_rh   = interp1d(sonde_z, sonde_raw["relative_humidity"].values,
                              bounds_error=False, fill_value=np.nan)

            sonde_temp_on_t   = f_temp(t_heights_m_asl)
            sonde_rh_on_t     = f_rh(t_heights_m_asl)
            sonde_abshum_on_t = humidity_rh_temp_to_abs(sonde_rh_on_t, sonde_temp_on_t)

            diff_t_trop = sonde_temp_on_t   - entry["tropoe_temp"][t_mask]
            diff_h_trop = sonde_abshum_on_t - entry["tropoe_hum"][t_mask]

            ax_t.plot(diff_t_trop, t_heights, color=colour, linewidth=1.2,
                      alpha=0.85, linestyle="--")
            ax_h.plot(diff_h_trop, t_heights, color=colour, linewidth=1.2,
                      alpha=0.85, linestyle="--")

            tropoe_temp_diffs.append(diff_t_trop)
            tropoe_hum_diffs.append(diff_h_trop)
            tropoe_heights_plot = t_heights

    # --- Mean lines ---
    h_heights_plot = entry["heights_km"][entry["heights_km"] <= max_height_km]

    ax_t.plot(np.nanmean(hatpro_temp_diffs, axis=0), h_heights_plot,
              color="black", linewidth=2.5, linestyle="-",  label="HATPRO mean")
    ax_h.plot(np.nanmean(hatpro_hum_diffs,  axis=0), h_heights_plot,
              color="black", linewidth=2.5, linestyle="-")

    if tropoe_temp_diffs:
        ax_t.plot(np.nanmean(tropoe_temp_diffs, axis=0), tropoe_heights_plot,
                  color="black", linewidth=2.5, linestyle="--", label="TROPoe mean")
        ax_h.plot(np.nanmean(tropoe_hum_diffs,  axis=0), tropoe_heights_plot,
                  color="black", linewidth=2.5, linestyle="--")

    # --- Colorbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_t, ax_h], orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("Launch time [UTC hour]", fontsize=9)
    cbar.set_ticks(range(0, 25, 3))

    # --- Legend (temperature panel only — solid=HATPRO, dashed=TROPoe) ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="-",  label="HATPRO"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--", label="TROPoe"),
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="-", label="Mean"),
    ]
    ax_t.legend(handles=legend_elements, fontsize=8, loc="upper right")

    # --- Formatting ---
    for ax in (ax_t, ax_h):
        ax.axvline(0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.set_ylabel("Height agl [km]", fontsize=10)
        ax.set_ylim(0, max_height_km)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    ax_t.set_xlabel("Sonde − Instrument [K]", fontsize=10)
    ax_h.set_xlabel("Sonde − Instrument [g/m³]", fontsize=10)
    ax_t.set_title("Temperature difference", fontsize=10)
    ax_h.set_title("Abs. humidity difference", fontsize=10)

    fig.suptitle(
        f"{comparison_table[0]['launch_time'].strftime('%Y-%m-%d')}",
        fontsize=11, fontweight="bold"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main — loop over every date in the date-list CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    date_list_csv_path = lookup.date_list_location
    df = pd.read_csv(date_list_csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    datestrings = df['datetime'].dt.strftime('%Y%m%d').unique().tolist()

    _DATA = lookup.data_location
    assert os.path.isdir(_DATA), f"Data folder not found: {_DATA}"

    skipped_dates = []

    for datestring in datestrings:

        print(datestring)

        output_dir = lookup.plot_save_location + 'HATPRO_TROPoe_raso_comparison/TOC/' + datestring + '/'

        # condition based on the month for sEOP or wEOP
        dt = datetime.strptime(datestring, '%Y%m%d')
        month = dt.month
        if month < 3:
            assert 1 <= month <= 2
            EOP = 'wEOP'
        else:
            assert 6 <= month <= 7
            EOP = 'sEOP'

        # --- HATPRO paths: season-wide file, same layout as compare_tropoe_hatpro.py ---
        T_CSV = os.path.join(_DATA, 'HATPRO_processed_Massaro/TOC/', EOP + "_temperature.csv")
        Q_CSV = os.path.join(_DATA, 'HATPRO_processed_Massaro/TOC/', EOP + "_humidity.csv")
        MET_CSV = os.path.join(_DATA, 'HATPRO_processed_Massaro/TOC/', EOP + "_met.csv")

        if not (os.path.isfile(T_CSV) and os.path.isfile(Q_CSV) and os.path.isfile(MET_CSV)):
            print(f'  [!] HATPRO CSV(s) not found for {EOP}, skipping {datestring}')
            skipped_dates.append((datestring, f'missing HATPRO CSV for {EOP}'))
            continue

        # --- TROPoe
        FILE_PATTERN = os.path.join(_DATA + 'TROPoe_output/TOC/' + datestring,
                                    "tropoe_innsbruck.c1." + datestring + ".*.nc")
        matches = glob.glob(FILE_PATTERN)
        if len(matches) != 1:
            print(f'  [!] Expected exactly 1 TROPoe file for {datestring}, found {len(matches)}, skipping')
            skipped_dates.append((datestring, f'{len(matches)} TROPoe file matches (expected 1)'))
            continue
        tropoe_path = matches[0]

        # --- Sonde
        SONDE_DIR = Path(lookup.data_location + 'radiosonde_processed_csv_data_TEAMx/')
        if not os.path.isdir(SONDE_DIR):
            print(f'  [!] Sonde folder not found, skipping {datestring}: {SONDE_DIR}')
            skipped_dates.append((datestring, 'sonde folder not found'))
            continue

        sonde_paths = sorted(SONDE_DIR.glob(f"raso_teamx_{EOP}_kolsass_{datestring}*.csv"))
        if not sonde_paths:
            print(f'  [!] No radiosonde files found for {datestring}, skipping')
            skipped_dates.append((datestring, 'no radiosonde files found'))
            continue

        # All checks passed -- only now create the output directory.
        os.makedirs(output_dir, exist_ok=True)

        try:
            # --- Load ---
            print("Loading HATPRO data...")
            hatpro_temp_full, hatpro_hum_full = load_hatpro_profiles(T_CSV, Q_CSV)
            hatpro_met = load_hatpro_met(MET_CSV)  # loaded for completeness; not plotted below

            # T_CSV/Q_CSV cover the whole EOP season, not just this day -- narrow to
            # the target date and align temp/hum onto matching timestamps.
            windowed = select_hatpro_window(datestring, temp=hatpro_temp_full, hum=hatpro_hum_full)
            hatpro_temp, hatpro_hum = windowed['temp'], windowed['hum']

            print("Loading TROPoe data...")
            tropoe_temp, tropoe_hum, tropoe_heights_km = load_tropoe_as_frames(tropoe_path)

            print(f"Loading {len(sonde_paths)} radiosonde files...")
            sondes = load_all_radiosondes(sonde_paths)

            # --- Match and interpolate ---
            print("Matching and interpolating...")
            comparison = build_comparison_table(
                sondes, hatpro_temp, hatpro_hum,
                tropoe_temp=tropoe_temp, tropoe_hum=tropoe_hum,
                tropoe_heights_km=tropoe_heights_km,
            )

            # --- Visualise ---
            print("Plotting...")
            plot_profile_comparison(comparison, max_height_km=10.0,
                                    save_path=output_dir + datestring + "_hatpro_raso_comparison.png")

            print("Plotting differences...")
            plot_difference_profiles(comparison, max_height_km=10.0,
                                     save_path=output_dir + datestring + "_hatpro_raso_differences.png")

        except Exception as e:
            print(f'  [!] Error processing {datestring}, skipping: {e}')
            skipped_dates.append((datestring, str(e)))
            continue

        print('end')

    print('All done.')

    if skipped_dates:
        print(f'\n{len(skipped_dates)} date(s) skipped:')
        for d, reason in skipped_dates:
            print(f'  - {d}: {reason}')
    else:
        print('\nNo dates skipped -- every date processed successfully.')