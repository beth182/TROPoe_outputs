"""
plot_tropoe_single.py
----------------------
Single-file TROPoe plotting. No comparison, no second file.

Produces:
  <prefix>_time_height.png          — T and WV time/height cross-sections
  <prefix>_lwp.png                  — LWP time series with uncertainty
  <prefix>_profile_NNNN.png         — Single T/WV profile with uncertainty

Requirements:
    pip install netCDF4 matplotlib numpy scipy

Split out of plot_tropoe.py: this file only ever loads one TROPoe file, so
the A-vs-B comparison/diff logic (and its own output location) now lives in
plot_tropoe_comparison.py instead.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob

# --- shared module import -----------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TROPoe_outputs import lookup
from TROPoe_outputs.functions.constants import H_MAX_T, H_MAX_WV
from TROPoe_outputs.functions.tropoe_io import load_tropoe

print('Imports complete')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS (script-specific: axis formatting, saving, single-index lookup)
# ══════════════════════════════════════════════════════════════════════════════

def nearest_idx(data, target_hour):
    return int(np.argmin(np.abs(data['hour'] - target_hour)))


def fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlabel('Time (UTC)')


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FILE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_height(data, out_prefix):
    """
    Produces *_time_height.png
    Two colour-filled contour plots stacked vertically, sharing a time axis. The top panel shows temperature in °C and
    the bottom shows water vapour mixing ratio in g/kg, both as a function of time (x-axis) and height above ground
    (y-axis). This is the standard way to visualise a day's worth of profiling data — you can immediately see things
    like the diurnal temperature cycle, the growth of the boundary layer, and where moisture is concentrated.
    The colourmaps are chosen deliberately: RdBu_r for temperature (red = warm, blue = cold) and YlGnBu for moisture.
    :param data:
    :param out_prefix: *
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('TROPoe — Time/Height Cross-sections', fontsize=13)
    times = mdates.date2num(data['timestamps'])
    H = data['height']

    cf = axes[0].contourf(times, H, data['temp'].T, levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=axes[0], label='Temperature (°C)')
    axes[0].set_ylabel('Height AGL (km)');
    axes[0].set_ylim(0, H_MAX_T)
    axes[0].set_title('Temperature')

    cf = axes[1].contourf(times, H, data['wv'].T, levels=20, cmap='YlGnBu')
    plt.colorbar(cf, ax=axes[1], label='Water Vapour (g/kg)')
    axes[1].set_ylabel('Height AGL (km)');
    axes[1].set_ylim(0, H_MAX_WV)
    axes[1].set_title('Water Vapour Mixing Ratio')

    fmt_xaxis(axes[1]);
    fig.autofmt_xdate();
    plt.tight_layout()
    _save(fig, f'{out_prefix}_time_height.png')


def plot_lwp(data, out_prefix):
    """
    Produces *_lwp.png
    A simple time series of liquid water path (g/m²) with ±1σ uncertainty shading. LWP tells you whether there's cloud
    liquid water overhead. Values near zero mean clear sky; positive values indicate cloud. The uncertainty band comes
    directly from TROPoe's posterior covariance — one of the things TROPoe does that the manufacturer retrieval doesn't.
    :param data:
    :param out_prefix: *
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    times = mdates.date2num(data['timestamps'])
    ax.plot(times, data['lwp'], 'k-', lw=1, label='LWP')
    ax.fill_between(times, data['lwp'] - data['sigma_lwp'],
                    data['lwp'] + data['sigma_lwp'],
                    alpha=0.3, color='steelblue', label='±1σ')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('LWP (g/m²)');
    ax.set_ylim(bottom=-20)
    ax.set_title('Liquid Water Path');
    ax.legend(fontsize=8)
    fmt_xaxis(ax);
    fig.autofmt_xdate();
    plt.tight_layout()
    _save(fig, f'{out_prefix}_lwp.png')


def plot_profile_with_uncertainty(data, target_hour, out_prefix):
    """
    Produces *_profile_NNNN.png
    Two side-by-side vertical profiles (temperature left, water vapour right) at a single moment in time — whichever
    hour you set in PROFILE_TIME. The solid line is the retrieved profile and the shading is ±1σ uncertainty.
    The four-digit number in the filename is the time index. This is useful for a quick sanity check and for
    illustrating what the retrieval looks like at a particular moment.
    :param data:
    :param target_hour: NNNN
    :param out_prefix: *
    :return:
    """
    idx = nearest_idx(data, target_hour)
    t_str = data['timestamps'][idx].strftime('%Y-%m-%d %H:%M UTC')
    H = data['height']
    fig, axes = plt.subplots(1, 2, figsize=(8, 9), sharey=True)
    fig.suptitle(f'TROPoe Profile — {t_str}', fontsize=12)
    for ax, var, sig, xlabel, hmax, col in [
        (axes[0], 'temp', 'sigma_t', 'Temperature (°C)', H_MAX_T, 'steelblue'),
        (axes[1], 'wv', 'sigma_wv', 'Water Vapour (g/kg)', H_MAX_WV, 'seagreen'),
    ]:
        V, S = data[var][idx, :], data[sig][idx, :]
        ax.plot(V, H, color=col, lw=2, label='Retrieved')
        ax.fill_betweenx(H, V - S, V + S, alpha=0.25, color=col, label='±1σ')
        ax.set_xlabel(xlabel);
        ax.set_ylabel('Height AGL (km)')
        ax.set_ylim(0, hmax);
        ax.legend(fontsize=8);
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f'{out_prefix}_profile_{idx:04d}.png')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ══════════════════════════════════════════════════════════════════════════════

date_list_csv_path = lookup.date_list_location
df = pd.read_csv(date_list_csv_path)

df['datetime'] = pd.to_datetime(df['datetime'])
datestrings = df['datetime'].dt.strftime('%Y%m%d').unique().tolist()

skipped_dates = []

for datestring in datestrings:

    print(datestring)

    # Path to your TROPoe output file
    # ToDo: change this to data location once the data from TROPoe runs has been moved to central location
    FILE_PATTERN = 'C:/Users/c7071147/Documents/TROPoe_run/TOC/' + datestring + '/tropoe/hatpro/tropoe_innsbruck.c1.' + datestring + '*'
    matches = glob.glob(FILE_PATTERN)

    if not matches:  # or os.path.isfile(FILE), depending on where you landed with the glob fix
        print(f'  [!] File not found, skipping: {FILE_PATTERN}')
        skipped_dates.append((datestring, 'file not found'))
        continue

    assert len(matches) == 1
    FILE = matches[0]

    PROFILE_TIME = 12.0  # UTC hour for single-profile plot; None = midpoint of file

    # Make output directory if it doesn't exist
    outdir = lookup.plot_save_location + 'TROPoe_output/TOC/' + datestring + '/'
    os.makedirs(outdir, exist_ok=True)

    OUT_PREFIX = outdir + datestring + '_tropoe'

    # ══════════════════════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════════════════════

    try:
        da = load_tropoe(FILE)

        t_profile = PROFILE_TIME
        if t_profile is None:
            t_profile = float(da['hour'][len(da['hour']) // 2])
            print(f'Using midpoint time: {t_profile:.2f} UTC')

        plot_time_height(da, OUT_PREFIX)
        plot_lwp(da, OUT_PREFIX)
        plot_profile_with_uncertainty(da, t_profile, OUT_PREFIX)

    except Exception as e:
        print(f'  [!] Error processing {datestring}, skipping: {e}')
        skipped_dates.append((datestring, str(e)))
        continue


if skipped_dates:
    print(f'\n{len(skipped_dates)} date(s) skipped:')
    for d, reason in skipped_dates:
        print(f'  - {d}: {reason}')
else:
    print('\nNo dates skipped -- a file was found and processed for every date.')

print('All done.')
print('end')