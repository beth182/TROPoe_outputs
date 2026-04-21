import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
import argparse
import numpy as np
from datetime import datetime, timezone
import netCDF4 as nc


def load_tropoe(filepath):
    """Load key variables from a TROPoe output NetCDF file."""
    ds = nc.Dataset(filepath, 'r')

    # Time: reconstruct as datetime objects
    base = int(ds.variables['base_time'][:])
    offset = ds.variables['time_offset'][:]
    timestamps = [datetime.fromtimestamp(base + float(o), tz=timezone.utc)
                  for o in offset]
    hour = ds.variables['hour'][:]

    height = ds.variables['height'][:]  # km AGL
    temp = ds.variables['temperature'][:, :]  # degC (time x height)
    wv = ds.variables['waterVapor'][:, :]  # g/kg  (time x height)
    sigma_t = ds.variables['sigma_temperature'][:, :]
    sigma_wv = ds.variables['sigma_waterVapor'][:, :]
    lwp = ds.variables['lwp'][:]  # g/m2
    sigma_lwp = ds.variables['sigma_lwp'][:]
    qc = ds.variables['qc_flag'][:]

    ds.close()

    # Mask bad retrievals (qc_flag != 0)
    bad = qc != 0
    temp[bad, :] = np.nan
    wv[bad, :] = np.nan
    lwp[bad] = np.nan

    return dict(
        timestamps=timestamps,
        hour=hour,
        height=height,
        temp=temp,
        wv=wv,
        sigma_t=sigma_t,
        sigma_wv=sigma_wv,
        lwp=lwp,
        sigma_lwp=sigma_lwp,
        qc=qc,
    )

def find_nearest_time_idx(data, target_hour):
    """Return index of timestamp closest to target_hour (UTC)."""
    diffs = np.abs(data['hour'] - target_hour)
    return int(np.argmin(diffs))


def plot_time_height(data, out_prefix='tropoe'):
    """Time-height cross-sections of temperature and water vapour."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('TROPoe Retrieval — Time/Height Cross-sections', fontsize=13)

    times = mdates.date2num(data['timestamps'])
    H = data['height']

    # Temperature
    ax = axes[0]
    cf = ax.contourf(times, H, data['temp'].T,
                     levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='Temperature (°C)')
    ax.set_ylabel('Height AGL (km)')
    ax.set_ylim(0, 10)
    ax.set_title('Temperature')

    # Water vapour
    ax = axes[1]
    cf = ax.contourf(times, H, data['wv'].T,
                     levels=20, cmap='YlGnBu')
    plt.colorbar(cf, ax=ax, label='Water Vapour (g/kg)')
    ax.set_ylabel('Height AGL (km)')
    ax.set_ylim(0, 6)
    ax.set_title('Water Vapour Mixing Ratio')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlabel('Time (UTC)')
    fig.autofmt_xdate()

    plt.tight_layout()
    outfile = f'{out_prefix}_time_height.png'
    plt.savefig(outfile, dpi=150)
    print(f'Saved: {outfile}')
    plt.close()


def plot_profile_with_uncertainty(data, target_hour, out_prefix='tropoe'):
    """Single profile of T and q with 1-sigma uncertainty shading."""
    idx = find_nearest_time_idx(data, target_hour)
    t_str = data['timestamps'][idx].strftime('%Y-%m-%d %H:%M UTC')

    fig, axes = plt.subplots(1, 2, figsize=(8, 9), sharey=True)
    fig.suptitle(f'TROPoe Profile — {t_str}', fontsize=12)

    H = data['height']

    # Temperature profile
    ax = axes[0]
    T = data['temp'][idx, :]
    sT = data['sigma_t'][idx, :]
    ax.plot(T, H, 'b-', lw=2, label='Retrieved')
    ax.fill_betweenx(H, T - sT, T + sT, alpha=0.25, color='blue',
                     label='±1σ uncertainty')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Height AGL (km)')
    ax.set_ylim(0, 10)
    ax.set_title('Temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Water vapour profile
    ax = axes[1]
    WV = data['wv'][idx, :]
    sWV = data['sigma_wv'][idx, :]
    ax.plot(WV, H, 'g-', lw=2, label='Retrieved')
    ax.fill_betweenx(H, WV - sWV, WV + sWV, alpha=0.25, color='green',
                     label='±1σ uncertainty')
    ax.set_xlabel('Water Vapour (g/kg)')
    ax.set_ylim(0, 6)
    ax.set_title('Water Vapour')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f'{out_prefix}_profile_{idx:04d}.png'
    plt.savefig(outfile, dpi=150)
    print(f'Saved: {outfile}')
    plt.close()


def plot_lwp(data, out_prefix='tropoe'):
    """LWP time series with uncertainty."""
    fig, ax = plt.subplots(figsize=(12, 3))

    times = mdates.date2num(data['timestamps'])
    lwp = data['lwp']
    slwp = data['sigma_lwp']

    ax.plot(times, lwp, 'k-', lw=1, label='LWP')
    ax.fill_between(times, lwp - slwp, lwp + slwp,
                    alpha=0.3, color='steelblue', label='±1σ')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('LWP (g/m²)')
    ax.set_ylim(bottom=-20)
    ax.set_title('Liquid Water Path')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlabel('Time (UTC)')
    fig.autofmt_xdate()
    plt.tight_layout()

    outfile = f'{out_prefix}_lwp.png'
    plt.savefig(outfile, dpi=150)
    print(f'Saved: {outfile}')
    plt.close()


def plot_comparison(data_a, label_a, data_b, label_b,
                    target_hour, out_prefix='comparison'):
    """
    Side-by-side profile comparison between two TROPoe outputs,
    or a TROPoe output vs. manufacturer retrieval (if you load that too).
    target_hour: float, e.g. 12.0 for noon UTC
    """
    idx_a = find_nearest_time_idx(data_a, target_hour)
    idx_b = find_nearest_time_idx(data_b, target_hour)

    fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True)
    t_str = data_a['timestamps'][idx_a].strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f'Profile Comparison — {t_str}', fontsize=12)

    H_a = data_a['height']
    H_b = data_b['height']

    colours = {'a': 'steelblue', 'b': 'tomato'}

    for ax, var, sigma_var, xlabel, ylim in [
        (axes[0], 'temp', 'sigma_t', 'Temperature (°C)', (0, 10)),
        (axes[1], 'wv', 'sigma_wv', 'Water Vapour (g/kg)', (0, 6)),
    ]:
        Va = data_a[var][idx_a, :]
        sVa = data_a[sigma_var][idx_a, :]
        Vb = data_b[var][idx_b, :]
        sVb = data_b[sigma_var][idx_b, :]

        ax.plot(Va, H_a, color=colours['a'], lw=2, label=label_a)
        ax.fill_betweenx(H_a, Va - sVa, Va + sVa,
                         alpha=0.2, color=colours['a'])
        ax.plot(Vb, H_b, color=colours['b'], lw=2, label=label_b,
                ls='--')
        ax.fill_betweenx(H_b, Vb - sVb, Vb + sVb,
                         alpha=0.2, color=colours['b'])

        ax.set_xlabel(xlabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Height AGL (km)')
    plt.tight_layout()

    outfile = f'{out_prefix}_profile_compare.png'
    plt.savefig(outfile, dpi=150)
    print(f'Saved: {outfile}')
    plt.close()










# ── configuration — edit these ─────────────────────────────────────────────────

# Path to your TROPoe output file
FILE_A = 'C:/Users/c7071147/Documents/TROPoe_run/tropoe/hatpro/tropoe.c1.20251201.000015.nc'

# Path to a second file for comparison (set to None to skip)
FILE_B = 'I:/User/Documents/Research/Running_TROPoe/Download from Dave/beth_saunders/tropoe/hatpro/tropoe.c1.20251201.000015.nc'

# Labels for the comparison plot
LABEL_A = 'Beth'
LABEL_B = 'Dave'

# UTC hour for the single profile plot, e.g. 12.0 for noon
# Set to None to use the midpoint of the file automatically
PROFILE_TIME = 12.0

# Prefix for output plot filenames
OUT_PREFIX = 'tropoe'

# ── run ────────────────────────────────────────────────────────────────────────

print(f'Loading {FILE_A} ...')
data_a = load_tropoe(FILE_A)

plot_time_height(data_a, out_prefix=OUT_PREFIX)
plot_lwp(data_a, out_prefix=OUT_PREFIX)

t_profile = PROFILE_TIME
if t_profile is None:
    t_profile = float(data_a['hour'][len(data_a['hour']) // 2])
    print(f'Using midpoint time: {t_profile:.2f} UTC')

plot_profile_with_uncertainty(data_a, t_profile, out_prefix=OUT_PREFIX)

if FILE_B is not None:
    print(f'Loading {FILE_B} ...')
    data_b = load_tropoe(FILE_B)
    plot_comparison(data_a, LABEL_A, data_b, LABEL_B,
                    t_profile, out_prefix=OUT_PREFIX + '_vs')
    print('Comparison plot done.')

print('All done.')


# filepath_beth = 'C:/Users/c7071147/Documents/TROPoe_run/tropoe/hatpro/tropoe.c1.20251201.000015.nc'
# data_beth = load_tropoe(filepath_beth)
#
# filepath_dave = 'I:/User/Documents/Research/Running_TROPoe/Download from Dave/beth_saunders/tropoe/hatpro/tropoe.c1.20251201.000015.nc'
#
# data_dave = load_tropoe(filepath_beth)
#
# # plt.plot(data_beth['timestamps'], data_beth['temp'])
# # plt.show(block=False)




print('end')