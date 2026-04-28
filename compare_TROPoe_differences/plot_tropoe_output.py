"""
plot_tropoe.py
--------------
All TROPoe plotting in one script.

Produces (when FILE_B is set):
  <prefix>_time_height.png          — T and WV time/height cross-sections (File A)
  <prefix>_lwp.png                  — LWP time series with uncertainty (File A)
  <prefix>_profile_NNNN.png         — Single T/WV profile with uncertainty (File A)
  <prefix>_compare_profiles.png     — 3x3 profile subplots, A vs B
  <prefix>_compare_timeseries.png   — All 1D variables stacked, A vs B
  <prefix>_diff_timehgt.png         — Time/height difference maps (A minus B)
  <prefix>_diff_timeseries.png      — 1D difference time series (A minus B)
  <prefix>_diff_mean_profile.png    — Mean difference profile with spread (A minus B)

Requirements:
    pip install netCDF4 matplotlib numpy scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.interpolate import interp1d
import netCDF4 as nc

# TODO: move out all single visualisation functions to separate place to be used again
# TODO: Change save plot function locations to plot directory

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get(ds, name):
    if name not in ds.variables:
        return None
    v = ds.variables[name][:]
    v = np.ma.filled(v, np.nan) if hasattr(v, 'filled') else np.array(v)
    return v.astype(float)


def load_tropoe(filepath):
    ds = nc.Dataset(filepath, 'r')
    base = int(ds.variables['base_time'][:])
    offset = np.array(ds.variables['time_offset'][:], dtype=float)
    hour = np.array(ds.variables['hour'][:], dtype=float)
    height = np.array(ds.variables['height'][:], dtype=float)
    qc = np.array(ds.variables['qc_flag'][:], dtype=float)
    timestamps = [datetime.fromtimestamp(base + o, tz=timezone.utc) for o in offset]
    bad = qc != 0

    def g2d(name):
        v = _get(ds, name)
        if v is not None:
            v = v.copy();
            v[bad, :] = np.nan
        return v

    def g1d(name):
        v = _get(ds, name)
        if v is not None:
            v = v.copy();
            v[bad] = np.nan
        return v

    data = dict(
        filepath=filepath, timestamps=timestamps, hour=hour, height=height, qc=qc,
        temp=g2d('temperature'), wv=g2d('waterVapor'),
        sigma_t=g2d('sigma_temperature'), sigma_wv=g2d('sigma_waterVapor'),
        rh=g2d('rh'), dewpt=g2d('dewpt'),
        theta=g2d('theta'), thetae=g2d('thetae'),
        vres_t=g2d('vres_temperature'), vres_wv=g2d('vres_waterVapor'),
        cdfs_t=g2d('cdfs_temperature'), cdfs_wv=g2d('cdfs_waterVapor'),
        lwp=g1d('lwp'), sigma_lwp=g1d('sigma_lwp'),
        pwv=g1d('pwv'), pblh=g1d('pblh'),
        sbih=g1d('sbih'), sbim=g1d('sbim'),
        sbLCL=g1d('sbLCL'), sbCAPE=g1d('sbCAPE'), sbCIN=g1d('sbCIN'),
        mlCAPE=g1d('mlCAPE'), mlCIN=g1d('mlCIN'),
        cbh=g1d('cbh'), rmsa=g1d('rmsa'),
        rmsp=g1d('rmsp'), sic=g1d('sic'),
    )
    ds.close()
    return data


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


def _interp_time(src_hour, src_data, tgt_hour):
    if src_data is None:
        return None
    if src_data.ndim == 1:
        return interp1d(src_hour, src_data, bounds_error=False, fill_value=np.nan)(tgt_hour)
    out = np.full((len(tgt_hour), src_data.shape[1]), np.nan)
    for j in range(src_data.shape[1]):
        out[:, j] = interp1d(src_hour, src_data[:, j],
                             bounds_error=False, fill_value=np.nan)(tgt_hour)
    return out


def _interp_height(data_2d, h_from, h_to):
    if data_2d is None:
        return None
    out = np.full((data_2d.shape[0], len(h_to)), np.nan)
    for i in range(data_2d.shape[0]):
        row = data_2d[i, :]
        if not np.all(np.isnan(row)):
            out[i, :] = interp1d(h_from, row, bounds_error=False,
                                 fill_value=np.nan)(h_to)
    return out


def align_to(da, db):
    """Re-grid db onto da's time and height grids for differencing."""
    ha, hb = da['height'], db['height']
    skip = {'timestamps', 'hour', 'height', 'filepath', 'qc'}
    aligned = {k: da[k] if k in ('timestamps', 'hour', 'height') else db[k]
               for k in skip}
    for key, val in db.items():
        if key in skip or val is None:
            aligned.setdefault(key, None)
            continue
        v = _interp_time(db['hour'], val, da['hour'])
        if v is not None and v.ndim == 2 and not np.allclose(ha, hb, atol=0.001):
            v = _interp_height(v, hb, ha)
        aligned[key] = v
    return aligned


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
# COMPARISON PLOTS (A vs B)
# ══════════════════════════════════════════════════════════════════════════════

def plot_compare_profiles(da, db, label_a, label_b, target_hour, out_prefix):
    """
    Produces *_compare_profiles.png
    A 3×3 grid of profile panels, both datasets overlaid in the same axes at your chosen PROFILE_TIME. The solid line
    is always File A, dashed is File B, with uncertainty shading where available. The nine panels cover: temperature,
    water vapour, relative humidity, dew point, potential temperature, equivalent potential temperature, vertical
    resolution of T, vertical resolution of WV, and cumulative degrees of freedom for signal in temperature. The last
    two rows are particularly interesting scientifically — vertical resolution tells you how sharp the retrieval
    actually is at each level (the retrieval grid and the true resolution are not the same thing), and cumulative
    DFS tells you how much of the information content in the profiles comes from below each height level.
    :param da:
    :param db:
    :param label_a:
    :param label_b:
    :param target_hour:
    :param out_prefix: *
    :return:
    """
    ia = nearest_idx(da, target_hour)
    ib = nearest_idx(db, target_hour)
    t_str = da['timestamps'][ia].strftime('%Y-%m-%d %H:%M UTC')
    Ha, Hb = da['height'], db['height']

    panels = [
        (0, 0, 'temp', 'sigma_t', 'Temperature (°C)', H_MAX_T),
        (0, 1, 'wv', 'sigma_wv', 'Water Vapour (g/kg)', H_MAX_WV),
        (0, 2, 'rh', None, 'Relative Humidity (%)', H_MAX_WV),
        (1, 0, 'dewpt', None, 'Dew Point (°C)', H_MAX_WV),
        (1, 1, 'theta', None, 'Potential Temp (K)', H_MAX_T),
        (1, 2, 'thetae', None, 'Equiv. Potential Temp (K)', H_MAX_WV),
        (2, 0, 'vres_t', None, 'Vert. Res. — Temp (km)', H_MAX_T),
        (2, 1, 'vres_wv', None, 'Vert. Res. — WV (km)', H_MAX_WV),
        (2, 2, 'cdfs_t', None, 'Cumul. DFS — Temp', H_MAX_T),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 16))
    fig.suptitle(f'Profile Comparison — {t_str}\n'
                 f'{label_a} (solid) vs {label_b} (dashed)', fontsize=12)

    for row, col, var, sig, xlabel, hmax in panels:
        ax = axes[row, col]
        Va, Vb = da.get(var), db.get(var)
        if Va is None and Vb is None:
            ax.text(0.5, 0.5, f'{var}\nnot available', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=9)
        else:
            if Va is not None:
                ax.plot(Va[ia, :], Ha, color=COL_A, lw=2, label=label_a)
                if sig and da.get(sig) is not None:
                    s = da[sig][ia, :]
                    ax.fill_betweenx(Ha, Va[ia, :] - s, Va[ia, :] + s, alpha=0.2, color=COL_A)
            if Vb is not None:
                ax.plot(Vb[ib, :], Hb, color=COL_B, lw=2, ls='--', label=label_b)
                if sig and db.get(sig) is not None:
                    s = db[sig][ib, :]
                    ax.fill_betweenx(Hb, Vb[ib, :] - s, Vb[ib, :] + s, alpha=0.2, color=COL_B)
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel, fontsize=9);
        ax.set_ylabel('Height AGL (km)', fontsize=8)
        ax.set_ylim(0, hmax);
        ax.grid(True, alpha=0.3);
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    _save(fig, f'{out_prefix}_compare_profiles.png')


def plot_compare_timeseries(da, db, label_a, label_b, out_prefix):
    """
    Produces *_compare_timeseries.png
    All the 1D (time-only) variables stacked as separate subplots, both datasets on each panel. Covers: LWP,
    precipitable water vapour, planetary boundary layer height, stable boundary layer inversion height and magnitude,
    surface-based and mixed-layer LCL/CAPE/CIN, cloud base height, RMSa, RMSp, and Shannon information content.
    This is useful for spotting systematic offsets or periods where the two retrievals diverge. The RMSa panel is
    particularly worth watching — it tells you how well each retrieval fits its own observations, so if one is
    consistently higher it may be struggling.
    :param da:
    :param db:
    :param label_a:
    :param label_b:
    :param out_prefix: *
    :return:
    """
    panels = [
        ('lwp', 'LWP (g/m²)', True),
        ('pwv', 'PWV (cm)', False),
        ('pblh', 'PBLH (km)', False),
        ('sbih', 'SB Inv. Height (km)', False),
        ('sbim', 'SB Inv. Mag. (°C)', False),
        ('sbLCL', 'SB LCL (km)', False),
        ('sbCAPE', 'SB CAPE (J/kg)', False),
        ('sbCIN', 'SB CIN (J/kg)', False),
        ('mlCAPE', 'ML CAPE (J/kg)', False),
        ('mlCIN', 'ML CIN (J/kg)', False),
        ('cbh', 'Cloud Base Ht (km)', False),
        ('rmsa', 'RMSa (unitless)', False),
        ('rmsp', 'RMSp (unitless)', False),
        ('sic', 'Shannon Info. Content', False),
    ]
    panels = [p for p in panels
              if da.get(p[0]) is not None or db.get(p[0]) is not None]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(f'Time-Series Comparison\n'
                 f'{label_a} (solid) vs {label_b} (dashed)', fontsize=12)
    times_a = mdates.date2num(da['timestamps'])
    times_b = mdates.date2num(db['timestamps'])

    for ax, (var, ylabel, show_unc) in zip(axes, panels):
        Va, Vb = da.get(var), db.get(var)
        if Va is not None:
            ax.plot(times_a, Va, color=COL_A, lw=1.2, label=label_a)
            if show_unc and da.get('sigma_' + var) is not None:
                sv = da['sigma_' + var]
                ax.fill_between(times_a, Va - sv, Va + sv, alpha=0.2, color=COL_A)
        if Vb is not None:
            ax.plot(times_b, Vb, color=COL_B, lw=1.2, ls='--', label=label_b)
            if show_unc and db.get('sigma_' + var) is not None:
                sv = db['sigma_' + var]
                ax.fill_between(times_b, Vb - sv, Vb + sv, alpha=0.2, color=COL_B)
        ax.set_ylabel(ylabel, fontsize=8);
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3);
        ax.tick_params(labelsize=8)

    fmt_xaxis(axes[-1]);
    fig.autofmt_xdate();
    plt.tight_layout()
    _save(fig, f'{out_prefix}_compare_timeseries.png')


# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENCE PLOTS (A minus B)
# ══════════════════════════════════════════════════════════════════════════════

def plot_diff_timehgt(da, db_a, label_a, label_b, out_prefix):
    """
    Produces *_diff_timehgt.png
    Time/height colour maps showing File A minus File B for the 2D profile variables: temperature, water vapour,
    RH, potential temperature, equivalent potential temperature, and vertical resolution. The colourmap is always
    diverging and centred on zero (RdBu_r), so red means A is larger than B, blue means B is larger than A, and
    white means they agree. This is the clearest way to see whether differences are systematic (a persistent colour)
    or random (patchy), and whether they're height-dependent.
    :param da:
    :param db_a:
    :param label_a:
    :param label_b:
    :param out_prefix: *
    :return:
    """
    panels = [
        ('temp', 'Temperature diff (°C)', (-5, 5)),
        ('wv', 'Water Vapour diff (g/kg)', (-2, 2)),
        ('rh', 'Relative Humidity diff (%)', (-20, 20)),
        ('theta', 'Potential Temp diff (K)', (-5, 5)),
        ('thetae', 'Equiv. Pot. Temp diff (K)', (-10, 10)),
        ('vres_t', 'Vert. Res. T diff (km)', (-2, 2)),
    ]
    panels = [(v, l, r) for v, l, r in panels
              if da.get(v) is not None and db_a.get(v) is not None]
    if not panels:
        print('  No 2D variables for time/height diff — skipping.');
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(f'Time/Height Difference: {label_a} minus {label_b}', fontsize=12)
    times = mdates.date2num(da['timestamps'])
    H = da['height']

    for ax, (var, ylabel, (vmin, vmax)) in zip(axes, panels):
        diff = da[var] - db_a[var]
        cf = ax.contourf(times, H, diff.T,
                         levels=np.linspace(vmin, vmax, 21),
                         cmap='RdBu_r', extend='both')
        plt.colorbar(cf, ax=ax, label=ylabel)
        ax.set_ylabel('Height AGL (km)');
        ax.set_ylim(0, H_MAX_T)
        ax.set_title(ylabel);
        ax.grid(True, alpha=0.2);
        ax.tick_params(labelsize=8)

    fmt_xaxis(axes[-1]);
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f'{out_prefix}_diff_timehgt.png')


def plot_diff_timeseries(da, db_a, label_a, label_b, out_prefix):
    """
    Produces *_diff_timeseries.png
    The same idea but for all the 1D variables — a time series of A minus B for each one. The zero line is marked as a
    dashed grey reference. Each panel has the mean difference and standard deviation annotated in the corner,
    which gives you an at-a-glance quantitative summary. If you end up writing a paper, these numbers are likely
    to appear in a results table.
    :param da:
    :param db_a:
    :param label_a:
    :param label_b:
    :param out_prefix: *
    :return:
    """
    panels = [
        ('lwp', 'LWP diff (g/m²)'),
        ('pwv', 'PWV diff (cm)'),
        ('pblh', 'PBLH diff (km)'),
        ('sbih', 'SB Inv. Height diff (km)'),
        ('sbim', 'SB Inv. Mag. diff (°C)'),
        ('sbLCL', 'SB LCL diff (km)'),
        ('sbCAPE', 'SB CAPE diff (J/kg)'),
        ('sbCIN', 'SB CIN diff (J/kg)'),
        ('mlCAPE', 'ML CAPE diff (J/kg)'),
        ('mlCIN', 'ML CIN diff (J/kg)'),
        ('cbh', 'Cloud Base Ht diff (km)'),
        ('rmsa', 'RMSa diff'),
        ('sic', 'Shannon Info. Content diff'),
    ]
    panels = [(v, l) for v, l in panels
              if da.get(v) is not None and db_a.get(v) is not None]
    if not panels:
        print('  No 1D variables for diff time-series — skipping.');
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(f'Time-Series Difference: {label_a} minus {label_b}', fontsize=12)
    times = mdates.date2num(da['timestamps'])

    for ax, (var, ylabel) in zip(axes, panels):
        diff = da[var] - db_a[var]
        ax.plot(times, diff, color='darkslateblue', lw=1.2)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_ylabel(ylabel, fontsize=8);
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        mu, std = np.nanmean(diff), np.nanstd(diff)
        ax.text(0.99, 0.92, f'mean={mu:+.3f}  σ={std:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                color='darkslateblue',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fmt_xaxis(axes[-1]);
    fig.autofmt_xdate();
    plt.tight_layout()
    _save(fig, f'{out_prefix}_diff_timeseries.png')


def plot_diff_mean_profile(da, db_a, label_a, label_b, out_prefix):
    """
    Produces *_diff_mean_profile.png
    Probably the most paper-ready of all the plots. Two panels (temperature and water vapour), each showing the
    mean difference profile — averaged over the whole time period — as a single bold line, with ±1 standard
    deviation as medium shading and the 10th–90th percentile range as lighter shading. The zero line is marked
    for reference. This collapses the whole day into a single summary of how the two retrievals differ as a
    function of height, and directly answers the question "do these two retrievals agree, and if not, at what
    heights and by how much?".
    :param da:
    :param db_a:
    :param label_a:
    :param label_b:
    :param out_prefix: *
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True)
    fig.suptitle(f'Mean Profile Difference: {label_a} minus {label_b}\n'
                 f'(shading = ±1 std;  whiskers = 10th–90th percentile)', fontsize=11)
    H = da['height']

    for ax, var, xlabel, hmax in [
        (axes[0], 'temp', 'Temperature diff (°C)', H_MAX_T),
        (axes[1], 'wv', 'Water Vapour diff (g/kg)', H_MAX_WV),
    ]:
        if da.get(var) is None or db_a.get(var) is None:
            ax.text(0.5, 0.5, 'not available', ha='center', va='center',
                    transform=ax.transAxes, color='gray');
            continue
        diff = da[var] - db_a[var]
        mu = np.nanmean(diff, axis=0)
        std = np.nanstd(diff, axis=0)
        p10 = np.nanpercentile(diff, 10, axis=0)
        p90 = np.nanpercentile(diff, 90, axis=0)
        ax.fill_betweenx(H, p10, p90, alpha=0.15, color='darkslateblue',
                         label='10th–90th percentile')
        ax.fill_betweenx(H, mu - std, mu + std, alpha=0.35,
                         color='darkslateblue', label='±1 std')
        ax.plot(mu, H, color='darkslateblue', lw=2.5, label='Mean diff')
        ax.axvline(0, color='gray', lw=1, ls='--')
        ax.set_xlabel(xlabel);
        ax.set_ylim(0, hmax)
        ax.legend(fontsize=8);
        ax.grid(True, alpha=0.3);
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel('Height AGL (km)')
    plt.tight_layout()
    _save(fig, f'{out_prefix}_diff_mean_profile.png')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ══════════════════════════════════════════════════════════════════════════════


# Path to your TROPoe output file
FILE_A = 'C:/Users/c7071147/Documents/TROPoe_run/tropoe/hatpro/tropoe.c1.20251201.000015.nc'

# Path to a second file for comparison (set to None to skip)
FILE_B = 'I:/User/Documents/Research/Running_TROPoe/Download from Dave/beth_saunders/tropoe/hatpro/tropoe.c1.20251201.000015.nc'

# Labels for the comparison plot
LABEL_A = 'Beth'
LABEL_B = 'Dave'

PROFILE_TIME = 12.0  # UTC hour for single-profile plot; None = midpoint of file
OUT_PREFIX = 'tropoe'

H_MAX_T = 10.0  # height ceiling for temperature/theta plots (km)
H_MAX_WV = 6.0  # height ceiling for moisture plots (km)

COL_A = 'steelblue'
COL_B = 'tomato'

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

print(f'Loading {FILE_A} ...')
da = load_tropoe(FILE_A)

t_profile = PROFILE_TIME
if t_profile is None:
    t_profile = float(da['hour'][len(da['hour']) // 2])
    print(f'Using midpoint time: {t_profile:.2f} UTC')

print('Single-file plots ...')
plot_time_height(da, OUT_PREFIX)
plot_lwp(da, OUT_PREFIX)
plot_profile_with_uncertainty(da, t_profile, OUT_PREFIX)

if FILE_B is not None:
    print(f'Loading {FILE_B} ...')
    db = load_tropoe(FILE_B)

    print('Comparison plots ...')
    plot_compare_profiles(da, db, LABEL_A, LABEL_B, t_profile, OUT_PREFIX)
    plot_compare_timeseries(da, db, LABEL_A, LABEL_B, OUT_PREFIX)

    print('Aligning datasets for difference plots ...')
    db_aligned = align_to(da, db)

    print('Difference plots ...')
    plot_diff_timehgt(da, db_aligned, LABEL_A, LABEL_B, OUT_PREFIX)
    plot_diff_timeseries(da, db_aligned, LABEL_A, LABEL_B, OUT_PREFIX)
    plot_diff_mean_profile(da, db_aligned, LABEL_A, LABEL_B, OUT_PREFIX)

print('All done.')
