"""
io/tropoe_io.py
---------------
One canonical load_tropoe(), replacing the 4 separate versions found in
plot_tropoe.py, compare_TROPoe_Massaro.py, and HATPRO_raso_visu.py (which
had two of its own — one for TROPoe-vs-TROPoe QC masking, one for the
sonde-matching pipeline).

Per your call: loads everything, always, in one consistent shape. Callers
just pick the keys they need.

Returns a plain dict (not a class) so it stays easy to inspect/print and
matches your existing "no argparse, functions not objects" style.
"""

import numpy as np
import netCDF4 as nc
import pandas as pd

from humidity import humidity_rh_temp_to_abs, humidity_mixing_ratio_to_abs


def _get(ds, name):
    """Read a variable if present, filling masked values with NaN."""
    if name not in ds.variables:
        return None
    v = ds.variables[name][:]
    v = np.ma.filled(v, np.nan) if hasattr(v, 'filled') else np.array(v)
    return v.astype(float)


def load_tropoe(filepath, mask_qc=True):
    """
    Load a TROPoe NetCDF retrieval file, returning every field used across
    your existing scripts, plus a couple of convenience derived fields.

    Parameters
    ----------
    filepath : str or Path
    mask_qc  : bool  if True (default), timesteps where qc_flag != 0 are set
               to NaN across all 2D/1D retrieved variables (this was the
               behaviour in plot_tropoe.py; the other loaders didn't do this
               — recommend keeping it on unless you have a specific reason
               to inspect flagged retrievals).

    Returns
    -------
    dict with keys:
        filepath, timestamps (list of tz-aware datetime, UTC),
        hour, height (km agl), qc,
        temp (degC), temp_k (K), sigma_t,
        wv (g/kg), sigma_wv,
        rh (%), dewpt, theta, thetae,
        vres_t, vres_wv, cdfs_t, cdfs_wv,
        pressure (mb),
        lwp, sigma_lwp, pwv, pblh, sbih, sbim,
        sbLCL, sbCAPE, sbCIN, mlCAPE, mlCIN,
        cbh, rmsa, rmsp, sic,
        abs_hum_from_rh (g/m3, via humidity_rh_temp_to_abs using rh + temp_k),
        abs_hum_from_mixing (g/m3, via humidity_mixing_ratio_to_abs using
            wv + pressure + temp, only computed if pressure is present)
    """
    ds = nc.Dataset(filepath, 'r')

    base = float(ds.variables['base_time'][:])
    offset = np.array(ds.variables['time_offset'][:], dtype=float)
    timestamps = pd.to_datetime(base + offset, unit='s', utc=True).to_pydatetime().tolist()

    hour = _get(ds, 'hour')
    height = _get(ds, 'height')
    qc = _get(ds, 'qc_flag')
    bad = (qc != 0) if (mask_qc and qc is not None) else None

    def g2d(name):
        v = _get(ds, name)
        if v is not None and bad is not None:
            v = v.copy()
            v[bad, :] = np.nan
        return v

    def g1d(name):
        v = _get(ds, name)
        if v is not None and bad is not None:
            v = v.copy()
            v[bad] = np.nan
        return v

    data = dict(
        filepath=str(filepath), timestamps=timestamps, hour=hour, height=height, qc=qc,
        temp=g2d('temperature'), sigma_t=g2d('sigma_temperature'),
        wv=g2d('waterVapor'), sigma_wv=g2d('sigma_waterVapor'),
        rh=g2d('rh'), dewpt=g2d('dewpt'),
        theta=g2d('theta'), thetae=g2d('thetae'),
        vres_t=g2d('vres_temperature'), vres_wv=g2d('vres_waterVapor'),
        cdfs_t=g2d('cdfs_temperature'), cdfs_wv=g2d('cdfs_waterVapor'),
        pressure=g2d('pressure'),
        lwp=g1d('lwp'), sigma_lwp=g1d('sigma_lwp'),
        pwv=g1d('pwv'), pblh=g1d('pblh'),
        sbih=g1d('sbih'), sbim=g1d('sbim'),
        sbLCL=g1d('sbLCL'), sbCAPE=g1d('sbCAPE'), sbCIN=g1d('sbCIN'),
        mlCAPE=g1d('mlCAPE'), mlCIN=g1d('mlCIN'),
        cbh=g1d('cbh'), rmsa=g1d('rmsa'),
        rmsp=g1d('rmsp'), sic=g1d('sic'),
    )
    ds.close()

    # Convenience: temperature in Kelvin (several scripts needed this)
    data['temp_k'] = data['temp'] + 273.15 if data['temp'] is not None else None

    # Convenience: absolute humidity via both methods, where inputs allow.
    if data['rh'] is not None and data['temp_k'] is not None:
        data['abs_hum_from_rh'] = humidity_rh_temp_to_abs(data['rh'], data['temp_k'])
    else:
        data['abs_hum_from_rh'] = None

    if (data['wv'] is not None and data['pressure'] is not None
            and data['temp'] is not None):
        data['abs_hum_from_mixing'] = humidity_mixing_ratio_to_abs(
            data['wv'], data['pressure'], data['temp'])
    else:
        data['abs_hum_from_mixing'] = None

    return data