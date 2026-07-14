"""
interpolation.py
----------------
Generalized versions of the interpolation helpers that appeared (in
slightly different forms) in plot_tropoe.py (_interp_time, _interp_height,
align_to) and HATPRO_raso_visu.py (interpolate_sonde_to_hatpro_levels).

These are pure numerical helpers — no file I/O, no plotting.
"""

import numpy as np
from scipy.interpolate import interp1d


def interp_time(src_time, src_data, tgt_time):
    """
    Interpolate a 1D or 2D (time, height) array from its own time axis onto
    a different target time axis.

    Was: _interp_time() in plot_tropoe.py

    Parameters
    ----------
    src_time : array_like, shape (n_src,)   source time axis (any consistent unit)
    src_data : array_like, shape (n_src,) or (n_src, n_levels), or None
    tgt_time : array_like, shape (n_tgt,)   target time axis (same unit as src_time)

    Returns
    -------
    ndarray shape (n_tgt,) or (n_tgt, n_levels), or None if src_data is None
    """
    if src_data is None:
        return None
    src_data = np.asarray(src_data, dtype=float)
    if src_data.ndim == 1:
        return interp1d(src_time, src_data, bounds_error=False, fill_value=np.nan)(tgt_time)
    out = np.full((len(tgt_time), src_data.shape[1]), np.nan)
    for j in range(src_data.shape[1]):
        out[:, j] = interp1d(src_time, src_data[:, j],
                              bounds_error=False, fill_value=np.nan)(tgt_time)
    return out


def interp_height(data_2d, h_from, h_to):
    """
    Interpolate a (time, height) array from its own height grid onto a
    different target height grid, row by row.

    Was: _interp_height() in plot_tropoe.py

    Parameters
    ----------
    data_2d : array_like, shape (n_time, n_from), or None
    h_from  : array_like, shape (n_from,)
    h_to    : array_like, shape (n_to,)

    Returns
    -------
    ndarray shape (n_time, n_to), or None if data_2d is None
    """
    if data_2d is None:
        return None
    data_2d = np.asarray(data_2d, dtype=float)
    out = np.full((data_2d.shape[0], len(h_to)), np.nan)
    for i in range(data_2d.shape[0]):
        row = data_2d[i, :]
        if not np.all(np.isnan(row)):
            out[i, :] = interp1d(h_from, row, bounds_error=False,
                                  fill_value=np.nan)(h_to)
    return out


def align_datasets(da, db, time_key='hour', height_key='height',
                    skip_keys=('timestamps', 'hour', 'height', 'filepath', 'qc')):
    """
    Re-grid dataset db onto dataset da's time and height axes, so the two
    can be directly differenced. Generalized from align_to() in
    plot_tropoe.py — works for any two dict-of-arrays datasets that share
    the same key names (e.g. two TROPoe loads), not just TROPoe-vs-TROPoe.

    Parameters
    ----------
    da, db      : dict   e.g. output of load_tropoe(); must both have
                  da[time_key], da[height_key] etc.
    time_key    : str    key holding the time axis (default 'hour')
    height_key  : str    key holding the height axis (default 'height')
    skip_keys   : tuple  keys to carry over from da unchanged rather than
                  interpolate (axes, metadata)

    Returns
    -------
    dict  db's data re-gridded onto da's time/height axes
    """
    ha, hb = da[height_key], db[height_key]
    aligned = {k: da[k] for k in skip_keys if k in da}

    for key, val in db.items():
        if key in skip_keys or val is None:
            aligned.setdefault(key, None)
            continue
        v = interp_time(db[time_key], val, da[time_key])
        heights_differ = (len(ha) != len(hb)) or not np.allclose(ha, hb, atol=0.001)
        if v is not None and np.ndim(v) == 2 and heights_differ:
            v = interp_height(v, hb, ha)
        aligned[key] = v
    return aligned


def interpolate_profile_to_heights(source_heights_m_asl, source_vars,
                                    target_heights_m_asl):
    """
    Linearly interpolate a set of profile variables (e.g. from a radiosonde)
    onto a different, typically coarser, height grid (e.g. HATPRO or TROPoe
    levels). Generalized from interpolate_sonde_to_hatpro_levels() in
    HATPRO_raso_visu.py — that function did the m-asl conversion internally
    and was hardcoded to sonde column names; this version takes already-
    converted heights (in the same units on both sides) and an arbitrary
    dict of variables, so it isn't tied to sonde-specific naming.

    Parameters
    ----------
    source_heights_m_asl : array_like, shape (n_source,)
    source_vars          : dict[str, array_like]  each shape (n_source,)
    target_heights_m_asl : array_like, shape (n_target,)

    Returns
    -------
    dict[str, ndarray]  each shape (n_target,), NaN outside source range
    """
    result = {}
    for name, values in source_vars.items():
        f = interp1d(source_heights_m_asl, values, bounds_error=False,
                      fill_value=np.nan)
        result[name] = f(target_heights_m_asl)
    return result