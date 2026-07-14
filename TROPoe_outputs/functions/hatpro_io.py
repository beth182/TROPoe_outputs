"""
io/hatpro_io.py
---------------
One canonical HATPRO CSV reader. compare_TROPoe_Massaro.py and
HATPRO_raso_visu.py each had their own reader for what looks like the same
underlying file format (data_temperature.csv, data_humidity.csv,
data_met.csv: semicolon-delimited, first line a '#' comment, then a header
row). This version follows HATPRO_raso_visu.py's approach (skiprows=1,
index_col=0) since that one also handled data_met.csv.

Worth double-checking the first time you swap this in: if your two original
readers ever produced subtly different DataFrames from the same file, that
was a latent inconsistency between scripts — this makes it consistent
everywhere going forward.
"""

import pandas as pd


def _read_hatpro_csv(path):
    df = pd.read_csv(path, sep=";", skiprows=1, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False)
    return df


def load_hatpro_profiles(temp_path, hum_path):
    """
    Load HATPRO temperature and humidity profile CSVs.

    Parameters
    ----------
    temp_path : str or Path   data_temperature.csv [K]
    hum_path  : str or Path   data_humidity.csv [g/m3]

    Returns
    -------
    temp_df : DataFrame  index=datetime, columns=0..n_levels-1, values [K]
    hum_df  : DataFrame  index=datetime, columns=0..n_levels-1, values [g/m3]
    """
    temp_df = _read_hatpro_csv(temp_path)
    hum_df = _read_hatpro_csv(hum_path)
    temp_df.columns = range(len(temp_df.columns))
    hum_df.columns = range(len(hum_df.columns))
    return temp_df, hum_df


def load_hatpro_met(met_path):
    """
    Load HATPRO surface met file.

    Parameters
    ----------
    met_path : str or Path   data_met.csv

    Returns
    -------
    DataFrame  index=datetime, columns=[hs, ps, rf, ts, dd, ff] (as in file)
    """
    return _read_hatpro_csv(met_path)


def select_hatpro_window(start, end=None, ndays=1, **hatpro_dfs):
    """
    Restrict one or more HATPRO DataFrames to a date/time window, and align
    them to their common timestamps.

    Some HATPRO files (e.g. season-wide operational retrieval CSVs covering
    a whole wEOP/sEOP campaign) span far more than the single day you
    actually want. Slicing each variable to the same window independently
    can also leave them with slightly different timestamps if one variable
    has small gaps the other doesn't -- this handles both problems at once.

    Parameters
    ----------
    start      : str or datetime-like   start of window (inclusive). A plain
                 datestring like '20250219' works via pd.Timestamp.
    end        : str or datetime-like or None   end of window (exclusive).
                 If None, computed as start + ndays.
    ndays      : int   window length in days, used only if end is None
    **hatpro_dfs : DataFrame   any number of named DataFrames sharing a
                 datetime index, e.g. temp=hat_temp_k, hum=hat_hum

    Returns
    -------
    dict[str, DataFrame]   same keys as input, each restricted to the
    window and aligned to the intersection of all the given DataFrames'
    indices (so e.g. windowed['temp'] and windowed['hum'] are guaranteed to
    have matching timestamps, row for row).

    Example
    -------
    >>> hat_temp_k, hat_hum = load_hatpro_profiles(T_CSV, Q_CSV)
    >>> windowed = select_hatpro_window(datestring, temp=hat_temp_k, hum=hat_hum)
    >>> hat_temp_k, hat_hum = windowed['temp'], windowed['hum']
    """
    if not hatpro_dfs:
        raise ValueError("Provide at least one named HATPRO DataFrame.")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start + pd.Timedelta(days=ndays)

    windowed = {name: df.loc[(df.index >= start) & (df.index < end)]
                for name, df in hatpro_dfs.items()}

    common_idx = None
    for df in windowed.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    return {name: df.loc[common_idx] for name, df in windowed.items()}