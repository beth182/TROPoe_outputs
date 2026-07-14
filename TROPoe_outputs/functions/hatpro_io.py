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