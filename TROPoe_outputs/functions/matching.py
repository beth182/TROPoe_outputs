"""
matching.py
-----------
HATPRO_raso_visu.py had match_hatpro_to_sonde() and match_tropoe_to_sonde()
as two near-identical, copy-pasted functions (same logic, different variable
names). This replaces both with one generic function.

Matching rule (Scheiber 2025): prefer the first timestep within
window_min minutes AFTER the target time; fall back to the closest
timestep within window_min minutes BEFORE if no post-target retrieval
exists.
"""

import pandas as pd

from .constants import MATCH_WINDOW_MIN


def find_nearest_time(available_times, target_time, window_min=MATCH_WINDOW_MIN):
    """
    Find the timestamp in `available_times` closest to `target_time`,
    preferring the first one at or after target_time within the window,
    falling back to the closest one before it within the window.

    Parameters
    ----------
    available_times : DatetimeIndex   e.g. a DataFrame's .index
    target_time      : datetime       may be tz-aware; will be compared as naive
    window_min       : int            max minutes away to accept a match

    Returns
    -------
    Timestamp or None
    """
    target_naive = target_time.replace(tzinfo=None) if target_time.tzinfo else target_time
    delta = available_times - target_naive
    window = pd.Timedelta(minutes=window_min)

    after_mask = (delta >= pd.Timedelta(0)) & (delta <= window)
    if after_mask.any():
        idx = delta[after_mask].argmin()
        return available_times[after_mask][idx]

    before_mask = (delta < pd.Timedelta(0)) & (delta >= -window)
    if before_mask.any():
        idx = (-delta[before_mask]).argmin()
        return available_times[before_mask][idx]

    return None


def match_profiles_to_time(target_time, window_min=MATCH_WINDOW_MIN, **profile_dfs):
    """
    Match one or more profile DataFrames (sharing a common DatetimeIndex) to
    a single target time (e.g. a sonde launch), and pull out the matched row
    from each.

    Replaces match_hatpro_to_sonde() and match_tropoe_to_sonde(), which did
    this same thing twice with different variable names.

    Parameters
    ----------
    target_time : datetime           time to match against (e.g. sonde launch)
    window_min  : int
    **profile_dfs : DataFrame        any number of named DataFrames, e.g.
                    temp=hatpro_temp_df, hum=hatpro_hum_df

    Returns
    -------
    matched_time : Timestamp or None
    profiles     : dict[str, ndarray] or dict[str, None]  matched row from
                   each input DataFrame, keyed the same way as the input

    Example
    -------
    >>> matched_time, profs = match_profiles_to_time(
    ...     launch_time, temp=hatpro_temp, hum=hatpro_hum)
    >>> profs['temp'], profs['hum']
    """
    if not profile_dfs:
        raise ValueError("Provide at least one named profile DataFrame.")

    # All inputs are assumed to share the same time index (as they did in
    # the original scripts — HATPRO temp/hum come from the same instrument).
    reference_df = next(iter(profile_dfs.values()))
    matched_time = find_nearest_time(reference_df.index, target_time, window_min)

    if matched_time is None:
        return None, {name: None for name in profile_dfs}

    profiles = {name: df.loc[matched_time].values.astype(float)
                for name, df in profile_dfs.items()}
    return matched_time, profiles