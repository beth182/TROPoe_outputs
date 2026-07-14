"""
humidity.py
-----------
Two genuinely different humidity conversions were found across your scripts.
They are NOT duplicates of each other — they take different inputs and rest
on different physics — so they're kept as two clearly-named function pairs
rather than merged into one.

METHOD 1 — RH + temperature  <->  absolute humidity
    Clausius-Clapeyron with temperature-dependent latent heat
    (Koutsoyiannis 2012 / Scheiber 2025). This is the method used throughout
    HATPRO_raso_visu.py (sonde_rh_to_abs_humidity, hatpro_abs_hum_to_rh).
    Forward and inverse are exact inverses of each other, same constants.

METHOD 2 — mixing ratio + pressure + temperature  ->  absolute humidity
    Ideal gas law via dry air density (rho_dry = P / (Rd * T)).
    This is the method used in compare_TROPoe_Massaro.py's load_tropoe(),
    to convert TROPoe's waterVapor (g/kg mixing ratio) into g/m3 so it's
    comparable against HATPRO's native g/m3 units.

Use Method 1 whenever you have RH directly. Use Method 2 whenever you only
have a mixing ratio + pressure (e.g. straight out of a TROPoe file) and need
g/m3 without going via RH.
"""

import numpy as np

from TROPoe_outputs.functions.constants import RV, ESAT0, T0, RD


# ══════════════════════════════════════════════════════════════════════════
# METHOD 1 — RH <-> absolute humidity (Clausius-Clapeyron, Koutsoyiannis 2012)
# ══════════════════════════════════════════════════════════════════════════

def humidity_rh_temp_to_abs(rh_percent, temp_k):
    """
    Convert relative humidity [%] + temperature [K] to absolute humidity [g/m3].

    Steps
    -----
    1. Temperature-dependent latent heat:      Lv = 3.139e6 - 2336 * T   [J/kg]
    2. Saturation vapour pressure (C-C):       esat = esat0 * exp(-Lv/Rv * (1/T - 1/T0))
    3. Actual vapour pressure:                 e = (RH/100) * esat
    4. Absolute humidity (ideal gas, e=rho*Rv*T): rho = e / (Rv * T)   -> g/m3

    Was: sonde_rh_to_abs_humidity() in HATPRO_raso_visu.py

    Parameters
    ----------
    rh_percent : array_like  relative humidity [%]
    temp_k     : array_like  temperature [K]

    Returns
    -------
    ndarray  absolute humidity [g/m3]
    """
    rh_percent = np.asarray(rh_percent, dtype=float)
    temp_k = np.asarray(temp_k, dtype=float)

    Lv = 3.139e6 - 2336.0 * temp_k
    esat = ESAT0 * np.exp(-Lv / RV * (1.0 / temp_k - 1.0 / T0))
    e = (rh_percent / 100.0) * esat
    rho = e / (RV * temp_k)  # kg/m3
    return rho * 1000.0  # -> g/m3


def humidity_abs_to_rh(abs_hum_gm3, temp_k):
    """
    Convert absolute humidity [g/m3] + temperature [K] to relative humidity [%].

    Exact inverse of humidity_rh_temp_to_abs(), same constants.

    Was: hatpro_abs_hum_to_rh() in HATPRO_raso_visu.py

    Parameters
    ----------
    abs_hum_gm3 : array_like  absolute humidity [g/m3]
    temp_k      : array_like  temperature [K]

    Returns
    -------
    ndarray  relative humidity [%]
    """
    abs_hum_gm3 = np.asarray(abs_hum_gm3, dtype=float)
    temp_k = np.asarray(temp_k, dtype=float)

    rho_kgm3 = abs_hum_gm3 / 1000.0
    e = rho_kgm3 * RV * temp_k

    Lv = 3.139e6 - 2336.0 * temp_k
    esat = ESAT0 * np.exp(-Lv / RV * (1.0 / temp_k - 1.0 / T0))

    return (e / esat) * 100.0


# ══════════════════════════════════════════════════════════════════════════
# METHOD 2 — mixing ratio + pressure + temperature -> absolute humidity
#            (ideal gas law via dry air density)
# ══════════════════════════════════════════════════════════════════════════

def humidity_mixing_ratio_to_abs(mixing_ratio_gkg, pressure_mb, temp_c):
    """
    Convert a water vapour mixing ratio [g/kg] to absolute humidity [g/m3]
    using dry-air density from the ideal gas law.

    Steps
    -----
    1. Dry air density:  rho_dry = (P * 100) / (Rd * (T_C + 273.15))   [kg/m3]
    2. Absolute humidity: wv_gm3 = mixing_ratio_gkg * rho_dry

    Was: the pressure-based conversion inside load_tropoe() in
    compare_TROPoe_Massaro.py.

    Parameters
    ----------
    mixing_ratio_gkg : array_like  water vapour mixing ratio [g/kg]
    pressure_mb      : array_like  pressure [mb / hPa]
    temp_c           : array_like  temperature [degC]

    Returns
    -------
    ndarray  absolute humidity [g/m3]
    """
    mixing_ratio_gkg = np.asarray(mixing_ratio_gkg, dtype=float)
    pressure_mb = np.asarray(pressure_mb, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)

    rho_dry = (pressure_mb * 100.0) / (RD * (temp_c + 273.15))  # kg/m3
    return mixing_ratio_gkg * rho_dry