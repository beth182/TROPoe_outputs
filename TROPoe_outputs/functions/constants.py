"""
constants.py
------------
Single source of truth for values that were previously hardcoded (and
duplicated) across plot_tropoe.py, compare_TROPoe_Massaro.py, and
HATPRO_raso_visu.py.

If any of these ever need correcting, this is the only place to change.
"""

import numpy as np

# --- Site / instrument geometry ---------------------------------------------

# HATPRO retrieval height grid [km above ground], 39 levels.
# Source: 26040823_CMP_TPC.NC altitude_layers (m -> km).
# Was hardcoded identically in compare_TROPoe_Massaro.py and HATPRO_raso_visu.py.
HATPRO_HEIGHTS_KM = np.array([
    0.000, 0.010, 0.030, 0.050, 0.075, 0.100, 0.125, 0.150,
    0.200, 0.250, 0.325, 0.400, 0.475, 0.550, 0.625, 0.700,
    0.800, 0.900, 1.000, 1.150, 1.300, 1.450, 1.600, 1.800,
    2.000, 2.200, 2.500, 2.800, 3.100, 3.500, 3.900, 4.400,
    5.000, 5.600, 6.200, 7.000, 8.000, 9.000, 10.000,
])

# HATPRO instrument site elevation [m asl]. From HATPRO_raso_visu.py.
# NOTE: that script's comment mentions both Kolsass (~545 m asl) and
# Innsbruck (~612 m asl) — the value used in the code was 612.0.
# Flagging this so you can confirm which site each dataset actually needs.
HATPRO_SITE_ELEV_M = 612.0

# --- Matching / processing defaults -----------------------------------------

# Max time difference (minutes) between sonde launch and instrument retrieval
# when looking for a match. Used identically for HATPRO and TROPoe matching
# in HATPRO_raso_visu.py.
MATCH_WINDOW_MIN = 10

# --- Physical constants ------------------------------------------------------

RD = 287.05     # specific gas constant, dry air [J kg-1 K-1]
RV = 461.5      # specific gas constant, water vapour [J kg-1 K-1]
ESAT0 = 611.7   # reference saturation vapour pressure [Pa] at T0
T0 = 273.16     # reference temperature [K]

# --- Plot styling defaults ----------------------------------------------------

H_MAX_T = 10.0   # height ceiling for temperature/theta plots (km)
H_MAX_WV = 6.0   # height ceiling for moisture plots (km)

# Standardised colours (as used in HATPRO_raso_visu.py / your comparison methodology)
COLOR_HATPRO = 'tab:red'
COLOR_TROPOE = 'tab:green'
COLOR_SONDE = 'black'