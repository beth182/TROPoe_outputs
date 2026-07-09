"""
Radiosonde visualisation – TEAMx wEOP, Kolsass, 19 Feb 2025

Requires: pandas, matplotlib, numpy
"""

import re
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "sonde_data_20250219")
DATA_DIR = Path(_DATA)
assert os.path.isdir(DATA_DIR), f"Data folder not found: {DATA_DIR}"

# derive date string from folder name for the output subfolder
_date_str = DATA_DIR.name.replace("sonde_data_", "")   # e.g. "20250219"
PLOT_DIR = Path(_HERE) / "plots" / _date_str
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────

HEADER_ROWS = 7
COL_NAMES = [
    "elapsed_time", "latitude", "longitude", "geopotential_height",
    "air_pressure", "wind_from_direction", "wind_speed",
    "u_wind", "v_wind", "air_temperature", "dew_point_temperature",
    "air_potential_temperature", "relative_humidity", "humidity_mixing_ratio",
]


def parse_launch_time(filepath):
    m = re.search(r"(\d{10})", str(filepath))
    return (m.group(1)[8:10] + ":00 UTC") if m else os.path.basename(str(filepath))


def load_raso(filepath):
    df = pd.read_csv(
        filepath,
        skiprows=HEADER_ROWS,
        header=None,
        names=COL_NAMES,
        na_values=["", " "],
    )
    df["T_C"]   = df["air_temperature"] - 273.15
    df["Td_C"]  = df["dew_point_temperature"] - 273.15
    df["p_hPa"] = df["air_pressure"] / 100.0
    df["z_km"]  = df["geopotential_height"] / 1000.0
    return df


asc_files  = sorted(DATA_DIR.glob("*ascent.csv"))
desc_files = sorted(DATA_DIR.glob("*descent.csv"))

assert asc_files,  f"No ascent CSVs found in {DATA_DIR}"
assert desc_files, f"No descent CSVs found in {DATA_DIR}"

asc_data  = [(parse_launch_time(f), load_raso(f)) for f in asc_files]
desc_data = [(parse_launch_time(f), load_raso(f)) for f in desc_files]

# ── plot ──────────────────────────────────────────────────────────────────────

asc_cmap = cm.get_cmap("tab10", len(asc_data))

# map hour string → colour so descents can reuse the matching ascent colour
hour_colour = {
    label.split(":")[0]: asc_cmap(i / max(len(asc_data) - 1, 1))
    for i, (label, _) in enumerate(asc_data)
}

fig, axes = plt.subplots(1, 5, figsize=(18, 9), sharey=True)
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
    ax.grid(True, color="#e0e0e0", lw=0.5)

ax_T, ax_Td, ax_RH, ax_ws, ax_wd = axes
LABEL_COLOR = "#333333"

for i, (label, df) in enumerate(asc_data):
    c  = asc_cmap(i / max(len(asc_data) - 1, 1))
    kw = dict(color=c, lw=1.4, alpha=0.9, label=label + " ↑")
    ax_T.plot(df["T_C"],  df["z_km"], **kw)
    ax_Td.plot(df["Td_C"], df["z_km"], **kw)
    ax_RH.plot(df["relative_humidity"], df["z_km"], **kw)
    ax_ws.plot(df["wind_speed"], df["z_km"], **kw)
    v = df[["wind_from_direction", "z_km"]].dropna()
    ax_wd.scatter(v["wind_from_direction"], v["z_km"], color=c, s=1.0, alpha=0.8)

for label, df in desc_data:
    c  = hour_colour.get(label.split(":")[0], "black")
    kw = dict(color=c, lw=1.1, alpha=0.8, linestyle="--", label=label + " ↓")
    ax_T.plot(df["T_C"],  df["z_km"], **kw)
    ax_Td.plot(df["Td_C"], df["z_km"], **kw)
    ax_RH.plot(df["relative_humidity"], df["z_km"], **kw)
    ax_ws.plot(df["wind_speed"], df["z_km"], **kw)
    v = df[["wind_from_direction", "z_km"]].dropna()
    ax_wd.scatter(v["wind_from_direction"], v["z_km"], color=c, s=1.0, alpha=0.6, marker="x")

# axis formatting
ax_T.set_ylabel("Geopotential Height (km)", color=LABEL_COLOR, fontsize=10)
ax_T.set_ylim(0, 22)
ax_T.set_yticks(range(0, 23, 2))

for ax, xlabel in zip(axes, [
    "Temperature (°C)", "Dew-point (°C)", "Relative Humidity (%)",
    "Wind Speed (m s⁻¹)", "Wind Direction (°)"
]):
    ax.set_xlabel(xlabel, color=LABEL_COLOR, fontsize=10)

ax_RH.set_xlim(0, 105)
ax_RH.set_xticks([0, 25, 50, 75, 100])
ax_wd.set_xlim(0, 360)
ax_wd.set_xticks([0, 90, 180, 270, 360])
ax_wd.set_xticklabels(["N", "E", "S", "W", "N"], fontsize=8)

for ax, title in zip(axes, ["Temperature", "Dew-point", "Rel. Humidity", "Wind Speed", "Wind Dir."]):
    ax.set_title(title, color="#111111", fontsize=11, fontweight="bold", pad=6)

handles, labels = ax_T.get_legend_handles_labels()
ax_T.legend(handles, labels, loc="upper right", fontsize=7.5, framealpha=0.7,
            facecolor="white", edgecolor="#bbbbbb", title_fontsize=8)

fig.suptitle(f"Radiosonde {_date_str}",
             color="#111111", fontsize=14, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.08)

# ── save ──────────────────────────────────────────────────────────────────────

out_path = PLOT_DIR / "raso_profiles.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")