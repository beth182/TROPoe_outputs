"""
Visual inspection of MWRpy 1C01 output NetCDF.
Run after create_inputs.py has produced the file.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

# ── config ────────────────────────────────────────────────────────────────────
NC_FILE = r"I:\User\Documents\Research\Running_TROPoe\Converting_RAW_HATPRO_for_TROPoe\output\20260408_innsbruck_1C01.nc"
# ─────────────────────────────────────────────────────────────────────────────


def unix_to_dt(unix_arr):
    """Convert array of unix timestamps to Python datetimes (UTC)."""
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_arr]


# ── 1. Print file summary ─────────────────────────────────────────────────────
ds = nc.Dataset(NC_FILE)

print("=" * 60)
print("FILE:", NC_FILE)
print("=" * 60)

print("\n── Global attributes ──")
for attr in ds.ncattrs():
    print(f"  {attr}: {getattr(ds, attr)}")

print("\n── Dimensions ──")
for name, dim in ds.dimensions.items():
    print(f"  {name}: {len(dim)}")

print("\n── Variables ──")
for name, var in ds.variables.items():
    print(f"  {name:30s}  shape={str(var.shape):20s}  units={getattr(var, 'units', '—')}")

# ── Pull core arrays ──────────────────────────────────────────────────────────
time_unix = ds["time"][:]
time_dt   = unix_to_dt(time_unix)
freq      = ds["frequency"][:]          # GHz

tb        = ds["tb"][:]                 # (time, frequency)  K
ele       = ds["elevation_angle"][:]    # degrees
azi       = ds["azimuth_angle"][:]      # degrees
qflag     = ds["quality_flag"][:]       # (time, frequency)

# MET variables (may not all be present — guard each)
def safe(var):
    return ds[var][:] if var in ds.variables else None

t_air  = safe("air_temperature")        # K
rh     = safe("relative_humidity")      # 1
p_air  = safe("air_pressure")           # Pa
rain   = safe("rainfall_rate")          # mm/h
irt    = safe("irt")                    # K  (infrared TB)

# ── 2. Brightness temperatures ───────────────────────────────────────────────
# Only zenith scans (elevation ~90°)
zenith_mask = (ele > 89.0) & (ele < 91.0)
t_zen  = [t for t, m in zip(time_dt, zenith_mask) if m]
tb_zen = tb[zenith_mask, :]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("Brightness Temperatures — Innsbruck HATPRO  2026-04-08", fontsize=12)

# K-band (frequencies < 35 GHz, receiver 1)
kband = freq < 35.0
ax = axes[0]
for i, f in enumerate(freq[kband]):
    ax.plot(t_zen, tb_zen[:, np.where(kband)[0][i]], label=f"{f:.2f} GHz", lw=0.8)
ax.set_ylabel("TB  [K]")
ax.set_title("K-band (22–31 GHz)")
ax.legend(fontsize=7, ncol=4, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# V-band (frequencies > 50 GHz, receiver 2)
vband = freq > 50.0
ax = axes[1]
for i, f in enumerate(freq[vband]):
    ax.plot(t_zen, tb_zen[:, np.where(vband)[0][i]], label=f"{f:.2f} GHz", lw=0.8)
ax.set_ylabel("TB  [K]")
ax.set_title("V-band (51–58 GHz)")
ax.legend(fontsize=7, ncol=4, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time (UTC)")

plt.tight_layout()
plt.savefig("tb_timeseries.png", dpi=150)
plt.show()

# ── 3. Scan geometry ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
fig.suptitle("Scan geometry", fontsize=12)
ax1.plot(time_dt, ele, lw=0.5, color="steelblue")
ax1.set_ylabel("Elevation  [°]")
ax2.plot(time_dt, azi, lw=0.5, color="darkorange")
ax2.set_ylabel("Azimuth  [°]")
ax2.set_xlabel("Time (UTC)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.savefig("scan_geometry.png", dpi=150)
plt.show()

# ── 4. Quality flags (bitmap heatmap) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.pcolormesh(
    time_dt,
    freq,
    qflag.T,
    cmap="Reds",
    shading="auto",
)
plt.colorbar(im, ax=ax, label="quality_flag  (0 = OK)")
ax.set_ylabel("Frequency  [GHz]")
ax.set_xlabel("Time (UTC)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Quality flags (all scans)")
plt.tight_layout()
plt.savefig("quality_flags.png", dpi=150)
plt.show()

# ── 5. MET data ───────────────────────────────────────────────────────────────
met_vars = {
    "air_temperature [K]":    t_air,
    "relative_humidity [1]":  rh,
    "air_pressure [Pa]":      p_air,
    "rainfall_rate [mm/h]":   rain,
}
met_available = {k: v for k, v in met_vars.items() if v is not None}

if met_available:
    fig, axes = plt.subplots(len(met_available), 1, figsize=(12, 2.5 * len(met_available)), sharex=True)
    if len(met_available) == 1:
        axes = [axes]
    fig.suptitle("MET surface data", fontsize=12)
    for ax, (label, data) in zip(axes, met_available.items()):
        ax.plot(time_dt, data, lw=0.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig("met_data.png", dpi=150)
    plt.show()

# ── 6. IRT (infrared TB) ──────────────────────────────────────────────────────
if irt is not None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_dt, irt if irt.ndim == 1 else irt[:, 0], lw=0.8, color="firebrick")
    ax.set_ylabel("IRT  [K]")
    ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("Infrared brightness temperature")
    plt.tight_layout()
    plt.savefig("irt.png", dpi=150)
    plt.show()

ds.close()
print("\nAll plots saved as PNG in the working directory.")

print('end')