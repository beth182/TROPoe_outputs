"""
Visual inspection of MWRpy 1C01 output NetCDF.
Run after create_inputs.py has produced the file(s).
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import os

print('Imports complete')

# ── config ────────────────────────────────────────────────────────────────────
base_output_dir = r"/TROPoe_outputs/create_TROPoe_inputs/output/TOC/"
_HERE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isdir(_HERE), f"Data folder not found: {_HERE}"
# ─────────────────────────────────────────────────────────────────────────────

# Find all subdirectories of base_output_dir that look like dates (YYYYMMDD)
# and actually contain the expected nc file.
datestrings = []
for name in sorted(os.listdir(base_output_dir)):
    full_path = os.path.join(base_output_dir, name)
    if not os.path.isdir(full_path):
        continue
    try:
        datetime.strptime(name, '%Y%m%d')
    except ValueError:
        continue  # not a date-named folder, skip it

    candidate_nc = os.path.join(full_path, f"innsbruck_1C01_{name}.nc")
    if os.path.isfile(candidate_nc):
        datestrings.append(name)
    else:
        print(f"Skipping {name}: no nc file found at {candidate_nc}")

print(f"Found {len(datestrings)} date folders with data: {datestrings}")


def unix_to_dt(unix_arr):
    """Convert array of unix timestamps to Python datetimes (UTC)."""
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_arr]


def process_date(datestring):
    NC_FILE = os.path.join(base_output_dir, datestring, f"innsbruck_1C01_{datestring}.nc")
    assert os.path.isfile(NC_FILE), f"NC file not found: {NC_FILE}"

    output_dir = os.path.join(_HERE, "plots/TOC/", datestring) + '/'
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Print file summary ─────────────────────────────────────────────────
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

    # ── Pull core arrays ──────────────────────────────────────────────────────
    time_unix = ds["time"][:]
    time_dt = unix_to_dt(time_unix)
    freq = ds["frequency"][:]  # GHz

    tb = ds["tb"][:]  # (time, frequency)  K
    ele = ds["elevation_angle"][:]  # degrees
    azi = ds["azimuth_angle"][:]  # degrees
    qflag = ds["quality_flag"][:]  # (time, frequency)

    def safe(var):
        return ds[var][:] if var in ds.variables else None

    t_air = safe("air_temperature")  # K
    rh = safe("relative_humidity")  # 1
    p_air = safe("air_pressure")  # Pa
    rain = safe("rainfall_rate")  # mm/h
    irt = safe("irt")  # K  (infrared TB)

    # ── 2. Brightness temperatures ─────────────────────────────────────────────
    zenith_mask = (ele > 89.0) & (ele < 91.0)
    t_zen = [t for t, m in zip(time_dt, zenith_mask) if m]
    tb_zen = tb[zenith_mask, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Brightness Temperatures — Innsbruck HATPRO  {datestring}", fontsize=12)

    kband = freq < 35.0
    ax = axes[0]
    for i, f in enumerate(freq[kband]):
        ax.plot(t_zen, tb_zen[:, np.where(kband)[0][i]], label=f"{f:.2f} GHz", lw=0.8)
    ax.set_ylabel("TB  [K]")
    ax.set_title("K-band (22–31 GHz)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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
    plt.savefig(output_dir + datestring + "_tb_timeseries.png", dpi=150)
    # plt.show()

    # ── 3. Scan geometry ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    fig.suptitle("Scan geometry", fontsize=12)
    ax1.plot(time_dt, ele, lw=0.5, color="steelblue")
    ax1.set_ylabel("Elevation  [°]")
    ax2.plot(time_dt, azi, lw=0.5, color="darkorange")
    ax2.set_ylabel("Azimuth  [°]")
    ax2.set_xlabel("Time (UTC)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig(output_dir + datestring + "_scan_geometry.png", dpi=150)
    # plt.show()

    # ── 4. Quality flags (bitmap heatmap) ──────────────────────────────────────
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
    plt.savefig(output_dir + datestring + "_quality_flags.png", dpi=150)
    # plt.show()

    # ── 5. MET data ─────────────────────────────────────────────────────────────
    met_vars = {
        "air_temperature [K]": t_air,
        "relative_humidity [1]": rh,
        "air_pressure [Pa]": p_air,
        "rainfall_rate [mm/h]": rain,
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
        plt.savefig(output_dir + datestring + "_met_data.png", dpi=150)
        # plt.show()

    # ── 6. IRT (infrared TB) ──────────────────────────────────────────────────
    if irt is not None:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time_dt, irt if irt.ndim == 1 else irt[:, 0], lw=0.8, color="firebrick")
        ax.set_ylabel("IRT  [K]")
        ax.set_xlabel("Time (UTC)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title("Infrared brightness temperature")
        plt.tight_layout()
        plt.savefig(output_dir + datestring + "_irt.png", dpi=150)
        # plt.show()

    ds.close()
    print(f"Plots saved for {datestring} in {output_dir}")


# ── Run for all found dates ────────────────────────────────────────────────────
for datestring in datestrings:
    print(f"\n--- Plotting {datestring} ---")
    try:
        process_date(datestring)
    except Exception as e:
        print(f"FAILED for {datestring}: {e}")
        continue

print('\nend')