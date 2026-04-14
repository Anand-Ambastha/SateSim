"""
core.py — Satellite Communication Simulation: Computation Engine
================================================================
Handles:
  • TLE-based orbital propagation via Skyfield (SGP4)
  • Ground-station visibility (elevation angle filtering)
  • Full RF link budget (FSPL, Pr, Noise, SNR)
  • BER for BPSK / QPSK modulation
  • Doppler shift from range-rate

Author: Anand Ambastha
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.special import erfc
from skyfield.api import load, EarthSatellite, wgs84

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
SPEED_OF_LIGHT: float = 2.998e8      # m/s
BOLTZMANN_K:    float = 1.380649e-23  # J/K

# ─────────────────────────────────────────────────────────────────────────────
# Ground Station — Delhi, India
# ─────────────────────────────────────────────────────────────────────────────
GROUND_STATION: dict = {
    "name": "Delhi Ground Station",
    "lat":   28.6139,   # °N
    "lon":   77.2090,   # °E
    "alt":  216.0,      # metres ASL
}

# ─────────────────────────────────────────────────────────────────────────────
# TLE Database  (representative LEO / Sun-synchronous orbits, epoch ~Jan 2024)
# ─────────────────────────────────────────────────────────────────────────────
TLE_DATABASE: dict[str, tuple[str, str]] = {
    "ISS (ZARYA)": (
        "1 25544U 98067A   24010.50000000  .00020000  00000-0  36000-3 0  9994",
        "2 25544  51.6400 100.0000 0001000  90.0000 270.0000 15.50000000000005",
    ),
    "NOAA-18": (
        "1 28654U 05018A   24010.50000000  .00000200  00000-0  11000-3 0  9990",
        "2 28654  99.0000  50.0000 0014000 200.0000 160.0000 14.10000000000008",
    ),
    "Aqua": (
        "1 27424U 02022A   24010.50000000  .00000050  00000-0  30000-4 0  9992",
        "2 27424  98.2000  30.0000 0001200 270.0000  90.0000 14.57000000000003",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Default link-budget parameters
# ─────────────────────────────────────────────────────────────────────────────
LINK_PARAMS: dict = {
    "frequency_hz":    2.0e9,   # Carrier frequency  [Hz]
    "tx_power_dbw":   10.0,     # Transmit power     [dBW]
    "tx_gain_dbi":    10.0,     # Tx antenna gain    [dBi]
    "rx_gain_dbi":     5.0,     # Rx antenna gain    [dBi]
    "system_temp_k":  290.0,    # System noise temp  [K]
    "bandwidth_hz":    1.0e6,   # Noise bandwidth    [Hz]
    "misc_losses_db":  3.0,     # Pointing + atm.    [dB]
    "modulation":     "BPSK",   # "BPSK" or "QPSK"
}


# ─────────────────────────────────────────────────────────────────────────────
# RF helper functions  (all accept NumPy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def fspl_db(range_m: np.ndarray | float, frequency_hz: float) -> np.ndarray | float:
    """Free-Space Path Loss [dB].

    FSPL = 20·log₁₀(d) + 20·log₁₀(f) + 20·log₁₀(4π/c)
    """
    return (
        20.0 * np.log10(range_m)
        + 20.0 * np.log10(frequency_hz)
        + 20.0 * np.log10(4.0 * np.pi / SPEED_OF_LIGHT)
    )


def noise_power_dbw(temp_k: float, bandwidth_hz: float) -> float:
    """Thermal noise power [dBW].  N = k·T·B"""
    return 10.0 * np.log10(BOLTZMANN_K * temp_k * bandwidth_hz)


def link_budget(range_m: np.ndarray | float, params: dict) -> tuple:
    """Compute (SNR_dB, Pr_dBW, Pn_dBW) from slant range and link params.

    Friis:  Pr = Pt + Gt + Gr − FSPL − L_misc
    """
    loss  = fspl_db(range_m, params["frequency_hz"])
    Pr    = (
        params["tx_power_dbw"]
        + params["tx_gain_dbi"]
        + params["rx_gain_dbi"]
        - loss
        - params["misc_losses_db"]
    )
    Pn    = noise_power_dbw(params["system_temp_k"], params["bandwidth_hz"])
    snr   = Pr - Pn
    return snr, Pr, np.full_like(Pr, Pn) if isinstance(Pr, np.ndarray) else Pn


def compute_ber(snr_db: np.ndarray | float, modulation: str) -> np.ndarray | float:
    """Bit Error Rate for BPSK or QPSK.

    BPSK:  BER = ½·erfc(√Eb/N0)
    QPSK:  BER = ½·erfc(√(Eb/N0 / 2))
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    if modulation == "BPSK":
        return 0.5 * erfc(np.sqrt(np.maximum(snr_lin, 0.0)))
    else:  # QPSK
        return 0.5 * erfc(np.sqrt(np.maximum(snr_lin / 2.0, 0.0)))


def doppler_shift_hz(range_rate_km_s: np.ndarray | float, frequency_hz: float) -> np.ndarray | float:
    """Doppler frequency shift [Hz].

    f_d = −f₀ · (dρ/dt) / c
    Negative → satellite approaching; Positive → satellite receding.
    """
    return -frequency_hz * (range_rate_km_s * 1.0e3) / SPEED_OF_LIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    duration_minutes: int = 90,
    step_seconds: int = 60,
    link_params: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full end-to-end simulation for all satellites.

    Parameters
    ----------
    duration_minutes : simulation window length
    step_seconds     : time resolution
    link_params      : RF link parameters (defaults to LINK_PARAMS)

    Returns
    -------
    dict mapping satellite name → DataFrame with columns:
        time, lat, lon, alt_km, elevation_deg, range_km,
        snr_db, ber, doppler_hz, rx_power_dbw, noise_power_dbw, visible
    """
    if link_params is None:
        link_params = LINK_PARAMS.copy()

    # ── Skyfield setup ──────────────────────────────────────────────────────
    ts = load.timescale(builtin=True)
    gs = wgs84.latlon(
        GROUND_STATION["lat"],
        GROUND_STATION["lon"],
        elevation_m=GROUND_STATION["alt"],
    )

    # ── Time vector ─────────────────────────────────────────────────────────
    t0        = datetime.utcnow().replace(microsecond=0)
    n_pts     = max(2, int(duration_minutes * 60 / step_seconds))
    datetimes = [t0 + timedelta(seconds=i * step_seconds) for i in range(n_pts)]

    sky_t = ts.utc(
        [t.year   for t in datetimes],
        [t.month  for t in datetimes],
        [t.day    for t in datetimes],
        [t.hour   for t in datetimes],
        [t.minute for t in datetimes],
        [t.second for t in datetimes],
    )

    results: dict[str, pd.DataFrame] = {}

    for sat_name, (line1, line2) in TLE_DATABASE.items():
        sat = EarthSatellite(line1, line2, sat_name, ts)

        # ── Geocentric subpoint ──────────────────────────────────────────────
        geo = sat.at(sky_t)
        sub = wgs84.subpoint(geo)

        lats    = np.asarray(sub.latitude.degrees)
        lons    = np.asarray(sub.longitude.degrees)
        alts_km = np.asarray(sub.elevation.km)

        # ── Topocentric (from ground station) ────────────────────────────────
        diff = sat - gs
        top  = diff.at(sky_t)

        el_obj, _, dist_obj = top.altaz()
        el_deg = np.asarray(el_obj.degrees)
        rng_km = np.asarray(dist_obj.km)
        rng_m  = rng_km * 1.0e3

        # Range rate: ρ̇ = (r̂ · v) / |r|   [km/s]
        pos_km  = top.position.km       # shape (3, n)
        vel_kms = top.velocity.km_per_s # shape (3, n)
        rr_kms  = np.einsum("ij,ij->j", pos_km, vel_kms) / rng_km

        # ── Link budget & derived quantities ─────────────────────────────────
        snr_db, Pr_dbw, Pn_dbw = link_budget(rng_m, link_params)
        ber     = compute_ber(snr_db, link_params["modulation"])
        dop_hz  = doppler_shift_hz(rr_kms, link_params["frequency_hz"])

        # Floor BER to avoid −∞ on log scale
        ber = np.maximum(ber, 1.0e-15)

        visible = el_deg > 0.0

        results[sat_name] = pd.DataFrame({
            "time":            datetimes,
            "lat":             lats,
            "lon":             lons,
            "alt_km":          alts_km,
            "elevation_deg":   el_deg,
            "range_km":        rng_km,
            "snr_db":          snr_db,
            "ber":             ber,
            "doppler_hz":      dop_hz,
            "rx_power_dbw":    Pr_dbw,
            "noise_power_dbw": Pn_dbw if np.isscalar(Pn_dbw) else Pn_dbw,
            "visible":         visible,
        })

    return results


def summary_stats(sim_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-satellite summary statistics for the simulation."""
    rows = []
    for name, df in sim_data.items():
        vis = df[df["visible"]]
        rows.append({
            "Satellite":      name,
            "Visibility (%)": round(df["visible"].mean() * 100, 1),
            "Passes":         int((df["visible"].astype(int).diff() == 1).sum()),
            "Peak SNR (dB)":  round(vis["snr_db"].max(), 1)   if not vis.empty else float("nan"),
            "Min SNR (dB)":   round(vis["snr_db"].min(), 1)   if not vis.empty else float("nan"),
            "Min BER":        f"{vis['ber'].min():.2e}"        if not vis.empty else "N/A",
            "Max Doppler (kHz)": round(vis["doppler_hz"].abs().max() / 1e3, 2) if not vis.empty else float("nan"),
            "Max Alt (km)":   round(df["alt_km"].max(), 1),
        })
    return pd.DataFrame(rows)
