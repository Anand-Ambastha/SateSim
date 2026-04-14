"""
run_simulation.py — Standalone Simulation Runner
=================================================
Run the full simulation pipeline from the command line without Streamlit:

    python run_simulation.py [--duration 90] [--step 60] [--modulation BPSK]

Outputs:
  • outputs/data/       — CSV files for each satellite + combined
  • outputs/plots/      — PNG figures
  • outputs/satellite_comm_report.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Satellite Communication Simulation — standalone runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration",    type=int,   default=90,    help="Simulation window (minutes)")
    parser.add_argument("--step",        type=int,   default=60,    help="Time step (seconds)")
    parser.add_argument("--freq",        type=float, default=2.0,   help="Carrier frequency (GHz)")
    parser.add_argument("--tx-power",    type=float, default=10.0,  help="Tx power (dBW)")
    parser.add_argument("--tx-gain",     type=float, default=10.0,  help="Tx antenna gain (dBi)")
    parser.add_argument("--rx-gain",     type=float, default=5.0,   help="Rx antenna gain (dBi)")
    parser.add_argument("--temp",        type=float, default=290.0, help="System noise temperature (K)")
    parser.add_argument("--bandwidth",   type=float, default=1000.0,help="Noise bandwidth (kHz)")
    parser.add_argument("--losses",      type=float, default=3.0,   help="Miscellaneous losses (dB)")
    parser.add_argument("--modulation",  type=str,   default="BPSK",choices=["BPSK", "QPSK"])
    parser.add_argument("--no-report",   action="store_true",       help="Skip PDF report generation")
    return parser.parse_args()


def banner(text: str) -> None:
    print(f"\n{'─'*60}\n  {text}\n{'─'*60}")


def main() -> None:
    args = parse_args()

    banner("🛰️  Satellite Communication Simulation System")
    print(f"  Duration    : {args.duration} minutes")
    print(f"  Time step   : {args.step} seconds")
    print(f"  Frequency   : {args.freq} GHz")
    print(f"  Modulation  : {args.modulation}")

    # ── Import after banner so startup errors surface cleanly ───────────────
    try:
        from core   import run_simulation, summary_stats, GROUND_STATION
        from report import save_all_plots, generate_pdf_report
    except ImportError as exc:
        print(f"\n[ERROR] Missing dependency: {exc}")
        print("        Run:  pip install -r requirements.txt")
        sys.exit(1)

    link_params = {
        "frequency_hz":    args.freq * 1e9,
        "tx_power_dbw":    args.tx_power,
        "tx_gain_dbi":     args.tx_gain,
        "rx_gain_dbi":     args.rx_gain,
        "system_temp_k":   args.temp,
        "bandwidth_hz":    args.bandwidth * 1e3,
        "misc_losses_db":  args.losses,
        "modulation":      args.modulation,
    }

    # ── Step 1: Orbital propagation + link budget ───────────────────────────
    banner("Step 1/4  Computing orbital positions and link budgets…")
    t0 = time.perf_counter()
    sim_data = run_simulation(
        duration_minutes=args.duration,
        step_seconds=args.step,
        link_params=link_params,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s  —  {len(sim_data)} satellites propagated")

    # ── Step 2: Summary statistics ──────────────────────────────────────────
    banner("Step 2/4  Summary statistics")
    stats = summary_stats(sim_data)
    print(stats.to_string(index=False))

    # ── Step 3: Save CSVs ───────────────────────────────────────────────────
    banner("Step 3/4  Exporting CSV data…")
    os.makedirs("outputs/data", exist_ok=True)
    combined = []
    for name, df in sim_data.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = f"outputs/data/{safe}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df)} rows)")
        tmp = df.copy()
        tmp.insert(0, "satellite", name)
        combined.append(tmp)

    import pandas as pd
    combined_path = "outputs/data/all_satellites.csv"
    pd.concat(combined, ignore_index=True).to_csv(combined_path, index=False)
    print(f"  Saved: {combined_path}")

    # ── Step 4: Plots + PDF ─────────────────────────────────────────────────
    banner("Step 4/4  Rendering plots and generating PDF report…")
    if args.no_report:
        print("  Saving plots only (--no-report flag set)…")
        plots = save_all_plots(sim_data)
        for key, path in plots.items():
            print(f"  Plot: {path}")
    else:
        pdf_path = generate_pdf_report(sim_data, link_params)
        print(f"  PDF report saved: {pdf_path}")

    banner("✅  Simulation complete")
    print(f"  Output directory: outputs/")
    print()


if __name__ == "__main__":
    main()
