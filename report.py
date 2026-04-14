"""
report.py — Satellite Communication Simulation: PDF Report Generator
====================================================================
Responsibilities:
  • Render all time-series / map plots as PNG via Matplotlib
  • Assemble a multi-section professional PDF using ReportLab

Sections:
  1. Cover / title
  2. Abstract
  3. Introduction
  4. System Model  (equations, parameters table)
  5. Results       (all plots + commentary)
  6. Discussion
  7. Conclusion
  8. References

Author: Anand Ambastha
"""

from __future__ import annotations

import os
import math
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from reportlab.lib                 import colors
from reportlab.lib.pagesizes       import A4
from reportlab.lib.styles          import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units           import cm
from reportlab.lib.enums           import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus            import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether,
)

from core import GROUND_STATION, LINK_PARAMS, SPEED_OF_LIGHT, BOLTZMANN_K

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PLOTS  = "outputs/plots"
OUTPUT_REPORT = "outputs"

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]    # one per satellite

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib style helpers
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK  = "#0d0d1a"
BG_AX    = "#111122"
GRID_COL = "#2a2a44"
TICK_COL = "#aabbcc"
LABEL_COL= "#ccddef"


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(BG_AX)
    ax.set_title(title,   color=LABEL_COL, fontsize=10, pad=6, fontweight="bold")
    ax.set_xlabel(xlabel, color=TICK_COL,  fontsize=8)
    ax.set_ylabel(ylabel, color=TICK_COL,  fontsize=8)
    ax.tick_params(colors=TICK_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)


def _new_figure(nrows: int = 1, figsize: tuple = (10, 3.6)) -> tuple:
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, facecolor=BG_DARK,
                             squeeze=False)
    fig.subplots_adjust(hspace=0.38)
    return fig, [ax for row in axes for ax in row]


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_PLOTS, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot renderers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ground_track(sim_data: dict) -> str:
    fig, (ax,) = _new_figure(figsize=(12, 4.5))

    # World border approximation (simple blue-grey fill)
    ax.axhspan(-90, 90, color="#1a1a3a", alpha=0.3)

    for i, (name, df) in enumerate(sim_data.items()):
        col = PALETTE[i % len(PALETTE)]
        ax.plot(df["lon"], df["lat"], color=col, linewidth=1.5, label=name, alpha=0.9)
        ax.scatter(df["lon"].iloc[-1], df["lat"].iloc[-1],
                   color=col, s=55, zorder=6, edgecolors="white", linewidths=0.6)

    # Ground station
    ax.scatter(GROUND_STATION["lon"], GROUND_STATION["lat"],
               color="#f1c40f", s=140, marker="^", zorder=7,
               edgecolors="white", linewidths=0.8, label="Delhi GS")
    ax.annotate(" Delhi GS",
                xy=(GROUND_STATION["lon"], GROUND_STATION["lat"]),
                color="#f1c40f", fontsize=8, fontweight="bold")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    leg = ax.legend(facecolor="#1a1a2e", labelcolor="white",
                    fontsize=8, loc="lower left", framealpha=0.7)
    _style_ax(ax, "Satellite Ground Tracks", "Longitude (°)", "Latitude (°)")
    return _save(fig, "ground_track.png")


def _plot_elevation(sim_data: dict) -> str:
    fig, (ax,) = _new_figure()
    for i, (name, df) in enumerate(sim_data.items()):
        ax.plot(df["time"], df["elevation_deg"],
                color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=name)
    ax.axhline(0, color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.7, label="Horizon")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.7)
    fig.autofmt_xdate(rotation=20, ha="right")
    _style_ax(ax, "Elevation Angle vs Time", "Time (UTC)", "Elevation (°)")
    return _save(fig, "elevation.png")


def _plot_snr(sim_data: dict) -> str:
    fig, (ax,) = _new_figure()
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if not vis.empty:
            ax.plot(vis["time"], vis["snr_db"],
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=name)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.7)
    fig.autofmt_xdate(rotation=20, ha="right")
    _style_ax(ax, "Signal-to-Noise Ratio vs Time (visible passes)", "Time (UTC)", "SNR (dB)")
    return _save(fig, "snr.png")


def _plot_ber(sim_data: dict) -> str:
    fig, (ax,) = _new_figure()
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if not vis.empty:
            ax.semilogy(vis["time"], vis["ber"],
                        color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=name)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all"))
    ax.grid(True, which="minor", color=GRID_COL, linewidth=0.3, alpha=0.5)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.7)
    fig.autofmt_xdate(rotation=20, ha="right")
    _style_ax(ax, "Bit Error Rate vs Time (visible passes, log scale)",
              "Time (UTC)", "BER")
    return _save(fig, "ber.png")


def _plot_doppler(sim_data: dict) -> str:
    fig, (ax,) = _new_figure()
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if not vis.empty:
            ax.plot(vis["time"], vis["doppler_hz"] / 1e3,
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=name)
    ax.axhline(0, color="#95a5a6", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.7)
    fig.autofmt_xdate(rotation=20, ha="right")
    _style_ax(ax, "Doppler Shift vs Time (visible passes)",
              "Time (UTC)", "Doppler Shift (kHz)")
    return _save(fig, "doppler.png")


def _plot_range(sim_data: dict) -> str:
    fig, (ax,) = _new_figure()
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if not vis.empty:
            ax.plot(vis["time"], vis["range_km"],
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=name)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.7)
    fig.autofmt_xdate(rotation=20, ha="right")
    _style_ax(ax, "Slant Range vs Time (visible passes)",
              "Time (UTC)", "Slant Range (km)")
    return _save(fig, "range.png")


def save_all_plots(sim_data: dict) -> dict[str, str]:
    """Render and persist all plots. Returns {key: filepath}."""
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)
    return {
        "ground_track": _plot_ground_track(sim_data),
        "elevation":    _plot_elevation(sim_data),
        "snr":          _plot_snr(sim_data),
        "ber":          _plot_ber(sim_data),
        "doppler":      _plot_doppler(sim_data),
        "range":        _plot_range(sim_data),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ReportLab PDF assembly
# ─────────────────────────────────────────────────────────────────────────────

# Brand colours
C_NAVY   = colors.HexColor("#1a3a5c")
C_BLUE   = colors.HexColor("#2c5f8a")
C_LBLUE  = colors.HexColor("#3a80b8")
C_GREY   = colors.HexColor("#5a6a7a")
C_LGREY  = colors.HexColor("#c8d4e0")
C_ROW_A  = colors.HexColor("#eef4fb")
C_ROW_B  = colors.white
C_HEAD   = C_NAVY


def _build_styles() -> dict:
    base = getSampleStyleSheet()

    def ps(name, **kw):
        parent = kw.pop("parent", "Normal")
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": ps("ct", parent="Title",
                           fontSize=24, textColor=C_NAVY,
                           alignment=TA_CENTER, spaceAfter=6, leading=28),
        "cover_sub":   ps("cs", fontSize=12, textColor=C_BLUE,
                           alignment=TA_CENTER, spaceAfter=4),
        "cover_meta":  ps("cm", fontSize=8.5, textColor=C_GREY,
                           alignment=TA_CENTER),
        "h1":          ps("h1", parent="Heading1",
                           fontSize=14, textColor=C_NAVY,
                           spaceBefore=16, spaceAfter=6, leading=18),
        "h2":          ps("h2", parent="Heading2",
                           fontSize=11, textColor=C_BLUE,
                           spaceBefore=10, spaceAfter=4),
        "body":        ps("bd", fontSize=9.5, leading=15, alignment=TA_JUSTIFY),
        "equation":    ps("eq", fontSize=9, fontName="Courier",
                           alignment=TA_CENTER, spaceBefore=5, spaceAfter=5,
                           textColor=C_NAVY, backColor=colors.HexColor("#f0f6ff"),
                           borderPadding=(4, 8, 4, 8)),
        "caption":     ps("cap", fontSize=8, alignment=TA_CENTER,
                           textColor=C_GREY, spaceAfter=10),
        "footer":      ps("ft",  fontSize=7.5, textColor=C_GREY,
                           alignment=TA_CENTER),
    }


def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=1.2,
                      color=C_NAVY, spaceAfter=6, spaceBefore=2)


def _img(path: str, width: float = 15 * cm) -> Image | Paragraph:
    if not path or not os.path.exists(path):
        styles = getSampleStyleSheet()
        return Paragraph(f"[Figure not found: {path}]", styles["Italic"])
    im = Image(path, width=width, height=width * 0.40)
    im.hAlign = "CENTER"
    return im


def _param_table(link_params: dict) -> Table:
    rows = [
        ["Parameter", "Symbol", "Value"],
        ["Carrier frequency",      "f₀",    f"{link_params.get('frequency_hz', 2e9)/1e9:.2f} GHz"],
        ["Transmit power",         "Pₜ",    f"{link_params.get('tx_power_dbw', 10):.1f} dBW"],
        ["Tx antenna gain",        "Gₜ",    f"{link_params.get('tx_gain_dbi', 10):.1f} dBi"],
        ["Rx antenna gain",        "Gᵣ",    f"{link_params.get('rx_gain_dbi', 5):.1f} dBi"],
        ["System noise temp.",     "T_sys", f"{link_params.get('system_temp_k', 290):.0f} K"],
        ["Noise bandwidth",        "B",     f"{link_params.get('bandwidth_hz', 1e6)/1e3:.0f} kHz"],
        ["Miscellaneous losses",   "L_misc",f"{link_params.get('misc_losses_db', 3):.1f} dB"],
        ["Modulation",             "—",     str(link_params.get("modulation", "BPSK"))],
        ["Speed of light",         "c",     "2.998 × 10⁸ m/s"],
        ["Boltzmann constant",     "k",     "1.381 × 10⁻²³ J/K"],
    ]
    col_w = [7 * cm, 3 * cm, 6 * cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_HEAD),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_ROW_A, C_ROW_B]),
        ("GRID",           (0, 0), (-1, -1), 0.5, C_LGREY),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (0, -1), 8),
    ]))
    return t


def _satellite_summary_table(sim_data: dict) -> Table:
    import numpy as _np
    rows = [["Satellite", "Visibility", "Passes", "Peak SNR", "Min BER", "Max |Doppler|"]]
    for name, df in sim_data.items():
        vis = df[df["visible"]]
        v_pct   = df["visible"].mean() * 100
        passes  = int((df["visible"].astype(int).diff() == 1).sum())
        pk_snr  = f"{vis['snr_db'].max():.1f} dB"  if not vis.empty else "N/A"
        mn_ber  = f"{vis['ber'].min():.2e}"         if not vis.empty else "N/A"
        mx_dop  = f"{vis['doppler_hz'].abs().max()/1e3:.2f} kHz" if not vis.empty else "N/A"
        rows.append([name, f"{v_pct:.1f}%", str(passes), pk_snr, mn_ber, mx_dop])

    col_w = [4.8*cm, 2.2*cm, 1.8*cm, 2.8*cm, 2.5*cm, 3*cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_ROW_A, C_ROW_B]),
        ("GRID",           (0, 0), (-1, -1), 0.5, C_LGREY),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (0, -1), 8),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Page numbering via canvas callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_GREY)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Satellite Communication Simulation System — Confidential")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm,
                           f"Page {doc.page}")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Master report builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(sim_data: dict, link_params: dict | None = None) -> str:
    """Generate the full PDF report and return the output file path."""
    if link_params is None:
        link_params = LINK_PARAMS.copy()

    os.makedirs(OUTPUT_PLOTS, exist_ok=True)
    os.makedirs(OUTPUT_REPORT, exist_ok=True)

    # Render all matplotlib figures
    plots = save_all_plots(sim_data)

    pdf_path = os.path.join(OUTPUT_REPORT, "satellite_comm_report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.8 * cm,
        bottomMargin=2.8 * cm,
    )

    S = _build_styles()
    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Spacer(1, 2.0 * cm),
        Paragraph("Satellite Communication", S["cover_title"]),
        Paragraph("Simulation System", S["cover_title"]),
        Spacer(1, 0.4 * cm),
        _hr(),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Orbital Mechanics · RF Link Budget · Doppler Analysis · BER Characterisation",
            S["cover_sub"],
        ),
        Spacer(1, 1.2 * cm),
        Paragraph(
            f"Generated: {datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')}",
            S["cover_meta"],
        ),
        Paragraph(
            f"Ground Station: {GROUND_STATION['name']} "
            f"({GROUND_STATION['lat']}°N, {GROUND_STATION['lon']}°E, "
            f"{GROUND_STATION['alt']:.0f} m ASL)",
            S["cover_meta"],
        ),
        Paragraph(
            f"Satellites tracked: {', '.join(sim_data.keys())}",
            S["cover_meta"],
        ),
        Spacer(1, 0.5 * cm),
        _hr(),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ABSTRACT
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("Abstract", S["h1"]),
        Paragraph(
            "This report presents results from an end-to-end satellite communication "
            "simulation that integrates real-time orbital mechanics with a complete "
            "radio-frequency (RF) link budget model. Using Two-Line Element (TLE) data "
            "and the Skyfield library (SGP4 propagator), the system computes precise "
            "geocentric positions, topocentric elevation angles, and slant ranges for "
            "three Earth-orbiting satellites — ISS (ZARYA), NOAA-18, and Aqua — as "
            "observed from the Delhi, India ground station over a configurable time "
            "window. A closed-form link budget yields received signal power, thermal "
            "noise power, signal-to-noise ratio (SNR), bit error rate (BER) for BPSK "
            "or QPSK modulation, and Doppler frequency shift at each time step. "
            "Results are presented as interactive time-series plots and exported as "
            "publication-quality PNG figures and structured CSV data files.",
            S["body"],
        ),
        Spacer(1, 0.4 * cm),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 2. INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("1  Introduction", S["h1"]),
        Paragraph(
            "Satellite communication systems underpin modern global telecommunications, "
            "remote sensing, navigation, and scientific data collection. Accurate "
            "simulation of such systems demands the tight integration of two distinct "
            "domains: orbital mechanics — to predict where a spacecraft will be at "
            "any instant — and RF propagation modelling — to characterise how a signal "
            "degrades across the vast free-space path between the satellite and the "
            "ground receiver.",
            S["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The simulation presented here addresses both domains simultaneously. "
            "Three low-Earth-orbit (LEO) satellites with diverse inclinations and "
            "altitudes are tracked concurrently from the Delhi ground station "
            "(28.61°N, 77.21°E). The International Space Station (ISS, ~408 km, "
            "51.6° inclination) represents a crewed platform; NOAA-18 (~854 km, "
            "99.0° inclination) is a polar-orbiting meteorological satellite; and "
            "Aqua (~705 km, 98.2° inclination) is a NASA Earth-science observatory. "
            "Together they provide a representative cross-section of LEO communication "
            "scenarios.",
            S["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The remainder of the report is organised as follows. Section 2 describes "
            "the mathematical system model. Section 3 documents the simulation "
            "parameters. Section 4 presents and discusses the computed results. "
            "Section 5 offers a broader discussion of the findings, and Section 6 "
            "concludes the report.",
            S["body"],
        ),
        Spacer(1, 0.3 * cm),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SYSTEM MODEL
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("2  System Model", S["h1"]),

        # 2.1 Orbital mechanics
        Paragraph("2.1  Orbital Mechanics (SGP4 / TLE)", S["h2"]),
        Paragraph(
            "Satellite state vectors are derived from TLE data using the SGP4 "
            "semi-analytic propagator embedded in the Skyfield Python library. "
            "The geocentric position vector r_sat is projected onto the WGS-84 "
            "reference ellipsoid to obtain geodetic latitude φ, longitude λ, and "
            "altitude h. The topocentric position vector (from the ground station) is:",
            S["body"],
        ),
        Paragraph("Δr = r_sat − r_gs", S["equation"]),
        Paragraph(
            "The slant range ρ [m] and elevation angle θ [°] follow directly from "
            "the Euclidean norm and the spherical altitude angle of Δr, respectively. "
            "A satellite is considered visible when θ > 0°. The radial range rate "
            "(used for Doppler computation) is obtained via the dot product:",
            S["body"],
        ),
        Paragraph("dρ/dt = (Δr · Δv) / |Δr|   [km/s]", S["equation"]),

        # 2.2 Link budget
        Paragraph("2.2  RF Link Budget", S["h2"]),
        Paragraph(
            "The received signal power Pᵣ at the ground station is given by the "
            "Friis transmission equation expressed in decibels:",
            S["body"],
        ),
        Paragraph(
            "Pᵣ [dBW] = Pₜ + Gₜ + Gᵣ − FSPL − L_misc",
            S["equation"],
        ),
        Paragraph(
            "where Pₜ is the transmit power, Gₜ and Gᵣ are the transmit and "
            "receive antenna gains, L_misc collects pointing error and atmospheric "
            "absorption losses, and FSPL is the Free-Space Path Loss:",
            S["body"],
        ),
        Paragraph(
            "FSPL [dB] = 20 log₁₀(ρ) + 20 log₁₀(f₀) + 20 log₁₀(4π/c)",
            S["equation"],
        ),
        Paragraph(
            "The thermal noise power floor at the receiver is:",
            S["body"],
        ),
        Paragraph("N [dBW] = 10 log₁₀(k · T_sys · B)", S["equation"]),
        Paragraph(
            "where k = 1.381 × 10⁻²³ J/K, T_sys is the system noise temperature, "
            "and B is the noise-equivalent bandwidth. The SNR at the receiver is:",
            S["body"],
        ),
        Paragraph("SNR [dB] = Pᵣ − N", S["equation"]),

        # 2.3 BER
        Paragraph("2.3  Bit Error Rate", S["h2"]),
        Paragraph(
            "For BPSK modulation, the theoretical BER over an AWGN channel is:",
            S["body"],
        ),
        Paragraph(
            "BER_BPSK = ½ · erfc(√SNR_linear)",
            S["equation"],
        ),
        Paragraph("For QPSK, the per-bit SNR is halved:", S["body"]),
        Paragraph("BER_QPSK = ½ · erfc(√(SNR_linear / 2))", S["equation"]),

        # 2.4 Doppler
        Paragraph("2.4  Doppler Shift", S["h2"]),
        Paragraph(
            "The Doppler frequency shift f_d [Hz] due to radial motion between "
            "the satellite and the ground station is:",
            S["body"],
        ),
        Paragraph("f_d = −f₀ · (dρ/dt) / c", S["equation"]),
        Paragraph(
            "A negative f_d indicates the satellite is approaching (decreasing "
            "slant range); a positive f_d indicates recession. At 2 GHz, LEO "
            "satellites with orbital velocities of ~7 km/s produce peak Doppler "
            "shifts of ±20–50 kHz.",
            S["body"],
        ),
        Spacer(1, 0.3 * cm),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SIMULATION PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("3  Simulation Parameters", S["h1"]),
        Paragraph(
            "Table 1 lists the RF link-budget parameters used in this simulation.",
            S["body"],
        ),
        Spacer(1, 0.2 * cm),
        _param_table(link_params),
        Paragraph("Table 1 — RF link-budget parameters.", S["caption"]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Table 2 summarises the key performance indicators for each satellite "
            "derived from the simulation run.",
            S["body"],
        ),
        Spacer(1, 0.2 * cm),
        _satellite_summary_table(sim_data),
        Paragraph("Table 2 — Per-satellite simulation summary.", S["caption"]),
        Spacer(1, 0.3 * cm),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 5. RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    story += [Paragraph("4  Results", S["h1"])]

    figures = [
        ("4.1  Satellite Ground Tracks", "ground_track",
         "Figure 1 — Ground tracks for all three satellites over the simulation window. "
         "The yellow triangle marks the Delhi ground station.",
         "Figure 1 shows the Mercator-projected ground tracks of all three satellites. "
         "The ISS ground track covers mid-latitude regions due to its 51.6° inclination, "
         "while NOAA-18 and Aqua sweep near-polar corridors in Sun-synchronous orbits."),

        ("4.2  Elevation Angle", "elevation",
         "Figure 2 — Elevation angle of each satellite as seen from Delhi. "
         "Arcs above the dashed horizon line (0°) represent active contact windows.",
         "The elevation profile captures the characteristic bell-shaped arc for each "
         "satellite pass. Contact windows are brief (typically 5–12 minutes for LEO) "
         "and the maximum elevation angle per pass determines the best link quality."),

        ("4.3  Signal-to-Noise Ratio", "snr",
         "Figure 3 — SNR during visible passes. Higher elevation → shorter range → "
         "lower FSPL → better SNR.",
         "SNR is dominated by free-space path loss, which scales with 20 log₁₀(ρ). "
         "Consequently, SNR peaks near the point of closest approach and degrades "
         "rapidly as the satellite approaches the horizon. The ISS, at the lowest "
         "altitude (~408 km), achieves the highest per-pass SNR."),

        ("4.4  Bit Error Rate", "ber",
         "Figure 4 — BER (log scale) during visible passes. BER is lowest at "
         "maximum elevation (closest range).",
         "The inverse relationship between BER and SNR is clearly visible: "
         "each dB of SNR improvement reduces the BER by approximately an order of "
         "magnitude in the relevant operating regime. BPSK delivers a 3 dB SNR "
         "advantage over QPSK for the same BER threshold."),

        ("4.5  Doppler Shift", "doppler",
         "Figure 5 — Doppler shift (kHz) during visible passes. The shift crosses "
         "zero at the time of closest approach (TCA).",
         "The S-shaped Doppler curve is the diagnostic signature of an LEO pass. "
         "The zero-crossing marks the TCA. Peak |Δf| values of 20–50 kHz at 2 GHz "
         "necessitate closed-loop frequency tracking in the receiver."),

        ("4.6  Slant Range", "range",
         "Figure 6 — Slant range (km) during visible passes.",
         "Slant range determines both FSPL and the Doppler rate of change (d²ρ/dt²). "
         "The ISS's low orbit produces the shortest minimum ranges, while NOAA-18 "
         "and Aqua — at higher altitudes — maintain longer but more stable ranges."),
    ]

    for sec_title, key, caption, commentary in figures:
        story += [
            Paragraph(sec_title, S["h2"]),
            Paragraph(commentary, S["body"]),
            Spacer(1, 0.15 * cm),
            _img(plots.get(key, ""), width=14.5 * cm),
            Paragraph(caption, S["caption"]),
        ]

    story += [PageBreak()]

    # ══════════════════════════════════════════════════════════════════════════
    # 6. DISCUSSION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("5  Discussion", S["h1"]),
        Paragraph(
            "The simulation confirms the fundamental geometry-driven behaviour of LEO "
            "satellite links. The most influential parameter is slant range ρ, which "
            "determines FSPL via a 20 log₁₀(ρ) relationship. This means that doubling "
            "the range degrades SNR by ~6 dB and can raise the BER by several orders "
            "of magnitude. The strong elevation-angle dependence of link quality "
            "underscores the importance of antenna pointing accuracy and the need for "
            "minimum elevation angle masks (typically 5°–10°) in operational ground "
            "stations.",
            S["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The Doppler analysis reveals that peak frequency offsets of 20–50 kHz "
            "at 2 GHz are common for LEO satellites. Without compensation, such offsets "
            "would cause severe inter-carrier interference in multi-carrier schemes "
            "(OFDM) and degrade coherent demodulation performance. Practical systems "
            "employ open-loop Doppler pre-compensation based on the predicted TLE "
            "trajectory and closed-loop AFC (Automatic Frequency Control) loops in the "
            "receiver to mitigate residual offsets.",
            S["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The link budget presented here is intentionally conservative. No Forward "
            "Error Correction (FEC) coding gain is included; modern turbo codes or "
            "Low-Density Parity-Check (LDPC) codes typically provide 5–10 dB of coding "
            "gain, substantially extending the usable elevation-angle range. Additionally, "
            "the 3 dB miscellaneous loss is a coarse approximation; detailed system "
            "design would decompose this into individual contributions from pointing "
            "error, polarisation mismatch, atmospheric absorption (rain fade, tropospheric "
            "scintillation), and cable/connector losses.",
            S["body"],
        ),
        Spacer(1, 0.3 * cm),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Paragraph("6  Conclusion", S["h1"]),
        Paragraph(
            "An end-to-end satellite communication simulation system has been developed "
            "and validated. The system seamlessly integrates SGP4 orbital propagation "
            "via Skyfield, a full Friis-equation link budget, BPSK/QPSK BER analysis, "
            "and Doppler characterisation into a unified interactive platform built with "
            "Streamlit. Simulation results for ISS, NOAA-18, and Aqua observed from "
            "the Delhi ground station confirm the strong dependence of link quality on "
            "elevation angle and slant range, and demonstrate the magnitude of Doppler "
            "shifts that must be compensated in operational systems.",
            S["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The modular Python architecture (core.py / app.py / report.py) facilitates "
            "straightforward extension to incorporate additional fidelity: multi-path "
            "fading channels, realistic antenna gain patterns, rain-fade margins, "
            "multiple-access protocols (FDMA, TDMA, CDMA), or machine-learning-based "
            "scheduling algorithms. The system is therefore a versatile foundation "
            "for both academic study and preliminary engineering analysis of LEO "
            "satellite communication links.",
            S["body"],
        ),
        Spacer(1, 0.6 * cm),
        _hr(),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "References: "
            "T. S. Kelso, Celestrak TLE Repository; "
            "B. Rhodes, Skyfield Astronomy Library (rhodesmill.org/skyfield); "
            "T. Pratt &amp; C. Bostian, Satellite Communications, Wiley 1986; "
            "J. G. Proakis, Digital Communications, 5th ed., McGraw-Hill 2007; "
            "ITU-R P.618-13, Propagation data for satellite systems, 2017.",
            S["footer"],
        ),
    ]

    # ── Build PDF ────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return pdf_path
