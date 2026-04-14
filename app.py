
from __future__ import annotations

import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core import (
    run_simulation,
    summary_stats,
    GROUND_STATION,
    LINK_PARAMS,
    TLE_DATABASE,
)
from report import generate_pdf_report, save_all_plots

# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PLOTS = "outputs/plots"
OUTPUT_DATA  = "outputs/data"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_DATA,  exist_ok=True)

# Satellite colour palette
COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


# ─────────────────────────────────────────────────────────────────────────────
# Data persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(sim_data: dict[str, pd.DataFrame]) -> None:
    """Export per-satellite CSVs and a combined CSV."""
    for name, df in sim_data.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(f"{OUTPUT_DATA}/{safe}.csv", index=False)

    combined = []
    for name, df in sim_data.items():
        tmp = df.copy()
        tmp.insert(0, "satellite", name)
        combined.append(tmp)
    pd.concat(combined, ignore_index=True).to_csv(
        f"{OUTPUT_DATA}/all_satellites.csv", index=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figure builders
# ─────────────────────────────────────────────────────────────────────────────

_GEO_LAYOUT = dict(
    showland=True,        landcolor="rgb(28, 32, 48)",
    showocean=True,       oceancolor="rgb(12, 18, 35)",
    showcoastlines=True,  coastlinecolor="rgb(90, 110, 170)",
    showlakes=True,       lakecolor="rgb(12, 18, 35)",
    lonaxis=dict(
        showgrid=True,
        gridcolor="rgba(80, 80, 120, 0.3)"
    ),
    lataxis=dict(
        showgrid=True,
        gridcolor="rgba(80, 80, 120, 0.3)"
    ),

    bgcolor="rgb(8,10,22)",
    projection_type="natural earth",
    showframe=False,
)

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,12,24,0)",
    plot_bgcolor="rgba(10,12,24,0)",
    font=dict(family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(80,80,120,0.4)", borderwidth=1),
    margin=dict(l=0, r=10, t=40, b=10),
)


def fig_world_map(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()

    for i, (name, df) in enumerate(sim_data.items()):
        col = COLOURS[i % len(COLOURS)]
        # Ground track line
        fig.add_trace(go.Scattergeo(
            lat=df["lat"], lon=df["lon"],
            mode="lines",
            name=name,
            line=dict(width=1.8, color=col),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Lat: %{lat:.2f}°<br>"
                "Lon: %{lon:.2f}°<extra></extra>"
            ),
        ))
        # Current position marker (last point)
        fig.add_trace(go.Scattergeo(
            lat=[df["lat"].iloc[-1]], lon=[df["lon"].iloc[-1]],
            mode="markers",
            showlegend=False,
            marker=dict(size=11, color=col, symbol="circle",
                        line=dict(width=2, color="white")),
            hovertemplate=f"<b>{name}</b><br>Current position<extra></extra>",
        ))

    # Ground station
    fig.add_trace(go.Scattergeo(
        lat=[GROUND_STATION["lat"]], lon=[GROUND_STATION["lon"]],
        mode="markers+text",
        name="Delhi GS",
        text=["  Delhi GS"],
        textposition="middle right",
        textfont=dict(size=11, color="yellow"),
        marker=dict(size=14, color="yellow", symbol="triangle-up",
                    line=dict(width=1.5, color="white")),
        hovertemplate="<b>Delhi Ground Station</b><br>"
                      f"Lat: {GROUND_STATION['lat']}°N<br>"
                      f"Lon: {GROUND_STATION['lon']}°E<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="🌍  Satellite Ground Tracks", font=dict(size=16), x=0.5),
        geo=_GEO_LAYOUT,
        paper_bgcolor="rgb(8,10,22)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(80,80,120,0.4)", borderwidth=1),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _time_series_base(title: str, yaxis_title: str, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.02),
        xaxis_title="Time (UTC)",
        yaxis_title=yaxis_title,
        yaxis_type="log" if log_y else "linear",
        height=320,
        **_DARK_LAYOUT,
    )
    return fig


def fig_snr(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = _time_series_base("📶  SNR vs Time (visible passes)", "SNR (dB)")
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if vis.empty:
            continue
        fig.add_trace(go.Scatter(
            x=vis["time"], y=vis["snr_db"],
            mode="lines", name=name,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2.2),
            hovertemplate=f"<b>{name}</b><br>SNR: %{{y:.1f}} dB<extra></extra>",
        ))
    return fig


def fig_ber(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = _time_series_base("⚠️  BER vs Time (visible passes)", "BER", log_y=True)
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if vis.empty:
            continue
        fig.add_trace(go.Scatter(
            x=vis["time"], y=vis["ber"],
            mode="lines", name=name,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2.2),
            hovertemplate=f"<b>{name}</b><br>BER: %{{y:.2e}}<extra></extra>",
        ))
    return fig


def fig_doppler(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = _time_series_base("🔄  Doppler Shift vs Time", "Doppler Shift (kHz)")
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if vis.empty:
            continue
        fig.add_trace(go.Scatter(
            x=vis["time"], y=vis["doppler_hz"] / 1e3,
            mode="lines", name=name,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2.2),
            hovertemplate=f"<b>{name}</b><br>Δf: %{{y:.2f}} kHz<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(160,160,160,0.4)", line_width=1)
    return fig


def fig_elevation(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = _time_series_base("📡  Elevation Angle vs Time", "Elevation (°)")
    for i, (name, df) in enumerate(sim_data.items()):
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["elevation_deg"],
            mode="lines", name=name,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2.2),
            hovertemplate=f"<b>{name}</b><br>El: %{{y:.1f}}°<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(220,80,80,0.7)",
                  annotation_text="Horizon  ", annotation_position="bottom right")
    return fig


def fig_range(sim_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = _time_series_base("📏  Slant Range vs Time", "Range (km)")
    for i, (name, df) in enumerate(sim_data.items()):
        vis = df[df["visible"]]
        if vis.empty:
            continue
        fig.add_trace(go.Scatter(
            x=vis["time"], y=vis["range_km"],
            mode="lines", name=name,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2.2),
            hovertemplate=f"<b>{name}</b><br>Range: %{{y:.0f}} km<extra></extra>",
        ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Satellite Comm Simulator",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] {
          background: linear-gradient(135deg, #080a16 0%, #0d1228 50%, #0a0e1e 100%);
      }
      [data-testid="stSidebar"] {
          background: rgba(14, 18, 38, 0.95);
          border-right: 1px solid rgba(80, 100, 180, 0.2);
      }
      .metric-card {
          background: linear-gradient(135deg, rgba(26,40,80,0.8), rgba(16,24,56,0.8));
          border: 1px solid rgba(80,120,220,0.3);
          border-radius: 10px;
          padding: 14px 18px;
          margin-bottom: 8px;
      }
      .section-title {
          color: #7eb8ff;
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 6px;
      }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
      div[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.75; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <span style="font-size:2.6rem;">🛰️</span>
        <h1 style="margin:4px 0; color:#e0eaff; font-size:2rem; font-weight:800;
                   letter-spacing:-0.02em;">Satellite Communication Simulator</h1>
        <p style="color:#7090c0; font-size:0.95rem; margin:0;">
            Real-time orbital mechanics &nbsp;·&nbsp; RF link budget &nbsp;·&nbsp;
            Delhi Ground Station (28.61°N, 77.21°E)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Parameters")
        st.markdown('<p class="section-title">Time Window</p>', unsafe_allow_html=True)
        duration = st.slider("Duration (minutes)", 30, 180, 90, 10)
        step     = st.selectbox("Time step (seconds)", [30, 60, 120], index=1)

        st.markdown('<p class="section-title">RF Link Budget</p>', unsafe_allow_html=True)
        freq_ghz = st.number_input("Frequency (GHz)", 0.4, 30.0, 2.0, 0.1, format="%.1f")
        tx_power = st.number_input("Tx Power (dBW)",  0.0, 30.0, 10.0, 1.0, format="%.1f")
        tx_gain  = st.number_input("Tx Gain (dBi)",   0.0, 50.0, 10.0, 1.0, format="%.1f")
        rx_gain  = st.number_input("Rx Gain (dBi)",   0.0, 50.0,  5.0, 1.0, format="%.1f")
        sys_temp = st.number_input("System Temp (K)", 50.0, 2000.0, 290.0, 10.0, format="%.0f")
        bw_khz   = st.number_input("Bandwidth (kHz)",  1.0, 50000.0, 1000.0, 100.0, format="%.0f")
        losses   = st.number_input("Misc Losses (dB)", 0.0, 20.0, 3.0, 0.5, format="%.1f")
        mod      = st.selectbox("Modulation", ["BPSK", "QPSK"])

        st.markdown("---")
        run_btn = st.button("🚀  Run Simulation", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("**Satellites tracked:**")
        for name in TLE_DATABASE:
            st.markdown(f"• {name}")
        st.markdown(f"**Ground station:** {GROUND_STATION['name']}")

    # ── Session state ────────────────────────────────────────────────────────
    if "sim_data"    not in st.session_state:
        st.session_state.sim_data    = None
    if "link_params" not in st.session_state:
        st.session_state.link_params = None

    # ── Run simulation ───────────────────────────────────────────────────────
    if run_btn:
        params = {
            "frequency_hz":    freq_ghz * 1e9,
            "tx_power_dbw":    tx_power,
            "tx_gain_dbi":     tx_gain,
            "rx_gain_dbi":     rx_gain,
            "system_temp_k":   sys_temp,
            "bandwidth_hz":    bw_khz * 1e3,
            "misc_losses_db":  losses,
            "modulation":      mod,
        }

        with st.spinner("⚙️  Propagating orbits and computing link budgets…"):
            sim_data = run_simulation(
                duration_minutes=duration,
                step_seconds=int(step),
                link_params=params,
            )

        st.session_state.sim_data    = sim_data
        st.session_state.link_params = params
        save_csv(sim_data)
        st.success(
            f"✅  Simulation complete — {len(sim_data)} satellites · "
            f"{duration} min · {int(step)}s steps"
        )

    # ── Dashboard ────────────────────────────────────────────────────────────
    if st.session_state.sim_data is not None:
        sim_data    = st.session_state.sim_data
        link_params = st.session_state.link_params or LINK_PARAMS

        # Summary metric cards
        stats_df = summary_stats(sim_data)
        cols = st.columns(len(sim_data) + 1)
        cols[0].metric("🏠 Ground Station", "Delhi, India", f"28.61°N · 77.21°E")
        for i, row in stats_df.iterrows():
            with cols[i + 1]:
                st.metric(
                    label=row["Satellite"],
                    value=f"Vis: {row['Visibility (%)']:.0f}%",
                    delta=f"SNR max: {row['Peak SNR (dB)']} dB",
                )

        st.divider()

        # ── World Map ─────────────────────────────────────────────────────
        st.plotly_chart(fig_world_map(sim_data), use_container_width=True)

        st.divider()

        # ── Time-series grid ──────────────────────────────────────────────
        st.markdown("### 📊  Communication Link Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_snr(sim_data),       use_container_width=True)
            st.plotly_chart(fig_doppler(sim_data),   use_container_width=True)
        with c2:
            st.plotly_chart(fig_ber(sim_data),       use_container_width=True)
            st.plotly_chart(fig_elevation(sim_data), use_container_width=True)

        st.plotly_chart(fig_range(sim_data), use_container_width=True)

        st.divider()

        # ── Summary stats table ───────────────────────────────────────────
        st.markdown("### 📋  Simulation Summary")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ── Raw data explorer ─────────────────────────────────────────────
        with st.expander("🔬  Raw Data Explorer"):
            sat_sel = st.selectbox("Select Satellite", list(sim_data.keys()))
            df_sel  = sim_data[sat_sel]

            filter_vis = st.checkbox("Show visible passes only", value=False)
            display_df = df_sel[df_sel["visible"]] if filter_vis else df_sel

            st.dataframe(
                display_df.style.format({
                    "lat": "{:.3f}", "lon": "{:.3f}", "alt_km": "{:.1f}",
                    "elevation_deg": "{:.2f}", "range_km": "{:.1f}",
                    "snr_db": "{:.2f}", "ber": "{:.2e}",
                    "doppler_hz": "{:.1f}", "rx_power_dbw": "{:.2f}",
                }),
                use_container_width=True, height=300,
            )

            buf = io.BytesIO()
            display_df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Download CSV",
                buf.getvalue(),
                file_name=f"{sat_sel.replace(' ', '_')}.csv",
                mime="text/csv",
            )

        # ── PDF Report ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📄  Report Generation")
        col_r1, col_r2 = st.columns([2, 3])
        with col_r1:
            if st.button("📄  Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Rendering plots and building PDF…"):
                    pdf_path = generate_pdf_report(sim_data, link_params)
                st.session_state.pdf_path = pdf_path
                st.success(f"Report saved → `{pdf_path}`")

        with col_r2:
            if "pdf_path" in st.session_state and os.path.exists(st.session_state.pdf_path):
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        f.read(),
                        file_name="satellite_comm_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #5070a0;">
            <div style="font-size:4rem; margin-bottom:16px;">🛰️</div>
            <h3 style="color:#7090c0; font-weight:600;">Ready to Simulate</h3>
            <p style="max-width:480px; margin:0 auto; line-height:1.6;">
                Configure link-budget parameters in the sidebar and click
                <strong style="color:#e0eaff;">Run Simulation</strong> to compute
                orbital tracks, SNR, BER, and Doppler for three satellites
                simultaneously.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
