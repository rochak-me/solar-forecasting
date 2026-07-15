"""
╔══════════════════════════════════════════════════════════════════╗
║  Solar Irradiance Forecasting — Interactive Dashboard            ║
║  Koshi Province, Nepal  |  LSTM / GRU / CNN-LSTM / PI-LSTM      ║
║  Python equivalent of RA's R Shiny app                           ║
╚══════════════════════════════════════════════════════════════════╝

Run with:
    streamlit run solar_forecasting_ui.py

Requirements:
    pip install streamlit plotly pandas
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecasting — Koshi Province",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1117 100%); }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2544 40%, #1a1a2e 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(251,191,36,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { color: #f0f9ff; font-size: 1.8rem; font-weight: 700; margin: 0; }
.hero p  { color: #94a3b8; font-size: 0.95rem; margin: 0.4rem 0 0 0; }
.badge {
    display: inline-block;
    background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4);
    color: #fbbf24; border-radius: 20px; padding: 2px 12px;
    font-size: 0.75rem; font-weight: 600; margin-right: 6px;
}
.metric-card {
    background: linear-gradient(135deg, #1e2d3d 0%, #162033 100%);
    border: 1px solid rgba(99,179,237,0.15); border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,179,237,0.4); }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #63b3ed; }
.metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
.metric-card .model { font-size: 0.75rem; color: #94a3b8; }
.metric-card-gold .value { color: #fbbf24 !important; }
.metric-card-green .value { color: #48bb78 !important; }
.metric-card-red .value   { color: #fc8181 !important; }
.section-header {
    color: #e2e8f0; font-size: 1.1rem; font-weight: 600;
    padding: 0.6rem 0; border-bottom: 1px solid rgba(99,179,237,0.15);
    margin-bottom: 1rem; margin-top: 1.2rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.1);
}
.log-box {
    background: #0d1117; border: 1px solid rgba(99,179,237,0.15);
    border-radius: 8px; padding: 1rem;
    font-family: 'Courier New', monospace; font-size: 0.78rem;
    color: #a0aec0; max-height: 300px; overflow-y: auto;
    white-space: pre-wrap; line-height: 1.5;
}
.chart-card {
    background: rgba(14,20,30,0.7);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
}
.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg,#fbbf24,#f59e0b);
    color: #0a0e1a; border-radius: 20px; padding: 2px 10px;
    font-size: 0.7rem; font-weight: 700; margin-left: 6px;
}
.monitor-row {
    background: rgba(14,20,30,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 10px; padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem; display: flex; align-items: center;
    gap: 1rem;
}
.status-done  { color: #48bb78; font-weight: 700; }
.status-run   { color: #fbbf24; font-weight: 700; }
.status-wait  { color: #64748b; }
.monitor-card {
    background: linear-gradient(135deg,#1a2535,#111827);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.monitor-card .big  { font-size: 2.2rem; font-weight: 800; color: #63b3ed; line-height: 1; }
.monitor-card .sub  { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
_LOCAL_ROOT = Path(r"D:\Semester Project and Notes\3rd Year Project")
if _LOCAL_ROOT.exists():
    PROJECT_ROOT = _LOCAL_ROOT
else:
    # Fallback to current directory for Cloud Deployment (GitHub)
    PROJECT_ROOT = Path(".")

RESULTS_CSV     = PROJECT_ROOT / "ra_pipeline_replication_v2" / "ra_pipeline_all_results.csv"
TRAINING_SCRIPT = PROJECT_ROOT / "Antigravity folder" / "ra_pipeline_replication_v2.py"
CONFIG_FILE     = PROJECT_ROOT / "ra_pipeline_replication_v2" / "ui_config.json"

SITE_INFO = {
    "Combined (All 6 Sites)": {
        "folder":  None,
        "label":   "Combined — All 6 Koshi Sites (current default)",
        "combined": True,
    },
    "Site 1": {"folder": PROJECT_ROOT / "Site1 data",  "label": "Site 1 (Koshi)", "combined": False},
    "Site 2": {"folder": PROJECT_ROOT / "Site 2 data", "label": "Site 2 (Koshi)", "combined": False},
    "Site 3": {"folder": PROJECT_ROOT / "Site 3 Data", "label": "Site 3 (Koshi)", "combined": False},
    "Site 4": {"folder": PROJECT_ROOT / "Site 4 data", "label": "Site 4 (Koshi)", "combined": False},
    "Site 5": {"folder": PROJECT_ROOT / "Site 5 data", "label": "Site 5 (Koshi)", "combined": False},
    "Site 6": {"folder": PROJECT_ROOT / "Site 6 data", "label": "Site 6 (Koshi)", "combined": False},
}

MODEL_COLORS = {
    "LSTM":     "#63b3ed",
    "GRU":      "#48bb78",
    "CNN-LSTM": "#f6ad55",
    "PI-LSTM":  "#fc8181",
}
PIPELINE_ORDER = ["Pipeline A","Pipeline B","Pipeline C","Pipeline D","Pipeline F","Pipeline G"]

PIPELINE_INFO = {
    "A": "Sparse Seasonal (17 anchors: 10 yearly + 7 daily)",
    "B": "Dense 168h window — 1 week of hourly history",
    "C": "Dense 336h window — 2 weeks of hourly history",
    "D": "Dense 24h window  — 1 day of hourly history",
    "F": "Seasonal-window: N yearly lags only (no trailing) — 2-week window specialist",
    "G": "Seasonal-window: N yearly lags + 14-day trailing hourly block",
}

EST_MINUTES = {"A": 12, "B": 33, "C": 60, "D": 8, "F": 15, "G": 25}

PLOT_BG   = "rgba(0,0,0,0)"
CHART_BG  = "rgba(14,20,30,0.6)"
FONT_CFG  = dict(color="#e2e8f0", family="Inter")
GRID_CLR  = "rgba(255,255,255,0.05)"

# ─── Session state ────────────────────────────────────────────────────────────
for key, val in [("training_running", False), ("log_lines", []),
                 ("expected_runs", 0), ("config_ts", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helper: apply dark layout to any figure ────────────────────────────────
def dark_layout(fig, title="", height=420, showlegend=True, yrange=None):
    upd = dict(
        title=title, height=height,
        paper_bgcolor=PLOT_BG, plot_bgcolor=CHART_BG,
        font=FONT_CFG,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
        margin=dict(t=55, b=30, l=10, r=10),
        showlegend=showlegend,
    )
    if yrange:
        upd["yaxis"] = dict(range=yrange, gridcolor=GRID_CLR)
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor=GRID_CLR)
    if not yrange:
        fig.update_yaxes(gridcolor=GRID_CLR)
    return fig

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>☀️ Solar Irradiance Forecasting Dashboard</h1>
  <p>Koshi Province, Nepal &nbsp;·&nbsp; 6-Site LSTM / GRU / CNN-LSTM / PI-LSTM Experiment</p>
  <br/>
  <span class="badge">LSTM</span><span class="badge">GRU</span>
  <span class="badge">CNN-LSTM</span><span class="badge">PI-LSTM</span>
  <span class="badge">Pipelines A → D</span>
  <span class="badge">Pipelines F &amp; G (Seasonal)</span>
  <span class="badge">1h &amp; 24h Horizons</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Experiment Configuration")
    st.markdown("---")

    # Site
    st.markdown("#### 📍 Site")
    selected_site = st.selectbox(
        "Choose site:",
        list(SITE_INFO.keys()),
        index=0,
        help="'Combined' uses df_clean already loaded in the notebook (current mode). "
             "Individual sites load a single site CSV."
    )
    site_info   = SITE_INFO[selected_site]
    is_combined = site_info["combined"]
    site_folder = site_info["folder"]

    if is_combined:
        st.success("🌐 **Combined mode** — uses all 6 sites merged (current default)")
    else:
        exists = site_folder.exists() if site_folder else False
        st.caption(f"{'✅ Found' if exists else '❌ Missing'}: `{site_folder.name if site_folder else 'N/A'}`")

    st.markdown("---")

    # Pipelines A-D
    st.markdown("#### 🔀 Pipelines A – D")
    cols = st.columns(2)
    run_A = cols[0].checkbox("A — Sparse",  value=True, help=PIPELINE_INFO["A"])
    run_B = cols[0].checkbox("B — 168h",    value=True, help=PIPELINE_INFO["B"])
    run_C = cols[1].checkbox("C — 336h",    value=True, help=PIPELINE_INFO["C"])
    run_D = cols[1].checkbox("D — 24h",     value=True, help=PIPELINE_INFO["D"])

    st.markdown("#### 🌊 Pipelines F & G (Seasonal Window)")
    fg_cols = st.columns(2)
    run_F = fg_cols[0].checkbox("F — Yearly only",    value=False, help=PIPELINE_INFO["F"])
    run_G = fg_cols[1].checkbox("G — + Trailing 14d", value=False, help=PIPELINE_INFO["G"])

    if run_F or run_G:
        st.markdown("**🗃 Seasonal Window Config**")
        split_doy     = st.slider("Window start (Day-of-Year)", 1, 355, 170,
                                  help="DOY 170 = June 19 (monsoon onset). "
                                       "DOY 1=Jan 1, 91=Apr 1, 182=Jul 1")
        window_days   = st.slider("Window length (days)", 7, 30, 14)
        n_yearly_lags = st.slider("Yearly lags (N)", 1, 10, 5,
                                  help="GHI/Kt from same hour, 1y to Ny back")
        trailing_days = st.slider("Trailing days (G only)", 7, 28, 14,
                                  help="Hours of raw history BEFORE window start (G only)")
        month_approx  = pd.Timestamp('2024-01-01') + pd.Timedelta(days=split_doy-1)
        st.caption(f"≈ {month_approx.strftime('%B %d')} each year")
    else:
        split_doy = 170; window_days = 14; n_yearly_lags = 5; trailing_days = 14

    selected_pipelines = [k for k, v in zip(["A","B","C","D","F","G"],
                                             [run_A,run_B,run_C,run_D,run_F,run_G]) if v]

    st.markdown("---")

    # Horizons
    st.markdown("#### ⏱️ Forecast Horizons")
    hc = st.columns(2)
    run_1h  = hc[0].checkbox("1h ahead",  value=True)
    run_24h = hc[1].checkbox("24h ahead", value=True)
    selected_horizons = [h for h, v in zip(["1h","24h"], [run_1h, run_24h]) if v]

    st.markdown("---")

    # Hyperparameters
    st.markdown("#### 🧠 Training Hyperparameters")
    max_epochs = st.slider("Max epochs",        10, 150, 50,  step=5)
    patience   = st.slider("Early-stop patience", 3,  20,  8)
    batch_size = st.slider("Batch size",         32, 512, 256, step=32)

    st.markdown("---")

    est = sum(EST_MINUTES.get(p, 0) for p in selected_pipelines) * len(selected_horizons)
    st.info(f"⏳ Estimated runtime: **~{est} min**")

    run_clicked = st.button(
        "🚀 Generate Config & Run",
        disabled=(not selected_pipelines or not selected_horizons),
        width='stretch', type="primary",
    )

# ─── Save config on Run ───────────────────────────────────────────────────────
if run_clicked:
    is_combined = SITE_INFO[selected_site]["combined"]
    site_folder  = SITE_INFO[selected_site]["folder"]

    config = {
        "site_name":      selected_site,
        "is_combined":    is_combined,
        "site_folder":    None if is_combined else str(site_folder),
        "all_site_folders": {
            k: str(v["folder"]) for k, v in SITE_INFO.items()
            if not v["combined"] and v["folder"] is not None
        },
        "run_pipelines":  selected_pipelines,
        "run_horizons":   selected_horizons,
        "max_epochs":     max_epochs,
        "patience":       patience,
        "batch_size":     batch_size,
        "split_doy":      split_doy,
        "window_days":    window_days,
        "n_yearly_lags":  n_yearly_lags,
        "trailing_days":  trailing_days,
    }
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # Store expected run count for the monitor
    n_models = 4  # LSTM, GRU, CNN-LSTM, PI-LSTM
    st.session_state.expected_runs = len(selected_pipelines) * len(selected_horizons) * n_models
    st.session_state.config_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.log_lines = [
        f"✅ Config saved → {CONFIG_FILE}",
        f"   Site      : {selected_site}",
        f"   Pipelines : {selected_pipelines}",
        f"   Horizons  : {selected_horizons}",
        f"   Epochs    : {max_epochs}  Patience: {patience}",
        f"   Expected  : {st.session_state.expected_runs} model runs",
        "─" * 55,
        "👇 Copy the command below and run it in a terminal:",
    ]
    st.toast("Config saved! See Training Monitor tab to track progress.", icon="✅")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_results, tab_deepdive, tab_monitor, tab_log, tab_howto = st.tabs([
    "📊 Results", "🔬 Deep Dive", "🔄 Training Monitor", "📋 Training Log", "📖 How to Use"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not RESULTS_CSV.exists():
        st.info(f"No results found at:\n`{RESULTS_CSV}`\n\nRun training first.")
        st.stop()

    df_raw = pd.read_csv(RESULTS_CSV).dropna(how="all")

    # ── Sidebar filters for this tab ─────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    hz_filter = fc1.multiselect("🕐 Horizon",  df_raw["Horizon"].unique().tolist(),
                                default=df_raw["Horizon"].unique().tolist(), key="hz1")
    pl_filter = fc2.multiselect("🔀 Pipeline", df_raw["Pipeline"].unique().tolist(),
                                default=df_raw["Pipeline"].unique().tolist(), key="pl1")
    mo_filter = fc3.multiselect("🤖 Model",    df_raw["Model"].unique().tolist(),
                                default=df_raw["Model"].unique().tolist(), key="mo1")
    df = df_raw[df_raw["Horizon"].isin(hz_filter) &
                df_raw["Pipeline"].isin(pl_filter) &
                df_raw["Model"].isin(mo_filter)].reset_index(drop=True)

    if df.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1 — KPI Scorecards
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Best Performance Highlights</div>',
                unsafe_allow_html=True)

    kpi_defs = [
        ("1h",  "R2_pp",   "Best R² — 1h",      True,  "{:.4f}", "metric-card-gold"),
        ("24h", "R2_pp",   "Best R² — 24h",     True,  "{:.4f}", "metric-card-gold"),
        ("1h",  "RMSE_pp", "Lowest RMSE — 1h",  False, "{:.1f} W/m²", "metric-card-green"),
        ("24h", "RMSE_pp", "Lowest RMSE — 24h", False, "{:.1f} W/m²", "metric-card-green"),
        ("1h",  "MAE_pp",  "Lowest MAE — 1h",   False, "{:.1f} W/m²", ""),
        ("24h", "MAE_pp",  "Lowest MAE — 24h",  False, "{:.1f} W/m²", ""),
    ]
    kpi_cols = st.columns(6)
    for (hz, col, label, maximize, fmt, cls), kc in zip(kpi_defs, kpi_cols):
        sub = df[df["Horizon"] == hz]
        if sub.empty or col not in sub.columns: continue
        idx  = sub[col].idxmax() if maximize else sub[col].idxmin()
        best = sub.loc[idx]
        kc.markdown(f"""
        <div class="metric-card {cls}">
          <div class="value">{fmt.format(best[col])}</div>
          <div class="label">{label}</div>
          <div class="model">{best['Model']} · {best['Pipeline']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2 — R² Heatmap (Pipeline × Model)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌡️ R² Heatmap — Pipeline × Model</div>',
                unsafe_allow_html=True)
    heat_cols = st.columns(len(df["Horizon"].unique()))
    for col_i, hz in enumerate(sorted(df["Horizon"].unique())):
        sub = df[df["Horizon"] == hz]
        pivot = sub.pivot_table(index="Model", columns="Pipeline",
                                values="R2_pp", aggfunc="mean")
        # Sort columns by pipeline order
        ordered_cols = [c for c in PIPELINE_ORDER if c in pivot.columns]
        pivot = pivot[ordered_cols] if ordered_cols else pivot
        if pivot.empty: continue

        z_min = max(0, pivot.values.min() - 0.05)
        fig = px.imshow(
            pivot, text_auto=".4f",
            color_continuous_scale=[[0,"#0d1117"],[0.3,"#1e3a5f"],[0.7,"#3b82f6"],[1,"#fbbf24"]],
            zmin=z_min, zmax=1.0,
            title=f"R² (post-processed) — {hz} Horizon",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=FONT_CFG, height=300, margin=dict(t=50, b=10, l=10, r=10),
            coloraxis_colorbar=dict(tickfont=dict(color="#e2e8f0")),
        )
        fig.update_traces(textfont=dict(color="white", size=11))
        heat_cols[col_i].plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Grouped bar: R² by Pipeline coloured by Model (per horizon)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 R² Comparison — All Pipelines & Models</div>',
                unsafe_allow_html=True)
    for hz in sorted(df["Horizon"].unique()):
        sub = df[df["Horizon"] == hz].copy()
        # Sort pipeline labels
        sub["_order"] = sub["Pipeline"].apply(lambda x: PIPELINE_ORDER.index(x) if x in PIPELINE_ORDER else 99)
        sub = sub.sort_values("_order")

        fig = go.Figure()
        for model in ["LSTM","GRU","CNN-LSTM","PI-LSTM"]:
            m = sub[sub["Model"] == model]
            if m.empty: continue
            fig.add_trace(go.Bar(
                name=model, x=m["Pipeline"], y=m["R2_pp"],
                marker_color=MODEL_COLORS.get(model, "#94a3b8"),
                marker_line_width=0,
                text=m["R2_pp"].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
                textfont=dict(size=9, color="white"),
                hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.4f}<extra></extra>",
            ))

        dark_layout(fig, title=f"R² (post-processed) — {hz} Horizon", height=440,
                    yrange=[max(0, sub["R2_pp"].min()-0.05), 1.02])
        fig.update_layout(barmode="group",
                          xaxis=dict(title="Pipeline", gridcolor=GRID_CLR))
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Bubble Chart: RMSE vs R² bubble = Violations
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 RMSE vs R² Bubble Chart (bubble = Violations)</div>',
                unsafe_allow_html=True)
    bubble_cols = st.columns(len(df["Horizon"].unique()))
    for col_i, hz in enumerate(sorted(df["Horizon"].unique())):
        sub = df[df["Horizon"] == hz].copy()
        if "Violations" not in sub.columns:
            sub["Violations"] = 0
        sub["Violations_sz"] = sub["Violations"].clip(lower=1) ** 0.5 + 5

        fig = px.scatter(
            sub, x="R2_pp", y="RMSE_pp",
            size="Violations_sz", color="Model",
            color_discrete_map=MODEL_COLORS,
            hover_data={"Pipeline": True, "Model": True,
                        "R2_pp": "{:.4f}", "RMSE_pp": "{:.1f}",
                        "Violations": True, "Violations_sz": False},
            symbol="Pipeline",
            title=f"RMSE vs R² — {hz} | bubble = physics violations",
        )
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.2)")))
        dark_layout(fig, height=420)
        bubble_cols[col_i].plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Horizon Degradation: 1h vs 24h R² drop
    # ──────────────────────────────────────────────────────────────────────────
    if "1h" in df["Horizon"].unique() and "24h" in df["Horizon"].unique():
        st.markdown('<div class="section-header">📉 Horizon Degradation — R² Drop: 1h → 24h</div>',
                    unsafe_allow_html=True)

        df_1h  = df[df["Horizon"] == "1h" ].set_index(["Pipeline","Model"])["R2_pp"].rename("R2_1h")
        df_24h = df[df["Horizon"] == "24h"].set_index(["Pipeline","Model"])["R2_pp"].rename("R2_24h")
        deg_df = pd.concat([df_1h, df_24h], axis=1).dropna().reset_index()
        deg_df["Drop"] = (deg_df["R2_1h"] - deg_df["R2_24h"]).round(4)
        deg_df["Label"] = deg_df["Pipeline"] + "<br>" + deg_df["Model"]
        deg_df = deg_df.sort_values("Drop", ascending=False)

        fig = go.Figure()
        for model in ["LSTM","GRU","CNN-LSTM","PI-LSTM"]:
            m = deg_df[deg_df["Model"] == model]
            if m.empty: continue
            fig.add_trace(go.Bar(
                name=model,
                x=m["Pipeline"] + " / " + m["Model"],
                y=m["Drop"],
                marker_color=MODEL_COLORS.get(model, "#94a3b8"),
                marker_line_width=0,
                text=m["Drop"].apply(lambda v: f"−{v:.3f}"),
                textposition="outside",
                textfont=dict(size=9, color="white"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "1h R²: %{customdata[0]:.4f}<br>"
                    "24h R²: %{customdata[1]:.4f}<br>"
                    "Drop: %{y:.4f}<extra></extra>"
                ),
                customdata=m[["R2_1h","R2_24h"]].values,
            ))
        dark_layout(fig, title="R² Degradation (1h → 24h) — smaller bar = more robust model", height=440)
        fig.update_layout(barmode="group",
                          yaxis_title="R² Drop (lower = better)",
                          xaxis=dict(tickangle=-30, gridcolor=GRID_CLR))
        st.plotly_chart(fig, use_container_width=True)

        # Summary degradation table
        deg_pivot = deg_df.pivot_table(index="Model", columns="Pipeline", values="Drop", aggfunc="mean")
        st.caption("Average R² drop per model (lower = more robust to longer horizons)")
        st.dataframe(
            deg_pivot.style
                .format("{:.4f}")
                .background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Physics Compliance (Violations)
    # ──────────────────────────────────────────────────────────────────────────
    if "Violations" in df.columns:
        st.markdown('<div class="section-header">⚠️ Physics Compliance — Violations per Model</div>',
                    unsafe_allow_html=True)
        st.caption("Violations = predictions below 0, above clearsky ceiling, or during night hours. "
                   "Lower is better. PI-LSTM is specifically trained to minimise these.")

        viol_cols = st.columns(len(df["Horizon"].unique()))
        for col_i, hz in enumerate(sorted(df["Horizon"].unique())):
            sub = df[df["Horizon"] == hz].copy()
            sub["_order"] = sub["Pipeline"].apply(
                lambda x: PIPELINE_ORDER.index(x) if x in PIPELINE_ORDER else 99)
            sub = sub.sort_values(["_order", "Model"])

            fig = go.Figure()
            for model in ["LSTM","GRU","CNN-LSTM","PI-LSTM"]:
                m = sub[sub["Model"] == model]
                if m.empty: continue
                fig.add_trace(go.Bar(
                    name=model,
                    x=m["Pipeline"],
                    y=m["Violations"],
                    marker_color=MODEL_COLORS.get(model,"#94a3b8"),
                    marker_line_width=0,
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y} violations<extra></extra>",
                ))
            dark_layout(fig, title=f"Physics Violations — {hz}", height=380)
            fig.update_layout(barmode="group",
                              yaxis_title="# Violations (lower = better)",
                              xaxis_title="Pipeline")
            viol_cols[col_i].plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Raw vs Post-Processed: Shift (improvement)
    # ──────────────────────────────────────────────────────────────────────────
    if "Shift" in df.columns:
        st.markdown('<div class="section-header">🔧 Post-Processing Benefit — RMSE Shift</div>',
                    unsafe_allow_html=True)
        st.caption("Shift = RMSE improvement from physics post-processing (zeroing night, clipping negatives). "
                   "Higher shift = post-processing helped more.")

        fig_shift = px.strip(
            df, x="Pipeline", y="Shift", color="Model",
            color_discrete_map=MODEL_COLORS,
            facet_col="Horizon",
            stripmode="overlay",
            hover_data=["Model","Pipeline","Horizon","R2_pp","RMSE_pp"],
            title="Post-Processing Shift (W/m² RMSE improvement) — each dot = one model run",
        )
        fig_shift.update_traces(marker=dict(size=10, opacity=0.85,
                                             line=dict(width=1, color="rgba(255,255,255,0.3)")))
        dark_layout(fig_shift, height=420)
        st.plotly_chart(fig_shift, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 8 — Best Model per Pipeline (winner table)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🥇 Best Model per Pipeline</div>',
                unsafe_allow_html=True)
    winner_rows = []
    for hz in sorted(df["Horizon"].unique()):
        for pl in df["Pipeline"].unique():
            sub = df[(df["Horizon"] == hz) & (df["Pipeline"] == pl)]
            if sub.empty: continue
            best_idx  = sub["R2_pp"].idxmax()
            best_rmse = sub["RMSE_pp"].idxmin()
            row = sub.loc[best_idx]
            winner_rows.append({
                "Horizon": hz, "Pipeline": pl,
                "🏆 Best Model (R²)": row["Model"],
                "R² (pp)":   round(row["R2_pp"],   4),
                "RMSE (pp)": round(row["RMSE_pp"],  1),
                "MAE (pp)":  round(row.get("MAE_pp", float("nan")), 1),
                "Violations": int(row.get("Violations", 0)),
            })
    if winner_rows:
        winner_df = pd.DataFrame(winner_rows)
        st.dataframe(
            winner_df.style
                .format({"R² (pp)": "{:.4f}", "RMSE (pp)": "{:.1f}", "MAE (pp)": "{:.1f}"})
                .applymap(lambda v: "color:#fbbf24;font-weight:700"
                          if isinstance(v, str) and v == "PI-LSTM" else "",
                          subset=["🏆 Best Model (R²)"]),
            use_container_width=True, height=320,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Full Results Table with Download
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Full Results Table</div>',
                unsafe_allow_html=True)
    show_cols = ["Pipeline","Horizon","Model","R2_pp","RMSE_pp","MAE_pp",
                 "R2_raw","RMSE_raw","Violations","Shift"]
    st.dataframe(
        df[[c for c in show_cols if c in df.columns]]
        .sort_values(["Horizon","R2_pp"], ascending=[True, False])
        .style.format({"R2_pp":"{:.4f}","R2_raw":"{:.4f}","RMSE_pp":"{:.1f}",
                       "RMSE_raw":"{:.1f}","MAE_pp":"{:.1f}","Shift":"{:.2f}"}),
        use_container_width=True, height=450,
    )
    st.download_button("⬇️ Download Results CSV",
                       df.to_csv(index=False).encode(),
                       "solar_forecasting_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_deepdive:
    if not RESULTS_CSV.exists():
        st.info("Run training first to see deep-dive charts.")
        st.stop()

    df_raw2 = pd.read_csv(RESULTS_CSV).dropna(how="all")

    fc1, fc2 = st.columns(2)
    hz2 = fc1.selectbox("Horizon", sorted(df_raw2["Horizon"].unique()), key="hz_dd")
    metric2 = fc2.selectbox("Metric", ["R2_pp","RMSE_pp","MAE_pp","R2_raw","RMSE_raw"], key="met_dd")

    df2 = df_raw2[df_raw2["Horizon"] == hz2].copy()
    df2["_order"] = df2["Pipeline"].apply(lambda x: PIPELINE_ORDER.index(x) if x in PIPELINE_ORDER else 99)
    df2 = df2.sort_values("_order")

    # ── Radar chart: model profile across metrics ─────────────────────────────
    st.markdown('<div class="section-header">🕸️ Model Radar — Strength Profile</div>',
                unsafe_allow_html=True)
    st.caption("Each axis is a performance dimension. Larger polygon = stronger model.")

    radar_metrics = ["R2_pp", "R2_raw"]
    if "Violations" in df2.columns:
        # Invert violations and RMSE so larger = better on radar
        df2["Viol_inv"]  = 1 / (1 + df2["Violations"])
        df2["RMSE_inv"]  = 1 / (1 + df2["RMSE_pp"] / 100)
        df2["Shift_norm"] = df2["Shift"].clip(0) / (df2["Shift"].max() + 1e-6)
        radar_dims = ["R2_pp","R2_raw","Viol_inv","RMSE_inv","Shift_norm"]
        radar_labels = ["R² (pp)","R² (raw)","Physics OK","RMSE score","PP Shift"]
    else:
        df2["RMSE_inv"] = 1 / (1 + df2["RMSE_pp"] / 100)
        radar_dims   = ["R2_pp","R2_raw","RMSE_inv"]
        radar_labels = ["R² (pp)","R² (raw)","RMSE score"]

    radar_fig = go.Figure()
    agg = df2.groupby("Model")[radar_dims].mean().reset_index()
    for _, row in agg.iterrows():
        vals = [row[d] for d in radar_dims]
        vals += [vals[0]]  # close polygon
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            fill='toself', name=row["Model"],
            line_color=MODEL_COLORS.get(row["Model"], "#94a3b8"),
            fillcolor=MODEL_COLORS.get(row["Model"], "#94a3b8").replace(")", ",0.15)").replace("rgb","rgba"),
        ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(14,20,30,0.6)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=GRID_CLR, tickfont=dict(color="#94a3b8", size=9)),
            angularaxis=dict(gridcolor=GRID_CLR, tickfont=dict(color="#e2e8f0")),
        ),
        paper_bgcolor=PLOT_BG, font=FONT_CFG, height=440,
        title=f"Model Radar Profile — {hz2} Horizon",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(t=60, b=20, l=40, r=40),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ── Parallel Coordinates ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">〰️ Parallel Coordinates — All Metrics at Once</div>',
                unsafe_allow_html=True)
    st.caption("Each line = one experiment run. Trace a line across all axes to see how one model performs everywhere.")

    model_idx = {m: i for i, m in enumerate(["LSTM","GRU","CNN-LSTM","PI-LSTM"])}
    df2["model_num"] = df2["Model"].map(model_idx).fillna(0)
    par_dims = [
        dict(label="R² (pp)",    values=df2["R2_pp"],    range=[0,1]),
        dict(label="RMSE (pp)",  values=df2["RMSE_pp"],  range=[df2["RMSE_pp"].min(), df2["RMSE_pp"].max()]),
        dict(label="MAE (pp)",   values=df2.get("MAE_pp", df2["RMSE_pp"]),
             range=[df2.get("MAE_pp", df2["RMSE_pp"]).min(), df2.get("MAE_pp", df2["RMSE_pp"]).max()]),
        dict(label="R² (raw)",   values=df2["R2_raw"],   range=[0,1]),
    ]
    if "Violations" in df2.columns:
        par_dims.append(dict(label="Violations", values=df2["Violations"],
                             range=[0, df2["Violations"].max()+1]))

    par_fig = go.Figure(go.Parcoords(
        line=dict(color=df2["model_num"],
                  colorscale=[[0,"#63b3ed"],[0.33,"#48bb78"],[0.67,"#f6ad55"],[1,"#fc8181"]],
                  showscale=True,
                  colorbar=dict(tickvals=[0,1,2,3],
                                ticktext=["LSTM","GRU","CNN-LSTM","PI-LSTM"],
                                tickfont=dict(color="#e2e8f0"), len=0.7)),
        dimensions=par_dims,
    ))
    par_fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=CHART_BG,
        font=FONT_CFG, height=420,
        title=f"Parallel Coordinates — {hz2} Horizon",
        margin=dict(t=60, b=20, l=80, r=80),
    )
    st.plotly_chart(par_fig, use_container_width=True)

    # ── RMSE vs R² scatter (both horizons) ───────────────────────────────────
    st.markdown('<div class="section-header">🎯 RMSE vs R² — Raw vs Post-Processed</div>',
                unsafe_allow_html=True)
    fig_scat = px.scatter(
        df_raw2, x="R2_raw", y="R2_pp",
        color="Model", symbol="Horizon",
        color_discrete_map=MODEL_COLORS,
        size=[12]*len(df_raw2),
        hover_data=["Pipeline","Model","Horizon","RMSE_pp","R2_pp","R2_raw"],
        title="Post-processing impact: dots above diagonal = PP improved R²",
    )
    # Diagonal reference line
    mn = min(df_raw2["R2_raw"].min(), df_raw2["R2_pp"].min()) - 0.02
    mx = max(df_raw2["R2_raw"].max(), df_raw2["R2_pp"].max()) + 0.02
    fig_scat.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                       line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5))
    fig_scat.add_annotation(x=mx-0.02, y=mx-0.02, text="y = x (no change)",
                             showarrow=False, font=dict(color="#64748b", size=10))
    dark_layout(fig_scat, height=460)
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── RMSE line chart by pipeline ───────────────────────────────────────────
    st.markdown('<div class="section-header">📉 RMSE Trend Across Pipelines</div>',
                unsafe_allow_html=True)
    fig_rmse = make_subplots(rows=1, cols=2,
                              subplot_titles=["1h Horizon", "24h Horizon"],
                              shared_yaxes=True)
    for ci, hz in enumerate(["1h","24h"], 1):
        sub = df_raw2[df_raw2["Horizon"] == hz].copy()
        sub["_order"] = sub["Pipeline"].apply(
            lambda x: PIPELINE_ORDER.index(x) if x in PIPELINE_ORDER else 99)
        sub = sub.sort_values("_order")
        for model in ["LSTM","GRU","CNN-LSTM","PI-LSTM"]:
            m = sub[sub["Model"] == model]
            if m.empty: continue
            fig_rmse.add_trace(go.Scatter(
                name=model if ci == 1 else None,
                x=m["Pipeline"], y=m["RMSE_pp"],
                mode="lines+markers",
                line=dict(color=MODEL_COLORS.get(model,"#94a3b8"), width=2.5),
                marker=dict(size=9, symbol="circle",
                            line=dict(width=1.5, color="rgba(255,255,255,0.4)")),
                showlegend=(ci == 1),
                hovertemplate=f"<b>{model}</b><br>%{{x}}<br>RMSE: %{{y:.1f}} W/m²<extra></extra>",
            ), row=1, col=ci)

    fig_rmse.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=CHART_BG,
        font=FONT_CFG, height=420,
        title="RMSE (W/m²) across pipelines — both horizons",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(t=60, b=30),
    )
    fig_rmse.update_xaxes(gridcolor=GRID_CLR, tickangle=-20)
    fig_rmse.update_yaxes(gridcolor=GRID_CLR, title_text="RMSE W/m²")
    st.plotly_chart(fig_rmse, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown('<div class="section-header">🔄 Live Training Monitor</div>',
                unsafe_allow_html=True)

    # ── Auto-refresh controls ────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 3])
    auto_refresh = ctrl_c1.checkbox("⟳ Auto-refresh", value=True,
                                    help="Page refreshes every N seconds while training")
    refresh_sec = ctrl_c2.selectbox("Interval", [15, 30, 60, 120], index=1,
                                    label_visibility="collapsed")
    manual_btn = ctrl_c3.button("🔁 Refresh Now", type="secondary")
    if manual_btn:
        st.rerun()

    # ── Load current results ─────────────────────────────────────────────────
    now_ts = pd.Timestamp.now()
    results_exist = RESULTS_CSV.exists()
    df_live = pd.read_csv(RESULTS_CSV).dropna(how="all") if results_exist else pd.DataFrame()

    completed_runs = len(df_live)
    expected_runs  = st.session_state.get("expected_runs", 0)
    config_ts      = st.session_state.get("config_ts", None)

    # If CSV has more rows than expected, update expected
    if completed_runs > expected_runs:
        expected_runs = completed_runs

    # ── Top status strip ─────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        is_training = results_exist and completed_runs < expected_runs and expected_runs > 0
        icon = "🟡" if is_training else ("🟢" if completed_runs > 0 else "⚫")
        status_txt = "Training…" if is_training else ("Complete ✅" if completed_runs > 0 else "Idle")
        st.markdown(f'<div class="monitor-card"><div class="big">{icon}</div>'
                    f'<div class="sub">{status_txt}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="monitor-card"><div class="big" style="color:#48bb78">{completed_runs}</div>'
                    f'<div class="sub">Runs completed</div></div>', unsafe_allow_html=True)
    with s3:
        remaining = max(0, expected_runs - completed_runs)
        st.markdown(f'<div class="monitor-card"><div class="big" style="color:#fbbf24">{remaining}</div>'
                    f'<div class="sub">Runs remaining (expected)</div></div>', unsafe_allow_html=True)
    with s4:
        if results_exist:
            mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(RESULTS_CSV))
            age   = int((now_ts - mtime).total_seconds())
            age_str = f"{age}s ago" if age < 120 else f"{age//60}m ago"
            st.markdown(f'<div class="monitor-card"><div class="big" style="font-size:1.1rem;color:#94a3b8;padding-top:0.5rem">{mtime.strftime("%H:%M:%S")}</div>'
                        f'<div class="sub">CSV last updated ({age_str})</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="monitor-card"><div class="big" style="color:#64748b">—</div>'
                        '<div class="sub">No results yet</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Progress bar ─────────────────────────────────────────────────────────
    if expected_runs > 0:
        pct = min(completed_runs / expected_runs, 1.0)
        st.markdown(f"**Progress: {completed_runs} / {expected_runs} model runs "
                    f"({pct*100:.0f}%)**")
        st.progress(pct)
        if config_ts:
            st.caption(f"Config generated at: {config_ts}")
    else:
        st.info("ℹ️ Click **🚀 Generate Config & Run** in the sidebar first "
                "to set up your experiment. The monitor will then track progress here.")

    # ── Live results table ────────────────────────────────────────────────────
    if not df_live.empty:
        st.markdown('<div class="section-header">📋 Completed Runs (live)</div>',
                    unsafe_allow_html=True)

        # Colour rows by R² quality
        show_cols = ["Pipeline","Horizon","Model","R2_pp","RMSE_pp","MAE_pp","Violations"]
        df_show = df_live[[c for c in show_cols if c in df_live.columns]].copy()

        def colour_r2(val):
            if not isinstance(val, float): return ""
            if val >= 0.90: return "color:#48bb78;font-weight:700"
            if val >= 0.80: return "color:#fbbf24;font-weight:600"
            return "color:#fc8181"

        styled = df_show.style
        if "R2_pp" in df_show.columns:
            styled = styled.applymap(colour_r2, subset=["R2_pp"])
        styled = styled.format({k: "{:.4f}" for k in ["R2_pp","R2_raw"] if k in df_show.columns})
        styled = styled.format({k: "{:.1f}"  for k in ["RMSE_pp","MAE_pp"] if k in df_show.columns})

        st.dataframe(styled, use_container_width=True, height=min(420, 55 + len(df_show)*35))

        # ── Mini leaderboard ─────────────────────────────────────────────────
        if "R2_pp" in df_live.columns and len(df_live) >= 2:
            st.markdown('<div class="section-header">🏅 Current Leaderboard (Top 5 by R²)</div>',
                        unsafe_allow_html=True)
            top5 = (df_live
                    .sort_values("R2_pp", ascending=False)
                    .head(5)[[c for c in show_cols if c in df_live.columns]]
                    .reset_index(drop=True))
            top5.index = ["🥇","🥈","🥉","4","5"][:len(top5)]
            st.dataframe(
                top5.style
                    .format({"R2_pp":"{:.4f}","RMSE_pp":"{:.1f}","MAE_pp":"{:.1f}"})
                    .applymap(colour_r2, subset=["R2_pp"] if "R2_pp" in top5.columns else []),
                use_container_width=True, height=230,
            )

        # ── Sparkline mini-chart: R² progress over runs ───────────────────────
        if "R2_pp" in df_live.columns and len(df_live) >= 3:
            st.markdown('<div class="section-header">📈 R² as Runs Complete (training timeline)</div>',
                        unsafe_allow_html=True)
            df_spark = df_live.copy().reset_index(drop=True)
            df_spark["Run #"] = df_spark.index + 1
            df_spark["Label"] = df_spark["Pipeline"] + " / " + df_spark["Model"]

            spark_fig = go.Figure()
            for hz in df_spark["Horizon"].unique():
                sub = df_spark[df_spark["Horizon"] == hz]
                spark_fig.add_trace(go.Scatter(
                    x=sub["Run #"], y=sub["R2_pp"],
                    mode="lines+markers+text",
                    name=f"R² — {hz}",
                    text=sub["Model"],
                    textposition="top center",
                    textfont=dict(size=8, color="#94a3b8"),
                    line=dict(width=2),
                    marker=dict(size=8, symbol="circle",
                                line=dict(width=1.5, color="rgba(255,255,255,0.4)")),
                    hovertemplate="<b>Run %{x}</b><br>%{customdata}<br>R²: %{y:.4f}<extra></extra>",
                    customdata=sub["Label"],
                ))

            spark_fig.add_hline(y=0.9, line_dash="dash",
                                line_color="rgba(72,187,120,0.4)",
                                annotation_text="R²=0.90",
                                annotation_font_color="#48bb78")
            dark_layout(spark_fig,
                        title="R² over time (each point = one model/pipeline finished)",
                        height=380)
            spark_fig.update_layout(
                xaxis_title="Completed Run #",
                yaxis_title="R² (post-processed)",
                yaxis=dict(range=[max(0, df_spark["R2_pp"].min()-0.05), 1.02],
                           gridcolor=GRID_CLR),
            )
            st.plotly_chart(spark_fig, use_container_width=True)

        # ── Per-pipeline progress ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📦 Per-Pipeline Status</div>',
                    unsafe_allow_html=True)
        models_expected = ["LSTM","GRU","CNN-LSTM","PI-LSTM"]
        horizons_expected = ["1h","24h"]
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as _f:
                    _cfg = json.load(_f)
                pipelines_expected = [f"Pipeline {p}" for p in _cfg.get("run_pipelines", [])]
                horizons_expected  = _cfg.get("run_horizons", ["1h","24h"])
            except Exception:
                pipelines_expected = df_live["Pipeline"].unique().tolist() if not df_live.empty else []
        else:
            pipelines_expected = df_live["Pipeline"].unique().tolist() if not df_live.empty else []

        for pl in (pipelines_expected or df_live["Pipeline"].unique().tolist()):
            pl_done = df_live[df_live["Pipeline"] == pl] if not df_live.empty else pd.DataFrame()
            done_cnt = len(pl_done)
            total_pl = len(models_expected) * len(horizons_expected)
            pct_pl   = done_cnt / total_pl if total_pl > 0 else 0

            status_icon = "✅" if done_cnt >= total_pl else ("⏳" if done_cnt > 0 else "⬜")
            best_r2_str = ""
            if not pl_done.empty and "R2_pp" in pl_done.columns:
                best_r2_str = f" · best R²={pl_done['R2_pp'].max():.4f}"

            st.markdown(
                f"{status_icon} **{pl}** — {done_cnt}/{total_pl} runs{best_r2_str}"
            )
            st.progress(pct_pl)

    else:
        st.markdown('<div class="section-header">📋 Completed Runs</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:2rem;text-align:center;color:#64748b;">
            <div style="font-size:2.5rem">📭</div>
            <div style="margin-top:0.5rem">No training results yet.</div>
            <div style="font-size:0.8rem;margin-top:0.3rem">
                Run your notebook cell or training script, then click Refresh.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Auto-refresh logic ────────────────────────────────────────────────────
    if auto_refresh and is_training if results_exist else auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAINING LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<div class="section-header">📋 Training Log & Commands</div>',
                unsafe_allow_html=True)

    log_text = "\n".join(st.session_state.log_lines) if st.session_state.log_lines \
               else "Click '🚀 Generate Config & Run' in the sidebar to start."
    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

    st.markdown("#### ▶️ Run this command in your terminal / Anaconda Prompt:")
    st.code(f'python "{TRAINING_SCRIPT}" --config "{CONFIG_FILE}"', language="bash")

    st.markdown("#### ⚙️ Current Config:")
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            st.json(json.load(f))
    else:
        st.info("No config saved yet. Click 'Generate Config & Run' first.")

    if RESULTS_CSV.exists():
        mtime = os.path.getmtime(RESULTS_CSV)
        ts = pd.Timestamp.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"✅ Last results file updated: `{ts}`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
with tab_howto:
    st.markdown("""
## 📖 How to Use This Dashboard

### Step 1 — Configure (Sidebar)
| Control | What it does |
|---|---|
| **Site** | Pick one of the 6 Koshi Province sites, or Combined |
| **Pipelines A–D** | Select feature engineering strategies |
| **Pipelines F & G** | Seasonal-window strategies from RA PDF |
| **Horizons** | 1h = next hour, 24h = next day |
| **Epochs / Patience** | Training duration controls |
| **Batch size** | Memory/speed tradeoff |

### Step 2 — Generate Config
Click **🚀 Generate Config & Run** in the sidebar.
This saves a `ui_config.json` file with all your settings.

### Step 3 — Run Training in Jupyter
Paste and run the `ra_pipeline_fg.py` code in a Jupyter cell after your preprocessing.

> ⏱️ Pipeline B alone takes ~33 min. Pipeline C takes ~60 min per horizon.

### Step 4 — View Results
Once training saves the results CSV, refresh this page and check
the **Results** and **Deep Dive** tabs.

---

## 🔬 Pipeline Summary

| Pipeline | Window | Feature strategy |
|---|---|---|
| **A** | 17 anchor points | 10 yearly + 7 daily sparse lags |
| **B** | 168h (1 week)  | Dense consecutive hourly window |
| **C** | 336h (2 weeks) | Dense consecutive hourly window |
| **D** | 24h (1 day)    | Dense consecutive hourly window |
| **F** | 14d seasonal   | N yearly lags only, no trailing |
| **G** | 14d seasonal   | N yearly lags + 14d trailing block |

All pipelines include **target-hour geometry** (clearsky GHI, cos_zenith,
hour_sin/cos, doy_sin/cos) from the **prediction time** — per RA didi's correction.

## 🧠 Model Summary

| Model | Description |
|---|---|
| **LSTM**    | Long Short-Term Memory — standard temporal baseline |
| **GRU**     | Gated Recurrent Unit — lighter, often matches LSTM |
| **CNN-LSTM**| CNN extracts local patterns → LSTM learns long-range |
| **PI-LSTM** | Physics-Informed: penalizes negative GHI, night predictions, ceiling violations |

## 📊 Chart Guide

| Chart | What to look for |
|---|---|
| **R² Heatmap** | Warm yellow = best performance |
| **Bubble chart** | Top-right corner (high R², low RMSE), small bubble = best |
| **Horizon degradation** | Small bar = model stays accurate even at 24h |
| **Radar** | Larger polygon = stronger all-round model |
| **Parallel coords** | Lines that stay high on all axes = best |
| **Physics violations** | PI-LSTM should have the fewest |
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: #475569; font-size: 0.78rem;">
  Solar Irradiance Forecasting · Koshi Province, Nepal<br/>
  LSTM / GRU / CNN-LSTM / PI-LSTM · Pipelines A–D, F, G · Python dashboard
</div>
""", unsafe_allow_html=True)
